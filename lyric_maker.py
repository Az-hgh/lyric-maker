#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lyric_maker.py —— 从音乐/视频自动生成标准 .lrc 歌词

流程：ffmpeg 解码 -> faster-whisper 本地语音识别(带时间戳) -> 质量过滤 -> 写出 .lrc

设计要点（针对"歌曲听写"这个真实难点）：
  * condition_on_previous_text=False —— 最关键的开关。默认 True 时，Whisper 在
    伴奏/间奏段落会陷入"重复上一句"的幻觉循环；关掉后大幅改善。
  * 段落级时间戳足够写 LRC（逐行），不需要逐字时间戳，省一半算力。
  * 幻觉黑名单 + 相邻重复段去重 + no_speech_prob 过滤，三道过滤专治听写垃圾。
  * --split-long 可按标点把过长段落拆成多行，时间按字数比例分配。

用法示例：
  python lyric_maker.py "C:\\path\\song.mp3"
  python lyric_maker.py song.flac --model medium --language zh
  python lyric_maker.py video.mp4 --model turbo --out out\\video.lrc
  python lyric_maker.py --gui
"""

import os
import sys
import re
import io
import json
import time
import glob
import queue
import shutil
import difflib
import argparse
import threading
import base64
import hashlib
import subprocess
import tempfile
import traceback
import urllib.parse
import urllib.request

# --- 依赖目录挂载 / 打包后的路径基准 ---
# 源码运行时：把 pip --target 装出来的 .deps / .deps-ocr 挂进 sys.path。
# 打包成 exe 后（sys.frozen）：依赖已经打进包里，不能再改 sys.path；
#   此时 _HERE 指向 exe 所在目录，这样 models/、out/ 就会落在 exe 旁边，
#   用户把 exe 拷到别处、旁边放个 models 文件夹就能用。
_FROZEN = bool(getattr(sys, "frozen", False))
if _FROZEN:
    _HERE = os.path.dirname(os.path.abspath(sys.executable))
else:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)

    _DEPS = os.path.join(_HERE, ".deps")
    if os.path.isdir(_DEPS) and _DEPS not in sys.path:
        sys.path.insert(0, _DEPS)
    # OCR 依赖放独立目录，且排在 .deps 之后：
    #   1) pip --target 到已装有同名包的目录会先清空再装，这里绕开；
    #   2) 排在后面意味着 numpy/onnxruntime/cv2 仍用 .deps 里已验证可用的版本，
    #      .deps-ocr 只补 rapidocr/shapely/pyclipper 这三个 .deps 里没有的。
    _DEPS_OCR = os.path.join(_HERE, ".deps-ocr")
    if os.path.isdir(_DEPS_OCR) and _DEPS_OCR not in sys.path:
        sys.path.append(_DEPS_OCR)
    # yt-dlp 也单独放一个目录：它自己会带 websockets / pycryptodomex 之类，
    # 混进 .deps 容易和已验证过的版本打架。排在最后，只在"视频下载"功能里才 import。
    _DEPS_YTDLP = os.path.join(_HERE, ".deps-ytdlp")
    if os.path.isdir(_DEPS_YTDLP) and _DEPS_YTDLP not in sys.path:
        sys.path.append(_DEPS_YTDLP)
    # yt-dlp-ejs：解 YouTube "n 参数"挑战的求解器脚本（不然只给低清格式）。
    # 单独一个目录，和 yt-dlp 并列。
    _DEPS_EJS = os.path.join(_HERE, ".deps-ejs")
    if os.path.isdir(_DEPS_EJS) and _DEPS_EJS not in sys.path:
        sys.path.append(_DEPS_EJS)


# 限制底层数值库（OpenBLAS / OpenMP / MKL 等）的线程池上限。
# 这是 CPU 模式下内存暴涨的主因之一：numpy/OpenBLAS 默认会按核心数开满线程，
# 每个线程各占一份缓冲区；而 faster-whisper 自己又按 cpu_threads 开一组。
# 这里统一把它们压到"实际使用的线程数"，避免双重开线程吃爆内存。
# 只在环境变量还没被显式设置时才填默认值，方便用户在 start.bat / 系统里手动覆盖。
_THREAD_CAP_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "GOTO_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def _set_thread_caps(n):
    try:
        n = max(1, int(n)) if n else 1
    except Exception:
        n = 1
    for k in _THREAD_CAP_KEYS:
        if k not in os.environ:
            os.environ[k] = str(n)


MODELS_DIR = os.path.join(_HERE, "models")

# 友好名 -> HuggingFace 仓库（经 hf-mirror.com 下载）
MODEL_REPOS = {
    "tiny":     "Systran/faster-whisper-tiny",
    "base":     "Systran/faster-whisper-base",
    "small":    "Systran/faster-whisper-small",
    "medium":   "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
    "turbo":    "deepdml/faster-whisper-large-v3-turbo-ct2",
}

# Whisper 在中文素材上常见的"训练集残留"幻觉，必须清掉
HALLUCINATION_PATTERNS = [
    r"请不吝?点赞", r"订阅(频道|转发)?", r"打赏", r"明镜与点点", r"点点栏目",
    r"字幕(由|志愿者|组)", r"Amara\.org", r"MING\s*PAO", r"谢谢(大家)?观看",
    r"感谢(您的)?观看", r"本字幕", r"翻译\s*[:：]", r"校对\s*[:：]", r"时间轴\s*[:：]",
    # Whisper 中文模型的经典幻觉：实测在真实素材开头直接吐出了这句
    r"优优独播剧场", r"独播剧场", r"YoYo\s*Television", r"YoYo\s*Television\s*Series",
    r"^\s*[（(【\[]?(完|结束|音乐|间奏|前奏|尾奏|纯音乐)[）)】\]]?\s*$",
]
HALLUCINATION_RE = re.compile("|".join(HALLUCINATION_PATTERNS), re.IGNORECASE)

AUDIO_EXT = {".mp3", ".flac", ".m4a", ".wav", ".aac", ".ogg", ".opus", ".wma", ".ape", ".alac"}
VIDEO_EXT = {".mp4", ".mkv", ".avi", ".flv", ".webm", ".mov", ".ts", ".m4v", ".wmv", ".mpg", ".mpeg"}


# =====================================================================
# ffmpeg
# =====================================================================
def find_ffmpeg(explicit=None):
    if explicit:
        if os.path.isfile(explicit):
            return explicit
        raise FileNotFoundError("指定的 ffmpeg 不存在: %s" % explicit)
    env = os.environ.get("FFMPEG")
    if env and os.path.isfile(env):
        return env
    # ① 随程序携带的 runtime/ffmpeg.exe —— 拷走即用的关键，绝对路径在本机，
    #    到了别的电脑上必然失效，所以自带件必须排在最前面。
    b = bundled_runtime("ffmpeg.exe")
    if b:
        return b
    from shutil import which
    found = which("ffmpeg")
    if found:
        return found
    for c in [
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        os.path.join(_HERE, "ffmpeg.exe"),
        os.path.join(_HERE, "bin", "ffmpeg.exe"),
    ]:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "找不到 ffmpeg。请把它放到 PATH，或用 --ffmpeg 指定路径，或设置环境变量 FFMPEG。")


def probe_duration(ffmpeg, path):
    """用 ffmpeg 读时长（不需要 ffprobe）。"""
    try:
        p = subprocess.run([ffmpeg, "-hide_banner", "-i", path],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           creationflags=_no_window())
        txt = p.stdout.decode("utf-8", "replace")
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+)(?:\.(\d+))?", txt)
        if m:
            h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
            frac = m.group(4) or "0"
            return h * 3600 + mi * 60 + float("%d.%s" % (s, frac))
    except Exception:
        pass
    return None


def _no_window():
    """Windows 下避免弹出黑窗口。"""
    if os.name == "nt":
        return 0x08000000  # CREATE_NO_WINDOW
    return 0


def decode_to_wav(ffmpeg, src, dst, start=None, duration=None, sample_rate=16000, log=print):
    """解码成 16kHz 单声道 WAV —— Whisper 的输入规格。"""
    cmd = [ffmpeg, "-hide_banner", "-nostdin", "-y"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", src]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-vn", "-ac", "1", "-ar", str(sample_rate), "-c:a", "pcm_s16le", dst]
    log("  解码: %s" % os.path.basename(src))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                       creationflags=_no_window())
    if p.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) == 0:
        tail = p.stdout.decode("utf-8", "replace").strip().splitlines()[-6:]
        raise RuntimeError("ffmpeg 解码失败:\n  " + "\n  ".join(tail))
    return dst


# =====================================================================
# 元数据（写进 LRC 头部标签）
# =====================================================================
def read_wav_as_float32(path):
    """用标准库 wave + numpy 把 16k/单声道/16bit WAV 读成 float32 数组。

    刻意绕开 faster-whisper 内置的 PyAV 解码路径：本环境装到的 av 19.0.1 与
    faster_whisper 1.2.1 的 decode_audio(metadata_errors=...) 调用不兼容
    （open() 不接受该参数）。把 numpy 数组直接交给 transcribe 可以完全跳过
    那段代码，还省掉一次多余的解码。
    """
    import wave
    import numpy as np
    with wave.open(path, "rb") as w:
        ch, sw, fr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise RuntimeError("预期 16 位 PCM，实际采样宽度 %d 字节" % sw)
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if fr != 16000:
        raise RuntimeError("预期 16kHz，实际 %d Hz" % fr)
    return a


# =====================================================================
# 歌词文本对齐模式（高精度路径）
#
# 为什么需要它：中文"歌唱"的语音识别本质上是不可靠的 —— 换再大的模型也会
# 输出"听着像但字不对"的结果。但如果你手里有**正确的歌词文本**，问题就从
# "识别"变成了"对齐"：识别结果只用来当时间锚点，文字一律用你提供的原文。
# 这样只要时间轴对，产出的 LRC 就是完美的。
# =====================================================================

def _norm_for_match(s):
    """只保留汉字与字母数字用于相似度匹配，丢掉标点和空格。"""
    out = []
    for ch in s:
        if "\u4e00" <= ch <= "\u9fff" or ch.isalnum():
            out.append(ch.lower())
    return "".join(out)


def read_lyrics_file(path):
    """读取歌词文本，容忍常见编码，并剥掉已有时间戳/元数据行。"""
    raw = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "gb18030", "utf-16", "latin-1"):
        try:
            with io.open(path, "r", encoding=enc) as f:
                raw = f.read()
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if raw is None:
        raise RuntimeError("无法识别歌词文件编码: %s" % path)

    lines = []
    for ln in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        t = ln.strip()
        if not t:
            continue
        # 已经是 LRC 的话剥掉时间戳，元数据行整行丢弃
        if re.match(r"^\[(ti|ar|al|by|offset|length|re|ve):", t, re.IGNORECASE):
            continue
        t = re.sub(r"^(\[\d{1,3}:\d{1,2}(?:[.:]\d{1,3})?\])+", "", t).strip()
        t = re.sub(r"^\d{1,3}[\.、]\s*", "", t)      # 去掉行首序号
        if t:
            lines.append(t)
    return lines


def _make_monotone_knots(pairs, min_gap=0.05):
    """把 (行号, 时间) 整理成时间严格递进的锚点序列。

    时间没有前进的锚点不可信（典型症状：好几行被压进几十毫秒的同一窗口），
    直接丢掉，交给后面的折线插值重建。
    """
    pairs = sorted(pairs, key=lambda p: p[0])
    out = []
    for i, t in pairs:
        if out and t <= out[-1][1] + min_gap:
            continue
        out.append((i, t))
    return out


def _interp_curve(knots, n, total):
    """按锚点折线插值出每一行的时间；两端用端点区间的斜率外推。"""
    def at(i):
        if len(knots) == 1:
            return max(0.0, knots[0][1] + (i - knots[0][0]) * 2.0)
        if i <= knots[0][0]:
            (i0, t0), (i1, t1) = knots[0], knots[1]
            slope = (t1 - t0) / max(1, i1 - i0)
            return max(0.0, t0 - slope * (i0 - i))
        if i >= knots[-1][0]:
            (i0, t0), (i1, t1) = knots[-2], knots[-1]
            slope = (t1 - t0) / max(1, i1 - i0)
            v = t1 + slope * (i - knots[-1][0])
            return min(total, v) if total else v
        for a in range(len(knots) - 1):
            (i0, t0), (i1, t1) = knots[a], knots[a + 1]
            if i0 <= i <= i1:
                return t0 if i1 == i0 else t0 + (t1 - t0) * (i - i0) / (i1 - i0)
        return knots[-1][1]
    return [at(i) for i in range(n)]


def align_lyrics_to_audio(lyric_lines, words, total_duration=None, log=print,
                          max_dev=4.0, min_gap=0.35):
    """把已知歌词文本对齐到音频时间轴。

    思路：识别出的词级时间戳给出"什么时候唱到了哪些字"，用 difflib 求
    歌词文本与识别文本的最长匹配块，从而知道每行歌词的首字出现在何时。
    未匹配上的行用相邻已知行的位置做线性插值。
    """
    if not lyric_lines:
        return []
    if not words:
        raise RuntimeError("对齐需要词级时间戳，但识别没有产出任何词。"
                           "请确认模型可用，或用更长的音频重试。")

    # 展平 ASR 词 -> 字符级时间
    asr_chars, asr_times = [], []
    for w in words:
        t = w.get("start", 0.0)
        for ch in _norm_for_match(w.get("w", "")):
            asr_chars.append(ch)
            asr_times.append(t)
    asr_text = "".join(asr_chars)

    # 歌词行 -> 字符级，同时记住每行的起始下标
    lyr_chars, line_at = [], []
    for line in lyric_lines:
        line_at.append(len(lyr_chars))
        lyr_chars.extend(_norm_for_match(line))
    lyr_text = "".join(lyr_chars)

    log("  对齐: 歌词 %d 行/%d 字 vs 识别 %d 字"
        % (len(lyric_lines), len(lyr_text), len(asr_text)))
    if not lyr_text or not asr_text:
        raise RuntimeError("歌词或识别结果为空，无法对齐。")

    sm = difflib.SequenceMatcher(None, lyr_text, asr_text, autojunk=False)
    blocks = [(i, j, n) for (i, j, n) in sm.get_matching_blocks() if n > 0]
    total_chars = len(lyr_text)
    matched = sum(n for (_, _, n) in blocks)
    log("  对齐: 字符匹配率 %.1f%%（%d 个匹配块）"
        % (100.0 * matched / max(1, total_chars), len(blocks)))

    line_end = [(line_at[i + 1] if i + 1 < len(line_at) else total_chars)
                for i in range(len(lyric_lines))]
    line_len = [line_end[i] - line_at[i] for i in range(len(lyric_lines))]

    # 原始锚点：取该行【第一个被匹配到的字符】对应的时间 —— 正常情况下这是最准的
    # 定义（锚点误差只受行首几个字识别错误影响，通常 <1s）。
    # 它唯一的弱点是：若行首几个字在很远的重复段（副歌）里命中，整行会被拖偏十几秒。
    # 这个弱点交给后面的"离群锚点迭代剔除"处理，所以这里同时累计覆盖度作为可信度。
    raw = [None] * len(lyric_lines)
    cover = [0] * len(lyric_lines)
    for (bi, bj, bn) in blocks:
        b_end = bi + bn
        for li in range(len(lyric_lines)):
            lo = bi if bi > line_at[li] else line_at[li]
            hi = b_end if b_end < line_end[li] else line_end[li]
            if hi <= lo:
                continue
            cover[li] += hi - lo
            if raw[li] is None:
                raw[li] = asr_times[bj + (lo - bi)]

    if not any(r is not None for r in raw):
        raise RuntimeError("歌词与音频内容完全不匹配（一个字都没对上）。\n"
                           "请确认歌词文件与音频是同一首歌。")

    # 只信覆盖度足够的锚点，然后迭代剔除与单调趋势严重不符的离群点
    accepted = [i for i in range(len(lyric_lines))
                if raw[i] is not None
                and cover[i] >= max(2, int(0.20 * max(1, line_len[i])))]
    if not accepted:
        accepted = [i for i in range(len(lyric_lines)) if raw[i] is not None]

    for rnd in range(6):
        knots = _make_monotone_knots([(i, raw[i]) for i in accepted])
        if len(knots) < 2:
            break
        curve = _interp_curve(knots, len(lyric_lines), total_duration)
        badset = set(knots)
        bad = [i for i in accepted if i in badset and abs(raw[i] - curve[i]) > max_dev]
        if not bad:
            break
        log("    稳健化第 %d 轮: 剔除 %d 个离群锚点" % (rnd + 1, len(bad)))
        accepted = [i for i in accepted if i not in set(bad)]

    if not accepted:
        accepted = [i for i in range(len(lyric_lines)) if raw[i] is not None]

    knots = _make_monotone_knots([(i, raw[i]) for i in accepted])
    if not knots:
        raise RuntimeError("没有可用于对齐的锚点。")
    starts = _interp_curve(knots, len(lyric_lines), total_duration)

    # 最终保证严格递增且非负
    if starts[0] < 0:
        starts[0] = 0.0
    for i in range(1, len(starts)):
        if starts[i] <= starts[i - 1]:
            starts[i] = starts[i - 1] + min_gap

    end_limit = total_duration or (starts[-1] + 4.0)
    segs = []
    for i, line in enumerate(lyric_lines):
        e = starts[i + 1] if i + 1 < len(starts) else min(end_limit, starts[i] + 4.0)
        if e <= starts[i]:
            e = starts[i] + min_gap
        segs.append({"start": starts[i], "end": e, "text": line, "no_speech": 0.0})
    log("  对齐完成: %d 行，%d 个锚点直接定位，其余按折线插值"
        % (len(segs), len(knots)))
    return segs


def read_tags(path):
    meta = {}
    try:
        from mutagen import File as MutaFile
        f = MutaFile(path, easy=True)
        if f is not None:
            for k, lrc_key in (("title", "ti"), ("artist", "ar"), ("album", "al")):
                v = f.get(k)
                if v:
                    meta[lrc_key] = v[0] if isinstance(v, (list, tuple)) else str(v)
            if getattr(f, "info", None) is not None and getattr(f.info, "length", None):
                meta["length"] = f.info.length
    except Exception:
        pass
    return meta


# =====================================================================
# 质量过滤
# =====================================================================
def is_junk(text):
    t = text.strip()
    if not t:
        return True
    # 去掉标点/空白后如果什么都不剩，就是垃圾
    if not re.sub(r"[\s\W_]+", "", t, flags=re.UNICODE):
        return True
    if HALLUCINATION_RE.search(t):
        return True
    return False


_OPENCC = {}


def _get_opencc(config):
    """按配置取 OpenCC 实例（带缓存）。取不到就返回 False。"""
    if config not in _OPENCC:
        try:
            from opencc import OpenCC
            _OPENCC[config] = OpenCC(config)
        except Exception:
            _OPENCC[config] = False
    return _OPENCC[config]


def to_simplified(text):
    """把繁体统一成简体。

    Whisper 在中文素材上会简繁混用（同一首歌里既有"局势"又有"局勢"），
    对歌词文件来说很难看。opencc 可用就转换，不可用则原样返回（不影响主流程）。
    """
    cv = _get_opencc("t2s")
    if cv:
        try:
            return cv.convert(text)
        except Exception:
            return text
    return text


def to_traditional(text):
    """把简体统一成繁体（粤语歌词常用繁体字形）。"""
    cv = _get_opencc("s2t")
    if cv:
        try:
            return cv.convert(text)
        except Exception:
            return text
    return text


# 输出字形模式：用户可选"要什么字形"，而不是由语言硬性决定。
#   "keep"  保持模型输出原样（粤语默认：识别成什么就是什么）
#   "s2t"   统一转简体
#   "t2s"   统一转繁体
SCRIPT_MODES = ("keep", "s2t", "t2s")

SCRIPT_LABEL = {
    "keep": "保持识别原样",
    "s2t":  "统一转简体",
    "t2s":  "统一转繁体",
}


def apply_script(text, mode):
    """按字形模式转换文本。未知模式或 keep 都原样返回。"""
    if mode == "s2t":
        return to_simplified(text)
    if mode == "t2s":
        return to_traditional(text)
    return text


def clean_text(text, script="keep"):
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"^[\-–—\s]+", "", t)
    return apply_script(t, script).strip()


def filter_segments(segs, drop_repeats=True, min_dur=0.30, log=print, script="keep"):
    """segs: [{'start','end','text','no_speech'}] -> 过滤后的列表"""
    out = []
    dropped_junk = dropped_rep = dropped_short = 0
    prev = None
    for s in segs:
        s["text"] = clean_text(s["text"], script=script)
        if is_junk(s["text"]):
            dropped_junk += 1
            continue
        if s["end"] - s["start"] < min_dur:
            dropped_short += 1
            continue
        if drop_repeats and prev is not None and s["text"] == prev:
            dropped_rep += 1
            continue
        out.append(s)
        prev = s["text"]
    log("  过滤: 丢弃 垃圾/幻觉 %d, 过短 %d, 相邻重复 %d，保留 %d 段"
        % (dropped_junk, dropped_short, dropped_rep, len(out)))
    return out


def merge_short(segs, max_gap=0.60, max_chars=28):
    """把间隔很小且合计不长的相邻段合并成一行，避免 LRC 碎成一片。"""
    if not segs:
        return segs
    out = [dict(segs[0])]
    for s in segs[1:]:
        last = out[-1]
        gap = s["start"] - last["end"]
        if gap <= max_gap and len(last["text"]) + len(s["text"]) <= max_chars:
            last["text"] = last["text"] + s["text"]
            last["end"] = s["end"]
        else:
            out.append(dict(s))
    return out


def split_long(segs, max_chars=30):
    """按标点把过长的一行拆开，时间按字数比例分配（近似但实用）。"""
    out = []
    for s in segs:
        text = s["text"]
        if len(text) <= max_chars:
            out.append(s)
            continue
        parts = [p for p in re.split(r"(?<=[，。！？；、,.!?;])", text) if p.strip()]
        if len(parts) <= 1:
            out.append(s)
            continue
        total = sum(len(p) for p in parts)
        dur = s["end"] - s["start"]
        t = s["start"]
        for p in parts:
            d = dur * (len(p) / total) if total else 0
            out.append({"start": t, "end": t + d, "text": p.strip(),
                        "no_speech": s.get("no_speech", 0.0)})
            t += d
    return out


# =====================================================================
# LRC 输出
# =====================================================================
def fmt_ts(sec):
    if sec < 0:
        sec = 0.0
    # 用百分秒做进位：59.999 不能让 %.2f 四舍五入出 "60.00" 这种非法分钟时间，
    # 必须先进位到分钟（59.999 -> 01:00.00）。
    total_cs = int(round(sec * 100))
    m, cs = divmod(total_cs, 6000)
    return "[%02d:%05.2f]" % (m, cs / 100.0)


def write_lrc(path, segs, tags=None, offset_ms=0, by="lyric-maker"):
    tags = tags or {}
    lines = []
    if tags.get("ti"):
        lines.append("[ti:%s]" % tags["ti"])
    if tags.get("ar"):
        lines.append("[ar:%s]" % tags["ar"])
    if tags.get("al"):
        lines.append("[al:%s]" % tags["al"])
    lines.append("[by:%s]" % by)
    lines.append("[offset:%d]" % int(offset_ms))
    if tags.get("length"):
        lines.append("[length:%s]" % fmt_ts(tags["length"]).strip("[]"))
    lines.append("")
    shift = offset_ms / 1000.0
    for s in segs:
        lines.append("%s%s" % (fmt_ts(s["start"] + shift), s["text"]))
    text = "\r\n".join(lines) + "\r\n"
    # LRC 用 UTF-8 带 BOM：Windows 上的旧播放器/网易云对此最兼容
    with io.open(path, "w", encoding="utf-8-sig", newline="") as f:
        f.write(text)
    return path


SUBTITLE_FORMATS = ("lrc", "srt", "ass", "ssa", "vtt")


def normalize_format(fmt, out=None):
    """把 --format / --out 的扩展名归一成最终格式名。

    优先级：显式 --format > --out 的扩展名 > lrc（默认，保持老行为不变）。
    """
    f = (fmt or "").strip().lower()
    if f and f in SUBTITLE_FORMATS:
        return f
    if out:
        ext = os.path.splitext(out)[1].lower().lstrip(".")
        if ext in SUBTITLE_FORMATS:
            return ext
    return "lrc"


def write_subtitle_output(path, segs, fmt="lrc", tags=None, offset_ms=0):
    """按格式写字幕文件。lrc 走原来的 write_lrc，其余走 subtitle 的渲染函数。"""
    tags = tags or {}
    fmt = (fmt or "lrc").lower()
    if fmt == "lrc":
        return write_lrc(path, segs, tags=tags, offset_ms=offset_ms)

    shift = offset_ms / 1000.0
    if fmt == "srt":
        body = render_srt(segs, offset_sec=shift)
    elif fmt == "vtt":
        body = render_vtt(segs, offset_sec=shift)
    elif fmt in ("ass", "ssa"):
        body = render_ass(segs, offset_sec=shift,
                                   title=tags.get("ti") or tags.get("title"))
    else:
        raise ValueError("不支持的字幕格式: %r（可选 lrc/srt/ass/ssa/vtt）" % fmt)

    text = body.replace("\n", "\r\n") + "\r\n"
    # srt/ass 用 UTF-8 BOM 兼容 Windows 老播放器；vtt 按 WebVTT 惯例不带 BOM
    encoding = "utf-8-sig" if fmt in ("srt", "ass", "ssa") else "utf-8"
    with io.open(path, "w", encoding=encoding, newline="") as f:
        f.write(text)
    return path


# =====================================================================
# 核心
# =====================================================================
def load_model(name_or_path, threads=None, log=print):
    from faster_whisper import WhisperModel
    path = name_or_path
    if not os.path.isdir(path):
        cand = os.path.join(MODELS_DIR, name_or_path)
        if os.path.isdir(cand):
            path = cand
        else:
            raise FileNotFoundError(
                "找不到模型 '%s'。\n  先运行 start.bat，或 python lyric_maker.py --get-model %s\n  （或用 --model 直接给模型目录）"
                % (name_or_path, name_or_path))
    if threads is None:
        threads = min(16, os.cpu_count() or 4)
    # 把底层数值库的线程池上限对齐到本次实际使用的线程数，避免内存被开满的
    # 线程池占满（OpenBLAS/OpenMP 默认按物理核心数开线程，与 whisper 的
    # cpu_threads 叠加后会显著膨胀常驻内存）。
    _set_thread_caps(threads)
    log("  加载模型: %s  (CPU int8, %d 线程)" % (os.path.basename(path), threads))
    t0 = time.time()
    model = WhisperModel(path, device="cpu", compute_type="int8", cpu_threads=threads)
    log("  模型就绪，用时 %.1fs" % (time.time() - t0))
    return model


def _load_subtitle_source(path):
    """把文件当字幕解析。成功返回 [(start,end,text)]，不是字幕就返回 []。"""
    try:
        subs = load_subtitle_file(path)
    except Exception:
        return []
    # 至少两行、且时间轴确实在往后走，才认为是带时间戳的字幕
    if len(subs) >= 2 and subs[-1][0] > 1.0:
        return subs
    return []


def _try_subtitles(media, ff, mode, region, fps, refine, subs_file, min_score, log,
                   start=None, duration=None):
    """按模式取字幕，返回 (subs, 来源描述)；拿不到返回 (None, None)。

    track : 只用现成字幕（内嵌轨 / 外挂文件），快且准
    ocr   : 只识别画面上烧录的硬字幕
    auto  : 先找现成字幕，没有再上 OCR
    """
    mode = (mode or "off").lower()
    if mode == "off":
        return None, None
    if mode not in ("track", "ocr", "auto"):
        raise ValueError("--subs 只支持 off/track/ocr/auto，收到 %r" % mode)

    if subs_file:
        if not os.path.isfile(subs_file):
            raise FileNotFoundError("指定的字幕文件不存在: %s" % subs_file)
        subs = load_subtitle_file(subs_file)
        if subs:
            return subs, "字幕文件 %s" % os.path.basename(subs_file)
        log("  %s 里没解析出带时间戳的字幕，按普通歌词文本处理" % os.path.basename(subs_file))
        return None, None

    if mode in ("track", "auto"):
        log("  检查内嵌字幕轨…")
        streams = probe_subtitle_streams(ff, media)
        text_tracks = [s for s in streams if not s["image"]]
        if text_tracks:
            import shutil as _sh
            tmpd = tempfile.mkdtemp(prefix="subtrack_", dir=_tmp_root())
            try:
                dst = os.path.join(tmpd, "track.srt")
                for st in text_tracks:
                    lang = (" / " + st["lang"]) if st["lang"] else ""
                    log("  发现字幕轨 #%d (%s%s)，提取中…" % (st["index"], st["codec"], lang))
                    if extract_subtitle_track(ff, media, st["index"], dst):
                        subs = load_subtitle_file(dst)
                        if subs:
                            return subs, "内嵌字幕轨 #%d (%s)" % (st["index"], st["codec"])
                        log("  轨 #%d 提取出来是空的，换下一条" % st["index"])
            finally:
                _sh.rmtree(tmpd, ignore_errors=True)
        elif streams:
            log("  只有图像字幕轨（%s），ffmpeg 转不出文本，改用画面 OCR"
                % ",".join(s["codec"] for s in streams))

        sidecar = find_sidecar_subtitle(media)
        if sidecar:
            subs = load_subtitle_file(sidecar)
            if subs:
                return subs, "外挂字幕 %s" % os.path.basename(sidecar)
            log("  外挂字幕 %s 没能解析出内容" % os.path.basename(sidecar))

        if mode == "track":
            log("  没有现成字幕可用")
            return None, None

    if os.path.splitext(media)[1].lower() not in VIDEO_EXT:
        log("  音频文件没有画面，跳过 OCR")
        return None, None
    if not ocr_available():
        log("  未安装 OCR 引擎（运行 start.bat --install-ocr 装），本次改用语音识别")
        return None, None

    log("  没找到现成字幕，开始识别画面上的硬字幕（要解码画面，请耐心）")
    subs = ocr_hard_subtitles(ff, media, region=region, scan_fps=fps,
                                       refine=refine, min_score=min_score, log=log,
                                       start=start, duration=duration)
    if subs:
        return subs, "画面 OCR"
    log("  画面里没识别出字幕")
    return None, None


def _finish_from_subtitles(media, subs, src, out=None, offset_ms=0, fmt="lrc",
                           split_long_lines=True, total=None, log=print, script="keep"):
    """字幕拿到了就到此为止：文字是现成的，不需要语音识别。"""
    segs = subs_to_segments(subs)
    for s in segs:
        s["text"] = clean_text(s["text"], script=script)
    segs = [s for s in segs if s["text"]]
    if not segs:
        raise RuntimeError("字幕里没解析出有效文字")

    # 不做 merge：字幕的断行是人工做好的，合并反而会丢行。
    # 只拆过长行（OCR 把两行并成一行时会出现）。
    before = len(segs)
    if split_long_lines:
        segs = split_long(segs)
        if len(segs) != before:
            log("  拆分过长行: %d -> %d 行" % (before, len(segs)))

    tags = read_tags(media)
    write_subtitle_output(out, segs, fmt=fmt, tags=tags, offset_ms=offset_ms)
    log("  已写出: %s  (%d 行，来自%s)" % (out, len(segs), src))
    return {"out": out, "segments": segs, "elapsed": 0.0,
            "duration": total or (segs[-1]["end"] if segs else 0),
            "tags": tags, "info": None, "mode": "subtitle", "source": src}


def transcribe(media, model_name="medium", language="zh", out=None, prompt=None,
               vad=False, threads=None, start=None, duration=None,
               split_long_lines=True, merge=True, offset_ms=0,
               ffmpeg=None, keep_wav=False, lyrics_file=None, log=print,
               subs_mode="auto", subs_region="auto", subs_fps=2.0,
               subs_refine=True, subs_file=None, subs_min_score=0.5,
               fmt=None, script=None):
    if not os.path.isfile(media):
        raise FileNotFoundError("输入文件不存在: %s" % media)

    ext = os.path.splitext(media)[1].lower()
    kind = "视频" if ext in VIDEO_EXT else ("音频" if ext in AUDIO_EXT else "未知类型")
    ff = find_ffmpeg(ffmpeg)
    total = probe_duration(ff, media)
    log("输入: %s  (%s, 时长 %s)" % (os.path.basename(media), kind,
                                    ("%.1fs" % total) if total else "未知"))

    # 输出格式：显式 --format > --out 扩展名 > lrc。默认仍是 lrc，老行为不变。
    fmt = normalize_format(fmt, out)
    if out is None:
        out = os.path.splitext(media)[0] + "." + fmt

    # 语言策略在这里解析一次，字幕路径和语音识别路径共用。
    # 字形优先级：用户显式指定 > 语言默认（粤语=保留原样）> 全局默认（简体）。
    _pol = lang_policy(language)
    _script = resolve_script(language, script)

    # ---- 字幕优先：现成的文字永远比语音识别准，也比语音识别快 ----
    if subs_mode != "off":
        subs, src = None, None
        if lyrics_file and os.path.isfile(lyrics_file):
            got = _load_subtitle_source(lyrics_file)
            if got:
                subs, src = got, "字幕文件 %s" % os.path.basename(lyrics_file)
        if subs is None:
            subs, src = _try_subtitles(media, ff, subs_mode, subs_region, subs_fps,
                                       subs_refine, subs_file, subs_min_score, log,
                                       start=start, duration=duration)
        if subs:
            return _finish_from_subtitles(media, subs, src, out=out,
                                          offset_ms=offset_ms, fmt=fmt,
                                          split_long_lines=split_long_lines,
                                          total=total, log=log, script=_script)

    tmpdir = tempfile.mkdtemp(prefix="lyric_", dir=_tmp_root())
    wav = os.path.join(tmpdir, "audio16k.wav")
    try:
        decode_to_wav(ff, media, wav, start=start, duration=duration, log=log)
        audio = read_wav_as_float32(wav)
        log("  音频载入: %.1f 秒（%d 采样点）" % (len(audio) / 16000.0, len(audio)))
        model = load_model(model_name, threads=threads, log=log)

        # 语言策略（粤语等）：换专属提示词。用户自己传了 --prompt 就以用户的为准。
        # 字形如果被用户改了（粤语选"转简体"），提示词要跟着改 ——
        # 提示词说"繁体"而转换又转成简体，两头打架会拉低识别准确度。
        if prompt is None:
            if _pol:
                prompt = _pol["prompt"]
                if _script == "s2t":
                    prompt = "以下是粵語歌曲的歌詞，請用簡體中文並帶標點。"
                elif _script == "t2s":
                    prompt = "以下是粵語歌曲的歌詞，請用繁體中文並帶標點。"
            else:
                prompt = "以下是普通话歌曲的歌词，请输出简体中文并带标点。"
        if _pol:
            log("  语言 %s：%s" % (language, SCRIPT_LABEL.get(_script, _script)))

        log("  识别中（CPU，请耐心；速度约为实时的 0.5~2 倍）...")
        t0 = time.time()
        seg_iter, info = model.transcribe(
            audio,
            language=None if language in ("auto", "", None) else language,
            beam_size=5,
            best_of=5,
            patience=1.0,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            condition_on_previous_text=False,   # 关键：关掉跨段条件，抑制重复幻觉
            vad_filter=bool(vad),
            initial_prompt=prompt,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            word_timestamps=bool(lyrics_file),
        )
        segs = []
        words = []
        last_report = 0.0
        for s in seg_iter:
            segs.append({"start": s.start, "end": s.end, "text": s.text,
                         "no_speech": getattr(s, "no_speech_prob", 0.0)})
            if lyrics_file:
                for w in (getattr(s, "words", None) or []):
                    words.append({"w": getattr(w, "word", ""),
                                  "start": getattr(w, "start", s.start),
                                  "end": getattr(w, "end", s.end)})
            if total and s.end - last_report >= 15:
                last_report = s.end
                log("    进度 %5.1f%%  (%.0fs / %.0fs)"
                    % (min(100.0, 100.0 * s.end / total), s.end, total))
        elapsed = time.time() - t0
        log("  识别完成: %d 段原始输出，用时 %.1fs" % (len(segs), elapsed))

        if lyrics_file:
            # 对齐模式：文字用你提供的原文，识别只提供时间锚点
            lines = read_lyrics_file(lyrics_file)
            segs = align_lyrics_to_audio(lines, words, total_duration=total, log=log)
            tags = read_tags(media)
            write_subtitle_output(out, segs, fmt=fmt, tags=tags, offset_ms=offset_ms)
            log("  已写出: %s  (%d 行，对齐模式)" % (out, len(segs)))
            return {"out": out, "segments": segs, "elapsed": elapsed,
                    "duration": total, "tags": tags, "info": info, "mode": "align"}

        segs = filter_segments(segs, log=log, script=_script)
        if merge:
            before = len(segs)
            segs = merge_short(segs)
            if len(segs) != before:
                log("  合并短段: %d -> %d 行" % (before, len(segs)))
        if split_long_lines:
            before = len(segs)
            segs = split_long(segs)
            if len(segs) != before:
                log("  拆分行: %d -> %d 行" % (before, len(segs)))

        if start:
            for s in segs:
                s["start"] += float(start)
                s["end"] += float(start)

        tags = read_tags(media)
        write_subtitle_output(out, segs, fmt=fmt, tags=tags, offset_ms=offset_ms)
        log("  已写出: %s  (%d 行)" % (out, len(segs)))
        return {"out": out, "segments": segs, "elapsed": elapsed,
                "duration": total, "tags": tags, "info": info}
    finally:
        if not keep_wav:
            try:
                import shutil
                shutil.rmtree(tmpdir, ignore_errors=True)
            except Exception:
                pass
        else:
            log("  WAV 保留在: %s" % wav)
        # 主动释放大对象，让 Python 尽快把内存还给系统：模型权重（约 1~3GB）
        # 和整段音频在任务结束后已不再需要，不显式清掉的话要等下次 GC 才回收，
        # 频繁批量跑任务时内存会持续累积。
        try:
            del audio
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass
        try:
            import gc
            gc.collect()
        except Exception:
            pass


def _tmp_root():
    """临时目录优先放项目内，避免受限环境对 %TEMP% 的写入限制。"""
    d = os.path.join(_HERE, ".tmp")
    if os.path.isdir(d):
        return d
    return tempfile.gettempdir()


# =====================================================================
# 语言相关策略：提示词 + 默认字形
# =====================================================================
# 粤语（yue）必须单独对待提示词，原因很实在：
#   默认提示词写死了"请输出简体中文"。Whisper 会被提示词带着走，把粤语歌
#   往普通话词上靠。粤语歌词惯用繁体，转简体后"聲""愛""們"会被换成
#   "声""爱""们"，读起来完全不是那首歌的写法。
#
# 字形（简/繁）则是**用户的选择**，不是语言能替你决定的：
#   * 粤语歌默认保留原样（keep）—— 很多粤语歌的繁体写法本身就是对的，
#     无差别转简会改错词（同音字靠字形区分）。
#   * 但用户可能就想要简体版歌词（方便在简中播放器里看），所以做成可选项。
# 因此这里只给出"默认字形"，用户可在设置里覆盖。
LANG_POLICY = {
    "yue": {
        "label": "粤语（yue）",
        "prompt": "以下是粵語歌曲的歌詞，請用繁體中文並帶標點。",
        "script": "keep",       # 默认保留识别原样（繁体）
    },
}


def lang_policy(language):
    """取该语言的策略；未知语言返回 None（表示走默认行为）。"""
    return LANG_POLICY.get((language or "").strip().lower())


# 各语言的**默认**字形。未列出的语言沿用历史行为：转简体。
DEFAULT_SCRIPT = "s2t"


def resolve_script(language, script=None):
    """决定最终使用的字形模式。

    优先级：用户显式指定 > 语言默认 > 全局默认（简体）。
    script 为 None/空/"auto" 时按语言默认走。
    """
    if script and script != "auto" and script in SCRIPT_MODES:
        return script
    pol = lang_policy(language)
    if pol and pol.get("script") in SCRIPT_MODES:
        return pol["script"]
    return DEFAULT_SCRIPT


# =====================================================================
# 内置自检（不需要音频，纯离线验证算法与文件解析）
# =====================================================================
def selftest():
    import random
    random.seed(20261003)
    passed = [0]
    failed = [0]

    def check(name, cond, detail=""):
        if cond:
            passed[0] += 1
            print("  [PASS] %s" % name)
        else:
            failed[0] += 1
            print("  [FAIL] %s   %s" % (name, detail))

    print("=== lyric_maker 自检 ===")

    # ---------- 1. 歌词文件解析 ----------
    print("\n--- 歌词文件解析 ---")
    tmpdir = _tmp_root()
    cases = [
        ("utf-8-sig", "utf-8-sig", "[ti:测试]\n[00:12.34]第一行\n\n第二行\n1. 第三行\n"),
        ("gbk", "gbk", "[ar:歌手]\n[00:01.00]简体歌词一\n[00:05.50]简体歌词二\n"),
    ]
    for label, enc, content in cases:
        p = os.path.join(tmpdir, "_selftest_lyrics_%s.txt" % label)
        with io.open(p, "w", encoding=enc, newline="") as f:
            f.write(content)
        try:
            lines = read_lyrics_file(p)
            check("解析 %s 歌词文件" % label, len(lines) > 0, "得到 0 行")
            check("  %s 已剥掉时间戳与元数据" % label,
                  all(not ln.startswith("[") and not re.match(r"^\d+\.", ln) for ln in lines),
                  "仍有残留: %s" % lines)
        except Exception as e:
            check("解析 %s 歌词文件" % label, False, str(e))
        finally:
            try:
                os.remove(p)
            except Exception:
                pass

    # ---------- 1b. 字幕解析 ----------
    print("\n--- 字幕解析 ---")
    try:
        srt = ("1\n00:00:01,500 --> 00:00:04,000\n第一行字幕\n第二行字幕\n\n"
               "2\n00:00:05,000 --> 00:00:08,250\n<i>斜体</i>也要留下\n")
        got = parse_srt(srt)
        check("SRT 解析出行数", len(got) == 2, "得到 %d 行" % len(got))
        check("SRT 起止时间正确", abs(got[0][0] - 1.5) < 1e-6 and abs(got[1][1] - 8.25) < 1e-6,
              str(got))
        check("SRT 多行合并成一行", got[0][2] == "第一行字幕 第二行字幕", got[0][2])
        check("SRT 去掉 <i> 标签", got[1][2] == "斜体也要留下", got[1][2])
    except Exception as e:
        check("SRT 解析", False, str(e))

    try:
        ass = ("[Script Info]\nScriptType: v4.00+\n\n[V4+ Styles]\n"
               "Format: Name, Fontname\nStyle: Default,Arial\n\n[Events]\n"
               "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
               "Dialogue: 0,0:00:02.10,0:00:05.00,Default,,0,0,0,,{\\pos(10,20)}第一句\\N第二句\n"
               "Dialogue: 0,0:00:06.00,0:00:09.50,Default,,0,0,0,,带逗号的, 文本\n")
        got = parse_ass(ass)
        check("ASS 解析出行数", len(got) == 2, "得到 %d 行" % len(got))
        check("ASS 百分秒时间正确",
              abs(got[0][0] - 2.10) < 1e-6 and abs(got[1][0] - 6.0) < 1e-6, str(got))
        check("ASS 去掉样式块与 \\N", got[0][2] == "第一句 第二句", got[0][2])
        check("ASS 正文里的逗号不丢", got[1][2] == "带逗号的, 文本", got[1][2])
    except Exception as e:
        check("ASS 解析", False, str(e))

    try:
        got = parse_vtt("WEBVTT\n\n00:00:03.000 --> 00:00:06.500\n字幕一行\n")
        check("VTT 解析", len(got) == 1 and abs(got[0][0] - 3.0) < 1e-6, str(got))
    except Exception as e:
        check("VTT 解析", False, str(e))

    try:
        got = parse_lrc("[ti:x]\n[00:12.34]甲\n[01:05.00]乙\n")
        check("LRC 解析并跳过元数据行", len(got) == 2 and abs(got[0][0] - 12.34) < 1e-6, str(got))
        check("LRC 结束时间取下一行起点", abs(got[0][1] - 65.0) < 1e-6, str(got[0][1]))
    except Exception as e:
        check("LRC 解析", False, str(e))

    try:
        y0, y1 = resolve_region("bottom", 1920, 1080)
        check("字幕区域 bottom 落在下半部", 648 < y0 < 918 and y1 <= 1080, "%d-%d" % (y0, y1))
        check("字幕区域比例写法", resolve_region("0.5,0.9", 1920, 1080) == (540, 972),
              str(resolve_region("0.5,0.9", 1920, 1080)))
        check("字幕区域像素写法", resolve_region("900,1000", 1920, 1080) == (900, 1000),
              str(resolve_region("900,1000", 1920, 1080)))
    except Exception as e:
        check("字幕区域解析", False, str(e))

    if ocr_available():
        print("  [INFO] OCR 引擎已安装，硬字幕识别可用")
    else:
        print("  [INFO] 未安装 OCR 引擎，硬字幕识别不可用（运行 start.bat --install-ocr 安装）")

    # ---------- 2. 对齐算法（合成数据）----------
    print("\n--- 对齐算法（合成数据：已知真值 + 15%% 字符错误 + 随机插入）---")
    lines = [
        "春天的花开秋天的风以及冬天的落阳",
        "忧郁的青春年少的我曾经无知的这么想",
        "风车在四季轮回的歌里它天天的流转",
        "风花雪月的诗句里我在年年的成长",
        "流水它带走光阴的故事改变了一个人",
        "就在那多愁善感而初次等待的青春",
        "发黄的相片古老的信以及褪色的圣诞卡",
        "年轻时为你写的歌恐怕你早已忘了吧",
        "过去的誓言就像那课本里缤纷的书签",
        "刻划着多少美丽的诗可是终究是一阵烟",
    ]
    truth = [8.0 + 6.0 * i for i in range(len(lines))]
    words = []
    for i, line in enumerate(lines):
        t = truth[i]
        for ch in line:
            if random.random() < 0.15:
                ch = random.choice("错别字替代品唔嗯啊")
            words.append({"w": ch, "start": t, "end": t + 0.30})
            t += 0.35
        if random.random() < 0.30:
            words.append({"w": "呃", "start": t, "end": t + 0.20})

    try:
        segs = align_lyrics_to_audio(lines, words, total_duration=90.0,
                                     log=lambda m: None)
        check("返回行数正确", len(segs) == len(lines),
              "得到 %d 行，预期 %d" % (len(segs), len(lines)))
        check("文字用的是原文（不是识别结果）",
              [s["text"] for s in segs] == lines, "文本被篡改")
        mono = all(segs[i]["start"] < segs[i + 1]["start"] for i in range(len(segs) - 1))
        check("时间轴严格单调递增", mono, "出现逆序")
        check("每行时长均为正", all(s["end"] > s["start"] for s in segs), "存在非正时长")
        errs = [abs(segs[i]["start"] - truth[i]) for i in range(len(lines))]
        maxerr = max(errs)
        check("时间误差 <= 1.2s（实测最大 %.2fs）" % maxerr, maxerr <= 1.2,
              "最大误差 %.2fs" % maxerr)
    except Exception as e:
        check("对齐算法执行", False, "%s: %s" % (type(e).__name__, e))

    # ---------- 2b. 对抗性测试：副歌重复导致的整行拖偏 ----------
    print("\n--- 对抗性测试：某行识别失败、同句在后方副歌重复出现 ---")
    verse = ["春风吹过山岗", "夏雨落在窗前", "秋叶飘向远方"]
    shared = "冬雪覆盖大地"                      # 这句会在后面副歌里再出现一次
    body = ["第五句继续唱下去", "第六句接着来", "第七句声音渐强", "第八句收尾",
            "第九句新段落开始", "第十句慢慢推进", "第十一句做个铺垫", "第十二句"]
    lines3 = verse + [shared] + body + [shared, "重复段第二句"]
    truth3 = [6.0 * i + 5.0 for i in range(len(lines3))]
    idx3 = 3                                     # 这句的早期出现"识别失败"
    words3 = []
    for i, line in enumerate(lines3):
        t = truth3[i]
        if i == idx3:
            # 只留下完全对不上的杂字，制造"早期出现无匹配"
            for _ in range(len(line)):
                words3.append({"w": "某某", "start": t, "end": t + 0.30})
                t += 0.35
            continue
        for ch in line:
            words3.append({"w": ch, "start": t, "end": t + 0.30})
            t += 0.35
    try:
        segs3 = align_lyrics_to_audio(lines3, words3, total_duration=140.0,
                                      log=lambda m: None)
        check("对抗测试行数正确", len(segs3) == len(lines3),
              "得到 %d，预期 %d" % (len(segs3), len(lines3)))
        err3 = abs(segs3[idx3]["start"] - truth3[idx3])
        check("识别失败的那行没被拖到副歌位置（偏差 %.2fs）" % err3, err3 <= 3.0,
              "偏差 %.2fs（真值 %.1fs，实际给了 %.1fs）"
              % (err3, truth3[idx3], segs3[idx3]["start"]))
        mono3 = all(segs3[i]["start"] < segs3[i + 1]["start"] for i in range(len(segs3) - 1))
        check("对抗测试时间轴仍单调", mono3, "出现逆序")
        max3 = max(abs(segs3[i]["start"] - truth3[i]) for i in range(len(lines3)))
        check("对抗测试整体最大偏差 <= 3s（实测 %.2fs）" % max3, max3 <= 3.0,
              "最大 %.2fs" % max3)
        # 副歌那句应当仍被正确定位（它自己是有匹配的）
        idxr = len(verse) + 1 + len(body)
        errr = abs(segs3[idxr]["start"] - truth3[idxr])
        check("副歌里那句本身定位准确（偏差 %.2fs）" % errr, errr <= 2.0,
              "偏差 %.2fs" % errr)
    except Exception as e:
        check("对抗性测试执行", False, "%s: %s" % (type(e).__name__, e))

    # ---------- 3. 完全不匹配时应报错而不是乱输出 ----------
    print("\n--- 安全性 ---")
    try:
        align_lyrics_to_audio(["完全无关的歌词内容甲乙丙丁"],
                              [{"w": "zzz", "start": 1.0, "end": 1.2}],
                              total_duration=10.0, log=lambda m: None)
        check("歌词与音频完全不匹配时应报错", False, "竟然没有报错")
    except Exception:
        check("歌词与音频完全不匹配时应报错", True)

    # ---------- 3b. 简繁字形转换 ----------
    print("\n--- 简繁字形转换 ---")
    # 用字选取说明：嘅 是粤语特有字，opencc 没有对应另一写法，转换时保持原样。
    # 係/系 虽长得像但是**两个不同的字**：係(繁) 的简体对应是 系，不是 是。
    # 断言必须用真正的一简一繁对应（們/们、声/声、愛/爱、係/系），否则会误报。
    _TRAD = "我們的聲音真係好聽"
    _SIMP = "我们的声音真系好听"
    if _get_opencc("t2s"):
        check("keep 保持原样",
              apply_script(_TRAD, "keep") == _TRAD, apply_script(_TRAD, "keep"))
        check("s2t 转简体（們/聲/愛/係→们/声/爱/系）",
              apply_script(_TRAD, "s2t") == _SIMP, apply_script(_TRAD, "s2t"))
        check("t2s 转繁体（选 1:1 对应的字做往返）",
              apply_script("我们爱听声音", "t2s") == "我們愛聽聲音",
              apply_script("我们爱听声音", "t2s"))
        check("粤语字 嘅/係 不被误转（无对应写法则原样）",
              "嘅" in apply_script("你嘅聲", "s2t"), apply_script("你嘅聲", "s2t"))
        check("clean_text 按 script 透传",
              clean_text(_TRAD, script="keep") == _TRAD,
              clean_text(_TRAD, script="keep"))
        check("clean_text(script='s2t') 真的转了",
              clean_text(_TRAD, script="s2t") == _SIMP,
              clean_text(_TRAD, script="s2t"))
        # 字形选择：用户显式指定 > 语言默认 > 全局默认
        check("粤语默认保持原样",
              resolve_script("yue") == "keep", resolve_script("yue"))
        check("粤语可被用户覆盖为简体",
              resolve_script("yue", "s2t") == "s2t", resolve_script("yue", "s2t"))
        check("粤语可被用户覆盖为繁体",
              resolve_script("yue", "t2s") == "t2s", resolve_script("yue", "t2s"))
        check("普通话默认转简体（老行为不变）",
              resolve_script("zh") == "s2t", resolve_script("zh"))
        check("auto 等同于不指定",
              resolve_script("yue", "auto") == "keep", resolve_script("yue", "auto"))
        check("坏值退回语言默认而不是崩",
              resolve_script("yue", "乱写") == "keep", resolve_script("yue", "乱写"))
        _fs = filter_segments([{"start": 0.0, "end": 3.0, "text": _TRAD,
                                "no_speech": 0.0}], log=lambda m: None, script="s2t")
        check("filter_segments 透传 script", _fs and _fs[0]["text"] == _SIMP,
              _fs[0]["text"] if _fs else "空")
    else:
        print("  [跳过] opencc 不可用，无法验证简繁转换")

    # ---------- 3c. 下载文件名模板 ----------
    print("\n--- 下载文件名模板 ---")
    _info = {"title": "测试标题", "id": "abc123", "uploader": "某某UP",
             "upload_date": "20260101", "duration": 213, "resolution": "1920x1080",
             "playlist": "某歌单", "playlist_index": 3}
    check("默认模板 = 标题 + ID",
          preview_filename("", _info) == "测试标题 [abc123].mp4",
          preview_filename("", _info))
    check("自定义模板",
          preview_filename("%(title)s - %(uploader)s", _info) == "测试标题 - 某某UP.mp4",
          preview_filename("%(title)s - %(uploader)s", _info))
    check("日期与分辨率变量",
          preview_filename("%(upload_date)s_%(resolution)s", _info)
          == "20260101_1920x1080.mp4",
          preview_filename("%(upload_date)s_%(resolution)s", _info))
    check("播放列表序号",
          preview_filename("%(playlist_index)s-%(title)s", _info) == "3-测试标题.mp4",
          preview_filename("%(playlist_index)s-%(title)s", _info))
    check("{n} 队列序号", preview_filename("第{n}首", _info, index=7) == "第7首.mp4",
          preview_filename("第{n}首", _info, index=7))
    check("artist 变量等同 uploader",
          preview_filename("%(artist)s", _info) == "某某UP.mp4",
          preview_filename("%(artist)s", _info))
    check("扩展名跟着选择走（音频）",
          preview_filename("%(title)s", _info, ext="mp3") == "测试标题.mp3",
          preview_filename("%(title)s", _info, ext="mp3"))
    check("模板里写 %(ext)s 不会变成 mp4.mp4",
          preview_filename("%(ext)s", _info) == "mp4",
          preview_filename("%(ext)s", _info))
    check("模板里写 x.%(ext)s 不会被重复加扩展名",
          preview_filename("x.%(ext)s", _info) == "x.mp4",
          preview_filename("x.%(ext)s", _info))
    check("Windows 非法字符被替换",
          "/" not in preview_filename("a/b:c*d", _info)
          and ":" not in preview_filename("a/b:c*d", _info),
          preview_filename("a/b:c*d", _info))
    check("结尾的点/空格被去掉（否则 Windows 改名失败）",
          not preview_filename("名字...", _info).endswith(". .mp4"),
          preview_filename("名字...", _info))
    check("未知变量原样保留（让用户看出自己写错了）",
          "%(nope)s" in preview_filename("%(nope)s", _info),
          preview_filename("%(nope)s", _info))
    check("没有解析信息也不崩",
          isinstance(preview_filename("%(title)s", None), str),
          preview_filename("%(title)s", None))
    check("模板为空用默认，不返回空文件名",
          bool(preview_filename("", None)), preview_filename("", None))
    # 目录拼接：用户模板里带路径也不该写到别处去
    _ot = build_outtmpl("D:/x/y", "%(title)s")
    check("outtmpl 固定在下载目录内", _ot.startswith("D:/x/y") or
          _ot.startswith("D:\\x\\y"), _ot)
    check("outtmpl 空模板回落到默认",
          "title" in build_outtmpl("D:/x/y", None), build_outtmpl("D:/x/y", None))
    check("outtmpl 丢掉模板里的 ../ 穿越",
          ".." not in build_outtmpl("D:/x/y", "../../evil"), build_outtmpl("D:/x/y", "../../evil"))

    # ---------- 3b-2. 解析取消要真的立刻生效 ----------
    # 之前靠 progress_hook 收取消信号，但解 n 挑战那段一次都不回调 ——
    # 实测点取消后还要等 18 秒，等于"取消没反应"。改成 _run_with_cancel
    # 由调用方轮询标记后，0.3 秒取消实测 0.5 秒就停。
    print("\n--- 解析取消响应 ---")
    import threading as _th
    _job = _Job()
    _box = {}

    def _slow():
        _t0 = time.time()
        while time.time() - _t0 < 30:      # 模拟一个很慢的解析
            time.sleep(0.05)
        return "不该走到这"

    def _go():
        try:
            _box["v"] = _run_with_cancel(_slow, _job)
        except Exception as e:
            _box["e"] = type(e).__name__

    _t = _th.Thread(target=_go, daemon=True)
    _t.start()
    time.sleep(0.3)
    _job.cancel()
    _t.join(timeout=10)
    check("取消后立刻抛出 _Canceled",
          _box.get("e") == "_Canceled", str(_box))
    check("取消线程已退出", not _t.is_alive())
    # 不取消时应正常返回
    _job2 = _Job()
    _box2 = {}
    _t2 = _th.Thread(target=lambda: _box2.update(
        {"v": _run_with_cancel(lambda: "done", _job2)}), daemon=True)
    _t2.start()
    _t2.join(timeout=5)
    check("不取消时正常返回结果", _box2.get("v") == "done", str(_box2))
    check("yt-dlp 超时/重试已收紧（避免长时间无响应）",
          _ytdlp_opts(os.path.join(_HERE, DOWNLOAD_REL), lambda s: None)
          .get("socket_timeout", 0) <= 30
          and _ytdlp_opts(os.path.join(_HERE, DOWNLOAD_REL), lambda s: None)
          .get("retries", 99) <= 3)

    # ---------- 3c. Cookie 自动导入监视器 ----------
    print("\n--- Cookie 自动导入 ---")
    try:
        import cookiescan as _cs
        _H = "# Netscape HTTP Cookie File\n"
        _YT = ".youtube.com\tTRUE\t/\tTRUE\t1900000000\tSID\tabc\n"
        _n_ok = _cs.looks_like_cookie_name("youtube.com_cookies.txt")
        check("按文件名认出 cookie 文件", _n_ok)
        check("非 txt 不认", not _cs.looks_like_cookie_name("cookies"))
        _td = tempfile.mkdtemp(prefix="_cscan_")
        try:
            _src = os.path.join(_td, "dl")
            _dst = os.path.join(_td, "dst")
            os.makedirs(_src)
            os.makedirs(_dst)
            with io.open(os.path.join(_src, "youtube.com_cookies.txt"),
                         "w", encoding="utf-8") as _f:
                _f.write(_H + _YT)
            with io.open(os.path.join(_src, "shopping.txt"),
                         "w", encoding="utf-8") as _f:
                _f.write("a\tb\n")
            _imp = _cs.CookieImporter(_dst, extra_dirs=[_src])
            # 只扫测试目录：CookieImporter 默认还会去找真实下载目录，
            # 自检不该把用户 D:\Downloads 里的文件也卷进来。
            _got = _imp._scan_dir(_src, _dst)
            check("扫到并导入插件导出的文件", len(_got) == 1, str(_got))
            if _got:
                check("原文件不被移动/删除", os.path.isfile(_got[0][0]))
            check("同一份不会重复导入", _imp._scan_dir(_src, _dst) == [])
        finally:
            shutil.rmtree(_td, ignore_errors=True)
    except Exception as e:
        check("cookiescan 可用", False, "%s: %s" % (type(e).__name__, e))

    # ---------- 3d. Cookie 登录态识别（插件导出那条路） ----------
    # Get cookies.txt LOCALLY 之类的插件导出的是标准 Netscape 格式，能被
    # 直接用；但它默认只导**当前页面**的 cookie，在别的网站点导出就会得到
    # 一个"格式合法但没有 YouTube"的文件 —— 以前只表现为"登录失效"。
    print("\n--- Cookie 登录态识别 ---")
    _ck_good = os.path.join(tmpdir, "_ck_good.txt")
    _ck_noyt = os.path.join(tmpdir, "_ck_noyt.txt")
    _ck_pref = os.path.join(tmpdir, "_ck_pref.txt")
    _ck_empty = os.path.join(tmpdir, "_ck_empty.txt")
    try:
        _hdr = "# Netscape HTTP Cookie File\n"
        for _f, _b in (
            (_ck_good, ".youtube.com\tTRUE\t/\tTRUE\t1900000000\tSID\tabc\n"
                       ".youtube.com\tTRUE\t/\tTRUE\t1900000000\tHSID\tdef\n"),
            (_ck_noyt, "b.example.com\tFALSE\t/\tFALSE\t1900000000\tfoo\tbar\n"),
            (_ck_pref, ".youtube.com\tTRUE\t/\tFALSE\t1900000000\tCONSENT\tYES\n"),
            (_ck_empty, "# nothing\n")):
            with io.open(_f, "w", encoding="utf-8") as _fh:
                _fh.write(_hdr + _b)
        check("含登录凭据的 cookie 判定为已登录",
              youtube_auth_state(_ck_good)[0] is True,
              youtube_auth_state(_ck_good)[1])
        check("只有别的网站的 cookie 判为未登录",
              youtube_auth_state(_ck_noyt)[0] is False,
              youtube_auth_state(_ck_noyt)[1])
        check("有 YT cookie 但无凭据判为未登录",
              youtube_auth_state(_ck_pref)[0] is False,
              youtube_auth_state(_ck_pref)[1])
        check("空文件判为未登录且不崩",
              youtube_auth_state(_ck_empty)[0] is False)
        check("文件不存在时判为未登录且不崩",
              youtube_auth_state(os.path.join(tmpdir, "_no_such.txt"))[0] is False)
        # 插件导出的文件必须能被当成合法 cookie 文件收下
        check("插件导出格式被识别为合法 cookie 文件",
              _looks_like_cookie_file(_ck_good))
        check("LANG/凭据常量完整",
              set(YT_AUTH_COOKIES) >= {"SID", "HSID", "SAPISID"})
    finally:
        for _f in (_ck_good, _ck_noyt, _ck_pref, _ck_empty):
            try:
                os.remove(_f)
            except Exception:
                pass

    # ---------- 3e. 磁盘占用体检的分级规则 ----------
    print("\n--- 磁盘占用体检分级 ---")
    try:
        import dupscan as _ds
        _keep_expect = ("_internal/ctranslate2.dll", ".deps/cv2/cv2.pyd",
                        "models/large-v3/model.bin", "runtime/ffmpeg.exe",
                        ".deps-ocr/x/y.onnx", ".git/config")
        _safe_expect = ("__pycache__/a.pyc", ".tmp/b.bin", "build/c",
                        "dist/d/e", "f.log", "g.pyc")
        _rev_expect = ("lyric_maker.py", "start.bat", "README.md")
        _bad = [p for p in _keep_expect
                if _ds.classify(os.path.join(_HERE, p), _HERE) != "keep"]
        check("依赖/模型/运行时判为保留", not _bad, str(_bad))
        _bad2 = [p for p in _safe_expect
                 if _ds.classify(os.path.join(_HERE, p), _HERE) != "safe"]
        check("缓存与中间产物判为可清理", not _bad2, str(_bad2))
        _bad3 = [p for p in _rev_expect
                 if _ds.classify(os.path.join(_HERE, p), _HERE) != "review"]
        check("业务文件判为需确认", not _bad3, str(_bad3))
        check("human() 可用", _ds.human(3 * 1024 * 1024).endswith("MB"),
              _ds.human(3 * 1024 * 1024))
    except Exception as e:
        check("dupscan 可用", False, "%s: %s" % (type(e).__name__, e))

    # ---------- 4. LRC 格式 ----------
    print("\n--- LRC 输出格式 ---")
    p = os.path.join(tmpdir, "_selftest_out.lrc")
    try:
        write_lrc(p, [{"start": 12.345, "end": 15.0, "text": "测试行", "no_speech": 0.0}],
                  tags={"ti": "标题", "ar": "歌手", "length": 200.0})
        body = io.open(p, encoding="utf-8-sig").read()
        check("含 ti/ar 标签", "[ti:标题]" in body and "[ar:歌手]" in body, body[:80])
        check("时间戳格式为 [mm:ss.xx]",
              re.search(r"\[00:12\.3[45]\]测试行", body) is not None, body)
        check("输出为 UTF-8 BOM（兼容旧播放器）",
              open(p, "rb").read(3) == b"\xef\xbb\xbf", "缺 BOM")
    except Exception as e:
        check("LRC 写出", False, str(e))
    finally:
        try:
            os.remove(p)
        except Exception:
            pass

    # ---------- 4b. 多格式输出 ----------
    print("\n--- 多格式输出（srt / vtt / ass / ssa）---")
    demo = [{"start": 1.5, "end": 4.25, "text": "第一句"},
            {"start": 5.0, "end": 8.75, "text": "第二句"}]
    for fmt_name, expect, enc, want_bom in [
        ("srt", "00:00:01,500 --> 00:00:04,250", "utf-8-sig", True),
        ("vtt", "00:00:01.500 --> 00:00:04.250", "utf-8", False),
        ("ass", "Dialogue: 0,0:00:01.50,0:00:04.25,Default,,0,0,0,,第一句", "utf-8-sig", True),
        ("ssa", "Dialogue: 0,0:00:01.50,0:00:04.25,Default,,0,0,0,,第一句", "utf-8-sig", True),
    ]:
        p = os.path.join(tmpdir, "_selftest_out." + fmt_name)
        try:
            write_subtitle_output(p, demo, fmt=fmt_name)
            raw = open(p, "rb").read()
            body = raw.decode(enc)
            check("写出 %s 且时间戳正确" % fmt_name, expect in body, body[:120])
            check("  %s BOM 策略正确" % fmt_name,
                  (raw[:3] == b"\xef\xbb\xbf") == want_bom, "BOM 不符")
        except Exception as e:
            check("写出 %s" % fmt_name, False, str(e))
        finally:
            try:
                os.remove(p)
            except Exception:
                pass

    # ---------- 4c. 格式转换：命令构造 ----------
    print("\n--- 格式转换（命令构造）---")
    _ff = None
    try:
        _ff = find_ffmpeg()
    except Exception:
        pass
    check("格式表覆盖常用的视频/音频容器", len(CONVERT_FORMATS) >= 15,
          "只有 %d 个" % len(CONVERT_FORMATS))
    for key, _disp, kind, ext, _vc, _ac in CONVERT_FORMATS:
        try:
            _c_enc = build_convert_cmd("ffmpeg", "in.mkv", "out." + ext,
                                       key, "23", "192k", False)
            _c_rmx = build_convert_cmd("ffmpeg", "in.mkv", "out." + ext,
                                       key, "23", "192k", True)
        except Exception as e:
            check("构造 %s 命令" % key, False, str(e))
            continue
        check("构造 %s 命令（重编码 + 转封装都以输出文件结尾）" % key,
              _c_enc[-1] == "out." + ext and _c_rmx[-1] == "out." + ext
              and _c_rmx[-2] == "copy", str(_c_rmx))
        if kind == "audio":
            # 音频目标必须丢画面，否则会把视频轨塞进 mp3 这种容器
            check("  %s 转音频时丢掉画面" % key,
                  "-vn" in _c_enc and "-vn" in _c_rmx, str(_c_enc))
        else:
            check("  %s 重编码时带视频编码器" % key, "-c:v" in _c_enc, str(_c_enc))
    check("进度解析 time=HH:MM:SS.ms",
          abs(_ffmpeg_seconds("time=00:01:30.50") - 90.5) < 1e-6,
          str(_ffmpeg_seconds("time=00:01:30.50")))
    check("进度解析 out_time_ms=",
          abs(_ffmpeg_seconds("out_time_ms=90500") - 90.5) < 1e-6,
          str(_ffmpeg_seconds("out_time_ms=90500")))
    check("没有进度信息时返回 0", _ffmpeg_seconds("frame=   12 fps=3") == 0.0, "")

    # ---------- 4c-2. 带封面的音频文件（真实用户踩过的坑）----------
    # mp3/m4a 里常嵌一张 png 封面，ffmpeg 报成 "Video: png (attached pic)"。
    # 若不排除，转成视频格式时会给封面也套上 -c:v libx264，写进 mp4 就报
    # "Could not find tag for codec h264" -> 退出码 -22，整个转换失败。
    print("\n--- 带封面的音频（封面流处理）---")
    _cover = os.path.join(tmpdir, "_selftest_cover.mp3")
    _made_cover = False
    if _ff:
        try:
            # 造一个"音频 + 封面"的 mp3：把一张图挂成 attached pic
            _okc, _whyc = run_ffmpeg(
                [_ff, "-y", "-hide_banner", "-loglevel", "error",
                 "-f", "lavfi", "-i", "color=c=red:s=320x240:d=1",
                 "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                 "-map", "0:v", "-map", "1:a",
                 "-c:v", "mjpeg", "-c:a", "libmp3lame", "-id3v2_version", "3",
                 "-metadata:s:v", "title=Album cover",
                 "-metadata:s:v", "comment=Cover (front)", _cover])
            _made_cover = _okc and os.path.isfile(_cover)
        except Exception:
            _made_cover = False
    if not _made_cover:
        print("  [跳过] 没造出带封面的测试音频")
    else:
        _st = probe_streams(_ff, _cover)
        check("探到音频+封面两条流", len(_st) >= 2, str(_st))
        check("封面被标记为 attached_pic",
              any(s["attached_pic"] for s in _st), str(_st))
        check("真实画面数排除封面后为 0",
              len(real_video_streams(_st)) == 0, str(_st))
        _w, _h, _ = probe_media_info(_ff, _cover)
        check("尺寸不误取封面图", not (_w == 320 and _h == 240),
              "%sx%s" % (_w, _h))
        _hv = bool(real_video_streams(_st))
        _dst = os.path.join(tmpdir, "_selftest_cover2mp4.mp4")
        _cmd = build_convert_cmd(_ff, _cover, _dst, "mp4", "28", "128k", False,
                                 has_video=_hv)
        _i = _cmd.index("-i")
        _cmd2 = _cmd[:_i + 2] + ["-t", "2"] + _cmd[_i + 2:]
        _okv, _whyv = run_ffmpeg(_cmd2, log=lambda m: None)
        check("带封面音频 -> mp4 成功（黑底承载）",
              _okv and os.path.isfile(_dst) and os.path.getsize(_dst) > 0, _whyv)
        check("转视频时用了 lavfi 造画面",
              "lavfi" in _cmd and "-shortest" in _cmd, str(_cmd[:6]))
        # 音频目标仍应走 -vn
        _dst2 = os.path.join(tmpdir, "_selftest_cover2m4a.m4a")
        _cmd3 = build_convert_cmd(_ff, _cover, _dst2, "m4a", "28", "128k", False,
                                  has_video=False)
        check("音频目标仍用 -vn", "-vn" in _cmd3, str(_cmd3[:8]))
        try:
            os.remove(_cover)
        except Exception:
            pass

    # models 页的 on_event：原来 else 分支直接 `name, rc, reason = payload`，
    # 队列里混进结构不同的事件就 ValueError 崩掉整页。事件来自后台线程，
    # 结构不受本页控制 —— 这里用一个假的 PageModels 实例验证不崩。
    _pmodels = None
    if _TK_OK:
        try:
            import tkinter as _tk2
            _r2 = _tk2.Tk()
            _r2.withdraw()
            _init_fonts()
            init_ui_scale(_r2)
            class _FakeApp2(object):
                settings = dict(DEFAULT_SETTINGS)
                q = __import__('queue').Queue()
                def refresh_storage(self):
                    pass
            _pmodels = PageModels(_tk2.Frame(_r2), _FakeApp2())
            try:
                _pmodels.on_event("__bogus__", None)
                check("models.on_event 对异常 payload 不崩", True)
            except Exception as _e:
                check("models.on_event 对异常 payload 不崩", False,
                      "%s: %s" % (type(_e).__name__, _e))
            _r2.destroy()
        except Exception as _e:
            print("  [跳过] 建不了页面：%s" % _e)

    # 短链还原：抖音分享的 v.douyin.com/xxx 里没有视频 ID，而 DouyinIE 只认
    # www.douyin.com/video/<数字> —— 不展开就 Unsupported URL。
    print("\n--- 短链还原 ---")
    check("已知短链域名会被识别",
          _host_is_short('v.douyin.com') and _host_is_short('b23.tv'))
    check("普通域名不算短链",
          (not _host_is_short('www.youtube.com'))
          and (not _host_is_short('www.bilibili.com')))
    # 关键：非短链必须原样返回（不做无谓的网络往返）
    for _u in ('https://www.youtube.com/watch?v=bWKOw9GauIs',
               'https://www.douyin.com/video/7505847520262720787'):
        check("非短链原样返回(%s)" % _u[:34], expand_short_url(_u) == _u)
    # 展开失败时不能返回空/抛异常，否则会把可用链接弄丢
    _bad = expand_short_url('https://v.douyin.com/NO_SUCH_TEST_123/')
    check("展开失败时返回非空且不抛异常", bool(_bad))
    check("空输入不崩", expand_short_url('') == '')
    check("None 输入不崩", expand_short_url(None) == '')

    # ---------- 4d-1. 仅音频下载必须真的只有音频 ----------
    # 抖音那类源的第一条流就是视频（download_addr-0 是音视频合流 mp4）。
    # 不显式 -map 0:a:0 时 ffmpeg 按默认规则映射，会把视频流一起写进 .m4a，
    # 文件名是 m4a、内容却不纯，很多播放器直接判"文件损坏"。
    print("\n--- 仅音频提取的流选择 ---")
    if _ff:
        _av = os.path.join(tmpdir, "_selftest_av.mp4")
        _okav, _wav = run_ffmpeg(
            [_ff, "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=10",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
             "-c:v", "libx264", "-c:a", "aac", "-shortest", _av])
        if _okav:
            _dstm4a = os.path.join(tmpdir, "_selftest_pure.m4a")
            _okx, _wx = run_ffmpeg(
                [_ff, "-y", "-hide_banner", "-loglevel", "error",
                 "-i", _av, "-map", "0:a:0", "-vn",
                 "-c:a", "aac", "-b:a", "192k", _dstm4a])
            _hasv = False
            if _okx and os.path.isfile(_dstm4a):
                _pp = subprocess.run([_ff, "-hide_banner", "-i", _dstm4a],
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE)
                _hasv = "Video:" in ((_pp.stderr or b"").decode("utf-8", "replace"))
            check("音视频源 -> m4a 只留音频流",
                  _okx and not _hasv, _wx)
            try:
                os.remove(_av)
            except Exception:
                pass
        else:
            print("  [跳过] 没造出测试源")
    # 上面已用真实 ffmpeg 验证了「音视频源 -> m4a 只留音频」这条路径。
    # 这里再确认「仅音频」转换选项也带 -vn（它走 build_convert_cmd，不走 yt-dlp）。
    _cmdvn = build_convert_cmd(
        _ff or "ffmpeg", "in.mp4", "out.mp3", "mp3", "23", "128k", False,
        has_video=True) if _ff else []
    check("仅音频转换选项带 -vn",
          "-vn" in _cmdvn, str(_cmdvn[:6]))

    # ---------- 4d-1. 文件名模板必须带扩展名 ----------
    # 抖音/西瓜这类站点给的是**已合流的单文件 mp4**，yt-dlp 不需要 ffmpeg 合并；
    # 而**只有合并那一步才会补扩展名**。模板里没有 %(ext)s 时，产物就是
    # 一个彻底没有后缀的文件，双击打不开、播放器不认（YouTube/B站因为要合并
    # 反而没事，所以只有抖音这类会暴露问题）。
    print("\n--- 文件名模板的扩展名 ---")
    check("默认模板含 %(ext)s", "%(ext)s" in DEFAULT_OUTTMPL)
    check("空模板走默认且含 ext",
          "%(ext)s" in build_outtmpl("D:/x", None))
    check("只写 %(title)s 时自动补 ext（修抖音无后缀）",
          build_outtmpl("D:/x", "%(title)s").endswith("%(title)s.%(ext)s"),
          build_outtmpl("D:/x", "%(title)s"))
    check("已含 ext 不会重复补",
          build_outtmpl("D:/x", "%(title)s.%(ext)s").count("%(ext)s") == 1,
          build_outtmpl("D:/x", "%(title)s.%(ext)s"))
    check("以点结尾的模板不再补",
          build_outtmpl("D:/x", "%(title)s.").count("%(ext)s") == 0,
          build_outtmpl("D:/x", "%(title)s."))
    check("补 ext 后目录部分不变",
          build_outtmpl("D:/x", "%(title)s").startswith("D:/x"),
          build_outtmpl("D:/x", "%(title)s"))
    # 预览端也要能显示扩展名，否则界面会骗用户
    check("预览能反映最终文件名带扩展名",
          preview_filename("%(title)s", {"title": "T", "id": "1"}).endswith(".mp4"),
          preview_filename("%(title)s", {"title": "T", "id": "1"}))

    # ---------- 4d-2. 下载页的输出格式选项 ----------
    # 下载页的"输出格式"和格式转换页共用同一批格式，所以它俩必须指向
    # 同一张表 —— 这里锁住这条约定，防止以后只改一边导致两边行为不一致。
    print("\n--- 下载输出格式 ---")
    _ckeys = [c[0] for c in CONTAINER_CHOICES]
    check("下载页含原始格式(不转码)", "auto" in _ckeys)
    check("下载页 key 不重复", len(_ckeys) == len(set(_ckeys)))
    check("每个下载选项都有显示名", all(c[1] for c in CONTAINER_CHOICES))
    _want = ("mp4", "mkv", "avi", "webm", "mov", "flv", "ts", "hevc",
             "vp9", "av1", "aonly-mp3", "aonly-flac", "aonly-opus")
    check("常用视频+音频格式都在下拉里",
          all(k in _ckeys for k in _want),
          str([k for k in _want if k not in _ckeys]))
    check("仅音频 key 都能映射到格式表",
          all(v in CONVERT_FORMAT_MAP for v in DL_AUDIO_KEYS.values()))
    # ProRes 必须显式给 10bit 4:2:2，否则 ffmpeg 报 "need YUV422P10 input"
    check("ProRes 指定了 yuv422p10le 像素格式",
          "yuv422p10le" in " ".join(_video_encode_args("prores", "23")))

    # ---------- 4d. 格式转换：真跑一遍 ffmpeg ----------
    print("\n--- 格式转换（实跑 ffmpeg）---")
    _src = os.path.join(tmpdir, "_selftest_src.mp4")
    if not _ff:
        print("  [跳过] 没找到 ffmpeg，跳过实跑用例")
        _src = None
    else:
        _ok0, _why0 = run_ffmpeg(
            [_ff, "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=10",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
             "-c:v", "libx264", "-c:a", "aac", "-shortest", _src])
        if not (_ok0 and os.path.exists(_src) and os.path.getsize(_src) > 0):
            # 素材都造不出来（缺 lavfi / 缺编码器）时不要算作失败，
            # 只跳过 —— 这是环境差异，不是代码错误。
            print("  [跳过] 造不出测试素材（%s），跳过实跑用例" % _why0)
            _src = None
        else:
            check("造出测试素材（1 秒 带画面的 mp4）", True)
    if _src:
        for key in ("mp3", "m4a", "wav", "flac", "mkv", "mov", "avi", "webm"):
            _spec = CONVERT_FORMAT_MAP[key]
            _dst = os.path.join(tmpdir, "_selftest_conv." + _spec[3])
            _okc, _whyc = run_ffmpeg(
                build_convert_cmd(_ff, _src, _dst, key, "28", "128k", False))
            check("转 %s" % key,
                  _okc and os.path.exists(_dst) and os.path.getsize(_dst) > 0,
                  _whyc)
            try:
                os.remove(_dst)
            except Exception:
                pass
        _dst = os.path.join(tmpdir, "_selftest_remux.mkv")
        _okc, _whyc = run_ffmpeg(
            build_convert_cmd(_ff, _src, _dst, "mkv", "28", "128k", True))
        check("mp4 → mkv 仅转封装（不重编码）",
              _okc and os.path.exists(_dst) and os.path.getsize(_dst) > 0, _whyc)
        try:
            os.remove(_dst)
        except Exception:
            pass
        # ProRes：必须实跑。缺了 yuv422p10le 会在转码阶段才炸，
        # 而且报错信息（"need YUV422P10 input"）对用户毫无指向性。
        _src2 = os.path.join(tmpdir, "_selftest_prores_src.mp4")
        _okp, _whyp = run_ffmpeg(
            [_ff, "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "testsrc=duration=1:size=160x120:rate=10",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
             "-c:v", "libx264", "-c:a", "aac", "-shortest", _src2])
        if _okp:
            _dstp = os.path.join(tmpdir, "_selftest_prores.mov")
            _okr, _whyr = run_ffmpeg(build_convert_cmd(
                _ff, _src2, _dstp, "prores", "28", "128k", False))
            check("转 ProRes 成功（需 10bit 4:2:2 像素格式）",
                  _okr and os.path.exists(_dstp) and os.path.getsize(_dstp) > 0,
                  _whyr)
            try:
                os.remove(_src2)
            except Exception:
                pass
        else:
            print("  [跳过] 没造出测试源，ProRes 实跑用例跳过")

        # 失败路径：输入不存在时要返回 (False, 原因)，而不是抛异常把界面搞崩
        _okc, _whyc = run_ffmpeg(build_convert_cmd(
            _ff, os.path.join(tmpdir, "_selftest_nope.mp4"),
            os.path.join(tmpdir, "_selftest_nope.mp3"), "mp3", "28", "128k", False))
        check("输入不存在时返回失败原因而不崩", (not _okc) and bool(_whyc), str(_whyc))
        try:
            os.remove(_src)
        except Exception:
            pass

    print("\n=== 自检结果: %d 通过, %d 失败 ===" % (passed[0], failed[0]))
    return 0 if failed[0] == 0 else 1
# =====================================================================
# CLI
# =====================================================================
def _one_line(text, limit=90):
    """把多行文本压成一行，给底部那行单行提示用（超长加省略号）。"""
    s = " ".join(str(text or "").split())
    return s[:limit] + ("…" if len(s) > limit else "")


def _dedup_tail(lines, limit=6):
    """把 yt-dlp 的日志尾部压成几行 —— 同一条错误它常常连着刷好几遍。

    原样贴出来就是一面墙（实测 Choome cookie 失败连刷 4 遍 8 行 URL），
    用户反而看不到真正有用的那行。
    """
    out = []
    for ln in lines:
        s = str(ln).rstrip()
        if not s:
            continue
        if s not in out:
            out.append(s)
    return "\n".join(out[-limit:])


def _short_path(p, limit=46):
    """路径太长时保留尾部（路径的关键信息在末尾），前面加省略号。"""
    s = str(p or "")
    return s if len(s) <= limit else ("…" + s[-(limit - 1):])


def _alert(title, message, error=False):
    """用弹窗把结果或错误告诉用户。

    打包成无控制台的 exe 之后根本没有 stdout，双击或拖放运行时用户什么都看不到，
    只能靠弹窗。优先走 Windows 原生对话框（不依赖 tkinter 初始化，更省事更稳），
    失败再退回 tkinter；两条路都不通就只留一行日志，绝不让提示本身把程序搞崩。
    """
    title, message = str(title), str(message)
    print("[弹窗] %s: %s" % (title, message.replace("\n", " ")))
    try:
        import ctypes
        # MB_ICONERROR = 0x10, MB_ICONINFORMATION = 0x40
        flags = 0x10 if error else 0x40
        ctypes.windll.user32.MessageBoxW(None, message, title, flags)
        return
    except Exception:
        pass
    try:
        import tkinter as tk
        from tkinter import messagebox
    except Exception:
        return
    try:
        root = tk.Tk()
        root.withdraw()
        (messagebox.showerror if error else messagebox.showinfo)(title, message)
        root.destroy()
    except Exception:
        pass


def download_diagnose(url):
    """`--dl-test URL`：把下载链路每一环的状态打出来，便于定位卡在哪。

    视频下载涉及的东西比看起来多（yt-dlp、JS 运行时、求解器、Cookie、代理），
    任何一环缺了都会表现成"下不动"，但提示语完全不同。这个命令一次全打出来。
    """
    print("=== 视频下载自检 ===")
    print("yt-dlp        : %s" % (ytdlp_version() or "未内置"))
    # 自带件优先显示，让"拷走即用"这件事一眼可见
    js, jsp = find_js_runtime()
    bundled = bundled_runtime("node.exe")
    print("JS 运行时     : %s" % (("%s -> %s" % (js, jsp)) if js else
                                 "未找到（YouTube 可能只给低清格式）"))
    if bundled and os.path.abspath(jsp).lower() == os.path.abspath(bundled).lower():
        print("              : ↑ 随程序携带，换电脑也能用")
    if _bundled_ytdlp_exe():
        print("自带 yt-dlp   : %s" % _bundled_ytdlp_exe())
    try:
        import yt_dlp_ejs
        print("求解器 ejs    : %s" % os.path.dirname(yt_dlp_ejs.__file__))
    except Exception as e:
        print("求解器 ejs    : 未安装（%s）" % e)
    cf = cookie_txt()
    print("Cookie 文件   : %s" % (cf if cf else "无（%s 里没有 cookies.txt）"
                                 % cookie_dir()))
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "NODE_OPTIONS"):
        v = os.environ.get(name)
        if v:
            print("环境变量      : %s=%s%s" % (name, v[:60],
                                              "  ← 会干扰 yt-dlp" if name == "NODE_OPTIONS" else ""))
    if not url:
        return 0
    print("\n=== 解析 %s ===" % url)
    logs = []
    t0 = time.time()
    ok, info = ytdlp_probe(url, log=lambda m: logs.append(str(m)))
    print("耗时 %.1fs  ok=%s" % (time.time() - t0, ok))
    if ok:
        print("标题   :", info.get("title"))
        print("时长   :", info.get("duration"))
        print("清晰度 :", ytdlp_heights(info))
        print("可选档位:", [k for k, _ in ytdlp_format_choices(info)])
    else:
        print("原因   :", info)
    if logs:
        print("--- yt-dlp 日志 ---")
        for l in logs[-8:]:
            print("  " + str(l)[:170])
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# 失败 / 诊断落盘：界面报错也写进 exe 旁的 lyric-maker.log。
# 之前用户只能看到红字、日志却是空的，排障白白绕一圈——根因是 GUI 日志只进
# 文本框、从不直接落盘。这里统一补一道，兼容两种运行形态：
#   · 双击 exe（无控制台）：main() 已把 sys.stdout 重定向到该日志文件，print 即落盘；
#   · 终端 / 调试（有控制台）：额外开 append 句柄落盘，并保留终端输出便于看。
# ---------------------------------------------------------------------------
_LOGFILE_PATH = os.path.join(_HERE, "lyric-maker.log")


def logfile_write(msg):
    line = str(msg)
    wrote = False
    try:
        so = sys.stdout
        if so is not None and getattr(so, "name", "") == _LOGFILE_PATH:
            print(line)
            try:
                so.flush()
            except Exception:
                pass
            wrote = True
    except Exception:
        pass
    if not wrote:
        try:
            with io.open(_LOGFILE_PATH, "a", encoding="utf-8", errors="replace") as fh:
                fh.write(line + "\n")
        except Exception:
            pass
        try:
            if sys.stdout is not None:
                print(line)
        except Exception:
            pass


def main(argv=None):
    # 2K/4K：必须在创建任何窗口之前声明 DPI 感知，否则 Windows 会把整个窗口画成
    # 逻辑分辨率的小位图再放大，文字和描边全糊。
    enable_dpi_awareness()
    # 分两种情况处理标准输出：
    #
    # 无控制台的构建（spec 里 console=False）：sys.stdout / stderr 是 None，
    # 任何 print 都会直接抛异常。把输出落到 exe 旁边的日志文件是两全的做法 ——
    # 既兜住了 print，界面模式下用户也不会被打扰，出问题时还有线索可翻。
    #
    # 有控制台时：把输出固定成 UTF-8。Windows 上输出一旦被重定向到管道或文件，
    # 默认会退回 OEM 代码页（本机 936），中文全变乱码。
    # 这行只影响控制台输出，不改动写出的字幕文件（一直是 UTF-8）。
    no_console = sys.stdout is None
    if no_console:
        _log = None
        try:
            _log = io.open(os.path.join(_HERE, "lyric-maker.log"), "w",
                           encoding="utf-8", errors="replace")
        except Exception:
            pass
        if _log is not None:
            sys.stdout = sys.stderr = _log
        else:
            for _name in ("stdout", "stderr"):
                setattr(sys, _name, io.StringIO())
    else:
        for _stream in (sys.stdout, sys.stderr):
            try:
                _stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    ap = argparse.ArgumentParser(
        description="从音乐/视频生成歌词与字幕（lrc/srt/ass/ssa/vtt，本地处理，无需联网）")
    ap.add_argument("media", nargs="?", help="音乐或视频文件路径")
    ap.add_argument("--model", default="medium",
                    help="模型名(%s)或模型目录，默认 medium" % "/".join(MODEL_REPOS))
    ap.add_argument("--language", default="zh", help="语言，默认 zh；auto 为自动")
    ap.add_argument("--script", default=None, choices=list(SCRIPT_MODES) + ["auto"],
                    help="输出字形：keep 保持识别原样（粤语默认）/ s2t 转简体 / "
                         "t2s 转繁体 / auto 按语言默认。不给则按语言默认，"
                         "非粤语语言默认转简体")
    ap.add_argument("--out", default=None, help="输出路径，默认与输入同名")
    ap.add_argument("--format", "--fmt", dest="format", default=None,
                    choices=["lrc", "srt", "ass", "ssa", "vtt"],
                    help="输出格式，默认 lrc。不给的话也能用 --out 的扩展名指定，"
                         "比如 --out 字幕.srt 会自动输出 srt")
    ap.add_argument("--prompt", default=None, help="初始提示词，用于引导用词与标点")
    ap.add_argument("--lyrics-file", default=None,
                    help="已知歌词文本(.txt/.lrc)。给出后用【对齐模式】：文字用你的原文，"
                         "识别只提供时间锚点 —— 这是出高质量歌词的推荐方式。"
                         "若传入的是 .srt/.ass/.vtt/.lrc 等带时间戳的字幕，则直接用其时间轴")
    ap.add_argument("--subs", default="auto", choices=["off", "track", "ocr", "auto"],
                    help="字幕模式(默认 auto)：off 不用字幕；track 只用现成字幕"
                         "(内嵌字幕轨或同目录外挂字幕)；ocr 只识别画面上的硬字幕；"
                         "auto 先找现成字幕，没有再 OCR")
    ap.add_argument("--subs-file", default=None,
                    help="指定字幕文件(.srt/.ass/.ssa/.vtt/.lrc)")
    ap.add_argument("--subs-region", default="auto",
                    help="硬字幕所在画面区域：auto(默认，自动探测字幕带) / bottom / top / "
                         "0.65,0.95(画面高度比例)")
    ap.add_argument("--subs-fps", type=float, default=2.0,
                    help="硬字幕粗扫帧率，默认 2；时间精度不够可调到 3~4（更慢）")
    ap.add_argument("--subs-no-refine", action="store_true",
                    help="关闭硬字幕起点细化（快一些，但行起始时间精度降到 1/subs-fps 秒）")
    ap.add_argument("--subs-min-score", type=float, default=0.5,
                    help="OCR 置信度阈值，默认 0.5；误识别多就调高")
    ap.add_argument("--vad", action="store_true",
                    help="开启语音活动检测（纯人声/视频建议开；歌曲建议关，默认关）")
    ap.add_argument("--threads", type=int, default=None, help="CPU 线程数，默认 min(16,核心数)")
    ap.add_argument("--start", type=float, default=None, help="只处理从第 N 秒开始")
    ap.add_argument("--duration", type=float, default=None, help="只处理 N 秒")
    ap.add_argument("--offset", type=int, default=0, help="时间轴整体平移(毫秒)")
    ap.add_argument("--no-split", action="store_true", help="不按标点拆分过长行")
    ap.add_argument("--no-merge", action="store_true", help="不合并过短的相邻段")
    ap.add_argument("--ffmpeg", default=None, help="ffmpeg 路径")
    ap.add_argument("--keep-wav", action="store_true", help="保留中间 WAV 便于排查")
    ap.add_argument("--gui", action="store_true", help="启动图形界面")
    ap.add_argument("--selftest", action="store_true", help="运行内置自检（不需要音频）")
    ap.add_argument("--get-model", nargs="?", const="", default=None, metavar="NAME",
                    help="下载模型：不带名字则列出全部与状态，给名字则下载"
                         "（tiny/base/small/medium/turbo/large-v3）")
    ap.add_argument("--source", choices=["auto", "modelscope", "hf"], default="auto",
                    help="--get-model 使用的下载源，默认 auto")
    ap.add_argument("--force", action="store_true", help="--get-model 时已存在也重新下载")
    ap.add_argument("--gui-check", action="store_true",
                    help="只构建一遍界面控件然后退出（不需要人看着）")
    ap.add_argument("--gui-capture", default=None, metavar="PNG",
                    help="构建界面、截图存到 PNG 然后退出")
    ap.add_argument("--dl-test", default=None, metavar="URL",
                    help="诊断视频下载：打印 yt-dlp / JS 运行时 / Cookie 状态，"
                         "并尝试解析这个链接")
    args = ap.parse_args(argv)

    if args.dl_test is not None:
        return download_diagnose(args.dl_test)

    if args.selftest:
        return selftest()

    if args.get_model is not None:
        if not args.get_model:
            show_list()
            return 0
        return download(args.get_model, force=args.force, source=args.source)

    if args.gui_check or args.gui_capture:
        return launch(check_only=args.gui_check or bool(args.gui_capture),
                      capture_png=args.gui_capture)

    if args.gui or not args.media:
        return run_gui()

    try:
        res = transcribe(args.media, model_name=args.model, language=args.language,
                         out=args.out, prompt=args.prompt, vad=args.vad,
                         threads=args.threads, start=args.start, duration=args.duration,
                         offset_ms=args.offset,
                         split_long_lines=not args.no_split, merge=not args.no_merge,
                         ffmpeg=args.ffmpeg, keep_wav=args.keep_wav,
                         lyrics_file=args.lyrics_file,
                         subs_mode=args.subs, subs_region=args.subs_region,
                         subs_fps=args.subs_fps, subs_refine=not args.subs_no_refine,
                         subs_file=args.subs_file, subs_min_score=args.subs_min_score,
                         fmt=args.format, script=args.script)
        if no_console and res:
            src = res.get("source")
            extra = ("\n来源：%s" % src) if src else ""
            _alert("完成", "已生成：\n%s\n\n共 %d 行%s"
                   % (res.get("out"), len(res.get("segments") or []), extra))
        return 0
    except Exception as e:
        print("\n[错误] %s" % e, file=sys.stderr)
        if no_console:
            _alert("出错了", str(e), error=True)
        return 1


# =====================================================================
# 图形界面入口（launch 的实现在本文件末尾）
# =====================================================================
def run_gui():
    return launch()



# =====================================================================
# 字幕源：内嵌字幕轨提取 / 字幕文件解析 / 画面硬字幕 OCR
# =====================================================================

SUB_TEXT_EXT = {".srt", ".ass", ".ssa", ".vtt", ".lrc"}

# 图像字幕（dvd/pgs）是位图，ffmpeg 转不出文本，只能走 OCR
IMAGE_SUB_CODECS = {
    "dvd_subtitle", "dvb_subtitle", "hdmv_pgs_subtitle", "xsub",
    "dvb_graphics_subtitle", "bluray_subtitle",
}

VIDEO_SIZE_RE = re.compile(r"Video:\s+.*?(\d{2,5})x(\d{2,5})")
DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):([\d.]+)")
# 例：Stream #0:2(chi): Subtitle: subrip
SUB_STREAM_RE = re.compile(
    r"Stream #(\d+):(\d+)(?:\(([A-Za-z\-]{2,12})\))?:\s*Subtitle:\s*([A-Za-z0-9_\-]+)")

LRC_TS_RE = re.compile(r"\[(\d{1,3}):(\d{1,2})(?:[.:](\d{1,3}))?\]")


# ---------------------------------------------------------------- 通用工具

def _decode(data):
    """字幕文件编码五花八门，按最可能的顺序试。"""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        try:
            return data.decode("utf-16")
        except Exception:
            pass
    if data[:3] == b"\xef\xbb\xbf":
        return data.decode("utf-8-sig")
    try:
        t = data.decode("utf-8")
        if "\ufffd" not in t:
            return t
    except Exception:
        pass
    for enc in ("gbk", "big5", "utf-16", "latin-1"):
        try:
            return data.decode(enc)
        except Exception:
            continue
    return data.decode("utf-8", "replace")


def norm_txt(s):
    """比较两段字幕是否"同一句"用的归一化：去掉空白和标点。"""
    return re.sub(r"[\s，。、！？；：,.!?;:~—\-…·\"'“”‘’()（）\[\]【】]+", "", s or "")


def similar(a, b):
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def plausible_subtitle_text(text):
    """判断一条 OCR 结果像不像"一句字幕"。

    画面上除了字幕还有水印、台标、印章、数值、装饰字，OCR 一律照读。
    它们通常是"很短 / 不含中文 / 单个汉字混字母"的碎片。实测漏出来的噪声长这样：
    `LY`、`2 1`、`公 M`、`7 M` —— 全部在这几条规则下被丢掉。

    代价是极短的歌词（单独一个"啊""嘿"）也会被滤掉。这个取舍是刻意的：
    往 lrc 里塞一堆水印碎片，比漏掉一个语气词的破坏力大得多。
    """
    t = norm_txt(text)
    if len(t) < 2:
        return False
    cjk = len(re.findall(r"[\u4e00-\u9fff\u3400-\u4dbf]", t))
    if cjk == 0 and len(t) < 6:      # 纯字母/数字且很短 -> 水印或画面上的数值
        return False
    if cjk == 1 and len(t) < 3:      # 单个汉字混字母 -> 碎屑
        return False
    if len(set(t)) <= 1:             # 同一个字符堆叠
        return False
    return True


def probe_streams(ffmpeg, media):
    """返回 [{"kind":"video"/"audio"/"other", "codec":str, "attached_pic":bool, "index":int}]。

    为什么要专门探：带封面的 mp3/m4a 里有**两条**流 —— 音频 + 一张 png 封面
    （ffmpeg 报成 "Video: png ... (attached pic)"）。它不是真正的画面，
    但 ffmpeg 默认会把所有流都带上，于是给封面也套上 -c:v libx264，
    再写进 mp4 就炸："Could not find tag for codec h264 ... not supported in
    container" -> 退出码 -22。实测用户转罗大佑的 mp3 就是这么失败的。
    """
    p = subprocess.run([ffmpeg, "-hide_banner", "-i", media],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       creationflags=_no_window())
    out = (p.stderr or b"").decode("utf-8", "replace")
    streams = []
    for m in re.finditer(r"Stream #\d+:\d+.*?:\s*(Video|Audio|Subtitle|Data):\s*([A-Za-z0-9_]+)",
                         out):
        kind = m.group(1).lower()
        line = out[m.start():out.find("\n", m.start()) if "\n" in out[m.start():] else len(out)]
        streams.append({
            "kind": kind,
            "codec": m.group(2),
            "attached_pic": "attached pic" in line,
            "index": len(streams),
        })
    return streams


def real_video_streams(streams):
    """筛掉封面图，只留真正的画面流。"""
    return [s for s in streams if s["kind"] == "video" and not s["attached_pic"]]


def probe_media_info(ffmpeg, media):
    """返回 (width, height, duration_sec)。解析失败用 None 填充。"""
    p = subprocess.run([ffmpeg, "-hide_banner", "-i", media],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       creationflags=_no_window())
    out = (p.stderr or b"").decode("utf-8", "replace")
    # 带封面的 mp3 里 png 封面会出现在 "Video: ... (attached pic)" 行上，
    # VIDEO_SIZE_RE 会把它当成真实画面尺寸。这里逐行找**不含 attached pic**
    # 的 Video 行；都没有才说明源里确实只有音频，此时尺寸留空（None）。
    w = h = None
    for line in out.split("\n"):
        if "Video:" in line and "attached pic" not in line:
            mm = VIDEO_SIZE_RE.search(line)
            if mm:
                w, h = int(mm.group(1)), int(mm.group(2))
                break
    dur = None
    m = DURATION_RE.search(out)
    if m:
        dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return w, h, dur


# ---------------------------------------------------------------- 内嵌字幕轨

def probe_subtitle_streams(ffmpeg, media):
    """返回 [{"index":int, "lang":str, "codec":str, "image":bool}, ...]"""
    p = subprocess.run([ffmpeg, "-hide_banner", "-i", media],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       creationflags=_no_window())
    out = (p.stderr or b"").decode("utf-8", "replace")
    res = []
    for m in SUB_STREAM_RE.finditer(out):
        idx = int(m.group(2))
        codec = m.group(4)
        res.append({"index": idx, "lang": (m.group(3) or ""), "codec": codec,
                    "image": codec in IMAGE_SUB_CODECS})
    return res


def extract_subtitle_track(ffmpeg, media, index, dst_srt):
    """把第 index 条字幕轨转成 srt。成功返回 True。

    统一转成 srt 再解析，这样下游只需要写一种解析器；代价是 ass 的样式会丢，
    但歌词只要文字和时间，无所谓。
    """
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
           "-i", media, "-map", "0:%d" % index, "-c:s", "srt", dst_srt]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       creationflags=_no_window())
    return os.path.isfile(dst_srt) and os.path.getsize(dst_srt) > 0


# ---------------------------------------------------------------- 字幕解析

def _ts_hms(s):
    """00:01:02,500 / 00:01:02.500 -> 秒"""
    s = s.strip().replace(",", ".")
    parts = s.split(":")
    if len(parts) == 3:
        h, m, sec = parts
    elif len(parts) == 2:
        h, m, sec = "0", parts[0], parts[1]
    else:
        return float(parts[0])
    return int(h) * 3600 + int(m) * 60 + float(sec)


def _ts_ass(s):
    """0:01:02.50 -> 秒（ass 用百分秒，且只有一位小时）"""
    s = s.strip()
    parts = s.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except Exception:
        return 0.0


def parse_srt(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    subs = []
    ts_re = re.compile(r"(\d{1,3}:\d{1,2}:\d{1,2}[,.]\d{1,3})\s*-->\s*(\d{1,3}:\d{1,2}:\d{1,2}[,.]\d{1,3})")
    for block in re.split(r"\n[ \t]*\n", text):
        m = ts_re.search(block)
        if not m:
            continue
        start, end = _ts_hms(m.group(1)), _ts_hms(m.group(2))
        body = []
        for ln in block.split("\n"):
            if "-->" in ln:
                continue
            # 序号行：块开头孤立的数字
            if re.fullmatch(r"\s*\d+\s*", ln) and not body:
                continue
            body.append(ln.strip())
        t = " ".join(x for x in body if x)
        t = re.sub(r"<[^>]*>", "", t)          # 去 <i> {\an8} 之类标签
        t = re.sub(r"\s+", " ", t).strip()
        if t and end >= start:
            subs.append((start, end, t))
    subs.sort(key=lambda x: x[0])
    return subs


def parse_vtt(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    subs = []
    ts_re = re.compile(r"(\d{1,3}:\d{1,2}:\d{1,2}[.]\d{1,3}|\d{1,2}:\d{1,2}[.]\d{1,3})\s*-->\s*"
                       r"(\d{1,3}:\d{1,2}:\d{1,2}[.]\d{1,3}|\d{1,2}:\d{1,2}[.]\d{1,3})")
    for block in re.split(r"\n[ \t]*\n", text):
        m = ts_re.search(block)
        if not m:
            continue
        start, end = _ts_hms(m.group(1)), _ts_hms(m.group(2))
        body = []
        for ln in block.split("\n"):
            if "-->" in ln or ln.strip().upper().startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
                continue
            body.append(ln.strip())
        t = " ".join(x for x in body if x)
        t = re.sub(r"<[^>]*>", "", t)
        t = re.sub(r"\s+", " ", t).strip()
        if t and end >= start:
            subs.append((start, end, t))
    subs.sort(key=lambda x: x[0])
    return subs


def parse_ass(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    subs = []
    fmt = None
    for ln in text.split("\n"):
        ln = ln.strip()
        if ln.lower().startswith("format:"):
            fmt = [x.strip().lower() for x in ln[len("format:"):].split(",")]
            continue
        if not ln.lower().startswith(("dialogue:", "comment:")):
            continue
        is_comment = ln.lower().startswith("comment:")
        payload = ln.split(":", 1)[1].strip()
        if not fmt:
            continue
        parts = payload.split(",")
        # Text 是最后一项且可能含逗号，多出来的要拼回去
        if len(parts) > len(fmt):
            parts = parts[:len(fmt) - 1] + [",".join(parts[len(fmt) - 1:])]
        d = dict(zip(fmt, parts))
        start = _ts_ass(d.get("start", "0"))
        end = _ts_ass(d.get("end", "0"))
        if is_comment or end <= start:
            continue
        t = d.get("text", "")
        t = re.sub(r"\{[^}]*\}", "", t)        # 去 {\pos(..)} 样式覆盖块
        t = t.replace("\\N", " ").replace("\\n", " ").replace("\\h", " ")
        t = re.sub(r"\s+", " ", t).strip()
        if t:
            subs.append((start, end, t))
    subs.sort(key=lambda x: x[0])
    return subs


def parse_lrc(text):
    """lrc 没有结束时间，end 填下一条的 start。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw = []
    for ln in text.split("\n"):
        hits = LRC_TS_RE.findall(ln)
        if not hits:
            continue
        body = LRC_TS_RE.sub("", ln).strip()
        if not body:
            continue
        for mm, ss, frac in hits:
            if frac == "":
                t = int(mm) * 60 + int(ss)
            elif len(frac) == 2:
                t = int(mm) * 60 + int(ss) + int(frac) / 100.0
            else:
                t = int(mm) * 60 + int(ss) + int(frac) / 1000.0
            raw.append((t, body))
    raw.sort(key=lambda x: x[0])
    subs = []
    for i, (t, body) in enumerate(raw):
        end = raw[i + 1][0] if i + 1 < len(raw) else t + 4.0
        subs.append((t, end, body))
    return subs


def load_subtitle_file(path):
    """按扩展名（并辅以内容嗅探）解析字幕文件，返回 [(start, end, text)]。"""
    with open(path, "rb") as f:
        text = _decode(f.read())
    ext = os.path.splitext(path)[1].lower()
    if ext == ".ass" or ext == ".ssa":
        return parse_ass(text)
    if ext == ".vtt":
        return parse_vtt(text)
    if ext == ".lrc":
        return parse_lrc(text)
    if ext == ".srt":
        return parse_srt(text)
    # 未知扩展名：看内容
    if "Dialogue:" in text or "[Script Info]" in text:
        return parse_ass(text)
    if "WEBVTT" in text:
        return parse_vtt(text)
    if "-->" in text and re.search(r"\d{1,2}:\d{2}:\d{2}", text):
        return parse_srt(text)
    if LRC_TS_RE.search(text):
        return parse_lrc(text)
    return []


def find_sidecar_subtitle(media):
    """找同目录下与媒体同名（或同前缀）的外挂字幕文件。"""
    base = os.path.splitext(media)[0]
    for ext in (".srt", ".ass", ".ssa", ".vtt", ".lrc"):
        p = base + ext
        if os.path.isfile(p):
            return p
        p = base + ext.upper()
        if os.path.isfile(p):
            return p
    d = os.path.dirname(os.path.abspath(media))
    stem = os.path.splitext(os.path.basename(media))[0].lower()
    try:
        for f in sorted(os.listdir(d)):
            if f.lower().startswith(stem) and os.path.splitext(f)[1].lower() in SUB_TEXT_EXT:
                return os.path.join(d, f)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------- 硬字幕 OCR

_OCR = None


def ocr_available():
    try:
        import rapidocr_onnxruntime  # noqa: F401
        return True
    except Exception:
        return False


def get_ocr(log=print, threads=None):
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        if threads is None:
            threads = min(16, os.cpu_count() or 4)
        # 实测：默认线程数下一帧要 1.5s，显式给到 16 线程降到 ~0.9s。
        # 再往上（24 线程 + mkldnn）反而更慢，所以封顶 16。
        try:
            _OCR = RapidOCR(intra_op_num_threads=int(threads))
        except Exception:
            _OCR = RapidOCR()
        log("  加载 OCR 引擎（%d 线程）..." % int(threads))
    return _OCR


def _imread(path, gray=False):
    """cv2.imread 不支持中文路径，用 fromfile + imdecode 绕开。"""
    import cv2
    import numpy as np
    data = np.fromfile(path, dtype=np.uint8)
    flag = 0 if gray else 1
    return cv2.imdecode(data, flag)


def _gray_small(path, size=(160, 40)):
    import cv2
    img = _imread(path, gray=True)
    if img is None:
        return None
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA).astype("float32")


def _ocr_items(engine, img, min_score=0.5):
    """OCR 一张图，返回 [(文本框中心y, 文本, 置信度), ...]。

    注意 rapidocr 的 __call__ 返回的是 (结果列表, 各阶段耗时列表) 二元组，
    直接遍历会把耗时（一堆 float）也当成结果，所以这里先拆一层。
    """
    if img is None:
        return []
    try:
        out = engine(img)
    except Exception:
        return []
    if out is None:
        return []
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], (list, tuple)):
        out = out[0]
    if not isinstance(out, (list, tuple)) or not out:
        return []
    items = []
    for it in out:
        try:
            box, txt, score = it[0], it[1], it[2]
            score = float(score)
        except Exception:
            continue
        if not isinstance(txt, str):
            continue
        txt = txt.strip()
        if not txt or score < min_score:
            continue
        try:
            ys = sum(float(p[1]) for p in box) / max(1, len(box))
        except Exception:
            continue
        items.append((ys, txt, score))
    items.sort()
    return items


def ocr_image_text(engine, img, min_score=0.5):
    """OCR 一张图，返回 (合并后的文本, 平均置信度)。多行按 y 排序后空格拼接。"""
    items = _ocr_items(engine, img, min_score=min_score)
    if not items:
        return "", 0.0
    text = " ".join(t for _, t, _ in items)
    avg = sum(s for _, _, s in items) / float(len(items))
    return re.sub(r"\s+", " ", text).strip(), avg


def resolve_region(region, w, h):
    """把 --subs-region 解析成 (y0, y1) 像素坐标。"""
    r = (region or "bottom").strip().lower()
    if r in ("bottom", "b", ""):
        a, b = 0.70, 0.98
    elif r in ("top", "t"):
        a, b = 0.02, 0.25
    else:
        m = re.match(r"^([\d.]+)\s*[,，]\s*([\d.]+)$", r)
        if not m:
            raise ValueError("无法解析字幕区域 %r（可用 bottom / top / 0.65,0.95）" % region)
        a, b = float(m.group(1)), float(m.group(2))
        if a > 1 or b > 1:      # 直接给像素
            y0, y1 = int(a), int(b)
            y0 = max(0, min(y0, h - 8))
            y1 = max(y0 + 8, min(y1, h))
            return y0, y1
    y0 = int(h * a)
    y1 = int(h * b)
    y0 = max(0, min(y0, h - 8))
    y1 = max(y0 + 8, min(y1, h))
    if (y1 - y0) % 2:
        y1 -= 1
    return y0, y1


def _extract_frames(ffmpeg, media, outdir, crop, fps, ss=None, dur=None, scale_max=None):
    """按 crop + fps 抽帧到 outdir，返回 [(time_sec, path)]。

    crop=None 表示整帧；scale_max 用于把输出缩到指定宽度（自动定位时要扫全画面，
    缩小能省不少 OCR 时间）。
    第 k 帧的时间按 ss + k/fps 估算，误差不超过 1/(2*fps)，后面还会细化。
    """
    os.makedirs(outdir, exist_ok=True)
    for f in glob.glob(os.path.join(outdir, "*.jpg")):
        try:
            os.remove(f)
        except Exception:
            pass
    parts = []
    if crop:
        parts.append("crop=%s" % crop)
    if scale_max:
        parts.append("scale=%d:-2" % scale_max)
    parts.append("fps=%.6f" % fps)
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    if ss is not None:
        cmd += ["-ss", "%.3f" % max(0.0, ss)]
    cmd += ["-i", media]
    if dur is not None:
        cmd += ["-t", "%.3f" % dur]
    cmd += ["-an", "-sn", "-vf", ",".join(parts),
            "-q:v", "3", "-start_number", "0",
            os.path.join(outdir, "f%06d.jpg")]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                   creationflags=_no_window())
    files = sorted(glob.glob(os.path.join(outdir, "f*.jpg")))
    base = 0.0 if ss is None else ss
    return [(base + i / float(fps), p) for i, p in enumerate(files)]


def _grab_stills(ffmpeg, media, outdir, times, scale_max=None):
    """按给定的时间点各抽一帧（每次单独 seek）。

    比"fps 滤镜跑一遍全片"划算：fps 滤镜必须解码所有帧才能均匀采样，
    而这里每次 seek 只解到目标位置附近，几十次加起来也快得多。
    """
    os.makedirs(outdir, exist_ok=True)
    for f in glob.glob(os.path.join(outdir, "*.jpg")):
        try:
            os.remove(f)
        except Exception:
            pass
    out = []
    for i, t in enumerate(times):
        p = os.path.join(outdir, "s%04d.jpg" % i)
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-ss", "%.3f" % max(0.0, t), "-i", media, "-frames:v", "1"]
        if scale_max:
            cmd += ["-vf", "scale=%d:-2" % scale_max]
        cmd += ["-q:v", "3", p]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       creationflags=_no_window())
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            out.append((t, p))
    return out


def _detect_items(engine, img):
    """只跑文字"检测"，不跑"识别"。

    返回 [(中心y, 框高, y上, y下), ...]，坐标是像素。
    定位字幕带只需要知道"哪里有条状文字"，不需要认出是什么字，
    而检测比完整 OCR 快（实测约 0.3~0.9s vs 1~3s），所以这条路更划算。
    拿不到内部检测器时回退到完整 OCR（只是慢些，结果一样）。

    注意返回结构：text_det 给的是 (盒子数组, 耗时)，且盒子是 numpy 数组
    shape=(N,4,2) 而不是 list —— 早期版本这里按 list 判断，结果一个框都拿不到。
    """
    if img is None:
        return []
    det = getattr(engine, "text_det", None)
    if det is None:
        items = _ocr_items(engine, img, min_score=0.5)
        return [(cy, 0.0, cy, cy) for cy, _t, _s in items]
    try:
        out = det(img)
    except Exception:
        return []
    if out is None:
        return []
    boxes = out[0] if isinstance(out, tuple) and len(out) >= 1 else out
    if boxes is None:
        return []
    import numpy as np
    try:
        allb = np.asarray(boxes, dtype="float32")
    except Exception:
        return []
    if allb.ndim != 3 or allb.shape[1] < 4 or allb.shape[2] < 2:
        return []
    items = []
    for b in allb:
        hh = float(b[:, 1].max() - b[:, 1].min())
        ww = float(b[:, 0].max() - b[:, 0].min())
        # 太小的当噪声
        if hh < 5 or ww < 12:
            continue
        items.append((float(b[:, 1].mean()), hh,
                      float(b[:, 1].min()), float(b[:, 1].max())))
    return items


def detect_subtitle_region(ffmpeg, media, w, h, engine=None, samples=16,
                           min_ratio=0.30, workdir=None, log=print,
                           start=None, duration=None):
    """自动定位字幕带，返回 (y0, y1) 像素；定位不出来返回 None。

    依据：字幕几乎每帧都出现在同一高度，而标题、水印、画面里偶发的文字
    只在个别帧出现。所以把检出文字的纵向范围按桶统计"出现帧数"，
    出现率最高的那条连续带就是字幕带。

    实测踩到的坑：不是所有视频都把字幕放底部 —— 实测两个填词视频的字幕都在
    画面正中，固定用 bottom 会一行都抓不到，所以默认走这个自动探测。
    """
    if not ocr_available():
        return None
    _w, _h, dur = probe_media_info(ffmpeg, media)
    if not dur or dur < 4:
        return None
    engine = engine or get_ocr(log=log)
    made = workdir is None
    d = workdir or tempfile.mkdtemp(prefix="subsdet_")
    try:
        base = max(0.0, float(start or 0.0))
        span_all = float(duration) if duration else max(0.0, dur - base)
        if span_all < 4:                      # 片段太短，没意义，退回整片
            base, span_all = 0.0, dur
        n = max(6, int(samples))
        t_start = base + span_all * 0.05
        span = span_all * 0.9
        times = [t_start + span * i / float(n - 1) for i in range(n)]
        log("  自动定位字幕带：抽查 %d 帧…" % n)
        # 定位只需要知道"文字大致在画面的哪一带"，不需要看清每个字，
        # 所以整帧缩到 640 宽，检测能快好几倍。
        frames = _grab_stills(ffmpeg, media, d, times, scale_max=640)
        if len(frames) < 3:
            return None
        BUCKETS = 20
        counts = [0] * BUCKETS
        used = 0
        for _t, p in frames:
            img = _imread(p)
            if img is None:
                continue
            items = _detect_items(engine, img)
            if not items:
                continue
            used += 1
            ih = float(img.shape[0]) or 1.0
            seen = set()
            for _cy, _hh, y0b, y1b in items:
                b0 = min(BUCKETS - 1, max(0, int(y0b / ih * BUCKETS)))
                b1 = min(BUCKETS - 1, max(0, int(y1b / ih * BUCKETS)))
                for b in range(b0, b1 + 1):
                    if b not in seen:
                        seen.add(b)
                        counts[b] += 1
            del img
        if used < 3:
            return None
        peak = max(counts)
        if peak < used * min_ratio:
            log("  没有哪条纵向带稳定出现文字（最高出现率 %.0f%%），不自动定位"
                % (100.0 * peak / used))
            return None
        ci = counts.index(peak)
        thr = peak * 0.55
        lo = ci
        while lo - 1 >= 0 and counts[lo - 1] >= thr:
            lo -= 1
        hi = ci
        while hi + 1 < BUCKETS and counts[hi + 1] >= thr:
            hi += 1
        pad = 0.012
        y0 = int(h * max(0.0, lo / float(BUCKETS) - pad))
        y1 = int(h * min(1.0, (hi + 1) / float(BUCKETS) + pad))
        if y1 - y0 < h * 0.05:
            y1 = min(h, y0 + int(h * 0.05))
        if y1 - y0 > h * 0.32:
            y1 = y0 + int(h * 0.32)
        if (y1 - y0) % 2:
            y1 -= 1
        log("  字幕带定位成功：抽查 %d 帧，其中 %d 帧检出文字，纵向 %.0f%%-%.0f%%（y %d-%d）"
            % (len(frames), peak, 100.0 * lo / BUCKETS,
               100.0 * (hi + 1) / BUCKETS, y0, y1))
        return (y0, y1)
    finally:
        if made:
            shutil.rmtree(d, ignore_errors=True)


def ocr_hard_subtitles(ffmpeg, media, region="auto", scan_fps=2.0,
                       refine=True, refine_fps=6.0, min_score=0.5,
                       diff_thresh=10.0, min_interval=0.45, max_scan_frames=4000,
                       scan_scale=896, workdir=None, log=print,
                       start=None, duration=None):
    """识别画面上烧录的硬字幕，返回 [(start, end, text)]。

    思路：字幕切换必然造成该区域像素突变，所以先用极便宜的"帧间灰度差"定位突变点，
    只对突变帧做 OCR（几百帧里通常只有几十帧要真正识别）。粗扫的时间分辨率是
    1/scan_fps，再对每个新行在突变点前 1/scan_fps 的窗口里用 refine_fps 细扫，
    把起点误差压到 1/refine_fps。

    start/duration 可以把处理范围限制在一个片段内（用于快速试效果）。
    此时输出的时间戳仍然是**绝对时间**，与视频对齐，可以直接和片段对上。
    """
    if not ocr_available():
        raise RuntimeError(
            "未安装 OCR 引擎。请运行 start.bat --install-ocr，或改用 --subs track 只处理内嵌字幕轨。")

    w, h, total = probe_media_info(ffmpeg, media)
    if not h:
        raise RuntimeError("无法读取视频分辨率，不能定位字幕区域。")

    win_start = max(0.0, float(start or 0.0))
    if duration:
        win_end = win_start + float(duration)
        if total:
            win_end = min(win_end, total)
    else:
        win_end = total
    win_dur = (win_end - win_start) if win_end else None
    if win_dur is not None and win_dur <= 0:
        win_start, win_dur, win_end = 0.0, None, total
    if win_start or (win_dur and total and win_dur < total - 0.5):
        log("  处理区间: %.1fs ~ %s" % (win_start,
                                      ("%.1fs" % (win_start + win_dur)) if win_dur else "片尾"))

    keep_tmp = bool(workdir)
    tmp = workdir or tempfile.mkdtemp(prefix="subsocr_")
    try:
        engine = get_ocr(log=log)

        if (region or "").strip().lower() == "auto":
            got = detect_subtitle_region(ffmpeg, media, w, h, engine=engine,
                                         workdir=os.path.join(tmp, "det"), log=log,
                                         start=win_start, duration=win_dur)
            if got:
                y0, y1 = got
            else:
                y0, y1 = resolve_region("bottom", w, h)
                log("  改用画面下部作为字幕区: y %d-%d" % (y0, y1))
        else:
            y0, y1 = resolve_region(region, w, h)
        log("  字幕区域: y %d-%d（画面高 %d），粗扫 %.2f fps" % (y0, y1, h, scan_fps))

        # 抽帧太多会又慢又占盘，按上限自动降帧率
        if win_dur:
            est = win_dur * scan_fps
            if est > max_scan_frames:
                scan_fps = max(0.5, max_scan_frames / float(win_dur))
                log("  预计抽帧过多，粗扫帧率降到 %.2f fps" % scan_fps)

        crop = "iw:%d:0:%d" % (y1 - y0, y0)
        scan_dir = os.path.join(tmp, "scan")
        log("  抽帧中（需解码画面，1080p 约需半分钟到几分钟）...")
        # 抽帧时就缩小。这一步不只是为了快：
        # 实测这个视频的字幕是超大字艺术排版，原分辨率下识别模型反而认不出来
        # （"谁曾四渡赤水破万难"只认出 1 个字），缩到 768~896 宽反而能认出 9/10 个。
        # 原因是识别模型对文字高度有最佳区间，太大的字同样会掉准确率。
        frames = _extract_frames(ffmpeg, media, scan_dir, crop, scan_fps,
                                 ss=win_start if win_start else None,
                                 dur=win_dur, scale_max=scan_scale)
        log("  粗扫 %d 帧，开始定位字幕变化点" % len(frames))
        if not frames:
            return []

        subs = []
        prev_gray = None
        last_ocr_t = -1e9
        last_new_t = -1e9      # 上一行"新增"时的粗扫时间（不是细化后的时间）
        ocr_calls = 0
        refine_dir = os.path.join(tmp, "refine")

        for t, path in frames:
            g = _gray_small(path)
            if g is None:
                continue
            if prev_gray is None:
                diff = 1e9
            else:
                import numpy as np
                diff = float(np.abs(g - prev_gray).mean())
            prev_gray = g
            if diff < diff_thresh:
                continue
            if t - last_ocr_t < min_interval:
                continue
            last_ocr_t = t
            text, score = ocr_image_text(engine, _imread(path), min_score=min_score)
            ocr_calls += 1
            if not text or not plausible_subtitle_text(text):
                continue
            nt = norm_txt(text)
            if subs:
                nt_prev = norm_txt(subs[-1][2])
                sim = similar(nt, nt_prev)
                if sim >= 0.85:
                    continue                       # 与上一行基本一致，忽略
                # 时间紧挨着、文字又相似 -> 同一句正在逐字/淡入，
                # 保留更完整的那一版，不另起一行。
                # （实测片头名单会被读成 "重新填词：钟表居士 虞兮叹 原曲： ："
                #   的几个残缺版本，这条规则把它们合成一句。）
                # 注意这里比的是"粗扫时间"：细化会把起点往前挪，
                # 拿挪过的时间来比会把本该合并的两帧判成隔得太远。
                if sim >= 0.55 and t - last_new_t <= 3.0:
                    if len(nt) > len(nt_prev):
                        subs[-1][2] = text
                    continue
                # 挨得极近就不再看相似度：字幕不会在 1.2 秒内换一句，
                # 这种只能是 OCR 把同一句读成了好几种（实测"六朝…泪眼已然"
                # 一度被拆成 6 行）。保留文字更完整的那一版。
                if t - last_new_t <= 1.2:
                    if len(nt) > len(nt_prev):
                        subs[-1][2] = text
                    continue
            start_t = t
            if refine:
                # 细化只需要判断"这句出现了没有"，不需要看清每个字，
                # 所以用比粗扫更小的图，能省下不少时间。
                got = _refine_start(ffmpeg, media, crop, t, scan_fps, nt,
                                    refine_dir, refine_fps, engine, min_score, log,
                                    scale_max=min(scan_scale or 768, 768))
                ocr_calls += 1
                if got is not None:
                    start_t = got
            subs.append([start_t, start_t, text])
            last_new_t = t
            if win_end:
                log("    %.1fs  %s" % (start_t, text[:36]))

        log("  OCR 调用 %d 次，得到 %d 行" % (ocr_calls, len(subs)))
        # 补结束时间
        for i in range(len(subs)):
            nxt = subs[i + 1][0] if i + 1 < len(subs) else (win_end or (subs[i][0] + 4.0))
            subs[i][1] = max(subs[i][0] + 0.3, min(nxt, subs[i][0] + 30.0))
        return [(float(a), float(b), c) for a, b, c in subs]
    finally:
        if not keep_tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def _refine_start(ffmpeg, media, crop, t_guess, scan_fps, target_norm, workdir,
                  refine_fps, engine, min_score, log, scale_max=None):
    """在 [t_guess - 1/scan_fps, t_guess] 窗口里细扫，找目标文本首次出现的时刻。

    窗口宽度取粗扫的时间分辨率 —— 真正的切换点必然落在"上一个粗扫帧"和
    "本次粗扫帧"之间，在这一格内用更高的帧率重扫就能把误差压到 1/refine_fps。
    """
    win = 1.0 / max(0.5, scan_fps)
    t0 = max(0.0, t_guess - win - 0.08)
    dur = win + 0.16
    try:
        frames = _extract_frames(ffmpeg, media, workdir, crop, refine_fps, ss=t0, dur=dur,
                                 scale_max=scale_max)
    except Exception:
        return t_guess
    for t, path in frames:
        if t > t_guess + 0.02:
            break
        text, _s = ocr_image_text(engine, _imread(path), min_score=min_score)
        nt = norm_txt(text)
        if not nt:
            continue
        if nt == target_norm or similar(nt, target_norm) >= 0.72 \
                or target_norm in nt or nt in target_norm:
            return t
    return t_guess


# ---------------------------------------------------------------- 输出适配

def subs_to_segments(subs):
    """[(start, end, text)] -> lyric_maker 用的 [{'start','end','text'}]"""
    out = []
    for i, (s, e, t) in enumerate(subs):
        out.append({"start": float(s), "end": float(e), "text": str(t)})
    return out


# ---------------------------------------------------------------- 输出渲染
# 下面是"segs -> 各字幕格式文本"的纯函数，不含文件写入（编码/换行在调用方处理）。

def fmt_srt_ts(sec):
    """秒 -> 00:00:01,500（srt 用逗号 + 毫秒）"""
    if sec < 0:
        sec = 0.0
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return "%02d:%02d:%02d,%03d" % (h, m, s, ms)


def fmt_vtt_ts(sec):
    """秒 -> 00:00:01.500（vtt 用点 + 毫秒）"""
    return fmt_srt_ts(sec).replace(",", ".")


def fmt_ass_ts(sec):
    """秒 -> 0:00:01.50（ass 小时可 1 位 + 百分秒）"""
    if sec < 0:
        sec = 0.0
    cs = int(round(sec * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return "%d:%02d:%02d.%02d" % (h, m, s, cs)


def render_srt(segs, offset_sec=0.0):
    """segs([{start,end,text}]) -> srt 文本（\n 连接，含结尾空行）。"""
    lines = []
    for i, s in enumerate(segs, 1):
        lines.append(str(i))
        lines.append("%s --> %s" % (fmt_srt_ts(s["start"] + offset_sec),
                                    fmt_srt_ts(s["end"] + offset_sec)))
        lines.append(s["text"])
        lines.append("")
    return "\n".join(lines)


def render_vtt(segs, offset_sec=0.0):
    """segs -> WebVTT 文本。"""
    lines = ["WEBVTT", ""]
    for s in segs:
        lines.append("%s --> %s" % (fmt_vtt_ts(s["start"] + offset_sec),
                                    fmt_vtt_ts(s["end"] + offset_sec)))
        lines.append(s["text"])
        lines.append("")
    return "\n".join(lines)


def render_ass(segs, offset_sec=0.0, title=None):
    """segs -> ASS(v4.00+) 文本。ssa 与 ass 共用这一份，内容相同。"""
    head = [
        "[Script Info]",
        "; Generated by lyric-maker",
        "ScriptType: v4.00+",
        "PlayResX: 1920",
        "PlayResY: 1080",
        "WrapStyle: 2",
    ]
    if title:
        head.append("Title: " + str(title).replace("\n", " ").strip())
    head += [
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
        "0,0,0,0,100,100,0,0,1,2,0,2,10,10,20,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for s in segs:
        text = str(s["text"])
        text = text.replace("\\", "").replace("\n", "\\N")
        # Text 是 Events 行的最后一个字段，逗号按规范可以直接保留；
        # 这里不替换，交给播放器按"贪婪匹配最后字段"解析。
        head.append("Dialogue: 0,%s,%s,Default,,0,0,0,,%s"
                    % (fmt_ass_ts(s["start"] + offset_sec),
                       fmt_ass_ts(s["end"] + offset_sec), text))
    return "\n".join(head)



# =====================================================================
# 模型下载（双源 + 断点续传）
# =====================================================================

HF_MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_ENDPOINT"] = HF_MIRROR
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# name -> (hf repo, size, desc)
HF_REPOS = {
    "tiny":   ("Systran/faster-whisper-tiny",   "~75 MB",  "最快，质量最差"),
    "base":   ("Systran/faster-whisper-base",   "~145 MB", "快，质量一般"),
    "small":  ("Systran/faster-whisper-small",  "~480 MB", "较快，可用"),
    "medium": ("Systran/faster-whisper-medium", "~1.5 GB", "平衡"),
    "turbo":  ("deepdml/faster-whisper-large-v3-turbo-ct2", "~1.6 GB", "大模型蒸馏版，快且不错"),
}

# name -> (modelscope repo, size, desc)   —— 稳定高速，放大的模型
MS_REPOS = {
    "large-v3": ("pengzhendong/faster-whisper-large-v3", "~2.9 GB", "质量最好（中文首选）"),
}
MS_ALT = "keepitsimple/faster-whisper-large-v3"   # 备用仓库

MS_FILES = ["config.json", "model.bin", "tokenizer.json",
            "vocabulary.json", "preprocessor_config.json"]

ALL = {}
ALL.update({k: ("hf-mirror", v[0], v[1], v[2]) for k, v in HF_REPOS.items()})
ALL.update({k: ("modelscope", v[0], v[1], v[2]) for k, v in MS_REPOS.items()})


def human(n):
    n = float(n)
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return "%.1f %s" % (n, u)
        n /= 1024.0
    return "%.1f TB" % n


def dir_size(d):
    tot = 0
    for root, _, files in os.walk(d):
        for f in files:
            try:
                tot += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return tot


def is_downloaded(name):
    return os.path.isfile(os.path.join(MODELS_DIR, name, "model.bin"))


# =====================================================================
# ModelScope：自写续传下载器
# =====================================================================
def ms_url(repo, filename):
    q = urllib.parse.urlencode({"Revision": "master", "FilePath": filename})
    return "https://www.modelscope.cn/api/v1/models/%s/repo?%s" % (repo, q)


def ms_download_file(repo, filename, dest, retries=10, log=print):
    """带 Range 断点续传 + 指数退避重试的单文件下载。"""
    url = ms_url(repo, filename)
    part = dest + ".part"

    for attempt in range(1, retries + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        headers = dict(UA)
        if have > 0:
            headers["Range"] = "bytes=%d-" % have
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as r:
                total = None
                cr = r.headers.get("Content-Range")
                if cr and "/" in cr:
                    total = int(cr.split("/")[-1])
                elif r.headers.get("Content-Length"):
                    total = have + int(r.headers["Content-Length"])

                # 服务端没按 Range 返回（200 而非 206）就必须从头写，不能追加
                if r.status != 206 and have > 0:
                    have = 0
                mode = "ab" if have > 0 else "wb"

                if attempt == 1 or have == 0:
                    log("    %s  %s" % (filename, human(total) if total else "(未知大小)"))

                last_log = time.time()
                with open(part, mode) as f:
                    while True:
                        buf = r.read(1 << 20)
                        if not buf:
                            break
                        f.write(buf)
                        f.flush()        # 立刻交给操作系统，不要留在 Python 缓冲区
                        have += len(buf)
                        if time.time() - last_log >= 5:
                            last_log = time.time()
                            try:
                                os.fsync(f.fileno())
                            except Exception:
                                pass
                            # 自检：磁盘上的字节数必须与统计值吻合。实测遇到过
                            # "write 不报错但文件不增长"的情况，这一步会立刻暴露它。
                            try:
                                on_disk = os.path.getsize(part)
                            except OSError:
                                on_disk = -1
                            warn = ""
                            if on_disk >= 0 and abs(on_disk - have) > (2 << 20):
                                warn = "  [警告] 磁盘实际只有 %.1f MB，写入被丢弃！" \
                                       % (on_disk / 1048576.0)
                            if total:
                                log("      %s  %5.1f%%  (%.1f/%.1f MB)%s"
                                    % (filename, 100.0 * have / total,
                                       have / 1048576.0, total / 1048576.0, warn))
                            else:
                                log("      %s  %.1f MB%s"
                                    % (filename, have / 1048576.0, warn))

            if total is None or os.path.getsize(part) >= total:
                os.replace(part, dest)
                log("    %s  完成 %s" % (filename, human(os.path.getsize(dest))))
                return True
            log("    连接提前结束（%s），将续传" % filename)
        except Exception as e:
            got = os.path.getsize(part) if os.path.exists(part) else 0
            log("    第 %d/%d 次中断: %s（已存 %.1f MB，将续传）"
                % (attempt, retries, type(e).__name__, got / 1048576.0))
            time.sleep(min(2 * attempt, 12))

    log("    [失败] %s 重试 %d 次仍未完成" % (filename, retries))
    return False


def download_from_modelscope(name, repo, target, log=print, alt=None):
    log("源      : ModelScope (%s)" % repo)
    os.makedirs(target, exist_ok=True)
    for fn in MS_FILES:
        dest = os.path.join(target, fn)
        if os.path.isfile(dest) and os.path.getsize(dest) > 0:
            log("    %s  已存在，跳过" % fn)
            continue
        ok = ms_download_file(repo, fn, dest, log=log)
        if not ok and alt:
            log("    改用备用仓库 %s 重试 %s" % (alt, fn))
            ok = ms_download_file(alt, fn, dest, log=log)
        if not ok:
            return False
    return True


def download_from_hf(name, repo, target, log=print):
    log("源      : hf-mirror (%s)  注意：实测较慢且大文件易断" % HF_MIRROR)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log("[错误] 缺少 huggingface_hub，请先运行 start.bat 安装依赖")
        return False
    os.makedirs(target, exist_ok=True)
    try:
        snapshot_download(repo_id=repo, local_dir=target, endpoint=HF_MIRROR,
                          allow_patterns=["config.json", "model.bin", "tokenizer.json",
                                          "vocabulary.*", "preprocessor_config.json"],
                          max_workers=2)
        return True
    except Exception as e:
        log("[错误] 下载失败: %s: %s" % (type(e).__name__, e))
        log("提示：改用 ModelScope 源试试 ->  python lyric_maker.py --get-model large-v3")
        return False


# =====================================================================
def show_list():
    print("模型目录: %s" % MODELS_DIR)
    print("hf-mirror: %s" % HF_MIRROR)
    print()
    print("  %-10s %-11s %-9s %-22s %s" % ("名称", "源", "大小", "状态", "说明"))
    print("  " + "-" * 88)
    for name in list(HF_REPOS) + list(MS_REPOS):
        src, repo, size, desc = ALL[name]
        d = os.path.join(MODELS_DIR, name)
        state = ("已下载 %s" % human(dir_size(d))) if is_downloaded(name) else "未下载"
        print("  %-10s %-11s %-9s %-22s %s" % (name, src, size, state, desc))
    print()
    print("推荐：中文歌曲/视频先用 small 验证流程，再下 large-v3 追求质量。")
    print("      python lyric_maker.py --get-model small")
    print("      python lyric_maker.py --get-model large-v3")


def download(name, force=False, source=None, log=print):
    # 这里所有消息一律走 `log`，不用 print。
    # 原因：`log` 是调用方传进来的输出通道（GUI 传的是"塞进队列回主线程"的回调，
    # 命令行传的是默认 print）。原来这些全用 print，于是**在打包的 exe 里
    # 无控制台模式下这些话只会进日志文件，界面上一个字都看不到** ——
    # 下载失败时用户只看到一句"下载失败，可重试"，完全不知道原因。
    if name not in ALL:
        log("[错误] 未知模型 '%s'。可选: %s" % (name, ", ".join(ALL)))
        return 1
    src, repo, size, desc = ALL[name]
    if source and source != "auto":
        if source == "modelscope" and name not in MS_REPOS:
            log("[错误] %s 没有 ModelScope 源，只有: %s" % (name, ", ".join(MS_REPOS)))
            return 1
        if source == "hf" and name not in HF_REPOS:
            log("[错误] %s 没有 hf-mirror 源，只有: %s" % (name, ", ".join(HF_REPOS)))
            return 1
        src = source

    target = os.path.join(MODELS_DIR, name)
    if is_downloaded(name) and not force:
        log("[跳过] %s 已存在（%s）。要重下加 --force。" % (name, human(dir_size(target))))
        return 0

    log("模型    : %s   (%s)" % (name, desc))
    log("预计大小: %s" % size)
    log("保存到  : %s" % target)
    log("")

    t0 = time.time()
    if src == "modelscope":
        ok = download_from_modelscope(name, repo, target, log=log, alt=MS_ALT)
    else:
        ok = download_from_hf(name, repo, target, log=log)

    if not ok:
        log("[失败] %s 下载没有完成（上面是具体原因）。可重试，会从断点续传。" % name)
        return 1

    missing = [f for f in ("model.bin", "config.json", "tokenizer.json")
               if not os.path.isfile(os.path.join(target, f))]
    if missing:
        log("[警告] 缺少关键文件: %s" % ", ".join(missing))
        return 1
    log("[完成] %s -> %s  共 %s，用时 %.1f 分钟"
        % (name, target, human(dir_size(target)), (time.time() - t0) / 60.0))
    return 0


def model_main():
    ap = argparse.ArgumentParser(description="下载 faster-whisper 模型（双源，带断点续传）")
    ap.add_argument("name", nargs="?", help="模型名: %s" % "/".join(ALL))
    ap.add_argument("--force", action="store_true", help="已存在也重新下载")
    ap.add_argument("--source", choices=["auto", "modelscope", "hf"], default="auto",
                    help="强制指定下载源")
    ap.add_argument("--list", action="store_true", help="只列出模型")
    args = ap.parse_args()

    if args.list or not args.name:
        show_list()
        return 0
    return download(args.name, force=args.force, source=args.source)


# =====================================================================
# 视频下载：内置 yt-dlp + Cookie 识别
# ---------------------------------------------------------------------
# 目标：下载视频**不用再开控制台、不用另外装 yt-dlp**。
#   * 用 yt-dlp 的 **Python API**（不是命令行），界面才能拿到结构化的
#     进度 / 速度 / 剩余时间，而不用去解析控制台文本。
#   * Cookie 两条路：① 让 yt-dlp 直接从本机浏览器读（只在内存里用、不落盘）；
#     ② 把手动导出的 cookies.txt 丢进主目录的 `cookies/` 文件夹。
#   * 全程"缺了也不崩"：yt-dlp 没装 / Cookie 读不到 / 没有 ffmpeg，
#     都只是功能降级 + 明确提示，绝不让整个软件起不来。
# =====================================================================

COOKIE_DIR_DEFAULT = os.path.join(_HERE, "cookies")
# 当前生效的 Cookie 目录。存在 settings["cookie_dir"] 里，默认就是 exe 旁边的 cookies/。
# 用变量而不是常量：用户能在界面上改（见 set_cookie_dir）。
_cookie_dir = COOKIE_DIR_DEFAULT


def cookie_dir():
    """当前生效的 Cookie 文件夹。"""
    return _cookie_dir


# YouTube 登录态真正依赖的几个 cookie。缺任何一个都可能被判成未登录：
#   SID / HSID / SSID / APISID / SAPISID —— 会话与登录凭据
#   __Secure-3PAPISID / __Secure-1PAPISID / __Secure-3PSID —— 新版加固凭据
# 判断"有没有登录"而不是"有没有 .youtube.com 的 cookie"：游客也会拿到
# CONSENT、PREF 之类的偏好 cookie，只有上面这些才是真正的登录态。
YT_AUTH_COOKIES = ("SID", "HSID", "SSID", "APISID", "SAPISID",
                   "__Secure-3PAPISID", "__Secure-1PAPISID",
                   "__Secure-3PSID", "__Secure-1PSID", "LOGIN_INFO")


def youtube_auth_state(path):
    """检查一个 cookie 文件里有没有可用的 YouTube 登录态。

    返回 (是否有登录, 说明)。为什么需要：Get cookies.txt LOCALLY 之类
    的插件默认导出**当前域名**的 cookie，如果用户在别的页面点的导出，
    文件里就只有那个站点的条目 —— 格式完全合法，但 yt-dlp 拿去用会被
    判成未登录。用户看到的是"登录失效"，真实原因却是"导错页面了"。
    """
    if not path or not os.path.isfile(path):
        return False, "文件不存在"
    try:
        with io.open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except Exception as e:
        return False, "读取失败：%s" % e

    total = yt = 0
    found = set()
    for ln in lines:
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        parts = ln.split("\t")
        if len(parts) < 7:
            continue
        total += 1
        domain = parts[0]
        if "youtube.com" not in domain:
            continue
        yt += 1
        name = parts[5].strip()
        if name in YT_AUTH_COOKIES:
            found.add(name)

    if total == 0:
        return False, "文件里没有 cookie 数据"
    if yt == 0:
        return False, ("只导出了 %d 个其他网站的 cookie，没有 YouTube 的 —— "
                       "请在**已登录 YouTube 的那个标签页**上重新导出" % total)
    if not found:
        return (False, "有 %d 条 YouTube cookie，但缺少登录凭据（%s）—— "
                       "多半是没登录，或导出时页面不对"
                % (yt, "、".join(YT_AUTH_COOKIES[:4])))
    return True, "已登录（%d 条 YouTube cookie，凭据含 %s）" % (
        yt, "、".join(sorted(found)[:4]))


def _looks_like_cookie_file(p):
    """判断一个文件是不是 Netscape 格式的 cookie 文件。

    yt-dlp 只认 Netscape 格式，但**文件名随便叫什么都行**。判定分两级：
      ① 文件头里出现 `Netscape HTTP Cookie File` —— 主流导出工具都会写；
      ② 退一步看**结构**：至少有一行是 7 个制表符分隔字段、且第 6/7 个非空
         （域名 / 含子域 / 路径 / 安全 / 过期 / 名 / 值）。

    为什么不能只看"有没有制表符"：一个制表符分隔的通讯录
    （`张三\t123`）会被误判成 cookie 文件 —— 实测踩过。
    """
    try:
        with io.open(p, "r", encoding="utf-8", errors="replace") as f:
            head = f.read(8192)
    except Exception:
        return False
    if "Netscape HTTP Cookie File" in head:
        return True
    for ln in head.splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        parts = ln.split("\t")
        if len(parts) >= 7 and parts[5].strip() and parts[6].strip():
            return True
    return False


def cookie_scan_dirs():
    """Cookie 文件的扫描范围：**只有 Cookie 文件夹**。

    之前一度把 exe 所在的主文件夹也扫进来（想少让用户搬文件），但主目录里
    乱七八糟的 .txt 太多、也容易让人搞不清"到底哪份在生效"。收回到
    Cookie 文件夹一处，规则简单：**这里放什么就用什么**。
    """
    return [_cookie_dir] if _cookie_dir else []


def collect_cookie_files():
    """收集 Cookie 文件夹里所有**内容合格**的 .txt（按修改时间**从新到旧**）。

    返回全部合格文件（不只是要用的那个）—— 界面要靠它说明
    "有 N 个候选、正在用哪个"。
    """
    out = []
    for d in cookie_scan_dirs():
        try:
            names = os.listdir(d)
        except Exception:
            continue
        for n in names:
            if not n.lower().endswith(".txt"):
                continue
            p = os.path.join(d, n)
            if not os.path.isfile(p) or p in out:
                continue
            if _looks_like_cookie_file(p):
                out.append(p)
    # 新的在前；同一时间戳按名字排，保证结果稳定（不随文件系统顺序抖动）
    def _key(p):
        try:
            return (os.path.getmtime(p), p)
        except Exception:
            return (0.0, p)
    out.sort(key=_key, reverse=True)
    return out


def pick_cookie_file():
    """选出生效的 cookie 文件：**最新的那个**。

    为什么按"最新"而不是"合并全部"：用户手上的多份导出往往是同一账号在不同时间
    导出的，越新的越可能还有效（旧的早被 YouTube 轮换掉了）。全合并会把一堆
    已失效的条目也塞进去，反而更容易被判成无效登录态。
    """
    files = collect_cookie_files()
    return files[0] if files else ""


def cookie_txt():
    """送给 yt-dlp 的 cookie 文件路径（没有可用文件时返回空串）。"""
    try:
        return pick_cookie_file()
    except Exception:
        return ""


def cookie_file():
    """当前生效的 cookie 文件路径（没有时返回约定的默认名，便于提示）。"""
    return pick_cookie_file() or os.path.join(_cookie_dir, "cookies.txt")


def cookie_problem():
    """Cookie 有问题时给一句人话说明（没问题返回 ""）。

    专门区分"一个 cookie 文件都没找到"和"文件明明在、却读不到"这两种 ——
    后者最容易让人以为程序坏了。
    """
    if cookie_txt():
        return ""
    stray = []
    for d in cookie_scan_dirs():
        try:
            for n in os.listdir(d):
                if n.lower().endswith(".txt") and n not in stray:
                    stray.append(n)
        except Exception:
            continue
    if not stray:
        return ""
    return "%s 不是有效的 cookie 文件（需要 Netscape 格式）" % "、".join(stray[:2])


# ---------------------------------------------------------------------------
# 自动更新 Cookie：从本机浏览器直接导出
# ---------------------------------------------------------------------------
# 为什么需要：手工导出的 cookies.txt 会过期，登录态一变（改密码 / 换账号 /
# 重新登录）就整份失效，于是又变成 "Sign in to confirm you're not a bot"。
# 能从本机浏览器里现取现用，就不存在过期问题。
#
# 只读浏览器数据库、不碰浏览器的任何其他数据；取完立刻关闭句柄。
# Cookie 值在 Chrome 130+ / Edge 里是 DPAPI 加密的，v20 起前缀是 "v10"/"v11"，
# 需要按 AES-GCM 解开（见 _decrypt_cookie_value）。

BROWSER_PROFILE_DIRS = {
    # key 与 BROWSER_CHOICES 里的取值一致。
    # (浏览器安装目录名, 还要不要再拼一级)
    #   Chrome/Edge/Brave/Chromium: LocalAppData\<安装目录名>\User Data\<profile>
    #   Opera 不一样 —— 用户数据目录本身就是 profile 根，没有 User Data 这一级。
    "chrome":   ("Google\\Chrome\\User Data",),
    "edge":     ("Microsoft\\Edge\\User Data",),
    "brave":    ("BraveSoftware\\Brave-Browser\\User Data",),
    "chromium": ("Chromium\\User Data",),
    "opera":    (None,),
}


def _local_appdata():
    return os.environ.get("LOCALAPPDATA", "")


def _browser_user_data_root(browser):
    """浏览器"用户数据"根目录（里面才是 Default / Profile 1 这些）。"""
    spec = BROWSER_PROFILE_DIRS.get(browser)
    if not spec:
        return ""
    rel = spec[0]
    if rel:
        return os.path.join(_local_appdata(), rel)
    return os.path.join(os.environ.get("APPDATA", ""),
                        "Opera Software", "Opera Stable")


def browser_cookie_db(browser, profile="Default"):
    """找到浏览器的 Cookies 数据库文件路径；找不到返回空串。"""
    if not browser or browser == "none":
        return ""
    if browser not in BROWSER_PROFILE_DIRS:
        return ""
    root = _browser_user_data_root(browser)
    if not root or not os.path.isdir(root):
        return ""
    # Chrome 系的 Cookies 在 <root>/<profile>/Network/Cookies
    profs = [profile] if profile else []
    profs += [n for n in (list_browser_profiles(browser) or [])
              if n not in profs]
    for pr in profs:
        for cand in (os.path.join(root, pr, "Network", "Cookies"),
                     os.path.join(root, pr, "Cookies"),
                     os.path.join(root, "Network", "Cookies")):
            if os.path.isfile(cand):
                return cand
    return ""


def list_browser_profiles(browser):
    """列出这个浏览器下实际存在的用户配置名（Default / Profile 1 …）。"""
    root = _browser_user_data_root(browser)
    if not root or not os.path.isdir(root):
        return []
    out = []
    try:
        for n in os.listdir(root):
            p = os.path.join(root, n)
            if not os.path.isdir(p):
                continue
            if n == "Default" or n.startswith("Profile "):
                out.append(n)
    except Exception:
        pass
    return sorted(out)


def _local_state_key():
    """Chrome/Edge 的 Local State 里那串加密主密钥。"""
    spec_paths = []
    local = _local_appdata()
    appdata = os.environ.get("APPDATA", "")
    for rel in ("Google\\Chrome\\User Data", "Microsoft\\Edge\\User Data",
                "BraveSoftware\\Brave-Browser\\User Data", "Chromium\\User Data"):
        spec_paths.append(os.path.join(local, rel, "Local State"))
    spec_paths.append(os.path.join(appdata, "Opera Software", "Opera Stable",
                                   "Local State"))
    for p in spec_paths:
        if os.path.isfile(p):
            return p
    return ""


def _dpapi_unprotect(data):
    """用 Windows DPAPI 解密（浏览器 v10 之前的加密方式）。"""
    import ctypes
    from ctypes import wintypes

    class BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    blob = BLOB()
    buf = ctypes.create_string_buffer(data, len(data))
    blob.cbData = len(data)
    blob.pbData = ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))
    # ppszDataOut 要的是"指向指针的指针"，用 create_string_buffer 拿到的长度
    # 永远是 0 —— 必须用 POINTER(c_char) 传进去，最后再按返回值切。
    out = ctypes.POINTER(ctypes.c_char)()
    n = wintypes.DWORD(0)
    crypt = ctypes.windll.crypt32
    if not crypt.CryptUnprotectData(ctypes.byref(blob), None, None, None, None,
                                    0x01, ctypes.byref(out), ctypes.byref(n)):
        raise OSError("CryptUnprotectData failed")
    try:
        return ctypes.string_at(out, n.value)
    finally:
        try:
            ctypes.windll.kernel32.LocalFree(out)
        except Exception:
            pass


def _aes_gcm_decrypt(key, nonce, data, aad):
    """AES-256-GCM 解密（Chrome 80+ 的 cookie 加密）。"""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    return AESGCM(key).decrypt(nonce, data, aad)


def _decrypt_cookie_value(enc, key_file_cache={}):
    """解一个 cookie 值。

    Chrome 130 起 cookie 前缀是 v10 / v11（v10 = AES-GCM，v11 前缀带 SHA256
    主机名校验）；更早的是纯 DPAPI。两种都支持，解不开就返回 None（跳过这条）。
    """
    if not enc:
        return None
    if isinstance(enc, str):
        enc = enc.encode("utf-8", "replace")
    if enc[:3] in (b"v10", b"v11") and len(enc) > 3:
        try:
            ls = _local_state_key()
            if not ls or ls not in key_file_cache:
                import json
                with io.open(ls, "r", encoding="utf-8") as f:
                    d = json.load(f)
                b64 = d.get("os_crypt", {}).get("encrypted_key", "")
                if not b64:
                    return None
                # 前缀是 "DPAPI" 的 **base64 编码**（即字符串 "RFBBU"），
                # 不是字面量 "DPAP" —— 按字面量切会切错，导致 base64 解不开。
                # 稳妥做法：先还原成字节，再看开头是不是 b"DPAPI" 是才剥掉。
                raw = base64.b64decode(b64)
                if raw[:5] == b"DPAPI":
                    raw = raw[5:]
                key_file_cache[ls] = _dpapi_unprotect(raw)
            master = key_file_cache[ls]
            if len(master) < 32:
                return None
            key = hashlib.sha256(master).digest()
            nonce, ct = enc[3:15], enc[15:]
            if enc[:3] == b"v10":
                return _aes_gcm_decrypt(key, nonce, ct, None)
            # v11：aad 是 SHA256(host_key)
            aad = hashlib.sha256(ct[:32]).digest()
            return _aes_gcm_decrypt(key, nonce, ct[32:], aad)
        except Exception:
            return None
    try:
        return _dpapi_unprotect(enc).decode("utf-8", "replace")
    except Exception:
        return None


# 只导出 YouTube 的 cookie：最克制，避免把其他站点（搜索/广告/邮箱等）的登录态搬出去。
# 用子串匹配，所以 *.youtube.com（www / accounts / music 等）都覆盖到。
COOKIE_DOMAINS = (
    "youtube.com",
)


def _is_file_locked(path):
    """文件是否正被别的进程独占（Windows 错误 32 = 共享冲突）。

    用来判断"浏览器还开着" —— 浏览器运行时它的 Cookie 库是独占锁着的，
    任何复制方式都会失败（yt-dlp 自己的 cookiesfrombrowser 也一样报错）。
    先探一下就能给用户一句准话，而不是甩一个 PermissionError。
    """
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.windll.kernel32
        k32.CreateFileW.restype = wintypes.HANDLE
        k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                    wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                                    wintypes.HANDLE]
        share = 0x1 | 0x2 | 0x4
        h = k32.CreateFileW(path, 0x80000000, share, None, 3, 0x80, None)
        if h == ctypes.c_void_p(-1).value or h is None:
            return k32.GetLastError() == 32
        k32.CloseHandle(h)
        return False
    except Exception:
        return False


BROWSER_DISPLAY = {"chrome": "Chrome", "edge": "Edge", "brave": "Brave",
                   "opera": "Opera", "chromium": "Chromium"}


def _copy_sqlite_db(src, browser=""):
    """把 SQLite 文件复制到临时目录，浏览器正在运行时也能读到。

    浏览器运行时数据库是**独占**的，shutil.copy2 会直接 WinError 32。
    独占锁连"共享读"都开不了（不是共享位没给对），所以这里做不到后台读 ——
    只能在探测到锁时给一句"请先退出浏览器"的准话，而不是让人看不懂报错。
    """
    import shutil
    import tempfile
    if _is_file_locked(src):
        name = BROWSER_DISPLAY.get(browser, browser or "浏览器")
        raise IOError(32,
                      "%s 正在运行，Cookie 数据库被它独占锁定，无法读取。\n\n"
                      "请完全退出 %s（任务管理器里确认没有它的进程），"
                      "再点一次【更新】。\n"
                      "其实关浏览器不损失什么 —— Cookie 只读，不会动你的浏览数据。"
                      % (name, name))
    dst = os.path.join(tempfile.gettempdir(),
                       "lm_cookies_%d.db" % int(time.time() * 1000))
    shutil.copy2(src, dst)
    return dst


def export_browser_cookies(browser, profile="Default", domains=COOKIE_DOMAINS):
    """从浏览器导出 Netscape 格式的 cookie 文本。

    只读浏览器、只写自己的 Cookie 文件夹，绝不回写浏览器的任何数据。

    **优先走 yt-dlp 自带的提取器**（extract_cookies_from_browser）。这不是图省事：
    Chrome 127+ 用了 "app-bound encryption" —— 主密钥不是 DPAPI，而是被绑定到
    chrome.exe 这一个可执行文件上（靠一个校验服务）。任何**别的进程**（包括本程序
    自己手搓的解密）都解不开，只有 Chrome 自己能解。yt-dlp 内部已经处理了这件事，
    所以能用它的就用它的；自己那套 DPAPI + AES-GCM 只作为老版本浏览器的兜底。

    返回 (文本, 条数)；失败抛异常。
    """
    if not browser or browser == "none":
        raise ValueError("没有选择浏览器")
    if browser not in BROWSER_PROFILE_DIRS:
        raise ValueError("不支持的浏览器：%s" % browser)
    if not _browser_user_data_root(browser) or not os.path.isdir(
            _browser_user_data_root(browser)):
        raise FileNotFoundError("找不到 %s 的用户数据目录"
                                % BROWSER_DISPLAY.get(browser, browser))

    # 浏览器开着时数据库被独占锁着，先探一下，给一句人话。
    # 注意 Chrome 127+ 即便**关掉**浏览器也读不出来（app-bound 加密只认
    # chrome.exe 自己），所以这里把两条路都摆出来，并优先推荐"不用关浏览器"的插件法。
    db = browser_cookie_db(browser, profile)
    if db and _is_file_locked(db):
        name = BROWSER_DISPLAY.get(browser, browser)
        raise IOError(32,
                      "%s 正在运行，Cookie 数据库被它独占锁定，读不到。\n\n"
                      "【推荐 · 不用关浏览器】用浏览器插件导出：\n"
                      "  装「Get cookies.txt LOCALLY」→ 在**已登录的 YouTube 页面**"
                      "点导出 → 把 cookies.txt 放进 Cookie 文件夹即可。\n\n"
                      "【或者】完全退出 %s（任务管理器确认无进程）再点【从浏览器更新】。\n"
                      "（注意 Chrome 127+ 即便关闭浏览器也可能因加密读不到，"
                      "此时插件法更可靠。）" % (name, name))

    jar = _extract_cookies_ytdlp(browser, profile)
    if jar:
        return jar

    # yt-dlp 提取不了（老版本浏览器等），退回自己读库
    return _export_cookies_own_sqlite(browser, profile, domains)


def _chrome_exp_to_unix(raw):
    """Chrome 把 cookie 过期时间存成「Windows 微秒计数」（1601-01-01 起）。
    Netscape cookie 文件需要 Unix 秒（1970 起）。换算：微秒/1e6 - 11644473600。

    为兼容两条来源、且「不要转换两次」，这里按数量级自动判定：
      - 已经是合理 Unix 秒的值（<= 1e11，约合 5138 年以前）原样返回；
      - 超过 1e11 必为 Chrome 原始微秒值，做换算。
    0 / 空 / 非数字都按「会话 cookie」返回 0。
    """
    if not raw:
        return 0
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return 0
    if v <= 0:
        return 0
    if v > 10 ** 11:  # 正常 Unix 秒不可能这么大，必为微秒
        return int(v / 1000000) - 11644473600
    return v


def _host_in_domains(host, domains):
    """判断 cookie 的 host 是否落在白名单域名内。

    必须是**域名后缀匹配**，不能子串包含 —— 否则 `evil-youtube.com`、
    `youtube.com.attacker.com` 这类域名会被误判命中、把 cookie 泄露出去。
    host 可能带前导点（`.youtube.com` 表示该 cookie 对子域也有效），
    两边统一去掉前导点再比。
    """
    h = (host or "").lower().lstrip(".")
    for d in domains:
        d0 = d.lower().lstrip(".")
        if h == d0 or h.endswith("." + d0):
            return True
    return False


def _export_cookies_netscape(rows, domains):
    """把 cookie 行写成 Netscape 文本。

    统一在这里做 expires 换算（两条导出路径都走这里，只转换一次）。
    rows 里的 e 既可能是 Chrome 原始微秒值，也可能是已经换算好的 Unix 秒，
    由 _chrome_exp_to_unix 自动判定。域名按后缀匹配（见 _host_in_domains）。
    """
    out = ["# Netscape HTTP Cookie File",
           "# 由「歌词生成器」自动导出", ""]
    n = 0
    for host, name, value, cpath, e, secure in rows:
        if domains and not _host_in_domains(host, domains):
            continue
        inc = "TRUE" if host.startswith(".") else "FALSE"
        sec = "TRUE" if secure else "FALSE"
        out.append("\t".join([host, inc, cpath or "/", sec,
                              str(_chrome_exp_to_unix(e)),
                              name or "", value or ""]))
        n += 1
    if not n:
        raise ValueError("没有导出到任何 cookie（该浏览器可能没登录过）")
    return "\n".join(out) + "\n", n


def _extract_cookies_ytdlp(browser, profile):
    """用 yt-dlp 自带的浏览器 cookie 提取器（支持 app-bound 加密）。"""
    try:
        from yt_dlp.cookies import extract_cookies_from_browser
    except Exception:
        return None
    try:
        args = (browser, profile) if profile else (browser,)
        jar = extract_cookies_from_browser(*args)
    except Exception:
        return None
    try:
        rows = []
        for c in jar:
            rows.append((c.domain, c.name, c.value, c.path,
                         getattr(c, "expires", 0), bool(getattr(c, "secure", False))))
        return _export_cookies_netscape(rows, COOKIE_DOMAINS)
    except ValueError:
        raise
    except Exception:
        return None


def _export_cookies_own_sqlite(browser, profile="Default", domains=COOKIE_DOMAINS):
    """兜底路径：自己读浏览器数据库 + DPAPI/AES-GCM 解密。"""
    db = browser_cookie_db(browser, profile)
    if not db or not os.path.isfile(db):
        raise FileNotFoundError("找不到浏览器的 Cookie 数据库")

    import sqlite3

    # 复制一份再读：直接读运行中的库既可能被锁，也可能读到写了一半的状态。
    tmp = _copy_sqlite_db(db, browser)
    rows = []
    try:
        con = sqlite3.connect(tmp)
        try:
            cur = con.execute(
                "select host_key, name, value, encrypted_value, path, "
                "expires_utc, is_secure from cookies")
            rows = cur.fetchall()
        finally:
            con.close()
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass

    vals = []
    for host, name, value, enc, cpath, exp, secure in rows:
        if not host:
            continue
        val = value
        if not val and enc:
            val = _decrypt_cookie_value(enc)
            if val is None:
                continue
        if val is None:
            continue
        e = (int(exp / 1000000) - 11644473600) if exp else 0
        vals.append((host, name, val, cpath, e, bool(secure)))
    return _export_cookies_netscape(vals, domains)


def refresh_cookies_from_browser(browser="none", profile="Default"):
    """导出并写进 Cookie 文件夹，返回 (路径, 条数)。"""
    txt, n = export_browser_cookies(browser, profile)
    d = _cookie_dir
    if not d or not ensure_dir(d):
        raise OSError("Cookie 文件夹不可写：%s" % (d or "(未设置)"))
    out = os.path.join(d, "cookies.txt")
    # 先写临时文件再替换：中途失败不会把原来那份好的 cookie 弄坏
    tmp = out + ".tmp"
    with io.open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(txt)
    try:
        if os.path.exists(out):
            os.remove(out)
        os.replace(tmp, out)
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        raise
    return out, n


def default_cookie_dir():
    return COOKIE_DIR_DEFAULT


def set_cookie_dir(p):
    """切换 Cookie 目录（界面上的【更改】）。空值回到默认目录。"""
    global _cookie_dir
    p = (p or "").strip() or COOKIE_DIR_DEFAULT
    try:
        p = os.path.abspath(p)
    except Exception:
        pass
    _cookie_dir = p
    return ensure_dir(_cookie_dir)
DOWNLOAD_REL = os.path.join("out", "downloads")

# 画质档位 -> yt-dlp format selector
#   best  = 不挑，让 yt-dlp 自己选最优
#   audio = 只要音频（转 m4a）
#   h<数字> = 该分辨率封顶，数字由链接里实际可用的清晰度动态生成
FMT_BEST = "best"
FMT_AUDIO = "audio"
FMT_CHOICES = ((FMT_BEST, "最佳画质"),
               ("h2160", "最高 2160p"),
               ("h1440", "最高 1440p"),
               ("h1080", "最高 1080p"),
               ("h720", "最高 720p"),
               ("h480", "最高 480p"),
               (FMT_AUDIO, "仅音频（m4a）"))
# 下载页的「输出格式」选项。
#   原始/不转码 —— 保留 YouTube 给的流，只改封装（最快，画质无损）
#   其余与格式转换页共用同一批格式 —— 下载完直接转成目标格式，不用再跑一次转换。
# 转码类格式需要 ffmpeg 重编码，会明显变慢，这一点在界面上有提示。
CONTAINER_CHOICES = (("auto", "原始格式（不转码，最快）"),
                     ("mp4", "MP4（H.264 + AAC）"),
                     ("mkv", "MKV（H.264 + AAC）"),
                     ("mov", "MOV（H.264 + AAC）"),
                     ("avi", "AVI（MPEG-4 + MP3）"),
                     ("webm", "WebM（VP9 + Opus）"),
                     ("m4v", "M4V（H.264 + AAC）"),
                     ("flv", "FLV（H.264 + AAC）"),
                     ("wmv", "WMV（WMV2 + WMA）"),
                     ("ts", "TS（MPEG-TS）"),
                     ("mpg", "MPG（MPEG-2 + MP2）"),
                     ("3gp", "3GP（手机）"),
                     ("hevc", "HEVC / H.265（MP4，省空间）"),
                     ("vp9", "VP9（WebM，画质好）"),
                     ("av1", "AV1（MKV，最新压缩）"),
                     ("prores", "ProRes 422（剪辑代理，体积大）"),
                     ("mjpeg", "MJPEG（AVI，序列帧）"),
                     # 转音频
                     ("aonly-mp3", "仅音频 MP3"),
                     ("aonly-m4a", "仅音频 M4A（AAC）"),
                     ("aonly-flac", "仅音频 FLAC（无损）"),
                     ("aonly-wav", "仅音频 WAV（无损）"),
                     ("aonly-opus", "仅音频 Opus"),
                     ("aonly-ogg", "仅音频 OGG（Vorbis）"),
                     ("aonly-wma", "仅音频 WMA"),
                     ("aonly-ac3", "仅音频 AC3（杜比）"),
                     )
# 这些 key 是"仅音频"快捷写法，需要映射回 CONVERT_FORMATS 的 key
DL_AUDIO_KEYS = {"aonly-mp3": "mp3", "aonly-m4a": "m4a", "aonly-flac": "flac",
                 "aonly-wav": "wav", "aonly-opus": "opus", "aonly-ogg": "ogg",
                 "aonly-wma": "wma", "aonly-ac3": "ac3"}
BROWSER_CHOICES = (("none", "不读取"), ("chrome", "Chrome"), ("edge", "Edge"),
                   ("firefox", "Firefox"), ("brave", "Brave"), ("opera", "Opera"))

# ---------------------------------------------------------------------------
# 格式转换：常用视频/音频格式表 + ffmpeg 命令构造
# 每条：(key, 显示名, 类型 video/audio, 容器扩展名, 视频编码器, 音频编码器)
# 视频编码器为空字符串代表"音频格式"（转音频时丢掉画面 -vn）。
#
# 选型说明：
#   * 前几个是绝大多数人日常会转的（微信/抖音/手机/剪辑软件通用），排在最前。
#   * AV1 / VP9 / HEVC 是近年播放器与流媒体平台的主力，画质同码率更好。
#   * ProRes / DNxHD / MJPEG 是剪辑代理与调色的中间格式（体积大、不压缩）。
#   * 音频补了 AC3/DTS 兼容老设备，以及 APE/WavPack/ALAC 等无损/高保真格式。
# ---------------------------------------------------------------------------
CONVERT_REL = os.path.join("out", "converts")

CONVERT_FORMATS = (
    # ---- 常用：兼容性最好，优先出现在下拉前列 ----
    ("mp4",  "MP4（H.264 + AAC）",  "video", "mp4",  "libx264",    "aac"),
    ("mkv",  "MKV（H.264 + AAC）",  "video", "mkv",  "libx264",    "aac"),
    ("mov",  "MOV（H.264 + AAC）",  "video", "mov",  "libx264",    "aac"),
    ("avi",  "AVI（MPEG-4 + MP3）", "video", "avi",  "mpeg4",      "libmp3lame"),
    ("webm", "WebM（VP9 + Opus）",  "video", "webm", "libvpx-vp9", "libopus"),
    ("m4v",  "M4V（H.264 + AAC）",  "video", "m4v",  "libx264",    "aac"),
    ("flv",  "FLV（H.264 + AAC）",  "video", "flv",  "libx264",    "aac"),
    ("wmv",  "WMV（WMV2 + WMA）",   "video", "wmv",  "wmv2",       "wmav2"),
    ("ts",   "TS（MPEG-TS）",       "video", "ts",   "libx264",    "aac"),
    ("mpg",  "MPG（MPEG-2 + MP2）", "video", "mpg",  "mpeg2video", "mp2"),
    ("3gp",  "3GP（手机格式）",     "video", "3gp",  "libx264",    "aac"),
    ("ogv",  "OGV（Theora + Vorbis）", "video", "ogv", "libtheora", "libvorbis"),
    ("m2ts", "M2TS（蓝光原盘截取）", "video", "m2ts", "libx264",    "aac"),
    # ---- 画质优先：同码率画质更好，体积更小 ----
    ("hevc",  "HEVC / H.265（省空间）", "video", "mp4", "libx265",  "aac"),
    ("hevc-mkv", "HEVC / H.265（MKV）", "video", "mkv", "libx265",  "aac"),
    ("vp9",    "VP9（WebM 画质）",     "video", "webm", "libvpx-vp9", "libopus"),
    ("av1",    "AV1（最新压缩）",      "video", "mkv", "libaom-av1", "libopus"),
    ("av1-webm", "AV1（WebM）",        "video", "webm", "libaom-av1", "libopus"),
    # ---- 剪辑 / 调色的中间格式（体积大）----
    ("prores", "ProRes 422（剪辑代理）", "video", "mov", "prores",   "pcm_s16le"),
    ("mjpeg",  "MJPEG（序列帧）",       "video", "avi", "mjpeg",     "mp2"),
    ("dv",     "DV（DVcam，自动缩到 720×480）", "video", "avi", "dvvideo", "pcm_s16le"),
    # ---- 音频 ----
    ("mp3",  "MP3",                "audio", "mp3",  "",           "libmp3lame"),
    ("m4a",  "M4A（AAC）",         "audio", "m4a",  "",           "aac"),
    ("aac",  "AAC",                "audio", "aac",  "",           "aac"),
    ("flac", "FLAC（无损）",        "audio", "flac", "",           "flac"),
    ("wav",  "WAV（无损）",        "audio", "wav",  "",           "pcm_s16le"),
    ("ogg",  "OGG（Vorbis）",      "audio", "ogg",  "",           "libvorbis"),
    ("opus", "Opus",               "audio", "opus", "",           "libopus"),
    ("wma",  "WMA",                "audio", "wma",  "",           "wmav2"),
    ("ac3",  "AC3（杜比，老设备兼容）", "audio", "ac3", "",          "ac3"),
    ("wv",   "WavPack（高压缩无损）", "audio", "wv",  "",           "wavpack"),
    ("alac", "ALAC（Apple 无损）",  "audio", "m4a",  "",           "alac"),
    ("aiff", "AIFF（Apple 未压缩）", "audio", "aiff", "",          "pcm_s16be"),
    ("amr",  "AMR（移动语音）",     "audio", "amr",  "",           "libopencore_amrnb"),
)
CONVERT_FORMAT_KEYS = [f[0] for f in CONVERT_FORMATS]
CONVERT_FORMAT_MAP = dict((f[0], f) for f in CONVERT_FORMATS)
# 视频质量（CRF，越小越好）；部分编码器不支持 CRF，用 q:v 映射
CONVERT_VQUALITY = (("18", "高画质（体积大）"), ("23", "标准"), ("28", "高压缩（体积小）"))
# 支持 -crf 的编码器。libaom-av1 / libtheora / prores 的 CRF 取值范围与 x264 不同，
# 下面 CONVERT_VQ_RANGE 会按编码器换算。
CONVERT_VQ_CRFTWO = {"libx264", "libx265", "libvpx-vp9", "libaom-av1",
                     "libtheora", "prores"}
# 不支持 -crf、只能用 -q:v 的（值越小越好）
CONVERT_VQ_Q = {"18": "3", "23": "5", "28": "8"}
# 无损/中间格式：画质与码率都不可调，用户给什么都不该影响输出
CONVERT_VQ_FIXED = {"pcm_s16le", "pcm_s16be", "dvvideo", "mjpeg", "mpeg2video"}
# 各编码器的 CRF 有效区间：AV1/VP9 的数值区间和 x264 不通用，直接照搬会让
# 输出要么糊成一团、要么大得离谱。
CONVERT_VQ_RANGE = {
    "libx264": (0, 51), "libx265": (0, 51), "libvpx-vp9": (0, 63),
    "libaom-av1": (0, 63), "libtheora": (0, 63),
    # prores 的 -profile:v 用 0..4，越小越好；映射到 CRF 语义上取固定档
    "prores": None,
}
CONVERT_ARATE = (("320k", "320 kbps"), ("256k", "256 kbps"),
                 ("192k", "192 kbps"), ("128k", "128 kbps"))
CONVERT_LOSSLESS_AUDIO = ("flac", "pcm_s16le", "pcm_s16be", "alac", "wavpack")
# ffmpeg 的编码器名 -> 给人看的写法（用在"输出预览"那排胶囊上）
CONVERT_CODEC_LABEL = {
    "libx264": "H.264", "libx265": "H.265", "libvpx-vp9": "VP9",
    "libaom-av1": "AV1", "libtheora": "Theora", "prores": "ProRes",
    "mpeg4": "MPEG-4", "wmv2": "WMV2", "mpeg2video": "MPEG-2",
    "mjpeg": "MJPEG", "dvvideo": "DV",
    "aac": "AAC", "libmp3lame": "MP3", "libopus": "Opus", "libvorbis": "Vorbis",
    "flac": "FLAC", "alac": "ALAC", "ac3": "AC3", "wmav2": "WMA",
    "pcm_s16le": "PCM", "pcm_s16be": "PCM", "mp2": "MP2",
    "libopencore_amrnb": "AMR-NB", "wavpack": "WavPack",
}


def convert_fmt_label(key):
    f = CONVERT_FORMAT_MAP.get(key)
    return f[1] if f else key


def build_convert_cmd(ff, src, out, fmt_key, crf, arate, remux, has_video=None):
    """构造一条 ffmpeg 转换命令。

    remux=True 时只转封装（-c copy），最快但要求原编码被目标容器支持。
    视频目标：重编码视频（CRF 或 q:v）+ 重编码音频（码率）。
    音频目标：丢掉画面（-vn）+ 重编码音频（无损格式不指定码率）。

    has_video：源文件里有没有**真正的**画面流。带封面的 mp3 会含一张 png
    封面（ffmpeg 报成 Video），不算画面 —— 那种源转视频目标时要另外造画面，
    见 has_video=False 的分支。
    """
    spec = CONVERT_FORMAT_MAP.get(fmt_key)
    if spec is None:
        raise ValueError("未知格式: %s" % fmt_key)
    kind, ext, vcodec, acodec = spec[2], spec[3], spec[4], spec[5]
    cmd = [ff, "-y", "-i", src]
    if has_video is False and kind == "video" and not remux:
        # 源只有音频（可能带封面图），却要转成视频格式：ffmpeg 默认会把封面
        # 也当画面流塞进 mp4，直接报 -22。正确做法是丢掉封面，并用一张纯色
        # 画面按音频长度铺满 —— 否则用户转个歌也失败，且看不出原因。
        w = h = 1280
        try:
            ff2 = find_ffmpeg()          # 探测尺寸不必，用固定分辨率即可
            pw, ph, _ = probe_media_info(ff2, src)
            if pw and ph:
                w, h = pw, ph
        except Exception:
            pass
        cmd = [ff, "-y", "-f", "lavfi", "-i",
               "color=c=black:s=%dx%d:r=25" % (w, h), "-i", src,
               "-map", "0:v:0", "-map", "1:a:0", "-shortest"]
        cmd += _video_encode_args(vcodec, crf)
        cmd += _audio_encode_args(acodec, arate)
        cmd.append(out)
        return cmd
    if remux:
        # 音频目标就算只转封装也得丢画面，不然会把视频轨塞进 mp3 这种容器里
        cmd += (["-vn", "-c:a", "copy"] if kind == "audio" else ["-c", "copy"])
    elif kind == "video":
        cmd += _video_encode_args(vcodec, crf)
        cmd += _audio_encode_args(acodec, arate)
    else:  # 音频目标
        cmd += ["-vn"]
        cmd += _audio_encode_args(acodec, arate)
    cmd.append(out)
    return cmd


def _video_encode_args(vcodec, crf):
    """按编码器给出 -c:v 相关的参数。

    不同编码器对"质量"的表达方式完全不同，不能一律套 -crf：
      * libx264/libx265  ：-crf 0~51，越小越好
      * libvpx-vp9/libaom-av1/libtheora：需要 "-b:v 0 -crf"（恒定质量模式），
        否则 ffmpeg 会退化成 VBR 且忽略 crf；范围也是 0~63，与 x264 不同量纲，
        直接照搬用户的 18/23/28 会让画质明显偏糊。
      * prores           ：用 -profile:v（0~4，越小越好），不认 -crf
      * mpeg4/wmv2 等    ：不认 -crf，用 -q:v（1~31，越小越好）
      * mjpeg/dv/mpeg2   ：中间格式，画质与码率都不可调，什么都不给
    """
    v = str(crf or "23")
    if vcodec == "dvvideo":
        # DVcam 只有 720x480 / 720x576 两种合法尺寸，任意分辨率直接报
        # "Found no DV profile"。这里统一缩到 720x480（不缩放时保持原样更自然，
        # 但那就只能碰运气），并转成 DV 需要的 yuv411p。
        return ["-c:v", "dvvideo",
                "-vf", "scale=720:480:flags=bicubic,format=yuv411p"]
    if vcodec in ("mjpeg", "mpeg2video"):
        return ["-c:v", vcodec]          # 无损/中间格式，不给质量参数
    if vcodec == "prores":
        # 0=Proxy 1=LT 2=422 3=HQ 4=4444。4444 是 4:4:4 无损，
        # 体积是 422 的数倍，放到"高压缩"档会让用户以为程序有 bug，
        # 所以最高只到 HQ(3)。
        prof = {"18": "1", "23": "2", "28": "3"}.get(v, "2")
        # ProRes 只吃 10-bit 4:2:2，不指定就会报
        # "encoding with ProRes ... need YUV422P10 input" 然后转码失败。
        # 常见源是 8-bit yuv420p，必须显式插一个格式转换。
        return ["-c:v", "prores", "-profile:v", prof, "-vendor", "apl0",
                "-pix_fmt", "yuv422p10le"]
    # x264 / x265 的 CRF 就是 0~51，用户那三档 18/23/28 是照着这个量纲挑的，
    # 必须原样透传。早先把它们也塞进下面的区间换算，结果"高画质"变成 crf=0
    # （又慢又大）、"高压缩"变成 51（糊成一团），完全反了。
    if vcodec in ("libx264", "libx265"):
        return ["-c:v", vcodec, "-crf", v, "-preset", "medium"]
    if vcodec in CONVERT_VQ_CRFTWO:
        # 把 18/23/28 这三档映射到该编码器的 CRF 区间。
        # VP9 / AV1 的区间是 0~63，与 x264 不同量纲，直接照搬会明显偏糊。
        rng = CONVERT_VQ_RANGE.get(vcodec)
        if rng is None:
            cv = v
        else:
            lo, hi = rng
            frac = (28.0 - float(v)) / 10.0            # 18->1.0  23->0.5  28->0.0
            frac = max(0.0, min(1.0, frac))
            cv = str(int(round(hi - frac * (hi - lo))))
        if vcodec in ("libvpx-vp9", "libaom-av1"):
            # 恒定质量模式必须配 -b:v 0，否则 crf 被忽略
            return ["-c:v", vcodec, "-b:v", "0", "-crf", cv]
        if vcodec == "libtheora":
            q = CONVERT_VQ_Q.get(v, "5")
            return ["-c:v", vcodec, "-q:v", q]
    # 其余（mpeg4 / wmv2 / libxvid …）用 -q:v
    return ["-c:v", vcodec, "-q:v", CONVERT_VQ_Q.get(v, "5")]


def _audio_encode_args(acodec, arate):
    """按音频编码器给出参数。无损/PCM 类不设码率（设了也没意义）。"""
    if acodec == "libopencore_amrnb":
        # AMR-NB 只支持 8000 Hz 单声道，且码率有固定档位；
        # 不显式指定 -ar/-ac 时 ffmmpeg 会直接报 "Invalid argument"。
        return ["-c:a", acodec, "-ar", "8000", "-ac", "1", "-b:a", "12.2k"]
    if acodec in CONVERT_LOSSLESS_AUDIO:
        return ["-c:a", acodec]
    return ["-c:a", acodec, "-b:a", str(arate)]


def _ffmpeg_seconds(txt):
    """从 ffmpeg 的一行 stderr 里抠出当前处理时间（秒），用于进度条。
    兼容两种写法：time=HH:MM:SS.ms 和 out_time_ms=1234567。找不到返回 0.0。"""
    try:
        m = re.search(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)", txt)
        if m:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        m = re.search(r"out_time_ms=(\d+)", txt)
        if m:
            return int(m.group(1)) / 1000.0
    except Exception:
        pass
    return 0.0


def run_ffmpeg(cmd, job=None, log=None, proc_holder=None):
    """跑一条 ffmpeg 命令：边跑边把输出交给 log，进度写进 job。

    ffmpeg 没有进度回调接口，只有往 stderr 打 `time=...`。这里逐行读、逐行
    解析，所以能拿到真实进度；同时留一份尾部输出，失败时好定位原因。
    返回 (是否成功, 失败原因)。
    """
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT,
                             creationflags=_no_window())
    except Exception as e:
        return False, "启动 ffmpeg 失败：%s" % e
    if proc_holder is not None:
        proc_holder.append(p)
    tail = []
    try:
        while True:
            raw = p.stdout.readline()
            if not raw:
                break
            txt = raw.decode("utf-8", "replace").rstrip()
            if not txt:
                continue
            tail.append(txt)
            if len(tail) > 12:
                del tail[:-12]
            if job is not None:
                sec = _ffmpeg_seconds(txt)
                if sec > 0:
                    job.done = sec
            if log is not None:
                log(txt)
            if job is not None and job.canceled:
                try:
                    p.kill()
                except Exception:
                    pass
                break
    except Exception:
        pass
    try:
        p.stdout.close()
    except Exception:
        pass
    rc = p.wait()
    if rc == 0:
        return True, ""
    # 从尾部挑一句"像是在报错"的行 —— 比把整屏输出糊给用户有用得多
    why = ""
    for ln in reversed(tail):
        low = ln.lower()
        if ("error" in low or "invalid" in low or "unsupported" in low
                or "no such" in low or "failed" in low or "not found" in low
                or "无法" in ln or "错误" in ln or "失败" in ln):
            why = ln.strip()
            break
    if not why and tail:
        why = tail[-1].strip()
    return False, (why[:200] or ("ffmpeg 退出码 %d" % rc))



def fmt_label(key):
    """把格式键变成人看的文字。"""
    if key == FMT_AUDIO:
        return "仅音频（m4a）"
    if key == FMT_BEST:
        return "最佳画质"
    h = str(key).lstrip("h")
    return "最高 %sp" % h if h.isdigit() else str(key)


def ensure_cookie_dir():
    """Cookie 文件夹：放手动导出的 cookies.txt。"""
    return ensure_dir(_cookie_dir)


def ytdlp_module():
    """拿到 yt_dlp 模块；没装返回 None（界面据此提示，不抛异常）。"""
    try:
        import yt_dlp
        return yt_dlp
    except Exception:
        return None


def ytdlp_version():
    m = ytdlp_module()
    if m is None:
        # 模块没进来（源码模式下依赖没装好）就退到随程序携带的 yt-dlp.exe，
        # 至少还能下 —— 拷走即用的兜底。
        return _bundled_ytdlp_version()
    return getattr(getattr(m, "version", None), "__version__",
                  None) or str(getattr(m, "version", "") or "")


def _bundled_ytdlp_exe():
    """随程序携带的 yt-dlp.exe 路径，没有返回空串。"""
    return bundled_runtime("yt-dlp.exe")


def _bundled_ytdlp_version():
    """问 yt-dlp.exe 自己要版本号；拿不到就返回空串（不抛异常）。"""
    exe = _bundled_ytdlp_exe()
    if not exe:
        return ""
    try:
        import subprocess
        r = subprocess.run([exe, "--version"], capture_output=True,
                           timeout=30, creationflags=_no_window())
        out = (r.stdout or b"").decode("utf-8", "replace").strip()
        return out.splitlines()[0].strip() if out else ""
    except Exception:
        return ""


def ytdlp_has_backend():
    """有没有可用的下载后端：内置 Python 模块，或随程序携带的 yt-dlp.exe。"""
    return bool(ytdlp_module()) or bool(_bundled_ytdlp_exe())


# --- YouTube 的 n challenge：要一个 JS 运行时 + 求解器脚本 -----------------
# yt-dlp 2026 起，YouTube 会下发 "n 参数" 挑战，不解就只给少数低清格式
# （报 `n challenge solving failed`）。解它需要：
#   ① 一个 JS 运行时（deno 是官方默认，node 要显式指定路径）；
#   ② yt-dlp-ejs 求解器（已随本程序一起装，见 .deps-ejs）。
JS_RUNTIME_NAMES = ("deno", "node", "nodejs", "bun", "qjs")

# 随程序一起走的 JS 运行时 / yt-dlp 可执行文件（都在程序目录的 runtime/ 下）。
# 全部用**相对路径**定位，所以整个文件夹拷到别的电脑照样能用 —— 这是
# "拷走即用"的前提：绝不能依赖 C:\Program Files\nodejs 或任何用户目录。
BUNDLED_RUNTIME = os.path.join(_HERE, "runtime")


def bundled_runtime(name):
    """取随程序携带的可执行文件（runtime/node.exe、runtime/yt-dlp.exe）。

    只认程序自己旁边的 runtime/，不写死任何绝对路径 —— 换台电脑、换个盘符
    都还能找到。文件不存在返回空串。
    """
    try:
        p = os.path.join(BUNDLED_RUNTIME, name)
        return p if os.path.isfile(p) else ""
    except Exception:
        return ""


def _js_runtime_candidates():
    """可能的 JS 运行时可执行文件路径，按优先级排。

    顺序很重要：**先找随程序携带的 runtime/**，再退到系统里找。
    自带的那个跟着文件夹走，换电脑也一定在；系统里那个（比如本机的
    ``C:\\Program Files\\nodejs`` 或 WorkBuddy 的隔离 node）到了别的电脑上
    根本不存在，所以只能当兜底。
    """
    from shutil import which
    import glob
    seen = set()

    # ① 随程序携带的（便携，优先）
    for name in ("node.exe", "deno.exe", "bun.exe", "qjs.exe", "node", "deno"):
        p = bundled_runtime(name)
        if p and p.lower() not in seen:
            seen.add(p.lower())
            yield p

    # ② PATH 里的
    for name in JS_RUNTIME_NAMES:
        try:
            p = which(name)
        except Exception:
            p = None
        if p and os.path.isfile(p):
            p = os.path.abspath(p)
            if p.lower() not in seen:
                seen.add(p.lower())
                yield p
    home = os.path.expanduser("~")
    appdata = os.environ.get("APPDATA", "")
    local = os.environ.get("LOCALAPPDATA", "")
    pats = [
        r"C:\Program Files\nodejs\node.exe",
        r"C:\Program Files (x86)\nodejs\node.exe",
        r"C:\nodejs\node.exe",
        os.path.join(home, ".workbuddy", "binaries", "node", "versions", "*", "node.exe"),
        os.path.join(appdata, "nvm", "node.exe"),
        os.path.join(appdata, "nvm", "v*", "node.exe"),
        os.path.join(home, ".nvm", "v*", "node.exe"),
        os.path.join(home, ".fnm", "node-versions", "*", "node.exe"),
        os.path.join(home, ".volta", "bin", "node.exe"),
        os.path.join(local, "fnm", "node-versions", "*", "node.exe"),
    ]
    for pat in pats:
        if "*" in pat:
            for gp in glob.glob(pat):
                gp = os.path.abspath(gp)
                if os.path.isfile(gp) and gp.lower() not in seen:
                    seen.add(gp.lower())
                    yield gp
        else:
            ap = os.path.abspath(pat)
            if os.path.isfile(ap) and ap.lower() not in seen:
                seen.add(ap.lower())
                yield ap


def _js_key_for(path):
    base = os.path.basename(path).lower()
    if base in ("node.exe", "node", "nodejs.exe", "nodejs"):
        return "node"
    if base.startswith("deno"):
        return "deno"
    if base.startswith("bun"):
        return "bun"
    if base.startswith("qjs"):
        return "qjs"
    return "node"


def _js_runtime_works(path):
    """轻量可用性校验：文件存在即可。

    不做 `node --version` 这类子进程校验——冻结后的 exe 起子进程容易因继承的
    环境变量 / 窗口标志出问题而误判成"不可用"，反而把明明在的运行时漏掉。
    候选路径本身已带 node/deno 等名字，文件存在基本就等于可用。
    """
    return bool(path) and os.path.isfile(path)


_JS_RUNTIME_CACHE = None  # 查过一次就缓存；中途新装运行时需重启生效


def find_js_runtime():
    """找一个可用的 JS 运行时。返回 (名字, 可执行文件路径)，找不到返回 ("", "")。

    找不到也**不算错误** —— 只是 YouTube 的部分清晰度会缺，别的站点不受影响。
    """
    global _JS_RUNTIME_CACHE
    if _JS_RUNTIME_CACHE is not None:
        return _JS_RUNTIME_CACHE
    found = ("", "")
    for p in _js_runtime_candidates():
        if _js_runtime_works(p):
            found = (_js_key_for(p), p)
            break
    _JS_RUNTIME_CACHE = found
    return found


def js_runtime_note():
    """界面用的一句话说明。"""
    name, path = find_js_runtime()
    if not name:
        return "未找到 JS 运行时（deno / node）；YouTube 可能只给低清格式"
    return "JS 运行时 %s" % name


def _prepare_ytdlp_env():
    """清理会影响 yt-dlp 起子进程的环境变量。

    踩过的坑：某些工具会往环境里塞 `NODE_OPTIONS=--require=...shim.cjs`，
    而 yt-dlp 起 node 时会**继承**它 —— shim 一旦限制文件读取，求解器就读不到
    自己的 JS 文件，报 `Access to this API has been restricted`，
    表现成"n challenge 解不开"，但真凶跟 yt-dlp 一点关系都没有。
    """
    for k in ("NODE_OPTIONS", "NODE_PATH"):
        try:
            os.environ.pop(k, None)
        except Exception:
            pass


def browser_cookie_installed(browser):
    """本机这个浏览器的用户目录在不在 —— 界面上据此提示"可用/没装"。"""
    if not browser or browser == "none":
        return False
    local = os.environ.get("LOCALAPPDATA", "")
    appdata = os.environ.get("APPDATA", "")
    pats = {
        "chrome":  [os.path.join(local, "Google", "Chrome", "User Data")],
        "edge":    [os.path.join(local, "Microsoft", "Edge", "User Data")],
        "brave":   [os.path.join(local, "BraveSoftware", "Brave-Browser",
                                  "User Data")],
        "opera":   [os.path.join(appdata, "Opera Software", "Opera Stable")],
        "firefox": [os.path.join(appdata, "Mozilla", "Firefox", "Profiles")],
    }
    for p in pats.get(browser, []):
        if p and os.path.isdir(p):
            return True
    return False


def _ytdlp_fmt(quality):
    """画质档位 -> format 字符串。支持 h<数字> 这种动态生成的档位。"""
    if quality == FMT_AUDIO:
        return "bestaudio/best"
    h = str(quality or "").lstrip("h")
    if h.isdigit():
        return "bv*[height<=%s]+ba/b[height<=%s]/b" % (h, h)
    return "bv*+ba/b"


def ytdlp_heights(info):
    """从解析结果里挑出这个链接**真实可用**的清晰度，降序。

    用户要的是"链接有什么就能下什么"，而不是写死几档。
    只统计带视频轨的格式，否则纯音频格式会把 height 报成 None 之外的值。
    """
    out = set()
    for f in (info or {}).get("formats") or []:
        try:
            if f.get("vcodec") in (None, "none"):
                continue
            h = f.get("height")
            if h:
                out.add(int(h))
        except Exception:
            continue
    return sorted(out, reverse=True)


def ytdlp_format_choices(info):
    """按链接实际可用的清晰度生成下拉项。

    只保留"有意义"的档位：从最高往下，相邻档位差别太小就并掉
    （YouTube 会给 1080/1080p60/1080 HDR 一堆重复高度）。
    """
    hs = ytdlp_heights(info)
    opts = [(FMT_BEST, "最佳画质")]
    for h in hs:
        opts.append(("h%d" % h, "最高 %dp" % h))
    opts.append((FMT_AUDIO, "仅音频（m4a）"))
    return opts if len(opts) > 2 else list(FMT_CHOICES)


def _ytdlp_logger(log):
    """把 yt-dlp 的日志接到我们自己的回调上。"""

    class _L(object):
        def debug(self, msg):
            pass                      # yt-dlp 的 debug 太吵，不往界面刷

        def info(self, msg):
            if msg:
                log(msg)

        def warning(self, msg):
            if msg:
                log("[警告] " + msg)

        def error(self, msg):
            if msg:
                log("[错误] " + msg)

    return _L()


def _ytdlp_opts(outdir, log, browser="none", template=None):
    _prepare_ytdlp_env()
    opts = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "logger": _ytdlp_logger(log),
        "outtmpl": build_outtmpl(outdir, template),
        # 重试次数与超时：默认值（重试 10 次、socket 不超时）在网络不畅时
        # 会把"点一下没反应"拖成一两分钟，用户只能干等或杀进程。
        # 这里收到 3 次，并给 socket 20 秒上限：够应付正常的网络抖动，
        # 又能让"真的连不上"快速失败、把错误显示出来让用户换链接/开代理。
        "retries": 3,
        "fragment_retries": 3,
        "extractor_retries": 2,
        "socket_timeout": 20,
        "ignoreerrors": "only_download",
        "restrictfilenames": False,
        "windowsfilenames": True,
    }
    # 喂给 yt-dlp 一个 JS 运行时，否则 YouTube 的 n challenge 解不开
    _js, _jspath = find_js_runtime()
    if _js:
        opts["js_runtimes"] = {_js: {"path": _jspath}}
    try:                              # 合流 / 转码要 ffmpeg，没有就只给单文件
        ff = find_ffmpeg()
        if ff:
            opts["ffmpeg_location"] = os.path.dirname(ff)
    except Exception:
        pass
    ct = cookie_txt()
    if ct:
        # 有 cookie 文件就用文件，**不再叠一层浏览器读取**。
        # 踩过的坑：浏览器那步失败会直接抛 CookieLoadError 把整个解析打断
        # （Chrome 在运行时必失败），而用户明明已经给了可靠的 cookies.txt。
        # 两条路同时开只会让报错信息互相淹没。
        opts["cookiefile"] = ct
    elif browser and browser != "none":
        opts["cookiesfrombrowser"] = (browser,)
    return opts


# =====================================================================
# 下载文件名模板
# =====================================================================
# 原来 outtmpl 写死成 "%(title).80s [%(id)s].%(ext)s"，用户想改名只能下完
# 再去资源管理器里重命名。这里把文件名做成模板：下载前就能定，也能预览。
#
# 可用变量（和 yt-dlp 的 outtmpl 语法一致）：
#   %(title)s   标题      %(id)s      视频 ID
#   %(uploader)s / %(artist)s  作者
#   %(playlist)s 所属播放列表  %(playlist_index)s 列表内序号
#   %(ext)s     扩展名    %(upload_date)s 上传日期 YYYYMMDD
#   %(duration)s 时长秒     %(resolution)s 分辨率
# 以及自定义字面量：{n}（本条在队列里的序号，从 1 开始）
#
# 保留 %(id)s 是有意的：ID 唯一，标题可能重复或含平台垃圾字符；
# 用户主动删掉它，那是他的选择。
DEFAULT_OUTTMPL = "%(title).80s [%(id)s].%(ext)s"

# 界面上的命名预设：点一下 = 换一整套命名（覆盖原来的），不是往上叠占位符。
# (显示名, 模板)
#
# 预设里**一律不写 %(ext)s**：扩展名由程序按实际容器/格式自动补。
# 写死的话换格式（mp4→mkv）时文件名会带着错的扩展名，yt-dlp 也会再拼一次。
TPL_PRESETS = (
    ("标题 + 视频ID", "%(title).80s [%(id)s]"),
    ("标题 - 作者", "%(title)s - %(uploader)s"),
    ("作者 - 标题", "%(uploader)s - %(title)s"),
    ("标题 + 序号", "%(playlist_index)s. %(title)s"),
    ("日期 - 标题", "%(upload_date)s - %(title)s"),
    ("仅标题", "%(title)s"),
    ("标题 + 分辨率", "%(title)s [%(resolution)s]"),
    ("序号_标题_作者", "%(playlist_index)s_%(title)s_%(uploader)s"),
)

# 界面上"插入变量"按钮的清单：(占位符, 说明)
TPL_VARS = (
    ("%(title)s", "标题"),
    ("%(uploader)s", "UP主"),
    ("%(id)s", "视频ID"),
    ("%(upload_date)s", "日期"),
    ("%(duration)s", "时长秒"),
    ("%(resolution)s", "分辨率"),
    ("%(playlist)s", "播放列表"),
    ("%(playlist_index)s", "列表序号"),
    ("%(artist)s", "作者"),
    ("{n}", "队列序号"),
)


def build_outtmpl(outdir, template=None):
    """把用户模板和下载目录拼成 yt-dlp 的 outtmpl。

    模板为空就用默认；用户写了模板就照用，但仍然把目录部分固定在 outdir，
    免得一个 ../.. 把文件写到别处去。

    **缺 %(ext)s 就自动补上**。yt-dlp 只有在模板里写了 %(ext)s（或占位符本身
    以点结尾）才会产出扩展名；只写 `%(title)s` 的话，落盘文件**完全没有扩展名**
    —— 用户看到的是一个没有后缀的文件，双击打不开、播放器不认，还得自己
    改名才能用（实测抖音下载就是这样）。所以这里主动补，别指望用户记得写。
    """
    tpl = (template or "").strip() or DEFAULT_OUTTMPL
    # 目录由 outdir 决定：模板里若含路径分隔符，只取最后一段文件名部分。
    # 这样用户可以写 "%(title)s - %(uploader)s" 而不必关心存哪。
    name_part = os.path.basename(tpl.replace("\\", "/"))
    if not name_part or name_part in (".", ".."):
        name_part = DEFAULT_OUTTMPL
    # 补扩展名：模板里既没有 %(ext)s 也没占位符以点收尾时才补，
    # 避免把用户特意写的 "%(title)s.v2" 变成 "...v2.mp4" 之外的意外结果。
    if "%(ext)" not in name_part and not name_part.endswith("."):
        name_part += ".%(ext)s"
    return os.path.join(outdir, name_part)


def preview_filename(template, info, ext="mp4", index=1):
    """按模板和已解析信息预览最终文件名（纯本地计算，不发网络请求）。

    只为"让用户下之前就看见叫什么"，所以对未知/异常字段一律退化成占位符，
    绝不能因为少一个字段就抛异常把界面搞崩。
    """
    tpl = (template or "").strip() or DEFAULT_OUTTMPL
    d = {}
    if isinstance(info, dict):
        uploader = info.get("uploader") or info.get("channel") or ""
        d["title"] = info.get("title") or "未命名"
        d["id"] = info.get("id") or ""
        d["uploader"] = uploader
        d["artist"] = uploader
        d["playlist"] = info.get("playlist") or ""
        d["playlist_index"] = info.get("playlist_index") or index or 1
        d["upload_date"] = (info.get("upload_date") or "")
        d["duration"] = info.get("duration") or 0
        d["resolution"] = info.get("resolution") or ""
    else:
        d.update({"title": "未命名", "id": "", "uploader": "", "artist": "",
                  "playlist": "", "playlist_index": index, "upload_date": "",
                  "duration": 0, "resolution": ""})
    d["ext"] = (ext or "mp4").lstrip(".")
    d["n"] = index

    out = _expand_template(tpl, d)
    # Windows 文件名非法字符 + 结尾的点/空格，这些会让 os.rename 直接失败
    out = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", out)
    out = out.rstrip(". ")
    if not out:
        out = "%s.%s" % (d["title"], d["ext"])
    # 只在**没有**扩展名时才补。注意两种情况都要判：
    #   整串就是扩展名     （模板写 %(ext)s     -> "mp4"）
    #   以 .ext 结尾        （模板写 x.%(ext)s   -> "x.mp4"）
    # 早先只判了后者，于是 %(ext)s 会变成 "mp4.mp4"。
    _e = d["ext"]
    if out.lower() == _e.lower() or out.lower().endswith("." + _e.lower()):
        return out
    return "%s.%s" % (out, _e)


def _expand_template(tpl, d):
    """展开 %(key)s 与 %(key).Ns 两种占位符形式。

    yt-dlp 的 outtmpl 支持 %(title).80s 这种"截断到 80 字符"的语法，
    内置默认模板就用了它，所以这里必须同样支持，否则默认模板会把
    ".80s" 原样留在文件名里。未知 key 原样保留，让用户能看出写错了什么。
    """
    def rep(m):
        key, trunc = m.group(1), m.group(2)
        val = d.get(key)
        if val is None:
            return m.group(0)                 # 未知 key：原样留着
        val = str(val)
        if trunc:                             # .Ns -> 截断到 N 个字符
            try:
                val = val[:max(1, int(float(trunc)))]
            except Exception:
                pass
        return val
    out = re.sub(r"%\(([^)]+)\)(?:\.([0-9]+))?s", rep, tpl)
    # {n}：本条在队列里的序号
    out = out.replace("{n}", str(d.get("n", 1)))
    return out


_URL_RE = re.compile(r"https?://[^\s<>\"'，。；、）)】\]]+", re.I)


def extract_urls(text):
    """从任意文本里挑出所有 http(s) 链接，按出现顺序去重。

    为什么需要：用户从浏览器 / 分享按钮复制的往往不是纯链接，而是整段文案
    （"【标题】 https://... 快来观看"）。整段塞给 yt-dlp 只会报
    "Unsupported URL"，用户根本不知道该删掉哪一段。
    顺带支持"一次贴一串链接"→ 变成下载队列。
    """
    out = []
    for m in _URL_RE.findall(str(text or "")):
        u = m.rstrip(".,;:!?)）】")
        if u and u not in out:
            out.append(u)
    return out


# 短链域名：这些域名本身不带视频 ID，必须跟随跳转才能拿到真实地址。
# 抖音分享出来的就是 v.douyin.com/xxxxx 这种，而 yt-dlp 的 DouyinIE 只认
# www.douyin.com/video/<数字> —— 不展开就报 Unsupported URL。
SHORT_LINK_HOSTS = (
    "v.douyin.com", "douyin.com", "vm.tiktok.com", "vt.tiktok.com",
    "ixigua.com", "xhslink.com", "b23.tv", "dwz.cn", "t.cn",
)


def _host_is_short(netloc):
    """判断一个主机名是不是短链域名。单独抽出来便于自检直接验证。"""
    host = (netloc or "").lower().lstrip("www.")
    for h in SHORT_LINK_HOSTS:
        b = h.lstrip("www.")
        if host == b or host.endswith("." + b):
            return True
    return False


def expand_short_url(url, timeout=15):
    """跟随跳转把短链还原成真实地址。

    返回还原后的 URL；失败或无需还原时返回原 URL（绝不抛异常、绝不返回空）。
    只做一次 HEAD/GET 跳转跟随，不下载任何内容。
    """
    u = (url or "").strip()
    if not u:
        return ""
    try:
        host = urllib.parse.urlparse(u).netloc.lower()
    except Exception:
        return u
    # 只对已知短链域名做展开，避免对每个链接都多一次网络往返
    if not _host_is_short(host):
        return u
    try:
        req = urllib.request.Request(u, method="GET", headers={
            "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                           "AppleWebKit/537.36 (KHTML, like Gecko) "
                           "Chrome/120.0 Safari/537.36"),
        })
        # youtube_dl/yt-dlp 有现成的 opener 时优先用它（能带上 cookie）。
        # 这里退化为标准库，够用且不引入依赖。
        with urllib.request.urlopen(req, timeout=timeout) as r:
            final = r.geturl()
        return final or u
    except Exception:
        return u     # 展开失败就用原链接，让 yt-dlp 自己试，别把路堵死


def _eta(sec):
    try:
        sec = max(0, int(sec))
    except Exception:
        return "--:--"
    h, m, s = sec // 3600, (sec % 3600) // 60, sec % 60
    return ("%d:%02d:%02d" % (h, m, s)) if h else ("%d:%02d" % (m, s))


class _Job(object):
    """一次下载任务的句柄：查进度 + 中途取消。"""

    def __init__(self):
        self.canceled = False
        self.total = 0
        self.done = 0
        self.speed = 0.0
        self.eta = 0
        self.filename = ""

    def cancel(self):
        self.canceled = True

    def ratio(self):
        return (self.done / float(self.total)) if self.total else 0.0


def _run_with_cancel(fn, job, poll=0.25, hard_limit=90):
    """在后台线程里跑 fn，主线程轮询 job.canceled，从而真正做到"立刻取消"。

    为什么不用 hook：YouTube 解析最慢的一段是解 n 挑战（在子进程里跑 JS），
    那期间 yt-dlp 一次 progress_hook 都不回调，hook 根本收不到取消信号 ——
    实测点了取消还要再等 18 秒。改成由调用方（worker）自己起线程跑、主线程
    每 0.25 秒看一眼标记，才是真的想停就停。

    hard_limit 是兜底：万一线程卡在不可中断的系统调用里，至少调用方不会
    被永久拖住（子线程是 daemon，进程退出即回收）。
    """
    import threading
    box = {}

    def target():
        try:
            box["val"] = fn()
        except BaseException as e:      # noqa: BLE001 - 原样转交给调用方
            box["err"] = e

    th = threading.Thread(target=target, daemon=True)
    th.start()
    waited = 0.0
    while th.is_alive():
        if job is not None and getattr(job, "canceled", False):
            raise _Canceled()
        th.join(poll)
        waited += poll
        if waited >= hard_limit:
            raise _Canceled()
    if "err" in box:
        raise box["err"]
    return box.get("val")


def ytdlp_probe(url, browser="none", log=print, job=None):
    """解析链接，只取元信息不下载。返回 (ok, info_or_error)。

    job 传进来才能响应取消：解析要访问多个页面（主页面 + 播放器配置 +
    player API），YouTube 慢的时候十几秒很正常，而这段时间里界面毫无反馈
    —— 用户只能以为程序卡死了。这里挂一个 progress hook，yt-dlp 每次做完
    一个步骤回调一次，canceled 就抛 _Canceled 打断，不至于等它跑完。
    """
    m = ytdlp_module()
    if m is None:
        return False, "未内置 yt-dlp，无法解析"
    url = (url or "").strip()
    if not url:
        return False, "请先填写视频链接"

    # 短链先还原：抖音分享的 https://v.douyin.com/xxxxx 里没有视频 ID，
    # DouyinIE 只认 www.douyin.com/video/<数字>，不展开就 Unsupported URL。
    # 展开失败也无所谓 —— 用原链接让 yt-dlp 再试一次。
    expanded = expand_short_url(url)
    if expanded and expanded != url:
        try:
            log("短链已还原：%s" % expanded)
        except Exception:
            pass
        url = expanded

    opts = _ytdlp_opts(os.path.join(_HERE, DOWNLOAD_REL), log, browser)
    opts["skip_download"] = True
    opts["noplaylist"] = False
    # 解析阶段**必须**让错误抛出来。
    # 下载用的 "ignoreerrors": "only_download" 会把"提取阶段"的异常一起吞掉，
    # extract_info 于是返回 None，界面上只剩一句"没有解析到任何信息"——
    # 而 yt-dlp 其实已经把真实原因（多半是 Sign in to confirm you're not a bot）
    # 打进日志了，用户却看不到，只能以为程序坏了。解析要的就是"成功或明确原因"。
    opts["ignoreerrors"] = False

    if job is not None:
        def _cancel_check(_d=None):
            if getattr(job, "canceled", False):
                raise _Canceled()

        opts["progress_hooks"] = [_cancel_check]
        # progress_hook 只在 yt-dlp **主动回调**时才跑，而 YouTube 解析最慢的
        # 那一段是解 n 挑战（跑 JS），期间一次都不回调 —— 实测点取消后仍等了
        # 18 秒，用户体感等于"取消没反应"。所以再挂一个 postprocessor hook：
        # 每个处理阶段切换时至少有机会检查一次取消。
        opts["postprocessor_hooks"] = [lambda _pp: _cancel_check()]

    try:
        with m.YoutubeDL(opts) as ydl:
            info = _run_with_cancel(
                lambda: ydl.extract_info(url, download=False), job)
    except _Canceled:
        return False, "已取消"
    except Exception as e:
        msg = str(e)
        # 错误文本通常很长（带 wiki 链接），界面上一行放不下；只留有用的前半段
        clean = msg.split(" See  https://")[0].strip() or msg.split(". See ")[0].strip()
        # 抖音现在强制要"新鲜 cookie"（2025 年起收紧），yt-dlp 原话是
        # "Fresh cookies (not necessarily logged in) are needed"，
        # 用户看到这句完全不知道下一步做什么 —— 补一段可操作的说明。
        if "Fresh cookies" in msg or ("Douyin" in msg and "403" in msg):
            clean = ("抖音现在要求有效的 Cookie 才能下载。\n\n"
                     "做法：用 Chrome 打开 douyin.com 并**登录**，然后：\n"
                     "  1. 装浏览器插件「Get cookies.txt LOCALLY」\n"
                     "  2. 在**已登录的抖音页面**点插件图标 → Export\n"
                     "  3. 把导出的 cookies.txt 放进 Cookie 文件夹\n\n"
                     "（注意：插件只导当前页面的 cookie，"
                     "所以必须在 douyin.com 页面上导出）\n\n"
                     "原始原因：" + clean)
        elif "Unsupported URL" in msg:
            clean = ("这个链接 yt-dlp 不认：%s\n\n"
                     "如果这是短链（如 v.douyin.com/xxx），"
                     "请确认网络能访问该站点——短链需要联网还原成真实地址。"
                     % (url[:60],))
        return False, clean
    if not info:
        return False, "没有解析到任何信息（yt-dlp 返回了空结果）"
    if info.get("_type") == "playlist" or info.get("entries") is not None:
        items = [x for x in (info.get("entries") or []) if x]
        return True, {"_playlist": True, "title": info.get("title") or "播放列表",
                      "uploader": info.get("uploader") or "",
                      "count": len(items), "duration": None}
    return True, info


class _Canceled(Exception):
    """用户中途点了取消 —— 用异常打断 yt-dlp 的下载循环。"""


def ytdlp_download(url, outdir, quality="best", container="auto", browser="none",
                   subs=False, thumb=False, playlist=False,
                   log=print, job=None, template=None):
    """真正下载。job 传进来就能查进度、还能中途取消。返回 (ok, 说明)。

    container 决定下载完之后得到什么：
      * "auto" 或空 —— 只换封装（`-c copy`），不重编码，最快、画质无损。
      * 其他值     —— 先按 YouTube 原始格式下载，再用 ffmpeg 转成目标格式。
        这样转码路径与「格式转换」页完全一致（共用 build_convert_cmd），
        不必在下载里另写一套编码参数、也不会两边行为不一致。

    先下后转而不是让 yt-dlp 直接转：yt-dlp 的 merge_output_format 只会换封装，
    想要 H.265/AV1/ProRes 这类真正的重编码，它做不到。
    """
    m = ytdlp_module()
    if m is None:
        return False, "未内置 yt-dlp"
    url = (url or "").strip()
    if not url:
        return False, "请先填写视频链接"
    # 直接下载（没先解析）时也要还原短链，理由同 ytdlp_probe。
    _exp = expand_short_url(url)
    if _exp and _exp != url:
        try:
            log("短链已还原：%s" % _exp)
        except Exception:
            pass
        url = _exp
    ensure_dir(outdir)

    # 目标格式：auto = 不转码；其余映射到 CONVERT_FORMATS 的 key
    target = (container or "auto").strip()
    if target in ("", "auto"):
        target = None
    fmt_key = DL_AUDIO_KEYS.get(target, target)
    spec = CONVERT_FORMAT_MAP.get(fmt_key) if fmt_key else None
    want_audio = bool(fmt_key and fmt_key.startswith("aonly-")) or \
        (spec is not None and spec[2] == "audio")

    opts = _ytdlp_opts(outdir, log, browser, template)
    opts["format"] = _ytdlp_fmt(quality)
    opts["noplaylist"] = not playlist
    if quality == "audio":
        # postprocessorargs 里显式 -map 0:a:0 是关键：抖音那类源的第一条流
        # **就是视频**（download_addr-0 是音视频合流 mp4）。不加 -map 时
        # ffmpeg 按默认规则映射，会把非纯音频的内容写进 .m4a —— 文件名是 .m4a、
        # 里面却带着视频流，很多播放器直接判为"文件损坏"，用户看着就像
        # "下载坏了，得自己改名/修复"。-vn + -map 0:a:0 双保险只留音频。
        opts["postprocessorargs"] = {
            "extractaudio": ["-map", "0:a:0", "-vn"],
        }
        opts["postprocessors"] = [{
            "key": "FFmpegExtractAudio", "preferredcodec": "m4a",
            "preferredquality": "192"}]
    elif target in ("mp4", "mkv", "webm"):
        # 这三个 yt-dlp 自己做得到，直接换封装即可，省掉一次转码
        opts["merge_output_format"] = target
        target = None
    if subs:
        opts["writesubtitles"] = True
        opts["writeautomaticsub"] = True
        opts["subtitleslangs"] = ["zh-Hans", "zh-CN", "zh", "en"]
        opts["subtitlesformat"] = "srt/vtt/ass/best"
    if thumb:
        opts["writethumbnail"] = True

    def hook(d):
        if job is None:
            return
        if d.get("status") == "downloading":
            job.total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            job.done = d.get("downloaded_bytes") or 0
            job.speed = d.get("speed") or 0.0
            job.eta = d.get("eta") or 0
            job.filename = d.get("filename") or ""
        elif d.get("status") == "finished":
            job.done = job.total or job.done
            job.filename = d.get("filename") or job.filename

    opts["progress_hooks"] = [hook]
    # yt-dlp 不会因为点"取消"自己停下来，靠一个会抛的 progress hook 中断
    if job is not None:
        def _cancel_check(d):
            if job.canceled:
                raise _Canceled()
            hook(d)
        opts["progress_hooks"] = [_cancel_check]
    try:
        with m.YoutubeDL(opts) as ydl:
            # 用 extract_info(download=True) 而不是 download()：
            # 前者会返回 info，能从 requested_downloads[].filepath 拿到
            # **合并/转码之后的最终文件名**。只靠 progress hook 拿到的是
            # 中间产物（实测标题会显示成 `xxx.f251.webm` 这种临时名，
            # 明明成品是同目录下的 .mp4，用户一看就以为下错了）。
            info = ydl.extract_info(url, download=True)
            final_path = None
            if job is not None and isinstance(info, dict):
                for rd in (info.get("requested_downloads") or []):
                    fp = rd.get("filepath")
                    if fp:
                        final_path = fp
                        job.filename = fp
    except _Canceled:
        return False, "已取消"
    except Exception as e:
        return False, "%s: %s" % (type(e).__name__, e)
    if job is not None and job.canceled:
        return False, "已取消"

    # ---- 需要真正的重编码时：下完再转一次 ----
    if target and spec is not None:
        src = final_path or (job.filename if job is not None else "")
        if not src or not os.path.isfile(src):
            return False, "下载完成但找不到成品文件，无法转换"
        if job is not None:
            job.total = job.total or probe_duration(find_ffmpeg(), src) or 0
            job.done = 0.0
        try:
            log("正在转换为 %s …" % spec[1])
        except Exception:
            pass
        dst = "%s.%s" % (os.path.splitext(src)[0], spec[3])
        try:
            ff = find_ffmpeg()
        except Exception as e:
            # 转不了就把原片留下，别把已下好的东西弄丢
            try:
                log("找不到 ffmpeg，跳过转换（保留原格式）：%s" % e)
            except Exception:
                pass
            return True, "下载完成（未转换：缺少 ffmpeg）"
        # 音频源转视频目标时要造黑底画面，否则 ffmpeg 会因为没有视频流而报错
        hv = True
        try:
            hv = bool(real_video_streams(probe_streams(ff, src)))
        except Exception:
            hv = True
        try:
            cmd = build_convert_cmd(ff, src, dst, fmt_key, "23", "192k", False,
                                    has_video=hv)
        except Exception as e:
            return False, "构造转换命令失败：%s" % e
        if job is not None:
            ok2, why2 = run_ffmpeg(cmd, job=job, log=log)
        else:
            ok2, why2 = run_ffmpeg(cmd, log=log)
        if job is not None and job.canceled:
            return False, "已取消"
        if not ok2:
            # 转换失败但原片已下好，明确告诉用户文件在哪
            return False, "已下载原格式（转换失败：%s）\n原文件：%s" % (why2, src)
        try:
            # 转码成功后删掉中间产物，避免用户看到两份、也不知道该用哪个
            if os.path.abspath(dst) != os.path.abspath(src):
                os.remove(src)
        except Exception:
            pass
        if job is not None:
            job.filename = dst
        return True, "下载并转换完成：%s" % spec[1]

    return True, "下载完成"



# =====================================================================
# 图形界面（tkinter）—— 深色主题
# =====================================================================
#
# 为什么控件全是自绘的：
#   ttk 的原生控件在 Windows 上跟着系统主题走，深色底上摆一排亮色输入框会非常
#   割裂；而且圆角、内边距、悬停态都改不动。所以这里只在 Canvas 上画圆角矩形
#   来拼控件。输入仍然用 tk.Entry / tk.Text(保留输入法、选择、快捷编辑这些能力)，
#   只是把边框拿掉、底色对齐主题。
#
# 取色和圆角直接沿用设计稿，改主题只要动这一个字典：
#   画布 #0A0D3A / 卡片 #1E2353 / 输入 #262C69 / 描边 #2E3583
#   主色 #5865F2 · 高意图动作绿 #35ED7E · 进行中品红 #EC48BD · 链接青 #00B0F4
# 文件对话框用的扩展名清单**由上面那份唯一事实来源生成**。
# 原来手工维护两份，`.alac` / `.mpeg` 只落在 CLI 那侧 —— 同一个文件，
# 命令行认得、界面却报"未知类型"。现在不会再漂移。
GUI_AUDIO_EXT = " ".join(sorted(AUDIO_EXT))
GUI_VIDEO_EXT = " ".join(sorted(VIDEO_EXT))

# 优先用质量高的模型，但**必须真的存在 model.bin** 才算可用。
# （踩过坑：只看目录存在会把下载失败残留的空目录当成可用模型，默认选中一个跑不起来的模型。）
MODEL_PREFERENCE = ["large-v3", "turbo", "medium", "small", "base", "tiny"]


def usable_models():
    out = []
    for n in MODEL_PREFERENCE:
        if n in MODEL_REPOS and os.path.isfile(os.path.join(MODELS_DIR, n, "model.bin")):
            out.append(n)
    return out


def default_model():
    have = usable_models()
    return have[0] if have else "small"


# 界面控件是模块级的类（要继承 tk.Canvas / tk.Frame），所以 tkinter 必须在模块
# 作用域就导入 —— 原来它是写在 launch() 里的。极少数没带 tkinter 的精简 Python
# 上也别让整个模块 import 就炸：给一组空壳类顶上去，命令行模式照常可用。
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
    _TK_OK = True
except Exception:
    _TK_OK = False

    class _Stub(object):
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, name):
            return lambda *a, **k: None

    class _StubNS(object):
        Canvas = Frame = Label = Entry = Text = Toplevel = StringVar = _Stub

        def __getattr__(self, name):
            return _Stub

    tk = _StubNS()
    filedialog = messagebox = _StubNS()

C = {
    "canvas":     "#0A0D3A",
    "titlebar":   "#070A2C",
    "surface":    "#1E2353",
    "surface_in": "#262C69",
    "surface_hi": "#2A3080",
    "bg_tint":    "#2A3080",
    "line":       "#2E3583",
    "line_soft":  "#22275E",
    "ink":        "#FFFFFF",
    "ink2":       "#B7BCE8",
    "ink3":       "#7C82BC",
    "ink4":       "#5A60A0",
    "ink5":       "#3E4490",
    "primary":    "#5865F2",
    "primary_dk": "#4752C4",
    "green":      "#35ED7E",
    "green_bg":   "#163D33",
    "magenta":    "#EC48BD",
    "magenta_bg": "#33163F",
    "link":       "#00B0F4",
    "on_green":   "#0A0D3A",
    "log_bg":     "#05071F",
    "log_fg":     "#8C92CE",
    "err":        "#FF6B81",
}

# 圆角：控件 11，按钮 14，卡片 22，窗口 24 —— 窗口 > 卡片 > 控件，层次不能倒
R_INPUT, R_BTN, R_CARD = 11, 14, 22

FONT_UI = "Microsoft YaHei UI"     # 会被 _init_fonts() 换成系统里真实存在的
FONT_NUM = "Consolas"              # 数字/时间码，等宽才对得齐
S_UI, S_NUM = 9, 9

# =====================================================================
# 2K / 4K：让界面真正按系统 DPI 渲染在物理像素上
# ---------------------------------------------------------------------
# 之前进程是 DPI-unaware 的：Windows 把整个窗口画成"逻辑分辨率"的小位图，
# 再按缩放比例整体拉大（本机 2560x1600 被当成 1280x800 再乘 2）——
# 所以在 2K/4K 屏上所有内容、尤其文字都是糊的。
#
# 做了 DPI 感知之后 Tk 改用物理像素 reporting，字号（单位是**磅**）会由
# `tk scaling` 自动按真实 DPI 放大；但布局尺寸全是写死的**像素**，必须整体
# 乘 DPI 系数，否则面板留白 / 侧栏宽度会跟放大后的字体脱节。
#
# 缩放统一收口在四个地方，避免满地改数字、也避免把"本来就是物理像素"的值
# 重复缩放（`winfo_width()` 之类的返回值一律不碰）：
#   1) pack / grid / place 的 padx / pady / ipadx / ipady
#   2) tk 控件构造时的 width / height
#   3) 自绘助手 rrect 的圆角半径与描边、icon 的尺寸、渐变块的圆角
#   4) 少量仍需手工收口的圆角矩形绘制（见各控件里的 U()）
# =====================================================================

UI_SCALE = 1.0
_UI_SCALED = False


def enable_dpi_awareness():
    """必须在创建任何窗口**之前**调用，否则设置不生效。

    返回是否成功。失败也不影响使用（只是回到低分辨率拉伸的老样子）。
    """
    if not sys.platform.startswith("win"):
        return False
    try:
        import ctypes
        try:      # Win10 1703+：per-monitor v2，跨屏 / 改缩放比例都能正确重算
            ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
            return True
        except Exception:
            pass
        try:
            ctypes.windll.user32.SetProcessDPIAware()
            return True
        except Exception:
            return False
    except Exception:
        return False


def U(v):
    """像素字面量 → 物理像素。布局里写死的尺寸都该过这里。"""
    try:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return v
        return int(round(v * UI_SCALE))
    except Exception:
        return v


def UP(v):
    """同 U()，但支持 (a, b) / [a, b] 这种成对的 padding。"""
    if isinstance(v, (tuple, list)):
        return tuple(U(x) for x in v)
    return U(v)


def init_ui_scale(root):
    """按显示器实际 DPI 算出缩放系数，并装上收口钩子。"""
    global UI_SCALE, _UI_SCALED, _CHROME_H
    try:
        dpi = float(root.winfo_fpixels("1i")) or 96.0
    except Exception:
        dpi = 96.0
    UI_SCALE = max(1.0, dpi / 96.0)
    _CHROME_H = U(32)                  # 顶部外壳高度也要跟着缩放（用字面量，幂等）
    if not _UI_SCALED:
        _install_ui_scale_hooks()
        _UI_SCALED = True
    return UI_SCALE


def _scale_pad_kw(kw):
    for k in ("padx", "pady", "ipadx", "ipady"):
        if k in kw:
            kw[k] = UP(kw[k])
    return kw


def _install_ui_scale_hooks():
    """给 tkinter 装一层"像素字面量自动放大"的薄壳（只碰数值字面量）。"""
    import tkinter as _tk

    # 1) 几何管理器：padx / pady / ipadx / ipady
    for name in ("pack", "grid", "place"):
        orig = getattr(_tk.Misc, name, None)
        if orig is None or getattr(orig, "_ui_scaled", False):
            continue

        def _make(_orig):
            def wrapper(self, *a, **kw):
                if a and isinstance(a[0], dict):
                    a = (_scale_pad_kw(dict(a[0])),) + a[1:]
                if kw:
                    kw = _scale_pad_kw(dict(kw))
                return _orig(self, *a, **kw)
            wrapper._ui_scaled = True
            wrapper._ui_orig = _orig
            return wrapper

        setattr(_tk.Misc, name, _make(orig))

    # 2) 控件构造：width / height（tk 里这两项一律是像素）
    widget = getattr(_tk, "Widget", None)
    orig_init = getattr(widget, "__init__", None) if widget else None
    if orig_init is not None and not getattr(orig_init, "_ui_scaled", False):
        def _win_init(self, master, widgetName, cnf={}, kw={}, extra=()):
            if kw:
                kw = dict(kw)
                for k in ("width", "height"):
                    if k in kw:
                        kw[k] = U(kw[k])
            if isinstance(cnf, dict) and cnf:
                cnf = dict(cnf)
                for k in ("width", "height"):
                    if k in cnf:
                        cnf[k] = U(cnf[k])
            return orig_init(self, master, widgetName, cnf, kw, extra)
        _win_init._ui_scaled = True
        _win_init._ui_orig = orig_init
        widget.__init__ = _win_init


def _init_fonts():
    """挑一个系统里真实存在的界面字体。

    直接用 "Microsoft YaHei UI" 在个别精简版 Windows 上会静默回退成点阵字体，
    中文看着发虚，所以按优先级试探一次。
    """
    global FONT_UI, FONT_NUM, S_UI, S_NUM
    try:
        from tkinter import font as tkfont
        fams = set(tkfont.families())
    except Exception:
        return
    for name in ("Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", "Arial"):
        if name in fams:
            FONT_UI = name
            break
    for name in ("Consolas", "Cascadia Mono", "Courier New"):
        if name in fams:
            FONT_NUM = name
            break


def F(size=9, bold=False):
    return (FONT_UI, size, "bold") if bold else (FONT_UI, size)


def FN(size=9, bold=False):
    return (FONT_NUM, size, "bold") if bold else (FONT_NUM, size)


# ---------------------------------------------------------------- 绘图基础

def rrect(cv, x1, y1, x2, y2, r, fill, outline=None, width=1, tags=()):
    """在 Canvas 上画一个圆角矩形（4 段 pieslice 圆弧 + 2 个交叉矩形）。

    不用 create_polygon(smooth=True)：那个走的是样条插值，圆角是"椭圆味儿"的，
    半径一大就不圆。圆弧是真正的圆，跟设计稿一致。
    描边靠"外圈填描边色 + 内圈填底色"两层叠出来，因为 canvas 的圆角矩形没有描边概念。

    `r` 与 `width` 是**像素字面量**，在这里统一按 DPI 放大；坐标不缩放
    （坐标要么是常量 0/1，要么来自 winfo，已经是物理像素）。
    """
    return _rrect_raw(cv, x1, y1, x2, y2, U(r), fill, outline,
                      U(width) if width else width, tags)


def _rrect_raw(cv, x1, y1, x2, y2, r, fill, outline=None, width=1, tags=()):
    """rrect 的本体，r/width 已是物理像素。递归只走这里，避免二次缩放。"""
    if x2 - x1 < 2 or y2 - y1 < 2:
        return []
    if outline and width > 0:
        _rrect_raw(cv, x1, y1, x2, y2, r, outline, None, 0, tags)
        i = width
        return _rrect_raw(cv, x1 + i, y1 + i, x2 - i, y2 - i, max(0, r - i), fill, None, 0, tags)
    r = max(0, min(r, (x2 - x1) / 2.0, (y2 - y1) / 2.0))
    ids = []
    if r <= 0.5:
        ids.append(cv.create_rectangle(x1, y1, x2, y2, fill=fill, outline="", tags=tags))
        return ids
    ids.append(cv.create_rectangle(x1 + r, y1, x2 - r, y2, fill=fill, outline="", tags=tags))
    ids.append(cv.create_rectangle(x1, y1 + r, x2, y2 - r, fill=fill, outline="", tags=tags))
    for (ax, ay, st) in ((x1, y1, 90), (x2 - 2 * r, y1, 0),
                         (x1, y2 - 2 * r, 180), (x2 - 2 * r, y2 - 2 * r, 270)):
        ids.append(cv.create_arc(ax, ay, ax + 2 * r, ay + 2 * r, start=st, extent=90,
                                 fill=fill, outline="", style="pieslice", tags=tags))
    return ids


def vgrad(cv, x1, y1, x2, y2, c1, c2, radius=0, tags=()):
    """竖直线条堆出来的线性渐变（Canvas 没有原生渐变填充）。

    每行 1px，颜色线性插值。只用在进度条和品牌标这种小面积上，几百行线条
    的开销可以忽略。
    """
    def hex2rgb(h):
        h = h.lstrip("#")
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    r1, g1, b1 = hex2rgb(c1)
    r2, g2, b2 = hex2rgb(c2)
    h = max(1, int(y2 - y1))
    for i in range(h):
        t = i / float(h - 1) if h > 1 else 0.0
        col = "#%02x%02x%02x" % (int(r1 + (r2 - r1) * t),
                                 int(g1 + (g2 - g1) * t),
                                 int(b1 + (b2 - b1) * t))
        cv.create_line(x1, y1 + i, x2, y1 + i, fill=col, tags=tags)


# ---------------------------------------------------------------- 图标（24x24 线性稿）

def icon(cv, name, cx, cy, size, color, width=None, tags=()):
    """按 24x24 的坐标格子画图标，再整体缩放到 size。

    不用 SVG/图片：图标要跟着主题色变，而且打包成 exe 时不该再多带一堆图片资源。
    `size` 是像素字面量，按 DPI 放大；中心坐标 cx/cy 来自 winfo，不缩放。
    """
    size = U(size)
    if width is None:
        width = max(1.5, size / 12.5)
    s = float(size) / 24.0

    def X(v):
        return cx + (v - 12.0) * s

    def Y(v):
        return cy + (v - 12.0) * s

    def line(pts, w=None, closed=False):
        flat = []
        for (a, b) in pts:
            flat += [X(a), Y(b)]
        if not closed:
            cv.create_line(*flat, fill=color, width=w or width,
                           capstyle="round", joinstyle="round", tags=tags)
        else:
            cv.create_polygon(*flat, fill=color, outline="", tags=tags)

    def ring(x1, y1, x2, y2, w=None):
        cv.create_oval(X(x1), Y(y1), X(x2), Y(y2), outline=color,
                       width=w or width, tags=tags)

    def dot(x, y, rad):
        cv.create_oval(X(x - rad), Y(y - rad), X(x + rad), Y(y + rad),
                       fill=color, outline="", tags=tags)

    def rbox(x1, y1, x2, y2, rad, w=None):
        rrect(cv, X(x1), Y(y1), X(x2), Y(y2), rad * s, fill="",
              outline=color, width=w or width, tags=tags)

    if name == "wave":
        for (x, ya, yb) in ((4, 10, 14), (8, 6, 18), (12, 3.5, 20.5),
                            (16, 6, 18), (20, 10, 14)):
            line([(x, ya), (x, yb)])
    elif name == "cube":
        line([(12, 2.8), (20.5, 7.4), (20.5, 16.6), (12, 21.2),
              (3.5, 16.6), (3.5, 7.4)], closed=False)
        line([(3.5, 7.4), (12, 12), (20.5, 7.4)])
        line([(12, 12), (12, 21.2)])
    elif name == "clock":
        ring(3.4, 3.4, 20.6, 20.6)
        line([(12, 7.2), (12, 12), (15.4, 14.1)])
    elif name == "gear":
        ring(9, 9, 15, 15, w=max(1.5, width * 0.9))
        for (x1, y1, x2, y2) in ((12, 2.5, 12, 5.1), (12, 18.9, 12, 21.5),
                                 (3.7, 7.2, 6.0, 8.5), (18.0, 15.5, 20.3, 16.8),
                                 (3.7, 16.8, 6.0, 15.5), (18.0, 8.5, 20.3, 7.2)):
            line([(x1, y1), (x2, y2)])
    elif name == "help":
        ring(3, 3, 21, 21)
        line([(9.6, 9.4), (10.2, 8.6)])
        cv.create_arc(X(8.8), Y(7.6), X(15.2), Y(14.0), start=200, extent=-200,
                      style="arc", outline=color, width=width, tags=tags)
        dot(12, 17.2, 1.1)
    elif name == "upload":
        line([(12, 15.5), (12, 4.5)])
        line([(7.8, 8.7), (12, 4.5), (16.2, 8.7)])
        line([(4, 15.2), (4, 18.3)])
        line([(4, 18.3), (6.2, 20.5), (17.8, 20.5), (20, 18.3), (20, 15.2)])
    elif name == "download":
        line([(12, 3.6), (12, 14.2)])
        line([(7.8, 10.2), (12, 14.4), (16.2, 10.2)])
        line([(4.2, 16.6), (4.2, 18.0), (6.6, 20.4), (17.4, 20.4), (19.8, 18.0), (19.8, 16.6)])
    elif name == "file":
        rbox(4, 3.2, 20, 20.8, 3)
        line([(8.4, 10.6), (15.6, 10.6)])
        line([(8.4, 14.2), (12.8, 14.2)])
    elif name == "folder":
        line([(3, 7.4), (3, 16.6), (5.4, 19), (18.6, 19), (21, 16.6),
              (21, 10), (18.6, 7.6), (10.3, 7.6), (8.3, 5)])
    elif name == "search":
        ring(4, 4, 17.6, 17.6)
        line([(16, 16), (20.4, 20.4)])
    elif name == "check":
        line([(4.8, 12.6), (9.4, 17.2), (19.2, 7.4)], w=max(2.0, width * 1.3))
    elif name == "play":
        line([(7.6, 4.9), (19.8, 12), (7.6, 19.1)], closed=True)
    elif name == "copy":
        rbox(8.6, 8.6, 19.6, 19.6, 2.6)
        line([(15.6, 5.8), (13.1, 4.2), (5.8, 4.2), (3.2, 6.8),
              (3.2, 14.1), (4.8, 16.5)])
    elif name == "refresh":
        cv.create_arc(X(4), Y(4), X(20), Y(20), start=310, extent=310,
                      style="arc", outline=color, width=width, tags=tags)
        line([(20.2, 6.4), (20.2, 11.6), (15, 11.6)])
    elif name == "x":
        line([(6.5, 6.5), (17.5, 17.5)])
        line([(17.5, 6.5), (6.5, 17.5)])
    elif name == "trash":
        line([(4, 6.6), (20, 6.6)])
        line([(9.4, 6.6), (9.4, 3.8), (14.6, 3.8), (14.6, 6.6)])
        line([(5.8, 6.6), (7.0, 20.2), (17.0, 20.2), (18.2, 6.6)])
    elif name == "pause":
        line([(9, 5), (9, 19)])
        line([(15, 5), (15, 19)])
    elif name == "down":
        line([(3, 4.5), (6, 7.5), (9, 4.5)])
    elif name == "right":
        line([(9.5, 6.5), (15, 12), (9.5, 17.5)])
    elif name == "left":
        line([(14.5, 6.5), (9, 12), (14.5, 17.5)])
    elif name == "chevrons_right":
        line([(6, 6.5), (11.5, 12), (6, 17.5)])
        line([(12.5, 6.5), (18, 12), (12.5, 17.5)])
    elif name == "lyrics":
        line([(4, 6.5), (20, 6.5)])
        line([(4, 12), (20, 12)])
        line([(4, 17.5), (13, 17.5)])
    elif name == "swap":
        # 上下两条反向箭头，表达"格式互换"
        line([(4, 9), (17, 9)])
        line([(13.5, 5.5), (17, 9), (13.5, 12.5)])
        line([(20, 15), (7, 15)])
        line([(10.5, 11.5), (7, 15), (10.5, 18.5)])


# ---------------------------------------------------------------- 控件基础

def _family_ok(family):
    try:
        from tkinter import font as tkfont
        return family in set(tkfont.families())
    except Exception:
        return False


class Panel(tk.Frame):
    """圆角卡片。内容统一放进 self.body。

    Canvas 只做背景（place 上去，不参与布局），真正的尺寸由 body 撑开 ——
    这样卡片能"内容多高就多高"，不用手算高度。
    """

    def __init__(self, master, radius=R_CARD, fill=None, outline=None, pad=22,
                 outer=None, height=None, **kw):
        self._fill = fill or C["surface"]
        self._outline = outline
        self._radius = radius
        outer = outer or C["canvas"]
        tk.Frame.__init__(self, master, bg=outer, **kw)
        self._outer = outer
        self._bgc = tk.Canvas(self, bg=outer, highlightthickness=0, bd=0)
        self._bgc.place(x=0, y=0, relwidth=1, relheight=1)
        self.body = tk.Frame(self, bg=self._fill)
        self.body.pack(fill="both", expand=True, padx=pad, pady=pad)
        if height is not None:
            # 构造时钩子已经把 height 缩放过了，这里必须再 U() 一次 ——
            # 否则 configure 会把**未缩放的原始值**写回去，控件退回 1x 大小。
            self.configure(height=U(height))
            self.pack_propagate(False)
        self._bgc.bind("<Configure>", self._draw)
        self.bind("<Configure>", self._draw)

    def _draw(self, _e=None):
        w, h = self._bgc.winfo_width(), self._bgc.winfo_height()
        if w < 2 or h < 2:
            return
        self._bgc.delete("all")
        rrect(self._bgc, 0, 0, w, h, self._radius, self._fill, self._outline, 1)

    def set_fill(self, fill):
        self._fill = fill
        self.body.configure(bg=fill)
        self._draw()


class RoundedBody(tk.Canvas):
    """一块自带圆角的内容区：内容放self.box（一个普通 Frame）。

    Panel 的圆角画在外框 canvas 上，而内容是直接铺在 body 里的实心控件；
    只要内容顶到卡片边缘（比如 pad=0），实心矩形就会把圆角重新盖成直角。
    RoundedBody 把圆角画在内容**自己这一层**，内容再放进 box，圆角就留住了。
    """

    def __init__(self, master, fill=None, radius=16, pad_x=18, pad_y=14, outer=None):
        self._fill = fill or C["surface"]
        self._radius = radius
        self._px, self._py = pad_x, pad_y
        outer = outer or self._fill
        tk.Canvas.__init__(self, master, bg=outer, highlightthickness=0, bd=0)
        self._outer = outer
        self.box = tk.Frame(self, bg=self._fill)
        self._box_id = self.create_window((0, 0), window=self.box, anchor="nw")
        self.bind("<Configure>", self._draw)

    def _draw(self, _e=None):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 4 or h < 4:
            return
        self.delete("bg")
        rrect(self, 0, 0, w, h, self._radius, self._fill, tags="bg")
        self.itemconfigure(self._box_id, width=max(1, w - 2 * U(self._px)))
        self.box.configure(bg=self._fill)
        self.tk.call("raise", self.box)


class RoundedLogPanel(Panel):
    """折叠式面板：标题条 + 可展开的日志区，四周保持圆角。

    为什么需要它：Panel 的圆角画在外框 canvas 上，而日志区是直接铺在 body 里的
    实心控件；pad=0 时 body 正好盖满整张卡片，于是把画好的圆角重新填成直角 ——
    "运行日志""高级参数"那种带棱角的观感就是这么来的。

    做法：日志区换成 RoundedBody（自带圆角底），Text 放进它的 box 里。
    收起时整条只露标题条，圆角由标题条自己带。
    """

    def __init__(self, master, head_bg, body_bg, radius=R_CARD, pad=0, **kw):
        Panel.__init__(self, master, pad=pad, **kw)
        self._radius = radius
        self._head_bg = head_bg
        self._body_bg = body_bg
        self.log_cv = RoundedBody(self.body, fill=body_bg, radius=radius,
                                  pad_x=16, pad_y=12, outer=head_bg)
        self.log_cv.pack(fill="both", expand=True)
        self.txt = tk.Text(self.log_cv.box, bg=body_bg, fg=C["log_fg"],
                           relief="flat", bd=0, highlightthickness=0,
                           font=FN(9), wrap="word", insertbackground=C["link"],
                           selectbackground=C["surface_hi"], height=7)
        self.txt.pack(fill="both", expand=True)
        self.txt.tag_configure("cur", foreground=C["link"])
        self.txt.tag_configure("err", foreground=C["err"])
        self.txt.configure(state="disabled")

    def set_open(self, ok):
        self._log_open = bool(ok)
        if ok:
            self.log_cv.pack(fill="both", expand=True)
        else:
            self.log_cv.pack_forget()


class Btn(tk.Canvas):
    """圆角按钮，宽度按文字自动算（和设计稿一样），也可以 fill="x" 撑满。"""

    _SKIN = {
        #       底          悬停        按下        文字        图标（None 表示跟随文字）
        "green":   ("#35ED7E", "#61F0A0", "#2BD26C", "#0A0D3A", None),
        "blue":    ("#5865F2", "#6C78F5", "#4A55D6", "#FFFFFF", None),
        "ghost":   ("#262C69", "#303A82", "#20255C", "#B7BCE8", None),
        "outline": (None,      "#1E2353", "#181D45", "#7C82BC", None),
        "quiet":   (None,      "#1E2353", "#181D45", "#7C82BC", None),
        "danger":  (None,      "#33163F", "#2A1234", "#EC48BD", None),
    }

    def __init__(self, master, text="", icon_name=None, kind="ghost", command=None,
                 height=38, radius=R_BTN, pad_x=16, size=9, bold=False, outer=None,
                 fill_width=False, gap=8):
        outer = outer or C["canvas"]
        self._kind = kind
        self._text = text
        self._icon = icon_name
        self._h = height
        self._r = radius
        self._pad = pad_x
        self._size = size
        self._bold = bold
        self._gap = gap
        self._enabled = True
        self._hover = False
        self._pressed = False
        self._cmd = command
        tk.Canvas.__init__(self, master, bg=outer, highlightthickness=0, bd=0,
                           height=height)
        self._outer = outer
        self._font = F(size, bold)
        self._fill_width = fill_width
        self.configure(width=1 if fill_width else self._natural())
        self.bind("<Configure>", lambda e: self._draw())
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.configure(cursor="hand2")

    def _natural(self):
        try:
            from tkinter import font as tkfont
            w = tkfont.Font(font=self._font).measure(self._text)
        except Exception:
            w = len(self._text) * self._size
        if self._icon:
            w += 15 + self._gap
        return int(w + self._pad * 2)

    def _draw(self):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 2 or h < 2:
            return
        base, hov, prs, fg, ic = self._SKIN.get(self._kind, self._SKIN["ghost"])
        self.delete("all")
        if not self._enabled:
            fill, tcol, icol = C["surface_in"], C["ink4"], C["ink4"]
        else:
            fill = prs if self._pressed else (hov if self._hover else base)
            tcol = icol = fg
            if base is None:
                fill = hov if self._hover else self._outer
        outline = C["line"] if (base is None and self._kind in ("outline", "danger")) else None
        rrect(self, 0, 0, w, h, self._r, fill, outline, 1)

        cx = w / 2.0
        iw = 15 if self._icon else 0
        total = iw + (self._gap if self._icon and self._text else 0)
        try:
            from tkinter import font as tkfont
            tw = tkfont.Font(font=self._font).measure(self._text) if self._text else 0
        except Exception:
            tw = 0
        total += tw
        x = cx - total / 2.0
        if self._icon:
            icon(self, self._icon, x + iw / 2.0, h / 2.0, iw, icol)
            x += iw + self._gap
        if self._text:
            self.create_text(x, h / 2.0, text=self._text, anchor="w",
                             fill=tcol, font=self._font)

    def _on_enter(self, _e=None):
        self._hover = True
        self._draw()

    def _on_leave(self, _e=None):
        self._hover = False
        self._pressed = False
        self._draw()

    def _on_press(self, _e=None):
        if not self._enabled:
            return
        self._pressed = True
        self._draw()

    def _on_release(self, _e=None):
        if not self._enabled:
            return
        was = self._pressed
        self._pressed = False
        self._draw()
        if was and self._cmd:
            self._cmd()

    def set_text(self, text, redraw_width=False):
        self._text = text
        if redraw_width and not self._fill_width:
            self.configure(width=self._natural())
        self._draw()

    def set_enabled(self, ok):
        self._enabled = bool(ok)
        self.configure(cursor="hand2" if ok else "arrow")
        self._draw()

    def set_kind(self, kind):
        self._kind = kind
        self._draw()


class Input(tk.Canvas):
    """圆角输入框。真正的输入交给内部的 tk.Entry，只把边框藏掉。"""

    def __init__(self, master, textvariable=None, placeholder="", height=38,
                 radius=R_INPUT, icon_name=None, outer=None, fill=None,
                 font=None, on_return=None, inner_pad=12, size=9, width=None):
        outer = outer or C["surface"]
        fill = fill or C["surface_in"]
        tk.Canvas.__init__(self, master, bg=outer, highlightthickness=0, bd=0,
                           height=height)
        if width:
            self.configure(width=U(width))   # 同上：不能写回未缩放的原始值
        self._fill, self._radius, self._h = fill, radius, height
        self._icon = icon_name
        self._font = font or F(size)
        self._bgc = self
        self._entry = tk.Entry(self, textvariable=textvariable, bg=fill, fg=C["ink"],
                               insertbackground=C["link"], relief="flat", bd=0,
                               highlightthickness=0, font=self._font,
                               # takefocus=True 才能用 Tab 键走到这个框；
                               # 缺了它键盘用户永远到不了输入框。
                               takefocus=True,
                               # 只读状态的 Entry 在 Windows 上不走 bg，而是走
                               # readonlybackground —— 不显式给就会渲染成系统默认的白底，
                               # 深色主题下就是一个刺眼的白块。
                               readonlybackground=fill,
                               disabledbackground=fill, disabledforeground=C["ink4"])
        left = U(inner_pad + (20 if icon_name else 0))
        self._entry.place(x=left, y=U(4), relwidth=1, width=-(left + U(inner_pad)),
                          height=height - 8)
        self._ph = None
        if placeholder:
            self._ph = tk.Label(self, text=placeholder, bg=fill, fg=C["ink4"],
                                font=self._font, anchor="w")
            self._ph.place(x=left, y=U(4), relwidth=1, width=-(left + U(inner_pad)),
                           height=height - 8)
            # 【关键】占位 Label 是 place 在 Entry **之后**的，叠放顺序上它
            # 盖住了整个输入区。Tk 的事件命中取最顶层的控件，所以鼠标点输入框
            # 时事件落在 Label 上，Entry 拿不到焦点 —— 表现就是"点不进去、
            # 打不了字"，只有当占位文字因为有内容而被 place_forget 之后
            # 输入框才可编辑（实测：先点别的按钮让文字出现，框就能打字了）。
            # 这里让 Label 把点击转交给 Entry，行为就与"有内容时"一致。
            self._ph.bind("<Button-1>", self._click_into, add="+")
            self._entry.bind("<FocusIn>", self._ph_hide, add="+")
            self._entry.bind("<FocusOut>", self._ph_show, add="+")
            self._entry.bind("<KeyRelease>", self._ph_sync, add="+")
            # 程序化赋值（粘贴、从设置还原、清空）既不会触发 <KeyRelease>
            # 也不一定伴随焦点变化 —— 不跟踪 textvariable 的话，占位文字会
            # **一直盖在真正的文字上面**，看起来就是"粘贴的链接看不见"。
            if textvariable is not None:
                try:
                    textvariable.trace_add("write", lambda *a: self._ph_sync())
                except Exception:
                    pass
        # 即使没有占位文字，点 Canvas 的空白处（比如右侧留白）也应该能进输入状态，
        # 否则用户会以为这一块不能点。
        self.bind("<Button-1>", self._click_into, add="+")
        if on_return:
            self._entry.bind("<Return>", lambda e: on_return())
        self.bind("<Configure>", lambda e: self._draw())
        self._entry.bind("<FocusIn>", lambda e: self._draw(), add="+")
        self._entry.bind("<FocusOut>", lambda e: self._draw(), add="+")

    def _click_into(self, _e=None):
        """把点击转交给内部 Entry 并给焦点，同时把光标放到点击处附近。

        只做 focus 不够：Entry 是 place 在 Canvas 里的子控件，不先 takefocus
        的话部分环境下第一次点击仍会落在 Canvas 上。
        """
        if str(self._entry.cget("state")) == "disabled":
            return "break"
        try:
            self._entry.focus_force()
        except Exception:
            try:
                self._entry.focus_set()
            except Exception:
                return "break"
        # 点击位置换算成 Entry 内的字符索引，让光标落在点中的那一段，
        # 而不是每次都跳到最末尾。
        try:
            x = self._entry.winfo_pointerx()
            idx = self._entry.index("@%d" % x)
            self._entry.icursor(idx)
        except Exception:
            pass
        self._draw()
        return "break"

    def _ph_hide(self, _e=None):
        if self._ph:
            self._ph.place_forget()

    def _ph_show(self, _e=None):
        if self._ph and not self._entry.get():
            left = U(12) + (U(20) if self._icon else 0)
            self._ph.place(x=left, y=U(4), relwidth=1, width=-(left + U(12)),
                           height=self._h - 8)

    def _ph_sync(self, _e=None):
        if not self._ph:
            return
        if self._entry.get():
            self._ph.place_forget()
        else:
            self._ph_show()

    def _draw(self):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 2 or h < 2:
            return
        self.delete("bg")
        focus = False
        try:
            focus = self.focus_get() is self._entry
        except Exception:
            pass
        outline = C["primary"] if focus else C["line"]
        ids = rrect(self, 0, 0, w, h, self._radius, self._fill, outline, 1, tags="bg")
        if self._icon:
            icon(self, self._icon, 12 + 7, h / 2.0, 14, C["ink3"], tags="bg")
        self.tag_lower("bg")

    def entry(self):
        return self._entry

    def get(self):
        return self._entry.get()

    def set(self, v):
        self._entry.delete(0, "end")
        self._entry.insert(0, v)
        self._ph_sync()


def set_popup_owner(pop, master):
    """把 overrideredirect 弹层挂成 master 的「从属窗口」(owned window)。

    为什么必须手动做：Tk 的 `wm transient` 对 overrideredirect 窗口会被**静默忽略**
    （实测 `GetWindow(hwnd, GW_OWNER)` 仍为 0），而自绘弹层又必须 overrideredirect
    （不能带标题栏）。原实现只能靠 `-topmost` 硬顶，副作用是弹层会一直压在
    **别的程序**窗口上面 —— 用户切到后台还盖着别人。

    自己调 Win32 设好 owner 之后：弹层只压在本程序主窗口之上，用户切到别的
    程序 / 桌面时它跟着一起退到后面，不再覆盖其他窗口。
    非 Windows 或调用失败时静默跳过，不影响其它功能。
    """
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes
        u = ctypes.windll.user32
        u.GetParent.restype = ctypes.c_void_p
        u.GetParent.argtypes = [ctypes.c_void_p]
        u.SetWindowLongPtrW.restype = ctypes.c_void_p
        u.SetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        u.SetWindowPos.restype = ctypes.c_bool
        u.SetWindowPos.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint]
        u.GetWindow.restype = ctypes.c_void_p
        u.GetWindow.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        GWLP_HWNDPARENT = -8
        SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE = 0x0001, 0x0002, 0x0010
        # Tk 的 winfo_id() 给的是内层子窗口；外面还有一层包装窗口才是真正带 owner 的顶层窗口
        me = int(u.GetParent(int(pop.winfo_id())) or 0)
        owner = int(u.GetParent(int(master.winfo_id())) or 0)
        if me and owner:
            u.SetWindowLongPtrW(me, GWLP_HWNDPARENT, owner)
            # 光设 owner 还不够：overrideredirect 弹层不受窗口管理器摆布，不会自动"举"到
            # 主窗口之上（实测会藏在主窗口后面）。显式把它插到「主窗口上面那一个窗口」之后
            # （hWndInsertAfter 的语义是"排在谁后面"，传主窗口自己会跑到主窗口底下 —— 踩过）。
            # 若主窗口已在最前，则退化为 HWND_TOP(0)。
            GW_HWNDPREV = 3
            above = int(u.GetWindow(owner, GW_HWNDPREV) or 0)
            u.SetWindowPos(me, above, 0, 0, 0, 0,
                           SWP_NOSIZE | SWP_NOMOVE | SWP_NOACTIVATE)
    except Exception:
        pass


class Select(tk.Canvas):
    """圆角下拉。列表用自绘的 Toplevel —— 原生 tk.Menu 在深色界面上是一块白。"""

    # 当前处于打开状态的弹层实例（唯一事实来源）。Scroll 的全局滚轮绑定据此让位，
    # 避免"弹层停在固定屏幕坐标、页面却在下面滚动，看着像下拉跟着页面上下飘"。
    _open_instances = []

    @classmethod
    def _prune_dead(cls):
        """清掉 _open_instances 里控件已被销毁的僵尸条目。

        `<Destroy>` 绑了也会清，但那是"正常路径"；万一某条路径没走到，
        僵尸条目会让 `_any_open()` 永远为真，页面滚轮让位判断永久失效。
        这里再兜一层，保证状态不会卡死。
        """
        alive = []
        for s in cls._open_instances:
            try:
                if s.winfo_exists():
                    alive.append(s)
            except Exception:
                pass
        cls._open_instances[:] = alive

    @classmethod
    def _any_open(cls):
        cls._prune_dead()
        return bool(cls._open_instances)

    @classmethod
    def reposition_all_open(cls):
        """页面滚动后，让所有打开的弹层重新贴住自己的字段。

        弹层坐标是 `open()` 那一刻算死的，页面一滚它就脱锚。所以页面每滚动一步
        就把弹层挪回字段旁边 —— 这样"页面可以正常滚动"和"下拉不会飘走"就不再冲突。
        字段被滚出可视区时弹层没有意义，直接收掉。
        """
        for s in list(cls._open_instances):
            try:
                s.reposition()
            except Exception:
                pass
        cls._prune_dead()

    def _field_in_view(self):
        """字段是否还在它所在 Scroll 容器的可视区内（沿 master 链往上找）。"""
        try:
            p = self.master
            while p is not None:
                if isinstance(p, Scroll):
                    cv = p.cv
                    top = cv.winfo_rooty()
                    bot = top + cv.winfo_height()
                    f0 = self.winfo_rooty()
                    f1 = f0 + self.winfo_height()
                    return (f1 >= top - 4) and (f0 <= bot + 4)
                p = getattr(p, "master", None)
        except Exception:
            pass
        return True

    def reposition(self):
        """把弹层重新贴到字段旁边。字段不可见时收掉弹层。"""
        pop = self._popup
        if pop is None:
            return False
        if not self._field_in_view():
            self.close()
            return False
        try:
            w, h = pop.winfo_width(), pop.winfo_height()
            if self._place_below:
                y = self.winfo_rooty() + self.winfo_height() + U(6)
            else:
                y = self.winfo_rooty() - h - U(6)
            geom = (w, h, self.winfo_rootx() - U(1), y - U(1))
            if geom != self._last_geom:      # 没动就不重设，避免无谓的重绘
                pop.geometry("%dx%d+%d+%d" % geom)
                self._last_geom = geom
        except Exception:
            return False
        return True

    @classmethod
    def close_all_open(cls):
        """收起全部弹层（页面被拖动滚动条 / 切页等场景的兜底）。"""
        for s in list(cls._open_instances):
            try:
                s.close()
            except Exception:
                pass

    def __init__(self, master, values, textvariable=None, command=None, height=38,
                 radius=R_INPUT, outer=None, value=None, width=None, pad_x=12,
                 size=9, fill=None, labeler=None, statuser=None, downloader=None):
        outer = outer or C["surface"]
        fill = fill or C["surface_in"]
        self._values = list(values)
        self._var = textvariable
        self._cmd = command
        self._h, self._r, self._pad = height, U(radius), U(pad_x)
        self._fill = fill
        self._size = size
        self._labeler = labeler or (lambda v: str(v))
        # statuser(v) -> 状态串（如 "已下载"/"未下载"/""）；downloader(v) -> 触发下载
        self._statuser = statuser
        self._downloader = downloader
        self._popup = None
        self._disabled = False        # 见 set_enabled()：Canvas 的 state 拦不住点击
        self._hover = False
        self._place_below = True     # 弹层展开在字段下方还是上方（滚动时保持不变）
        self._last_geom = None
        tk.Canvas.__init__(self, master, bg=outer, highlightthickness=0, bd=0,
                           height=height)
        if width:
            self.configure(width=U(width))   # 同上（当前调用点未传 width，属潜伏问题）
        else:
            self._autosize()
        self.bind("<Configure>", lambda e: self._draw())
        self.bind("<Enter>", lambda e: (setattr(self, "_hover", True), self._draw()))
        self.bind("<Leave>", lambda e: (setattr(self, "_hover", False), self._draw()))
        self.bind("<Button-1>", lambda e: self.toggle())
        # 页面 reload() 会 destroy 掉旧的 Select。它若还留在类级 `_open_instances` 里，
        # `_any_open()` 就永远为真 —— 页面滚轮"让位"判断从此永久失效，且只能重启进程恢复。
        self.bind("<Destroy>", self._on_destroy, add="+")
        self.configure(cursor="hand2")

    def _on_destroy(self, _e=None):
        try:
            Select._open_instances.remove(self)
        except ValueError:
            pass
        # 控件此刻正在销毁，不能走 close()（里面的 _draw() 会 TclError）。
        # 直接把弹层销毁掉，否则它作为独立 Toplevel 会变成查无主人的孤儿窗口。
        pop = self._popup
        self._popup = None
        if pop is not None:
            try:
                pop.grab_release()
            except Exception:
                pass
            try:
                pop.destroy()
            except Exception:
                pass

    def _autosize(self):
        try:
            from tkinter import font as tkfont
            fnt = tkfont.Font(font=F(self._size))
            w = max([fnt.measure(self._labeler(v)) for v in self._values] or [40])
        except Exception:
            w = 90
        if self._statuser:
            w += 22  # 给左侧状态点留位置
        self.configure(width=int(w + self._pad * 2 + U(22)))

    def _draw(self):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 2 or h < 2:
            return
        self.delete("all")
        bg = C["surface_hi"] if (self._hover or self._popup) else self._fill
        rrect(self, 0, 0, w, h, self._r, bg, C["primary"] if self._popup else None, 1)
        x0 = self._pad
        if self._statuser:
            st = self._statuser(self.get()) or ""
            if st:
                ok = (st == "已下载")
                self.create_oval(self._pad + U(2), h / 2.0 - U(4), self._pad + U(10),
                                h / 2.0 + U(4), fill=C["green"] if ok else C["ink4"],
                                outline="")
                x0 = self._pad + U(20)
        txt = self._labeler(self.get())
        self.create_text(x0, h / 2.0, text=txt, anchor="w",
                         fill=C["ink"], font=F(self._size))
        icon(self, "down", w - self._pad - 1, h / 2.0, 11, C["ink3"])

    def get(self):
        if self._var is not None:
            return self._var.get()
        return self._value

    _value = None

    def set(self, v):
        if self._var is not None:
            self._var.set(v)
        else:
            self._value = v
        self._draw()

    def set_enabled(self, ok):
        """真正禁用 / 启用。

        注意：光 `configure(state="disabled")` 是**没用的** —— 那是 Canvas 自己的
        状态位，管的是图元插入和选中，`<Button-1>` 照样会送到 toggle() 把弹层展开。
        必须自己加一个标志位拦住它。
        """
        self._disabled = not ok
        try:
            self.configure(state="disabled" if self._disabled else "normal")
        except Exception:
            pass
        self._draw()

    def toggle(self):
        if self._disabled:
            return
        if self._popup:
            self.close()
        else:
            self.open()

    def close(self, _e=None):
        if self._popup:
            try:
                self._popup.grab_release()
            except Exception:
                pass
            try:
                self._popup.destroy()
            except Exception:
                pass
            self._popup = None
            try:
                Select._open_instances.remove(self)
            except ValueError:
                pass
        self._draw()

    def open(self):
        top = self.winfo_toplevel()
        row_h = 34
        pad = 6
        n = len(self._values)
        inner_w = max(self.winfo_width(), U(160))
        need = n * row_h + pad * 2
        maxpop = 360
        use_scroll = need > maxpop
        h = U(min(need, maxpop))
        x = self.winfo_rootx()
        below = True
        y = self.winfo_rooty() + self.winfo_height() + U(6)
        if y + h + U(2) > top.winfo_rooty() + top.winfo_height():
            below = False
            y = self.winfo_rooty() - h - U(6)
        # 记下这次展开的方向：页面滚动后要"贴着字段移动"，但不能因为空间变化
        # 就在上/下之间来回翻转（那样会一直抖）。
        self._place_below = below
        pop = tk.Toplevel(top)
        # 立刻登记，别等建完再记：下面 100 多行里任何一处抛异常（比如细滚动条样式
        # 没注册成功，`ttk.Scrollbar(style=...)` 会抛 TclError），如果不先登记，
        # 这个已经 map 出来的空弹层既不在 `_open_instances` 里、self._popup 也是 None，
        # close() 够不着它 → **每点一次下拉就永久泄漏一个顶层窗口**。
        self._popup = pop
        if self not in Select._open_instances:
            Select._open_instances.append(self)
        pop.overrideredirect(True)
        pop.configure(bg=C["line"])
        geom_w = inner_w + (U(10) if use_scroll else U(2))   # 细滚动条约 8px
        pop.geometry("%dx%d+%d+%d" % (geom_w, h + U(2), x - U(1), y - U(1)))
        self._last_geom = (geom_w, h + U(2), x - U(1), y - U(1))
        # 成为主窗口的从属窗口：只压在本程序主窗口上面，切到别的程序时跟着一起退后。
        # （不能用 -topmost —— 那会一直盖在别人的窗口上；transient 对 overrideredirect
        #  无效，真正的机制见 set_popup_owner 里的 Win32 设 owner。）
        try:
            pop.transient(top)
        except Exception:
            pass
        # 弹层的"包装窗口"要等它真正 map 之后才存在（实测 map 前 GetParent 返回 0），
        # 所以 owner 必须在 <Map> 里设；再补一个短延时重设兜底。
        # 注意 _e 必须给默认值：弹层被销毁时若还有排队的 <Map> 回调，
        # Tk 会以 0 个参数调它（写成 lambda e: 就会偶发报"missing argument 'e'"）。
        def _set_owner(_e=None, _p=pop, _t=top):
            set_popup_owner(_p, _t)

        pop.bind("<Map>", _set_owner)
        try:
            pop.after(60, lambda _p=pop, _t=top: set_popup_owner(_p, _t))
        except Exception:
            pass
        cv = tk.Canvas(pop, bg=C["surface"], highlightthickness=0, bd=0)
        sb = None
        # 滚动条必须先 pack（Tk 按顺序分配空间），否则会被 expand 的画布挤没
        if use_scroll:
            from tkinter import ttk
            sb = ttk.Scrollbar(pop, orient="vertical", style="Thin.Vertical.TScrollbar",
                               command=cv.yview)
            sb.pack(side="right", fill="y")
            cv.configure(yscrollcommand=sb.set)
        cv.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(cv, bg=C["surface"])
        win = cv.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e=None: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind("<Configure>",
                lambda e=None: cv.itemconfigure(
                    win, width=e.width if e is not None else cv.winfo_width()))
        # 滚轮必须绑在 **Toplevel(pop)** 上，不能绑在内部的 cv 上：
        # Tk 的事件只沿「控件自身 → 控件类 → 所属 Toplevel → all」这四个绑定标签传播，
        # **不经过中间的父控件**。列表项 row/r 的所属 Toplevel 正是 pop，
        # 所以只有绑在 pop 上，鼠标悬停在**列表项**上时的滚轮才能收到；
        # 绑在 cv 上只有鼠标落在列表的空隙里才有效（几乎等于没有）。
        # 滚轮路由（比之前多考虑了"页面也能滚"）：
        # grab_set 期间所有滚轮事件都先落到弹层上，所以这里按**指针位置**分流 ——
        #   · 指针在弹层矩形内 → 滚弹层自己的列表，并 return "break" 吃掉事件；
        #   · 指针在页面其它位置 → 不 break，让事件继续冒泡到 "all" 标签，
        #     由 Scroll._wheel 滚页面，随后 _on_scroll 会把弹层重新贴回字段。
        # 列表不可滚（项数少）时同样放行给页面。
        def _wheel_pop(e, _cv=cv, _p=pop, _scrollable=use_scroll):
            try:
                px, py = _p.winfo_rootx(), _p.winfo_rooty()
                inside = (px <= e.x_root <= px + _p.winfo_width()
                          and py <= e.y_root <= py + _p.winfo_height())
            except Exception:
                inside = True
            if not (inside and _scrollable):
                return
            d = getattr(e, "delta", 0)
            if d:
                _cv.yview_scroll(int(-d / 120) * 3, "units")
            else:
                _cv.yview_scroll(-3 if getattr(e, "num", 0) == 4 else 3, "units")
            return "break"

        for _seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            pop.bind(_seq, _wheel_pop)
        cur = self.get()
        for idx, v in enumerate(self._values):
            active = (v == cur)
            status = (self._statuser(v) if self._statuser else "") or ""
            downloaded = (status == "已下载")
            row = tk.Frame(inner, bg=C["surface"], cursor="hand2")
            row.pack(fill="x", padx=pad, pady=(U(2) if idx == 0 else 0, 0))
            r = tk.Canvas(row, bg=C["surface"], highlightthickness=0, bd=0, height=row_h)
            r.pack(fill="x")
            lbl = self._labeler(v)
            # 已下载：右侧绿色「已下载」徽标；未下载且有下载器：右侧蓝色「下载」按钮
            pill_w, pill_h, pill_x0 = 0, 0, 0

            def paint(cv2, ww, hh, hov=False, _v=v, _a=active, _l=lbl,
                      _ok=downloaded, _st=status, _dl=bool(self._downloader and not downloaded)):
                cv2.delete("all")
                fillc = C["surface_in"] if (hov or _a) else C["surface"]
                rrect(cv2, 0, 0, ww, hh, 9, fillc)
                cv2.create_text(U(12), hh / 2.0, text=_l, anchor="w",
                                fill=C["ink"] if _a else C["ink2"], font=F(self._size))
                if _a:
                    icon(cv2, "check", ww - 20, hh / 2.0, 13, C["link"])
                px1 = ww - U(12)
                if _ok:
                    txt = "已下载"
                    bw, bh = U(50), U(18)
                    px0 = px1 - bw
                    rrect(cv2, px0, hh / 2.0 - bh / 2.0, px1, hh / 2.0 + bh / 2.0, 7,
                          C["green_bg"])
                    cv2.create_text((px0 + px1) / 2.0, hh / 2.0, text=txt, anchor="center",
                                    fill=C["green"], font=F(7))
                elif _dl:
                    txt = "下载"
                    bw, bh = U(46), U(20)
                    px0 = px1 - bw
                    rrect(cv2, px0, hh / 2.0 - bh / 2.0, px1, hh / 2.0 + bh / 2.0, 8,
                          C["primary"])
                    cv2.create_text((px0 + px1) / 2.0, hh / 2.0, text=txt, anchor="center",
                                    fill=C["ink"], font=F(7, True))
                    cv2._pill = (px0, px1)

            # 注意：必须把 paint 函数对象本身绑进默认参数。若写成 lambda ...: paint(...)，
            # paint 是按“名字”解析的，循环结束后所有 lambda 都会指向最后一次定义的 paint，
            # 结果每行都渲染成最后一项的文字与状态（真实踩过的坑）。
            # e 一律给默认值：控件被销毁时若还有排队的回调，Tk 可能以非预期参数个数调用它。
            r.bind("<Configure>",
                   lambda e=None, c=r, _p=paint: _p(
                       c,
                       e.width if e is not None else c.winfo_width(),
                       e.height if e is not None else c.winfo_height()))
            r.bind("<Enter>",
                   lambda e=None, c=r, _p=paint: _p(c, c.winfo_width(), c.winfo_height(), True))
            r.bind("<Leave>",
                   lambda e=None, c=r, _p=paint: _p(c, c.winfo_width(), c.winfo_height()))

            def on_click(e, _v=v):
                # 点到了右侧「下载」按钮区域 → 触发下载；否则选中该项
                pill = getattr(e.widget, "_pill", None)
                if pill and pill[0] <= e.x <= pill[1]:
                    self.close()
                    if self._downloader:
                        self._downloader(_v)
                    return "break"
                self._choose(_v)
                return "break"

            for wgt in (row, r):
                wgt.bind("<Button-1>", on_click)
        self._draw()
        # 弹层自己吃下所有点击：点在行上由 on_click 处理（已 return "break"，不会冒泡到这里），
        # 点空白区域 / 滚动条以外的区域一律关闭。用 grab_set 把指针事件收归弹层，
        # 避免主窗口抢事件导致"点了没反应、弹层卡住关不掉"的问题。
        def _bg(e=None, _sb=sb):
            if e is not None and _sb is not None and e.widget is _sb:
                return
            self.close()

        pop.bind("<Button-1>", _bg)
        pop.bind("<Escape>", lambda e=None: self.close())
        try:
            pop.grab_set()
        except Exception:
            pass

        # ---------------------------------------------------------------
        # grab 必须"让位"给窗口顶部外壳，否则关不掉程序、也拖不动窗口。
        #
        # Tk 的局部 grab 在 Windows 上靠 SetCapture 实现：一旦捕获，本进程所有鼠标
        # 消息都会被送到弹层，主窗口顶部外壳一带（原生标题栏，或摘掉标题栏后我们自己
        # 画的顶栏）就收不到点击 —— 关闭 / 最小化 / 最大化按钮全部失灵，
        # 表现就是"下拉开着时点 × 关不掉程序"。
        #
        # 指针一进入顶部外壳就收掉弹层：close() 先 grab_release() 再 destroy，
        # 鼠标随即还给系统 / 主窗口，按钮就正常了。之所以整个收掉而不是只释放 grab，
        # 是因为只释放的话，用户从这块区域拖动窗口时弹层会留在原地，
        # 变成"脱锚的野弹层"（它的坐标是 open() 那一刻算死的）。
        # ---------------------------------------------------------------
        def _titlebar_release(e=None, _t=top):
            try:
                pop2 = self._popup
                if pop2 is None or e is None:
                    return
                # 指针还在弹层自己的范围内就别管 —— 弹层向上展开（字段靠窗口顶部）时，
                # 它可能正好压在外壳那条带上，此时悬停列表不该把弹层关掉。
                px, py = pop2.winfo_rootx(), pop2.winfo_rooty()
                if (px <= e.x_root <= px + pop2.winfo_width()
                        and py <= e.y_root <= py + pop2.winfo_height()):
                    return
                cy = _t.winfo_rooty()          # 客户区顶边
                if e.y_root >= cy + _CHROME_H:  # 还在内容区里，保持 grab
                    return
                cx0 = _t.winfo_rootx()
                if cx0 - 12 <= e.x_root <= cx0 + _t.winfo_width() + 12:
                    self.close()
            except Exception:
                pass

        pop.bind("<Motion>", _titlebar_release)
        # Alt+F4 同样会被 grab 吞掉（键盘事件也被收归弹层）。收到就按"点窗口 ×"处理：
        # 先收弹层，再退出主窗口 —— 与标题栏关闭按钮的行为保持一致。
        pop.bind("<Alt-F4>",
                 lambda e=None: (self.close(), self.winfo_toplevel().destroy()))

    def refresh(self):
        """外部（如下载完成后）要求重画折叠态，以反映最新状态。"""
        self._draw()

    def set_values(self, values, labeler=None, keep=True):
        """换掉选项列表（解析完拿到真实清晰度后要用）。

        当前值还在新列表里就保留，否则退回第一项 —— 不能让 textvariable
        指着一个已经不存在的选项。宽度交给外面的布局（grid/pack）决定。
        """
        self._values = list(values)
        if labeler is not None:
            self._labeler = labeler
        cur = self._var.get() if self._var is not None else None
        if self._values and (not keep or cur not in self._values):
            self.set(self._values[0])
        self._draw()
        return self._values

    def _choose(self, v):
        self.set(v)
        self.close()
        if self._cmd:
            self._cmd(v)


class Segmented(tk.Canvas):
    """分段控件（对齐模式 / 自动听写 那种）。"""

    def __init__(self, master, options, value=None, command=None, height=32,
                 radius=R_INPUT, pad=3, size=8, outer=None, fill_width=False):
        outer = outer or C["surface"]
        self._opts = [(o, o) if isinstance(o, str) else (o[0], o[1]) for o in options]
        self._val = value if value is not None else self._opts[0][0]
        self._cmd = command
        self._h, self._r, self._pad, self._size = height, U(radius), U(pad), size
        self._hover = None
        self._rects = []
        tk.Canvas.__init__(self, master, bg=outer, highlightthickness=0, bd=0,
                           height=height)
        self._fill_width = fill_width
        self.configure(width=1 if fill_width else self._natural())
        self.bind("<Configure>", lambda e: self._draw())
        self.bind("<Button-1>", self._click)
        self.bind("<Motion>", self._motion)
        self.bind("<Leave>", lambda e: (setattr(self, "_hover", None), self._draw()))
        self.configure(cursor="hand2")

    def _natural(self):
        try:
            from tkinter import font as tkfont
            fnt = tkfont.Font(font=F(self._size))
            w = sum(fnt.measure(l) + 28 for _, l in self._opts)
        except Exception:
            w = 160
        return int(w + self._pad * 2)

    def _draw(self):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 2 or h < 2:
            return
        self.delete("all")
        rrect(self, 0, 0, w, h, self._r, C["surface_in"])
        n = len(self._opts)
        seg_w = (w - self._pad * 2) / float(n)
        self._rects = []
        for i, (val, lbl) in enumerate(self._opts):
            x1 = self._pad + i * seg_w
            x2 = x1 + seg_w
            on = (val == self._val)
            if on:
                rrect(self, x1, self._pad, x2, h - self._pad, self._r - 2, C["primary"])
            elif self._hover == i:
                rrect(self, x1, self._pad, x2, h - self._pad, self._r - 2, C["surface_hi"])
            self.create_text((x1 + x2) / 2.0, h / 2.0, text=lbl,
                             fill=C["ink"] if on else C["ink3"],
                             font=F(self._size, on))
            self._rects.append((x1, x2, val))

    def _motion(self, e):
        i = None
        for idx, (x1, x2, _v) in enumerate(self._rects):
            if x1 <= e.x <= x2:
                i = idx
                break
        if i != self._hover:
            self._hover = i
            self._draw()

    def _click(self, e):
        for (x1, x2, v) in self._rects:
            if x1 <= e.x <= x2:
                self.set(v)
                if self._cmd:
                    self._cmd(v)
                return

    def get(self):
        return self._val

    def set(self, v):
        self._val = v
        self._draw()

    def set_options(self, options):
        self._opts = [(o, o) if isinstance(o, str) else (o[0], o[1]) for o in options]
        if not self._fill_width:
            self.configure(width=self._natural())
        self._draw()


class Switch(tk.Canvas):
    """开关。整行是"胶囊 + 文案"，点哪儿都能切。"""

    def __init__(self, master, text="", value=False, command=None, outer=None,
                 pw=38, ph=21, size=8, mute_when_off=True):
        outer = outer or C["canvas"]
        self._text = text
        self._val = bool(value)
        self._cmd = command
        self._enabled = True
        self._pw, self._ph, self._size = U(pw), U(ph), size
        self._hover = False
        self._mute = mute_when_off
        try:
            from tkinter import font as tkfont
            tw = tkfont.Font(font=F(size)).measure(text)
        except Exception:
            tw = len(text) * size
        tk.Canvas.__init__(self, master, bg=outer, highlightthickness=0, bd=0,
                           height=max(ph, 20), width=int(pw + 10 + tw + 4))
        self.bind("<Configure>", lambda e: self._draw())
        self.bind("<Button-1>", lambda e: self.toggle())
        self.bind("<Enter>", lambda e: (setattr(self, "_hover", True), self._draw()))
        self.bind("<Leave>", lambda e: (setattr(self, "_hover", False), self._draw()))
        self.configure(cursor="hand2")

    def set_enabled(self, ok):
        """禁用时点不动，并变灰 —— 用于"当前模式下这个开关不参与"。

        没有这个能力的话，某个模式下不适用的开关只能留在界面上，
        用户会以为它生效（其实被忽略了），属于会误导人的界面。
        """
        ok = bool(ok)
        if ok == self._enabled:
            return
        self._enabled = ok
        self.configure(cursor="hand2" if ok else "arrow")
        self._draw()

    def _draw(self):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 2 or h < 2:
            return
        self.delete("all")
        cy = h / 2.0
        muted = not self._enabled
        track = C["primary"] if self._val else C["surface_in"]
        if self._hover and not self._val:
            track = C["surface_hi"]
        rrect(self, 0, cy - self._ph / 2.0, self._pw, cy + self._ph / 2.0,
              self._ph / 2.0, track)
        kr = self._ph / 2.0 - U(3)
        kx = ((self._pw - self._ph / 2.0 - U(1)) if self._val
              else (self._ph / 2.0 + U(1)))
        self.create_oval(kx - kr, cy - kr, kx + kr, cy + kr,
                         fill=C["ink"] if self._val else C["ink3"], outline="")
        self.create_text(self._pw + U(10), cy, text=self._text, anchor="w",
                         fill=(C["ink4"] if muted
                               else C["ink2"] if (self._val or not self._mute)
                               else C["ink4"]),
                         font=F(self._size))

    def get(self):
        return self._val

    def set(self, v, notify=False):
        self._val = bool(v)
        self._draw()
        if notify and self._cmd:
            self._cmd(self._val)

    def toggle(self):
        if not self._enabled:
            return          # 禁用态点不动，免得用户以为改了会生效
        self._val = not self._val
        self._draw()
        if self._cmd:
            self._cmd(self._val)


class Bar(tk.Canvas):
    """进度条。填充是主色 -> 品红的横向渐变，和设计稿一致。"""

    def __init__(self, master, height=8, radius=4, outer=None):
        outer = outer or C["surface"]
        tk.Canvas.__init__(self, master, bg=outer, highlightthickness=0, bd=0,
                           height=height)
        self._h, self._r = height, radius
        self._ratio = 0.0
        self.bind("<Configure>", lambda e: self._draw())

    def _draw(self):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 2 or h < 2:
            return
        self.delete("all")
        rrect(self, 0, 0, w, h, self._r, C["surface_in"])
        fw = int((w - 2) * max(0.0, min(1.0, self._ratio)))
        if fw >= 4:
            rrect(self, 1, 1, 1 + fw, h - 1, max(1, self._r - 1), "#5865F2")
            vgrad(self, 1, 1, 1 + fw, h - 1, "#5865F2", "#EC48BD")
            rrect(self, 1, 1, 1 + fw, h - 1, max(1, self._r - 1), "", "#5865F2", 0)

    def set(self, ratio):
        self._ratio = ratio
        self._draw()


class Scroll(tk.Frame):
    """纵向滚动容器。内容放进 self.inner。"""

    def __init__(self, master, bg=None, style=None):
        bg = bg or C["canvas"]
        tk.Frame.__init__(self, master, bg=bg)
        self._bg = bg
        from tkinter import ttk
        self.cv = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0)
        self.sb = ttk.Scrollbar(self, orient="vertical", style=style or "Dark.Vertical.TScrollbar",
                                command=self.cv.yview)
        self.cv.configure(yscrollcommand=self._on_scroll)
        self.sb.pack(side="right", fill="y")
        self.cv.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(self.cv, bg=bg)
        self._win = self.cv.create_window((0, 0), window=self.inner, anchor="nw")
        self._wheel_ids = []
        self.inner.bind("<Configure>", self._on_inner)
        self.cv.bind("<Configure>", self._on_canvas)
        self.bind("<Destroy>", self._on_destroy, add="+")
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            # bind_all 是全局注册，**只有 unbind_all 能摘**。页面内容重建（切页、
            # 筛选、搜索）会反复销毁重建 Scroll，不摘的话每个僵尸实例都要在每次
            # 滚轮事件里被分发一次，对已销毁控件调 winfo_* 抛 TclError 再被
            # except 吞掉 —— 越用越慢、错误日志越堆越多。
            self._wheel_ids.append((seq, self.cv.bind_all(seq, self._wheel, add="+")))

    def _on_destroy(self, _e=None):
        """控件被销毁时摘掉全局滚轮绑定，别让僵尸实例一直吃事件。"""
        for seq, fid in getattr(self, "_wheel_ids", []):
            try:
                self.cv.unbind_all(seq, fid)
            except Exception:
                pass
        self._wheel_ids = []

    def _on_scroll(self, a, b):
        # 页面滚动了：让打开的下拉弹层重新贴住它的字段（而不是收掉），
        # 这样"页面能正常滚动"和"下拉不会脱锚"可以同时成立。
        if Select._any_open():
            Select.reposition_all_open()
        self.sb.set(a, b)
        if float(a) <= 0.0 and float(b) >= 1.0:
            self.sb.pack_forget()
        else:
            self.sb.pack(side="right", fill="y")

    def _on_inner(self, _e=None):
        self.cv.configure(scrollregion=self.cv.bbox("all"))

    def _on_canvas(self, e):
        self.cv.itemconfigure(self._win, width=e.width)

    def _wheel(self, e):
        try:
            # 注意：这里**不再**因为下拉开着就 return —— 页面滚动是允许的，
            # 弹层由 _on_scroll → Select.reposition_all_open 负责跟随字段。
            # 鼠标停在弹层上时滚轮给弹层自己（Select.open 里绑在 pop 上并 return "break"），
            # 所以"滚下拉"与"滚页面"互不干扰。
            if not self.winfo_ismapped():
                return
            x, y = self.winfo_pointerxy()
            if not (self.winfo_rootx() <= x <= self.winfo_rootx() + self.winfo_width()
                    and self.winfo_rooty() <= y <= self.winfo_rooty() + self.winfo_height()):
                return
            d = e.delta if getattr(e, "delta", 0) else (120 if e.num == 4 else -120)
            self.cv.yview_scroll(int(-d / 120) * 3, "units")
        except Exception:
            pass

    def scroll_top(self):
        self.cv.yview_moveto(0)


# ---------------------------------------------------------------- 组合小件

def label(parent, text, size=9, color=None, bold=False, bg=None, wrap=None,
          anchor="w", num=False):
    lb = tk.Label(parent, text=text, bg=bg or C["surface"],
                  fg=color or C["ink2"], font=FN(size, bold) if num else F(size, bold),
                  anchor=anchor, justify="left")
    if wrap:
        lb.configure(wraplength=wrap)
    return lb


def field_label(parent, text, bg=None):
    """字段上方的小标签（设计稿里是 11.5px 的次级文字）。"""
    return tk.Label(parent, text=text, bg=bg or C["surface"], fg=C["ink3"],
                    font=F(8), anchor="w")

# =====================================================================
# 本地存储：设置 / 历史
# =====================================================================
# 数据存储：历史记录统一放 exe 目录下的 data/ 子文件夹，别在主目录堆一堆散 JSON。
# 都是小 JSON。写坏了宁可当没有，也绝不让它把程序卡住 —— 所以读写全包在 try 里，
# 失败就退回默认值。

DATA_DIR = os.path.join(_HERE, "data")
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")
# 旧版本把 history.json 直接扔在 exe 旁边，首次运行搬进 data/（见 _migrate_history）
LEGACY_HISTORY_FILE = os.path.join(_HERE, "history.json")
SETTINGS_FILE = os.path.join(_HERE, "settings.json")

DEFAULT_PROMPT = "以下是普通话歌曲的歌词，请输出简体中文并带标点。"

DEFAULT_SETTINGS = {
    "model": "",
    "language": "zh",
    "fmt": "lrc",
    "out_dir": "",
    "threads": min(16, os.cpu_count() or 4),
    "subs": "auto",
    "region": "auto",
    "subs_fps": "2.0",
    "min_score": "0.5",
    "vad": False,
    "split": True,
    "merge": True,
    "keep_wav": False,
    "ffmpeg": "",
    "prompt": DEFAULT_PROMPT,
    "dl_browser": "none",
    "dl_dir": "",
    "audio_only": False,
    "cookie_dir": "",
    # 输出字形：auto = 按语言默认（粤语保留原样，其余转简体）
    "script": "auto",
    # 下载文件名模板；空 = 用内置默认 "%(title).80s [%(id)s].%(ext)s"
    "dl_name_tpl": "",
    # Cookie 自动导入：后台把插件导出的文件复制进 Cookie 目录
    "cookie_autoimport": True,
}


def clean_cache(dry=False):
    """清理目录下可安全删除的残留缓存，返回 (移除项数, 释放字节数, [(路径, 大小)])。

    清理对象都是代码不引用、可重建的临时/缓存文件：
      * 整棵目录树的 __pycache__/（Python 字节码，下次运行自动重建）
      * .tmp/（临时目录：解码出的 WAV 等中间文件都在这里，可整棵删）
      * out/_guitest/、output/（早期测试残留，代码不使用）
      * .deps-build/（历史版本的 PyInstaller 构建缓存）

    **绝不碰**：out/downloads、out/converts（正常输出）、data/、cookies/、
    models/（语音模型）、runtime/（ffmpeg/node）、_internal/、
    以及所有 .deps* 依赖目录和 *.exe / *.py / *.spec。

    特意排除 .deps* —— 那是依赖（重装要联网下载几 GB），不是缓存。
    按用户要求，清理范围也不包括 .deps。
    dry=True 时只统计、不删除。错误项会被跳过而非中断。
    """
    import shutil

    def dir_size(p):
        tot = 0
        try:
            for _r, _d, fs in os.walk(p):
                for f in fs:
                    try:
                        tot += os.path.getsize(os.path.join(_r, f))
                    except Exception:
                        pass
        except Exception:
            pass
        return tot

    # 受保护的顶层目录：依赖 / 用户数据 / 打包产物，整棵都不碰。
    #
    # 刻意**不含 out/ 和 runtime/**：
    #   * out/    —— 清理目标只有 __pycache__ 和几个写死的目录名，
    #               out/downloads 永远不会成为目标，业务文件天然安全。
    #               早先把整个 out/ 保护了，导致 out/downloads/__pycache__
    #               这种真缓存清不掉。
    #   * runtime/ —— 里面只有 ffmpeg/node/python 本体和 python 的标准库；
    #               本体不在清理目标里，而 runtime\python\Lib 下的 __pycache__
    #               是纯字节码缓存，删了下次运行自动重建（约 6.7MB）。
    PROTECTED = (".deps", ".deps-ocr", ".deps-ytdlp", ".deps-ejs",
                 ".deps-build",
                 "models", "_internal", "data", "cookies",
                 "dist", "build", ".git")

    def _skip(path):
        """path 是否落在受保护目录里。"""
        try:
            rel = os.path.relpath(path, _HERE)
        except Exception:
            return True
        if rel.startswith(".."):
            return True
        return rel.replace("\\", "/").split("/")[0] in PROTECTED

    def iter_dirs(name, protect=True):
        found = []
        for root, dirs, _f in os.walk(_HERE):
            if os.path.basename(root) == name:
                continue
            if protect:
                # 剪枝：受保护子树不再往下钻（省时间，也避免误收集）
                dirs[:] = [d for d in dirs
                           if not _skip(os.path.join(root, d))]
            for d in dirs:
                if d == name:
                    p = os.path.join(root, d)
                    if protect and _skip(p):
                        continue
                    found.append(p)
        found.sort(key=len, reverse=True)  # 先删深层
        return found

    targets = list(iter_dirs("__pycache__"))
    # .tmp 是整棵临时目录：解码 WAV 等中间文件都在这儿，可整体删
    _tmp = os.path.join(_HERE, ".tmp")
    if os.path.isdir(_tmp):
        targets.append(_tmp)
    for p in (os.path.join(_HERE, ".deps-build"),
              os.path.join(_HERE, "out", "_guitest"),
              os.path.join(_HERE, "output")):
        if os.path.isdir(p):
            targets.append(p)

    # 去重 + 去掉"父目录已包含"的子项（否则删两次、统计也重复）
    uniq, seen = [], set()
    for p in sorted(targets, key=len):
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        # 已经有祖先在列表里就跳过
        parent = os.path.dirname(ap)
        covered = False
        while len(parent) >= len(_HERE):
            if parent in seen:
                covered = True
                break
            if parent == os.path.dirname(parent):
                break
            parent = os.path.dirname(parent)
        if covered:
            continue
        seen.add(ap)
        uniq.append(ap)
    targets = uniq

    removed, freed, items = 0, 0, []
    for p in targets:
        sz = dir_size(p)
        if dry:
            items.append((p, sz))
            removed += 1
            continue
        try:
            shutil.rmtree(p, ignore_errors=True)
            removed += 1
            freed += sz
            items.append((p, sz))
        except Exception:
            items.append((p, None))  # 跳过，不阻塞其余项
    return removed, freed, items



def ensure_dir(path):
    """确保目录存在。失败只返回 False，绝不抛 —— 存储层的任何问题都不该让程序崩。"""
    if not path:
        return False
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception:
        return False


def ensure_data_dir():
    return ensure_dir(DATA_DIR)


def _migrate_history():
    """把旧版直接放在 exe 旁边的 history.json 搬进 data/。

    先完整解析一遍再搬：文件坏了宁可留在原地，也绝不能因为搬家把用户的记录弄丢。
    目标已存在就什么都不做（避免每次启动重复搬）。
    """
    try:
        if not os.path.isfile(LEGACY_HISTORY_FILE):
            return False
        with io.open(LEGACY_HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)                   # 校验：坏文件不搬
        if not isinstance(data, list):
            return False                           # 不是历史数组（dict/字符串）也别搬
        # 目标已存在时，只有"目标有内容"才放弃。目标只是个空壳（[]）仍然搬 ——
        # 否则用户升级后先跑过一次（生成空 history.json），旧记录就永远看不到了。
        if os.path.isfile(HISTORY_FILE):
            try:
                with io.open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    if json.load(f):
                        return False
            except Exception:
                return False
        if not ensure_data_dir():
            return False
        os.replace(LEGACY_HISTORY_FILE, HISTORY_FILE)
        return True
    except Exception:
        return False


def load_json(path, default):
    try:
        with io.open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return dict(default) if isinstance(default, dict) else list(default)
    if isinstance(default, dict):
        out = dict(default)
        if isinstance(data, dict):
            out.update(data)
        return out
    return data if isinstance(data, list) else list(default)


def save_json(path, data):
    """原子写。

    这里原来有两个真 bug，都修掉了：
    - **先 `os.remove` 再 `os.rename`**：两步之间进程被杀（任务管理器结束、关机、
      断电），旧文件已经没了、新文件还没就位 → 配置/历史**直接消失**，而
      `load_json` 的兜底会静默退回默认值，用户完全察觉不到。
      同一个文件里模型下载（`os.replace`）和历史迁移本来就用的是原子替换，只有这里不是。
    - **固定名 `<path>.tmp`**：同时开两个 exe 会互相踩 —— 一边把 tmp 改名走了，
      另一边 `PermissionError`（被吞掉、返回 False、调用方也不看返回值），
      两份内容还可能拼进同一个文件（实测能读出 `Extra data` 的坏 JSON）。
      改成带进程号的名字，两个进程各写各的。
    """
    tmp = None
    try:
        d = os.path.dirname(path)
        if d:
            ensure_dir(d)                     # data/ 不存在就现建
        tmp = "%s.%d.tmp" % (path, os.getpid())
        with io.open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)                  # 原子替换，不需要先删旧的
        return True
    except Exception:
        try:
            if tmp and os.path.isfile(tmp):
                os.remove(tmp)                 # 别把半截文件留在盘上
        except Exception:
            pass
        return False


def _clamp_int(v, lo, hi, default):
    try:
        n = int(float(v))
    except Exception:
        return default
    return max(lo, min(hi, n))


def _clamp_float(v, lo, hi, default):
    try:
        n = float(v)
    except Exception:
        return default
    return max(lo, min(hi, n))


def sanitize_settings(st):
    """把 settings 里可能被外部写坏的字段拉回可用状态。

    实测过的后果：`out_dir` 存成数字 → `os.path.join` 抛 TypeError，
    `StringVar(value=123)` 抛 TclError，**整个设置页打不开、程序起不来**。
    这里统一按类型和范围纠正一次，比在每个使用点加防御更省事。
    """
    st["out_dir"] = str(st.get("out_dir") or "")
    for k in ("model", "language", "fmt", "subs", "region", "ffmpeg", "prompt",
              "dl_browser", "dl_dir", "cookie_dir", "script", "dl_name_tpl"):
        st[k] = str(st.get(k) or "")
    # script 只认 SCRIPT_MODES + auto，坏值退回 auto（否则 apply_script 会静默失效）
    if st["script"] not in list(SCRIPT_MODES) + ["auto"]:
        st["script"] = "auto"
    # threads 不夹范围的话：填 0 会被 faster-whisper 当成"自动"（和用户预期相反），
    # 填负数直接透传给 ctranslate2，填超大值则绕过了"封顶 16"的设计意图。
    st["threads"] = _clamp_int(st.get("threads"), 1, 32,
                              min(16, os.cpu_count() or 4))
    st["subs_fps"] = _clamp_float(st.get("subs_fps"), 0.5, 60.0, 2.0)
    st["min_score"] = _clamp_float(st.get("min_score"), 0.0, 1.0, 0.5)
    for k in ("vad", "split", "merge", "keep_wav", "audio_only"):
        st[k] = bool(st.get(k))
    return st


def history_load():
    _migrate_history()                        # 旧文件顺手搬进 data/
    items = load_json(HISTORY_FILE, [])
    if not isinstance(items, list):
        return []
    # 逐条校验类型：文件被外部写坏时（网盘同步 / 多进程竞争 / 手改），只要有一条
    # 不是 dict，历史页 reload() 里的 r.get() 就会抛 AttributeError，**整页崩成空白**。
    # 这里把坏条目丢掉，保住其余记录。
    return [r for r in items if isinstance(r, dict)]


def history_add(rec):
    items = history_load()
    items.insert(0, rec)
    # 返回成败：写盘失败（目录只读/被占用）时调用方要能告诉用户，
    # 否则用户以为记上了，历史页却是空的。
    return save_json(HISTORY_FILE, items[:200])


def fmt_dur(sec):
    try:
        sec = int(round(float(sec)))
    except Exception:
        return "--:--"
    if sec < 0:
        return "--:--"
    if sec >= 3600:
        return "%d:%02d:%02d" % (sec // 3600, (sec % 3600) // 60, sec % 60)
    return "%02d:%02d" % (sec // 60, sec % 60)


def fmt_cost(sec):
    if not sec:
        return "--"
    if sec < 60:
        return "%.1fs" % sec
    return "%dm%02ds" % (int(sec) // 60, int(sec) % 60)


def fmt_when(ts):
    """把时间戳说成"今天 21:04 / 昨天 21:04 / 10-03 21:04"，比一串数字好认。"""
    try:
        lt = time.localtime(ts)
        now = time.time()
        d = time.strftime("%Y-%m-%d", lt)
        if d == time.strftime("%Y-%m-%d", time.localtime(now)):
            return "今天 " + time.strftime("%H:%M", lt)
        if d == time.strftime("%Y-%m-%d", time.localtime(now - 86400)):
            return "昨天 " + time.strftime("%H:%M", lt)
        return time.strftime("%m-%d %H:%M", lt)
    except Exception:
        return "--"


# =====================================================================
# 窗口外壳：标题栏区的原生处理
# =====================================================================

def _hex_to_colorref(h):
    """#RRGGBB -> Win32 COLORREF（0x00BBGGRR）。DWM 的颜色属性要这个格式。"""
    try:
        h = h.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return (b << 16) | (g << 8) | r
    except Exception:
        return 0


def set_window_border_color(win, hexcolor):
    """把窗口外框那一圈 1px 边框染成应用自己的底色。

    摘掉标题栏后 `WS_THICKFRAME` 还在（缩放/分屏要用），DWM 会照旧在客户区外侧
    画一圈系统默认的深色边框 —— 视觉上就是"最上层还残留一条黑边"。
    这里显式把它染成 `C["canvas"]`，边框就彻底融进界面里。
    `DWMWA_BORDER_COLOR` 在部分 Windows 构建上不存在，调用失败也不影响功能。
    """
    try:
        import ctypes
        hwnd = top_hwnd(win)
        v = ctypes.c_int(_hex_to_colorref(hexcolor))
        ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 34, ctypes.byref(v),
                                                   ctypes.sizeof(v))
        return True
    except Exception:
        return False


def top_hwnd(win):
    """Tk 窗口真正的**顶层** HWND。

    `winfo_id()` 给的是 Tk 内部那个**子**窗口 —— 实测对它调 `GetWindowTextW`
    得到的是空串，而它的父窗口才有真正的窗口标题。圆角、摘标题栏、移动窗口
    都必须发给后者，发错对象（尤其发同步消息）会一直等一个不存在的回应。

    另外 ctypes 默认按 32 位 `c_int` 返回，64 位句柄会被截断成错的值，
    所以这里必须显式声明 `restype / argtypes`。
    """
    try:
        import ctypes
        wid = int(win.winfo_id())
        u = ctypes.windll.user32
        gp = u.GetParent
        gp.restype = ctypes.c_void_p
        gp.argtypes = [ctypes.c_void_p]
        top = gp(ctypes.c_void_p(wid))
        return int(top) if top else wid
    except Exception:
        try:
            return int(win.winfo_id())
        except Exception:
            return 0


def round_native_window(win):
    """把原生窗口的四角做圆，并把标题栏切成深色。

    设计稿的窗口是 24px 圆角、深色标题栏。tkinter 画不到窗口外框，但 Windows 11
    的 DWM 可以：角偏好 = 2（圆角），沉浸式深色标题栏 = 1。Win10 上这些属性
    不存在，调用会失败，那就退回直角 + 系统色标题栏，不影响使用。

    注意：属性必须在窗口**真正映射之后**再设一次才生效，刚创建时设完，
    系统画第一帧会把标题栏颜色覆盖回去。所以这里既设值，也顺手触发一次
    非客户区重画（SWP_FRAMECHANGED），调用方还会在 after() 里再调一次。
    """
    try:
        import ctypes
        win.update_idletasks()
        hwnd = top_hwnd(win)
        # 深色标题栏：DWM 的属性号在部分 Windows 构建上"调用成功但不生效"——
        # 本机实测就是这样（返回 0，标题栏依然白）。必须再配合 uxtheme 的两个
        # 未公开序号函数才真的变黑：135 = SetPreferredAppMode(ForceDark)，
        # 133 = AllowDarkModeForWindow(hwnd, True)。序号调用是社区通行做法，
        # 任何一步失败都不影响功能，只是标题栏保持系统色。
        try:
            ux = ctypes.WinDLL("uxtheme")
            ux[135](2)
            ux[133](hwnd, True)
        except Exception:
            pass
        # 20 是 Win10 1809+ / Win11 的属性号，19 是更早的 Win10 构建用的；两个都设一遍，
        # 不存在的那个调用会失败，不影响后面的。33 = 窗口圆角偏好。
        for attr, val in ((20, 1), (19, 1), (33, 2)):
            v = ctypes.c_int(val)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr,
                                                       ctypes.byref(v), ctypes.sizeof(v))
        # 外框那圈 1px 边框也染成底色，否则会残留一条系统深色"黑边"
        set_window_border_color(win, C["canvas"])
        SWP_NOSIZE, SWP_NOMOVE, SWP_NOZORDER, SWP_NOACTIVATE, SWP_FRAMECHANGED = (
            0x0001, 0x0002, 0x0004, 0x0010, 0x0020)
        ctypes.windll.user32.SetWindowPos(
            hwnd, 0, 0, 0, 0, 0,
            SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED)
    except Exception:
        pass


# 原生标题栏是否已被摘掉。摘掉后截图不要再往上留 32px 的标题栏高度。
_FRAMELESS = False
# 窗口顶部"外壳区"的高度：还有原生标题栏时约 32px；摘掉之后顶栏(68px)自己就是
# 标题栏。指针进入这块区域就收掉下拉弹层 —— 里面的窗口控制按钮会被 grab 挡住，
# 而且从这里拖动窗口会让弹层脱锚。
# 存的是"逻辑像素"，比较前由 init_ui_scale() 换算 —— 模块加载时 UI_SCALE 还不存在。
_CHROME_H = 32


def frameless_top(win):
    """把窗口做成**完全无框**：摘掉 `WS_CAPTION` + `WS_THICKFRAME`。

    设计上标题栏本来就该和应用同一套色板（`C["titlebar"]`），而且原生那条
    深色标题栏会把大标题重复画一遍（顶栏里已经有一遍），颜色也对不齐。

    **为什么连 `WS_THICKFRAME` 一起摘**（实测踩出来的）：只摘 caption 时，
    那圈看不见的缩放边框仍然会被 DWM 画成 **#202020 的深灰**（本机每边 7px），
    正好压在顶栏上方 —— 就是用户看到的"最上层残留黑边"。
    - `DWMWA_BORDER_COLOR`(属性 34) 在本机 Windows 上**不支持**
      （DwmSet/DwmGet 都返回 E_INVALIDARG），染不了色；
    - `DwmExtendFrameIntoClientArea(-1)` 只是把那一圈变成**透明**，
      露出桌面颜色（浅色桌面上就是一圈白边），并不算解决；
    - 摘掉 `WS_THICKFRAME` 后 `_frame_insets()` 变成 (0,0,0,0)，
      客户区直接顶到窗口边缘，应用自己的底色铺满整个窗口 —— 干净。
    代价是失去"拖边缘缩放 / 贴边分屏"，改用右下角的自绘缩放手柄
    （见 `ResizeGrip`），双击顶栏最大化、拖顶栏移动都还在。

    拖动窗口不自己算 move，而是发 `WM_NCLBUTTONDOWN(HTCAPTION)` 给系统
    （见 `App._bind_title_drag`），这样拖到边缘的贴边分屏、双击最大化都由系统处理。
    """
    global _FRAMELESS
    global _CHROME_H
    try:
        import ctypes
        win.update_idletasks()
        u = ctypes.windll.user32
        u.GetWindowLongPtrW.restype = ctypes.c_void_p
        u.GetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int]
        u.SetWindowLongPtrW.restype = ctypes.c_void_p
        u.SetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        hwnd = top_hwnd(win)
        GWL_STYLE = -16
        WS_CAPTION, WS_THICKFRAME = 0x00C00000, 0x00040000
        style = int(u.GetWindowLongPtrW(hwnd, GWL_STYLE) or 0)
        new = style & ~(WS_CAPTION | WS_THICKFRAME)
        if new == style:
            _FRAMELESS = True
            _CHROME_H = U(68)                     # 顶栏自己接管了标题栏
            return True
        u.SetWindowLongPtrW(hwnd, GWL_STYLE, new)
        SWP_NOSIZE, SWP_NOMOVE, SWP_NOZORDER, SWP_NOACTIVATE, SWP_FRAMECHANGED = (
            0x0001, 0x0002, 0x0004, 0x0010, 0x0020)
        u.SetWindowPos(hwnd, 0, 0, 0, 0, 0,
                       SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE
                       | SWP_FRAMECHANGED)
        _FRAMELESS = True
        _CHROME_H = U(68)                      # 顶栏自己接管了标题栏
        return True
    except Exception:
        return False


def init_scrollbar_style(root):
    try:
        from tkinter import ttk
        st = ttk.Style(root)
        try:
            st.theme_use("clam")
        except Exception:
            pass
        st.configure("Dark.Vertical.TScrollbar", background=C["surface_in"],
                     troughcolor=C["canvas"], bordercolor=C["canvas"],
                     darkcolor=C["surface_in"], lightcolor=C["surface_in"],
                     arrowcolor=C["ink4"], relief="flat", arrowsize=10)
        st.map("Dark.Vertical.TScrollbar",
               background=[("active", C["surface_hi"])])
        # 下拉弹层专用的"细"滚动条：比页面滚动条窄一半（width=6 → 实测 8px），
        # 颜色跟弹层底色/次级文字对齐，箭头颜色取弹层底色让箭头"隐形"，
        # 只剩一根细长的圆润滑杆 —— 既变细又跟深色 UI 匹配。
        st.configure("Thin.Vertical.TScrollbar", background=C["ink4"],
                     troughcolor=C["surface"], bordercolor=C["surface"],
                     darkcolor=C["ink4"], lightcolor=C["ink4"],
                     arrowcolor=C["surface"], relief="flat", arrowsize=8, width=6)
        st.map("Thin.Vertical.TScrollbar",
               background=[("active", C["ink3"])],
               troughcolor=[("active", C["surface"])])
        # 去掉上下箭头按钮，只留一条细滑轨+滑杆（clam 默认布局带两个箭头按钮，
        # 会在细滚动条上留下小方块状残影，反而更不像深色 UI）
        st.layout("Thin.Vertical.TScrollbar",
                  [("Vertical.Scrollbar.trough",
                    {"children": [("Vertical.Scrollbar.thumb",
                                   {"expand": "1", "sticky": "nswe"})],
                     "sticky": "ns"})])
    except Exception:
        pass


# =====================================================================
# 侧栏导航项
# =====================================================================

class NavBtn(tk.Canvas):
    def __init__(self, master, icon_name, text, command, active=False, collapsed=False):
        self._icon, self._text, self._cmd = icon_name, text, command
        self._active, self._collapsed = active, collapsed
        self._hover = False
        w = 44 if collapsed else 208
        tk.Canvas.__init__(self, master, bg=C["canvas"], highlightthickness=0, bd=0,
                           width=w, height=42)
        self.bind("<Configure>", lambda e: self._draw())
        self.bind("<Enter>", lambda e: (setattr(self, "_hover", True), self._draw()))
        self.bind("<Leave>", lambda e: (setattr(self, "_hover", False), self._draw()))
        self.bind("<Button-1>", lambda e: self._cmd() if self._cmd else None)
        self.configure(cursor="hand2")

    def _draw(self):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 2 or h < 2:
            return
        self.delete("all")
        if self._active:
            rrect(self, 0, 0, w, h, R_BTN, C["surface"])
        elif self._hover:
            rrect(self, 0, 0, w, h, R_BTN, "#171C46")
        icol = C["magenta"] if self._active else C["ink3"]
        if self._collapsed:
            icon(self, self._icon, w / 2.0, h / 2.0, 18, icol)
        else:
            icon(self, self._icon, U(13) + U(8.5), h / 2.0, 17, icol)
            self.create_text(U(13) + U(17) + U(11), h / 2.0, text=self._text, anchor="w",
                             fill=C["ink"] if self._active else C["ink3"],
                             font=F(9, self._active))

    def set_active(self, on):
        self._active = bool(on)
        self._draw()

    def set_collapsed(self, on):
        self._collapsed = bool(on)
        self.configure(width=U(44) if on else U(208))
        self._draw()


def _knock_corners(cv, x1, y1, x2, y2, r, outer):
    """把圆角之外的四个小三角用底色盖掉 —— Canvas 没有"裁剪"能力，只能这样补。"""
    import math
    if r <= 0.5:
        return
    jobs = ((x1 + r, y1 + r, 180, 270, (x1, y1)),
            (x2 - r, y1 + r, 270, 360, (x2, y1)),
            (x2 - r, y2 - r, 0, 90, (x2, y2)),
            (x1 + r, y2 - r, 90, 180, (x1, y2)))
    for (cx, cy, a0, a1, corner) in jobs:
        pts = list(corner)
        for i in range(a0, a1 + 1, 6):
            t = math.radians(i)
            pts += [cx + r * math.cos(t), cy + r * math.sin(t)]
        cv.create_polygon(*pts, fill=outer, outline="")


def grad_round_box(cv, x1, y1, x2, y2, r, c1, c2, outer, vertical=True):
    if vertical:
        vgrad(cv, x1, y1, x2, y2, c1, c2)
    else:
        vgrad(cv, x1, y1, x1 + max(1, int(x2 - x1)), y2, c1, c2)
    _knock_corners(cv, x1, y1, x2, y2, U(r), outer)


class IconBtn(tk.Canvas):
    """方形的图标按钮（顶栏的设置 / 帮助）。"""

    def __init__(self, master, icon_name, command=None, size=34, icon_size=16,
                 outer=None, fill=None, radius=12, tip=None):
        outer = outer or C["canvas"]
        fill = fill or C["surface"]
        tk.Canvas.__init__(self, master, bg=outer, highlightthickness=0, bd=0,
                           width=size, height=size)
        self._icon, self._size, self._isize = icon_name, size, icon_size
        self._fill, self._hover, self._cmd = fill, False, command
        self._outer, self._r = outer, radius
        self.bind("<Configure>", lambda e: self._draw())
        self.bind("<Enter>", lambda e: (setattr(self, "_hover", True), self._draw()))
        self.bind("<Leave>", lambda e: (setattr(self, "_hover", False), self._draw()))
        self.bind("<Button-1>", lambda e: command() if command else None)
        self.configure(cursor="hand2")

    def _draw(self):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 2:
            return
        self.delete("all")
        rrect(self, 0, 0, w, h, self._r, C["surface_hi"] if self._hover else self._fill)
        icon(self, self._icon, w / 2.0, h / 2.0, self._isize, C["ink2"])

    def set_icon(self, name):
        self._icon = name
        self._draw()


class Chip(tk.Canvas):
    """胶囊标签。模型状态、模式标签都用它。"""

    def __init__(self, master, text="", dot=None, icon_name=None, command=None,
                 height=34, outer=None, fill=None, fg=None, radius=None, size=9,
                 bold=False, pad_l=13, pad_r=12, chevron=False):
        outer = outer or C["canvas"]
        fill = fill or C["surface"]
        self._text = text
        self._dot = dot
        self._icon = icon_name
        self._fg = fg or C["ink2"]
        self._h = height
        self._r = radius if radius is not None else height // 2
        self._size, self._bold = size, bold
        self._pl, self._pr = pad_l, pad_r
        self._chev = chevron
        self._hover = False
        self._cmd = command
        tk.Canvas.__init__(self, master, bg=outer, highlightthickness=0, bd=0,
                           height=height, width=self._natural())
        self._outer, self._fill = outer, fill
        if command:
            self.bind("<Enter>", lambda e: (setattr(self, "_hover", True), self._draw()))
            self.bind("<Leave>", lambda e: (setattr(self, "_hover", False), self._draw()))
            self.bind("<Button-1>", lambda e: command())
            self.configure(cursor="hand2")
        self.bind("<Configure>", lambda e: self._draw())

    def _natural(self):
        try:
            from tkinter import font as tkfont
            w = tkfont.Font(font=F(self._size, self._bold)).measure(self._text)
        except Exception:
            w = len(self._text) * self._size
        w += self._pl + self._pr
        if self._dot:
            w += 15
        if self._icon:
            w += 22
        if self._chev:
            w += 19
        return int(w)

    def _draw(self):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 2:
            return
        self.delete("all")
        fill = C["surface_hi"] if self._hover else self._fill
        rrect(self, 0, 0, w, h, self._r, fill)
        x = self._pl
        if self._dot:
            cy = h / 2.0
            self.create_oval(x - U(3.5), cy - U(3.5), x + U(3.5), cy + U(3.5),
                             fill=self._dot, outline="")
            x += U(15)
        if self._icon:
            icon(self, self._icon, x + U(7), h / 2.0, 14, self._fg)
            x += U(22)
        self.create_text(x, h / 2.0, text=self._text, anchor="w",
                         fill=self._fg, font=F(self._size, self._bold))
        if self._chev:
            icon(self, "down", w - self._pr - 1, h / 2.0, 11, C["ink3"])

    def set_text(self, text):
        self._text = text
        self.configure(width=self._natural())
        self._draw()

    def set_fill(self, fill):
        self._fill = fill
        self._draw()


# =====================================================================
# 应用外壳
# =====================================================================

NAV_ITEMS = (("generate", "wave", "生成歌词"),
             ("models", "cube", "模型管理"),
             ("download", "download", "视频下载"),
             ("convert", "swap", "格式转换"),
             ("history", "clock", "历史记录"),
             ("settings", "gear", "设置"))

COLLAPSE_AT = 1200          # 逻辑像素：窄于此宽度侧栏收成图标轨道（比较时用 U() 换算）

SNAP_EDGE = 8               # 逻辑像素：光标进入屏幕边缘这个距离内就吸附（比较时用 U() 换算）
SNAP_MOVE = 5               # 逻辑像素：拖动超过这个位移才算"真在拖"，用于最大化状态下先还原再跟随

# 换几何（最大化 / 还原 / 半屏）时的过渡参数。
# 本机实测：Tk 改一次窗口尺寸要 ~330ms，其中**布局只占 5.6ms，剩下全是重绘**；
# 而且拆成补间动画没用 —— 每帧都是整窗重排，N 帧就是 N×250ms（8 帧 = 2.1 秒）。
# 所以不做逐帧缩放，改成：淡出 -> 在屏幕外按新尺寸排好版（不重绘，5ms）-> 挪回来
# 一次画完（~75ms，此时窗口是透明的）-> 淡入。全程 ~240ms，看不到没画完的半截画面。
MORPH_FADE_OUT = 45         # 毫秒：换几何之前先淡出到透明
MORPH_FADE_IN = 95          # 毫秒：几何就位之后再淡回来
MORPH_STEP = 26             # 毫秒：alpha 每一步的间隔
# 每次 attributes("-alpha", …) 都会打一次窗口管理器往返，并让整个窗口重新
# 分层合成 —— 是过渡里最贵的一步。原来按 16ms 走，淡出+淡入要 ~11 次；
# 降到 26ms 只需 ~6 次，肉眼看不出台阶，卡顿却少一半。
# （30fps 的 alpha 渐变在半透明窗口上看不出闪烁，实测可接受。）
MORPH_OFF = (-30000, -30000)  # 排版时临时挪到的屏幕外坐标（任何多屏布局之外）


class WinCtl(tk.Canvas):
    """顶栏右侧的窗口控制按钮（最小化 / 最大化 / 关闭）。

    原生标题栏被摘掉后，这三个按钮由顶栏接管。画法沿用整体的自绘语言：
    纯图标 + 悬停整块高亮，关闭按钮悬停用 `err` 色（和 danger 语义一致）。
    按钮通高贴在窗口右缘，交互面积和 Windows 11 标题栏一致。
    """

    def __init__(self, master, kind, command, width=46, height=68, bg=None):
        self.kind = kind
        self._cmd = command
        self._bg = bg or C["canvas"]
        self._hover = False
        self._maxed = False
        tk.Canvas.__init__(self, master, bg=self._bg, width=width, height=height,
                           highlightthickness=0, bd=0)
        self.bind("<Configure>", lambda e: self._draw())
        self.bind("<Enter>", lambda e: self._set(True))
        self.bind("<Leave>", lambda e: self._set(False))
        self.bind("<Button-1>", self._click)
        self.configure(cursor="hand2")
        self._draw()

    def _set(self, h):
        self._hover = h
        self._draw()

    def set_maximized(self, flag):
        """最大化时把"方框"换成"还原"的双层方框，和系统一致。"""
        flag = bool(flag)
        if flag == self._maxed:
            return          # 状态没变就别重画（Configure 高频触发时省掉大量重绘）
        self._maxed = flag
        self._draw()

    def _draw(self):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 2 or h < 2:
            return
        self.delete("all")
        danger = (self.kind == "close")
        if self._hover:
            self.create_rectangle(0, 0, w, h,
                                  fill=C["err"] if danger else C["surface_hi"],
                                  outline="")
            col = C["ink"] if danger else C["ink"]
        else:
            col = C["ink3"]
        cx, cy = w / 2.0, h / 2.0
        lw = U(1.4)
        if self.kind == "min":
            self.create_line(cx - U(6), cy, cx + U(6), cy, fill=col, width=lw)
        elif self.kind == "max":
            r = U(5.5)
            if self._maxed:      # 还原：两个错开的方框
                self.create_rectangle(cx - r + U(2.5), cy - r - U(2.5), cx + r + U(2.5),
                                      cy + r - U(2.5), outline=col, width=lw)
                self.create_rectangle(cx - r, cy - r + U(2), cx + r - U(3), cy + r,
                                      outline=col, width=lw, fill=self._bg)
                self.create_rectangle(cx - r, cy - r + U(2), cx + r - U(3), cy + r,
                                      outline=col, width=lw)
            else:
                self.create_rectangle(cx - r, cy - r, cx + r, cy + r,
                                      outline=col, width=lw)
        else:
            d = 5.5
            self.create_line(cx - d, cy - d, cx + d, cy + d, fill=col, width=lw)
            self.create_line(cx + d, cy - d, cx - d, cy + d, fill=col, width=lw)

    def _click(self, _e=None):
        if self._cmd:
            self._cmd()
        return "break"


class ResizeGrip(tk.Canvas):
    """右下角的缩放手柄。

    摘掉 `WS_THICKFRAME` 后窗口没有可拖的边缘了，边角缩放也用不了，
    所以自己画一个：按住拖动改窗口尺寸 respects 最小尺寸。
    平时只有三道很淡的斜线提示，悬停才明显 —— 跟整体的自绘语言一致。
    """

    def __init__(self, master, app, size=18):
        self.app = app
        self._hover = False
        self._drag = None
        tk.Canvas.__init__(self, master, width=size, height=size,
                           bg=C["canvas"], highlightthickness=0, bd=0)
        # Tk 默认光标里没有斜向缩放 curs（nwse-resize 之类在 Windows 上不存在），
        # 用右下角光标，语义最接近
        self.configure(cursor="bottom_right_corner")
        self.bind("<Enter>", lambda e: self._set(True))
        self.bind("<Leave>", lambda e: self._set(False))
        self.bind("<Button-1>", self._down)
        self.bind("<B1-Motion>", self._move)
        self.bind("<ButtonRelease-1>", self._up)
        self._draw()

    def _set(self, h):
        self._hover = h
        self._draw()

    def _draw(self):
        self.delete("all")
        col = C["ink3"] if self._hover else C["ink5"]
        for i in range(3):
            off = U(4) + i * U(4)
            self.create_line(self.winfo_width() - off, self.winfo_height() - U(3),
                             self.winfo_width() - U(3), self.winfo_height() - off,
                             fill=col, width=U(1.2))

    def _down(self, e):
        r = self.app._win_rect()
        if r:
            self._drag = (e.x_root, e.y_root, r)
        return "break"

    def _move(self, e):
        if not self._drag:
            return "break"
        x0, y0, r = self._drag
        dx, dy = e.x_root - x0, e.y_root - y0
        w = max(U(1024), r[2] + dx)       # 与 root.minsize(1024, 680) 对齐
        h = max(U(680), r[3] + dy)
        self.app._set_win_rect((r[0], r[1], w, h))
        return "break"

    def _up(self, _e=None):
        self._drag = None
        return "break"

    def set_visible(self, on):
        try:
            if on:
                self.place(relx=1.0, rely=1.0, anchor="se")
            else:
                self.place_forget()
        except Exception:
            pass


class App(object):
    """界面外壳 + 后台任务。页面各管各的，这里只负责顶栏 / 侧栏 / 页面切换。"""

    def __init__(self, root, check_only=False):
        self.root = root
        self.check_only = check_only
        self.settings = sanitize_settings(load_json(SETTINGS_FILE, DEFAULT_SETTINGS))
        # Cookie 目录可以在「视频下载」页上改，存在 settings 里 —— 这里恢复上次的选择
        set_cookie_dir(self.settings.get("cookie_dir"))
        self.q = queue.Queue()
        self.state = {"running": False, "t0": 0.0, "out": None, "last": None}
        self.collapsed = False
        self._maxed = False
        self._restore_geom = None
        self._restore_rect = None
        self._mv = None              # 拖顶栏移动窗口时的 (起始光标, 起始矩形)
        self._pend = None            # 最大化状态下按下但还没真的开始拖：(x, y)
        self._snap = None            # 当前吸附态：'max' / 'left' / 'right' / None
        self._snap_prev = None       # 进入吸附前的自由尺寸 (rect, offy, offx)
        self._morph_phase = None     # 换几何的过渡：None / 'out'(淡出) / 'in'(淡入)
        self._morph_final = None     # 过渡的目标矩形
        self._morph_after = None     # 过渡排着的 after 回调 id
        self._morph_t0 = 0.0         # 当前这半段的起始时刻
        self._download_busy = None
        self.pages = {}
        self.current = None
        self.navs = {}
        self._build_header()
        tk.Frame(root, height=1, bg=C["line_soft"]).pack(fill="x")
        self._build_body()
        self._build_pages()
        self.select("generate")
        self.refresh_model_chip()
        root.bind("<Configure>", self._on_resize, add="+")
        root.after(120, self.tick)
        # 右下角缩放手柄（窗口已无边框，边缘拖拽缩放不可用了）
        self.grip = ResizeGrip(self.root, self)
        self.grip.set_visible(True)

    # ---------------- 顶栏（同时也是窗口标题栏）----------------
    def _build_header(self):
        # 原生标题栏已被 frameless_top 摘掉，所以顶栏就是窗口最上层那一层：
        # 背景换成色板里的 titlebar 色（比 canvas 更深的一档），和整体同一套色。
        hd = tk.Frame(self.root, bg=C["canvas"], height=68)
        hd.pack(fill="x")
        hd.pack_propagate(False)
        left = tk.Frame(hd, bg=C["canvas"])
        left.pack(side="left", padx=24)
        mk = tk.Canvas(left, width=38, height=38, bg=C["canvas"],
                       highlightthickness=0, bd=0)
        mk.pack(side="left", pady=15)

        def draw_mark(_e=None):
            mk.delete("all")
            grad_round_box(mk, 0, 0, 38, 38, 13, "#5966F3", "#EC48BD", C["canvas"])
            icon(mk, "wave", 19, 19, 18, C["ink"])

        mk.bind("<Configure>", draw_mark)
        draw_mark()

        txt = tk.Frame(left, bg=C["canvas"])
        txt.pack(side="left", padx=(12, 0), pady=15)
        tk.Label(txt, text="歌词生成器", bg=C["canvas"], fg=C["ink"],
                 font=F(13, True)).pack(anchor="w")
        tk.Label(txt, text="从音乐 / 视频生成歌词与字幕 · 全程本地处理",
                 bg=C["canvas"], fg=C["ink3"], font=F(8)).pack(anchor="w")

        # 窗口控制：贴在窗口右缘、顶栏通高。先 pack 的靠右，所以先建它。
        wc = tk.Frame(hd, bg=C["canvas"])
        wc.pack(side="right")
        self.btn_close = WinCtl(wc, "close", self._win_close, bg=C["canvas"])
        self.btn_close.pack(side="right", fill="y")
        self.btn_max = WinCtl(wc, "max", self._win_toggle_max, bg=C["canvas"])
        self.btn_max.pack(side="right", fill="y")
        self.btn_min = WinCtl(wc, "min", self._win_min, bg=C["canvas"])
        self.btn_min.pack(side="right", fill="y")

        right = tk.Frame(hd, bg=C["canvas"])
        right.pack(side="right", padx=24)
        self.chip_model = Chip(right, text="检测中…", dot=C["ink4"],
                               command=lambda: self.select("models"), chevron=True,
                               outer=C["canvas"])
        self.chip_model.pack(side="left", pady=17)
        IconBtn(right, "gear", command=lambda: self.select("settings"),
                outer=C["canvas"]).pack(side="left", padx=(10, 0), pady=17)
        IconBtn(right, "help", command=self.show_help,
                outer=C["canvas"]).pack(side="left", padx=(10, 0), pady=17)

        self._bind_title_drag(hd)

    # ---------------- 窗口控制 / 拖动 ----------------
    def _win_hwnd(self):
        """顶层窗口 HWND。

        `winfo_id()` 给的是 Tk 内部那个**子**窗口（拿它的窗口标题是空的），
        真正的顶层窗口要再往上一层 —— 拖动 / 最大化 / 圆角都必须发给它。

        两点必须注意：
        * ctypes 默认按 32 位 `c_int` 返回，64 位句柄会被截断成一个**错的**值。
          拿错的句柄去发同步消息，会一直等一个根本不会回你的窗口 —— 表现就是卡死。
          所以这里显式声明 `restype / argtypes`。
        * 取一次就缓存：窗口不会重建，没必要每次都进 ctypes。
        """
        h = getattr(self, "_hwnd", 0)
        if h:
            return h
        self._hwnd = top_hwnd(self.root)
        return self._hwnd

    @staticmethod
    def _work_area():
        """屏幕工作区（已排除任务栏/停靠栏）：(x, y, w, h)。取不到返回 None。"""
        try:
            import ctypes

            class RECT(ctypes.Structure):
                _fields_ = [("l", ctypes.c_long), ("t", ctypes.c_long),
                            ("r", ctypes.c_long), ("b", ctypes.c_long)]

            r = RECT()
            # SPI_GETWORKAREA = 0x0030
            if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0,
                                                           ctypes.byref(r), 0):
                return (r.l, r.t, r.r - r.l, r.b - r.t)
        except Exception:
            pass
        return None

    def _apply_round(self, pref):
        """DWM 窗口圆角偏好：2 = 圆角，0 = 直角（最大化时用直角，边角不漏黑缝）。"""
        try:
            import ctypes
            v = ctypes.c_int(pref)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                self._win_hwnd(), 33, ctypes.byref(v), ctypes.sizeof(v))
        except Exception:
            pass

    def _win_min(self):
        try:
            import ctypes
            ctypes.windll.user32.ShowWindow(self._win_hwnd(), 6)      # SW_MINIMIZE
        except Exception:
            self.root.iconify()

    def _win_rect(self):
        """当前窗口外框矩形 (x, y, w, h)（用外框坐标，和 SetWindowPos 同一套）。"""
        try:
            import ctypes

            class RECT(ctypes.Structure):
                _fields_ = [("l", ctypes.c_long), ("t", ctypes.c_long),
                            ("r", ctypes.c_long), ("b", ctypes.c_long)]

            u = ctypes.windll.user32
            gr = u.GetWindowRect
            gr.restype = ctypes.c_bool
            gr.argtypes = [ctypes.c_void_p, ctypes.POINTER(RECT)]
            r = RECT()
            if gr(ctypes.c_void_p(self._win_hwnd()), ctypes.byref(r)):
                return (r.l, r.t, r.r - r.l, r.b - r.t)
        except Exception:
            pass
        return None

    def _set_win_rect(self, rect):
        """把**外框**放到指定矩形。最大化必须用外框坐标：Tk 的 geometry() 说的是
        客户区，外框会差一个边框宽度（实测 7px），贴不满工作区就会露缝。"""
        x, y, w, h = rect
        try:
            import ctypes
            u = ctypes.windll.user32
            sp = u.SetWindowPos
            sp.restype = ctypes.c_bool
            sp.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                           ctypes.c_int, ctypes.c_int, ctypes.c_int,
                           ctypes.c_int, ctypes.c_uint]
            SWP_NOZORDER, SWP_NOACTIVATE = 0x0004, 0x0010
            sp(ctypes.c_void_p(self._win_hwnd()), None, int(x), int(y),
               int(w), int(h), SWP_NOZORDER | SWP_NOACTIVATE)
            return True
        except Exception:
            return False

    def _frame_insets(self):
        """外框相对客户区四边的留白 (left, top, right, bottom)。

        摘掉标题栏但保留 `WS_THICKFRAME` 时，窗口仍有一圈不可见的缩放边框
        （本机实测每边 7px）。所以要让**客户区**铺满工作区，外框必须向左上探出
        这些留白、尺寸再加上它们 —— 这也正是系统自己的最大化算法。
        """
        try:
            import ctypes

            class POINT(ctypes.Structure):
                _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

            class RECT(ctypes.Structure):
                _fields_ = [("l", ctypes.c_long), ("t", ctypes.c_long),
                            ("r", ctypes.c_long), ("b", ctypes.c_long)]

            h = self._win_hwnd()
            u = ctypes.windll.user32
            wr, cr = RECT(), RECT()
            u.GetWindowRect(h, ctypes.byref(wr))
            u.GetClientRect(h, ctypes.byref(cr))
            p = POINT(0, 0)
            u.ClientToScreen(h, ctypes.byref(p))
            left, top = p.x - wr.l, p.y - wr.t
            right = (wr.r - wr.l) - (cr.r - cr.l) - left
            bottom = (wr.b - wr.t) - (cr.b - cr.t) - top
            return (left, top, right, bottom)
        except Exception:
            return (0, 0, 0, 0)

    def _client_rect(self):
        """客户区在屏幕上的矩形 (x, y, w, h) —— 也就是应用内容真正占的地方。"""
        try:
            import ctypes

            class POINT(ctypes.Structure):
                _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

            class RECT(ctypes.Structure):
                _fields_ = [("l", ctypes.c_long), ("t", ctypes.c_long),
                            ("r", ctypes.c_long), ("b", ctypes.c_long)]

            h = self._win_hwnd()
            u = ctypes.windll.user32
            cr = RECT()
            u.GetClientRect(h, ctypes.byref(cr))
            p = POINT(cr.l, cr.t)
            u.ClientToScreen(h, ctypes.byref(p))
            return (p.x, p.y, cr.r - cr.l, cr.b - cr.t)
        except Exception:
            return None

    def _win_toggle_max(self):
        """最大化/还原：**自己算几何**，不走 ShowWindow(SW_MAXIMIZE)。

        三个原因，都是实测踩出来的：
        1. `SW_MAXIMIZE` 会触发 DWM 缩放动画，而启动后还有几个 `after()` 里排着
           `SWP_FRAMECHANGED`（重画圆角、补设样式），撞上就可能把窗口卡死；
        2. 自己摆位没有动画，也就不会有黑边/卡死；
        3. **定位要让客户区而不是外框铺满工作区** —— 带 `WS_THICKFRAME` 的窗口外框
           每边多出约 7px 不可见边框，按外框摆位会让应用内容内缩一圈、四边露出
           系统的深色边框（实测这正是"占屏没占满"的原因）。所以外框要向左上探出
           `_frame_insets()`、尺寸再加上它们，等价于系统自己的最大化算法。

        最大化时顺手把圆角关掉，四角就不会有缺口；再点一次（或双击顶栏）恢复。

        摆位统一走 `_morph_to()`（过渡），状态同步走 `_set_maxed()` —— 两者分开，
        是为了让"先退出最大化、再摆到半屏"这种连续动作只做一次过渡。
        """
        if self._maxed:
            self._set_maxed(False)
            r = getattr(self, "_restore_rect", None)
            if r:
                self._morph_to(r)
            elif getattr(self, "_restore_geom", None):
                try:
                    self.root.geometry(self._restore_geom)
                except Exception:
                    pass
            return
        try:
            self._restore_geom = self.root.geometry()
        except Exception:
            self._restore_geom = None
        self._restore_rect = self._win_rect()
        self._set_maxed(True)
        wa = self._work_area()
        if wa:
            x, y, w, h = wa
            l, t, r, b = self._frame_insets()
            self._morph_to((x - l, y - t, w + l + r, h + t + b))

    def _set_maxed(self, flag):
        """只同步"最大化"这个状态的**视觉部分**，不动几何。

        按钮图标 / 圆角 / 缩放手柄都在这里切换。几何由调用方决定怎么摆
        （`_win_toggle_max` 用 `_morph_to`，拖动相关的路径直接 `SetWindowPos`）。
        """
        self._maxed = bool(flag)
        self._snap = "max" if flag else None
        b = getattr(self, "btn_max", None)
        if b is not None:
            b.set_maximized(bool(flag))
        self._apply_round(0 if flag else 2)     # 最大化时直角，边角不漏黑缝
        if getattr(self, "grip", None) is not None:
            self.grip.set_visible(not flag)     # 占屏时缩放手柄没有意义

    # ---------------- 换几何的过渡 ----------------
    def _set_alpha(self, v):
        """窗口整体不透明度。切 WS_EX_LAYERED 实测只要 1ms，步进 0.4ms，很便宜。"""
        try:
            self.root.attributes("-alpha", min(max(float(v), 0.0), 1.0))
        except Exception:
            pass

    def _drop_layered(self):
        """过渡结束后把 `WS_EX_LAYERED` 摘掉，让窗口回到普通合成路径。

        Tk 设过 `-alpha` 之后不会主动摘掉分层样式，而 layered 窗口有可能拿不到
        DWM 的圆角（四角会变成直角）。所以只在过渡期间借用一下，用完就摘。
        """
        try:
            import ctypes
            u = ctypes.windll.user32
            u.GetWindowLongPtrW.restype = ctypes.c_void_p
            u.GetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int]
            u.SetWindowLongPtrW.restype = ctypes.c_void_p
            u.SetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                            ctypes.c_void_p]
            GWL_EXSTYLE, WS_EX_LAYERED = -20, 0x00080000
            h = ctypes.c_void_p(self._win_hwnd())
            ex = int(u.GetWindowLongPtrW(h, GWL_EXSTYLE) or 0)
            if not (ex & WS_EX_LAYERED):
                return
            u.SetWindowLongPtrW(h, GWL_EXSTYLE,
                                ctypes.c_void_p(ex & ~WS_EX_LAYERED))
        except Exception:
            pass

    def _morph_finish(self):
        """立刻结束进行中的过渡：窗口摆到最终位置、恢复不透明。

        任何会重新摆位的入口（下一次最大化、开始拖动、程序退出前）都该先调它，
        否则窗口可能被留在屏幕外或半透明状态。
        """
        if getattr(self, "_morph_phase", None) is None:
            return False
        if getattr(self, "_morph_after", None):
            try:
                self.root.after_cancel(self._morph_after)
            except Exception:
                pass
        r = getattr(self, "_morph_final", None)
        if r:
            self._set_win_rect(r)
        self._set_alpha(1.0)
        self._drop_layered()
        self._morph_phase = None
        self._morph_final = None
        self._morph_after = None
        return True

    def _morph_to(self, rect):
        """换到新几何，但**不露出没画完的画面**。

        直接 `SetWindowPos` 改尺寸的话，窗口框先变大、内容要 330ms 才排完，
        中间那段时间看到的是背景色和陆续长出来的控件。这里绕开它：

        1. 先淡出到透明（60ms）；
        2. 把窗口挪到屏幕外并改成目标尺寸 —— Tk 只在可见时才绘制，屏幕外这一步
           **只排版不重绘**（实测 5.6ms vs 屏幕内 330ms）；
        3. 挪回目标位置（只要重绘一次，~75ms），此时窗口还是透明的；
        4. 淡入（120ms）。

        全程 ~240ms，和系统最大化动画一个量级，而且看不到撕裂。
        """
        self._morph_finish()
        cur = self._win_rect()
        if not cur or not rect:
            self._set_win_rect(rect)
            return
        # 位移/尺寸变化很小就别折腾了（比如吸附后微调），直接摆
        if all(abs(cur[i] - rect[i]) < 4 for i in range(4)):
            self._set_win_rect(rect)
            return
        self._morph_final = tuple(rect)
        self._morph_phase = "out"
        self._morph_t0 = time.perf_counter()
        self._morph_step()

    def _morph_step(self):
        self._morph_after = None
        ph = getattr(self, "_morph_phase", None)
        if ph is None:
            return
        try:
            if ph == "out":
                p = (time.perf_counter() - self._morph_t0) / (MORPH_FADE_OUT / 1000.0)
                if p >= 1.0:
                    self._set_alpha(0.0)
                    r = self._morph_final
                    if r:
                        # 屏幕外按新尺寸排版（不重绘）
                        self._set_win_rect((MORPH_OFF[0], MORPH_OFF[1],
                                            r[2], r[3]))
                        self.root.update_idletasks()
                        # 挪回目标位置（只重绘一次）
                        self._set_win_rect(r)
                        self.root.update_idletasks()
                    self._morph_phase = "in"
                    self._morph_t0 = time.perf_counter()
                    self._morph_after = self.root.after(0, self._morph_step)
                    return
                self._set_alpha(1.0 - p * p)        # 前段慢后段快，收得更干净
                self._morph_after = self.root.after(MORPH_STEP, self._morph_step)
                return
            # ph == "in"
            p = (time.perf_counter() - self._morph_t0) / (MORPH_FADE_IN / 1000.0)
            if p >= 1.0:
                self._set_alpha(1.0)
                self._drop_layered()        # 用完就摘，别一直挂着分层样式
                self._morph_phase = None
                self._morph_final = None
                return
            self._set_alpha(p * (2 - p))            # ease-out：先快后稳
            self._morph_after = self.root.after(MORPH_STEP, self._morph_step)
        except Exception:
            # 窗口已经在关了：别让 after 回调再碰 Tk
            self._morph_phase = None
            self._morph_final = None

    # ---------------- 贴边吸附 ----------------
    def _snap_zone(self, x, y):
        """光标 (x, y) 落在工作区哪条边上。

        返回 `'max'`（顶边 = 最大化）/ `'left'` / `'right'`（左右 = 半屏），
        不在任何边上返回 `''`。判据用的是**光标**位置而不是窗口位置 —— 和系统
        的贴边行为一致：手把窗口甩到边上就吸附。

        用工作区而不是整块屏幕：任务栏占掉的那条边不该触发吸附，否则拖到底边
        会误判成"贴近屏幕边缘"。
        """
        wa = self._work_area()
        if not wa:
            return ""
        d = U(SNAP_EDGE)
        wx, wy, ww, wh = wa
        if y <= wy + d:
            return "max"
        if x <= wx + d:
            return "left"
        if x >= wx + ww - 1 - d:
            return "right"
        return ""

    def _snap_apply(self, zone, cx, cy):
        """把窗口摆到 `zone` 指定的吸附位。

        * `'max'`：走 `_win_toggle_max()`，还原矩形自动记成当前位置，
          之后点最大化按钮能退回拖动前的地方；
        * `'left'` / `'right'`：半个工作区，同样按 `_frame_insets()` 外扩，
          保证**客户区**正好占半屏（不然四边会露系统的深色边框）。

        传 `''` 进来什么都不做 —— 离开吸附区走的是 `_snap_release()`。
        """
        if zone == "max":
            if not getattr(self, "_maxed", False):
                self._win_toggle_max()
            return
        if getattr(self, "_maxed", False):
            # 只退出最大化状态，几何由下面这次 morph 一步到位（不摆两次）
            self._set_maxed(False)
        if zone not in ("left", "right"):
            return
        wa = self._work_area()
        if not wa:
            return
        x, y, w, h = wa
        half = w // 2
        nx = x if zone == "left" else x + w - half
        l, t, r, b = self._frame_insets()
        self._morph_to((nx - l, y - t, half + l + r, h + t + b))

    def _snap_release(self, cx, cy):
        """离开吸附区：还原成吸附前的大小，并把它挪到光标底下跟着走。

        返回新的拖动基准矩形；拿不到吸附前的记录就返回 None（那就维持现状）。
        """
        prev = getattr(self, "_snap_prev", None)
        if not prev:
            return None
        rect, offy, offx = prev
        if getattr(self, "_maxed", False):
            self._set_maxed(False)          # 退出最大化状态；位置由下面这次过渡一并摆好
        nx, ny = int(cx - offx), int(cy - offy)
        wa = self._work_area()
        if wa:
            # 别让窗口整块飞出工作区：至少留一截能抓回来的标题栏
            wx, wy, ww, wh = wa
            nx = max(min(nx, wx + ww - U(64)), wx - rect[2] + U(64))
            ny = max(min(ny, wy + wh - U(48)), wy)
        # 过渡期间窗口由 morph 摆位，这里只把"该在哪"记下来，等动画结束再交给拖动
        self._morph_to((nx, ny, rect[2], rect[3]))
        return (nx, ny, rect[2], rect[3])

    def _maxed_drag_out(self, e):
        """最大化状态下往里拖：先还原，再让还原后的窗口跟着光标走。

        横向按光标在最大化宽度里的**相对位置**摆放（和 Windows 一致：从左边
        拖出来窗口靠左，从右边拖出来靠右），纵向保持光标离窗口顶边的距离。

        这里是**拖动**，要跟手，所以不走过渡：直接按还原后的尺寸摆到光标下。
        """
        mr = self._win_rect()
        self._morph_finish()
        self._set_maxed(False)          # 只退状态，几何马上自己摆
        r = getattr(self, "_restore_rect", None) or self._win_rect()
        if not r:
            return
        if mr and mr[2]:
            fx = min(max((e.x_root - mr[0]) / float(mr[2]), 0.0), 1.0)
            dy = e.y_root - mr[1]
            nx = int(e.x_root - fx * r[2])
            ny = int(e.y_root - max(0, min(dy, r[3] - 1)))
            self._set_win_rect((nx, ny, r[2], r[3]))
            r = (nx, ny, r[2], r[3])
        self._mv = (e.x_root, e.y_root, r)
        self._snap_prev = (r, e.y_root - r[1], e.x_root - r[0])

    def _win_close(self):
        self.root.destroy()

    def _on_state(self, _e=None):
        """同步标题栏"最大化"按钮的图标。"""
        b = getattr(self, "btn_max", None)
        if b is not None:
            b.set_maximized(bool(getattr(self, "_maxed", False)))

    def _bind_title_drag(self, hd):
        """顶栏空白处按住拖动 = 移动窗口，双击 = 最大化/还原。

        这里**不再发系统的 `WM_NCLBUTTONDOWN(HTCAPTION)`**，改成自己算位移，
        和右下角缩放手柄（`ResizeGrip`）同一套做法。原来那条路有两个实测的坑：

        1. **同步消息把界面卡死。** 发 `WM_NCLBUTTONDOWN` 用的是 `SendMessageW`，
           它要等拖动结束才返回 —— 这期间 Tk 的 mainloop 完全停摆，表现就是
           "一拖窗口程序就没反应"；更糟的是它会在 Tk 处理事件的过程中再开一个
           模态消息循环，Tk 不支持重入，容易直接把进程带崩。
        2. **系统那条路自带最大化。** 双击标题栏、或把窗口拖到屏幕顶边，都会触发
           系统的 `SW_MAXIMIZE`；而这个无框窗口碰上它会卡死（原因见
           `_win_toggle_max` 的说明）。自己算位移就没有这些副作用。

        现在按下时记下窗口矩形和光标位置，移动时用 `SetWindowPos` 跟着摆，
        全程在自己的事件里走完，不进任何模态循环。

        系统那条路上的**贴边吸附**（拖到顶边最大化、拖到左右半屏）也因此没了，
        所以在这里自己补了一份：拖动时算光标离工作区各条边的距离，进到
        `SNAP_EDGE` 内就吸附，拖出来就还原（见 `_snap_zone` / `_snap_apply`）。
        半屏同样按 `_frame_insets()` 外扩，保证客户区正好占半屏。

        注意 Tk 的事件只沿「控件 → 控件类 → 所属 Toplevel → all」传播，**不经过
        中间的父控件**，所以光绑在 hd 上只有它自己的空白处生效；必须递归给顶栏里
        每个非交互子控件各绑一次（交互控件要跳过，否则点芯片/齿轮会变成拖窗口）。
        """
        def begin(e=None):
            # 还在过渡中就先收尾：否则窗口可能停在半透明或屏幕外的中间态
            self._morph_finish()
            self._mv = None
            self._pend = None
            self._snap = None
            if e is None:
                return "break"
            if getattr(self, "_maxed", False):
                # 最大化时**先不动**：光标位移超过 SNAP_MOVE 才还原并跟随，
                # 这样只是单击一下顶栏不会把窗口还原掉（和系统一致）。
                self._pend = (e.x_root, e.y_root)
                return "break"
            r = self._win_rect()
            if not r:
                return "break"
            try:
                self._mv = (e.x_root, e.y_root, r)
                # 吸附前的"自由尺寸"：进入吸附态后拖出来就还原成它
                self._snap_prev = (r, e.y_root - r[1], e.x_root - r[0])
            except Exception:
                self._mv = None
            return "break"

        def move(e=None):
            if e is None:
                return "break"
            # 最大化状态下的起手：位移够大才真正开始拖
            pend = getattr(self, "_pend", None)
            if pend is not None:
                px, py = pend
                if (abs(e.x_root - px) < U(SNAP_MOVE)
                        and abs(e.y_root - py) < U(SNAP_MOVE)):
                    return "break"
                self._pend = None
                if getattr(self, "_maxed", False):
                    self._maxed_drag_out(e)
                return "break"

            st = getattr(self, "_mv", None)
            if not st:
                return "break"
            # 拖动途中被双击最大化了：立刻放弃这次拖动，别再拿旧矩形去摆位
            if getattr(self, "_maxed", False) and getattr(self, "_snap", None) is None:
                self._mv = None
                return "break"

            x0, y0, r = st
            cx, cy = e.x_root, e.y_root
            z = self._snap_zone(cx, cy)
            cur = getattr(self, "_snap", None)
            # 过渡还在进行时，位置由 morph 自己摆，别拿拖动去抢
            morphing = getattr(self, "_morph_phase", None) is not None
            if (z or None) == (cur or None):
                if z:
                    return "break"        # 已经吸附在这个位置，别再反复摆位
                if morphing:
                    return "break"
                self._set_win_rect((r[0] + (cx - x0),
                                    r[1] + (cy - y0), r[2], r[3]))
                return "break"
            if z:
                if cur is None:
                    fr = (r[0] + (cx - x0), r[1] + (cy - y0), r[2], r[3])
                    self._snap_prev = (fr, cy - fr[1], cx - fr[0])
                self._snap_apply(z, cx, cy)
            elif cur is not None:
                # 从吸附态拖出来：还原成吸附前的大小，重新起算继续跟着光标
                nr = self._snap_release(cx, cy)
                if nr:
                    self._mv = (cx, cy, nr)
            elif not morphing:
                self._set_win_rect((r[0] + (cx - x0),
                                    r[1] + (cy - y0), r[2], r[3]))
            self._snap = z or None
            return "break"

        def end(_e=None):
            self._mv = None
            self._pend = None
            return "break"

        def dbl(_e=None):
            # 双击顶栏 = 最大化/还原。走自己的实现，不用系统的双击缩放，
            # 免得又回到 SW_MAXIMIZE 那条会卡死的路径。
            self._morph_finish()
            self._mv = None
            self._pend = None
            self._win_toggle_max()
            return "break"

        skip = (WinCtl, Chip, IconBtn, Btn, Select, Segmented, Switch)

        def walk(w):
            for ch in w.winfo_children():
                if isinstance(ch, skip):
                    continue
                ch.bind("<Button-1>", begin, add="+")
                ch.bind("<B1-Motion>", move, add="+")
                ch.bind("<ButtonRelease-1>", end, add="+")
                ch.bind("<Double-Button-1>", dbl, add="+")
                walk(ch)

        hd.bind("<Button-1>", begin, add="+")
        hd.bind("<B1-Motion>", move, add="+")
        hd.bind("<ButtonRelease-1>", end, add="+")
        hd.bind("<Double-Button-1>", dbl, add="+")
        walk(hd)

    def refresh_model_chip(self):
        have = usable_models()
        cur = self.settings.get("model") or default_model()
        ok = cur in have
        self.chip_model.set_text("%s · %s" % (cur, "已就绪" if ok else "未下载"))
        self.chip_model._dot = C["green"] if ok else C["magenta"]
        self.chip_model._draw()

    # ---------------- 侧栏 + 页面宿主 ----------------
    def _build_body(self):
        body = tk.Frame(self.root, bg=C["canvas"])
        body.pack(fill="both", expand=True)
        self.body = body

        self.sidebar = tk.Frame(body, bg=C["canvas"], width=232)
        self.sidebar.pack(side="left", fill="y", padx=(24, 0), pady=24)
        self.sidebar.pack_propagate(False)

        nav_holder = tk.Frame(self.sidebar, bg=C["canvas"])
        nav_holder.pack(fill="x")
        for key, ic, label in NAV_ITEMS:
            b = NavBtn(nav_holder, ic, label, command=lambda k=key: self.select(k))
            b.pack(fill="x", pady=(0, 4))
            self.navs[key] = b

        self.side_spacer = tk.Frame(self.sidebar, bg=C["canvas"])
        self.side_spacer.pack(fill="both", expand=True)

        self.storage = Panel(self.sidebar, radius=16, pad=14, outer=C["canvas"])
        self.storage.pack(fill="x")
        st = self.storage.body
        row = tk.Frame(st, bg=C["surface"])
        row.pack(fill="x")
        tk.Label(row, text="本地模型", bg=C["surface"], fg=C["ink2"],
                 font=F(8)).pack(side="left")
        self.lbl_storage = tk.Label(row, text="", bg=C["surface"], fg=C["ink3"],
                                    font=FN(8))
        self.lbl_storage.pack(side="right")
        self.bar_storage = tk.Canvas(st, height=5, bg=C["surface"],
                                     highlightthickness=0, bd=0)
        self.bar_storage.pack(fill="x", pady=(10, 0))
        self.bar_storage.bind("<Configure>", self._draw_storage_bar)
        tk.Label(st, text="离线运行 · 不上传任何音频", bg=C["surface"],
                 fg=C["ink4"], font=F(7)).pack(anchor="w", pady=(10, 0))

        self.expand_btn = tk.Frame(self.sidebar, bg=C["canvas"])

        self.host = tk.Frame(body, bg=C["canvas"])
        self.host.pack(side="left", fill="both", expand=True, padx=24, pady=24)
        self.refresh_storage()

    def _draw_storage_bar(self, _e=None):
        cv = self.bar_storage
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 2:
            return
        cv.delete("all")
        rrect(cv, 0, 0, w, h, 3, C["surface_in"])
        have = usable_models()
        n = len(have)
        ratio = min(1.0, n / 6.0) if n else 0.0
        fw = int((w - 1) * ratio)
        if fw >= 4:
            vgrad(cv, 0, 0, fw, h, "#5966F3", "#EC48BD")
            _knock_corners(cv, 0, 0, fw, h, 3, C["surface"])

    def refresh_storage(self):
        have = usable_models()
        total = 0
        for n in have:
            total += dir_size(os.path.join(MODELS_DIR, n))
        self.lbl_storage.configure(text=human(total) if total else "0 B")
        self._draw_storage_bar()
        self.refresh_model_chip()

    def set_collapsed(self, on):
        on = bool(on)
        if on == self.collapsed:
            return
        self.collapsed = on
        # 这两处以前是裸数字：轨道宽度/内缩没跟着 DPI 放大，而轨道里的图标、
        # 圆角都是 2x —— 结果就是一条 1x 宽的窄轨里塞着 2x 的内容，看着"不平整"。
        self.sidebar.configure(width=U(64) if on else U(232))
        self.sidebar.pack_configure(padx=(U(24) if not on else U(18), 0))
        for b in self.navs.values():
            b.set_collapsed(on)
        if on:
            self.storage.pack_forget()
            self.expand_btn.pack(side="bottom", fill="x")
        else:
            self.expand_btn.pack_forget()
            self.storage.pack(fill="x")

    def _on_resize(self, e):
        if e.widget is not self.root:
            return
        w = e.width
        self.set_collapsed(w < U(COLLAPSE_AT))
        # 最大化/还原都会带一次 Configure，顺手同步标题栏"最大化"按钮的图标
        if getattr(self, "btn_max", None) is not None:
            self._on_state()
        # 窗口真的移动/缩放了才收掉下拉：弹层那套"打开时算死的屏幕坐标"只在
        # 窗口几何变化后才失效。`<Configure>` 还会因为别的原因触发（最大化过程中的
        # 一连串中间态、DWM 重绘等），那些时候弹层坐标仍然有效，不该被误关 ——
        # 否则下拉会"刚打开就被自己关掉"。
        #
        # 这里**不能**再调 winfo_rootx/rooty/width/height：那是 4 次到窗口管理器的
        # 同步往返，而最大化过程中 Configure 会连着来几十次，每次都同步查询就是
        # 几十次阻塞 —— 表现为最大化/还原"一顿一顿"。e 里已经带了 width/height，
        # 只有 rootx/rooty 拿不到，所以把整套几何查询挪到空闲时再做（见 _sync_geo）。
        if getattr(self, "_geo_pending", False):
            return
        self._geo_pending = True
        try:
            self.root.after_idle(self._sync_geo)
        except Exception:
            self._geo_pending = False

    def _sync_geo(self):
        """空闲时读一次真实几何，只为判断"下拉弹层是否需要收掉"。

        挂在 after_idle 上：Configure 风暴期间只会排队一个回调，几十次事件
        被合并成一次同步查询；等真正要收下拉时坐标也是最新的。
        """
        self._geo_pending = False
        try:
            geo = (self.root.winfo_rootx(), self.root.winfo_rooty(),
                   self.root.winfo_width(), self.root.winfo_height())
        except Exception:
            return
        if geo != getattr(self, "_last_geo", None):
            self._last_geo = geo
            if Select._any_open():
                Select.close_all_open()

    # ---------------- 页面 ----------------
    def _build_pages(self):
        self.pages["generate"] = PageGenerate(self.host, self)
        self.pages["models"] = PageModels(self.host, self)
        self.pages["download"] = PageDownload(self.host, self)
        self.pages["convert"] = PageConvert(self.host, self)
        self.pages["history"] = PageHistory(self.host, self)
        self.pages["settings"] = PageSettings(self.host, self)
        IconBtn(self.expand_btn, "chevrons_right", command=lambda: self.set_collapsed(False),
                size=44, icon_size=18, fill=C["canvas"], outer=C["canvas"],
                radius=R_BTN).pack()

    def dispatch_drop(self, data):
        """拖放进来的文件：优先给**当前这一页**（格式转换页也能接），
        当前页不收才回落到生成页。

        之前写死给生成页，于是人在格式转换页拖文件、文件却跑到生成页去了。
        """
        pg = self.pages.get(getattr(self, "current", ""))
        if pg is not None and hasattr(pg, "drop_files"):
            pg.drop_files(data)
            return
        gp = self.pages.get("generate")
        if gp is not None:
            gp.drop_files(data)

    def select(self, key):
        if key not in self.pages:
            return
        for k, p in self.pages.items():
            if k == key:
                p.pack(fill="both", expand=True)
                p.on_show()
            else:
                p.pack_forget()
        for k, b in self.navs.items():
            b.set_active(k == key)
        self.current = key
        if key == "history":
            self.pages["history"].reload()
        if key == "models":
            self.pages["models"].reload()

    def show_help(self):
        _alert("歌词生成器 · 使用提示",
               "1. 选一个音乐或视频文件（也可以直接把文件拖到窗口里）\n"
               "2. 有歌词文本就上传它 —— 走【对齐模式】，文字 100% 是你的原文，"
               "识别只负责算时间点，这是歌曲的正确用法\n"
               "3. 视频里本来就有字幕的，字幕来源选【自动】即可，秒级完成且文字全对\n"
               "4. 没有歌词、也没字幕，才用【自动听写】（口播/访谈适用，歌曲会有谐音错字）\n\n"
               "参数说明见 start.bat --help，常见问题见 README.md")

    # ---------------- 后台任务 ----------------
    def log(self, msg):
        self.q.put(("log", str(msg)))

    def run_job(self, kwargs, media, out):
        def worker():
            try:
                res = transcribe(log=self.log, **kwargs)
                self.q.put(("done", res))
            except Exception:
                self.q.put(("error", traceback.format_exc()))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def download_model(self, name):
        """统一的模型下载入口：后台线程跑 download()，进度/完成都走 self.q。

        生成页和模型管理页都复用这里，保证「同一时刻只有一个下载在跑」，
        且下载完成后两个页面都能刷新状态。
        """
        if self._download_busy:
            _alert("正在下载", "已经有一个下载在跑了（%s），等它结束再说。"
                   % self._download_busy)
            return
        self._download_busy = name
        # 失败原因：把这一轮下载的日志留一份底，失败时连同异常一起回传给界面。
        # 原来这里 `except Exception: rc = 1` 把异常对象整个丢掉，界面只能显示
        # "下载失败，可重试" —— 用户既不知道是磁盘满了、目录只读还是网络断了，
        # 也看不到 download() 打出来的那些细节（在无控制台的 exe 里那些 print
        # 只会进日志文件，界面上一个字都没有）。
        lines = []

        def worker():
            def lg(msg):
                s = str(msg)
                lines.append(s)
                self.q.put(("modellog", (name, s)))

            try:
                rc = download(name, force=False, source="auto", log=lg)
                reason = "" if rc == 0 else "\n".join(lines[-12:])
            except Exception as e:
                rc = 1
                reason = "%s: %s" % (type(e).__name__, e)
                lg("[异常] " + reason)
            self.q.put(("modeldone", (name, rc, reason)))

        threading.Thread(target=worker, daemon=True).start()

    def tick(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                page = self.pages.get("generate")
                if kind == "log":
                    if page:
                        page.push_log(payload)
                        page.on_log_tick()
                elif kind == "done":
                    cancelled = self.state.pop("cancelled", False)
                    self.state["running"] = False
                    if cancelled:
                        # 用户已经取消：别把界面从"正在中止"强行切到结果页，
                        # 也别把这次结果当成有效产出记下来。
                        self.q.put(("log", "[已取消] 任务已中止，结果未保存。"))
                    else:
                        self.state["last"] = payload
                        if page:
                            page.on_done(payload)
                    self.refresh_storage()
                elif kind == "error":
                    cancelled = self.state.pop("cancelled", False)
                    self.state["running"] = False
                    if cancelled:
                        self.q.put(("log", "[已取消] 任务已中止。"))
                    elif page:
                        page.on_error(payload)
                elif kind in ("modellog", "modeldone"):
                    mp = self.pages.get("models")
                    if mp:
                        mp.on_event(kind, payload)
                    gp = self.pages.get("generate")
                    if gp:
                        gp.on_event(kind, payload)
                    if kind == "modeldone":
                        self._download_busy = None
                        self.refresh_storage()
                        if gp and hasattr(gp, "refresh_models"):
                            gp.refresh_models()
                elif kind in ("dl_log", "dl_parsed", "dl_done", "cookie_updated"):
                    # 视频下载页的后台事件（yt-dlp 跑在 worker 线程里）
                    dp = self.pages.get("download")
                    if dp:
                        if kind == "dl_log":
                            dp.log(payload)
                        elif kind == "dl_parsed":
                            dp.on_parsed(*payload)
                        elif kind == "cookie_updated":
                            dp.on_cookie_updated(payload)
                        else:
                            dp.on_done(*payload)
                elif kind == "cookie_autoimported":
                    # 后台监视器发现了插件导出的 Cookie 文件并复制进来了
                    dp = self.pages.get("download")
                    if dp:
                        dp.on_cookie_autoimported(payload)
                elif kind == "dupscan_done":
                    # 设置页「占用体检」的后台结果
                    sp = self.pages.get("settings")
                    if sp:
                        sp.on_dupscan_done(payload)
                elif kind in ("conv_log", "conv_progress", "conv_file", "conv_done"):
                    # 格式转换页的后台事件（ffmpeg 跑在 worker 线程里）
                    cp = self.pages.get("convert")
                    if cp:
                        if kind == "conv_log":
                            cp.on_conv_log(payload)
                        elif kind == "conv_progress":
                            cp.on_progress()
                        elif kind == "conv_file":
                            cp.on_file_done(*payload)
                        else:
                            cp.on_all_done(payload)
        except queue.Empty:
            pass
        # 下载进度：worker 线程只往 job 上写数字，这里（主线程）按帧刷新界面
        dp = self.pages.get("download")
        if dp is not None:
            try:
                dp.on_progress()
            except Exception:
                pass
        # 转换进度：同理
        cp = self.pages.get("convert")
        if cp is not None:
            try:
                cp.on_progress()
            except Exception:
                pass
        try:
            self.root.after(120, self.tick)
        except Exception:
            pass


# =====================================================================
# 通用小件
# =====================================================================

def dashed_rrect(cv, x1, y1, x2, y2, r, color, width=1.5, on=8, off=6):
    """手绘虚线圆角矩形。

    专门不用 canvas 的 dash 属性：虚线要沿"直边 + 圆角"连续走一圈，
    分别用直线 dash 和弧 dash 拼出来接缝处对不齐，一眼能看出断裂。
    这里把整条路径参数化，按弧长统一打点，虚线是连续的。
    """
    import math
    # r / width / on / off 都是像素字面量，统一按 DPI 放大（坐标不动）
    r, width = U(r), U(width)
    on, off = U(on), U(off)
    path = []

    def arc(cx, cy, a0, a1, n=14):
        for i in range(n + 1):
            t = math.radians(a0 + (a1 - a0) * i / float(n))
            path.append((cx + r * math.cos(t), cy + r * math.sin(t)))

    path.append((x1 + r, y1))
    path.append((x2 - r, y1))
    arc(x2 - r, y1 + r, 270, 360)
    path.append((x2, y2 - r))
    arc(x2 - r, y2 - r, 0, 90)
    path.append((x1 + r, y2))
    arc(x1 + r, y2 - r, 90, 180)
    path.append((x1, y1 + r))
    arc(x1 + r, y1 + r, 180, 270)
    path.append((x1 + r, y1))

    drawing, remain = True, float(on)
    cur = path[0]
    for nxt in path[1:]:
        seg = math.hypot(nxt[0] - cur[0], nxt[1] - cur[1])
        if seg <= 0.001:
            cur = nxt
            continue
        t = 0.0
        while t < seg:
            step = min(remain, seg - t)
            if drawing and step > 0.2:
                ax = cur[0] + (nxt[0] - cur[0]) * (t / seg)
                ay = cur[1] + (nxt[1] - cur[1]) * (t / seg)
                bx = cur[0] + (nxt[0] - cur[0]) * ((t + step) / seg)
                by = cur[1] + (nxt[1] - cur[1]) * ((t + step) / seg)
                cv.create_line(ax, ay, bx, by, fill=color, width=width,
                               capstyle="round")
            t += step
            remain -= step
            if remain <= 0.01:
                drawing = not drawing
                remain = float(on) if drawing else float(off)
        cur = nxt


class DropZone(tk.Canvas):
    """拖拽/点选区域。框是虚线的 —— 这是"可以把文件放进来"的唯一视觉暗号。"""

    def __init__(self, master, on_click, on_hover=None, outer=None, height=150,
                 title="拖入音乐或视频，或", link="选择文件", caption=""):
        outer = outer or C["surface"]
        tk.Canvas.__init__(self, master, bg=outer, highlightthickness=0, bd=0,
                           height=height)
        self._outer, self._title, self._link = outer, title, link
        self._caption = caption
        self._hover = False
        self._cmd = on_click
        self.bind("<Configure>", lambda e: self._draw())
        self.bind("<Enter>", lambda e: (setattr(self, "_hover", True), self._draw()))
        self.bind("<Leave>", lambda e: (setattr(self, "_hover", False), self._draw()))
        self.bind("<Button-1>", lambda e: on_click())
        self.configure(cursor="hand2")

    def _draw(self):
        w, h = self.winfo_width(), self.winfo_height()
        if w < 4 or h < 4:
            return
        self.delete("all")
        if self._hover:
            rrect(self, 1, 1, w - 1, h - 1, 16, "#232A66")
        dashed_rrect(self, 1.2, 1.2, w - 1.2, h - 1.2, 16,
                     "#4A52B8" if self._hover else "#3A4199", 1.5)
        cy = h / 2.0
        top = cy - U(34)
        rrect(self, w / 2.0 - U(21), top, w / 2.0 + U(21), top + U(42), 13, C["bg_tint"])
        icon(self, "upload", w / 2.0, top + U(21), 22, C["primary"])
        try:
            from tkinter import font as tkfont
            f1 = tkfont.Font(font=F(10))
            tw = f1.measure(self._title) + U(4) + f1.measure(self._link)
        except Exception:
            tw = 200
        x = w / 2.0 - tw / 2.0
        self.create_text(x, top + U(62), text=self._title, anchor="w",
                         fill=C["ink2"], font=F(10))
        self.create_text(x + (tw - f1.measure(self._link)), top + U(62), text=self._link,
                         anchor="w", fill=C["link"], font=F(10, True))
        if self._caption:
            self.create_text(w / 2.0, top + U(84), text=self._caption, anchor="center",
                             fill=C["ink4"], font=F(8))

    def set_caption(self, t):
        self._caption = t
        self._draw()

    def set_title(self, t, link=""):
        self._title, self._link = t, link
        self._draw()


# =====================================================================
# 页面基类
# =====================================================================

class Page(tk.Frame):
    """统一的"标题行 + 内容区 + 底部动作条"三段式。"""

    def __init__(self, master, app, title, desc="", with_action=True):
        tk.Frame.__init__(self, master, bg=C["canvas"])
        self.app = app
        self.head = tk.Frame(self, bg=C["canvas"])
        self.head.pack(fill="x")
        hl = tk.Frame(self.head, bg=C["canvas"])
        hl.pack(side="left")
        tk.Label(hl, text=title, bg=C["canvas"], fg=C["ink"],
                 font=F(13, True)).pack(anchor="w")
        self.lbl_desc = tk.Label(hl, text=desc, bg=C["canvas"], fg=C["ink3"], font=F(8))
        self.lbl_desc.pack(anchor="w", pady=(5, 0))
        self.head_right = tk.Frame(self.head, bg=C["canvas"])
        self.head_right.pack(side="right")
        self.action = self.action_left = self.action_right = None
        if with_action:
            # 动作条必须先占位（side="bottom"），再放内容。
            # tkinter 的 pack 是按调用顺序分配空间的：先给每个子件它要的尺寸，
            # 空间不够时**排在后面的会被压成 0**。内容里一个 Text 的"自然高度"
            # 动辄几百像素，动作条要是排在它后面，就会整个消失。
            self.action = tk.Frame(self, bg=C["canvas"], height=64)
            self.action.pack(side="bottom", fill="x")
            self.action.pack_propagate(False)
            self.action_left = tk.Frame(self.action, bg=C["canvas"])
            self.action_left.pack(side="left")
            self.action_right = tk.Frame(self.action, bg=C["canvas"])
            self.action_right.pack(side="right")
        self.content = tk.Frame(self, bg=C["canvas"])
        self.content.pack(fill="both", expand=True, pady=(16, 0))

    def on_show(self):
        pass

    def set_desc(self, t):
        self.lbl_desc.configure(text=t)

    def action_hint(self, text, color=None):
        if not self.action_left:
            return
        for c in self.action_left.winfo_children():
            c.destroy()
        tk.Label(self.action_left, text=text, bg=C["canvas"],
                 fg=color or C["ink4"], font=F(8)).pack(side="left")


# =====================================================================
# 页面一：生成歌词
# =====================================================================

MODE_ALIGN, MODE_DICTATE = "align", "dictate"

SUBS_LABEL = {"auto": "自动（先找现成字幕）", "track": "只用现成字幕",
              "ocr": "只识别画面硬字幕", "off": "只识别音频（不用字幕）"}
REGION_LABEL = {"auto": "自动探测", "bottom": "画面底部", "top": "画面顶部"}
# 语言切换项。Whisper 是「多语种模型」——所有语言识别能力都打包在 model.bin 里，
# 不存在单独的「语言下载 / 语言文件夹」。下面这些语种只要用的是多语种模型就能识别，
# 无需任何额外下载。生成页与设置页共用这一份，避免两处写死不一致。
LANG_LABEL = {
    "auto": "自动检测",
    "zh":   "中文（zh）",
    # 粤语：Whisper 原生支持 yue。选中后会自动改用粤语提示词并保留繁体
    # （见 LANG_POLICY），否则默认的"输出简体中文"会把粤语歌带偏。
    "yue":  "粤语（yue）",
    "en":   "English（en）",
    "ja":   "日本語（ja）",
    "ko":   "한국어（ko）",
    "fr":   "Français（fr）",
    "es":   "Español（es）",
    "de":   "Deutsch（de）",
    "ru":   "Русский（ru）",
    "pt":   "Português（pt）",
    "it":   "Italiano（it）",
    "th":   "ไทย（th）",
    "vi":   "Tiếng Việt（vi）",
}


class PageGenerate(Page):
    def __init__(self, master, app):
        Page.__init__(self, master, app, "生成歌词",
                      "把音乐或视频交给它，产出 lrc / srt / ass / vtt")
        # 记下原始描述：音频开关会临时改写它，关掉时要能还原
        self._desc0 = self.lbl_desc.cget("text")
        self._subs_backup = None
        self._log_lines = 0
        self._pending_download = None
        self.views = {}
        self.mode = MODE_ALIGN
        self.var_media = tk.StringVar()
        self.var_lyrics = tk.StringVar()
        self.var_out = tk.StringVar()
        self.var_prompt = tk.StringVar(value=app.settings.get("prompt", DEFAULT_PROMPT))
        self.var_start = tk.StringVar()
        self.var_dur = tk.StringVar()
        self.var_offset = tk.StringVar(value="0")
        self.var_fps = tk.StringVar(value=str(app.settings.get("subs_fps", "2.0")))
        self.var_score = tk.StringVar(value=str(app.settings.get("min_score", "0.5")))
        self.var_ffmpeg = tk.StringVar(value=app.settings.get("ffmpeg", ""))
        self.var_threads = tk.StringVar(value=str(app.settings.get("threads", 4)))
        self.var_fmt = tk.StringVar(value=app.settings.get("fmt", "lrc"))
        self.var_model = tk.StringVar(value=app.settings.get("model") or default_model())
        self.var_lang = tk.StringVar(value=app.settings.get("language", "zh"))
        self.var_script = tk.StringVar(value=app.settings.get("script", "auto") or "auto")
        self.var_subs = tk.StringVar(value="auto")
        self.var_region = tk.StringVar(value="auto")

        self.holder = tk.Frame(self.content, bg=C["canvas"])
        self.holder.pack(fill="both", expand=True)
        self._build_input()
        self._build_running()
        self._build_done()
        self._build_logview()
        self.show("input")

    # ---------------- 状态机 ----------------
    def show(self, key):
        for k, v in self.views.items():
            if k == key:
                v.pack(fill="both", expand=True)
            else:
                v.pack_forget()
        self.current = key
        self._sync_actions()

    def _sync_actions(self):
        """动作条整条重建 —— 每个状态该有哪些按钮完全不同，与其原地改不如重建。

        注意：重建会把上一批按钮对象**销毁**，所以 self.btn_run 这类引用不能跨状态用。
        真需要改状态，得在重建时就把目标状态建出来（比如"运行中"直接建一个禁用按钮）。
        """
        for c in self.action_right.winfo_children():
            c.destroy()
        self.btn_run = None
        if self.current == "input":
            Btn(self.action_right, "打开输出目录", icon_name="folder",
                command=self.open_dir, outer=C["canvas"]).pack(side="right", padx=(10, 0))
            model = self.var_model.get()
            if is_downloaded(model):
                self.btn_run = Btn(self.action_right, "开始生成歌词", icon_name="play",
                                   kind="green", bold=True, size=10,
                                   command=self.start, outer=C["canvas"], pad_x=22)
                self.btn_run.pack(side="right")
                self.action_hint("预计耗时 ≈ 音频长度 ÷ 模型倍速（large-v3 约 1.4 倍实时）")
            else:
                self.btn_run = Btn(self.action_right, "下载并继续", icon_name="download",
                                   kind="blue", bold=True, size=10,
                                   command=self._download_and_continue,
                                   outer=C["canvas"], pad_x=22)
                self.btn_run.pack(side="right")
                self.action_hint("模型「%s」尚未下载 · 点此下载后自动开始" % model,
                                C["magenta"])
        elif self.current == "running":
            Btn(self.action_right, "取消任务", icon_name="x", command=self.cancel,
                outer=C["canvas"]).pack(side="right")
            busy = Btn(self.action_right, "开始生成歌词", icon_name="play", kind="green",
                       bold=True, size=10, outer=C["canvas"], pad_x=22)
            busy.pack(side="right", padx=(10, 0))
            busy.set_enabled(False)
            self.action_hint("任务进行中 · 关闭窗口会中断处理")
        elif self.current == "error":
            Btn(self.action_right, "打开输出目录", icon_name="folder",
                command=self.open_dir, outer=C["canvas"]).pack(side="right", padx=(10, 0))
            Btn(self.action_right, "返回重试", icon_name="refresh", kind="green",
                bold=True, size=10, command=lambda: self.show("input"),
                outer=C["canvas"], pad_x=22).pack(side="right")
            self.action_hint("处理失败 · 上面的日志里有完整堆栈", C["err"])
        elif self.current == "done":
            Btn(self.action_right, "重新生成", icon_name="refresh", command=self.back_to_input,
                outer=C["canvas"]).pack(side="right", padx=(10, 0))
            Btn(self.action_right, "查看日志", command=lambda: self.show("logview"),
                outer=C["canvas"]).pack(side="right", padx=(10, 0))
            Btn(self.action_right, "打开输出目录", icon_name="folder", kind="green",
                bold=True, size=10, command=self.open_dir,
                outer=C["canvas"], pad_x=22).pack(side="right")
            self.action_hint("文件已写入本地 · 未上传任何数据")
        else:
            Btn(self.action_right, "返回结果", command=lambda: self.show("done"),
                outer=C["canvas"]).pack(side="right", padx=(10, 0))
            Btn(self.action_right, "打开输出目录", icon_name="folder",
                command=self.open_dir, outer=C["canvas"]).pack(side="right")
            self.action_hint("日志同时写入 exe 旁边的 lyric-maker.log")

    # ---------------- 模型状态 / 下载联动 ----------------
    def _on_model(self, v):
        """下拉框选了某个模型：同步顶栏芯片，并按是否已下载刷新动作条。"""
        self.var_model.set(v)
        self.app.refresh_model_chip()
        if self.current == "input":
            self._sync_actions()

    def request_download(self, name):
        """从下拉框的「下载」链接或动作条「下载并继续」触发。

        把待办记下来，交给 App 统一下载；下载完成后 on_event 会自动刷新状态，
        若下的是当前选中的模型且仍在输入态，就自动开始生成。
        """
        if self.app._download_busy:
            _alert("正在下载", "已经有一个下载在跑了（%s），等它结束再说。"
                   % self.app._download_busy)
            return
        self._pending_download = name
        self.var_model.set(name)
        self.app.refresh_model_chip()
        self.action_hint("正在下载 %s …" % name, C["magenta"])
        if self.current == "input":
            self._sync_actions()
        self.app.download_model(name)

    def _download_and_continue(self):
        self.request_download(self.var_model.get())

    def on_event(self, kind, payload):
        """接收 App 转发的 modellog / modeldone 事件（与模型管理页共享同一下载流）。"""
        if kind == "modellog":
            name, msg = payload
            if self._pending_download and name == self._pending_download:
                self.action_hint("正在下载 %s：%s"
                                % (name, (msg or "").strip()[:64]), C["magenta"])
        elif kind == "modeldone":
            name, rc, reason = payload
            self.refresh_models()
            self.app.refresh_model_chip()
            if self._pending_download and name == self._pending_download:
                self._pending_download = None
                if rc == 0:
                    self.action_hint("%s 下载完成，准备开始…" % name, C["green"])
                    if self.current == "input" and self.var_model.get() == name:
                        self._sync_actions()
                        self.start()
                else:
                    self.action_hint("%s 下载失败：%s" % (name, _one_line(reason)),
                                     C["err"])
                    if self.current == "input":
                        self._sync_actions()

    def refresh_models(self):
        """下载完成等时机调用：重画模型下拉框状态点，并同步动作条。"""
        sel = self.f_model.get("ctl") if isinstance(self.f_model, dict) else None
        if sel is not None and hasattr(sel, "refresh"):
            sel.refresh()
        if self.current == "input":
            self._sync_actions()

    # ---------------- 视图一：输入 ----------------
    def _build_input(self):
        sc = Scroll(self.holder, bg=C["canvas"])
        self.views["input"] = sc
        pad = sc.inner
        st = self.app.settings

        # 输入素材
        c1 = Panel(pad, pad=22)
        c1.pack(fill="x", pady=(0, 16))
        row = tk.Frame(c1.body, bg=C["surface"])
        row.pack(fill="x")
        self.drop = DropZone(row, on_click=self.pick_media, outer=C["surface"],
                             height=150,
                             caption="mp3 / wav / flac / m4a / mp4 / mkv / webm · 不上传，全部本地处理")
        self.drop.pack(side="left", fill="both", expand=True)
        side = tk.Frame(row, bg=C["surface"], width=292)
        side.pack(side="left", fill="y", padx=(22, 0))
        side.pack_propagate(False)
        tk.Label(side, text="可选 · 对齐模式", bg=C["surface"], fg=C["magenta"],
                 font=F(8, True)).pack(anchor="w", pady=(24, 9))
        tk.Label(side, text="已有歌词文本？上传 .txt / .lrc 走对齐模式 —— "
                            "文字 100% 是你的原文，识别只负责算时间点。",
                 bg=C["surface"], fg=C["ink2"], font=F(8), wraplength=270,
                 justify="left").pack(anchor="w")
        Btn(side, "选择歌词文本", icon_name="file", command=self.pick_lyrics,
            outer=C["surface"], height=34).pack(anchor="w", pady=(12, 24), fill="x")

        # 识别参数
        c2 = Panel(pad, pad=22)
        c2.pack(fill="x", pady=(0, 16))
        head = tk.Frame(c2.body, bg=C["surface"])
        head.pack(fill="x")
        tk.Label(head, text="识别参数", bg=C["surface"], fg=C["ink"],
                 font=F(10, True)).pack(side="left")
        self.seg_mode = Segmented(head, [(MODE_ALIGN, "对齐模式"),
                                         (MODE_DICTATE, "自动听写")],
                                  value=MODE_ALIGN, command=self._on_mode,
                                  outer=C["surface"])
        self.seg_mode.pack(side="right")
        # 「只识别音频」：一键走"只听声音生成歌词"这条路，不去碰视频字幕 / 画面 OCR。
        # 等价于把「字幕来源」设成 off，但摆在标题行才是用户找得到的地方 ——
        # 之前这个能力藏在"字幕来源"下拉的最后一项，没人会往那儿找。
        self.sw_audio = Switch(head, "只识别音频", value=bool(st.get("audio_only")),
                               command=self._on_audio_only, outer=C["surface"])
        self.sw_audio.pack(side="right", padx=(0, U(18)))

        self.grid = tk.Frame(c2.body, bg=C["surface"])
        self.grid.pack(fill="x", pady=(16, 0))
        for i in range(3):
            self.grid.columnconfigure(i, weight=1, uniform="fld")

        self.f_model = self._field(self.grid, "识别模型", lambda p: Select(
            p, list(MODEL_REPOS.keys()), textvariable=self.var_model,
            outer=C["surface"],
            statuser=lambda v: "已下载" if is_downloaded(v) else "未下载",
            downloader=lambda v: self.request_download(v),
            command=self._on_model))
        self.f_lang = self._field(self.grid, "语言", lambda p: Select(
            p, list(LANG_LABEL.keys()), textvariable=self.var_lang,
            outer=C["surface"], labeler=lambda v: LANG_LABEL.get(v, v)))
        self.f_threads = self._field(self.grid, "CPU 线程", lambda p: Input(
            p, textvariable=self.var_threads, outer=C["surface"]))
        self.f_script = self._field(self.grid, "输出字形", lambda p: Select(
            p, ["auto"] + list(SCRIPT_MODES), textvariable=self.var_script,
            outer=C["surface"],
            labeler=lambda v: ("自动（按语言）" if v == "auto"
                               else SCRIPT_LABEL.get(v, v))))
        self.f_subs = self._field(self.grid, "字幕来源", lambda p: Select(
            p, list(SUBS_LABEL.keys()), textvariable=self.var_subs,
            outer=C["surface"], command=self._on_subs,
            labeler=lambda v: SUBS_LABEL.get(v, v)))
        self.f_region = self._field(self.grid, "字幕区域", lambda p: Select(
            p, list(REGION_LABEL.keys()), textvariable=self.var_region,
            outer=C["surface"], labeler=lambda v: REGION_LABEL.get(v, v)))
        self.f_offset = self._field(self.grid, "时间轴偏移", lambda p: Input(
            p, textvariable=self.var_offset, outer=C["surface"],
            placeholder="0 ms"))
        fields = (self.f_model, self.f_lang, self.f_script,
                  self.f_threads, self.f_subs, self.f_region)
        for idx, f in enumerate(fields):
            r, c = divmod(idx, 3)
            f["frame"].grid(row=r, column=c, sticky="ew",
                            padx=(0, 14) if c < 2 else 0, pady=(0, 14))
        self._grid_cols = 3
        self._on_audio_only()      # 按保存的设置还原开关，并同步被禁用的字段

        # 歌词文本行
        lr = tk.Frame(c2.body, bg=C["surface"])
        lr.pack(fill="x")
        field_label(lr, "歌词文本（对齐模式下必填）").pack(anchor="w", pady=(0, 7))
        lrr = tk.Frame(lr, bg=C["surface"])
        lrr.pack(fill="x")
        self.inp_lyrics = Input(lrr, textvariable=self.var_lyrics, outer=C["surface"],
                                placeholder="选择 .txt / .lrc 文件…")
        self.inp_lyrics.pack(side="left", fill="x", expand=True)
        Btn(lrr, "浏览…", command=self.pick_lyrics, outer=C["surface"]).pack(
            side="left", padx=(10, 0))
        self.lbl_mode = tk.Label(c2.body, text="", bg=C["surface"], fg=C["ink4"],
                                 font=F(7), anchor="w", justify="left", wraplength=1000)
        self.lbl_mode.pack(anchor="w", pady=(10, 0))
        self.var_lyrics.trace_add("write", lambda *a: self._on_mode(self.mode))

        # 高级参数
        c3 = Panel(pad, pad=0)
        c3.pack(fill="x", pady=(0, 16))
        self.adv_head = tk.Canvas(c3.body, bg=C["surface"], height=48,
                                  highlightthickness=0, bd=0)
        self.adv_head.pack(fill="x")
        self.adv_head.bind("<Configure>", lambda e: self._draw_adv_head())
        self.adv_head.bind("<Button-1>", lambda e: self.toggle_adv())
        self.adv_head.configure(cursor="hand2")
        self.adv_open = False
        # 高级参数的正文裹一层 RoundedBody：原来直接是一个 surface 色的 Frame，
        # 展开时底边和卡片边缘齐平，把卡片的圆角切成直角。这里把正文自己画成
        # 圆角矩形，展开后四周都是圆的。
        self.adv_body = RoundedBody(c3.body, fill=C["surface"], radius=16)
        self._build_advanced(self.adv_body.box)

        # 输出
        c4 = Panel(pad, pad=22)
        c4.pack(fill="x", pady=(0, 4))
        oh = tk.Frame(c4.body, bg=C["surface"])
        oh.pack(fill="x")
        tk.Label(oh, text="输出", bg=C["surface"], fg=C["ink"],
                 font=F(10, True)).pack(side="left")
        tk.Label(oh, text="默认写入源文件同目录 · 支持 lrc / srt / ass / ssa / vtt",
                 bg=C["surface"], fg=C["ink4"], font=F(8)).pack(side="right")
        orow = tk.Frame(c4.body, bg=C["surface"])
        orow.pack(fill="x", pady=(14, 0))
        self.sel_fmt = Select(orow, ["lrc", "srt", "ass", "ssa", "vtt"],
                              textvariable=self.var_fmt, outer=C["surface"],
                              command=self._on_fmt, width=104)
        self.sel_fmt.pack(side="left")
        self.inp_out = Input(orow, textvariable=self.var_out, outer=C["surface"])
        self.inp_out.pack(side="left", fill="x", expand=True, padx=(10, 10))
        Btn(orow, "另存为…", command=self.pick_out, outer=C["surface"]).pack(side="left")
        self.var_fmt.trace_add("write", lambda *a: self._on_fmt(self.var_fmt.get()))
        self._on_mode(MODE_ALIGN)
        self._on_subs("auto")

    def _field(self, parent, text, make):
        """一个字段 = 上方小标签 + 下方控件，放进 grid 的同一格里。

        控件用工厂函数现场创建、master 就是这一格 —— 不能先在别处建好再塞进来：
        tkinter 的 pack/grid 认的是控件的真实 master，混着用会直接报
        "cannot use geometry manager grid inside ... already has slaves managed by pack"。
        """
        f = tk.Frame(parent, bg=C["surface"])
        field_label(f, text).pack(anchor="w", pady=(0, 7))
        ctl = make(f)
        ctl.pack(fill="x")
        return {"frame": f, "ctl": ctl}

    def _build_advanced(self, parent):
        st = self.app.settings
        g = tk.Frame(parent, bg=C["surface"])
        g.pack(fill="x", padx=22, pady=(0, 22))
        for i in range(3):
            g.columnconfigure(i, weight=1, uniform="adv")

        pa = tk.Frame(g, bg=C["surface"])
        pa.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 14))
        field_label(pa, "初始提示词（引导用词与标点）").pack(anchor="w", pady=(0, 7))
        Input(pa, textvariable=self.var_prompt, outer=C["surface"]).pack(fill="x")

        self._field(g, "从第 N 秒开始", lambda p: Input(
            p, textvariable=self.var_start, outer=C["surface"],
            placeholder="0"))["frame"].grid(
            row=1, column=0, sticky="ew", padx=(0, 14), pady=(0, 14))
        self._field(g, "处理时长（秒，空=全部）", lambda p: Input(
            p, textvariable=self.var_dur, outer=C["surface"]))["frame"].grid(
            row=1, column=1, sticky="ew", padx=(0, 14), pady=(0, 14))
        self._field(g, "硬字幕粗扫帧率", lambda p: Input(
            p, textvariable=self.var_fps, outer=C["surface"]))["frame"].grid(
            row=1, column=2, sticky="ew", pady=(0, 14))
        self._field(g, "OCR 置信度阈值", lambda p: Input(
            p, textvariable=self.var_score, outer=C["surface"]))["frame"].grid(
            row=2, column=0, sticky="ew", padx=(0, 14), pady=(0, 14))

        sw = tk.Frame(g, bg=C["surface"])
        sw.grid(row=3, column=0, columnspan=3, sticky="w")
        self.sw_vad = Switch(sw, "语音活动检测（口播 / 视频建议开，歌曲建议关）",
                             value=bool(st.get("vad")), outer=C["surface"])
        self.sw_vad.pack(anchor="w", pady=(0, 6))
        self.sw_split = Switch(sw, "按标点拆分过长行", value=bool(st.get("split")),
                               outer=C["surface"])
        self.sw_split.pack(anchor="w", pady=(0, 6))
        self.sw_merge = Switch(sw, "合并过短的相邻段", value=bool(st.get("merge")),
                               outer=C["surface"])
        self.sw_merge.pack(anchor="w", pady=(0, 6))
        self.sw_wav = Switch(sw, "保留中间 WAV（排查用）", value=bool(st.get("keep_wav")),
                             outer=C["surface"])
        self.sw_wav.pack(anchor="w")

        ff = tk.Frame(g, bg=C["surface"])
        ff.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(14, 0))
        field_label(ff, "ffmpeg 路径（留空则从 PATH 找）").pack(anchor="w", pady=(0, 7))
        fr = tk.Frame(ff, bg=C["surface"])
        fr.pack(fill="x")
        Input(fr, textvariable=self.var_ffmpeg, outer=C["surface"]).pack(
            side="left", fill="x", expand=True)
        Btn(fr, "浏览…", command=self.pick_ffmpeg, outer=C["surface"]).pack(
            side="left", padx=(10, 0))

    def _draw_adv_head(self):
        cv = self.adv_head
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 4:
            return
        cv.delete("all")
        cv.create_text(U(22), h / 2.0, text="高级参数", anchor="w", fill=C["ink"],
                       font=F(10, True))
        cv.create_text(U(96), h / 2.0, text="提示词 / 帧率 / 阈值 / 开关 / ffmpeg",
                       anchor="w", fill=C["ink4"], font=F(8))
        icon(cv, "down" if not self.adv_open else "check", w - U(30), h / 2.0, 12,
             C["ink3"])

    def toggle_adv(self):
        self.adv_open = not self.adv_open
        if self.adv_open:
            self.adv_body.pack(fill="x")
        else:
            self.adv_body.pack_forget()
        self._draw_adv_head()

    def _relayout_fields(self, cols):
        if cols == self._grid_cols:
            return
        self._grid_cols = cols
        for i in range(3):
            self.grid.columnconfigure(i, weight=1 if i < cols else 0,
                                      uniform="fld" if i < 3 else "")
        fields = (self.f_model, self.f_lang, self.f_threads,
                  self.f_subs, self.f_region, self.f_offset)
        for idx, f in enumerate(fields):
            f["frame"].grid_forget()
            r, c = divmod(idx, cols)
            f["frame"].grid(row=r, column=c, sticky="ew", padx=(0, 14), pady=(0, 14))

    # ---------------- 联动 ----------------
    def _on_mode(self, _v=None):
        has_lyrics = bool(self.var_lyrics.get().strip())
        if self.mode == MODE_ALIGN:
            if has_lyrics:
                self.lbl_mode.configure(
                    text="对齐模式：文字用你的原文，识别只提供时间锚点 —— 歌曲请务必用这个模式。",
                    fg=C["green"])
            else:
                self.lbl_mode.configure(
                    text="已选对齐模式，但还没指定歌词文本 —— 请上传 .txt / .lrc，"
                         "否则会自动退回听写。", fg=C["magenta"])
        else:
            self.lbl_mode.configure(
                text="自动听写：工具自己听、自己写字。口播 / 访谈适用；"
                     "歌曲会有大量谐音错字，这是原理决定的。", fg=C["ink4"])

    def _on_audio_only(self, _v=None):
        """「只识别音频」开关：把字幕来源锁到"不用字幕"，只走声音识别。

        记住切换前的字幕来源，关掉开关时还原 —— 用户不该因为试一下开关
        就把原来的设置弄丢。
        """
        on = self.sw_audio.get()
        if on:
            if self.var_subs.get() != "off":
                self._subs_backup = self.var_subs.get()
            self.var_subs.set("off")
        else:
            self.var_subs.set(getattr(self, "_subs_backup", None) or "auto")
        self.f_subs["ctl"].set_enabled(not on)
        self.app.settings["audio_only"] = on
        self._on_subs()
        # 开关状态也反映到页面描述上，省得用户猜它到底改了什么
        self.set_desc("只识别音频：跳过视频字幕与画面 OCR，只用声音生成歌词"
                      if on else self._desc0)

    def _on_subs(self, v=None):
        v = v or self.var_subs.get()
        if self.sw_audio.get():
            self.f_subs["ctl"].set_enabled(False)
        else:
            self.f_subs["ctl"].set_enabled(True)
        # 只有"自动"才用得上"字幕区域"（track/off 都不会去 OCR 画面）
        self.f_region["ctl"].set_enabled(v not in ("track", "off"))

    def _on_fmt(self, v=None):
        fmt = (v or self.var_fmt.get() or "lrc").lower()
        cur = self.var_out.get().strip()
        if cur:
            self.var_out.set(os.path.splitext(cur)[0] + "." + fmt)

    def _on_media_set(self):
        p = self.var_media.get().strip().strip('"')
        if not p:
            self.drop.set_caption("mp3 / wav / flac / m4a / mp4 / mkv / webm · "
                                  "不上传，全部本地处理")
            return
        ext = os.path.splitext(p)[1].lower()
        kind = "视频" if ext in GUI_VIDEO_EXT.split() else (
            "音频" if ext in GUI_AUDIO_EXT.split() else "未知类型")
        self.drop.set_caption("%s · %s" % (os.path.basename(p), kind))
        # 纯音频文件根本不存在视频字幕可提取，自动切到「只识别音频」——
        # 省得用户还要自己去想"字幕来源该选哪个"
        if kind == "音频" and not self.sw_audio.get():
            self.sw_audio.set(True)
            self._on_audio_only()
        if not self.var_out.get().strip():
            base = os.path.splitext(p)[0]
            if self.app.settings.get("out_dir"):
                base = os.path.join(self.app.settings["out_dir"],
                                    os.path.basename(base))
            self.var_out.set(base + "." + self.var_fmt.get())

    # ---------------- 选文件 ----------------
    def pick_media(self):
        p = filedialog.askopenfilename(
            title="选择音乐或视频",
            filetypes=[("音乐/视频", "*" + " *".join(
                (GUI_AUDIO_EXT + " " + GUI_VIDEO_EXT).split())),
                ("所有文件", "*.*")])
        if p:
            self.var_media.set(p)
            self._on_media_set()

    def pick_lyrics(self):
        p = filedialog.askopenfilename(
            title="选择歌词文本（txt 或 lrc）",
            filetypes=[("歌词/文本", "*.txt *.lrc"), ("所有文件", "*.*")])
        if p:
            self.var_lyrics.set(p)
            self.seg_mode.set(MODE_ALIGN)
            self.mode = MODE_ALIGN
            self._on_mode()

    def pick_out(self):
        ext = "." + self.var_fmt.get()
        p = filedialog.asksaveasfilename(
            title="保存为", defaultextension=ext,
            initialdir=self.app.settings.get("out_dir") or None,
            filetypes=[("%s 字幕" % self.var_fmt.get().upper(), "*" + ext),
                       ("所有文件", "*.*")])
        if p:
            self.var_out.set(p)
            low = p.lower()
            for k in ("lrc", "srt", "ass", "ssa", "vtt"):
                if low.endswith("." + k):
                    self.var_fmt.set(k)
                    break

    def pick_ffmpeg(self):
        p = filedialog.askopenfilename(
            title="选择 ffmpeg.exe", filetypes=[("ffmpeg", "ffmpeg.exe"),
                                                ("所有文件", "*.*")])
        if p:
            self.var_ffmpeg.set(p)

    def open_dir(self):
        out = self.var_out.get().strip()
        d = os.path.dirname(out) if out else _HERE
        if not os.path.isdir(d):
            d = _HERE
        try:
            os.startfile(d)
        except Exception:
            try:
                subprocess.Popen(["explorer", d])
            except Exception:
                _alert("打不开目录", d, error=True)

    def back_to_input(self):
        self.show("input")

    # ---------------- 开始 / 取消 ----------------
    def start(self):
        app = self.app
        if app.state["running"]:
            return
        media = self.var_media.get().strip().strip('"')
        if not media:
            messagebox.showwarning("缺少输入", "请先选择音乐或视频文件。")
            return
        if not os.path.isfile(media):
            messagebox.showerror("文件不存在", media)
            return
        model = self.var_model.get()
        subs_mode = self.var_subs.get()
        # 走纯字幕路径（track）时不需要模型；其余模式失败后会回退到语音识别，仍需模型
        if subs_mode != "track":
            if not os.path.isfile(os.path.join(MODELS_DIR, model, "model.bin")):
                if not messagebox.askyesno(
                        "模型未下载",
                        "模型 '%s' 还没下载。\n\n"
                        "可以到【模型管理】页下载，或先用已下载的模型继续。\n\n"
                        "是否改用已下载的模型？" % model):
                    app.select("models")
                    return
                have = usable_models()
                if not have:
                    messagebox.showerror("没有可用模型",
                                         "一个模型都没下载，请先到【模型管理】页下载。")
                    app.select("models")
                    return
                model = have[0]
                self.var_model.set(model)

        out = self.var_out.get().strip()
        if not out:
            out = os.path.splitext(media)[0] + "." + self.var_fmt.get()
            self.var_out.set(out)

        # 提前验证输出目录能不能写。原来是等 transcribe() 把解码 + 全部 OCR/ASR
        # 跑完（几十分钟到几小时），到写文件那一刻才抛 FileNotFoundError ——
        # 成果全丢，用户白等。几毫秒的检查换掉几小时的白跑，值。
        out_dir = os.path.dirname(os.path.abspath(out))
        if not os.path.isdir(out_dir):
            messagebox.showerror(
                "输出目录不存在",
                "输出目录不存在，请先创建或换一个：\n\n%s" % out_dir)
            return
        if not os.access(out_dir, os.W_OK):
            messagebox.showerror(
                "输出目录不可写",
                "输出目录不可写（可能是只读盘或权限不足）：\n\n%s" % out_dir)
            return

        lyrics = self.var_lyrics.get().strip()
        if self.mode == MODE_ALIGN and not lyrics:
            lyrics = ""

        def num(sv, cast, default=None):
            try:
                t = str(sv.get()).strip()
                return cast(t) if t else default
            except Exception:
                return default

        # 数值参数统一夹范围。原来只防"解析不了"，不防"解析出来不合理"：
        #   threads 填 0 会被 faster-whisper 当成"自动"（和用户预期相反），
        #   负数直接透传给 ctranslate2，超大值则绕过了"封顶 16"的设计意图；
        #   subs_fps 填 0 被 `or 2.0` 悄悄改成默认值，填负数会让抽帧时间戳算错、
        #   还会绕过降帧保护，最终 OCR 静默失败。
        thr = _clamp_int(num(self.var_threads, int), 1, 32,
                        min(16, os.cpu_count() or 4))
        fps = _clamp_float(num(self.var_fps, float, 2.0), 0.5, 60.0, 2.0)
        score = _clamp_float(num(self.var_score, float, 0.5), 0.0, 1.0, 0.5)
        kwargs = dict(
            media=media, model_name=model, language=self.var_lang.get(), out=out,
            prompt=self.var_prompt.get() or None,
            vad=self.sw_vad.get(), threads=thr,
            start=num(self.var_start, float), duration=num(self.var_dur, float),
            offset_ms=num(self.var_offset, int, 0) or 0,
            split_long_lines=self.sw_split.get(), merge=self.sw_merge.get(),
            ffmpeg=self.var_ffmpeg.get().strip() or None,
            keep_wav=self.sw_wav.get(), lyrics_file=lyrics or None,
            subs_mode=subs_mode, subs_region=self.var_region.get(),
            subs_fps=fps,
            subs_min_score=score,
            fmt=self.var_fmt.get(), script=self.var_script.get())

        app.state["running"] = True
        app.state["t0"] = time.time()
        app.state["media"] = media
        app.state["out"] = out
        app.state["kwargs"] = kwargs
        # 这次的参数顺手记成新的默认值：下次打开还是这套，不用重填
        try:
            app.settings.update(self.snapshot())
            save_json(SETTINGS_FILE, app.settings)
        except Exception:
            pass
        self.clear_log()
        self.lbl_task_name.configure(text=os.path.basename(media))
        ext = os.path.splitext(media)[1].lstrip(".").upper() or "未知"
        self.lbl_task_spec.configure(
            text="%s · 模型 %s · 字幕来源 %s · 输出 .%s"
                 % (ext, model, SUBS_LABEL.get(subs_mode, subs_mode),
                    self.var_fmt.get()))
        self.bar.set(0.0)
        self.lbl_stage.configure(text="正在准备…")
        self.lbl_pct.configure(text="0%")
        self.show("running")
        app.run_job(kwargs, media, out)

    def cancel(self):
        app = self.app
        if not app.state.get("running"):
            return
        if not messagebox.askyesno(
                "取消任务",
                "确定要取消吗？已经解出来的部分不会保留。\n\n"
                "当前这一步（解码 / 识别）没法中途打断，跑完就会停下；"
                "这段时间里不能再开新任务。"):
            return
        # 这里**不能**把 state["running"] 置 False。
        # 置了之后用户能立刻再点一次「开始生成歌词」，而后台那个 worker 还在跑
        # ffmpeg / 识别 —— 于是两个 transcribe() 同时跑同一个文件：抢 CPU、
        # 还会同时写同一个输出文件。真正的 running=False 要等 worker 真的退出，
        # 由 tick() 收到 done/error 时统一处理。
        app.state["cancelled"] = True
        app.q.put(("log", "[用户取消] 已请求中止，等当前步骤结束后停下…"))
        self.action_hint("正在中止…（当前步骤结束后停止）", C["magenta"])

    # ---------------- 日志 ----------------
    def clear_log(self):
        self._log_lines = 0
        for t in self.log_targets:
            t.configure(state="normal")
            t.delete("1.0", "end")
            t.configure(state="disabled")
        self.app.state["t0"] = time.time()

    def push_log(self, msg):
        # t0 为 0 表示"还没开始计时"（比如日志比 start() 先到），
        # 这时候要给 00:00 而不是拿 epoch 去减，否则会印出 29852218:41 这种鬼时间
        t0 = self.app.state.get("t0") or 0.0
        now = (time.time() - t0) if t0 else 0.0
        stamp = "%02d:%05.2f" % (int(now) // 60, now % 60)
        for t in self.log_targets:
            t.configure(state="normal")
            try:
                t.insert("end", stamp + "   ", "cur")
            except Exception:
                pass
            t.insert("end", msg + "\n")
            t.see("end")
            t.configure(state="disabled")
        self._log_lines += 1
        # 同时落盘：界面日志面板里的内容（含转写失败堆栈）也写进 lyric-maker.log，
        # 用户不用再开 --dl-test 才能拿到失败详情。
        logfile_write(msg)

    def on_log_tick(self):
        t0 = self.app.state.get("t0") or 0.0
        el = (time.time() - t0) if t0 else 0.0
        self.action_hint("识别中… 已用 %.0f 秒" % el, C["magenta"])
        self.lbl_stage.configure(text="正在处理… 已用 %.0f 秒" % el)
        # 进度没有真实回调，用时间做一个保守估算（large-v3 约 1.4 倍实时）
        try:
            idx = self._log_lines
            self.bar.set(min(0.96, idx / 40.0))
            self.lbl_pct.configure(text="%d%%" % int(min(0.96, idx / 40.0) * 100))
        except Exception:
            pass

    def on_error(self, tb):
        self.push_log("=== 错误 ===")
        for line in tb.rstrip().splitlines():
            self.push_log(line)
        self.lbl_stage.configure(text="处理失败")
        # 停在日志页让人能看到堆栈；动作条换成"失败"那一套
        self.show("logview")
        self.current = "error"
        self._sync_actions()
        messagebox.showerror("出错了", tb.strip().splitlines()[-1] if tb else "未知错误")

    def on_done(self, res):
        segs = res.get("segments") or []
        n = len(segs)
        el = res.get("elapsed") or 0
        dur = res.get("duration") or 0
        self._fill_result(res, segs)
        self.show("done")

        # 记一条历史
        try:
            history_add({
                "ts": time.time(),
                "media": self.app.state.get("media") or self.var_media.get(),
                "out": res.get("out") or self.app.state.get("out") or "",
                "mode": res.get("mode") or ("subtitle" if res.get("source") else "asr"),
                "source": res.get("source") or "",
                "subs": self.var_subs.get(),
                "model": self.var_model.get(),
                "fmt": self.var_fmt.get(),
                "lines": n,
                "elapsed": el,
                "duration": dur,
                "status": "done",
                "kwargs": {},
            })
        except Exception:
            pass
        if self.app.current == "history":
            self.app.pages["history"].reload()

    # ---------------- 视图二：运行中 ----------------
    def _make_log_panel(self, parent, subtitle):
        p = RoundedLogPanel(parent, head_bg=C["log_bg"], body_bg=C["log_bg"],
                            outline=C["line"])
        head = tk.Frame(p.body, bg=C["log_bg"], height=46)
        head.pack(fill="x")
        head.pack_propagate(False)
        tk.Label(head, text="运行日志", bg=C["log_bg"], fg=C["ink2"],
                 font=F(9, True)).pack(side="left", padx=20)
        tk.Label(head, text=subtitle, bg=C["log_bg"], fg=C["ink5"],
                 font=F(7)).pack(side="right", padx=20)
        txt = p.txt
        txt.tag_configure("cur", foreground=C["link"])
        txt.tag_configure("err", foreground=C["err"])
        txt.configure(state="disabled")
        return p, txt

    def _build_running(self):
        f = tk.Frame(self.holder, bg=C["canvas"])
        self.views["running"] = f
        card = Panel(f, pad=22)
        card.pack(fill="x", pady=(0, 16))
        top = tk.Frame(card.body, bg=C["surface"])
        top.pack(fill="x")
        fic = tk.Canvas(top, width=42, height=42, bg=C["surface"],
                        highlightthickness=0, bd=0)
        fic.pack(side="left")
        fic.bind("<Configure>", lambda e: (fic.delete("all"),
                                           rrect(fic, 0, 0, 42, 42, 13, C["bg_tint"]),
                                           icon(fic, "file", 21, 21, 20, C["magenta"])))
        meta = tk.Frame(top, bg=C["surface"])
        meta.pack(side="left", padx=(14, 0), fill="x", expand=True)
        self.lbl_task_name = tk.Label(meta, text="", bg=C["surface"], fg=C["ink"],
                                      font=F(10, True), anchor="w")
        self.lbl_task_name.pack(anchor="w")
        self.lbl_task_spec = tk.Label(meta, text="", bg=C["surface"], fg=C["ink3"],
                                      font=F(8), anchor="w")
        self.lbl_task_spec.pack(anchor="w", pady=(4, 0))
        self.chip_mode = Chip(top, "对齐模式", fill=C["green_bg"], fg=C["green"],
                              outer=C["surface"], height=28, size=8)
        self.chip_mode.pack(side="right")

        pr = tk.Frame(card.body, bg=C["surface"])
        pr.pack(fill="x", pady=(18, 0))
        self.bar = Bar(pr, height=8, radius=4, outer=C["surface"])
        self.bar.pack(fill="x")
        row = tk.Frame(card.body, bg=C["surface"])
        row.pack(fill="x", pady=(10, 0))
        self.lbl_stage = tk.Label(row, text="", bg=C["surface"], fg=C["ink2"],
                                  font=F(8), anchor="w")
        self.lbl_stage.pack(side="left")
        self.lbl_pct = tk.Label(row, text="0%", bg=C["surface"], fg=C["magenta"],
                                font=FN(8, True))
        self.lbl_pct.pack(side="right")

        chips = tk.Frame(card.body, bg=C["surface"])
        chips.pack(fill="x", pady=(14, 0))
        self.run_chips = []
        for txt in ("large-v3", "中文", "16 线程", "输出 .lrc"):
            c = Chip(chips, txt, fill=C["surface_in"], fg=C["ink3"], outer=C["surface"],
                     height=26, size=8, radius=8, pad_l=11, pad_r=11)
            c.pack(side="left", padx=(0, 8))
            self.run_chips.append(c)

        lp, self.txt_running = self._make_log_panel(f, "自动滚动 · 输出至 lyric-maker.log")
        lp.pack(fill="both", expand=True)

    # ---------------- 视图三：完成 ----------------
    def _build_done(self):
        f = tk.Frame(self.holder, bg=C["canvas"])
        self.views["done"] = f
        card = Panel(f, pad=22)
        card.pack(fill="x", pady=(0, 16))
        top = tk.Frame(card.body, bg=C["surface"])
        top.pack(fill="x")
        ck = tk.Canvas(top, width=38, height=38, bg=C["surface"],
                       highlightthickness=0, bd=0)
        ck.pack(side="left")
        ck.bind("<Configure>", lambda e: (ck.delete("all"),
                                          rrect(ck, 0, 0, 38, 38, 19, C["green_bg"]),
                                          icon(ck, "check", 19, 19, 20, C["green"])))
        meta = tk.Frame(top, bg=C["surface"])
        meta.pack(side="left", padx=(12, 0), fill="x", expand=True)
        self.lbl_done_title = tk.Label(meta, text="生成完成", bg=C["surface"],
                                       fg=C["ink"], font=F(10, True), anchor="w")
        self.lbl_done_title.pack(anchor="w")
        self.lbl_done_path = tk.Label(meta, text="", bg=C["surface"], fg=C["ink3"],
                                      font=F(8), anchor="w")
        self.lbl_done_path.pack(anchor="w", pady=(4, 0))
        self.chip_done = Chip(top, "对齐模式 · 零错字", fill=C["green_bg"],
                              fg=C["green"], outer=C["surface"], height=28, size=8)
        self.chip_done.pack(side="right")

        stats = tk.Frame(card.body, bg=C["surface"])
        stats.pack(fill="x", pady=(16, 0))
        self.stat_vals = {}
        for i, (key, cap) in enumerate((("lines", "行歌词"), ("dur", "音频时长"),
                                        ("cost", "端到端耗时"), ("speed", "相对实时倍速"))):
            box = tk.Frame(stats, bg=C["surface_in"], bd=0)
            box.pack(side="left", fill="x", expand=True,
                     padx=(0, 12) if i < 3 else (0, 0))
            inner = tk.Frame(box, bg=C["surface_in"])
            inner.pack(fill="x", padx=14, pady=11)
            v = tk.Label(inner, text="—", bg=C["surface_in"], fg=C["ink"],
                         font=FN(14, True), anchor="w")
            v.pack(anchor="w")
            tk.Label(inner, text=cap, bg=C["surface_in"], fg=C["ink4"],
                     font=F(7), anchor="w").pack(anchor="w", pady=(4, 0))
            self.stat_vals[key] = v

        lc = Panel(f, pad=22)
        lc.pack(fill="both", expand=True)
        lh = tk.Frame(lc.body, bg=C["surface"])
        lh.pack(fill="x")
        self.lbl_prev_head = tk.Label(lh, text="歌词预览", bg=C["surface"],
                                      fg=C["ink2"], font=F(9, True))
        self.lbl_prev_head.pack(side="left")
        Btn(lh, "复制全部", icon_name="copy", command=self.copy_all,
            outer=C["surface"], height=30, pad_x=12, size=8).pack(side="right")
        self.txt_prev = tk.Text(lc.body, bg=C["surface"], fg="#E4E6F7", relief="flat",
                                bd=0, highlightthickness=0, font=F(9), wrap="none",
                                selectbackground=C["surface_hi"], padx=6, pady=12,
                                height=6)
        self.txt_prev.pack(fill="both", expand=True)
        self.txt_prev.tag_configure("ts", foreground=C["link"], font=FN(9))
        self.txt_prev.tag_configure("hi", foreground=C["ink"], font=F(9, True))
        self.txt_prev.configure(state="disabled")

    def _build_logview(self):
        f = tk.Frame(self.holder, bg=C["canvas"])
        self.views["logview"] = f
        lp, self.txt_logview = self._make_log_panel(f, "完整输出 · 也写入 lyric-maker.log")
        lp.pack(fill="both", expand=True)
        self.log_targets = [self.txt_running, self.txt_logview]

    def _fill_result(self, res, segs):
        n = len(segs)
        el = res.get("elapsed") or 0
        dur = res.get("duration") or 0
        self.lbl_done_title.configure(text="生成完成 · %d 行歌词已写入" % n)
        self.lbl_done_path.configure(text="%s" % (res.get("out") or ""))
        src = res.get("source")
        self.chip_done._text = ("取自 %s" % src) if src else (
            "对齐模式 · 零错字" if res.get("mode") == "align" else "自动听写")
        self.chip_done.set_text(self.chip_done._text)
        self.stat_vals["lines"].configure(text=str(n))
        self.stat_vals["dur"].configure(text=fmt_dur(dur))
        self.stat_vals["cost"].configure(text=fmt_cost(el))
        self.stat_vals["speed"].configure(
            text=("%.1fx" % (dur / el)) if (el and dur) else "—")
        self.lbl_prev_head.configure(
            text="歌词预览 · 前 %d / %d 行" % (min(20, n), n))
        t = self.txt_prev
        t.configure(state="normal")
        t.delete("1.0", "end")
        for s in segs[:20]:
            t.insert("end", fmt_ts(s.get("start", 0)), "ts")
            t.insert("end", "     " + str(s.get("text", "")).strip() + "\n", "hi")
        t.configure(state="disabled")
        self._copy_text = "".join(
            "%s  %s\n" % (fmt_ts(s.get("start", 0)), str(s.get("text", "")).strip())
            for s in segs)

    def copy_all(self):
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(getattr(self, "_copy_text", ""))
            self.action_hint("已复制到剪贴板", C["green"])
        except Exception:
            pass

    def prefill(self, rec):
        """从历史里"重跑"：把媒体、歌词、输出路径填回来，参数不还原（保持当前设置）。"""
        media = rec.get("media") or ""
        if media:
            self.var_media.set(media)
            self._on_media_set()
        out = rec.get("out") or ""
        if out:
            self.var_out.set(out)
        fmt = (rec.get("fmt") or "lrc").lower()
        if fmt in ("lrc", "srt", "ass", "ssa", "vtt"):
            self.var_fmt.set(fmt)
        if rec.get("model") in MODEL_REPOS:
            self.var_model.set(rec["model"])
            self.f_model["ctl"].refresh()
        self.show("input")

    def drop_files(self, data):
        """拖放进来的可能是多个文件，只取第一个能识别的音视频。"""
        try:
            items = self.root.tk.splitlist(data)
        except Exception:
            items = [str(data).strip("{}")]
        for it in items:
            ext = os.path.splitext(it)[1].lower()
            if ext in (GUI_AUDIO_EXT + " " + GUI_VIDEO_EXT).split():
                self.var_media.set(it)
                self._on_media_set()
                return
            if ext in (".txt", ".lrc"):
                self.var_lyrics.set(it)
                self.seg_mode.set(MODE_ALIGN)
                self.mode = MODE_ALIGN
                self._on_mode()


    def apply_settings(self, st):
        """设置页保存后把默认值同步到本页。"""
        if st.get("model") in MODEL_REPOS:
            self.var_model.set(st["model"])
            self.f_model["ctl"].refresh()
        self.var_lang.set(st.get("language", "zh"))
        _sc = st.get("script", "auto") or "auto"
        if _sc not in list(SCRIPT_MODES) + ["auto"]:
            _sc = "auto"
        self.var_script.set(_sc)
        if getattr(self, "f_script", None):
            self.f_script["ctl"].refresh()
        self.var_fmt.set(st.get("fmt", "lrc"))
        self.var_threads.set(str(st.get("threads", 4)))
        self.var_ffmpeg.set(st.get("ffmpeg", ""))
        self.var_prompt.set(st.get("prompt", DEFAULT_PROMPT))
        for sw, key in ((self.sw_vad, "vad"), (self.sw_split, "split"),
                        (self.sw_merge, "merge"), (self.sw_wav, "keep_wav")):
            sw.set(bool(st.get(key)), notify=False)
        if self.var_media.get().strip() and not self.var_out.get().strip():
            self._on_media_set()

    def snapshot(self):
        """把这次的参数记成新的默认值 —— 下次打开还是这套。"""
        return {
            "model": self.var_model.get(),
            "language": self.var_lang.get(),
            "script": self.var_script.get(),
            "fmt": self.var_fmt.get(),
            "threads": self.var_threads.get(),
            "ffmpeg": self.var_ffmpeg.get().strip(),
            "prompt": self.var_prompt.get(),
            "vad": self.sw_vad.get(),
            "split": self.sw_split.get(),
            "merge": self.sw_merge.get(),
            "keep_wav": self.sw_wav.get(),
            "subs_fps": self.var_fps.get(),
            "min_score": self.var_score.get(),
        }

    def refresh_models(self):
        self.app.refresh_model_chip()
        if self._grid_cols:
            self._on_mode()


# =====================================================================
# 页面二：模型管理
# =====================================================================

class PageModels(Page):
    def __init__(self, master, app):
        Page.__init__(self, master, app, "本地模型",
                      "模型不打进程序包 · 只在首次使用时下载 · 大文件走 ModelScope 源")
        Btn(self.head_right, "打开模型目录", icon_name="folder",
            command=self.open_dir, outer=C["canvas"]).pack(side="right")
        self.sc = Scroll(self.content, bg=C["canvas"])
        self.sc.pack(fill="both", expand=True)
        self.busy = None
        self._rows = {}
        # 已经建好的行对应哪一组"已下载"状态；变了才重建整页（见 reload）
        self._rows_built_for = None
        Btn(self.action_right, "重新检查", icon_name="refresh", command=self.reload,
            outer=C["canvas"]).pack(side="right", padx=(10, 0))
        Btn(self.action_right, "下载未完成的模型", icon_name="download", kind="green",
            bold=True, size=10, command=self.download_missing,
            outer=C["canvas"], pad_x=22).pack(side="right")
        self.action_hint("下载可中断 · 会自动断点续传，下次继续")
        self.reload()

    def on_show(self):
        self.reload()

    def open_dir(self):
        try:
            os.makedirs(MODELS_DIR, exist_ok=True)
            os.startfile(MODELS_DIR)
        except Exception:
            _alert("模型目录", MODELS_DIR)

    def reload(self):
        """刷新模型列表。

        原来每次都把 6 行全部 destroy 掉再重建，实测每次 ~57ms —— 而切到这一页
        就调一次，切一次卡一下（整页控件 83 个，全带 <Configure> 重绘）。
        现在改成**就地更新**：行只在第一次建，之后只改状态文字、按钮文案和
        样式。真正的下载/删除会改变行集合，那时才整体重建。
        """
        if self._rows and self._rows_built_for == self._row_signature():
            self._refresh_rows()
            return
        for c in self.sc.inner.winfo_children():
            c.destroy()
        self._rows = {}
        card = Panel(self.sc.inner, pad=12)
        card.pack(fill="both", expand=True)
        head = tk.Frame(card.body, bg=C["surface"])
        head.pack(fill="x", padx=(18, 18), pady=(6, 8))
        cols = (("模型", 0, 1), ("大小", 104, 0), ("来源", 150, 0),
                ("状态", 150, 0), ("操作", 90, 0))
        for i, (txt, w, weight) in enumerate(cols):
            head.grid_columnconfigure(i, weight=weight, minsize=w)
            tk.Label(head, text=txt, bg=C["surface"], fg=C["ink4"], font=F(7),
                     anchor="w").grid(row=0, column=i, sticky="ew")
        self._head = head

        order = ["large-v3", "turbo", "medium", "small", "base", "tiny"]
        for i, name in enumerate(order):
            self._rows[name] = self._row(card.body, name)
        self._rows_built_for = self._row_signature()

        tk.Label(card.body, text="模型目录 %s" % MODELS_DIR, bg=C["surface"],
                 fg=C["ink4"], font=F(7), anchor="w").pack(
            fill="x", padx=(18, 18), pady=(10, 4))

    def _row_signature(self):
        """当前哪些模型已下载。变了才值得重建整页。"""
        return tuple((n, is_downloaded(n)) for n in
                     ("large-v3", "turbo", "medium", "small", "base", "tiny"))

    def _refresh_rows(self):
        """就地刷新每行的状态与按钮，不重建控件。"""
        for name, rec in self._rows.items():
            have = is_downloaded(name)
            src, repo, size, desc = ALL[name]
            rec["status"].configure(
                text="已下载 · %s" % desc if have else "未下载",
                fg=C["green"] if have else C["ink4"])
            if self.busy:
                # 正在下载/删除：这一行正由下载进度改着，别抢。
                continue
            btn = rec["btn"]
            # 按钮的文案/样式/命令都在 _draw/_natural 里读这几个字段，
            # 就地改完重画一次，比销毁重建快一个数量级。
            btn._text = "删除" if have else "下载"
            btn._kind = "outline" if have else "blue"
            btn._cmd = ((lambda n=name: self._remove(n)) if have
                        else (lambda n=name: self._download(n)))
            btn._enabled = True
            if not btn._fill_width:
                btn.configure(width=btn._natural())
            btn._draw()

    def _row(self, parent, name):
        src, repo, size, desc = ALL[name]
        have = is_downloaded(name)
        row = Panel(parent, radius=16, pad=18, fill=C["surface_in"], outer=C["surface"])
        row.pack(fill="x", pady=(0, 6))
        g = row.body
        for i, w in ((1, 104), (2, 150), (3, 150), (4, 90)):
            g.grid_columnconfigure(i, minsize=w)
        g.grid_columnconfigure(0, weight=1)
        nm = tk.Frame(g, bg=C["surface_in"])
        nm.grid(row=0, column=0, sticky="ew")
        tk.Label(nm, text=name, bg=C["surface_in"], fg=C["ink"], font=FN(10, True),
                 anchor="w").pack(side="left")
        if name == "large-v3":
            Chip(nm, "中文首选", fill=C["magenta_bg"], fg=C["magenta"],
                 outer=C["surface_in"], height=22, size=7, radius=7, pad_l=9,
                 pad_r=9).pack(side="left", padx=(10, 0))
        tk.Label(g, text=size, bg=C["surface_in"], fg=C["ink2"], font=FN(9),
                 anchor="w").grid(row=0, column=1, sticky="ew")
        tk.Label(g, text=("%s 源" % src), bg=C["surface_in"], fg=C["ink3"], font=F(8),
                 anchor="w").grid(row=0, column=2, sticky="ew")
        st = tk.Label(g, text="已下载 · %s" % desc if have else "未下载",
                      bg=C["surface_in"], fg=C["green"] if have else C["ink4"],
                      font=F(8), anchor="w")
        st.grid(row=0, column=3, sticky="ew")
        act = Btn(g, "删除" if have else "下载", kind="outline" if have else "blue",
                  height=34, pad_x=14, size=8,
                  command=(lambda n=name: self._remove(n)) if have
                  else (lambda n=name: self._download(n)),
                  outer=C["surface_in"], fill_width=True)
        act.grid(row=0, column=4, sticky="ew")
        return {"status": st, "btn": act, "row": row}

    def _remove(self, name):
        if not messagebox.askyesno("删除模型",
                                   "确定删除 %s 吗？下次要用得重新下载。" % name):
            return
        try:
            shutil.rmtree(os.path.join(MODELS_DIR, name))
        except Exception as e:
            _alert("删不掉", str(e), error=True)
        self.reload()
        self.app.refresh_storage()

    def _download(self, name):
        if self.busy:
            _alert("正在下载", "已经有一个下载在跑了（%s），等它结束再说。" % self.busy)
            return
        self.busy = name
        rec = self._rows.get(name)
        if rec:
            rec["status"].configure(text="排队中…", fg=C["magenta"])
            rec["btn"].set_enabled(False)
        self.action_hint("正在下载 %s …" % name, C["magenta"])
        # 真正的下载线程交给 App 统一管理（统一并发闸门 + 事件回传）
        self.app.download_model(name)

    def download_missing(self):
        for name in ("large-v3", "turbo", "medium", "small", "base", "tiny"):
            if not is_downloaded(name):
                self._download(name)
                return
        _alert("都齐了", "所有模型都已经下载好了。")

    def on_event(self, kind, payload):
        if kind == "modellog":
            name, msg = payload
            rec = self._rows.get(name)
            if rec:
                txt = msg.strip()
                if txt:
                    rec["status"].configure(text=txt[:34], fg=C["magenta"])
            self.action_hint(msg.strip()[:80] or "下载中…", C["magenta"])
        else:
            # 这里原来是 `name, rc, reason = payload`，一旦 payload 不是
            # 三元组（比如队列里混进了别的事件）就 ValueError 崩掉整页。
            # 事件来自后台线程，结构不由本页控制，必须防御式取。
            try:
                name, rc, reason = payload
            except Exception:
                name, rc, reason = str(payload), 1, "事件数据异常"
            self.busy = None
            self.reload()
            self.app.refresh_storage()
            if rc == 0:
                self.action_hint("%s 下载完成" % name, C["green"])
            else:
                self.action_hint("%s 下载失败：%s" % (name, _one_line(reason)),
                                 C["err"])
                # 光在底下写一行字太容易被忽略，失败原因直接弹出来
                _alert("下载失败", "%s\n\n%s" % (name, reason or "没有更多细节了。"))


# =====================================================================
# 页面三：视频下载
# =====================================================================


class PageDownload(Page):
    """粘贴链接 → 解析 → 下载。

    不用控制台、不用另外装 yt-dlp：内置的 Python API 直接跑。
    需要登录态的视频可以读本机浏览器 Cookie（内存使用），也可以把
    手动导出的 cookies.txt 丢进主目录的 cookies/ 文件夹。
    """

    def __init__(self, master, app):
        Page.__init__(self, master, app, "视频下载",
                      "粘贴链接即可下载；登录才能看的内容会读取浏览器 Cookie")
        st = app.settings
        self.url = tk.StringVar()
        self.fmt = tk.StringVar(value="best")
        self.container = tk.StringVar(value="auto")
        self.browser = tk.StringVar(value=st.get("dl_browser") or "none")
        self.outdir = tk.StringVar(
            value=st.get("dl_dir") or os.path.join(_HERE, DOWNLOAD_REL))
        self._job = None
        self._probe_job = None     # 解析阶段用的可取消 job（下载时 _job 才非空）
        self._busy = False
        self._parsed = None
        self._log_tail = []
        self._queue = []           # 一次粘了多个链接时，剩下的排这里逐个下

        Btn(self.head_right, "打开输出目录", icon_name="folder",
            command=self.open_outdir, outer=C["canvas"], height=36).pack(side="right")

        self.sc = Scroll(self.content, bg=C["canvas"])
        self.sc.pack(fill="both", expand=True)
        inner = self.sc.inner

        # ---------------- 视频链接 ----------------
        c1 = Panel(inner, pad=18)
        c1.pack(fill="x", pady=(0, 12))
        field_label(c1.body, "视频链接").pack(anchor="w", pady=(0, 7))
        r1 = tk.Frame(c1.body, bg=C["surface"])
        r1.pack(fill="x")
        self.inp = Input(r1, textvariable=self.url, height=40,
                         placeholder="粘贴视频链接，或按 Ctrl+V",
                         icon_name="download",
                         outer=C["surface"], on_return=self.parse)
        self.inp.pack(side="left", fill="x", expand=True)
        # Ctrl+V 直接粘贴链接。两条路都要有，缺一个就会出现"焦点不在框里就贴不了"：
        #   ① 绑在 Entry 上：焦点在框里时，用 after_idle 让 Entry 自带的 Ctrl+V
        #      先跑完，再把整段替换成提取出来的链接；
        #   ② 绑在 toplevel 上：焦点在页面任何别处时也能贴（见 _on_ctrl_v）。
        self.inp.entry().bind(
            "<Control-v>",
            lambda e: self.winfo_toplevel().after_idle(self._paste_hotkey),
            add="+")
        self.winfo_toplevel().bind("<Control-v>", self._on_ctrl_v, add="+")
        Btn(r1, "解析", kind="blue", command=self.parse, outer=C["surface"],
            height=40).pack(side="left", padx=(10, 0))
        Btn(r1, "粘贴", command=self.paste, outer=C["surface"],
            height=40).pack(side="left", padx=(10, 0))
        self.lbl_info = label(c1.body, "", size=8, color=C["ink3"])
        self.lbl_info.pack(fill="x", pady=(10, 0))
        self.set_desc(
            "粘贴链接即可下载；登录才能看的内容会读取浏览器 Cookie。"
            "可以直接粘整段分享文案，会自动挑出链接；一次粘多个则排成队列依次下载。")

        # ---------------- 下载选项 ----------------
        c2 = Panel(inner, pad=18)
        c2.pack(fill="x", pady=(0, 12))
        field_label(c2.body, "下载选项").pack(anchor="w", pady=(0, 7))
        g = tk.Frame(c2.body, bg=C["surface"])
        g.pack(fill="x")
        g.columnconfigure(0, weight=0, minsize=U(190))
        g.columnconfigure(1, weight=0, minsize=U(190))
        g.columnconfigure(2, weight=1)
        self.sel_fmt = Select(g, [c[0] for c in FMT_CHOICES], textvariable=self.fmt,
                              command=self._sync_labels, outer=C["surface"],
                              labeler=fmt_label, height=40)
        self.sel_fmt.grid(row=0, column=0, sticky="ew")
        self.sel_cont = Select(g, [c[0] for c in CONTAINER_CHOICES],
                               textvariable=self.container, outer=C["surface"],
                               labeler=lambda v: dict(CONTAINER_CHOICES)[v],
                               command=self._sync_fmthint, height=40)
        self.sel_cont.grid(row=0, column=1, sticky="ew", padx=(U(12), 0))
        holder = tk.Frame(g, bg=C["surface"])
        holder.grid(row=0, column=2, sticky="ew", padx=(U(12), 0))
        field_label(holder, "保存位置").pack(anchor="w", pady=(0, 7))
        rr = tk.Frame(holder, bg=C["surface"])
        rr.pack(fill="x")
        # 只读路径框：之前用 tk.Label 画，方角 + 高度对不上，
        # 夹在一堆圆角控件里特别跳（就是"风格对不上"的那处）。
        # 改用 Input + state="readonly" —— 圆角、40px 高都和旁边一致，
        # 而且是只读 Entry，用户还能选中复制路径。
        self.var_dir = tk.StringVar(value=self._short_dir())
        self.inp_dir = Input(rr, textvariable=self.var_dir, height=40,
                             outer=C["surface"], fill=C["surface_in"])
        self.inp_dir.pack(side="left", fill="x", expand=True)
        self.inp_dir.entry().configure(state="readonly")
        Btn(rr, "选择…", command=self.pick_dir, outer=C["surface"],
            height=40).pack(side="left", padx=(U(10), 0))
        sw = tk.Frame(c2.body, bg=C["surface"])
        sw.pack(fill="x", pady=(U(14), 0))
        self.sw_subs = Switch(sw, "同时下载字幕", value=True, outer=C["surface"])
        self.sw_subs.pack(side="left")
        self.sw_thumb = Switch(sw, "保存封面图", value=False, outer=C["surface"])
        self.sw_thumb.pack(side="left", padx=(U(22), 0))
        # 格式选择的代价说明：转封装很快，重编码很慢，用户不该踩完才知道。
        self.lbl_fmthint = label(c2.body, "", size=8, color=C["ink3"])
        self.lbl_fmthint.pack(fill="x", pady=(U(10), 0))

        # ---------------- 文件名模板 ----------------
        # 下完之后再去资源管理器改名很别扭，所以把文件名做成模板：
        # 解析完立刻能看到"会存成什么名字"，不满意就改模板再下。
        # 紧贴在「下载选项」下面 —— 命名和格式是同一件事的一部分，
        # 放太下面会被挤到要滚动才看得到（实测下载页内容比窗口高）。
        c2b = Panel(inner, pad=14)
        c2b.pack(fill="x", pady=(0, 10))
        field_label(c2b.body, "文件名（模板）").pack(anchor="w", pady=(0, 6))
        nr = tk.Frame(c2b.body, bg=C["surface"])
        nr.pack(fill="x")
        self.var_tpl = tk.StringVar(value=st.get("dl_name_tpl") or "")
        self.inp_tpl = Input(nr, textvariable=self.var_tpl, height=36,
                             outer=C["surface"],
                             placeholder="解析后自动填入原文件名，也可点下方方案")
        self.inp_tpl.pack(side="left", fill="x", expand=True)
        Btn(nr, "恢复默认", command=self.reset_tpl, outer=C["surface"],
            height=36).pack(side="left", padx=(U(10), 0))
        # 模板改动 -> 立即刷新预览
        self.var_tpl.trace_add("write", lambda *a: self._sync_labels())
        # 命名预设：点一下换一整套（覆盖原模板），不是往上叠占位符。
        vb = tk.Frame(c2b.body, bg=C["surface"])
        vb.pack(fill="x", pady=(U(8), 0))
        field_label(vb, "命名方案（点一下即替换）").pack(anchor="w", pady=(0, 6))
        pg = tk.Frame(vb, bg=C["surface"])
        pg.pack(fill="x")
        for i, (disp, tpl) in enumerate(TPL_PRESETS):
            Btn(pg, disp, command=(lambda t=tpl: self.apply_preset(t)),
                outer=C["surface"], size=8, height=28).grid(
                row=i // 4, column=i % 4, sticky="ew",
                padx=(0, U(5)), pady=(0, U(5)))
        for i in range(4):
            pg.columnconfigure(i, weight=1, uniform="p")
        # 变量按钮收进「更多变量」：手动微调时才用得上，默认不占地方
        vrow = tk.Frame(c2b.body, bg=C["surface"])
        vrow.pack(fill="x", pady=(U(4), 0))
        Btn(vrow, "更多变量…", command=self._toggle_vars, outer=C["surface"],
            size=8, height=24).pack(side="left")
        self.frm_vars = tk.Frame(c2b.body, bg=C["surface"])
        vtip = label(self.frm_vars, "点一下加入模板，再点一次可移除",
                     size=8, color=C["ink3"])
        vtip.pack(anchor="w", pady=(U(6), U(4)))
        vg = tk.Frame(self.frm_vars, bg=C["surface"])
        vg.pack(fill="x", pady=(0, 0))
        for i, (tok, _tip) in enumerate(TPL_VARS):
            Btn(vg, tok.replace("%(", "").replace(")s", ""), command=
                (lambda t=tok: self._insert_token(t)),
                outer=C["surface"], size=8, height=24).grid(
                row=i // 5, column=i % 5, sticky="ew",
                padx=(0, U(5)), pady=(0, U(4)))
        for i in range(5):
            vg.columnconfigure(i, weight=1, uniform="v")
        self._vars_open = False
        self.frm_vars.pack_forget()
        # 实时预览
        self.lbl_prev = label(c2b.body, "", size=8, color=C["ink3"])
        self.lbl_prev.pack(fill="x", pady=(U(6), 0))

        # ---------------- 登录与 Cookie ----------------
        c3 = Panel(inner, pad=18)
        c3.pack(fill="x")
        field_label(c3.body, "登录与 Cookie").pack(anchor="w", pady=(0, 7))
        g3 = tk.Frame(c3.body, bg=C["surface"])
        g3.pack(fill="x")
        g3.columnconfigure(0, weight=0, minsize=U(190))
        g3.columnconfigure(1, weight=1)
        self.sel_browser = Select(g3, [c[0] for c in BROWSER_CHOICES],
                                 textvariable=self.browser, outer=C["surface"],
                                 labeler=lambda v: dict(BROWSER_CHOICES)[v],
                                 command=self._sync_browser, height=40)
        self.sel_browser.grid(row=0, column=0, sticky="ew")
        ck = tk.Frame(g3, bg=C["surface"])
        ck.grid(row=0, column=1, sticky="ew", padx=(U(12), 0))
        field_label(ck, "Cookie 文件夹").pack(anchor="w", pady=(0, 7))
        cr = tk.Frame(ck, bg=C["surface"])
        cr.pack(fill="x")
        # 同上：方角 Label 换成只读 Input，风格统一
        self.var_cookie = tk.StringVar(value=_short_path(cookie_dir()))
        self.inp_cookie = Input(cr, textvariable=self.var_cookie, height=40,
                               outer=C["surface"], fill=C["surface_in"])
        self.inp_cookie.pack(side="left", fill="x", expand=True)
        self.inp_cookie.entry().configure(state="readonly")
        Btn(cr, "更改…", command=self.pick_cookie_dir, outer=C["surface"],
            height=40).pack(side="left", padx=(U(10), 0))
        Btn(cr, "打开", command=self.open_cookie_dir, outer=C["surface"],
            height=40).pack(side="left", padx=(U(10), 0))
        Btn(cr, "从浏览器更新", kind="blue", icon_name="refresh",
            command=self.refresh_cookie, outer=C["surface"],
            height=40).pack(side="left", padx=(U(10), 0))
        # Chrome 127+ 读浏览器库的路在运行期必然失败，插件导出才是能走通的，
        # 所以给它一个和「从浏览器更新」并列的入口，而不是藏在报错弹窗里。
        Btn(cr, "插件导出说明", icon_name="help",
            command=self.open_cookie_help, outer=C["surface"],
            height=40).pack(side="left", padx=(U(10), 0))
        self.lbl_cookie = label(c3.body, "", size=8, color=C["ink3"])
        self.lbl_cookie.pack(fill="x", pady=(U(12), 0))
        # 自动导入开关：默认开。放在状态行下面而不是动作条上，
        # 因为它和"插件导出说明"是同一件事的两半（怎么导出 / 导出后怎么收）。
        self.sw_autoimport = Switch(
            c3.body, "自动导入插件导出的 cookies.txt（盯着下载文件夹，不用手动拖）",
            value=bool(st.get("cookie_autoimport", True)),
            outer=C["surface"], command=self._toggle_autoimport)
        self.sw_autoimport.pack(fill="x", pady=(U(10), 0))

        # ---------------- 动作条 ----------------
        # 进度条要横跨动作条左侧（设计稿里就是 fill 满的）。放进 action_left 的话
        # action_left 是 side="left" 的 hug 容器，进度条只会缩成一小截。
        self.bar = Bar(self.action, height=6, outer=C["canvas"])
        self.bar.pack(side="left", fill="x", expand=True, padx=(0, U(14)))
        self.btn_go = Btn(self.action_right, "开始下载", kind="green",
                          icon_name="play", command=self.start, height=44)
        self.btn_go.pack(side="right")
        self.btn_stop = Btn(self.action_right, "取消", icon_name="pause",
                            command=self.cancel, height=44)
        self.btn_stop.pack(side="right", padx=(0, 10))
        self._sync_browser()
        # 浏览器是 StringVar，代码里改它（比如从设置恢复）不会触发 Select 的 command，
        # 补一条 trace 让提示文字始终跟着变
        self.browser.trace_add("write", lambda *a: self._sync_browser())
        self.on_show()

    def log(self, msg):
        """yt-dlp 的日志：留一份尾部，失败时能说清到底卡在哪一步。"""
        msg = str(msg or "").strip()
        if not msg:
            return
        self._log_tail.append(msg)
        if len(self._log_tail) > 40:
            del self._log_tail[:-40]
        if not self._busy:
            return
        low = msg.lower()
        if any(k in low for k in ("destination", "merging", "fragment",
                                  "downloading", "下载", "已下载")):
            self.action_hint(_one_line(msg, 72), C["magenta"])

    # ---------------- 状态同步 ----------------
    def on_show(self):
        self._sync_browser()
        self._sync_actions()

    def _short_dir(self):
        return _short_path(self.outdir.get())

    def _sync_fmthint(self, _v=None):
        """说明当前选的输出格式要不要重编码、代价多大。

        不说清楚的话，用户选了 AV1 才发现要等很久，会以为程序卡死了。
        """
        if not hasattr(self, "lbl_fmthint"):
            return
        c = (self.container.get() or "auto").strip()
        disp = dict(CONTAINER_CHOICES).get(c, c)
        if c in ("", "auto"):
            txt, fg = "原始格式：只换封装、不重编码，最快且画质无损。", C["green"]
        elif c in ("mp4", "mkv", "webm"):
            txt = "%s：下载时直接换封装，不重编码，速度几乎不变。" % disp
            fg = C["green"]
        elif c.startswith("aonly-"):
            txt = "%s：下完转成音频文件（丢弃画面）。" % disp
            fg = C["ink2"]
        else:
            txt = ("%s：需要**重新编码**，耗时可能比下载本身还长"
                   "（HEVC / AV1 尤其慢）。" % disp)
            fg = C["ink3"]
        self.lbl_fmthint.configure(text=txt, fg=fg)

    def _sync_labels(self, _v=None):
        self.var_dir.set(self._short_dir())
        self._sync_preview()
        self._sync_fmthint()

    # ---------------- 文件名模板 ----------------
    def _tpl(self):
        return (self.var_tpl.get() or "").strip()

    def _toggle_vars(self):
        """展开/收起「更多变量」。默认收起，避免下载页内容过长要滚动。"""
        self._vars_open = not self._vars_open
        if self._vars_open:
            self.frm_vars.pack(fill="x", pady=(U(6), 0))
        else:
            self.frm_vars.pack_forget()

    def _fill_tpl_from_parsed(self):
        """解析成功后，把视频原本的文件名填进「文件名」框。

        填的是**真实标题的字面文本**，不是 `%(title)s` 这种占位符 ——
        占位符对用户没有信息量（他要看的就是"这文件到底叫什么"），
        而真实标题一眼就能读懂，想改直接在这行字上改，改坏了也能立刻看出来。

        只在框为空时自动填：用户已经手动写过模板就别覆盖，
        否则每解析一个链接就把人家写好的命名规则冲掉（批量下载时尤其烦人）。
        """
        if not self._parsed:
            return
        title = (self._parsed.get("title") or "").strip()
        if not title:
            return
        if self._tpl():
            return                      # 用户已有模板，尊重它
        # Windows 文件名非法字符先替掉，否则填进去就是一串下划线，
        # 用户还得自己再删。
        safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title).rstrip(". ")
        if not safe:
            return
        self.var_tpl.set(safe)
        self.action_hint("已填入原文件名，可直接修改或点下方命名方案", C["green"])

    def reset_tpl(self):
        self.var_tpl.set("")
        self.action_hint("已恢复默认命名（标题 + 视频ID）", C["green"])

    def _insert_token(self, tok):
        """点变量按钮 = 切换该占位符。

        * 模板里没有  -> 插入
        * 模板里已经有了 -> 删掉它

        做成"开关"而不是"只加不删"：拼模板时经常要反复调整，
        想去掉某个占位符时，连点两下就行，不必再手动退格删一长串。
        之前只加不删，连点同一个会得到 `%(title)s%(title)s` —— 既没意义，
        看着还像坏了。
        """
        try:
            cur = self._tpl()
            if tok in cur:
                # 移除（含它前后紧邻的连接符，免得留下 "A - " 这种空壳）
                new = cur.replace(tok, "", 1)
                new = re.sub(r"[-_]\s*$", "", new)      # 尾部连接符
                new = re.sub(r"^\s*[-_]\s*", "", new)   # 头部连接符
                new = re.sub(r"\s{2,}", " ", new).strip()
                self.var_tpl.set(new)
                # 光标留在删除处，这样连续点删/点加时顺序是可控的
                # （原来一律推到末尾，用户点两下就没法把变量插回中间了）。
                try:
                    self.inp_tpl.entry().icursor(min(len(new), max(0, cur.index(tok))))
                except Exception:
                    pass
                self.action_hint("已移除 %s" % tok, C["ink3"])
                return
            ent = self.inp_tpl.entry()
            pos = ent.index("insert")
            self.var_tpl.set(cur[:pos] + tok + cur[pos:])
            ent.icursor(pos + len(tok))
            self.action_hint("已加入 %s" % tok, C["green"])
        except Exception:
            self.var_tpl.set(self._tpl() + tok)

    def apply_preset(self, tpl):
        """套用一套命名预设：**替换**原有模板，而不是往上叠。

        原来点变量按钮是在光标处插入，连点几个就拼成一长串
        `%(title)s%(uploader)s%(id)s` —— 用户想要的是"换一套命名"，
        不是"把几个占位符糊在一起"。所以点预设直接覆盖，点一次一个结果。
        """
        self.var_tpl.set(tpl)
        self.action_hint("已套用命名方案：%s" % (
            dict((t, d) for t, d in TPL_PRESETS).get(tpl, "自定义")), C["green"])

    def _sync_preview(self):
        """按当前模板 + 已解析信息算出最终文件名，显示给用户看。

        没有解析过也显示：那时候用占位信息，意思是"模板长这样"，
        免得用户以为预览坏了。
        """
        if not hasattr(self, "lbl_prev"):
            return
        ext = "mp3" if self.fmt.get() == "audio" else (
            "m4a" if self.fmt.get() == "audio" else
            (self.container.get() if self.container.get() != "auto" else "mp4"))
        try:
            name = preview_filename(self._tpl(), self._parsed, ext=ext)
        except Exception as e:
            self.lbl_prev.configure(text="模板有问题：%s" % e, fg=C["err"])
            return
        if not self._parsed:
            self.lbl_prev.configure(
                text="预览（先点【解析】可显示真实标题）：%s" % name, fg=C["ink3"])
        else:
            self.lbl_prev.configure(text="将保存为：%s" % name, fg=C["ink2"])

    def _sync_browser(self, _v=None):
        """Cookie 来源可用性提示 —— 说清楚"为什么没生效"，而不是默默失败。

        这里会真的检查 cookie 文件里有没有 YouTube 登录凭据：Get cookies.txt
        LOCALLY 这类插件默认只导出**当前页面**的 cookie，用户在别的标签页导出，
        文件格式完全合法但里面没有 YouTube —— 这种情况以前只表现为"登录失效"，
        根本看不出是自己导错了页面。
        """
        if ytdlp_module() is None:
            # yt-dlp 没装/没打进来 —— 直接说清楚，别让"点了没反应"
            self.lbl_cookie.configure(
                text="未内置 yt-dlp，下载功能不可用；运行 start.bat 可自动安装",
                fg=C["err"])
            return
        b = self.browser.get()
        js_name, _ = find_js_runtime()
        bits = ["yt-dlp %s" % ytdlp_version(),
                ("JS 运行时 %s" % js_name) if js_name else "缺少 JS 运行时（YouTube 只给低清）"]
        authed = False
        cur = cookie_txt()
        if cur:
            # 有文件时浏览器那条路会被跳过，直说，免得用户以为选的浏览器没生效
            _files = collect_cookie_files()
            _cur = os.path.basename(cur)
            if len(_files) > 1:
                bits.append("已加载 %s（另有 %d 个较旧的 .txt 未用）"
                            % (_cur, len(_files) - 1))
            else:
                bits.append("已加载 %s" % _cur)
            # 关键：文件"能用"和"有登录态"是两件事，分开说
            authed, why = youtube_auth_state(cur)
            bits.append(why)
            if b != "none":
                bits.append("（已优先用文件，忽略浏览器 Cookie）")
        else:
            if b != "none":
                bits.append("浏览器 %s %s" % (
                    dict(BROWSER_CHOICES).get(b, b),
                    "已就绪" if browser_cookie_installed(b) else
                    "未检测到（可能没装或没登录）"))
            # 文件明明在、却读不到 —— 最容易让人以为程序有 bug，把原因点出来
            prob = cookie_problem()
            if prob:
                bits.append(prob)
            else:
                bits.append("%s 里没有 cookie 文件"
                            % os.path.basename(cookie_dir() or "cookies"))
        self.lbl_cookie.configure(text=" · ".join(bits),
                                  fg=C["green"] if authed
                                  else (C["ink3"] if b == "none" else C["ink2"]))

    def open_cookie_help(self):
        """【怎么用插件导出 Cookie】—— 一步步说清楚，别让用户自己摸索。

        为什么需要单独的入口：Chrome 127+ 之后浏览器数据库被 app-bound 加密
        且独占锁定，"从浏览器更新"在不关浏览器时必然失败。Get cookies.txt LOCALLY
        这类插件走的是浏览器扩展自己的权限，**不用关浏览器**，是现在唯一顺畅的路。
        """
        d = ensure_cookie_dir() or cookie_dir()
        _alert("用插件导出 Cookie（不用关浏览器）",
               "Chrome 127 之后浏览器数据库被加密并独占锁定，"
               "不关浏览器就读不出来。用插件导出是现在最省事的做法：\n\n"
               "1. 在 Chrome 扩展商店安装「Get cookies.txt LOCALLY」\n"
               "2. **先登录 YouTube**（打开 youtube.com 确认是已登录状态）\n"
               "3. 停留在 **youtube.com 的页面**上（不要在别的网站点导出）\n"
               "4. 点浏览器工具栏里的插件图标 → Export\n"
               "5. 把下载到的 cookies.txt 拖进下面这个文件夹：\n\n"
               "     %s\n\n"
               "（导出后程序会自动识别，不需要再点别的按钮）\n\n"
               "之后 Cookie 过期了，重复第 2~5 步即可。"
               % d,
               )
        try:
            os.startfile(d)
        except Exception:
            pass

    def _sync_actions(self):
        self.btn_go.set_enabled(not self._busy)
        self.btn_stop.set_enabled(self._busy)
        if not self._busy:
            self.bar.set(0)

    # ---------------- 输入 ----------------
    def paste(self):
        """【粘贴】按钮：和 Ctrl+V 走同一条路。"""
        urls = self._clip_urls()
        if not urls:
            _alert("剪贴板", "剪贴板里没有找到链接。\n\n"
                             "请复制以 http:// 或 https:// 开头的地址。")
            return
        self._take_urls(urls)

    def _clip_urls(self):
        try:
            txt = self.root_clipboard()
        except Exception:
            txt = ""
        return extract_urls(txt)

    def _take_urls(self, urls):
        """把提取到的链接填进输入框。

        **只填不解析** —— 解析必须由用户点【解析】触发，这样他能先看到
        贴进来的是什么、必要时改一下，而不是被直接拖进一次网络请求。
        多于一个链接时其余排成队列，下完一个自动下一个。
        """
        self.url.set(urls[0])
        self._queue = list(urls[1:])
        if len(urls) > 1:
            self.action_hint("已粘贴 %d 个链接 · 点【解析】开始，其余 %d 个随后依次下载"
                             % (len(urls), len(urls) - 1), C["magenta"])
        else:
            self.action_hint("已粘贴链接 · 确认后点【解析】", C["green"])
        try:
            self.inp.entry().focus_set()      # 光标进框，贴进来的内容一眼可见
        except Exception:
            pass

    def _paste_hotkey(self):
        """输入框里按 Ctrl+V：默认粘贴会塞进整段文案，这里改成
        提取其中的链接并自动解析。"""
        urls = self._clip_urls()
        if urls:
            self._take_urls(urls)

    def _focus_in_entry(self, event=None):
        """当前焦点是否落在某个可编辑的 tk.Entry 上。

        优先用事件对象自带的 widget（最可靠：它就是被点/被按键的那个控件）。
        退化到 focus_get()/focus_displayof()。

        为什么不用"模拟点击后读 Entry.get()"那种间接判断：焦点在某些环境
        （窗口未映射、焦点在别的进程、远程桌面）下 focus_get() 会返回根窗口
        而不是真实控件，这时"焦点不在输入框"会被误判，于是又开始劫持粘贴。
        """
        if event is not None:
            w = getattr(event, "widget", None)
            if isinstance(w, tk.Entry):
                return True
            # 事件可能落在 Entry 的父 Canvas 上（Input 就是 Canvas 包 Entry）
            if w is not None:
                try:
                    kids = w.winfo_children()
                except Exception:
                    kids = ()
                if any(isinstance(k, tk.Entry) for k in kids):
                    return True
        for getter in (self.focus_get, self.focus_displayof):
            try:
                w = getter()
            except Exception:
                w = None
            if isinstance(w, tk.Entry):
                return True
        return False

    def _on_ctrl_v(self, _e=None):
        """在整个窗口范围内接管 Ctrl+V。

        之前只绑在输入框上，**焦点不在输入框时按 Ctrl+V 就毫无反应** ——
        用户看到的就是"只能点按钮粘贴"。绑到 toplevel 上（控件自己的
        bindtags 里含所属 toplevel），焦点在页面任何位置都能贴。

        但不能无条件劫持：焦点在**别的**输入框里（比如命名模板、保存位置
        自定义框）时，必须把事件还给控件自己，否则会出现
        "在模板框里按 Ctrl+V，文字却跑进链接栏" —— 粘贴的内容被送去了
        完全无关的字段，用户只看到链接栏莫名其妙多出一段字。
        """
        if self.app.current != "download":
            return None            # 别的页面放行，交给控件自己的粘贴
        if self._focus_in_entry(_e):
            return None            # 焦点在可编辑输入框 → 放行，Entry 自己粘贴
        self._paste_hotkey()
        return "break"

    def root_clipboard(self):
        return self.winfo_toplevel().clipboard_get()

    def pick_dir(self):
        d = filedialog.askdirectory(parent=self.winfo_toplevel(),
                                    title="选择保存位置",
                                    initialdir=self.outdir.get() or _HERE)
        if d:
            self.outdir.set(d)
            self._sync_labels()

    def open_outdir(self):
        d = self.outdir.get() or os.path.join(_HERE, DOWNLOAD_REL)
        ensure_dir(d)
        try:
            os.startfile(d)
        except Exception as e:
            _alert("打不开", "%s\n\n%s" % (d, e), error=True)

    def open_cookie_dir(self):
        ensure_cookie_dir()
        try:
            os.startfile(cookie_dir())
        except Exception as e:
            _alert("打不开", "%s\n\n%s" % (cookie_dir(), e), error=True)

    def refresh_cookie(self):
        """【从浏览器更新 Cookie】—— 直接从本机浏览器现取登录态。

        手工导出的 cookies.txt 会过期、也会在换账号后失效，于是又变成
        "Sign in to confirm you're not a bot"。这里从浏览器只读取一次并写回
        Cookie 文件夹，之后解析/下载都直接用新的。

        浏览器要选哪个用上面那个【浏览器】下拉；一个浏览器有多个用户配置
        （Default / Profile 1 …）时会让用户挑，选错会导出另一个账号的登录态。

        Chrome 127+ 在**运行时**数据库被 app-bound 加密且独占锁定，外部读不出来
        （yt-dlp 官方同样失败，见其 issue #7271）。所以检测到锁住时不再让用户
        干等，直接把人引到"插件导出"那条不需要关浏览器的路上。
        """
        browser = self.browser.get()
        if browser == "none":
            _alert("选一种方式拿 Cookie",
                   "Chrome 127 之后，浏览器开着时读不到它的 Cookie 数据库"
                   "（官方 yt-dlp 也一样）。所以有两种做法：\n\n"
                   "【推荐】用浏览器插件导出，不用关浏览器：\n"
                   "  安装「Get cookies.txt LOCALLY」→ 在已登录的 YouTube 页面"
                   " 点导出 → 把 cookies.txt 放进 Cookie 文件夹。\n"
                   "  点下面的【插件导出说明】看详细步骤。\n\n"
                   "【或者】完全退出浏览器后，再点【从浏览器更新】。")
            return
        profs = list_browser_profiles(browser)
        if not profs:
            _alert("没找到浏览器",
                   "在这台电脑上找不到 %s 的用户数据目录。\n\n"
                   "如果浏览器装在非默认位置，可能读不到；"
                   "也可以继续用【更改…】里已有的 cookies.txt。"
                   % dict(BROWSER_CHOICES).get(browser, browser))
            return
        if len(profs) > 1:
            d = self._ask_profile(browser, profs)
            if not d:
                return
            profile = d
        else:
            profile = profs[0]

        if getattr(self, "_cookie_busy", False):
            return
        self._cookie_busy = True
        self.lbl_cookie.configure(text="正在从 %s 读取 Cookie…"
                                  % dict(BROWSER_CHOICES).get(browser, browser),
                                  fg=C["link"])
        app = self.app

        def worker():
            try:
                path, n = refresh_cookies_from_browser(browser, profile)
                app.q.put(("cookie_updated", (True, path, n, profile, browser)))
            except Exception as e:
                app.q.put(("cookie_updated", (False, str(e), 0, profile, browser)))

        threading.Thread(target=worker, daemon=True).start()

    def _ask_profile(self, browser, profs):
        """多用户配置时弹一个小选择框。返回选中的名字，取消返回 ""。"""
        from tkinter import simpledialog
        d = simpledialog.askstring(
            "选择浏览器用户配置",
            "%s 里有多个用户配置，请输入要用哪一个：\n\n%s\n\n"
            "（一般用 Default；不确定就选它）"
            % (dict(BROWSER_CHOICES).get(browser, browser),
               "   ".join(profs)),
            initialvalue=profs[0], parent=self.winfo_toplevel())
        if d is None:
            return ""
        d = d.strip()
        return d if d in profs else profs[0]

    def on_cookie_updated(self, payload):
        """worker 回来的结果。必须在主线程更新界面。"""
        self._cookie_busy = False
        ok, info, n, profile, browser = payload
        name = dict(BROWSER_CHOICES).get(browser, browser)
        if ok:
            self.lbl_cookie.configure(
                text="已更新：%d 条 Cookie（来自 %s / %s）"
                     % (n, name, profile), fg=C["green"])
            self.action_hint("Cookie 已更新（%d 条）" % n, C["green"])
        else:
            self.lbl_cookie.configure(text="更新失败：%s" % _one_line(info, 60),
                                      fg=C["err"])
            _alert("更新 Cookie 失败", str(info), error=True)
        self._sync_browser()

    def _toggle_autoimport(self, val=None):
        """开关「自动导入 Cookie」—— 立刻生效，不用重启。

        开启时顺带立刻扫一遍：用户往往刚导出一个文件就来回拨开关，
        让他不用干等下一次轮询。
        """
        on = bool(self.sw_autoimport.get()) if val is None else bool(val)
        try:
            self.app.settings["cookie_autoimport"] = on
            save_json(SETTINGS_FILE, self.app.settings)
        except Exception:
            pass
        imp = getattr(self.app, "_cookie_importer", None)
        if not on:
            if imp is not None:
                try:
                    imp.stop()
                except Exception:
                    pass
                self.app._cookie_importer = None
            self.action_hint("已关闭 Cookie 自动导入", C["ink3"])
            return
        started = _start_cookie_watcher(self.app)
        if started is None:
            self.action_hint("自动导入启动失败（缺少 cookiescan 模块）", C["err"])
            return
        self.action_hint("已开启 Cookie 自动导入", C["green"])

    def on_cookie_autoimported(self, paths):
        """后台监视器自动导入 Cookie 后的界面反馈。

        只在**确实带来可用的 YouTube 登录态**时才出声打扰：监视器也可能捡到
        别人站点的 cookie 或格式合法的空文件，那些静默收下即可，不必弹窗。
        """
        if not paths:
            return
        good = []
        for p in paths:
            try:
                authed, why = youtube_auth_state(p)
            except Exception:
                continue
            if authed:
                good.append((os.path.basename(p), why))
        self._sync_browser()
        if not good:
            return
        names = "、".join(n for n, _ in good[:2])
        self.action_hint("已自动导入 Cookie（%s）" % names, C["green"])

    def pick_cookie_dir(self):
        """改 Cookie 识别目录 —— 之前这里只有一个只读的路径 + 【打开】，
        用户想换个位置（比如放到同步盘上）没有任何入口。"""
        d = filedialog.askdirectory(parent=self.winfo_toplevel(),
                                    title="选择 Cookie 文件夹",
                                    initialdir=cookie_dir())
        if not d:
            return
        if not set_cookie_dir(d):
            _alert("这个文件夹不能用", "创建或写入失败：\n\n%s" % d, error=True)
            return
        self.app.settings["cookie_dir"] = cookie_dir()
        if not save_json(SETTINGS_FILE, self.app.settings):
            self.action_hint("目录已切换，但设置没能保存（下次启动会回到默认）",
                             C["err"])
        self.var_cookie.set(_short_path(cookie_dir()))
        self._sync_browser()
        self.action_hint("Cookie 目录已改为 %s" % cookie_dir(), C["green"])

    # ---------------- 解析 ----------------
    def parse(self):
        if self._busy:
            return
        url = self.url.get().strip()
        if not url:
            _alert("缺少链接", "请先粘贴视频链接。")
            return
        # 所有 Tk 变量必须在主线程读完再交给 worker：
        # StringVar.get() 会碰 Tcl 解释器，从子线程调会抛
        # "RuntimeError: main thread is not in main loop"。
        browser = self.browser.get()
        self._busy = True
        self._sync_actions()
        self.action_hint("正在解析…", C["magenta"])
        self.lbl_info.configure(text="", fg=C["ink3"])
        app = self.app
        # 解析也给一份 job，解析期间点【取消】才能立刻打断（实测能省十几秒等待）
        job = _Job()
        self._probe_job = job

        def worker():
            lines = []

            def lg(m):
                lines.append(str(m))
                app.q.put(("dl_log", str(m)))

            ok, info = ytdlp_probe(url, browser, lg, job=job)
            if getattr(job, "canceled", False):
                ok, info = False, "已取消"
            app.q.put(("dl_parsed", (ok, info, "\n".join(lines[-3:]))))

        threading.Thread(target=worker, daemon=True).start()

    def on_parsed(self, ok, info, tail):
        self._busy = False
        self._sync_actions()
        self._probe_job = None            # 解析结束，job 用完即弃
        if not ok:
            if str(info) == "已取消":
                self.action_hint("已取消解析", C["ink3"])
                self.lbl_info.configure(text="", fg=C["ink3"])
                return
            self.action_hint("解析失败", C["err"])
            self.lbl_info.configure(text=str(info)[:160], fg=C["err"])
            # 失败也落盘：把链接 / JS 运行时 / Cookie / 原因 / yt-dlp 末几行
            # 一次性写进 lyric-maker.log，用户不用再开 --dl-test 就能看到全貌。
            logfile_write("")
            logfile_write("=== 视频解析失败 ===")
            try:
                logfile_write("链接   : %s" % (self.url.get().strip() or "(空)"))
            except Exception:
                pass
            js, jsp = find_js_runtime()
            logfile_write("JS 运行时 : %s"
                          % (("%s -> %s" % (js, jsp)) if js
                             else "未找到（n 挑战解不开 → YouTube 不给格式）"))
            ecf = cookie_file()
            try:
                cdir = cookie_scan_dirs()[0] if cookie_scan_dirs() else "cookies/"
            except Exception:
                cdir = "cookies/"
            logfile_write("Cookie  : %s" % (ecf if ecf else "无（%s 里没有可用的 .txt）" % cdir))
            logfile_write("原因   : %s" % str(info))
            if tail:
                logfile_write("--- yt-dlp 末几行 ---")
                for ln in tail.splitlines():
                    logfile_write("  " + ln[:170])
            if tail:
                self.action_hint(_one_line(tail, 70), C["err"])
            return
        self._parsed = info
        self._fill_tpl_from_parsed()
        self._sync_preview()      # 解析完立刻显示真实标题下的文件名
        # 把下拉换成这个链接**真实可用**的清晰度（用户要的"自动识别视频格式"）
        opts = ytdlp_format_choices(info)
        self.sel_fmt.set_values([k for k, _ in opts], labeler=fmt_label)
        if info.get("_playlist"):
            n = info.get("count") or 0
            self.lbl_info.configure(
                text="播放列表：%s · 共 %d 个视频%s"
                     % (info.get("title") or "未命名", n,
                        (" · %s" % info["uploader"]) if info.get("uploader") else ""),
                fg=C["ink2"])
        else:
            dur = info.get("duration")
            hs = ytdlp_heights(info)
            self.lbl_info.configure(
                text="%s · 时长 %s · 可用清晰度 %s%s"
                     % (info.get("title") or "未命名",
                        _eta(dur) if dur else "未知",
                        "、".join(("%dp" % h) for h in hs) or "仅音频",
                        (" · %s" % info["uploader"]) if info.get("uploader") else ""),
                fg=C["ink2"])
        self.action_hint("解析完成，可以开始下载", C["green"])

    # ---------------- 下载 ----------------
    def start(self):
        if self._busy:
            return
        url = self.url.get().strip()
        if not url:
            _alert("缺少链接", "请先粘贴视频链接并解析。")
            return
        out = self.outdir.get().strip() or os.path.join(_HERE, DOWNLOAD_REL)
        self.outdir.set(out)
        # 默认目录 out/downloads 第一次用时并不存在，这里直接建掉 ——
        # 否则新用户第一次点【开始下载】就撞上一个莫名其妙的"目录不存在"，
        # 批量队列更是在第一个就断了。
        if not ensure_dir(out):
            _alert("保存位置不可用", "无法创建保存位置：\n\n%s" % out, error=True)
            return
        # 命名模板顺手存下来：下次打开还是这套，不用重填
        try:
            self.app.settings["dl_name_tpl"] = self._tpl()
            save_json(SETTINGS_FILE, self.app.settings)
        except Exception:
            pass
        self._job = _Job()
        self._busy = True
        self._sync_actions()
        self.action_hint("准备下载…", C["magenta"])
        app = self.app
        job = self._job
        # 同上：Tk 变量一律在主线程取好，worker 只用纯 Python 值
        kw = dict(quality=self.fmt.get(), container=self.container.get(),
                  browser=self.browser.get(), subs=bool(self.sw_subs.get()),
                  thumb=bool(self.sw_thumb.get()),
                  playlist=bool(self._parsed and self._parsed.get("_playlist")),
                  template=self._tpl())

        def worker():
            def lg(m):
                app.q.put(("dl_log", str(m)))

            ok, msg = ytdlp_download(url, out, log=lg, job=job, **kw)
            app.q.put(("dl_done", (ok, msg, job.filename or "")))

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def cancel(self):
        # 解析和下载是两段各自独立的等待，两个都要能取消：
        # 解析阶段 _job 是 None、只有 _probe_job；下载阶段反过来。
        hit = False
        if self._job and not self._job.canceled:
            self._job.cancel()
            hit = True
        _pj = getattr(self, "_probe_job", None)
        if _pj is not None and not _pj.canceled:
            _pj.cancel()
            hit = True
        if hit:
            self.action_hint("正在取消…（不等它跑完）", C["magenta"])

    def on_progress(self):
        """由 App.tick 定时调用，把 job 的进度刷到进度条上。"""
        j = self._job
        if j is None or not self._busy:
            return
        self.bar.set(j.ratio())
        if j.speed:
            self.action_hint(
                "下载中 %s · %s/s · 剩余 %s"
                % ("%.0f%%" % (j.ratio() * 100),
                   human(int(j.speed)), _eta(j.eta)), C["magenta"])
        elif j.total:
            self.action_hint("下载中 %.0f%%" % (j.ratio() * 100), C["magenta"])

    def on_done(self, ok, msg, filename):
        self._busy = False
        self._job = None
        self._sync_actions()
        # 队列里还有链接就接着下 —— 用户一次粘一串链接时不用守着点
        if self._queue and ok:
            nxt = self._queue.pop(0)
            self.url.set(nxt)
            self.action_hint("还剩 %d 个，继续下一个…" % len(self._queue),
                             C["magenta"])
            self.start()
            return
        if ok:
            self.bar.set(1.0)
            self.action_hint("下载完成 · %s" % os.path.basename(filename or ""),
                             C["green"])
        else:
            self.action_hint(msg, C["err"])
            if msg != "已取消":
                tail = _dedup_tail(self._log_tail)
                _alert("下载未完成",
                       msg + (("\n\n最后几步：\n" + tail) if tail else ""),
                       error=True)
                # 失败也落盘：把原因 + yt-dlp 末几步写进 lyric-maker.log
                logfile_write("")
                logfile_write("=== 视频下载失败 ===")
                logfile_write("原因   : %s" % msg)
                if tail:
                    logfile_write("--- 末几步 ---")
                    for ln in tail.splitlines():
                        logfile_write("  " + ln[:170])


# =====================================================================
# 页面四：格式转换
# =====================================================================

class PageConvert(Page):
    """视频 / 音频格式互转。

    多选文件排队转换，ffmpeg 跑在后台线程里。进度来自 ffmpeg 往 stderr 打的
    `time=` —— 它没有进度回调接口，这是唯一稳定可得的进度来源。
    """

    def __init__(self, master, app):
        Page.__init__(self, master, app, "格式转换",
                      "视频与音频互转，常用的容器和编码格式都在下拉里")
        st = app.settings
        self.fmt = tk.StringVar(value=st.get("conv_fmt") or "mp4")
        self.vq = tk.StringVar(value=st.get("conv_vq") or "23")
        self.arate = tk.StringVar(value=st.get("conv_arate") or "192k")
        self.outdir = tk.StringVar(
            value=st.get("conv_dir") or os.path.join(_HERE, CONVERT_REL))
        self._paths = []          # 待转换文件完整路径，与列表框的行一一对应
        self._busy = False
        self._job = None
        self._thread = None
        self._proc_holder = []    # 正在跑的 ffmpeg 进程，取消时直接 kill
        self._t0 = 0.0
        self._idx = 0
        self._total_files = 0
        self._log_open = False

        Btn(self.head_right, "打开输出目录", icon_name="folder",
            command=self.open_outdir, outer=C["canvas"], height=36).pack(side="right")
        Btn(self.head_right, "清空列表", icon_name="trash",
            command=self.clear_files, outer=C["canvas"],
            height=36).pack(side="right", padx=(0, 10))

        self.sc = Scroll(self.content, bg=C["canvas"])
        self.sc.pack(fill="both", expand=True)
        inner = self.sc.inner

        # ---------------- ① 待转换文件 ----------------
        c1 = Panel(inner, pad=18)
        c1.pack(fill="x", pady=(0, 12))
        self._section_head(c1.body, "upload", "待转换文件", "可一次选多个")
        # 空 / 非空两套界面占同一个位置：空的时候给一块虚线区（点它就能选文件），
        # 比一个空荡荡的深色框友好得多，也让页面不至于从上黑到下。
        self.slot = tk.Frame(c1.body, bg=C["surface"])
        self.slot.pack(fill="x")
        self.empty_wrap = tk.Frame(self.slot, bg=C["surface"])
        self.drop = DropZone(self.empty_wrap, on_click=self.pick_files,
                             outer=C["surface"], height=U(150),
                             title="添加视频或音频，或", link="选择文件",
                             caption="支持一次选多个，会按顺序依次转换")
        self.drop.pack(fill="x")
        self.list_wrap = tk.Frame(self.slot, bg=C["surface"])
        # 深色界面里 tk.Listbox 是唯一既能多选、又不用自己画滚动条的控件。
        # 底色用 surface_in 而不是 log_bg —— 后者接近纯黑，一整块贴在这儿
        # 会把整页压得很闷；选中色也换成主色，扫一眼就知道选了哪几个。
        self.lst = tk.Listbox(self.list_wrap, bg=C["surface_in"], fg=C["ink2"],
                              relief="flat", bd=0, highlightthickness=0,
                              font=FN(9), height=5, activestyle="none",
                              selectmode=tk.EXTENDED,
                              selectbackground=C["primary"],
                              selectforeground=C["ink"])
        self.lst.pack(fill="x")
        r1 = tk.Frame(self.list_wrap, bg=C["surface"])
        r1.pack(fill="x", pady=(U(10), 0))
        Btn(r1, "添加文件…", kind="blue", icon_name="upload",
            command=self.pick_files, outer=C["surface"],
            height=38).pack(side="left")
        Btn(r1, "移除选中", icon_name="x", command=self.remove_selected,
            outer=C["surface"], height=38).pack(side="left", padx=(U(10), 0))
        self.lbl_count = label(c1.body, "", size=8, color=C["ink3"])
        self.lbl_count.pack(fill="x", pady=(10, 0))

        # ---------------- ② 转换设置 ----------------
        c2 = Panel(inner, pad=18)
        c2.pack(fill="x", pady=(0, 12))
        self._section_head(c2.body, "swap", "转换设置")
        g = tk.Frame(c2.body, bg=C["surface"])
        g.pack(fill="x")
        g.columnconfigure(0, weight=0, minsize=U(190))
        g.columnconfigure(1, weight=0, minsize=U(190))
        g.columnconfigure(2, weight=1)
        self.sel_fmt = Select(g, CONVERT_FORMAT_KEYS, textvariable=self.fmt,
                              command=self._sync_mode, outer=C["surface"],
                              labeler=convert_fmt_label, height=40)
        self.sel_fmt.grid(row=0, column=0, sticky="ew")
        # 质量/码率两个下拉占同一格：目标格式决定哪一个生效（另一个收起来），
        # 这样两者的取值互不干扰 —— 共用一个下拉会把 "23" 和 "192k" 混在一起。
        qbox = tk.Frame(g, bg=C["surface"])
        qbox.grid(row=0, column=1, sticky="ew", padx=(U(12), 0))
        self.sel_vq = Select(qbox, [c[0] for c in CONVERT_VQUALITY],
                             textvariable=self.vq, outer=C["surface"],
                             labeler=lambda v: dict(CONVERT_VQUALITY)[v],
                             height=40)
        self.sel_ar = Select(qbox, [c[0] for c in CONVERT_ARATE],
                             textvariable=self.arate, outer=C["surface"],
                             labeler=lambda v: dict(CONVERT_ARATE)[v],
                             height=40)
        holder = tk.Frame(g, bg=C["surface"])
        holder.grid(row=0, column=2, sticky="ew", padx=(U(12), 0))
        field_label(holder, "保存位置").pack(anchor="w", pady=(0, 7))
        rr = tk.Frame(holder, bg=C["surface"])
        rr.pack(fill="x")
        # 同下载页：用只读 Input 而不是 Label，圆角和高度才跟旁边的控件一致，
        # 而且用户还能选中复制路径。
        self.var_dir = tk.StringVar(value=self._short_dir())
        self.inp_dir = Input(rr, textvariable=self.var_dir, height=40,
                             outer=C["surface"], fill=C["surface_in"])
        self.inp_dir.pack(side="left", fill="x", expand=True)
        self.inp_dir.entry().configure(state="readonly")
        Btn(rr, "选择…", command=self.pick_dir, outer=C["surface"],
            height=40).pack(side="left", padx=(U(10), 0))
        sw = tk.Frame(c2.body, bg=C["surface"])
        sw.pack(fill="x", pady=(U(14), 0))
        self.sw_remux = Switch(sw, "仅转封装（不重编码，极快）", value=False,
                               command=self._sync_mode, outer=C["surface"])
        self.sw_remux.pack(side="left")
        self.sw_over = Switch(sw, "覆盖已存在的文件", value=True,
                              outer=C["surface"])
        self.sw_over.pack(side="left", padx=(U(22), 0))

        # 输出预览：把"这套设置到底会产出什么"摊成几个彩色胶囊。
        # 一来信息量比一行灰字大，二来给整页添几块颜色，不至于一片深蓝。
        pv = tk.Frame(c2.body, bg=C["surface"])
        pv.pack(fill="x", pady=(U(16), 0))
        field_label(pv, "输出预览").pack(anchor="w", pady=(0, 8))
        self.chips = tk.Frame(pv, bg=C["surface"])
        self.chips.pack(fill="x")
        self.lbl_hint = label(c2.body, "", size=8, color=C["ink3"])
        self.lbl_hint.pack(fill="x", pady=(U(12), 0))

        # ---------------- ③ 运行日志（默认收起） ----------------
        # 日志区是整页最暗的一块（终端底色），摊开着会占掉半屏、把页面压黑。
        # 默认折起来只留一条标题，点开始转换时才自动展开。
        lp = RoundedLogPanel(inner, head_bg=C["surface"], body_bg=C["log_bg"],
                            outline=C["line"])
        self.lp = lp
        lp.pack(fill="x", pady=(0, 12))
        lh = tk.Frame(lp.body, bg=C["surface"], height=46)
        lh.pack(fill="x")
        lh.pack_propagate(False)
        lh.configure(cursor="hand2")
        self.log_chev = tk.Canvas(lh, width=22, height=22, bg=C["surface"],
                                  highlightthickness=0, bd=0)
        self.log_chev.pack(side="left", padx=(16, 6))
        self.log_chev.bind("<Configure>", lambda e: self._draw_chev())
        tk.Label(lh, text="运行日志", bg=C["surface"], fg=C["ink2"],
                 font=F(9, True)).pack(side="left")
        self.lbl_log_sub = tk.Label(lh, text="", bg=C["surface"],
                                    fg=C["ink5"], font=F(7))
        self.lbl_log_sub.pack(side="right", padx=16)

        def _bind_all(w, cb):
            """整行都要能点：只绑在容器上，点到子控件时不会触发。"""
            w.bind("<Button-1>", cb, add="+")
            for ch in w.winfo_children():
                _bind_all(ch, cb)

        _bind_all(lh, lambda e: self._toggle_log())
        # 日志正文由 RoundedLogPanel 提供（自带圆角底 + 叠在上面的 Text）
        self.txt = lp.txt
        self.txt.tag_configure("cur", foreground=C["link"])
        self.txt.tag_configure("err", foreground=C["err"])
        self.txt.configure(state="disabled")

        # ---------------- 动作条 ----------------
        self.bar = Bar(self.action, height=6, outer=C["canvas"])
        self.bar.pack(side="left", fill="x", expand=True, padx=(0, U(14)))
        self.btn_go = Btn(self.action_right, "开始转换", kind="green",
                          icon_name="swap", command=self.start, height=44)
        self.btn_go.pack(side="right")
        self.btn_stop = Btn(self.action_right, "取消", icon_name="pause",
                            command=self.cancel, height=44)
        self.btn_stop.pack(side="right", padx=(0, 10))
        self._sync_mode()
        self._sync_count()
        self._set_log_open(False)
        self._sync_actions()
        self.on_show()

    # ---------------- 界面小件 ----------------
    def _section_head(self, parent, icon_name, text, hint=""):
        """分区标题：小圆角图标底 + 标题。

        原来是满色蓝→粉渐变的大徽标，那是全应用最跳的一处：同一页里
        「添加文件」等按钮才是实心蓝，标题反而比按钮更抢眼，深色底上像贴了两块糖。
        现在改成低饱和的 surface_hi 底 + 主色线性图标 —— 保留"能扫到分区"的作用，
        但和周围控件是同一个语气。圆角用控件档 R_INPUT，和输入框/下拉一致。
        """
        row = tk.Frame(parent, bg=C["surface"])
        row.pack(fill="x", pady=(0, 12))
        badge = tk.Canvas(row, width=U(26), height=U(26), bg=C["surface"],
                          highlightthickness=0, bd=0)
        badge.pack(side="left")

        def draw(_e=None):
            w, h = badge.winfo_width(), badge.winfo_height()
            if w < 4 or h < 4:
                return
            badge.delete("all")
            # 纯色底 + 1px 描边，不铺渐变：四角天然就是圆的，也就不需要敲角补画，
            # 从根上避免了"渐变盖掉圆角"那个老问题。
            rrect(badge, 0, 0, w, h, R_INPUT, C["surface_in"], C["line"], 1)
            icon(badge, icon_name, w / 2.0, h / 2.0, 14, C["primary"])

        badge.bind("<Configure>", draw)
        tk.Label(row, text=text, bg=C["surface"], fg=C["ink"],
                 font=F(10, True)).pack(side="left", padx=(10, 0))
        if hint:
            tk.Label(row, text=hint, bg=C["surface"], fg=C["ink4"],
                     font=F(8)).pack(side="right")
        return row

    def _draw_chev(self):
        c = self.log_chev
        c.delete("all")
        w, h = c.winfo_width(), c.winfo_height()
        if w < 4 or h < 4:
            return
        icon(c, "right" if not self._log_open else "down", w / 2.0, h / 2.0,
             13, C["ink3"])

    def _set_log_open(self, ok):
        self._log_open = bool(ok)
        self.lp.set_open(bool(ok))
        self.lbl_log_sub.configure(
            text="失败原因也会写入 lyric-maker.log" if ok else "点这里展开")
        self._draw_chev()

    def _toggle_log(self):
        self._set_log_open(not self._log_open)

    def drop_files(self, data):
        """拖进来的文件（装了 tkinterdnd2 才走得到这儿；没装时用【选择文件】）。"""
        if self._busy:
            return
        try:
            items = self.winfo_toplevel().tk.splitlist(data)
        except Exception:
            items = [str(data).strip("{}")]
        self._add_files([it for it in items if it])

    # ---------------- 日志 ----------------
    def log(self, msg):
        msg = str(msg or "")
        if not msg.strip():
            return
        now = (time.time() - self._t0) if self._t0 else 0.0
        stamp = "%02d:%05.2f" % (int(now) // 60, now % 60)
        t = self.txt
        t.configure(state="normal")
        try:
            t.insert("end", stamp + "   ", "cur")
        except Exception:
            pass
        t.insert("end", msg + "\n")
        t.see("end")
        t.configure(state="disabled")
        # ffmpeg 每一帧都往 stderr 打一行，全量落盘会把 lyric-maker.log 冲爆，
        # 这里只把"看起来在报错"的行写进去，失败的完整上下文另外单独写一块。
        low = msg.lower()
        if any(k in low for k in ("error", "invalid", "unsupported", "failed",
                                  "no such", "not found", "incorrect")) or \
           any(k in msg for k in ("错误", "失败", "无法", "不支持")):
            logfile_write(msg)

    # ---------------- 文件列表 ----------------
    def pick_files(self):
        if self._busy:
            return
        types = [("音视频文件",
                  "*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.ts *.flv *.wmv *.mpg "
                  "*.mpeg *.3gp *.mp3 *.m4a *.aac *.flac *.wav *.ogg *.opus "
                  "*.wma *.aiff *.aif"),
                 ("视频文件",
                  "*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.ts *.flv *.wmv *.mpg "
                  "*.mpeg *.3gp"),
                 ("音频文件",
                  "*.mp3 *.m4a *.aac *.flac *.wav *.ogg *.opus *.wma *.aiff *.aif"),
                 ("所有文件", "*.*")]
        try:
            paths = filedialog.askopenfilenames(parent=self.winfo_toplevel(),
                                                title="选择要转换的文件",
                                                filetypes=types)
        except Exception as e:
            _alert("打不开选择框", str(e), error=True)
            return
        self._add_files(list(paths or []))

    def _add_files(self, paths):
        n = 0
        for p in paths:
            if not p:
                continue
            ap = os.path.abspath(p)
            if ap in self._paths:
                continue
            self._paths.append(ap)
            self.lst.insert("end", os.path.basename(ap))
            n += 1
        self._sync_count()
        if n:
            self.action_hint("已添加 %d 个文件" % n, C["green"])

    def remove_selected(self):
        if self._busy:
            return
        for i in reversed(list(self.lst.curselection())):
            try:
                self.lst.delete(i)
                del self._paths[i]
            except Exception:
                pass
        self._sync_count()

    def clear_files(self):
        if self._busy:
            return
        self.lst.delete(0, "end")
        self._paths = []
        self._sync_count()
        self.action_hint("已清空文件列表", C["ink3"])

    def _sync_count(self):
        n = len(self._paths)
        # 空列表时给虚线区，有文件时才换成列表 —— 两者占同一个 slot，互不打架
        if n:
            self.empty_wrap.pack_forget()
            self.list_wrap.pack(fill="x")
        else:
            self.list_wrap.pack_forget()
            self.empty_wrap.pack(fill="x")
        if not n:
            self.lbl_count.configure(text="还没选文件 · 点上面的区域，或【添加文件…】",
                                     fg=C["ink3"])
        else:
            self.lbl_count.configure(
                text="共 %d 个文件 · 会按顺序依次转换" % n, fg=C["ink2"])

    # ---------------- 输出目录 ----------------
    def _short_dir(self):
        return _short_path(self.outdir.get())

    def pick_dir(self):
        d = filedialog.askdirectory(parent=self.winfo_toplevel(),
                                    title="选择保存位置",
                                    initialdir=self.outdir.get() or _HERE)
        if d:
            self.outdir.set(d)
            self.var_dir.set(self._short_dir())

    def open_outdir(self):
        d = self.outdir.get() or os.path.join(_HERE, CONVERT_REL)
        ensure_dir(d)
        try:
            os.startfile(d)
        except Exception as e:
            _alert("打不开", "%s\n\n%s" % (d, e), error=True)

    # ---------------- 状态同步 ----------------
    def _sync_mode(self, _v=None):
        """目标格式决定"视频质量"还是"音频码率"生效；仅转封装时两者都不用。"""
        spec = CONVERT_FORMAT_MAP.get(self.fmt.get())
        kind = spec[2] if spec else "video"
        remux = bool(self.sw_remux.get())
        if kind == "audio":
            self.sel_vq.pack_forget()
            self.sel_ar.pack(fill="x", expand=True)
        else:
            self.sel_ar.pack_forget()
            self.sel_vq.pack(fill="x", expand=True)
        self.sel_vq.set_enabled(not remux)
        self.sel_ar.set_enabled(not remux)
        # ---- 输出预览胶囊：把当前设置摊开给人看 ----
        if remux:
            tags = [("直接复制流 · 不重编码", C["green_bg"], C["green"]),
                    ("容器 %s" % ((spec[3] if spec else "?").upper()),
                     C["bg_tint"], C["ink2"])]
            if kind == "audio":
                tags.append(("丢掉画面", C["magenta_bg"], C["magenta"]))
        elif kind == "video":
            tags = [("%s 画面" % CONVERT_CODEC_LABEL.get(spec[4], spec[4]),
                     C["primary_dk"], C["ink"]),
                    ("%s 声音" % CONVERT_CODEC_LABEL.get(spec[5], spec[5]),
                     C["bg_tint"], C["ink2"])]
            if spec[4] in CONVERT_VQ_FIXED:
                # 中间格式不提供画质档位，显示"不可调"比显示一个假的 CRF 更诚实
                tags.append(("画质不可调", C["green_bg"], C["green"]))
            else:
                tags.append(("CRF %s" % self.vq.get(), C["magenta_bg"], C["ink2"]))
            if spec[5] in CONVERT_LOSSLESS_AUDIO:
                tags.append(("音频无损", C["green_bg"], C["green"]))
            else:
                tags.append(("音频 %s" % self.arate.get(), C["bg_tint"], C["ink2"]))
            if spec[4] == "dvvideo":
                tags.append(("缩放到 720×480", C["magenta_bg"], C["ink2"]))
        else:
            tags = [("%s 音频" % CONVERT_CODEC_LABEL.get(spec[5], spec[5]),
                     C["primary_dk"], C["ink"]),
                    ("丢掉画面", C["magenta_bg"], C["magenta"])]
            if spec[5] in CONVERT_LOSSLESS_AUDIO:
                tags.append(("无损 · 不设码率", C["green_bg"], C["green"]))
            elif spec[5] == "libopencore_amrnb":
                tags.append(("8000 Hz 单声道", C["magenta_bg"], C["ink2"]))
            else:
                tags.append(("码率 %s" % self.arate.get(), C["bg_tint"], C["ink2"]))
        for w in self.chips.winfo_children():
            w.destroy()
        for i, (txt, fill, fg) in enumerate(tags):
            Chip(self.chips, text=txt, fill=fill, fg=fg, height=30, size=8,
                 outer=C["surface"]).pack(
                side="left", padx=((0 if i == 0 else U(8)), 0))
        if remux:
            self.lbl_hint.configure(
                text="仅转封装：不重新编码，速度极快且画质无损，"
                     "但要求原文件的编码被目标容器支持（不支持时会失败）。")
        elif kind == "audio":
            self.lbl_hint.configure(
                text="转成音频：丢掉画面只留声音。FLAC / WAV / ALAC / WavPack 为无损，不设码率；"
                     "AMR 固定 8000 Hz 单声道。")
        elif spec[4] in CONVERT_VQ_FIXED:
            self.lbl_hint.configure(
                text="中间格式：不压缩、画质无损，画质档位不适用——适合做剪辑代理，"
                     "但体积会明显变大。")
        else:
            self.lbl_hint.configure(
                text="转成视频：画面和声音都重新编码。CRF 越小画质越好、体积越大。"
                 "（H.265 / VP9 / AV1 同码率画质更好但更慢，兼容性也略差。）")

    def _sync_actions(self):
        self.btn_go.set_enabled(not self._busy)
        self.btn_stop.set_enabled(self._busy)
        if not self._busy:
            self.bar.set(0)

    def on_show(self):
        self._sync_mode()
        self._sync_actions()

    # ---------------- 转换 ----------------
    def start(self):
        if self._busy:
            return
        if not self._paths:
            _alert("没有文件", "请先点【添加文件…】选择要转换的文件。")
            return
        out = self.outdir.get().strip() or os.path.join(_HERE, CONVERT_REL)
        self.outdir.set(out)
        # 默认目录 out/converts 第一次用时并不存在，这里直接建掉 ——
        # 否则新用户第一次点【开始转换】就撞上一个莫名其妙的"目录不存在"。
        if not ensure_dir(out):
            _alert("保存位置不可用", "无法创建保存位置：\n\n%s" % out, error=True)
            return
        try:
            ff = find_ffmpeg()
        except Exception as e:
            _alert("找不到 ffmpeg", "%s" % e, error=True)
            return
        # Tk 变量一律在主线程读完再交给 worker：StringVar.get() 会碰 Tcl
        # 解释器，从子线程调会抛 "main thread is not in main loop"。
        fmt = self.fmt.get()
        vq = self.vq.get()
        ar = self.arate.get()
        remux = bool(self.sw_remux.get())
        overwrite = bool(self.sw_over.get())
        files = list(self._paths)
        # 记住这一套选择，下次打开还是它
        st = self.app.settings
        st["conv_fmt"], st["conv_vq"] = fmt, vq
        st["conv_arate"], st["conv_dir"] = ar, out
        save_json(SETTINGS_FILE, st)

        self._job = _Job()
        self._proc_holder = []
        self._idx = 0
        self._total_files = len(files)
        self._t0 = time.time()
        self._busy = True
        self._sync_actions()
        # 开始跑了才把日志区展开：平时收着，页面不会一直挂着一大块近黑色
        self._set_log_open(True)
        self.action_hint("准备转换…", C["magenta"])
        app, job = self.app, self._job

        def worker():
            done = fail = 0
            msgs = []
            for src in files:
                if job.canceled:
                    break
                name = os.path.basename(src)
                spec = CONVERT_FORMAT_MAP.get(fmt)
                ext = spec[3] if spec else fmt
                base = os.path.splitext(name)[0]
                dst = os.path.join(out, base + "." + ext)
                # 源和目标撞名（比如同目录 mp4 -> mp4）：必须改名，
                # 让 ffmpeg 边读边写会把原文件覆盖成半个坏文件。
                if os.path.abspath(dst) == os.path.abspath(src):
                    dst = os.path.join(out, base + "_converted." + ext)
                if os.path.exists(dst) and not overwrite:
                    app.q.put(("conv_file", (True, src, dst, "已存在，跳过")))
                    done += 1
                    continue
                app.q.put(("conv_log", "▶ %s  →  %s" % (name, os.path.basename(dst))))
                # 探测流构成：带封面的 mp3 里的 png 封面不算画面，
                # 若不排除，转视频格式时 ffmpeg 会给封面也套上视频编码器而报 -22。
                try:
                    _has_video = bool(real_video_streams(probe_streams(ff, src)))
                except Exception:
                    _has_video = None      # 探测失败就按原样处理，不阻断
                if _has_video is False and spec[2] == "video" and not remux:
                    app.q.put(("conv_log", "  源文件没有画面（可能是纯音频或只带封面），"
                                            "将生成黑底画面承载音频"))
                cmd = build_convert_cmd(ff, src, dst, fmt, vq, ar, remux,
                                        has_video=_has_video)
                job.filename = name
                job.done = 0.0
                job.total = probe_duration(ff, src) or 0.0
                ok, why = run_ffmpeg(cmd, job=job, proc_holder=self._proc_holder,
                                     log=lambda m: app.q.put(("conv_log", m)))
                if job.canceled:
                    # 半截的输出文件留着只会让人以为转好了，删掉
                    try:
                        if os.path.exists(dst) and os.path.getsize(dst) == 0:
                            os.remove(dst)
                    except Exception:
                        pass
                    break
                if ok:
                    done += 1
                    app.q.put(("conv_file", (True, src, dst, "完成")))
                else:
                    fail += 1
                    msgs.append("%s：%s" % (name, why))
                    app.q.put(("conv_file", (False, src, dst, why)))
            app.q.put(("conv_done", {"done": done, "fail": fail,
                                     "canceled": job.canceled,
                                     "msgs": msgs[-5:]}))

        self._thread = threading.Thread(target=worker, daemon=True)
        self._thread.start()

    def cancel(self):
        if not self._busy:
            return
        if self._job and not self._job.canceled:
            self._job.cancel()
        # 光置 canceled 不够：ffmpeg 要等当前帧编码完才会去看它，
        # 直接 kill 进程才能立刻停。
        for p in self._proc_holder:
            try:
                p.kill()
            except Exception:
                pass
        self.action_hint("正在取消…", C["magenta"])

    # ---------------- 后台事件回调（由 App.tick 派发） ----------------
    def on_conv_log(self, payload):
        if self._busy:
            self.log(payload)

    def on_progress(self):
        j = self._job
        if j is None or not self._busy:
            return
        n = self._total_files or 1
        r = min(1.0, j.done / float(j.total)) if j.total else 0.0
        # 进度条走的是"整批"的量：单个文件内部的百分比会让条反复回退
        self.bar.set((self._idx + r) / float(n))
        name = _one_line(j.filename or "", 26)
        if j.total:
            self.action_hint("转换中 %s · %.0f%% · 剩余 %s"
                             % (name, r * 100,
                                _eta(max(0.0, j.total - j.done))), C["magenta"])
        else:
            self.action_hint("转换中 %s" % name, C["magenta"])

    def on_file_done(self, ok, src, out, msg):
        name = os.path.basename(src or "")
        self._idx += 1
        if ok:
            self.log("  完成：%s" % (("（已存在，跳过）" if msg == "已存在，跳过" else "")
                                     + name))
            return
        self.log("  失败：%s · %s" % (name, msg))
        self._set_log_open(True)      # 失败了就把日志摊开，别让人去猜
        # 失败也落盘：文件和原因一次性写进 lyric-maker.log
        logfile_write("")
        logfile_write("=== 格式转换失败 ===")
        logfile_write("文件   : %s" % (src or ""))
        logfile_write("目标   : %s" % (out or ""))
        logfile_write("原因   : %s" % msg)

    def on_all_done(self, summary):
        self._busy = False
        self._job = None
        self._proc_holder = []
        self._sync_actions()
        summary = summary or {}
        done = summary.get("done", 0)
        fail = summary.get("fail", 0)
        if summary.get("canceled"):
            self.bar.set(0)
            self.action_hint("已取消 · 已完成 %d 个" % done, C["err"])
            self.log("已取消")
            return
        self.bar.set(1.0)
        if fail == 0:
            self.action_hint("转换完成 · %d 个文件已输出到 %s"
                             % (done, self._short_dir()), C["green"])
            self.log("全部完成：%d 个文件" % done)
        else:
            self.action_hint("完成 %d 个，失败 %d 个" % (done, fail), C["err"])
            self.log("完成 %d 个，失败 %d 个" % (done, fail))
            detail = "\n".join(summary.get("msgs") or [])
            _alert("有文件没转成功",
                   "完成 %d 个，失败 %d 个：\n\n%s\n\n"
                   "详细原因已写入 lyric-maker.log" % (done, fail, detail),
                   error=True)


# =====================================================================
# 页面五：历史记录
# =====================================================================

class PageHistory(Page):
    def __init__(self, master, app):
        Page.__init__(self, master, app, "历史记录", "每次生成都会记一条", with_action=False)
        self.var_search = tk.StringVar()
        self.filter = "all"
        inp = Input(self.head_right, textvariable=self.var_search, width=240,
                    placeholder="搜索文件名或歌词…", icon_name="search",
                    outer=C["canvas"], height=36)
        inp.pack(side="right")
        inp.entry().bind("<KeyRelease>", lambda e: self.reload())
        Btn(self.head_right, "清空记录", command=self.clear_all,
            outer=C["canvas"], height=36).pack(side="right", padx=(10, 0))

        self.seg = Segmented(self.content, [("all", "全部"), ("done", "已完成"),
                                            ("fail", "失败")],
                             value="all", command=self._on_filter, outer=C["canvas"])
        self.seg.pack(anchor="w", pady=(0, 12))
        self.sc = Scroll(self.content, bg=C["canvas"])
        self.sc.pack(fill="both", expand=True)
        self.reload()

    def on_show(self):
        self.reload()

    def _on_filter(self, v):
        self.filter = v
        self.reload()

    def clear_all(self):
        if not messagebox.askyesno("清空记录", "确定清空全部历史记录？只删记录，不动字幕文件。"):
            return
        if not save_json(HISTORY_FILE, []):
            messagebox.showerror(
                "清空失败",
                "无法写入历史记录文件。\n"
                "可能是 data/ 目录只读或被占用，请检查权限后重试。")
            return
        self.reload()

    def _list_signature(self, items, kw):
        """"这一页要画的内容"指纹。

        数据、筛选、搜索词三者任一变化才会重画；全都没变就说明屏幕上那份
        列表还是对的，直接跳过整段重建。
        """
        return (self.filter, kw, len(items),
                tuple(str(r.get("out")) for r in items[:60]))

    def reload(self):
        if not hasattr(self, "sc"):
            return
        items = history_load()
        kw = self.var_search.get().strip().lower()
        if kw:
            items = [r for r in items
                     if kw in str(r.get("media", "")).lower()
                     or kw in str(r.get("out", "")).lower()]
        if self.filter == "done":
            items = [r for r in items if r.get("status") == "done"]
        elif self.filter == "fail":
            items = [r for r in items if r.get("status") != "done"]
        # 列表已经建好、且"数据 + 筛选条件"都没变，就别重画。
        # 这一页每次最多重建 60 行控件（实测 ~38ms），而切页就会调一次；
        # 历史记录只在跑完任务时才变，所以绝大多数切页都能整段跳过。
        sig = self._list_signature(items, kw)
        if getattr(self, "_list_sig", None) == sig and self.sc.inner.winfo_children():
            return
        self._list_sig = sig
        for c in self.sc.inner.winfo_children():
            c.destroy()
        if not items:
            self._empty(kw or self.filter != "all")
            return
        card = Panel(self.sc.inner, pad=12)
        card.pack(fill="both", expand=True)
        head = tk.Frame(card.body, bg=C["surface"])
        head.pack(fill="x", padx=(18, 18), pady=(6, 8))
        cols = (("状态", 66), ("文件", 0), ("模式", 100), ("时长", 74),
                ("输出", 74), ("耗时", 74), ("处理时间", 112), ("操作", 56))
        for i, (txt, w) in enumerate(cols):
            head.grid_columnconfigure(i, weight=1 if w == 0 else 0, minsize=w)
            tk.Label(head, text=txt, bg=C["surface"], fg=C["ink4"], font=F(7),
                     anchor="w").grid(row=0, column=i, sticky="ew")
        shown = items[:60]
        for rec in shown:
            self._row(card.body, rec, cols)
        # 只渲染前 60 条，所以这里报**实际渲染**的条数；
        # 原来按总数报，用户有 187 条时界面写"显示 187 条"却只有 60 行。
        _more = ("，仅列出最近 60 条" if len(items) > len(shown) else "")
        tk.Label(card.body, text="显示 %d 条%s · 最多保留最近 200 条 · 记录存在 %s"
                 % (len(shown), _more, DATA_DIR),
                 bg=C["surface"], fg=C["ink4"], font=F(7), anchor="w").pack(
            fill="x", padx=(18, 18), pady=(10, 4))

    def _empty(self, filtered):
        box = Panel(self.sc.inner, pad=30)
        box.pack(fill="both", expand=True)
        tk.Label(box.body, text="没有匹配的记录" if filtered else "还没有生成记录",
                 bg=C["surface"], fg=C["ink2"], font=F(10, True)).pack(pady=(40, 8))
        tk.Label(box.body, text=("换个关键词试试" if filtered else
                                 "到【生成歌词】页跑一次，这里就会自动记下来"),
                 bg=C["surface"], fg=C["ink4"], font=F(8)).pack()
        tk.Label(box.body, text="记录保存在 %s" % DATA_DIR,
                 bg=C["surface"], fg=C["ink5"], font=F(7)).pack(pady=(14, 0))

    def _row(self, parent, rec, cols):
        ok = rec.get("status") == "done"
        row = Panel(parent, radius=16, pad=16, fill=C["surface_in"], outer=C["surface"])
        row.pack(fill="x", pady=(0, 6))
        g = row.body
        for i, (_t, w) in enumerate(cols):
            g.grid_columnconfigure(i, weight=1 if w == 0 else 0,
                                   minsize=w if w else 0)
        tk.Label(g, text="已完成" if ok else "失败", bg=C["surface_in"],
                 fg=C["green"] if ok else C["magenta"], font=F(8), anchor="w").grid(
            row=0, column=0, sticky="w")
        tk.Label(g, text=os.path.basename(str(rec.get("media", ""))) or "—",
                 bg=C["surface_in"], fg=C["ink"], font=F(9, True), anchor="w").grid(
            row=0, column=1, sticky="ew")
        tk.Label(g, text=self._mode_text(rec), bg=C["surface_in"], fg=C["ink2"],
                 font=F(8), anchor="w").grid(row=0, column=2, sticky="ew")
        tk.Label(g, text=fmt_dur(rec.get("duration")), bg=C["surface_in"],
                 fg=C["ink2"], font=FN(8), anchor="w").grid(row=0, column=3, sticky="ew")
        tk.Label(g, text=("." + str(rec.get("fmt") or "")) if ok else "未产出",
                 bg=C["surface_in"], fg=C["ink2"] if ok else C["ink4"], font=FN(8),
                 anchor="w").grid(row=0, column=4, sticky="ew")
        tk.Label(g, text=fmt_cost(rec.get("elapsed")), bg=C["surface_in"], fg=C["ink2"],
                 font=FN(8), anchor="w").grid(row=0, column=5, sticky="ew")
        tk.Label(g, text=fmt_when(rec.get("ts", 0)), bg=C["surface_in"], fg=C["ink3"],
                 font=F(8), anchor="w").grid(row=0, column=6, sticky="ew")
        tk.Label(g, text="重跑" if ok else "重试", bg=C["surface_in"], fg=C["link"],
                 font=F(8, True), anchor="w", cursor="hand2").grid(
            row=0, column=7, sticky="w")
        for w in row.body.winfo_children():
            w.bind("<Button-1>", lambda e, r=rec: self._rerun(r))

    def _mode_text(self, rec):
        mode = rec.get("mode") or ""
        if mode == "align":
            return "对齐模式"
        if mode == "subtitle":
            src = str(rec.get("source") or "")
            if "轨" in src:
                return "字幕轨"
            if "OCR" in src or "ocr" in src:
                return "硬字幕 OCR"
            return "字幕"
        subs = rec.get("subs") or ""
        return SUBS_LABEL.get(subs, "自动听写")

    def _rerun(self, rec):
        self.app.pages["generate"].prefill(rec)
        self.app.select("generate")


# =====================================================================
# 页面六：设置
# =====================================================================

class PageSettings(Page):
    def __init__(self, master, app):
        Page.__init__(self, master, app, "设置", "这些值会作为新任务的默认参数")
        sc = Scroll(self.content, bg=C["canvas"])
        sc.pack(fill="both", expand=True)
        st = app.settings
        self.v = {
            "model": tk.StringVar(value=st.get("model") or default_model()),
            "language": tk.StringVar(value=st.get("language", "zh")),
            "fmt": tk.StringVar(value=st.get("fmt", "lrc")),
            "out_dir": tk.StringVar(value=st.get("out_dir", "")),
            "threads": tk.StringVar(value=str(st.get("threads", 4))),
            "subs": tk.StringVar(value=st.get("subs", "auto")),
            "fps": tk.StringVar(value=str(st.get("subs_fps", "2.0"))),
            "score": tk.StringVar(value=str(st.get("min_score", "0.5"))),
            "ffmpeg": tk.StringVar(value=st.get("ffmpeg", "")),
        }
        pad = sc.inner

        c1 = Panel(pad, pad=22)
        c1.pack(fill="x", pady=(0, 16))
        self._card_title(c1.body, "默认参数")
        g1 = self._grid(c1.body)
        self._put(g1, 0, 0, "默认模型", lambda p: Select(
            p, list(MODEL_REPOS.keys()), textvariable=self.v["model"],
            outer=C["surface"]))
        self._put(g1, 0, 1, "默认语言", lambda p: Select(
            p, list(LANG_LABEL.keys()), textvariable=self.v["language"],
            outer=C["surface"], labeler=lambda x: LANG_LABEL.get(x, x)))
        self._put(g1, 0, 2, "默认输出格式", lambda p: Select(
            p, ["lrc", "srt", "ass", "ssa", "vtt"], textvariable=self.v["fmt"],
            outer=C["surface"]))
        row = tk.Frame(c1.body, bg=C["surface"])
        row.pack(fill="x", pady=(14, 0))
        field_label(row, "默认输出目录（留空 = 源文件同目录）").pack(anchor="w", pady=(0, 7))
        rr = tk.Frame(row, bg=C["surface"])
        rr.pack(fill="x")
        Input(rr, textvariable=self.v["out_dir"], outer=C["surface"]).pack(
            side="left", fill="x", expand=True)
        Btn(rr, "浏览…", command=self.pick_dir, outer=C["surface"]).pack(
            side="left", padx=(10, 0))

        c2 = Panel(pad, pad=22)
        c2.pack(fill="x", pady=(0, 16))
        self._card_title(c2.body, "默认处理参数")
        g2 = self._grid(c2.body)
        self._put(g2, 0, 0, "CPU 线程数", lambda p: Input(
            p, textvariable=self.v["threads"], outer=C["surface"]))
        self._put(g2, 0, 1, "默认字幕来源", lambda p: Select(
            p, list(SUBS_LABEL.keys()), textvariable=self.v["subs"],
            outer=C["surface"], labeler=lambda x: SUBS_LABEL.get(x, x)))
        self._put(g2, 0, 2, "硬字幕粗扫帧率", lambda p: Input(
            p, textvariable=self.v["fps"], outer=C["surface"]))
        self._put(g2, 1, 0, "OCR 置信度阈值", lambda p: Input(
            p, textvariable=self.v["score"], outer=C["surface"]))
        sw = tk.Frame(c2.body, bg=C["surface"])
        sw.pack(fill="x", pady=(4, 0))
        self.sw = {}
        for key, txt in (("vad", "语音活动检测"), ("split", "按标点拆分过长行"),
                         ("merge", "合并过短的相邻段"), ("keep_wav", "保留中间 WAV")):
            s = Switch(sw, txt, value=bool(st.get(key)), outer=C["surface"])
            s.pack(anchor="w", pady=(0, 6))
            self.sw[key] = s

        c3 = Panel(pad, pad=22)
        c3.pack(fill="x", pady=(0, 16))
        self._card_title(c3.body, "路径与依赖")
        tk.Label(c3.body, text="模型目录", bg=C["surface"], fg=C["ink3"],
                 font=F(8)).pack(anchor="w", pady=(4, 7))
        tk.Label(c3.body, text=MODELS_DIR, bg=C["surface"], fg=C["ink2"],
                 font=FN(8), anchor="w").pack(fill="x")
        row = tk.Frame(c3.body, bg=C["surface"])
        row.pack(fill="x", pady=(14, 0))
        field_label(row, "ffmpeg 路径（留空则从 PATH 找）").pack(anchor="w", pady=(0, 7))
        rr = tk.Frame(row, bg=C["surface"])
        rr.pack(fill="x")
        Input(rr, textvariable=self.v["ffmpeg"], outer=C["surface"]).pack(
            side="left", fill="x", expand=True)
        Btn(rr, "浏览…", command=self.pick_ffmpeg, outer=C["surface"]).pack(
            side="left", padx=(10, 0))

        c4 = Panel(pad, pad=22)
        c4.pack(fill="x")
        self._card_title(c4.body, "数据")
        row = tk.Frame(c4.body, bg=C["surface"])
        row.pack(fill="x", pady=(14, 0))
        Btn(row, "重新运行自检", icon_name="check", command=self.run_selftest,
            outer=C["surface"]).pack(side="left")
        Btn(row, "打开程序目录", icon_name="folder", command=self.open_here,
            outer=C["surface"]).pack(side="left", padx=(10, 0))
        Btn(row, "清空历史记录", icon_name="trash", kind="danger", command=self.clear_hist,
            outer=C["surface"]).pack(side="left", padx=(10, 0))
        Btn(row, "清理无用缓存", icon_name="refresh", command=self.clear_cache_run,
            outer=C["surface"]).pack(side="left", padx=(10, 0))
        Btn(row, "占用体检", icon_name="search", command=self.run_dupscan,
            outer=C["surface"]).pack(side="left", padx=(10, 0))

        Btn(self.action_right, "恢复默认", command=self.reset, outer=C["canvas"]).pack(
            side="right", padx=(10, 0))
        Btn(self.action_right, "保存设置", icon_name="check", kind="green", bold=True,
            size=10, command=self.save, outer=C["canvas"], pad_x=22).pack(side="right")
        self.action_hint("保存后立即生效")

    def _card_title(self, parent, text):
        tk.Label(parent, text=text, bg=C["surface"], fg=C["ink"], font=F(10, True),
                 anchor="w").pack(fill="x")

    def _grid(self, parent):
        g = tk.Frame(parent, bg=C["surface"])
        g.pack(fill="x", pady=(14, 0))
        for i in range(3):
            g.columnconfigure(i, weight=1, uniform="set")
        return g

    def _put(self, g, r, c, label, make):
        f = tk.Frame(g, bg=C["surface"])
        field_label(f, label).pack(anchor="w", pady=(0, 7))
        make(f).pack(fill="x")
        f.grid(row=r, column=c, sticky="ew", padx=(0, 14) if c < 2 else 0,
               pady=(0, 14))

    def pick_dir(self):
        d = filedialog.askdirectory(title="默认输出目录")
        if d:
            self.v["out_dir"].set(d)

    def pick_ffmpeg(self):
        p = filedialog.askopenfilename(title="选择 ffmpeg.exe",
                                       filetypes=[("ffmpeg", "ffmpeg.exe"),
                                                  ("所有文件", "*.*")])
        if p:
            self.v["ffmpeg"].set(p)

    def open_here(self):
        try:
            os.startfile(_HERE)
        except Exception:
            _alert("程序目录", _HERE)

    def clear_hist(self):
        if not messagebox.askyesno("清空记录", "确定清空全部历史记录？"):
            return
        save_json(HISTORY_FILE, [])
        self.app.pages["history"].reload()
        self.action_hint("历史记录已清空", C["green"])

    def clear_cache_run(self):
        """清理无用缓存（__pycache__ 树、.tmp、out/_guitest、output）。

        先用干跑统计并让用户确认，再真正删除；删除时逐个跳过错误项，绝不中断。
        依赖目录（.deps*）和模型（models/）、运行时（runtime/）都不在清理范围内。
        """
        removed, _freed, items = clean_cache(dry=True)
        if removed == 0:
            self.action_hint("没有可清理的缓存", C["green"])
            return
        mb = sum((s or 0) for _, s in items) / 1048576.0
        names = "\n".join("  · " + os.path.relpath(p, _HERE) for p, _ in items)
        if not messagebox.askyesno(
                "清理无用缓存",
                "将清理以下 %d 项（约 %.1f MB）：\n%s\n\n"
                "都是可重建的临时/缓存文件，不影响功能。\n"
                "依赖目录（.deps*）、模型（models/）、运行时（runtime/）不会被删。"
                % (removed, mb, names)):
            return
        n, freed, _ = clean_cache(dry=False)
        self.action_hint("已清理 %d 项，释放 %.1f MB" % (n, freed / 1048576.0), C["green"])

    def run_dupscan(self):
        """磁盘占用体检：统计重复文件并按"能不能删"分类。

        项目里同时有两套依赖（_internal 给 exe、.deps* 给源码/打包），
        所以必然存在大量重复。这不是故障，但用户会以为是"占了双份空间"。
        这里给出分级结论，并明确告诉用户：哪部分能删、删了会失去什么。

        只做统计与说明，不删除任何东西 —— 要删的话用 dupscan.py --clean-safe，
        且默认仍是干跑。
        """
        self.action_hint("正在统计重复文件…", C["magenta"])
        app = self.app

        def worker():
            try:
                import dupscan
                groups, nfiles, nbytes = dupscan.scan(_HERE, min_size=1024)
                tally = {}
                for kind, same, sz in groups:
                    e = tally.setdefault(kind, [0, 0])
                    e[0] += 1
                    e[1] += sz * (len(same) - 1)
                msg = self._dupscan_summary(nfiles, nbytes, tally)
                app.q.put(("dupscan_done", msg))
            except Exception as e:
                app.q.put(("dupscan_done", "体检失败：%s: %s" % (type(e).__name__, e)))

        threading.Thread(target=worker, daemon=True).start()

    @staticmethod
    def _dupscan_summary(nfiles, nbytes, tally):
        def hb(n):
            for u in ("B", "KB", "MB", "GB"):
                if n < 1024 or u == "GB":
                    return "%.1f %s" % (n, u)
                n /= 1024.0
            return "%.1f GB" % n

        keep = tally.get("keep", [0, 0])
        safe = tally.get("safe", [0, 0])
        review = tally.get("review", [0, 0])
        lines = [
            "共 %d 个文件（%s）" % (nfiles, hb(nbytes)),
            "",
            "依赖重叠（保留）  %d 组，可省 %s" % (keep[0], hb(keep[1])),
            "  _internal 是 exe 运行必需，.deps* 是源码模式/重新打包必需；",
            "  两边装同一批库所以文件相同 —— 这是设计使然，不是故障。",
            "",
            "可清理（缓存）    %d 组，可省 %s" % (safe[0], hb(safe[1])),
            "  重复的 __pycache__ / 临时文件，删了会自动重建。",
        ]
        if review[0]:
            lines += ["", "需确认（业务区）  %d 组，可省 %s"
                      % (review[0], hb(review[1])),
                      "  业务文件重复，请自行判断后再处理。"]
        return "\n".join(lines)

    def on_dupscan_done(self, msg):
        """展示占用体检结果（弹窗 + 状态栏）。"""
        self.action_hint("占用体检完成", C["green"])
        _alert("磁盘占用体检", str(msg))

    def run_selftest(self):
        self.action_hint("正在运行自检…", C["magenta"])
        # Page 继承自 tk.Frame，窗口句柄在 _root 上（不是 root）。
        # 原来这里写成 self.root，点「重新运行自检」必定抛 AttributeError。
        self._root.after(80, lambda: self._selftest_go())

    def _selftest_go(self):
        try:
            rc = selftest()
            self.action_hint("自检通过" if rc == 0 else "自检有失败项，详见日志",
                             C["green"] if rc == 0 else C["err"])
        except Exception as e:
            self.action_hint("自检出错：%s" % e, C["err"])

    def collect(self):
        def num(k, cast, d):
            try:
                t = str(self.v[k].get()).strip()
                return cast(t) if t else d
            except Exception:
                return d

        return {
            "model": self.v["model"].get(),
            "language": self.v["language"].get(),
            "fmt": self.v["fmt"].get(),
            "out_dir": self.v["out_dir"].get().strip(),
            "threads": num("threads", int, 4),
            "subs": self.v["subs"].get(),
            "subs_fps": num("fps", float, 2.0),
            "min_score": num("score", float, 0.5),
            "ffmpeg": self.v["ffmpeg"].get().strip(),
            "vad": self.sw["vad"].get(),
            "split": self.sw["split"].get(),
            "merge": self.sw["merge"].get(),
            "keep_wav": self.sw["keep_wav"].get(),
        }

    def save(self, quiet=False):
        self.app.settings.update(self.collect())
        ok = save_json(SETTINGS_FILE, self.app.settings)
        self.app.pages["generate"].apply_settings(self.app.settings)
        self.app.refresh_storage()
        if not quiet:
            self.action_hint("已保存" if ok else "保存失败（目录只读？）",
                             C["green"] if ok else C["err"])

    def reset(self):
        self.app.settings.update(dict(DEFAULT_SETTINGS))
        save_json(SETTINGS_FILE, self.app.settings)
        for k, v in self.v.items():
            v.set(str(self.app.settings.get(k, "")))
        for key, s in self.sw.items():
            s.set(bool(self.app.settings.get(key)), notify=False)
        self.app.pages["generate"].apply_settings(self.app.settings)
        self.action_hint("已恢复默认", C["green"])
# =====================================================================
# 入口
# =====================================================================

def _start_cookie_watcher(app):
    """启动 Cookie 自动导入监视器（失败也不影响程序启动）。

    为什么需要：Chrome 127+ 之后读不了浏览器的 Cookie 库，只能靠
    「Get cookies.txt LOCALLY」这类插件导出；而插件走 chrome.downloads，
    **不支持指定目录**，文件落在 Chrome 的下载目录，还得用户手动拖进程序目录。
    这里后台盯着那些目录，发现新导出的就自动复制进来 —— 被监视的是普通文件夹，
    不受 Chrome 是否运行影响，所以"不用关浏览器"这条依然成立。
    """
    try:
        st = app.settings
        if not st.get("cookie_autoimport", True):
            return None
        import cookiescan
    except Exception:
        return None
    dest = cookie_dir()
    try:
        imp = cookiescan.CookieImporter(dest)
    except Exception:
        return None

    def on_import(got, _msg):
        # 运行在监视线程里，只能往队列里塞消息，由 App.tick 切回主线程刷新界面。
        try:
            app.q.put(("cookie_autoimported", [d for _s, d in got]))
        except Exception:
            pass

    imp.on_import = on_import
    try:
        imp.start()
        app._cookie_importer = imp      # 挂到 app 上，退出时能停掉
    except Exception:
        return None
    return imp


def launch(check_only=False, capture_png=None):
    if not _TK_OK:
        print("无法加载 tkinter，请改用命令行模式：python lyric_maker.py <文件>")
        return 1

    _init_fonts()
    # 主目录里先把历史记录的存储文件夹建好，用户一眼就能看到 data/ 在哪
    ensure_data_dir()
    ensure_cookie_dir()      # 手动导出的 cookies.txt 放这里
    _migrate_history()

    # 拖放是可选能力：装了 tkinterdnd2 就能把文件直接拖进窗口；
    # 没装也不影响点选（打包时没带上它，源码模式下可以用 pip 装上）。
    has_dnd = False
    try:
        from tkinterdnd2 import TkinterDnD, DND_FILES
        root = TkinterDnD.Tk()
        has_dnd = True
    except Exception:
        root = tk.Tk()

    root.title("歌词生成器 — 音乐/视频 生成歌词与字幕")
    # 2K/4K：先按显示器 DPI 算出缩放系数，后面所有像素字面量都靠它放大。
    # 必须在建任何控件之前调用。
    init_ui_scale(root)
    # 默认按屏幕大小开窗并居中：1320x880 是设计稿的舒适尺寸，
    # 小屏上退到屏幕内，最小尺寸由设计断点决定（1024x680，再小信息就不完整了）。
    # DPI 感知下 winfo_screenwidth() 已经是物理像素，所以这里要用 U() 把
    # "设计稿逻辑尺寸"换算成物理像素，窗口才能在高分屏上占比和低分屏一致。
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    w = max(U(1024), min(U(1320), sw - U(160)))
    h = max(U(680), min(U(880), sh - U(180)))
    root.geometry("%dx%d+%d+%d" % (w, h, max(0, (sw - w) // 2), max(0, (sh - h) // 3)))
    root.minsize(U(1024), U(680))
    root.configure(bg=C["canvas"])
    init_scrollbar_style(root)

    app = App(root, check_only=check_only)

    # Cookie 自动导入：后台盯着 Chrome 下载目录 / 系统下载 / 桌面，
    # 发现插件导出的 cookies.txt 就复制进 Cookie 文件夹并校验登录态。
    # 必须在 App 建好之后启动 —— 它要通过 app.queue 把结果送回主线程刷新界面。
    _start_cookie_watcher(app)

    if has_dnd:
        try:
            root.drop_target_register(DND_FILES)
            root.dnd_bind("<<Drop>>", lambda e: app.dispatch_drop(e.data))
        except Exception:
            pass

    round_native_window(root)
    try:
        # 映射之后再补一次（见 round_native_window 的说明）
        root.after(120, lambda: round_native_window(root))
    except Exception:
        pass
    # 摘掉原生标题栏，让顶栏成为最上层 UI。必须在窗口创建之后做（要拿到 HWND）。
    # 之后再补一次，防止系统重画时把样式写回去。
    try:
        frameless_top(root)
        root.after(120, lambda: frameless_top(root))
        # 反复设一次：frameless_top 触发 SWP_FRAMECHANGED 会让系统重画外框，
        # 那之后要把边框色重新染一遍，否则第一次设的会被重置掉
        set_window_border_color(root, C["canvas"])
        root.after(200, lambda: set_window_border_color(root, C["canvas"]))
    except Exception:
        pass

    if check_only or capture_png:
        # 自检模式：真正把四个页面都构建一遍（能抓出控件选项写错之类的运行时错误），
        # 但不进入事件循环，构建完即退出。
        root.update_idletasks()
        root.update()
        n = [0]

        def count(w):
            n[0] += 1
            for c in w.winfo_children():
                count(c)

        count(root)
        print("UI 构建成功: 窗口 %dx%d，控件 %d 个"
              % (root.winfo_width(), root.winfo_height(), n[0]))
        if capture_png:
            shots = [("generate", capture_png)]
            if "{page}" in capture_png:
                shots = [(k, capture_png.replace("{page}", k))
                         for k in ("generate", "models", "download", "convert",
                                   "history", "settings")]
            for key, path in shots:
                try:
                    app.select(key)
                    root.update_idletasks()
                    root.update()
                    round_native_window(root)
                    from PIL import ImageGrab
                    root.lift()
                    root.attributes("-topmost", True)
                    root.update()
                    round_native_window(root)
                    # 让 DWM 有时间把标题栏重画完，否则截到的是改色之前的那一帧
                    time.sleep(0.35)
                    root.update()
                    # 本进程不是 DPI 感知的：Tk 的坐标是"逻辑像素"，而 ImageGrab 按
                    # "物理像素"截屏。缩放不是 100% 的机器上直接用 Tk 坐标会整体偏移。
                    # 用"全屏实际宽度 / Tk 报告的逻辑宽度"求出真实缩放比再换算。
                    full = ImageGrab.grab()
                    scale = full.width / float(root.winfo_screenwidth())
                    del full
                    x, y = root.winfo_rootx(), root.winfo_rooty()
                    w, h = root.winfo_width(), root.winfo_height()
                    title = 0 if _FRAMELESS else int(32 * scale)   # 无标题栏时不用留高
                    bbox = (int(x * scale) - 4, int(y * scale) - title,
                            int((x + w) * scale) + 4, int((y + h) * scale) + 4)
                    img = ImageGrab.grab(bbox=bbox)
                    img.save(path)
                    print("界面截图: %s  (%dx%d 物理像素, 缩放比 %.2f, 页面 %s)"
                          % (path, img.width, img.height, scale, key))
                except Exception as e:
                    print("截图失败(%s): %s: %s" % (key, type(e).__name__, e))
        root.destroy()
        return 0

    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())





