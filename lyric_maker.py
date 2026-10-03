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
import difflib
import argparse
import subprocess
import tempfile

# --- 让 pip install --target .deps 安装的依赖可被导入 ---
_HERE = os.path.dirname(os.path.abspath(__file__))
_DEPS = os.path.join(_HERE, ".deps")
if os.path.isdir(_DEPS) and _DEPS not in sys.path:
    sys.path.insert(0, _DEPS)

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
    from shutil import which
    found = which("ffmpeg")
    if found:
        return found
    for c in [
        r"C:\Users\az_\.workbuddy\bin\ffmpeg.exe",
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
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", txt)
        if m:
            return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
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


_OPENCC = None


def to_simplified(text):
    """把繁体统一成简体。

    Whisper 在中文素材上会简繁混用（同一首歌里既有"局势"又有"局勢"），
    对歌词文件来说很难看。opencc 可用就转换，不可用则原样返回（不影响主流程）。
    """
    global _OPENCC
    if _OPENCC is None:
        try:
            from opencc import OpenCC
            _OPENCC = OpenCC("t2s")
        except Exception:
            _OPENCC = False
    if _OPENCC:
        try:
            return _OPENCC.convert(text)
        except Exception:
            return text
    return text


def clean_text(text):
    t = text.strip()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"^[\-–—\s]+", "", t)
    t = to_simplified(t)
    return t.strip()


def filter_segments(segs, drop_repeats=True, min_dur=0.30, log=print):
    """segs: [{'start','end','text','no_speech'}] -> 过滤后的列表"""
    out = []
    dropped_junk = dropped_rep = dropped_short = 0
    prev = None
    for s in segs:
        s["text"] = clean_text(s["text"])
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
    m = int(sec // 60)
    s = sec - m * 60
    return "[%02d:%05.2f]" % (m, s)


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
                "找不到模型 '%s'。\n  先运行: python get_model.py %s\n  （或用 --model 直接给模型目录）"
                % (name_or_path, name_or_path))
    if threads is None:
        threads = min(16, os.cpu_count() or 4)
    log("  加载模型: %s  (CPU int8, %d 线程)" % (os.path.basename(path), threads))
    t0 = time.time()
    model = WhisperModel(path, device="cpu", compute_type="int8", cpu_threads=threads)
    log("  模型就绪，用时 %.1fs" % (time.time() - t0))
    return model


def transcribe(media, model_name="medium", language="zh", out=None, prompt=None,
               vad=False, threads=None, start=None, duration=None,
               split_long_lines=True, merge=True, offset_ms=0,
               ffmpeg=None, keep_wav=False, lyrics_file=None, log=print):
    if not os.path.isfile(media):
        raise FileNotFoundError("输入文件不存在: %s" % media)

    ext = os.path.splitext(media)[1].lower()
    kind = "视频" if ext in VIDEO_EXT else ("音频" if ext in AUDIO_EXT else "未知类型")
    ff = find_ffmpeg(ffmpeg)
    total = probe_duration(ff, media)
    log("输入: %s  (%s, 时长 %s)" % (os.path.basename(media), kind,
                                    ("%.1fs" % total) if total else "未知"))

    tmpdir = tempfile.mkdtemp(prefix="lyric_", dir=_tmp_root())
    wav = os.path.join(tmpdir, "audio16k.wav")
    try:
        decode_to_wav(ff, media, wav, start=start, duration=duration, log=log)
        audio = read_wav_as_float32(wav)
        log("  音频载入: %.1f 秒（%d 采样点）" % (len(audio) / 16000.0, len(audio)))
        model = load_model(model_name, threads=threads, log=log)

        if prompt is None:
            prompt = "以下是普通话歌曲的歌词，请输出简体中文并带标点。"

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
            if out is None:
                out = os.path.splitext(media)[0] + ".lrc"
            tags = read_tags(media)
            write_lrc(out, segs, tags=tags, offset_ms=offset_ms)
            log("  已写出: %s  (%d 行，对齐模式)" % (out, len(segs)))
            return {"out": out, "segments": segs, "elapsed": elapsed,
                    "duration": total, "tags": tags, "info": info, "mode": "align"}

        segs = filter_segments(segs, log=log)
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
        if out is None:
            out = os.path.splitext(media)[0] + ".lrc"
        write_lrc(out, segs, tags=tags, offset_ms=offset_ms)
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


def _tmp_root():
    """临时目录优先放项目内，避免受限环境对 %TEMP% 的写入限制。"""
    d = os.path.join(_HERE, ".tmp")
    if os.path.isdir(d):
        return d
    return tempfile.gettempdir()


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

    print("\n=== 自检结果: %d 通过, %d 失败 ===" % (passed[0], failed[0]))
    return 0 if failed[0] == 0 else 1


# =====================================================================
# CLI
# =====================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="从音乐/视频自动生成标准 .lrc 歌词（本地识别，无需联网）")
    ap.add_argument("media", nargs="?", help="音乐或视频文件路径")
    ap.add_argument("--model", default="medium",
                    help="模型名(%s)或模型目录，默认 medium" % "/".join(MODEL_REPOS))
    ap.add_argument("--language", default="zh", help="语言，默认 zh；auto 为自动")
    ap.add_argument("--out", default=None, help="输出 .lrc 路径，默认与输入同名")
    ap.add_argument("--prompt", default=None, help="初始提示词，用于引导用词与标点")
    ap.add_argument("--lyrics-file", default=None,
                    help="已知歌词文本(.txt/.lrc)。给出后用【对齐模式】：文字用你的原文，"
                         "识别只提供时间锚点 —— 这是出高质量歌词的推荐方式")
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
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    if args.gui or not args.media:
        return run_gui()

    try:
        transcribe(args.media, model_name=args.model, language=args.language,
                   out=args.out, prompt=args.prompt, vad=args.vad, threads=args.threads,
                   start=args.start, duration=args.duration, offset_ms=args.offset,
                   split_long_lines=not args.no_split, merge=not args.no_merge,
                   ffmpeg=args.ffmpeg, keep_wav=args.keep_wav,
                   lyrics_file=args.lyrics_file)
        return 0
    except Exception as e:
        print("\n[错误] %s" % e, file=sys.stderr)
        return 1


# =====================================================================
# 图形界面（延迟导入，CLI 不受影响）
# =====================================================================
def run_gui():
    from lyric_gui import launch
    return launch()


if __name__ == "__main__":
    sys.exit(main())
