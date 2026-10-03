#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
get_model.py —— 下载 faster-whisper 模型（双源，带断点续传）

为什么是两个源：
  * huggingface.co 在本机被 DNS 污染（解析到 31.13.75.12），不可达。
  * hf-mirror.com 可达，但实测只有 ~0.5 MB/s，且大文件传输极不稳定
    （连续 4 段请求 3 段被远端强断：handshake timeout / WinError 10054）。
  * modelscope.cn 实测 ~2.6 MB/s 且连测 5/5 全成功，还支持 Range 断点续传。
  => 大模型默认走 ModelScope，小模型走 hf-mirror（huggingface_hub 处理）。
     ModelScope 路径用自写的续传下载器：每段失败就带着已下载字节数重试。

用法：
  python get_model.py                      列出全部模型与状态
  python get_model.py small                用 hf-mirror 下 small
  python get_model.py large-v3             用 ModelScope 下 large-v3（推荐，质量最好）
  python get_model.py large-v3 --source hf 强制走 hf-mirror
"""

import os
import sys
import time
import argparse
import urllib.parse
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEPS = os.path.join(_HERE, ".deps")
if os.path.isdir(_DEPS) and _DEPS not in sys.path:
    sys.path.insert(0, _DEPS)

MODELS_DIR = os.path.join(_HERE, "models")

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
        log("[错误] 缺少 huggingface_hub，请先运行 install_deps.bat")
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
        log("提示：改用 ModelScope 源试试 ->  python get_model.py large-v3")
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
    print("      python get_model.py small")
    print("      python get_model.py large-v3")


def download(name, force=False, source=None, log=print):
    if name not in ALL:
        print("[错误] 未知模型 '%s'。可选: %s" % (name, ", ".join(ALL)))
        return 1
    src, repo, size, desc = ALL[name]
    if source and source != "auto":
        if source == "modelscope" and name not in MS_REPOS:
            print("[错误] %s 没有 ModelScope 源，只有: %s" % (name, ", ".join(MS_REPOS)))
            return 1
        if source == "hf" and name not in HF_REPOS:
            print("[错误] %s 没有 hf-mirror 源，只有: %s" % (name, ", ".join(HF_REPOS)))
            return 1
        src = source

    target = os.path.join(MODELS_DIR, name)
    if is_downloaded(name) and not force:
        print("[跳过] %s 已存在（%s）。要重下加 --force。" % (name, human(dir_size(target))))
        return 0

    print("模型    : %s   (%s)" % (name, desc))
    print("预计大小: %s" % size)
    print("保存到  : %s" % target)
    print()

    t0 = time.time()
    if src == "modelscope":
        ok = download_from_modelscope(name, repo, target, log=log, alt=MS_ALT)
    else:
        ok = download_from_hf(name, repo, target, log=log)

    if not ok:
        return 1

    missing = [f for f in ("model.bin", "config.json", "tokenizer.json")
               if not os.path.isfile(os.path.join(target, f))]
    if missing:
        print("\n[警告] 缺少关键文件: %s" % ", ".join(missing))
        return 1
    print("\n[完成] %s -> %s  共 %s，用时 %.1f 分钟"
          % (name, target, human(dir_size(target)), (time.time() - t0) / 60.0))
    return 0


def main():
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


if __name__ == "__main__":
    sys.exit(main())
