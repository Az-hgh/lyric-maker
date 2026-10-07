# -*- coding: utf-8 -*-
r"""
dupscan.py —— 磁盘占用体检：找出项目里的重复文件，并按"能不能删"分类。

为什么需要：
项目里同时存在两套依赖 —— `_internal`（exe 运行时用，PyInstaller onedir）
和 `.deps*`（源码模式 / 重新打包用）。两边装的是同一批库，所以**必然有大量
逐字节重复的文件**。这不是"文件被复制错了"，而是两种运行方式各带一份依赖。

光看重复数没用，真正要回答的是：哪些是设计使然（别删）、哪些是真能回收。
所以本工具按"安全等级"分类，并且**默认只报告不删除**。

三个等级：
  * 保留    —— 依赖区（_internal/.deps*/models/runtime）。删了 exe 或源码模式就没了。
  * 可清理  —— 缓存与中间产物（__pycache__、临时目录、构建残留）。
              这些删了会自动重建，属于"本来就不该留"的东西。
  * 需确认  —— 业务区的重复文件（如多份备份的同名文件）。工具不自动删，
              只列出来让人自己判断。

用法：
  python dupscan.py              报告（默认，不删）
  python dupscan.py --clean-safe 只清理「可清理」等级（干跑）
  python dupscan.py --clean-safe --yes  真正删除
  python dupscan.py --min-size 1   只看 >= 1 字节的重复
  python dupscan.py --json out.json  额外输出 JSON
"""
import os
import sys
import json
import time
import shutil
import argparse
import hashlib
from collections import defaultdict

# --------------------------------------------------------------------------
# 分级规则
# --------------------------------------------------------------------------
# 依赖区：exe 运行必需(_internal) 或源码模式/打包必需(.deps*)。
# models 是语音模型，runtime 是 ffmpeg/node —— 都不是"重复"，单独保护。
DEPENDENCY_TOPS = {
    ".deps", ".deps-ocr", ".deps-ytdlp", ".deps-ejs", ".deps-build",
    "_internal", "models", "runtime", ".git",
}
# 可清理：可重建的缓存与中间产物。
SAFE_DIRS = ("__pycache__", ".tmp", ".pytest_cache", "build", "dist")
SAFE_NAMES = (".deps-build",)
SAFE_EXTS = (".pyc", ".pyo", ".log", ".tmp")

_BLOCK = 1 << 20


def sha256(path, chunk=_BLOCK):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def top_of(path, root):
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)
    return parts[0] if len(parts) > 1 else ""


def is_safe_cleanup(path, root):
    """是否是可安全清理的缓存/中间产物。"""
    parts = os.path.relpath(path, root).split(os.sep)
    for p in parts[:-1]:
        if p in SAFE_DIRS:
            return True
    name = parts[-1]
    if name in SAFE_NAMES:
        return True
    return name.lower().endswith(SAFE_EXTS)


def classify(path, root):
    top = top_of(path, root)
    if top in DEPENDENCY_TOPS:
        return "keep"
    return "safe" if is_safe_cleanup(path, root) else "review"


def scan(root, min_size=1024):
    by_size = defaultdict(list)
    total_files = 0
    total_bytes = 0
    for base, dirs, names in os.walk(root):
        for n in names:
            p = os.path.join(base, n)
            try:
                sz = os.path.getsize(p)
            except OSError:
                continue
            total_files += 1
            total_bytes += sz
            if sz >= min_size:
                by_size[sz].append(p)

    groups = []   # (kind, [paths], size)
    for sz, paths in by_size.items():
        if len(paths) < 2:
            continue
        by_hash = defaultdict(list)
        for p in paths:
            try:
                by_hash[sha256(p)].append(p)
            except OSError:
                continue
        for h, same in by_hash.items():
            if len(same) < 2:
                continue
            kind = classify(same[0], root)
            # 组内只要有一个是"保留"，整组都按保留处理（宁可保守）
            if any(classify(p, root) == "keep" for p in same):
                kind = "keep"
            groups.append((kind, same, sz))
    groups.sort(key=lambda g: g[2] * (len(g[1]) - 1), reverse=True)
    return groups, total_files, total_bytes


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return "%.1f %s" % (n, u)
        n /= 1024.0
    return "%.1f GB" % n


def report(groups, root, out=print):
    tally = defaultdict(lambda: [0, 0, 0])   # kind -> [组数, 可省字节, 涉及文件]
    for kind, same, sz in groups:
        waste = sz * (len(same) - 1)
        tally[kind][0] += 1
        tally[kind][1] += waste
        tally[kind][2] += len(same)

    out("=" * 66)
    out("磁盘占用体检：%s" % root)
    out("=" * 66)
    names = {"keep": "保留（依赖/模型/运行时，删了会坏）",
             "safe": "可清理（缓存/中间产物，删了自动重建）",
             "review": "需确认（业务区重复，自己判断）"}
    for k in ("keep", "safe", "review"):
        g, w, f = tally[k]
        if g:
            out("  %-34s %5d 组   可省 %9s   涉及 %d 个文件"
                % (names[k], g, human(w), f))
        else:
            out("  %-34s 无" % names[k])
    out("")

    for kind in ("review", "safe", "keep"):
        sel = [g for g in groups if g[0] == kind]
        if not sel:
            continue
        out("-" * 66)
        out("【%s】%d 组，显示前 25 组" % (names[kind], len(sel)))
        for k, same, sz in sel[:25]:
            out("  %9s  x%d  每个 %s" % (human(sz * (len(same) - 1)),
                                            len(same), human(sz)))
            for p in same[:5]:
                out("      " + os.path.relpath(p, root))
            if len(same) > 5:
                out("      ... 另有 %d 个" % (len(same) - 5))
        if len(sel) > 25:
            out("  ... 另有 %d 组未显示" % (len(sel) - 25))
        out("")


def clean_safe(groups, root, dry=True, out=print):
    n = w = 0
    for kind, same, sz in groups:
        if kind != "safe":
            continue
        # 保留一份，其余删除
        for p in same[1:]:
            try:
                if dry:
                    out("  [干跑] 删除 " + os.path.relpath(p, root))
                else:
                    os.remove(p)
                    out("  已删除 " + os.path.relpath(p, root))
                n += 1
                w += sz
            except OSError as e:
                out("  跳过 %s：%s" % (p, e))
    return n, w


def main(argv=None):
    ap = argparse.ArgumentParser(description="重复文件体检与安全清理")
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)),
                    help="要扫描的目录（默认是本脚本所在目录）")
    ap.add_argument("--min-size", type=int, default=1024,
                    help="只看 >= N 字节的重复文件（默认 1024）")
    ap.add_argument("--clean-safe", action="store_true",
                    help="清理「可清理」等级（默认干跑，只打印不删）")
    ap.add_argument("--yes", action="store_true",
                    help="配合 --clean-safe 真正执行删除（否则只干跑）")
    ap.add_argument("--json", default=None, help="把结果写成 JSON")
    a = ap.parse_args(argv)

    root = os.path.abspath(a.root)
    t0 = time.time()
    groups, nfiles, nbytes = scan(root, a.min_size)

    report(groups, root)

    if a.clean_safe:
        n, w = clean_safe(groups, root, dry=not a.yes)
        print("-" * 66)
        if a.yes:
            print("已清理 %d 个文件，释放 %s" % (n, human(w)))
        else:
            print("干跑：将清理 %d 个文件，释放 %s（加 --yes 才会真删）"
                  % (n, human(w)))

    print("扫描 %d 个文件（%.2f GB），耗时 %.1fs"
          % (nfiles, nbytes / (1 << 30), time.time() - t0))

    if a.json:
        data = {
            "root": root,
            "files": nfiles,
            "bytes": nbytes,
            "groups": [
                {"kind": k, "size": sz, "waste": sz * (len(same) - 1),
                 "paths": [os.path.relpath(p, root) for p in same]}
                for k, same, sz in groups
            ],
        }
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        print("JSON 已写入 %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
