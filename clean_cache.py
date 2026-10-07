# -*- coding: utf-8 -*-
"""
clean_cache.py —— 清理「歌词生成器」目录下可安全删除的残留缓存

清理对象（都是代码不引用、可重建的临时/缓存文件）：
  * 整棵目录树里的 __pycache__/ 目录（Python 字节码缓存，启动时会自动重建）
  * .tmp/（临时目录：解码的 WAV 等中间文件，可整棵删）
  * .deps-build/（历史版本的 PyInstaller 构建缓存，运行/启动都不引用）
  * out/_guitest/（GUI 测试残留，代码不使用）
  * output/（早期测试残留的目录，代码不使用，但 out/downloads、out/converts 会保留）

不会动：out/downloads、out/converts（程序正常输出目录）、
        data/、cookies/、models/、runtime/、_internal/、
        以及所有 .deps* 依赖目录、*.exe、*.py、*.spec 等。

用法：
  python clean_cache.py            正常清理（清理前会打印将要移除的项）
  python clean_cache.py --quiet    只打印结果
  python clean_cache.py --dry      只列出来、不真删

也用作「设置 -> 清理无用缓存」按钮的后端。
"""
import os
import sys
import shutil
import argparse

# 这里只依赖平面目录结构，不 import 主程序，避免拖动一整套依赖。

# 冻结成 exe 后，__file__ 指向 PyInstaller 解出的临时目录（_MEI...），
# 必须像主程序一样用 sys.executable 定位到真正的安装目录。
_FROZEN = bool(getattr(sys, "frozen", False))
if _FROZEN:
    _HERE = os.path.dirname(os.path.abspath(sys.executable))
else:
    _HERE = os.path.dirname(os.path.abspath(__file__))


# 受保护的顶层目录：依赖 / 用户数据 / 打包产物，整棵都不碰。
#
# 刻意不含 out/ 和 runtime/：
#   * out/     —— 清理目标只有 __pycache__ 和几个写死的目录名，
#                 out/downloads 永远不会成为目标，业务文件天然安全。
#   * runtime/ —— 只有 ffmpeg/node/python 本体和标准库；本体不是清理目标，
#                 而 runtime\python\Lib 下的 __pycache__ 是可重建的字节码缓存。
_PROTECTED = (".deps", ".deps-ocr", ".deps-ytdlp", ".deps-ejs", ".deps-build",
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
    return rel.replace("\\", "/").split("/")[0] in _PROTECTED


def _iter_dirs(name):
    """递归产出根目录下所有名为 name 的目录（含子目录），跳过受保护目录。"""
    out = []
    for root, dirs, _files in os.walk(_HERE):
        if os.path.basename(root) in ("__pycache__",):
            continue
        dirs[:] = [d for d in dirs if not _skip(os.path.join(root, d))]
        for d in dirs:
            if d == name:
                p = os.path.join(root, d)
                if _skip(p):
                    continue
                out.append(p)
    # 反向排序：先删深层的，避免父目录删除时冲突
    out.sort(key=len, reverse=True)
    return out


def clean_cache(dry=False, quiet=False):
    """执行清理，返回 (removed_count, freed_bytes, items)。

    items 是 [(path, size_bytes_or_None), ...]，便于 UI 展示。
    dry=True 时只统计不删除。
    """
    targets = []

    # 1) 全树 __pycache__
    for p in _iter_dirs("__pycache__"):
        targets.append(p)

    # 2) .tmp 整棵临时目录（解码 WAV 等中间文件都在里面）
    _tmp = os.path.join(_HERE, ".tmp")
    if os.path.isdir(_tmp):
        targets.append(_tmp)

    # 3) 代码不引用的残留目录（位于根目录）
    for p in (os.path.join(_HERE, ".deps-build"),
              os.path.join(_HERE, "out", "_guitest"),
              os.path.join(_HERE, "output")):
        if os.path.isdir(p):
            targets.append(p)

    # 去重 + 去掉父目录已包含的子项
    uniq, seen = [], set()
    for p in sorted(targets, key=len):
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        parent, covered = os.path.dirname(ap), False
        while len(parent) >= len(_HERE) and parent != os.path.dirname(parent):
            if parent in seen:
                covered = True
                break
            parent = os.path.dirname(parent)
        if covered:
            continue
        seen.add(ap)
        uniq.append(ap)
    targets = uniq

    removed, freed, items = 0, 0, []
    for p in targets:
        try:
            sz = _dir_size(p)
        except Exception:
            sz = 0
        if not quiet:
            rel = os.path.relpath(p, _HERE)
            print(("  [干跑] " if dry else "  删除 ") + rel +
                  ("  (%.1f MB)" % (sz / 1048576.0) if sz else ""))
        if not dry:
            try:
                shutil.rmtree(p, ignore_errors=True)
                removed += 1
                freed += sz
                items.append((p, sz))
                continue
            except Exception as e:
                items.append((p, None))
                print("  跳过 %s: %s" % (os.path.relpath(p, _HERE), e))
                continue
        items.append((p, sz))
    return removed, freed, items


def _dir_size(p):
    total = 0
    for _root, _dirs, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(_root, f))
            except Exception:
                pass
    return total


def main(argv=None):
    ap = argparse.ArgumentParser(description="清理歌词生成器目录下的无用缓存")
    ap.add_argument("--dry", action="store_true", help="只列出、不删除")
    ap.add_argument("--quiet", action="store_true", help="不打印逐项信息")
    args = ap.parse_args(argv)

    if not args.quiet:
        print("清理目录：%s" % _HERE)

    removed, freed, _items = clean_cache(dry=args.dry, quiet=args.quiet)

    if args.dry:
        print("干跑完成：以上 %d 项将被清理。" % removed)
    else:
        mb = freed / 1048576.0
        print("完成：清理 %d 项，释放 %.1f MB 空间。" % (removed, mb))
    return 0


if __name__ == "__main__":
    sys.exit(main())
