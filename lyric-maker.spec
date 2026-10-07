# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 打包配置 —— 把整个工具做成独立可运行的 exe。

设计取舍：
  * 用 onedir（一个文件夹 + 里面的 exe），不用 onefile。
    这个工具依赖 ctranslate2 / onnxruntime / opencv 等一堆原生 DLL，
    总计数百 MB，onefile 每次启动都要把这么多东西解压到临时目录，慢得没法用。
  * **模型不打包**。large-v3 单独就 2.9GB，而很多人只用 small。
    exe 运行时会去自己旁边的 models/ 目录找模型（代码里 _HERE 在打包后
    指向 exe 所在目录），把整个文件夹拷走、旁边放个 models 就能用。
  * **console=False：双击不弹控制台窗口，直接进图形界面。**
    代价是命令行模式下完全看不到 stdout，所以代码里做了两层兜底：
    无控制台时把 sys.stdout 换成空流（否则 print 会直接抛异常），
    处理完成或出错改用弹窗告知。
    如果你要频繁用命令行、需要看 --get-model 的下载进度，
    把下面 console 改成 True 重新打包，就回到带控制台的版本。

用法：
    python -m PyInstaller lyric-maker.spec --noconfirm
产物：
    dist/lyric-maker/lyric-maker.exe
    它和同目录的 _internal/ 是一对，必须一起移动，单独拿 exe 会启动失败。
    本仓库把两者都放在了项目根目录（见 README 的"打包成独立 exe"）。
"""
import os
import sys
from PyInstaller.utils.hooks import collect_all

HERE = os.path.abspath(SPECPATH)

# 关键：collect_all() 必须能在打包时真的 import 到这些包，才能收集它们的
# 数据文件、原生 DLL 和子模块。而 collect_all 是在本文件求值时执行的，
# Analysis(pathex=...) 是之后才生效的 —— 只写 pathex 的话 collect_all
# 照样 import 失败，于是"看起来打包成功"，实际漏掉整个包。
#
# 实测踩过的坑：不把 .deps 挂进 sys.path 时，faster_whisper / cv2 /
# rapidocr_onnxruntime 全部静默漏掉，产出的 exe 一启动就 ModuleNotFoundError。
# 所以这里把 .deps、.deps-ocr 与 yt-dlp 的两个目录一起挂上。
# 顺序与运行期保持一致：.deps 最前，OCR 依赖排后（只补 .deps 里没有的包）。
_DEPS = os.path.join(HERE, ".deps")
_DEPS_OCR = os.path.join(HERE, ".deps-ocr")
_YTDLP = os.path.join(HERE, ".deps-ytdlp")
_EJS = os.path.join(HERE, ".deps-ejs")
for _d in (_DEPS, _DEPS_OCR, _YTDLP, _EJS):
    if os.path.isdir(_d) and _d not in sys.path:
        sys.path.insert(0, _d)
# .deps-ocr 必须在 .deps 之后：它只用来补 rapidocr/shapely/pyclipper，
# numpy/onnxruntime/cv2 一律以 .deps 里的版本为准。
if os.path.isdir(_DEPS) and _DEPS in sys.path:
    sys.path.remove(_DEPS)
    sys.path.insert(0, _DEPS)

# 这些包带原生 DLL、模型文件或数据文件，必须整包收集，
# 只靠静态分析会漏掉运行时才加载的东西（onnx 模型、cuDNN 之类的 DLL）。
_PACKAGES = (
    "ctranslate2", "onnxruntime", "av", "tokenizers", "faster_whisper",
    "rapidocr_onnxruntime", "shapely", "pyclipper", "opencc",
    "huggingface_hub", "cv2",
    "yt_dlp",
    "yt_dlp_ejs",
    # 「从浏览器更新 Cookie」要解浏览器加密的 cookie 值；yt-dlp 自带的提取器
    # 会用到 cryptography（AES-GCM）。不收的话按钮点下去会报 No module named。
    "cryptography",
)

datas, binaries, hiddenimports = [], [], []
for _pkg in _PACKAGES:
    try:
        _d, _b, _h = collect_all(_pkg)
    except Exception as _e:          # 某个可选包没装就跳过，不阻断整个打包
        print("[spec] 跳过 %s: %s" % (_pkg, _e))
        continue
    datas += _d
    binaries += _b
    hiddenimports += _h

# certifi 要特别照顾：它真正的代码只有 __init__.py + core.py 两个小文件，
# 而 cacert.pem 会被 collect_all 当成数据收走。PyInstaller 分析 certifi 时
# 有时两个 .py 没进包、只留下一个空壳 certifi 目录 —— 结果 `import certifi`
# 能过（当成命名空间包）却没有 where()，而 yt_dlp/__init__.py 一上来就 import
# yt_dlp.cookies → yt_dlp.dependencies 里要 certifi.where()，于是**整个 yt_dlp
# 都 import 失败**。这里把源码目录整个当数据塞进去，运行时就能正常找到。
_DEPS = os.path.join(HERE, ".deps")
for _mod in ("certifi", "cryptography"):
    _src = os.path.join(_DEPS, _mod)
    if os.path.isdir(_src):
        for _root, _dirs, _files in os.walk(_src):
            if "__pycache__" in _root or os.path.basename(_root) == "tests":
                continue
            _rel = os.path.relpath(_root, _DEPS)
            for _f in _files:
                if _f.endswith((".py", ".pem", ".typed")):
                    datas.append((os.path.join(_root, _f), _rel))
        print("[spec] 补收 %s 源码" % _mod)

hiddenimports += ["mutagen", "yaml", "numpy", "tqdm", "requests", "certifi",
                  "huggingface_hub", "tkinter", "tkinter.ttk",
                  "tkinter.filedialog", "tkinter.messagebox",
                  # 【从浏览器更新 Cookie】用到：yt-dlp 的浏览器 cookie 提取器
                  # 是按需 import 的，静态分析扫不到，不显式列出来 exe 里会缺
                  "yt_dlp.cookies",
                  # certifi 真正的代码在 __init__.py / core.py 里，而 cacert.pem
                  # 是数据。PyInstaller 偶尔只把 pem 收成数据文件、留下一个空壳
                  # certifi 目录 —— 那样 `import certifi` 能过但没有 where()，
                  # yt_dlp.cookies 一用就炸。显式点名把 .py 收进来。
                  "certifi.core",
                  # 只在 --gui-capture（界面截图）里用，是写在函数内部的 import，
                  # 静态分析有时扫不到，显式列出来免得 exe 里缺
                  "PIL", "PIL.ImageGrab",
                  # 设置页「占用体检」按钮在 worker 线程里才 import dupscan，
                  # 静态分析扫不到，纯标准库无二进制依赖，显式点名。
                  "dupscan",
                  # Cookie 自动导入监视器（后台线程里 import），同理。
                  "cookiescan"]

a = Analysis(
    [os.path.join(HERE, "lyric_maker.py")],
    pathex=[HERE, os.path.join(HERE, ".deps"), os.path.join(HERE, ".deps-ocr"),
            _YTDLP, _EJS],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # 这些是开发期才用得上的东西，排除掉能少不少体积
    excludes=["tkinter.test", "unittest", "pydoc_data", "test", "setuptools",
              "pip", "PyInstaller", "matplotlib", "pandas", "scipy", "sphinx"],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="lyric-maker",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="lyric-maker",
)
