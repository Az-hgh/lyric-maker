#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lyric_gui.py —— 歌词生成器的图形界面（tkinter）

刻意把识别放在后台线程 + 队列回传日志，避免界面假死。
界面本身不碰识别逻辑，全部委托给 lyric_maker.transcribe()。
"""

import os
import sys
import queue
import threading
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))

AUDIO_EXT = ".mp3 .flac .m4a .wav .aac .ogg .opus .wma .ape"
VIDEO_EXT = ".mp4 .mkv .avi .flv .webm .mov .ts .m4v .wmv .mpg"

# 优先用质量高的模型，但**必须真的存在 model.bin** 才算可用。
# （踩过坑：只看目录存在会把下载失败残留的空目录当成可用模型，默认选中一个跑不起来的模型。）
MODEL_PREFERENCE = ["large-v3", "turbo", "medium", "small", "base", "tiny"]


def usable_models():
    from lyric_maker import MODEL_REPOS, MODELS_DIR
    out = []
    for n in MODEL_PREFERENCE:
        if n in MODEL_REPOS and os.path.isfile(os.path.join(MODELS_DIR, n, "model.bin")):
            out.append(n)
    return out


def default_model():
    have = usable_models()
    return have[0] if have else "small"


def launch(check_only=False, capture_png=None):
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
    except Exception as e:
        print("无法加载 tkinter，请改用命令行模式：python lyric_maker.py <文件>")
        print("原因:", e)
        return 1

    from lyric_maker import MODEL_REPOS, MODELS_DIR, transcribe

    root = tk.Tk()
    root.title("歌词生成器 — 音乐/视频 自动生成 .lrc")
    root.geometry("760x620")
    root.minsize(700, 560)

    q = queue.Queue()
    state = {"running": False, "t0": 0.0, "out": None}

    # ---------------- 顶部：文件选择 ----------------
    frm_file = ttk.LabelFrame(root, text="1. 选择音乐或视频")
    frm_file.pack(fill="x", padx=10, pady=(10, 6))

    var_file = tk.StringVar()
    ent = ttk.Entry(frm_file, textvariable=var_file)
    ent.pack(side="left", fill="x", expand=True, padx=(8, 6), pady=8)

    def pick():
        p = filedialog.askopenfilename(
            title="选择音乐或视频",
            filetypes=[("音乐/视频", "*" + " *".join((AUDIO_EXT + " " + VIDEO_EXT).split())),
                       ("所有文件", "*.*")])
        if p:
            var_file.set(p)
            if not var_out.get():
                var_out.set(os.path.splitext(p)[0] + ".lrc")
            refresh_hint()

    ttk.Button(frm_file, text="浏览…", command=pick).pack(side="left", padx=(0, 8), pady=8)

    # ---------------- 中部左：参数 ----------------
    frm_opt = ttk.LabelFrame(root, text="2. 识别参数")
    frm_opt.pack(fill="x", padx=10, pady=6)

    g = ttk.Frame(frm_opt)
    g.pack(fill="x", padx=8, pady=8)

    ttk.Label(g, text="模型:").grid(row=0, column=0, sticky="w", pady=3)
    var_model = tk.StringVar(value=default_model())
    cb_model = ttk.Combobox(g, textvariable=var_model, width=12, state="readonly",
                            values=list(MODEL_REPOS.keys()))
    cb_model.grid(row=0, column=1, sticky="w", padx=(4, 16), pady=3)

    ttk.Label(g, text="语言:").grid(row=0, column=2, sticky="w", pady=3)
    var_lang = tk.StringVar(value="zh")
    ttk.Combobox(g, textvariable=var_lang, width=8, state="readonly",
                 values=["zh", "en", "ja", "ko", "auto"]).grid(row=0, column=3, sticky="w", padx=(4, 16), pady=3)

    ttk.Label(g, text="线程:").grid(row=0, column=4, sticky="w", pady=3)
    var_threads = tk.IntVar(value=min(16, os.cpu_count() or 4))
    ttk.Spinbox(g, from_=1, to=(os.cpu_count() or 8), textvariable=var_threads,
                width=5).grid(row=0, column=5, sticky="w", padx=(4, 0), pady=3)

    var_vad = tk.BooleanVar(value=False)
    ttk.Checkbutton(g, text="语音活动检测（纯人声/视频建议勾选；歌曲一般不用）",
                    variable=var_vad).grid(row=1, column=0, columnspan=6, sticky="w", pady=(6, 0))

    ttk.Label(g, text="提示词:").grid(row=2, column=0, sticky="w", pady=(6, 0))
    var_prompt = tk.StringVar(value="以下是普通话歌曲的歌词，请输出简体中文并带标点。")
    ttk.Entry(g, textvariable=var_prompt).grid(row=2, column=1, columnspan=5,
                                               sticky="ew", padx=(4, 0), pady=(6, 0))
    g.columnconfigure(5, weight=1)

    # 歌词文本 -> 对齐模式（推荐路径）
    ttk.Label(g, text="歌词文本:").grid(row=3, column=0, sticky="w", pady=(8, 0))
    var_lyrics = tk.StringVar()
    ttk.Entry(g, textvariable=var_lyrics).grid(row=3, column=1, columnspan=4,
                                               sticky="ew", padx=(4, 0), pady=(8, 0))

    def pick_lyrics():
        p = filedialog.askopenfilename(
            title="选择歌词文本（txt 或 lrc）",
            filetypes=[("歌词/文本", "*.txt *.lrc"), ("所有文件", "*.*")])
        if p:
            var_lyrics.set(p)

    ttk.Button(g, text="浏览…", command=pick_lyrics).grid(row=3, column=5,
                                                          sticky="e", padx=(4, 0), pady=(8, 0))

    lbl_mode = ttk.Label(g, text="", foreground="#1a7f37")
    lbl_mode.grid(row=4, column=0, columnspan=6, sticky="w", pady=(2, 0))

    def upd_mode(*_a):
        if var_lyrics.get().strip():
            lbl_mode.configure(
                text="★ 已填歌词文本 → 对齐模式：文字用你的原文，识别只提供时间锚点。"
                     "歌曲请务必用这个模式")
        else:
            lbl_mode.configure(text="（未填歌词文本 → 自动听写模式；口播/视频适用，"
                                    "歌曲会有大量谐音错字）")

    var_lyrics.trace_add("write", upd_mode)
    upd_mode()

    lbl_hint = ttk.Label(frm_opt, text="", foreground="#555")
    lbl_hint.pack(fill="x", padx=10, pady=(0, 8))

    # ---------------- 输出路径 ----------------
    frm_out = ttk.LabelFrame(root, text="3. 输出 .lrc 路径")
    frm_out.pack(fill="x", padx=10, pady=6)
    var_out = tk.StringVar()
    ttk.Entry(frm_out, textvariable=var_out).pack(side="left", fill="x", expand=True, padx=(8, 6), pady=8)

    def pick_out():
        p = filedialog.asksaveasfilename(title="保存为", defaultextension=".lrc",
                                        filetypes=[("LRC 歌词", "*.lrc")])
        if p:
            var_out.set(p)

    ttk.Button(frm_out, text="另存为…", command=pick_out).pack(side="left", padx=(0, 8), pady=8)

    # ---------------- 操作按钮 ----------------
    frm_run = ttk.Frame(root)
    frm_run.pack(fill="x", padx=10, pady=(2, 6))

    btn_run = ttk.Button(frm_run, text="开始生成歌词")
    btn_run.pack(side="left")

    lbl_status = ttk.Label(frm_run, text="就绪", foreground="#333")
    lbl_status.pack(side="left", padx=12)

    def open_dir():
        import subprocess
        d = os.path.dirname(var_out.get()) or _HERE
        try:
            os.startfile(d)
        except Exception:
            subprocess.Popen(["explorer", d])

    ttk.Button(frm_run, text="打开输出目录", command=open_dir).pack(side="right")

    # ---------------- 日志 ----------------
    frm_log = ttk.LabelFrame(root, text="日志")
    frm_log.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    txt = tk.Text(frm_log, wrap="word", height=14, font=("Consolas", 9))
    sb = ttk.Scrollbar(frm_log, command=txt.yview)
    txt.configure(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y")
    txt.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
    txt.configure(state="disabled")

    def log(msg):
        q.put(("log", str(msg)))

    def refresh_hint():
        p = var_file.get()
        if not p:
            lbl_hint.configure(text="")
            return
        ext = os.path.splitext(p)[1].lower()
        kind = "视频" if ext in VIDEO_EXT.split() else ("音频" if ext in AUDIO_EXT.split() else "未知")
        have = [n for n in MODEL_REPOS if os.path.isdir(os.path.join(MODELS_DIR, n))
                and os.path.isfile(os.path.join(MODELS_DIR, n, "model.bin"))]
        lbl_hint.configure(
            text="类型: %s   已下载模型: %s%s" % (
                kind, ", ".join(have) if have else "（无）",
                "" if have else "   请先运行: python get_model.py small"))

    # ---------------- 运行 ----------------
    def worker(media, out, model, lang, vad, threads, prompt, lyrics_file):
        try:
            res = transcribe(media, model_name=model, language=lang, out=out,
                             prompt=prompt or None, vad=vad, threads=threads,
                             lyrics_file=lyrics_file or None, log=log)
            q.put(("done", res))
        except Exception:
            q.put(("error", traceback.format_exc()))

    def start():
        if state["running"]:
            return
        media = var_file.get().strip().strip('"')
        if not media:
            messagebox.showwarning("缺少输入", "请先选择音乐或视频文件。")
            return
        if not os.path.isfile(media):
            messagebox.showerror("文件不存在", media)
            return
        model = var_model.get()
        mdir = os.path.join(MODELS_DIR, model)
        if not os.path.isfile(os.path.join(mdir, "model.bin")):
            if not messagebox.askyesno(
                    "模型未下载",
                    "模型 '%s' 还没下载。\n\n"
                    "请在命令行运行下把命令后再试：\n"
                    "    python get_model.py %s\n\n"
                    "是否改用已下载的模型继续？" % (model, model)):
                return
            have = [n for n in MODEL_REPOS
                    if os.path.isfile(os.path.join(MODELS_DIR, n, "model.bin"))]
            if not have:
                messagebox.showerror("没有可用模型", "一个模型都没下载，请先运行 get_model.py")
                return
            model = have[0]
            var_model.set(model)

        out = var_out.get().strip() or (os.path.splitext(media)[0] + ".lrc")
        var_out.set(out)

        state["running"] = True
        state["t0"] = time.time()
        btn_run.configure(state="disabled")
        lbl_status.configure(text="识别中…请勿关闭窗口", foreground="#b35c00")
        txt.configure(state="normal")
        txt.delete("1.0", "end")
        txt.configure(state="disabled")
        log("开始: %s" % os.path.basename(media))
        t = threading.Thread(target=worker, args=(media, out, model, var_lang.get(),
                                                  var_vad.get(), var_threads.get(),
                                                  var_prompt.get(),
                                                  var_lyrics.get().strip()), daemon=True)
        t.start()

    btn_run.configure(command=start)
    ent.bind("<Return>", lambda e: refresh_hint())
    var_file.trace_add("write", lambda *a: refresh_hint())

    def tick():
        try:
            while True:
                kind, payload = q.get_nowait()
                if kind == "log":
                    txt.configure(state="normal")
                    txt.insert("end", payload + "\n")
                    txt.see("end")
                    txt.configure(state="disabled")
                    lbl_status.configure(
                        text="识别中… 已用 %.0f 秒" % (time.time() - state["t0"]),
                        foreground="#b35c00")
                elif kind == "done":
                    state["running"] = False
                    state["out"] = payload.get("out")
                    btn_run.configure(state="normal")
                    n = len(payload.get("segments", []))
                    el = payload.get("elapsed", 0)
                    dur = payload.get("duration") or 0
                    speed = (dur / el) if el else 0
                    lbl_status.configure(
                        text="完成：%d 行，识别耗时 %.0fs（约 %.1f 倍实时）" % (n, el, speed),
                        foreground="#1a7f37")
                    if state["out"] and os.path.isfile(state["out"]):
                        try:
                            os.startfile(state["out"])
                        except Exception:
                            pass
                elif kind == "error":
                    state["running"] = False
                    btn_run.configure(state="normal")
                    lbl_status.configure(text="失败", foreground="#c1121f")
                    txt.configure(state="normal")
                    txt.insert("end", "\n=== 错误 ===\n" + payload + "\n")
                    txt.see("end")
                    txt.configure(state="disabled")
        except queue.Empty:
            pass
        root.after(120, tick)

    tick()
    refresh_hint()

    if check_only or capture_png:
        # 自检模式：真正构建一遍全部控件（能抓出控件选项写错之类的运行时错误），
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
            try:
                from PIL import ImageGrab
                root.lift()
                root.attributes("-topmost", True)
                root.update()
                # 本进程不是 DPI 感知的：Tk 的坐标是"逻辑像素"，而 ImageGrab 按"物理像素"
                # 截屏。缩放不是 100% 的机器上直接用 Tk 坐标会整体偏移（实测偏过两次）。
                # 用"全屏实际宽度 / Tk 报告的逻辑宽度"求出真实缩放比，再换算。
                # 全屏图只用来读尺寸，不保存，不落盘。
                full = ImageGrab.grab()
                scale = full.width / float(root.winfo_screenwidth())
                del full
                x, y = root.winfo_rootx(), root.winfo_rooty()
                w, h = root.winfo_width(), root.winfo_height()
                title = int(32 * scale)      # 标题栏在客户区上方，逻辑约 32px
                bbox = (int(x * scale) - 4, int(y * scale) - title,
                        int((x + w) * scale) + 4, int((y + h) * scale) + 4)
                img = ImageGrab.grab(bbox=bbox)
                img.save(capture_png)
                print("界面截图: %s  (%dx%d 物理像素, 缩放比 %.2f)"
                      % (capture_png, img.width, img.height, scale))
            except Exception as e:
                print("截图失败: %s: %s" % (type(e).__name__, e))
        root.destroy()
        return 0

    root.mainloop()
    return 0


if __name__ == "__main__":
    png = None
    if "--capture" in sys.argv:
        i = sys.argv.index("--capture")
        if i + 1 < len(sys.argv):
            png = sys.argv[i + 1]
    sys.exit(launch(check_only=("--check" in sys.argv) or (png is not None),
                    capture_png=png))
