# -*- coding: utf-8 -*-
"""
cookiescan.py —— 自动发现并导入插件导出的 Cookie 文件。

为什么需要它
------------
Chrome 127+ 之后浏览器自己的 Cookie 数据库被 app-bound 加密、且运行期独占锁定，
不关浏览器就读不出来（yt-dlp 官方同样失败，见其 issue #7271）。于是唯一能走通的
办法是用「Get cookies.txt LOCALLY」这类插件导出 —— 但插件只调
``chrome.downloads.download({url, filename})``，**不带目录参数**，文件落在
Chrome 的下载目录，用户还得手动拖进程序目录。

这个模块做的就是把那一步接上：盯着几个常见目录，一旦发现新的 Cookie 文件就
**复制**进程序目录（不动原文件），并验证里面有没有 YouTube 登录态。

为什么可行
----------
被监视的是 Downloads / Chrome 下载目录 / 桌面这些**普通文件夹**，不是 Chrome
那个加密数据库 —— 读普通文件不受 Chrome 是否运行影响。所以"不用关浏览器"这条
依然成立。

安全边界
--------
  * 只**复制**，从不移动或删除用户的原文件
  * 文件名含 cookie **且**内容是 Netscape 格式，两个条件都满足才收
  * 用内容哈希去重，导入过的不会重复处理
  * 只在程序运行期间监听；启动时先补扫一次，避免漏掉上次退出后导出的
"""
import os
import re
import io
import time
import shutil
import hashlib

# 文件名里出现这些词之一才可能是 cookie 文件
_NAME_HINTS = ("cookie", "cookies")
# 扫描间隔（秒）。太密会频繁读盘，太疏会让用户等。
POLL_SECONDS = 5.0
# 单个文件超过这个大小就不看（cookie 文件正常几百 KB，几十 MB 的一定不是）
MAX_SIZE = 32 * 1024 * 1024


def looks_like_cookie_name(name):
    """文件名像不像 cookie 导出文件。"""
    low = str(name or "").lower()
    if not low.endswith(".txt"):
        return False
    return any(h in low for h in _NAME_HINTS)


def looks_like_cookie_content(path, head=8192):
    """内容是不是 Netscape 格式的 cookie 文件。

    与主程序里的判定一致：优先认文件头 `Netscape HTTP Cookie File`，
    否则看有没有"7 个制表符字段且第 6/7 个非空"的结构行。
    只读文件头 —— cookie 文件可能几百 KB，逐个全读会拖慢轮询。
    """
    try:
        if os.path.getsize(path) > MAX_SIZE:
            return False
        with io.open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read(head)
    except Exception:
        return False
    if "Netscape HTTP Cookie File" in text:
        return True
    for ln in text.splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        parts = ln.split("\t")
        if len(parts) >= 7 and parts[5].strip() and parts[6].strip():
            return True
    return False


def file_digest(path):
    """内容哈希，用于判断"这份和上次导入的是同一个"。"""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                b = f.read(1 << 20)
                if not b:
                    break
                h.update(b)
    except Exception:
        return ""
    return h.hexdigest()


def candidate_dirs(dirs=None):
    """要监视的目录列表（自动 + 调用方补充）。

    自动收集的是"插件导出最可能落地的几个地方"：
      * Chrome/Edge 当前设置的下载目录（从它们自己的 Preferences 里读）
      * 系统「下载」文件夹（注册表里的 Known Folder）
      * 桌面
    任何一步读不到就跳过，不影响其余。
    """
    out = []
    seen = set()

    def add(p):
        if not p:
            return
        try:
            p = os.path.abspath(p)
            # Windows 路径不区分大小写，去重也要按不敏感来 ——
            # 否则同一个下载文件夹会被注册表和 fallback 各报一次，白扫一遍。
            key = os.path.normcase(p)
        except Exception:
            return
        if not os.path.isdir(p) or key in seen:
            return
        seen.add(key)
        out.append(p)

    # 1) Chrome / Edge 的下载目录
    for env, prof in (
            ("LOCALAPPDATA", r"Google\Chrome\User Data\Default\Preferences"),
            ("LOCALAPPDATA", r"Microsoft\Edge\User Data\Default\Preferences")):
        base = os.environ.get(env)
        if not base:
            continue
        p = os.path.join(base, prof)
        d = _read_pref_download_dir(p)
        if d:
            add(os.path.expandvars(d))
            add(os.path.expanduser(d))

    # 2) 系统「下载」文件夹（拿注册表里的真实路径，而不是猜 %USERPROFILE%\Downloads）
    #    注意：raw string 不能以反斜杠结尾，所以这里拆成两段拼。
    _uf_key = (r"SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer"
               "\\User Shell Folders")
    _dl_guid = "{374DE290-123F-4565-9164-39C4925E467B}"
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _uf_key) as k:
            val, _ = winreg.QueryValueEx(k, _dl_guid)
        if val:
            add(os.path.expandvars(val))
    except Exception:
        pass
    add(os.path.join(os.path.expanduser("~"), "Downloads"))
    add(os.path.join(os.path.expanduser("~"), "Desktop"))

    for extra in (dirs or []):
        add(extra)
    return out


def _read_pref_download_dir(pref_path):
    """从 Chrome/Edge 的 Preferences 里取下载目录（读不到返回 ""）。"""
    try:
        if not os.path.isfile(pref_path):
            return ""
        with io.open(pref_path, "r", encoding="utf-8", errors="replace") as f:
            raw = f.read()
        # Preferences 是个大 JSON，用正则只抠需要的两处，避免整个解析。
        m = re.search(r'"download"\s*:\s*\{[^{}]*?"default_directory"\s*:\s*"([^"]*)"', raw)
        if not m:
            m = re.search(r'"savefile"\s*:\s*\{[^{}]*?"default_directory"\s*:\s*"([^"]*)"', raw)
        d = (m.group(1) if m else "").strip()
        # Chrome 未设置时该字段可能是空串、或以反斜杠开头的裸路径
        return d if d else ""
    except Exception:
        return ""


class CookieImporter(object):
    """后台监视器：定时扫描候选目录，把新出现的 Cookie 文件复制进来。

    用法：
        imp = CookieImporter(dest_dir, on_import=cb)
        imp.start()          # 启动线程 + 先补扫一次
        imp.stop()           # 退出时调用

    on_import(imported_list, msg) 在**主线程之外**被调用，界面相关的更新
    要自己通过队列切回主线程（主程序里就是这么做的）。
    """

    def __init__(self, dest_dir, on_import=None, extra_dirs=None,
                 poll=POLL_SECONDS, log=None):
        self.dest = dest_dir
        self.on_import = on_import
        self.extra_dirs = list(extra_dirs or [])
        self.poll = float(poll)
        self.log = log
        self._thread = None
        self._stop_evt = None
        self._seen = set()      # 已处理过的 (路径, mtime, size)
        self._digests = set()   # 已导入内容的哈希，防止同一份重复导入
        self._busy = False

    # ---------- 对外 ----------
    def start(self):
        if self._thread is not None:
            return
        import threading
        self._stop_evt = threading.Event()
        # 启动时先把已知 cookie 目录里的现有文件登记为"已见过"，
        # 免得把用户早就手动放好的文件又当成新导出再提示一遍。
        self._prime_existing()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._stop_evt is not None:
            self._stop_evt.set()
        self._thread = None

    def scan_once(self):
        """扫一轮，返回本次新导入的 [(源路径, 目标路径)]。"""
        if self._busy:
            return []
        self._busy = True
        got = []
        try:
            dest = self.dest
            try:
                os.makedirs(dest, exist_ok=True)
            except Exception:
                return []
            for d in candidate_dirs(self.extra_dirs):
                # 目标目录本身不用扫（里面的文件是我们自己放的）
                try:
                    if os.path.abspath(d) == os.path.abspath(dest):
                        continue
                except Exception:
                    pass
                got.extend(self._scan_dir(d, dest))
        except Exception:
            pass
        finally:
            self._busy = False
        return got

    # ---------- 内部 ----------
    def _prime_existing(self):
        """把 dest 里已有的文件按内容哈希登记，避免重复导入提示。"""
        try:
            for n in os.listdir(self.dest):
                p = os.path.join(self.dest, n)
                if os.path.isfile(p):
                    h = file_digest(p)
                    if h:
                        self._digests.add(h)
        except Exception:
            pass

    def _scan_dir(self, d, dest):
        out = []
        try:
            names = os.listdir(d)
        except Exception:
            return out
        for n in names:
            src = os.path.join(d, n)
            try:
                st = os.stat(src)
            except OSError:
                continue
            if not os.path.isfile(src):
                continue
            key = (src, st.st_mtime, st.st_size)
            if key in self._seen:
                continue
            self._seen.add(key)
            if not looks_like_cookie_name(n):
                continue
            if not looks_like_cookie_content(src):
                continue
            h = file_digest(src)
            if not h or h in self._digests:
                continue          # 同一份内容已经导过了
            dst = self._unique_dst(dest, n)
            try:
                shutil.copy2(src, dst)     # 复制，不动原文件
            except Exception as e:
                self._say("复制失败 %s：%s" % (n, e))
                continue
            self._digests.add(h)
            out.append((src, dst))
        return out

    @staticmethod
    def _unique_dst(dest, name):
        """目标同名时加序号，避免覆盖用户已有文件。"""
        base, ext = os.path.splitext(name)
        cand = os.path.join(dest, name)
        i = 1
        while os.path.exists(cand):
            cand = os.path.join(dest, "%s_%d%s" % (base, i, ext))
            i += 1
        return cand

    def _loop(self):
        while self._stop_evt is not None and not self._stop_evt.is_set():
            try:
                got = self.scan_once()
                if got and self.on_import:
                    self.on_import(got, "")
            except Exception:
                pass
            if self._stop_evt is not None:
                self._stop_evt.wait(self.poll)

    def _say(self, msg):
        if self.log:
            try:
                self.log(msg)
            except Exception:
                pass
