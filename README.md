# 歌词生成器 — 从音乐/视频自动生成标准 .lrc

本地语音识别，不联网、不上传任何音频。输出网易云音乐/绝大多数播放器都能导入的标准 `.lrc`。

---

## 快速开始

按顺序双击三个批处理：

| 步骤 | 文件 | 说明 |
|---|---|---|
| 1 | `install_deps.bat` | 装 Python 依赖（走清华源）。约 150MB |
| 2 | `download_model.bat` | 下模型。先下 `small`（480MB）验证流程 |
| 3 | `run_gui.bat` | 启动图形界面 |

也可以把**音视频文件直接拖到 `run_gui.bat` 上**，直接开始识别并生成同名 `.lrc`。

---

## 两种模式（**关键**）

### 模式一：自动听写（`--lyrics-file` 不传）

工具自己听，自己写字。适合**人声清晰的口播/访谈/视频**。

```
python lyric_maker.py "讲座.mp4" --model large-v3 --vad
```

⚠️ **对"歌曲"效果有限，这是原理决定的，不是 bug。** 歌唱的元音拖长、音高变化、伴奏掩蔽，
会让任何语音识别模型输出"听着像但字不对"的结果。实测 `small` 模型听写一首中文歌曲，
大量句子是谐音乱码。换更大的模型会好一些，但不可能根治。

### 模式二：对齐模式（推荐，`--lyrics-file` 传歌词文本）

**你提供正确歌词文本，工具只负责算出每句的精确时间点。** 文字 100% 是你的原文，
零错字；识别结果仅用作时间锚点。只要歌词和音频是同一首歌，产出的 LRC 就是可用的成品。

```
python lyric_maker.py "歌曲.mp3" --lyrics-file "歌词.txt" --model small
```

- 歌词文件支持 `.txt` / `.lrc`，编码自动识别（UTF-8 / GBK / UTF-16 都行）
- 已经带时间戳的 `.lrc` 也能直接喂进来，时间戳会被剥掉重新对齐
- 歌词里多余的空行、`[ti:]` 之类的元数据行、行首序号都会被自动清理

**对齐算法的可靠性（有测试支撑）。** 内置自检用合成数据验证，并专门构造了对抗场景：

| 场景 | 结果 |
|---|---|
| 干净数据（已知真值 + 15% 字符错误 + 随机插入杂字） | 最大误差 **0.35 秒** |
| **对抗场景**：某行在音频里识别失败，但同一句在后方副歌里重复出现 | 偏差 **0.00 秒**（老算法会被拖到副歌位置，偏十几秒） |
| 真实歌曲实测：相邻行的最小间隔 | 老算法 **0.04 秒**（多行被压成一团）→ 新算法 **2.24 秒** |

对抗场景是实测踩出来的真实 bug：最初我按"该行第一个匹配到的字符"取锚点，结果某行首字
恰好在很远的副歌里命中，整行被拖偏 13 秒。修法是三层稳健化 —— 锚点覆盖度门槛、
与单调趋势严重不符的离群锚点迭代剔除、以及时间没前进的锚点直接丢弃后折线插值。

`python lyric_maker.py --selftest` 可以随时自己跑这 18 项检查。

---

## 命令行参数

> **本机注意：`python` 不在 PATH 里**，而且 `py` 启动器指向一个已损坏的 Python 3.7，
> 所以下面示例里的 `python` **不能直接用**。请用下面任一种方式调用：
>
> **方式一（推荐，单文件）**：把文件拖到 `run_gui.bat` 上，或
> ```bat
> cd /d D:\A\lyric-maker
> run_gui.bat "D:\A\某首歌.webm"
> ```
>
> **方式二（要传参数时）**：用 `install_deps.bat` 记录下来的解释器
> ```powershell
> cd D:\A\lyric-maker
> & (Get-Content .python-path.txt) lyric_maker.py "D:\A\某首歌.webm" --lyrics-file "D:\A\歌词.txt"
> ```
> 下面的示例为了简洁仍写作 `python`，实际请按上面两种方式替换。

```
python lyric_maker.py <音视频文件> [选项]

  --model NAME      模型名或模型目录，默认 medium
                    可选: tiny base small medium turbo large-v3
  --lyrics-file F   给出歌词文本 -> 进入对齐模式（强烈推荐用于歌曲）
  --language L      zh(默认) / en / ja / ko / auto
  --out PATH        输出 .lrc 路径，默认与输入文件同名同目录
  --prompt TEXT     初始提示词，可引导用词、标点、简繁
  --vad             开启语音活动检测。口播/视频建议开；歌曲建议关（默认关）
  --threads N       CPU 线程数，默认 min(16, 核心数)
  --start S         只处理从第 S 秒开始
  --duration S      只处理 S 秒（快速试听效果用）
  --offset MS       时间轴整体平移毫秒
  --no-split        不按标点拆分过长行
  --no-merge        不合并过短的相邻段
  --keep-wav        保留中间 WAV 便于排查
  --gui             启动图形界面
  --selftest        运行内置自检（不需要音频）
```

### 模型选择

| 模型 | 大小 | 源 | 建议 |
|---|---|---|---|
| `small` | 480 MB | hf-mirror | 流程验证、对齐模式的锚点够用 |
| `turbo` | 1.6 GB | hf-mirror | 速度与质量折中 |
| `large-v3` | 2.9 GB | **ModelScope** | 质量最好，中文首选 |

```
python get_model.py              # 列出全部模型与状态
python get_model.py small
python get_model.py large-v3
```

---

## 实测性能（本机 i9-14900HX，CPU int8，16 线程）

以一首 3 分 45 秒的 mp3 为例：

| 项目 | 结果 |
|---|---|
| 解码 + 载入 | 约 1 秒 |
| 识别（`small`） | 24.5 秒 → **约 9.2 倍实时速度** |
| 端到端 | 27.5 秒 |
| 对齐模式时间误差 | 合成数据实测最大 **0.35 秒** |

CPU 上跑，完全不需要 GPU。（本机是 RTX 5060 Laptop / Blackwell sm_120，
需要 CUDA 12.8+ 才能用上，收益不值当，故一律走 CPU int8。）

### `small` vs `large-v3` 实测对比（同一首歌的第 60–120 秒，输入完全相同）

| 模型 | 耗时 | 速度 | 听写质量 |
|---|---|---|---|
| `small` | 9.6 秒 | **6.3 倍实时** | 谐音乱码多；且在 01:00–01:30 的**纯伴奏段幻觉出了整句** |
| `large-v3` | 43.2 秒 | **1.4 倍实时** | 明显更连贯，用词更像真句子；且**正确识别出伴奏段不出字** |

具体差异（两边都是错的，但错的程度不同）：

```
small     [01:00.00] 背后一回拥有了背后马脸足异歌无声音      ← 这一段其实没人唱
large-v3  （01:00–01:30 无输出，正确跳过）                  ← 分辨出了纯伴奏

small     [01:45.00] 说来说出海关毛巨星当初革命格的有点机会注意
large-v3  [01:45.52] 说来说去还怪毛主席当初革命歌队有点机会主义   ← 更接近真实
```

**结论：`large-v3` 明显更好，但仍然不能直接当歌词用。** 中文歌唱的听写是原理性难题，
不要指望换模型解决。**歌曲请一律用对齐模式**（`--lyrics-file`）：
文字 100% 是你的原文零错字，识别只用来提供时间锚点。
`large-v3` 也值得下，因为它的词级时间戳更准，对齐结果更稳。

---

## 实现要点（为什么这么写）

**为什么不直接用 faster-whisper 自带的音频解码。**
它内部走 PyAV，而本环境装到的 `av 19.0.1` 与 `faster_whisper 1.2.1` 的
`decode_audio(metadata_errors=...)` 调用不兼容，会直接抛
`open() got an unexpected keyword argument 'metadata_errors'`。
所以先用 ffmpeg 解码成 16kHz 单声道 WAV，再用标准库 `wave` + numpy 读成数组喂给
`transcribe()` —— 整段 PyAV 代码根本不执行，还省一次重复解码。

**反幻觉三件套。** Whisper 在伴奏/间奏段会陷入"重复上一句"的循环，所以：
`condition_on_previous_text=False`（最关键的开关）、幻觉黑名单正则
（"请不吝点赞/订阅/字幕由…提供"这类训练集残留）、相邻重复段去重。

**繁简归一。** Whisper 在中文素材上会简繁混用（同一首歌里"局势"和"局勢"并存），
用 opencc 统一成简体。

**网络双源。** `huggingface.co` 被 DNS 污染（解析到 31.13.75.12）不可达；
`pypi.org`、`github.com` 也不可达。实测 `hf-mirror.com` 可用但仅 0.5 MB/s 且大文件
极不稳定（连续 4 段请求 3 段被远端强断）；`modelscope.cn` 实测 2.6 MB/s 且 5/5 稳定。
所以大模型走 ModelScope，并且**自写了带 Range 断点续传 + 指数退避重试的下载器**，
专门对付这种会中途断流的网络。

---

## 目录结构

```
lyric-maker/
  lyric_maker.py       引擎 + 命令行（核心）
  lyric_gui.py         图形界面
  get_model.py         模型下载（双源 + 断点续传）
  install_deps.bat     装依赖
  download_model.bat   下模型
  run_gui.bat          启动界面（支持拖放文件）
  .deps/               第三方依赖（pip --target 安装，与系统 Python 隔离）
  models/              语音识别模型
  out/                 输出目录
  .tmp/                临时文件
```

---

## 常见问题

**双击 .bat 报 "Unable to create process using ...python.exe" 或 Python 版本不对。**
`.deps` 里的 `ctranslate2`、`av`、`numpy`、`onnxruntime` 都是**编译好的 wheel，
锁死在某一个 Python 小版本上**（本机是 3.12）。所以启动脚本必须用**当初装依赖的那个
解释器**，用别的版本会直接失败。

为此 `install_deps.bat` 会把选中的解释器完整路径记进 `.python-path.txt`，另外两个
脚本优先读它。如果那个 Python 被卸载/移动了，删掉 `.python-path.txt` 重新运行
`install_deps.bat` 即可。

本机还遇到过一个坑：系统里注册了一个**指向已不存在的 Python 3.7** 的 `py` 启动器项
（`py -3` 会解析到它），所以脚本里加了 3.9+ 的版本校验，不满足就跳过继续找下一个。

**报 "找不到 ffmpeg"。**
把 ffmpeg 放进 PATH，或用 `--ffmpeg "C:\path\ffmpeg.exe"` 指定。

**模型下载卡住或失败。**
网络会自动续传重试。想换源：`python get_model.py large-v3 --source hf`。
若 `hf-mirror.com` 也不通，检查 hosts 是否把它钉到了 `127.0.0.1`。

**识别结果全是乱码/谐音。**
歌曲听写的固有问题。改用对齐模式：`--lyrics-file 歌词.txt`。

**时间轴整体偏了。**
用 `--offset 毫秒` 平移。比如歌词比画面晚 0.5 秒出现，就 `--offset -500`。

**想只试听一小段效果。**
`--start 60 --duration 30` 只处理第 60 秒起的 30 秒。

**自检。**
`python lyric_maker.py --selftest` —— 不需要音频，验证对齐算法、歌词解析、LRC 格式。
