<div align="center">

# 🎧 Qwen3-aligner

**音频转录 · 字幕对齐 · SRT 修复**

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)]()
[![Gradio](https://img.shields.io/badge/Gradio-6.x-F97316?style=for-the-badge&logo=gradio&logoColor=white)]()
[![Qwen](https://img.shields.io/badge/Qwen3--ASR-6B_|_1.7B-6B21F2?style=for-the-badge&logo=alibabacloud&logoColor=white)]()
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge&logo=open-source-initiative&logoColor=white)]()

[📖 介绍](#-介绍) •
[🚀 快速开始](#-快速开始) •
[💻 CLI 指南](#-cli-指南) •
[🖥️ GUI 指南](#️-gui-指南) •
[📦 项目结构](#-项目结构) •
[🤖 模型说明](#-模型说明)

---

https://github.com/user-attachments/assets/7b1ee0c5-f9d8-4d3e-a8ad-4f177de44b60

</div>

## 📖 介绍

Qwen3-aligner 是基于通义千问 **Qwen3-ASR** 与 **Qwen3-ForcedAligner** 的一站式音频处理工具。它将强大的语音识别能力封装为简洁的 CLI 和友好的 GUI，让音频处理变得简单高效。

| 能力 | 说明 |
|------|------|
| 🎤 **音频转录** | 将音频/视频文件转为文字，支持 30+ 种语言 |
| 📝 **字幕对齐** | 将已有文本与音频对齐，生成 SRT 字幕文件 |
| 🔧 **SRT 修复** | 修复字幕中指定行的错误时间戳 |

---

## 🚀 快速开始

### 环境要求

| 依赖 | 版本要求 | 说明 |
|------|----------|------|
| Python | ≥ 3.12 | 推荐 3.12 |
| CUDA | ≥ 12.0 | GPU 加速（推荐） |
| ffmpeg | 系统安装 | 音频/视频处理 |
| 显存 | ≥ 4GB | 0.6B 模型 |
| 显存 | ≥ 6GB | 1.7B 模型 |

### 安装步骤

<details>
<summary><b>📦 基础安装</b></summary>

```bash
# 克隆项目
git clone https://github.com/your/Qwen3-aligner.git
cd Qwen3-aligner

# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt

# 确认 ffmpeg 可用
ffmpeg -version
```

</details>

<details>
<summary><b>🐳 Docker 部署（可选）</b></summary>

```dockerfile
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["python", "gui/app.py"]
```

</details>

---

## 💻 CLI 指南

### 🎤 `transcribe` — 音频转录

将音频/视频文件中的语音转为文字。

```bash
python main.py transcribe <文件> [语言] [选项]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `文件` | 音频或视频文件路径 | **必填** |
| `语言` | 识别语言（English / Japanese / Chinese / ...） | `English` |
| `-m` | 模型大小：`0.6B`（快速）或 `1.7B`（精确） | `0.6B` |
| `--segment-length` | 音频分段长度（秒） | `180` |
| `--output` | 输出文本文件路径 | `文件名.txt` |

```bash
# 快速转录中文音频
python main.py transcribe meeting.mp3 Chinese -m 0.6B --output meeting.txt

# 高精度转录日语
python main.py transcribe interview.wav Japanese -m 1.7B

# 指定分段长度（处理长音频）
python main.py transcribe podcast.mp3 English --segment-length 300
```

---

### 📝 `align` — 文本对齐

将已有文本与音频对齐，生成带精确时间戳的 SRT 字幕。

```bash
python main.py align <文件> <文本> <语言> [选项]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `文件` | 音频或视频文件路径 | **必填** |
| `文本` | 文本文件路径 或 直接输入文本 | **必填** |
| `语言` | 对齐语言：`English` / `Japanese` / `Chinese` | **必填** |
| `--output` | 输出 SRT 文件路径 | 不保存 |

```bash
# 英文对齐
python main.py align speech.wav transcript.txt English --output speech.srt

# 日文对齐（文本直接输入）
python main.py align audio.wav "こんにちは、今日はいい天気ですね。" Japanese

# 中文对齐
python main.py align lecture.mp3 讲义.txt Chinese --output 字幕.srt
```

---

### 🔧 `fix-srt` — SRT 修复

当 SRT 文件中某些行的时间戳不准确时，指定行号重新对齐修复。

```bash
python main.py fix-srt <文件> <字幕> <行号...> [选项]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `文件` | 原始音频或视频文件 | **必填** |
| `字幕` | 待修复的 SRT 文件 | **必填** |
| `行号` | 错误行号（支持多种格式） | **必填** |
| `--language` | 语言 | `English` |
| `--output` | 输出文件路径 | `原文件_fixed.srt` |

行号格式灵活：

```bash
# 空格分隔
python main.py fix-srt audio.wav sub.srt 37 38 39 --language Japanese

# 逗号分隔
python main.py fix-srt audio.wav sub.srt 37,38,39

# 列表格式
python main.py fix-srt audio.wav sub.srt [37,38,39]

# 混合连续行号和单行号
python main.py fix-srt audio.wav sub.srt 15 16 17 23 24 --output fixed.srt
```

---

## 🖥️ GUI 指南

图形界面基于 [Gradio](https://gradio.app/)，无需记忆命令，拖拽上传即可使用。

### 启动

```bash
# 默认中文界面
.venv/bin/python gui/app.py

# 英文界面
.venv/bin/python gui/app.py --lang English
```

启动后浏览器自动打开 `http://localhost:7860`。

### 界面一览

```
┌──────────────────────────────────────────────────────────────┐
│  🎧 Qwen3 音频处理工坊                                        │
│  ┌─────────────────────────────────────┬─────────────────────┐ │
│  │  🌐 中文 / English                  │                     │ │
│  ├─────────────────────────────────────┤                     │ │
│  │  🚀 快速转录/对齐  │  🔧 SRT 修复   │                     │ │
│  ├─────────────────────────────────────┤                     │ │
│  │  Step 1: 音频转录                    │                     │ │
│  │  ┌─────────────────────────────────┐│                     │ │
│  │  │    拖拽或点击上传音频文件        ││                     │ │
│  │  └─────────────────────────────────┘│                     │ │
│  │  [English ▾]  [⚡快速 0.6B]         │                     │ │
│  │  [ ▶ 开始转录 ]                     │                     │ │
│  │  ┌─────────────────────────────────┐│                     │ │
│  │  │ 转录结果预览...                  ││                     │ │
│  │  └─────────────────────────────────┘│                     │ │
│  │  ─────────────────────────────────── │                     │ │
│  │  Step 2: 生成字幕（可选）            │                     │ │
│  │  ┌─────────────────────────────────┐│                     │ │
│  │  │ 粘贴文本或等待转录结果自动填充   ││                     │ │
│  │  └─────────────────────────────────┘│                     │ │
│  │  [ 🎬 生成 SRT 字幕 ]              │                     │ │
│  └─────────────────────────────────────┴─────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### 功能说明

| 标签页 | 工作流程 |
|--------|----------|
| 🚀 **快速转录/对齐** | **Step 1** 上传音频 → 选择语言 → 开始转录 → **Step 2** 自动填充文本 → 生成 SRT 字幕 |
| 🔧 **SRT 修复** | 上传音频 + SRT 文件 → 输入错误行号 → 一键修复 → 预览并下载 |

---

## 📦 项目结构

```
Qwen3-aligner/
│
├── 📄 main.py                  # CLI 入口（唯一根级启动文件）
│
├── 📁 qwen3_aligner/           # 核心库包
│   ├── __init__.py             # 公共 API 导出
│   ├── aligner.py              # 统一对齐器（EN / JA / ZH）
│   ├── audio_utils.py          # 音频加载 · 视频提音轨 · VAD 分割
│   ├── text_utils.py           # 文本分割（英文缩写感知 / 中日文标点）
│   ├── srt_utils.py            # SRT 解析 · 格式化 · 保存
│   ├── model_loader.py         # 模型加载（本地 → HuggingFace → ModelScope）
│   ├── transcribe.py           # ASR 音频转录
│   └── fix_srt.py              # SRT 时间戳修复
│
├── 📁 gui/                     # Gradio 图形界面
│   ├── app.py                  # 入口（支持 --lang 参数）
│   ├── i18n.py                 # 中英双语国际化
│   ├── worker.py               # 后端任务调度
│   └── tabs/                   # 标签页组件
│       ├── main_tab.py         # 快速转录/对齐
│       └── fixsrt_tab.py       # SRT 修复
│
├── 📁 Qwen/                    # 本地模型文件（自动下载）
├── 📁 FireRedVAD/              # VAD 人声检测模型
├── 📁 example/                 # 示例音频与字幕
├── 📁 docs/                    # 设计文档
│
├── 📄 requirements.txt         # Python 依赖清单
└── 📄 README.md                # 📖 本文件
```

---

## 🤖 模型说明

| 模型 | 用途 | 参数量 | 推荐显存 |
|------|------|--------|----------|
| <img src="https://img.shields.io/badge/Qwen3--ASR-0.6B-6B21F2?style=flat-square"> | 音频转录（快速模式） | ~6 亿 | ≥ 4GB |
| <img src="https://img.shields.io/badge/Qwen3--ASR-1.7B-6B21F2?style=flat-square"> | 音频转录（精确模式） | ~17 亿 | ≥ 8GB |
| <img src="https://img.shields.io/badge/Qwen3--Aligner-0.6B-0891B2?style=flat-square"> | 文本对齐 / SRT 修复 | ~6 亿 | ≥ 4GB |

**下载策略**（自动回退）：
1. 优先加载 `Qwen/` 目录下的本地模型
2. 从 HuggingFace 下载并缓存到本地
3. 从 ModelScope 下载（国内用户加速）

---

## ⚙️ 注意事项

<table>
<tr>
<th width="120">项目</th>
<th>说明</th>
</tr>
<tr>
<td>🎵 音频格式</td>
<td>mp3 · wav · flac · m4a · ogg · wma · aac</td>
</tr>
<tr>
<td>🎬 视频格式</td>
<td>mp4 · mkv · avi · mov · flv · webm（自动提取音轨）</td>
</tr>
<tr>
<td>⏱️ 长音频</td>
<td>超过 180s 自动分段，VAD 检测人声边界切割</td>
</tr>
<tr>
<td>🧠 显存占用</td>
<td>0.6B ≈ 4GB · 1.7B ≈ 8GB · 可加 `--segment-length` 降低占用</td>
</tr>
<tr>
<td>🐍 运行环境</td>
<td>建议使用 `.venv/` 虚拟环境隔离依赖</td>
</tr>
</table>

---

<div align="center">

**Qwen3-aligner** · 基于 Qwen3 生态构建

</div>
