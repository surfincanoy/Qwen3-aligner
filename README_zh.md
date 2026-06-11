<div align="center">

# 🎧 Qwen3-aligner

**音频转录 · 字幕对齐 · SRT 修复 · 批量处理**

<br>

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Gradio](https://img.shields.io/badge/Gradio-6.5-F97316?logo=gradio&logoColor=white)](https://gradio.app)
[![Qwen3-ASR](https://img.shields.io/badge/Qwen3--ASR-0.6B%20|%201.7B-6B21F2?logo=alibabacloud&logoColor=white)](https://huggingface.co/Qwen)
[![Qwen3-Aligner](https://img.shields.io/badge/Qwen3--Aligner-0.6B-0891B2?logo=alibabacloud&logoColor=white)](https://huggingface.co/Qwen)
[![License](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)

</div>

<br>

---

<div align="center">

[🇬🇧 English](README.md) · [🇯🇵 日本語](README_ja.md)

</div>

## 📖 &nbsp;介绍

Qwen3-aligner 基于通义千问 **Qwen3-ASR** 与 **Qwen3-ForcedAligner** 模型，提供一站式音频处理能力。支持单文件转录对齐、批量处理、SRT 时间戳修复，并内置 **中英双语 Gradio 界面**，对音频无时长限制。

| 功能 | 说明 |
|------|------|
| 🎤 **音频转录** | 将音频/视频转为文本和字幕 |
| 📝 **字幕对齐** | 将已有文本与音频对齐，生成精准 SRT 字幕 |
| 🔧 **SRT 修复** | 指定行号重新对齐，修复错误时间戳 |
| 📦 **批量处理** | 一键转录/对齐/匹配多个文件 |

<br>

---

## 🚀 &nbsp;快速开始

### 📋 环境要求

| 依赖 | 说明 |
|------|------|
| Python | ≥ 3.12 |
| CUDA | ≥ 12.0（GPU 加速，推荐） |
| ffmpeg | 音频/视频处理 |
| 显存 | ≥ 6GB（ASR + Align 模型） |

### 🔧 安装

```bash
uv venv --python 3.12
# source .venv/bin/activate   # Linux/macOS

uv pip install -r requirements.txt    # 推荐 uv

ffmpeg -version   # 确认 ffmpeg 可用
```

### ▶️ 启动

```bash
# 方式一：Python 直接启动
uv run gui/app.py

```

浏览器自动打开 **http://localhost:7860**。

<br>

---

## 🖥️ &nbsp;GUI 指南

### 📑 标签页一览

| 标签页 | 功能 |
|--------|------|
| 🚀 **转录/对齐** | 单文件：上传音频 → 转录文本 → 对齐生成 SRT |
| 📦 **批量转录/对齐** | 多文件：一键完成所有音频的转写 + 对齐 + ZIP 打包 |
| 📦 **批量匹配** | 已有 TXT + 音频，批量对齐生成 SRT |
| 🔧 **SRT 修复** | 上传音频 + SRT，重新对齐时间戳 |

### ▶️ 使用流程

**快速转录/对齐**

```
Step 1  上传音频 → 选择语言 → 选择模型 → 开始转录 → 纠正识别错误或重新断句
Step 2  自动填充或粘贴文本 → 生成 SRT 字幕 → 下载
```

**批量转录/对齐**

```
上传多个音频 → 选择语言/模型 → 一键执行
→ 自动转录全部 → 自动对齐全部 → ZIP 打包下载
```

**批量匹配**

```
上传多个 TXT + 同名音频 → 选择语言 → 批量对齐 → ZIP 下载
```

### ⚡ Flash Attention 加速

勾选界面右上角 **Flash Attention 加速** 复选框，在支持硬件上启用 FA2 加速（NVIDIA Turing SM 7.5+，含 RTX 20（部分）/30/40 系列，部分 GTX 16 系列）。

<br>

---

## 🤖 &nbsp;模型说明

| 模型 | 用途 | 显存 |
|------|------|------|
| `Qwen3-ASR-0.6B` | 转录（快速） | ≥ 6GB |
| `Qwen3-ASR-1.7B` | 转录（精确） | ≥ 6GB |
| `Qwen3-ForcedAligner-0.6B` | 对齐 / 修复 | ≥ 6GB |

**加载策略**（自动回退）：
1. 优先读取 `Qwen/` 本地已下载模型
2. 从 **HuggingFace** 下载并自动缓存到本地
3. 从 **ModelScope** 下载（国内网络加速）

<br>

---

## ⚙️ &nbsp;注意

| 项目 | 说明 |
|------|------|
| 🎵 **音频格式** | mp3 · wav · flac · m4a · ogg · wma · aac |
| 🎬 **视频格式** | mp4 · mkv · avi · mov · flv · webm（自动提取音轨） |
| ⏱️ **长音频** | 自动分段，不限制音频时长 |
| 🧠 **显存** | ≥ 6GB |
| ⚡ **Flash Attention** | GTX 1060 等 Pascal 架构不支持，勾选无效但不会报错 |

<br>

---

<div align="center">

**Qwen3-aligner** · 基于 Qwen3 生态构建

</div>
