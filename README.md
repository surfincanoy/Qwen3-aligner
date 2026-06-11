<div align="center">

# 🎧 Qwen3-aligner

**Audio Transcription · Subtitle Alignment · SRT Fix · Batch Processing**

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

[🇨🇳 中文](README_zh.md) · [🇯🇵 日本語](README_ja.md)

</div>

## 📖 &nbsp;Introduction

Qwen3-aligner is an all-in-one audio processing tool powered by **Qwen3-ASR** and **Qwen3-ForcedAligner** from Alibaba's Tongyi Qianwen series. It supports single-file transcription & alignment, batch processing, SRT timestamp repair, and comes with a **bilingual (Chinese/English) Gradio UI**. No limit on audio duration.

| Feature | Description |
|---------|-------------|
| 🎤 **Transcribe** | Convert audio/video to text and subtitles |
| 📝 **Align** | Align existing text with audio to generate precise SRT subtitles |
| 🔧 **SRT Fix** | Re-align specific subtitle lines to fix incorrect timestamps |
| 📦 **Batch Processing** | Batch transcribe, align, and match multiple files in one click |

<br>

---

## 🚀 &nbsp;Quick Start

### 📋 Requirements

| Dependency | Notes |
|------------|-------|
| Python | ≥ 3.12 |
| CUDA | ≥ 12.0 (GPU acceleration recommended) |
| ffmpeg | Audio/video processing |
| VRAM | ≥ 6GB (ASR + Align models) |

### 🔧 Installation

```bash
uv venv --python 3.12
# source .venv/bin/activate   # Linux/macOS

uv pip install -r requirements.txt    # uv recommended

ffmpeg -version   # verify ffmpeg is available
```

### ▶️ Launch

```bash
# Option 1: Python direct launch
uv run gui/app.py
```

The browser will automatically open **http://localhost:7860**.

<br>

---

## 🖥️ &nbsp;GUI Guide

### 📑 Tabs Overview

| Tab | Function |
|-----|----------|
| 🚀 **Transcribe/Align** | Single file: upload audio → transcribe → align → generate SRT |
| 📦 **Batch Transcribe/Align** | Multiple files: batch transcribe + align + ZIP download |
| 📦 **Batch Align** | TXT + audio pairs: batch align to generate SRT |
| 🔧 **SRT Fix** | Upload audio + SRT, specify bad line numbers to re-align |

### ▶️ Workflow

**Quick Transcribe & Align**

```
Step 1  Upload audio → select language → select model → transcribe → correct errors or re-segment
Step 2  Auto-fill or paste text → generate SRT → download
```

**Batch Transcribe & Align**

```
Upload multiple audio files → select language/model → one-click execute
→ auto-transcribe all → auto-align all → ZIP download
```

**Batch Align**

```
Upload TXT + matching audio files → select language → batch align → ZIP download
```

### ⚡ Flash Attention

Check the **Flash Attention** checkbox in the top-right corner to enable FA2 acceleration on supported hardware (NVIDIA Turing SM 7.5+, including RTX 20/30/40 series, some GTX 16 series).

<br>

---

## 🤖 &nbsp;Model Details

| Model | Purpose | VRAM |
|-------|---------|------|
| `Qwen3-ASR-0.6B` | Transcription (fast) | ≥ 6GB |
| `Qwen3-ASR-1.7B` | Transcription (accurate) | ≥ 6GB |
| `Qwen3-ForcedAligner-0.6B` | Alignment / Fix | ≥ 6GB |

**Loading strategy** (automatic fallback):
1. Load local model from `Qwen/` directory first
2. Download from **HuggingFace** and cache locally
3. Download from **ModelScope** (faster for mainland China users)

<br>

---

## ⚙️ &nbsp;Notes

| Item | Notes |
|------|-------|
| 🎵 **Audio formats** | mp3 · wav · flac · m4a · ogg · wma · aac |
| 🎬 **Video formats** | mp4 · mkv · avi · mov · flv · webm (audio auto-extracted) |
| ⏱️ **Long audio** | Auto-segmented, no duration limit |
| 🧠 **VRAM** | ≥ 6GB |
| ⚡ **Flash Attention** | Not supported on Pascal architecture (e.g. GTX 1060); checking it has no effect but won't crash |

<br>

---

<div align="center">

**Qwen3-aligner** · Built on the Qwen3 ecosystem

</div>
