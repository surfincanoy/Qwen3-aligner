<div align="center">

# 🎧 Qwen3-aligner

**音声文字起こし · 字幕アライメント · SRT 修正 · 一括処理**

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

[🇨🇳 中文](README_zh.md) · [🇬🇧 English](README.md)

</div>

## 📖 &nbsp;はじめに

Qwen3-aligner は、阿里巴巴（アリババ）通義千問の **Qwen3-ASR** および **Qwen3-ForcedAligner** モデルをベースにしたオールインワン音声処理ツールです。単一ファイルの文字起こし・アライメント、一括処理、SRT タイムスタンプ修正に対応し、**日中バイリンガル Gradio UI** を内蔵。音声の長さに制限はありません。

| 機能 | 説明 |
|------|------|
| 🎤 **文字起こし** | 音声・動画ファイルをテキストや字幕に変換 |
| 📝 **字幕アライメント** | 既存のテキストを音声に同期させ、正確な SRT 字幕を生成 |
| 🔧 **SRT 修正** | 指定した行番号のタイムスタンプを再アライメントして修正 |
| 📦 **一括処理** | 複数ファイルの文字起こし・アライメント・マッチングをワンクリックで実行 |

<br>

---

## 🚀 &nbsp;クイックスタート

### 📋 動作環境

| 依存環境 | 説明 |
|---------|------|
| Python | ≥ 3.12 |
| CUDA | ≥ 12.0（GPU アクセラレーション推奨） |
| ffmpeg | 音声・動画処理 |
| VRAM | ≥ 6GB（ASR + Align モデル） |

### 🔧 インストール

```bash
uv venv --python 3.12
# source .venv/bin/activate   # Linux/macOS

uv pip install -r requirements.txt    # uv 推奨

ffmpeg -version   # ffmpeg が利用可能か確認
```

### ▶️ 起動

```bash
# 方法 1：Python から直接起動
uv run gui/app.py

```

ブラウザが自動的に **http://localhost:7860** を開きます。

<br>

---

## 🖥️ &nbsp;GUI ガイド

### 📑 タブ一覧

| タブ | 機能 |
|------|------|
| 🚀 **文字起こし/アライメント** | 単一ファイル：音声アップロード → 文字起こし → アライメント → SRT 生成 |
| 📦 **一括文字起こし/アライメント** | 複数ファイル：一括文字起こし + アライメント + ZIP ダウンロード |
| 📦 **一括マッチング** | TXT + 音声ペアを一括アライメントして SRT を生成 |
| 🔧 **SRT 修正** | 音声 + SRT をアップロードし、問題のある行番号を指定して再アライメント |

### ▶️ ワークフロー

**クイック文字起こし & アライメント**

```
Step 1  音声をアップロード → 言語を選択 → モデルを選択 → 文字起こし → 誤認識の修正または再セグメント
Step 2  テキストを自動入力または貼り付け → SRT 生成 → ダウンロード
```

**一括文字起こし & アライメント**

```
複数の音声ファイルをアップロード → 言語/モデルを選択 → ワンクリック実行
→ すべて自動文字起こし → すべて自動アライメント → ZIP ダウンロード
```

**一括マッチング**

```
TXT + 同名の音声ファイルをアップロード → 言語を選択 → 一括アライメント → ZIP ダウンロード
```

### ⚡ Flash Attention 加速

右上の **Flash Attention 加速** チェックボックスをオンにすると、対応ハードウェアで FA2 アクセラレーションが有効になります（NVIDIA Turing SM 7.5+、RTX 20（一部）/30/40 シリーズ、一部の GTX 16 シリーズを含む）。

<br>

---

## 🤖 &nbsp;モデル説明

| モデル | 用途 | VRAM |
|-------|------|------|
| `Qwen3-ASR-0.6B` | 文字起こし（高速） | ≥ 6GB |
| `Qwen3-ASR-1.7B` | 文字起こし（高精度） | ≥ 6GB |
| `Qwen3-ForcedAligner-0.6B` | アライメント / 修正 | ≥ 6GB |

**ロード戦略**（自動フォールバック）：
1. 最初に `Qwen/` ディレクトリのローカルモデルを読み込む
2. **HuggingFace** からダウンロードし、ローカルにキャッシュ
3. **ModelScope** からダウンロード（中国国内のユーザー向け高速化）

<br>

---

## ⚙️ &nbsp;注意事項

| 項目 | 説明 |
|------|------|
| 🎵 **音声形式** | mp3 · wav · flac · m4a · ogg · wma · aac |
| 🎬 **動画形式** | mp4 · mkv · avi · mov · flv · webm（音声を自動抽出） |
| ⏱️ **長音声** | 自動セグメント分割、長さ制限なし |
| 🧠 **VRAM** | ≥ 6GB |
| ⚡ **Flash Attention** | Pascal アーキテクチャ（GTX 1060 等）では非対応。チェックしても無効ですがクラッシュしません |

<br>

---

<div align="center">

**Qwen3-aligner** · Qwen3 エコシステムに基づいて構築

</div>
