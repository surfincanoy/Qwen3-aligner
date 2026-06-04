#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os

from qwen3_aligner import align_text, fix_srt_file
from qwen3_aligner import (
    clear_memory_cache,
    load_and_segment_audio,
    load_asr_model,
    transcribe_audio,
)
from qwen3_aligner.srt_utils import save_srt_entries, sentences_to_entries


def read_text_input(text_input: str) -> str:
    if os.path.exists(text_input) and os.path.isfile(text_input):
        with open(text_input, "r", encoding="utf-8") as f:
            return f.read()
    return text_input


def run_align(args: argparse.Namespace) -> int:
    text = read_text_input(args.text_input)
    sentences = align_text(args.audio_path, text, language=args.language)
    if args.output:
        entries = sentences_to_entries(sentences)
        save_srt_entries(entries, args.output)

    print(f"\n对齐完成，共 {len(sentences)} 个句子")
    return 0


def run_fix_srt(args: argparse.Namespace) -> int:
    bad_indices = []
    for token in args.bad_indices:
        if token.startswith("[") and token.endswith("]"):
            inner = token[1:-1]
            for part in inner.split(","):
                part = part.strip()
                if part.isdigit():
                    bad_indices.append(int(part))
        elif "," in token:
            for part in token.split(","):
                part = part.strip()
                if part.isdigit():
                    bad_indices.append(int(part))
        elif token.isdigit():
            bad_indices.append(int(token))
        else:
            raise ValueError(f"无法解析错误行号: {token}")

    if not bad_indices:
        raise ValueError("需要至少一个错误行号")

    output_path = fix_srt_file(
        args.audio_path,
        args.srt_path,
        bad_indices,
        language=args.language,
        output_path=args.output,
    )
    print(f"\n修复完成，输出文件: {output_path}")
    return 0


def run_transcribe(args: argparse.Namespace) -> int:
    if not os.path.exists(args.audio_path):
        raise FileNotFoundError(f"音频文件不存在: {args.audio_path}")

    chunks = load_and_segment_audio(
        args.audio_path, segment_length_s=args.segment_length
    )
    model_name = f"Qwen/Qwen3-ASR-{args.model}"
    clear_memory_cache()
    model = load_asr_model(model_name)
    results = transcribe_audio(chunks, model, language=args.language)
    del model
    clear_memory_cache()

    full_text = " ".join([r["text"] for r in results])
    if args.output:
        output_path = args.output
    else:
        base, _ = os.path.splitext(args.audio_path)
        output_path = f"{base}.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"\n转录完成，共 {len(full_text)} 字符")
    print(f"已保存: {output_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="统一调用 English/Japanese/Chinese 对齐和 SRT 修复功能"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_align = subparsers.add_parser("align", help="通用语言对齐，可指定语言")
    p_align.add_argument("audio_path", help="音频或视频文件路径")
    p_align.add_argument("text_input", help="文本文件路径或直接文本")
    p_align.add_argument(
        "language",
        help="语言名称",
        choices=["English", "Japanese", "Chinese"],
        default="English",
    )
    p_align.add_argument("--output", help="输出 SRT 文件路径", default=None)
    p_align.set_defaults(func=run_align)

    p_fix = subparsers.add_parser("fix-srt", help="修复 SRT 时间戳")
    p_fix.add_argument("audio_path", help="音频或视频文件路径")
    p_fix.add_argument("srt_path", help="SRT 文件路径")
    p_fix.add_argument(
        "bad_indices", nargs="+", help="错误行号，支持 37 38 39、37,38,39 或 [37,38,39]"
    )
    p_fix.add_argument(
        "--language",
        help="修复语言",
        choices=["English", "Japanese", "Chinese"],
        default="English",
    )
    p_fix.add_argument("--output", help="输出 SRT 文件路径", default=None)
    p_fix.set_defaults(func=run_fix_srt)

    p_trans = subparsers.add_parser("transcribe", help="音频转录")
    p_trans.add_argument("audio_path", help="音频文件路径")
    p_trans.add_argument(
        "language",
        nargs="?",
        default="English",
        help="语言 (默认: English)",
    )
    p_trans.add_argument(
        "-m",
        "--model",
        choices=["0.6B", "1.7B"],
        default="0.6B",
        help="ASR 模型大小 (默认: 0.6B)",
    )
    p_trans.add_argument(
        "--segment-length",
        type=int,
        default=180,
        help="音频分段长度，单位秒 (默认: 180)",
    )
    p_trans.add_argument("--output", help="输出文本文件路径", default=None)
    p_trans.set_defaults(func=run_transcribe)

    args = parser.parse_args()

    try:
        return args.func(args)
    except Exception as e:
        print(f"错误: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
