import os

from qwen3_aligner import (
    clear_memory_cache,
    ensure_audio,
    load_and_segment_audio,
    load_asr_model,
    transcribe_audio,
    align_text,
    fix_srt_file,
)
from qwen3_aligner.srt_utils import format_time


def run_transcribe(audio_path: str, language: str, model_size: str) -> tuple[str, str]:
    try:
        audio_path = ensure_audio(audio_path)
        model_name = f"Qwen/Qwen3-ASR-{model_size}"
        chunks = load_and_segment_audio(audio_path, segment_length_s=180)
        model = load_asr_model(model_name)
        results = transcribe_audio(chunks, model, language=language)
        del model
        clear_memory_cache()
        full_text = " ".join([r["text"] for r in results])
        return full_text, ""
    except Exception as e:
        return "", str(e)


def run_align(audio_path: str, text: str, language: str) -> tuple[str, str]:
    try:
        sentences = align_text(audio_path, text, language=language)
        if not sentences:
            return "", "对齐未生成任何结果"

        lines = []
        for i, sent in enumerate(sentences, 1):
            start = format_time(sent["start_time"])
            end = format_time(sent["end_time"])
            lines.append(f"{i}\n{start} --> {end}\n{sent['text']}\n")

        return "\n".join(lines), ""
    except Exception as e:
        return "", str(e)


def run_fixsrt(
    audio_path: str, srt_path: str, bad_indices: list[int], language: str
) -> tuple[str, str]:
    try:
        base, ext = os.path.splitext(srt_path)
        output_path = f"{base}_fixed{ext}"
        fix_srt_file(audio_path, srt_path, bad_indices, language, output_path)

        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()

        return content, ""
    except Exception as e:
        return "", str(e)
