import os

from qwen3_aligner.aligner import Aligner
from qwen3_aligner.audio_utils import WAV_SAMPLE_RATE, ensure_audio, load_audio
from qwen3_aligner.srt_utils import (
    load_srt,
    save_srt_entries,
    join_text_segments,
)


def fix_srt_line_timestamps(
    audio_path: str,
    srt_path: str,
    bad_indices: list[int],
    language: str = "English",
    output_path: str = None,
):
    if not bad_indices:
        raise ValueError("Need at least one bad index")

    audio_path = ensure_audio(audio_path)
    aligner = Aligner(language=language)
    aligner.load_model()

    entries = load_srt(srt_path)
    index_map = {entry["index"]: entry for entry in entries}
    bad_indices = sorted(set(bad_indices))
    audio = load_audio(audio_path)

    consecutive_groups = []
    single_indices = []
    i = 0
    while i < len(bad_indices):
        is_consecutive = False
        if i + 1 < len(bad_indices) and bad_indices[i] + 1 == bad_indices[i + 1]:
            is_consecutive = True
        if is_consecutive:
            start = bad_indices[i]
            end = start
            while i < len(bad_indices) - 1 and bad_indices[i] + 1 == bad_indices[i + 1]:
                end = bad_indices[i + 1]
                i += 1
            consecutive_groups.append((start, end))
        else:
            single_indices.append(bad_indices[i])
        i += 1

    print(f"Consecutive groups: {consecutive_groups}, Single: {single_indices}")

    for first_bad, last_bad in consecutive_groups:
        prev_entry = index_map.get(first_bad - 1)
        next_entry = index_map.get(last_bad + 1)
        if prev_entry is None or next_entry is None:
            raise ValueError(f"Cannot fix consecutive {first_bad}-{last_bad}: need prev and next")

        audio_start = prev_entry["start_time"]
        audio_end = next_entry["end_time"]
        start_sample = int(audio_start * WAV_SAMPLE_RATE)
        end_sample = int(audio_end * WAV_SAMPLE_RATE)
        audio_segment = audio[start_sample:end_sample]

        text_segments = [prev_entry["text"].replace("\n", " ")]
        current_group = list(range(first_bad, last_bad + 1))
        for bad_idx in current_group:
            text_segments.append(index_map[bad_idx]["text"].replace("\n", " "))
        text_segments.append(next_entry["text"].replace("\n", " "))
        full_text = join_text_segments(text_segments, language)

        print(f"Re-aligning sentences {first_bad - 1} to {last_bad + 1}")
        results = aligner.model.align(
            audio=(audio_segment, WAV_SAMPLE_RATE),
            text=full_text,
            language=language,
        )
        word_timestamps = [
            {
                "text": item.text,
                "start_time": float(item.start_time) + audio_start,
                "end_time": float(item.end_time) + audio_start,
            }
            for item in results[0]
        ]
        matched = aligner._words_to_sentences(word_timestamps, text_segments)
        expected_count = len(current_group) + 2
        if len(matched) != expected_count:
            raise RuntimeError(
                f"Re-alignment of {current_group} generated {len(matched)} sentences, expected {expected_count}"
            )

        prev_entry["start_time"] = matched[0]["start_time"]
        prev_entry["end_time"] = matched[0]["end_time"]
        for j, bad_idx in enumerate(current_group):
            index_map[bad_idx]["start_time"] = matched[j + 1]["start_time"]
            index_map[bad_idx]["end_time"] = matched[j + 1]["end_time"]
        next_entry["start_time"] = matched[-1]["start_time"]
        next_entry["end_time"] = matched[-1]["end_time"]
        print(f"Fixed consecutive: {current_group}")

    for bad_index in single_indices:
        if bad_index not in index_map:
            raise ValueError(f"Index {bad_index} not found in SRT")
        bad_entry = index_map[bad_index]
        prev_entry = index_map.get(bad_index - 1)
        next_entry = index_map.get(bad_index + 1)
        if prev_entry is None or next_entry is None:
            raise ValueError(f"Index {bad_index}: need prev and next entries")

        print(f"Fixing index {bad_index}")
        start_sample = int(prev_entry["start_time"] * WAV_SAMPLE_RATE)
        end_sample = int(next_entry["end_time"] * WAV_SAMPLE_RATE)
        audio_segment = audio[start_sample:end_sample]

        text_segments = [
            prev_entry["text"].replace("\n", " "),
            bad_entry["text"].replace("\n", " "),
            next_entry["text"].replace("\n", " "),
        ]
        full_text = join_text_segments(text_segments, language)

        results = aligner.model.align(
            audio=(audio_segment, WAV_SAMPLE_RATE),
            text=full_text,
            language=language,
        )
        word_timestamps = [
            {
                "text": item.text,
                "start_time": float(item.start_time) + prev_entry["start_time"],
                "end_time": float(item.end_time) + prev_entry["start_time"],
            }
            for item in results[0]
        ]
        matched = aligner._words_to_sentences(word_timestamps, text_segments)
        if len(matched) != 3:
            raise RuntimeError(f"Re-alignment of {bad_index} generated {len(matched)} sentences, expected 3")

        prev_entry["start_time"] = matched[0]["start_time"]
        prev_entry["end_time"] = matched[0]["end_time"]
        bad_entry["start_time"] = matched[1]["start_time"]
        bad_entry["end_time"] = matched[1]["end_time"]
        next_entry["start_time"] = matched[2]["start_time"]
        next_entry["end_time"] = matched[2]["end_time"]
        print(f"Fixed index {bad_index}")

    base, ext = os.path.splitext(srt_path)
    if output_path is None:
        output_path = base + "x" + ext
    save_srt_entries(entries, output_path)
    print(f"Saved fixed SRT to {output_path}")


def fix_srt_file(
    audio_path: str,
    srt_path: str,
    bad_indices: list[int],
    language: str = "English",
    output_path: str = None,
) -> str:
    fix_srt_line_timestamps(audio_path, srt_path, bad_indices, language, output_path)
    if output_path is None:
        base, ext = os.path.splitext(srt_path)
        output_path = base + "x" + ext
    return output_path
