import os
import re


def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def parse_srt_time(time_str: str) -> float:
    hours, minutes, seconds = time_str.split(":")
    seconds, millis = seconds.split(",")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def load_srt(srt_path: str) -> list[dict]:
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    entries = []
    for block in re.split(r"\r?\n\r?\n", content):
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue
        time_line = lines[1].strip()
        match = re.match(
            r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})",
            time_line,
        )
        if not match:
            continue
        start_time = parse_srt_time(match.group(1))
        end_time = parse_srt_time(match.group(2))
        text = "\n".join(lines[2:]).strip()
        entries.append({
            "index": index,
            "start_time": start_time,
            "end_time": end_time,
            "text": text,
        })
    return entries


def save_srt_entries(entries: list[dict], output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in entries:
            start = format_time(entry["start_time"])
            end = format_time(entry["end_time"])
            f.write(f"{entry['index']}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{entry['text']}\n\n")
    print(f"SRT saved: {output_path}")


def sentences_to_srt(sentences: list[dict]) -> str:
    lines = []
    for i, sent in enumerate(sentences, 1):
        start = format_time(sent["start_time"])
        end = format_time(sent["end_time"])
        lines.append(f"{i}\n{start} --> {end}\n{sent['text']}\n")
    return "\n".join(lines)


def sentences_to_entries(sentences: list[dict]) -> list[dict]:
    return [
        {
            "index": i + 1,
            "start_time": s["start_time"],
            "end_time": s["end_time"],
            "text": s["text"],
        }
        for i, s in enumerate(sentences)
    ]


def parse_bad_indices(tokens: list[str]) -> list[int]:
    combined = ",".join(tokens)
    combined = combined.replace("[", "").replace("]", "")
    indices = []
    for num in combined.split(","):
        num = num.strip()
        if num.isdigit():
            indices.append(int(num))
    return sorted(set(indices))


def join_text_segments(text_segments: list[str], language: str) -> str:
    if language in {"Chinese", "Japanese", "Korean", "Thai"}:
        return "".join(text_segments)
    return " ".join(text_segments)
