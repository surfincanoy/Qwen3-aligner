# Copyright 2026 Alibaba Cloud (Qwen3-ASR)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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


def sentences_to_srt(sentences: list[dict]) -> str:
    lines = []
    for i, sent in enumerate(sentences, 1):
        start = format_time(sent["start_time"])
        end = format_time(sent["end_time"])
        lines.append(f"{i}\n{start} --> {end}\n{sent['text']}\n")
    return "\n".join(lines)


def merge_sentences(sentences: list[dict]) -> list[dict]:
    if len(sentences) < 2:
        return list(sentences)
    result = [dict(s) for s in sentences]
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(result):
            s = result[i]
            text = s.get("text", "").strip()

            # Merge right: sentence ending with ? (merge left from next's perspective)
            if i + 1 < len(result):
                nxt = result[i + 1]
                nxt_text = nxt.get("text", "").strip()
                if nxt_text.endswith("?") and len(nxt_text) < 20:
                    s["text"] = text + " " + nxt_text
                    s["end_time"] = nxt["end_time"]
                    if "words" in s and "words" in nxt:
                        s["words"].extend(nxt["words"])
                    result.pop(i + 1)
                    changed = True
                    continue

            # Merge right: comma + short
            if text.endswith(",") and len(text) < 20 and i + 1 < len(result):
                nxt = result[i + 1]
                s["text"] = text + " " + nxt["text"].strip()
                s["end_time"] = nxt["end_time"]
                if "words" in s and "words" in nxt:
                    s["words"].extend(nxt["words"])
                result.pop(i + 1)
                changed = True
                continue

            i += 1
    return result


def sentences_to_srt(sentences: list[dict]) -> str:
    lines = []
    for i, sent in enumerate(sentences, 1):
        start = format_time(sent["start_time"])
        end = format_time(sent["end_time"])
        lines.append(f"{i}\n{start} --> {end}\n{sent['text']}\n")
    return "\n".join(lines)


def sentences_to_entries(sentences: list[dict], min_duration: float = 0.0) -> list[dict]:
    """Convert aligned sentences to SRT entries.

    Broken entries (empty text or zero duration) are NOT merged into
    the previous entry; they get a minimum duration and appear as
    separate subtitles so the text is never corrupted by concatenation.
    """
    if not sentences:
        return []

    merged: list[dict] = []
    for s in sentences:
        start = float(s.get("start_time", 0.0))
        end = float(s.get("end_time", start))
        text = (s.get("text", "") or "").strip()
        duration = max(0.0, end - start)

        is_broken = (not text) or (end <= start and min_duration == 0)

        if is_broken:
            if end <= start:
                end = start + max(0.2, min_duration)
            if not text:
                continue  # drop completely empty entries
            merged.append({"start_time": start, "end_time": end, "text": text})
        else:
            merged.append({"start_time": start, "end_time": end, "text": text})

    # Fix CTC-compressed entries (too short per character).
    # Extend end_time to a more readable duration without
    # overlapping the next entry.
    for i, entry in enumerate(merged):
        start = entry["start_time"]
        end = entry["end_time"]
        text = entry["text"]
        clean = re.sub(r"[^\w]", "", text)
        n = len(clean)
        if n >= 2:
            dur = end - start
            if dur / n < 0.1:  # <100ms/char => compressed
                desired = max(0.5, n * 0.25)  # at least 500ms or 250ms/char
                new_end = start + desired
                if i + 1 < len(merged):
                    new_end = min(new_end, merged[i + 1]["start_time"])
                if new_end > end + 1e-6:
                    entry["end_time"] = new_end

    return [
        {
            "index": i + 1,
            "start_time": m["start_time"],
            "end_time": m["end_time"],
            "text": m["text"],
        }
        for i, m in enumerate(merged)
    ]



