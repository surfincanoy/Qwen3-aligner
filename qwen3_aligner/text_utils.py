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

import json
import re
from pathlib import Path


def load_abbreviations(json_path: str = None) -> set:
    if json_path is None:
        json_path = Path(__file__).parent.parent / "abbreviations.json"
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(a.lower() for a in data.get("abbreviations", []))
    except FileNotFoundError:
        return set()


def extract_urls_and_emails(text: str) -> list[tuple[int, int, str]]:
    results = []
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    urls = [
        r"https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/[a-zA-Z0-9./_%-]*)?",
        r"http?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/[a-zA-Z0-9./_%-]*)?",
        r"www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(/[a-zA-Z0-9./_%-]*)?",
    ]
    for match in re.finditer(email_pattern, text):
        results.append((match.start(), match.end(), match.group()))
    for url_pattern in urls:
        for match in re.finditer(url_pattern, text):
            if match.start() not in {r[0] for r in results}:
                results.append((match.start(), match.end(), match.group()))
    results.sort(key=lambda x: x[0])
    return results


def is_in_url_or_email(text: str, pos: int, url_list: list) -> bool:
    for start, end, _ in url_list:
        if start <= pos < end:
            return True
    return False


def is_abbreviation_context(text: str, pos: int, abbreviations: set) -> bool:
    if pos == 0 or not text[pos - 1].isalpha():
        return False
    prefix = text[max(0, pos - 10): pos]
    prefix_lower = prefix.lower().strip()
    for abbrev in abbreviations:
        abbrev_clean = abbrev.rstrip(".").lower()
        if len(abbrev_clean) < 2:
            continue
        if prefix_lower.endswith(abbrev_clean):
            idx = prefix_lower.rfind(abbrev_clean)
            if idx > 0 and (prefix[idx - 1] == " " or prefix[idx - 1] in ",;:'\"("):
                return True
    return False


def _looks_like_file_ext(text: str, pos: int) -> bool:
    if pos <= 0 or not text[pos - 1].isalnum():
        return False
    i = pos + 1
    ext_len = 0
    while i < len(text) and text[i].isalpha():
        ext_len += 1
        i += 1
    if ext_len < 1:
        return False
    return i >= len(text) or text[i] in ' \t\n\r,;:)]'


def add_spaces_before_split(text: str, abbreviations: set) -> str:
    url_list = extract_urls_and_emails(text)
    separators = {".", "?", "!"}
    result = []
    i = 0
    while i < len(text):
        if i + 2 < len(text) and text[i:i + 3] == "...":
            result.append("...")
            if i + 3 < len(text) and text[i + 3] != " ":
                result.append(" ")
            i += 3
            continue
        char = text[i]
        if char == "." and _looks_like_file_ext(text, i):
            result.append(char)
        elif char in separators:
            if is_in_url_or_email(text, i, url_list) or is_abbreviation_context(text, i, abbreviations):
                result.append(char)
            else:
                result.append(char)
                if i + 1 < len(text) and text[i + 1] != " ":
                    result.append(" ")
        else:
            result.append(char)
        i += 1
    return "".join(result)


def split_english_text(text: str, abbreviations: set = None) -> list[str]:
    text = text.replace("\n", " ").replace("\t", " ").replace("\r", " ")
    if abbreviations is None:
        abbreviations = load_abbreviations()
    text = add_spaces_before_split(text, abbreviations)
    seps = ['?" ', '!" ', '." ', ". ", "? ", "! ", ", ", "\n", "\t", "\r"]
    segments = []
    last_sep_idx = -1
    i = 0
    while i < len(text):
        matched = False
        for sep in seps:
            if text[i:i+len(sep)] == sep:
                segment = text[last_sep_idx + 1: i + len(sep)].strip()
                if segment:
                    segments.append(segment)
                last_sep_idx = i + len(sep) - 1
                i += len(sep)
                matched = True
                break
        if not matched:
            i += 1
    if last_sep_idx < len(text):
        final_segment = text[last_sep_idx + 1:].strip()
        if final_segment:
            segments.append(final_segment)
    segments = [s for s in segments if s]
    merged = []
    i = 0
    while i < len(segments):
        seg = segments[i]
        if len(seg) < 20 and seg.endswith("?") and merged:
            merged[-1] = merged[-1] + " " + seg
        else:
            merged.append(seg)
        i += 1
    return merged


def split_text_by_punctuation(text: str, punctuation: list[str] = None) -> list[str]:
    if punctuation is None:
        punctuation = [
            '。', '。"', '。”', '，”',
            '！', '！"', '！”',
            '？', '？"', '？”',
            '，', '、',
            '．', '；', '──', "\n", "\t", "\r",
        ]
        # ensure longer (compound) patterns are matched first
        punctuation = sorted(punctuation, key=len, reverse=True)
    text = text.replace("\n", " ").replace("\t", " ").replace("\r", " ")
    pattern = f"({'|'.join(map(re.escape, punctuation))})"
    parts = re.split(pattern, text)
    sentences = []
    current = ""
    for part in parts:
        if part.strip():
            current += part
            if part in punctuation:
                sentences.append(current.strip())
                current = ""
    if current.strip():
        sentences.append(current.strip())
    return [s for s in sentences if s]


def split_text(text: str, language: str) -> list[str]:
    if language == "English":
        return split_english_text(text)
    return split_text_by_punctuation(text)


def match_words_to_sentences(
    sentences: list[str],
    words: list[tuple[str, float, float]],
) -> list[dict]:
    _KEEP = re.compile(r'[^\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\u3005a-zA-Z0-9]')

    def _norm(t: str) -> str:
        return _KEEP.sub('', t).strip()

    result = []
    word_idx = 0
    n_words = len(words)

    for sent in sentences:
        sent_norm = _norm(sent)
        if not sent_norm:
            result.append({"sentence": sent, "start": None, "end": None, "words": []})
            continue

        sent_words = []
        acc_text = ""
        sent_start = None
        sent_end = None

        while word_idx < n_words:
            w_text, w_start, w_end = words[word_idx]
            sent_words.append((w_text, w_start, w_end))
            acc_text += w_text

            if sent_start is None:
                sent_start = w_start
            sent_end = w_end

            word_idx += 1

            acc_norm = _norm(acc_text)
            if len(acc_norm) >= len(sent_norm) and acc_norm.startswith(sent_norm):
                break

        result.append({
            "sentence": sent,
            "start": sent_start,
            "end": sent_end,
            "words": sent_words,
        })

    return result


