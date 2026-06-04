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
        if char in separators:
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
    separators = [".", "?", "!"]
    segments = []
    last_sep_idx = -1
    for i, letter in enumerate(text):
        next_letter = text[i + 1] if i + 1 < len(text) else ""
        if letter in separators and next_letter == " ":
            segment = text[last_sep_idx + 1: i + 1].strip()
            if segment:
                segments.append(segment)
            last_sep_idx = i + 1
    if last_sep_idx < len(text):
        final_segment = text[last_sep_idx + 1:].strip()
        if final_segment:
            segments.append(final_segment)
    segments = [s for s in segments if s]
    merged = []
    i = 0
    while i < len(segments):
        seg = segments[i].strip()
        if len(seg) <= 15 and i + 1 < len(segments):
            combined = seg
            short_count = 1
            j = i + 1
            while j < len(segments) and short_count < 3:
                next_seg = segments[j].strip()
                if len(next_seg) <= 15:
                    combined = f"{combined} {next_seg}"
                    short_count += 1
                    j += 1
                else:
                    break
            if short_count < 3 and j < len(segments):
                next_seg = segments[j].strip()
                if len(next_seg) <= 20:
                    combined = f"{combined} {next_seg}"
                    j += 1
            merged.append(combined)
            i = j
        else:
            merged.append(seg)
            i += 1
    return merged


def split_text_by_punctuation(text: str, punctuation: list[str] = None) -> list[str]:
    if punctuation is None:
        punctuation = [
            "、", "。", "！", "？", ",", ".", "?", "，", "!", "\n", "\t", "\r", "──",
        ]
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
