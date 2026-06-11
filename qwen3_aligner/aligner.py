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

import os
import re

from tqdm import tqdm

from qwen3_aligner.audio_utils import (
    WAV_SAMPLE_RATE,
    clear_memory_cache,
    create_vad_model,
    ensure_audio,
    load_audio,
    parse_vad_result,
    write_audio_temp,
)
from qwen3_aligner.model_loader import load_aligner_model
from qwen3_aligner.text_utils import split_text

try:
    import fireredvad
    HAS_VAD = True
except ImportError:
    HAS_VAD = False


class Aligner:
    def __init__(self, language: str = "English", attn_implementation: str = None):
        self.language = language
        self.attn_implementation = attn_implementation
        self.model = None
        self.SEGMENT_SECONDS = 90
        self._retried_last_segment = False

    def load_model(self):
        if self.model is not None:
            return
        clear_memory_cache()
        self.model = load_aligner_model(attn_implementation=self.attn_implementation)

    def _unload_model(self):
        if self.model is not None:
            if hasattr(self.model, "model") and hasattr(self.model.model, "to"):
                self.model.model.to("cpu")
                self.model.model = None
            self.model = None
        clear_memory_cache()

    def split_text(self, text: str) -> list[str]:
        return split_text(text, self.language)

    def _join_text(self, segments: list[str]) -> str:
        if self.language in {"Chinese", "Japanese", "Korean", "Thai"}:
            return "".join(segments)
        return " ".join(segments)

    def _clean_text_for_matching(self, text: str) -> str:
        return re.sub(r"[^\w]", "", text).lower()

    def align(self, audio_path: str, text: str) -> list[dict]:
        clear_memory_cache()
        self.load_model()
        audio_path = ensure_audio(audio_path)
        text_segments = self.split_text(text)
        audio = load_audio(audio_path)
        duration = len(audio) / WAV_SAMPLE_RATE
        if duration > self.SEGMENT_SECONDS:
            return self.align_long_audio(audio_path, text)

        full_text = self._join_text(text_segments)
        audio_input = (audio, WAV_SAMPLE_RATE)
        results = self.model.align(
            audio=audio_input,
            text=full_text,
            language=self.language,
        )
        word_timestamps = [
            {
                "text": item.text,
                "start_time": float(item.start_time),
                "end_time": float(item.end_time),
            }
            for item in results[0]
        ]
        sentences, _ = self._words_to_sentences(
            word_timestamps, text_segments, language=self.language
        )
        clear_memory_cache()
        if not sentences:
            expected_sample = text_segments[0][:80] if text_segments else ""
            model_sample = [w["text"] for w in word_timestamps[:10]] if word_timestamps else []
            raise RuntimeError(
                f"模型输出了 {len(word_timestamps)} 个词，但无法与文本句子匹配。\n"
                f"预期文本开头: '{expected_sample}'\n"
                f"模型输出前 10 词: {model_sample}\n"
                f"文本与音频无法互相匹配，程序终止。"
            )
        return sentences

    def _words_to_sentences(
        self,
        word_timestamps: list[dict],
        text_segments: list[str],
        segment_duration: float = None,
        language: str = "English",
    ) -> tuple[list[dict], bool]:
        if not word_timestamps or not text_segments:
            return [], False
        cjk = language in {"Chinese", "Japanese", "Korean", "Thai"}
        sentences = []
        word_idx = 0
        total_words = len(word_timestamps)
        consumed = 0
        overflow_detected = False

        for text_segment in text_segments:
            if word_idx >= total_words:
                remaining_count = len(text_segments) - len(sentences)
                first_failing = (
                    text_segments[len(sentences)]
                    if len(sentences) < len(text_segments)
                    else ""
                )
                print(
                    f"[_words_to_sentences] Word timestamps exhausted after {len(sentences)} sentences; "
                    f"{remaining_count} remaining, first: '{first_failing[:50]}'"
                )
                break
            consumed += 1

            if cjk:
                clean_sent = self._clean_text_for_matching(text_segment)
                if not clean_sent:
                    continue
                target_len = len(clean_sent)
                all_words = []
                current_match_len = 0
                while word_idx < total_words and current_match_len < target_len:
                    word = word_timestamps[word_idx]
                    word_idx += 1
                    word_text = self._clean_text_for_matching(word.get("text", ""))
                    if not word_text:
                        continue
                    all_words.append(word)
                    current_match_len += len(word_text)
                if current_match_len != target_len:
                    if consumed <= 3:
                        print(
                            f"[_words_to_sentences] CJK char count mismatch: "
                            f"sentence='{text_segment[:30]}' target={target_len} "
                            f"got={current_match_len} "
                            f"model_words={[w.get('text', '') for w in all_words[:5]]}"
                        )
                    break
                has_overflow = (
                    segment_duration is not None
                    and segment_duration >= 60
                    and all_words[-1]["end_time"] > 70
                )
                if has_overflow:
                    overflow_detected = True
                    break
                valid_words = [w for w in all_words if w["end_time"] > w["start_time"]]
                if consumed <= 3 and not valid_words:
                    sample_ends = [
                        f"{all_words[i]['end_time']:.3f}"
                        for i in range(min(3, len(all_words)))
                    ]
                    sample_starts = [
                        f"{all_words[i]['start_time']:.3f}"
                        for i in range(min(3, len(all_words)))
                    ]
                    print(
                        f"[_words_to_sentences] No valid words for sentence '{text_segment[:30]}': "
                        f"end_times={sample_ends} start_times={sample_starts}"
                    )
                if not valid_words:
                    break
                sentences.append(
                    {
                        "text": text_segment,
                        "start_time": valid_words[0]["start_time"],
                        "end_time": valid_words[-1]["end_time"],
                        "words": valid_words,
                        "_input_idx": consumed - 1,
                    }
                )
                continue
            else:
                sent_words = [
                    w for w in text_segment.split() if self._clean_text_for_matching(w)
                ]
                if not sent_words:
                    continue
                target_count = len(sent_words)
                all_words = []
                while word_idx < total_words and len(all_words) < target_count:
                    word = word_timestamps[word_idx]
                    word_idx += 1
                    word_text = self._clean_text_for_matching(word.get("text", ""))
                    if not word_text:
                        continue
                    all_words.append(word)
                if len(all_words) != target_count:
                    break
                has_overflow = (
                    segment_duration is not None
                    and segment_duration >= 60
                    and all_words[-1]["end_time"] > 70
                )
                if has_overflow:
                    overflow_detected = True
                    break
                valid_words = [w for w in all_words if w["end_time"] > w["start_time"]]
                if valid_words:
                    sentences.append(
                        {
                            "text": text_segment,
                            "start_time": valid_words[0]["start_time"],
                            "end_time": valid_words[-1]["end_time"],
                            "words": valid_words,
                            "_input_idx": consumed - 1,
                        }
                    )
                    continue
            break
        return sentences, overflow_detected



    def split_audio_for_alignment(self, audio_path: str, segment_seconds: int = None):
        if segment_seconds is None:
            segment_seconds = self.SEGMENT_SECONDS
        wav = load_audio(audio_path)
        total_duration = len(wav) / WAV_SAMPLE_RATE
        print(
            f"Splitting audio: {audio_path} ({total_duration:.2f}s, chunk={segment_seconds}s)"
        )

        if total_duration <= segment_seconds:
            return [
                {
                    "start_time": 0.0,
                    "end_time": total_duration,
                    "start_sample": 0,
                    "end_sample": len(wav),
                    "audio": wav,
                }
            ]

        if not HAS_VAD:
            raise RuntimeError("fireredvad is required for long audio splitting")

        print("Using FireRedVAD for speech detection...")
        tmp_path = write_audio_temp(wav)
        try:
            vad = create_vad_model()
            result = vad.detect(tmp_path)
            speech_timestamps = parse_vad_result(result)
        finally:
            os.unlink(tmp_path)

        segments = []
        current_pos = 0
        while current_pos < total_duration:
            next_boundary = current_pos + segment_seconds
            split_point = next_boundary
            if speech_timestamps:
                for idx, (start, end) in enumerate(speech_timestamps):
                    if start <= next_boundary <= end:
                        if idx + 1 < len(speech_timestamps):
                            # 60s anchor falls inside speech; split at
                            # the next speech interval's start.
                            split_point = speech_timestamps[idx + 1][0]
                        else:
                            # Already in the last speech interval:
                            # no further split needed.
                            split_point = total_duration
                        break
                    elif next_boundary < start:
                        # 60s anchor already in silence; keep it.
                        break
            split_point = min(split_point, total_duration)
            start_sample = int(current_pos * WAV_SAMPLE_RATE)
            end_sample = int(split_point * WAV_SAMPLE_RATE)
            segments.append(
                {
                    "start_time": current_pos,
                    "end_time": split_point,
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "audio": wav[start_sample:end_sample],
                }
            )
            current_pos = split_point
        print(f"Audio split into {len(segments)} segments")
        return segments

    def _get_vad_timestamps(self, audio_path: str) -> list:
        """Run VAD and return speech interval list [(start, end), ...]."""
        if not HAS_VAD:
            return []
        print("Using FireRedVAD for speech detection...")
        wav = load_audio(audio_path)
        tmp_path = write_audio_temp(wav)
        try:
            vad = create_vad_model()
            result = vad.detect(tmp_path)
            return parse_vad_result(result)
        finally:
            os.unlink(tmp_path)

    def align_long_audio(self, audio_path: str, text: str) -> list[dict]:
        clear_memory_cache()
        self.load_model()
        text_segments = self.split_text(text)
        full_audio = load_audio(audio_path)
        total_duration = len(full_audio) / WAV_SAMPLE_RATE

        remaining_segments = list(text_segments)
        all_sentences = []
        current_start = 0.0
        progress = tqdm(total=total_duration, unit="s", desc="Aligning")
        while current_start < total_duration and remaining_segments:
            segment_end = min(current_start + self.SEGMENT_SECONDS, total_duration)
            seg_duration = segment_end - current_start

            if seg_duration < 5:
                current_start = segment_end
                continue

            start_sample = int(current_start * WAV_SAMPLE_RATE)
            end_sample = int(segment_end * WAV_SAMPLE_RATE)
            audio_chunk = full_audio[start_sample:end_sample]

            max_sentences = min(20, len(remaining_segments))
            segment_text = self._join_text(remaining_segments[:max_sentences])
            clear_memory_cache()
            try:
                results = self.model.align(
                    audio=(audio_chunk, WAV_SAMPLE_RATE),
                    text=segment_text,
                    language=self.language,
                )
            except Exception as e:
                progress.close()
                raise RuntimeError(
                    f"模型对齐失败: {e}\n"
                    f"文本与音频无法互相匹配，程序终止。"
                )
            word_timestamps = [
                {
                    "text": item.text,
                    "start_time": float(item.start_time),
                    "end_time": float(item.end_time),
                }
                for item in results[0]
            ]
            matched, overflow_detected = self._words_to_sentences(
                word_timestamps,
                remaining_segments[:max_sentences],
                segment_duration=seg_duration,
                language=self.language,
            )
            if not matched:
                expected_sample = remaining_segments[0][:80] if remaining_segments else ""
                model_sample = [w["text"] for w in word_timestamps[:10]] if word_timestamps else []
                progress.close()
                raise RuntimeError(
                    f"模型输出了 {len(word_timestamps)} 个词，但无法与文本句子匹配。\n"
                    f"预期文本开头: '{expected_sample}'\n"
                    f"模型输出前 10 词: {model_sample}\n"
                    f"文本与音频无法互相匹配，程序终止。"
                )

            # Convert timestamps to absolute and clamp to valid range
            for sent in matched:
                sent["start_time"] = min(
                    sent["start_time"] + current_start, total_duration
                )
                sent["end_time"] = min(sent["end_time"] + current_start, total_duration)
                for w in sent.get("words", []):
                    w["start_time"] = min(
                        w["start_time"] + current_start, total_duration
                    )
                    w["end_time"] = min(w["end_time"] + current_start, total_duration)

            last_idx = matched[-1].get("_input_idx", len(matched) - 1)
            remaining_segments = remaining_segments[last_idx + 1 :]

            if segment_end >= total_duration and remaining_segments:
                if self._retried_last_segment:
                    progress.close()
                    raise RuntimeError(
                        f"音频末尾仍有 {len(remaining_segments)} 句未能对齐，重试后仍然失败。\n"
                        f"剩余文本开头: '{remaining_segments[0][:80]}'"
                    )
                last_valid = None
                for s in reversed(matched):
                    if s["end_time"] > s["start_time"]:
                        last_valid = s
                        break
                if last_valid is None:
                    progress.close()
                    raise RuntimeError(
                        f"音频末尾 {len(remaining_segments)} 句未能对齐，且最后一句无有效时间戳。\n"
                        f"剩余文本开头: '{remaining_segments[0][:80]}'"
                    )
                keep_idx = matched.index(last_valid)
                all_sentences.extend(matched[:keep_idx])
                remaining_segments.insert(0, last_valid["text"])
                current_start = last_valid["start_time"]
                self._retried_last_segment = True
            else:
                current_start = min(matched[-1]["end_time"], segment_end)
                all_sentences.extend(matched)
            clear_memory_cache()
            progress.n = int(current_start)
            progress.refresh()
        progress.close()

        return all_sentences

    def fix_sentences_timestamps(
        self, sentences: list[dict], audio_path: str
    ) -> list[dict]:
        fixed = list(sentences)
        for i, sent in enumerate(sentences):
            if sent["end_time"] > sent["start_time"]:
                continue
            print(f"Bad timestamp at sentence {i + 1}: {sent['text'][:20]}...")
            if i == 0 or i >= len(sentences) - 1:
                continue
            prev_sent = sentences[i - 1]
            next_sent = sentences[i + 1]

            audio_start = prev_sent["start_time"]
            audio_end = next_sent["end_time"]
            text_segments = [prev_sent["text"], sent["text"], next_sent["text"]]
            full_text = self._join_text(text_segments)

            full_audio = load_audio(audio_path)
            start_sample = int(audio_start * WAV_SAMPLE_RATE)
            end_sample = int(audio_end * WAV_SAMPLE_RATE)
            audio_segment = full_audio[start_sample:end_sample]

            results = self.model.align(
                audio=(audio_segment, WAV_SAMPLE_RATE),
                text=full_text,
                language=self.language,
            )
            word_timestamps = [
                {
                    "text": item.text,
                    "start_time": float(item.start_time) + audio_start,
                    "end_time": float(item.end_time) + audio_start,
                }
                for item in results[0]
            ]
            matched, _ = self._words_to_sentences(
                word_timestamps, text_segments, language=self.language
            )
            if len(matched) == 3:
                fixed[i - 1] = matched[0]
                fixed[i] = matched[1]
                fixed[i + 1] = matched[2]
                print(f"Re-aligned sentences {i} to {i + 2}")
        return fixed


def align_text(
    audio_path: str,
    text: str,
    language: str = "English",
    attn_implementation: str = None,
) -> list[dict]:
    aligner = Aligner(language=language, attn_implementation=attn_implementation)
    try:
        return aligner.align(audio_path, text)
    finally:
        aligner._unload_model()
