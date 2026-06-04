import os
import re
import string
import tempfile

import numpy as np
import soundfile as sf
import torch

from qwen3_aligner.audio_utils import (
    WAV_SAMPLE_RATE,
    clear_memory_cache,
    load_audio,
    ensure_audio,
)
from qwen3_aligner.model_loader import load_aligner_model
from qwen3_aligner.text_utils import split_text

try:
    from fireredvad import FireRedVad, FireRedVadConfig
    HAS_VAD = True
except ImportError:
    HAS_VAD = False


class Aligner:
    def __init__(self, language: str = "English"):
        self.language = language
        self.model = None
        self.SEGMENT_SECONDS = 180

    def load_model(self):
        if self.model is not None:
            return
        self.model = load_aligner_model()

    def split_text(self, text: str) -> list[str]:
        return split_text(text, self.language)

    def _join_text(self, segments: list[str]) -> str:
        if self.language in {"Chinese", "Japanese", "Korean", "Thai"}:
            return "".join(segments)
        return " ".join(segments)

    def _clean_text_for_matching(self, text: str) -> str:
        if self.language == "English":
            return re.sub(r"[^\w\s]", "", text).lower()
        return re.sub(r"[^\w]", "", text).lower()

    def align(self, audio_path: str, text: str) -> list[dict]:
        print("=" * 60)
        print("Starting alignment")
        print("=" * 60)
        self.load_model()
        audio_path = ensure_audio(audio_path)
        text_segments = self.split_text(text)
        audio = load_audio(audio_path)
        duration = len(audio) / WAV_SAMPLE_RATE
        if duration > self.SEGMENT_SECONDS:
            return self.align_long_audio(audio_path, text)

        full_text = self._join_text(text_segments)
        print("\nRunning alignment...")
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
        print(f"Aligned {len(word_timestamps)} words")
        sentences = self._words_to_sentences(word_timestamps, text_segments)
        print(f"Generated {len(sentences)} sentences")
        clear_memory_cache()
        print("=" * 60)
        print("Done")
        print("=" * 60)
        return sentences

    def _words_to_sentences(
        self, word_timestamps: list[dict], text_segments: list[str]
    ) -> list[dict]:
        if not word_timestamps or not text_segments:
            return []
        sentences = []
        word_idx = 0
        total_words = len(word_timestamps)

        for text_segment in text_segments:
            if word_idx >= total_words:
                break

            clean_sent = text_segment.strip()
            punct = set(string.punctuation)
            for p in punct:
                clean_sent = clean_sent.rstrip(p)
            clean_sent = clean_sent.strip()

            if not clean_sent:
                continue

            clean_sent_no_space = self._clean_text_for_matching(clean_sent).replace(" ", "")
            if not clean_sent_no_space:
                continue
            target_len = len(clean_sent_no_space)

            matched_words = []
            matched_clean = ""
            temp_idx = word_idx

            while temp_idx < total_words and len(matched_clean) < target_len:
                word = word_timestamps[temp_idx]
                word_text = self._clean_text_for_matching(word.get("text", ""))
                if not word_text:
                    temp_idx += 1
                    continue
                matched_words.append(word)
                matched_clean += word_text.replace(" ", "")
                temp_idx += 1

            if len(matched_clean) >= target_len:
                last_valid_word = None
                for w in reversed(matched_words):
                    if w["end_time"] > w["start_time"]:
                        last_valid_word = w
                        break
                if last_valid_word is not None:
                    sentences.append({
                        "text": text_segment,
                        "start_time": matched_words[0]["start_time"],
                        "end_time": last_valid_word["end_time"],
                        "words": matched_words,
                    })
                    word_idx = temp_idx
        return sentences

    def get_vad_model(self):
        vad_config = FireRedVadConfig(
            use_gpu=False, smooth_window_size=5, speech_threshold=0.4,
            min_speech_frame=20, max_speech_frame=2000, min_silence_frame=20,
            merge_silence_frame=0, extend_speech_frame=0, chunk_max_frame=30000,
        )
        return FireRedVad.from_pretrained("./FireRedVAD", vad_config)

    def split_audio_for_alignment(self, audio_path: str, segment_seconds: int = None):
        if segment_seconds is None:
            segment_seconds = self.SEGMENT_SECONDS
        wav = load_audio(audio_path)
        total_duration = len(wav) / WAV_SAMPLE_RATE
        print(f"Splitting audio: {audio_path} ({total_duration:.2f}s, chunk={segment_seconds}s)")

        if total_duration <= segment_seconds:
            return [{
                "start_time": 0.0, "end_time": total_duration,
                "start_sample": 0, "end_sample": len(wav), "audio": wav,
            }]

        if not HAS_VAD:
            raise RuntimeError("fireredvad is required for long audio splitting")

        print("Using FireRedVAD for speech detection...")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            audio_np = wav.numpy().flatten() if hasattr(wav, "numpy") else wav.flatten()
            sf.write(tmp_path, audio_np, WAV_SAMPLE_RATE)

        vad = self.get_vad_model()
        result = vad.detect(tmp_path)
        if isinstance(result, tuple):
            speech_timestamps = result[0].get("timestamps", []) if hasattr(result[0], "get") else []
        else:
            speech_timestamps = result.get("timestamps", [])
        os.unlink(tmp_path)

        segments = []
        current_pos = 0
        chunk_idx = 0
        while current_pos < total_duration:
            next_boundary = (chunk_idx + 1) * segment_seconds
            split_point = next_boundary
            if speech_timestamps:
                for start, end in speech_timestamps:
                    if start <= next_boundary <= end:
                        split_point = end
                        break
                    elif next_boundary < start:
                        break
            split_point = min(split_point, total_duration)
            if split_point - current_pos < 10:
                split_point = min(current_pos + segment_seconds, total_duration)
            start_sample = int(current_pos * WAV_SAMPLE_RATE)
            end_sample = int(split_point * WAV_SAMPLE_RATE)
            segments.append({
                "start_time": current_pos, "end_time": split_point,
                "start_sample": start_sample, "end_sample": end_sample,
                "audio": wav[start_sample:end_sample],
            })
            current_pos = split_point
            chunk_idx += 1
        print(f"Audio split into {len(segments)} segments")
        return segments

    def align_long_audio(self, audio_path: str, text: str) -> list[dict]:
        print("=" * 60)
        print("Starting long audio segmented alignment")
        print("=" * 60)
        self.load_model()
        text_segments = self.split_text(text)
        print(f"Text split into {len(text_segments)} sentences")
        segments = self.split_audio_for_alignment(audio_path)
        remaining_segments = list(text_segments)
        all_sentences = []

        for seg_idx, seg in enumerate(segments):
            if not remaining_segments:
                print("All sentences aligned, stopping")
                break

            seg_start = seg["start_time"]
            seg_end = seg["end_time"]
            seg_duration = seg_end - seg_start
            print(f"Segment {seg_idx + 1}/{len(segments)}: {seg_start:.2f}s - {seg_end:.2f}s")

            remaining_text_full = self._join_text(remaining_segments)
            try:
                results = self.model.align(
                    audio=(seg["audio"], WAV_SAMPLE_RATE),
                    text=remaining_text_full,
                    language=self.language,
                )
            except Exception as e:
                print(f"Align failed (seg {seg_idx + 1}): {e}")
                continue

            word_timestamps = [
                {
                    "text": item.text,
                    "start_time": float(item.start_time),
                    "end_time": float(item.end_time),
                }
                for item in results[0]
            ]
            matched_sentences = self._words_to_sentences(word_timestamps, remaining_segments)
            if not matched_sentences:
                continue

            valid_sentences = [
                s for s in matched_sentences
                if s["end_time"] <= seg_duration + 1e-6
            ]
            if not valid_sentences:
                continue

            for sent in valid_sentences:
                sent["start_time"] += seg_start
                sent["end_time"] += seg_start
                for w in sent.get("words", []):
                    w["start_time"] += seg_start
                    w["end_time"] += seg_start

            num_valid = len(valid_sentences)
            all_sentences.extend(valid_sentences)
            print(f"Segment matched {num_valid} sentences")
            remaining_segments = remaining_segments[num_valid:]
            clear_memory_cache()

        clear_memory_cache()
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
            matched = self._words_to_sentences(word_timestamps, text_segments)
            if len(matched) == 3:
                fixed[i - 1] = matched[0]
                fixed[i] = matched[1]
                fixed[i + 1] = matched[2]
                print(f"Re-aligned sentences {i} to {i + 2}")
        return fixed


def align_text(audio_path: str, text: str, language: str = "English") -> list[dict]:
    aligner = Aligner(language=language)
    return aligner.align(audio_path, text)
