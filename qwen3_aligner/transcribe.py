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

import contextlib
import os

import torchaudio
from tqdm import tqdm

from qwen3_aligner.audio_utils import (
    WAV_SAMPLE_RATE,
    clear_memory_cache,
    create_vad_model,
    parse_vad_result,
    write_audio_temp,
)
from qwen3_aligner.model_loader import load_asr_model

try:
    import fireredvad
    HAS_VAD = True
except ImportError:
    HAS_VAD = False


def load_and_segment_audio(audio_path: str, segment_length_s: int = 180) -> list[dict]:
    print(f"Loading audio: {audio_path}")
    waveform, sr = torchaudio.load(audio_path)
    duration = waveform.shape[1] / sr
    print(f"Audio duration: {duration:.2f}s")

    if sr != WAV_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, WAV_SAMPLE_RATE)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    tmp_path = write_audio_temp(waveform)
    try:
        if not HAS_VAD:
            raise RuntimeError("fireredvad is required for audio segmentation")

        print("Using FireRedVAD for speech detection...")
        vad = create_vad_model()
        result = vad.detect(tmp_path)
        speech_timestamps = parse_vad_result(result)

        chunks = []
        total_duration = duration
        current_pos = 0
        chunk_idx = 0
        while current_pos < total_duration:
            next_boundary = (chunk_idx + 1) * segment_length_s
            split_point = next_boundary
            for start, end in speech_timestamps:
                if start <= next_boundary <= end:
                    split_point = end
                    break
                elif next_boundary < start:
                    break
            split_point = min(split_point, total_duration)
            if split_point - current_pos < 10:
                split_point = min(current_pos + segment_length_s, total_duration)
            chunk = waveform[
                :, int(current_pos * WAV_SAMPLE_RATE): int(split_point * WAV_SAMPLE_RATE)
            ]
            chunks.append({"audio": chunk, "offset": current_pos})
            current_pos = split_point
            chunk_idx += 1
        print(f"Audio segmented into {len(chunks)} chunks")
        return chunks
    finally:
        if tmp_path and os.path.exists(tmp_path):
            with contextlib.suppress(OSError):
                os.remove(tmp_path)


def _audio_segment_to_numpy(seg_tensor):
    return seg_tensor.numpy().flatten()


def transcribe_audio(chunks: list, model, language: str = "Japanese") -> list[dict]:
    results = []
    for chunk in tqdm(chunks, unit="chunk", desc="Transcribing"):
        samples = _audio_segment_to_numpy(chunk["audio"])
        audio_input = (samples, WAV_SAMPLE_RATE)
        result = model.transcribe(audio_input, language=language, context=None)
        if isinstance(result, list):
            result = result[0]
        text = getattr(result, "text", "")
        lang = getattr(result, "language", language)
        results.append({"text": text, "lang": lang, "offset": chunk["offset"]})
        del samples, audio_input
    return results


def run_transcribe(audio_path: str, language: str, model_size: str = "0.6B") -> tuple[str, list[dict]]:
    model_name = f"Qwen/Qwen3-ASR-{model_size}"
    clear_memory_cache()
    chunks = load_and_segment_audio(audio_path, segment_length_s=180)
    model = load_asr_model(model_name)
    results = transcribe_audio(chunks, model, language=language)
    del model
    clear_memory_cache()
    full_text = " ".join([r["text"] for r in results])
    return full_text, results



