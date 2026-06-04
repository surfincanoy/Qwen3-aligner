import contextlib
import os
import tempfile

import numpy as np
import soundfile as sf
import torch
import torchaudio

from qwen3_aligner.audio_utils import WAV_SAMPLE_RATE, clear_memory_cache
from qwen3_aligner.model_loader import load_asr_model

try:
    from fireredvad import FireRedVad, FireRedVadConfig
    HAS_VAD = True
except ImportError:
    HAS_VAD = False


def _get_vad_model():
    vad_config = FireRedVadConfig(
        use_gpu=False, smooth_window_size=5, speech_threshold=0.4,
        min_speech_frame=20, max_speech_frame=2000, min_silence_frame=20,
        merge_silence_frame=0, extend_speech_frame=0, chunk_max_frame=30000,
    )
    return FireRedVad.from_pretrained("./FireRedVAD", vad_config)


def load_and_segment_audio(audio_path: str, segment_length_s: int = 180) -> list[dict]:
    print(f"Loading audio: {audio_path}")
    waveform, sr = torchaudio.load(audio_path)
    duration = waveform.shape[1] / sr
    print(f"Audio duration: {duration:.2f}s")

    if sr != WAV_SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, WAV_SAMPLE_RATE)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            audio_np = waveform.numpy().flatten()
            sf.write(tmp_path, audio_np, WAV_SAMPLE_RATE)

        if not HAS_VAD:
            raise RuntimeError("fireredvad is required for audio segmentation")

        print("Using FireRedVAD for speech detection...")
        vad = _get_vad_model()
        result = vad.detect(tmp_path)
        if isinstance(result, tuple):
            speech_timestamps = (
                result[0].get("timestamps", []) if hasattr(result[0], "get") else []
            )
        else:
            speech_timestamps = result.get("timestamps", [])

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


def _audio_segment_to_numpy(seg_tensor) -> np.ndarray:
    return seg_tensor.numpy().flatten()


def transcribe_audio(chunks: list, model, language: str = "Japanese") -> list[dict]:
    results = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i + 1}/{len(chunks)} (offset: {chunk['offset']:.1f}s)")
        samples = _audio_segment_to_numpy(chunk["audio"])
        audio_input = (samples, WAV_SAMPLE_RATE)
        result = model.transcribe(audio_input, language=language, context=None)
        if isinstance(result, list):
            result = result[0]
        text = getattr(result, "text", "")
        lang = getattr(result, "language", language)
        results.append({"text": text, "lang": lang, "offset": chunk["offset"]})
        print(f"  Text length: {len(text)} chars")
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
