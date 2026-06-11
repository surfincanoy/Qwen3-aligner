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

import gc
import io
import os
import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch


WAV_SAMPLE_RATE = 16000

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VAD_DIR = os.path.join(_PROJECT_DIR, "FireRedVAD")



def load_audio(file_path: str) -> np.ndarray:
    try:
        if file_path.startswith(("http://", "https://")):
            raise ValueError("Using ffmpeg to load remote file.")
        wav_data, _ = librosa.load(file_path, sr=WAV_SAMPLE_RATE, mono=True)
        return wav_data
    except Exception as e:
        print(e)
        try:
            command = [
                "ffmpeg",
                "-i",
                file_path,
                "-ar",
                str(WAV_SAMPLE_RATE),
                "-ac",
                "1",
                "-c:a",
                "pcm_s16le",
                "-f",
                "wav",
                "-",
            ]
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout_data, stderr_data = process.communicate()
            if process.returncode != 0:
                raise RuntimeError(
                    f"FFmpeg error: {stderr_data.decode('utf-8', errors='ignore')}"
                )
            with io.BytesIO(stdout_data) as data_io:
                wav_data, _ = sf.read(data_io, dtype="float32")
            return wav_data
        except Exception as ffmpeg_e:
            raise RuntimeError(
                f"Failed to load audio: {ffmpeg_e}"
            )


def extract_audio_from_video(video_path: str) -> str:
    video_ext = Path(video_path).suffix.lower()
    if video_ext not in VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video format: {video_ext}")

    video_dir = Path(video_path).parent
    video_name = Path(video_path).stem
    temp_audio_path = str(video_dir / f"{video_name}_temp_audio.wav")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        temp_audio_path,
    ]
    print(f"Extracting audio from video: {video_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Audio extraction failed: {result.stderr}")
    print(f"Audio extracted to: {temp_audio_path}")
    return temp_audio_path


def ensure_audio(audio_path: str) -> str:
    if Path(audio_path).suffix.lower() in VIDEO_EXTENSIONS:
        return extract_audio_from_video(audio_path)
    return audio_path


def clear_memory_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def write_audio_temp(wav) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_np = wav.numpy().flatten() if hasattr(wav, "numpy") else wav.flatten()
        sf.write(tmp.name, audio_np, WAV_SAMPLE_RATE)
        return tmp.name


def create_vad_model():
    from fireredvad import FireRedVad, FireRedVadConfig

    config = FireRedVadConfig(
        use_gpu=False, smooth_window_size=5, speech_threshold=0.4,
        min_speech_frame=20, max_speech_frame=2000, min_silence_frame=20,
        merge_silence_frame=0, extend_speech_frame=0, chunk_max_frame=30000,
    )
    return FireRedVad.from_pretrained(_VAD_DIR, config)


def parse_vad_result(result) -> list:
    if isinstance(result, tuple):
        return result[0].get("timestamps", []) if hasattr(result[0], "get") else []
    return result.get("timestamps", [])
