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
import os
import tempfile

import torch
from qwen3_aligner import (
    Aligner,
    align_text,
    clear_memory_cache,
    ensure_audio,
    load_and_segment_audio,
    load_asr_model,
    transcribe_audio,
)
from qwen3_aligner.srt_utils import (
    format_time,
    load_srt,
    merge_sentences,
    sentences_to_entries,
    sentences_to_srt,
)
from qwen3_aligner.text_utils import split_text


def _entries_to_rows(entries: list[dict]) -> list[list]:
    return [
        [
            e["index"],
            format_time(e["start_time"]),
            format_time(e["end_time"]),
            e["text"],
        ]
        for e in entries
    ]


def run_transcribe(
    audio_path: str, language: str, model_size: str, attn_implementation: str = None
) -> tuple[str, str]:
    model = None
    results = None
    chunks = None
    try:
        clear_memory_cache()
        audio_path = ensure_audio(audio_path)
        model_name = f"Qwen/Qwen3-ASR-{model_size}"
        chunks = load_and_segment_audio(audio_path, segment_length_s=120)
        model = load_asr_model(model_name, attn_implementation=attn_implementation)
        results = transcribe_audio(chunks, model, language=language)
        full_text = " ".join([r["text"] for r in results])
        return full_text, ""
    except Exception as e:
        return "", str(e)
    finally:
        torch.cuda.synchronize()
        del results
        del chunks
        if model is not None:
            if hasattr(model, "model"):
                if hasattr(model.model, "to"):
                    model.model.to("cpu")
                model.model = None
            del model
        clear_memory_cache()


def batch_transcribe(
    files: list, language: str, model_size: str, attn_implementation: str = None
) -> tuple[dict[str, dict], str]:
    model = None
    all_chunks = []
    try:
        clear_memory_cache()
        model_name = f"Qwen/Qwen3-ASR-{model_size}"
        model = load_asr_model(model_name, attn_implementation=attn_implementation)

        results = {}
        for f in files:
            base = os.path.splitext(os.path.basename(f.name))[0]
            try:
                audio_path = ensure_audio(f.name)
                chunks = load_and_segment_audio(audio_path, segment_length_s=120)
                texts = transcribe_audio(chunks, model, language=language)
                full_text = " ".join([r["text"] for r in texts])
                results[base] = {"text": full_text, "audio_path": audio_path}
            except Exception as e:
                results[base] = None
        return results, ""
    except Exception as e:
        return {}, str(e)
    finally:
        torch.cuda.synchronize()
        del all_chunks
        if model is not None:
            if hasattr(model, "model"):
                if hasattr(model.model, "to"):
                    model.model.to("cpu")
                model.model = None
            del model
        clear_memory_cache()


def batch_align(
    entries: dict[str, dict], language: str, attn_implementation: str = None,
    orig_entry_texts: dict[str, list[str]] = None,
) -> dict[str, tuple[str, str, str]]:
    aligner = None
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        aligner = Aligner(language=language, attn_implementation=attn_implementation)
        aligner._keep_model = True
        aligner.load_model()

        results = {}
        for base, entry in entries.items():
            try:
                sentences = aligner.align(entry["audio_path"], entry["text"])
                if not sentences:
                    results[base] = ("", "", "对齐未生成任何结果")
                    continue
                if orig_entry_texts and base in orig_entry_texts:
                    orig_texts = orig_entry_texts[base]
                    all_words = []
                    for s in sentences:
                        for w in s.get("words", []):
                            all_words.append(w)
                    if all_words:
                        remapped, _ = aligner._words_to_sentences(all_words, orig_texts, language)
                        entries_list = []
                        for i, t in enumerate(orig_texts):
                            if i < len(remapped):
                                entries_list.append({
                                    "index": i + 1, "text": t,
                                    "start_time": remapped[i]["start_time"],
                                    "end_time": remapped[i]["end_time"],
                                })
                            else:
                                entries_list.append({
                                    "index": i + 1, "text": t, "start_time": 0, "end_time": 0,
                                })
                    else:
                        entries_list = [
                            {"index": i + 1, "text": t, "start_time": 0, "end_time": 0}
                            for i, t in enumerate(orig_texts)
                        ]
                else:
                    entries_list = sentences_to_entries(merge_sentences(sentences))
                srt_content = sentences_to_srt(entries_list)
                results[base] = (srt_content, "", "")
            except Exception as e:
                results[base] = ("", "", str(e))
        return results
    except Exception as e:
        return {base: ("", "", str(e)) for base in entries}
    finally:
        if aligner is not None:
            aligner._unload_model()
        del aligner
        clear_memory_cache()


def run_align(
    audio_path: str, text: str, language: str, attn_implementation: str = None
) -> tuple[str, str, list[list], str]:
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        sentences = align_text(
            audio_path, text, language=language, attn_implementation=attn_implementation
        )
        if not sentences:
            return "", "", [], "对齐未生成任何结果"

        entries = sentences_to_entries(merge_sentences(sentences))
        srt_content = sentences_to_srt(entries)
        fd, srt_path = tempfile.mkstemp(suffix=".srt")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(srt_content)

        df_rows = _entries_to_rows(entries)
        clear_memory_cache()
        return srt_content, srt_path, df_rows, ""
    except Exception as e:
        clear_memory_cache()
        return "", "", [], str(e)


def run_fixsrt(
    audio_path: str, srt_path: str, language: str,
    resegment: bool = False, attn_implementation: str = None,
) -> tuple[str, str, list[list], str]:
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        orig_entries = load_srt(srt_path)
        if not orig_entries:
            return "", "", [], "SRT 文件为空或格式错误"

        srt_texts = [e["text"].replace("\n", " ") for e in orig_entries]
        if language in {"Chinese", "Japanese", "Korean", "Thai"}:
            full_text = "".join(srt_texts)
        else:
            full_text = " ".join(srt_texts)

        if resegment:
            segs = split_text(full_text, language)
            full_text = "\n".join(segs)

        sentences = align_text(
            audio_path,
            full_text,
            language=language,
            attn_implementation=attn_implementation,
        )
        if not sentences:
            return "", "", [], "对齐未生成任何结果"

        if resegment:
            out_entries = sentences_to_entries(merge_sentences(sentences))
        else:
            all_words = []
            for s in sentences:
                for w in s.get("words", []):
                    all_words.append(w)
            if all_words:
                aligner = Aligner(language=language)
                remapped, _ = aligner._words_to_sentences(all_words, srt_texts, language=language)
                out_entries = []
                for i, entry_text in enumerate(srt_texts):
                    if i < len(remapped):
                        out_entries.append({
                            "index": i + 1,
                            "text": entry_text,
                            "start_time": remapped[i]["start_time"],
                            "end_time": remapped[i]["end_time"],
                        })
                    else:
                        out_entries.append({
                            "index": i + 1, "text": entry_text, "start_time": 0, "end_time": 0,
                        })
            else:
                out_entries = [
                    {"index": i + 1, "text": t, "start_time": 0, "end_time": 0}
                    for i, t in enumerate(srt_texts)
                ]

        srt_content = sentences_to_srt(out_entries)

        base, ext = os.path.splitext(srt_path)
        output_path = f"{base}_fixed{ext}"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt_content)

        df_rows = _entries_to_rows(out_entries)
        clear_memory_cache()
        return srt_content, output_path, df_rows, ""
    except Exception as e:
        clear_memory_cache()
        return "", "", [], str(e)
