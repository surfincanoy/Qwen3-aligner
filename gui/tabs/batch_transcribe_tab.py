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
import tempfile
import zipfile
from datetime import datetime

import gradio as gr

from gui.i18n import LANGUAGES, t


def create_batch_transcribe_tab(batch_transcribe, batch_align, global_fa2):
    comps = {}
    label_map = {}

    comps["batch_asr_title"] = gr.Markdown(t("batch_asr_title"))
    label_map["batch_asr_title"] = "batch_asr_title"

    comps["batch_asr_audio_files"] = gr.File(
        label=t("batch_asr_audio_label"),
        file_types=["audio", "video"],
        file_count="multiple",
    )
    label_map["batch_asr_audio_files"] = "batch_asr_audio_label"

    with gr.Row():
        comps["batch_asr_lang"] = gr.Dropdown(
            label=t("lang_label"),
            choices=LANGUAGES,
            value="English",
        )
        label_map["batch_asr_lang"] = "lang_label"

        comps["batch_asr_model"] = gr.Radio(
            label=t("model_label"),
            choices=[t("model_fast"), t("model_precise")],
            value=t("model_fast"),
        )
        label_map["batch_asr_model"] = "batch_asr_model"

    with gr.Row():
        comps["batch_asr_btn"] = gr.Button(t("btn_batch_asr"), variant="primary")
        comps["batch_asr_download"] = gr.DownloadButton(
            label=t("btn_download_zip"), visible=False,
        )
    label_map["batch_asr_btn"] = "btn_batch_asr"
    label_map["batch_asr_download"] = "btn_download_zip"

    batch_asr_status = gr.Markdown(visible=False)

    comps["batch_asr_results"] = gr.DataFrame(
        headers=[t("batch_file"), t("batch_status")],
        datatype=["str", "str"],
        column_count=2,
        interactive=False,
        label=t("batch_asr_results_label"),
    )
    label_map["batch_asr_results"] = "batch_asr_results_label"

    batch_outputs = [batch_asr_status, comps["batch_asr_results"], comps["batch_asr_download"]]

    def show_running():
        return [
            gr.update(value="⏳ " + t("batch_asr_running"), visible=True),
            gr.update(value=[]),
            gr.update(visible=False),
        ]

    def do_batch(audio_files, lang, model, fa2):
        if not audio_files:
            return [
                gr.update(value="⚠️ " + t("no_audio"), visible=True),
                gr.update(value=[]),
                gr.update(visible=False),
            ]

        model_size = "1.7B" if model == t("model_precise") else "0.6B"
        attn_impl = "flash_attention_2" if fa2 else None
        out_dir = tempfile.mkdtemp()
        rows = []
        success_count = 0

        # Phase 1: transcribe all files (loads ASR model once)
        texts, err = batch_transcribe(audio_files, lang, model_size, attn_implementation=attn_impl)
        if err:
            return [
                gr.update(value="❌ " + err, visible=True),
                gr.update(value=[]),
                gr.update(visible=False),
            ]

        total = len(texts)

        # save .txt files, prepare alignment entries (only from transcribed dict)
        align_entries = {}
        for base, entry in texts.items():
            if entry is None:
                rows.append([f"❌ {base}", t("transcribe_fail")])
                continue
            text = entry["text"]
            txt_path = os.path.join(out_dir, f"{base}.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            align_entries[base] = entry

        if not align_entries:
            status_msg = t("batch_asr_done").format(success=0, total=total)
            return [gr.update(value="❌ " + status_msg, visible=True), gr.update(value=rows), gr.update(visible=False)]

        # Phase 2: align all transcribed files (loads aligner model once)
        srt_results = batch_align(align_entries, lang, attn_implementation=attn_impl)

        for base in align_entries:
            srt_content, json_content, err = srt_results.get(base, ("", "", "未知错误"))
            if err:
                rows.append([f"❌ {base}", t("align_fail") + ": " + err])
                continue
            dst = os.path.join(out_dir, f"{base}.srt")
            with open(dst, "w", encoding="utf-8") as f:
                f.write(srt_content)
            rows.append([f"✅ {base}.txt + .srt", t("batch_success")])
            success_count += 1

        status_msg = t("batch_asr_done").format(success=success_count, total=total)

        zip_path = None
        if success_count > 0:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_name = f"batch_output_{ts}.zip"
            zip_path = os.path.join(tempfile.gettempdir(), zip_name)
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                    for fname in os.listdir(out_dir):
                        fpath = os.path.join(out_dir, fname)
                        z.write(fpath, fname)

        return [
            gr.update(value="✅ " + status_msg, visible=True),
            gr.update(value=rows),
            gr.update(value=zip_path, visible=True) if zip_path else gr.update(visible=False),
        ]

    comps["batch_asr_btn"].click(
        show_running,
        outputs=batch_outputs,
    ).then(
        do_batch,
        inputs=[comps["batch_asr_audio_files"], comps["batch_asr_lang"], comps["batch_asr_model"], global_fa2],
        outputs=batch_outputs,
    )

    return comps, label_map

