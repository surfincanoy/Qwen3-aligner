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


def create_batch_tab(batch_align, global_fa2):
    comps = {}
    label_map = {}

    comps["batch_title"] = gr.Markdown(t("batch_title"))
    label_map["batch_title"] = "batch_title"

    with gr.Row():
        comps["batch_txt_files"] = gr.File(
            label=t("batch_txt_label"),
            file_types=[".txt"],
            file_count="multiple",
        )
        label_map["batch_txt_files"] = "batch_txt_label"

    with gr.Row():
        comps["batch_audio_files"] = gr.File(
            label=t("batch_audio_label"),
            file_types=["audio", "video"],
            file_count="multiple",
        )
        label_map["batch_audio_files"] = "batch_audio_label"

    with gr.Row():
        comps["batch_lang"] = gr.Dropdown(
            label=t("lang_label"),
            choices=LANGUAGES,
            value="English",
        )
        label_map["batch_lang"] = "lang_label"

    comps["batch_hint"] = gr.Markdown(t("batch_hint"))
    label_map["batch_hint"] = "batch_hint"

    with gr.Row():
        comps["batch_btn"] = gr.Button(t("btn_batch_align"), variant="primary")
        label_map["batch_btn"] = "btn_batch_align"

        comps["batch_download"] = gr.DownloadButton(
            label=t("btn_download_zip"), visible=False,
        )
        label_map["batch_download"] = "btn_download_zip"

    batch_status = gr.Markdown(visible=False)

    comps["batch_results"] = gr.DataFrame(
        headers=[t("batch_file"), t("batch_status")],
        datatype=["str", "str"],
        column_count=2,
        interactive=False,
        label=t("batch_status"),
    )
    label_map["batch_results"] = "batch_status"

    batch_outputs = [batch_status, comps["batch_results"], comps["batch_download"]]

    def show_batch_running():
        return [
            gr.update(value="⏳ " + t("batch_running"), visible=True),
            gr.update(value=[]),
            gr.update(visible=False),
        ]

    def do_batch(txt_files, audio_files, lang, fa2):
        if not txt_files:
            return [
                gr.update(value="⚠️ " + t("batch_no_txt"), visible=True),
                gr.update(value=[]),
                gr.update(visible=False),
            ]
        if not audio_files:
            return [
                gr.update(value="⚠️ " + t("batch_no_audio"), visible=True),
                gr.update(value=[]),
                gr.update(visible=False),
            ]

        txt_map = {}
        for f in txt_files:
            base = os.path.splitext(os.path.basename(f.name))[0]
            txt_map[base] = f.name

        audio_map = {}
        for f in audio_files:
            base = os.path.splitext(os.path.basename(f.name))[0]
            audio_map[base] = f.name

        matched = []
        for name in txt_map:
            if name in audio_map:
                matched.append((name, txt_map[name], audio_map[name]))

        if not matched:
            return [
                gr.update(value="⚠️ " + t("batch_no_match"), visible=True),
                gr.update(value=[]),
                gr.update(visible=False),
            ]

        out_dir = tempfile.mkdtemp()
        rows = []
        success_count = 0
        attn_impl = "flash_attention_2" if fa2 else None

        # Phase 1: read all texts, build entries
        align_entries = {}
        for name, txt_path, audio_path in matched:
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    text = f.read()
                if not text.strip():
                    rows.append([f"❌ {name}.srt", t("no_text")])
                    continue
                align_entries[name] = {"audio_path": audio_path, "text": text}
            except Exception as e:
                rows.append([f"❌ {name}.srt", str(e)])

        if not align_entries:
            status_msg = t("batch_done").format(success=0, total=len(matched))
            return [gr.update(value="❌ " + status_msg, visible=True), gr.update(value=rows), gr.update(visible=False)]

        # Phase 2: align all (loads model once)
        srt_results = batch_align(align_entries, lang, attn_implementation=attn_impl)

        for name in align_entries:
            srt_content, json_content, err = srt_results.get(name, ("", "", "未知错误"))
            if err:
                rows.append([f"❌ {name}.srt", err])
                continue
            dst = os.path.join(out_dir, f"{name}.srt")
            with open(dst, "w", encoding="utf-8") as f:
                f.write(srt_content)
            rows.append([f"✅ {name}.srt", t("batch_success")])
            success_count += 1

        total = len(matched)
        status_msg = t("batch_done").format(success=success_count, total=total)

        zip_path = None
        if success_count > 0:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_name = f"batch_align_{ts}.zip"
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

    comps["batch_btn"].click(
        show_batch_running,
        outputs=batch_outputs,
    ).then(
        do_batch,
        inputs=[comps["batch_txt_files"], comps["batch_audio_files"], comps["batch_lang"], global_fa2],
        outputs=batch_outputs,
    )

    return comps, label_map

