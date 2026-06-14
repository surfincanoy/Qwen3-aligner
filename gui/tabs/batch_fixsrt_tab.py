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
from qwen3_aligner.srt_utils import load_srt
from qwen3_aligner.text_utils import split_text


def create_batch_fixsrt_tab(batch_align, global_fa2):
    comps = {}
    label_map = {}

    comps["batch_fixsrt_title"] = gr.Markdown(t("batch_fixsrt_title"))
    label_map["batch_fixsrt_title"] = "batch_fixsrt_title"

    with gr.Row():
        comps["batch_fixsrt_srt_files"] = gr.File(
            label=t("batch_fixsrt_srt_label"),
            file_types=[".srt"],
            file_count="multiple",
        )
        label_map["batch_fixsrt_srt_files"] = "batch_fixsrt_srt_label"

    with gr.Row():
        comps["batch_fixsrt_audio_files"] = gr.File(
            label=t("batch_fixsrt_audio_label"),
            file_types=["audio", "video"],
            file_count="multiple",
        )
        label_map["batch_fixsrt_audio_files"] = "batch_fixsrt_audio_label"

    with gr.Row():
        comps["batch_fixsrt_lang"] = gr.Dropdown(
            label=t("lang_label"),
            choices=LANGUAGES,
            value="English",
            scale=2,
        )
        label_map["batch_fixsrt_lang"] = "lang_label"

        comps["batch_fixsrt_resegment"] = gr.Checkbox(
            label=t("resegment_label"), value=False,
            elem_classes="resegment-label",
        )
        label_map["batch_fixsrt_resegment"] = "resegment_label"

    comps["batch_fixsrt_hint"] = gr.Markdown(t("batch_hint"))
    label_map["batch_fixsrt_hint"] = "batch_hint"

    with gr.Row():
        comps["batch_fixsrt_btn"] = gr.Button(t("btn_batch_fixsrt"), variant="primary")
        label_map["batch_fixsrt_btn"] = "btn_batch_fixsrt"

        comps["batch_fixsrt_download"] = gr.DownloadButton(
            label=t("btn_download_zip"), visible=False,
        )
        label_map["batch_fixsrt_download"] = "btn_download_zip"

    batch_fixsrt_status = gr.Markdown(visible=False)

    comps["batch_fixsrt_results"] = gr.DataFrame(
        headers=[t("batch_file"), t("batch_status")],
        datatype=["str", "str"],
        column_count=2,
        interactive=False,
        label=t("batch_status"),
    )
    label_map["batch_fixsrt_results"] = "batch_status"

    batch_outputs = [batch_fixsrt_status, comps["batch_fixsrt_results"], comps["batch_fixsrt_download"], comps["batch_fixsrt_btn"]]

    def show_running():
        return [
            gr.update(value="⏳ " + t("batch_fixsrt_running"), visible=True),
            gr.update(value=[]),
            gr.update(visible=False),
            gr.update(interactive=False),
        ]

    def do_batch(srt_files, audio_files, lang, resegment, fa2):
        if not srt_files:
            return [
                gr.update(value="⚠️ " + t("batch_fixsrt_no_srt"), visible=True),
                gr.update(value=[]),
                gr.update(visible=False),
                gr.update(interactive=True),
            ]
        if not audio_files:
            return [
                gr.update(value="⚠️ " + t("batch_fixsrt_no_audio"), visible=True),
                gr.update(value=[]),
                gr.update(visible=False),
                gr.update(interactive=True),
            ]

        srt_map = {}
        for f in srt_files:
            base = os.path.splitext(os.path.basename(f.name))[0]
            srt_map[base] = f.name

        audio_map = {}
        for f in audio_files:
            base = os.path.splitext(os.path.basename(f.name))[0]
            audio_map[base] = f.name

        matched = []
        for name in srt_map:
            if name in audio_map:
                matched.append((name, srt_map[name], audio_map[name]))

        if not matched:
            return [
                gr.update(value="⚠️ " + t("batch_fixsrt_no_match"), visible=True),
                gr.update(value=[]),
                gr.update(visible=False),
                gr.update(interactive=True),
            ]

        out_dir = tempfile.mkdtemp()
        rows = []
        success_count = 0
        attn_impl = "flash_attention_2" if fa2 else None

        align_entries = {}
        preserve_texts = {} if not resegment else None
        for name, srt_path, audio_path in matched:
            try:
                entries = load_srt(srt_path)
                if not entries:
                    rows.append([f"❌ {name}.srt", t("no_srt")])
                    continue
                srt_texts = [e["text"].replace("\n", " ") for e in entries]
                if resegment:
                    segs = split_text(" ".join(srt_texts) if lang not in {"Chinese", "Japanese", "Korean", "Thai"} else "".join(srt_texts), lang)
                    full_text = "\n".join(segs)
                else:
                    if lang in {"Chinese", "Japanese", "Korean", "Thai"}:
                        full_text = "".join(srt_texts)
                    else:
                        full_text = " ".join(srt_texts)
                    preserve_texts[name] = srt_texts
                if not full_text.strip():
                    rows.append([f"❌ {name}.srt", t("no_text")])
                    continue
                align_entries[name] = {"audio_path": audio_path, "text": full_text}
            except Exception as e:
                rows.append([f"❌ {name}.srt", str(e)])

        if not align_entries:
            status_msg = t("batch_fixsrt_done").format(success=0, total=len(matched))
            return [gr.update(value="❌ " + status_msg, visible=True), gr.update(value=rows), gr.update(visible=False), gr.update(interactive=True)]

        srt_results = batch_align(align_entries, lang, attn_implementation=attn_impl, orig_entry_texts=preserve_texts)

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
        status_msg = t("batch_fixsrt_done").format(success=success_count, total=total)

        zip_path = None
        if success_count > 0:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_name = f"batch_fixsrt_{ts}.zip"
            zip_path = os.path.join(tempfile.gettempdir(), zip_name)
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                for fname in os.listdir(out_dir):
                    fpath = os.path.join(out_dir, fname)
                    z.write(fpath, fname)

        return [
            gr.update(value="✅ " + status_msg, visible=True),
            gr.update(value=rows),
            gr.update(value=zip_path, visible=True) if zip_path else gr.update(visible=False),
            gr.update(interactive=True),
        ]

    comps["batch_fixsrt_btn"].click(
        show_running,
        outputs=batch_outputs,
    ).then(
        do_batch,
        inputs=[comps["batch_fixsrt_srt_files"], comps["batch_fixsrt_audio_files"], comps["batch_fixsrt_lang"], comps["batch_fixsrt_resegment"], global_fa2],
        outputs=batch_outputs,
    )

    return comps, label_map
