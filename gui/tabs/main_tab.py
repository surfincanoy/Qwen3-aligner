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

import gradio as gr

from gui.i18n import LANGUAGES, t


def create_main_tab(run_transcribe, run_align, global_fa2):
    comps = {}
    label_map = {}

    comps["step1_title"] = gr.Markdown(t("step1_title"))
    label_map["step1_title"] = "step1_title"
    comps["audio_input"] = gr.File(
        label=t("audio_input"), file_types=["audio", "video"]
    )
    label_map["audio_input"] = "audio_input"

    with gr.Row():
        with gr.Column(min_width="300px"):
            comps["lang_dropdown"] = gr.Dropdown(
                label=t("lang_label"),
                choices=LANGUAGES,
                value="English",
            )
            label_map["lang_dropdown"] = "lang_label"

        comps["model_radio"] = gr.Radio(
            label=t("model_label"),
            choices=[t("model_fast"), t("model_precise")],
            value=t("model_fast"),
        )
        label_map["model_radio"] = "model_radio"

    with gr.Row():
        comps["transcribe_btn"] = gr.Button(t("btn_transcribe"), variant="primary")
        comps["download_text"] = gr.DownloadButton(label=t("btn_save_text"), visible=False)

    label_map["transcribe_btn"] = "btn_transcribe"
    label_map["download_text"] = "btn_save_text"

    gr.Markdown("---")
    comps["step2_title"] = gr.Markdown(t("step2_title"))
    label_map["step2_title"] = "step2_title"
    comps["model_hint_align"] = gr.Markdown(t("model_hint_align"))
    label_map["model_hint_align"] = "model_hint_align"

    comps["text_input"] = gr.Textbox(
        label=t("text_input"), lines=6, placeholder=t("placeholder_text")
    )
    label_map["text_input"] = "text_input"

    with gr.Row():
        comps["align_btn"] = gr.Button(t("btn_align"), variant="primary")
        comps["download_srt"] = gr.DownloadButton(label=t("btn_download_srt"), visible=False)

    label_map["align_btn"] = "btn_align"
    label_map["download_srt"] = "btn_download_srt"

    align_status = gr.Markdown(visible=False)
    transcribe_status = gr.Markdown(visible=False)

    comps["srt_output"] = gr.DataFrame(
        elem_id="srt-table-main",
        headers=["#", "Start", "End", "Text"],
        datatype=["number", "str", "str", "str"],
        column_count=4,
        interactive=False,
        label=t("srt_preview"),
    )
    label_map["srt_output"] = "srt_preview"

    t_outputs = [transcribe_status, comps["text_input"], comps["download_text"], comps["transcribe_btn"]]

    def show_transcribing():
        return [
            gr.update(value="⏳ " + t("transcribing"), visible=True),
            gr.update(value=""),
            gr.update(visible=False),
            gr.update(interactive=False),
        ]

    def do_transcribe(file, lang, model, fa2):
        if not file:
            return [
                gr.update(value="⚠️ " + t("no_audio"), visible=True),
                gr.update(value=""),
                gr.update(visible=False),
                gr.update(interactive=True),
            ]
        model_size = "1.7B" if model == t("model_precise") else "0.6B"
        attn_impl = "flash_attention_2" if fa2 else None
        result, error = run_transcribe(file.name, lang, model_size, attn_implementation=attn_impl)
        if error:
            return [
                gr.update(value="❌ " + t("transcribe_fail") + ": " + error, visible=True),
                gr.update(value=""),
                gr.update(visible=False),
                gr.update(interactive=True),
            ]
        base = os.path.splitext(os.path.basename(file))[0]
        download_name = base + ".txt"
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, download_name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(result)
        return [
            gr.update(visible=False),
            gr.update(value=result),
            gr.update(value=path, visible=True),
            gr.update(interactive=True),
        ]

    def show_aligning():
        return [
            gr.update(value=[]),
            gr.update(value="⏳ " + t("aligning"), visible=True),
            gr.update(visible=False),
            gr.update(interactive=False),
        ]

    def do_align(file, text, lang, fa2):
        if not file:
            return [
                gr.update(value=[]),
                gr.update(value="⚠️ " + t("no_audio"), visible=True),
                gr.update(visible=False),
                gr.update(interactive=True),
            ]
        if not text or not text.strip():
            return [
                gr.update(value=[]),
                gr.update(value="⚠️ " + t("no_text"), visible=True),
                gr.update(visible=False),
                gr.update(interactive=True),
            ]
        attn_impl = "flash_attention_2" if fa2 else None
        srt_content, srt_path, df_rows, error = run_align(file.name, text, lang, attn_implementation=attn_impl)
        if error:
            return [
                gr.update(value=[]),
                gr.update(value="❌ " + t("align_fail") + ": " + error, visible=True),
                gr.update(visible=False),
                gr.update(interactive=True),
            ]
        base = os.path.splitext(os.path.basename(file))[0]
        download_name = base + ".srt"
        tmpdir = tempfile.mkdtemp()
        dst = os.path.join(tmpdir, download_name)
        with open(dst, "w", encoding="utf-8") as f:
            f.write(srt_content)
        os.unlink(srt_path)
        return [
            gr.update(value=df_rows),
            gr.update(value="✅ " + t("align_done"), visible=True),
            gr.update(value=dst, visible=True),
            gr.update(interactive=True),
        ]

    comps["transcribe_btn"].click(
        show_transcribing,
        outputs=t_outputs,
    ).then(
        do_transcribe,
        inputs=[comps["audio_input"], comps["lang_dropdown"], comps["model_radio"], global_fa2],
        outputs=t_outputs,
    )

    a_status_outputs = [comps["srt_output"], align_status, comps["download_srt"], comps["align_btn"]]
    a_result_outputs = [comps["srt_output"], align_status, comps["download_srt"], comps["align_btn"]]

    comps["align_btn"].click(
        show_aligning,
        outputs=a_status_outputs,
    ).then(
        do_align,
        inputs=[comps["audio_input"], comps["text_input"], comps["lang_dropdown"], global_fa2],
        outputs=a_result_outputs,
    )

    return comps, label_map


