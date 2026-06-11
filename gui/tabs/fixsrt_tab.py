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

import gradio as gr

from gui.i18n import LANGUAGES, t


def create_fixsrt_tab(run_fixsrt, global_fa2):
    comps = {}
    label_map = {}

    comps["audio_input"] = gr.File(
        label=t("audio_input"), file_types=["audio", "video"]
    )
    label_map["audio_input"] = "audio_input"

    comps["srt_file"] = gr.File(label=t("srt_input"), file_types=[".srt"])
    label_map["srt_file"] = "srt_input"

    with gr.Row():
        comps["lang_dropdown"] = gr.Dropdown(
            label=t("lang_label"),
            choices=LANGUAGES,
            value="English",
            scale=2,
        )
        label_map["lang_dropdown"] = "lang_label"

        comps["resegment_checkbox"] = gr.Checkbox(
            label=t("resegment_label"), value=False,
            elem_classes="resegment-label",
        )
        label_map["resegment_checkbox"] = "resegment_label"

    with gr.Row():
        comps["fix_btn"] = gr.Button(t("btn_fix"), variant="primary")
        label_map["fix_btn"] = "btn_fix"

        comps["download_fixed"] = gr.DownloadButton(
            label=t("btn_download_fixed"), visible=False
        )
        label_map["download_fixed"] = "btn_download_fixed"

    fix_status = gr.Markdown(visible=False)

    comps["srt_output"] = gr.DataFrame(
        elem_id="srt-table-fix",
        headers=["#", "Start", "End", "Text"],
        datatype=["number", "str", "str", "str"],
        column_count=4,
        interactive=False,
        label=t("srt_preview"),
    )
    label_map["srt_output"] = "srt_preview"

    def show_fixing():
        return [
            gr.update(value=[]),
            gr.update(value="⏳ " + t("fixing"), visible=True),
            gr.update(visible=False),
        ]

    def do_fix(file, srt, lang, resegment, fa2):
        if not file:
            return [
                gr.update(value=[]),
                gr.update(value="⚠️ " + t("no_audio"), visible=True),
                gr.update(visible=False),
            ]
        if not srt:
            return [
                gr.update(value=[]),
                gr.update(value="⚠️ " + t("no_srt"), visible=True),
                gr.update(visible=False),
            ]

        attn_impl = "flash_attention_2" if fa2 else None
        result, out_path, df_rows, error = run_fixsrt(file.name, srt.name, lang, resegment=resegment, attn_implementation=attn_impl)
        if error:
            return [
                gr.update(value=[]),
                gr.update(value="❌ " + t("fix_fail") + ": " + error, visible=True),
                gr.update(visible=False),
            ]
        return [
            gr.update(value=df_rows),
            gr.update(value="✅ " + t("fix_done"), visible=True),
            gr.update(value=out_path, visible=True),
        ]

    comps["fix_btn"].click(
        show_fixing,
        outputs=[comps["srt_output"], fix_status, comps["download_fixed"]],
    ).then(
        do_fix,
        inputs=[
            comps["audio_input"],
            comps["srt_file"],
            comps["lang_dropdown"],
            comps["resegment_checkbox"],
            global_fa2,
        ],
        outputs=[comps["srt_output"], fix_status, comps["download_fixed"]],
    )

    return comps, label_map
