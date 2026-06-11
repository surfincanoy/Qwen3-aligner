#!/usr/bin/env python3
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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import torch

from gui.i18n import LANG, set_lang, t
from gui.tabs.batch_fixsrt_tab import create_batch_fixsrt_tab
from gui.tabs.batch_tab import create_batch_tab
from gui.tabs.batch_transcribe_tab import create_batch_transcribe_tab
from gui.tabs.fixsrt_tab import create_fixsrt_tab
from gui.tabs.main_tab import create_main_tab
from gui.worker import (
    batch_align,
    batch_transcribe,
    run_align,
    run_fixsrt,
    run_transcribe,
)


def _fa2_supported() -> bool:
    try:
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            return cap[0] + cap[1] / 10 >= 7.5
        return False
    except Exception:
        return False


def _fa2_status() -> str:
    supported = _fa2_supported()
    if supported:
        return f'<span style="color:#4CAF50">✅ {t("fa2_supported")}</span>'
    else:
        return f'<span style="color:#e6a817">⚠️ {t("fa2_unsupported")}</span>'


def build_app():
    all_comps = []
    all_i18n_keys = []

    def switch_lang(lang):
        if lang not in LANG:
            lang = "English"
        set_lang(lang)
        updates = []
        for i18n_key, comp in zip(all_i18n_keys, all_comps, strict=False):
            if isinstance(comp, gr.Button):
                updates.append(gr.update(value=t(i18n_key)))
            elif isinstance(comp, gr.DownloadButton):
                updates.append(gr.update(label=t(i18n_key)))
            elif isinstance(comp, gr.Markdown):
                if i18n_key == "fa2_req":
                    updates.append(gr.update(value=_fa2_status()))
                else:
                    updates.append(gr.update(value=t(i18n_key)))
            elif isinstance(comp, gr.Radio) and i18n_key in (
                "model_radio",
                "batch_asr_model",
            ):
                updates.append(
                    gr.update(
                        label=t("model_label"),
                        choices=[t("model_fast"), t("model_precise")],
                    )
                )
            else:
                updates.append(gr.update(label=t(i18n_key)))
        return updates

    with gr.Blocks(title="Qwen3 音频与文稿匹配工具") as app:
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    "## 🎧 Qwen3 音频与文稿匹配工具 <small style='font-weight:normal;font-size:0.6em'>（B站）创作者：水橙汁</small>"
                )
                gr.Markdown("基于 Qwen3-ASR 与 Qwen3-ForcedAligner 的音频转录工具")
            with gr.Column(min_width=200, elem_id="lang-col"), gr.Row():
                with gr.Column():
                    fa2_checkbox = gr.Checkbox(label=t("fa2_checkbox"), value=False)
                    fa2_req = gr.Markdown(_fa2_status())
                lang_radio = gr.Dropdown(
                    elem_id="lang-dropdown",
                    choices=list(LANG.keys()),
                    value="English",
                    label=t("lang_switch"),
                )
        with gr.Tabs():
            with gr.TabItem(t("tab_main"), id="main"):
                comps, label_map = create_main_tab(
                    run_transcribe, run_align, fa2_checkbox
                )
                for k, v in comps.items():
                    all_comps.append(v)
                    all_i18n_keys.append(label_map[k])

            with gr.TabItem(t("tab_batch_asr"), id="batch_asr"):
                comps, label_map = create_batch_transcribe_tab(
                    batch_transcribe, batch_align, fa2_checkbox
                )
                for k, v in comps.items():
                    all_comps.append(v)
                    all_i18n_keys.append(label_map[k])

            with gr.TabItem(t("tab_batch"), id="batch"):
                comps, label_map = create_batch_tab(batch_align, fa2_checkbox)
                for k, v in comps.items():
                    all_comps.append(v)
                    all_i18n_keys.append(label_map[k])

            with gr.TabItem(t("tab_fixsrt"), id="fixsrt"):
                comps, label_map = create_fixsrt_tab(run_fixsrt, fa2_checkbox)
                for k, v in comps.items():
                    all_comps.append(v)
                    all_i18n_keys.append(label_map[k])

            with gr.TabItem(t("tab_batch_fixsrt"), id="batch_fixsrt"):
                comps, label_map = create_batch_fixsrt_tab(batch_align, fa2_checkbox)
                for k, v in comps.items():
                    all_comps.append(v)
                    all_i18n_keys.append(label_map[k])

        all_comps.append(fa2_checkbox)
        all_i18n_keys.append("fa2_checkbox")
        all_comps.append(fa2_req)
        all_i18n_keys.append("fa2_req")

        lang_radio.change(
            fn=switch_lang,
            inputs=lang_radio,
            outputs=list(all_comps),
        )

        lang_radio.change(
            fn=lambda lang: None,
            inputs=lang_radio,
            outputs=[],
            js="""(lang) => {
                const tabs = document.querySelectorAll('[role="tab"]');
                const labels = [lang === '中文' ? '🚀 转录/对齐' : '🚀 Quick Transcribe/Align',
               lang === '中文' ? '📦 批量转录/对齐' : '📦 Batch Transcribe/Align',
               lang === '中文' ? '📦 批量匹配' : '📦 Batch Align',
               lang === '中文' ? '🔧 SRT 修复' : '🔧 SRT Fix',
               lang === '中文' ? '📦 批量 SRT 修复' : '📦 Batch SRT Fix'];
                tabs.forEach((btn, i) => { if (i < 5) btn.textContent = labels[i]; });
            }""",
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.queue(default_concurrency_limit=1)
    app.launch(
        inbrowser=True,
        theme=gr.themes.Soft(),
        css="""
#srt-table-main table th:nth-child(1),
#srt-table-main table td:nth-child(1) { width: 20px; }
#srt-table-main table th:nth-child(2),
#srt-table-main table td:nth-child(2) { width: 130px; }
#srt-table-main table th:nth-child(3),
#srt-table-main table td:nth-child(3) { width: 130px; }

#srt-table-fix table th:nth-child(1),
#srt-table-fix table td:nth-child(1) { width: 20px; }
#srt-table-fix table th:nth-child(2),
#srt-table-fix table td:nth-child(2) { width: 130px; }
#srt-table-fix table th:nth-child(3),
#srt-table-fix table td:nth-child(3) { width: 130px; }

.resegment-label label span { font-size: 1.15em !important; font-weight: 600 !important; }
""",
    )
