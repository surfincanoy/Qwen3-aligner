#!/usr/bin/env python3
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr

from gui.i18n import t, set_lang, LANG
from gui.tabs.main_tab import create_main_tab
from gui.tabs.fixsrt_tab import create_fixsrt_tab
from gui.worker import run_transcribe, run_align, run_fixsrt


def build_app():
    all_comps = []
    all_i18n_keys = []

    def switch_lang(lang):
        if lang not in LANG:
            lang = "中文"
        set_lang(lang)
        updates = []
        for i18n_key, comp in zip(all_i18n_keys, all_comps):
            if isinstance(comp, (gr.Button, gr.DownloadButton)):
                updates.append(gr.update(value=t(i18n_key)))
            elif isinstance(comp, gr.Radio) and i18n_key == "model_radio":
                updates.append(
                    gr.update(
                        label=t("model_label"),
                        choices=[t("model_fast"), t("model_precise")],
                    )
                )
            else:
                updates.append(gr.update(label=t(i18n_key)))
        return updates

    with gr.Blocks(title="Qwen3 音频处理工坊") as app:
        gr.Markdown("# 🎧 Qwen3 音频处理工坊")
        gr.Markdown(
            "基于 Qwen3-ASR 与 Qwen3-ForcedAligner 的音频处理工具"
        )

        lang_radio = gr.Radio(
            choices=list(LANG.keys()),
            value="中文",
            label=t("lang_switch"),
        )

        with gr.Tabs():
            with gr.TabItem(t("tab_main"), id="main") as tab_main:
                comps, label_map = create_main_tab(run_transcribe, run_align)
                for k, v in comps.items():
                    all_comps.append(v)
                    all_i18n_keys.append(label_map[k])

            with gr.TabItem(t("tab_fixsrt"), id="fixsrt") as tab_fixsrt:
                comps, label_map = create_fixsrt_tab(run_fixsrt)
                for k, v in comps.items():
                    all_comps.append(v)
                    all_i18n_keys.append(label_map[k])

        all_comps.extend([tab_main, tab_fixsrt])
        all_i18n_keys.extend(["tab_main", "tab_fixsrt"])

        lang_radio.change(
            switch_lang,
            lang_radio,
            all_comps,
            js="""(lang) => {
                const isCN = lang === '中文';
                const tabs = document.querySelectorAll('[role="tab"]');
                if (tabs.length >= 2) {
                    tabs[0].textContent = isCN ? '🚀 快速转录/对齐' : '🚀 Quick Transcribe/Align';
                    tabs[1].textContent = isCN ? '🔧 SRT 修复' : '🔧 SRT Fix';
                }
            }""",
        )

    return app


def main():
    parser = argparse.ArgumentParser(description="Qwen3 音频处理工坊 GUI")
    parser.add_argument(
        "--lang",
        choices=list(LANG.keys()),
        default="中文",
        help="界面语言",
    )
    args = parser.parse_args()
    set_lang(args.lang)

    app = build_app()
    app.launch(inbrowser=True, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
