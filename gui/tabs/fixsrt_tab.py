import gradio as gr

from gui.i18n import t


def create_fixsrt_tab(run_fixsrt):
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
            choices=["English", "Japanese", "Chinese"],
            value="English",
        )
        label_map["lang_dropdown"] = "lang_label"

        comps["indices_input"] = gr.Textbox(
            label=t("bad_indices"), placeholder="37, 38, 39"
        )
        label_map["indices_input"] = "bad_indices"

    gr.Markdown("*" + t("model_hint_fix") + "*")

    comps["fix_btn"] = gr.Button(t("btn_fix"), variant="primary")
    label_map["fix_btn"] = "btn_fix"

    comps["log_output"] = gr.Textbox(
        label=t("log_output"),
        lines=4,
        interactive=False,
        placeholder=t("placeholder_log"),
    )
    label_map["log_output"] = "log_output"

    comps["srt_output"] = gr.Textbox(
        label=t("srt_preview"),
        lines=10,
        interactive=False,
        placeholder=t("placeholder_srt"),
    )
    label_map["srt_output"] = "srt_preview"

    comps["download_fixed"] = gr.DownloadButton(
        t("btn_download_fixed"), visible=False
    )
    label_map["download_fixed"] = "btn_download_fixed"

    def fix_click(file, srt, lang, indices_str):
        if not file:
            yield [
                gr.update(value=t("no_audio")),
                gr.update(value=""),
                gr.update(visible=False),
            ]
            return
        if not srt:
            yield [
                gr.update(value=t("no_srt")),
                gr.update(value=""),
                gr.update(visible=False),
            ]
            return
        if not indices_str or not indices_str.strip():
            yield [
                gr.update(value=t("no_indices")),
                gr.update(value=""),
                gr.update(visible=False),
            ]
            return
        try:
            cleaned = indices_str.replace("[", "").replace("]", "")
            bad_indices = [
                int(x.strip()) for x in cleaned.split(",") if x.strip().isdigit()
            ]
            if not bad_indices:
                yield [
                    gr.update(value=t("no_indices")),
                    gr.update(value=""),
                    gr.update(visible=False),
                ]
                return
        except ValueError:
            yield [
                gr.update(value="无效的行号格式"),
                gr.update(value=""),
                gr.update(visible=False),
            ]
            return

        yield [
            gr.update(value=t("fixing")),
            gr.update(value=""),
            gr.update(visible=False),
        ]
        result, error = run_fixsrt(file.name, srt.name, bad_indices, lang)
        if error:
            yield [
                gr.update(value=""),
                gr.update(value=t("fix_fail") + ": " + error),
                gr.update(visible=False),
            ]
        else:
            yield [
                gr.update(value=t("fix_done")),
                gr.update(value=result),
                gr.update(value=result, visible=True),
            ]

    comps["fix_btn"].click(
        fix_click,
        inputs=[
            comps["audio_input"],
            comps["srt_file"],
            comps["lang_dropdown"],
            comps["indices_input"],
        ],
        outputs=[comps["log_output"], comps["srt_output"], comps["download_fixed"]],
    )

    return comps, label_map
