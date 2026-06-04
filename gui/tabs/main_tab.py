import gradio as gr

from gui.i18n import t


def create_main_tab(run_transcribe, run_align):
    comps = {}
    label_map = {}

    gr.Markdown("## " + t("step1_title"))
    comps["audio_input"] = gr.File(
        label=t("audio_input"), file_types=["audio", "video"]
    )
    label_map["audio_input"] = "audio_input"

    with gr.Row():
        comps["lang_dropdown"] = gr.Dropdown(
            label=t("lang_label"),
            choices=["English", "Japanese", "Chinese"],
            value="English",
        )
        label_map["lang_dropdown"] = "lang_label"

        comps["model_radio"] = gr.Radio(
            label=t("model_label"),
            choices=[t("model_fast"), t("model_precise")],
            value=t("model_fast"),
        )
        label_map["model_radio"] = "model_radio"

    comps["transcribe_btn"] = gr.Button(t("btn_transcribe"), variant="primary")
    label_map["transcribe_btn"] = "btn_transcribe"

    comps["log_output"] = gr.Textbox(
        label=t("log_output"),
        lines=4,
        interactive=False,
        placeholder=t("placeholder_log"),
    )
    label_map["log_output"] = "log_output"

    comps["transcript_output"] = gr.Textbox(
        label=t("transcript_result"),
        lines=8,
        interactive=False,
        placeholder=t("placeholder_text"),
    )
    label_map["transcript_output"] = "transcript_result"

    gr.Markdown("---")
    gr.Markdown("## " + t("step2_title"))
    gr.Markdown("*" + t("model_hint_align") + "*")

    comps["text_input"] = gr.Textbox(
        label=t("text_input"), lines=6, placeholder=t("placeholder_text")
    )
    label_map["text_input"] = "text_input"

    with gr.Row():
        comps["align_lang"] = gr.Dropdown(
            label=t("lang_label"),
            choices=["English", "Japanese", "Chinese"],
            value="English",
        )
        label_map["align_lang"] = "lang_label"

    comps["align_btn"] = gr.Button(t("btn_align"), variant="primary")
    label_map["align_btn"] = "btn_align"

    comps["srt_output"] = gr.Textbox(
        label=t("srt_preview"),
        lines=8,
        interactive=False,
        placeholder=t("placeholder_srt"),
    )
    label_map["srt_output"] = "srt_preview"

    comps["download_srt"] = gr.DownloadButton(t("btn_download_srt"), visible=False)
    label_map["download_srt"] = "btn_download_srt"

    def transcribe_click(file, lang, model):
        if not file:
            yield [
                gr.update(value=t("no_audio")),
                gr.update(value=""),
                gr.update(value=""),
            ]
            return
        model_size = "1.7B" if model == t("model_precise") else "0.6B"
        yield [
            gr.update(value=t("transcribing")),
            gr.update(value=""),
            gr.update(value=""),
        ]
        result, error = run_transcribe(file.name, lang, model_size)
        if error:
            yield [
                gr.update(value=""),
                gr.update(value=t("transcribe_fail") + ": " + error),
                gr.update(value=""),
            ]
        else:
            yield [
                gr.update(value=t("transcribe_done") + f"，共 {len(result)} 字符"),
                gr.update(value=""),
                gr.update(value=result),
            ]

    def align_click(file, text, lang):
        if not file:
            yield [
                gr.update(value=t("no_audio")),
                gr.update(value=""),
                gr.update(visible=False),
            ]
            return
        if not text or not text.strip():
            yield [
                gr.update(value=t("no_text")),
                gr.update(value=""),
                gr.update(visible=False),
            ]
            return
        yield [
            gr.update(value=t("aligning")),
            gr.update(value=""),
            gr.update(visible=False),
        ]
        srt_content, error = run_align(file.name, text, lang)
        if error:
            yield [
                gr.update(value=""),
                gr.update(value=t("align_fail") + ": " + error),
                gr.update(visible=False),
            ]
        else:
            yield [
                gr.update(value=t("align_done") + f"，{len(srt_content)} 字符"),
                gr.update(value=srt_content),
                gr.update(value=srt_content, visible=True),
            ]

    def auto_fill_text(transcript):
        return gr.update(value=transcript if transcript else "")

    comps["transcribe_btn"].click(
        transcribe_click,
        inputs=[comps["audio_input"], comps["lang_dropdown"], comps["model_radio"]],
        outputs=[
            comps["log_output"],
            comps["log_output"],
            comps["transcript_output"],
        ],
    )

    comps["transcript_output"].change(
        auto_fill_text,
        inputs=[comps["transcript_output"]],
        outputs=[comps["text_input"]],
    )

    comps["align_btn"].click(
        align_click,
        inputs=[comps["audio_input"], comps["text_input"], comps["align_lang"]],
        outputs=[comps["log_output"], comps["srt_output"], comps["download_srt"]],
    )

    return comps, label_map
