"""Browser UI (Gradio): chat with images, uploaded videos and a live webcam stream.

    vlm-assistant-web                     # http://127.0.0.1:7860
    vlm-assistant-web --host 0.0.0.0      # reachable from other devices (e.g. the robot's screen)

The webcam runs in the *browser*: frames are streamed to the server into a rolling buffer, so
"what just happened?" questions can use the last few seconds as a video clip.
"""

from __future__ import annotations

import argparse
import logging

import numpy as np

from .assistant import VisionAssistant
from .cli import LIVE_PROMPT, build_parser, config_from_args
from .media import FrameBuffer, MediaItem, guess_kind, load_image, sample_video

CAM_MODES = ["No camera", "Snapshot", "Last N seconds (clip)"]


class WebApp:
    def __init__(self, assistant: VisionAssistant):
        self.a = assistant
        self.frames = FrameBuffer(assistant.config.media)

    # ------------------------------------------------------------------ callbacks
    def on_frame(self, frame: np.ndarray | None) -> None:
        if frame is not None:
            self.frames.push(np.asarray(frame)[:, :, :3])

    def _collect_media(self, files, cam_mode: str, clip_seconds: float) -> list[MediaItem]:
        media: list[MediaItem] = []
        for f in files or []:
            path = f if isinstance(f, str) else getattr(f, "name", str(f))
            if guess_kind(path) == "video":
                media.append(sample_video(path, self.a.config.media))
            else:
                media.append(load_image(path, self.a.config.media))
        if cam_mode == CAM_MODES[1]:
            media.append(self.frames.snapshot())
        elif cam_mode == CAM_MODES[2]:
            media.append(self.frames.clip(float(clip_seconds)))
        return media

    def respond(self, message: str, files, cam_mode: str, clip_seconds: float, history: list[dict]):
        history = list(history or [])
        message = (message or "").strip()
        try:
            media = self._collect_media(files, cam_mode, clip_seconds)
        except (RuntimeError, FileNotFoundError, ValueError) as exc:
            history.append({"role": "assistant", "content": f"⚠️ {exc}"})
            yield history, message, files, ""
            return
        if not message and not media:
            yield history, "", files, ""
            return
        label = message or "(no text)"
        if media:
            label += "\n\n" + "\n".join(f"📎 {m.describe()}" for m in media)
        history.append({"role": "user", "content": label})
        history.append({"role": "assistant", "content": ""})
        yield history, "", None, ""
        reply = ""
        try:
            for chunk in self.a.chat(message or "Describe this.", media):
                reply += chunk
                history[-1]["content"] = reply
                yield history, "", None, ""
        except Exception as exc:  # show backend errors in the chat instead of crashing the UI
            history[-1]["content"] = reply + f"\n\n⚠️ {type(exc).__name__}: {exc}"
        yield history, "", None, self.a.backend.last_stats.summary()

    def live_comment(self, prompt: str, clip_seconds: float) -> str:
        try:
            clip = self.frames.clip(float(clip_seconds))
        except RuntimeError as exc:
            return f"⚠️ {exc} - start the webcam first."
        return "".join(self.a.oneshot(prompt or LIVE_PROMPT, [clip], max_new_tokens=160))

    def reset(self):
        self.a.reset()
        return [], ""

    # ------------------------------------------------------------------ layout
    def build(self):
        import gradio as gr

        with gr.Blocks(title="Vision Assistant") as demo:
            gr.Markdown(
                f"### Vision Assistant · `{self.a.config.model.model_id}` ({self.a.backend.name} backend)"
            )
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(height=560, label="Conversation")
                    with gr.Row():
                        msg = gr.Textbox(placeholder="Ask something...", show_label=False, scale=5)
                        send = gr.Button("Send", variant="primary", scale=1)
                    files = gr.File(
                        label="Attach images / videos", file_count="multiple",
                        file_types=["image", "video"],
                    )
                    with gr.Row():
                        stats = gr.Textbox(label="Last reply", interactive=False, scale=4)
                        clear = gr.Button("New conversation", scale=1)
                with gr.Column(scale=2):
                    cam = gr.Image(sources=["webcam"], streaming=True, label="Live webcam", type="numpy")
                    cam_mode = gr.Radio(CAM_MODES, value=CAM_MODES[0], label="Include webcam in next message")
                    clip_seconds = gr.Slider(2, self.a.config.media.webcam_buffer_seconds, value=6, step=1,
                                             label="Clip length (s)")
                    with gr.Accordion("Live commentary", open=False):
                        live_prompt = gr.Textbox(value=LIVE_PROMPT, lines=3, label="Prompt")
                        live_on = gr.Checkbox(label="Comment on the live feed every 10 s")
                        live_out = gr.Textbox(label="Latest observation", lines=4, interactive=False)
                        timer = gr.Timer(10.0, active=False)

            cam.stream(self.on_frame, inputs=cam, outputs=None, stream_every=0.25, concurrency_limit=None)
            inputs = [msg, files, cam_mode, clip_seconds, chatbot]
            outputs = [chatbot, msg, files, stats]
            msg.submit(self.respond, inputs, outputs)
            send.click(self.respond, inputs, outputs)
            clear.click(self.reset, None, [chatbot, stats])
            live_on.change(lambda on: gr.Timer(active=on), live_on, timer)
            timer.tick(self.live_comment, [live_prompt, clip_seconds], live_out)
        return demo


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.prog = "vlm-assistant-web"
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a public Gradio link (off by default)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    app = WebApp(VisionAssistant(config_from_args(args)))
    demo = app.build()
    demo.queue(default_concurrency_limit=1).launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
