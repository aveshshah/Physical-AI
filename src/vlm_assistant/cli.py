"""Terminal chat interface.

    vlm-assistant                                   # interactive chat (local GPU)
    vlm-assistant --quant 4bit                      # fits in ~8 GB of VRAM
    vlm-assistant --backend openai --base-url http://jetson.local:8080/v1
    vlm-assistant --image photo.jpg -p "What is on the table?"   # one-shot
    vlm-assistant --video run.mp4  -p "Summarise this video."
"""

from __future__ import annotations

import argparse
import logging
import shlex
import sys
import time
from pathlib import Path

from .assistant import VisionAssistant
from .config import AppConfig
from .media import MediaItem, guess_kind, load_image, sample_video

HELP = """\
Type a message and press Enter. Attached media is sent with your next message.

  /image <path|url> [question]   attach an image (asks right away if a question follows)
  /video <path> [start] [end]    attach a recorded video (optional window in seconds)
  /cam [question]                attach a webcam snapshot
  /clip [seconds] [question]     attach the last N seconds of webcam footage (default 6)
  /live [interval] [prompt]      continuous webcam commentary every N s (Ctrl+C to stop)
  /attached                      list pending attachments    /clear  drop them
  /reset                         forget the conversation
  /system <prompt>               replace the system prompt (also resets)
  /save <file.json>              save the transcript
  /stats                         toggle speed statistics after each reply
  /help                          this help                  /quit  exit
"""

LIVE_PROMPT = (
    "This is a short clip from a live camera. In one or two sentences, describe what is happening "
    "right now, focusing on people, objects and any movement or change."
)


def _print_stream(chunks, prefix: str = "assistant> ") -> str:
    sys.stdout.write(prefix)
    sys.stdout.flush()
    parts = []
    for chunk in chunks:
        parts.append(chunk)
        sys.stdout.write(chunk)
        sys.stdout.flush()
    sys.stdout.write("\n")
    return "".join(parts)


def _float(s: str | None, default: float | None = None) -> float | None:
    try:
        return float(s) if s is not None else default
    except ValueError:
        return default


class Repl:
    def __init__(self, assistant: VisionAssistant, show_stats: bool = False):
        self.a = assistant
        self.pending: list[MediaItem] = []
        self.show_stats = show_stats

    # ------------------------------------------------------------------ helpers
    def send(self, text: str) -> None:
        media, self.pending = self.pending, []
        if media:
            print("  (sending " + "; ".join(m.describe() for m in media) + ")")
        try:
            _print_stream(self.a.chat(text, media))
        except KeyboardInterrupt:
            print("\n  [interrupted]")
            return
        if self.show_stats:
            print(f"  [{self.a.backend.last_stats.summary()}]")

    def attach(self, item: MediaItem, question: str) -> None:
        self.pending.append(item)
        print(f"  attached {item.describe()}")
        if question:
            self.send(question)

    def live(self, interval: float, prompt: str) -> None:
        cam = self.a.webcam
        seconds = max(2.0, interval)
        print(f"  live mode: every {interval:.0f}s, clips of {seconds:.0f}s. Ctrl+C to stop.")
        time.sleep(min(seconds, cam.cfg.webcam_buffer_seconds))
        try:
            while True:
                t0 = time.monotonic()
                clip = cam.clip(seconds)
                stamp = time.strftime("%H:%M:%S")
                _print_stream(self.a.oneshot(prompt, [clip], max_new_tokens=160), prefix=f"[{stamp}] ")
                time.sleep(max(0.0, interval - (time.monotonic() - t0)))
        except KeyboardInterrupt:
            print("\n  live mode stopped")

    # ------------------------------------------------------------------ commands
    def handle(self, line: str) -> bool:
        """Process one input line. Returns False to quit."""
        line = line.strip()
        if not line:
            return True
        if not line.startswith("/"):
            self.send(line)
            return True

        cmd, _, rest = line.partition(" ")
        rest = rest.strip()
        try:
            args = shlex.split(rest)
        except ValueError:
            args = rest.split()
        cmd = cmd.lower()

        try:
            if cmd in ("/quit", "/exit", "/q"):
                return False
            elif cmd in ("/help", "/h", "/?"):
                print(HELP)
            elif cmd == "/image":
                if not args:
                    print("  usage: /image <path|url> [question]")
                else:
                    self.attach(load_image(args[0], self.a.config.media), " ".join(args[1:]))
            elif cmd == "/video":
                if not args:
                    print("  usage: /video <path> [start_s] [end_s]")
                else:
                    start, end = _float(args[1] if len(args) > 1 else None), _float(args[2] if len(args) > 2 else None)
                    print("  sampling frames...")
                    self.attach(sample_video(args[0], self.a.config.media, start=start, end=end), "")
            elif cmd == "/cam":
                self.attach(self.a.webcam.snapshot(), rest)
            elif cmd == "/clip":
                seconds = _float(args[0]) if args else None
                question = " ".join(args[1:]) if seconds is not None else rest
                seconds = seconds or 6.0
                cam = self.a.webcam
                if cam.buffered_seconds() < min(seconds, cam.cfg.webcam_buffer_seconds) - 0.5:
                    wait = min(seconds, cam.cfg.webcam_buffer_seconds) - cam.buffered_seconds()
                    print(f"  recording {wait:.0f}s more...")
                    time.sleep(wait)
                self.attach(cam.clip(seconds), question)
            elif cmd == "/live":
                interval = _float(args[0]) if args else None
                prompt = " ".join(args[1:]) if interval is not None else rest
                self.live(interval or 5.0, prompt or LIVE_PROMPT)
            elif cmd == "/attached":
                print("\n".join("  " + m.describe() for m in self.pending) or "  (nothing attached)")
            elif cmd == "/clear":
                self.pending.clear()
                print("  attachments cleared")
            elif cmd == "/reset":
                self.a.reset()
                self.pending.clear()
                print("  conversation reset")
            elif cmd == "/system":
                if rest:
                    self.a.conversation.system_prompt = rest
                    self.a.reset()
                    print("  system prompt updated; conversation reset")
                else:
                    print(self.a.conversation.system_prompt)
            elif cmd == "/save":
                path = self.a.conversation.save(rest or f"transcript_{time.strftime('%Y%m%d_%H%M%S')}.json")
                print(f"  saved to {path}")
            elif cmd == "/stats":
                self.show_stats = not self.show_stats
                print(f"  stats {'on' if self.show_stats else 'off'}")
            else:
                print(f"  unknown command {cmd}; /help for the list")
        except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
            print(f"  error: {exc}")
        return True

    def run(self) -> None:
        try:
            import readline  # noqa: F401  (line editing + history)
        except ImportError:
            pass
        print(f"Vision assistant ready ({self.a.backend.name}: {self.a.config.model.model_id}). /help for commands.")
        while True:
            try:
                line = input("you> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not self.handle(line):
                break


# ---------------------------------------------------------------------------- entry point
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vlm-assistant", description="Local vision-language chat assistant (Qwen3-VL).")
    p.add_argument("-c", "--config", help="YAML config file (see configs/)")
    p.add_argument("--backend", choices=["transformers", "openai", "mock"])
    p.add_argument("--model", dest="model_id", help="HF model id / local path, or server model name")
    p.add_argument("--quant", choices=["none", "4bit", "8bit"], help="bitsandbytes quantization (transformers)")
    p.add_argument("--attn", dest="attn_implementation", help="attention implementation, e.g. flash_attention_2")
    p.add_argument("--base-url", help="OpenAI-compatible server URL (openai backend)")
    p.add_argument("--max-new-tokens", type=int)
    p.add_argument("--temperature", type=float)
    p.add_argument("--camera", help="webcam index, video file, or stream URL (RTSP/HTTP/GStreamer)")
    p.add_argument("--image", action="append", default=[], help="one-shot: image to analyse (repeatable)")
    p.add_argument("--video", help="one-shot: video file to analyse")
    p.add_argument("-p", "--prompt", help="one-shot question; exits after answering")
    p.add_argument("--stats", action="store_true", help="print speed statistics")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def config_from_args(args: argparse.Namespace) -> AppConfig:
    cfg = AppConfig.load(args.config)
    m = cfg.model
    for key in ("backend", "model_id", "attn_implementation", "base_url", "max_new_tokens", "temperature"):
        val = getattr(args, key)
        if val is not None:
            setattr(m, key, val)
    if args.quant is not None:
        m.quantization = None if args.quant == "none" else args.quant
    if args.camera is not None:
        cfg.media.webcam_source = int(args.camera) if args.camera.isdigit() else args.camera
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    cfg = config_from_args(args)

    oneshot = bool(args.prompt or args.image or args.video)
    media: list[MediaItem] = []
    try:
        for path in args.image:
            media.append(load_image(path, cfg.media))
        if args.video:
            if guess_kind(args.video) == "image":
                media.append(load_image(args.video, cfg.media))
            else:
                media.append(sample_video(args.video, cfg.media))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not Path(str(cfg.model.model_id)).exists() and cfg.model.backend == "transformers":
        print(f"Loading {cfg.model.model_id} (first run downloads ~17 GB)...", file=sys.stderr)
    assistant = VisionAssistant(cfg)
    try:
        if oneshot:
            prompt = args.prompt or "Describe this in detail."
            _print_stream(assistant.chat(prompt, media), prefix="")
            if args.stats:
                print(f"[{assistant.backend.last_stats.summary()}]", file=sys.stderr)
        else:
            repl = Repl(assistant, show_stats=args.stats)
            repl.pending.extend(media)
            repl.run()
    finally:
        assistant.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
