"""Qwen3-VL grounding service for the simulator (runs in the VLM venv, not in Isaac's).

    .venv/bin/python scripts/vlm_ground_server.py --config configs/sim_4b.yaml     # port 8765

POST /ground  {"image_png_b64": "...", "query": "the blue barrel"}
  -> {"boxes": [{"bbox_2d": [x1, y1, x2, y2], "label": "..."}], "raw": "...", "ms": 812.0}
GET  /health  -> {"model": "...", "ready": true}

Boxes are normalised to 0..1000 (Qwen3-VL's grounding convention). One request at a time.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image  # noqa: E402

from vlm_assistant.assistant import VisionAssistant  # noqa: E402
from vlm_assistant.config import AppConfig  # noqa: E402
from vlm_assistant.grounding import GROUND_PROMPT, parse_boxes  # noqa: E402
from vlm_assistant.media import load_image  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sim_4b.yaml")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--adapter", default=None, help="LoRA adapter to load on top of the model (scripts/finetune_planner.py)")
    args = ap.parse_args()

    cfg = AppConfig.load(args.config)
    cfg.model.temperature = 0.0  # grounding: deterministic
    t0 = time.time()
    assistant = VisionAssistant(cfg)
    if args.adapter:
        from peft import PeftModel

        assistant.backend.model = PeftModel.from_pretrained(assistant.backend.model, args.adapter)
        assistant.backend.model.eval()
        print(f"[vlm] adapter {args.adapter}", flush=True)
    print(f"[vlm] {cfg.model.model_id} ({cfg.model.quantization or cfg.model.dtype}) loaded in {time.time() - t0:.0f} s", flush=True)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path == "/health":
                self._send(200, {"model": cfg.model.model_id + (f" + {args.adapter}" if args.adapter else ""), "ready": True})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path != "/ground":
                self._send(404, {"error": "not found"})
                return
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                img = Image.open(io.BytesIO(base64.b64decode(req["image_png_b64"]))).convert("RGB")
                prompt = req.get("prompt") or GROUND_PROMPT.format(query=req["query"])
                with lock:
                    t = time.perf_counter()
                    raw = "".join(assistant.oneshot(prompt, [load_image(img, cfg.media)]))
                    ms = (time.perf_counter() - t) * 1e3
                self._send(200, {"boxes": parse_boxes(raw), "raw": raw, "ms": ms})
            except Exception as e:  # report, keep serving
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt, *a):
            pass

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[vlm] serving on http://{args.host}:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
