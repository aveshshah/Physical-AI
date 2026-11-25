# Physical AI: Vision Assistant (Qwen3-VL-8B, fully local)

Phase 1 of the robot-assistant project: a **local, GPU-accelerated conversational assistant** that
can discuss **images, a live webcam feed and recorded videos**, built on
[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct). Nothing leaves your
machine after the one-time model download.

Memory, task planning and physical actions come later. See [docs/ROADMAP.md](docs/ROADMAP.md).
The code is already split so the same perception/chat core can be moved onto a small robot.

```
                ┌──────────── media.py ────────────┐
 image file ───►│ load_image()       ImageItem      │
 video file ───►│ sample_video()     VideoItem      │──┐
 webcam/RTSP ──►│ Webcam ─► FrameBuffer (rolling)   │  │   conversation.py         backends/
 browser cam ──►│ FrameBuffer (web UI)              │  ├─► Conversation ─► messages ─► transformers (local GPU)
                └───────────────────────────────────┘  │   (history, media trim)    ├─► openai  (llama.cpp / vLLM / Ollama)
                                                       │                            └─► mock    (tests, UI work)
                          cli.py / webui.py ◄──────────┘── assistant.py (VisionAssistant)
```

## Install

Python 3.10+, an NVIDIA GPU, and a CUDA build of PyTorch.

```bash
git clone https://github.com/aveshshah/Physical-AI.git
```
```bash
cd Physical-AI
```
```bash
python3 -m venv .venv
```
```bash
source .venv/bin/activate
```

Install PyTorch for your CUDA version first, using the command from https://pytorch.org/get-started/locally/. For example:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

```bash
pip install -e ".[gpu,web,dev]"
```

Only for GPUs with less than 20 GB of VRAM:

```bash
pip install -e ".[quant]"
```

Check what you have and which config to use:

```bash
python scripts/check_env.py
```

| GPU VRAM | Config | Notes |
|---|---|---|
| ≥ 20 GB (3090/4090/5090, A5000…) | `configs/default.yaml` | bf16, best quality |
| 8–16 GB (3060 12 GB, 4060 Ti, 4070…) | `configs/low_vram.yaml` | 4-bit NF4, ~7 GB |
| < 8 GB, or no CUDA | `configs/edge_llamacpp.yaml` | GGUF via llama.cpp/Ollama, or a smaller Qwen3-VL (4B/2B) |

The first run downloads the model (~17 GB) into the Hugging Face cache. To pre-download it:

```bash
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct
```

Optional faster attention (Ampere or newer): `pip install flash-attn --no-build-isolation`, then use `--attn flash_attention_2`.

## Use

### Terminal chat

```bash
vlm-assistant
```

```bash
vlm-assistant -c configs/low_vram.yaml --stats
```

```
you> /image ~/Pictures/desk.jpg What tools are on the desk?
you> Which one would you use to tighten an M4 bolt?
you> /video ~/Videos/test_run.mp4 12 30          # seconds 12-30 of the video
you> Did the robot hit the obstacle?
you> /cam Is anyone in the room?                 # webcam snapshot
you> /clip 6 What did I just do with my hands?   # last 6 s of webcam as a video
you> /live 5                                     # running commentary every 5 s (Ctrl+C)
```

`/help` lists all commands (`/reset`, `/system`, `/save`, `/stats`…).

One-shot (scriptable):

```bash
vlm-assistant --image photo.jpg -p "List every object with its approximate position."
```

```bash
vlm-assistant --video run.mp4 -p "Summarise this video with timestamps."
```

The camera can be a device index, a video file, an RTSP/HTTP stream or a GStreamer pipeline:

```bash
vlm-assistant --camera rtsp://192.168.1.50:8554/cam
```

### Browser UI

```bash
vlm-assistant-web -c configs/default.yaml
```

Open http://127.0.0.1:7860. You can:

* attach images and videos;
* start the live webcam (it runs in the browser and streams frames to a rolling server-side buffer);
* choose whether the next message includes a **snapshot** or the **last N seconds as a clip**;
* turn on **live commentary** (a short description of the feed every 10 s).

Add `--host 0.0.0.0` to open it from another device on your LAN, such as a tablet strapped to the robot.

### Python API

```python
from vlm_assistant import AppConfig, VisionAssistant, load_image, sample_video

a = VisionAssistant(AppConfig.load("configs/default.yaml"))
print(a.ask("What is this?", [load_image("part.jpg")]))
print(a.ask("Is it damaged?"))                       # follow-up, same context
for chunk in a.chat("Summarise.", [sample_video("run.mp4")]):
    print(chunk, end="", flush=True)
snap = a.webcam.snapshot()                           # camera starts lazily
```

## Running against a model server (and later on the robot)

`--backend openai` talks to any OpenAI-compatible server. That is how the assistant will run on the
small robot: the robot's computer runs the model server, or a GPU desktop on the network does.

```bash
llama-server -hf Qwen/Qwen3-VL-8B-Instruct-GGUF --port 8080 -ngl 99 -c 16384
```

```bash
vlm-assistant -c configs/edge_llamacpp.yaml
```

With **Ollama**, run `ollama pull qwen3-vl:8b` and set `base_url: http://127.0.0.1:11434/v1` and `model_id: qwen3-vl:8b`.
With **vLLM**, run `vllm serve Qwen/Qwen3-VL-8B-Instruct` and set `base_url: http://127.0.0.1:8000/v1`.

Over this backend, videos are sent as a sequence of time-stamped JPEG frames, which every server
supports. So keep `video_max_frames` small (8–16).

## Settings that matter

| Setting | Effect |
|---|---|
| `media.image_max_side` | Qwen3-VL uses one visual token per 32×32 px. 1280 px (16:9) is about 1.6k tokens. Lower it to save VRAM and latency; raise it for small text. |
| `media.video_fps`, `video_max_frames` | Frames sampled from videos. Long videos are spread evenly over the whole duration. |
| `media.webcam_buffer_seconds/fps` | How far back `/clip` and the web UI clip can look. |
| `chat.keep_media_turns` | Only the last N user turns re-send their images/videos; older ones become text placeholders. This is the main control on context growth. |
| `chat.max_history_turns` | Number of exchanges kept. |

## Tests

```bash
pytest -q
```

The tests use the mock backend, synthetic videos, a fake camera and a fake SSE server, so they need
neither a GPU nor a model.

```bash
python scripts/tiny_qwen3vl_smoke.py
```

This script builds a tiny **random-weight** Qwen3-VL locally, with no download, and runs the real
transformers backend on an image and a video. It checks that preprocessing, the chat template,
video timestamps and streaming generation work with the installed `transformers` and
`qwen-vl-utils` versions. Run it after upgrading either library.

## Layout

```
src/vlm_assistant/
  config.py          dataclass config (YAML + CLI overrides)
  media.py           images, video sampling, Webcam thread, FrameBuffer
  conversation.py    history, media trimming, neutral message format
  assistant.py       VisionAssistant: chat / ask / oneshot / webcam
  backends/          transformers (local GPU), openai (servers), mock
  cli.py             terminal REPL + one-shot mode
  webui.py           Gradio UI with live browser webcam
configs/             default / low_vram / edge_llamacpp
scripts/             check_env.py, tiny_qwen3vl_smoke.py
docs/ROADMAP.md      memory → planning → actions (NMPC) → robot
```
