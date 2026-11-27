# Physical AI: Vision Assistant (Qwen3-VL-8B, fully local)

Phase 1 of the robot-assistant project: a **local, GPU-accelerated conversational assistant** that
can discuss **images, a live webcam feed and recorded videos**, built on
[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct). Nothing leaves your
machine after the one-time model download.

Memory, task planning and physical actions come later. See [docs/ROADMAP.md](docs/ROADMAP.md).

**Talk-to-Drive (Isaac Sim project):** NMPC-driven Jetbot in NVIDIA Isaac Sim, guided by
Qwen3-VL-4B. See [docs/TALK2DRIVE.md](docs/TALK2DRIVE.md) and the setup guide
[docs/ISAAC_SIM_SETUP.md](docs/ISAAC_SIM_SETUP.md).
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

## Talk-to-Drive: VLM planner + NMPC on the Magna robot (Isaac Sim 6.2, Mcity)

The vision model here also drives a robot in simulation. The Magna tadpole delivery robot (two
front wheels, one rear) drives a 1,041 m route in the Mcity test track:

* the speed and path tracker is the CARLA tadpole NMPC from `~/Documents/carla_nmpc_project`,
  imported unchanged;
* the behaviour and local planner is a fine-tuned **Qwen3-VL-4B**: it looks at the front camera
  and decides whether to keep following the route, swerve left or right (and by how much), or stop.

Plan and earlier milestones: [docs/TALK2DRIVE.md](docs/TALK2DRIVE.md). Isaac setup:
[docs/ISAAC_SIM_SETUP.md](docs/ISAAC_SIM_SETUP.md).

```
 front camera ──► Qwen3-VL-4B + LoRA ──► {"blocker": {label, box}, "action": bypass_left, "offset_m": 1.2}
 (route drawn                │
  on the road)               ▼
             depth in the box ──► where the blocker is ──► smooth sideways-shift path
                                                              │  checked: stays on pavement,
                                                              │  depth camera sees it clear
                                                              ▼
                                        CARLA tadpole NMPC (speed + steering) ──► robot
             depth safety check: brake only (never steers)
```

### Road-load disturbances (M2)

`sim/m2_mcity_tadpole_nmpc.py --disturbance <scenario>` changes the road and the air without
telling the NMPC:

* road friction (wet, low-µ, ice patches);
* grade ±6 %;
* wind with gusts;
* extra drag and rolling resistance.

`sim/run_disturbance_sweep.sh` runs every scenario, and `sim/summarise_disturbances.py` tabulates
them. The NMPC holds its target speed within about 0.1–0.15 m/s RMS under grade, wind, drag and a
combination of them; lap time changes by under 0.5 s. It fails on:

* µ 0.35: the rear wheel spins;
* µ 0.2 patches: the front wheels lock, because there is no ABS;
* a 14 m/s storm with 21 m/s gusts: the robot rolls over.

### Suspension

`sim/build_robot_from_fbx.py --suspension` adds prismatic struts and a front anti-roll bar, scaled
down from passenger-car data:

| Setting | Value |
|---|---|
| Ride frequency | 1.8 Hz front, 2.0 Hz rear |
| Damping ratio | 0.4 |
| Roll gradient | 3°/g |
| Wheel travel | 80 mm |

The sprung robot is `--robot magna_susp`; `sim/check_suspension.py` checks it on flat ground. With
the bend speed limit `--ay-curve 2.4`, the lap takes 125.5 s.

### VLM behaviour planner (M3)

1. **Data.** `sim/gen_planner_data.py` renders 4,000 Mcity camera images with obstacles placed
   on the route. Each image is labelled from the scene geometry with follow / bypass_left /
   bypass_right / stop, the offset and the blocker's box. `sim/m3_planner_testset.py` builds a
   separate 36-image test set.
2. **Fine-tune.** `scripts/finetune_planner.py` runs QLoRA on Qwen3-VL-4B:
   * 4-bit NF4 base, LoRA r=16 on the language model, 2 epochs, about 1 h 45 min;
   * the answer gives the blocker box first, then the action, and the action and offset count
     5× in the loss.
3. **Serve.** `scripts/vlm_ground_server.py --adapter runs/planner_lora`.
4. **Drive.** `sim/m3_vlm_planner.py --prompt ft` places 5 obstacles on the route and runs the
   lap with the window open.

Accuracy on 400 held-out images:

| Model | follow | bypass_left | bypass_right | stop | Overall | Offset error |
|---|---|---|---|---|---|---|
| Base Qwen3-VL-8B, no fine-tuning (36-image test set) | – | – | – | – | 64 % | – |
| Fine-tuned 4B, first run (action written first) | 97 % | 94 % | 1 % | 13 % | 60 % | 0.42 m |
| Fine-tuned 4B, box first, action weighted 5× | 97.5 % | 78 % | 68 % | 45 % | **77 %** | 0.21 m |

Obstacle lap on Mcity (magna_susp, 4 blockers plus 2 side cones):

| Run | Result |
|---|---|
| 1 | VLM switched side 13 m from the box and hit a cone |
| 2 | Swerved over a 15 cm kerb (too low for the depth check) and got stuck |
| 3 | Stopped cleanly before a box at a kerb corner: the VLM's offset was too small |
| 4 | **Full lap, 0 collisions, 134.1 s**, closest gap 0.30 m |
| Live rerun | **Full lap, 0 collisions, 127.5 s** (obstacle-free lap: 125.5 s); VLM about 1.06 s per answer |

Fixes added after runs 1–3:

* within 18 m of an obstacle, a change of side needs the depth camera to agree;
* a swerve must keep the robot's edge on pavement (road map);
* a swerve that is too tight is first widened on the VLM's side, in 0.4 m steps up to 1.2 m.

What the VLM does: it decides every swerve, its side and its rough size, and which object is in the
way. Without it the robot stops at the first obstacle, because the depth check can only brake.

Not yet done:

* the VLM's side choice is still imperfect: it handled 1 of 4 blockers without any check stepping
  in during run 4;
* it does not see obstacles closer than about 2 m;
* each answer takes about 1–2 s.

Run the lap (Isaac venv in `~/isaac-env`, VLM venv in `.venv`):

```bash
.venv/bin/python scripts/vlm_ground_server.py --config configs/sim_4b.yaml --adapter runs/planner_lora
```

```bash
OMNI_KIT_ACCEPT_EULA=YES DISPLAY=:1 ~/isaac-env/bin/python sim/m3_vlm_planner.py --prompt ft
```

`runs/` is not committed: rebuild the data and the adapter with the scripts above.

### Talk to drive (M4)

`sim/m4_talk_to_drive.py` drives the robot round Mcity on spoken or typed commands. The robot plans
the global route on the OpenDRIVE road graph and the CARLA NMPC tracks it.

| Say | What happens |
|---|---|
| go / stop / emergency stop | drive at 8 m/s, normal stop in the lane, full emergency brake |
| accelerate / brake (more) | speed up or slow down, 0.5 m/s² harder with each command |
| hold speed / set speed 20 km/h | keep the speed, or go to a given speed |
| turn left / right / go straight | at the first junction within 200 m that allows it (or "no left turn possible") |
| u-turn / turn around | turns where it is, also from a stop or a parking spot; on one-way or narrow roads it goes round a roundabout or block |
| change lane left / right | into the next lane going the same way (never into oncoming traffic) |
| steer left / right | 0.5 m to the side inside the lane, back to the centre before the next junction |
| pull over | parks at the right-hand edge, away from junctions |
| what's ahead / status / help / reset | junctions coming up; speed, mode and position; the list of commands; back on the wheels after tipping |
| put a cone 30 m ahead / remove the objects | puts a cone, box, barrel or ball on the road ahead (the robot can't see it and will hit it); removes them |
| save the video | keeps the dash cam footage from 60 s before to 60 s after |

A move asked for while stopped also starts the robot. A collision is announced once, followed by
emergency braking.

In the Isaac window, shapes made with Create → Shape become solid on their own. If a shape is left
at the map centre, it is moved onto the road 20 m ahead of the robot. This replaces Isaac 6.2's
Add → Physics menu, which is broken. Pause, Stop and Play in the window are safe: after a Stop, the
robot is put back where it was, stopped. Phrases the keyword parser doesn't know
("let me out here", "give the cyclist more room") go to the local Qwen3-VL-4B model, which picks
one of the commands above or explains why it can't.

```bash
.venv/bin/python scripts/vlm_ground_server.py --config configs/sim_4b.yaml --adapter runs/planner_lora   # for unknown phrases
OMNI_KIT_ACCEPT_EULA=YES DISPLAY=:1 ~/isaac-env/bin/python sim/m4_talk_to_drive.py
.venv/bin/python scripts/voice.py          # push-to-talk (Enter), spoken replies; or:
.venv/bin/python scripts/say.py "turn left"
```

Voice runs offline: Whisper small.en for speech recognition and Piper for the replies (`pip install -e .[voice]`).
Speak over the robot to interrupt it.

### Dash cam (loop recording)

`scripts/dashcam.py` records the robot's front camera like a dash cam, as its own process (if the
assistant or the language model fails, recording goes on). It follows section 5 of the AI Road
Companion report:

* **Loop recording:** 640 × 360 at 10 frames per simulated second, in 30-second MP4 videos in
  `runs/dashcam/rolling/`. The last 10 minutes are kept and older videos are deleted
  (`--keep` and `--segment` change this).
* **Incidents:** a collision (the robot's body touching anything, or a tyre touching an obstacle),
  a rollover, "save the video" or `dashcam.py trigger` keeps the footage from 60 s before to 60 s
  after (`--pre` and `--post` change this), in whole 30 s videos. Another event during that time
  extends it to 60 s after the last one. Incidents are never deleted automatically.
* **Sealing:** the clip goes to `runs/dashcam/incidents/<id>/`, with `incident.mp4` joined from the
  segments, every frame's simulation, wall-clock and monotonic time, and `manifest.json` with the
  SHA-256 of every file. The files are then made read-only.
* **Verification:** the clip is marked `incomplete`, with the reason, when the footage doesn't
  reach 60 s back, the simulation stops early, or frames are missing. After a crash, the recorder
  seals unfinished incidents as incomplete when it starts again.
* **Deletion:** loop cleanup only touches `rolling/`. `dashcam.py delete ID` refuses unless a
  person types the incident ID at a terminal.

```bash
.venv/bin/python scripts/dashcam.py                 # 1. the recorder (start before or after the simulation)
OMNI_KIT_ACCEPT_EULA=YES DISPLAY=:1 ~/isaac-env/bin/python sim/m4_talk_to_drive.py   # 2. sends the camera
.venv/bin/python scripts/dashcam.py list            # incidents, complete or not, and why
.venv/bin/python scripts/dashcam.py verify ID       # re-check the hashes
```

The simulation never waits for the recorder. If the recorder is slow or not running, frames are
dropped and the gap in frame numbers shows in the manifest. `--box-at 520` puts a box on the road
for a crash test.

Tested by driving into a box at 74.7 s: incident complete, footage 14.0–134.9 s, 1210 frames,
1500 sent and 0 dropped. The `tests/test_dashcam.py` tests cover coverage, early trigger, repeated
collisions, early exit, dropped frames, tampering, deletion and crash recovery.

Protection is limited: read-only files stop ordinary writes by the same user, not root. For real
evidence, run the recorder as a separate `roadrec` account that owns the store, as the report
describes.

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
src/talk2drive/      Isaac robot wrapper, Mcity route, tadpole NMPC driver, disturbances, VLM planner
sim/                 Isaac Sim scripts (M1–M4, data generation, suspension check, sweeps)
```
