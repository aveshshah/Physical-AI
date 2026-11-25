# Roadmap: from vision chat to a small robot

The guiding idea: **the VLM decides *what* to do; your own planners and NMPC decide *how*.**
We are not adopting a VLA (vision-language-action) model that outputs joint commands end to end.
Qwen3-VL stays a slow, deliberative layer (~0.2–2 Hz). It emits structured goals, and existing
motion-planning and NMPC code turns them into safe, smooth, constraint-respecting motion at control
rate (50–1000 Hz).

```
 camera(s) ─► Perception (Qwen3-VL)  ──►  Memory  ──►  Task planner (Qwen3-VL / LLM, tool calls)
                 ▲  scene description        ▲              │ structured skill calls (JSON)
                 │  object detections + 2D   │              ▼
                 │  boxes / points           │        Skill layer  (navigate_to, look_at, pick, say …)
                 │                           │              │ goals, constraints, references
                 │                           │              ▼
                 │                           │        Motion planner + NMPC  (your existing stack)
                 │                           │              │ torques / velocities @ control rate
                 └──── feedback: success/failure, new images ◄── Hardware + safety supervisor
```

## Phase 1: local vision chat ✅ (this repository)

* Qwen3-VL-8B on a local GPU (bf16 or 4-bit), streaming replies.
* Images, recorded videos (uniform frame sampling with timestamps), webcam snapshots, rolling
  webcam clips and live commentary.
* Terminal + browser UI; an OpenAI-compatible backend so the model can run on a separate server.

**Exit criteria:** answer quality and latency you are happy with on your GPU. Use `--stats` to
record time-to-first-token and tokens/s for typical image, clip and video prompts.

## Phase 2: structured perception

Make the model's output machine-readable before giving it any authority.

* **Grounding:** Qwen3-VL can return 2D boxes and points as JSON. Its coordinates are on a 0–1000
  normalised grid. Add `perceive(image) -> list[Detection]` with schema validation and retry.
* **Depth/3D:** project boxes to 3D with an RGB-D camera (RealSense/OAK-D) or stereo, and transform
  them into the robot frame using the camera extrinsics.
* **Change detection:** cheap frame differencing gates when the VLM is called on the live feed, which
  saves GPU time and power on the robot.

## Phase 3: memory

* **Short-term:** the existing conversation window plus a rolling log of scene descriptions with
  timestamps ("event memory").
* **Long-term:** a small local vector store such as SQLite + `sqlite-vec` or FAISS. Store text
  embeddings of observations, user facts and task outcomes, plus thumbnail paths. Retrieve the
  top-k entries and put them in the prompt.
* **Spatial memory:** object → last seen pose (map frame) + time. This is what lets the robot
  answer "where did I leave my keys?" and plan "go to the kitchen table".

## Phase 4: autonomous task planning

* Give the model **tools** (function calling with a JSON schema) instead of free text:
  `look_at(target)`, `navigate_to(pose|object)`, `pick(object_id)`, `place(location)`, `say(text)`,
  `ask_user(question)`, `wait(seconds)`.
* Use a **plan → act → observe → replan** loop: after each skill, capture a new image and ask the
  model whether the step succeeded before continuing. Bound the number of steps, time and retries.
* Keep a symbolic **task state** (goals, completed steps, failures) outside the model. It is the
  source of truth, and the model only proposes the next step.
* Evaluate in simulation first (Isaac Sim / Gazebo / MuJoCo) with scripted scenarios.

## Phase 5: physical actions, reusing motion planning + NMPC

The skill layer is where the AI meets your control experience:

| Skill | Model provides | Your stack does |
|---|---|---|
| `navigate_to(goal)` | target object/pose from perception + memory | global path (A*/RRT*/lattice) → NMPC tracking with obstacle, velocity and acceleration constraints |
| `look_at(target)` | pixel/3D target | pan-tilt or base yaw reference → MPC/PID |
| `pick(object)` | object 3D pose, grasp hint | grasp pose selection, IK, trajectory optimisation, NMPC with joint and collision constraints |
| `follow(person)` | person track | NMPC with distance-keeping cost and safety constraints |

Design rules:

1. **The model never commands actuators directly.** It calls skills, and each skill validates its
   inputs (workspace limits, reachable poses, known objects).
2. **Safety supervisor:** an independent, model-free layer for e-stop, geofence, speed and force
   limits, collision checks and a watchdog on the planner heartbeat. NMPC constraints are the first
   line; the supervisor is the last.
3. **Asynchronous rates:** the VLM can take 0.5–3 s per call, so the NMPC must keep tracking the
   last valid reference and degrade gracefully (stop, hold) when no new goal arrives.
4. **Interfaces:** a ROS 2 action server per skill (goal / feedback / result) fits this naturally.
   A plain Python interface also works for a first prototype.

## Phase 6: deployment on the small robot

Options, from most to least on-board compute:

| Robot computer | Suggested model | Notes |
|---|---|---|
| Jetson AGX Orin 32/64 GB | Qwen3-VL-8B (Q4/Q8 GGUF via llama.cpp, or vLLM/TensorRT-LLM) | comfortable; room for planner + NMPC |
| Jetson Orin NX 16 GB | Qwen3-VL-8B Q4_K_M or Qwen3-VL-4B | good balance |
| Jetson Orin Nano 8 GB | Qwen3-VL-4B / 2B Q4 | 8B fits only with a small context and little else running |
| Raspberry Pi 5 / small SBC | none on-board | stream the camera to a GPU desktop over Wi-Fi (`--backend openai` + RTSP camera) |

What already carries over from Phase 1:

* `--backend openai` → `llama-server` on the robot or on the LAN.
* `--camera` accepts RTSP or GStreamer pipelines, for example a Jetson CSI camera:
  `"nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! appsink"`
  (this needs OpenCV built with GStreamer, which JetPack provides).
* The web UI can be opened from a phone or tablet as the robot's face/screen (`--host 0.0.0.0`).

Still needed for a "chatbot" robot: **speech**. Use local ASR (faster-whisper / whisper.cpp) in
front of `assistant.chat()` and local TTS (Piper / Kokoro) after it, with wake-word detection
(openWakeWord) and barge-in handling.

## Suggested next steps

1. Run Phase 1 on your GPU and collect `--stats` numbers for images, clips and videos.
2. Pick the target robot computer. That decides 8B vs 4B and llama.cpp vs TensorRT-LLM.
3. Start Phase 2 with a `perceive()` function that returns validated JSON detections, plus a small
   evaluation set of your own images.
