# Talk-to-Drive: project plan

A small robot in Isaac Sim that **understands spoken/written goals through Qwen3-VL**, drives with
an **energy-aware NMPC that respects motor and battery limits**, and **explains its own failures**.
The simulator's exact ground truth lets every part be measured.

```
 Isaac Sim (physics, scene, Jetbot, camera, fault injection)
     │ camera images, pose, wheel speeds                ▲ wheel-speed commands
     ▼                                                  │
 Qwen3-VL-4B ──"goal: blue pallet at (x, y)"──►  NMPC (motor + battery limits, energy cost)
     ▲                                                  │ tracking error, motor current
     └──────── "what went wrong?" (clip + telemetry) ◄──┘
```

## Milestones

| # | Milestone | Status | Test |
|---|---|---|---|
| M0 | Jetbot drives in Isaac Sim, camera frames saved | **code ready** (`sim/m0_hello_jetbot.py`) | runs, frames saved |
| M1 | NMPC + Kanayama baseline track figure-8 / slalom / circle | **code ready**, validated offline (`sim/m1_offline.py`, `sim/m1_isaac_tracking.py`) | position/heading RMSE, solve time |
| M2 | Motor torque–speed, current and battery model; energy cost; eco vs fast | next | Wh per run, constraint violations |
| M3 | Camera → Qwen3-VL-4B → object box → depth → goal pose | | success rate and position error vs ground truth |
| M4 | Fault injection (slip, motor derating, moving obstacle) + VLM diagnosis | | detection rate, explanation correctness |
| M5 | Replicator domain randomisation (lighting, textures) stress test | | VLM accuracy vs lighting |

## Code layout

```
src/talk2drive/
  robot.py            diff-drive params, kinematic simulator, DiffDriveRobot interface
  references.py       figure-8 / slalom / circle with smooth start/stop ramps
  controllers/nmpc.py CasADi + IPOPT NMPC (RK4, multiple shooting, wheel-lag model, motor limits)
  controllers/baseline.py  Kanayama tracking controller (baseline)
  runner.py           closed loop (same for offline + Isaac), metrics, CSV, plots
  isaac/jetbot.py     Isaac Sim adapter (scene, Jetbot, camera)
sim/                  runnable milestone scripts
```

The controllers only see the `DiffDriveRobot` interface (`get_state`, `set_wheel_speeds`, `step`),
so the same code runs against the offline model, Isaac Sim, and later the real robot.

## M1 offline results (kinematic model, 2 mm pose noise)

| Trajectory | NMPC RMSE | Kanayama RMSE | NMPC heading RMSE | Kanayama heading RMSE | NMPC mean solve |
|---|---|---|---|---|---|
| figure-8 | 0.30 cm | 0.39 cm | 0.64° | 0.77° | 3.5 ms |
| slalom | **1.81 cm** | 2.91 cm | **1.52°** | 3.21° | 4.7 ms |
| circle | 0.29 cm | 0.31 cm | 0.64° | 0.62° | 2.9 ms |
| figure-8, 15 % wheel slip | **1.89 cm** | 2.37 cm | **1.53°** | 3.12° | 3.5 ms |

The NMPC's advantage grows with path curvature and model mismatch. Isaac Sim's physics will
show how much of that survives contact dynamics, which is the motivation for M2.

## What to record from each Isaac run

* `runs/isaac/*_metrics.json`: compare with `runs/offline/`.
* GPU FPS and NMPC solve time while rendering.
* Anything that looks physically off (wheel hop, drift on spin-in-place).
