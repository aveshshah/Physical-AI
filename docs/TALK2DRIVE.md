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
| M0 | Jetbot drives in Isaac Sim, camera frames saved | **done** on Isaac Sim 6.2 (`sim/m0_hello_jetbot.py`) | runs, frames saved |
| M1 | NMPC + Kanayama baseline track figure-8 / slalom / circle | **done**: offline and Isaac Sim 6.2 (`sim/m1_offline.py`, `sim/m1_isaac_tracking.py`) | position/heading RMSE, solve time |
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
  isaac/jetbot.py     Isaac Sim adapter (scene, Jetbot, camera, encoder wheel speeds, calibrated params)
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

## Jetbot calibration (Isaac Sim 6.2, `sim/calibrate_jetbot.py`)

| Quantity | Nominal model | Isaac Jetbot | How measured |
|---|---|---|---|
| wheel radius | 30.0 mm | **33.6 mm** | path speed / encoder speed, straight runs at 2-12 rad/s (spread < 0.1 mm); matches the 67.25 mm wheel geometry |
| wheel base | 112.5 mm | **121.2 mm** (effective) | spin in place at 2, 4, 8 rad/s; wheel centres are 113.9 mm apart, tyre scrub widens it |
| wheel-speed tracking | 1.0 | 1.002 | encoder / command; no gain correction needed |
| wheel lag | 50 ms | < 1 physics step (17 ms) | step response; the NMPC keeps 50 ms (RK4 at 50 ms is unstable below ~18 ms, and tau also sets its acceleration limit) |

Isaac's reported wheel-joint velocities are biased by about +0.35 rad/s on 6.2 (joint positions advance at
the commanded rate), so `IsaacJetbot` differences joint positions like encoders. A held-out arc (4, 8 rad/s)
checks the fit: the nominal model is off by -10.6 % (speed) and -4.3 % (yaw rate); the calibrated one by
0.0 % and -0.6 %.

## M1 Isaac Sim 6.2 results (figure-8, grid, headless, 60 Hz physics)

| Model in controller | NMPC RMSE | Kanayama RMSE | NMPC heading RMSE | Kanayama heading RMSE | NMPC mean solve |
|---|---|---|---|---|---|
| nominal (30 mm / 112.5 mm, reported joint velocities) | 1.03 cm | 1.40 cm | 0.54° | 1.61° | 2.6 ms |
| **calibrated (33.6 mm / 121.2 mm, encoder speeds)** | **0.21 cm** | **0.19 cm** | **0.19°** | 0.25° | 2.5 ms |

With the nominal model, Isaac errors match the offline model run with a 10 % speed excess (0.99 / 1.24 cm),
so the gap was almost entirely calibration, not contact dynamics. Once calibrated, both controllers track
to about 2 mm and NMPC's edge on the gentle figure-8 disappears. The max error (0.95 cm) is the robot
shifting while it settles on the ground before t = 0; after 3 s it is under 0.4 cm. Slalom and fault
injection (M4) should separate the controllers again.

## CARLA tadpole NMPC on Mcity (25 Nov 2026, `sim/m2_mcity_tadpole_nmpc.py`)

The NMPC stack from the CARLA project (acados full-SQP pose/dynamics NMPC, RouteSpeedPlanner with
stitching, force-disturbance observer, 2-DOF force split) drives the Magna robot round a 1041 m
Mcity loop (57 OpenDRIVE roads, 3.5 m lanes, grades up to 9 %) at that project's V_MAX of 15 m/s.

| Configuration | Lap | Peak / mean speed | a_x (0.2 s) | max a_y | max CTE | max heading err | steering reversals | Rules |
|---|---|---|---|---|---|---|---|---|
| comfort (project default) | 170.0 s | 15.1 / 6.1 m/s | +2.82 / -3.36 | 1.38 | 10.7 cm | 7.9 deg | 4 | pass |
| sport | 140.1 s | 15.1 / 7.4 | **+3.85** / -3.42 | 1.86 | 11.6 cm | 7.5 deg | 24 | 1B fail |
| sport + grade governor | 140.1 s | 15.1 / 7.4 | **+3.43** / -3.40 | 1.86 | 11.6 cm | 7.5 deg | 23 | 1B fail |
| sport + governor + CG velocities | 140.1 s | 15.0 / 7.4 | +2.65 / -3.39 | 1.86 | 11.7 cm | 7.7 deg | 22 | pass |
| **+ speed-scheduled horizon (default)** | **140.1 s** | 15.0 / 7.4 | +2.64 / -3.39 | 1.82 | 18.0 cm | 8.1 deg | **1** | **pass** |

Limits: V_MAX 15 m/s (1A), +3.0 / -3.5 m/s^2 (1B), AY_MAX 3.0 m/s^2 rollover envelope, CTE 10 % of the
lane = 35 cm and heading 15 deg (3C). The sport lap is bounded by the plan (V_MAX, 2.2 / 3.1 m/s^2,
AY_CURVE 1.65 m/s^2), not by the controller.

Findings on the way:

* **Rollovers on "flat" road** were a simulator artefact: Mcity's road is several render meshes,
  and PhysX treats the open boundary of the next mesh as a ledge, so wheels were kicked up at
  every seam (front wheels lost contact, the robot was launched at ~2.5 m/s). The environment now
  collides with one seamless surface resampled from those meshes (0.3 m grid, road over terrain,
  0.45 m smoothing of cm-scale mesh offsets): `LocalEnvironment.drive_surface_grid`.
* **Grades.** `FX_DRIVE_MAX = M * AX_ACCEL_MAX` bounds acceleration only on flat ground (CARLA
  towns). Downhill the runner caps drive force so drive + gravity stays 0.2 m/s^2 inside 1B.
* **Where speed is measured.** The pose point is on the ground under the rear axle, 0.45 m below
  the CG; body pitch over a crest swings it fore-aft and read as +2 m/s^2. Velocities are now
  taken at the CG, as the CARLA actor reports them.
* NMPC mean solve time is 27 ms (max ~60 ms) in-process with Isaac rendering, above the 20 ms
  tick; the simulation is synchronous so this costs wall time, not control quality, but it would
  matter on hardware.

## What to record from each Isaac run

* `runs/isaac/*_metrics.json`: compare with `runs/offline/`.
* GPU FPS and NMPC solve time while rendering.
* Anything that looks physically off (wheel hop, drift on spin-in-place).
