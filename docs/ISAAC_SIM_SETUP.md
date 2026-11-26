# Isaac Sim setup (Ubuntu + RTX 5060 Ti)

Isaac Sim gets its **own** Python environment, separate from the vision assistant's `.venv`.
Recent Isaac Sim releases on PyPI (6.x) require **Python 3.12**, which Ubuntu 24.04 already has.
If NVIDIA's install page for the version you pick says otherwise, follow it.

Needs roughly 25–30 GB of disk (packages + extension cache + shader cache) and 32 GB of RAM or more.

## 1. Create the environment

```bash
cd ~/Physical-AI
```
```bash
python3.12 -m venv ~/isaac-env
```
(Name the version explicitly: if a conda env is active, `python3` may be an older Python.)
```bash
source ~/isaac-env/bin/activate
```
```bash
pip install --upgrade pip
```

## 2. Install Isaac Sim

```bash
pip install "isaacsim[all,extscache]==6.2.0.0" --extra-index-url https://pypi.nvidia.com
```

This is a large download. Then install this project's simulation code into the same environment:

```bash
pip install -e ".[sim]"
```

## 3. First launch (accept the licence, warm the caches)

```bash
isaacsim
```

* It asks you to accept the NVIDIA Omniverse licence (EULA). Type `Yes`.
  For scripts, `export OMNI_KIT_ACCEPT_EULA=YES` accepts it non-interactively.
* **The first start is slow (5–15 minutes)** while it compiles shaders for your GPU. Later starts
  take well under a minute.
* An Isaac Sim window opens. Close it once it is up.

## 4. M0: Hello Jetbot

```bash
python sim/m0_hello_jetbot.py
```

A small Jetbot drives a scripted route on a flat ground plane. It prints its pose every second and
saves front-camera images to `runs/m0/frames/`. Then try a real scene:

```bash
python sim/m0_hello_jetbot.py --env warehouse
```

(Robot and scene assets are streamed from NVIDIA's servers, so the first load of each one takes a while.)

## 5. M1: your NMPC drives the Jetbot

```bash
python sim/m1_isaac_tracking.py --controller nmpc,kanayama --draw
```

The green line is the reference and the orange line is the NMPC's predicted path. Results go to
`runs/isaac/`: CSV logs, metrics JSON and `figure8_isaac_comparison.png`. Compare them with the
offline run:

```bash
python sim/m1_offline.py
```

## 5b. Mcity (University of Michigan test facility)

The [Mcity digital twin](https://github.com/mcity/mcity-digital-twin) (MIT licence) ships an Omniverse
USD version. Fetch only that folder (about 1.8 GB):

```bash
mkdir -p ~/assets && cd ~/assets
```
```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/mcity/mcity-digital-twin.git
```
```bash
cd mcity-digital-twin && git sparse-checkout set Omniverse && cd ~/Physical-AI
```

Then use `--env mcity` (set `MCITY_DIR` if you put it elsewhere):

```bash
python sim/m1_isaac_tracking.py --env mcity --traj slalom --draw --view follow
```

The robot starts in the middle of the downtown north-south street (the street runs along +x). The adapter
scales the centimetre stage to metres, adds colliders to roads, ground, sidewalks and the roundabout, and
adds a sun and sky (the stage has only street lamps). The road has a real crown and grade of a few cm.

## 5c. The Magna delivery robot (and other FBX vehicles)

`assets/robots/magna_robot.usd` is the Magna delivery robot. It has two steered front wheels (Ackermann,
±35°, position servos) 0.980 m apart, one motor-driven rear wheel on a fixed axle 1.292 m behind them, and
0.22 m wheels. The cargo bed is at 0.8 m and the roof at 1.42 m, with the flag mast reaching 2.0 m. It is
built from the FBX's wheel geometry plus a modelled shell:

```bash
python sim/build_robot_from_fbx.py ~/Documents/magna_robot_2wheeler.fbx assets/robots/magna_robot.usd --layout trike --shell magna --chassis-mass 278 --wheel-mass 6 --max-torque 400 --drive-damping 300 --max-steer 60 --steer-stiffness 20000 --steer-damping 600 --steer-torque 2000 --com 0.642,0,0.45 --chassis-inertia 90,140,470
```

Drive it with the car-like NMPC / Kanayama controllers (commands: speed and steering angle):

```bash
python sim/m1_isaac_tracking.py --robot magna --env mcity --traj roundabout --draw --view follow
```

`--traj slalom` gives a lane change down the downtown street. Masses, motor torque and servo gains are
assumptions (not in the FBX); change them on the build command line.

## 5d. The CARLA tadpole NMPC on an Mcity route

`sim/m2_mcity_tadpole_nmpc.py` drives the Magna robot round a 1.04 km lane-following loop built from
Mcity's OpenDRIVE map, using the NMPC stack from `~/Documents/carla_nmpc_project`. That project is
imported as-is and not edited. Its NMPC is compiled with acados, so install acados' Python template
once:

```bash
pip install -e ~/acados/interfaces/acados_template
```
```bash
python sim/m2_mcity_tadpole_nmpc.py
```

Results (metrics against that project's rules, a plot and a GIF with `--record`) go to
`runs/mcity_tadpole_nmpc/`. The robot model matches that NMPC (300 kg, CG 0.65 m behind the front
axle and 0.45 m high); only the wheelbase is set to the Magna's 1.292 m.

## 6. Calibrate the Jetbot model (after Isaac Sim upgrades)

```bash
python sim/calibrate_jetbot.py --headless
```

It measures the effective wheel radius and wheel base and writes `runs/calib/jetbot_isaac.json`.
If the suggested values differ from `ISAAC_JETBOT_PARAMS` in `src/talk2drive/isaac/jetbot.py`, update them.

## 7. Running the VLM at the same time (M3 and later)

Isaac Sim uses most of the 16 GB of VRAM. Run the assistant with the 4B model in its own terminal
and environment:

```bash
source ~/Physical-AI/.venv/bin/activate
```
```bash
vlm-assistant -c configs/sim_4b.yaml --stats
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `No matching distribution found for isaacsim-...` | Wrong Python version for that Isaac Sim release. Check `python --version` inside `~/isaac-env`. |
| `ModuleNotFoundError: isaacsim.core.api` (or another `isaacsim.*` module) | The Python API was renamed in your Isaac Sim version. The fix is usually a one-line import change in `src/talk2drive/isaac/jetbot.py`. |
| Black viewport / crash at start | Update the NVIDIA driver; close other GPU-heavy apps (games, the 8B assistant). |
| Very low FPS | Use `--env grid` (the default) or `--headless`; warehouse scenes are heavier. |
| `Could not find the Isaac Sim assets root` | Assets are streamed from the internet; check your connection or proxy. |
