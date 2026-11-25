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
python3 -m venv ~/isaac-env
```
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

## 6. Running the VLM at the same time (M3 and later)

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
| `ModuleNotFoundError: isaacsim.core.api` (or another `isaacsim.*` module) | The Python API was renamed in your Isaac Sim version. Send the full error to Claude; the fix is a one-line import change in `src/talk2drive/isaac/jetbot.py`. |
| Black viewport / crash at start | Update the NVIDIA driver; close other GPU-heavy apps (games, the 8B assistant). |
| Very low FPS | Use `--env grid` (the default) or `--headless`; warehouse scenes are heavier. |
| `Could not find the Isaac Sim assets root` | Assets are streamed from the internet; check your connection or proxy. |
