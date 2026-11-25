"""M0 - Hello Isaac: a Jetbot drives a scripted route in a scene and saves camera frames.

Run inside the Isaac Sim Python environment (see docs/ISAAC_SIM_SETUP.md):

    python sim/m0_hello_jetbot.py                       # flat ground, with GUI
    python sim/m0_hello_jetbot.py --env warehouse       # warehouse scene
    python sim/m0_hello_jetbot.py --headless            # no window

Frames go to runs/m0/frames/, the driven path to runs/m0/path.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--env", default="grid", help="grid | warehouse | warehouse_full | room | mcity")
ap.add_argument("--robot", default="jetbot", help="jetbot | magna")
ap.add_argument("--headless", action="store_true")
ap.add_argument("--seconds", type=float, default=20.0)
ap.add_argument("--save-every", type=float, default=1.0, help="seconds between saved camera frames")
ap.add_argument("--out", default="runs/m0")
args = ap.parse_args()

# SimulationApp must be created before any other isaacsim / omni import.
from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from talk2drive.isaac.jetbot import IsaacJetbot  # noqa: E402


def route(t: float) -> tuple[float, float]:
    """Scripted body velocities (v [m/s], w [rad/s]): straight, left arc, straight, spin, right arc."""
    if t < 4:
        return 0.2, 0.0
    if t < 8:
        return 0.15, 0.6
    if t < 11:
        return 0.2, 0.0
    if t < 14:
        return 0.0, 1.2
    return 0.15, -0.5


def main() -> None:
    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    bot = IsaacJetbot(env=args.env, render=True, robot=args.robot)
    cam = None
    try:
        cam = bot.make_camera()
    except Exception as exc:  # noqa: BLE001
        print(f"[m0] camera not available ({exc}); continuing without frames")

    steps = int(args.seconds / bot.physics_dt)
    save_every = max(1, int(args.save_every / bot.physics_dt))
    rows, saved = [], 0
    for i in range(steps):
        if not app.is_running():
            break
        t = i * bot.physics_dt
        v, w = route(t)
        if bot.is_car:  # speed + steering; the spin-in-place segment becomes a stop
            bot.set_command(v, np.arctan(bot.params.wheelbase * w / v) if v > 1e-3 else 0.0)
        else:
            bot.set_wheel_speeds(*bot.params.body_to_wheels(v, w))
        bot.step()
        s = bot.get_state()
        rows.append([t, *s])
        if cam is not None and i % save_every == 0 and i > 10:
            rgba = cam.get_rgba()
            if rgba is not None and np.asarray(rgba).size:
                Image.fromarray(np.asarray(rgba)[:, :, :3].astype(np.uint8)).save(out / "frames" / f"{saved:04d}.png")
                saved += 1
        if i % 60 == 0:
            print(f"[m0] t={t:5.1f}s  x={s[0]:+.2f} y={s[1]:+.2f} yaw={np.degrees(s[2]):+6.1f} deg  "
                  f"wheels=({s[3]:+.1f}, {s[4]:+.1f}) rad/s")

    with open(out / "path.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "x", "y", "theta", "wl", "wr"])
        w.writerows(rows)
    print(f"[m0] done: {len(rows)} steps, {saved} frames -> {out.resolve()}")
    app.close()


if __name__ == "__main__":
    main()
