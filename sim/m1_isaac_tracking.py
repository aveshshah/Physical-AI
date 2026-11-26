"""M1 in Isaac Sim: the NMPC (and the Kanayama baseline) drive the Jetbot along a reference.

Run inside the Isaac Sim Python environment:

    python sim/m1_isaac_tracking.py                                # figure-8, NMPC
    python sim/m1_isaac_tracking.py --controller kanayama
    python sim/m1_isaac_tracking.py --traj slalom --env warehouse
    python sim/m1_isaac_tracking.py --headless --controller nmpc,kanayama   # both, back to back
    python sim/m1_isaac_tracking.py --draw --record                # viewport frames + GIF in runs/isaac/

Results (CSV, metrics JSON, comparison plot) go to runs/isaac/. Compare with sim/m1_offline.py:
the gap between the kinematic model and Isaac's physics (tyre contact, motor drive, mass) is
exactly what M2 is about.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--traj", default="figure8", help="figure8 | slalom | circle | roundabout (mcity)")
ap.add_argument("--controller", default="nmpc", help="nmpc | kanayama | nmpc,kanayama")
ap.add_argument("--env", default="grid", help="grid | warehouse | room | mcity")
ap.add_argument("--robot", default="jetbot", help="jetbot | magna | carla_trike (the last two are built from FBX files)")
ap.add_argument("--scale", type=float, default=None, help="trajectory size vs the Jetbot paths (default 1)")
ap.add_argument("--start", default=None, help="x,y,yaw_deg: start the path at this pose (scene coordinates)")
ap.add_argument("--headless", action="store_true")
ap.add_argument("--draw", action="store_true", help="draw reference and NMPC prediction in the viewport")
ap.add_argument("--record", action="store_true", help="save viewport frames and a GIF per controller")
ap.add_argument("--view", default="overview", help="overview (whole path) | follow (chase camera)")
ap.add_argument("--record-every", type=float, default=0.2, help="seconds between recorded frames")
ap.add_argument("--out", default="runs/isaac")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

from talk2drive import make_trajectory  # noqa: E402
from talk2drive.controllers import make_controller  # noqa: E402
from talk2drive.isaac.jetbot import IsaacJetbot  # noqa: E402
from talk2drive.runner import compute_metrics, plot_runs, run_tracking, save_run  # noqa: E402


# Paths that need a specific place or a car-sized shape.
PRESETS = {
    # one counter-clockwise lap of Mcity's roundabout lane (radius 20.5 m around the block-M island,
    # ~2 m/s), starting at its south side; scene coordinates of --env mcity
    "roundabout": dict(traj="circle", shape=dict(radius=20.5, period=64.4), start=(-93.72, -18.88, 0.0)),
}
# Path shapes per robot (car-like robots cannot follow the tight Jetbot paths). Keys are --traj names.
ROBOT_SHAPES = {
    "magna": {  # min turn radius 1.85 m, sidewalk speeds
        "slalom": dict(speed=1.2, amplitude=0.8, wavelength=10.0, length=40.0),
        "roundabout": dict(radius=20.5, period=85.9),  # ~1.5 m/s
    },
    "carla_trike": {"slalom": dict(speed=1.5, amplitude=1.5, wavelength=20.0, length=60.0)},  # min radius 4.3 m
}


def make_drawer(traj, bot):
    """Optional viewport overlay: reference path (green) and NMPC prediction (orange), on the terrain."""
    try:
        from isaacsim.util.debug_draw import _debug_draw
    except ImportError:
        print("[m1] debug_draw extension not available; --draw ignored")
        return None
    draw = _debug_draw.acquire_debug_draw_interface()
    pts = [traj.sample(t) for t in [i * 0.1 for i in range(int(traj.duration / 0.1) + 1)]]
    ref = [(p.x, p.y, bot.ground_height(p.x, p.y) + 0.06) for p in pts]  # draped over crowns and grades

    def on_step(t, s, ctrl):
        draw.clear_lines()
        draw.draw_lines(ref[:-1], ref[1:], [(0.1, 0.9, 0.2, 1)] * (len(ref) - 1), [2] * (len(ref) - 1))
        pred = getattr(ctrl, "last_prediction", None)
        if pred is not None and len(pred) > 1:
            P = [(float(x), float(y), bot.ground_height(float(x), float(y)) + 0.08) for x, y in pred[:, :2]]
            draw.draw_lines(P[:-1], P[1:], [(1.0, 0.5, 0.0, 1)] * (len(P) - 1), [4] * (len(P) - 1))

    return on_step


def set_overview(traj, bot) -> None:
    """Angled overhead view framing the whole reference path."""
    import numpy as np
    from isaacsim.core.utils.viewports import set_camera_view

    P = np.array([[traj.sample(t).x, traj.sample(t).y] for t in np.arange(0.0, traj.duration, 0.2)])
    (x0, y0), (x1, y1) = P.min(axis=0), P.max(axis=0)
    cx, cy, span = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, 2.0 * (y1 - y0), 0.5)
    z = bot.ground_height(cx, cy)
    set_camera_view(eye=[cx, cy - 0.95 * span, z + 0.95 * span], target=[cx, cy, z])


def make_follower(bot, scale: float = 1.0):
    """Chase camera: behind and above the robot, looking slightly ahead of it."""
    import math

    from isaacsim.core.utils.viewports import set_camera_view

    def on_step(t, s, ctrl):
        x, y, th = s[0], s[1], s[2]
        c, sn = math.cos(th), math.sin(th)
        z = float(bot.robot.get_world_pose()[0][2])
        k = scale ** 0.75  # pull back for bigger robots / paths
        set_camera_view(eye=[x - 2.2 * k * c, y - 2.2 * k * sn, z + 1.3 * k],
                        target=[x + 1.0 * k * c, y + 1.0 * k * sn, z])

    return on_step


def make_recorder(frames_dir: Path, every: float):
    """Capture the active viewport (including debug-draw lines) every ``every`` seconds."""
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("*.png"):
        old.unlink()
    vp = get_active_viewport()
    state = {"next": 0.0, "n": 0}

    def on_step(t, s, ctrl):
        if t + 1e-9 >= state["next"]:
            capture_viewport_to_file(vp, file_path=str(frames_dir / f"{state['n']:04d}.png"))
            state["n"] += 1
            state["next"] += every

    return on_step


def write_gif(frames_dir: Path, path: Path, fps: float, width: int = 720) -> Path | None:
    from PIL import Image

    files = sorted(frames_dir.glob("*.png"))
    if not files:
        return None
    imgs = []
    for f in files:
        im = Image.open(f).convert("RGB")
        imgs.append(im.resize((width, round(im.height * width / im.width))))
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=round(1000 / fps), loop=0, optimize=True)
    return path


def main() -> None:
    logs = []
    preset = PRESETS.get(args.traj, {})
    traj_name = preset.get("traj", args.traj)
    shape = {**preset.get("shape", {}), **ROBOT_SHAPES.get(args.robot, {}).get(args.traj, {})}
    scale = args.scale if args.scale is not None else 1.0
    start_at = tuple(float(v) for v in args.start.split(",")) if args.start else preset.get("start")

    def new_traj():
        tr = make_trajectory(traj_name, scale, **shape)
        if start_at is not None:
            x, y, yaw = start_at
            tr.anchored(x, y, math.radians(yaw))
        return tr

    start = new_traj().start_pose()
    bot = IsaacJetbot(env=args.env, start_pose=start, render=True, robot=args.robot)  # one World per process
    if bot.is_car and new_traj().min_turn_radius() < 1.1 * bot.params.min_turn_radius:
        print(f"[m1] WARNING: the path turns tighter ({new_traj().min_turn_radius():.1f} m) than this robot can "
              f"({bot.params.min_turn_radius:.1f} m); expect large errors")
    bot.reset_to(start)  # raycasts the terrain, so the camera and overlay below see the real ground
    set_overview(new_traj(), bot)
    for name in [c.strip() for c in args.controller.split(",")]:
        traj = new_traj()
        ctrl = make_controller(name, bot.params)
        bot.reset_to(start)  # same start pose for every controller, robot settled on the ground
        print(f"[m1] running {name} on {traj.name} ({traj.duration:.0f} s)...")
        frames_dir = Path(args.out) / f"frames_{args.traj}_{args.robot}_{name}"
        # things to do after every control step: draw the path, move the camera, save a frame
        hooks = []
        if args.draw:
            drawer = make_drawer(traj, bot)
            if drawer:
                hooks.append(drawer)
        if args.view == "follow":
            hooks.append(make_follower(bot, 3.0 if bot.is_car else scale))
        if args.record:
            hooks.append(make_recorder(frames_dir, args.record_every))

        def after_each_step(t, s, c):
            for hook in hooks:
                hook(t, s, c)

        log = run_tracking(bot, ctrl, traj, on_control_step=after_each_step if hooks else None)
        save_run(log, args.out)
        logs.append(log)
        print("[m1] " + compute_metrics(log).table_row())
        if args.record:
            for _ in range(10):  # let the last viewport captures finish writing
                app.update()
            gif = write_gif(frames_dir, Path(args.out) / f"{args.traj}_{args.env}_{args.robot}_{name}.gif",
                            fps=3 / args.record_every)  # 3x real time
            print(f"[m1] recording: {gif}")
    plot = plot_runs(logs, Path(args.out) / f"{args.traj}_{args.robot}_isaac_comparison.png",
                      title=f"{args.traj} x{scale:g} - {args.robot} - Isaac Sim ({args.env})")
    if plot:
        print(f"[m1] plot: {plot}")
    app.close()


if __name__ == "__main__":
    main()
