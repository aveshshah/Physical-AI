"""M1 in Isaac Sim: the NMPC (and the Kanayama baseline) drive the Jetbot along a reference.

Run inside the Isaac Sim Python environment:

    python sim/m1_isaac_tracking.py                                # figure-8, NMPC
    python sim/m1_isaac_tracking.py --controller kanayama
    python sim/m1_isaac_tracking.py --traj slalom --env warehouse
    python sim/m1_isaac_tracking.py --headless --controller nmpc,kanayama   # both, back to back

Results (CSV, metrics JSON, comparison plot) go to runs/isaac/. Compare with sim/m1_offline.py:
the gap between the kinematic model and Isaac's physics (tyre contact, motor drive, mass) is
exactly what M2 is about.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--traj", default="figure8", help="figure8 | slalom | circle")
ap.add_argument("--controller", default="nmpc", help="nmpc | kanayama | nmpc,kanayama")
ap.add_argument("--env", default="grid", help="grid | warehouse | room")
ap.add_argument("--headless", action="store_true")
ap.add_argument("--draw", action="store_true", help="draw reference and NMPC prediction in the viewport")
ap.add_argument("--out", default="runs/isaac")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

from talk2drive import TRAJECTORIES  # noqa: E402
from talk2drive.controllers import make_controller  # noqa: E402
from talk2drive.isaac.jetbot import IsaacJetbot  # noqa: E402
from talk2drive.runner import compute_metrics, plot_runs, run_tracking, save_run  # noqa: E402


def make_drawer(traj):
    """Optional viewport overlay: reference path (green) and NMPC prediction (orange)."""
    try:
        from isaacsim.util.debug_draw import _debug_draw
    except ImportError:
        print("[m1] debug_draw extension not available; --draw ignored")
        return None
    draw = _debug_draw.acquire_debug_draw_interface()
    pts = [traj.sample(t) for t in [i * 0.1 for i in range(int(traj.duration / 0.1) + 1)]]
    ref = [(p.x, p.y, 0.01) for p in pts]

    def on_step(t, s, ctrl):
        draw.clear_lines()
        draw.draw_lines(ref[:-1], ref[1:], [(0.1, 0.9, 0.2, 1)] * (len(ref) - 1), [2] * (len(ref) - 1))
        pred = getattr(ctrl, "last_prediction", None)
        if pred is not None and len(pred) > 1:
            P = [(float(x), float(y), 0.03) for x, y in pred[:, :2]]
            draw.draw_lines(P[:-1], P[1:], [(1.0, 0.5, 0.0, 1)] * (len(P) - 1), [4] * (len(P) - 1))

    return on_step


def main() -> None:
    logs = []
    start = TRAJECTORIES[args.traj]().start_pose()
    bot = IsaacJetbot(env=args.env, start_pose=start, render=True)  # Isaac allows one World per process
    for name in [c.strip() for c in args.controller.split(",")]:
        traj = TRAJECTORIES[args.traj]()
        ctrl = make_controller(name, bot.params)
        bot.reset_to(start)  # same start pose for every controller, robot settled on the ground
        print(f"[m1] running {name} on {traj.name} ({traj.duration:.0f} s)...")
        log = run_tracking(bot, ctrl, traj, on_control_step=make_drawer(traj) if args.draw else None)
        save_run(log, args.out)
        logs.append(log)
        print("[m1] " + compute_metrics(log).table_row())
    plot = plot_runs(logs, Path(args.out) / f"{args.traj}_isaac_comparison.png", title=f"{args.traj} - Isaac Sim ({args.env})")
    if plot:
        print(f"[m1] plot: {plot}")
    app.close()


if __name__ == "__main__":
    main()
