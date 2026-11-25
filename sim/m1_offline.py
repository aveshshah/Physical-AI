"""M1 without Isaac Sim: NMPC vs Kanayama on the kinematic robot model.

    python sim/m1_offline.py                      # figure-8, both controllers
    python sim/m1_offline.py --traj slalom --noise 0.005 --slip 0.1

Writes CSV logs, metrics JSON and a comparison plot to runs/offline/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from talk2drive import TRAJECTORIES, DiffDriveParams, KinematicDiffDrive  # noqa: E402
from talk2drive.controllers import make_controller  # noqa: E402
from talk2drive.runner import compute_metrics, plot_runs, run_tracking, save_run  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traj", choices=sorted(TRAJECTORIES), default="figure8")
    ap.add_argument("--controllers", default="nmpc,kanayama")
    ap.add_argument("--noise", type=float, default=0.002, help="pose measurement noise std [m]")
    ap.add_argument("--slip", type=float, default=0.0, help="fraction of wheel speed lost to slip (M4 preview)")
    ap.add_argument("--out", default="runs/offline")
    args = ap.parse_args()

    params = DiffDriveParams()
    logs = []
    for name in args.controllers.split(","):
        traj = TRAJECTORIES[args.traj]()
        robot = KinematicDiffDrive(params, initial_state=traj.start_pose(), pose_noise_std=args.noise, slip=args.slip)
        ctrl = make_controller(name.strip(), params)
        log = run_tracking(robot, ctrl, traj)
        save_run(log, args.out)
        logs.append(log)
        print(compute_metrics(log).table_row())
    plot = plot_runs(logs, Path(args.out) / f"{args.traj}_comparison.png",
                     title=f"{args.traj} - kinematic model (noise {args.noise} m, slip {args.slip:.0%})")
    if plot:
        print(f"plot: {plot}")


if __name__ == "__main__":
    main()
