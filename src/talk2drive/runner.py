"""Closed-loop tracking runner, logging and metrics, identical for the offline model and Isaac Sim."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .references import Trajectory
from .robot import DiffDriveRobot


@dataclass
class RunLog:
    controller: str
    trajectory: str
    t: list[float] = field(default_factory=list)
    state: list[list[float]] = field(default_factory=list)  # measured [x, y, th, wl, wr]
    ref: list[list[float]] = field(default_factory=list)  # [x, y, th, v, w]
    cmd: list[list[float]] = field(default_factory=list)  # [wl_cmd, wr_cmd]
    solve_time: list[float] = field(default_factory=list)
    status: list[str] = field(default_factory=list)


@dataclass
class Metrics:
    controller: str
    trajectory: str
    duration_s: float
    rmse_pos_cm: float
    max_pos_err_cm: float
    final_pos_err_cm: float
    rmse_heading_deg: float
    mean_solve_ms: float
    max_solve_ms: float
    not_converged: int
    path_length_m: float

    def table_row(self) -> str:
        return (
            f"{self.controller:<9} {self.trajectory:<8} RMSE {self.rmse_pos_cm:6.2f} cm | "
            f"max {self.max_pos_err_cm:6.2f} cm | final {self.final_pos_err_cm:6.2f} cm | "
            f"heading RMSE {self.rmse_heading_deg:5.2f} deg | solve {self.mean_solve_ms:6.2f} ms "
            f"(max {self.max_solve_ms:6.2f}) | not converged {self.not_converged}"
        )


def run_tracking(
    robot: DiffDriveRobot,
    controller,
    traj: Trajectory,
    settle_time: float = 1.0,
    on_control_step=None,
) -> RunLog:
    """Run ``controller`` on ``robot`` along ``traj`` (already anchored at the robot's pose).

    The controller runs every ``controller.control_dt`` seconds (zero-order hold between),
    physics runs every ``robot.physics_dt`` seconds.
    """
    controller.reset()
    log = RunLog(controller=controller.name, trajectory=traj.name)
    steps_per_control = max(1, round(controller.control_dt / robot.physics_dt))
    total_steps = int(math.ceil((traj.duration + settle_time) / robot.physics_dt))

    t = 0.0
    for i in range(total_steps):
        if i % steps_per_control == 0:
            s = robot.get_state()
            wl, wr = controller.compute(t, s, traj)
            robot.set_wheel_speeds(wl, wr)
            r = traj.sample(t)
            log.t.append(t)
            log.state.append([float(v) for v in s])
            log.ref.append([r.x, r.y, r.theta, r.v, r.w])
            log.cmd.append([wl, wr])
            log.solve_time.append(controller.last_solve_time)
            log.status.append(controller.last_status)
            if on_control_step is not None:
                on_control_step(t, s, controller)
        robot.step()
        t += robot.physics_dt
    return log


def compute_metrics(log: RunLog) -> Metrics:
    S, R = np.array(log.state), np.array(log.ref)
    pos_err = np.hypot(S[:, 0] - R[:, 0], S[:, 1] - R[:, 1])
    head_err = np.degrees(np.angle(np.exp(1j * (S[:, 2] - R[:, 2]))))
    moving = R[:, 3] > 1e-3  # heading is undefined once the reference has stopped
    st = np.array(log.solve_time) * 1e3
    return Metrics(
        controller=log.controller,
        trajectory=log.trajectory,
        duration_s=float(log.t[-1]),
        rmse_pos_cm=float(np.sqrt(np.mean(pos_err**2)) * 100),
        max_pos_err_cm=float(pos_err.max() * 100),
        final_pos_err_cm=float(pos_err[-1] * 100),
        rmse_heading_deg=float(np.sqrt(np.mean(head_err[moving] ** 2))) if moving.any() else 0.0,
        mean_solve_ms=float(st.mean()),
        max_solve_ms=float(st.max()),
        not_converged=sum(1 for s in log.status if s != "ok"),
        path_length_m=float(np.sum(np.hypot(np.diff(S[:, 0]), np.diff(S[:, 1])))),
    )


def save_run(log: RunLog, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{log.trajectory}_{log.controller}"
    with open(out / f"{stem}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "x", "y", "theta", "wl", "wr", "x_ref", "y_ref", "theta_ref", "v_ref", "w_ref",
                    "wl_cmd", "wr_cmd", "solve_ms", "status"])
        for i in range(len(log.t)):
            w.writerow([log.t[i], *log.state[i], *log.ref[i], *log.cmd[i], log.solve_time[i] * 1e3, log.status[i]])
    (out / f"{stem}_metrics.json").write_text(json.dumps(asdict(compute_metrics(log)), indent=2))
    return out


def plot_runs(logs: list[RunLog], path: str | Path, title: str = "") -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    R = np.array(logs[0].ref)
    ax1.plot(R[:, 0], R[:, 1], "k--", lw=1.5, label="reference")
    for log in logs:
        S = np.array(log.state)
        Rl = np.array(log.ref)
        m = compute_metrics(log)
        ax1.plot(S[:, 0], S[:, 1], lw=1.6, label=f"{log.controller} (RMSE {m.rmse_pos_cm:.1f} cm)")
        ax2.plot(log.t, np.hypot(S[:, 0] - Rl[:, 0], S[:, 1] - Rl[:, 1]) * 100, lw=1.4, label=log.controller)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="best", fontsize=9)
    ax1.set_title("Path")
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("position error [cm]")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)
    ax2.set_title("Tracking error")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
