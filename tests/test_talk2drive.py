from __future__ import annotations

import math

import numpy as np
import pytest

from talk2drive import DiffDriveParams, KinematicDiffDrive, circle, figure_eight, slalom
from talk2drive.controllers import KanayamaController, make_controller
from talk2drive.robot import wrap_angle, yaw_from_quat_wxyz
from talk2drive.runner import compute_metrics, run_tracking, save_run


def test_wheel_body_roundtrip_and_saturation():
    p = DiffDriveParams()
    wl, wr = p.body_to_wheels(0.2, 0.5)
    assert p.wheels_to_body(wl, wr) == pytest.approx((0.2, 0.5))
    sl, sr = p.saturate_wheels(30.0, 15.0)
    assert max(abs(sl), abs(sr)) == pytest.approx(p.max_wheel_speed)
    assert sl / sr == pytest.approx(2.0)  # curvature preserved


def test_yaw_and_wrap():
    yaw = 2.5
    q = (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))
    assert yaw_from_quat_wxyz(q) == pytest.approx(yaw)
    assert wrap_angle(3 * math.pi) == pytest.approx(-math.pi)


def test_kinematic_robot_drives_circle():
    p = DiffDriveParams(wheel_time_constant=1e-6, max_wheel_accel=1e6)
    bot = KinematicDiffDrive(p, physics_dt=0.01)
    v, w = 0.2, 0.5  # radius 0.4 m
    bot.set_wheel_speeds(*p.body_to_wheels(v, w))
    for _ in range(int(2 * math.pi / w / 0.01)):
        bot.step()
    assert np.hypot(*bot.s[:2]) < 0.02  # back near the start after one full lap


def test_trajectory_ramps_and_derivatives():
    tr = figure_eight(size=1.0, period=30.0)
    assert tr.sample(0.0).v == 0.0 and tr.sample(tr.duration).v == 0.0
    mid = tr.sample(tr.ramp + 5.0)
    assert 0.05 < mid.v < 0.4
    assert tr.max_speed() < DiffDriveParams().max_speed  # feasible for the Jetbot
    # heading matches direction of motion
    a, b = tr.sample(10.0), tr.sample(10.05)
    assert math.atan2(b.y - a.y, b.x - a.x) == pytest.approx(a.theta, abs=0.05)


def test_anchored_trajectory_starts_at_pose():
    tr = slalom().anchored(1.0, -2.0, 0.7)
    s = tr.sample(0.0)
    assert (s.x, s.y, s.theta) == pytest.approx((1.0, -2.0, 0.7), abs=1e-6)


@pytest.mark.parametrize("make_traj", [circle, slalom])
def test_kanayama_tracks(make_traj):
    traj = make_traj()
    bot = KinematicDiffDrive(initial_state=traj.start_pose())
    m = compute_metrics(run_tracking(bot, KanayamaController(), traj))
    assert m.rmse_pos_cm < 5.0


def test_nmpc_tracks_better_than_baseline_on_slalom(tmp_path):
    pytest.importorskip("casadi")
    res = {}
    for name in ("nmpc", "kanayama"):
        traj = slalom()
        bot = KinematicDiffDrive(initial_state=traj.start_pose())
        log = run_tracking(bot, make_controller(name), traj)
        res[name] = compute_metrics(log)
        save_run(log, tmp_path)
    assert res["nmpc"].rmse_pos_cm < 3.0
    assert res["nmpc"].rmse_pos_cm < res["kanayama"].rmse_pos_cm
    assert res["nmpc"].mean_solve_ms < 50.0  # real-time at 20 Hz with margin
    assert (tmp_path / "slalom_nmpc.csv").exists() and (tmp_path / "slalom_nmpc_metrics.json").exists()


def test_nmpc_respects_wheel_limits():
    pytest.importorskip("casadi")
    p = DiffDriveParams(max_wheel_speed=6.0)  # too slow for the reference -> must saturate
    traj = figure_eight(period=15.0)
    bot = KinematicDiffDrive(p, initial_state=traj.start_pose())
    log = run_tracking(bot, make_controller("nmpc", p), traj)
    cmds = np.abs(np.array(log.cmd))
    assert cmds.max() <= 6.0 + 1e-6
