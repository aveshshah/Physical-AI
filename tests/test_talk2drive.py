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


def test_ackermann_angles_roundtrip_and_inner_wheel():
    import math

    from talk2drive import AckermannParams

    p = AckermannParams(wheelbase=1.292, track=0.98)
    for d in (-0.5, -0.1, 0.0, 0.2, 0.55):
        left, right = p.ackermann_angles(d)
        assert abs(p.delta_from_wheels(left, right) - d) < 1e-9
        if d > 0:  # turning left: the left (inner) wheel steers more
            assert left > d > right
    assert abs(p.min_turn_radius - 1.292 / math.tan(math.radians(35))) < 1e-9


def test_car_controllers_track_lane_change():
    pytest.importorskip("casadi")
    from talk2drive import AckermannParams, KinematicBicycle, compute_metrics, make_trajectory, run_tracking
    from talk2drive.controllers import make_controller

    p = AckermannParams(wheelbase=1.292, track=0.98, wheel_radius=0.22, max_speed=2.0, max_accel=1.0)
    rmse = {}
    for name in ("nmpc", "kanayama"):
        tr = make_trajectory("slalom", speed=1.2, amplitude=0.8, wavelength=10.0, length=40.0)
        log = run_tracking(KinematicBicycle(p, initial_state=tr.start_pose()), make_controller(name, p), tr)
        rmse[name] = compute_metrics(log).rmse_pos_cm
    assert rmse["nmpc"] < 5.0 and rmse["kanayama"] < 15.0
    assert rmse["nmpc"] < rmse["kanayama"]


def test_route_path_circle_curvature_and_projection():
    import numpy as np

    from talk2drive.opendrive import RoutePath

    R = 20.0
    a = np.linspace(0, 2 * np.pi, 800, endpoint=False)
    path = RoutePath(np.c_[R * np.cos(a), R * np.sin(a)], closed=True)
    s = np.linspace(5, path.length - 5, 50)
    assert np.allclose(path.kappa_many(s), 1 / R, rtol=0.03)  # counter-clockwise: positive curvature
    s0, cte = path.project(R + 0.3, 0.0, s_hint=0.0)  # 0.3 m outside the circle = right of travel
    assert s0 < 1.0 and abs(cte + 0.3) < 0.02


def test_opendrive_line_and_arc_sampling():
    import xml.etree.ElementTree as ET

    import numpy as np

    from talk2drive.opendrive import _sample_geometry

    line = ET.fromstring('<geometry s="0" x="1" y="2" hdg="0.5" length="10"><line/></geometry>')
    p = _sample_geometry(line, 0.5)
    assert np.allclose(p[-1, :2], [1 + 10 * np.cos(0.5), 2 + 10 * np.sin(0.5)])
    arc = ET.fromstring('<geometry s="0" x="0" y="0" hdg="0" length="31.4159265" curvature="0.1"><arc curvature="0.1"/></geometry>')
    q = _sample_geometry(arc, 0.5)  # half circle of radius 10, turning left
    assert np.allclose(q[-1, :2], [0.0, 20.0], atol=1e-6) and abs(q[-1, 2] - np.pi) < 1e-6
