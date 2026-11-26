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


# ---------------------------------------------------------------- road-load disturbances
class _P:
    M, G, RHO_AIR, CD, A_FRONTAL, CRR, ROLLING_RESISTANCE_BLEND_SPEED = 300.0, 9.81, 1.225, 0.5, 1.0, 0.015, 0.3


def test_nominal_road_load_has_no_unmodelled_force():
    from talk2drive.disturbances import RoadLoadModel, parse_scenario

    m = RoadLoadModel(parse_scenario("nominal"), _P, 1000.0)
    for v in (0.5, 5.0, 15.0):
        f = m.forces(100.0, v, 0.0, 0.7)
        assert abs(f["dF_true"]) < 1e-9
        assert abs(f["f_aero_y"]) < 1e-9


def test_grade_wind_and_drag_signs():
    import math

    from talk2drive.disturbances import RoadLoadModel, parse_scenario

    up = RoadLoadModel(parse_scenario("nominal:grade_offset_pct=5"), _P, 1000.0)
    f = up.forces(0.0, 10.0, 0.0, 0.0)
    assert f["dF_true"] == pytest.approx(-300 * 9.81 * 0.05 / math.sqrt(1.0025))  # uphill slows the car
    head = RoadLoadModel(parse_scenario("nominal:wind_speed=10,wind_from_deg=0"), _P, 1000.0)
    assert head.forces(0.0, 10.0, 0.0, 0.0)["dF_true"] < -50  # driving along +x into wind from +x
    tail = RoadLoadModel(parse_scenario("nominal:wind_speed=10,wind_from_deg=180"), _P, 1000.0)
    assert tail.forces(0.0, 10.0, 0.0, 0.0)["dF_true"] > 20
    cross = RoadLoadModel(parse_scenario("nominal:wind_speed=10,wind_from_deg=90"), _P, 1000.0)
    assert cross.forces(0.0, 10.0, 0.0, 0.0)["f_aero_y"] < -50  # wind from +y pushes towards -y
    g = RoadLoadModel(parse_scenario("grade"), _P, 1000.0)
    assert max(abs(g.grade(s)) for s in range(0, 1000, 2)) == pytest.approx(0.06, rel=0.02)
    drag = RoadLoadModel(parse_scenario("drag"), _P, 1000.0)
    assert drag.forces(0.0, 10.0, 0.0, 0.0)["dF_true"] < 0


# ---------------------------------------------------------------- talk to drive
def test_parse_command():
    from talk2drive.commands import parse_command

    cases = {
        "turn left": ("turn", "left"), "take the next right please": ("turn", "right"), "go straight": ("turn", "straight"),
        "EMERGENCY STOP": ("emergency", None), "brake hard!": ("emergency", None), "pull over": ("pull_over", None),
        "park on the side": ("pull_over", None), "accelerate more": ("accelerate", None), "speed up": ("accelerate", None),
        "brake more": ("brake", None), "slow down": ("brake", None), "hold speed": ("hold", None),
        "set speed to 6": ("set_speed", 6.0), "drive at 36 km/h": ("set_speed", 10.0), "stop": ("stop", None),
        "go": ("go", None), "keep going": ("go", None), "what's ahead?": ("ahead", None), "status": ("status", None),
        "sing a song": ("unknown", None), "u-turn": ("u_turn", None), "turn around": ("u_turn", None),
        "change lane left": ("lane", "left"), "switch to the right lane": ("lane", "right"), "overtake": ("lane", None),
        "steer left": ("nudge", "left"), "move a bit to the right": ("nudge", "right"), "keep left": ("nudge", "left"),
        "back to the middle": ("centre", None), "take left": ("turn", "left"),
    }
    for text, (kind, value) in cases.items():
        c = parse_command(text)
        assert (c.kind, c.value) == (kind, value), text


def test_speed_commander_steps():
    from talk2drive.commands import SpeedCommander

    sc = SpeedCommander(v_max=15.0)
    sc.go(0.0)
    assert sc.rate == 1.0 and sc.goal == 8.0
    for _ in range(1000):
        sc.update(0.02)
    assert sc.v_set == 8.0 and sc.rate == 0.0
    sc.accelerate(8.0)
    sc.accelerate(8.0)
    sc.accelerate(8.0)
    assert sc.rate == 1.5              # three steps of 0.5 m/s^2
    for _ in range(5):
        sc.accelerate(8.0)
    assert sc.rate == sc.MAX_ACCEL     # capped
    sc.brake(9.0)
    assert sc.rate == -0.5 and sc.v_set == 9.0   # braking starts from the actual speed
    sc.brake(9.0)
    assert sc.rate == -1.0
    sc.hold(8.5)
    assert sc.rate == 0.0 and sc.v_set == 8.5


def test_road_router_turns():
    import os

    import pytest

    from talk2drive.mcity import MCITY_XODR, mcity_to_sim
    from talk2drive.opendrive import OpenDrive
    from talk2drive.road_router import RoadRouter

    if not os.path.exists(MCITY_XODR):
        pytest.skip("Mcity map not installed")
    router = RoadRouter(OpenDrive(MCITY_XODR), mcity_to_sim)
    route = router.build(router.keep_going([("104", 1)], 700))
    assert route.path.length > 600
    # from the start the first junction is 225 m away: nothing within 200 m
    assert router.plan_turn(route, 0.0, "left", 20.0) == (None, None, None)
    # from 300 m: left at the fork 35 m ahead, right at the junction 124 m ahead
    roads, junction, move = router.plan_turn(route, 300.0, "left", 20.0)
    assert move.kind == "left" and 30 < junction.s - 300 < 40
    roads, junction, move = router.plan_turn(route, 300.0, "right", 20.0)
    assert move.kind == "right" and 115 < junction.s - 300 < 130
    new = router.build(roads)
    assert new.roads[0] == route.roads[route.road_index_at(300.0)]


def test_router_skips_broken_links_and_turns_back():
    from talk2drive.mcity import MCITY_XODR, mcity_to_sim
    from talk2drive.opendrive import OpenDrive
    from talk2drive.road_router import RoadRouter

    router = RoadRouter(OpenDrive(MCITY_XODR), mcity_to_sim)
    assert ("107", -1) in router.od.successors("100", -1)       # linked in the map ...
    assert ("107", -1) not in router.next_roads(("100", -1))    # ... but its lanes start 45 m away
    route = router.build(router.keep_going([("104", 1)], 600))
    lanes = router.lanes_beside(route.roads[0], *route.path.xy[40], route.path.heading(10.0))
    assert sorted(rank for rank, _, _ in lanes) == [-1, 0, 1]   # two lanes my way, one the other way
    back = router.route_back(route, 290.0)                     # on a one-way road: round the block
    assert back is not None and back[0] == route.roads[route.road_index_at(290.0)]
