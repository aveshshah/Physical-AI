"""The CARLA tadpole NMPC (~/Documents/carla_nmpc_project) drives the Magna robot round an Mcity route.

    python sim/m2_mcity_tadpole_nmpc.py                      # one lap, GUI, chase camera
    python sim/m2_mcity_tadpole_nmpc.py --headless --record  # GIF + logs, no window
    python sim/m2_mcity_tadpole_nmpc.py --style comfort      # the project's default comfort plan

Defaults are the fastest configuration that passes every rule (tuned 25 Nov 2026): the project's
``sport`` style (its lap-time tuning), its speed-scheduled horizon, V_MAX 15 m/s.

Nothing in the CARLA project is copied or edited: its modules are imported from ``--nmpc-dir``
and called the way its route-following mode calls them in run_nmpc_carla.py, every 20 ms:

    stitch_start -> RouteSpeedPlanner.plan (v_cap, AY_CURVE) -> TadpoleNMPCController.step
    -> InnerLoopTracker.track_speed_2dof (drive / brake split)
    -> LongitudinalForceDisturbanceObserver (dF_unmod for the next solve)

Rules from that project: V_MAX = 15 m/s (compiled vx bound and cruise cap), the drive style's
planned acceleration / braking / jerk, curve speeds from AY_CURVE, the AY_MAX rollover envelope,
steering command limit 44 deg at up to 180 deg/s, 65/35 front/rear brake split. Only the vehicle
geometry is set to the Magna robot's (wheelbase 1.292 m, CG 0.65 m behind the front axle), because
the NMPC model must match the plant it drives; mass, inertia, CG height, tyres and road load are the
project's own values, and the Isaac robot is built to match them.

The run is scored against the rules (see ``score``) and written to ``--out``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--nmpc-dir", default=os.path.expanduser("~/Documents/carla_nmpc_project"))
ap.add_argument("--xodr", default=os.path.expanduser(
    "~/assets/mcity-digital-twin/Omniverse/Collected_McityMap_NSR_v4_1_6/mcity_map_v6_nvidia.xodr"))
ap.add_argument("--route-start", default="104,1", help="road id,direction (+1 along the road's s)")
ap.add_argument("--route-seed", type=int, default=1)
ap.add_argument("--v-max", type=float, default=15.0, help="NMPC_V_MAX [m/s]")
ap.add_argument("--style", default="sport", help="DRIVE_STYLE of the CARLA project: sport (lap time) | comfort")
ap.add_argument("--no-speed-horizon", action="store_true",
                help="fixed 0.8 s horizon instead of the project's opt-in speed-scheduled one (0.8 -> 1.6 s)")
ap.add_argument("--max-time", type=float, default=400.0, help="s of simulated time before giving up")
ap.add_argument("--start-s", type=float, default=0.0, help="start this far along the route (debugging a stretch)")
ap.add_argument("--contacts", action="store_true", help="print every contact that is not tyre-on-road (debugging)")
ap.add_argument("--no-grade-governor", action="store_true",
                help="do not cap drive force on downhill grades (Mcity has up to 9 %%; the CARLA towns are flat)")
ap.add_argument("--headless", action="store_true")
ap.add_argument("--record", action="store_true")
ap.add_argument("--record-every", type=float, default=0.5)
ap.add_argument("--out", default="runs/mcity_tadpole_nmpc")
args = ap.parse_args()

# The CARLA project's parameters read these at import.
os.environ["NMPC_V_MAX"] = f"{args.v_max:g}"
os.environ["DRIVE_STYLE"] = args.style
os.environ["NMPC_SPEED_HORIZON"] = "0" if args.no_speed_horizon else "1"
os.environ.setdefault("ACADOS_SOURCE_DIR", os.path.expanduser("~/acados"))

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

import numpy as np  # noqa: E402

from talk2drive.isaac.jetbot import LOCAL_ENVIRONMENTS, IsaacJetbot  # noqa: E402
from talk2drive.opendrive import OpenDrive, RoutePath  # noqa: E402

MAGNA_L, MAGNA_LF = 1.292, 0.65  # wheelbase, CG behind the front axle (tadpole_params: LF = 0.65/1.292 L)


def load_tadpole_stack(nmpc_dir: str, build_dir: Path):
    """Import the CARLA project's NMPC stack with the Magna geometry; compile its solver in build_dir."""
    sys.path.insert(0, nmpc_dir)
    import tadpole_params as P

    P.L, P.LF = MAGNA_L, MAGNA_LF
    P.LR = P.L - P.LF
    P.KUS = P.M * (P.LR * P.CR - P.LF * P.CF) / (P.L * P.CF * P.CR)
    P.POSE_POINT_TO_CG_X = P.LR
    P.CARLA_ACTOR_TO_REAR_X = P.LR
    build_dir.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(build_dir)  # acados writes c_generated_code/ and the solver json into the working directory
    try:
        from nmpc_controller import TadpoleNMPCController

        ctrl = TadpoleNMPCController()
    finally:
        os.chdir(cwd)
    from force_disturbance_observer import LongitudinalForceDisturbanceObserver
    from inner_loop import InnerLoopTracker
    from trajectory import RouteSpeedPlanner, stitch_start

    inner = InnerLoopTracker(m=P.M, Tw=P.TW_FRONT, kp_v=1000.0, ki_v=200.0, ki_v_2dof=30.0,
                             v_integral_max_2dof=20.0, Fx_drive_max=P.FX_DRIVE_MAX, Fx_brake_max=P.FX_BRAKE_MAX)
    return P, ctrl, RouteSpeedPlanner(), stitch_start, LongitudinalForceDisturbanceObserver(Ts=P.NMPC_TS), inner


def mcity_route(xodr: str, start: str, seed: int) -> tuple[RoutePath, float, list]:
    """One lap of a lane-following loop, in the Isaac scene frame, plus 40 m run-out past the start."""
    od = OpenDrive(xodr)
    rid, d = start.split(",")
    route = od.find_loop((rid, int(d)), min_length=500, max_length=1100, tries=3000, seed=seed)
    pts, widths = od.route_points(route)
    env = LOCAL_ENVIRONMENTS["mcity"]
    c, s = math.cos(math.radians(env.yaw_deg)), math.sin(math.radians(env.yaw_deg))
    q = pts - np.asarray(env.origin[:2])
    sim = np.c_[c * q[:, 0] - s * q[:, 1], s * q[:, 0] + c * q[:, 1]]
    lap = RoutePath(sim, closed=True)
    runout = int(40.0 / lap.ds)
    path = RoutePath(np.r_[lap.xy, lap.xy[1:runout]], smooth_m=0.0)
    return path, lap.length, [float(np.median(widths))], route


def make_chase_camera(bot):
    from isaacsim.core.utils.viewports import set_camera_view

    def update(x, y, psi):
        z = float(bot.robot.get_world_pose()[0][2])
        c, s = math.cos(psi), math.sin(psi)
        set_camera_view(eye=[x - 7.0 * c, y - 7.0 * s, z + 3.2], target=[x + 4.0 * c, y + 4.0 * s, z + 0.6])

    return update


def score(log: dict, P, lane_width: float, lap_length: float) -> dict:
    """The CARLA project's rules, measured on this run (audit.py definitions where they exist)."""
    t = np.asarray(log["t"])
    vx = np.asarray(log["vx"])
    dt = float(P.NMPC_TS)
    w = max(1, int(round(0.2 / dt)))  # 0.2 s averages, as the project judges realised acceleration

    def avg(a):
        return np.convolve(a, np.ones(w) / w, mode="valid")

    ax = avg(np.gradient(vx, dt))
    ay = avg(vx * np.asarray(log["r"]))
    cte = np.abs(log["cte"])
    epsi = np.degrees(np.abs(log["epsi"]))
    steer = np.asarray(log["delta_cmd"])
    flips = sum(1 for i in range(2, len(steer))
                if (steer[i - 1] - steer[i - 2]) * (steer[i] - steer[i - 1]) < 0 and abs(steer[i] - steer[i - 1]) > 0.02)
    cte_limit = 0.10 * lane_width
    moving = vx > 0.5
    crawl = (vx < 1.5) & (np.asarray(log["v_ref"]) > 2.5)  # slow while the plan wants to go (Rule 3E)
    done = log["s"][-1] >= lap_length - 0.5
    lap_t = float(t[np.argmax(np.asarray(log["s"]) >= lap_length - 0.5)]) if done else float("nan")
    return {
        "lap_completed": bool(done),
        "lap_length_m": lap_length,
        "lap_time_s": lap_t,
        "rule_1A_max_speed": {"max_vx": float(vx.max()), "v_max": float(P.V_MAX), "mean_speed_moving": float(vx[moving].mean()),
                              "pass": bool(vx.max() <= P.V_MAX + 0.2)},
        "rule_1B_long_accel": {"max_accel": float(ax.max()), "max_decel": float(-ax.min()),
                               "accel_limit": float(P.AX_ACCEL_MAX), "decel_limit": float(P.AX_DECEL_MAX),
                               "ticks_over": int(np.sum((ax > P.AX_ACCEL_MAX) | (ax < -P.AX_DECEL_MAX))),
                               "pass": bool(ax.max() <= P.AX_ACCEL_MAX + 0.05 and -ax.min() <= P.AX_DECEL_MAX + 0.05)},
        "rollover_envelope_ay": {"max_abs_ay": float(np.abs(ay).max()), "ay_max": float(P.AY_MAX),
                                 "ay_curve_plan": float(P.AY_CURVE), "pass": bool(np.abs(ay).max() <= P.AY_MAX)},
        "rule_3C_lane_keeping": {"lane_width": lane_width, "cte_limit": cte_limit, "cte_mean": float(cte.mean()),
                                 "cte_max": float(cte.max()), "cte_ticks_over": int(np.sum(cte > cte_limit)),
                                 "cte_time_over_s": float(np.sum(cte > cte_limit) * dt),
                                 "epsi_max_deg": float(epsi.max()), "epsi_ticks_over_15deg": int(np.sum(epsi > 15.0)),
                                 "pass": bool(np.sum(cte > cte_limit) == 0 and np.sum(epsi > 15.0) == 0)},
        "rule_3D_steering_reversals": {"count": flips, "per_minute": flips / max(t[-1] / 60.0, 1e-9)},
        "rule_3E_crawling_s": float(np.sum(crawl) * dt),
        "nmpc": {"mean_solve_ms": float(np.mean(log["solve_ms"])), "max_solve_ms": float(np.max(log["solve_ms"])),
                 "non_zero_status": int(np.sum(np.asarray(log["status"]) != 0))},
    }


def plot(log: dict, path: RoutePath, lap_length: float, P, out: Path, metrics: dict) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = np.asarray(log["s"])
    fig = plt.figure(figsize=(15, 9))
    a = fig.add_subplot(1, 2, 1)
    a.plot(path.xy[:, 0], path.xy[:, 1], color="0.8", lw=1, zorder=1, label="lane centre (OpenDRIVE)")
    sc = a.scatter(log["x"], log["y"], c=log["vx"], s=3, cmap="turbo", vmin=0, vmax=P.V_MAX, zorder=2)
    plt.colorbar(sc, ax=a, label="speed [m/s]", shrink=0.7)
    a.set_aspect("equal")
    a.set_title(f"Mcity lap {lap_length:.0f} m in {metrics['lap_time_s']:.1f} s")
    a.legend(loc="lower left")
    b = fig.add_subplot(4, 2, 2)
    b.plot(s, log["vx"], lw=1.2, label="speed")
    b.plot(s, log["v_ref"], lw=0.8, label="planned speed")
    k = np.abs(path.kappa_many(s))
    b.plot(s, np.minimum(P.V_MAX, np.sqrt(P.AY_CURVE / np.maximum(k, 1e-4))), "k:", lw=0.8, label="curve limit (AY_CURVE)")
    b.axhline(P.V_MAX, color="r", lw=0.8, ls="--", label="V_MAX")
    b.set_ylabel("m/s")
    b.legend(fontsize=7, ncol=4)
    c = fig.add_subplot(4, 2, 4, sharex=b)
    c.plot(s, np.asarray(log["cte"]) * 100, lw=1)
    lim = metrics["rule_3C_lane_keeping"]["cte_limit"] * 100
    c.axhline(lim, color="r", ls="--", lw=0.8)
    c.axhline(-lim, color="r", ls="--", lw=0.8)
    c.set_ylabel("CTE [cm]")
    d = fig.add_subplot(4, 2, 6, sharex=b)
    d.plot(s, np.degrees(log["delta_cmd"]), lw=1, label="steer cmd")
    d.plot(s, np.degrees(log["delta"]), lw=0.8, label="steer measured")
    d.set_ylabel("deg")
    d.legend(fontsize=7)
    e = fig.add_subplot(4, 2, 8, sharex=b)
    e.plot(s, np.gradient(log["vx"], P.NMPC_TS), lw=0.6, label="a_x")
    e.plot(s, np.asarray(log["vx"]) * np.asarray(log["r"]), lw=0.6, label="a_y")
    e.axhline(-P.AX_DECEL_MAX, color="r", ls="--", lw=0.8)
    e.axhline(P.AY_MAX, color="m", ls="--", lw=0.8)
    e.axhline(-P.AY_MAX, color="m", ls="--", lw=0.8)
    e.set_ylim(-5, 5)
    e.set_ylabel("m/s^2")
    e.set_xlabel("distance along route [m]")
    e.legend(fontsize=7)
    for ax_ in (b, c, d, e):
        ax_.grid(alpha=0.3)
    fig.suptitle("CARLA tadpole NMPC on the Magna robot in Mcity (Isaac Sim)")
    fig.tight_layout()
    p = out / "lap.png"
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


def main() -> None:
    out = (REPO / args.out) if not os.path.isabs(args.out) else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    P, ctrl, planner, stitch_start, observer, inner = load_tadpole_stack(args.nmpc_dir, out / "solver_build")
    path, lap_length, (lane_width,), route = mcity_route(args.xodr, args.route_start, args.route_seed)
    print(f"[m2] route: {len(route)} roads, lap {lap_length:.0f} m, lane width {lane_width:.2f} m; "
          f"V_MAX {P.V_MAX:g} m/s, style {P.DRIVE_STYLE}, AY_CURVE {P.AY_CURVE:.2f}, AY_MAX {P.AY_MAX:.2f} m/s^2")

    TS = float(P.NMPC_TS)
    physics_dt = TS / 2
    i0 = int(args.start_s / path.ds)
    x0, y0 = path.xy[i0]
    psi0 = path.heading(args.start_s)
    bot = IsaacJetbot(env="mcity", robot="magna", physics_dt=physics_dt, start_pose=(x0, y0, psi0), render=True)
    bot.enable_torque_mode()
    bot.reset_to((x0, y0, psi0))
    bot.params.max_steer = float(np.deg2rad(P.MAX_STEER_DEG))  # the servo may go to the physical limit
    contact_reader = None
    if args.contacts:
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysxSchema, Usd, UsdPhysics

        for prim in Usd.PrimRange(bot.world.stage.GetPrimAtPath("/World/Jetbot")):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.0)
        bot.reset_to((x0, y0, psi0))
        sim_if = get_physx_simulation_interface()

        def contact_reader(t):
            from pxr import PhysicsSchemaTools

            headers, data = sim_if.get_contact_report()
            for h in headers:
                a = str(PhysicsSchemaTools.intToSdfPath(h.collider0))
                b = str(PhysicsSchemaTools.intToSdfPath(h.collider1))
                pair = sorted([a, b])
                pts = data[h.contact_data_offset:h.contact_data_offset + h.num_contact_data]
                imp = sum(float(np.linalg.norm(np.asarray(c.impulse))) for c in pts)
                pos = np.asarray(pts[0].position) if len(pts) else np.zeros(3)
                nrm = np.asarray(pts[0].normal) if len(pts) else np.zeros(3)
                sep = min((float(c.separation) for c in pts), default=0.0)
                tyre_road = any("wheel/collision" in q for q in pair) and any("/World/Environment" in q for q in pair)
                if tyre_road and imp < 400.0 * TS and abs(nrm[2]) > 0.9:
                    continue  # ordinary tyre support: under ~400 N, normal near vertical
                print(f"[contact] t={t:6.2f} {pair[0].replace('/World/','')} <-> {pair[1].replace('/World/','')} "
                      f"n={len(pts)} impulse={imp:7.2f} N s sep={sep*100:+.1f} cm normal=({nrm[0]:+.2f},{nrm[1]:+.2f},{nrm[2]:+.2f}) "
                      f"at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
    camera = make_chase_camera(bot)
    ctrl.reset()
    observer.reset(vx=0.0)
    inner.reset()

    frames = out / "frames"
    if args.record:
        frames.mkdir(exist_ok=True)
        for f in frames.glob("*.png"):
            f.unlink()
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

        vp = get_active_viewport()

    keys = ("t", "x", "y", "psi", "vx", "vy", "r", "delta", "delta_cmd", "fx_drive", "fx_brake", "s", "cte", "epsi",
            "v_ref", "status", "solve_ms", "dF", "roll", "pitch", "z")
    log = {k: [] for k in keys}
    prev = None
    s_hint, followed, fx_prev, ax_f, n_frame = args.start_s, None, 0.0, 0.0, 0
    s_stop = lap_length
    wall0 = time.perf_counter()
    k = 0
    while k * TS < args.max_time and app.is_running():
        t = k * TS
        pos, quat = bot.robot.get_world_pose()
        X, Y = float(pos[0]), float(pos[1])
        st = bot.get_state()
        psi, delta = float(st[2]), float(st[4])
        # Velocities at the centre of mass (as the CARLA actor reports them). The pose point is on the
        # ground under the rear axle, 0.45 m below the CG, so body pitch/roll would swing it fore-aft
        # and sideways and read as phantom acceleration.
        qw_, qx_, qy_, qz_ = (float(v) for v in quat)
        Rb = np.array([[1 - 2 * (qy_ * qy_ + qz_ * qz_), 2 * (qx_ * qy_ - qz_ * qw_), 2 * (qx_ * qz_ + qy_ * qw_)],
                       [2 * (qx_ * qy_ + qz_ * qw_), 1 - 2 * (qx_ * qx_ + qz_ * qz_), 2 * (qy_ * qz_ - qx_ * qw_)],
                       [2 * (qx_ * qz_ - qy_ * qw_), 2 * (qy_ * qz_ + qx_ * qw_), 1 - 2 * (qx_ * qx_ + qy_ * qy_)]])
        cg = np.asarray(pos, dtype=float) + Rb @ np.array([P.LR, 0.0, P.H_CG])
        if prev is None:
            vx = vy = r = 0.0
        else:
            vwx, vwy = (cg[0] - prev[0]) / TS, (cg[1] - prev[1]) / TS
            r = ((psi - prev[2] + math.pi) % (2 * math.pi) - math.pi) / TS
            c, s_ = math.cos(psi), math.sin(psi)
            vx = c * vwx + s_ * vwy
            vy = -s_ * vwx + c * vwy
        ax_meas = 0.0 if prev is None else (vx - prev[3]) / TS
        ax_f += 0.3 * (ax_meas - ax_f)
        prev = (float(cg[0]), float(cg[1]), psi, vx)

        s0, cte = path.project(X, Y, s_hint)
        s_hint = s0
        epsi = (psi - path.heading(s0) + math.pi) % (2 * math.pi) - math.pi

        obs = observer.update(vx=vx, vy=vy, r=r, delta=delta, Fx_command_previous_interval=fx_prev,
                              ax_body_measured=ax_meas if k > 1 else None, mu_est=P.MU)
        dF = float(obs.estimate)

        # route end: stop on the start line after one lap, braking at the 1B command limit
        v_cap = min(P.TARGET_V, math.sqrt(max(0.0, 2.0 * P.AX_DECEL_CMD * (s_stop - s0 - 0.5))))
        t0, s_st, v_st, a_st, parent, _mode = stitch_start(followed, path, t, 0.0, s0, vx, ax_f, cte)
        traj = planner.plan(path, t0, s_st, v_st, a_st, v_cap=v_cap, ay_plan=float(P.AY_CURVE),
                            duration=float(P.NMPC_HORIZON_T_MAX) + 1.4, parent=parent,
                            decel_max=float(P.TRAJ_DECEL_MAX))
        t_solve = time.perf_counter()
        cmds = ctrl.step(X_global=X, Y_global=Y, psi_global=psi, vx=vx, vy=vy, r=r, delta=delta,
                         s0=s0, path=path, v_plan=float(P.V_MAX), dF_unmod_est=dF,
                         ay_max=P.AY_MAX, ay_plan=float(P.AY_CURVE), mu_est=P.MU,
                         Mz_min_avail=0.0, Mz_max_avail=0.0, Mz_applied_est=0.0, sim_time=t,
                         trajectory=traj, v_plan_target=None, v_plan_rate=0.0, mu_front_est=P.MU)
        solve_ms = (time.perf_counter() - t_solve) * 1e3
        followed = traj
        fx_drive, fx_brake = inner.track_speed_2dof(Fx_ff=cmds["Fx_net_nmpc"], v_cmd=cmds["v_cmd"], v_meas=vx, dt=TS)
        fx_drive = float(np.clip(fx_drive, 0.0, P.FX_DRIVE_MAX))
        if not args.no_grade_governor:
            # GRADE_GOVERNOR: FX_DRIVE_MAX = M * AX_ACCEL_MAX bounds the acceleration on flat ground only.
            # Downhill, gravity adds g*sin(pitch); cap the drive force so drive + gravity stays within
            # AX_ACCEL_MAX less the same 0.2 m/s^2 headroom the project keeps for braking (Rule 1B).
            qw_, qx_, qy_, qz_ = (float(v) for v in quat)
            a_grav = P.G * max(-1.0, min(1.0, 2 * (qw_ * qy_ - qz_ * qx_)))  # +ve nose down (downhill)
            fx_drive = min(fx_drive, max(0.0, P.M * (P.AX_ACCEL_MAX - 0.2 - a_grav)))
        fx_brake = float(np.clip(fx_brake, 0.0, P.FX_BRAKE_MAX))
        delta_cmd = float(np.clip(cmds["delta_cmd"], -np.deg2rad(P.MAX_STEER_DEG), np.deg2rad(P.MAX_STEER_DEG)))
        road_load = 0.5 * P.RHO_AIR * P.CD * P.A_FRONTAL * vx * vx + P.CRR * P.M * P.G * math.tanh(
            vx / P.ROLLING_RESISTANCE_BLEND_SPEED)
        bot.apply_forces(delta_cmd, fx_drive, fx_brake, front_brake_frac=P.FRONT_BRAKE_FRAC,
                         fx_resist=abs(road_load), dt=TS)
        fx_prev = fx_drive - fx_brake

        qw, qx, qy, qz = (float(v) for v in quat)
        roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
        for i, v in zip(keys, (t, X, Y, psi, vx, vy, r, delta, delta_cmd, fx_drive, fx_brake, s0, cte, epsi,
                               float(cmds["v_ref"]), int(cmds["status"]), solve_ms, dF, roll, pitch, float(pos[2]))):
            log[i].append(v)
        if k % 50 == 0:
            print(f"[m2] t={t:6.1f}s s={s0:7.1f}/{lap_length:.0f} m v={vx:5.2f} (ref {cmds['v_ref']:5.2f}) m/s "
                  f"cte={cte * 100:+6.1f} cm epsi={math.degrees(epsi):+5.1f} deg steer={math.degrees(delta_cmd):+6.1f} deg "
                  f"Fx={fx_drive - fx_brake:+6.0f} N solve={solve_ms:4.1f} ms status={cmds['status']}")

        if not args.headless:
            camera(X, Y, psi)
        if args.record and t + 1e-9 >= n_frame * args.record_every:
            capture_viewport_to_file(vp, file_path=str(frames / f"{n_frame:04d}.png"))
            n_frame += 1
        bot.step(render=False)
        bot.step(render=not args.headless or args.record)
        if contact_reader is not None:
            contact_reader(t)
        k += 1
        if s0 >= s_stop - 1.0 and vx < 0.05:
            break

    wall = time.perf_counter() - wall0
    metrics = score(log, P, lane_width, lap_length)
    metrics["sim_time_s"] = k * TS
    metrics["wall_time_s"] = wall
    metrics["route_roads"] = [f"{r}{'+' if d > 0 else '-'}" for r, d in route]
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(out / "log.npz", **{k_: np.asarray(v) for k_, v in log.items()}, path_xy=path.xy)
    fig = plot(log, path, lap_length, P, out, metrics)
    print("[m2] " + json.dumps({k_: v for k_, v in metrics.items() if k_ != "route_roads"}, indent=1))
    print(f"[m2] plot {fig}  ({k * TS:.0f} s simulated in {wall:.0f} s)")
    if args.record:
        for _ in range(10):
            app.update()
        from PIL import Image

        files = sorted(frames.glob("*.png"))
        if files:
            imgs = [Image.open(f).convert("RGB").resize((640, 360)) for f in files]
            gif = out / "lap.gif"
            imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=round(1000 * args.record_every / 4), loop=0)
            print(f"[m2] recording {gif} (4x real time)")
    app.close()


if __name__ == "__main__":
    main()
