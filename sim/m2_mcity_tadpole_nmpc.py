"""The CARLA tadpole NMPC (~/Documents/carla_nmpc_project) drives the Magna robot round an Mcity route.

    python sim/m2_mcity_tadpole_nmpc.py                      # one lap, GUI, chase camera
    python sim/m2_mcity_tadpole_nmpc.py --headless --record  # GIF + logs, no window
    python sim/m2_mcity_tadpole_nmpc.py --style comfort      # the project's default comfort plan
    python sim/m2_mcity_tadpole_nmpc.py --disturbance wind   # road-load uncertainty (see talk2drive.disturbances)
    python sim/m2_mcity_tadpole_nmpc.py --disturbance "nominal:mu=0.5,grade_pct=3"

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
ap.add_argument("--ay-curve", type=float, default=None,
                help="lateral acceleration the speed plan uses on bends [m/s^2] (sport: 0.55*AY_MAX = 1.65); "
                     "capped at the AY_MAX rollover envelope")
ap.add_argument("--no-speed-horizon", action="store_true",
                help="fixed 0.8 s horizon instead of the project's opt-in speed-scheduled one (0.8 -> 1.6 s)")
ap.add_argument("--max-time", type=float, default=400.0, help="s of simulated time before giving up")
ap.add_argument("--start-s", type=float, default=0.0, help="start this far along the route (debugging a stretch)")
ap.add_argument("--contacts", action="store_true", help="print every contact that is not tyre-on-road (debugging)")
ap.add_argument("--no-grade-governor", action="store_true",
                help="do not cap drive force on downhill grades (Mcity has up to 9 %%; the CARLA towns are flat)")
ap.add_argument("--disturbance", default="nominal",
                help="road-load scenario the NMPC is not told about: nominal | wet | low_mu | ice_patches | grade | "
                     "wind | storm | drag | combined, optionally with overrides, e.g. wind:wind_speed=12")
ap.add_argument("--robot", default="magna", help="magna (rigid chassis) | magna_susp (sprung struts, front anti-roll bar)")
ap.add_argument("--baseline", default=None, help="log.npz of an undisturbed lap, to compare speed at the same s")
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

from talk2drive.disturbances import BodyForceApplier, RoadLoadModel, apply_road_friction, parse_scenario  # noqa: E402
from talk2drive.isaac.jetbot import IsaacJetbot  # noqa: E402
from talk2drive.tadpole import TadpoleDriver, load_tadpole_stack  # noqa: E402
from talk2drive.mcity import mcity_route  # noqa: E402
from talk2drive.opendrive import RoutePath  # noqa: E402

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


def _baseline_speed(baseline: str | None, s: np.ndarray):
    if not baseline or not os.path.exists(baseline):
        return None
    b = np.load(baseline)
    bs, bv = np.asarray(b["s"]), np.asarray(b["vx"])
    order = np.argsort(bs)
    return np.interp(s, bs[order], bv[order])


def speed_tracking(log: dict, P, baseline: str | None, metrics: dict) -> dict:
    """Does the NMPC still hold the target speed? Against its own reference and against an undisturbed lap."""
    vx, vref, s = (np.asarray(log[k]) for k in ("vx", "v_ref", "s"))
    moving = vref > 0.5
    e = (vx - vref)[moving]
    mode = np.asarray(log["plan_mode"])
    dF, dF_true = np.asarray(log["dF"]), np.asarray(log["dF_true"])
    wheel = np.asarray(log["wheel_rear"]) * 0.22  # rear-wheel surface speed (r = 0.22 m)
    fast = vx > 2.0
    slip = (wheel[fast] - vx[fast]) / vx[fast]
    out = {
        "vs_nmpc_reference": {"rmse": float(np.sqrt(np.mean(e ** 2))), "mean": float(e.mean()),
                              "p95_abs": float(np.percentile(np.abs(e), 95)), "max_below": float(-e.min()),
                              "max_above": float(e.max()), "time_off_by_more_than_0.5_s": float(np.sum(np.abs(e) > 0.5) * P.NMPC_TS)},
        "plan_restarts": {"description": "plans restarted from the measured state because the robot was not on the previous plan (stitch_start)",
                          "count": int(np.sum((mode[1:] != 0) & (mode[:-1] == 0))),
                          "time_off_plan_s": float(np.sum(mode[1:] != 0) * P.NMPC_TS)},
        "observer": {"description": "LongitudinalForceDisturbanceObserver estimate vs the injected unmodelled force [N]",
                     "rmse_N": float(np.sqrt(np.mean((dF - dF_true)[fast] ** 2))) if fast.any() else float("nan"),
                     "true_mean_N": float(dF_true[fast].mean()) if fast.any() else float("nan"),
                     "true_range_N": [float(dF_true[fast].min()), float(dF_true[fast].max())] if fast.any() else [],
                     "correlation": float(np.corrcoef(dF[fast], dF_true[fast])[0, 1]) if fast.sum() > 2 and np.std(dF_true[fast]) > 1 else None},
        "rear_wheel_slip": {"max_drive": float(slip.max()) if fast.any() else 0.0,
                            "max_brake": float(-slip.min()) if fast.any() else 0.0},
    }
    vb = _baseline_speed(baseline, s)
    if vb is not None:
        d = (vx - vb)[moving]
        b = np.load(baseline)
        bt, bs = np.asarray(b["t"]), np.asarray(b["s"])
        lap = metrics["lap_length_m"]
        b_lap = float(bt[np.argmax(bs >= lap - 0.5)]) if bs[-1] >= lap - 0.5 else float("nan")
        out["vs_undisturbed_lap"] = {"rmse": float(np.sqrt(np.mean(d ** 2))), "mean": float(d.mean()),
                                     "max_below": float(-d.min()), "max_above": float(d.max()),
                                     "lap_time_baseline_s": b_lap, "lap_time_delta_s": metrics["lap_time_s"] - b_lap}
    return out


def plot_disturbance(log: dict, P, out: Path, metrics: dict, baseline: str | None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = np.asarray(log["s"])
    fig, ax = plt.subplots(5, 1, figsize=(13, 12), sharex=True)
    vb = _baseline_speed(baseline, s)
    ax[0].plot(s, log["v_ref"], lw=0.9, label="NMPC reference")
    if vb is not None:
        ax[0].plot(s, vb, lw=0.9, color="0.55", ls="--", label="undisturbed lap")
    ax[0].plot(s, log["vx"], lw=1.2, label="speed")
    ax[0].set_ylabel("m/s")
    ax[0].legend(fontsize=8, ncol=3)
    ax[1].plot(s, np.asarray(log["vx"]) - np.asarray(log["v_ref"]), lw=1, label="speed - reference")
    if vb is not None:
        ax[1].plot(s, np.asarray(log["vx"]) - vb, lw=0.8, color="0.5", label="speed - undisturbed")
    off = np.asarray(log["plan_mode"]) != 0
    if off.any():
        ax[1].scatter(s[off], np.zeros(off.sum()), s=4, color="r", zorder=3, label="plan restarted")
    ax[1].set_ylabel("m/s")
    ax[1].legend(fontsize=8, ncol=3)
    ax[2].plot(s, log["dF_true"], lw=1, color="k", label="injected unmodelled force (incl. Mcity grade)")
    ax[2].plot(s, log["dF"], lw=1, label="observer estimate dF")
    ax[2].set_ylabel("N")
    ax[2].legend(fontsize=8)
    ax[3].plot(s, 100 * np.asarray(log["grade_extra"]), lw=1, label="extra grade [%]")
    ax[3].plot(s, -100 * np.tan(np.asarray(log["pitch"])), lw=0.8, label="Mcity grade [%]")
    ax[3].plot(s, np.hypot(log["wind_x"], log["wind_y"]), lw=0.8, label="wind speed [m/s]")
    ax[3].plot(s, 10 * np.asarray(log["mu_road"]), lw=0.8, label="road mu x10")
    ax[3].legend(fontsize=8, ncol=4)
    ax[4].plot(s, np.degrees(log["delta_cmd"]), lw=0.8, label="steer cmd [deg]")
    ax[4].plot(s, 100 * np.asarray(log["cte"]), lw=0.8, label="CTE [cm]")
    ax[4].plot(s, np.degrees(log["roll"]), lw=0.8, label="roll [deg]")
    ax[4].legend(fontsize=8, ncol=3)
    ax[4].set_xlabel("distance along route [m]")
    for a in ax:
        a.grid(alpha=0.3)
    st = metrics["speed_tracking"]["vs_nmpc_reference"]
    fig.suptitle(f"{metrics['disturbance']['name']}: {metrics['disturbance']['description']}\n"
                 f"lap {metrics['lap_time_s']:.1f} s, speed error RMSE {st['rmse']:.2f} m/s (p95 {st['p95_abs']:.2f})")
    fig.tight_layout()
    p = out / "disturbance.png"
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return p


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
    stack = load_tadpole_stack(args.nmpc_dir, out / "solver_build", ay_curve=args.ay_curve)
    P = stack.P
    path, lap_length, (lane_width,), route = mcity_route(args.xodr, args.route_start, args.route_seed)
    print(f"[m2] route: {len(route)} roads, lap {lap_length:.0f} m, lane width {lane_width:.2f} m; "
          f"V_MAX {P.V_MAX:g} m/s, style {P.DRIVE_STYLE}, AY_CURVE {P.AY_CURVE:.2f}, AY_MAX {P.AY_MAX:.2f} m/s^2")

    TS = float(P.NMPC_TS)
    physics_dt = TS / 2
    i0 = int(args.start_s / path.ds)
    x0, y0 = path.xy[i0]
    psi0 = path.heading(args.start_s)
    bot = IsaacJetbot(env="mcity", robot=args.robot, physics_dt=physics_dt, start_pose=(x0, y0, psi0), render=True)
    bot.enable_torque_mode()
    scenario = parse_scenario(args.disturbance)
    patches = apply_road_friction(bot.world.stage, scenario, path)
    loads = RoadLoadModel(scenario, P, lap_length, dt=TS)
    body_forces = BodyForceApplier("/World/Jetbot/chassis")
    print(f"[m2] road loads: {scenario.name}: {scenario.describe()}"
          + (f"; patches {[(round(a), round(b), m) for a, b, m in patches]}" if patches else ""))
    bot.reset_to((x0, y0, psi0))  # the world reset re-parses physics, picking up the road materials
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

    frames = out / "frames"
    if args.record:
        frames.mkdir(exist_ok=True)
        for f in frames.glob("*.png"):
            f.unlink()
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

        vp = get_active_viewport()

    keys = ("t", "x", "y", "psi", "vx", "vy", "r", "delta", "delta_cmd", "fx_drive", "fx_brake", "s", "cte", "epsi",
            "v_ref", "status", "solve_ms", "dF", "roll", "pitch", "z",
            "dF_true", "grade_extra", "wind_x", "wind_y", "f_aero_y", "plan_mode", "wheel_rear", "mu_road",
            "susp_fl", "susp_fr", "susp_r")
    patch_s = [(a, b, m) for a, b, m in patches]
    log = {k: [] for k in keys}
    driver = TadpoleDriver(stack, bot, loads, body_forces, grade_governor=not args.no_grade_governor)
    driver.set_path(path, s_hint=args.start_s)
    n_frame = 0
    s_stop = lap_length
    wall0 = time.perf_counter()
    k = 0
    while k * TS < args.max_time and app.is_running():
        t = k * TS
        row = driver.tick(t, s_stop)
        s0, vx, cte, roll = row["s"], row["vx"], row["cte"], row["roll"]
        row["mu_road"] = next((m for a, b, m in patch_s if a <= s0 <= b), scenario.mu)
        for key in keys:
            log[key].append(row[key])
        if k % 50 == 0:
            print(f"[m2] t={t:6.1f}s s={s0:7.1f}/{lap_length:.0f} m v={vx:5.2f} (ref {row['v_ref']:5.2f}) m/s "
                  f"cte={cte * 100:+6.1f} cm epsi={math.degrees(row['epsi']):+5.1f} deg "
                  f"steer={math.degrees(row['delta_cmd']):+6.1f} deg Fx={row['fx_drive'] - row['fx_brake']:+6.0f} N "
                  f"solve={row['solve_ms']:4.1f} ms status={row['status']}")

        if not args.headless:
            camera(row["x"], row["y"], row["psi"])
        if args.record and t + 1e-9 >= n_frame * args.record_every:
            capture_viewport_to_file(vp, file_path=str(frames / f"{n_frame:04d}.png"))
            n_frame += 1
        driver.step(render=not args.headless or args.record)
        if contact_reader is not None:
            contact_reader(t)
        k += 1
        if s0 >= s_stop - 1.0 and vx < 0.05:
            break
        if abs(roll) > math.radians(45) or abs(cte) > 3.0:
            print(f"[m2] FAILED at t={t:.1f} s, s={s0:.0f} m: roll {math.degrees(roll):.0f} deg, cte {cte:.2f} m")
            break

    wall = time.perf_counter() - wall0
    metrics = score(log, P, lane_width, lap_length)
    metrics["disturbance"] = {"name": scenario.name, "description": scenario.describe(), "patches": patches}
    metrics["speed_tracking"] = speed_tracking(log, P, args.baseline, metrics)
    metrics["sim_time_s"] = k * TS
    metrics["wall_time_s"] = wall
    metrics["route_roads"] = [f"{r}{'+' if d > 0 else '-'}" for r, d in route]
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(out / "log.npz", **{k_: np.asarray(v) for k_, v in log.items()}, path_xy=path.xy)
    fig = plot(log, path, lap_length, P, out, metrics)
    plot_disturbance(log, P, out, metrics, args.baseline)
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
