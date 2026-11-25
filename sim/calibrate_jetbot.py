"""Calibrate the Isaac Sim Jetbot against the talk2drive diff-drive model.

    python sim/calibrate_jetbot.py --headless

On a flat ground plane it measures:

* wheel tracking  encoder wheel speed / commanded wheel speed (straight runs)
* wheel radius    path speed / encoder wheel speed (straight runs)
* wheel base      r * (wr - wl) / yaw rate (spin-in-place runs)
* wheel lag       63 % rise time of a wheel-speed step
* velocity bias   Isaac's reported joint velocity minus the encoder speed

It also prints the wheel-joint drive settings and wheel geometry from the USD, and checks the
fitted model on an arc that was not used for fitting. Results go to runs/calib/jetbot_isaac.json;
copy ``suggested`` into ``ISAAC_JETBOT_PARAMS`` in src/talk2drive/isaac/jetbot.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ap = argparse.ArgumentParser()
ap.add_argument("--robot", default="jetbot", help="jetbot | magna")
ap.add_argument("--headless", action="store_true")
ap.add_argument("--hold", type=float, default=4.0, help="seconds per constant-speed test")
ap.add_argument("--out", default="runs/calib")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": args.headless})

import numpy as np  # noqa: E402

from talk2drive.isaac.jetbot import IsaacJetbot  # noqa: E402
from talk2drive.robot import DiffDriveParams  # noqa: E402


def usd_report(stage) -> dict:
    """Wheel-joint drive gains and wheel geometry as authored in the Jetbot asset."""
    from pxr import Usd, UsdGeom, UsdPhysics

    info: dict = {"joints": {}, "wheels": {}}
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Jetbot")):
        name = prim.GetName()
        if prim.IsA(UsdPhysics.RevoluteJoint) and "wheel" in name:
            d = UsdPhysics.DriveAPI.Get(prim, "angular")
            info["joints"][name] = {
                "localPos0": [round(float(v), 5) for v in UsdPhysics.RevoluteJoint(prim).GetLocalPos0Attr().Get()],
                "stiffness": d.GetStiffnessAttr().Get() if d else None,
                "damping": d.GetDampingAttr().Get() if d else None,
                "maxForce": str(d.GetMaxForceAttr().Get()) if d else None,
            }
        elif name in ("left_wheel", "right_wheel") and prim.GetParent().GetName() == "Jetbot":
            box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            info["wheels"][name] = {
                "diameter": round(float(box.GetSize()[0]), 5),
                "width": round(float(box.GetSize()[1]), 5),
                "center_y": round(float(box.GetMidpoint()[1]), 5),
            }
    return info


def hold(bot: IsaacJetbot, wl: float, wr: float, seconds: float) -> np.ndarray:
    """From rest, command constant wheel speeds; log [t, x, y, yaw, wl, wr, wl_rep, wr_rep] per step."""
    bot.reset_to((0.0, 0.0, 0.0))
    idx = bot.robot.wheel_dof_indices
    rows = []
    for i in range(int(seconds / bot.physics_dt)):
        bot.set_wheel_speeds(wl, wr)
        bot.step()
        reported = np.asarray(bot.robot.get_joint_velocities(), dtype=float)[idx]
        rows.append([(i + 1) * bot.physics_dt, *bot.get_state(), *reported])
    d = np.array(rows)
    d[:, 3] = np.unwrap(d[:, 3])
    return d


def steady(d: np.ndarray) -> dict:
    """Rates over the second half of a run (wheels settled). Speed is along the path, not the chord."""
    s = d[len(d) // 2:]
    t = s[:, 0]
    path = np.sum(np.hypot(np.diff(s[:, 1]), np.diff(s[:, 2])))
    return {
        "v": float(path / (t[-1] - t[0])),
        "yaw_rate": float(np.polyfit(t, s[:, 3], 1)[0]),
        "wl": float(s[:, 4].mean()),
        "wr": float(s[:, 5].mean()),
        "reported_bias": [float(v) for v in (s[:, 6:8] - s[:, 4:6]).mean(axis=0)],
    }


def main() -> None:
    bot = IsaacJetbot(env="grid", render=not args.headless, robot=args.robot)
    nominal = bot.params
    report = {"isaac_physics_dt": bot.physics_dt, "usd": usd_report(bot.world.stage), "straight": [], "spin": []}
    print("[calib] USD:", json.dumps(report["usd"]))

    # 1. straight runs -> wheel tracking and effective wheel radius
    wmax = bot.params.max_wheel_speed  # test speeds are fractions of the robot's limit (12 rad/s on the Jetbot)
    for cmd in (wmax * f for f in (1 / 6, 1 / 3, 1 / 2, 2 / 3, 5 / 6, 1.0)):
        r = steady(hold(bot, cmd, cmd, args.hold))
        w = 0.5 * (r["wl"] + r["wr"])
        row = {"cmd": cmd, **r, "tracking": w / cmd, "radius": r["v"] / w}
        report["straight"].append(row)
        print(f"[calib] straight {cmd:5.1f}  encoder {r['wl']:6.3f} {r['wr']:6.3f}  tracking {row['tracking']:.4f}  "
              f"v {r['v']:.4f} m/s  radius {row['radius'] * 1000:.2f} mm  "
              f"reported-velocity bias {r['reported_bias'][0]:+.3f} {r['reported_bias'][1]:+.3f}")
    radius = float(np.median([r["radius"] for r in report["straight"]]))

    # 2. spin in place -> effective wheel base
    for cmd in (wmax * f for f in (1 / 6, 1 / 3, 2 / 3)):
        d = hold(bot, -cmd, cmd, args.hold)
        r = steady(d)
        base = radius * (r["wr"] - r["wl"]) / r["yaw_rate"]
        drift = float(np.hypot(d[-1, 1] - d[0, 1], d[-1, 2] - d[0, 2]))
        report["spin"].append({"cmd": cmd, **r, "wheel_base": base, "drift_m": drift})
        print(f"[calib] spin {cmd:4.1f}  encoder {r['wl']:+6.3f} {r['wr']:+6.3f}  yaw rate {r['yaw_rate']:.4f} rad/s  "
              f"wheel base {base * 1000:.2f} mm  centre drift {drift * 100:.2f} cm")
    base = float(np.median([r["wheel_base"] for r in report["spin"]]))

    # 3. wheel-speed step response
    step = 2 / 3 * wmax
    d = hold(bot, step, step, 1.0)
    w = 0.5 * (d[:, 4] + d[:, 5])
    final = w[len(w) // 2:].mean()
    t63 = float(d[int(np.argmax(w >= 0.632 * final)), 0])
    print(f"[calib] step 0 -> {step:.1f} rad/s: settles at {final:.3f}, 63 % after {t63 * 1000:.0f} ms "
          f"(one physics step = {bot.physics_dt * 1000:.1f} ms)")

    # 4. check on an arc not used for fitting
    cal = DiffDriveParams(wheel_radius=radius, wheel_base=base)
    r = steady(hold(bot, wmax / 3, 2 * wmax / 3, args.hold))
    meas = (r["v"], r["yaw_rate"])
    pred = {"nominal": nominal.wheels_to_body(r["wl"], r["wr"]), "calibrated": cal.wheels_to_body(r["wl"], r["wr"])}
    print(f"[calib] arc ({wmax / 3:.1f}, {2 * wmax / 3:.1f}): measured v={meas[0]:.4f} w={meas[1]:.4f} | " + " | ".join(
        f"{k} v={v:.4f} ({(v / meas[0] - 1) * 100:+.1f} %) w={om:.4f} ({(om / meas[1] - 1) * 100:+.1f} %)"
        for k, (v, om) in pred.items()))

    report["step_t63_s"] = t63
    report["arc_check"] = {"measured": meas, **pred}
    report["suggested"] = {"wheel_radius": round(radius, 4), "wheel_base": round(base, 4)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.robot}_isaac.json").write_text(json.dumps(report, indent=2))
    print("[calib] suggested ISAAC_JETBOT_PARAMS:", json.dumps(report["suggested"]))
    app.close()


if __name__ == "__main__":
    main()
