"""Rigid vs sprung Magna robot on flat ground: ride height, roll gradient, side-force tipping limit.

    python sim/check_suspension.py --headless              # magna (rigid) and magna_susp
    python sim/check_suspension.py --robots magna_susp

Tests (each from rest on the default ground plane):

1. settle      -- strut travel and chassis height at rest (should be ~0 travel: preload holds design height)
2. steady turn -- 6 m/s on a 12 m circle (a_y 3 m/s^2): body roll per g
3. side push   -- robot at rest, lateral force at the 1.15 m centre of pressure ramped 0 -> 1500 N over
                  15 s; the force at which the up-wind front wheel unloads (roll runs away) is the
                  quasi-static tipping limit
4. gust        -- 1 s, 650 N side-force pulse at rest: peak roll
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
ap = argparse.ArgumentParser()
ap.add_argument("--robots", default="magna,magna_susp")
ap.add_argument("--headless", action="store_true")
ap.add_argument("--out", default="runs/suspension")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": args.headless})

import numpy as np  # noqa: E402

from talk2drive.disturbances import BodyForceApplier  # noqa: E402
from talk2drive.isaac.jetbot import IsaacJetbot  # noqa: E402

DT = 0.01
H_CP, LR = 1.15, 0.642


def pose(bot):
    pos, q = bot.robot.get_world_pose()
    w, x, y, z = (float(v) for v in q)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.asarray(pos, dtype=float), roll, yaw


def side_point(bot):
    pos, _, yaw = pose(bot)
    c, s = math.cos(yaw), math.sin(yaw)
    return pos + np.array([LR * c, LR * s, H_CP]), np.array([-s, c, 0.0])  # CP, body +y (left) in world


def run(robot: str) -> dict:
    bot = IsaacJetbot(env="grid", robot=robot, physics_dt=DT, render=not args.headless)
    bot.params.max_speed = 15.0
    push = BodyForceApplier("/World/Jetbot/chassis")
    res = {}

    bot.reset_to((0.0, 0.0, 0.0), settle_steps=200)
    res["settle"] = {"travel_mm": [round(1000 * v, 1) for v in bot.suspension_travel()],
                     "chassis_z_mm": round(1000 * pose(bot)[0][2], 1)}

    # steady turn: ramp to 6 m/s on R = 12 m, then measure
    R, v = 12.0, 6.0
    delta = math.atan(bot.params.wheelbase / R)
    rolls = []
    for i in range(int(14.0 / DT)):
        bot.set_command(min(v, 0.6 * i * DT), delta)
        bot.step()
        if i * DT > 9.0:
            rolls.append(pose(bot)[1])
    w = v / R
    ay = v * w
    res["steady_turn"] = {"speed": v, "radius": R, "a_y": ay, "roll_deg": round(math.degrees(float(np.mean(rolls))), 2),
                          "roll_gradient_deg_per_g": round(abs(math.degrees(float(np.mean(rolls)))) / (ay / 9.81), 2)}

    # quasi-static side push, robot held at rest
    bot.reset_to((0.0, 0.0, 0.0), settle_steps=100)
    trace = []
    tip_force = None
    for i in range(int(15.0 / DT)):
        F = 1500.0 * i * DT / 15.0
        cp, left = side_point(bot)
        push.set([(F * left, cp)])
        push.apply()
        bot.set_command(0.0, 0.0)
        bot.step()
        roll = pose(bot)[1]
        trace.append((F, math.degrees(roll)))
        if tip_force is None and abs(math.degrees(roll)) > 15.0:
            tip_force = F
            break
    tr = np.array(trace)
    at = {f: float(np.interp(f, tr[:, 0], tr[:, 1])) for f in (200.0, 400.0, 600.0) if f <= tr[-1, 0]}
    res["side_push"] = {"tipping_force_N": tip_force, "roll_deg_at": {int(k): round(v, 2) for k, v in at.items()},
                        "crosswind_to_tip_m_s": round(math.sqrt(tip_force / (0.5 * 1.225 * 0.9 * 3.0)), 1) if tip_force else None}

    # gust pulse
    bot.reset_to((0.0, 0.0, 0.0), settle_steps=100)
    peak = 0.0
    for i in range(int(4.0 / DT)):
        F = 650.0 if i * DT < 1.0 else 0.0
        cp, left = side_point(bot)
        push.set([(F * left, cp)])
        push.apply()
        bot.set_command(0.0, 0.0)
        bot.step()
        peak = max(peak, abs(math.degrees(pose(bot)[1])))
    res["gust_650N_1s"] = {"peak_roll_deg": round(peak, 2), "tipped": peak > 30}
    bot.world.clear()
    return res


def main() -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for r in args.robots.split(","):
        report[r] = run(r)
        print(f"[susp] {r}: {json.dumps(report[r])}", flush=True)
    (out / "check.json").write_text(json.dumps(report, indent=2))
    app.close()


if __name__ == "__main__":
    main()
