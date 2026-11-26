"""Labelled camera images for scoring the VLM planner offline (no driving).

    ~/isaac-env/bin/python sim/m3_planner_testset.py --layouts 12      # -> runs/planner_testset/

Each layout puts obstacles on Mcity's long straight: a blocker on or near the lane centre (or none),
sometimes a cone on one side of it so only the other side is free, sometimes a harmless object well off
the route. The robot is placed 12, 20 and 30 m before it and the front camera image is saved with the
route painted on the road (exactly what the planner sees), with the correct answer:

  follow        nothing within 1.5 m of the route
  bypass_left   blocked, and the left side is free (or both are free: either side is correct)
  bypass_right  blocked, right side free
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
ap = argparse.ArgumentParser()
ap.add_argument("--layouts", type=int, default=12)
ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--robot", default="magna_susp")
ap.add_argument("--headless", action="store_true")
ap.add_argument("--out", default="runs/planner_testset")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from talk2drive.isaac.jetbot import IsaacJetbot, drive_surface_height  # noqa: E402
from talk2drive.isaac.props import CAM_RES, make_front_camera, spawn_object  # noqa: E402
from talk2drive.mcity import MCITY_XODR, mcity_route  # noqa: E402
from talk2drive.vlm_planner import draw_route  # noqa: E402

BLOCKERS = ["brown cardboard box", "blue barrel", "red box", "green box", "purple barrel", "yellow ball"]


def road_h(x, y):
    return drive_surface_height(np.atleast_1d(x), np.atleast_1d(y))


def main() -> None:
    out = REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    route, _, _, _ = mcity_route(MCITY_XODR, "104,1", 1)
    x0, y0 = route.xy[0]
    bot = IsaacJetbot(env="mcity", robot=args.robot, physics_dt=0.01, start_pose=(x0, y0, route.heading(0.0)), render=True)
    cam = make_front_camera(bot)
    stage = bot.world.stage
    rng = np.random.default_rng(args.seed)
    labels = []
    s_obj = 150.0  # obstacle position along the route (long straight)
    c = route.xy[int(s_obj / route.ds)]
    hd = route.heading(s_obj)
    nrm = np.array([-math.sin(hd), math.cos(hd)])  # + = left
    for li in range(args.layouts):
        kind = ["clear", "blocked_free_both", "blocked_left_free", "blocked_right_free"][li % 4]
        objs = []
        if kind != "clear":
            objs.append((BLOCKERS[rng.integers(len(BLOCKERS))], float(rng.uniform(-0.5, 0.5)), "blocker"))
            lat = objs[0][1]
            if kind == "blocked_left_free":
                objs.append(("orange traffic cone", lat - 1.8, "side"))
            elif kind == "blocked_right_free":
                objs.append(("orange traffic cone", lat + 1.8, "side"))
        if kind == "clear" or rng.random() < 0.4:  # harmless object off the route
            objs.append((BLOCKERS[rng.integers(len(BLOCKERS))], float(rng.choice([-1, 1]) * rng.uniform(3.2, 4.5)), "off_route"))
        truth = {"clear": ["follow"], "blocked_free_both": ["bypass_left", "bypass_right"],
                 "blocked_left_free": ["bypass_left"], "blocked_right_free": ["bypass_right"]}[kind]
        if stage.GetPrimAtPath("/World/Obstacles"):
            stage.RemovePrim("/World/Obstacles")
        for i, (n, lat, _) in enumerate(objs):
            xy = c + lat * nrm
            spawn_object(stage, f"/World/Obstacles/obj{i}", n, xy, bot.ground_height(*xy))
        for dist in (12.0, 20.0, 30.0):
            s_r = s_obj - dist - 1.6  # bumper ``dist`` before the obstacle
            p = route.xy[int(s_r / route.ds)]
            bot.reset_to((p[0], p[1], route.heading(s_r)), settle_steps=20)
            for _ in range(4):
                bot.step(render=True)
            rgb = np.asarray(cam.get_rgba())[:, :, :3].astype(np.uint8).copy()
            ss = np.arange(s_r + 2.0, s_r + 45.0, 0.5)
            R = route.xy[(ss / route.ds).astype(int)]
            im = draw_route(Image.fromarray(rgb), cam, np.c_[R, road_h(R[:, 0], R[:, 1]) + 0.02],
                            np.asarray(cam.get_depth(), dtype=float))
            name = f"{li:02d}_{kind}_{int(dist)}m.png"
            im.save(out / name)
            labels.append(dict(image=name, kind=kind, distance_m=dist, correct=truth,
                               objects=[dict(name=n, lateral_m=lat, role=r) for n, lat, r in objs]))
            print(f"[set] {name}: {', '.join(f'{n} {r} {lat:+.1f} m' for n, lat, r in objs)} -> {truth}", flush=True)
    (out / "labels.json").write_text(json.dumps(labels, indent=1))
    app.close()


if __name__ == "__main__":
    main()
