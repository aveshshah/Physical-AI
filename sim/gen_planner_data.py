"""Labelled camera images for fine-tuning the VLM planner (behaviour + local plan around obstacles).

    ~/isaac-env/bin/python sim/gen_planner_data.py --n 4000 --headless      # -> runs/planner_data/

Each sample places the Magna robot somewhere on the Mcity lap (straight-ish stretches) with 0-4
objects ahead, renders the front camera, paints the route on the road (hidden behind objects, exactly
as the planner sees it at run time) and writes the answer the planner should give:

    {"action": "follow" | "bypass_left" | "bypass_right" | "stop",
     "offset_m": lateral distance from the route to drive at while passing,
     "blocker": {"label": "...", "bbox_2d": [x1, y1, x2, y2] (0..1000)} or null}

Labels come from geometry (route frame, + = left):
  blocker   nearest object 2-40 m ahead whose edge is within 1.1 m of the route (the robot is 1.16 m wide)
  a side is free when the robot can pass at ``offset`` = enough for 0.6 m clearance, nothing else is in
            its way within +-5 m of the blocker, and the robot stays on pavement (Mcity road mesh)
  both free -> the side needing the smaller swerve (ties: left); none free -> stop
In 30 % of samples the robot is already off the route (mid-manoeuvre) by up to 1.8 m.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--n", type=int, default=4000)
ap.add_argument("--val-frac", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--robot", default="magna_susp")
ap.add_argument("--headless", action="store_true")
ap.add_argument("--out", default="runs/planner_data")
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from talk2drive.isaac import jetbot as J  # noqa: E402
from talk2drive.isaac.jetbot import IsaacJetbot, quat_wxyz_from_yaw  # noqa: E402
from talk2drive.isaac.props import CAM_RES, CATALOG, FRONT_BUMPER_X, HALF_WIDTH, footprint, make_front_camera, object_height, spawn_object  # noqa: E402
from talk2drive.isaac.simtools import road_height  # noqa: E402
from talk2drive.mcity import MCITY_XODR, mcity_route  # noqa: E402
from talk2drive.vlm_planner import draw_route  # noqa: E402

CLEAR = 0.6  # clearance kept from the blocker when passing [m]
BLOCK_EDGE = HALF_WIDTH + 0.5  # an object whose edge is this close to the route is in the way
POOL = 3  # prims per object type


def paved_extent(route, s: float) -> tuple[float, float]:
    """Paved road to the left and right of the route at ``s`` [m] (contiguous with the lane centre)."""
    c = route.xy[int(s / route.ds)]
    hd = route.heading(s)
    nrm = np.array([-math.sin(hd), math.cos(hd)])
    lat = np.arange(0.0, 8.0, 0.15)
    ext = []
    for sign in (1.0, -1.0):
        P = c + np.outer(sign * lat, nrm)
        ok = J.is_paved(P[:, 0], P[:, 1])
        bad = np.nonzero(~ok)[0]
        ext.append(float(lat[bad[0] - 1]) if len(bad) and bad[0] > 0 else (0.0 if len(bad) else float(lat[-1])))
    return ext[0], ext[1]


def label(objs, s_bumper: float, ext_l: float, ext_r: float):
    """objs: dicts with s, lat, name. Returns (action, offset, blocker index or None)."""
    ahead = [i for i, o in enumerate(objs) if 2.0 < o["s"] - s_bumper < 40.0
             and abs(o["lat"]) - footprint(o["name"]) < BLOCK_EDGE]
    if not ahead:
        return "follow", 0.0, None
    b = min(ahead, key=lambda i: objs[i]["s"])
    ob = objs[b]
    r = footprint(ob["name"])
    opts = []
    for side, off in (("bypass_left", ob["lat"] + r + HALF_WIDTH + CLEAR), ("bypass_right", -(ob["lat"] - r - HALF_WIDTH - CLEAR))):
        signed = off if side == "bypass_left" else -off
        if off < 0.3:
            off, signed = 0.3, (0.3 if side == "bypass_left" else -0.3)
        room = ext_l if side == "bypass_left" else ext_r
        if abs(signed) + HALF_WIDTH + 0.2 > room:
            continue
        clash = any(j != b and abs(o["s"] - ob["s"]) < 5.0 and abs(o["lat"] - signed) < footprint(o["name"]) + HALF_WIDTH + 0.4
                    for j, o in enumerate(objs))
        if not clash:
            opts.append((off, side))
    if not opts:
        return "stop", 0.0, b
    opts.sort(key=lambda t: (round(t[0] / 0.2), 0 if t[1] == "bypass_left" else 1))
    return opts[0][1], round(opts[0][0], 1), b


def main() -> None:
    from pxr import Gf, UsdGeom

    out = REPO / args.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    route, lap_length, _, _ = mcity_route(MCITY_XODR, "104,1", 1)
    x0, y0 = route.xy[0]
    bot = IsaacJetbot(env="mcity", robot=args.robot, physics_dt=0.01, start_pose=(x0, y0, route.heading(0.0)), render=True)
    J.build_paved_mask(bot.world.stage, "mcity")
    cam = make_front_camera(bot)
    stage = bot.world.stage
    rng = np.random.default_rng(args.seed)

    # candidate obstacle positions: stretches that are straight-ish from 45 m before to 10 m after
    s_all = np.arange(60.0, lap_length - 60.0, 2.0)
    cand = np.array([s for s in s_all if np.abs(route.kappa_many(np.arange(s - 45, s + 10, 1.0))).max() < 0.025])
    print(f"[gen] {len(cand)} candidate obstacle positions on the {lap_length:.0f} m lap", flush=True)
    ext_cache = {}

    # object pool: POOL prims per type, parked far away and hidden
    pool = {}
    names = list(CATALOG)
    for n in names:
        for k in range(POOL):
            path = f"/World/Pool/{n.replace(' ', '_')}_{k}"
            spawn_object(stage, path, n, (0.0, 0.0), -100.0, collide=False)
            prim = stage.GetPrimAtPath(path)
            UsdGeom.Imageable(prim).MakeInvisible()
            pool.setdefault(n, []).append(prim)
    bot.reset_to((x0, y0, route.heading(0.0)), settle_steps=5)

    def place(prim, name, xy):
        zc = object_height(name) / 2
        op = UsdGeom.Xformable(prim).GetOrderedXformOps()[0]
        op.Set(Gf.Vec3d(float(xy[0]), float(xy[1]), float(road_height(*xy)[0]) + zc))
        UsdGeom.Imageable(prim).MakeVisible()

    def gt_box(name, xy):
        r, h = footprint(name), object_height(name)
        z0 = float(road_height(*xy)[0])
        corners = np.array([[xy[0] + dx, xy[1] + dy, z0 + dz] for dx in (-r, r) for dy in (-r, r) for dz in (0.0, h)])
        uv = np.asarray(cam.get_image_coords_from_world_points(corners), dtype=float)
        (u0, v0), (u1, v1) = uv.min(axis=0), uv.max(axis=0)
        u0, u1 = np.clip([u0, u1], 0, CAM_RES[0] - 1)
        v0, v1 = np.clip([v0, v1], 0, CAM_RES[1] - 1)
        return [round(1000 * u0 / CAM_RES[0]), round(1000 * v0 / CAM_RES[1]), round(1000 * u1 / CAM_RES[0]), round(1000 * v1 / CAM_RES[1])]

    n_val = int(args.n * args.val_frac)
    f_train, f_val = open(out / "train.jsonl", "w"), open(out / "val.jsonl", "w")
    stats = {}
    quota = {"follow": 0.30, "bypass_left": 0.30, "bypass_right": 0.25, "stop": 0.15}
    def sample_layout():
        s_b = float(rng.choice(cand))
        if s_b not in ext_cache:
            ext_cache[s_b] = paved_extent(route, s_b)
        ext_l, ext_r = ext_cache[s_b]
        kind = rng.choice(["clear", "offroute", "single", "side", "both", "far"], p=[0.15, 0.1, 0.37, 0.25, 0.08, 0.05])
        objs = []
        if kind in ("single", "side", "both", "far"):
            objs.append(dict(name=names[rng.integers(len(names))], s=s_b + (rng.uniform(42, 60) if kind == "far" else 0.0),
                             lat=float(rng.uniform(-0.9, 0.9))))
            lb = objs[0]["lat"]
            if kind == "side":
                sgn = rng.choice([-1.0, 1.0])
                objs.append(dict(name=names[rng.integers(len(names))], s=s_b + rng.uniform(-1.5, 1.5), lat=lb + sgn * rng.uniform(1.3, 2.4)))
            if kind == "both":
                for sgn in (-1.0, 1.0):
                    objs.append(dict(name=names[rng.integers(len(names))], s=s_b + rng.uniform(-1.0, 1.0), lat=lb + sgn * rng.uniform(1.3, 2.0)))
        for _ in range(rng.integers(0, 3) if kind != "offroute" else rng.integers(1, 4)):  # clutter off the route
            objs.append(dict(name=names[rng.integers(len(names))], s=s_b + rng.uniform(-10, 25),
                             lat=float(rng.choice([-1, 1]) * rng.uniform(2.8, 5.5))))
        # no more than POOL of one type
        cnt = {}
        objs = [o for o in objs if cnt.setdefault(o["name"], 0) < POOL and not cnt.update({o["name"]: cnt[o["name"]] + 1})]
        d = float(rng.uniform(6.0, 40.0))  # bumper to the obstacle position
        s_bumper = s_b - d
        s_r = s_bumper - FRONT_BUMPER_X
        robot_lat = float(rng.normal(0, 0.15) if rng.random() < 0.7 else rng.uniform(-1.8, 1.8))
        hd = route.heading(s_r)
        nrm = np.array([-math.sin(hd), math.cos(hd)])
        p = route.xy[int(s_r / route.ds)] + robot_lat * nrm
        yaw = hd + math.radians(rng.normal(0, 2.0)) - 0.15 * robot_lat / 3.0 * (1 if rng.random() < 0.5 else 0)
        action, offset, b = label(objs, s_bumper, ext_l, ext_r)
        return s_b, kind, objs, d, s_bumper, s_r, robot_lat, p, yaw, ext_l, ext_r, action, offset, b

    for i in range(args.n):
        for _attempt in range(200):  # draw layouts until the action is one still below its share
            s_b, kind, objs, d, s_bumper, s_r, robot_lat, p, yaw, ext_l, ext_r, action, offset, b = sample_layout()
            if stats.get(action, 0) <= quota[action] * (i + 1):
                break

        for prims in pool.values():
            for prim in prims:
                UsdGeom.Imageable(prim).MakeInvisible()
        used = {}
        for o in objs:
            k = used.setdefault(o["name"], 0)
            used[o["name"]] = k + 1
            c = route.xy[int(o["s"] / route.ds)]
            hdo = route.heading(o["s"])
            o["xy"] = c + o["lat"] * np.array([-math.sin(hdo), math.cos(hdo)])
            place(pool[o["name"]][k], o["name"], o["xy"])
        z = float(road_height(*p)[0]) + 0.02
        bot.robot.set_world_pose(position=np.array([p[0], p[1], z]), orientation=quat_wxyz_from_yaw(yaw))
        bot.robot.set_linear_velocity(np.zeros(3))
        bot.robot.set_angular_velocity(np.zeros(3))
        for _ in range(4):  # a teleport takes effect on the next physics step; the camera needs a few frames
            bot.world.step(render=True)
        rgb = np.asarray(cam.get_rgba())[:, :, :3].astype(np.uint8)
        depth = np.asarray(cam.get_depth(), dtype=float)
        if rgb.size == 0 or depth.size == 0:
            continue
        ss = np.arange(s_r + 2.0, s_r + 45.0, 0.5)
        R = route.xy[np.clip((ss / route.ds).astype(int), 0, len(route.xy) - 1)]
        im = draw_route(Image.fromarray(rgb.copy()), cam, np.c_[R, road_height(R[:, 0], R[:, 1]) + 0.02], depth)
        im = im.resize((960, 540))
        name = f"{i:05d}.jpg"
        im.save(out / "images" / name, quality=92)
        target = {"action": action, "offset_m": offset,
                  "blocker": None if b is None else {"label": objs[b]["name"], "bbox_2d": gt_box(objs[b]["name"], objs[b]["xy"])}}
        rec = {"image": f"images/{name}", "target": target, "kind": str(kind), "distance_m": d, "robot_lat": robot_lat,
               "paved_left": ext_l, "paved_right": ext_r,
               "objects": [{k_: (v.tolist() if isinstance(v, np.ndarray) else v) for k_, v in o.items()} for o in objs]}
        (f_val if i < n_val else f_train).write(json.dumps(rec) + "\n")
        stats[action] = stats.get(action, 0) + 1
        if i % 200 == 0:
            print(f"[gen] {i}/{args.n}  {stats}", flush=True)
    f_train.close()
    f_val.close()
    print(f"[gen] done: {stats}", flush=True)
    app.close()


if __name__ == "__main__":
    main()
