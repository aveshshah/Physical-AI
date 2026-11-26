"""
Makes labelled camera images for fine-tuning the VLM planner.

Each sample puts the robot somewhere on the Mcity lap with 0-4 objects ahead, takes the front camera
image, paints the route on the road, and writes the answer the planner should give:
    {"action": "follow" | "bypass_left" | "bypass_right" | "stop", "offset_m": ..., "blocker": {"label", "bbox_2d"}}

    ~/isaac-env/bin/python sim/gen_planner_data.py --n 4000 --headless      # -> runs/planner_data/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--n", type=int, default=4000, help="number of images")
parser.add_argument("--val-frac", type=float, default=0.1, help="share of images kept for validation")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--robot", default="magna_susp")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--out", default="runs/planner_data")
args = parser.parse_args()

from isaacsim import SimulationApp  # noqa: E402  (must start before the other Isaac imports)

app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from talk2drive.isaac import jetbot as J  # noqa: E402
from talk2drive.isaac.jetbot import IsaacJetbot, quat_wxyz_from_yaw  # noqa: E402
from talk2drive.isaac.props import CAM_RES, CATALOG, FRONT_BUMPER_X, HALF_WIDTH, footprint, make_front_camera, object_height, spawn_object  # noqa: E402
from talk2drive.isaac.simtools import road_height  # noqa: E402
from talk2drive.mcity import MCITY_XODR, mcity_route  # noqa: E402
from talk2drive.vlm_planner import draw_route  # noqa: E402

CLEAR = 0.6                    # gap kept from the blocker when passing [m]
BLOCK_EDGE = HALF_WIDTH + 0.5  # an object whose edge is this close to the route is in the way [m]
POOL = 3                       # copies of each object type in the scene (so at most 3 of a type per image)
NAMES = list(CATALOG)
SHARE = {"follow": 0.30, "bypass_left": 0.30, "bypass_right": 0.25, "stop": 0.15}   # wanted mix of answers


# ---------------------------------------------------------------- labels (pure geometry)
def paved_width(route, s):
    """How far the paved road goes to the left and to the right of the route at distance s [m]."""
    centre = route.xy[int(s / route.ds)]
    heading = route.heading(s)
    left = np.array([-math.sin(heading), math.cos(heading)])
    steps = np.arange(0.0, 8.0, 0.15)
    widths = []
    for sign in (1.0, -1.0):   # left, then right
        points = centre + np.outer(sign * steps, left)
        off_road = np.nonzero(~J.is_paved(points[:, 0], points[:, 1]))[0]
        if len(off_road) == 0:
            widths.append(float(steps[-1]))                  # paved all the way out to 8 m
        elif off_road[0] == 0:
            widths.append(0.0)                               # not even the centre is paved
        else:
            widths.append(float(steps[off_road[0] - 1]))     # last paved step
    return widths[0], widths[1]


def label(objects, s_bumper, room_left, room_right):
    """The right answer for this layout. Returns (action, offset, index of the blocker or None).

    blocker  = nearest object 2-40 m ahead whose edge is within BLOCK_EDGE of the route
    a side is possible if the robot fits on the paved road and no other object is in the way there
    both possible -> the smaller swerve (ties: left);  none possible -> stop
    """
    ahead = []
    for i, o in enumerate(objects):
        if 2.0 < o["s"] - s_bumper < 40.0 and abs(o["lat"]) - footprint(o["name"]) < BLOCK_EDGE:
            ahead.append(i)
    if not ahead:
        return "follow", 0.0, None

    b = min(ahead, key=lambda i: objects[i]["s"])
    blocker = objects[b]
    r = footprint(blocker["name"])
    options = []   # (offset, side)
    for side in ("bypass_left", "bypass_right"):
        if side == "bypass_left":
            offset = blocker["lat"] + r + HALF_WIDTH + CLEAR
            signed = offset
            room = room_left
        else:
            offset = -(blocker["lat"] - r - HALF_WIDTH - CLEAR)
            signed = -offset
            room = room_right
        if offset < 0.3:   # always move at least 0.3 m
            offset = 0.3
            signed = 0.3 if side == "bypass_left" else -0.3
        if abs(signed) + HALF_WIDTH + 0.2 > room:
            continue       # the robot would leave the paved road
        other_object_in_the_way = False
        for j, o in enumerate(objects):
            if (j != b and abs(o["s"] - blocker["s"]) < 5.0
                    and abs(o["lat"] - signed) < footprint(o["name"]) + HALF_WIDTH + 0.4):
                other_object_in_the_way = True
        if not other_object_in_the_way:
            options.append((offset, side))

    if not options:
        return "stop", 0.0, b
    options.sort(key=lambda opt: (round(opt[0] / 0.2), 0 if opt[1] == "bypass_left" else 1))
    best_offset, best_side = options[0]
    return best_side, round(best_offset, 1), b


# ---------------------------------------------------------------- random layouts
def random_name(rng):
    return NAMES[rng.integers(len(NAMES))]


def random_layout(rng, route, candidates, paved_cache):
    """A random scene: obstacle position, objects, robot position, and the right answer.

    kind:  clear     no blocker
           offroute  only objects well off the route
           single    one blocker
           side      a blocker with a second object beside it
           both      a blocker with objects on both sides
           far       a blocker more than 40 m ahead (too far to matter yet)
    """
    s_obstacle = float(rng.choice(candidates))
    if s_obstacle not in paved_cache:
        paved_cache[s_obstacle] = paved_width(route, s_obstacle)
    room_left, room_right = paved_cache[s_obstacle]
    kind = rng.choice(["clear", "offroute", "single", "side", "both", "far"], p=[0.15, 0.1, 0.37, 0.25, 0.08, 0.05])

    objects = []   # each: name, s (distance along route), lat (sideways, + = left)
    if kind in ("single", "side", "both", "far"):
        name = random_name(rng)
        s = s_obstacle + (rng.uniform(42, 60) if kind == "far" else 0.0)
        objects.append(dict(name=name, s=s, lat=float(rng.uniform(-0.9, 0.9))))
        blocker_lat = objects[0]["lat"]
        if kind == "side":
            sign = rng.choice([-1.0, 1.0])
            name = random_name(rng)
            s = s_obstacle + rng.uniform(-1.5, 1.5)
            objects.append(dict(name=name, s=s, lat=blocker_lat + sign * rng.uniform(1.3, 2.4)))
        if kind == "both":
            for sign in (-1.0, 1.0):
                name = random_name(rng)
                s = s_obstacle + rng.uniform(-1.0, 1.0)
                objects.append(dict(name=name, s=s, lat=blocker_lat + sign * rng.uniform(1.3, 2.0)))

    # clutter well off the route
    if kind == "offroute":
        n_clutter = rng.integers(1, 4)
    else:
        n_clutter = rng.integers(0, 3)
    for _ in range(n_clutter):
        name = random_name(rng)
        s = s_obstacle + rng.uniform(-10, 25)
        objects.append(dict(name=name, s=s, lat=float(rng.choice([-1, 1]) * rng.uniform(2.8, 5.5))))

    # at most POOL objects of one type (that is all the scene has)
    kept, count = [], {}
    for o in objects:
        if count.get(o["name"], 0) < POOL:
            kept.append(o)
            count[o["name"]] = count.get(o["name"], 0) + 1
    objects = kept

    # the robot: its front bumper 6-40 m before the obstacle, usually near the lane centre
    distance = float(rng.uniform(6.0, 40.0))
    s_bumper = s_obstacle - distance
    s_robot = s_bumper - FRONT_BUMPER_X
    if rng.random() < 0.7:
        robot_lat = float(rng.normal(0, 0.15))
    else:
        robot_lat = float(rng.uniform(-1.8, 1.8))   # mid-swerve
    heading = route.heading(s_robot)
    left = np.array([-math.sin(heading), math.cos(heading)])
    robot_xy = route.xy[int(s_robot / route.ds)] + robot_lat * left
    yaw = heading + math.radians(rng.normal(0, 2.0)) - 0.15 * robot_lat / 3.0 * (1 if rng.random() < 0.5 else 0)

    action, offset, blocker = label(objects, s_bumper, room_left, room_right)
    return dict(kind=kind, objects=objects, distance=distance, s_robot=s_robot, robot_lat=robot_lat,
                robot_xy=robot_xy, yaw=yaw, room_left=room_left, room_right=room_right,
                action=action, offset=offset, blocker=blocker)


# ---------------------------------------------------------------- scene
def make_object_pool(stage):
    """POOL hidden copies of every object type; each image shows and moves the ones it needs."""
    from pxr import UsdGeom

    pool = {}
    for name in NAMES:
        pool[name] = []
        for k in range(POOL):
            path = f"/World/Pool/{name.replace(' ', '_')}_{k}"
            spawn_object(stage, path, name, (0.0, 0.0), -100.0, collide=False)
            prim = stage.GetPrimAtPath(path)
            UsdGeom.Imageable(prim).MakeInvisible()
            pool[name].append(prim)
    return pool


def show_objects(pool, objects, route):
    """Hide every pooled object, then move and show one copy per object in the layout."""
    from pxr import Gf, UsdGeom

    for prims in pool.values():
        for prim in prims:
            UsdGeom.Imageable(prim).MakeInvisible()
    used = {}
    for o in objects:
        k = used.get(o["name"], 0)
        used[o["name"]] = k + 1
        heading = route.heading(o["s"])
        o["xy"] = route.xy[int(o["s"] / route.ds)] + o["lat"] * np.array([-math.sin(heading), math.cos(heading)])
        prim = pool[o["name"]][k]
        height = object_height(o["name"]) / 2
        UsdGeom.Xformable(prim).GetOrderedXformOps()[0].Set(
            Gf.Vec3d(float(o["xy"][0]), float(o["xy"][1]), float(road_height(*o["xy"])[0]) + height))
        UsdGeom.Imageable(prim).MakeVisible()


def box_in_image(camera, name, xy):
    """The object's box in the image, as the VLM gives it: [x1, y1, x2, y2] scaled to 0..1000."""
    r, h = footprint(name), object_height(name)
    z0 = float(road_height(*xy)[0])
    corners = np.array([[xy[0] + dx, xy[1] + dy, z0 + dz] for dx in (-r, r) for dy in (-r, r) for dz in (0.0, h)])
    uv = np.asarray(camera.get_image_coords_from_world_points(corners), dtype=float)
    (u0, v0), (u1, v1) = uv.min(axis=0), uv.max(axis=0)
    u0, u1 = np.clip([u0, u1], 0, CAM_RES[0] - 1)
    v0, v1 = np.clip([v0, v1], 0, CAM_RES[1] - 1)
    return [round(1000 * u0 / CAM_RES[0]), round(1000 * v0 / CAM_RES[1]), round(1000 * u1 / CAM_RES[0]), round(1000 * v1 / CAM_RES[1])]


# ---------------------------------------------------------------- main
def main():
    out = REPO / args.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    route, lap_length, _, _ = mcity_route(MCITY_XODR, "104,1", 1)
    start = (route.xy[0][0], route.xy[0][1], route.heading(0.0))
    bot = IsaacJetbot(env="mcity", robot=args.robot, physics_dt=0.01, start_pose=start, render=True)
    J.build_paved_mask(bot.world.stage, "mcity")
    camera = make_front_camera(bot)
    rng = np.random.default_rng(args.seed)

    # obstacle positions: places that are straight from 45 m before to 10 m after
    candidates = []
    for s in np.arange(60.0, lap_length - 60.0, 2.0):
        if np.abs(route.kappa_many(np.arange(s - 45, s + 10, 1.0))).max() < 0.025:
            candidates.append(s)
    candidates = np.array(candidates)
    print(f"[gen] {len(candidates)} candidate obstacle positions on the {lap_length:.0f} m lap", flush=True)
    paved_cache = {}

    pool = make_object_pool(bot.world.stage)
    bot.reset_to(start, settle_steps=5)

    n_val = int(args.n * args.val_frac)
    train_file = open(out / "train.jsonl", "w")
    val_file = open(out / "val.jsonl", "w")
    counts = {}   # how many of each answer so far
    for i in range(args.n):
        # draw layouts until the answer is one that is still below its share
        for _ in range(200):
            layout = random_layout(rng, route, candidates, paved_cache)
            if counts.get(layout["action"], 0) <= SHARE[layout["action"]] * (i + 1):
                break

        # set up the scene and move the robot there
        show_objects(pool, layout["objects"], route)
        x, y = layout["robot_xy"]
        z = float(road_height(x, y)[0]) + 0.02
        bot.robot.set_world_pose(position=np.array([x, y, z]), orientation=quat_wxyz_from_yaw(layout["yaw"]))
        bot.robot.set_linear_velocity(np.zeros(3))
        bot.robot.set_angular_velocity(np.zeros(3))
        for _ in range(4):   # the move takes effect on the next physics step; the camera needs a few frames
            bot.world.step(render=True)

        # camera image with the route painted on it
        rgb = np.asarray(camera.get_rgba())[:, :, :3].astype(np.uint8)
        depth = np.asarray(camera.get_depth(), dtype=float)
        if rgb.size == 0 or depth.size == 0:
            continue
        s_ahead = np.arange(layout["s_robot"] + 2.0, layout["s_robot"] + 45.0, 0.5)
        xy = route.xy[np.clip((s_ahead / route.ds).astype(int), 0, len(route.xy) - 1)]
        image = draw_route(Image.fromarray(rgb.copy()), camera, np.c_[xy, road_height(xy[:, 0], xy[:, 1]) + 0.02], depth)
        image = image.resize((960, 540))
        file_name = f"{i:05d}.jpg"
        image.save(out / "images" / file_name, quality=92)

        # the answer, and what was in the scene
        b = layout["blocker"]
        if b is None:
            blocker = None
        else:
            o = layout["objects"][b]
            blocker = {"label": o["name"], "bbox_2d": box_in_image(camera, o["name"], o["xy"])}
        target = {"action": layout["action"], "offset_m": layout["offset"], "blocker": blocker}
        objects = []
        for o in layout["objects"]:
            objects.append({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in o.items()})
        record = {"image": f"images/{file_name}", "target": target, "kind": str(layout["kind"]),
                  "distance_m": layout["distance"], "robot_lat": layout["robot_lat"],
                  "paved_left": layout["room_left"], "paved_right": layout["room_right"], "objects": objects}
        if i < n_val:
            val_file.write(json.dumps(record) + "\n")
        else:
            train_file.write(json.dumps(record) + "\n")
        counts[layout["action"]] = counts.get(layout["action"], 0) + 1
        if i % 200 == 0:
            print(f"[gen] {i}/{args.n}  {counts}", flush=True)

    train_file.close()
    val_file.close()
    print(f"[gen] done: {counts}", flush=True)
    app.close()


if __name__ == "__main__":
    main()
