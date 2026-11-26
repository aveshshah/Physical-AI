"""
"Drive to the blue barrel": the VLM finds the named object in the camera image, the depth camera
tells how far away it is, and the NMPC drives the robot to stop just in front of it.

Each trial puts a target and a few other objects 12-35 m ahead of the robot on Mcity's long straight.
While still far away, the robot looks again every --relook seconds and corrects the goal.
Scored against the truth: right object? position error? where did it stop?

    .venv/bin/python scripts/vlm_ground_server.py --config configs/sim_4b.yaml &     # start the VLM first
    ~/isaac-env/bin/python sim/m3_vlm_goal.py                                         # 10 trials
    ~/isaac-env/bin/python sim/m3_vlm_goal.py --trials 1 --target "blue barrel"
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

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--trials", type=int, default=10)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--target", default=None, help="always use this target (catalog name), e.g. 'blue barrel'")
parser.add_argument("--distractors", type=int, default=3, help="number of other objects")
parser.add_argument("--robot", default="magna_susp")
parser.add_argument("--vlm-url", default="http://127.0.0.1:8765")
parser.add_argument("--standoff", type=float, default=2.0, help="stop with the front bumper this far from the object centre [m]")
parser.add_argument("--relook", type=float, default=1.5, help="seconds between VLM looks while driving (0 = look once)")
parser.add_argument("--relook-until", type=float, default=9.0, help="stop re-looking within this distance of the object [m]")
parser.add_argument("--v-max", type=float, default=15.0)
parser.add_argument("--nmpc-dir", default=os.path.expanduser("~/Documents/carla_nmpc_project"))
parser.add_argument("--xodr", default=None, help="Mcity OpenDRIVE map (default: talk2drive.mcity.MCITY_XODR)")
parser.add_argument("--start-s", type=float, default=15.0, help="start this far along the Mcity route [m]")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--out", default="runs/m3")
args = parser.parse_args()

# settings read by the CARLA NMPC project when it is imported
os.environ["NMPC_V_MAX"] = f"{args.v_max:g}"
os.environ["DRIVE_STYLE"] = "sport"
os.environ["NMPC_SPEED_HORIZON"] = "1"
os.environ.setdefault("ACADOS_SOURCE_DIR", os.path.expanduser("~/acados"))

from isaacsim import SimulationApp  # noqa: E402  (must start before the other Isaac imports)

app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from talk2drive.disturbances import BodyForceApplier, RoadLoadModel, parse_scenario  # noqa: E402
from talk2drive.isaac.jetbot import IsaacJetbot  # noqa: E402
from talk2drive.isaac.props import CAM_RES, CATALOG, FRONT_BUMPER_X, footprint, make_front_camera, spawn_object  # noqa: E402
from talk2drive.isaac.simtools import VlmClient, chase_view  # noqa: E402
from talk2drive.mcity import MCITY_XODR, mcity_lap  # noqa: E402
from talk2drive.opendrive import RoutePath  # noqa: E402
from talk2drive.tadpole import TadpoleDriver, load_tadpole_stack  # noqa: E402
from talk2drive.vlm_planner import locate_box  # noqa: E402
from vlm_assistant.grounding import to_pixels  # noqa: E402

MAX_TRIAL_TIME = 40.0   # give up a trial after this much simulated time [s]


# ---------------------------------------------------------------- objects
def place_objects(rng, road, s_start):
    """The target and the other objects, 12-35 m ahead, at least 3 m apart, none on the way to the target.
    Returns (target name, [(name, xy), ...]) with the target first."""
    names = list(CATALOG)
    target = args.target or names[rng.integers(len(names))]
    others = [n for n in names if n != target]
    rng.shuffle(others)
    start = road.xy[int(s_start / road.ds)]

    for _ in range(200):   # try random layouts until one fits the rules
        objects = []
        for name in [target, *others[:args.distractors]]:
            s = s_start + rng.uniform(12.0, 35.0)
            sideways = rng.uniform(-2.2, 2.2)
            heading = road.heading(s)
            xy = road.xy[int(s / road.ds)] + sideways * np.array([-math.sin(heading), math.cos(heading)])
            objects.append((name, xy))

        positions = np.array([xy for _, xy in objects])
        if too_close_together(positions, 3.0):
            continue
        if any(distance_to_line(start, positions[0], p) < 1.8 for p in positions[1:]):
            continue   # another object sits on the straight line from the start to the target
        return target, objects
    raise RuntimeError("could not place objects")


def too_close_together(positions, min_gap):
    gaps = [np.linalg.norm(positions[i] - positions[j]) for i in range(len(positions)) for j in range(i)]
    return min(gaps, default=9) < min_gap


def distance_to_line(a, b, p):
    """Distance from point p to the line segment a-b."""
    along = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1)
    return np.linalg.norm(a + along * (b - a) - p)


def true_box(camera, prim):
    """The object's real box in the image (its 3D bounding box corners projected), or None if off screen."""
    from pxr import Usd, UsdGeom

    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(prim).ComputeAlignedRange()
    lo, hi = np.array(bounds.GetMin()), np.array(bounds.GetMax())
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    uv = camera.get_image_coords_from_world_points(corners)
    (u0, v0), (u1, v1) = uv.min(axis=0), uv.max(axis=0)
    if u1 < 0 or v1 < 0 or u0 > CAM_RES[0] or v0 > CAM_RES[1]:
        return None
    return float(u0), float(v0), float(u1), float(v1)


# ---------------------------------------------------------------- path
def approach_path(x, y, heading, object_xy, standoff):
    """A smooth curve from the robot to where its front bumper is `standoff` m from the object,
    arriving pointed at the object, plus 15 m beyond. Returns (path, where on it to stop, end point)."""
    obj = np.asarray(object_xy, dtype=float)
    p0 = np.array([x, y])
    d = obj - p0
    towards_object = d / max(float(np.linalg.norm(d)), 1e-6)
    end = obj - (standoff + FRONT_BUMPER_X) * towards_object
    length = float(np.linalg.norm(end - p0))

    # cubic Hermite curve: starts along the robot's heading, ends pointing at the object
    t0 = np.array([math.cos(heading), math.sin(heading)]) * 0.45 * length
    t1 = towards_object * 0.45 * length
    s = np.linspace(0, 1, max(20, int(length / 0.1)))[:, None]
    h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
    curve = h00 * p0 + h10 * t0 + h01 * end + h11 * t1
    beyond = end + np.linspace(0.1, 15.0, 150)[:, None] * towards_object
    path = RoutePath(np.r_[curve, beyond], smooth_m=0.0)
    s_stop = path.project(end[0], end[1], length)[0]
    return path, float(s_stop), end


# ---------------------------------------------------------------- one VLM look
def look(vlm, camera, stage, query, t, file):
    """Ask the VLM where the target is; save the image (magenta = VLM's box, green = truth)."""
    rgb = np.asarray(camera.get_rgba())[:, :, :3].astype(np.uint8)
    depth = np.asarray(camera.get_depth(), dtype=float)
    answer = vlm.ask(Image.fromarray(rgb), query, timeout=120)
    boxes = answer.get("boxes", [])
    estimate = None
    box = to_pixels(boxes[0]["bbox_2d"], CAM_RES[0], CAM_RES[1]) if boxes else None
    if box is not None:
        estimate = locate_box(camera, depth, box)   # (xy, radius) from the box and the depth image

    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    truth = true_box(camera, stage.GetPrimAtPath("/World/M3Objects/obj0"))
    if truth:
        draw.rectangle(truth, outline=(0, 255, 0), width=2)
    if box:
        draw.rectangle(box, outline=(255, 0, 255), width=3)
    draw.text((10, 10), f"t={t:.1f}s  query: {query}  ({answer.get('ms', 0):.0f} ms)  magenta = VLM, green = truth", fill=(255, 255, 255))
    image.save(file)
    return dict(t=t, ms=answer.get("ms"), raw=answer.get("raw"), n_boxes=len(boxes), box_px=box, gt_box=truth,
                est=None if estimate is None else [float(estimate[0][0]), float(estimate[0][1])],
                est_radius=None if estimate is None else estimate[1])


# ---------------------------------------------------------------- one trial
def run_trial(trial, rng, road, start, bot, camera, driver, vlm, line_drawer, TS, out):
    x0, y0, heading0 = start
    stage = bot.world.stage
    target, objects = place_objects(rng, road, args.start_s)
    query = f"the {target}"
    if stage.GetPrimAtPath("/World/M3Objects"):
        stage.RemovePrim("/World/M3Objects")
    for i, (name, xy) in enumerate(objects):
        spawn_object(stage, f"/World/M3Objects/obj{i}", name, xy, bot.ground_height(*xy))
    bot.reset_to(start, settle_steps=60)
    driver.reset()
    for _ in range(4):   # let the camera catch up
        bot.step(render=True)
    true_xy = objects[0][1]
    print(f"[m3] trial {trial}: target '{target}' at ({true_xy[0]:.1f}, {true_xy[1]:.1f}), "
          f"{len(objects) - 1} distractors: {[o[0] for o in objects[1:]]}")
    folder = out / f"trial_{trial:02d}"
    folder.mkdir(exist_ok=True)

    # first look: where is the target?
    looks = [look(vlm, camera, stage, query, 0.0, folder / "look_00.png")]
    if looks[0]["est"] is None:
        print(f"[m3]   VLM did not find it: {looks[0]['raw']!r}")
        return dict(trial=trial, target=target, found=False, looks=looks)
    goal = np.array(looks[0]["est"])
    path, s_stop, _ = approach_path(x0, y0, heading0, goal, args.standoff)
    driver.set_path(path)

    # drive, looking again every --relook seconds until close
    log = {k: [] for k in ("t", "x", "y", "psi", "vx", "v_ref", "cte", "delta_cmd", "s", "solve_ms")}
    step, last_look, hit = 0, 0.0, False
    while step * TS < MAX_TRIAL_TIME and app.is_running():
        t = step * TS
        row = driver.tick(t, s_stop, stop_lag=0.6)
        for key in log:
            log[key].append(row[key])
        bumper = np.array([row["x"], row["y"]]) + FRONT_BUMPER_X * np.array([math.cos(row["psi"]), math.sin(row["psi"])])

        far_away = np.linalg.norm(goal - bumper) > args.relook_until
        if args.relook > 0 and t - last_look >= args.relook and far_away:
            new_look = look(vlm, camera, stage, query, t, folder / f"look_{len(looks):02d}.png")
            looks.append(new_look)
            last_look = t
            if new_look["est"] is not None:
                goal = np.array(new_look["est"])
                path, s_stop, _ = approach_path(row["x"], row["y"], row["psi"], goal, args.standoff)
                driver.set_path(path)

        if line_drawer is not None:   # path in green, goal as a magenta dot
            line_drawer.clear_lines()
            line_drawer.clear_points()
            z = bot.ground_height(row["x"], row["y"]) + 0.08
            points = [(float(a), float(b), z) for a, b in path.xy[::8]]
            line_drawer.draw_lines(points[:-1], points[1:], [(0.1, 0.9, 0.2, 1)] * (len(points) - 1), [3] * (len(points) - 1))
            line_drawer.draw_points([(float(goal[0]), float(goal[1]), z + 1.2)], [(1, 0, 1, 1)], [18])
        if not args.headless:
            chase_view(row["x"], row["y"], row["psi"], bot.ground_height(row["x"], row["y"]), 6.5, 3.4, 6, 0.5)

        driver.step(render=True)
        step += 1
        for name, xy in objects:   # touching any object?
            if np.linalg.norm(np.asarray(xy) - bumper) < footprint(name) + 0.1:
                hit = True
        stopped = row["s"] >= s_stop - 1.0 and abs(row["vx"]) < 0.05
        if hit or stopped:
            break

    result = score_trial(trial, target, objects, looks, goal, log, hit, x0, y0)
    np.savez_compressed(folder / "log.npz", **{k: np.asarray(v) for k, v in log.items()}, path_xy=path.xy,
                        objects=np.array([o[1] for o in objects]))
    print(f"[m3]   picked '{result['picked']}' ({'right' if result['correct_object'] else 'WRONG'}), position error "
          f"{result['est_error_first_look_m']:.2f} m first look -> {result['est_error_final_m']:.2f} m final; stopped "
          f"{result['stop_distance_to_object_m']:.2f} m from it (wanted {args.standoff:.1f}), lateral {result['lateral_miss_m']:.2f} m, "
          f"{result['time_s']:.1f} s, top speed {result['max_speed']:.1f} m/s, VLM {result['vlm_ms_mean']:.0f} ms/look, "
          f"{'SUCCESS' if result['success'] else 'fail'}{' (HIT)' if hit else ''}")
    return result


def score_trial(trial, target, objects, looks, goal, log, hit, x0, y0):
    heading = np.array([math.cos(log["psi"][-1]), math.sin(log["psi"][-1])])
    bumper = np.array((log["x"][-1], log["y"][-1])) + FRONT_BUMPER_X * heading
    true = np.asarray(objects[0][1])
    to_object = true - bumper
    picked = min(objects, key=lambda o: np.linalg.norm(np.asarray(o[1]) - goal))[0]   # object nearest the goal
    r = dict(trial=trial, target=target, found=True, picked=picked, correct_object=picked == target,
             est_error_first_look_m=float(np.linalg.norm(np.array(looks[0]["est"]) - true)),
             est_error_final_m=float(np.linalg.norm(goal - true)),
             true_range_m=float(np.linalg.norm(true - np.array([x0, y0]))),
             stop_distance_to_object_m=float(np.linalg.norm(to_object)),
             stop_error_m=float(np.linalg.norm(to_object) - args.standoff),
             lateral_miss_m=float(abs(heading[0] * to_object[1] - heading[1] * to_object[0])),
             collided=hit, time_s=float(log["t"][-1]), max_speed=float(max(log["vx"])),
             vlm_ms_mean=float(np.mean([lk["ms"] for lk in looks if lk["ms"]])), looks=looks)
    r["success"] = bool(r["correct_object"] and not hit and abs(r["stop_error_m"]) < 0.75 and r["lateral_miss_m"] < 0.75)
    return r


def summarise(results, model):
    found = [r for r in results if r.get("found")]

    def mean(key, of=found, fn=float):
        return fn(np.mean([r[key] for r in of])) if of else None

    def biggest(key):
        return float(np.max([r[key] for r in found])) if found else None

    return {
        "trials": len(results), "vlm": model,
        "success_rate": float(np.mean([r.get("success", False) for r in results])) if results else 0.0,
        "found_rate": float(np.mean([r["found"] for r in results])) if results else 0.0,
        "correct_object_rate": mean("correct_object") if found else 0.0,
        "position_error_first_look_m": {"mean": mean("est_error_first_look_m"), "max": biggest("est_error_first_look_m")},
        "position_error_final_m": {"mean": mean("est_error_final_m"), "max": biggest("est_error_final_m")},
        "stop_error_m": {"mean_abs": float(np.mean([abs(r["stop_error_m"]) for r in found])) if found else None},
        "lateral_miss_m": {"mean": mean("lateral_miss_m")},
        "collisions": int(sum(r.get("collided", False) for r in found)),
        "vlm_ms_mean": mean("vlm_ms_mean"),
    }


# ---------------------------------------------------------------- main
def main():
    out = Path(args.out) if os.path.isabs(args.out) else REPO / args.out
    out.mkdir(parents=True, exist_ok=True)
    vlm = VlmClient(args.vlm_url)
    try:
        health = vlm.health()
    except OSError as e:
        sys.exit(f"[m3] grounding service not reachable at {args.vlm_url} ({e}); start scripts/vlm_ground_server.py")
    print(f"[m3] VLM: {health['model']}")

    nmpc = load_tadpole_stack(args.nmpc_dir, out / "solver_build", ay_curve=2.4)
    TS = float(nmpc.P.NMPC_TS)
    road, _, _ = mcity_lap(args.xodr or MCITY_XODR, "104,1", 1)   # the m2 lap; it starts on a long straight
    x0, y0 = road.xy[int(args.start_s / road.ds)]
    start = (x0, y0, road.heading(args.start_s))
    bot = IsaacJetbot(env="mcity", robot=args.robot, physics_dt=TS / 2, start_pose=start, render=True)
    bot.enable_torque_mode()
    bot.params.max_steer = float(np.deg2rad(nmpc.P.MAX_STEER_DEG))
    camera = make_front_camera(bot)
    loads = RoadLoadModel(parse_scenario("nominal"), nmpc.P, 100.0, dt=TS)
    driver = TadpoleDriver(nmpc, bot, loads, BodyForceApplier("/World/Jetbot/chassis"))
    try:
        from isaacsim.util.debug_draw import _debug_draw

        line_drawer = _debug_draw.acquire_debug_draw_interface()
    except ImportError:
        line_drawer = None

    rng = np.random.default_rng(args.seed)
    results = []
    for trial in range(args.trials):
        results.append(run_trial(trial, rng, road, start, bot, camera, driver, vlm, line_drawer, TS, out))

    summary = summarise(results, health["model"])
    (out / "results.json").write_text(json.dumps({"summary": summary, "trials": results}, indent=1, default=float))
    print("[m3] " + json.dumps(summary, indent=1))
    app.close()


if __name__ == "__main__":
    main()
