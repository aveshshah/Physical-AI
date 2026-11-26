"""M3: camera -> Qwen3-VL -> object box -> depth -> goal pose -> the CARLA tadpole NMPC drives there.

Start the grounding service first (VLM venv), then run this in the Isaac venv:

    .venv/bin/python scripts/vlm_ground_server.py --config configs/sim_4b.yaml &
    ~/isaac-env/bin/python sim/m3_vlm_goal.py                     # 10 trials, window open
    ~/isaac-env/bin/python sim/m3_vlm_goal.py --trials 1 --target "blue barrel"

Each trial puts the Magna robot on Mcity's long straight and scatters a few coloured objects 12-35 m
ahead (a target and distractors). The robot's front camera (RGB + depth, 1.45 m up behind the
windscreen) sees the road; the VLM is asked for a box around "the <target>"; the median depth inside
the box's centre and the camera intrinsics put the object in the world; a smooth path to a stop
``--standoff`` metres short of it is handed to the NMPC. While still far away the robot looks again
every ``--relook`` seconds and refines the goal.

Scored against ground truth: did the VLM find the right object, how far was the estimated position
from the real one, and where did the robot stop relative to the intended stop point.
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

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--trials", type=int, default=10)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--target", default=None, help="always use this target (catalog name), e.g. 'blue barrel'")
ap.add_argument("--distractors", type=int, default=3)
ap.add_argument("--robot", default="magna_susp")
ap.add_argument("--vlm-url", default="http://127.0.0.1:8765")
ap.add_argument("--standoff", type=float, default=2.0, help="stop with the front bumper this far from the object centre [m]")
ap.add_argument("--relook", type=float, default=1.5, help="seconds between VLM looks while driving (0 = look once)")
ap.add_argument("--relook-until", type=float, default=9.0, help="stop re-looking within this distance of the object [m]")
ap.add_argument("--v-max", type=float, default=15.0)
ap.add_argument("--nmpc-dir", default=os.path.expanduser("~/Documents/carla_nmpc_project"))
ap.add_argument("--xodr", default=None, help="Mcity OpenDRIVE map (default: talk2drive.mcity.MCITY_XODR)")
ap.add_argument("--start-s", type=float, default=15.0, help="start this far along the Mcity route [m]")
ap.add_argument("--headless", action="store_true")
ap.add_argument("--out", default="runs/m3")
args = ap.parse_args()

os.environ["NMPC_V_MAX"] = f"{args.v_max:g}"
os.environ["DRIVE_STYLE"] = "sport"
os.environ["NMPC_SPEED_HORIZON"] = "1"
os.environ.setdefault("ACADOS_SOURCE_DIR", os.path.expanduser("~/acados"))

from isaacsim import SimulationApp  # noqa: E402

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


# ---------------------------------------------------------------- perception
def gt_box(cam, prim) -> tuple[float, float, float, float] | None:
    """Ground-truth 2D box of an object (its 3D bounding box corners projected into the image)."""
    from pxr import Usd, UsdGeom

    rng = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(prim).ComputeAlignedRange()
    lo, hi = np.array(rng.GetMin()), np.array(rng.GetMax())
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    uv = cam.get_image_coords_from_world_points(corners)
    (u0, v0), (u1, v1) = uv.min(axis=0), uv.max(axis=0)
    if u1 < 0 or v1 < 0 or u0 > CAM_RES[0] or v0 > CAM_RES[1]:
        return None
    return float(u0), float(v0), float(u1), float(v1)


# ---------------------------------------------------------------- planning
def approach_path(x0, y0, psi0, obj_xy, standoff: float) -> tuple[RoutePath, float, np.ndarray]:
    """Smooth path for the robot origin (rear contact) to the point where its front bumper is
    ``standoff`` from the object centre, arriving pointed at the object; 15 m run-out beyond."""
    obj = np.asarray(obj_xy, dtype=float)
    p0 = np.array([x0, y0])
    d = obj - p0
    dist = float(np.linalg.norm(d))
    # end heading: aim at the object from a point a little before the stop (straight final approach)
    u = d / max(dist, 1e-6)
    end = obj - (standoff + FRONT_BUMPER_X) * u
    L = float(np.linalg.norm(end - p0))
    t0 = np.array([math.cos(psi0), math.sin(psi0)]) * 0.45 * L
    t1 = u * 0.45 * L
    s = np.linspace(0, 1, max(20, int(L / 0.1)))[:, None]
    h00, h10, h01, h11 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s, -2 * s**3 + 3 * s**2, s**3 - s**2
    pts = h00 * p0 + h10 * t0 + h01 * end + h11 * t1
    run = end + np.linspace(0.1, 15.0, 150)[:, None] * u
    path = RoutePath(np.r_[pts, run], smooth_m=0.0)
    s_stop = path.project(end[0], end[1], L)[0]
    return path, float(s_stop), end


# ---------------------------------------------------------------- one trial
def scenario_layout(rng, start_path, s_start):
    names = list(CATALOG)
    target = args.target or names[rng.integers(len(names))]
    others = [n for n in names if n != target]
    rng.shuffle(others)
    for _ in range(200):
        objs = []
        for n in [target, *others[:args.distractors]]:
            s_o = s_start + rng.uniform(12.0, 35.0)
            lat = rng.uniform(-2.2, 2.2)
            c = start_path.xy[int(s_o / start_path.ds)]
            hd = start_path.heading(s_o)
            objs.append((n, c + lat * np.array([-math.sin(hd), math.cos(hd)])))
        P = np.array([o[1] for o in objs])
        gaps = [np.linalg.norm(P[i] - P[j]) for i in range(len(P)) for j in range(i)]
        if min(gaps, default=9) < 3.0:
            continue
        # distractors must not sit on the approach (closer than 1.8 m to the straight line start -> target)
        p0 = start_path.xy[int(s_start / start_path.ds)]
        tgt = P[0]
        ok = True
        for q in P[1:]:
            a = np.clip(np.dot(q - p0, tgt - p0) / np.dot(tgt - p0, tgt - p0), 0, 1)
            if np.linalg.norm(p0 + a * (tgt - p0) - q) < 1.8:
                ok = False
        if ok:
            return target, objs
    raise RuntimeError("could not place objects")


def main() -> None:
    out = (REPO / args.out) if not os.path.isabs(args.out) else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vlm = VlmClient(args.vlm_url)
    try:
        health = vlm.health()
    except OSError as e:
        sys.exit(f"[m3] grounding service not reachable at {args.vlm_url} ({e}); start scripts/vlm_ground_server.py")
    print(f"[m3] VLM: {health['model']}")

    stack = load_tadpole_stack(args.nmpc_dir, out / "solver_build", ay_curve=2.4)
    P = stack.P
    TS = float(P.NMPC_TS)
    road, _, _ = mcity_lap(args.xodr or MCITY_XODR, "104,1", 1)  # the m2 lap; it starts on a long straight

    x0, y0 = road.xy[int(args.start_s / road.ds)]
    psi0 = road.heading(args.start_s)
    bot = IsaacJetbot(env="mcity", robot=args.robot, physics_dt=TS / 2, start_pose=(x0, y0, psi0), render=True)
    bot.enable_torque_mode()
    bot.params.max_steer = float(np.deg2rad(P.MAX_STEER_DEG))
    cam = make_front_camera(bot)
    loads = RoadLoadModel(parse_scenario("nominal"), P, 100.0, dt=TS)
    body_forces = BodyForceApplier("/World/Jetbot/chassis")
    driver = TadpoleDriver(stack, bot, loads, body_forces)
    try:
        from isaacsim.util.debug_draw import _debug_draw

        draw = _debug_draw.acquire_debug_draw_interface()
    except ImportError:
        draw = None
    rng = np.random.default_rng(args.seed)
    results = []
    stage = bot.world.stage
    for trial in range(args.trials):
        target, objs = scenario_layout(rng, road, args.start_s)
        query = f"the {target}"
        if stage.GetPrimAtPath("/World/M3Objects"):
            stage.RemovePrim("/World/M3Objects")
        for i, (n, xy) in enumerate(objs):
            spawn_object(stage, f"/World/M3Objects/obj{i}", n, xy, bot.ground_height(*xy))
        bot.reset_to((x0, y0, psi0), settle_steps=60)
        driver.reset()
        for _ in range(4):  # let the camera's render product catch up
            bot.step(render=True)
        true_xy = objs[0][1]
        print(f"[m3] trial {trial}: target '{target}' at ({true_xy[0]:.1f}, {true_xy[1]:.1f}), "
              f"{len(objs) - 1} distractors: {[o[0] for o in objs[1:]]}")
        tdir = out / f"trial_{trial:02d}"
        tdir.mkdir(exist_ok=True)

        def look(t, tag):
            rgb = np.asarray(cam.get_rgba())[:, :, :3].astype(np.uint8)
            depth = np.asarray(cam.get_depth(), dtype=float)
            ans = vlm.ask(Image.fromarray(rgb), query, timeout=120)
            boxes = ans.get("boxes", [])
            est = None
            box_px = to_pixels(boxes[0]["bbox_2d"], CAM_RES[0], CAM_RES[1]) if boxes else None
            if box_px is not None:
                est = locate_box(cam, depth, box_px)
            im = Image.fromarray(rgb)
            dr = ImageDraw.Draw(im)
            gtb = gt_box(cam, stage.GetPrimAtPath("/World/M3Objects/obj0"))
            if gtb:
                dr.rectangle(gtb, outline=(0, 255, 0), width=2)
            if box_px:
                dr.rectangle(box_px, outline=(255, 0, 255), width=3)
            dr.text((10, 10), f"t={t:.1f}s  query: {query}  ({ans.get('ms', 0):.0f} ms)  magenta = VLM, green = truth", fill=(255, 255, 255))
            im.save(tdir / f"look_{tag}.png")
            return dict(t=t, ms=ans.get("ms"), raw=ans.get("raw"), n_boxes=len(boxes), box_px=box_px, gt_box=gtb,
                        est=None if est is None else [float(est[0][0]), float(est[0][1])],
                        est_radius=None if est is None else est[1])

        looks = [look(0.0, "00")]
        if looks[0]["est"] is None:
            print(f"[m3]   VLM did not find it: {looks[0]['raw']!r}")
            results.append(dict(trial=trial, target=target, found=False, looks=looks))
            continue
        goal_obj = np.array(looks[0]["est"])
        path, s_stop, end = approach_path(x0, y0, psi0, goal_obj, args.standoff)
        driver.set_path(path)
        log = {k: [] for k in ("t", "x", "y", "psi", "vx", "v_ref", "cte", "delta_cmd", "s", "solve_ms")}
        k, last_look, hit = 0, 0.0, False
        while k * TS < 40.0 and app.is_running():
            t = k * TS
            row = driver.tick(t, s_stop, stop_lag=0.6)
            for key in log:
                log[key].append(row[key])
            # robot front bumper -> estimated object distance
            hd = np.array([math.cos(row["psi"]), math.sin(row["psi"])])
            bumper = np.array([row["x"], row["y"]]) + FRONT_BUMPER_X * hd
            if args.relook > 0 and t - last_look >= args.relook and np.linalg.norm(goal_obj - bumper) > args.relook_until:
                lk = look(t, f"{len(looks):02d}")
                looks.append(lk)
                last_look = t
                if lk["est"] is not None:
                    goal_obj = np.array(lk["est"])
                    path, s_stop, end = approach_path(row["x"], row["y"], row["psi"], goal_obj, args.standoff)
                    driver.set_path(path)
            if draw is not None:
                draw.clear_lines()
                draw.clear_points()
                zz = bot.ground_height(row["x"], row["y"]) + 0.08
                pp = [(float(a), float(b), zz) for a, b in path.xy[::8]]
                draw.draw_lines(pp[:-1], pp[1:], [(0.1, 0.9, 0.2, 1)] * (len(pp) - 1), [3] * (len(pp) - 1))
                draw.draw_points([(float(goal_obj[0]), float(goal_obj[1]), zz + 1.2)], [(1, 0, 1, 1)], [18])
            if not args.headless:
                chase_view(row["x"], row["y"], row["psi"], bot.ground_height(row["x"], row["y"]), 6.5, 3.4, 6, 0.5)
            driver.step(render=True)
            k += 1
            for n, xy in objs:  # contact with any object (centre distance below footprint + half the pod width)
                if np.linalg.norm(np.asarray(xy) - bumper) < footprint(n) + 0.1:
                    hit = True
            if hit or (row["s"] >= s_stop - 1.0 and abs(row["vx"]) < 0.05):
                break

        fin = log["x"][-1], log["y"][-1], log["psi"][-1]
        hd = np.array([math.cos(fin[2]), math.sin(fin[2])])
        bumper = np.array(fin[:2]) + FRONT_BUMPER_X * hd
        true = np.asarray(true_xy)
        est0 = np.array(looks[0]["est"])
        nearest = min(objs, key=lambda o: np.linalg.norm(np.asarray(o[1]) - goal_obj))[0]
        r = dict(trial=trial, target=target, found=True, picked=nearest, correct_object=nearest == target,
                 est_error_first_look_m=float(np.linalg.norm(est0 - true)),
                 est_error_final_m=float(np.linalg.norm(goal_obj - true)),
                 true_range_m=float(np.linalg.norm(true - np.array([x0, y0]))),
                 stop_distance_to_object_m=float(np.linalg.norm(true - bumper)),
                 stop_error_m=float(np.linalg.norm(true - bumper) - args.standoff),
                 lateral_miss_m=float(abs(hd[0] * (true - bumper)[1] - hd[1] * (true - bumper)[0])),
                 collided=hit, time_s=float(log["t"][-1]), max_speed=float(max(log["vx"])),
                 vlm_ms_mean=float(np.mean([lk["ms"] for lk in looks if lk["ms"]])), looks=looks)
        r["success"] = bool(r["correct_object"] and not hit and abs(r["stop_error_m"]) < 0.75 and r["lateral_miss_m"] < 0.75)
        results.append(r)
        np.savez_compressed(tdir / "log.npz", **{k_: np.asarray(v) for k_, v in log.items()}, path_xy=path.xy,
                            objects=np.array([o[1] for o in objs]))
        print(f"[m3]   picked '{nearest}' ({'right' if r['correct_object'] else 'WRONG'}), position error "
              f"{r['est_error_first_look_m']:.2f} m first look -> {r['est_error_final_m']:.2f} m final; stopped "
              f"{r['stop_distance_to_object_m']:.2f} m from it (wanted {args.standoff:.1f}), lateral {r['lateral_miss_m']:.2f} m, "
              f"{r['time_s']:.1f} s, top speed {r['max_speed']:.1f} m/s, VLM {r['vlm_ms_mean']:.0f} ms/look, "
              f"{'SUCCESS' if r['success'] else 'fail'}{' (HIT)' if hit else ''}")

    ok = [r for r in results if r.get("found")]
    summary = {
        "trials": len(results), "vlm": health["model"],
        "success_rate": float(np.mean([r.get("success", False) for r in results])) if results else 0.0,
        "found_rate": float(np.mean([r["found"] for r in results])) if results else 0.0,
        "correct_object_rate": float(np.mean([r["correct_object"] for r in ok])) if ok else 0.0,
        "position_error_first_look_m": {"mean": float(np.mean([r["est_error_first_look_m"] for r in ok])) if ok else None,
                                        "max": float(np.max([r["est_error_first_look_m"] for r in ok])) if ok else None},
        "position_error_final_m": {"mean": float(np.mean([r["est_error_final_m"] for r in ok])) if ok else None,
                                   "max": float(np.max([r["est_error_final_m"] for r in ok])) if ok else None},
        "stop_error_m": {"mean_abs": float(np.mean([abs(r["stop_error_m"]) for r in ok])) if ok else None},
        "lateral_miss_m": {"mean": float(np.mean([r["lateral_miss_m"] for r in ok])) if ok else None},
        "collisions": int(sum(r.get("collided", False) for r in ok)),
        "vlm_ms_mean": float(np.mean([r["vlm_ms_mean"] for r in ok])) if ok else None,
    }
    (out / "results.json").write_text(json.dumps({"summary": summary, "trials": results}, indent=1, default=float))
    print("[m3] " + json.dumps(summary, indent=1))
    app.close()


if __name__ == "__main__":
    main()
