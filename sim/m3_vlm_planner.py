"""
The robot drives one lap of Mcity with obstacles on the road.
The VLM looks through the front camera and decides: follow, bypass_left, bypass_right or stop.
This script turns that decision into a swerve path, and the NMPC drives the robot along it.
The depth camera and the road map can only brake or reject a swerve; they never steer.

    .venv/bin/python scripts/vlm_ground_server.py --adapter runs/planner_lora     # start the VLM first
    ~/isaac-env/bin/python sim/m3_vlm_planner.py
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

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--obstacles", type=int, default=5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--robot", default="magna_susp")
parser.add_argument("--vlm-url", default="http://127.0.0.1:8765")
parser.add_argument("--prompt", default="ft", help="ft (fine-tuned planner) | zeroshot (base model, long prompt)")
parser.add_argument("--no-gate", action="store_true", help="no depth safety gate: the VLM alone must avoid everything")
parser.add_argument("--vlm-range", type=float, default=30.0, help="how far the VLM is assumed to see (used without the gate) [m]")
parser.add_argument("--v-max", type=float, default=15.0)
parser.add_argument("--ay-curve", type=float, default=2.4)
parser.add_argument("--max-time", type=float, default=400.0)
parser.add_argument("--nmpc-dir", default=os.path.expanduser("~/Documents/carla_nmpc_project"))
parser.add_argument("--headless", action="store_true")
parser.add_argument("--record", action="store_true", help="save the viewport every 0.5 s and a GIF")
parser.add_argument("--out", default="runs/m3_planner")
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
from talk2drive.isaac import jetbot as J  # noqa: E402
from talk2drive.isaac.jetbot import IsaacJetbot  # noqa: E402
from talk2drive.isaac.props import CAM_RES, FRONT_BUMPER_X, HALF_WIDTH, footprint, make_front_camera, spawn_object  # noqa: E402
from talk2drive.isaac.simtools import ViewportRecorder, VlmClient, chase_view, road_height  # noqa: E402
from talk2drive.mcity import MCITY_XODR, mcity_route  # noqa: E402
from talk2drive.tadpole import TadpoleDriver, load_tadpole_stack  # noqa: E402
from talk2drive.vlm_planner import (FT_PROMPT, PLANNER_PROMPT, DepthGate, draw_route, locate_box, parse_plan,  # noqa: E402
                                    parse_plan_ft, shift_path)

OBSTACLE_TYPES = ["brown cardboard box", "blue barrel", "red box", "orange traffic cone", "green box", "purple barrel", "yellow ball"]
COMMIT_DIST = 18.0   # closer than this to an obstacle, the VLM may not switch sides unless the depth camera agrees [m]
ROBOT_REAR = 0.33    # robot back end, behind its reference point [m]
INF = float("inf")


# ---------------------------------------------------------------- setup
def place_obstacles(rng, route, lap_length, how_many):
    """Obstacles on straight parts of the lap, 90 m apart. Half get a cone 1.8 m to one side."""
    straight_spots = []
    for s in np.arange(70.0, lap_length - 70.0, 1.0):
        if np.abs(route.kappa_many(np.arange(s - 45, s + 15, 2.0))).max() < 0.02:
            straight_spots.append(s)
    rng.shuffle(straight_spots)

    chosen = []
    for s in straight_spots:
        if all(abs(s - c) >= 90 for c in chosen):
            chosen.append(float(s))
        if len(chosen) == how_many:
            break

    obstacles = []
    for i, s in enumerate(sorted(chosen)):
        centre = route.xy[int(s / route.ds)]
        heading = route.heading(s)
        left = np.array([-math.sin(heading), math.cos(heading)])
        sideways = float(rng.uniform(-0.6, 0.6))
        obstacles.append(dict(name=OBSTACLE_TYPES[i % len(OBSTACLE_TYPES)], s=s, xy=centre + sideways * left, role="blocker"))
        if rng.random() < 0.5:
            side = 1.0 if rng.random() < 0.5 else -1.0
            cone_s = s + float(rng.uniform(-1, 1))
            obstacles.append(dict(name="orange traffic cone", s=cone_s, xy=centre + (sideways + side * 1.8) * left, role="side"))
    return obstacles


class Run:
    """Everything that changes while the robot drives."""

    def __init__(self, route, lap_length, n_obstacles):
        self.path = route                  # the path the NMPC follows now (the route, or a swerve)
        self.stop_at = lap_length          # where on self.path to stop
        self.swerve = None                 # current swerve: {"s_blocker", "offset", "s_end"}
        self.vlm_stop_at = None            # VLM said "stop": stop here (distance along the route)
        self.waiting = None                # VLM answer waiting for its latency to pass
        self.latency = 1.0                 # VLM answer time [s]
        self.gate_braking = False
        self.gate_events = 0
        self.vetoes = {"offset_widened": 0, "off_pavement": 0, "flip_kept": 0, "side_corrected": 0, "no_side_stop": 0}
        self.looks = []
        self.smallest_gap = [INF] * n_obstacles
        self.collision = None
        self.log = {k: [] for k in ("t", "x", "y", "psi", "vx", "v_ref", "s_route", "cte_route", "delta_cmd",
                                    "v_cap", "gate", "on_shift", "clear_min")}


# ---------------------------------------------------------------- VLM
def read_plan(text):
    """VLM text -> {action, offset_m, blocker} (or None)."""
    if args.prompt == "ft":
        return parse_plan_ft(text)
    plan = parse_plan(text)   # zero-shot prompt: longer answer format
    if plan is None:
        return None
    blocker = None
    if plan["obstacles"]:
        name = json.loads(text[text.index("{"):text.rindex("}") + 1]).get("blocker", "") if "{" in text else ""
        with_box = [o for o in plan["obstacles"] if o["bbox_2d"]]
        same_name = [o for o in with_box if name and o["label"] == name]
        blocker = (same_name or with_box or [None])[0]
    return {"action": plan["action"], "offset_m": plan.get("lateral_offset_m", 0.0), "blocker": blocker}


def ask_vlm(run, vlm, prompt, camera, route, s_route, t, speed, out):
    """Take a camera image, paint the route on it, ask the VLM, and find the blocker on the map."""
    rgb = np.asarray(camera.get_rgba())[:, :, :3].astype(np.uint8).copy()
    depth = np.asarray(camera.get_depth(), dtype=float).copy()
    s_ahead = np.arange(s_route + 2.0, min(s_route + 45.0, route.length), 0.5)
    xy = route.xy[(s_ahead / route.ds).astype(int)]
    image = draw_route(Image.fromarray(rgb), camera, np.c_[xy, road_height(xy[:, 0], xy[:, 1]) + 0.02], depth)

    answer = vlm.ask(image, prompt=prompt)
    plan = read_plan(answer.get("raw", ""))
    run.latency = answer.get("ms", 1000.0) / 1000.0

    blocker = None   # (xy, radius)
    box = None
    if plan and plan["blocker"] and plan["blocker"].get("bbox_2d"):
        x1, y1, x2, y2 = plan["blocker"]["bbox_2d"]   # 0..1000 -> pixels
        box = (x1 * CAM_RES[0] / 1000, y1 * CAM_RES[1] / 1000, x2 * CAM_RES[0] / 1000, y2 * CAM_RES[1] / 1000)
        found = locate_box(camera, depth, box)
        if found is not None:
            blocker = (found[0], found[1])

    save_look_image(image, t, speed, plan, box, answer.get("ms", 0), out / "looks" / f"{len(run.looks):04d}.jpg")
    blocker_xy = None if blocker is None else [float(blocker[0][0]), float(blocker[0][1])]
    run.looks.append(dict(t=t, s_route=s_route, ms=answer.get("ms"), raw=answer.get("raw"), plan=plan, blocker_xy=blocker_xy))
    run.waiting = dict(t_ready=t + run.latency, plan=plan, blk=blocker)


def save_look_image(image, t, speed, plan, box, ms, file):
    if box is not None:
        ImageDraw.Draw(image).rectangle(box, outline=(255, 0, 255), width=4)
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, CAM_RES[0], 30], fill=(0, 0, 0))
    text = f"t={t:5.1f}s  v={speed:4.1f} m/s  "
    if plan:
        text += f"{plan['action']}  offset {plan['offset_m']:.1f} m  {(plan['blocker'] or {}).get('label', '')}"
    else:
        text += "UNPARSED"
    draw.text((8, 8), text + f"   ({ms:.0f} ms)", fill=(255, 255, 255))
    image.resize((960, 540)).save(file, quality=88)


# ---------------------------------------------------------------- swerves
def swerve_stays_on_road(path, length, offset):
    """True if the robot's outer edge stays on paved road along the swerve (kerbs are too low for depth)."""
    s = np.arange(0.0, min(path.length, length + 5.0), 0.5)
    points = path.xy[np.clip((s / path.ds).astype(int), 0, len(path.xy) - 1)]
    d = np.gradient(points, axis=0)
    left = np.c_[-d[:, 1], d[:, 0]] / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
    edge = points + np.sign(offset) * (HALF_WIDTH + 0.15) * left
    return bool(J.is_paved(edge[:, 0], edge[:, 1]).all())


def choose_swerve(route, s_route, sideways, s_blocker, radius, vlm_offset, speed, gate, vetoes):
    """The VLM's swerve, widened 0.4 m at a time if needed, else the other side.
    Returns (offset, path, length) of the first safe one, or None."""
    points = gate.obstacle_points() if gate is not None else None
    if gate is None:
        candidates = [vlm_offset]
    else:
        side = np.sign(vlm_offset)
        candidates = [side * (abs(vlm_offset) + k * 0.4) for k in range(4)]
        candidates += [-side * (max(abs(vlm_offset), 0.8) + k * 0.4) for k in range(4)]

    for offset in candidates:
        path, length = shift_path(route, s_route, sideways, s_blocker, radius, offset, max(speed, 4.0))
        if not swerve_stays_on_road(path, length, offset):
            vetoes["off_pavement"] += 1
        elif gate is None or gate.first_blocked(path, 0.0, min(path.length, length + 5.0), points) is None:
            return offset, path, length
    return None


def use_vlm_answer(run, plan, blocker, route, s_route, sideways, x, y, speed, gate, driver):
    """Turn a VLM answer into a swerve, a stop, or nothing."""
    if plan is None:
        return
    if plan["action"] == "follow":
        run.vlm_stop_at = None
        return
    if blocker is None:
        return

    xy, radius = blocker
    s_blocker = route.project(xy[0], xy[1], s_route, window=60.0)[0]
    stop_before_blocker = s_blocker - radius - 0.8 - FRONT_BUMPER_X
    if plan["action"] == "stop":
        run.vlm_stop_at = stop_before_blocker
        return

    run.vlm_stop_at = None
    vlm_offset = plan["offset_m"] * (1.0 if plan["action"] == "bypass_left" else -1.0)

    # already swerving round this blocker?
    same_blocker = run.swerve is not None and abs(run.swerve["s_blocker"] - s_blocker) < 3.0
    keep = same_blocker and abs(run.swerve["offset"] - vlm_offset) < 0.4
    flips_side = same_blocker and np.sign(vlm_offset) != np.sign(run.swerve["offset"])
    if flips_side and s_blocker - s_route < COMMIT_DIST and gate is not None:
        s_on_path = run.path.project(x, y, driver.s_hint, window=20.0)[0]
        if gate.first_blocked(run.path, s_on_path, s_blocker - s_route + 6.0) is None:
            keep = True   # our side is still clear: don't switch
            run.vetoes["flip_kept"] += 1
    if keep or s_blocker <= s_route:
        return

    chosen = choose_swerve(route, s_route, sideways, s_blocker, radius, vlm_offset, speed, gate, run.vetoes)
    if chosen is None:
        run.vlm_stop_at = stop_before_blocker   # no safe way past
        run.vetoes["no_side_stop"] += 1
        return
    offset, path, length = chosen
    if np.sign(offset) != np.sign(vlm_offset):
        run.vetoes["side_corrected"] += 1
    elif abs(offset - vlm_offset) > 1e-6:
        run.vetoes["offset_widened"] += 1
    run.swerve = dict(s_blocker=s_blocker, offset=offset, s_end=s_route + length)
    run.path = path
    run.stop_at = path.length - 1.0
    driver.set_path(path, 0.0)


# ---------------------------------------------------------------- speed
def speed_to_stop_within(distance, braking, reaction_time):
    """Highest speed that can still stop within `distance` (wait `reaction_time`, then brake)."""
    distance = max(0.0, distance)
    return braking * (math.sqrt(reaction_time * reaction_time + 2.0 * distance / braking) - reaction_time)


def speed_limit(run, step, s_route, x, y, speed, braking, gate, driver):
    limit = INF
    if run.vlm_stop_at is not None:
        limit = speed_to_stop_within(run.vlm_stop_at - s_route, braking, 0.4)

    if gate is not None and step % 3 == 0:   # depth gate, every 3rd step
        s_on_path = run.path.project(x, y, driver.s_hint, window=20.0)[0]
        look_ahead = speed * speed / (2 * braking) + speed * 0.6 + 12.0
        distance = gate.first_blocked(run.path, s_on_path, look_ahead)
        gate_limit = INF if distance is None else speed_to_stop_within(distance - FRONT_BUMPER_X - 1.5, braking, 0.5)
        if gate_limit < speed + 0.3:
            if not run.gate_braking:
                run.gate_events += 1
            run.gate_braking = True
            limit = min(limit, gate_limit)
        else:
            run.gate_braking = False
    elif run.gate_braking and run.log["v_cap"]:
        limit = min(limit, run.log["v_cap"][-1])   # keep the gate's last limit between checks

    if gate is None:   # no gate: only as fast as the VLM can see, answer and still stop
        limit = min(limit, speed_to_stop_within(args.vlm_range - FRONT_BUMPER_X - 1.0, braking, 2 * run.latency + 0.6))
    return limit


# ---------------------------------------------------------------- logging and drawing
def gap_to_each_obstacle(x, y, heading, obstacles):
    """Gap between the robot's rectangle and each obstacle [m]; negative = touching."""
    c, s = math.cos(heading), math.sin(heading)
    gaps = []
    for o in obstacles:
        dx, dy = o["xy"][0] - x, o["xy"][1] - y
        forward, left = c * dx + s * dy, -s * dx + c * dy
        nearest_forward = min(max(forward, -ROBOT_REAR), FRONT_BUMPER_X)
        nearest_left = min(max(left, -HALF_WIDTH), HALF_WIDTH)
        gaps.append(math.hypot(forward - nearest_forward, left - nearest_left) - footprint(o["name"]))
    return gaps


def draw_path(line_drawer, path, swerving):
    """Green line = route, magenta = swerve, in the 3D view."""
    line_drawer.clear_lines()
    points = path.xy[::6]
    heights = road_height(points[:, 0], points[:, 1]) + 0.08
    p3 = [(float(px), float(py), float(pz)) for (px, py), pz in zip(points, heights) if np.isfinite(pz)]
    colour = (1.0, 0.2, 1.0, 1) if swerving else (0.1, 0.9, 0.2, 1)
    if len(p3) > 1:
        line_drawer.draw_lines(p3[:-1], p3[1:], [colour] * (len(p3) - 1), [3] * (len(p3) - 1))


def save_results(run, obstacles, route, lap_length, s_route, TS, health, wall_start, out):
    log = run.log
    finished = s_route >= lap_length - 1.0
    s_log = np.asarray(log["s_route"])
    lap_time = float(np.asarray(log["t"])[np.argmax(s_log >= lap_length - 1.0)]) if finished else float("nan")

    def action(look):
        return look["plan"]["action"] if look["plan"] else "unparsed"

    actions = [action(look) for look in run.looks]
    per_obstacle = []
    for o, gap in zip(obstacles, run.smallest_gap):
        before = (s_log > o["s"] - 35) & (s_log < o["s"])
        per_obstacle.append(dict(
            name=o["name"], role=o["role"], s=o["s"], min_clearance_m=gap, hit=gap < 0.0,
            gate_braked=bool(np.any(np.asarray(log["gate"])[before] > 0)),
            reached=bool(s_log.max() > o["s"] + 2.0),
            vlm_actions=[action(look) for look in run.looks if 0 < o["s"] - look["s_route"] < 35]))
    blockers = [p for p in per_obstacle if p["role"] == "blocker"]

    metrics = dict(
        vlm=health["model"], prompt=args.prompt, gate=not args.no_gate, lap_completed=bool(finished), lap_time_s=lap_time,
        lap_length_m=lap_length, collisions=sum(p["hit"] for p in per_obstacle), first_collision=run.collision,
        blockers=len(blockers),
        blockers_reached=sum(p["reached"] for p in blockers),
        blockers_passed_by_vlm_alone=sum(p["reached"] and not p["hit"] and not p["gate_braked"] for p in blockers),
        depth_checks=run.vetoes,
        min_clearance_m=float(min(run.smallest_gap)) if run.smallest_gap else None,
        gate_brake_events=run.gate_events, gate_time_s=float(np.sum(log["gate"]) * TS),
        vlm_looks=len(run.looks), vlm_mean_ms=float(np.mean([look["ms"] for look in run.looks if look["ms"]])),
        vlm_actions={a: actions.count(a) for a in sorted(set(actions))},
        max_speed=float(max(log["vx"])), mean_speed=float(np.mean(log["vx"])),
        wall_time_s=time.perf_counter() - wall_start, obstacles=per_obstacle)

    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    (out / "looks.json").write_text(json.dumps(run.looks, indent=1, default=float))
    np.savez_compressed(out / "log.npz", **{k: np.asarray(v) for k, v in log.items()}, route_xy=route.xy,
                        obstacles=np.array([o["xy"] for o in obstacles]))
    print("[m3p] " + json.dumps({k: v for k, v in metrics.items() if k != "obstacles"}, indent=1, default=float))
    for p in per_obstacle:
        print(f"[m3p]   {p['name']:20s} {p['role']:7s} s={p['s']:5.0f}  min clearance {p['min_clearance_m']:5.2f} m  "
              f"gate braked {p['gate_braked']!s:5s}  VLM said {p['vlm_actions']}")


# ---------------------------------------------------------------- main
def main():
    out = Path(args.out) if os.path.isabs(args.out) else REPO / args.out
    (out / "looks").mkdir(parents=True, exist_ok=True)
    for f in (out / "looks").glob("*.jpg"):
        f.unlink()

    vlm = VlmClient(args.vlm_url)
    try:
        health = vlm.health()
    except OSError as e:
        sys.exit(f"[m3p] VLM service not reachable at {args.vlm_url} ({e}); start scripts/vlm_ground_server.py")
    print(f"[m3p] VLM: {health['model']}, prompt {args.prompt}; gate {'off' if args.no_gate else 'on'}")
    prompt = FT_PROMPT if args.prompt == "ft" else PLANNER_PROMPT.format(width=2 * HALF_WIDTH)

    # NMPC, route, robot
    nmpc = load_tadpole_stack(args.nmpc_dir, out / "solver_build", ay_curve=args.ay_curve)
    TS = float(nmpc.P.NMPC_TS)
    braking = float(nmpc.P.AX_DECEL_CMD)
    route, lap_length, _, _ = mcity_route(MCITY_XODR, "104,1", 1)
    start = (route.xy[0][0], route.xy[0][1], route.heading(0.0))
    bot = IsaacJetbot(env="mcity", robot=args.robot, physics_dt=TS / 2, start_pose=start, render=True)
    J.build_paved_mask(bot.world.stage, "mcity")
    bot.enable_torque_mode()
    bot.params.max_steer = float(np.deg2rad(nmpc.P.MAX_STEER_DEG))

    # obstacles
    obstacles = place_obstacles(np.random.default_rng(args.seed), route, lap_length, args.obstacles)
    for i, o in enumerate(obstacles):
        spawn_object(bot.world.stage, f"/World/Obstacles/obj{i}", o["name"], o["xy"], bot.ground_height(*o["xy"]))
    print("[m3p] obstacles: " + "; ".join(f"{o['name']} ({o['role']}) at s={o['s']:.0f} m" for o in obstacles))

    # camera, driver, safety gate
    camera = make_front_camera(bot)
    bot.reset_to(start, settle_steps=60)
    for _ in range(4):
        bot.step(render=True)
    loads = RoadLoadModel(parse_scenario("nominal"), nmpc.P, lap_length, dt=TS)
    driver = TadpoleDriver(nmpc, bot, loads, BodyForceApplier("/World/Jetbot/chassis"))
    driver.set_path(route, 0.0)
    gate = None if args.no_gate else DepthGate(camera, road_height, half_width=HALF_WIDTH)
    try:
        from isaacsim.util.debug_draw import _debug_draw

        line_drawer = _debug_draw.acquire_debug_draw_interface()
    except ImportError:
        line_drawer = None
    recorder = ViewportRecorder(out / "frames", 0.5) if args.record else None

    run = Run(route, lap_length, len(obstacles))
    log = run.log
    s_route = 0.0
    step = 0
    wall_start = time.perf_counter()
    while step * TS < args.max_time and app.is_running():
        t = step * TS

        # 1. where is the robot?
        position, _ = bot.robot.get_world_pose()
        x, y, heading = float(position[0]), float(position[1]), float(bot.get_state()[2])
        s_route, sideways = route.project(x, y, s_route, window=30.0)
        speed = log["vx"][-1] if log["vx"] else 0.0

        # 2. ask the VLM (next look as soon as the last answer is in)
        if run.waiting is None:
            ask_vlm(run, vlm, prompt, camera, route, s_route, t, speed, out)

        # 3. use the answer once its latency has passed
        if run.waiting is not None and t >= run.waiting["t_ready"]:
            plan, blocker = run.waiting["plan"], run.waiting["blk"]
            run.waiting = None
            use_vlm_answer(run, plan, blocker, route, s_route, sideways, x, y, speed, gate, driver)

        # swerve done and back on the route line: follow the route again
        if run.swerve is not None and s_route > run.swerve["s_end"] and abs(sideways) < 0.3:
            run.swerve, run.path, run.stop_at = None, route, lap_length
            driver.set_path(route, s_route)

        # 4. speed limit, then the NMPC drives one step
        limit = speed_limit(run, step, s_route, x, y, speed, braking, gate, driver)
        row = driver.tick(t, run.stop_at, v_cap=max(limit, 0.0) if math.isfinite(limit) else None, stop_lag=0.6)

        # 5. collisions and log
        gaps = gap_to_each_obstacle(x, y, heading, obstacles)
        run.smallest_gap = [min(a, b) for a, b in zip(run.smallest_gap, gaps)]
        if min(gaps) < 0.0 and run.collision is None:
            run.collision = (t, obstacles[int(np.argmin(gaps))]["name"])
            print(f"[m3p] COLLISION at t={t:.1f}s with {run.collision[1]}")
        values = (t, x, y, heading, row["vx"], row["v_ref"], s_route, sideways, row["delta_cmd"],
                  limit if math.isfinite(limit) else -1.0, float(run.gate_braking), float(run.swerve is not None), min(gaps))
        for key, value in zip(log, values):
            log[key].append(value)
        if step % 50 == 0:
            last = run.looks[-1]["plan"]["action"] if run.looks and run.looks[-1]["plan"] else "-"
            print(f"[m3p] t={t:6.1f}s s={s_route:6.1f}/{lap_length:.0f} v={row['vx']:5.2f} m/s  plan={last:12s} "
                  f"shift={'yes' if run.swerve else 'no':3s} gate={'BRAKE' if run.gate_braking else '-':5s} "
                  f"nearest obstacle {min(gaps):5.1f} m", flush=True)

        # 6. view and recording
        if line_drawer is not None:
            draw_path(line_drawer, run.path, run.swerve is not None)
        if not args.headless:
            chase_view(x, y, heading, bot.ground_height(x, y), 7.5, 3.8, 7, 0.5)
        if recorder is not None:
            recorder.capture(t)

        # 7. physics step; stop when done, stuck or rolled over
        driver.step(render=True)
        step += 1
        if s_route >= lap_length - 1.0 and abs(row["vx"]) < 0.05 and run.swerve is None:
            break
        n30 = int(30.0 / TS)
        if len(log["s_route"]) > n30 and log["s_route"][-1] - log["s_route"][-n30] < 0.5:
            print("[m3p] stalled: no progress for 30 s")
            break
        if abs(row["roll"]) > math.radians(45):
            print("[m3p] rolled over")
            break

    save_results(run, obstacles, route, lap_length, s_route, TS, health, wall_start, out)
    if recorder is not None:
        recorder.save_gif(app, out / "run.gif", 125)
    app.close()


if __name__ == "__main__":
    main()
