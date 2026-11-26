"""Qwen3-VL drives the Mcity lap around obstacles as behaviour + local planner; the CARLA NMPC tracks.

Start the VLM service first (VLM venv), then run this in the Isaac venv:

    .venv/bin/python scripts/vlm_ground_server.py --adapter runs/planner_lora &     # fine-tuned 4B
    ~/isaac-env/bin/python sim/m3_vlm_planner.py                 # window open, 5 obstacles on the lap
    ~/isaac-env/bin/python sim/m3_vlm_planner.py --no-gate       # VLM only, no depth safety gate
    ~/isaac-env/bin/python sim/m3_vlm_planner.py --prompt zeroshot --vlm-url http://127.0.0.1:8766   # base model

Obstacles (boxes, barrels, cones, balls) are placed on straight stretches of the lap, on or near the
lane, some with a second object beside them so only one side is free. The robot is never told where
they are. Every look, the VLM gets the front camera image with the route painted on the road (hidden
behind anything standing on it) and answers what to do about the nearest object in the way:

    {"action": follow | bypass_left | bypass_right | stop, "offset_m": ..., "blocker": {label, bbox_2d}}

The blocker's box and the depth image put it on the map; a quintic lateral shift to ``offset_m`` on the
chosen side, held past it and back onto the route, is the path the NMPC tracks. The VLM's real latency
is honoured: the plan from an image taken at t is applied at t + latency; until then the robot follows
the previous plan.

The depth gate (talk2drive.vlm_planner.DepthGate) can only brake. With it on, the robot may drive at
full speed, because the gate can always stop it in time; whether the VLM steers round obstacles before
the gate has to brake is what the run measures. With --no-gate the speed is capped so the VLM alone can
react: see the obstacle, answer, and still have braking distance left.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--obstacles", type=int, default=5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--robot", default="magna_susp")
ap.add_argument("--vlm-url", default="http://127.0.0.1:8765")
ap.add_argument("--prompt", default="ft", help="ft (fine-tuned planner) | zeroshot (base model, long prompt)")
ap.add_argument("--no-gate", action="store_true", help="no depth safety gate: the VLM alone must avoid everything")
ap.add_argument("--vlm-range", type=float, default=30.0, help="distance at which the VLM is assumed to see an obstacle (speed cap without the gate) [m]")
ap.add_argument("--v-max", type=float, default=15.0)
ap.add_argument("--ay-curve", type=float, default=2.4)
ap.add_argument("--max-time", type=float, default=400.0)
ap.add_argument("--nmpc-dir", default=os.path.expanduser("~/Documents/carla_nmpc_project"))
ap.add_argument("--headless", action="store_true")
ap.add_argument("--record", action="store_true", help="save the viewport every 0.5 s and a GIF")
ap.add_argument("--out", default="runs/m3_planner")
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
from talk2drive.isaac import jetbot as J  # noqa: E402
from talk2drive.isaac.jetbot import IsaacJetbot, drive_surface_height  # noqa: E402
from talk2drive.isaac.props import CAM_RES, FRONT_BUMPER_X, HALF_WIDTH, footprint, make_front_camera, spawn_object  # noqa: E402
from talk2drive.mcity import MCITY_XODR, mcity_route  # noqa: E402
from talk2drive.tadpole import TadpoleDriver, load_tadpole_stack  # noqa: E402
from talk2drive.vlm_planner import (FT_PROMPT, PLANNER_PROMPT, DepthGate, draw_route, locate_box, parse_plan,  # noqa: E402
                                    parse_plan_ft, shift_path)

OBSTACLE_TYPES = ["brown cardboard box", "blue barrel", "red box", "orange traffic cone", "green box", "purple barrel", "yellow ball"]


def road_h(x, y):
    h = drive_surface_height(np.atleast_1d(x), np.atleast_1d(y))
    return h if np.ndim(x) else float(h[0])


def place_obstacles(rng, route, lap_length, n):
    """Obstacles on or near the lane on straight stretches, >= 90 m apart; half get a side blocker."""
    s_all = np.arange(70.0, lap_length - 70.0, 1.0)
    straight = np.array([np.abs(route.kappa_many(np.arange(s - 45, s + 15, 2.0))).max() < 0.02 for s in s_all])
    cand = list(s_all[straight])
    rng.shuffle(cand)
    chosen = []
    for s in cand:
        if all(abs(s - c) >= 90 for c in chosen):
            chosen.append(float(s))
        if len(chosen) == n:
            break
    objs = []
    for i, s in enumerate(sorted(chosen)):
        c = route.xy[int(s / route.ds)]
        hd = route.heading(s)
        nrm = np.array([-math.sin(hd), math.cos(hd)])
        name = OBSTACLE_TYPES[i % len(OBSTACLE_TYPES)]
        lat = float(rng.uniform(-0.6, 0.6))
        objs.append(dict(name=name, s=s, xy=c + lat * nrm, role="blocker"))
        if rng.random() < 0.5:  # a cone ~1.8 m to one side: only the other side is free
            side = 1.0 if rng.random() < 0.5 else -1.0
            objs.append(dict(name="orange traffic cone", s=s + float(rng.uniform(-1, 1)), xy=c + (lat + side * 1.8) * nrm, role="side"))
    return objs


def ask_vlm(img: Image.Image, prompt: str) -> dict:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    req = urllib.request.Request(args.vlm_url + "/ground", method="POST", headers={"Content-Type": "application/json"},
                                 data=json.dumps({"image_png_b64": base64.b64encode(buf.getvalue()).decode(), "query": "",
                                                  "prompt": prompt}).encode())
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def read_plan(raw: str) -> dict | None:
    """Both prompt styles -> {action, offset_m, blocker: {label, bbox_2d} | None}."""
    if args.prompt == "ft":
        return parse_plan_ft(raw)
    p = parse_plan(raw)
    if p is None:
        return None
    blk = None
    if p["obstacles"]:
        name = json.loads(raw[raw.index("{"):raw.rindex("}") + 1]).get("blocker", "") if "{" in raw else ""
        cands = [o for o in p["obstacles"] if o["bbox_2d"]]
        match = [o for o in cands if name and o["label"] == name]
        blk = (match or cands or [None])[0]
    return {"action": p["action"], "offset_m": p.get("lateral_offset_m", 0.0), "blocker": blk}


def robot_clearance(x, y, psi, objs) -> list[float]:
    """Gap between the robot's plan-view rectangle and each object's footprint circle [m]."""
    c, s = math.cos(psi), math.sin(psi)
    out = []
    for o in objs:
        dx, dy = o["xy"][0] - x, o["xy"][1] - y
        lx, ly = c * dx + s * dy, -s * dx + c * dy  # object centre in the robot frame
        qx = min(max(lx, -0.33), FRONT_BUMPER_X)
        qy = min(max(ly, -HALF_WIDTH), HALF_WIDTH)
        out.append(math.hypot(lx - qx, ly - qy) - footprint(o["name"]))
    return out


def stop_speed(d: float, a: float, lag: float) -> float:
    """Highest speed from which the robot still stops within ``d`` braking at ``a`` after ``lag`` s."""
    d = max(0.0, d)
    return a * (math.sqrt(lag * lag + 2.0 * d / a) - lag)


def paved_ok(path, span: float, side: float) -> bool:
    """The swerve keeps the robot's outer edge (on the ``side`` it moves to) on pavement: kerbs are too low for depth."""
    s = np.arange(0.0, min(path.length, span + 5.0), 0.5)
    i = np.clip((s / path.ds).astype(int), 0, len(path.xy) - 1)
    P = path.xy[i]
    d = np.gradient(P, axis=0)
    n = np.c_[-d[:, 1], d[:, 0]] / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]
    E = P + np.sign(side) * (HALF_WIDTH + 0.15) * n
    return bool(J.is_paved(E[:, 0], E[:, 1]).all())


COMMIT_DIST = 18.0  # within this of the blocker, a side flip needs the depth camera to agree [m]


def main() -> None:
    out = (REPO / args.out) if not os.path.isabs(args.out) else Path(args.out)
    (out / "looks").mkdir(parents=True, exist_ok=True)
    for f in (out / "looks").glob("*.jpg"):
        f.unlink()
    try:
        health = json.loads(urllib.request.urlopen(args.vlm_url + "/health", timeout=5).read())
    except OSError as e:
        sys.exit(f"[m3p] VLM service not reachable at {args.vlm_url} ({e}); start scripts/vlm_ground_server.py")
    print(f"[m3p] VLM: {health['model']}, prompt {args.prompt}; gate {'off' if args.no_gate else 'on'}")
    prompt = FT_PROMPT if args.prompt == "ft" else PLANNER_PROMPT.format(width=2 * HALF_WIDTH)

    stack = load_tadpole_stack(args.nmpc_dir, out / "solver_build", ay_curve=args.ay_curve)
    P = stack.P
    TS = float(P.NMPC_TS)
    route, lap_length, _, _ = mcity_route(MCITY_XODR, "104,1", 1)
    x0, y0 = route.xy[0]
    psi0 = route.heading(0.0)
    bot = IsaacJetbot(env="mcity", robot=args.robot, physics_dt=TS / 2, start_pose=(x0, y0, psi0), render=True)
    J.build_paved_mask(bot.world.stage, "mcity")
    bot.enable_torque_mode()
    bot.params.max_steer = float(np.deg2rad(P.MAX_STEER_DEG))
    rng = np.random.default_rng(args.seed)
    objs = place_obstacles(rng, route, lap_length, args.obstacles)
    for i, o in enumerate(objs):
        spawn_object(bot.world.stage, f"/World/Obstacles/obj{i}", o["name"], o["xy"], bot.ground_height(*o["xy"]))
    print("[m3p] obstacles: " + "; ".join(f"{o['name']} ({o['role']}) at s={o['s']:.0f} m" for o in objs))
    cam = make_front_camera(bot)
    bot.reset_to((x0, y0, psi0), settle_steps=60)
    for _ in range(4):
        bot.step(render=True)
    loads = RoadLoadModel(parse_scenario("nominal"), P, lap_length, dt=TS)
    driver = TadpoleDriver(stack, bot, loads, BodyForceApplier("/World/Jetbot/chassis"))
    driver.set_path(route, 0.0)
    gate = None if args.no_gate else DepthGate(cam, road_h, half_width=HALF_WIDTH)
    vetoes = {"offset_widened": 0, "off_pavement": 0, "flip_kept": 0, "side_corrected": 0, "no_side_stop": 0}  # depth checks on the VLM's swerves
    from isaacsim.core.utils.viewports import set_camera_view

    try:
        from isaacsim.util.debug_draw import _debug_draw

        draw = _debug_draw.acquire_debug_draw_interface()
    except ImportError:
        draw = None
    if args.record:
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

        vp = get_active_viewport()
        (out / "frames").mkdir(exist_ok=True)
        for f in (out / "frames").glob("*.png"):
            f.unlink()

    a_brake = float(P.AX_DECEL_CMD)
    keys = ("t", "x", "y", "psi", "vx", "v_ref", "s_route", "cte_route", "delta_cmd", "v_cap", "gate", "on_shift", "clear_min")
    log = {k: [] for k in keys}
    looks, pending = [], None
    cur_path, s_stop = route, lap_length
    shift = None  # active manoeuvre: dict(s_blocker, offset, s_end)
    s_route, last_latency = 0.0, 1.0
    vlm_stop = None  # (s_route where the robot must stop)
    gate_events, gate_active = 0, False
    min_clear = [float("inf")] * len(objs)
    k, n_frame, hit = 0, 0, None
    wall0 = time.perf_counter()
    while k * TS < args.max_time and app.is_running():
        t = k * TS
        pos, _ = bot.robot.get_world_pose()
        st = bot.get_state()
        X, Y, psi = float(pos[0]), float(pos[1]), float(st[2])
        s_route, cte_route = route.project(X, Y, s_route, window=30.0)
        vx_now = log["vx"][-1] if log["vx"] else 0.0

        # ---- VLM look (back to back: the next image is taken when the previous answer arrives)
        if pending is None:
            rgb = np.asarray(cam.get_rgba())[:, :, :3].astype(np.uint8).copy()
            depth = np.asarray(cam.get_depth(), dtype=float).copy()
            ss = np.arange(s_route + 2.0, min(s_route + 45.0, route.length), 0.5)
            R = route.xy[(ss / route.ds).astype(int)]
            im = draw_route(Image.fromarray(rgb), cam, np.c_[R, road_h(R[:, 0], R[:, 1]) + 0.02], depth)
            ans = ask_vlm(im, prompt)
            plan = read_plan(ans.get("raw", ""))
            last_latency = ans.get("ms", 1000.0) / 1000.0
            blk_xy = None
            if plan and plan["blocker"] and plan["blocker"].get("bbox_2d"):
                x1, y1, x2, y2 = plan["blocker"]["bbox_2d"]
                box = (x1 * CAM_RES[0] / 1000, y1 * CAM_RES[1] / 1000, x2 * CAM_RES[0] / 1000, y2 * CAM_RES[1] / 1000)
                loc = locate_box(cam, depth, box)
                if loc is not None:
                    blk_xy = (loc[0], loc[1])
                ImageDraw.Draw(im).rectangle(box, outline=(255, 0, 255), width=4)
            dr = ImageDraw.Draw(im)
            dr.rectangle([0, 0, CAM_RES[0], 30], fill=(0, 0, 0))
            txt = f"t={t:5.1f}s  v={vx_now:4.1f} m/s  " + (f"{plan['action']}  offset {plan['offset_m']:.1f} m  "
                                                          f"{(plan['blocker'] or {}).get('label', '')}" if plan else "UNPARSED")
            dr.text((8, 8), txt + f"   ({ans.get('ms', 0):.0f} ms)", fill=(255, 255, 255))
            im.resize((960, 540)).save(out / "looks" / f"{len(looks):04d}.jpg", quality=88)
            looks.append(dict(t=t, s_route=s_route, ms=ans.get("ms"), raw=ans.get("raw"), plan=plan,
                              blocker_xy=None if blk_xy is None else [float(blk_xy[0][0]), float(blk_xy[0][1])]))
            pending = dict(t_ready=t + last_latency, plan=plan, blk=blk_xy)

        # ---- apply the plan once its latency has passed
        if pending is not None and t >= pending["t_ready"]:
            plan, blk = pending["plan"], pending["blk"]
            pending = None
            if plan is not None and plan["action"] != "follow" and blk is not None:
                s_b = route.project(blk[0][0], blk[0][1], s_route, window=60.0)[0]
                if plan["action"] == "stop":
                    vlm_stop = s_b - blk[1] - 0.8 - FRONT_BUMPER_X
                else:
                    vlm_stop = None
                    off = plan["offset_m"] * (1.0 if plan["action"] == "bypass_left" else -1.0)
                    same_blk = shift is not None and abs(shift["s_blocker"] - s_b) < 3.0
                    same = same_blk and abs(shift["offset"] - off) < 0.4
                    flip = same_blk and np.sign(off) != np.sign(shift["offset"])
                    if flip and s_b - s_route < COMMIT_DIST and gate is not None:
                        # committed: keep the side we are on unless the depth camera says it is blocked
                        s_c = cur_path.project(X, Y, driver.s_hint, window=20.0)[0]
                        if gate.first_blocked(cur_path, s_c, s_b - s_route + 6.0) is None:
                            same = True
                            vetoes["flip_kept"] += 1
                    if not same and s_b > s_route:
                        pts = gate.obstacle_points() if gate is not None else None
                        chosen = None
                        # the VLM's side first, widened in 0.4 m steps if the depth camera says it is too tight; then the other side
                        cands = [np.sign(off) * (abs(off) + k * 0.4) for k in range(4)]
                        cands += [-np.sign(off) * (max(abs(off), 0.8) + k * 0.4) for k in range(4)]
                        for o in (cands if gate is not None else [off]):
                            lp, span = shift_path(route, s_route, cte_route, s_b, blk[1], o, max(vx_now, 4.0))
                            if not paved_ok(lp, span, o):
                                vetoes["off_pavement"] += 1
                                continue
                            if gate is None or gate.first_blocked(lp, 0.0, min(lp.length, span + 5.0), pts) is None:
                                chosen = (o, lp, span)
                                break
                        if chosen is None:  # neither side passes: stop short of the blocker
                            vlm_stop = s_b - blk[1] - 0.8 - FRONT_BUMPER_X
                            vetoes["no_side_stop"] += 1
                        else:
                            if np.sign(chosen[0]) != np.sign(off):
                                vetoes["side_corrected"] += 1
                            elif abs(chosen[0] - off) > 1e-6:
                                vetoes["offset_widened"] += 1
                            o, lp, span = chosen
                            shift = dict(s_blocker=s_b, offset=o, s_end=s_route + span)
                            cur_path, s_stop = lp, lp.length - 1.0
                            driver.set_path(lp, 0.0)
            elif plan is not None and plan["action"] == "follow":
                vlm_stop = None
        if shift is not None and s_route > shift["s_end"] and abs(cte_route) < 0.3:
            shift, cur_path, s_stop = None, route, lap_length  # manoeuvre done, back on the route
            driver.set_path(route, s_route)

        # ---- speed caps: VLM stop, gate (brake only), or (no gate) what the VLM alone can react to
        cap = float("inf")
        if vlm_stop is not None:
            cap = stop_speed(vlm_stop - s_route, a_brake, 0.4)
        if gate is not None and k % 3 == 0:
            s_here = cur_path.project(X, Y, driver.s_hint, window=20.0)[0]
            look = vx_now * vx_now / (2 * a_brake) + vx_now * 0.6 + 12.0
            gate_d = gate.first_blocked(cur_path, s_here, look)
            cap_g = stop_speed(gate_d - FRONT_BUMPER_X - 1.5, a_brake, 0.5) if gate_d is not None else float("inf")
            if cap_g < vx_now + 0.3:
                if not gate_active:
                    gate_events += 1
                gate_active = True
                cap = min(cap, cap_g)
            else:
                gate_active = False
        elif gate_active and log["v_cap"]:
            cap = min(cap, log["v_cap"][-1])
        if gate is None:
            cap = min(cap, stop_speed(args.vlm_range - FRONT_BUMPER_X - 1.0, a_brake, 2 * last_latency + 0.6))

        row = driver.tick(t, s_stop, v_cap=None if not math.isfinite(cap) else max(cap, 0.0), stop_lag=0.6)
        clears = robot_clearance(X, Y, psi, objs)
        min_clear = [min(a, b) for a, b in zip(min_clear, clears)]
        if min(clears) < 0.0 and hit is None:
            hit = (t, objs[int(np.argmin(clears))]["name"])
            print(f"[m3p] COLLISION at t={t:.1f}s with {hit[1]}")
        for key, v in zip(keys, (t, X, Y, psi, row["vx"], row["v_ref"], s_route, cte_route, row["delta_cmd"],
                                 cap if math.isfinite(cap) else -1.0, float(gate_active), float(shift is not None), min(clears))):
            log[key].append(v)
        if k % 50 == 0:
            last = looks[-1]["plan"]["action"] if looks and looks[-1]["plan"] else "-"
            print(f"[m3p] t={t:6.1f}s s={s_route:6.1f}/{lap_length:.0f} v={row['vx']:5.2f} m/s  plan={last:12s} "
                  f"shift={'yes' if shift else 'no':3s} gate={'BRAKE' if gate_active else '-':5s} nearest obstacle {min(clears):5.1f} m",
                  flush=True)

        if draw is not None:
            draw.clear_lines()
            pts = cur_path.xy[::6]
            zz = road_h(pts[:, 0], pts[:, 1]) + 0.08
            P3 = [(float(a), float(b), float(c)) for (a, b), c in zip(pts, zz) if np.isfinite(c)]
            col = (1.0, 0.2, 1.0, 1) if shift else (0.1, 0.9, 0.2, 1)
            if len(P3) > 1:
                draw.draw_lines(P3[:-1], P3[1:], [col] * (len(P3) - 1), [3] * (len(P3) - 1))
        if not args.headless:
            z = bot.ground_height(X, Y)
            set_camera_view(eye=[X - 7.5 * math.cos(psi), Y - 7.5 * math.sin(psi), z + 3.8],
                            target=[X + 7 * math.cos(psi), Y + 7 * math.sin(psi), z + 0.5])
        if args.record and t + 1e-9 >= n_frame * 0.5:
            capture_viewport_to_file(vp, file_path=str(out / "frames" / f"{n_frame:04d}.png"))
            n_frame += 1
        driver.step(render=True)
        k += 1
        if s_route >= lap_length - 1.0 and abs(row["vx"]) < 0.05 and shift is None:
            break
        n30 = int(30.0 / TS)
        if len(log["s_route"]) > n30 and log["s_route"][-1] - log["s_route"][-n30] < 0.5:
            print("[m3p] stalled: no progress for 30 s")
            break
        if abs(row["roll"]) > math.radians(45):
            print("[m3p] rolled over")
            break

    done = s_route >= lap_length - 1.0
    S = np.asarray(log["s_route"])
    lap_t = float(np.asarray(log["t"])[np.argmax(S >= lap_length - 1.0)]) if done else float("nan")
    acts = [lk["plan"]["action"] if lk["plan"] else "unparsed" for lk in looks]
    per_obj = []
    for o, c in zip(objs, min_clear):
        near = [lk for lk in looks if 0 < o["s"] - lk["s_route"] < 35]
        before = (S > o["s"] - 35) & (S < o["s"])
        per_obj.append(dict(name=o["name"], role=o["role"], s=o["s"], min_clearance_m=c, hit=c < 0.0,
                            gate_braked=bool(np.any(np.asarray(log["gate"])[before] > 0)),
                            reached=bool(S.max() > o["s"] + 2.0),
                            vlm_actions=[lk["plan"]["action"] if lk["plan"] else "unparsed" for lk in near]))
    blockers = [p for p in per_obj if p["role"] == "blocker"]
    metrics = dict(
        vlm=health["model"], prompt=args.prompt, gate=not args.no_gate, lap_completed=bool(done), lap_time_s=lap_t,
        lap_length_m=lap_length, collisions=sum(p["hit"] for p in per_obj), first_collision=hit,
        blockers=len(blockers),
        blockers_reached=sum(p["reached"] for p in blockers),
        blockers_passed_by_vlm_alone=sum(p["reached"] and not p["hit"] and not p["gate_braked"] for p in blockers),
        depth_checks=vetoes,
        min_clearance_m=float(min(min_clear)) if min_clear else None,
        gate_brake_events=gate_events, gate_time_s=float(np.sum(log["gate"]) * TS),
        vlm_looks=len(looks), vlm_mean_ms=float(np.mean([lk["ms"] for lk in looks if lk["ms"]])),
        vlm_actions={a: acts.count(a) for a in sorted(set(acts))},
        max_speed=float(max(log["vx"])), mean_speed=float(np.mean(log["vx"])),
        wall_time_s=time.perf_counter() - wall0, obstacles=per_obj)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    (out / "looks.json").write_text(json.dumps(looks, indent=1, default=float))
    np.savez_compressed(out / "log.npz", **{k_: np.asarray(v) for k_, v in log.items()}, route_xy=route.xy,
                        obstacles=np.array([o["xy"] for o in objs]))
    print("[m3p] " + json.dumps({k_: v for k_, v in metrics.items() if k_ != "obstacles"}, indent=1, default=float))
    for p in per_obj:
        print(f"[m3p]   {p['name']:20s} {p['role']:7s} s={p['s']:5.0f}  min clearance {p['min_clearance_m']:5.2f} m  "
              f"gate braked {p['gate_braked']!s:5s}  VLM said {p['vlm_actions']}")
    if args.record:
        for _ in range(10):
            app.update()
        files = sorted((out / "frames").glob("*.png"))
        if files:
            imgs = [Image.open(f).convert("RGB").resize((640, 360)) for f in files]
            imgs[0].save(out / "run.gif", save_all=True, append_images=imgs[1:], duration=125, loop=0)
    app.close()


if __name__ == "__main__":
    main()
