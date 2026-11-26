"""Qwen3-VL as behaviour + local planner; the NMPC tracks what it draws; a depth gate can only brake.

Each look:
  camera image with the route ahead painted on the road (green)
  -> VLM: {"action": follow | bypass_left | bypass_right | stop, "waypoints": [[u, v], ...] (0..1000,
     image coordinates of points on the road surface, near to far), "obstacles": [...], "reason": "..."}
  -> each waypoint's camera ray intersected with the road surface -> world waypoints
  -> local path: robot -> waypoints -> back onto the route, handed to the NMPC

The gate (DepthGate) runs every few ticks on the depth image, independent of the VLM: points standing
above the road inside the robot's corridor along the current path, within stopping distance, make it
cap the speed so the robot stops short of them. It never steers -- it is the check that keeps a wrong
VLM answer from becoming a collision, and how often it acts measures how good the VLM's plans are.
"""

from __future__ import annotations

import json
import math
import re

import numpy as np

PLANNER_PROMPT = """You are the behaviour planner of a delivery robot {width:.1f} m wide. It drives along the thick green line painted on the road (its route); the line disappears behind anything standing on it.
Step 1 - list every object standing on the road surface: boxes, cones, barrels, balls, debris, people, vehicles. Ignore lane markings, signs, poles and anything off the paved road. Give each a bounding box [x1, y1, x2, y2] normalised to 0-1000.
Step 2 - find the BLOCKER: the nearest object on the green line or within 1.5 m of it (where the green line is hidden, something is standing on it). If there is none, the action is "follow".
Step 3 - if there is a blocker, look at each side of it: "left_side" is "free" if there is open paved road to the LEFT of the blocker (from the robot's point of view) with no other object, otherwise "blocked"; same for "right_side".
Step 4 - action: "bypass_left" if left_side is free, else "bypass_right" if right_side is free, else "stop". "lateral_offset_m": how far from the green line to drive while passing (enough to keep 1 m between the robot and the blocker; 0 for follow).
Answer with JSON only:
{{"obstacles": [{{"label": "...", "bbox_2d": [x1, y1, x2, y2]}}], "blocker": "label or none", "left_side": "free|blocked", "right_side": "free|blocked", "action": "...", "lateral_offset_m": 0.0, "reason": "..."}}"""


# Prompt of the fine-tuned planner (scripts/finetune_planner.py); it answers
# {"action": ..., "offset_m": ..., "blocker": {"label": ..., "bbox_2d": [x1, y1, x2, y2]} | null}
FT_PROMPT = ("You drive a delivery robot along the green route painted on the road. Decide the next manoeuvre "
             "for the nearest object in the way. JSON only.")


def parse_plan_ft(text: str) -> dict | None:
    """The fine-tuned planner's answer; None if unusable."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    a = str(d.get("action", "")).lower()
    if a not in ("follow", "bypass_left", "bypass_right", "stop"):
        return None
    try:
        off = abs(float(d.get("offset_m", 0.0) or 0.0))
    except (TypeError, ValueError):
        off = 0.0
    b = d.get("blocker") if isinstance(d.get("blocker"), dict) else None
    if b is not None:
        bb = b.get("bbox_2d")
        ok = isinstance(bb, (list, tuple)) and len(bb) == 4 and all(isinstance(v, (int, float)) for v in bb)
        b = {"label": str(b.get("label", "")), "bbox_2d": [float(v) for v in bb] if ok else None}
    return {"action": a, "offset_m": off, "blocker": b}


def parse_plan(text: str) -> dict | None:
    """The planner's JSON answer (code fences tolerated); None if unusable."""
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    action = str(d.get("action", "")).strip().lower()
    if action not in ("follow", "bypass_left", "bypass_right", "stop"):
        return None
    wps = []
    for p in d.get("waypoints") or []:
        if isinstance(p, (list, tuple)) and len(p) >= 2 and all(isinstance(v, (int, float)) for v in p[:2]):
            wps.append((float(p[0]), float(p[1])))
    obstacles = []
    for o in d.get("obstacles") or []:
        if isinstance(o, dict):
            bb = o.get("bbox_2d")
            ok = isinstance(bb, (list, tuple)) and len(bb) == 4 and all(isinstance(v, (int, float)) for v in bb)
            obstacles.append({"label": str(o.get("label", "")), "bbox_2d": [float(v) for v in bb] if ok else None})
        else:
            obstacles.append({"label": str(o), "bbox_2d": None})
    try:
        offset = abs(float(d.get("lateral_offset_m", 0.0) or 0.0))
    except (TypeError, ValueError):
        offset = 0.0
    return {"action": action, "waypoints": wps, "obstacles": obstacles, "lateral_offset_m": offset,
            "reason": str(d.get("reason", ""))}


# ---------------------------------------------------------------- camera geometry
def camera_rays(cam, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera position and unit ray directions (world) through pixels ``uv`` (n, 2)."""
    origin = np.asarray(cam.get_world_pose()[0], dtype=float)
    far = np.asarray(cam.get_world_points_from_image_coords(np.asarray(uv, dtype=float), np.ones(len(uv))), dtype=float)
    d = far - origin
    return origin, d / np.linalg.norm(d, axis=1, keepdims=True)


def pixels_to_road(cam, uv: np.ndarray, road_height, max_range: float = 60.0) -> np.ndarray:
    """Where the camera rays through ``uv`` meet the road (n, 2); NaN rows if they miss within range."""
    o, d = camera_rays(cam, uv)
    out = np.full((len(uv), 2), np.nan)
    for i, di in enumerate(d):
        if di[2] >= -1e-3:
            continue  # at or above the horizon
        z0 = road_height(o[0], o[1])
        t = (o[2] - (z0 if np.isfinite(z0) else o[2] - 1.45)) / -di[2]
        for _ in range(4):  # refine against the local road height (grades)
            p = o + t * di
            zg = road_height(p[0], p[1])
            if not np.isfinite(zg):
                break
            t = (o[2] - zg) / -di[2]
        if 0 < t * math.hypot(di[0], di[1]) <= max_range:
            p = o + t * di
            out[i] = p[:2]
    return out


def project_to_image(cam, pts_xyz: np.ndarray, width: int, height: int) -> np.ndarray:
    """Pixel coordinates of world points in front of the camera (rows NaN otherwise)."""
    o = np.asarray(cam.get_world_pose()[0], dtype=float)
    fwd = camera_rays(cam, np.array([[width / 2, height / 2]]))[1][0]
    uv = np.asarray(cam.get_image_coords_from_world_points(pts_xyz), dtype=float)
    ahead = (pts_xyz - o) @ fwd > 1.0
    uv[~ahead] = np.nan
    return uv


def draw_route(image, cam, route_xyz: np.ndarray, depth: np.ndarray, color=(0, 230, 0), width: int = 7):
    """Paint the route on the road in ``image`` (PIL), hidden wherever something stands in front of it:
    a route point is drawn only if the depth image at its pixel is not nearer than the point itself."""
    from PIL import ImageDraw

    H, W = depth.shape
    uv = project_to_image(cam, route_xyz, W, H)
    o = np.asarray(cam.get_world_pose()[0], dtype=float)
    fwd = camera_rays(cam, np.array([[W / 2, H / 2]]))[1][0]
    z = (route_xyz - o) @ fwd  # distance to the image plane
    vis = np.zeros(len(uv), dtype=bool)
    for i, (u, v) in enumerate(uv):
        if np.isfinite(u) and 0 <= u < W and 0 <= v < H:
            d = depth[int(v), int(u)]
            vis[i] = (not np.isfinite(d)) or d >= z[i] - 0.4
    dr = ImageDraw.Draw(image)
    for i in range(len(uv) - 1):
        if vis[i] and vis[i + 1]:
            dr.line([tuple(uv[i]), tuple(uv[i + 1])], fill=color, width=width)
    return image


# ---------------------------------------------------------------- paths
def local_path(route, s_route: float, pose, wps_world: np.ndarray, rejoin: float = 8.0, tail: float = 60.0):
    """Robot pose -> VLM waypoints -> back onto ``route`` ``rejoin`` m past the last waypoint, then
    ``tail`` m along the route. Waypoints behind the robot or going backwards are dropped."""
    from talk2drive.opendrive import RoutePath

    x, y, psi = pose
    h = np.array([math.cos(psi), math.sin(psi)])
    pts = [np.array([x, y]), np.array([x, y]) + 1.5 * h]
    s_last = s_route
    for w in wps_world:
        if not np.all(np.isfinite(w)):
            continue
        if (w - pts[-1]) @ h < 1.0 and len(pts) > 2:
            continue
        if (w - np.array([x, y])) @ h < 2.5:
            continue  # too close to steer to
        s_w = route.project(w[0], w[1], s_last, window=40.0)[0]
        if s_w < s_last - 0.5:
            continue
        pts.append(w)
        s_last = s_w
    s_join = s_last + rejoin
    i0, i1 = int(s_join / route.ds), int(min(route.length, s_join + tail) / route.ds)
    P = np.r_[np.array(pts), route.xy[i0:i1]]
    # Catmull-Rom through the points, then the uniform resampling / smoothing of RoutePath
    Q = [P[0]]
    for k in range(len(P) - 1):
        p0, p1, p2, p3 = P[max(k - 1, 0)], P[k], P[k + 1], P[min(k + 2, len(P) - 1)]
        n = max(2, int(np.linalg.norm(p2 - p1) / 0.2))
        for tt in np.linspace(0, 1, n)[1:]:
            Q.append(0.5 * ((2 * p1) + (-p0 + p2) * tt + (2 * p0 - 5 * p1 + 4 * p2 - p3) * tt**2 + (-p0 + 3 * p1 - 3 * p2 + p3) * tt**3))
    return RoutePath(np.array(Q), smooth_m=1.0), len(pts) - 2


def _quintic(u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    return u**3 * (10 - 15 * u + 6 * u * u)


def shift_path(route, s_now: float, lat_now: float, s_blocker: float, r_blocker: float, offset: float, v: float,
               front: float = 1.6, rear: float = 0.33, margin: float = 0.5, a_lat: float = 1.5, tail: float = 80.0):
    """Path for the robot origin: from its current lateral position ``lat_now`` (route frame, + left) to
    ``offset`` before the blocker, held until its tail is past, then back onto the route.

    Each transition is a quintic (zero slope and curvature at both ends) long enough that the peak
    lateral acceleration 5.77*D*v^2/L^2 stays under ``a_lat``. ``offset`` is signed (+ = left)."""
    from talk2drive.opendrive import RoutePath

    v = max(v, 3.0)

    def length(d):
        return max(8.0, v * math.sqrt(5.77 * abs(d) / a_lat))

    s_full = s_blocker - r_blocker - margin - front  # origin here: shift complete
    s_hold = s_blocker + r_blocker + margin + rear  # origin here: tail past the blocker
    L_in = length(offset - lat_now)
    s_in0 = max(s_now, s_full - L_in)
    L_out = length(offset)
    s_end = min(route.length - 1.0, s_hold + L_out + tail)
    ss = np.arange(s_now, s_end, 0.25)
    lat = np.where(ss < s_in0, lat_now,
                   np.where(ss < s_full, lat_now + (offset - lat_now) * _quintic((ss - s_in0) / max(s_full - s_in0, 1e-3)),
                            np.where(ss < s_hold, offset, offset * (1 - _quintic((ss - s_hold) / L_out)))))
    hd = np.interp(ss, route._s, route._heading)
    base = route.xy[np.clip((ss / route.ds).astype(int), 0, len(route.xy) - 1)]
    pts = base + lat[:, None] * np.c_[-np.sin(hd), np.cos(hd)]
    return RoutePath(pts, smooth_m=0.0), float(s_hold + L_out - s_now)


def locate_box(cam, depth: np.ndarray, box_px) -> tuple[np.ndarray, float] | None:
    """World (x, y) of an object from its pixel box: median depth over the box's middle, pushed back by
    half the object's width (estimated from the box width)."""
    x1, y1, x2, y2 = box_px
    w, h = x2 - x1, y2 - y1
    if w < 2 or h < 2:
        return None
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    sub = depth[int(cy - 0.2 * h):int(cy + 0.2 * h) + 1, int(cx - 0.2 * w):int(cx + 0.2 * w) + 1]
    sub = sub[np.isfinite(sub) & (sub > 0.1)]
    if sub.size == 0:
        return None
    d = float(np.median(sub))
    p = np.asarray(cam.get_world_points_from_image_coords(np.array([[cx, cy]]), np.array([d])), dtype=float)[0]
    fx = float(cam.get_intrinsics_matrix()[0, 0])
    radius = 0.5 * w * d / fx
    o = np.asarray(cam.get_world_pose()[0], dtype=float)
    ray = p[:2] - o[:2]
    ray /= max(np.linalg.norm(ray), 1e-6)
    return p[:2] + radius * ray, radius


class DepthGate:
    """Brake-only safety check on the depth image along the path the NMPC is following."""

    def __init__(self, cam, road_height, half_width: float = 0.58, margin: float = 0.35, h_min: float = 0.2,
                 h_max: float = 2.2, stride: int = 6, max_range: float = 40.0):
        self.cam, self.road_height = cam, road_height
        self.half = half_width + margin
        self.h_min, self.h_max, self.stride, self.max_range = h_min, h_max, stride, max_range
        self._uv = None

    def obstacle_points(self) -> np.ndarray:
        """World (x, y) of depth pixels standing h_min..h_max above the road, within range."""
        depth = np.asarray(self.cam.get_depth(), dtype=float)
        H, W = depth.shape
        if self._uv is None or self._uv[0] != (H, W):
            vv, uu = np.mgrid[0:H:self.stride, 0:W:self.stride]
            self._uv = ((H, W), np.c_[uu.ravel(), vv.ravel()].astype(float), vv.ravel(), uu.ravel())
        _, uv, vi, ui = self._uv
        d = depth[vi, ui]
        ok = np.isfinite(d) & (d > 0.3) & (d < self.max_range)
        if not ok.any():
            return np.zeros((0, 2))
        p = np.asarray(self.cam.get_world_points_from_image_coords(uv[ok], d[ok]), dtype=float)
        zg = self.road_height(p[:, 0], p[:, 1])
        hgt = p[:, 2] - zg
        keep = np.isfinite(hgt) & (hgt > self.h_min) & (hgt < self.h_max)
        return p[keep, :2]

    def first_blocked(self, path, s0: float, look: float, pts: np.ndarray | None = None) -> float | None:
        """Distance along ``path`` from ``s0`` to the first obstacle point inside the robot corridor."""
        pts = self.obstacle_points() if pts is None else pts
        if len(pts) == 0:
            return None
        s = np.arange(s0, min(path.length, s0 + look), 0.5)
        if len(s) < 2:
            return None
        C = path.xy[np.clip((s / path.ds).astype(int), 0, len(path.xy) - 1)]
        d2 = ((pts[:, None, :] - C[None, :, :]) ** 2).sum(-1)  # points x corridor samples
        hit = (d2 < self.half ** 2).any(axis=0)
        if not hit.any():
            return None
        return float(s[np.argmax(hit)] - s0)
