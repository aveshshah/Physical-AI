"""Sideways manoeuvres: pull over to the side of the road and back, drive beside the lane centre
(a nudge or a lane change), and turn around.

The sideways moves are a smooth shift of the route (a quintic: no jump in heading or curvature), long
enough that the sideways acceleration stays gentle.
"""

from __future__ import annotations

import math

import numpy as np

from talk2drive.opendrive import RoutePath

SIDE_ACCEL = 1.0          # peak sideways acceleration of a shift [m/s^2]
KERB_GAP = 0.3            # gap left between the robot and the edge of the pavement [m]
JUNCTION_CLEARANCE = 15.0  # never park closer than this to a junction [m]
PARK_BRAKE = 1.2          # gentle braking to the parking spot [m/s^2]


def _quintic(u):
    u = np.clip(u, 0.0, 1.0)
    return u**3 * (10 - 15 * u + 6 * u * u)


def shift_length(offset: float, speed: float) -> float:
    """Distance needed to move sideways by ``offset`` at ``speed`` with peak side acceleration SIDE_ACCEL."""
    return max(12.0, max(speed, 3.0) * math.sqrt(5.77 * abs(offset) / SIDE_ACCEL))


def sideways_path(route: RoutePath, s_now: float, lat_now: float, s_start: float, lat_target: float,
                  length: float, s_end: float) -> RoutePath:
    """A path that keeps ``lat_now`` until ``s_start``, moves smoothly to ``lat_target`` over ``length``
    metres and then stays there until ``s_end`` (all distances along ``route``; lat + = left)."""
    ss = np.arange(s_now, min(s_end, route.length - 1.0), route.ds)
    lat = lat_now + (lat_target - lat_now) * _quintic((ss - s_start) / length)
    heading = np.interp(ss, route._s, route._heading)
    base = route.xy[np.clip((ss / route.ds).astype(int), 0, len(route.xy) - 1)]
    return RoutePath(base + lat[:, None] * np.c_[-np.sin(heading), np.cos(heading)], smooth_m=0.0)


def offset_path(route: RoutePath, s_now: float, lat_now: float, target: float, speed: float,
                s_back: float | None = None) -> tuple[RoutePath, float]:
    """Drive beside the route centre line: move smoothly from ``lat_now`` to ``target`` (+ = left),
    stay there, and, if ``s_back`` is given, be back on the centre line by ``s_back`` (a junction).
    Returns (path starting at s_now, route distance where the first shift ends)."""
    ss = np.arange(s_now, route.length - 1.0, route.ds)
    length = shift_length(target - lat_now, speed)
    shift_end = s_now + 1.0 + length
    lat = lat_now + (target - lat_now) * _quintic((ss - s_now - 1.0) / length)
    if s_back is not None:
        back_length = shift_length(target, speed)
        lat = lat - target * _quintic((ss - (s_back - back_length)) / back_length)
    heading = np.interp(ss, route._s, route._heading)
    base = route.xy[np.clip((ss / route.ds).astype(int), 0, len(route.xy) - 1)]
    return RoutePath(base + lat[:, None] * np.c_[-np.sin(heading), np.cos(heading)], smooth_m=0.0), shift_end


def paved_room_right(route: RoutePath, s: float, is_paved) -> float:
    """How far the pavement continues to the right of the route at s [m] (0 if the route itself is off it)."""
    heading = route.heading(s)
    right = np.array([math.sin(heading), -math.cos(heading)])
    centre = route.xy[min(int(s / route.ds), len(route.xy) - 1)]
    steps = np.arange(0.0, 8.0, 0.15)
    points = centre + np.outer(steps, right)
    off = np.nonzero(~is_paved(points[:, 0], points[:, 1]))[0]
    if len(off) == 0:
        return float(steps[-1])
    return float(steps[off[0] - 1]) if off[0] > 0 else 0.0


def find_pull_over_spot(planned, router, s_now: float, speed: float, is_paved, half_width: float,
                        max_dist: float = 200.0):
    """The first place ahead where the robot can stop at the right-hand edge of the pavement.

    It must be reachable with gentle braking, away from junctions, and have room on the right along
    the whole shift. Returns (s_stop, offset, shift_start, shift_length) or None."""
    route = planned.path
    junctions = []   # (start, end) of every junction road on the route
    for i, node in enumerate(planned.roads):
        if router.is_junction_road(node):
            end = planned.road_start_s[i + 1] if i + 1 < len(planned.roads) else route.length
            junctions.append((planned.road_start_s[i], end))

    nearest = s_now + max(25.0, speed * speed / (2 * PARK_BRAKE) + 10.0)
    for s_stop in np.arange(nearest, min(s_now + max_dist, route.length - 20.0), 2.0):
        room = paved_room_right(route, s_stop, is_paved)
        offset = -(room - half_width - KERB_GAP)
        if offset > -0.15:
            continue   # no space beside the lane here
        offset = max(offset, -6.0)
        length = shift_length(offset, min(speed, 5.0))
        s_start = s_stop - length - 3.0
        if s_start < s_now + 2.0:
            continue
        window = np.arange(s_start, s_stop + 6.0, 1.0)
        if any(a - JUNCTION_CLEARANCE < s < b + JUNCTION_CLEARANCE for s in (window[0], window[-1]) for a, b in junctions):
            continue
        if any(a < s < b for s in window for a, b in junctions):
            continue
        room_along = min(paved_room_right(route, s, is_paved) for s in window[window >= s_start + 0.5 * length])
        if room_along < abs(offset) + half_width + KERB_GAP - 0.05:
            continue
        return float(s_stop), float(offset), float(s_start), float(length)
    return None


def u_turn_path(route: RoutePath, s_now: float, s_start: float, opposite: RoutePath,
                min_radius: float = 1.5, max_width: float = 11.0, lat_now: float = 0.0):
    """Turn around: follow ``route`` to ``s_start``, then a half circle to the left into the lane going
    the other way (``opposite``), then along it. ``lat_now``: how far the robot is beside the route
    now (+ left; e.g. parked at the kerb); a U-turn close by starts from there.
    Returns (path, where the half circle ends on the path, where it joins ``opposite``) or None if
    the other lane is not beside the robot, or the road is too narrow for the robot's turning circle
    (``min_radius``) or too wide (a median)."""
    lat_start = lat_now if s_start - s_now < shift_length(lat_now, 3.0) else 0.0
    h = route.heading(s_start)
    left = np.array([-math.sin(h), math.cos(h)])
    a = route.xy[min(int(s_start / route.ds), len(route.xy) - 1)] + lat_start * left
    s_join, _ = opposite.project(a[0], a[1])
    b = opposite.xy[min(int(s_join / opposite.ds), len(opposite.xy) - 1)]
    if math.cos(opposite.heading(s_join) - h) > -0.8:
        return None   # that lane does not run the other way here
    width = float(np.dot(b - a, left))
    if width < 2 * min_radius or width > max_width:
        return None
    r = width / 2
    centre = a + r * left
    angles = (h - math.pi / 2) + np.linspace(0.0, math.pi, max(12, int(math.pi * r / 0.2)))
    arc = centre + r * np.c_[np.cos(angles), np.sin(angles)]
    ss = np.arange(s_now, s_start, route.ds)
    lat = lat_now + (lat_start - lat_now) * _quintic((ss - s_now) / max(s_start - s_now, 1.0))
    hs = np.interp(ss, route._s, route._heading)
    before = route.xy[np.clip((ss / route.ds).astype(int), 0, len(route.xy) - 1)] + lat[:, None] * np.c_[-np.sin(hs), np.cos(hs)]
    after = opposite.xy[int(s_join / opposite.ds) + 1:]
    path = RoutePath(np.r_[before.reshape(-1, 2), arc, after], smooth_m=0.0)
    return path, (s_start - s_now) + math.pi * r, float(s_join)
