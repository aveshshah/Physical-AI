"""The Mcity lane-following lap used by the NMPC runs (OpenDRIVE route in the Isaac scene frame)."""

from __future__ import annotations

import math
import os

import numpy as np

from talk2drive.isaac.jetbot import LOCAL_ENVIRONMENTS
from talk2drive.opendrive import OpenDrive, RoutePath

MCITY_XODR = os.path.expanduser(
    "~/assets/mcity-digital-twin/Omniverse/Collected_McityMap_NSR_v4_1_6/mcity_map_v6_nvidia.xodr")


def mcity_lap(xodr: str, start: str, seed: int) -> tuple[RoutePath, list, list]:
    """A lane-following loop as a closed path in the Isaac scene frame, its lane widths and its roads."""
    od = OpenDrive(xodr)
    rid, d = start.split(",")
    route = od.find_loop((rid, int(d)), min_length=500, max_length=1100, tries=3000, seed=seed)
    pts, widths = od.route_points(route)
    env = LOCAL_ENVIRONMENTS["mcity"]
    c, s = math.cos(math.radians(env.yaw_deg)), math.sin(math.radians(env.yaw_deg))
    q = pts - np.asarray(env.origin[:2])
    sim = np.c_[c * q[:, 0] - s * q[:, 1], s * q[:, 0] + c * q[:, 1]]
    return RoutePath(sim, closed=True), widths, route


def mcity_route(xodr: str, start: str, seed: int) -> tuple[RoutePath, float, list]:
    """One lap of a lane-following loop, in the Isaac scene frame, plus 40 m run-out past the start."""
    lap, widths, route = mcity_lap(xodr, start, seed)
    runout = int(40.0 / lap.ds)
    path = RoutePath(np.r_[lap.xy, lap.xy[1:runout]], smooth_m=0.0)
    return path, lap.length, [float(np.median(widths))], route
