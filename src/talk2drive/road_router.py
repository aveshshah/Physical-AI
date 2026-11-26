"""Global routing on the OpenDRIVE road graph: where are the junctions ahead, which turns do they
allow, and the route that takes a requested turn.

A route is a list of (road id, direction) pairs. With no command the robot keeps going "straight"
(at each junction it takes the exit closest to straight ahead). "Turn left" finds the first junction
ahead, between ``min_dist`` and ``max_dist`` metres, that has a left exit, and re-plans through it.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from talk2drive.opendrive import OpenDrive, RoutePath

STRAIGHT_DEG = 25.0   # exits within +-25 deg of straight ahead count as "straight"
U_TURN_DEG = 150.0    # exits turning more than this are U-turns (never offered)
SAMPLE = 0.5          # lane-point spacing used to measure headings [m]
JOIN_GAP = 4.0        # roads whose lanes are further apart than this at the join are not joined [m]


@dataclass
class Move:
    """One way out of a junction."""
    kind: str            # "left", "right" or "straight"
    angle_deg: float     # heading change, + = left
    connector: tuple     # (road, direction) of the junction's connecting road
    exit: tuple          # (road, direction) of the road after the junction


@dataclass
class PlannedRoute:
    roads: list           # [(road id, direction), ...]
    path: RoutePath       # lane-centre path in the scene frame
    road_start_s: list    # distance along ``path`` where each road starts [m]
    force: dict = field(default_factory=dict)   # {road index: lane rank} kept after a lane change

    def road_index_at(self, s: float) -> int:
        """Index in ``roads`` of the road at distance s along the path."""
        return max(0, int(np.searchsorted(self.road_start_s, s, side="right")) - 1)


@dataclass
class Junction:
    s: float              # distance along the route where the junction starts [m]
    road_index: int       # index in route.roads of the road that leads into it
    moves: list           # [Move, ...] available there
    taken: Move | None    # the move the route takes now


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _smooth_joins(points: np.ndarray, starts: list) -> np.ndarray:
    """Some joins between roads jump sideways (up to a lane width, where a short connector meets the
    other lane). Replace the stretch around each such jump with a cubic curve that leaves and rejoins
    the lanes in their own direction, long enough to be steerable (6x the jump, at least 8 m each way).
    Clean joins are left exactly as they are; jumps close together share one curve."""
    P = np.array(points, dtype=float)
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    windows = []   # [start s, end s] around each jump, overlapping ones merged
    for j in starts[1:]:
        if j < 1:
            continue
        gap = float(np.linalg.norm(P[j] - P[j - 1]))
        if gap < 0.5:
            continue
        span = max(8.0, 6.0 * gap) / 2
        lo, hi = s[j] - span, s[j] + span
        if windows and lo <= windows[-1][1]:
            windows[-1][1] = hi
        else:
            windows.append([lo, hi])
    smoothed = P.copy()
    for lo, hi in windows:
        a = int(np.searchsorted(s, lo))
        b = int(np.searchsorted(s, hi))
        if a < 2 or b > len(P) - 3 or b - a < 4:
            continue
        length = s[b] - s[a]
        t0 = P[a] - P[a - 2]
        t1 = P[b + 2] - P[b]
        t0 = t0 / max(np.linalg.norm(t0), 1e-9) * length
        t1 = t1 / max(np.linalg.norm(t1), 1e-9) * length
        u = np.linspace(0.0, 1.0, b - a + 1)[:, None]
        h00, h10, h01, h11 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2
        smoothed[a:b + 1] = h00 * P[a] + h10 * t0 + h01 * P[b] + h11 * t1
    return smoothed


class RoadRouter:
    def __init__(self, od: OpenDrive, to_scene):
        self.od = od
        self.to_scene = to_scene   # map coordinates (n, 2) -> scene coordinates (n, 2)
        self._lanes = {}

    # ------------------------------------------------------------------ headings and moves
    def _lane(self, node):
        if node not in self._lanes:
            self._lanes[node] = self.od._lane_points(node[0], node[1], 0, SAMPLE)[0]
        return self._lanes[node]

    def _heading(self, node, at_end: bool, span: float = 4.0) -> float:
        """Direction of travel at the start or end of a road [deg]."""
        P = self._lane(node)
        k = min(len(P) - 1, int(span / SAMPLE))
        a, b = (P[-1 - k], P[-1]) if at_end else (P[0], P[k])
        return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))

    def is_junction_road(self, node) -> bool:
        return self.od.roads[node[0]].junction != "-1"

    def _lane_ends(self, node):
        """(start, end) points of each driving lane of ``node`` in travel direction (map frame)."""
        key = ("ends", node)
        if key not in self._lanes:
            ends = []
            for rank in range(max(1, self.od.lane_count(*node))):
                P = self.od._lane_points(node[0], node[1], rank, SAMPLE)[0]
                if len(P):
                    ends.append((P[0], P[-1]))
            self._lanes[key] = ends
        return self._lanes[key]

    def next_roads(self, node) -> list:
        """Roads that follow ``node`` and whose lanes really meet its lanes (a few links in the map
        join roads whose lanes in that direction only start further along, metres away)."""
        out = []
        for nxt in self.od.successors(*node):
            gaps = [np.hypot(*(b[0] - a[1])) for a in self._lane_ends(node) for b in self._lane_ends(nxt)]
            if gaps and min(gaps) <= JOIN_GAP:
                out.append(nxt)
        return out

    def moves_after(self, node) -> list:
        """The ways on at the end of road ``node`` (one Move per exit road; U-turns left out)."""
        moves = {}
        for nxt in self.next_roads(node):
            if self.is_junction_road(nxt):
                after = self.next_roads(nxt)
                if not after:
                    continue
                exit_road = after[0]
            else:
                exit_road = nxt
            angle = _wrap_deg(self._heading(exit_road, at_end=False) - self._heading(node, at_end=True))
            if abs(angle) > U_TURN_DEG:
                continue
            if abs(angle) <= STRAIGHT_DEG:
                kind = "straight"
            elif angle > 0:
                kind = "left"
            else:
                kind = "right"
            connector = nxt if self.is_junction_road(nxt) else None
            # several connectors can lead to the same exit (one per lane): keep the first
            moves.setdefault(exit_road, Move(kind, angle, connector, exit_road))
        return list(moves.values())

    def _roads_for(self, move: Move) -> list:
        return [move.connector, move.exit] if move.connector is not None else [move.exit]

    # ------------------------------------------------------------------ building routes
    def keep_going(self, roads: list, min_length: float) -> list:
        """Extend ``roads`` until it is at least ``min_length`` long, going as straight as possible."""
        roads = list(roads)
        length = sum(self.od.roads[r].length for r, _ in roads)
        while length < min_length:
            moves = self.moves_after(roads[-1])
            if not moves:
                break   # dead end
            best = min(moves, key=lambda m: abs(m.angle_deg))
            for node in self._roads_for(best):
                roads.append(node)
                length += self.od.roads[node[0]].length
        return roads

    def build(self, roads: list, force: dict | None = None) -> PlannedRoute:
        """The drivable path along ``roads``; ``force`` {road index: lane rank} pins lanes."""
        points, _, starts = self.od.route_points(roads, with_starts=True, force=force)
        scene = _smooth_joins(self.to_scene(points), starts)
        path = RoutePath(scene)
        raw_s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(scene, axis=0).T))]
        return PlannedRoute(list(roads), path, [float(raw_s[i]) for i in starts], dict(force or {}))

    # ------------------------------------------------------------------ lanes beside the robot
    def lanes_beside(self, node, x: float, y: float, heading: float) -> list:
        """The driving lanes of road ``node`` at the point (x, y): [(rank, sideways distance of the
        lane centre from the point (+ left), lane width), ...] for the travel direction, then the same
        for the lanes going the other way with rank -1."""
        out = []
        left = np.array([-math.sin(heading), math.cos(heading)])
        for direction, ranks in ((node[1], range(self.od.lane_count(*node))),
                                 (-node[1], [-1] if self.od.drivable(node[0], -node[1]) else [])):
            for rank in ranks:
                P, widths = self.od._lane_points(node[0], direction, max(rank, 0), SAMPLE)
                if len(P) == 0:
                    continue
                P = self.to_scene(P)
                i = int(np.argmin(np.hypot(P[:, 0] - x, P[:, 1] - y)))
                out.append((rank, float(np.dot(P[i] - (x, y), left)), float(widths[min(i, len(widths) - 1)])))
        return out

    # ------------------------------------------------------------------ turning back
    def route_back(self, route: PlannedRoute, s_now: float, max_length: float = 1200.0, near: float = 40.0):
        """The shortest way (over junctions, a roundabout or round a block) from the current road onto
        a road that passes within ``near`` metres of the robot going the opposite way (the same road
        reversed, or the parallel one of a one-way pair). Returns the roads from the current one to
        that road, or None."""
        x, y = route.path.xy[min(int(s_now / route.path.ds), len(route.path.xy) - 1)]
        heading = route.path.heading(s_now)
        start = route.roads[route.road_index_at(s_now)]

        def goes_back(node):
            if self.is_junction_road(node):
                return False
            P = self.to_scene(self._lane(node))
            if len(P) < 2:
                return False
            k = int(np.argmin(np.hypot(P[:, 0] - x, P[:, 1] - y)))
            if np.hypot(P[k, 0] - x, P[k, 1] - y) > near:
                return False
            a, b = P[max(k - 2, 0)], P[min(k + 2, len(P) - 1)]
            return math.cos(math.atan2(b[1] - a[1], b[0] - a[0]) - heading) < -0.7

        queue, seen, count = [(0.0, 0, [start])], set(), 1
        while queue:
            length, _, roads = heapq.heappop(queue)
            node = roads[-1]
            if len(roads) > 1 and goes_back(node):
                return roads
            if node in seen or length > max_length:
                continue
            seen.add(node)
            for nxt in self.next_roads(node):
                count += 1
                heapq.heappush(queue, (length + self.od.roads[nxt[0]].length, count, roads + [nxt]))
        return None

    # ------------------------------------------------------------------ junctions ahead
    def junctions_ahead(self, route: PlannedRoute, s_now: float, max_dist: float) -> list:
        """Every place ahead (within ``max_dist``) where the route leaves a road into a junction
        with a choice of exits, or with a single exit that is a turn."""
        out = []
        for i in range(route.road_index_at(s_now), len(route.roads) - 1):
            node = route.roads[i]
            if self.is_junction_road(node):
                continue
            s_junction = route.road_start_s[i + 1]
            if s_junction - s_now > max_dist:
                break
            if s_junction <= s_now:
                continue
            moves = self.moves_after(node)
            if len(moves) < 2 and all(m.kind == "straight" for m in moves):
                continue   # just a road joint, not a junction with a choice
            nxt = route.roads[i + 1]
            taken = next((m for m in moves if nxt in self._roads_for(m)), None)
            out.append(Junction(s_junction, i, moves, taken))
        return out

    def plan_turn(self, route: PlannedRoute, s_now: float, kind: str, min_dist: float,
                  max_dist: float = 200.0, keep_length: float = 500.0):
        """Re-plan to take ``kind`` ("left" / "right" / "straight") at the first junction between
        ``min_dist`` and ``max_dist`` metres ahead that allows it.

        Returns (new roads, junction, move) or (None, junction-or-None, None) when not possible:
        the junction is then the one that already takes that move (nothing to change), or None."""
        for junction in self.junctions_ahead(route, s_now, max_dist):
            if junction.s - s_now < min_dist:
                continue   # too close to make the turn safely
            options = [m for m in junction.moves if m.kind == kind]
            if not options:
                continue
            if junction.taken is not None and junction.taken.kind == kind:
                return None, junction, junction.taken   # already going that way
            # left: the exit nearest +90 deg; right: nearest -90; straight: nearest 0
            target = {"left": 90.0, "right": -90.0, "straight": 0.0}[kind]
            move = min(options, key=lambda m: abs(m.angle_deg - target))
            first = route.road_index_at(s_now)
            roads = route.roads[first:junction.road_index + 1] + self._roads_for(move)
            return self.keep_going(roads, keep_length), junction, move
        return None, None, None
