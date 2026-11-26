"""Minimal OpenDRIVE (.xodr) reader: lane-centre routes through a road network.

Enough of the standard for driving routes on maps such as the Mcity digital twin:

* reference-line geometry: line, arc, spiral (clothoid), poly3, paramPoly3
* lanes: laneOffset, lane sections, lane widths (cubic polynomials), driving lanes
* road links (predecessor/successor) and junction connections

Routes are built at road level in right-hand traffic: along a road's +s direction the
vehicle uses the innermost right driving lane (id -1, or the nearest driving lane on the
right); against +s it uses the innermost left one. The result is a ``RoutePath`` with the
interface the CARLA tadpole NMPC expects (``xy``, ``ds``, ``length``, ``heading(s)``,
``kappa(s)``, ``kappa_many(s)``).
"""

from __future__ import annotations

import math
import random
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- geometry
def _poly(coefs, u):
    a, b, c, d = coefs
    return a + b * u + c * u * u + d * u * u * u


def _sample_geometry(g: ET.Element, step: float) -> np.ndarray:
    """(n, 3) samples [x, y, heading] along one planView geometry record, s from 0 to length."""
    s0, x0, y0, h0, L = (float(g.get(k)) for k in ("s", "x", "y", "hdg", "length"))
    n = max(2, int(math.ceil(L / step)) + 1)
    t = np.linspace(0.0, L, n)
    kind = g[0].tag
    if kind == "line":
        return np.c_[x0 + t * math.cos(h0), y0 + t * math.sin(h0), np.full(n, h0)]
    if kind == "arc":
        k = float(g[0].get("curvature"))
        if abs(k) < 1e-12:
            return np.c_[x0 + t * math.cos(h0), y0 + t * math.sin(h0), np.full(n, h0)]
        h = h0 + k * t
        return np.c_[x0 + (np.sin(h) - math.sin(h0)) / k, y0 - (np.cos(h) - math.cos(h0)) / k, h]
    if kind == "spiral":  # curvature linear in s: integrate numerically
        k0, k1 = float(g[0].get("curvStart")), float(g[0].get("curvEnd"))
        fine = np.linspace(0.0, L, max(50, 10 * n))
        h = h0 + k0 * fine + 0.5 * (k1 - k0) / max(L, 1e-9) * fine**2
        dx, dy = np.cos(h), np.sin(h)
        ds = np.diff(fine)
        x = x0 + np.r_[0.0, np.cumsum(0.5 * (dx[1:] + dx[:-1]) * ds)]
        y = y0 + np.r_[0.0, np.cumsum(0.5 * (dy[1:] + dy[:-1]) * ds)]
        return np.c_[np.interp(t, fine, x), np.interp(t, fine, y), np.interp(t, fine, h)]
    if kind in ("poly3", "paramPoly3"):  # local (u, v) curve, then rotate into place
        fine = np.linspace(0.0, 1.0, max(200, 20 * n))
        if kind == "poly3":
            c = [float(g[0].get(k)) for k in "abcd"]
            # u parameter: approximate arc length ~ u for road geometry (small slopes)
            u = fine * L
            v = _poly(c, u)
        else:
            cu = [float(g[0].get(k)) for k in ("aU", "bU", "cU", "dU")]
            cv = [float(g[0].get(k)) for k in ("aV", "bV", "cV", "dV")]
            p = fine * (L if g[0].get("pRange", "normalized") == "arcLength" else 1.0)
            u, v = _poly(cu, p), _poly(cv, p)
        arc = np.r_[0.0, np.cumsum(np.hypot(np.diff(u), np.diff(v)))]
        arc *= L / max(arc[-1], 1e-9)
        uu, vv = np.interp(t, arc, u), np.interp(t, arc, v)
        du, dv = np.gradient(uu, t, edge_order=1), np.gradient(vv, t, edge_order=1)
        c0, s0_ = math.cos(h0), math.sin(h0)
        return np.c_[x0 + c0 * uu - s0_ * vv, y0 + s0_ * uu + c0 * vv, h0 + np.arctan2(dv, du)]
    raise ValueError(f"unsupported geometry {kind}")


@dataclass
class Lane:
    id: int
    type: str
    widths: list  # [(sOffset, a, b, c, d)] relative to the lane section start

    def width(self, ds: float) -> float:
        rec = self.widths[0]
        for w in self.widths:
            if w[0] <= ds + 1e-9:
                rec = w
        return _poly(rec[1:], ds - rec[0])


@dataclass
class Road:
    id: str
    length: float
    junction: str
    geometry: list
    lane_offsets: list  # [(s, a, b, c, d)]
    sections: list  # [(s, {id: Lane})]
    links: dict = field(default_factory=dict)  # "predecessor"/"successor" -> (type, id, contactPoint)

    def reference(self, step: float = 0.25) -> np.ndarray:
        """(n, 4) [s, x, y, heading] along the reference line."""
        out = []
        for g in self.geometry:
            p = _sample_geometry(g, step)
            s = float(g.get("s")) + np.linspace(0.0, float(g.get("length")), len(p))
            out.append(np.c_[s, p])
        return np.vstack(out)

    def _section(self, s: float):
        sec = self.sections[0]
        for q in self.sections:
            if q[0] <= s + 1e-9:
                sec = q
        return sec

    def lane_offset(self, s: float) -> float:
        rec = None
        for r in self.lane_offsets:
            if r[0] <= s + 1e-9:
                rec = r
        return 0.0 if rec is None else _poly(rec[1:], s - rec[0])

    def driving_lane(self, side: int, s: float) -> int | None:
        """Innermost driving lane on ``side`` (-1 right, +1 left) at s, or None."""
        s0, lanes = self._section(s)
        ids = sorted((i for i, ln in lanes.items() if i * side > 0 and ln.type == "driving"), key=abs)
        return ids[0] if ids else None

    def lane_centre_t(self, lane_id: int, s: float) -> tuple[float, float]:
        """Lateral offset of a lane's centre from the reference line, and its width."""
        s0, lanes = self._section(s)
        sign = 1 if lane_id > 0 else -1
        t = self.lane_offset(s)
        for k in range(1, abs(lane_id)):
            ln = lanes.get(sign * k)
            if ln is not None:
                t += sign * ln.width(s - s0)
        w = lanes[lane_id].width(s - s0)
        return t + sign * 0.5 * w, w


class OpenDrive:
    def __init__(self, path: str):
        root = ET.parse(path).getroot()
        self.roads: dict[str, Road] = {}
        for rd in root.findall("road"):
            lanes_el = rd.find("lanes")
            offsets = [(float(o.get("s")), *(float(o.get(k)) for k in "abcd")) for o in lanes_el.findall("laneOffset")]
            sections = []
            for ls in lanes_el.findall("laneSection"):
                lanes = {}
                for side in ("left", "right"):
                    el = ls.find(side)
                    if el is None:
                        continue
                    for ln in el.findall("lane"):
                        ws = [(float(w.get("sOffset")), *(float(w.get(k)) for k in "abcd")) for w in ln.findall("width")]
                        lanes[int(ln.get("id"))] = Lane(int(ln.get("id")), ln.get("type"), ws or [(0, 0, 0, 0, 0)])
                sections.append((float(ls.get("s")), lanes))
            road = Road(rd.get("id"), float(rd.get("length")), rd.get("junction", "-1"),
                        rd.find("planView").findall("geometry"), offsets, sections)
            link = rd.find("link")
            if link is not None:
                for kind in ("predecessor", "successor"):
                    el = link.find(kind)
                    if el is not None:
                        road.links[kind] = (el.get("elementType"), el.get("elementId"), el.get("contactPoint"))
            self.roads[road.id] = road
        # junction connections: incoming road -> [(connecting road, contact point)]
        self.connections: dict[tuple[str, str], list] = {}
        for j in root.findall("junction"):
            for c in j.findall("connection"):
                self.connections.setdefault((j.get("id"), c.get("incomingRoad")), []).append(
                    (c.get("connectingRoad"), c.get("contactPoint")))

    # ------------------------------------------------------------------ graph
    def drivable(self, rid: str, direction: int) -> bool:
        road = self.roads[rid]
        side = -1 if direction > 0 else 1
        s = 0.5 * road.length
        return road.driving_lane(side, s) is not None

    def successors(self, rid: str, direction: int) -> list[tuple[str, int]]:
        """Road-level moves from the end of (road, direction)."""
        road = self.roads[rid]
        link = road.links.get("successor" if direction > 0 else "predecessor")
        if link is None:
            return []
        etype, eid, contact = link
        out = []
        if etype == "road":
            if eid in self.roads:
                out.append((eid, 1 if contact == "start" else -1))
        else:  # junction
            for conn, cp in self.connections.get((eid, rid), []):
                if conn in self.roads:
                    out.append((conn, 1 if cp == "start" else -1))
        return [(r, d) for r, d in out if self.drivable(r, d)]

    def find_loop(self, start: tuple[str, int], min_length: float = 300.0, max_length: float = 1200.0,
                  tries: int = 4000, seed: int = 0, prefer_straight: bool = True) -> list[tuple[str, int]]:
        """Random-walk search for a closed road-level loop through ``start``; longest-straight wins."""
        rng = random.Random(seed)
        best, best_score = None, -1.0
        for _ in range(tries):
            route, seen, length = [start], {start[0]}, self.roads[start[0]].length
            node = start
            while length < max_length:
                nxt = self.successors(*node)
                if any(n == start for n in nxt) and length >= min_length:
                    straight = sum(self._straight_length(r) for r, _ in route)
                    score = straight if prefer_straight else length
                    if score > best_score:
                        best, best_score = list(route), score
                    break
                nxt = [n for n in nxt if n[0] not in seen]
                if not nxt:
                    break
                node = rng.choice(nxt)
                route.append(node)
                seen.add(node[0])
                length += self.roads[node[0]].length
        if best is None:
            raise RuntimeError(f"no loop found through road {start}")
        return best

    def _straight_length(self, rid: str) -> float:
        return sum(float(g.get("length")) for g in self.roads[rid].geometry if g[0].tag == "line")

    def _lane_points(self, rid: str, direction: int, rank: int, step: float,
                     rank_out: int | None = None) -> tuple[np.ndarray, list[float]]:
        """Centre line of the ``rank``-th driving lane (0 = innermost) on the travel side of a road,
        changing smoothly to the ``rank_out``-th lane over the middle of the road if they differ."""
        road = self.roads[rid]
        side = -1 if direction > 0 else 1
        ref = road.reference(step)
        if direction < 0:  # walk in travel direction so lane continuity follows the vehicle
            ref = ref[::-1]
        rank_out = rank if rank_out is None else rank_out

        def offsets(r):
            out, t_prev, sec_prev, lane = [], None, None, None
            for s, *_ in ref:
                s0, lanes = road._section(s)
                ids = sorted((i for i, ln in lanes.items() if i * side > 0 and ln.type == "driving"), key=abs)
                if not ids:
                    out.append(None)
                    continue
                if t_prev is None:
                    lane = ids[min(r, len(ids) - 1)]
                elif s0 != sec_prev or lane not in ids:  # new lane section: stay in the nearest lane
                    lane = min(ids, key=lambda i: abs(road.lane_centre_t(i, s)[0] - t_prev))
                t_prev, sec_prev = road.lane_centre_t(lane, s)[0], s0
                out.append(road.lane_centre_t(lane, s))
            return out

        a = offsets(rank)
        b = a if rank_out == rank else offsets(rank_out)
        n = len(ref)
        seg, widths = [], []
        for k, ((s, x, y, h), ta, tb) in enumerate(zip(ref, a, b)):
            if ta is None or tb is None:
                continue
            f = float(np.clip((k / max(n - 1, 1) - 0.2) / 0.6, 0.0, 1.0))  # lane change over the middle 60 %
            f = f * f * (3 - 2 * f)
            t = (1 - f) * ta[0] + f * tb[0]
            seg.append((x - t * math.sin(h), y + t * math.cos(h)))
            widths.append(ta[1])
        return np.asarray(seg, dtype=float).reshape(-1, 2), widths

    def lane_count(self, rid: str, direction: int) -> int:
        road = self.roads[rid]
        side = -1 if direction > 0 else 1
        return max(sum(1 for i, ln in lanes.items() if i * side > 0 and ln.type == "driving") for _, lanes in road.sections)

    def route_points(self, route: list[tuple[str, int]], step: float = 0.25, with_starts: bool = False,
                     force: dict | None = None):
        """Lane-centre polyline (n, 2) for a road-level route and the lane widths met along it
        (and, with ``with_starts``, the index of each road's first point in the polyline).
        ``force`` {route index: lane rank} keeps those roads in that lane all along (a lane change).

        Junction connectors join specific lanes (a right turn leaves from the outer lane), so the
        lane on each road is chosen by dynamic programming to minimise the jumps at every join."""
        lane_change_cost = 1.0  # m-equivalent: prefer staying in lane when joins allow it
        cands = []
        force = force or {}
        for k, (r, d) in enumerate(route):
            n = max(1, self.lane_count(r, d))
            opts = [(i, o) for i in range(n) for o in range(n)] if self.roads[r].length > 15.0 else [(i, i) for i in range(n)]
            if k in force:
                rank = min(force[k], n - 1)
                opts = [(rank, rank)]
            cands.append([(self._lane_points(r, d, i, step, o), float(i != o) * lane_change_cost) for i, o in opts])
        cost = [np.array([pen for _, pen in cands[0]])]
        back = []
        for i in range(1, len(cands)):
            c = np.full(len(cands[i]), np.inf)
            b = np.zeros(len(cands[i]), dtype=int)
            for k, ((pk, _), pen) in enumerate(cands[i]):
                for j, ((pj, _), _) in enumerate(cands[i - 1]):
                    gap = float(np.hypot(*(pk[0] - pj[-1]))) if len(pk) and len(pj) else 0.0
                    if cost[-1][j] + gap + pen < c[k]:
                        c[k], b[k] = cost[-1][j] + gap + pen, j
            cost.append(c)
            back.append(b)
        k = int(np.argmin(cost[-1]))
        choice = [k]
        for b in reversed(back):
            k = int(b[k])
            choice.append(k)
        choice.reverse()
        cands = [[pts for pts, _ in cand] for cand in cands]
        pts, widths, starts = [], [], []
        for cand, k in zip(cands, choice):
            P, w = cand[k]
            starts.append(len(pts))
            pts.extend(P.tolist())
            widths.extend(w)
        P = np.asarray(pts, dtype=float)
        keep = np.r_[True, np.hypot(*np.diff(P, axis=0).T) > 1e-3]
        if not with_starts:
            return P[keep], widths
        kept_index = np.cumsum(keep) - 1  # raw point index -> index after dropping duplicates
        return P[keep], widths, [int(kept_index[min(i, len(keep) - 1)]) for i in starts]


# --------------------------------------------------------------------------- path for the NMPC
class RoutePath:
    """Uniformly resampled, smoothed path with heading and curvature tables (NMPC interface)."""

    def __init__(self, points: np.ndarray, ds: float = 0.25, smooth_m: float = 1.5, closed: bool = False):
        P = np.asarray(points, dtype=float)
        seg = np.hypot(*np.diff(P, axis=0).T)
        s = np.r_[0.0, np.cumsum(seg)]
        n = int(s[-1] / ds) + 1
        grid = np.linspace(0.0, (n - 1) * ds, n)
        xy = np.c_[np.interp(grid, s, P[:, 0]), np.interp(grid, s, P[:, 1])]
        k = max(1, int(round(smooth_m / ds)))  # moving average removes joint kinks between roads
        if k > 1:
            ker = np.ones(2 * k + 1) / (2 * k + 1)
            pad = np.r_[xy[-k:], xy, xy[:k]] if closed else np.r_[np.repeat(xy[:1], k, 0), xy, np.repeat(xy[-1:], k, 0)]
            xy = np.c_[np.convolve(pad[:, 0], ker, "valid"), np.convolve(pad[:, 1], ker, "valid")]
        # re-parameterise by arc length after smoothing
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
        n = int(s[-1] / ds) + 1
        grid = np.linspace(0.0, (n - 1) * ds, n)
        self.xy = np.c_[np.interp(grid, s, xy[:, 0]), np.interp(grid, s, xy[:, 1])]
        self.ds = ds
        self.length = float((len(self.xy) - 1) * ds)
        d = np.gradient(self.xy, ds, axis=0)
        self._heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
        dh = np.gradient(self._heading, ds)
        m = max(1, int(round(1.0 / ds)))  # 1 m smoothing on curvature
        self._kappa = np.convolve(np.r_[np.repeat(dh[:1], m, 0), dh, np.repeat(dh[-1:], m, 0)],
                                  np.ones(2 * m + 1) / (2 * m + 1), "valid")
        self._s = grid

    def heading(self, s: float) -> float:
        return float(np.interp(s, self._s, self._heading))

    def kappa(self, s: float) -> float:
        return float(np.interp(s, self._s, self._kappa))

    def kappa_many(self, s) -> np.ndarray:
        return np.interp(np.asarray(s, dtype=float), self._s, self._kappa)

    def project(self, x: float, y: float, s_hint: float | None = None, window: float = 15.0) -> tuple[float, float]:
        """Arc position of the closest path point (searched near ``s_hint``) and signed lateral error."""
        if s_hint is None:
            idx = np.arange(len(self.xy))
        else:
            i0 = max(0, int((s_hint - window) / self.ds))
            i1 = min(len(self.xy), int((s_hint + window) / self.ds) + 1)
            idx = np.arange(i0, i1)
        d = np.hypot(self.xy[idx, 0] - x, self.xy[idx, 1] - y)
        i = int(idx[np.argmin(d)])
        j = min(i, len(self.xy) - 2)
        a, b = self.xy[j], self.xy[j + 1]
        ab = b - a
        f = float(np.clip(np.dot([x - a[0], y - a[1]], ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0))
        s = (j + f) * self.ds
        h = self.heading(s)
        cte = -(x - (a[0] + f * ab[0])) * math.sin(h) + (y - (a[1] + f * ab[1])) * math.cos(h)
        return s, cte

    def transformed(self, fn) -> "RoutePath":
        """Copy with points mapped through ``fn((n,2)) -> (n,2)`` (e.g. into the sim frame)."""
        return RoutePath(fn(self.xy), ds=self.ds, smooth_m=0.0)
