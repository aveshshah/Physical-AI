"""Time-parameterised reference trajectories (figure-8, slalom, circle).

A trajectory is defined by a smooth path ``p(t) = (x(t), y(t))``. Heading, forward speed, yaw
rate and the matching wheel speeds are derived from it by finite differences, so new shapes
only need a position function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .robot import DiffDriveParams


@dataclass
class RefPoint:
    x: float
    y: float
    theta: float
    v: float
    w: float

    def as_state(self, params: DiffDriveParams) -> np.ndarray:
        wl, wr = params.body_to_wheels(self.v, self.w)
        return np.array([self.x, self.y, self.theta, wl, wr])


class Trajectory:
    """``path`` is traversed in ``path_duration`` seconds; ``ramp`` seconds of smooth (constant
    acceleration) time-scaling are added at the start and the end so the reference begins and
    ends at standstill - a real robot cannot jump to cruise speed."""

    def __init__(
        self,
        path: Callable[[float], tuple[float, float]],
        path_duration: float,
        name: str = "trajectory",
        ramp: float = 2.0,
    ):
        self._path = path
        self.path_duration = path_duration
        self.ramp = max(0.0, min(ramp, path_duration))
        self.duration = path_duration + self.ramp
        self.name = name
        # rigid transform applied on top of the path (see anchored())
        self._origin = np.zeros(2)
        self._rot = 0.0

    # ---------------------------------------------------------------- geometry
    def _tau(self, t: float) -> float:
        """Wall-clock time -> path time, with smooth start/stop ramps."""
        Tr, D, Dw = self.ramp, self.path_duration, self.duration
        if Tr <= 0:
            return min(max(t, 0.0), D)
        if t <= 0:
            return 0.0
        if t < Tr:
            return t * t / (2 * Tr)
        if t <= Dw - Tr:
            return t - Tr / 2
        if t < Dw:
            return D - (Dw - t) ** 2 / (2 * Tr)
        return D

    def _p(self, t: float) -> np.ndarray:
        return self._pr(self._tau(t))

    def sample(self, t: float, h: float = 1e-3) -> RefPoint:
        t = min(max(t, 0.0), self.duration)
        p = self._p(t)
        d1 = (self._p(t + h) - self._p(t - h)) / (2 * h)
        d2 = (self._p(t + h) - 2 * p + self._p(t - h)) / (h * h)
        v = float(np.hypot(*d1))
        # Heading from the path tangent (well defined even where the ramp makes v -> 0).
        tau = self._tau(t)
        g = (self._pr(tau + h) - self._pr(tau - h)) / (2 * h)
        theta = math.atan2(g[1], g[0])
        w = float((d1[0] * d2[1] - d1[1] * d2[0]) / v**2) if v > 1e-4 else 0.0
        if t <= 0.0 or t >= self.duration:
            v, w = 0.0, 0.0
        return RefPoint(float(p[0]), float(p[1]), theta, v, w)

    def _pr(self, tau: float) -> np.ndarray:
        """Transformed path at path-time ``tau`` (no ramp)."""
        x, y = self._path(min(max(tau, 0.0), self.path_duration))
        c, s = math.cos(self._rot), math.sin(self._rot)
        return self._origin + np.array([c * x - s * y, s * x + c * y])

    def horizon(self, t0: float, dt: float, n: int, params: DiffDriveParams) -> np.ndarray:
        """(n+1, 5) array of reference states at t0, t0+dt, ..., t0+n*dt."""
        return np.stack([self.sample(t0 + k * dt).as_state(params) for k in range(n + 1)])

    def anchored(self, x0: float, y0: float, theta0: float) -> "Trajectory":
        """Move/rotate the trajectory so it starts exactly at the robot's current pose."""
        start = self.sample(0.0)
        self._rot += theta0 - start.theta
        self._origin = np.zeros(2)
        self._origin = np.array([x0, y0]) - self._pr(0.0)
        return self

    def start_pose(self) -> tuple[float, float, float]:
        r = self.sample(0.0)
        return r.x, r.y, r.theta

    def max_speed(self, n: int = 2000) -> float:
        return max(self.sample(t).v for t in np.linspace(0, self.duration * 0.999, n))


# ---------------------------------------------------------------------------- library
def figure_eight(size: float = 1.0, period: float = 30.0, laps: float = 1.0) -> Trajectory:
    """Lemniscate of Gerono: width 2*size, height size."""
    om = 2 * math.pi / period
    return Trajectory(
        lambda t: (size * math.sin(om * t), 0.5 * size * math.sin(2 * om * t)),
        period * laps,
        name="figure8",
    )


def slalom(speed: float = 0.25, amplitude: float = 0.3, wavelength: float = 1.5, length: float = 6.0) -> Trajectory:
    k = 2 * math.pi / wavelength
    return Trajectory(
        lambda t: (speed * t, amplitude * math.sin(k * speed * t)),
        length / speed,
        name="slalom",
    )


def circle(radius: float = 0.75, period: float = 25.0, laps: float = 1.0) -> Trajectory:
    om = 2 * math.pi / period
    return Trajectory(
        lambda t: (radius * math.sin(om * t), radius * (1 - math.cos(om * t))),
        period * laps,
        name="circle",
    )


TRAJECTORIES = {"figure8": figure_eight, "slalom": slalom, "circle": circle}
