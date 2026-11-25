"""Differential-drive robot model, parameters and the robot interface shared by all simulators.

State convention used everywhere in talk2drive::

    s = [x, y, theta, omega_L, omega_R]     (m, m, rad, rad/s, rad/s)

Wheel angular speeds are part of the state so the controller can respect motor speed limits now
(M1) and motor torque / current / energy later (M2) without changing interfaces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class DiffDriveParams:
    wheel_radius: float = 0.03  # m  (NVIDIA Jetbot)
    wheel_base: float = 0.1125  # m  distance between wheel contact points
    max_wheel_speed: float = 12.0  # rad/s  (~0.36 m/s)
    max_wheel_accel: float = 30.0  # rad/s^2
    # First-order lag of the wheel-speed loop (motor + velocity controller).
    wheel_time_constant: float = 0.05  # s

    @property
    def max_speed(self) -> float:
        return self.max_wheel_speed * self.wheel_radius

    def body_to_wheels(self, v: float, w: float) -> tuple[float, float]:
        """(v [m/s], w [rad/s]) -> (omega_L, omega_R) [rad/s]."""
        r, L = self.wheel_radius, self.wheel_base
        return (v - 0.5 * w * L) / r, (v + 0.5 * w * L) / r

    def wheels_to_body(self, wl: float, wr: float) -> tuple[float, float]:
        r, L = self.wheel_radius, self.wheel_base
        return 0.5 * r * (wl + wr), r * (wr - wl) / L

    def saturate_wheels(self, wl: float, wr: float) -> tuple[float, float]:
        """Scale both wheels together so the larger one hits the limit (keeps path curvature)."""
        m = max(abs(wl), abs(wr))
        if m > self.max_wheel_speed:
            k = self.max_wheel_speed / m
            return wl * k, wr * k
        return wl, wr


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat_wxyz(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class DiffDriveRobot(Protocol):
    """What a simulator (offline kinematic model or Isaac Sim) must provide."""

    params: DiffDriveParams
    physics_dt: float

    def get_state(self) -> np.ndarray:
        """Measured [x, y, theta, omega_L, omega_R]."""

    def set_wheel_speeds(self, omega_l: float, omega_r: float) -> None:
        """Wheel angular-velocity targets (rad/s)."""

    def step(self) -> None:
        """Advance the simulation by one physics step of ``physics_dt``."""


class KinematicDiffDrive:
    """Lightweight stand-in for Isaac Sim: unicycle kinematics + wheel-speed lag + noise + slip.

    Used for developing and unit-testing controllers without a GPU, and as a reference
    to compare Isaac Sim physics against.
    """

    def __init__(
        self,
        params: DiffDriveParams | None = None,
        physics_dt: float = 1.0 / 60.0,
        initial_state=(0.0, 0.0, 0.0),
        pose_noise_std: float = 0.0,
        slip: float = 0.0,
        seed: int = 0,
    ):
        self.params = params or DiffDriveParams()
        self.physics_dt = physics_dt
        self.pose_noise_std = pose_noise_std
        self.slip = slip  # 0 = perfect grip, 0.3 = wheels lose 30% of their speed to the floor
        self.rng = np.random.default_rng(seed)
        x, y, th = initial_state
        self.s = np.array([x, y, th, 0.0, 0.0], dtype=float)
        self.cmd = np.zeros(2)

    def get_state(self) -> np.ndarray:
        s = self.s.copy()
        if self.pose_noise_std:
            s[:2] += self.rng.normal(0.0, self.pose_noise_std, 2)
            s[2] += self.rng.normal(0.0, self.pose_noise_std * 5.0)
        return s

    def set_wheel_speeds(self, omega_l: float, omega_r: float) -> None:
        p = self.params
        self.cmd = np.clip([omega_l, omega_r], -p.max_wheel_speed, p.max_wheel_speed)

    def step(self) -> None:
        p, dt = self.params, self.physics_dt
        # wheel-speed loop: first-order lag, acceleration-limited
        dw = (self.cmd - self.s[3:5]) * (dt / max(p.wheel_time_constant, dt))
        dw = np.clip(dw, -p.max_wheel_accel * dt, p.max_wheel_accel * dt)
        self.s[3:5] += dw
        v, w = p.wheels_to_body(*(self.s[3:5] * (1.0 - self.slip)))
        th = self.s[2]
        # exact integration for constant (v, w) over dt
        if abs(w) > 1e-9:
            self.s[0] += v / w * (math.sin(th + w * dt) - math.sin(th))
            self.s[1] += -v / w * (math.cos(th + w * dt) - math.cos(th))
        else:
            self.s[0] += v * dt * math.cos(th)
            self.s[1] += v * dt * math.sin(th)
        self.s[2] = wrap_angle(th + w * dt)


# ---------------------------------------------------------------------------------------------
# Car-like robots: front-steered, rear-driven (bicycle model about the rear axle)
#
# State convention for these robots:   s = [x, y, theta, v, delta]   (m, m, rad, m/s, rad)
# where (x, y) is the rear-axle point on the ground, v the rear-wheel ground speed and delta the
# equivalent single-track steering angle (the Ackermann pair is derived from it).
# Commands are (v_cmd [m/s], delta_cmd [rad]).
# ---------------------------------------------------------------------------------------------


@dataclass
class AckermannParams:
    wheelbase: float = 3.0  # m, rear axle -> front axle
    track: float = 2.0  # m, between the front wheels' steering axes
    wheel_radius: float = 0.4  # m, rear (driven) wheel
    max_speed: float = 3.0  # m/s
    max_accel: float = 1.5  # m/s^2
    max_steer: float = math.radians(35.0)  # rad
    max_steer_rate: float = math.radians(60.0)  # rad/s
    speed_time_constant: float = 0.3  # s, rear-wheel speed loop
    steer_time_constant: float = 0.15  # s, steering servo

    @property
    def min_turn_radius(self) -> float:
        return self.wheelbase / math.tan(self.max_steer)

    def steer_from_curvature(self, kappa: float) -> float:
        return math.atan(self.wheelbase * kappa)

    def ackermann_angles(self, delta: float) -> tuple[float, float]:
        """Single-track angle -> (left, right) front-wheel steering angles (positive = turn left)."""
        if abs(delta) < 1e-9:
            return 0.0, 0.0
        L, h = self.wheelbase, 0.5 * self.track
        R = L / math.tan(delta)  # signed turn radius of the rear-axle centre
        return math.atan(L / (R - h)), math.atan(L / (R + h))

    def delta_from_wheels(self, left: float, right: float) -> float:
        """Inverse of ``ackermann_angles`` (cot delta is the mean of the two cotangents)."""
        tl, tr = math.tan(left), math.tan(right)
        if abs(tl) < 1e-9 and abs(tr) < 1e-9:
            return 0.0
        denom = tl + tr
        return math.atan(2 * tl * tr / denom) if abs(denom) > 1e-12 else 0.0


class KinematicBicycle:
    """Offline stand-in for car-like robots: bicycle kinematics + speed lag + rate-limited steering."""

    is_car = True

    def __init__(self, params: AckermannParams | None = None, physics_dt: float = 1.0 / 60.0,
                 initial_state=(0.0, 0.0, 0.0), pose_noise_std: float = 0.0, seed: int = 0):
        self.params = params or AckermannParams()
        self.physics_dt = physics_dt
        self.pose_noise_std = pose_noise_std
        self.rng = np.random.default_rng(seed)
        x, y, th = initial_state
        self.s = np.array([x, y, th, 0.0, 0.0], dtype=float)
        self.cmd = np.zeros(2)

    def get_state(self) -> np.ndarray:
        s = self.s.copy()
        if self.pose_noise_std:
            s[:2] += self.rng.normal(0.0, self.pose_noise_std, 2)
            s[2] += self.rng.normal(0.0, self.pose_noise_std * 0.5)
        return s

    def set_command(self, v: float, delta: float) -> None:
        p = self.params
        self.cmd = np.array([np.clip(v, -p.max_speed, p.max_speed), np.clip(delta, -p.max_steer, p.max_steer)])

    def step(self) -> None:
        p, dt = self.params, self.physics_dt
        dv = np.clip((self.cmd[0] - self.s[3]) * dt / max(p.speed_time_constant, dt), -p.max_accel * dt, p.max_accel * dt)
        dd = np.clip((self.cmd[1] - self.s[4]) * dt / max(p.steer_time_constant, dt),
                     -p.max_steer_rate * dt, p.max_steer_rate * dt)
        self.s[3] += dv
        self.s[4] += dd
        v, th = self.s[3], self.s[2]
        w = v * math.tan(self.s[4]) / p.wheelbase
        self.s[0] += v * dt * math.cos(th + 0.5 * w * dt)
        self.s[1] += v * dt * math.sin(th + 0.5 * w * dt)
        self.s[2] = wrap_angle(th + w * dt)
