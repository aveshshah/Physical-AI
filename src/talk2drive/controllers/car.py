"""Tracking controllers for car-like (front-steered, rear-driven) robots.

Both return ``(v_cmd [m/s], delta_cmd [rad])`` for the bicycle model in ``talk2drive.robot``.

* ``CarNMPC``      CasADi + IPOPT NMPC on the bicycle model with speed lag, steering servo lag and
                   limits on speed, acceleration, steering angle and steering rate.
* ``CarKanayama``  Kanayama's (v, w) tracking law, converted to a steering angle with the bicycle
                   relation delta = atan(L w / v) - the natural baseline for the NMPC.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from ..references import Trajectory
from ..robot import AckermannParams, wrap_angle

COMMAND_NAMES = ("v_cmd", "steer_cmd")


def car_reference(traj: Trajectory, t0: float, dt: float, n: int, p: AckermannParams) -> np.ndarray:
    """(n+1, 5) reference states [x, y, theta, v, delta] along the trajectory."""
    rows = []
    for k in range(n + 1):
        r = traj.sample(t0 + k * dt)
        kappa = r.w / r.v if r.v > 1e-3 else 0.0
        delta = float(np.clip(p.steer_from_curvature(kappa), -p.max_steer, p.max_steer))
        rows.append([r.x, r.y, r.theta, r.v, delta])
    return np.array(rows)


@dataclass
class CarNMPCConfig:
    horizon: int = 20
    dt: float = 0.1  # s per step -> 2 s look-ahead (steering needs more preview than wheel speeds)
    q_pos: float = 50.0  # per m^2
    q_theta: float = 20.0
    q_v: float = 1.0  # reference speed (feed-forward)
    q_delta: float = 0.5  # reference steering (feed-forward)
    r_dv: float = 0.5  # command smoothness
    r_ddelta: float = 5.0
    terminal_factor: float = 5.0
    max_iter: int = 60


class CarNMPC:
    name = "nmpc"
    command_names = COMMAND_NAMES

    def __init__(self, params: AckermannParams | None = None, cfg: CarNMPCConfig | None = None):
        import casadi as ca

        self.ca = ca
        self.p = params or AckermannParams()
        self.cfg = cfg or CarNMPCConfig()
        self.control_dt = self.cfg.dt
        self._build()
        self.reset()

    def _dynamics(self, s, u):
        ca, p = self.ca, self.p
        return ca.vertcat(
            s[3] * ca.cos(s[2]),
            s[3] * ca.sin(s[2]),
            s[3] * ca.tan(s[4]) / p.wheelbase,
            (u[0] - s[3]) / p.speed_time_constant,
            (u[1] - s[4]) / p.steer_time_constant,
        )

    def _rk4(self, s, u, h):
        k1 = self._dynamics(s, u)
        k2 = self._dynamics(s + h / 2 * k1, u)
        k3 = self._dynamics(s + h / 2 * k2, u)
        k4 = self._dynamics(s + h * k3, u)
        return s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def _build(self):
        ca, c, p = self.ca, self.cfg, self.p
        N = c.horizon
        opti = ca.Opti()
        S, U = opti.variable(5, N + 1), opti.variable(2, N)
        s0, R = opti.parameter(5), opti.parameter(5, N + 1)
        cost = 0
        for k in range(N + 1):
            wgt = c.terminal_factor if k == N else 1.0
            cost += wgt * c.q_pos * ca.sumsqr(S[0:2, k] - R[0:2, k])
            cost += wgt * c.q_theta * 2.0 * (1 - ca.cos(S[2, k] - R[2, k]))
            cost += c.q_v * (S[3, k] - R[3, k]) ** 2 + c.q_delta * (S[4, k] - R[4, k]) ** 2
            if 0 < k < N:
                cost += c.r_dv * (U[0, k] - U[0, k - 1]) ** 2 + c.r_ddelta * (U[1, k] - U[1, k - 1]) ** 2
        opti.minimize(cost)
        opti.subject_to(S[:, 0] == s0)
        for k in range(N):
            opti.subject_to(S[:, k + 1] == self._rk4(S[:, k], U[:, k], c.dt))
        opti.subject_to(opti.bounded(-p.max_speed, U[0, :], p.max_speed))
        opti.subject_to(opti.bounded(-p.max_steer, U[1, :], p.max_steer))
        opti.subject_to(opti.bounded(-p.max_steer, S[4, 1:], p.max_steer))
        # acceleration and steering-rate limits through the first-order loops
        opti.subject_to(opti.bounded(-p.max_accel * p.speed_time_constant, U[0, :] - S[3, :-1],
                                     p.max_accel * p.speed_time_constant))
        opti.subject_to(opti.bounded(-p.max_steer_rate * p.steer_time_constant, U[1, :] - S[4, :-1],
                                     p.max_steer_rate * p.steer_time_constant))
        opti.solver("ipopt", {"print_time": False, "expand": True},
                    {"print_level": 0, "max_iter": c.max_iter, "tol": 1e-4, "acceptable_tol": 1e-3,
                     "warm_start_init_point": "yes", "sb": "yes"})
        self.opti, self.S, self.U, self.s0, self.R = opti, S, U, s0, R

    def reset(self) -> None:
        self._S_guess = self._U_guess = None
        self.last_solve_time = 0.0
        self.last_status = "init"
        self.last_prediction: np.ndarray | None = None

    def compute(self, t: float, state: np.ndarray, traj: Trajectory) -> tuple[float, float]:
        c, p = self.cfg, self.p
        ref = car_reference(traj, t, c.dt, c.horizon, p)
        ref[:, 2] = np.unwrap(np.concatenate([[state[2]], ref[:, 2]]))[1:]
        state = np.asarray(state, dtype=float).copy()
        state[4] = np.clip(state[4], -p.max_steer, p.max_steer)
        self.opti.set_value(self.s0, state)
        self.opti.set_value(self.R, ref.T)
        if self._S_guess is None:
            S_guess = ref.T.copy()
            S_guess[:, 0] = state
            U_guess = ref[:-1, 3:5].T.copy()
        else:
            S_guess = np.hstack([self._S_guess[:, 1:], self._S_guess[:, -1:]])
            S_guess[:, 0] = state
            U_guess = np.hstack([self._U_guess[:, 1:], self._U_guess[:, -1:]])
        self.opti.set_initial(self.S, S_guess)
        self.opti.set_initial(self.U, U_guess)
        t0 = time.perf_counter()
        try:
            sol = self.opti.solve()
            S_opt, U_opt = sol.value(self.S), sol.value(self.U)
            self.last_status = "ok"
        except RuntimeError:
            S_opt, U_opt = self.opti.debug.value(self.S), self.opti.debug.value(self.U)
            self.last_status = "max_iter"
        self.last_solve_time = time.perf_counter() - t0
        self._S_guess, self._U_guess = np.asarray(S_opt), np.asarray(U_opt).reshape(2, -1)
        self.last_prediction = self._S_guess.T.copy()
        return float(self._U_guess[0, 0]), float(self._U_guess[1, 0])


@dataclass
class CarKanayamaConfig:
    kx: float = 1.0  # 1/s
    ky: float = 1.0  # 1/m^2  (Jetbot: 40; gains scale with 1 / size^2)
    ktheta: float = 1.5  # 1/m  (Jetbot: 6)
    dt: float = 0.1


class CarKanayama:
    name = "kanayama"
    command_names = COMMAND_NAMES

    def __init__(self, params: AckermannParams | None = None, cfg: CarKanayamaConfig | None = None):
        self.p = params or AckermannParams()
        self.cfg = cfg or CarKanayamaConfig()
        self.control_dt = self.cfg.dt
        self.last_solve_time, self.last_status, self.last_prediction = 0.0, "ok", None

    def reset(self) -> None:
        pass

    def compute(self, t: float, state: np.ndarray, traj: Trajectory) -> tuple[float, float]:
        c, p = self.cfg, self.p
        ref = traj.sample(t)
        x, y, th = state[0], state[1], state[2]
        dx, dy = ref.x - x, ref.y - y
        ex = math.cos(th) * dx + math.sin(th) * dy
        ey = -math.sin(th) * dx + math.cos(th) * dy
        eth = wrap_angle(ref.theta - th)
        v = float(np.clip(ref.v * math.cos(eth) + c.kx * ex, -p.max_speed, p.max_speed))
        w = ref.w + ref.v * (c.ky * ey + c.ktheta * math.sin(eth))
        v_eff = max(abs(v), 0.05)  # steering is undefined at standstill
        delta = math.atan(p.wheelbase * w / v_eff) * (1 if v >= 0 else -1)
        return v, float(np.clip(delta, -p.max_steer, p.max_steer))
