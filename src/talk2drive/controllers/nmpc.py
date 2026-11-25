"""Nonlinear MPC for a differential-drive robot (CasADi + IPOPT).

Model (continuous time), state s = [x, y, theta, wL, wR], input u = [uL, uR] = wheel-speed commands
(exactly what the robot / Isaac Sim velocity drive accepts):

    v      = r (wL + wR) / 2
    w      = r (wR - wL) / L
    x'     = v cos(theta)
    y'     = v sin(theta)
    theta' = w
    wL'    = (uL - wL) / tau        first-order wheel-speed loop (motor + drive controller)
    wR'    = (uR - wR) / tau

Discretised with RK4, multiple shooting, horizon N. Constraints: |wheel speed|, |command| <= w_max
and |wheel accel| = |u - w| / tau <= a_max, i.e. the motor speed and torque limits. M2 will extend the model
with motor current, battery power and an energy cost.

Heading error uses 1 - cos(dtheta), so it is smooth and free of +/-pi wrap-around problems.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..references import Trajectory
from ..robot import DiffDriveParams


@dataclass
class NMPCConfig:
    horizon: int = 20  # steps
    dt: float = 0.05  # s per step  -> 1.0 s look-ahead
    q_pos: float = 200.0  # position error weight  (per m^2)
    q_theta: float = 20.0  # heading weight (on 1 - cos)
    q_wheel: float = 0.02  # wheel-speed tracking weight (feed-forward)
    r_du: float = 0.01  # weight on command changes (smoothness)
    terminal_factor: float = 5.0
    max_iter: int = 60


class NMPCController:
    name = "nmpc"

    def __init__(self, params: DiffDriveParams | None = None, cfg: NMPCConfig | None = None):
        import casadi as ca

        self.ca = ca
        self.p = params or DiffDriveParams()
        self.cfg = cfg or NMPCConfig()
        self.control_dt = self.cfg.dt
        self._build()
        self.reset()

    # ------------------------------------------------------------------ problem
    def _dynamics(self, s, u):
        ca = self.ca
        r, L = self.p.wheel_radius, self.p.wheel_base
        v = 0.5 * r * (s[3] + s[4])
        w = r * (s[4] - s[3]) / L
        tau = max(self.p.wheel_time_constant, 1e-3)
        return ca.vertcat(v * ca.cos(s[2]), v * ca.sin(s[2]), w, (u[0] - s[3]) / tau, (u[1] - s[4]) / tau)

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
        S = opti.variable(5, N + 1)
        U = opti.variable(2, N)
        s0 = opti.parameter(5)
        R = opti.parameter(5, N + 1)

        cost = 0
        for k in range(N + 1):
            wgt = c.terminal_factor if k == N else 1.0
            e = S[0:2, k] - R[0:2, k]
            cost += wgt * c.q_pos * ca.sumsqr(e)
            cost += wgt * c.q_theta * 2.0 * (1 - ca.cos(S[2, k] - R[2, k]))
            cost += c.q_wheel * ca.sumsqr(S[3:5, k] - R[3:5, k])
            if 0 < k < N:
                cost += c.r_du * ca.sumsqr(U[:, k] - U[:, k - 1])
        opti.minimize(cost)

        opti.subject_to(S[:, 0] == s0)
        for k in range(N):
            opti.subject_to(S[:, k + 1] == self._rk4(S[:, k], U[:, k], c.dt))
        opti.subject_to(opti.bounded(-p.max_wheel_speed, ca.vec(S[3:5, 1:]), p.max_wheel_speed))
        opti.subject_to(opti.bounded(-p.max_wheel_speed, ca.vec(U), p.max_wheel_speed))
        tau = max(p.wheel_time_constant, 1e-3)
        for k in range(N):  # torque limit: wheel acceleration of the first-order loop
            opti.subject_to(opti.bounded(-p.max_wheel_accel * tau, U[:, k] - S[3:5, k], p.max_wheel_accel * tau))

        opti.solver(
            "ipopt",
            {"print_time": False, "expand": True},
            {"print_level": 0, "max_iter": c.max_iter, "tol": 1e-4, "acceptable_tol": 1e-3,
             "warm_start_init_point": "yes", "sb": "yes"},
        )
        self.opti, self.S, self.U, self.s0, self.R = opti, S, U, s0, R

    # ------------------------------------------------------------------ runtime
    def reset(self) -> None:
        self._S_guess = None
        self._U_guess = None
        self.last_solve_time = 0.0
        self.last_status = "init"
        self.last_prediction: np.ndarray | None = None

    def compute(self, t: float, state: np.ndarray, traj: Trajectory) -> tuple[float, float]:
        c, p = self.cfg, self.p
        ref = traj.horizon(t, c.dt, c.horizon, p)  # (N+1, 5)
        # unwrap reference heading next to the measured heading (cost uses cos, but the
        # initial guess and logs are nicer without 2*pi jumps)
        ref[:, 2] = np.unwrap(np.concatenate([[state[2]], ref[:, 2]]))[1:]
        state = np.asarray(state, dtype=float).copy()
        state[3:5] = np.clip(state[3:5], -p.max_wheel_speed, p.max_wheel_speed)

        self.opti.set_value(self.s0, state)
        self.opti.set_value(self.R, ref.T)
        if self._S_guess is None:
            S_guess = ref.T.copy()
            S_guess[:, 0] = state
            U_guess = ref[:-1, 3:5].T.copy()
        else:  # shift previous solution by one step
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
            # Not converged within max_iter: use the last iterate (usually still good).
            S_opt, U_opt = self.opti.debug.value(self.S), self.opti.debug.value(self.U)
            self.last_status = "max_iter"
        self.last_solve_time = time.perf_counter() - t0

        self._S_guess, self._U_guess = np.asarray(S_opt), np.asarray(U_opt).reshape(2, -1)
        self.last_prediction = self._S_guess.T.copy()
        wl, wr = self._U_guess[0, 0], self._U_guess[1, 0]
        return p.saturate_wheels(float(wl), float(wr))
