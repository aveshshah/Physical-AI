"""Classic Kanayama trajectory-tracking controller: the baseline the NMPC has to beat.

Y. Kanayama et al., "A stable tracking control method for an autonomous mobile robot", ICRA 1990.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..references import Trajectory
from ..robot import DiffDriveParams, wrap_angle


@dataclass
class KanayamaConfig:
    kx: float = 1.5  # 1/s
    ky: float = 40.0  # 1/m^2
    ktheta: float = 6.0  # 1/m
    dt: float = 0.05


class KanayamaController:
    name = "kanayama"

    def __init__(self, params: DiffDriveParams | None = None, cfg: KanayamaConfig | None = None):
        self.p = params or DiffDriveParams()
        self.cfg = cfg or KanayamaConfig()
        self.control_dt = self.cfg.dt
        self.last_solve_time = 0.0
        self.last_status = "ok"
        self.last_prediction = None

    def reset(self) -> None:
        pass

    def compute(self, t: float, state: np.ndarray, traj: Trajectory) -> tuple[float, float]:
        c = self.cfg
        ref = traj.sample(t)
        x, y, th = state[0], state[1], state[2]
        dx, dy = ref.x - x, ref.y - y
        ex = math.cos(th) * dx + math.sin(th) * dy
        ey = -math.sin(th) * dx + math.cos(th) * dy
        eth = wrap_angle(ref.theta - th)
        v = ref.v * math.cos(eth) + c.kx * ex
        w = ref.w + ref.v * (c.ky * ey + c.ktheta * math.sin(eth))
        return self.p.saturate_wheels(*self.p.body_to_wheels(v, w))
