from __future__ import annotations

from ..robot import AckermannParams, DiffDriveParams
from .baseline import KanayamaConfig, KanayamaController


def make_controller(name: str, params: DiffDriveParams | AckermannParams | None = None):
    if isinstance(params, AckermannParams):  # car-like robot: (speed, steering) commands
        from .car import CarKanayama, CarNMPC

        if name == "nmpc":
            return CarNMPC(params)
        if name == "kanayama":
            return CarKanayama(params)
        raise ValueError(f"Unknown controller {name!r} (nmpc | kanayama)")
    if name == "nmpc":
        from .nmpc import NMPCController  # needs casadi

        return NMPCController(params)
    if name == "kanayama":
        return KanayamaController(params)
    raise ValueError(f"Unknown controller {name!r} (nmpc | kanayama)")


__all__ = ["make_controller", "KanayamaController", "KanayamaConfig"]
