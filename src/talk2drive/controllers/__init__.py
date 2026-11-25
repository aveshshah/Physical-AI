from __future__ import annotations

from ..robot import DiffDriveParams
from .baseline import KanayamaConfig, KanayamaController


def make_controller(name: str, params: DiffDriveParams | None = None):
    if name == "nmpc":
        from .nmpc import NMPCController  # needs casadi

        return NMPCController(params)
    if name == "kanayama":
        return KanayamaController(params)
    raise ValueError(f"Unknown controller {name!r} (nmpc | kanayama)")


__all__ = ["make_controller", "KanayamaController", "KanayamaConfig"]
