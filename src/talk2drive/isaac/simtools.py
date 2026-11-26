"""Small helpers shared by the sim/ scripts (need a running Isaac Sim app)."""

from __future__ import annotations

import base64
import io
import json
import math
import urllib.request
from pathlib import Path

import numpy as np

from talk2drive.isaac.jetbot import drive_surface_height


def road_height(x, y) -> np.ndarray:
    """Height of the drivable surface under scene points (array, NaN off the grid)."""
    return drive_surface_height(np.atleast_1d(x), np.atleast_1d(y))


def chase_view(x: float, y: float, psi: float, z: float, back: float, up: float, ahead: float, target_up: float) -> None:
    """Point the viewport from ``back`` m behind and ``up`` m above (x, y, z) at ``ahead`` m in front."""
    from isaacsim.core.utils.viewports import set_camera_view

    c, s = math.cos(psi), math.sin(psi)
    set_camera_view(eye=[x - back * c, y - back * s, z + up], target=[x + ahead * c, y + ahead * s, z + target_up])


class VlmClient:
    """Client of scripts/vlm_ground_server.py."""

    def __init__(self, url: str):
        self.url = url

    def health(self) -> dict:
        return json.loads(urllib.request.urlopen(self.url + "/health", timeout=5).read())

    def ask(self, image, query: str = "", prompt: str | None = None, timeout: float = 180) -> dict:
        """POST a PIL image; the server answers {"boxes", "raw", "ms"}."""
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        body = {"image_png_b64": base64.b64encode(buf.getvalue()).decode(), "query": query}
        if prompt is not None:
            body["prompt"] = prompt
        req = urllib.request.Request(self.url + "/ground", method="POST", headers={"Content-Type": "application/json"},
                                     data=json.dumps(body).encode())
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())


class ViewportRecorder:
    """Saves the viewport every ``every`` s of sim time to ``frames/`` and turns the frames into a GIF."""

    def __init__(self, frames: Path, every: float):
        from omni.kit.viewport.utility import get_active_viewport

        self.frames, self.every, self.n = frames, every, 0
        frames.mkdir(parents=True, exist_ok=True)
        for f in frames.glob("*.png"):
            f.unlink()
        self.vp = get_active_viewport()

    def capture(self, t: float) -> None:
        from omni.kit.viewport.utility import capture_viewport_to_file

        if t + 1e-9 >= self.n * self.every:
            capture_viewport_to_file(self.vp, file_path=str(self.frames / f"{self.n:04d}.png"))
            self.n += 1

    def save_gif(self, app, gif: Path, duration_ms: int) -> bool:
        """Let the last captures finish writing, then write the GIF (640x360). False if there were no frames."""
        from PIL import Image

        for _ in range(10):
            app.update()
        files = sorted(self.frames.glob("*.png"))
        if not files:
            return False
        imgs = [Image.open(f).convert("RGB").resize((640, 360)) for f in files]
        imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=duration_ms, loop=0)
        return True
