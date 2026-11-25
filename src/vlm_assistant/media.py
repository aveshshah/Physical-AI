"""Image, video-file and webcam input handling.

Everything is normalised to two media types that the backends understand:

* :class:`ImageItem` - a single RGB ``PIL.Image``.
* :class:`VideoItem` - a list of RGB frames with their timestamps (seconds).

Frames are sampled and downscaled here, once, so every backend sees the same input.
"""

from __future__ import annotations

import io
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from .config import MediaConfig

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}


@dataclass
class ImageItem:
    image: Image.Image
    source: str = "image"

    def describe(self) -> str:
        w, h = self.image.size
        return f"image from {self.source} ({w}x{h})"


@dataclass
class VideoItem:
    frames: list[Image.Image]
    timestamps: list[float]
    source: str = "video"
    # Effective sampling rate of ``frames`` (frames per second of real time).
    fps: float = 2.0
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        if not self.timestamps:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]

    def describe(self) -> str:
        return (
            f"video from {self.source} ({len(self.frames)} frames, "
            f"{self.duration:.1f}s, ~{self.fps:.2g} fps)"
        )


MediaItem = Union[ImageItem, VideoItem]


# ----------------------------------------------------------------------------- helpers
def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    """Downscale so that max(width, height) <= max_side (never upscales)."""
    w, h = img.size
    longest = max(w, h)
    if max_side <= 0 or longest <= max_side:
        return img
    scale = max_side / float(longest)
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BICUBIC)


def bgr_to_pil(frame: np.ndarray) -> Image.Image:
    """OpenCV BGR (or grayscale) ndarray -> RGB PIL image."""
    if frame.ndim == 2:
        return Image.fromarray(frame).convert("RGB")
    return Image.fromarray(np.ascontiguousarray(frame[:, :, 2::-1]))


def pil_to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _even_frames(frames: list, timestamps: list[float]) -> tuple[list, list[float]]:
    """Qwen3-VL groups frames in temporal pairs; pad with the last frame to an even count."""
    if frames and len(frames) % 2 == 1:
        frames = frames + [frames[-1]]
        timestamps = timestamps + [timestamps[-1]]
    return frames, timestamps


def is_url(s: str) -> bool:
    return s.startswith(("http://", "https://"))


def guess_kind(path: str) -> str:
    """Return "image", "video" or "unknown" based on the file extension."""
    ext = Path(path.split("?")[0]).suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    return "unknown"


# ----------------------------------------------------------------------------- images
def load_image(src: str | Path | Image.Image | np.ndarray, cfg: MediaConfig | None = None) -> ImageItem:
    cfg = cfg or MediaConfig()
    source = "array"
    if isinstance(src, Image.Image):
        img, source = src, "pil"
    elif isinstance(src, np.ndarray):
        img = Image.fromarray(src)
    else:
        source = str(src)
        if is_url(source):
            import requests

            resp = requests.get(source, timeout=30)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content))
        else:
            path = Path(source).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Image not found: {path}")
            img = Image.open(path)
            img.load()
    img = resize_max_side(img.convert("RGB"), cfg.image_max_side)
    return ImageItem(image=img, source=source)


# ----------------------------------------------------------------------------- video files
def sample_video(
    path: str | Path,
    cfg: MediaConfig | None = None,
    start: float | None = None,
    end: float | None = None,
    fps: float | None = None,
    max_frames: int | None = None,
) -> VideoItem:
    """Uniformly sample frames from a video file.

    Frames are taken at ``fps`` (default ``cfg.video_fps``) but never more than ``max_frames``;
    for long videos the effective rate drops so the whole [start, end] window is still covered.
    """
    import cv2

    cfg = cfg or MediaConfig()
    fps = fps or cfg.video_fps
    max_frames = max_frames or cfg.video_max_frames
    path = str(Path(path).expanduser()) if not is_url(str(path)) else str(path)
    if not is_url(path) and not Path(path).is_file():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path}")
    try:
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if not native_fps or math.isnan(native_fps) or native_fps <= 0:
            native_fps = 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            # Some containers do not report a frame count; count by decoding.
            total = 0
            while cap.grab():
                total += 1
            cap.release()
            cap = cv2.VideoCapture(path)
        if total <= 0:
            raise RuntimeError(f"Video has no decodable frames: {path}")

        full_duration = total / native_fps
        t0 = max(0.0, start or 0.0)
        t1 = min(full_duration, end if end is not None else full_duration)
        if t1 <= t0:
            raise ValueError(f"Empty time window [{t0}, {t1}] for video of {full_duration:.2f}s")
        f0 = int(t0 * native_fps)
        f1 = max(f0, min(total - 1, int(math.ceil(t1 * native_fps)) - 1))

        n = max(1, min(max_frames, int(math.ceil((t1 - t0) * fps))))
        if n == 1:
            wanted = [(f0 + f1) // 2]
        else:
            wanted = sorted({round(f0 + i * (f1 - f0) / (n - 1)) for i in range(n)})

        frames: list[Image.Image] = []
        stamps: list[float] = []
        wanted_set = set(wanted)
        last = wanted[-1]
        if f0 > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
            idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) or f0
        else:
            idx = 0
        while idx <= last:
            ok = cap.grab()
            if not ok:
                break
            if idx in wanted_set:
                ok, frame = cap.retrieve()
                if ok:
                    frames.append(resize_max_side(bgr_to_pil(frame), cfg.video_frame_max_side))
                    stamps.append(idx / native_fps)
            idx += 1
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"Could not decode any frames from {path}")
    eff_fps = (len(frames) - 1) / (stamps[-1] - stamps[0]) if len(frames) > 1 and stamps[-1] > stamps[0] else fps
    frames, stamps = _even_frames(frames, stamps)
    return VideoItem(
        frames=frames,
        timestamps=stamps,
        source=path,
        fps=eff_fps,
        meta={"native_fps": native_fps, "total_frames": total, "duration": full_duration},
    )


def load_media(src: str, cfg: MediaConfig | None = None) -> MediaItem:
    """Load an image or a video depending on the file extension."""
    kind = guess_kind(src)
    if kind == "video":
        return sample_video(src, cfg)
    return load_image(src, cfg)


# ----------------------------------------------------------------------------- live frames
class FrameBuffer:
    """Thread-safe store of the latest frame plus a rolling, time-stamped buffer of recent frames.

    Frames are RGB ``numpy`` arrays (H, W, 3). ``push`` can be called at any rate; frames are kept
    in the rolling buffer at most ``fps`` times per second for ``seconds`` seconds.
    """

    def __init__(self, cfg: MediaConfig | None = None):
        self.cfg = cfg or MediaConfig()
        self._lock = threading.Lock()
        self._latest: tuple[float, np.ndarray] | None = None
        maxlen = max(2, int(math.ceil(self.cfg.webcam_buffer_seconds * self.cfg.webcam_buffer_fps)) + 1)
        self._buffer: deque[tuple[float, np.ndarray]] = deque(maxlen=maxlen)
        self._period = 1.0 / max(self.cfg.webcam_buffer_fps, 1e-3)
        self._last_buffered = -1e9

    def push(self, frame_rgb: np.ndarray, t: float | None = None) -> None:
        t = time.monotonic() if t is None else t
        with self._lock:
            self._latest = (t, frame_rgb)
            if t - self._last_buffered >= self._period - 1e-6:
                self._buffer.append((t, frame_rgb))
                self._last_buffered = t

    def clear(self) -> None:
        with self._lock:
            self._latest = None
            self._buffer.clear()
            self._last_buffered = -1e9

    @property
    def has_frames(self) -> bool:
        return self._latest is not None

    def latest_frame(self) -> np.ndarray:
        with self._lock:
            if self._latest is None:
                raise RuntimeError("No camera frame available yet")
            return self._latest[1].copy()

    def snapshot(self, source: str = "webcam") -> ImageItem:
        img = resize_max_side(Image.fromarray(self.latest_frame()), self.cfg.image_max_side)
        return ImageItem(image=img, source=source)

    def clip(self, seconds: float | None = None, max_frames: int | None = None, source: str = "webcam") -> VideoItem:
        """Frames from the last ``seconds`` seconds, uniformly thinned to ``max_frames``."""
        seconds = seconds or self.cfg.webcam_buffer_seconds
        max_frames = max_frames or self.cfg.video_max_frames
        with self._lock:
            if not self._buffer:
                raise RuntimeError("Camera buffer is empty")
            now = self._buffer[-1][0]
            items = [(t, f) for (t, f) in self._buffer if now - t <= seconds + 1e-6]
        if len(items) > max_frames:
            step = len(items) / max_frames
            items = [items[int(i * step)] for i in range(max_frames - 1)] + [items[-1]]
        t0 = items[0][0]
        frames = [resize_max_side(Image.fromarray(f), self.cfg.video_frame_max_side) for _, f in items]
        stamps = [t - t0 for t, _ in items]
        fps = (len(stamps) - 1) / stamps[-1] if len(stamps) > 1 and stamps[-1] > 0 else self.cfg.webcam_buffer_fps
        frames, stamps = _even_frames(frames, stamps)
        return VideoItem(frames=frames, timestamps=stamps, source=source, fps=fps)

    def buffered_seconds(self) -> float:
        with self._lock:
            if len(self._buffer) < 2:
                return 0.0
            return self._buffer[-1][0] - self._buffer[0][0]


# ----------------------------------------------------------------------------- webcam
class Webcam:
    """Background-threaded camera reader feeding a :class:`FrameBuffer`.

    Works with USB webcams (int index), video files, RTSP/HTTP streams or GStreamer pipelines,
    which is what you will want on a robot (e.g. a CSI camera on a Jetson).
    """

    def __init__(self, cfg: MediaConfig | None = None, capture_factory=None):
        self.cfg = cfg or MediaConfig()
        self._capture_factory = capture_factory  # injectable for tests
        self._cap = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._first_frame = threading.Event()
        self.buffer = FrameBuffer(self.cfg)
        self.error: str | None = None

    # -- lifecycle
    def _open(self):
        if self._capture_factory is not None:
            return self._capture_factory()
        import cv2

        src = self.cfg.webcam_source
        if isinstance(src, str) and src.isdigit():
            src = int(src)
        cap = cv2.VideoCapture(src)
        if isinstance(src, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.webcam_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.webcam_height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def start(self, timeout: float = 5.0) -> "Webcam":
        if self.running:
            return self
        self._cap = self._open()
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open camera source {self.cfg.webcam_source!r}")
        self._stop.clear()
        self._first_frame.clear()
        self.error = None
        self._thread = threading.Thread(target=self._loop, name="webcam-reader", daemon=True)
        self._thread.start()
        if not self._first_frame.wait(timeout):
            self.stop()
            raise RuntimeError(f"No frames received from camera {self.cfg.webcam_source!r} within {timeout}s")
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        if self._cap is not None:
            self._cap.release()
        self._cap = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def __enter__(self) -> "Webcam":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def _loop(self) -> None:
        failures = 0
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if not ok or frame is None:
                failures += 1
                if failures > 50:
                    self.error = "camera stopped delivering frames"
                    break
                time.sleep(0.02)
                continue
            failures = 0
            rgb = frame[:, :, 2::-1] if frame.ndim == 3 else np.stack([frame] * 3, axis=-1)
            self.buffer.push(np.ascontiguousarray(rgb))
            self._first_frame.set()

    # -- access (delegates to the buffer)
    def latest_frame(self) -> np.ndarray:
        """Latest frame as an RGB array."""
        return self.buffer.latest_frame()

    def snapshot(self) -> ImageItem:
        return self.buffer.snapshot("webcam")

    def clip(self, seconds: float | None = None, max_frames: int | None = None) -> VideoItem:
        """Return the frames buffered during the last ``seconds`` seconds."""
        return self.buffer.clip(seconds, max_frames, "webcam")

    def buffered_seconds(self) -> float:
        return self.buffer.buffered_seconds()
