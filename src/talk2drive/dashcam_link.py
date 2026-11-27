"""The link between the simulation and the dash cam recorder (scripts/dashcam.py).

Messages go over a Unix socket. Each message is:
    4 bytes: length of the header (big-endian)
    header:  JSON, e.g. {"type": "frame", "index": 12, "sim_t": 1.2, ..., "size": 691200}
    payload: "size" bytes (for a frame: the RGB pixels, height x width x 3)

Message types: "hello" (a simulation starts), "frame", "event" (collision, rollover, manual save),
"bye" (the simulation ends).

DashcamFeed is used by the simulation. It never makes the simulation wait: frames go into a small
queue and a background thread sends them. If the recorder is slow or not running, frames are
dropped and counted (their numbers are skipped, so the recorder can see the gap).
"""

from __future__ import annotations

import json
import os
import queue
import socket
import struct
import threading
import time


def default_socket() -> str:
    return os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"), "talk2drive-dashcam.sock")


def send(sock, header: dict, payload: bytes = b"") -> None:
    header = dict(header, size=len(payload))
    text = json.dumps(header).encode()
    sock.sendall(struct.pack(">I", len(text)) + text + payload)


def _read_exactly(sock, n: int):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data


def receive(sock):
    """The next (header, payload), or (None, None) when the other side has closed the connection."""
    size = _read_exactly(sock, 4)
    if size is None:
        return None, None
    text = _read_exactly(sock, struct.unpack(">I", size)[0])
    if text is None:
        return None, None
    header = json.loads(text)
    payload = _read_exactly(sock, header.get("size", 0)) if header.get("size", 0) else b""
    if payload is None:
        return None, None
    return header, payload


def ask(header: dict, path: str | None = None, timeout: float = 5.0) -> dict:
    """Send one message that wants an answer (used by `dashcam.py trigger`); returns the answer."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(path or default_socket())
        send(sock, dict(header, want_reply=True))
        answer, _ = receive(sock)
        return answer or {}


class DashcamFeed:
    """Simulation side: hands camera frames and events to the recorder without blocking."""

    def __init__(self, path: str | None = None, fps: float = 10.0, queue_size: int = 30):
        self.path = path or default_socket()
        self.fps = fps
        self.queue = queue.Queue(maxsize=queue_size)
        self.index = 0          # number of the next frame (dropped frames use up a number too)
        self.sent = 0
        self.dropped = 0
        self.connected = False
        self.sock = None
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def frame(self, rgb, sim_t: float) -> None:
        """Queue one RGB frame (numpy uint8, height x width x 3) taken at simulation time sim_t."""
        header = {"type": "frame", "index": self.index, "sim_t": float(sim_t), "wall_t": time.time(),
                  "mono_t": time.monotonic(), "height": int(rgb.shape[0]), "width": int(rgb.shape[1])}
        self.index += 1
        if not self.connected:
            self.dropped += 1
            return
        try:
            self.queue.put_nowait((header, rgb.tobytes()))
        except queue.Full:
            self.dropped += 1

    def event(self, kind: str, sim_t: float, note: str = "") -> bool:
        """Tell the recorder to save an incident. Returns False if the recorder is not running."""
        if not self.connected:
            return False
        header = {"type": "event", "kind": kind, "sim_t": float(sim_t), "wall_t": time.time(), "note": note}
        while True:
            try:
                self.queue.put_nowait((header, b""))
                return True
            except queue.Full:   # an event matters more than a frame: make room
                try:
                    self.queue.get_nowait()
                    self.dropped += 1
                except queue.Empty:
                    pass

    def close(self) -> None:
        if self.connected:
            try:
                self.queue.put(({"type": "bye", "dropped": self.dropped}, b""), timeout=1.0)
            except queue.Full:
                pass
            deadline = time.time() + 3.0
            while not self.queue.empty() and time.time() < deadline:
                time.sleep(0.05)
        self.running = False
        self.thread.join(timeout=2.0)

    # -------- background thread
    def _connect(self) -> bool:
        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.connect(self.path)
            send(sock, {"type": "hello", "fps": self.fps, "first_index": self.index})
        except OSError:
            return False
        self.sock, self.connected = sock, True
        print(f"[dashcam] connected to the recorder ({self.path})", flush=True)
        return True

    def _run(self) -> None:
        last_try = 0.0
        while self.running:
            if not self.connected:
                if time.time() - last_try > 1.0:
                    last_try = time.time()
                    self._connect()
                time.sleep(0.05)
                continue
            try:
                header, payload = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                send(self.sock, header, payload)
                self.sent += header["type"] == "frame"
            except OSError:
                print("[dashcam] lost the recorder; frames are dropped until it is back", flush=True)
                self.connected = False
                self.sock.close()
        if self.sock is not None:
            self.sock.close()
