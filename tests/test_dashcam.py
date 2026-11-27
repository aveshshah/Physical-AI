"""Dash cam recorder (scripts/dashcam.py) with synthetic frames: no simulator needed.

Small frames at 10 per simulated second are fed straight into the Recorder, so minutes of
simulated driving take a few seconds.
"""

import importlib.util
import json
import os
import socket
import stat
import threading
import time
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dashcam", REPO / "scripts" / "dashcam.py")
dashcam = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dashcam)

W, H, FPS = 64, 36, 10


def feed(recorder, t_from, t_to, skip=(), start_index=None):
    """Frames from t_from to t_to (simulated seconds); frame numbers in ``skip`` are lost on the way."""
    for i in range(int(round(t_from * FPS)), int(round(t_to * FPS))):
        if i in skip:
            continue
        t = i / FPS
        pixels = np.full((H, W, 3), i % 255, np.uint8).tobytes()
        recorder.frame({"index": i, "sim_t": t, "wall_t": time.time(), "mono_t": time.monotonic(),
                        "width": W, "height": H}, pixels)


def manifests(store):
    return [json.loads((f / "manifest.json").read_text()) for f in sorted((store / "incidents").iterdir())]


def unlock(store):   # let pytest clean the temporary folder
    for folder in (store / "incidents").iterdir():
        folder.chmod(stat.S_IRWXU)


@pytest.fixture
def store(tmp_path):
    yield tmp_path
    unlock(tmp_path)


def make(store):
    """A recorder with 1-second videos and a 90 s loop, so the tests run quickly."""
    r = dashcam.Recorder(store, keep=90.0, segment_length=1.0, log=lambda *_: None)
    r.start_session(FPS)
    return r


def test_collision_keeps_60s_before_and_after(store):
    r = make(store)
    feed(r, 0, 70)
    r.event({"kind": "collision", "sim_t": 70.0})
    feed(r, 70, 140)
    (m,) = manifests(store)
    assert m["status"] == "complete", m["problems"]
    assert m["window_needed"] == [10.0, 130.0]
    assert m["window_recorded"][0] <= 10.0 and m["window_recorded"][1] >= 129.9
    assert m["frames"] >= 1200                       # 120 s at 10 frames/s
    folder = store / "incidents" / m["incident"]
    assert dashcam.verify_folder(folder) == (True, [])
    # loop recording: about 90 s of segments left in the rolling folder
    assert 88 <= len(list((store / "rolling").glob("*.mp4"))) <= 92


def test_protected_files_cannot_be_deleted_or_changed(store):
    r = make(store)
    feed(r, 0, 70)
    r.event({"kind": "manual", "sim_t": 65.0})
    feed(r, 70, 130)
    folder = next((store / "incidents").iterdir())
    video = folder / "incident.mp4"
    with pytest.raises(PermissionError):
        video.unlink()
    with pytest.raises(PermissionError):
        open(video, "ab")
    # rolling cleanup goes on for a long time and never touches the incident
    feed(r, 130, 400)
    assert dashcam.verify_folder(folder) == (True, [])
    # someone forces a change: verify notices
    folder.chmod(stat.S_IRWXU)
    video.chmod(stat.S_IRUSR | stat.S_IWUSR)
    with open(video, "ab") as f:
        f.write(b"x")
    ok, findings = dashcam.verify_folder(folder)
    assert not ok and any("incident.mp4 was changed" in f for f in findings)


def test_early_trigger_is_marked_incomplete(store):
    r = make(store)
    feed(r, 0, 20)
    r.event({"kind": "collision", "sim_t": 20.0})
    feed(r, 20, 90)
    (m,) = manifests(store)
    assert m["status"] == "incomplete"
    assert any("starts at 0.0 s" in p for p in m["problems"])


def test_second_collision_extends_the_window(store):
    r = make(store)
    feed(r, 0, 70)
    r.event({"kind": "collision", "sim_t": 70.0})
    feed(r, 70, 100)
    r.event({"kind": "collision", "sim_t": 100.0})
    feed(r, 100, 170)
    (m,) = manifests(store)
    assert m["status"] == "complete", m["problems"]
    assert m["window_needed"] == [10.0, 160.0] and len(m["events"]) == 2
    assert m["window_recorded"][1] >= 159.9


def test_early_exit_is_marked_incomplete(store):
    r = make(store)
    feed(r, 0, 70)
    r.event({"kind": "collision", "sim_t": 70.0})
    feed(r, 70, 100)
    r.end_session("the simulation ended")
    (m,) = manifests(store)
    assert m["status"] == "incomplete"
    assert any("ends at" in p for p in m["problems"]) and any("ended early" in p for p in m["problems"])


def test_dropped_frames_are_reported(store):
    r = make(store)
    feed(r, 0, 70)
    r.event({"kind": "collision", "sim_t": 70.0})
    feed(r, 70, 140, skip={1005, 1006, 1007})        # three frames lost at 100.5 s
    (m,) = manifests(store)
    assert m["status"] == "incomplete" and any("3 frames missing" in p for p in m["problems"])


def test_default_loop_is_10_minutes_of_30_second_videos(store):
    r = dashcam.Recorder(store, log=lambda *_: None)
    r.start_session(FPS)
    feed(r, 0, 900)                                  # 15 minutes of driving
    sides = [json.loads(f.read_text()) for f in sorted((store / "rolling").glob("*.json"))]   # finished videos
    assert len(sides) == 20
    assert all(len(side["frames"]) == 300 for side in sides)      # 30 s at 10 frames/s each
    assert sides[0]["start"] == 270.0 and sides[-1]["end"] == 870.0   # the last 10 minutes (the 870-900 s video is still being written)
    assert not list((store / "incidents").iterdir())               # the loop never makes incidents


def test_unfinished_incident_is_sealed_after_a_crash(store):
    r = make(store)
    feed(r, 0, 70)
    r.event({"kind": "collision", "sim_t": 70.0})
    feed(r, 70, 80)
    r.close_segment()
    del r                                            # the recorder dies here, nothing sealed
    dashcam.Recorder(store, keep=90.0, segment_length=1.0, log=lambda *_: None)     # restart
    (m,) = manifests(store)
    assert m["status"] == "incomplete"
    assert any("recorder stopped" in p for p in m["problems"])


def test_socket_server_end_to_end(store, tmp_path_factory):
    from talk2drive.dashcam_link import DashcamFeed, ask

    path = str(tmp_path_factory.mktemp("sock") / "dashcam.sock")
    recorder = dashcam.Recorder(store, log=lambda *_: None)
    threading.Thread(target=dashcam.serve, args=(recorder, path), daemon=True).start()
    for _ in range(50):
        if os.path.exists(path):
            break
        time.sleep(0.05)
    cam = DashcamFeed(path, fps=FPS)
    for _ in range(50):
        if cam.connected:
            break
        time.sleep(0.05)
    assert cam.connected
    for i in range(30):
        cam.frame(np.zeros((H, W, 3), np.uint8), i / FPS)
        time.sleep(0.01)
    assert cam.event("manual", 2.9, "test")
    time.sleep(0.5)
    assert "Saving incident" in ask({"type": "status"}, path)["reply"]
    cam.close()
    time.sleep(0.5)
    (m,) = manifests(store)
    assert m["status"] == "incomplete"     # 3 s of driving: not 60 s before or after
    assert m["frames"] == 30 and cam.dropped == 0
    with socket.socket(socket.AF_UNIX) as s:
        s.connect(path)                    # the recorder is still running for the next simulation
