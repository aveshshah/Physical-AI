"""Dash cam for the talk-to-drive robot: loop recording of the robot's front camera, and protected
incident clips (collision, rollover, or saved on request).

    .venv/bin/python scripts/dashcam.py                # start the recorder (keep it running)
    .venv/bin/python scripts/dashcam.py trigger        # save an incident now (the dash cam's button)
    .venv/bin/python scripts/dashcam.py list           # saved incidents and whether they are complete
    .venv/bin/python scripts/dashcam.py verify ID      # check an incident's files and coverage again
    .venv/bin/python scripts/dashcam.py delete ID      # remove an incident (you must type its ID)

The simulation (sim/m4_talk_to_drive.py) sends the camera frames (640 x 360, 10 per simulated
second) and events here over a Unix socket. The recorder is its own process: if the assistant or
the language model fails, recording goes on.

Loop recording: frames are written into 30-second MP4 videos in <store>/rolling. Only the last
10 minutes are kept: older videos are deleted. Nothing else is ever deleted automatically.

Incidents: an event at time t0 saves the footage from t0 - 60 s to t0 + 60 s into
<store>/incidents/<id>/. Another event before the end extends it to 60 s after that event.

    NORMAL      record, delete only rolling videos older than 10 minutes
    TRIGGER     new incident folder; copy the segments already recorded (up to 60 s before)
    POST_EVENT  keep copying segments until 60 s after the last event
    SEAL        join them into incident.mp4, SHA-256 of every file into manifest.json, read-only
    VERIFY      check the time covered, missing frames and the hashes; "incomplete" if any fail

Times are simulation seconds (each frame also carries wall-clock and monotonic time).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import stat
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from talk2drive.dashcam_link import ask, default_socket, receive, send  # noqa: E402

SEGMENT = 30.0    # length of one loop video [s]
KEEP = 600.0      # loop recording keeps this much (10 minutes) [s]
PRE = 60.0        # an incident keeps this much before the event [s]
POST = 60.0       # ... and this much after the last event [s]
DEFAULT_STORE = REPO / "runs" / "dashcam"


# ---------------------------------------------------------------- small helpers
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json_safely(path: Path, data) -> None:
    """Write a JSON file so that a crash leaves either the old file or the new one, never half of one."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    folder = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(folder)
    finally:
        os.close(folder)


def make_read_only(folder: Path) -> None:
    for f in folder.iterdir():
        f.chmod(stat.S_IRUSR | stat.S_IRGRP)                  # r--r-----
    folder.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP)   # r-xr-x---: no new or removed files


def join_videos(videos: list, out: Path, fps: float) -> int:
    """Join MP4 segments into one video; returns the number of frames written."""
    import cv2

    writer, count = None, 0
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if writer is None:
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (image.shape[1], image.shape[0]))
            writer.write(image)
            count += 1
        cap.release()
    if writer is not None:
        writer.release()
    return count


def check_coverage(frames: list, need_from: float, need_to: float, fps: float) -> list:
    """Problems with the footage: not reaching back to need_from or on to need_to, missing frames, gaps."""
    if not frames:
        return ["no frames recorded"]
    problems = []
    step = 1.0 / fps
    first, last = frames[0]["sim_t"], frames[-1]["sim_t"]
    if first > need_from + step:
        problems.append(f"footage starts at {first:.1f} s, {first - need_from:.1f} s later than needed ({need_from:.1f} s)")
    if last < need_to - step:
        problems.append(f"footage ends at {last:.1f} s, {need_to - last:.1f} s earlier than needed ({need_to:.1f} s)")
    missing = sum(max(0, b["index"] - a["index"] - 1) for a, b in zip(frames, frames[1:]))
    if missing:
        problems.append(f"{missing} frames missing (dropped before they reached the recorder)")
    gaps = [b["sim_t"] - a["sim_t"] for a, b in zip(frames, frames[1:]) if b["sim_t"] - a["sim_t"] > 1.5 * step]
    if gaps:
        problems.append(f"{len(gaps)} gaps in time, the longest {max(gaps):.1f} s")
    return problems


# ---------------------------------------------------------------- sealing and checking an incident
def seal_folder(folder: Path, note: str | None = None, log=None) -> dict:
    """SEAL and VERIFY an incident folder: join the video, hash every file, write the manifest, make
    everything read-only, then read the files back and check them."""
    info = json.loads((folder / "events.json").read_text())
    events, fps = info["events"], info["fps"]
    frames, videos = [], []
    for side_file in sorted(folder.glob("*_seg*.json")):
        segment = json.loads(side_file.read_text())
        frames += segment["frames"]
        videos.append(side_file.with_suffix(".mp4"))
    frames.sort(key=lambda f: f["sim_t"])
    need_from = events[0]["sim_t"] - info["pre"]
    need_to = max(e["sim_t"] for e in events) + info["post"]
    joined = join_videos(videos, folder / "incident.mp4", fps) if videos else 0

    problems = check_coverage(frames, need_from, need_to, fps)
    if note:
        problems.append(note)
    if joined != len(frames):
        problems.append(f"incident.mp4 has {joined} frames but {len(frames)} were recorded")
    files = sorted(f for f in folder.iterdir() if not f.name.startswith("manifest"))
    manifest = {
        "incident": folder.name,
        "status": "complete" if not problems else "incomplete",
        "problems": problems,
        "events": events,
        "window_needed": [need_from, need_to],
        "window_recorded": [frames[0]["sim_t"], frames[-1]["sim_t"]] if frames else None,
        "frames": len(frames),
        "fps": fps,
        "sealed_wall_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": {f.name: {"bytes": f.stat().st_size, "sha256": sha256(f)} for f in files},
    }
    write_json_safely(folder / "manifest.json", manifest)
    (folder / "manifest.sha256").write_text(sha256(folder / "manifest.json") + "  manifest.json\n")
    make_read_only(folder)
    ok, findings = verify_folder(folder)
    status = manifest["status"] if ok else "FAILED VERIFY"
    (log or (lambda text: print(text, flush=True)))(f"[dashcam] sealed incident {folder.name}: {status}"
        + ("" if not problems else " (" + "; ".join(problems) + ")") + ("" if ok else " " + "; ".join(findings)))
    return manifest


def verify_folder(folder: Path):
    """Read every file back and compare with the manifest. Returns (ok, findings)."""
    findings = []
    manifest_file = folder / "manifest.json"
    if not manifest_file.exists():
        return False, ["no manifest (not sealed)"]
    expected = (folder / "manifest.sha256").read_text().split()[0] if (folder / "manifest.sha256").exists() else None
    if expected != sha256(manifest_file):
        findings.append("manifest.json was changed after sealing")
    manifest = json.loads(manifest_file.read_text())
    for name, entry in manifest["files"].items():
        path = folder / name
        if not path.exists():
            findings.append(f"{name} is missing")
        elif sha256(path) != entry["sha256"]:
            findings.append(f"{name} was changed")
    extra = {f.name for f in folder.iterdir()} - set(manifest["files"]) - {"manifest.json", "manifest.sha256"}
    if extra:
        findings.append(f"unexpected files: {', '.join(sorted(extra))}")
    if folder.stat().st_mode & stat.S_IWUSR:
        findings.append("the folder is writable again")
    return not findings, findings


# ---------------------------------------------------------------- the recorder
class Recorder:
    """Loop recording and incidents. Not thread-safe: the server calls it under one lock."""

    def __init__(self, store: Path, keep: float = KEEP, pre: float = PRE, post: float = POST, log=None,
                 segment_length: float = SEGMENT):
        self.store = Path(store)
        self.rolling = self.store / "rolling"
        self.incidents = self.store / "incidents"
        self.rolling.mkdir(parents=True, exist_ok=True)
        self.incidents.mkdir(parents=True, exist_ok=True)
        self.keep, self.pre, self.post, self.segment_length = keep, pre, post, segment_length
        self.log = log or (lambda text: print(text, flush=True))
        self.fps = 10.0
        self.session = None      # name of the current simulation run (None: no simulation connected)
        self.segment = None      # the segment being written
        self.segments = []       # finished segments still in the rolling folder, oldest first
        self.incident = None     # the incident being recorded
        self.last_t = None       # simulation time of the latest frame
        self.state = "NORMAL"
        self.recover()

    # -------- simulation runs
    def start_session(self, fps: float = 10.0) -> None:
        self.end_session("a new simulation run started")
        self.session, self.fps = time.strftime("%Y%m%d-%H%M%S"), float(fps)
        for f in self.rolling.iterdir():   # loop footage of earlier runs (incidents are kept elsewhere)
            f.unlink()
        self.log(f"[dashcam] recording run {self.session} at {self.fps:g} frames/s")

    def end_session(self, why: str) -> None:
        if self.session is None:
            return
        self.close_segment()
        if self.incident is not None:
            self.seal(note=f"recording ended early: {why}")
        self.log(f"[dashcam] run {self.session} ended: {why}")
        self.session, self.segments, self.last_t = None, [], None

    # -------- frames
    def frame(self, header: dict, pixels: bytes) -> None:
        import cv2

        t = header["sim_t"]
        if self.session is None or (self.last_t is not None and t < self.last_t - 1.0):
            self.start_session(self.fps)   # frames without a hello, or the simulation time went back
        k = int(t // self.segment_length)
        if self.segment is not None and k != self.segment["k"]:
            self.close_segment()
        if self.segment is None:
            self.open_segment(k, header["width"], header["height"])
        image = np.frombuffer(pixels, np.uint8).reshape(header["height"], header["width"], 3)
        self.segment["writer"].write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        self.segment["frames"].append({key: header[key] for key in ("index", "sim_t", "wall_t", "mono_t")})
        self.last_t = t

    def open_segment(self, k: int, width: int, height: int) -> None:
        import cv2

        name = f"{self.session}_seg{k:06d}"
        video = self.rolling / f"{name}.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height))
        if not writer.isOpened():
            self.log(f"[dashcam] ALARM: cannot write {video} (disk full?)")
        self.segment = {"k": k, "name": name, "video": video, "frames": [], "writer": writer}

    def close_segment(self) -> None:
        segment, self.segment = self.segment, None
        if segment is None:
            return
        segment.pop("writer").release()
        segment["start"] = segment["k"] * self.segment_length
        segment["end"] = segment["start"] + self.segment_length
        segment["info"] = self.rolling / f"{segment['name']}.json"
        try:
            write_json_safely(segment["info"], {key: segment[key] for key in ("name", "start", "end", "frames")})
        except OSError as e:
            self.log(f"[dashcam] ALARM: cannot write {segment['info']}: {e} (disk full?)")
        self.segments.append(segment)
        if self.incident is not None:
            self.copy_into_incident(segment)
            if segment["end"] >= self.incident["deadline"]:
                self.seal()
        self.delete_old_segments()
        if int(segment["end"]) % 30 == 0:
            self.log(f"[dashcam] t={segment['end']:.0f} s, loop {segment['end'] - self.segments[0]['start']:.0f} s"
                     + (f", saving incident {self.incident['id']} until {self.incident['deadline']:.0f} s" if self.incident else ""))

    def delete_old_segments(self) -> None:
        """Loop recording: keep the newest KEEP seconds of finished videos; delete older ones.
        Only the rolling folder is touched."""
        newest_end = self.segments[-1]["end"] if self.segments else 0.0
        while self.segments and self.segments[0]["end"] <= newest_end - self.keep:
            old = self.segments.pop(0)
            old["video"].unlink(missing_ok=True)
            old["info"].unlink(missing_ok=True)

    # -------- incidents
    def event(self, header: dict) -> str:
        kind = header.get("kind", "manual")
        t = header.get("sim_t")
        t = self.last_t if t is None else t
        if self.session is None or t is None:
            return "No footage to save: the simulation is not sending camera frames."
        event = {"kind": kind, "sim_t": t, "wall_t": header.get("wall_t", time.time()), "note": header.get("note", "")}
        if self.incident is None:   # TRIGGER
            name = time.strftime("%Y%m%d-%H%M%S") + f"_{kind}"
            folder = self.incidents / name
            n = 1
            while folder.exists():
                n += 1
                folder = self.incidents / f"{name}_{n}"
            folder.mkdir()
            self.incident = {"id": folder.name, "dir": folder, "events": [], "deadline": t + self.post, "copied": set()}
            self.state = "TRIGGER"
            self.incident["events"].append(event)
            self.save_events()
            for segment in self.segments:
                if segment["end"] > t - self.pre:
                    self.copy_into_incident(segment)
            first = next((s["start"] for s in self.segments if s["end"] > t - self.pre), self.segment["k"] * self.segment_length if self.segment else t)
            reply = (f"Saving incident {folder.name}: {kind} at {t:.1f} s; footage from {max(first, t - self.pre):.0f} s "
                     f"to {self.incident['deadline']:.0f} s.")
        else:                       # another event: keep recording longer
            self.incident["events"].append(event)
            self.incident["deadline"] = max(self.incident["deadline"], t + self.post)
            self.save_events()
            reply = f"Added to incident {self.incident['id']}: now saving until {self.incident['deadline']:.0f} s."
        self.state = "POST_EVENT"
        self.log(f"[dashcam] {reply}")
        return reply

    def save_events(self) -> None:
        write_json_safely(self.incident["dir"] / "events.json",
                          {"pre": self.pre, "post": self.post, "fps": self.fps, "events": self.incident["events"]})

    def copy_into_incident(self, segment: dict) -> None:
        if segment["name"] in self.incident["copied"]:
            return
        try:
            shutil.copy2(segment["video"], self.incident["dir"])
            shutil.copy2(segment["info"], self.incident["dir"])
            self.incident["copied"].add(segment["name"])
        except OSError as e:
            self.log(f"[dashcam] ALARM: cannot copy {segment['name']} into the incident: {e} (disk full?)")

    def seal(self, note: str | None = None) -> None:
        incident, self.incident = self.incident, None
        self.state = "SEAL"
        seal_folder(incident["dir"], note, self.log)
        self.state = "NORMAL"

    def recover(self) -> None:
        """After a crash or power cut: seal incidents that were never sealed, marked incomplete."""
        for folder in sorted(self.incidents.iterdir()):
            if folder.is_dir() and not (folder / "manifest.json").exists() and (folder / "events.json").exists():
                seal_folder(folder, "the recorder stopped before this incident was finished", self.log)

    def status(self) -> str:
        if self.session is None:
            return "Recorder running; no simulation connected."
        loop = (self.last_t or 0) - (self.segments[0]["start"] if self.segments else (self.last_t or 0))
        text = f"Recording run {self.session}: t={self.last_t or 0:.1f} s, loop {loop:.0f} s in {len(self.segments)} videos."
        if self.incident:
            text += f" Saving incident {self.incident['id']} until {self.incident['deadline']:.0f} s."
        return text


# ---------------------------------------------------------------- the socket server
def serve(recorder: Recorder, path: str) -> None:
    if os.path.exists(path):
        try:
            ask({"type": "status"}, path, timeout=1.0)
            sys.exit(f"A dash cam recorder is already running on {path}.")
        except OSError:
            os.unlink(path)   # left over from a recorder that crashed
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)   # only this user can send frames or events
    server.listen(4)
    lock = threading.Lock()

    def process(header, payload):
        """Do what one message asks; returns the reply text."""
        kind = header.get("type")
        if kind == "hello":
            recorder.start_session(header.get("fps", 10.0))
        elif kind == "frame":
            recorder.frame(header, payload)
        elif kind == "event":
            return recorder.event(header)
        elif kind == "bye":
            if header.get("dropped"):
                recorder.log(f"[dashcam] the simulation dropped {header['dropped']} frames")
            recorder.end_session("the simulation ended")
        elif kind == "status":
            return recorder.status()
        return ""

    def handle(conn):
        """One connection: a simulation sending frames, or a one-off trigger/status request."""
        from_simulation = False
        with conn:
            while True:
                try:
                    header, payload = receive(conn)
                except (OSError, ValueError):
                    break
                if header is None:
                    break
                with lock:
                    try:
                        reply = process(header, payload)
                    except Exception as e:   # a bad message must not stop the recorder
                        recorder.log(f"[dashcam] ERROR handling {header.get('type')}: {e!r}")
                        reply = f"Recorder error: {e}"
                if header.get("type") in ("hello", "bye"):
                    from_simulation = header["type"] == "hello"
                if header.get("want_reply"):
                    send(conn, {"reply": reply})
        if from_simulation:
            with lock:
                recorder.end_session("the simulation disconnected")

    def stop(*_):
        raise KeyboardInterrupt

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, stop)   # `kill` stops it as cleanly as Ctrl+C
    print(f"[dashcam] recorder ready on {path}; saving to {recorder.store}. Ctrl+C stops.", flush=True)
    try:
        while True:
            conn, _ = server.accept()
            threading.Thread(target=handle, args=(conn,), daemon=True).start()
    except KeyboardInterrupt:
        pass
    finally:
        with lock:
            recorder.end_session("the recorder was stopped")
        server.close()
        os.unlink(path)


# ---------------------------------------------------------------- commands
def list_incidents(store: Path) -> None:
    folders = sorted(f for f in (store / "incidents").glob("*") if f.is_dir()) if (store / "incidents").exists() else []
    if not folders:
        print("No incidents saved.")
    for folder in folders:
        manifest_file = folder / "manifest.json"
        if not manifest_file.exists():
            print(f"{folder.name}: being recorded")
            continue
        m = json.loads(manifest_file.read_text())
        kinds = ", ".join(f"{e['kind']} at {e['sim_t']:.1f} s" for e in m["events"][:3])
        if len(m["events"]) > 3:
            kinds += f" and {len(m['events']) - 3} more events"
        span = m["window_recorded"]
        print(f"{folder.name}: {m['status'].upper()}  {kinds}; footage {span[0]:.1f}-{span[1]:.1f} s ({m['frames']} frames)"
              if span else f"{folder.name}: {m['status'].upper()}  {kinds}; no footage")
        for problem in m["problems"]:
            print(f"    - {problem}")


def delete_incident(store: Path, name: str) -> None:
    """Manual removal only: someone at the keyboard has to type the incident's name."""
    folder = store / "incidents" / name
    if not folder.is_dir():
        sys.exit(f"No incident {name}.")
    if not sys.stdin.isatty():
        sys.exit("Refusing: deleting an incident must be confirmed by a person at the terminal.")
    typed = input(f"This permanently deletes the protected footage of {name}.\nType the incident name to confirm: ")
    if typed.strip() != name:
        sys.exit("Not deleted.")
    folder.chmod(stat.S_IRWXU)
    for f in folder.iterdir():
        f.chmod(stat.S_IRUSR | stat.S_IWUSR)
    shutil.rmtree(folder)
    print(f"Deleted {name}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="run", choices=["run", "trigger", "status", "list", "verify", "delete"])
    parser.add_argument("incident", nargs="?", help="incident name (verify, delete)")
    parser.add_argument("--store", default=str(DEFAULT_STORE), help="folder for the loop and the incidents")
    parser.add_argument("--socket", default=default_socket())
    parser.add_argument("--keep", type=float, default=KEEP, help="seconds of loop recording kept (600 = 10 minutes)")
    parser.add_argument("--segment", type=float, default=SEGMENT, help="length of each loop video [s]")
    parser.add_argument("--pre", type=float, default=PRE, help="seconds kept before a collision or rollover")
    parser.add_argument("--post", type=float, default=POST, help="seconds kept after the last collision or rollover")
    parser.add_argument("--note", default="", help="trigger: a note saved with the event")
    args = parser.parse_args()
    store = Path(args.store)

    if args.command == "run":
        serve(Recorder(store, keep=args.keep, pre=args.pre, post=args.post, segment_length=args.segment), args.socket)
    elif args.command in ("trigger", "status"):
        header = {"type": "event", "kind": "manual", "note": args.note} if args.command == "trigger" else {"type": "status"}
        try:
            print(ask(header, args.socket).get("reply", ""))
        except OSError:
            sys.exit(f"The dash cam recorder is not running ({args.socket}). Start it with: .venv/bin/python scripts/dashcam.py")
    elif args.command == "list":
        list_incidents(store)
    elif args.command == "verify":
        if not args.incident:
            sys.exit("Which incident? See: dashcam.py list")
        ok, findings = verify_folder(store / "incidents" / args.incident)
        print("OK: every file matches its SHA-256 in the manifest." if ok else "PROBLEM: " + "; ".join(findings))
    elif args.command == "delete":
        if not args.incident:
            sys.exit("Which incident? See: dashcam.py list")
        delete_incident(store, args.incident)


if __name__ == "__main__":
    main()
