"""Voice control for the talk-to-drive simulation (sim/m4_talk_to_drive.py).

    .venv/bin/python scripts/voice.py

Press Enter and speak a command ("turn left", "pull over", "brake more", ...); recording stops by
itself when you go quiet. Whisper (small.en, on the CPU) turns it into text, the simulation answers,
and the answer is spoken with Piper. The robot's own announcements are spoken too.

Barge-in: the microphone is always open. Say a command while the robot is speaking ("stop", "turn
left", ...) and it stops talking at once and does it (pressing Enter also stops it and listens).
What the robot itself is saying is never taken as a command, even through a loudspeaker.
Everything runs offline.
Type q and Enter to quit.
"""

from __future__ import annotations

import difflib
import json
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from say import say  # noqa: E402

SIM = "http://127.0.0.1:8770"
VOICE = Path.home() / ".cache/talk2drive/piper/en_US-lessac-medium.onnx"
RATE = 16000
CHUNK = 480                # 30 ms of audio
SILENCE_END = 0.8          # a command ends after this long quiet [s]
NO_SPEECH_GIVE_UP = 5.0    # stop listening if nothing is said for this long [s]
MAX_LENGTH = 10.0          # never record longer than this [s]
FIXES = {r"\bbreak": "brake", r"\bbreaking\b": "braking", r"\bpole over\b": "pull over", r"\bpool over\b": "pull over"}


# ---------------------------------------------------------------- microphone (always on)
class Microphone:
    """Reads the default microphone continuously (arecord) and keeps the last few seconds."""

    def __init__(self):
        self.chunks = []           # recent chunks (float32 arrays); chunk number n is chunks[n - dropped]
        self.levels = []           # their loudness (RMS)
        self.dropped = 0           # chunks already thrown away (older than about a minute)
        self.lock = threading.Lock()
        self.process = subprocess.Popen(["arecord", "-q", "-f", "S16_LE", "-r", str(RATE), "-c", "1", "-t", "raw"],
                                        stdout=subprocess.PIPE)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        while True:
            raw = self.process.stdout.read(CHUNK * 2)
            if not raw:
                return
            chunk = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
            with self.lock:
                self.chunks.append(chunk)
                self.levels.append(float(np.sqrt(np.mean(chunk * chunk))))
                if len(self.chunks) > 2000:   # keep about a minute
                    del self.chunks[:500], self.levels[:500]
                    self.dropped += 500

    def position(self) -> int:
        """Number of the next chunk to arrive."""
        with self.lock:
            return self.dropped + len(self.chunks)

    def noise_level(self) -> float:
        """Background noise: a low percentile of the last 10 s of loudness."""
        with self.lock:
            recent = self.levels[-330:]
        return max(float(np.percentile(recent, 20)) if recent else 0.003, 0.002)

    def wait_chunk(self, number: int):
        """Chunk number ``number`` and its loudness (waits until it has been recorded)."""
        while True:
            with self.lock:
                i = max(0, number - self.dropped)
                if i < len(self.chunks):
                    return self.chunks[i], self.levels[i]
            time.sleep(0.005)

    def record_command(self, start_back: float = 0.0) -> np.ndarray:
        """Audio from ``start_back`` seconds ago until the speaker goes quiet (empty if nothing said)."""
        threshold = max(3.0 * self.noise_level(), 0.01)
        index = max(0, self.position() - int(start_back * RATE / CHUNK))
        audio, speaking, quiet, t = [], start_back > 0, 0.0, 0.0
        while t < MAX_LENGTH:
            chunk, level = self.wait_chunk(index)
            index += 1
            audio.append(chunk)
            t += CHUNK / RATE
            if level > threshold:
                speaking, quiet = True, 0.0
            elif speaking:
                quiet += CHUNK / RATE
                if quiet > SILENCE_END:
                    break
            elif t > NO_SPEECH_GIVE_UP:
                return np.zeros(0, np.float32)
        return np.concatenate(audio) if speaking else np.zeros(0, np.float32)


# ---------------------------------------------------------------- speaker (can be interrupted)
class Speaker:
    def __init__(self, voice):
        self.voice = voice
        self.queue = queue.Queue()
        self.player = None
        self.started = 0.0         # when the current sentence started playing
        self.recent = []           # what was said lately (to ignore the robot hearing itself)
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def say(self, text: str) -> None:
        self.queue.put(text)

    @property
    def talking(self) -> bool:
        with self.lock:
            return self.player is not None and self.player.poll() is None

    def stop(self) -> None:
        """Stop talking now and forget anything still waiting to be said."""
        while not self.queue.empty():
            self.queue.get_nowait()
        with self.lock:
            if self.player is not None and self.player.poll() is None:
                self.player.kill()

    def _run(self):
        while True:
            text = self.queue.get()
            self.recent = (self.recent + [text])[-5:]
            spoken = text.replace("m/s^2", "metres per second squared").replace("m/s", "metres per second")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                path = f.name
            with wave.open(path, "wb") as w:
                self.voice.synthesize_wav(spoken, w)
            with self.lock:
                self.player = subprocess.Popen(["pw-play", path])
                self.started = time.time()
            self.player.wait()
            Path(path).unlink(missing_ok=True)

    def is_echo(self, heard: str) -> bool:
        """True if ``heard`` sounds like something the robot said itself just now."""
        h = re.sub(r"[^a-z0-9 ]", "", heard.lower())
        for said in self.recent:
            s = re.sub(r"[^a-z0-9 ]", "", said.lower().replace("m/s", "meters per second"))
            if len(h) > 6 and (h in s or difflib.SequenceMatcher(None, h, s).ratio() > 0.6):
                return True
        return False


# ---------------------------------------------------------------- listening triggers
def watch_enter(events: queue.Queue) -> None:
    while True:
        try:
            line = input()
        except EOFError:
            events.put(("quit", ""))
            return
        events.put(("quit", "") if line.strip().lower() == "q" else ("enter", ""))


# words that start a command; heard while the robot talks (and not part of what it is saying), they interrupt it
COMMAND_WORDS = {"stop", "emergency", "brake", "braking", "slow", "left", "right", "straight", "go", "pull",
                 "park", "accelerate", "faster", "hold", "speed", "turn", "halt", "wait", "lane", "steer",
                 "around", "middle", "centre", "center", "overtake"}


def words(text: str) -> list:
    return re.findall(r"[a-z]+", clean(text).lower())


def watch_barge_in(mic: Microphone, speaker: Speaker, events: queue.Queue, listening: threading.Event, quick) -> None:
    """While the robot talks, keep transcribing the last 1.6 s of microphone audio with a small, fast
    Whisper model. A command word that is not in what the robot is saying means you are talking:
    the robot stops at once and your words are passed on. (Loudness alone cannot tell you from the
    robot when it plays through a loudspeaker next to the microphone.)"""
    while True:
        time.sleep(0.3)
        if not speaker.talking or listening.is_set():
            continue
        end = mic.position()
        n = int(1.6 * RATE / CHUNK)
        chunks = [mic.wait_chunk(i)[0] for i in range(max(0, end - n), end)]
        if not chunks:
            continue
        audio = np.concatenate(chunks)
        if float(np.sqrt(np.mean(audio * audio))) < 2.5 * mic.noise_level():
            continue   # quiet: nobody is talking
        segments, _ = quick.transcribe(audio, language="en", beam_size=1, vad_filter=True, condition_on_previous_text=False)
        heard = words(" ".join(s.text for s in segments))
        robot = set(words(" ".join(speaker.recent[-2:]))) | {"metres", "meters", "metre", "meter", "second", "per", "squared"}
        mine = [w for w in heard if w not in robot]
        if speaker.talking and any(w in COMMAND_WORDS for w in mine):
            speaker.stop()
            events.put(("barge", " ".join(mine)))


def watch_news(speaker: Speaker) -> None:
    """Speak what the robot announces on its own ("Parked at the side of the road.")."""
    after = None
    while True:
        try:
            reply = json.loads(urllib.request.urlopen(f"{SIM}/news?after={after or 0}", timeout=2).read())
            if after is not None:
                for message in reply["news"]:
                    print(f"\nbot (announces)> {message}", flush=True)
                    speaker.say(message)
            after = reply["next"]
        except OSError:
            pass
        time.sleep(1.0)


# ---------------------------------------------------------------- main
def clean(text: str) -> str:
    for pattern, word in FIXES.items():
        text = re.sub(pattern, word, text, flags=re.I)
    return text.strip()


def main() -> None:
    from faster_whisper import WhisperModel
    from piper import PiperVoice

    print("[voice] loading speech recognition (Whisper small.en) and voice (Piper) ...", flush=True)
    whisper = WhisperModel("small.en", device="cpu", compute_type="int8")
    quick = WhisperModel("base.en", device="cpu", compute_type="int8")   # fast one, for interruptions
    speaker = Speaker(PiperVoice.load(str(VOICE)))
    mic = Microphone()
    events, listening = queue.Queue(), threading.Event()
    threading.Thread(target=watch_enter, args=(events,), daemon=True).start()
    threading.Thread(target=watch_barge_in, args=(mic, speaker, events, listening, quick), daemon=True).start()
    threading.Thread(target=watch_news, args=(speaker,), daemon=True).start()
    time.sleep(1.0)   # a second of background noise before the first command
    print("[voice] ready. Press Enter and speak a command. Talk over the robot to interrupt it. q + Enter quits.")
    print("\n[Enter to talk] ", end="", flush=True)

    while True:
        event, heard_so_far = events.get()
        if event == "quit":
            break
        speaker.stop()
        listening.set()
        if event == "enter":
            print("\nlistening ...", flush=True)
        else:
            print(f"\n(interrupted: heard '{heard_so_far}') listening ...", flush=True)
        # after an interruption, keep listening in case the command is still going on
        audio = mic.record_command(start_back=0.0)
        listening.clear()
        t0 = time.time()
        text = heard_so_far
        if audio.size:
            segments, _ = whisper.transcribe(audio, language="en", beam_size=1, vad_filter=True,
                                             condition_on_previous_text=False)
            segments = [s for s in segments if s.no_speech_prob < 0.6 and s.avg_logprob > -1.0]
            text = clean((heard_so_far + " " + " ".join(s.text for s in segments)).strip())
        if not text and audio.size == 0:
            print("(heard nothing)")
            print("\n[Enter to talk] ", end="", flush=True)
            continue
        if not text:
            print("(could not make out any words)")
        elif speaker.is_echo(text):
            print(f"(ignored: that was the robot's own voice: '{text}')")
        else:
            print(f"you> {text}   ({time.time() - t0:.1f} s to understand)")
            reply = say(text)
            print(f"bot> {reply}")
            speaker.say(reply)
        print("\n[Enter to talk] ", end="", flush=True)
    speaker.stop()
    mic.process.terminate()


if __name__ == "__main__":
    main()
