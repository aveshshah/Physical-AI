"""Typed driving commands: text -> Command, and the stepwise speed control they drive.

    parse_command("turn left at the next junction")  -> Command("turn", "left")
    parse_command("accelerate more")                 -> Command("accelerate")
    parse_command("set speed to 6")                  -> Command("set_speed", 6.0)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

HELP = ("go | stop | emergency stop | pull over | turn left | turn right | go straight | u-turn | "
        "change lane left/right | steer left/right (a bit) | back to the centre | "
        "accelerate (more) | brake (more) | hold speed | set speed 8 | what's ahead | status | reset")


@dataclass
class Command:
    kind: str            # go, stop, emergency, pull_over, turn, u_turn, lane, nudge, centre, accelerate,
                         # brake, hold, set_speed, ahead, status, reset, help, unknown
    value: object = None  # turn: "left"/"right"/"straight"; lane, nudge: "left"/"right"/None; set_speed: m/s


def _side(t: str):
    if re.search(r"\bleft\b", t):
        return "left"
    if re.search(r"\bright\b", t):
        return "right"
    return None


def parse_command(text: str) -> Command:
    t = text.lower().strip()
    if re.search(r"\breset\b|put (me|it|yourself) back|recover|get up", t):
        return Command("reset")
    if re.search(r"what can you do|\bhelp\b|what commands|what do you understand", t):
        return Command("help")
    if re.search(r"u[- ]?turn|\byou turn|turn (a)?round|turn back|go back|other way|opposite direction|head back", t):
        return Command("u_turn")
    if re.search(r"emergenc|e-?stop|stop now|full brake|brake hard|slam", t):
        return Command("emergency")
    if re.search(r"pull over|park|side of the road|to the side|kerb|curb", t) and not re.search(r"away from|off the", t):
        return Command("pull_over")
    if re.search(r"\blanes?\b", t) and re.search(r"change|switch|move|merge|get|go|into|take|over", t) \
            or re.search(r"overtake|\bpass (the|this|that|him|her|it)\b", t):
        if not re.search(r"stay|keep|middle|cent(er|re)", t):
            return Command("lane", _side(t))
    if re.search(r"middle|cent(er|re)|stay in (the|your|my) lane|straighten", t):
        return Command("centre")
    if _side(t) and not re.search(r"\bturn\b|junction|exit|take|corner|street|road", t) and \
            re.search(r"steer|nudge|shift|drift|veer|edge|keep|stay|hug|dodge|avoid|swerve|slightly|a (bit|little|touch)|move( over)?\b", t):
        return Command("nudge", _side(t))
    m = re.search(r"(\d+(?:\.\d+)?)\s*(km/?h|kph|km per hour|kilometers per hour|kilometres per hour|mph|miles per hour|m/s|mps|meters per second|metres per second)?", t)
    if m and (m.group(2) or re.search(r"speed|drive at|go at", t)):
        value = float(m.group(1))
        unit = m.group(2) or ""
        if unit.startswith("k"):
            value /= 3.6
        elif unit.startswith("m") and ("ph" in unit or "mile" in unit):
            value *= 0.447
        return Command("set_speed", round(value, 2))
    if re.search(r"\bleft\b", t):
        return Command("turn", "left")
    if re.search(r"\bright\b", t):
        return Command("turn", "right")
    if re.search(r"straight", t):
        return Command("turn", "straight")
    if re.search(r"accelerat|faster|speed up|quicker", t):
        return Command("accelerate")
    if re.search(r"brake|braking|slow|decelerat|slower", t):
        return Command("brake")
    if re.search(r"hold|maintain|cruise|keep (this|the|your) speed|constant speed", t):
        return Command("hold")
    if re.search(r"\bstop\b|halt", t):
        return Command("stop")
    if re.search(r"ahead|junction|turns?\b|options", t):
        return Command("ahead")
    if re.search(r"status|how fast|speed\?|where", t):
        return Command("status")
    if re.search(r"\bgo\b|keep going|start|drive|resume|continue|move|let'?s go", t):
        return Command("go")
    return Command("unknown")


class SpeedCommander:
    """The speed set-point the NMPC is capped to, moved by commands.

    accelerate: speed up at a rate that grows by STEP (0.5 m/s^2) with every "accelerate" command
    brake:      slow down at a rate that grows by STEP with every "brake" command
    hold:       keep the current speed;  set_speed: go to a given speed;  go: speed up to GO_SPEED
    Each command starts from the robot's actual speed, so "brake" acts at once even if the set-point
    was higher than the road allowed."""

    STEP = 0.5          # m/s^2 per command
    MAX_ACCEL = 2.0     # m/s^2 (the CARLA plan's own limit is 2.2)
    MAX_BRAKE = 3.0     # m/s^2 (the CARLA comfort-braking limit is 3.1)
    GO_SPEED = 8.0      # m/s, what "go" drives at
    GO_ACCEL = 1.0      # m/s^2
    STOP_BRAKE = 1.5    # m/s^2, a normal stop

    def __init__(self, v_max: float):
        self.v_max = v_max
        self.v_set = 0.0     # current set-point [m/s]
        self.rate = 0.0      # how fast the set-point moves [m/s^2], + up, - down
        self.goal = 0.0      # where the set-point stops moving [m/s]

    def go(self, v_now: float) -> str:
        v_now = max(v_now, 0.0)
        self.v_set = v_now
        self.goal = max(self.GO_SPEED, v_now)
        self.rate = self.GO_ACCEL if self.goal > v_now else 0.0
        return f"Driving: speeding up to {self.goal:.0f} m/s."

    def accelerate(self, v_now: float) -> str:
        self.v_set = max(v_now, self.v_set if self.rate > 0 else v_now)
        self.rate = min(max(self.rate, 0.0) + self.STEP, self.MAX_ACCEL)
        self.goal = self.v_max
        return f"Accelerating at {self.rate:.1f} m/s^2 (say 'hold speed' to stop accelerating)."

    def brake(self, v_now: float) -> str:
        self.v_set = min(v_now, self.v_set) if self.rate < 0 else v_now
        self.rate = max(min(self.rate, 0.0) - self.STEP, -self.MAX_BRAKE)
        self.goal = 0.0
        return f"Braking at {-self.rate:.1f} m/s^2 (say 'hold speed' to keep the speed)."

    def hold(self, v_now: float) -> str:
        self.v_set, self.rate, self.goal = v_now, 0.0, v_now
        return f"Holding {v_now:.1f} m/s."

    def stop(self, v_now: float) -> str:
        self.v_set, self.rate, self.goal = v_now, -self.STOP_BRAKE, 0.0
        return "Stopping in the lane."

    def set_speed(self, v_now: float, value: float) -> str:
        value = max(0.0, min(value, self.v_max))
        self.v_set, self.goal = v_now, value
        if value > v_now:
            self.rate = self.GO_ACCEL
        elif value < v_now:
            self.rate = -self.STOP_BRAKE
        else:
            self.rate = 0.0
        return f"Going to {value:.1f} m/s."

    def update(self, dt: float) -> str | None:
        """Move the set-point; returns a message when it reaches its goal."""
        if self.rate == 0.0:
            return None
        self.v_set += self.rate * dt
        if (self.rate > 0 and self.v_set >= self.goal) or (self.rate < 0 and self.v_set <= self.goal):
            self.v_set, self.rate = self.goal, 0.0
            return None   # the robot says "Reached ..." itself, once its real speed gets there
        return None


# ---------------------------------------------------------------- free-form commands (language model)
INTENT_PROMPT = """You turn what a passenger says to a small self-driving delivery robot into ONE robot command.
The robot drives on public roads in Mcity (right-hand traffic: the kerb is on its RIGHT, oncoming traffic
on its LEFT, so moving away from the kerb means steer_left). It cannot reverse, fly, honk or open doors.
Commands it can do:
  go            start driving / continue / drive on
  stop          normal stop in the lane
  emergency     brake as hard as possible right now
  pull_over     park at the side of the road
  turn_left / turn_right / go_straight     at the next junction that allows it
  u_turn        turn around and drive back the other way (from where it is, even when stopped)
  lane_left / lane_right       change into the next lane on that side (same direction of travel)
  steer_left / steer_right     move a little (0.5 m) to that side inside its own lane
  centre        go back to the middle of its lane
  accelerate    speed up (each command a bit harder)
  brake         slow down (each command a bit harder)
  hold          keep the current speed
  set_speed     drive at a given speed; put it in "value" in m/s (convert km/h: divide by 3.6, mph: x0.447)
  ahead         tell what junctions and turns are coming up
  status        tell speed and what it is doing
  reset         put the robot back on its wheels after tipping over
  none          nothing above matches, it is impossible or unsafe, or it names a place or destination
                (the robot does not know addresses or places; it only takes turns when told)
Robot now: {context}
Passenger said: "{said}"
Answer with JSON only: {{"command": "...", "value": null, "say": "one short sentence to the passenger: what you understood, or why you cannot do it (and what you can do instead)"}}"""

LLM_COMMANDS = {
    "go": Command("go"), "stop": Command("stop"), "emergency": Command("emergency"), "pull_over": Command("pull_over"),
    "turn_left": Command("turn", "left"), "turn_right": Command("turn", "right"), "go_straight": Command("turn", "straight"),
    "u_turn": Command("u_turn"), "lane_left": Command("lane", "left"), "lane_right": Command("lane", "right"),
    "steer_left": Command("nudge", "left"), "steer_right": Command("nudge", "right"), "centre": Command("centre"), "accelerate": Command("accelerate"), "brake": Command("brake"), "hold": Command("hold"),
    "ahead": Command("ahead"), "status": Command("status"), "reset": Command("reset"),
}


def interpret_with_llm(said: str, context: str, url: str = "http://127.0.0.1:8765", timeout: float = 8.0):
    """Ask the local language model what an unrecognised sentence means.
    Returns (Command or None, sentence to say), or (None, None) if the model cannot be reached."""
    import json
    import urllib.request

    prompt = INTENT_PROMPT.format(context=context, said=said.replace('"', "'"))
    req = urllib.request.Request(url + "/text", data=json.dumps({"prompt": prompt}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = json.loads(r.read())["text"]
        m = re.search(r"\{.*\}", text, re.S)
        answer = json.loads(m.group(0)) if m else {}
    except (OSError, ValueError):
        return None, None
    name = str(answer.get("command", "none")).strip().lower()
    say = str(answer.get("say", "")).strip()
    if name == "set_speed":
        try:
            return Command("set_speed", float(answer.get("value"))), say
        except (TypeError, ValueError):
            return None, say or "I didn't catch the speed."
    return LLM_COMMANDS.get(name), say
