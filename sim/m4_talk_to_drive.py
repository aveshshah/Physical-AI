"""
Talk to drive: tell the robot what to do and it plans and drives it.

    ~/isaac-env/bin/python sim/m4_talk_to_drive.py                  # window open, waits for commands
    .venv/bin/python scripts/say.py "go"                             # from any other terminal
    .venv/bin/python scripts/say.py "turn left"
    ~/isaac-env/bin/python sim/m4_talk_to_drive.py --script "1:go; 20:turn left; 45:pull over"

Commands (see talk2drive.commands): go | stop | emergency stop | pull over | turn left | turn right |
go straight | u-turn | change lane left/right | steer left/right | back to the centre |
accelerate (more) | brake (more) | hold speed | set speed 8 | what's ahead | status | quit

The robot follows the road; at junctions it goes straight unless told to turn. "Turn left" picks the
first junction with a left exit far enough ahead to slow down for it and no more than 200 m away,
re-plans the global route through it, or answers that there is no left turn. "U-turn" turns around
where the robot is (also from a stop or from a parking spot), or, where the road is too narrow or one
way, goes round the nearest roundabout or block and comes back. "Change lane left" moves into the next
lane going the same way; "steer left" moves 0.5 m to the side inside the lane (until the next
junction). A move asked for while stopped also starts the robot. The CARLA NMPC drives.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--start", default="104,1", help="start road id,direction")
parser.add_argument("--robot", default="magna_susp")
parser.add_argument("--port", type=int, default=8770, help="command port (scripts/say.py talks to it)")
parser.add_argument("--llm-url", default="http://127.0.0.1:8765", help="language model for unfamiliar phrasing (scripts/vlm_ground_server.py)")
parser.add_argument("--script", default="", help='timed commands, e.g. "1:go; 20:turn left" (sim seconds)')
parser.add_argument("--v-max", type=float, default=15.0)
parser.add_argument("--ay-curve", type=float, default=2.4)
parser.add_argument("--max-time", type=float, default=1800.0, help="end the run after this much simulated time [s]")
parser.add_argument("--nmpc-dir", default=os.path.expanduser("~/Documents/carla_nmpc_project"))
parser.add_argument("--headless", action="store_true")
parser.add_argument("--record", action="store_true", help="save the viewport every 0.5 s and a GIF")
parser.add_argument("--out", default="runs/m4_talk")
args = parser.parse_args()

# settings read by the CARLA NMPC project when it is imported
os.environ["NMPC_V_MAX"] = f"{args.v_max:g}"
os.environ["DRIVE_STYLE"] = "sport"
os.environ["NMPC_SPEED_HORIZON"] = "1"
os.environ.setdefault("ACADOS_SOURCE_DIR", os.path.expanduser("~/acados"))

from isaacsim import SimulationApp  # noqa: E402  (must start before the other Isaac imports)

app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

import numpy as np  # noqa: E402

from talk2drive.commands import HELP, SpeedCommander, interpret_with_llm, parse_command  # noqa: E402
from talk2drive.disturbances import BodyForceApplier, RoadLoadModel, parse_scenario  # noqa: E402
from talk2drive.isaac import jetbot as J  # noqa: E402
from talk2drive.isaac.jetbot import IsaacJetbot  # noqa: E402
from talk2drive.isaac.props import HALF_WIDTH  # noqa: E402
from talk2drive.isaac.simtools import ViewportRecorder, chase_view, road_height  # noqa: E402
from talk2drive.maneuvers import PARK_BRAKE, find_pull_over_spot, offset_path, shift_length, sideways_path, u_turn_path  # noqa: E402
from talk2drive.mcity import MCITY_XODR, mcity_to_sim  # noqa: E402
from talk2drive.opendrive import OpenDrive  # noqa: E402
from talk2drive.road_router import RoadRouter  # noqa: E402
from talk2drive.tadpole import TadpoleDriver, load_tadpole_stack  # noqa: E402

LOOK_AHEAD = 200.0      # turns and parking spots are searched this far ahead [m]
KEEP_ROUTE = 600.0      # length of route kept planned ahead [m]
MIN_TURN_DIST = 1.0     # a commanded turn is taken if its junction starts at least this far ahead [m]
NORMAL_DECEL = 3.5      # beyond this braking need, a sharp bend ahead is approached with the emergency brake [m/s^2]
MODES = {"drive": 0, "pull_over": 1, "parked": 2, "rejoin": 3, "emergency": 4, "tipped": 5, "u_turn": 6}
HOLD_BRAKE = 600.0      # brake force that holds the robot still [N] (the full emergency brake chatters at standstill)
STILL = 0.5             # below this speed the robot is held with HOLD_BRAKE instead of braking dynamically [m/s]
NUDGE = 0.5             # one "steer left" moves this far to the side [m]
NEAR_U_TURN = 60.0      # a U-turn spot this close is used before going round a roundabout or block [m]
MOVES = ("turn", "u_turn", "lane", "nudge", "centre")   # commands that change where the robot drives


def describe(move):
    """'turning left (90 deg)', 'bearing right (32 deg)' or 'going straight'."""
    if move.kind == "straight":
        return "going straight"
    verb = "turning" if abs(move.angle_deg) >= 45 else "bearing"
    return f"{verb} {move.kind} ({abs(move.angle_deg):.0f} deg)"


def speed_to_stop_within(distance, braking, reaction_time):
    distance = max(0.0, distance)
    return braking * (math.sqrt(reaction_time * reaction_time + 2.0 * distance / braking) - reaction_time)


# ---------------------------------------------------------------- command input
class CommandServer:
    """Commands arrive over HTTP (scripts/say.py) or from --script; the sim loop answers them."""

    def __init__(self, port):
        self.inbox = queue.Queue()
        self.news = []   # things the robot says on its own ("Parked ...", "Back in the lane.")
        inbox, news = self.inbox, self.news

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802   /news?after=N -> messages N, N+1, ...
                after = int(self.path.split("after=")[1]) if "after=" in self.path else 0
                body = json.dumps({"news": news[after:], "next": len(news)}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):  # noqa: N802
                text = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}").get("text", "")
                done, answer = threading.Event(), {}
                inbox.put((text, answer, done))
                done.wait(timeout=10.0)
                body = json.dumps({"reply": answer.get("reply", "(no answer: is the simulation still running?)")}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def take(self):
        """All commands that arrived since the last call: [(text, answer dict, done event), ...]."""
        out = []
        while not self.inbox.empty():
            out.append(self.inbox.get())
        return out


def parse_script(text):
    """ "1:go; 20:turn left" -> [(1.0, "go"), (20.0, "turn left")] """
    out = []
    for part in text.split(";"):
        if ":" in part:
            t, cmd = part.split(":", 1)
            out.append((float(t), cmd.strip()))
    return sorted(out)


# ---------------------------------------------------------------- the robot's state
class Robot:
    """What the robot is doing: its route, its current path (the route or a manoeuvre), and its mode."""

    def __init__(self, router, route, driver, speed, bot, ay_curve):
        self.router, self.route, self.driver, self.speed = router, route, driver, speed
        self.bot, self.ay_curve = bot, ay_curve
        self.path = route.path      # path the NMPC follows
        self.stop_at = route.path.length - 1.0
        self.mode = "drive"
        self.maneuver_end = None    # pull over: where to stop; rejoin: where the shift back ends (route s)
        self.s = 0.0                # distance along the route [m]
        self.lat = 0.0              # sideways offset from the route [m], + left
        self.v = 0.0                # speed from progress along the path being driven, last 0.5 s [m/s]
        self.said_dead_end = False
        self.history = []           # recent (t, distance along the driven path) for the speed
        self.history_path = None    # the path those distances are on
        self.u_turn = None          # during a U-turn: (route after it, where the half circle ends, where it joins)
        self.announce_stop = False  # say "Stopped." once the robot has actually stopped
        self.announced_turns = set()  # junctions already announced with "Turning left now."
        self.speed_goal = None        # the speed the robot was last told to reach
        self.announced_speed = False  # has it said "Reached ..." for that goal
        self.nudge = 0.0              # how far to drive beside the lane centre (+ left) until the next junction [m]
        self.offset_done = None       # route distance where the current sideways path is finished
        self.shift_message = None     # said when a lane change is done
        self.lane_force = None        # after a lane change: (road, lane rank) kept on that road
        self.turnback_index = None    # going round a block to turn back: index of the road going back
        self.pending = None           # a move asked for during a U-turn, rejoin or emergency stop
        self.is_paved = None

    # -------- route
    def use_route(self, roads, keep_lane=False):
        """Switch to a new route that starts at the current road (``keep_lane``: in the lane the robot is in)."""
        first = self.route.road_index_at(self.s)
        force = {0: self.lane_force[1]} if self.lane_force and roads[0] == self.lane_force[0] else None
        if keep_lane and force is None:
            _, lanes = self.lanes_here()
            mine = [lane for lane in lanes or [] if lane[0] >= 0]
            if mine:
                force = {0: min(mine, key=lambda lane: abs(lane[1]))[0]}
        new = self.router.build(roads, force)
        print(f"[talk] new route: {' '.join(f'{r}{d:+d}'[:-1] for r, d in roads[:10])} ...", flush=True)
        self.s = self.s - self.route.road_start_s[first]
        if self.turnback_index is not None:
            self.turnback_index -= first
        self.route = new
        self.s, self.lat = new.path.project(*self.position, self.s, window=40.0)
        if self.mode == "drive":
            self.follow_route()

    def follow_route(self):
        """Drive mode: follow the route's lane centre, or a path beside it while the robot is off the
        centre (after a lane change) or was told to keep to one side (until the next junction)."""
        if abs(self.lat) < 0.5 and self.nudge == 0.0:   # small errors: the NMPC closes them itself
            self.path, self.offset_done = self.route.path, None
            self.driver.set_path(self.route.path, self.s)
        else:
            junction = self.next_junction_s()
            s_back = junction - 2.0 if self.nudge and junction is not None else None
            self.path, shift_end = offset_path(self.route.path, self.s, self.lat, self.nudge, max(self.v, 3.0), s_back)
            self.offset_done = s_back if s_back is not None else (shift_end if self.nudge == 0.0 else None)
            self.driver.set_path(self.path, 0.0)
        self.stop_at = self.path.length - 1.0

    def next_junction_s(self):
        """Route distance where the next junction starts (None if there is none on the route)."""
        for i in range(self.route.road_index_at(self.s) + 1, len(self.route.roads)):
            if self.router.is_junction_road(self.route.roads[i]):
                return self.route.road_start_s[i]
        return None

    def extend_route_if_short(self):
        if self.mode != "drive" or self.route.path.length - self.s > KEEP_ROUTE / 2:
            return None
        first = self.route.road_index_at(self.s)
        roads = self.router.keep_going(self.route.roads[first:], KEEP_ROUTE + self.s - self.route.road_start_s[first])
        if len(roads) == len(self.route.roads) - first:
            if not self.said_dead_end and self.route.path.length - self.s < 100:
                self.said_dead_end = True
                return f"The road ends in {self.route.path.length - self.s:.0f} m; I will stop there."
            return None
        self.use_route(roads, keep_lane=True)
        return None

    # -------- commands
    def handle(self, text, is_paved, llm_url=None):
        self.is_paved = is_paved
        cmd = parse_command(text)
        understood = ""
        if cmd.kind == "unknown" and llm_url:
            # not a phrase the robot knows: ask the language model what it means
            cmd, say = interpret_with_llm(text, self.context(), llm_url)
            if cmd is None:
                return say or f"Sorry, I didn't understand '{text}'. Try: {HELP}"
            understood = f"({say}) " if say else ""
        return understood + self.do(cmd, is_paved, text)

    def context(self):
        return f"speed {self.v:.1f} m/s, mode {self.mode}. {self.whats_ahead()}"

    def do(self, cmd, is_paved, text=""):
        v = self.v
        if cmd.kind == "help":
            return ("I can go, stop, do an emergency stop, pull over, turn left or right or go straight at the "
                    "next junction, do a U-turn, change lanes, steer a little left or right in my lane, "
                    "accelerate or brake a bit more each time you ask, hold or set a speed, and tell you "
                    "what's ahead or my status. Say it in your own words.")
        if cmd.kind == "reset":
            return self.reset()
        if self.mode == "tipped":
            return "I've tipped over. Say 'reset' to put me back on the road."
        if cmd.kind in ("emergency", "stop"):
            self.pending = None
        if cmd.kind in MOVES:
            return self.move(cmd, v)
        if cmd.kind == "emergency":
            self.mode = "emergency"
            self.speed.hold(0.0)
            return "EMERGENCY BRAKING."
        if cmd.kind == "go":
            return self.go()
        if cmd.kind == "stop":
            self.announce_stop = True
            if self.mode == "pull_over":
                self.mode = "drive"   # give up the parking spot: stop where we are, in the lane we are in
                self.follow_route()
            return self.speed.stop(v)
        if cmd.kind in ("accelerate", "brake", "hold", "set_speed"):
            if self.mode in ("emergency", "parked", "pull_over"):
                return f"I'm {self.mode.replace('_', ' ')}; say 'go' first."
            if cmd.kind == "accelerate":
                return self.speed.accelerate(v)
            if cmd.kind == "brake":
                self.announce_stop = True
                return self.speed.brake(v)
            if cmd.kind == "hold":
                return self.speed.hold(v)
            return self.speed.set_speed(v, cmd.value)
        if cmd.kind == "pull_over":
            return self.pull_over(is_paved)
        if cmd.kind == "ahead":
            return self.whats_ahead()
        if cmd.kind == "status":
            return (f"Speed {v:.1f} m/s (set-point {self.speed.v_set:.1f}, rate {self.speed.rate:+.1f} m/s^2), "
                    f"mode {self.mode}. " + self.whats_ahead())
        return f"Sorry, I didn't understand '{text}'. Try: {HELP}"

    def go(self):
        if self.mode == "parked":
            # pull back into the lane: shift from the kerb to the route over a gentle distance
            length = shift_length(self.lat, 5.0)
            self.path = sideways_path(self.route.path, self.s, self.lat, self.s + 2.0, 0.0, length, self.s + length + 40.0)
            self.stop_at = self.path.length - 1.0
            self.maneuver_end = self.s + length + 2.0
            self.driver.set_path(self.path, 0.0)
            self.mode = "rejoin"
            return self.speed.go(self.v).replace("Driving", "Pulling back into the lane")
        if self.mode == "emergency":
            self.mode = "drive"
        if self.mode == "pull_over":
            return "I'm pulling over; wait until I've stopped, then say 'go'."
        return self.speed.go(self.v)

    def move(self, cmd, v):
        """Turns, U-turns, lane changes and steering: done now if possible (starting off if the robot
        is stopped), or right after the U-turn / pulling back into the lane / emergency stop."""
        if self.mode == "u_turn" or self.mode == "rejoin" or (self.mode == "emergency" and abs(v) > STILL):
            self.pending = cmd
            after = {"u_turn": "I've turned around", "rejoin": "I'm back in the lane", "emergency": "I've stopped"}
            return f"OK, I'll do that as soon as {after[self.mode]}."
        if self.mode == "parked" and cmd.kind != "u_turn":
            self.pending = cmd
            return self.go() + " Then I'll do that."
        saved = (self.mode, self.path, self.stop_at, self.driver.s_hint)
        if self.mode in ("emergency", "pull_over", "parked"):
            self.mode = "drive"   # a U-turn from a parking spot starts at the kerb (turn_around uses self.lat)
            self.announce_stop = False
        was_stopped = self.speed.goal <= 0.0
        if cmd.kind == "turn":
            ok, reply = self.turn(cmd.value)
        elif cmd.kind == "u_turn":
            ok, reply = self.turn_around()
        elif cmd.kind == "lane":
            ok, reply = self.change_lane(cmd.value)
        elif cmd.kind == "nudge":
            ok, reply = self.steer(cmd.value)
        else:
            ok, reply = self.centre()
        if not ok:
            if saved[0] != self.mode:   # nothing changed: stay as we were
                self.mode, self.path, self.stop_at = saved[:3]
                self.driver.set_path(self.path, saved[3])
            return reply
        if self.mode == "drive" and self.path is saved[1] and saved[0] == "pull_over":
            self.follow_route()         # gave up the parking spot: back to the lane
        if was_stopped:
            reply += " " + self.speed.go(v).replace("Driving: s", "S")
        return reply

    def turn(self, kind):
        min_dist = MIN_TURN_DIST   # however close: the robot brakes as hard as it must (see hard_brake_target)
        roads, junction, move = self.router.plan_turn(self.route, self.s, kind, min_dist, LOOK_AHEAD, KEEP_ROUTE)
        if roads is not None:
            distance = junction.s - self.s
            self.use_route(roads)
            return True, f"OK: {describe(move)} at the junction in {distance:.0f} m."
        if junction is not None:
            return True, f"Already going {kind} at the junction in {junction.s - self.s:.0f} m."
        too_close = [j for j in self.router.junctions_ahead(self.route, self.s, min_dist)
                     if any(m.kind == kind for m in j.moves)]
        if too_close and too_close[0].taken is not None and too_close[0].taken.kind == kind:
            return True, f"Already going {kind} at the junction in {too_close[0].s - self.s:.0f} m."
        if too_close:
            what = "go straight" if kind == "straight" else f"turn {kind}"
            return False, (f"Too close to {what} at the junction in {too_close[0].s - self.s:.0f} m at this speed, "
                           f"and there is no other {kind} turn within {LOOK_AHEAD:.0f} m.")
        return False, f"No {kind} turn possible within the next {LOOK_AHEAD:.0f} m."

    def lanes_here(self):
        """(road index, lanes beside the robot from RoadRouter.lanes_beside) or (index, None) in a junction."""
        i = self.route.road_index_at(self.s)
        node = self.route.roads[i]
        if self.router.is_junction_road(node):
            return i, None
        x, y = self.position
        return i, self.router.lanes_beside(node, x, y, self.route.path.heading(self.s))

    def change_lane(self, side):
        i, lanes = self.lanes_here()
        if lanes is None:
            return False, "I'm crossing a junction; ask again once I'm through it."
        mine = [lane for lane in lanes if lane[0] >= 0]          # lanes going my way
        others = [lane for lane in mine if abs(lane[1]) > 1.0]
        if side is None:   # "change lanes": the nearest other lane
            if not others:
                return False, "There is only one lane going my way here."
            side = "left" if min(others, key=lambda lane: abs(lane[1]))[1] > 0 else "right"
        sign = 1.0 if side == "left" else -1.0
        choices = [lane for lane in others if lane[1] * sign > 0]
        if not choices:
            if side == "left" and any(lane[0] == -1 and lane[1] > 0 for lane in lanes):
                return False, ("The lane on my left is for oncoming traffic, so I won't drive in it. "
                               "Say 'steer left' to move over a little in my lane.")
            return False, f"There is no lane to my {side} on this road."
        rank, distance, _ = min(choices, key=lambda lane: abs(lane[1]))
        road_end = (self.route.road_start_s[i + 1] if i + 1 < len(self.route.roads) else self.route.path.length) - self.s
        if road_end < shift_length(distance, max(self.v, 3.0)) + 5.0:
            return False, f"Too close to the end of this road to change lanes ({road_end:.0f} m left); ask again after the junction."
        self.lane_force, self.nudge = (self.route.roads[i], rank), 0.0
        self.use_route(self.route.roads[i:])
        self.shift_message = f"In the {side} lane now."
        return True, f"Changing to the {side} lane."

    def steer(self, side):
        """Move NUDGE metres to the side inside the lane (back to the centre at the next junction)."""
        _, lanes = self.lanes_here()
        if lanes is None:
            return False, "I'm crossing a junction; ask again once I'm through it."
        width = min((lane for lane in lanes if lane[0] >= 0), key=lambda lane: abs(lane[1]))[2]
        limit = max(0.0, width / 2 - HALF_WIDTH - 0.1)
        target = float(np.clip(self.nudge + (NUDGE if side == "left" else -NUDGE), -limit, limit))
        if abs(target - self.nudge) < 0.05:
            return False, (f"I'm already as far {side} as I can go in my lane ({width:.1f} m wide). "
                           f"Say 'change lane {side}' to move into the next lane.")
        junction = self.next_junction_s()
        if junction is not None and junction - self.s < 2 * shift_length(target, max(self.v, 3.0)) + 5.0:
            return False, f"The junction is only {junction - self.s:.0f} m ahead; I'll stay in the middle of my lane for it."
        self.nudge = target
        self.follow_route()
        where = "the lane centre" if target == 0.0 else f"{abs(target):.1f} m {'left' if target > 0 else 'right'} of the lane centre"
        return True, f"Moving over to the {side}: now {where}, until the next junction."

    def centre(self):
        if self.nudge == 0.0 and abs(self.lat) < 0.3:
            return True, "I'm in the middle of my lane."
        self.nudge = 0.0
        self.follow_route()
        return True, "Back to the middle of the lane."

    def pull_over(self, is_paved):
        if self.mode in ("parked", "pull_over"):
            return "Already pulling over." if self.mode == "pull_over" else "Already parked."
        spot = find_pull_over_spot(self.route, self.router, self.s, self.v, is_paved, HALF_WIDTH, LOOK_AHEAD)
        if spot is None:
            return f"No safe place to pull over within the next {LOOK_AHEAD:.0f} m (junctions or no room at the side)."
        s_stop, offset, s_start, length = spot
        self.path = sideways_path(self.route.path, self.s, self.lat, s_start, offset, length, s_stop + 15.0)
        self.stop_at = s_stop - self.s    # the new path starts where the robot is
        self.maneuver_end = s_stop
        self.driver.set_path(self.path, 0.0)
        self.mode = "pull_over"
        self.speed.hold(max(self.v, 3.0))   # from standstill, creep to the spot at 3 m/s
        return f"Pulling over to the right in {s_stop - self.s:.0f} m (moving {abs(offset):.1f} m to the side)."

    def turn_around(self):
        """U-turn into the lane going the other way, starting where the robot is (from the kerb if
        parked). Where this road is one way or too narrow: the next road ahead where it fits (if
        close), otherwise round the nearest roundabout or block and back."""
        spot = self.u_turn_spot()
        loop = None if spot is not None and spot[0] <= NEAR_U_TURN else \
            (self.router.route_back(self.route, self.s) or self.router.route_back(self.route, self.s, near=80.0))
        if loop is not None:
            i = self.route.road_index_at(self.s)
            length = (self.route.road_start_s[i + 1] - self.s if i + 1 < len(self.route.roads) else 0.0) + \
                sum(self.router.od.roads[r].length for r, _ in loop[1:-1])
            if spot is not None and spot[0] < length / 2:
                loop = None   # the U-turn spot further on is still the shorter way back
        if loop is not None:
            self.nudge = 0.0
            self.use_route(self.router.keep_going(loop, KEEP_ROUTE))
            self.turnback_index = len(loop) - 1
            junctions = sum(1 for node in loop if self.router.is_junction_road(node))
            return True, (f"No room to turn around here, so I'll go round and come back the other way "
                          f"({junctions} junction{'s' if junctions != 1 else ''}, about {length:.0f} m).")
        if spot is not None:
            return self.start_u_turn(*spot)
        return False, f"No place to turn around within the next {LOOK_AHEAD:.0f} m (one-way or too narrow)."

    def u_turn_spot(self):
        """The nearest place ahead for a U-turn on the road itself:
        (distance to it, path, where the half circle ends, where it joins the other lane, route back)."""
        od = self.router.od
        lead = 2.0 + 0.5 * max(self.v, 0.0)    # where the half circle can start at the earliest
        for i in range(self.route.road_index_at(self.s), len(self.route.roads)):
            rid, d = self.route.roads[i]
            if self.router.is_junction_road((rid, d)) or not od.drivable(rid, -d):
                continue
            start = max(self.route.road_start_s[i] + 2.0, self.s + lead)
            end = (self.route.road_start_s[i + 1] if i + 1 < len(self.route.roads) else self.route.path.length) - 2.0
            if start - self.s > LOOK_AHEAD:
                break
            if end <= start:
                continue
            back = self.router.build(self.router.keep_going([(rid, -d)], KEEP_ROUTE), force={0: 0})   # nearest lane back
            result = u_turn_path(self.route.path, self.s, start, back.path, lat_now=self.lat)
            if result is not None:
                return (start - self.s, *result, back)
        return None

    def start_u_turn(self, distance, path, arc_end, join, back):
        self.u_turn = (back, arc_end, join)
        self.path, self.stop_at = path, path.length - 1.0
        self.driver.set_path(path, 0.0)
        self.mode, self.nudge, self.lane_force = "u_turn", 0.0, None
        return True, ("OK: turning around now." if distance < 5.0 else f"OK: turning around in {distance:.0f} m.")

    def reset(self):
        """Put the robot back on its wheels on the route where it is, stopped."""
        s = max(self.s, 0.0)
        x, y = self.route.path.xy[min(int(s / self.route.path.ds), len(self.route.path.xy) - 1)]
        self.bot.reset_to((x, y, self.route.path.heading(s)), settle_steps=30)
        self.driver.reset(s)
        self.mode, self.nudge, self.lat, self.lane_force = "drive", 0.0, 0.0, None
        self.follow_route()
        self.speed.hold(0.0)
        self.history, self.u_turn, self.pending, self.turnback_index = [], None, None, None
        return "Back on my wheels on the road. Say 'go' to drive."

    def hard_brake_target(self):
        """A bend on the path ahead too sharp to reach with normal braking (a turn or U-turn commanded
        at the last moment): the safe speed for it (the emergency brake gets there), otherwise None."""
        if self.mode in ("emergency", "parked", "tipped"):
            return None
        path, s0, v = self.driver.path, self.driver.s_hint, max(self.v, 0.0)
        ahead = np.arange(s0, min(path.length, s0 + 15.0 + v * v / (2 * NORMAL_DECEL)), 0.5)
        if len(ahead) < 2:
            return None
        safe = np.sqrt(self.ay_curve / np.maximum(np.abs(path.kappa_many(ahead)), 1e-3))
        need = (v * v - safe * safe) / (2 * np.maximum(ahead - s0, 0.5))
        too_fast = (need > NORMAL_DECEL) & (v > safe + 0.5)
        return float(safe[too_fast].min()) if too_fast.any() else None

    def whats_ahead(self):
        junctions = self.router.junctions_ahead(self.route, self.s, LOOK_AHEAD)
        if not junctions:
            return f"No junctions in the next {LOOK_AHEAD:.0f} m."
        parts = []
        for j in junctions:
            exits = ", ".join(sorted({m.kind for m in j.moves}))
            parts.append(f"in {j.s - self.s:.0f} m: {exits} (I'll go {j.taken.kind if j.taken else '?'})")
        return "Junctions " + "; ".join(parts) + "."

    # -------- every control step
    def update(self, position, t, dt):
        """Track where the robot is, finish manoeuvres, and give the speed cap for this step."""
        self.position = position
        self.s, self.lat = self.route.path.project(*position, self.s, window=30.0)
        # speed from how far the robot got along the route in the last 0.5 s (the body's own
        # velocity reading jitters when the robot stands with its brakes on)
        if self.driver.path is not self.history_path:   # a new path: distances start again
            self.history, self.history_path = [], self.driver.path
        self.history = [(tt, ss) for tt, ss in self.history if t - tt <= 0.5] + [(t, self.driver.s_hint)]
        t0, s0 = self.history[0]
        if t - t0 > 0.2:
            self.v = (self.driver.s_hint - s0) / (t - t0)
        messages = []
        self.speed.update(dt)
        # "Reached ..." only when the robot itself (not its set-point, not its wheels) is at that speed
        goal = self.speed.goal if self.speed.rate == 0.0 and self.speed.goal > 0.0 else None
        if goal != self.speed_goal:            # a new goal: announce it once it is really reached
            self.speed_goal, self.announced_speed = goal, False
        if goal is not None and not self.announced_speed and abs(self.v - goal) < 0.3 and self.mode in ("drive", "rejoin"):
            self.announced_speed = True
            messages.append(f"Reached {goal:.1f} m/s, holding.")
        if self.announce_stop and abs(self.v) < 0.05 and self.speed.v_set == 0.0:
            self.announce_stop = False
            messages.append("Stopped.")

        if self.mode == "pull_over":
            s_on_path = self.driver.s_hint
            cap = speed_to_stop_within(self.stop_at - s_on_path, PARK_BRAKE, 0.6)
            if abs(self.v) < 0.05 and self.stop_at - s_on_path < 1.5:
                self.mode = "parked"
                self.speed.hold(0.0)
                messages.append("Parked at the side of the road. Say 'go' to continue.")
            return min(cap, self.speed.v_set), messages
        if self.mode in ("parked", "tipped"):
            return 0.0, messages
        if self.mode == "u_turn":
            back, arc_end, join = self.u_turn
            if self.driver.s_hint > arc_end + 3.0:    # round the half circle: drive on in the other lane
                self.route, self.u_turn, self.mode = back, None, "drive"
                self.s, self.lat = back.path.project(*position, join + self.driver.s_hint - arc_end, window=30.0)
                self.follow_route()
                self.announced_turns = set()
                messages.append("Turned around.")
            return self.speed.v_set, messages
        if self.mode == "emergency":
            if abs(self.v) < 0.05 and self.pending is not None:   # stopped: now do what was asked meanwhile
                command, self.pending = self.pending, None
                messages.append(self.do(command, self.is_paved))
                return self.speed.v_set, messages
            if abs(self.v) < 0.05 and not getattr(self, "_said_stopped", False):
                self._said_stopped = True
                messages.append("Stopped. Say 'go' to continue.")
            return 0.0, messages
        self._said_stopped = False
        if self.mode == "rejoin" and self.s > self.maneuver_end and abs(self.lat) < 0.3:
            self.mode = "drive"
            self.follow_route()
            messages.append("Back in the lane.")
        if self.mode == "drive" and self.offset_done is not None and self.s > self.offset_done + 1.0:
            if self.nudge != 0.0:
                self.nudge = 0.0
                messages.append("Back in the middle of the lane for the junction.")
            self.follow_route()
            if self.shift_message:
                messages.append(self.shift_message)
                self.shift_message = None
        if self.lane_force and self.route.roads[self.route.road_index_at(self.s)] != self.lane_force[0]:
            self.lane_force = None
        if self.turnback_index is not None and self.route.road_index_at(self.s) >= self.turnback_index:
            self.turnback_index = None
            messages.append("Turned around.")
        if self.mode == "drive" and self.pending is not None:   # asked for during a U-turn or rejoin
            command, self.pending = self.pending, None
            messages.append(self.do(command, self.is_paved))
        for j in self.router.junctions_ahead(self.route, self.s, 15.0 + 1.5 * max(self.v, 0.0)):
            key = j.taken.connector if j.taken is not None else None
            if key is not None and j.taken.kind != "straight" and key not in self.announced_turns:
                self.announced_turns.add(key)
                messages.append(f"{describe(j.taken).capitalize()} now.")
        extended = self.extend_route_if_short()
        if extended:
            messages.append(extended)
        return self.speed.v_set, messages


# ---------------------------------------------------------------- drawing
def draw_route(line_drawer, robot):
    """Green: the route ahead; magenta: a manoeuvre; yellow dots: junctions where the route turns."""
    line_drawer.clear_lines()
    line_drawer.clear_points()
    path = robot.path
    s0 = robot.driver.s_hint if path is not robot.route.path else robot.s
    points = path.xy[int(s0 / path.ds)::8][:120]
    heights = road_height(points[:, 0], points[:, 1]) + 0.1
    p3 = [(float(x), float(y), float(z)) for (x, y), z in zip(points, heights) if np.isfinite(z)]
    colour = (0.1, 0.9, 0.2, 1) if path is robot.route.path else (1.0, 0.2, 1.0, 1)
    if len(p3) > 1:
        line_drawer.draw_lines(p3[:-1], p3[1:], [colour] * (len(p3) - 1), [4] * (len(p3) - 1))
    dots = []
    for j in robot.router.junctions_ahead(robot.route, robot.s, LOOK_AHEAD):
        if j.taken is not None and j.taken.kind != "straight":
            x, y = robot.route.path.xy[int(j.s / robot.route.path.ds)]
            dots.append((float(x), float(y), float(road_height(x, y)[0]) + 1.5))
    if dots:
        line_drawer.draw_points(dots, [(1.0, 0.85, 0.0, 1)] * len(dots), [20] * len(dots))


# ---------------------------------------------------------------- main
def main():
    out = Path(args.out) if os.path.isabs(args.out) else REPO / args.out
    out.mkdir(parents=True, exist_ok=True)

    nmpc = load_tadpole_stack(args.nmpc_dir, out / "solver_build", ay_curve=args.ay_curve)
    TS = float(nmpc.P.NMPC_TS)
    router = RoadRouter(OpenDrive(MCITY_XODR), mcity_to_sim)
    start_road = (args.start.split(",")[0], int(args.start.split(",")[1]))
    route = router.build(router.keep_going([start_road], KEEP_ROUTE))
    start = (route.path.xy[0][0], route.path.xy[0][1], route.path.heading(0.0))

    bot = IsaacJetbot(env="mcity", robot=args.robot, physics_dt=TS / 2, start_pose=start, render=True)
    J.build_paved_mask(bot.world.stage, "mcity")
    bot.enable_torque_mode()
    bot.params.max_steer = float(np.deg2rad(nmpc.P.MAX_STEER_DEG))
    bot.reset_to(start, settle_steps=60)
    loads = RoadLoadModel(parse_scenario("nominal"), nmpc.P, route.path.length, dt=TS)
    driver = TadpoleDriver(nmpc, bot, loads, BodyForceApplier("/World/Jetbot/chassis"))
    driver.set_path(route.path, 0.0)
    robot = Robot(router, route, driver, SpeedCommander(float(nmpc.P.V_MAX)), bot, float(nmpc.P.AY_CURVE))
    robot.position = start[:2]

    try:
        from isaacsim.util.debug_draw import _debug_draw

        line_drawer = _debug_draw.acquire_debug_draw_interface()
    except ImportError:
        line_drawer = None
    recorder = ViewportRecorder(out / "frames", 0.5) if args.record else None
    server = CommandServer(args.port)
    script = parse_script(args.script)
    print(f"[talk] ready. Send commands with:  .venv/bin/python scripts/say.py \"go\"   (port {args.port})", flush=True)
    print(f"[talk] commands: {HELP}", flush=True)

    conversation = []
    log = {k: [] for k in ("t", "x", "y", "psi", "vx", "v_ref", "v_set", "mode", "s_route", "lat", "fx_brake",
                           "s_path", "path_length", "cte", "delta_cmd", "status", "plan_mode", "v_cap")}
    step, quit_now = 0, False
    while step * TS < args.max_time and app.is_running() and not quit_now:
        t = step * TS

        # 1. commands: typed ones (HTTP) and scripted ones that are due
        incoming = server.take()
        while script and script[0][0] <= t:
            incoming.append((script.pop(0)[1], {}, threading.Event()))
        for text, answer, done in incoming:
            if text.strip().lower() in ("quit", "exit"):
                reply, quit_now = "Bye.", True
            else:
                reply = robot.handle(text, J.is_paved, args.llm_url)
            print(f"[you] {text}\n[bot] {reply}", flush=True)
            conversation.append(dict(t=t, said=text, reply=reply))
            answer["reply"] = reply
            done.set()

        # 2. where is the robot, what speed is allowed, any news
        position, _ = bot.robot.get_world_pose()
        v_cap, messages = robot.update((float(position[0]), float(position[1])), t, TS)
        for message in messages:
            print(f"[bot] {message}", flush=True)
            conversation.append(dict(t=t, said=None, reply=message))
            server.news.append(message)

        # 3. the NMPC drives one step
        # emergency: the full emergency brake while moving; once (nearly) stopped, or whenever the
        # set-point is zero and the robot is slow, hold it still with HOLD_BRAKE
        slow = abs(robot.v) < STILL
        emergency = robot.mode == "emergency" and not slow
        hold = (robot.mode in ("emergency", "parked", "tipped") or v_cap <= 0.0) and slow
        turn_speed = robot.hard_brake_target()
        row = driver.tick(t, robot.stop_at, v_cap=max(v_cap, 0.0), stop_lag=0.6, emergency=emergency,
                          hold_brake=HOLD_BRAKE if hold else 0.0, hard_brake_to=turn_speed)
        for key, value in (("t", t), ("x", row["x"]), ("y", row["y"]), ("psi", row["psi"]), ("vx", row["vx"]),
                           ("v_ref", row["v_ref"]), ("v_set", robot.speed.v_set), ("mode", MODES[robot.mode]),
                           ("s_route", robot.s), ("lat", robot.lat), ("fx_brake", row["fx_brake"]),
                           ("s_path", row["s"]), ("path_length", driver.path.length), ("cte", row["cte"]),
                           ("delta_cmd", row["delta_cmd"]), ("status", row["status"]), ("plan_mode", row["plan_mode"]),
                           ("v_cap", v_cap)):
            log[key].append(value)
        if step % 250 == 0:
            print(f"[talk] t={t:6.1f}s v={row['vx']:5.2f} m/s set {robot.speed.v_set:5.2f} mode {robot.mode}", flush=True)

        # 4. view
        if line_drawer is not None and step % 10 == 0:
            draw_route(line_drawer, robot)
        if not args.headless:
            chase_view(row["x"], row["y"], row["psi"], bot.ground_height(row["x"], row["y"]), 8.0, 4.0, 8, 0.5)
        if recorder is not None:
            recorder.capture(t)
        driver.step(render=True)
        step += 1
        if robot.mode != "tipped" and (abs(row["roll"]) > math.radians(45) or abs(row["pitch"]) > math.radians(45)):
            robot.mode = "tipped"
            robot.speed.hold(0.0)
            message = "I've tipped over. Say 'reset' to put me back on the road."
            print(f"[bot] {message}", flush=True)
            conversation.append(dict(t=t, said=None, reply=message))
            server.news.append(message)

    (out / "conversation.json").write_text(json.dumps(conversation, indent=1))
    np.savez_compressed(out / "log.npz", **{k: np.asarray(v) for k, v in log.items()})
    print(f"[talk] saved {out}", flush=True)
    if recorder is not None:
        recorder.save_gif(app, out / "run.gif", 125)
    server.server.shutdown()
    app.close()


if __name__ == "__main__":
    main()
