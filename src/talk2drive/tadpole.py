"""The CARLA tadpole NMPC stack (~/Documents/carla_nmpc_project) driving a car-like Isaac robot.

Nothing in the CARLA project is copied or edited: its modules are imported and called the way its
route-following mode calls them in run_nmpc_carla.py, every NMPC_TS (20 ms):

    stitch_start -> RouteSpeedPlanner.plan (v_cap, AY_CURVE) -> TadpoleNMPCController.step
    -> InnerLoopTracker.track_speed_2dof (drive / brake split)
    -> LongitudinalForceDisturbanceObserver (dF_unmod for the next solve)

``TadpoleDriver.tick`` does one such cycle on an ``IsaacJetbot`` in torque mode and returns what
happened; ``TadpoleDriver.step`` advances the physics by one NMPC period.
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

MAGNA_L, MAGNA_LF = 1.292, 0.65  # wheelbase, CG behind the front axle (tadpole_params: LF = 0.65/1.292 L)
PLAN_MODES = {"stitched": 0, "speed": 1, "measured": 2}


def load_tadpole_stack(nmpc_dir: str, build_dir: Path, ay_curve: float | None = None) -> SimpleNamespace:
    """Import the project's NMPC stack with the Magna geometry; compile its solver in ``build_dir``.

    Set NMPC_V_MAX / DRIVE_STYLE / NMPC_SPEED_HORIZON in the environment first (the project reads
    them at import). ``ay_curve`` overrides the speed plan's bend limit (capped at AY_MAX)."""
    sys.path.insert(0, nmpc_dir)
    import tadpole_params as P

    P.L, P.LF = MAGNA_L, MAGNA_LF
    P.LR = P.L - P.LF
    P.KUS = P.M * (P.LR * P.CR - P.LF * P.CF) / (P.L * P.CF * P.CR)
    P.POSE_POINT_TO_CG_X = P.LR
    P.CARLA_ACTOR_TO_REAR_X = P.LR
    if ay_curve is not None:  # planning value only; AY_MAX (the rollover envelope rule) is unchanged
        P.AY_CURVE = P.AY_PLAN = min(float(ay_curve), float(P.AY_MAX))
    build_dir.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(build_dir)  # acados writes c_generated_code/ and the solver json into the working directory
    try:
        from nmpc_controller import TadpoleNMPCController

        ctrl = TadpoleNMPCController()
    finally:
        os.chdir(cwd)
    from force_disturbance_observer import LongitudinalForceDisturbanceObserver
    from inner_loop import InnerLoopTracker
    from trajectory import RouteSpeedPlanner, stitch_start

    inner = InnerLoopTracker(m=P.M, Tw=P.TW_FRONT, kp_v=1000.0, ki_v=200.0, ki_v_2dof=30.0,
                             v_integral_max_2dof=20.0, Fx_drive_max=P.FX_DRIVE_MAX, Fx_brake_max=P.FX_BRAKE_MAX)
    return SimpleNamespace(P=P, ctrl=ctrl, planner=RouteSpeedPlanner(), stitch_start=stitch_start,
                           observer=LongitudinalForceDisturbanceObserver(Ts=P.NMPC_TS), inner=inner)


class TadpoleDriver:
    """One NMPC cycle per ``tick``: measure, plan, solve, split drive/brake, apply to the robot."""

    def __init__(self, stack: SimpleNamespace, bot, loads, body_forces, grade_governor: bool = True):
        self.k = stack
        self.P = stack.P
        self.bot = bot
        self.loads = loads  # talk2drive.disturbances.RoadLoadModel (the plant's road loads)
        self.body_forces = body_forces  # BodyForceApplier on the chassis
        self.grade_governor = grade_governor
        self.TS = float(self.P.NMPC_TS)
        self.path = None
        self.reset()

    def reset(self, s_hint: float = 0.0) -> None:
        self.k.ctrl.reset()
        self.k.observer.reset(vx=0.0)
        self.k.inner.reset()
        self.prev = None
        self.followed = None
        self.fx_prev = 0.0
        self.ax_f = 0.0
        self.n = 0
        self.s_hint = s_hint

    def set_path(self, path, s_hint: float = 0.0) -> None:
        """Follow a new path (the next plan restarts from the measured state if the geometry changed)."""
        self.path = path
        self.s_hint = s_hint

    def measure(self) -> dict:
        """Pose and velocities at the centre of mass (as the CARLA actor reports them). The robot's
        origin is on the ground under the rear axle, 0.45 m below the CG, so body pitch/roll would swing
        it fore-aft and sideways and read as phantom acceleration."""
        P, TS = self.P, self.TS
        pos, quat = self.bot.robot.get_world_pose()
        st = self.bot.get_state()
        psi, delta = float(st[2]), float(st[4])
        qw, qx, qy, qz = (float(v) for v in quat)
        Rb = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                       [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                       [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
        pos = np.asarray(pos, dtype=float)
        cg = pos + Rb @ np.array([P.LR, 0.0, P.H_CG])
        if self.prev is None:
            vx = vy = r = 0.0
        else:
            vwx, vwy = (cg[0] - self.prev[0]) / TS, (cg[1] - self.prev[1]) / TS
            r = ((psi - self.prev[2] + math.pi) % (2 * math.pi) - math.pi) / TS
            c, s_ = math.cos(psi), math.sin(psi)
            vx, vy = c * vwx + s_ * vwy, -s_ * vwx + c * vwy
        ax = 0.0 if self.prev is None else (vx - self.prev[3]) / TS
        self.ax_f += 0.3 * (ax - self.ax_f)
        self.prev = (float(cg[0]), float(cg[1]), psi, vx)
        roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))  # + = nose down
        return dict(pos=pos, Rb=Rb, cg=cg, X=float(pos[0]), Y=float(pos[1]), psi=psi, delta=delta,
                    vx=vx, vy=vy, r=r, ax=ax, roll=roll, pitch=pitch)

    def tick(self, t: float, s_stop: float, v_cap: float | None = None, stop_lag: float = 0.0) -> dict:
        """One control cycle along ``self.path``, stopping at ``s_stop``. Returns the log row.

        ``stop_lag`` [s]: the planner reaches a falling cap with a jerk-limited ramp, so the speed at
        which the robot can still stop by ``s_stop`` allows this long before full braking (0: none,
        fine for a lap that only has to end somewhere near the line)."""
        P, k, TS, path = self.P, self.k, self.TS, self.path
        m = self.measure()
        vx, vy, r, psi, delta = m["vx"], m["vy"], m["r"], m["psi"], m["delta"]
        s0, cte = path.project(m["X"], m["Y"], self.s_hint)
        self.s_hint = s0
        epsi = (psi - path.heading(s0) + math.pi) % (2 * math.pi) - math.pi

        obs = k.observer.update(vx=vx, vy=vy, r=r, delta=delta, Fx_command_previous_interval=self.fx_prev,
                                ax_body_measured=m["ax"] if self.n > 1 else None, mu_est=P.MU)
        dF = float(obs.estimate)

        # path end: stop there, braking at the 1B command limit
        a, d = float(P.AX_DECEL_CMD), max(0.0, s_stop - s0 - 0.5)
        cap = min(P.TARGET_V, a * (math.sqrt(stop_lag * stop_lag + 2.0 * d / a) - stop_lag))
        if v_cap is not None:
            cap = min(cap, v_cap)
        t0, s_st, v_st, a_st, parent, plan_mode = k.stitch_start(self.followed, path, t, 0.0, s0, vx, self.ax_f, cte)
        traj = k.planner.plan(path, t0, s_st, v_st, a_st, v_cap=cap, ay_plan=float(P.AY_CURVE),
                              duration=float(P.NMPC_HORIZON_T_MAX) + 1.4, parent=parent,
                              decel_max=float(P.TRAJ_DECEL_MAX))
        t_solve = time.perf_counter()
        cmds = k.ctrl.step(X_global=m["X"], Y_global=m["Y"], psi_global=psi, vx=vx, vy=vy, r=r, delta=delta,
                           s0=s0, path=path, v_plan=float(P.V_MAX), dF_unmod_est=dF,
                           ay_max=P.AY_MAX, ay_plan=float(P.AY_CURVE), mu_est=P.MU,
                           Mz_min_avail=0.0, Mz_max_avail=0.0, Mz_applied_est=0.0, sim_time=t,
                           trajectory=traj, v_plan_target=None, v_plan_rate=0.0, mu_front_est=P.MU)
        solve_ms = (time.perf_counter() - t_solve) * 1e3
        self.followed = traj
        fx_drive, fx_brake = k.inner.track_speed_2dof(Fx_ff=cmds["Fx_net_nmpc"], v_cmd=cmds["v_cmd"], v_meas=vx, dt=TS)
        fx_drive = float(np.clip(fx_drive, 0.0, P.FX_DRIVE_MAX))
        fx_brake = float(np.clip(fx_brake, 0.0, P.FX_BRAKE_MAX))
        if self.grade_governor:
            # GRADE_GOVERNOR: FX_DRIVE_MAX = M * AX_ACCEL_MAX bounds the acceleration on flat ground only.
            # Downhill, gravity adds g*sin(pitch): cap drive so drive + gravity stays within AX_ACCEL_MAX
            # less the 0.2 m/s^2 headroom the project keeps for braking (Rule 1B). Uphill, gravity and
            # road load add to the brakes: cap brake force so the total stays within AX_DECEL_MAX - 0.2.
            a_grav = P.G * math.sin(m["pitch"])  # + = downhill
            g_x = self.loads.grade(s0)  # an IMU would see the extra grade too (the NMPC never does)
            a_grav -= P.G * g_x / math.sqrt(1 + g_x * g_x)
            fx_drive = min(fx_drive, max(0.0, P.M * (P.AX_ACCEL_MAX - 0.2 - a_grav)))
            resist = self.loads.nominal_road_load(max(vx, 0.0))
            fx_brake = min(fx_brake, max(0.0, P.M * (P.AX_DECEL_MAX - 0.2 + a_grav) - resist))
        delta_cmd = float(np.clip(cmds["delta_cmd"], -np.deg2rad(P.MAX_STEER_DEG), np.deg2rad(P.MAX_STEER_DEG)))
        # Plant road loads: rolling resistance at the wheel, drag / side wind on the body at its centre of
        # pressure, extra grade at the CG (all nominal unless a disturbance scenario says otherwise).
        rl = self.loads.forces(s0, vx, vy, psi)
        cp = m["pos"] + m["Rb"] @ np.array([P.LR, 0.0, self.loads.h_cp])
        self.body_forces.set([(rl["f_aero_w"], cp), (rl["f_grade_w"], m["cg"])])
        self.bot.apply_forces(delta_cmd, fx_drive, fx_brake, front_brake_frac=P.FRONT_BRAKE_FRAC,
                              fx_resist=rl["f_rr"], dt=TS)
        self.fx_prev = fx_drive - fx_brake
        self.n += 1
        susp = list(self.bot.suspension_travel()) or [0.0, 0.0, 0.0]
        return dict(t=t, x=m["X"], y=m["Y"], psi=psi, vx=vx, vy=vy, r=r, delta=delta, delta_cmd=delta_cmd,
                    fx_drive=fx_drive, fx_brake=fx_brake, s=s0, cte=cte, epsi=epsi, v_ref=float(cmds["v_ref"]),
                    status=int(cmds["status"]), solve_ms=solve_ms, dF=dF, roll=m["roll"], pitch=m["pitch"],
                    z=float(m["pos"][2]), dF_true=rl["dF_true"] + P.M * P.G * math.sin(m["pitch"]),
                    grade_extra=rl["grade"], wind_x=float(rl["air_w"][0]), wind_y=float(rl["air_w"][1]),
                    f_aero_y=rl["f_aero_y"], plan_mode=PLAN_MODES.get(plan_mode, 3),
                    wheel_rear=float(getattr(self.bot, "last_wheel_speeds", [0.0])[0]),
                    susp_fl=susp[0], susp_fr=susp[1], susp_r=susp[2])

    def step(self, render: bool) -> None:
        """Advance the physics by one NMPC period (two physics steps), body forces on each."""
        self.body_forces.apply()
        self.bot.step(render=False)
        self.body_forces.apply()  # external forces last one physics step
        self.bot.step(render=render)
