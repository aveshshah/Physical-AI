"""Isaac Sim adapter: a Jetbot in a scene, exposed through the talk2drive ``DiffDriveRobot`` interface.

Import this module only *after* ``SimulationApp`` has been created (Isaac Sim requirement).
Written against the Isaac Sim 5.x Python API (``isaacsim.*`` namespaces); runs unchanged on 6.2, where
those extensions live under ``extsDeprecated/``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from pathlib import Path

import numpy as np

from ..robot import AckermannParams, DiffDriveParams, yaw_from_quat_wxyz

# Asset locations changed between Isaac Sim releases; the first that exists is used.
JETBOT_USD_CANDIDATES = [
    "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd",
    "/Isaac/Robots/Jetbot/jetbot.usd",
]
ENVIRONMENTS = {
    "grid": None,  # flat ground plane, lightest on the GPU
    "warehouse": "/Isaac/Environments/Simple_Warehouse/warehouse.usd",
    "warehouse_full": "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd",
    "room": "/Isaac/Environments/Simple_Room/simple_room.usd",
}
JETBOT_CAMERA_PRIM = "/World/Jetbot/chassis/rgb_camera/jetbot_camera"


@dataclass
class LocalEnvironment:
    """A scene from a local USD file, placed so ``origin`` (in the file's metres) is the world origin."""

    usd_path: str
    meters_per_unit: float = 1.0
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)  # metres, in the file's frame
    yaw_deg: float = 0.0  # rotate the scene about the origin
    collider_prims: list[str] = field(default_factory=list)  # meshes under these get static colliders
    add_sky: bool = False  # the file has no sun / sky light of its own


# Mcity Test Facility (University of Michigan) digital twin, Omniverse version, MIT licence:
#   git clone --depth 1 --filter=blob:none --sparse https://github.com/mcity/mcity-digital-twin.git
#   cd mcity-digital-twin && git sparse-checkout set Omniverse
# The stage is in centimetres, Z-up and georeferenced (roads at ~272 m elevation) and has no colliders.
# The origin is the middle of the 13 m-wide north-south street in the downtown block, rotated so the
# street runs along +x: the slalom (6 m) and figure-8 (2 x 1 m) both fit on the road surface.
MCITY_DIR = os.environ.get(
    "MCITY_DIR", os.path.expanduser("~/assets/mcity-digital-twin/Omniverse/Collected_McityMap_NSR_v4_1_6")
)
LOCAL_ENVIRONMENTS = {
    "mcity": LocalEnvironment(
        usd_path=os.path.join(MCITY_DIR, "McityMap_Main.usdc"),
        meters_per_unit=0.01,
        origin=(52.1, 4.0, 271.813),
        yaw_deg=-90.0,
        collider_prims=[
            "Terrain/PedestrianMeshes/Road_Road_MainRoads/SM_Roads_v2",
            "Terrain/StaticTerrainMeshes/SM_Ground",
            "Terrain/StaticTerrainMeshes/SM_Sidewalks",
            "Terrain/StaticTerrainMeshes/SM_Curbs",
            "Terrain/StaticTerrainMeshes/SM_McityRoundabout_Concrete",
        ],
        add_sky=True,
    ),
}


def load_local_environment(stage, env: LocalEnvironment, prim_path: str = "/World/Environment") -> None:
    """Reference ``env`` into the stage, place it, add static triangle-mesh colliders and lights."""
    from isaacsim.core.utils.stage import add_reference_to_stage
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

    if not os.path.exists(env.usd_path):
        raise FileNotFoundError(f"{env.usd_path} not found; see the download notes above LOCAL_ENVIRONMENTS.")
    prim = add_reference_to_stage(usd_path=env.usd_path, prim_path=prim_path)
    # Points are first scaled to metres, then shifted so ``origin`` is at 0, then rotated about it.
    # Kit's units resolver adds an ``xformOp:scale:unitsResolve`` op to references whose
    # metersPerUnit differs from the stage's; authoring that op ourselves keeps the scale from being
    # applied twice whether or not the resolver runs.
    m = Gf.Matrix4d().SetTranslate(-Gf.Vec3d(*env.origin)) * Gf.Matrix4d().SetRotate(
        Gf.Rotation(Gf.Vec3d(0, 0, 1), env.yaw_deg))
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTransformOp().Set(m)
    xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble, "unitsResolve").Set(Gf.Vec3d(env.meters_per_unit))

    n = 0
    for rel in env.collider_prims:
        root = stage.GetPrimAtPath(f"{prim_path}/{rel}")
        if not root.IsValid():
            print(f"[env] collider prim {rel} not found; skipped")
            continue
        for p in Usd.PrimRange(root):
            if p.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(p)
                UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("none")  # exact triangles (static)
                n += 1
    print(f"[env] {n} collider meshes added")

    if env.add_sky:
        sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Sun"))
        sun.CreateIntensityAttr(3000.0)
        sun.CreateAngleAttr(0.5)
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-35.0, 25.0, 40.0))
        UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Sky")).CreateIntensityAttr(800.0)

# Effective kinematics of the Isaac Jetbot, measured with sim/calibrate_jetbot.py on Isaac Sim 6.2
# (runs/calib/jetbot_isaac.json). The radius matches the wheel collision geometry (67.2 mm diameter);
# the wheel base is the effective value from spin-in-place runs (wider than the 113.9 mm between
# wheel centres because the 19.5 mm-wide tyres scrub). The wheels follow velocity targets within
# one physics step, so the controller's 50 ms lag model is kept (see calibrate_jetbot.py).
ISAAC_JETBOT_PARAMS = dict(wheel_radius=0.0336, wheel_base=0.1212)

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class RobotSpec:
    """A differential-drive robot asset: where it lives, its wheel joints and its model parameters."""

    usd_path: str  # absolute path, or relative to the Isaac assets root when ``from_assets_root``
    params: dict
    wheel_dofs: tuple[str, str] = ("left_wheel_joint", "right_wheel_joint")
    spawn_height: float = 0.05  # drop height of the robot origin above the ground [m]
    from_assets_root: bool = False
    camera_prim: str | None = None  # relative to the robot prim
    kind: str = "diff"  # "diff": (left, right) wheel speeds | "ackermann": rear-wheel speed + front steering
    steer_dofs: tuple[str, str] | None = None  # ackermann: (left, right) steering joints
    steer_gains: tuple[float, float] = (3000.0, 300.0)  # steering servo stiffness [N m/rad], damping [N m s/rad]


ROBOTS = {
    "jetbot": RobotSpec(
        usd_path=JETBOT_USD_CANDIDATES[0],
        params=ISAAC_JETBOT_PARAMS,
        from_assets_root=True,
        camera_prim="chassis/rgb_camera/jetbot_camera",
    ),
    # The Magna delivery robot (avesh-shah.com): built by sim/build_robot_from_fbx.py from
    # ~/Documents/magna_robot_2wheeler.fbx with --layout trike --shell magna. Two steered front wheels
    # (Ackermann, +-35 deg servo) 0.980 m apart, one driven rear wheel on a fixed axle 1.292 m behind
    # them, 0.22 m wheels, roof 1.42 m, cargo bed 0.8 m, flag mast to 2.0 m; ~174 kg, 150 N m rear motor.
    "magna": RobotSpec(
        usd_path=str(REPO_ROOT / "assets/robots/magna_robot.usd"),
        params=dict(wheelbase=1.292, track=0.980, wheel_radius=0.22, max_speed=2.0, max_accel=1.0,
                    max_steer=math.radians(35.0), max_steer_rate=math.radians(90.0)),
        wheel_dofs=("rear_wheel_joint",),
        steer_dofs=("front_left_steer_joint", "front_right_steer_joint"),
        steer_gains=(1500.0, 150.0),
        kind="ackermann",
        spawn_height=0.03,
    ),
    # Built from ~/Documents/carla_3wheeler.fbx by sim/build_robot_from_fbx.py (--layout trike):
    # two steered front wheels (Ackermann, +-35 deg servo) 2.0 m apart, one driven rear wheel on a
    # fixed axle 3.0 m behind them, 0.40 m wheels, ~236 kg, 250 N m rear motor. Origin: rear axle.
    "carla_trike": RobotSpec(
        usd_path=str(REPO_ROOT / "assets/robots/carla_trike.usd"),
        params=dict(wheelbase=3.0, track=2.0, wheel_radius=0.4, max_speed=3.0, max_accel=1.5,
                    max_steer=math.radians(35.0), max_steer_rate=math.radians(60.0)),
        wheel_dofs=("rear_wheel_joint",),
        steer_dofs=("front_left_steer_joint", "front_right_steer_joint"),
        kind="ackermann",
        spawn_height=0.05,
    ),
}


def assets_root() -> str:
    try:
        from isaacsim.storage.native import get_assets_root_path
    except ImportError:  # older releases
        from isaacsim.core.utils.nucleus import get_assets_root_path
    root = get_assets_root_path()
    if not root:
        raise RuntimeError(
            "Could not find the Isaac Sim assets root. Check your internet connection (assets are "
            "streamed from NVIDIA's S3 by default) or configure a local asset pack."
        )
    return root


def _first_existing(root: str, candidates: list[str]) -> str:
    import omni.client

    for rel in candidates:
        result, _ = omni.client.stat(root + rel)
        if result == omni.client.Result.OK:
            return root + rel
    return root + candidates[0]


def quat_wxyz_from_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


class IsaacJetbot:
    """Builds the scene and implements ``get_state`` / ``set_wheel_speeds`` / ``step``."""

    def __init__(
        self,
        env: str = "grid",
        start_pose: tuple[float, float, float] = (0.0, 0.0, 0.0),
        physics_dt: float = 1.0 / 60.0,
        render: bool = True,
        params: DiffDriveParams | None = None,
        robot: str = "jetbot",
    ):
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.robot.wheeled_robots.robots import WheeledRobot

        if robot not in ROBOTS:
            raise ValueError(f"Unknown robot {robot!r}; choose from {sorted(ROBOTS)}")
        self.spec = spec = ROBOTS[robot]
        self.is_car = spec.kind == "ackermann"
        self.params = params or (AckermannParams if self.is_car else DiffDriveParams)(**spec.params)
        self.physics_dt = physics_dt
        self.render = render
        self.world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=physics_dt)

        root = assets_root()
        if env in LOCAL_ENVIRONMENTS:
            load_local_environment(self.world.stage, LOCAL_ENVIRONMENTS[env])
        elif env not in ENVIRONMENTS:
            raise ValueError(f"Unknown env {env!r}; choose from {sorted([*ENVIRONMENTS, *LOCAL_ENVIRONMENTS])}")
        elif ENVIRONMENTS[env] is None:
            self.world.scene.add_default_ground_plane()
        else:
            add_reference_to_stage(usd_path=root + ENVIRONMENTS[env], prim_path="/World/Environment")

        x, y, yaw = start_pose
        if spec.from_assets_root:
            usd = _first_existing(root, JETBOT_USD_CANDIDATES) if robot == "jetbot" else root + spec.usd_path
        else:
            usd = spec.usd_path
            if not os.path.exists(usd):
                raise FileNotFoundError(f"{usd} not found; build it with sim/build_robot_from_fbx.py")
        self.robot = self.world.scene.add(
            WheeledRobot(
                prim_path="/World/Jetbot",  # prim name kept for every robot so existing paths still work
                name="robot",
                wheel_dof_names=list(spec.wheel_dofs),
                create_robot=True,
                usd_path=usd,
                position=np.array([x, y, spec.spawn_height]),
                orientation=quat_wxyz_from_yaw(yaw),
            )
        )
        self.world.reset()
        self._steer_idx = (np.array([self.robot.get_dof_index(n) for n in spec.steer_dofs])
                           if spec.steer_dofs else None)
        self._restore_steering_gains()
        self._cmd = np.zeros(2)
        self._reset_encoders()

    # ---------------------------------------------------------------- robot interface
    def get_state(self) -> np.ndarray:
        """Diff drive: [x, y, theta, wl, wr]. Car-like: [x, y, theta, v, delta] (rear axle, single-track angle)."""
        pos, quat = self.robot.get_world_pose()
        th = yaw_from_quat_wxyz(quat)
        if self.is_car:
            q = np.asarray(self.robot.get_joint_positions(), dtype=float)[self._steer_idx]
            v = float(self._wheel_speed[0]) * self.params.wheel_radius
            return np.array([pos[0], pos[1], th, v, self.params.delta_from_wheels(q[0], q[1])], dtype=float)
        wl, wr = self._wheel_speed
        return np.array([pos[0], pos[1], th, wl, wr], dtype=float)

    def set_command(self, v: float, delta: float) -> None:
        """Car-like robots: rear-wheel ground speed [m/s] and single-track steering angle [rad]."""
        from isaacsim.core.utils.types import ArticulationAction

        p = self.params
        v = float(np.clip(v, -p.max_speed, p.max_speed))
        delta = float(np.clip(delta, -p.max_steer, p.max_steer))
        self._cmd = np.array([v, delta])
        self.robot.apply_wheel_actions(ArticulationAction(joint_velocities=np.array([v / p.wheel_radius])))
        self.robot.apply_action(ArticulationAction(joint_positions=np.array(p.ackermann_angles(delta)),
                                                   joint_indices=self._steer_idx))

    def _restore_steering_gains(self) -> None:
        """WheeledRobot.post_reset() switches every joint to velocity control (stiffness 0), which
        disables position-servo steering; put the steering joints' servo gains back after each reset."""
        if self._steer_idx is None:
            return
        ctrl = self.robot.get_articulation_controller()
        kps, kds = (np.array(g, dtype=float) for g in ctrl.get_gains())
        kps[self._steer_idx], kds[self._steer_idx] = self.spec.steer_gains
        ctrl.set_gains(kps=kps, kds=kds)

    def ground_height(self, x: float, y: float, default: float = 0.0) -> float:
        """Height of the scene (not the robot) below (x, y), by a physics raycast."""
        from omni.physx import get_physx_scene_query_interface

        hits = []

        def report(hit):
            if not str(hit.rigid_body).startswith("/World/Jetbot"):
                hits.append(hit.position[2])
            return True

        get_physx_scene_query_interface().raycast_all((x, y, 200.0), (0.0, 0.0, -1.0), 1000.0, report)
        return max(hits) if hits else default

    # Wheel speeds come from differencing joint positions, like wheel encoders. On Isaac Sim 6.2 the
    # reported joint velocities are biased (about +0.35 rad/s while driving straight, while the joint
    # positions advance exactly at the commanded rate and the chassis moves accordingly).
    def _wheel_positions(self) -> np.ndarray:
        return np.asarray(self.robot.get_joint_positions(), dtype=float)[self.robot.wheel_dof_indices]

    def _reset_encoders(self) -> None:
        self._last_wheel_pos = self._wheel_positions()
        self._wheel_speed = np.zeros(len(self._last_wheel_pos))

    def _update_encoders(self) -> None:
        q = self._wheel_positions()
        dq = (q - self._last_wheel_pos + np.pi) % (2 * np.pi) - np.pi  # robust to angle wrapping
        self._wheel_speed = dq / self.physics_dt
        self._last_wheel_pos = q

    def set_wheel_speeds(self, omega_l: float, omega_r: float) -> None:
        from isaacsim.core.utils.types import ArticulationAction

        self._cmd = np.array([omega_l, omega_r], dtype=float)
        self.robot.apply_wheel_actions(ArticulationAction(joint_velocities=self._cmd))

    def step(self) -> None:
        self.world.step(render=self.render)
        self._update_encoders()

    def reset_to(self, pose: tuple[float, float, float], settle_steps: int = 30) -> None:
        """Reset the world and place the robot at (x, y, yaw), at rest."""
        x, y, yaw = pose
        self.world.reset()
        self._restore_steering_gains()
        z = self.ground_height(x, y) + self.spec.spawn_height
        self.robot.set_world_pose(position=np.array([x, y, z]), orientation=quat_wxyz_from_yaw(yaw))
        if self.is_car:
            self.set_command(0.0, 0.0)
        else:
            self.set_wheel_speeds(0.0, 0.0)
        self._reset_encoders()
        for _ in range(settle_steps):
            self.step()

    # ---------------------------------------------------------------- extras
    def make_camera(self, resolution=(640, 480)):
        """The robot's built-in front RGB camera (call before stepping to get frames)."""
        from isaacsim.sensors.camera import Camera

        if self.spec.camera_prim is None:
            raise RuntimeError("this robot has no camera")
        cam = Camera(prim_path=f"/World/Jetbot/{self.spec.camera_prim}", resolution=resolution)
        cam.initialize()
        return cam
