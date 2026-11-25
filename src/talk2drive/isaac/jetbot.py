"""Isaac Sim adapter: a Jetbot in a scene, exposed through the talk2drive ``DiffDriveRobot`` interface.

Import this module only *after* ``SimulationApp`` has been created (Isaac Sim requirement).
Written against the Isaac Sim 5.x Python API (``isaacsim.*`` namespaces).
"""

from __future__ import annotations

import math

import numpy as np

from ..robot import DiffDriveParams, yaw_from_quat_wxyz

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
    ):
        from isaacsim.core.api import World
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.robot.wheeled_robots.robots import WheeledRobot

        self.params = params or DiffDriveParams()
        self.physics_dt = physics_dt
        self.render = render
        self.world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=physics_dt)

        root = assets_root()
        if env not in ENVIRONMENTS:
            raise ValueError(f"Unknown env {env!r}; choose from {sorted(ENVIRONMENTS)}")
        if ENVIRONMENTS[env] is None:
            self.world.scene.add_default_ground_plane()
        else:
            add_reference_to_stage(usd_path=root + ENVIRONMENTS[env], prim_path="/World/Environment")

        x, y, yaw = start_pose
        self.robot = self.world.scene.add(
            WheeledRobot(
                prim_path="/World/Jetbot",
                name="jetbot",
                wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                create_robot=True,
                usd_path=_first_existing(root, JETBOT_USD_CANDIDATES),
                position=np.array([x, y, 0.05]),
                orientation=quat_wxyz_from_yaw(yaw),
            )
        )
        self.world.reset()
        self._cmd = np.zeros(2)

    # ---------------------------------------------------------------- DiffDriveRobot
    def get_state(self) -> np.ndarray:
        pos, quat = self.robot.get_world_pose()
        try:
            wheel_vel = np.asarray(self.robot.get_wheel_velocities(), dtype=float)
        except AttributeError:
            wheel_vel = np.asarray(self.robot.get_joint_velocities(), dtype=float)[self.robot.wheel_dof_indices]
        return np.array([pos[0], pos[1], yaw_from_quat_wxyz(quat), wheel_vel[0], wheel_vel[1]], dtype=float)

    def set_wheel_speeds(self, omega_l: float, omega_r: float) -> None:
        from isaacsim.core.utils.types import ArticulationAction

        self._cmd = np.array([omega_l, omega_r], dtype=float)
        self.robot.apply_wheel_actions(ArticulationAction(joint_velocities=self._cmd))

    def step(self) -> None:
        self.world.step(render=self.render)

    def reset_to(self, pose: tuple[float, float, float], settle_steps: int = 30) -> None:
        """Reset the world and place the robot at (x, y, yaw), at rest."""
        x, y, yaw = pose
        self.world.reset()
        self.robot.set_world_pose(position=np.array([x, y, 0.05]), orientation=quat_wxyz_from_yaw(yaw))
        self.set_wheel_speeds(0.0, 0.0)
        for _ in range(settle_steps):
            self.step()

    # ---------------------------------------------------------------- extras
    def make_camera(self, resolution=(640, 480)):
        """The Jetbot's built-in front RGB camera (call before stepping to get frames)."""
        from isaacsim.sensors.camera import Camera

        cam = Camera(prim_path=JETBOT_CAMERA_PRIM, resolution=resolution)
        cam.initialize()
        return cam
