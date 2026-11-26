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
    # Collide with one seamless surface resampled from the collider meshes (grid spacing, m) instead
    # of the meshes themselves. Render meshes are split into pieces whose open boundary edges act as
    # small ledges for a rolling wheel (PhysX cannot tell two flush meshes form one surface), which
    # kicks vehicles up at every seam. None = collide with the render meshes directly.
    drive_surface_grid: float | None = None
    drive_surface_margin: float = 3.0  # m kept around drivable (non-terrain) meshes
    # Gaussian smoothing of the surface heights (m). The render meshes meet with cm-scale offsets
    # that a real tyre and suspension absorb; a rigid robot on sphere wheels would feel every one.
    drive_surface_smoothing: float = 0.45
    terrain_prims: tuple[str, ...] = ()  # collider meshes that are terrain (height only, not drivable area)


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
        drive_surface_grid=0.3,
        terrain_prims=("Terrain/StaticTerrainMeshes/SM_Ground",),
    ),
}


def _surface_heights(stage, env: LocalEnvironment, prim_path: str):
    """Height of the collider meshes at every node of a regular grid (scene frame), and a mask of the
    nodes within ``drive_surface_margin`` of a drivable (non-terrain) mesh. Where a drivable mesh
    exists it defines the height (the highest drivable one, so curbs stay curbs); terrain only fills
    the rest, because the terrain mesh pokes a few cm through the asphalt in places."""
    from pxr import Usd, UsdGeom

    g = float(env.drive_surface_grid)
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    tris = []  # (n, 3, 3) per prim, and whether drivable
    for rel in env.collider_prims:
        root = stage.GetPrimAtPath(f"{prim_path}/{rel}")
        if not root.IsValid():
            continue
        for p in Usd.PrimRange(root):
            if not p.IsA(UsdGeom.Mesh):
                continue
            m = UsdGeom.Mesh(p)
            P = np.asarray(m.GetPointsAttr().Get(), dtype=float)
            M = np.asarray(cache.GetLocalToWorldTransform(p), dtype=float)
            W = np.c_[P, np.ones(len(P))] @ M
            cnt = np.asarray(m.GetFaceVertexCountsAttr().Get())
            idx = np.asarray(m.GetFaceVertexIndicesAttr().Get())
            starts = np.r_[0, np.cumsum(cnt)[:-1]]
            T = [(idx[s0], idx[s0 + k], idx[s0 + k + 1]) for s0, c in zip(starts, cnt) for k in range(1, c - 1)]
            tris.append((W[:, :3][np.asarray(T)], rel not in env.terrain_prims))
    allp = np.concatenate([t.reshape(-1, 3) for t, d in tris if d])
    lo, hi = allp[:, :2].min(axis=0) - env.drive_surface_margin, allp[:, :2].max(axis=0) + env.drive_surface_margin
    nx, ny = (np.ceil((hi - lo) / g).astype(int) + 1)
    Zd = np.full((nx, ny), -np.inf)  # drivable meshes
    Zt = np.full((nx, ny), -np.inf)  # terrain
    drivable = np.zeros((nx, ny), dtype=bool)
    for T, is_drivable in tris:
        for A, B, C in T:
            i0, j0 = np.floor((np.minimum(np.minimum(A, B), C)[:2] - lo) / g).astype(int)
            i1, j1 = np.ceil((np.maximum(np.maximum(A, B), C)[:2] - lo) / g).astype(int)
            i0, j0, i1, j1 = max(i0, 0), max(j0, 0), min(i1, nx - 1), min(j1, ny - 1)
            if i1 < i0 or j1 < j0:
                continue
            X, Y = np.meshgrid(lo[0] + g * np.arange(i0, i1 + 1), lo[1] + g * np.arange(j0, j1 + 1), indexing="ij")
            d = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
            if abs(d) < 1e-12:
                continue
            l1 = ((B[1] - C[1]) * (X - C[0]) + (C[0] - B[0]) * (Y - C[1])) / d
            l2 = ((C[1] - A[1]) * (X - C[0]) + (A[0] - C[0]) * (Y - C[1])) / d
            l3 = 1 - l1 - l2
            inside = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
            if not inside.any():
                continue
            z = l1 * A[2] + l2 * B[2] + l3 * C[2]
            sub = (Zd if is_drivable else Zt)[i0:i1 + 1, j0:j1 + 1]
            sub[inside] = np.maximum(sub[inside], z[inside])
            if is_drivable:
                drivable[i0:i1 + 1, j0:j1 + 1] |= inside
    Z = np.where(drivable, Zd, Zt)
    if env.drive_surface_smoothing > 0:  # normalised Gaussian blur over the valid nodes only
        sig = env.drive_surface_smoothing / g
        r = int(np.ceil(3 * sig))
        ker = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2)
        ok = np.isfinite(Z)
        num, den = np.where(ok, Z, 0.0), ok.astype(float)
        for axis in (0, 1):
            num = np.apply_along_axis(lambda v: np.convolve(v, ker, "same"), axis, num)
            den = np.apply_along_axis(lambda v: np.convolve(v, ker, "same"), axis, den)
        Z = np.where(ok, num / np.maximum(den, 1e-9), -np.inf)
    k = int(round(env.drive_surface_margin / g))  # grow the drivable area by the margin
    grown = drivable.copy()
    for _ in range(k):
        g2 = grown.copy()
        g2[1:] |= grown[:-1]
        g2[:-1] |= grown[1:]
        g2[:, 1:] |= grown[:, :-1]
        g2[:, :-1] |= grown[:, 1:]
        grown = g2
    return lo, g, Z, grown & np.isfinite(Z)


def build_drive_surface(stage, env: LocalEnvironment, prim_path: str, path: str = "/World/DriveSurface") -> None:
    """One seamless static collision mesh resampled from the environment's collider meshes."""
    import hashlib
    import time as _time

    from pxr import Gf, UsdGeom, UsdPhysics, Vt

    key = hashlib.sha1(repr((env.usd_path, os.path.getmtime(env.usd_path), env.origin, env.yaw_deg,
                             env.meters_per_unit, env.collider_prims, env.terrain_prims,
                             env.drive_surface_grid, env.drive_surface_margin, env.drive_surface_smoothing, 3)).encode()).hexdigest()[:16]
    cache = Path(os.path.expanduser("~/.cache/talk2drive")) / f"drive_surface_{key}.npz"
    t0 = _time.time()
    was_cached = cache.exists()
    if was_cached:
        d = np.load(cache)
        lo, g, Z, valid = d["lo"], float(d["g"]), d["Z"], d["valid"]
    else:
        lo, g, Z, valid = _surface_heights(stage, env, prim_path)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, lo=lo, g=g, Z=Z, valid=valid)
    global DRIVE_SURFACE
    DRIVE_SURFACE = (np.asarray(lo, dtype=float), g, Z, valid)
    nx, ny = Z.shape
    vid = np.full((nx, ny), -1, dtype=np.int64)
    vid[valid] = np.arange(int(valid.sum()))
    I, J = np.nonzero(valid)
    pts = np.c_[lo[0] + g * I, lo[1] + g * J, Z[I, J]]
    a, b, c, e = vid[:-1, :-1], vid[1:, :-1], vid[1:, 1:], vid[:-1, 1:]
    quad = (a >= 0) & (b >= 0) & (c >= 0) & (e >= 0)
    tri = np.r_[np.c_[a[quad], b[quad], c[quad]], np.c_[a[quad], c[quad], e[quad]]]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(tri), 3, dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(tri.reshape(-1).astype(np.int32)))
    mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)  # collision only, never rendered
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
    print(f"[env] drive surface: {len(pts)} vertices, {len(tri)} triangles at {g:.2f} m "
          f"({'from cache' if was_cached else 'built'} in {_time.time() - t0:.1f} s)")


DRIVE_SURFACE = None  # (lo, grid, Z, valid) of the last drive surface built


def drive_surface_height(x, y) -> np.ndarray:
    """Road height at scene points (vectorised lookup in the drive-surface grid; NaN off the grid)."""
    if DRIVE_SURFACE is None:
        raise RuntimeError("no drive surface built")
    lo, g, Z, valid = DRIVE_SURFACE
    i = np.rint((np.asarray(x, dtype=float) - lo[0]) / g).astype(int)
    j = np.rint((np.asarray(y, dtype=float) - lo[1]) / g).astype(int)
    ok = (i >= 0) & (j >= 0) & (i < Z.shape[0]) & (j < Z.shape[1])
    out = np.full(np.shape(i), np.nan)
    out[ok] = np.where(valid[i[ok], j[ok]], Z[i[ok], j[ok]], np.nan)
    return out


PAVED_PRIMS = {"mcity": ("Terrain/PedestrianMeshes/Road_Road_MainRoads/SM_Roads_v2",
                         "Terrain/StaticTerrainMeshes/SM_McityRoundabout_Concrete")}
PAVED = None  # bool grid aligned with DRIVE_SURFACE: covered by a road mesh (not sidewalk, curb or ground)


def build_paved_mask(stage, env_name: str, prim_path: str = "/World/Environment") -> np.ndarray:
    """Which drive-surface grid nodes lie on road pavement (cached next to the drive surface)."""
    import hashlib

    from pxr import Usd, UsdGeom

    global PAVED
    lo, g, Z, _ = DRIVE_SURFACE
    env = LOCAL_ENVIRONMENTS[env_name]
    key = hashlib.sha1(repr((env.usd_path, os.path.getmtime(env.usd_path), tuple(lo), g, Z.shape,
                             PAVED_PRIMS[env_name], 1)).encode()).hexdigest()[:16]
    cache = Path(os.path.expanduser("~/.cache/talk2drive")) / f"paved_{key}.npz"
    if cache.exists():
        PAVED = np.load(cache)["paved"]
        return PAVED
    nx, ny = Z.shape
    paved = np.zeros((nx, ny), dtype=bool)
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    for rel in PAVED_PRIMS[env_name]:
        root = stage.GetPrimAtPath(f"{prim_path}/{rel}")
        for p in (Usd.PrimRange(root) if root.IsValid() else []):
            if not p.IsA(UsdGeom.Mesh):
                continue
            m = UsdGeom.Mesh(p)
            P = np.asarray(m.GetPointsAttr().Get(), dtype=float)
            M = np.asarray(xc.GetLocalToWorldTransform(p), dtype=float)
            W = (np.c_[P, np.ones(len(P))] @ M)[:, :2]
            cnt = np.asarray(m.GetFaceVertexCountsAttr().Get())
            idx = np.asarray(m.GetFaceVertexIndicesAttr().Get())
            starts = np.r_[0, np.cumsum(cnt)[:-1]]
            for s0, c in zip(starts, cnt):
                for k in range(1, c - 1):
                    A, B, C = W[idx[s0]], W[idx[s0 + k]], W[idx[s0 + k + 1]]
                    i0, j0 = np.floor((np.minimum(np.minimum(A, B), C) - lo) / g).astype(int)
                    i1, j1 = np.ceil((np.maximum(np.maximum(A, B), C) - lo) / g).astype(int)
                    i0, j0, i1, j1 = max(i0, 0), max(j0, 0), min(i1, nx - 1), min(j1, ny - 1)
                    if i1 < i0 or j1 < j0:
                        continue
                    d = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1])
                    if abs(d) < 1e-12:
                        continue
                    X, Y = np.meshgrid(lo[0] + g * np.arange(i0, i1 + 1), lo[1] + g * np.arange(j0, j1 + 1), indexing="ij")
                    l1 = ((B[1] - C[1]) * (X - C[0]) + (C[0] - B[0]) * (Y - C[1])) / d
                    l2 = ((C[1] - A[1]) * (X - C[0]) + (A[0] - C[0]) * (Y - C[1])) / d
                    paved[i0:i1 + 1, j0:j1 + 1] |= (l1 >= -1e-6) & (l2 >= -1e-6) & (1 - l1 - l2 >= -1e-6)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, paved=paved)
    PAVED = paved
    return paved


def is_paved(x, y) -> np.ndarray:
    """Whether scene points are on road pavement (False off the grid)."""
    lo, g, Z, _ = DRIVE_SURFACE
    i = np.rint((np.asarray(x, dtype=float) - lo[0]) / g).astype(int)
    j = np.rint((np.asarray(y, dtype=float) - lo[1]) / g).astype(int)
    ok = (i >= 0) & (j >= 0) & (i < Z.shape[0]) & (j < Z.shape[1])
    out = np.zeros(np.shape(i), dtype=bool)
    out[ok] = PAVED[i[ok], j[ok]]
    return out


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

    if env.drive_surface_grid:
        build_drive_surface(stage, env, prim_path)
    else:
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
    susp_dofs: tuple[str, ...] = ()  # sprung struts (prismatic); rates, preload and anti-roll bar come from the USD


ROBOTS = {
    "jetbot": RobotSpec(
        usd_path=JETBOT_USD_CANDIDATES[0],
        params=ISAAC_JETBOT_PARAMS,
        from_assets_root=True,
        camera_prim="chassis/rgb_camera/jetbot_camera",
    ),
    # The Magna delivery robot (avesh-shah.com): built by sim/build_robot_from_fbx.py from
    # ~/Documents/magna_robot_2wheeler.fbx with --layout trike --shell magna (command in
    # docs/ISAAC_SIM_SETUP.md). Two steered front wheels (Ackermann servos, 60 deg joint limit) 0.980 m
    # apart, one driven rear wheel on a fixed axle 1.292 m behind them, 0.22 m wheels, roof 1.42 m,
    # cargo bed 0.8 m, flag mast to 2.0 m. Mass properties match the CARLA tadpole NMPC model
    # (~/Documents/carla_nmpc_project/tadpole_params.py): 300 kg, CG 0.65 m behind the front axle and
    # 0.45 m high, Izz ~480 kg m^2, tyre mu 0.9.
    "magna": RobotSpec(
        usd_path=str(REPO_ROOT / "assets/robots/magna_robot.usd"),
        params=dict(wheelbase=1.292, track=0.980, wheel_radius=0.22, max_speed=2.0, max_accel=1.0,
                    max_steer=math.radians(35.0), max_steer_rate=math.radians(90.0)),
        wheel_dofs=("rear_wheel_joint",),
        steer_dofs=("front_left_steer_joint", "front_right_steer_joint"),
        steer_gains=(20000.0, 600.0),  # ~30 ms servo: the tadpole NMPC assumes direct steering
        kind="ackermann",
        spawn_height=0.03,
    ),
    # The same robot on sprung, damped struts with a front anti-roll bar (build_robot_from_fbx.py
    # --suspension; rates from ride frequency 1.8/2.0 Hz, damping ratio 0.3, roll gradient 3 deg/g).
    "magna_susp": RobotSpec(
        usd_path=str(REPO_ROOT / "assets/robots/magna_robot_susp.usd"),
        params=dict(wheelbase=1.292, track=0.980, wheel_radius=0.22, max_speed=2.0, max_accel=1.0,
                    max_steer=math.radians(35.0), max_steer_rate=math.radians(90.0)),
        wheel_dofs=("rear_wheel_joint",),
        steer_dofs=("front_left_steer_joint", "front_right_steer_joint"),
        steer_gains=(20000.0, 600.0),
        susp_dofs=("front_left_susp_joint", "front_right_susp_joint", "rear_susp_joint"),
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
        self.torque_mode = False
        self._load_suspension()
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
        if getattr(self, "torque_mode", False):  # rear motor driven by effort, not a speed target
            kps[self.robot.wheel_dof_indices], kds[self.robot.wheel_dof_indices] = 0.0, 0.0
        if self._susp_idx is not None:
            kps[self._susp_idx], kds[self._susp_idx] = self._susp_k, self._susp_c
        ctrl.set_gains(kps=kps, kds=kds)
        self._hold_suspension()

    # ---------------------------------------------------------------- suspension
    def _load_suspension(self) -> None:
        """Strut springs (preloaded drive targets), dampers and the anti-roll bar rate from the USD."""
        self._susp_idx = None
        if not self.spec.susp_dofs:
            return
        from pxr import UsdPhysics

        stage = self.world.stage
        k, c, x0 = [], [], []
        for name in self.spec.susp_dofs:
            prim = stage.GetPrimAtPath(f"/World/Jetbot/joints/{name}")
            d = UsdPhysics.DriveAPI.Get(prim, "linear")
            k.append(float(d.GetStiffnessAttr().Get()))
            c.append(float(d.GetDampingAttr().Get()))
            x0.append(float(d.GetTargetPositionAttr().Get()))
        self._susp_idx = np.array([self.robot.get_dof_index(n) for n in self.spec.susp_dofs])
        self._susp_k, self._susp_c, self._susp_target = np.array(k), np.array(c), np.array(x0)
        self.arb_rate = float(stage.GetPrimAtPath("/World/Jetbot").GetCustomDataByKey("talk2drive:arb_rate") or 0.0)

    def _hold_suspension(self) -> None:
        """Spring free lengths as the struts' drive targets (resets clear them)."""
        if self._susp_idx is None:
            return
        from isaacsim.core.utils.types import ArticulationAction

        self.robot.apply_action(ArticulationAction(joint_positions=self._susp_target, joint_indices=self._susp_idx))

    def _apply_anti_roll_bar(self) -> None:
        """Front anti-roll bar: equal and opposite strut forces in proportion to the travel difference."""
        if self._susp_idx is None or self.arb_rate <= 0.0:
            return
        from isaacsim.core.utils.types import ArticulationAction

        z = np.asarray(self.robot.get_joint_positions(), dtype=float)[self._susp_idx[:2]]
        f = self.arb_rate * (z[0] - z[1])
        self.robot.apply_action(ArticulationAction(joint_efforts=np.array([-f, f]), joint_indices=self._susp_idx[:2]))

    def suspension_travel(self) -> np.ndarray:
        """Strut travel from design height [m], + = bump (front left, front right, rear)."""
        if self._susp_idx is None:
            return np.zeros(0)
        return np.asarray(self.robot.get_joint_positions(), dtype=float)[self._susp_idx]

    # ---------------------------------------------------------------- force-level actuation
    def enable_torque_mode(self) -> None:
        """Car-like robots: drive the rear wheel by torque and brake all wheels (see apply_forces)."""
        if not self.is_car:
            raise RuntimeError("torque mode is for car-like robots")
        self.torque_mode = True
        self._front_wheel_idx = np.array([self.robot.get_dof_index(n) for n in
                                          ("front_left_wheel_joint", "front_right_wheel_joint")])
        self._restore_steering_gains()

    def apply_forces(self, delta: float, fx_drive: float, fx_brake: float, front_brake_frac: float = 0.65,
                     fx_resist: float = 0.0, dt: float | None = None) -> None:
        """Steering angle [rad] plus longitudinal tyre forces [N]: rear-wheel drive force, total brake
        force (split front/rear) and an extra resistive force (road load) acting at the rear wheel."""
        from isaacsim.core.utils.types import ArticulationAction

        p = self.params
        self._all_dt = dt or self.physics_dt  # time since the previous call (wheel-speed estimate)
        delta = float(np.clip(delta, -p.max_steer, p.max_steer))
        self._cmd = np.array([fx_drive - fx_brake, delta])
        self.robot.apply_action(ArticulationAction(joint_positions=np.array(p.ackermann_angles(delta)),
                                                   joint_indices=self._steer_idx))
        q_dot = self._wheel_speed_all()  # rear, front-left, front-right [rad/s]
        self.last_wheel_speeds = q_dot
        sgn = np.tanh(q_dot / 0.5)  # brakes oppose rotation, smoothly through standstill
        r = p.wheel_radius
        rear = fx_drive * r - (1 - front_brake_frac) * fx_brake * r * sgn[0] - fx_resist * r * sgn[0]
        front = -0.5 * front_brake_frac * fx_brake * r * sgn[1:]
        idx = np.r_[self.robot.wheel_dof_indices, self._front_wheel_idx]
        self.robot.apply_action(ArticulationAction(joint_efforts=np.r_[rear, front], joint_indices=idx))

    def _wheel_speed_all(self) -> np.ndarray:
        q = np.asarray(self.robot.get_joint_positions(), dtype=float)
        idx = np.r_[self.robot.wheel_dof_indices, self._front_wheel_idx]
        prev = getattr(self, "_all_prev", None)
        self._all_prev = q[idx]
        if prev is None or len(prev) != len(idx):
            return np.zeros(len(idx))
        dt = getattr(self, "_all_dt", self.physics_dt)
        return ((q[idx] - prev + np.pi) % (2 * np.pi) - np.pi) / dt

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

    def step(self, render: bool | None = None) -> None:
        self._apply_anti_roll_bar()
        self.world.step(render=self.render if render is None else render)
        self._update_encoders()

    def reset_to(self, pose: tuple[float, float, float], settle_steps: int = 30) -> None:
        """Reset the world and place the robot at (x, y, yaw), at rest."""
        x, y, yaw = pose
        self.world.reset()
        self._restore_steering_gains()
        self._all_prev = None
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
