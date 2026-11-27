"""Scene props and the Magna robot's front camera (needs a running Isaac Sim app)."""

from __future__ import annotations

import numpy as np

# Objects placed in scenes: name -> (shape, size, colour). Sizes in metres (radius, height / edge).
CATALOG = {
    "orange traffic cone": ("cone", (0.25, 0.75), (1.0, 0.35, 0.0)),
    "blue barrel": ("cylinder", (0.30, 0.95), (0.05, 0.2, 0.85)),
    "red box": ("cube", (0.70,), (0.85, 0.05, 0.05)),
    "green box": ("cube", (0.70,), (0.1, 0.65, 0.15)),
    "yellow ball": ("sphere", (0.35,), (1.0, 0.85, 0.0)),
    "purple barrel": ("cylinder", (0.30, 0.95), (0.5, 0.1, 0.65)),
    "brown cardboard box": ("cube", (0.90,), (0.55, 0.38, 0.2)),
}
CAM_POS, CAM_PITCH_DEG, CAM_RES = (1.62, 0.0, 1.45), 8.0, (1280, 720)
FRONT_BUMPER_X = 1.60  # Magna front face, m ahead of the robot origin (rear contact)
HALF_WIDTH = 0.58  # Magna half width [m]


def object_height(name: str) -> float:
    shape, size, _ = CATALOG[name]
    return size[1] if shape in ("cone", "cylinder") else (2 * size[0] if shape == "sphere" else size[0])


def footprint(name: str) -> float:
    """Radius of the object's plan-view footprint (half the edge for boxes)."""
    shape, size, _ = CATALOG[name]
    return size[0] if shape in ("cone", "cylinder", "sphere") else size[0] / 2


def spawn_object(stage, path: str, name: str, xy, ground_z: float, collide: bool = True) -> None:
    """A coloured primitive standing on the ground at ``xy`` (with a static collider unless ``collide`` is False)."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    shape, size, rgb = CATALOG[name]
    if shape == "cone":
        g = UsdGeom.Cone.Define(stage, path)
        g.CreateRadiusAttr(size[0])
        g.CreateHeightAttr(size[1])
        z = size[1] / 2
    elif shape == "cylinder":
        g = UsdGeom.Cylinder.Define(stage, path)
        g.CreateRadiusAttr(size[0])
        g.CreateHeightAttr(size[1])
        z = size[1] / 2
    elif shape == "cube":
        g = UsdGeom.Cube.Define(stage, path)
        g.CreateSizeAttr(size[0])
        z = size[0] / 2
    else:
        g = UsdGeom.Sphere.Define(stage, path)
        g.CreateRadiusAttr(size[0])
        z = size[0]
    if shape in ("cone", "cylinder"):
        g.CreateAxisAttr("Z")
    g.AddTranslateOp().Set(Gf.Vec3d(float(xy[0]), float(xy[1]), ground_z + z))
    mat = UsdShade.Material.Define(stage, path + "_look")
    sh = UsdShade.Shader.Define(stage, path + "_look/shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat)
    if collide:
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())


def make_front_camera(bot, with_depth: bool = True, resolution=CAM_RES):
    """RGB(-D) camera behind the windscreen (moves with the body), 60 deg horizontal field of view."""
    from isaacsim.core.utils.rotations import euler_angles_to_quat
    from isaacsim.sensors.camera import Camera
    from pxr import UsdGeom

    path = "/World/Jetbot/chassis/front_cam"
    cam = Camera(prim_path=path, resolution=resolution)
    cam.initialize()
    cam.set_local_pose(translation=np.array(CAM_POS),
                       orientation=euler_angles_to_quat(np.array([0.0, CAM_PITCH_DEG, 0.0]), degrees=True), camera_axes="world")
    c = UsdGeom.Camera(bot.world.stage.GetPrimAtPath(path))
    c.GetFocalLengthAttr().Set(18.15)  # with the 20.955 mm aperture: 60 deg horizontal FOV
    c.GetClippingRangeAttr().Set((0.1, 300.0))
    if with_depth:
        cam.add_distance_to_image_plane_to_frame()
    return cam
