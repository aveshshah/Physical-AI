"""Turn a CARLA-style three-wheeled vehicle FBX into a physics-enabled robot USD for Isaac Sim.

    python sim/build_robot_from_fbx.py ~/Documents/carla_3wheeler.fbx assets/robots/carla_trike.usd
    python sim/build_robot_from_fbx.py ~/Documents/magna_robot_2wheeler.fbx assets/robots/magna_robot.usd --layout diff

The FBX (exported from Blender) holds one skinned mesh bound to a skeleton with a ``Root`` bone and
``Wheel_*`` bones (two front wheels and one rear wheel with geometry); it has no materials, masses or
joints. This script converts it with Isaac's asset converter, splits the skinned mesh by bone, and
builds an articulation in a Z-up, metre, +x-forward frame with:

* ``--layout trike`` (default): front wheels on steering knuckles (revolute about z, position-servo
  "steering motor" with +-max-steer limits, Ackermann angles set by the controller) that roll freely;
  the rear wheel is on a fixed axle and is the only driven wheel (velocity drive with a torque limit).
  Origin: rear-wheel contact point (the bicycle-model reference point).
* ``--layout diff``: the two front wheels are driven differentially and the rear wheel is a passive
  caster (swivel joint with trail). Origin: drive-axle midpoint on the ground.

``--suspension`` (trike) puts each wheel on a sprung, damped vertical strut (prismatic joint with a
linear drive preloaded so the chassis sits at its design height) and records a front anti-roll bar
rate that the Isaac adapter applies between the two front struts. Rates come from ride frequency,
damping ratio and target roll gradient, the way suspensions of real vehicles are specified.

Colliders are a box for the body and spheres for the wheels (smooth rolling, one contact point),
with a rubber material on the driven/steered tyres. Masses, motor torque and servo gains are not in
the FBX: the defaults are assumptions and can be changed on the command line.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("fbx")
ap.add_argument("out_usd")
ap.add_argument("--layout", choices=["trike", "diff"], default="trike")
ap.add_argument("--chassis-mass", type=float, default=None, help="kg (trike 200, diff 40)")
ap.add_argument("--wheel-mass", type=float, default=None, help="kg per wheel (trike 10, diff 3)")
ap.add_argument("--max-torque", type=float, default=None, help="N m per driven wheel (trike 250, diff 40)")
ap.add_argument("--drive-damping", type=float, default=None, help="velocity-drive gain, N m per rad/s (trike 400, diff 80)")
ap.add_argument("--max-steer", type=float, default=35.0, help="deg, trike steering limit")
ap.add_argument("--steer-stiffness", type=float, default=3000.0, help="N m per rad, trike steering servo")
ap.add_argument("--steer-damping", type=float, default=300.0, help="N m s per rad, trike steering servo")
ap.add_argument("--steer-torque", type=float, default=500.0, help="N m, trike steering servo limit")
ap.add_argument("--caster-mass", type=float, default=1.5, help="kg (diff layout caster wheel; the fork gets 0.5 kg)")
ap.add_argument("--caster-trail", type=float, default=0.06, help="m between swivel axis and caster wheel centre")
ap.add_argument("--tyre-friction", type=float, default=0.9, help="tyre-road friction (combine mode max, so the road's own value does not lower it)")
ap.add_argument("--com", default=None, help="x,y,z of the chassis mass centre in the robot frame [m] (default: body centre, low)")
ap.add_argument("--chassis-inertia", default=None, help="Ixx,Iyy,Izz of the chassis about its mass centre [kg m^2]")
ap.add_argument("--suspension", action="store_true", help="trike: sprung, damped struts and a front anti-roll bar")
ap.add_argument("--ride-freq", default="1.8,2.0", help="front,rear ride frequency [Hz] (cars 1-1.5 Hz, scaled up for size)")
ap.add_argument("--damping-ratio", type=float, default=0.3, help="suspension damping ratio (cars 0.2-0.4)")
ap.add_argument("--roll-gradient", type=float, default=3.0, help="target body roll per g of lateral acceleration [deg/g]")
ap.add_argument("--travel", type=float, default=0.05, help="bump and rebound travel from design height [m]")
ap.add_argument("--strut-mass", type=float, default=1.5, help="kg per strut (unsprung, with knuckle and wheel)")
ap.add_argument("--shell", choices=["fbx", "magna"], default="fbx",
                help="body visual: the FBX's own mesh, or a modelled Magna delivery-robot shell")
ap.add_argument("--overall-height", type=float, default=2.0, help="m, top of the flag mast (magna shell)")
ap.add_argument("--bed-height", type=float, default=0.8, help="m, cargo floor / bottom of the lockers (magna shell)")
ap.add_argument("--overhang", type=float, default=0.3, help="m, body beyond each axle (magna shell)")
args = ap.parse_args()
LAYOUT_DEFAULTS = {"trike": dict(chassis_mass=200.0, wheel_mass=10.0, max_torque=250.0, drive_damping=400.0),
                   "diff": dict(chassis_mass=40.0, wheel_mass=3.0, max_torque=40.0, drive_damping=80.0)}
for _k, _v in LAYOUT_DEFAULTS[args.layout].items():
    if getattr(args, _k) is None:
        setattr(args, _k, _v)

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import math  # noqa: E402

import numpy as np  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("omni.kit.asset_converter")
for _ in range(5):
    app.update()

import omni.kit.asset_converter as ac  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, UsdSkel, Vt  # noqa: E402


def convert(fbx: Path, dst: Path) -> None:
    async def run():
        ctx = ac.AssetConverterContext()
        ctx.ignore_animations = ctx.ignore_camera = ctx.ignore_light = True
        ctx.use_meter_as_world_unit = True
        ctx.baking_scales = True
        task = ac.get_instance().create_converter_task(str(fbx), str(dst), None, ctx)
        if not await task.wait_until_finished():
            raise RuntimeError(f"FBX conversion failed: {task.get_error_message()}")

    fut = asyncio.ensure_future(run())
    while not fut.done():
        app.update()
    fut.result()


def split_parts(raw_usd: Path) -> dict[str, dict]:
    """Faces of the skinned mesh grouped by their bone, in the mesh's own frame (Z-up, front = +y)."""
    st = Usd.Stage.Open(str(raw_usd))
    mesh_prim = next(p for p in st.Traverse() if p.IsA(UsdGeom.Mesh))
    mesh = UsdGeom.Mesh(mesh_prim)
    P = np.array(mesh.GetPointsAttr().Get(), dtype=float)
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
    idx = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    bind = UsdSkel.BindingAPI(mesh_prim)
    skel = UsdSkel.Skeleton(bind.GetSkeletonRel().GetTargets() and st.GetPrimAtPath(bind.GetSkeletonRel().GetTargets()[0])
                            or next(p for p in st.Traverse() if p.IsA(UsdSkel.Skeleton)))
    joints = [str(j).rsplit("/", 1)[-1] for j in (bind.GetJointsAttr().Get() or skel.GetJointsAttr().Get())]
    n_inf = bind.GetJointIndicesPrimvar().GetElementSize()
    ji = np.array(bind.GetJointIndicesAttr().Get()).reshape(len(P), n_inf)
    jw = np.array(bind.GetJointWeightsAttr().Get()).reshape(len(P), n_inf)
    bone_of_point = ji[np.arange(len(P)), jw.argmax(axis=1)]
    parts: dict[str, dict] = {}
    start = 0
    for c in counts:
        face = idx[start:start + c]
        start += c
        bone = joints[bone_of_point[face[0]]]
        parts.setdefault(bone, {"faces": []})["faces"].append(face)
    for bone, part in parts.items():
        used = np.unique(np.concatenate(part["faces"]))
        remap = {old: new for new, old in enumerate(used)}
        part["points"] = P[used]
        part["counts"] = [len(f) for f in part["faces"]]
        part["indices"] = [remap[i] for f in part["faces"] for i in f]
        part["min"], part["max"] = P[used].min(axis=0), P[used].max(axis=0)
        print(f"[build] bone {bone:20s} {len(used):4d} points  bbox {part['min'].round(3)} .. {part['max'].round(3)}")
    return parts


def _ring_mesh(stage, path, rings, mat, normals=True):
    """Closed mesh through ``rings`` (k rings of m points each, bottom to top), capped at both ends."""
    R = np.asarray(rings, dtype=float)
    k, m, _ = R.shape
    pts = list(R.reshape(-1, 3)) + [R[0].mean(axis=0), R[-1].mean(axis=0)]
    bot, top = k * m, k * m + 1
    counts, idx = [], []
    for i in range(k - 1):
        for j in range(m):
            a, b = i * m + j, i * m + (j + 1) % m
            counts.append(4)
            idx += [a, b, b + m, a + m]
    for j in range(m):
        counts += [3, 3]
        idx += [bot, (j + 1) % m, j, top, (k - 1) * m + j, (k - 1) * m + (j + 1) % m]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in pts]))
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateSubdivisionSchemeAttr("none")
    if normals:  # smooth vertex normals from area-weighted face normals
        P = np.array(pts)
        N = np.zeros_like(P)
        o = 0
        for c in counts:
            f = idx[o:o + c]
            o += c
            n = np.cross(P[f[1]] - P[f[0]], P[f[-1]] - P[f[0]])
            N[f] += n
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
        mesh.CreateNormalsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, n)) for n in N]))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
    return mesh


def _box_rings(c, size):
    (cx, cy, cz), (sx, sy, sz) = c, size
    sq = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    return [[(cx + u * sx / 2, cy + v * sy / 2, cz + w * sz / 2) for u, v in sq] for w in (-1, 1)]


def superellipse_xy(a, b, n, m=64):
    t = np.linspace(0, 2 * np.pi, m, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    return a * np.sign(c) * np.abs(c) ** (2 / n), b * np.sign(s) * np.abs(s) ** (2 / n)


def build_magna_shell(stage, x_rear, x_front, roof, bed, mast_top, look) -> dict:
    """Rounded delivery-pod shell like the Magna robot: white body flaring over the wheels, dark
    glass cargo lockers on both sides above the bed, camera window, headlight and amber markers on
    the front, tail lights, roof lidar and an orange safety flag on a mast. Returns its extents."""
    cx, a = (x_front + x_rear) / 2, (x_front - x_rear) / 2
    z0, rt, rb = 0.28, 0.20, 0.07  # shell bottom, roof / bottom edge rounding radii
    b_low, b_high, n = 0.58, 0.50, 6.0  # half-widths: skirt over the wheels, upper body; squareness
    rings = []
    for z in np.r_[np.linspace(z0, z0 + rb, 5), np.linspace(z0 + rb + 0.02, roof - rt, 14), np.linspace(roof - rt, roof, 10)[1:]]:
        inset = 0.0
        if z > roof - rt:
            inset = rt - math.sqrt(max(rt**2 - (z - (roof - rt)) ** 2, 0.0))
        elif z < z0 + rb:
            inset = rb - math.sqrt(max(rb**2 - (z0 + rb - z) ** 2, 0.0))
        blend = float(np.clip((z - 0.55) / 0.35, 0.0, 1.0))  # flare below ~0.55 m, upper width above 0.9 m
        b = b_low + (b_high - b_low) * (3 * blend**2 - 2 * blend**3)
        x, y = superellipse_xy(a - inset, b - inset, n)
        rings.append(np.c_[x + cx, y, np.full_like(x, z)])
    _ring_mesh(stage, "/robot/chassis/visual/shell", rings, look["white"])

    def side_x(y):  # front face x at lateral offset y on the flat part of the body
        return cx + a * max(1 - abs(y / b_high) ** n, 0.0) ** (1 / n)

    def panel(name, centre, size, mat):
        _ring_mesh(stage, f"/robot/chassis/visual/{name}", _box_rings(centre, size), mat, normals=False)

    # cargo lockers: two doors per side between the bed and just under the roof rounding
    zl0, zl1 = bed + 0.03, roof - rt + 0.02
    door_len = (x_front - x_rear) * 0.30
    for side, sgn in (("left", 1), ("right", -1)):
        for k, xc in enumerate((cx - 0.5 * door_len - 0.02, cx + 0.5 * door_len + 0.02)):
            panel(f"locker_{side}_{k}", (xc - 0.08, sgn * (b_high + 0.004), (zl0 + zl1) / 2), (door_len, 0.012, zl1 - zl0), look["glass"])
    # front face
    xf = side_x(0.0) + 0.004
    panel("camera_window", (xf, 0.0, roof - rt - 0.07), (0.012, 0.36, 0.17), look["glass"])
    panel("headlight", (xf, 0.0, roof - rt - 0.27), (0.012, 0.15, 0.045), look["light"])
    for sgn in (1, -1):
        y = sgn * 0.40
        panel(f"marker_front_{'l' if sgn > 0 else 'r'}", (side_x(y) + 0.004, y, roof - rt - 0.08), (0.012, 0.035, 0.11), look["amber"])
        panel(f"taillight_{'l' if sgn > 0 else 'r'}", (2 * cx - side_x(y) - 0.004, y, roof - rt - 0.08), (0.012, 0.035, 0.11), look["red"])
    # roof lidar, flag mast and flag
    panel("lidar", (x_front - 0.42, 0.0, roof + 0.05), (0.16, 0.24, 0.10), look["black"])
    mx, my = x_rear + 0.22, 0.30
    panel("mast", (mx, my, (roof + mast_top) / 2 - 0.05), (0.02, 0.02, mast_top - roof + 0.1), look["black"])
    flag = UsdGeom.Mesh.Define(stage, "/robot/chassis/visual/flag")
    flag.CreatePointsAttr([Gf.Vec3f(mx, my, mast_top), Gf.Vec3f(mx, my, mast_top - 0.22), Gf.Vec3f(mx - 0.25, my, mast_top - 0.11)])
    flag.CreateFaceVertexCountsAttr([3])
    flag.CreateFaceVertexIndicesAttr([0, 1, 2])
    flag.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(flag.GetPrim()).Bind(look["orange"])
    return {"lo": np.array([x_rear, -b_low, z0]), "hi": np.array([x_front, b_low, roof])}


def build_trike(stage, link, visual, wheel_collider, revolute, chassis, wheel_parts, centre, to_robot,
                left_name, right_name, rear_name, wheel_I, tyre_look, prismatic=None) -> None:
    """Front: steering knuckle (servo about z) + free wheel; rear: driven wheel on a fixed axle."""
    Iw = (wheel_I[1], wheel_I[0], wheel_I[1])
    filtered = []
    susp = suspension_rates(chassis, to_robot(centre[left_name])[0], to_robot(centre[right_name])[0]) if args.suspension else None

    def mount(name, c, end):
        """Chassis-side attachment of a wheel assembly: the chassis itself, or a sprung strut."""
        if susp is None:
            return "/robot/chassis", c
        strut = link(f"/robot/{name}_strut", c, args.strut_mass, (0.01, 0.01, 0.01))
        prismatic(f"/robot/joints/{name}_susp_joint", "/robot/chassis", strut.GetPath(), c, *susp[end])
        filtered.append(str(strut.GetPath()))
        return strut.GetPath(), (0, 0, 0)

    for side, bone in (("front_left", left_name), ("front_right", right_name)):
        c = to_robot(centre[bone])[0]
        parent, at = mount(side, c, "front")
        knuckle = link(f"/robot/{side}_knuckle", c, 2.0, (0.02, 0.02, 0.02))
        revolute(f"/robot/joints/{side}_steer_joint", parent, knuckle.GetPath(), at, "Z",
                 drive=False, servo=True, limit_deg=args.max_steer)
        w = link(f"/robot/{side}_wheel", c, args.wheel_mass, Iw)
        visual(f"/robot/{side}_wheel/visual", wheel_parts[bone], c, tyre_look)
        wheel_collider(f"/robot/{side}_wheel/collision")
        revolute(f"/robot/joints/{side}_wheel_joint", knuckle.GetPath(), w.GetPath(), (0, 0, 0), "Y", drive=False)
        filtered += [str(knuckle.GetPath()), str(w.GetPath())]
    c = to_robot(centre[rear_name])[0]
    w = link("/robot/rear_wheel", c, args.wheel_mass, Iw)
    visual("/robot/rear_wheel/visual", wheel_parts[rear_name], c, tyre_look)
    wheel_collider("/robot/rear_wheel/collision")
    parent, at = mount("rear", c, "rear")
    revolute("/robot/joints/rear_wheel_joint", parent, w.GetPath(), at, "Y", drive=True)
    filtered.append(str(w.GetPath()))
    UsdPhysics.FilteredPairsAPI.Apply(chassis.GetPrim()).CreateFilteredPairsRel().SetTargets(filtered)
    if susp is not None:
        stage.GetPrimAtPath("/robot").SetCustomDataByKey("talk2drive:arb_rate", susp["arb_rate"])
        stage.GetPrimAtPath("/robot").SetCustomDataByKey("talk2drive:suspension", susp["report"])


def suspension_rates(chassis, c_left, c_right) -> dict:
    """Spring, damper and anti-roll-bar rates from ride frequency, damping ratio and roll gradient.

    Sprung mass is the chassis; its share on each axle follows the mass centre. Only the front axle
    can resist roll (one rear wheel on the centreline), so the anti-roll bar is at the front. The
    struts slide vertically, so the roll centre is on the ground and the roll arm is the CG height.
    """
    m = UsdPhysics.MassAPI(chassis.GetPrim())
    ms = float(m.GetMassAttr().Get())
    com = m.GetCenterOfMassAttr().Get()
    L, t, h = float(c_left[0]), float(c_left[1] - c_right[1]), float(com[2])
    f_front, f_rear = (float(v) for v in args.ride_freq.split(","))
    m_front_corner = ms * float(com[0]) / L / 2  # rear axle at x = 0
    m_rear = ms * (1 - float(com[0]) / L)
    k_f = m_front_corner * (2 * math.pi * f_front) ** 2
    k_r = m_rear * (2 * math.pi * f_rear) ** 2
    c_f = 2 * args.damping_ratio * math.sqrt(k_f * m_front_corner)
    c_r = 2 * args.damping_ratio * math.sqrt(k_r * m_rear)
    roll_moment_per_g = ms * 9.81 * h
    k_phi = roll_moment_per_g / math.radians(args.roll_gradient)
    k_phi_springs = k_f * t * t / 2
    arb_rate = max(0.0, (k_phi - k_phi_springs) / (t * t))  # N/m on the left-right strut difference
    report = {"ride_freq_hz": [f_front, f_rear], "damping_ratio": args.damping_ratio, "roll_gradient_deg_per_g": args.roll_gradient,
              "spring_N_per_m": [round(k_f), round(k_r)], "damper_N_s_per_m": [round(c_f), round(c_r)],
              "static_deflection_mm": [round(1000 * 9.81 / (2 * math.pi * f_front) ** 2), round(1000 * 9.81 / (2 * math.pi * f_rear) ** 2)],
              "roll_stiffness_springs_Nm_per_rad": round(k_phi_springs), "roll_stiffness_total_Nm_per_rad": round(k_phi),
              "arb_rate_N_per_m": round(arb_rate), "travel_m": args.travel}
    print("[build] suspension:", report)
    load_f, load_r = m_front_corner * 9.81, m_rear * 9.81  # spring preload holds the chassis at design height
    return {"front": (k_f, c_f, load_f), "rear": (k_r, c_r, load_r), "arb_rate": arb_rate, "report": str(report)}


def main() -> None:
    fbx, out = Path(args.fbx).expanduser().resolve(), Path(args.out_usd).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_name(out.stem + "_raw.usd")
    convert(fbx, raw)
    parts = split_parts(raw)

    wheel_parts = {k: v for k, v in parts.items() if k.startswith("Wheel_")}
    body = parts["Root"]
    front = {k: v for k, v in wheel_parts.items() if "Front" in k}
    rear = {k: v for k, v in wheel_parts.items() if "Rear" in k}
    if len(front) != 2 or len(rear) != 1:
        raise RuntimeError(f"expected 2 front wheels and 1 rear wheel with geometry, got {sorted(wheel_parts)}")
    centre = {k: (v["min"] + v["max"]) / 2 for k, v in wheel_parts.items()}
    radius = float(np.mean([(v["max"][2] - v["min"][2]) / 2 for v in wheel_parts.values()]))
    width = float(np.mean([v["max"][0] - v["min"][0] for v in front.values()]))
    left_name = min(front, key=lambda k: centre[k][0])  # mesh frame: front = +y, so left = -x
    right_name = max(front, key=lambda k: centre[k][0])
    caster_name = rear_name = next(iter(rear))
    if args.layout == "trike":  # origin under the rear (driven) wheel
        axle_y, axle_z = centre[rear_name][1], centre[rear_name][2]
    else:  # origin at the front (driven) axle midpoint
        axle_y, axle_z = (centre[left_name][1] + centre[right_name][1]) / 2, centre[left_name][2]

    def to_robot(p):
        """Mesh frame (x right, y forward, z up) -> robot frame (x forward, y left, z up), origin on the ground."""
        p = np.atleast_2d(p)
        return np.c_[p[:, 1] - axle_y, -p[:, 0], p[:, 2] - axle_z + radius]

    track = float(abs(centre[right_name][0] - centre[left_name][0]))
    caster_c = to_robot(centre[caster_name])[0]
    wheelbase = float(abs(centre[left_name][1] - centre[rear_name][1]))
    print(f"[build] layout {args.layout}: wheel radius {radius:.3f} m, width {width:.3f} m, front track {track:.3f} m, "
          f"wheelbase {wheelbase:.3f} m")

    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/robot")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    art = PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
    art.CreateEnabledSelfCollisionsAttr(False)  # wheel spheres overlap the body box by design
    art.CreateSolverPositionIterationCountAttr(16)

    # materials: a look for the meshes and rubber for the tyres
    def look(name, rgb, rough):
        mat = UsdShade.Material.Define(stage, f"/robot/Looks/{name}")
        sh = UsdShade.Shader.Define(stage, f"/robot/Looks/{name}/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat

    body_look, tyre_look = look("Body", (0.85, 0.42, 0.08), 0.5), look("Tyre", (0.012, 0.012, 0.012), 0.8)
    shell_looks = {"white": look("ShellWhite", (0.92, 0.93, 0.94), 0.35), "glass": look("LockerGlass", (0.03, 0.035, 0.04), 0.15),
                   "black": look("Black", (0.02, 0.02, 0.02), 0.5), "light": look("Headlight", (1.0, 1.0, 0.95), 0.1),
                   "amber": look("Amber", (1.0, 0.55, 0.05), 0.3), "red": look("Red", (0.8, 0.05, 0.03), 0.3),
                   "orange": look("Flag", (1.0, 0.35, 0.0), 0.6)}
    rubber = UsdShade.Material.Define(stage, "/robot/PhysicsMaterials/Rubber")
    rm = UsdPhysics.MaterialAPI.Apply(rubber.GetPrim())
    rm.CreateStaticFrictionAttr(args.tyre_friction)
    rm.CreateDynamicFrictionAttr(args.tyre_friction * 0.9)
    rm.CreateRestitutionAttr(0.0)
    pm = PhysxSchema.PhysxMaterialAPI.Apply(rubber.GetPrim())
    pm.CreateFrictionCombineModeAttr("max")

    def link(path, pos, mass, inertia=None):
        x = UsdGeom.Xform.Define(stage, path)
        x.AddTranslateOp().Set(Gf.Vec3d(*map(float, pos)))
        UsdPhysics.RigidBodyAPI.Apply(x.GetPrim())
        m = UsdPhysics.MassAPI.Apply(x.GetPrim())
        m.CreateMassAttr(mass)
        if inertia is not None:
            m.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia))
        return x

    def visual(path, part, origin, mat):
        m = UsdGeom.Mesh.Define(stage, path)
        m.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, p)) for p in to_robot(part["points"]) - origin]))
        m.CreateFaceVertexCountsAttr(part["counts"])
        m.CreateFaceVertexIndicesAttr(part["indices"])
        m.CreateSubdivisionSchemeAttr("none")
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)

    def wheel_collider(path, mat_bind=True):
        s = UsdGeom.Sphere.Define(stage, path)  # spheres roll smoothly and touch the road at one point
        s.CreateRadiusAttr(radius)
        s.CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(s.GetPrim())
        if mat_bind:
            UsdShade.MaterialBindingAPI.Apply(s.GetPrim()).Bind(rubber, UsdShade.Tokens.weakerThanDescendants, "physics")

    def revolute(path, b0, b1, pos0, axis, drive: bool, servo: bool = False, limit_deg: float | None = None):
        j = UsdPhysics.RevoluteJoint.Define(stage, path)
        j.CreateBody0Rel().SetTargets([b0])
        j.CreateBody1Rel().SetTargets([b1])
        j.CreateAxisAttr(axis)
        j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, pos0)))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        if limit_deg is not None:
            j.CreateLowerLimitAttr(-limit_deg)
            j.CreateUpperLimitAttr(limit_deg)
        if servo:  # position-controlled steering motor
            d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
            d.CreateTypeAttr("force")
            d.CreateStiffnessAttr(args.steer_stiffness * math.pi / 180.0)
            d.CreateDampingAttr(args.steer_damping * math.pi / 180.0)
            d.CreateMaxForceAttr(args.steer_torque)
            d.CreateTargetPositionAttr(0.0)
        if drive:
            d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
            d.CreateTypeAttr("force")
            d.CreateStiffnessAttr(0.0)
            d.CreateDampingAttr(args.drive_damping * math.pi / 180.0)  # USD angular drives are per degree
            d.CreateMaxForceAttr(args.max_torque)
            d.CreateTargetVelocityAttr(0.0)
        return j

    def prismatic(path, b0, b1, pos0, k, c, preload):
        """Vertical strut: + = wheel moves up (bump). The drive's target is the free length, so the
        spring carries ``preload`` at zero travel (design ride height)."""
        j = UsdPhysics.PrismaticJoint.Define(stage, path)
        j.CreateBody0Rel().SetTargets([b0])
        j.CreateBody1Rel().SetTargets([b1])
        j.CreateAxisAttr("Z")
        j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, pos0)))
        j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
        j.CreateLowerLimitAttr(-args.travel)
        j.CreateUpperLimitAttr(args.travel)
        d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
        d.CreateTypeAttr("force")
        d.CreateStiffnessAttr(k)
        d.CreateDampingAttr(c)
        d.CreateMaxForceAttr(1e6)
        d.CreateTargetPositionAttr(-preload / k)
        return j

    # chassis: body box collider; mass centre at the body centre, low
    bmin, bmax = to_robot(body["min"])[0], to_robot(body["max"])[0]
    lo, hi = np.minimum(bmin, bmax), np.maximum(bmin, bmax)
    chassis = link("/robot/chassis", (0, 0, 0), args.chassis_mass)
    UsdPhysics.MassAPI(chassis.GetPrim()).CreateCenterOfMassAttr(
        Gf.Vec3f(float((lo[0] + hi[0]) / 2), 0.0, float(lo[2] + 0.35 * (hi[2] - lo[2]))))
    if args.shell == "magna":
        x_wheels = [to_robot(centre[k])[0][0] for k in wheel_parts]
        ext = build_magna_shell(stage, min(x_wheels) - args.overhang, max(x_wheels) + args.overhang,
                                float(hi[2]), args.bed_height, args.overall_height, shell_looks)
        lo, hi = ext["lo"], ext["hi"]  # collide with the shell, not the FBX's placeholder box
        UsdPhysics.MassAPI(chassis.GetPrim()).GetCenterOfMassAttr().Set(
            Gf.Vec3f(float((lo[0] + hi[0]) / 2), 0.0, 0.55))  # batteries and drive low in the pod
    else:
        visual("/robot/chassis/visual", body, np.zeros(3), body_look)
    box = UsdGeom.Cube.Define(stage, "/robot/chassis/collision")
    box.CreateSizeAttr(1.0)
    box.AddTranslateOp().Set(Gf.Vec3d(*map(float, (lo + hi) / 2)))
    box.AddScaleOp().Set(Gf.Vec3f(*map(float, hi - lo)))
    box.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())

    if args.com:
        UsdPhysics.MassAPI(chassis.GetPrim()).GetCenterOfMassAttr().Set(Gf.Vec3f(*map(float, args.com.split(","))))
    if args.chassis_inertia:
        UsdPhysics.MassAPI(chassis.GetPrim()).CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, args.chassis_inertia.split(","))))

    # wheels spin about +y (left axis), so positive speed rolls the robot forward (+x)
    wheel_I = (0.5 * args.wheel_mass * radius**2, 0.25 * args.wheel_mass * radius**2 + args.wheel_mass * width**2 / 12)
    if args.layout == "trike":
        build_trike(stage, link, visual, wheel_collider, revolute, chassis, wheel_parts, centre, to_robot,
                    left_name, right_name, rear_name, wheel_I, tyre_look, prismatic=prismatic)
        meta = root.GetPrim()
        for k, v in {"wheel_radius": radius, "wheelbase": wheelbase, "track": track}.items():
            meta.SetCustomDataByKey(f"talk2drive:{k}", v)
        stage.GetRootLayer().Save()
        raw.unlink(missing_ok=True)
        print(f"[build] wrote {out}  (radius {radius:.3f} m, wheelbase {wheelbase:.3f} m, track {track:.3f} m)")
        app.close()
        return

    for side, bone in (("left", left_name), ("right", right_name)):
        c = to_robot(centre[bone])[0]
        w = link(f"/robot/{side}_wheel", c, args.wheel_mass, (wheel_I[1], wheel_I[0], wheel_I[1]))
        visual(f"/robot/{side}_wheel/visual", wheel_parts[bone], c, tyre_look)
        wheel_collider(f"/robot/{side}_wheel/collision")
        revolute(f"/robot/joints/{side}_wheel_joint", "/robot/chassis", w.GetPath(), c, "Y", drive=True)

    # caster: fork swivels about z through a point ``trail`` ahead of the wheel centre
    pivot = caster_c + np.array([args.caster_trail, 0.0, 0.0])
    fork = link("/robot/caster_fork", pivot, 0.5, (0.002, 0.002, 0.002))
    revolute("/robot/joints/caster_swivel_joint", "/robot/chassis", fork.GetPath(), pivot, "Z", drive=False)
    cw = link("/robot/caster_wheel", caster_c, args.caster_mass,
              (0.25 * args.caster_mass * radius**2, 0.5 * args.caster_mass * radius**2, 0.25 * args.caster_mass * radius**2))
    visual("/robot/caster_wheel/visual", wheel_parts[caster_name], caster_c, tyre_look)
    wheel_collider("/robot/caster_wheel/collision", mat_bind=False)
    revolute("/robot/joints/caster_wheel_joint", fork.GetPath(), cw.GetPath(), caster_c - pivot, "Y", drive=False)

    # belt and braces: the body never collides with its own wheels or caster
    UsdPhysics.FilteredPairsAPI.Apply(chassis.GetPrim()).CreateFilteredPairsRel().SetTargets(
        ["/robot/left_wheel", "/robot/right_wheel", "/robot/caster_fork", "/robot/caster_wheel"])

    meta = root.GetPrim()
    for k, v in {"wheel_radius": radius, "wheel_base": track, "caster_offset": float(-caster_c[0])}.items():
        meta.SetCustomDataByKey(f"talk2drive:{k}", v)
    stage.GetRootLayer().Save()
    raw.unlink(missing_ok=True)
    print(f"[build] wrote {out}  (radius {radius:.3f} m, track {track:.3f} m)")
    app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        app.close()
        sys.exit(1)
