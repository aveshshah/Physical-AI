"""Road-load uncertainty for the Isaac Sim plant: road friction, unmodelled grade, wind, drag/rolling error.

The controller is never told about any of this: the NMPC keeps its nominal model (mu 0.9, flat road,
still air, nominal CdA / Crr) and has only its longitudinal-force disturbance observer to notice the
difference. The plant gets the real thing:

* road friction  -- physics materials on the drive surface (per-triangle, so patches can be local);
                    the tyre's material is switched to the ``min`` combine mode, so mu = min(tyre, road)
* grade          -- a smooth random grade profile along the route, applied as M g sin(theta) at the CG
                    (on top of Mcity's own terrain, which is real geometry)
* wind           -- mean wind (fixed world direction, so a loop sees head-, tail- and crosswind) plus
                    first-order gusts; drag along the body from the airspeed, a side force on the
                    2 m tall pod at its centre of pressure
* CdA, Crr       -- scale factors on the project's drag area and rolling-resistance coefficient
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np


@dataclass
class RoadLoadScenario:
    name: str = "nominal"
    mu: float = 0.9  # road friction everywhere (tyre rubber is 0.9)
    mu_patches: tuple = ()  # ((s_start, s_end, mu), ...) along the route, or "auto" (before the tightest bends)
    patch_mu: float = 0.3  # friction of the "auto" patches
    grade_pct: float = 0.0  # peak extra grade [%], smooth random profile along s
    grade_wavelength: float = 120.0  # [m] typical length of a rise + fall
    grade_offset_pct: float = 0.0  # constant extra grade [%] (+ = uphill in the driving direction)
    wind_speed: float = 0.0  # mean wind [m/s]
    wind_from_deg: float = 0.0  # direction the wind blows FROM, scene frame (0 = from +x)
    gust_std: float = 0.0  # gust standard deviation [m/s] (each horizontal axis)
    gust_tau: float = 2.0  # gust correlation time [s]
    cda_scale: float = 1.0  # true drag area / modelled drag area
    crr_scale: float = 1.0  # true rolling resistance / modelled
    seed: int = 0

    def describe(self) -> str:
        parts = []
        if self.mu != 0.9:
            parts.append(f"road mu {self.mu:g}")
        if self.mu_patches:
            parts.append("low-mu patches" + (f" (mu {self.patch_mu:g})" if self.mu_patches == "auto" else ""))
        if self.grade_pct or self.grade_offset_pct:
            parts.append(f"extra grade +-{self.grade_pct:g} %" + (f" {self.grade_offset_pct:+g} %" if self.grade_offset_pct else ""))
        if self.wind_speed or self.gust_std:
            parts.append(f"wind {self.wind_speed:g} m/s from {self.wind_from_deg:g} deg, gusts {self.gust_std:g} m/s")
        if self.cda_scale != 1.0 or self.crr_scale != 1.0:
            parts.append(f"CdA x{self.cda_scale:g}, Crr x{self.crr_scale:g}")
        return ", ".join(parts) or "nominal (matches the NMPC model)"


SCENARIOS: dict[str, RoadLoadScenario] = {s.name: s for s in (
    RoadLoadScenario("nominal"),
    RoadLoadScenario("wet", mu=0.6),
    RoadLoadScenario("low_mu", mu=0.35),
    RoadLoadScenario("ice_patches", mu_patches="auto", patch_mu=0.2),
    RoadLoadScenario("grade", grade_pct=6.0, grade_wavelength=120.0, seed=1),
    RoadLoadScenario("wind", wind_speed=8.0, wind_from_deg=30.0, gust_std=3.0, seed=2),
    RoadLoadScenario("storm", wind_speed=14.0, wind_from_deg=30.0, gust_std=5.0, seed=2),
    RoadLoadScenario("drag", cda_scale=1.6, crr_scale=2.0),
    RoadLoadScenario("combined", mu=0.6, grade_pct=4.0, wind_speed=8.0, wind_from_deg=30.0, gust_std=3.0,
                     cda_scale=1.3, crr_scale=1.5, seed=3),
)}


def parse_scenario(spec: str) -> RoadLoadScenario:
    """A preset name, optionally with overrides: ``wind``, ``wind:wind_speed=12,gust_std=4``, ``nominal:mu=0.5``."""
    name, _, rest = spec.partition(":")
    if name not in SCENARIOS:
        raise ValueError(f"unknown disturbance {name!r}; choose from {sorted(SCENARIOS)}")
    sc = SCENARIOS[name]
    if rest:
        kw = {}
        for item in rest.split(","):
            k, v = (p.strip() for p in item.split("="))
            cur = getattr(sc, k)
            kw[k] = v if k == "mu_patches" else (int(float(v)) if isinstance(cur, int) else float(v))
        sc = replace(sc, name=spec, **kw)
    return sc


@dataclass
class RoadLoadModel:
    """The plant's road loads for one scenario. Call ``forces`` once per control tick."""

    sc: RoadLoadScenario
    P: object  # tadpole_params (M, G, RHO_AIR, CD, A_FRONTAL, CRR, ROLLING_RESISTANCE_BLEND_SPEED)
    lap_length: float
    side_area: float = 3.0  # [m^2] pod side: 1.93 m long, shell from 0.28 to 2.0 m, rounded corners
    side_cy: float = 0.9  # side-force coefficient of a box-like body
    h_cp: float = 1.15  # [m] centre of pressure above the ground (middle of the shell)
    dt: float = 0.02
    _gust: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def __post_init__(self):
        rng = np.random.default_rng(self.sc.seed)
        self._rng = rng
        # grade: three sines with random wavelengths around the nominal and random phases, normalised
        n = 3
        self._gk = 2 * math.pi / (self.sc.grade_wavelength * rng.uniform(0.6, 1.6, n))
        self._gph = rng.uniform(0, 2 * math.pi, n)
        ss = np.linspace(0, max(self.lap_length, 1.0), 4000)
        peak = np.max(np.abs(np.sin(np.outer(ss, self._gk) + self._gph).sum(axis=1))) or 1.0
        self._gamp = self.sc.grade_pct / 100.0 / peak
        w = self.sc.wind_speed
        a = math.radians(self.sc.wind_from_deg)
        self.wind_mean = np.array([-w * math.cos(a), -w * math.sin(a)])  # air velocity (blows TOWARD)

    def grade(self, s: float) -> float:
        """Extra (unmodelled) grade as a slope tan(theta), + = uphill along the route."""
        return float(self._gamp * np.sin(s * self._gk + self._gph).sum() + self.sc.grade_offset_pct / 100.0)

    def wind(self) -> np.ndarray:
        """Air velocity in the scene frame; advances the gust process by one tick."""
        if self.sc.gust_std > 0:
            a = math.exp(-self.dt / self.sc.gust_tau)
            self._gust = a * self._gust + self.sc.gust_std * math.sqrt(1 - a * a) * self._rng.standard_normal(2)
        return self.wind_mean + self._gust

    def nominal_road_load(self, vx: float) -> float:
        """What the NMPC model assumes: still-air drag plus rolling resistance [N, resisting]."""
        P = self.P
        return 0.5 * P.RHO_AIR * P.CD * P.A_FRONTAL * vx * vx + P.CRR * P.M * P.G * math.tanh(
            vx / P.ROLLING_RESISTANCE_BLEND_SPEED)

    def forces(self, s: float, vx: float, vy: float, psi: float) -> dict:
        """Plant road loads this tick.

        Returns rolling resistance (applied as wheel torque), the aerodynamic body force (scene frame,
        at the centre of pressure), the grade force (scene frame, at the CG), and ``dF_true``: the
        longitudinal force the NMPC's model does not know about (same sign as the project's dFx).
        """
        P = self.P
        c, sn = math.cos(psi), math.sin(psi)
        air = self.wind()
        air_b = np.array([c * air[0] + sn * air[1], -sn * air[0] + c * air[1]])
        rel = np.array([vx, vy]) - air_b  # vehicle velocity relative to the air, body frame
        q = 0.5 * P.RHO_AIR * float(np.hypot(*rel))
        f_aero_b = np.array([-q * P.CD * P.A_FRONTAL * self.sc.cda_scale * rel[0], -q * self.side_cy * self.side_area * rel[1]])
        f_aero_w = np.array([c * f_aero_b[0] - sn * f_aero_b[1], sn * f_aero_b[0] + c * f_aero_b[1], 0.0])
        f_rr = self.sc.crr_scale * P.CRR * P.M * P.G * math.tanh(vx / P.ROLLING_RESISTANCE_BLEND_SPEED)
        g = self.grade(s)
        f_grade = -P.M * P.G * g / math.sqrt(1 + g * g)  # along the heading, + = pushes forward
        f_grade_w = np.array([f_grade * c, f_grade * sn, 0.0])
        nominal = self.nominal_road_load(vx)
        return {
            "f_rr": abs(f_rr),
            "f_aero_w": f_aero_w,
            "f_grade_w": f_grade_w,
            "air_w": air,
            "grade": g,
            "f_aero_x": float(f_aero_b[0]),
            "f_aero_y": float(f_aero_b[1]),
            "dF_true": float(f_grade + f_aero_b[0] - abs(f_rr) + nominal),
        }


# ---------------------------------------------------------------- Isaac side (needs a running Kit app)
def apply_road_friction(stage, sc: RoadLoadScenario, path, surface: str = "/World/DriveSurface",
                        tyre_material: str = "/World/Jetbot/PhysicsMaterials/Rubber",
                        lane_halfwidth: float = 4.0) -> list[tuple[float, float, float]]:
    """Bind friction materials to the drive surface (whole mesh plus optional patches along ``path``).

    Returns the patches actually used as (s_start, s_end, mu)."""
    from pxr import PhysxSchema, UsdGeom, UsdPhysics, UsdShade, Vt

    def material(name: str, mu: float):
        m = UsdShade.Material.Define(stage, f"/World/RoadMaterials/{name}")
        api = UsdPhysics.MaterialAPI.Apply(m.GetPrim())
        api.CreateStaticFrictionAttr(float(mu))
        api.CreateDynamicFrictionAttr(float(0.9 * mu))  # same static/dynamic ratio as the tyre rubber
        api.CreateRestitutionAttr(0.0)
        PhysxSchema.PhysxMaterialAPI.Apply(m.GetPrim()).CreateFrictionCombineModeAttr("min")
        return m

    tyre = stage.GetPrimAtPath(tyre_material)
    if not tyre.IsValid():
        raise RuntimeError(f"tyre material {tyre_material} not found")
    PhysxSchema.PhysxMaterialAPI.Apply(tyre).GetFrictionCombineModeAttr().Set("min")

    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(surface))
    base = material("road", sc.mu)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(base, UsdShade.Tokens.weakerThanDescendants, "physics")

    patches = sc.mu_patches
    if patches == "auto":
        patches = auto_patches(path, sc.patch_mu)
    if not patches:
        return []
    from scipy.spatial import cKDTree

    pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
    tri = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
    cen = pts[tri, :2].mean(axis=1)
    dist, idx = cKDTree(path.xy).query(cen, distance_upper_bound=lane_halfwidth)
    hit = np.isfinite(dist)
    s_tri = np.full(len(tri), -1.0)
    s_tri[hit] = idx[hit] * path.ds
    used = np.zeros(len(tri), dtype=bool)
    for i, (s0, s1, mu) in enumerate(patches):
        sel = hit & (s_tri >= s0) & (s_tri <= s1) & ~used
        used |= sel
        sub = UsdGeom.Subset.CreateGeomSubset(mesh, f"mu_patch_{i}", "face", Vt.IntArray(np.nonzero(sel)[0].tolist()),
                                              "materialBind", "nonOverlapping")
        UsdShade.MaterialBindingAPI.Apply(sub.GetPrim()).Bind(material(f"patch_{i}", mu),
                                                              UsdShade.Tokens.weakerThanDescendants, "physics")
    rest = UsdGeom.Subset.CreateGeomSubset(mesh, "mu_rest", "face", Vt.IntArray(np.nonzero(~used)[0].tolist()),
                                           "materialBind", "nonOverlapping")
    UsdShade.MaterialBindingAPI.Apply(rest.GetPrim()).Bind(base, UsdShade.Tokens.weakerThanDescendants, "physics")
    return [tuple(map(float, p)) for p in patches]


def auto_patches(path, mu: float, count: int = 3, before: float = 25.0, after: float = 10.0,
                 spacing: float = 120.0) -> list[tuple[float, float, float]]:
    """Low-friction patches over the braking zone and entry of the ``count`` tightest bends."""
    s = np.arange(0.0, path.length, path.ds)
    k = np.abs(path.kappa_many(s))
    out = []
    for i in np.argsort(-k):
        if len(out) == count:
            break
        if s[i] < before + 5 or all(abs(s[i] - (p[0] + before)) > spacing for p in out):
            if s[i] >= before + 5:
                out.append((float(s[i] - before), float(s[i] + after), mu))
    return sorted(out)


class BodyForceApplier:
    """Applies scene-frame forces to a rigid body for the next physics step (call before every step)."""

    def __init__(self, body_path: str):
        import omni.usd
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysicsSchemaTools

        self._sim = get_physx_simulation_interface()
        self._stage_id = omni.usd.get_context().get_stage_id()
        self._body = PhysicsSchemaTools.sdfPathToInt(body_path)
        self.loads: list[tuple[np.ndarray, np.ndarray]] = []

    def set(self, loads: list[tuple[np.ndarray, np.ndarray]]) -> None:
        self.loads = loads  # [(force_world, position_world), ...]

    def apply(self) -> None:
        from pxr import Gf

        for f, p in self.loads:
            self._sim.apply_force_at_pos(self._stage_id, self._body, Gf.Vec3f(*map(float, f)), Gf.Vec3f(*map(float, p)), "Force")
