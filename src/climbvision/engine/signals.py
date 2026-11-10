"""Named per-frame signals: the vocabulary recipes are written in.

A `Clip` bundles one analysed clip's layers (track, angles, support). Signals are
computed lazily by name from it; `clip.signal("knee")` etc. Adding a measurement to
the engine = registering one function here; recipes can then use it by name.

Naming: `<angle>` is the view-appropriate angle (biomech.Angles.primary),
`<angle>_l` / `<angle>_r` one side, `<angle>_l_2d` / `_3d` the raw versions.
Image y points down, so a *smaller* y is higher up.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from climbvision.engine.biomech import JOINT_ANGLES, Angles, Sex, compute_angles
from climbvision.engine.camera import CameraTrack, WallImage
from climbvision.engine.skeleton import J, side_joint
from climbvision.engine.support import Stability, compute_stability
from climbvision.engine.track import Track
from climbvision.engine.wall import LIMBS, WallSupport, compute_wall_support

ANGLE_NAMES = (*JOINT_ANGLES, "thigh")


@dataclass
class Clip:
    track: Track
    angles: Angles
    stability: Stability
    bar: bool = False
    camera: CameraTrack | None = None  # None = fixed camera assumed
    wall_image: WallImage | None = None  # the wall without the climber (camera layer)
    sex: str = "male"
    cleanup: dict | None = None  # clean-up layer report (None = not run)
    _cache: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @classmethod
    def build(cls, track: Track, sex: Sex = "male",
              frame_size: tuple[int, int] | None = None, bar: bool = False,
              camera: CameraTrack | None = None, wall_image: WallImage | None = None) -> Clip:
        ang = compute_angles(track)
        st = compute_stability(track, ang.view, sex, frame_size, bar=bar)
        return cls(track, ang, st, bar, camera, wall_image, sex)

    @property
    def wall_support(self) -> WallSupport:
        """Wall layer: anchors on holds, holds used, support on the wall (computed on
        first use)."""
        if "_wall_support" not in self._cache:
            sc = self.stability.scale
            self._cache["_wall_support"] = compute_wall_support(
                self.wall, self.t, sc.body_px, sc.px_per_m, self.sex,
                camera_ok=self.camera.ok if self.camera is not None else None,
                wall_image=self.wall_image)
        return self._cache["_wall_support"]  # type: ignore[return-value]

    @property
    def route(self):
        """Route layer: moves, rests, feet cut, hold usage (computed on first use)."""
        if "_route" not in self._cache:
            from climbvision.engine.route import compute_route

            self._cache["_route"] = compute_route(self.wall_support, self.t)
        return self._cache["_route"]

    @property
    def wall(self) -> np.ndarray:
        """(n, N_JOINTS, 2) joints in wall coordinates: camera motion removed. Without a
        camera layer the camera is assumed fixed (wall = image)."""
        if "_wall" not in self._cache:
            self._cache["_wall"] = (self.camera.to_wall(self.track.image)
                                    if self.camera is not None else self.track.image.copy())
        return self._cache["_wall"]

    @property
    def t(self) -> np.ndarray:
        return self.track.time_s

    def signal(self, name: str) -> np.ndarray:
        if name not in self._cache:
            self._cache[name] = _compute(self, name)
        return self._cache[name]

    def has(self, name: str) -> bool:
        try:
            self.signal(name)
            return True
        except KeyError:
            return False


SIGNALS: dict[str, tuple[Callable[[Clip], np.ndarray], str]] = {}


def register(name: str, doc: str):
    def deco(fn):
        SIGNALS[name] = (fn, doc)
        return fn
    return deco


def describe() -> dict[str, str]:
    """Every signal name with a one-line description (for docs / `climbvision signals`)."""
    out = {name: doc for name, (_, doc) in SIGNALS.items()}
    for a in (*ANGLE_NAMES, "trunk_lean"):
        out[a] = f"{a} angle (deg), best version for the view"
    for a in ANGLE_NAMES:
        out[f"{a}_l / {a}_r"] = f"{a} angle of one side (deg)"
        out[f"{a}_l_2d / _3d ..."] = f"{a} angle from image (2d) or 3D estimate (3d)"
    return dict(sorted(out.items()))


def _nanmean(*xs: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():  # frames with no pose are all-NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack(xs), axis=0)


def _compute(clip: Clip, name: str) -> np.ndarray:
    if name in SIGNALS:
        return SIGNALS[name][0](clip)
    ang = clip.angles
    if name in ang.values:  # raw: knee_l_2d
        return ang.values[name]
    if name == "trunk_lean" or name in ANGLE_NAMES:
        return ang.primary(name)
    base, _, s = name.rpartition("_")
    if base in ANGLE_NAMES and s in ("l", "r"):
        return ang.primary(base, "left" if s == "l" else "right")
    raise KeyError(f"unknown signal {name!r}")


# ------------------------------------------------------------------ body geometry

def _world_y(clip: Clip, joint: str, side: str | None) -> np.ndarray:
    w = clip.track.world
    if side:
        return w[:, side_joint(side, joint), 1]
    return (w[:, side_joint("left", joint), 1] + w[:, side_joint("right", joint), 1]) / 2


for _s, _suffix in ((None, ""), ("left", "_l"), ("right", "_r")):
    register(f"hip_below_knee_cm{_suffix}",
             "hip height below the knee (cm, 3D estimate; + = hip below knee)"
             + (f", {_s} leg" if _s else ", mean of both legs"))(
        lambda c, s=_s: (_world_y(c, "HIP", s) - _world_y(c, "KNEE", s)) * 100)


def hip_offset_pct(points: np.ndarray, side: str | None = None) -> np.ndarray:
    """Signed hip distance from the shoulder-ankle line, % of line length
    (+ = hips below the line / sag, - = pike). `points` (n, N_JOINTS, D), y down;
    side None = mid-points of both sides."""
    def mid(name: str) -> np.ndarray:
        if side:
            return points[:, side_joint(side, name)]
        return (points[:, side_joint("left", name)] + points[:, side_joint("right", name)]) / 2

    a, p, b = mid("SHOULDER"), mid("HIP"), mid("ANKLE")
    ab = b - a
    length = np.linalg.norm(ab, axis=-1)
    tpar = np.sum((p - a) * ab, axis=-1) / np.maximum(length**2, 1e-12)
    off = p - (a + tpar[:, None] * ab)
    mag = np.linalg.norm(off, axis=-1)
    return 100 * np.sign(off[:, 1]) * mag / np.maximum(length, 1e-12)


@register("hip_offset_pct", "hips off the shoulder-ankle line, % of its length "
          "(+ sag / - pike); side view: image, near side; front: 3D")
def _hip_offset(c: Clip) -> np.ndarray:
    if c.angles.view == "side":
        return hip_offset_pct(c.track.image, c.angles.side)
    return hip_offset_pct(c.track.world)


@register("feet_width", "ankle distance / shoulder width, in the image")
def _feet_width(c: Clip) -> np.ndarray:
    img = c.track.image
    sw = np.linalg.norm(img[:, J.L_SHOULDER] - img[:, J.R_SHOULDER], axis=-1)
    return np.linalg.norm(img[:, J.L_ANKLE] - img[:, J.R_ANKLE], axis=-1) / np.maximum(sw, 1e-6)


@register("shoulder_y", "image height of the mid-shoulders (px, down = larger)")
def _shoulder_y(c: Clip) -> np.ndarray:
    img = c.track.image
    return (img[:, J.L_SHOULDER, 1] + img[:, J.R_SHOULDER, 1]) / 2


@register("toe_y", "image height of the higher toe (px)")
def _toe_y(c: Clip) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmin(c.track.image[:, [J.L_TOE, J.R_TOE], 1], axis=1)


# ------------------------------------------------------------------ COM and support

register("com_x_px", "centre of mass, image x (px)")(lambda c: c.stability.com_px[:, 0])
register("com_y_px", "centre of mass, image y (px)")(lambda c: c.stability.com_px[:, 1])
register("com_x_cm", "COM horizontal position (cm, smoothed), for sway")(
    lambda c: c.stability.com_x_cm)
register("margin_cm", "COM to the nearest edge of the support (cm, + inside)")(
    lambda c: c.stability.margin_cm)
register("on_floor", "any body part on the floor")(
    lambda c: c.stability.contacts.on_floor())
register("stance", "what supports the body, e.g. 'L foot', 'feet + hands'")(
    lambda c: np.array(c.stability.labels, dtype=object))
register("ppm", "pixels per metre per frame")(
    lambda c: np.where(np.isfinite(c.stability.scale.ppm), c.stability.scale.ppm,
                       c.stability.scale.px_per_m))

# bar scenes
register("on_bar", "at least one hand gripping the bar")(lambda c: c.stability.on_bar)
register("hanging", "gripping with nothing on the floor")(lambda c: c.stability.hanging)
register("swing_cm", "COM x relative to the grip (cm)")(lambda c: c.stability.swing_cm)


@register("bar_y", "image height of the bar: mean y of the gripping hands (px)")
def _bar_y(c: Clip) -> np.ndarray:
    ct = c.stability.contacts
    ys = [np.where(ct.grip[s], ct.hand_px[s][:, 1], np.nan) for s in ct.grip]
    if not ys:
        return np.full(c.track.n, np.nan)
    return _nanmean(*ys)


# ------------------------------------------------------------------ wall coordinates (camera layer)

_PARTS = {"l_hand": (J.L_WRIST, J.L_HAND), "r_hand": (J.R_WRIST, J.R_HAND),
          "l_foot": (J.L_HEEL, J.L_TOE), "r_foot": (J.R_HEEL, J.R_TOE)}


def _part_speed(c: Clip, joints: tuple[J, ...], wall: bool) -> np.ndarray:
    from climbvision.engine.track import smooth

    pts = c.wall if wall else c.track.image
    p = _nanmean(*(pts[:, j] for j in joints))
    win = max(1, int(c.track.fps * 0.1))
    p = np.column_stack([smooth(p[:, 0], win), smooth(p[:, 1], win)])
    v = np.linalg.norm(np.gradient(p, c.t, axis=0), axis=1)
    return v / c.stability.scale.body_px


for _name, _joints in _PARTS.items():
    register(f"wall_speed_{_name}", f"speed of the {_name.replace('_', ' ')} on the wall "
             "(body lengths/s, camera motion removed)")(
        lambda c, j=_joints: _part_speed(c, j, wall=True))
    register(f"image_speed_{_name}", f"speed of the {_name.replace('_', ' ')} in the image "
             "(body lengths/s, includes camera motion)")(
        lambda c, j=_joints: _part_speed(c, j, wall=False))


@register("camera_ok", "frame registered to the wall (always true without a camera layer)")
def _camera_ok(c: Clip) -> np.ndarray:
    return c.camera.ok if c.camera is not None else np.ones(c.track.n, bool)


@register("camera_speed", "how fast the view moves over the wall (body lengths/s)")
def _camera_speed(c: Clip) -> np.ndarray:
    if c.camera is None:
        return np.zeros(c.track.n)
    from climbvision.engine.track import smooth

    p = c.camera.centre_path()
    win = max(1, int(c.track.fps * 0.2))
    p = np.column_stack([smooth(p[:, 0], win), smooth(p[:, 1], win)])
    return np.linalg.norm(np.gradient(p, c.t, axis=0), axis=1) / c.stability.scale.body_px


# ------------------------------------------------------------------ wall layer (anchors)

for _limb in LIMBS:
    register(f"on_hold_{_limb}", f"{_limb.replace('_', ' ')} anchored on a hold (still >= 0.5 s "
             "in wall coordinates, not on the ground)")(
        lambda c, limb=_limb: c.wall_support.on_hold[limb])
register("n_contacts", "number of limbs on holds (0-4)")(
    lambda c: c.wall_support.n_contacts.astype(float))
register("on_wall", "climbing: between the first and last foot on a hold, nothing on the "
         "ground")(lambda c: c.wall_support.on_wall)
register("on_ground", "standing on the ground (floor seen in the wall image)")(
    lambda c: c.wall_support.on_ground)
register("com_over_feet_cm", "COM to the nearest edge of the anchored feet span, "
         "horizontally (cm, + over the feet)")(lambda c: c.wall_support.com_over_feet_cm)
register("polygon_margin_cm", "COM to the edge of the hull of anchored limbs (cm, + inside; "
         ">= 3 limbs)")(lambda c: c.wall_support.polygon_margin_cm)
register("resting", "resting on the wall: 4 limbs on holds and the COM still for >= 1 s")(
    lambda c: c.route.resting)
register("feet_cut", "on the wall with a hand on a hold and both feet off holds (>= 0.3 s)")(
    lambda c: c.route.feet_off)
register("limbs_moving", "number of limbs travelling between holds (0-4)")(
    lambda c: c.route.in_move.astype(float))
register("wall_contacts", "label of the limbs on holds, e.g. '2 hands + R foot'")(
    lambda c: np.array(c.wall_support.labels, dtype=object))


@register("com_height_m", "COM height on the wall (m) above its lowest point while climbing")
def _com_height(c: Clip) -> np.ndarray:
    ws = c.wall_support
    y = ws.com[:, 1]
    base = np.nanmax(np.where(ws.on_wall, y, np.nan)) if ws.on_wall.any() else np.nanmax(y)
    return (base - y) / ws.px_per_m
