"""Biomechanics layer: whole-body centre of mass and joint angles, on canonical joints.

Centre of mass — 14-segment model, De Leva (1996), J. Biomech. 29(9):1223-1230, Table 4:
    segment COM = proximal + ratio * (distal - proximal)
    whole-body COM = sum(mass_fraction_i * segment_COM_i)
Approximations when the backend lacks a joint: head COM at HEAD (centre of the head;
De Leva measures from the vertex), C7 -> mid-shoulders, hip joint centres -> hip points,
3rd knuckle -> HAND, lateral malleolus -> ANKLE. Few-cm absolute offsets, negligible for
how the COM moves.

Angles (degrees):
  knee, hip, elbow, body_line: 180 = straight;  ankle: knee-ankle-toe (~90-110 standing)
  shoulder: hip-shoulder-elbow (0 = arm by the side, 180 = overhead)
  thigh: hip->knee vs horizontal (+90 standing, 0 parallel, < 0 hip below knee)
  trunk_lean: hip->shoulder vs straight up (0 upright, 90 horizontal); camera assumed level
Each in two versions: *_2d from image pixels (exact for movement parallel to the image),
*_3d from the backend's 3D estimate (any view, noisier). `Angles.primary` picks: 2D on the
near side for side views, 3D (mean of both sides) for front views.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from climbvision.engine.skeleton import SIDES, J, side_joint
from climbvision.engine.track import Track

Sex = Literal["male", "female"]
Side = Literal["left", "right"]
View = Literal["side", "front"]

# ------------------------------------------------------------------ centre of mass

Endpoint = tuple[J, ...]  # averaged joints


@dataclass(frozen=True)
class Segment:
    name: str
    proximal: Endpoint
    distal: Endpoint
    male: tuple[float, float]  # (mass fraction, COM ratio from proximal)
    female: tuple[float, float]


_MID_SHOULDER = (J.L_SHOULDER, J.R_SHOULDER)
_MID_HIP = (J.L_HIP, J.R_HIP)


def _build_segments() -> list[Segment]:
    segs = [
        Segment("head", (J.HEAD,), _MID_SHOULDER, (0.0694, 0.0), (0.0668, 0.0)),
        Segment("trunk", _MID_SHOULDER, _MID_HIP, (0.4346, 0.4486), (0.4257, 0.4151)),
    ]
    for side in SIDES:
        def s(name: str, side: str = side) -> J:
            return side_joint(side, name)
        segs += [
            Segment(f"{side}_upper_arm", (s("SHOULDER"),), (s("ELBOW"),),
                    (0.0271, 0.5772), (0.0255, 0.5754)),
            Segment(f"{side}_forearm", (s("ELBOW"),), (s("WRIST"),),
                    (0.0162, 0.4574), (0.0138, 0.4559)),
            Segment(f"{side}_hand", (s("WRIST"),), (s("HAND"),),
                    (0.0061, 0.7900), (0.0056, 0.7474)),
            Segment(f"{side}_thigh", (s("HIP"),), (s("KNEE"),),
                    (0.1416, 0.4095), (0.1478, 0.3612)),
            Segment(f"{side}_shank", (s("KNEE"),), (s("ANKLE"),),
                    (0.0433, 0.4459), (0.0481, 0.4416)),
            Segment(f"{side}_foot", (s("HEEL"),), (s("TOE"),),
                    (0.0137, 0.4415), (0.0129, 0.4014)),
        ]
    return segs


SEGMENTS: list[Segment] = _build_segments()


def _point(points: np.ndarray, endpoint: Endpoint) -> np.ndarray:
    return points[..., list(endpoint), :].mean(axis=-2)


def segment_params(sex: Sex = "male") -> tuple[np.ndarray, np.ndarray]:
    """(mass fractions normalised to sum 1, COM ratios), in SEGMENTS order."""
    table = np.array([getattr(s, sex) for s in SEGMENTS])
    return table[:, 0] / table[:, 0].sum(), table[:, 1]


def segment_coms(points: np.ndarray, sex: Sex = "male") -> np.ndarray:
    """COM of every segment, (..., 14, D), for points (..., N_JOINTS, D)."""
    points = np.asarray(points, dtype=float)
    _, ratios = segment_params(sex)
    segs = []
    for seg, r in zip(SEGMENTS, ratios, strict=True):
        p, d = _point(points, seg.proximal), _point(points, seg.distal)
        segs.append(p + r * (d - p))
    return np.stack(segs, axis=-2)


def whole_body_com(points: np.ndarray, sex: Sex = "male") -> np.ndarray:
    """Whole-body centre of mass, (..., D)."""
    masses, _ = segment_params(sex)
    return np.einsum("s,...sd->...d", masses, segment_coms(points, sex))


def com_confidence(visibility: np.ndarray, sex: Sex = "male") -> np.ndarray:
    """Mass-weighted visibility of the joints the COM depends on (each segment counts
    its least visible joint): a hidden foot costs ~1.4 %, a hidden trunk ~43 %."""
    visibility = np.asarray(visibility, dtype=float)
    masses, _ = segment_params(sex)
    seg_vis = np.stack([visibility[..., list(s.proximal + s.distal)].min(axis=-1)
                        for s in SEGMENTS], axis=-1)
    return seg_vis @ masses


# ------------------------------------------------------------------ angles

# name -> (a, vertex, b) joint-name suffixes, angle measured at the vertex
JOINT_ANGLES: dict[str, tuple[str, str, str]] = {
    "knee": ("HIP", "KNEE", "ANKLE"),
    "hip": ("SHOULDER", "HIP", "KNEE"),
    "ankle": ("KNEE", "ANKLE", "TOE"),
    "elbow": ("SHOULDER", "ELBOW", "WRIST"),
    "shoulder": ("HIP", "SHOULDER", "ELBOW"),
    "body_line": ("SHOULDER", "HIP", "ANKLE"),
}


def angle_at(a: np.ndarray, v: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle a-v-b in degrees for arrays of points (..., D)."""
    u, w = a - v, b - v
    cos = np.sum(u * w, axis=-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(w, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def body_yaw_deg(track: Track) -> float:
    """How far the body is turned from facing the camera, 0..90 deg (clip median), from
    the depth difference between left and right shoulders/hips in the 3D estimate."""
    w = track.world[track.detected]
    yaws = []
    for a, b in ((J.L_SHOULDER, J.R_SHOULDER), (J.L_HIP, J.R_HIP)):
        v = w[:, a] - w[:, b]
        yaws.append(np.degrees(np.arctan2(np.abs(v[:, 2]), np.abs(v[:, 0]))))
    y = np.concatenate(yaws) if yaws else np.array([])
    return float(np.nanmedian(y)) if np.isfinite(y).any() else float("nan")


def _shoulder_ratio(track: Track) -> float:
    img = track.image[track.detected]
    sw = np.linalg.norm(img[:, J.L_SHOULDER] - img[:, J.R_SHOULDER], axis=-1)
    mid_s = (img[:, J.L_SHOULDER] + img[:, J.R_SHOULDER]) / 2
    mid_h = (img[:, J.L_HIP] + img[:, J.R_HIP]) / 2
    torso = np.linalg.norm(mid_s - mid_h, axis=-1)
    return float(np.nanmedian(sw / np.maximum(torso, 1e-6)))


def detect_view(track: Track) -> View:
    """Side or front view from the body's yaw (< 30 front, > 60 side); image shoulder
    width breaks ties. Test clips: front 3-11 deg, side 78-88 deg."""
    if not track.detected.any():
        return "front"
    yaw = body_yaw_deg(track)
    if np.isfinite(yaw) and yaw < 30:
        return "front"
    if np.isfinite(yaw) and yaw > 60:
        return "side"
    return "side" if _shoulder_ratio(track) < 0.3 else "front"


def near_side(track: Track) -> Side:
    """The side facing the camera = higher mean visibility of the main joints."""
    names = ("SHOULDER", "ELBOW", "HIP", "KNEE", "ANKLE")
    vis = {s: np.mean([track.visibility[track.detected, side_joint(s, n)].mean()
                       for n in names]) for s in SIDES}
    return "left" if vis["left"] >= vis["right"] else "right"


@dataclass
class Angles:
    view: View
    yaw_deg: float
    side: Side  # near side
    time_s: np.ndarray
    values: dict[str, np.ndarray] = field(default_factory=dict)  # "knee_l_2d" -> (n,)

    def primary(self, joint: str, side: Side | None = None) -> np.ndarray:
        """Best available angle for this clip's view (optionally for one side). Without a
        3D estimate (2D-only backends) the 2D angles are used for every view."""
        has_3d = np.isfinite(self.values.get("trunk_lean_3d", np.array([np.nan]))).any()
        if joint == "trunk_lean":
            return self.values["trunk_lean_2d" if self.view == "side" or not has_3d
                               else "trunk_lean_3d"]
        if not has_3d and self.view != "side":
            if side is not None:
                return self.values[f"{joint}_{side[0]}_2d"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                return np.nanmean([self.values[f"{joint}_l_2d"], self.values[f"{joint}_r_2d"]],
                                  axis=0)
        if self.view == "side":
            return self.values[f"{joint}_{(side or self.side)[0]}_2d"]
        if side is not None:
            return self.values[f"{joint}_{side[0]}_3d"]
        with warnings.catch_warnings():  # frames with no pose are all-NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean([self.values[f"{joint}_l_3d"], self.values[f"{joint}_r_3d"]],
                              axis=0)


def compute_angles(track: Track) -> Angles:
    out = Angles(view=detect_view(track), yaw_deg=round(body_yaw_deg(track), 1),
                 side=near_side(track), time_s=track.time_s)
    for joint, (a, v, b) in JOINT_ANGLES.items():
        for side in SIDES:
            ia, iv, ib = (side_joint(side, x) for x in (a, v, b))
            for tag, pts in (("2d", track.image), ("3d", track.world)):
                out.values[f"{joint}_{side[0]}_{tag}"] = angle_at(pts[:, ia], pts[:, iv],
                                                                  pts[:, ib])
    for side in SIDES:
        for tag, pts in (("2d", track.image), ("3d", track.world)):
            d = pts[:, side_joint(side, "KNEE")] - pts[:, side_joint(side, "HIP")]
            horiz = np.abs(d[:, 0]) if tag == "2d" else np.hypot(d[:, 0], d[:, 2])
            out.values[f"thigh_{side[0]}_{tag}"] = np.degrees(np.arctan2(d[:, 1], horiz))
    for tag, pts in (("2d", track.image), ("3d", track.world)):
        vec = (pts[:, J.L_SHOULDER] + pts[:, J.R_SHOULDER]) / 2 \
            - (pts[:, J.L_HIP] + pts[:, J.R_HIP]) / 2
        up = np.zeros(vec.shape[1])
        up[1] = -1.0  # y points down in image and world coordinates
        cos = vec @ up / np.linalg.norm(vec, axis=-1)
        out.values[f"trunk_lean_{tag}"] = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return out
