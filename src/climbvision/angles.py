"""Joint angles for a whole clip.

Conventions (degrees):
  knee, hip, elbow, body_line: 180 = straight, smaller = more bent
  ankle: angle knee-ankle-toe (~90-110 standing, smaller = more dorsiflexed)
  shoulder: angle hip-shoulder-elbow (0 = arm by the side, 180 = overhead)
  trunk_lean: hip->shoulder vs straight up (0 = upright, 90 = horizontal); assumes
        the camera is level
  thigh: hip->knee vs horizontal (+90 standing, 0 = thigh parallel to the floor,
        negative = hip below knee, i.e. below parallel in a squat)

Two versions of each joint angle:
  *_2d  from image pixels. Accurate for movement parallel to the camera
        (side view of a squat), wrong when limbs point towards the camera.
  *_3d  from MediaPipe's 3D estimate. Works from any view but is noisier.
The `primary` angles use 2D for side views (near side only) and 3D for front views.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from climbvision.skeleton import LM
from climbvision.track import Track

Side = Literal["left", "right"]
View = Literal["side", "front"]

# name -> (a, vertex, b) landmark-name suffixes, angle measured at the vertex
JOINTS: dict[str, tuple[str, str, str]] = {
    "knee": ("HIP", "KNEE", "ANKLE"),
    "hip": ("SHOULDER", "HIP", "KNEE"),
    "ankle": ("KNEE", "ANKLE", "FOOT_INDEX"),
    "elbow": ("SHOULDER", "ELBOW", "WRIST"),
    "shoulder": ("HIP", "SHOULDER", "ELBOW"),
    "body_line": ("SHOULDER", "HIP", "ANKLE"),
}


def angle_at(a: np.ndarray, v: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle a-v-b in degrees for arrays of points shaped (..., D)."""
    u, w = a - v, b - v
    cos = np.sum(u * w, axis=-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(w, axis=-1))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _lm(side: Side, name: str) -> LM:
    return LM[f"{side.upper()}_{name}"]


def body_yaw_deg(track: Track) -> float:
    """How far the body is turned from facing the camera, 0..90 degrees (clip median).

    Uses the depth difference between left and right shoulders/hips in MediaPipe's
    3D estimate: facing the camera they are at the same depth (0), side-on they
    differ by the full body width (90).
    """
    w = track.world[track.detected]
    yaws = []
    for a, b in ((LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER), (LM.LEFT_HIP, LM.RIGHT_HIP)):
        v = w[:, a] - w[:, b]
        yaws.append(np.degrees(np.arctan2(np.abs(v[:, 2]), np.abs(v[:, 0]))))
    y = np.concatenate(yaws) if yaws else np.array([])
    return float(np.nanmedian(y)) if np.isfinite(y).any() else float("nan")


def _shoulder_ratio(track: Track) -> float:
    img = track.image[track.detected]
    sw = np.linalg.norm(img[:, LM.LEFT_SHOULDER] - img[:, LM.RIGHT_SHOULDER], axis=-1)
    mid_s = (img[:, LM.LEFT_SHOULDER] + img[:, LM.RIGHT_SHOULDER]) / 2
    mid_h = (img[:, LM.LEFT_HIP] + img[:, LM.RIGHT_HIP]) / 2
    torso = np.linalg.norm(mid_s - mid_h, axis=-1)
    return float(np.nanmedian(sw / np.maximum(torso, 1e-6)))


def detect_view(track: Track) -> View:
    """Side or front view, from the body's 3D yaw; image shoulder width breaks ties.

    On test clips: front views 3-11 deg, side views 78-88 deg.
    """
    if not track.detected.any():
        return "front"
    yaw = body_yaw_deg(track)
    if np.isfinite(yaw) and yaw < 30:
        return "front"
    if np.isfinite(yaw) and yaw > 60:
        return "side"
    return "side" if _shoulder_ratio(track) < 0.3 else "front"


def near_side(track: Track) -> Side:
    """The side facing the camera = the one with higher leg/arm visibility."""
    names = ("SHOULDER", "ELBOW", "HIP", "KNEE", "ANKLE")
    vis = {s: np.mean([track.visibility[track.detected, _lm(s, n)].mean() for n in names])
           for s in ("left", "right")}
    return "left" if vis["left"] >= vis["right"] else "right"


@dataclass
class Angles:
    view: View
    yaw_deg: float  # body turned from facing the camera, 0 = front, 90 = side
    side: Side  # near side (used for 2D primary angles in side view)
    time_s: np.ndarray
    values: dict[str, np.ndarray] = field(default_factory=dict)  # column -> (n,)

    def primary(self, joint: str, side: Side | None = None) -> np.ndarray:
        """Best available angle for this clip's view."""
        if joint == "trunk_lean":
            return self.values["trunk_lean_2d" if self.view == "side" else "trunk_lean_3d"]
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
    for joint, (a, v, b) in JOINTS.items():
        for side in ("left", "right"):
            ia, iv, ib = _lm(side, a), _lm(side, v), _lm(side, b)
            for tag, pts in (("2d", track.image), ("3d", track.world)):
                out.values[f"{joint}_{side[0]}_{tag}"] = angle_at(pts[:, ia], pts[:, iv],
                                                                  pts[:, ib])
    # thigh vs horizontal: +90 standing, 0 = parallel, negative = hip below knee
    for side in ("left", "right"):
        for tag, pts in (("2d", track.image), ("3d", track.world)):
            d = pts[:, _lm(side, "KNEE")] - pts[:, _lm(side, "HIP")]
            horiz = np.abs(d[:, 0]) if tag == "2d" else np.hypot(d[:, 0], d[:, 2])
            out.values[f"thigh_{side[0]}_{tag}"] = np.degrees(np.arctan2(d[:, 1], horiz))

    for tag, pts in (("2d", track.image), ("3d", track.world)):
        mid_s = (pts[:, LM.LEFT_SHOULDER] + pts[:, LM.RIGHT_SHOULDER]) / 2
        mid_h = (pts[:, LM.LEFT_HIP] + pts[:, LM.RIGHT_HIP]) / 2
        vec = mid_s - mid_h
        up = np.zeros(vec.shape[1])
        up[1] = -1.0  # y points down in both image and MediaPipe world coords
        cos = vec @ up / np.linalg.norm(vec, axis=-1)
        out.values[f"trunk_lean_{tag}"] = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return out
