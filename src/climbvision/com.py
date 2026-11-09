"""Whole-body centre of mass from a 14-segment model.

Segment parameters come from De Leva (1996), "Adjustments to Zatsiorsky-Seluyanov's
segment inertia parameters", J. Biomech. 29(9):1223-1230, Table 4.

For each segment:
    segment COM = proximal + ratio * (distal - proximal)
    whole-body COM = sum(mass_fraction_i * segment_COM_i)

Approximations forced by the 33 MediaPipe landmarks (no skull vertex, no C7,
no true joint centres):
  * Head:  COM taken at the midpoint of the ears (De Leva measures from the
           vertex, which pose models don't give; the head COM lies close to
           ear-canal level).
  * Trunk: C7 -> mid-shoulders, hip joint centres -> mid-hip landmarks.
  * Hand:  3rd metacarpal head -> midpoint of index and pinky landmarks.
  * Ankle landmark stands in for the lateral malleolus.
These are fine for tracking how COM moves; the absolute position has a few cm
of systematic error. Revisit if we ever compare against force-plate data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from climbvision.skeleton import LM

Sex = Literal["male", "female"]

Endpoint = tuple[LM, ...]  # averaged landmarks


@dataclass(frozen=True)
class Segment:
    name: str
    proximal: Endpoint
    distal: Endpoint
    # (mass fraction, COM ratio from proximal) for male / female
    male: tuple[float, float]
    female: tuple[float, float]


def _side(prefix: str) -> dict[str, LM]:
    p = prefix.upper()
    return {k: LM[f"{p}_{k}"] for k in
            ("SHOULDER", "ELBOW", "WRIST", "INDEX", "PINKY", "HIP", "KNEE",
             "ANKLE", "HEEL", "FOOT_INDEX")}


_MID_SHOULDER = (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER)
_MID_HIP = (LM.LEFT_HIP, LM.RIGHT_HIP)
_MID_EARS = (LM.LEFT_EAR, LM.RIGHT_EAR)


def _build_segments() -> list[Segment]:
    segs = [
        # ratio 0 => COM at the proximal point (mid-ears); see module docstring
        Segment("head", _MID_EARS, _MID_SHOULDER, (0.0694, 0.0), (0.0668, 0.0)),
        Segment("trunk", _MID_SHOULDER, _MID_HIP, (0.4346, 0.4486), (0.4257, 0.4151)),
    ]
    for side in ("left", "right"):
        s = _side(side)
        segs += [
            Segment(f"{side}_upper_arm", (s["SHOULDER"],), (s["ELBOW"],),
                    (0.0271, 0.5772), (0.0255, 0.5754)),
            Segment(f"{side}_forearm", (s["ELBOW"],), (s["WRIST"],),
                    (0.0162, 0.4574), (0.0138, 0.4559)),
            Segment(f"{side}_hand", (s["WRIST"],), (s["INDEX"], s["PINKY"]),
                    (0.0061, 0.7900), (0.0056, 0.7474)),
            Segment(f"{side}_thigh", (s["HIP"],), (s["KNEE"],),
                    (0.1416, 0.4095), (0.1478, 0.3612)),
            Segment(f"{side}_shank", (s["KNEE"],), (s["ANKLE"],),
                    (0.0433, 0.4459), (0.0481, 0.4416)),
            Segment(f"{side}_foot", (s["HEEL"],), (s["FOOT_INDEX"],),
                    (0.0137, 0.4415), (0.0129, 0.4014)),
        ]
    return segs


SEGMENTS: list[Segment] = _build_segments()


def _point(points: np.ndarray, endpoint: Endpoint) -> np.ndarray:
    return points[..., list(endpoint), :].mean(axis=-2)


def segment_params(sex: Sex = "male") -> tuple[np.ndarray, np.ndarray]:
    """Return (mass fractions normalised to sum 1, COM ratios) in SEGMENTS order."""
    table = np.array([getattr(s, sex) for s in SEGMENTS])
    masses = table[:, 0] / table[:, 0].sum()  # De Leva female sums to 0.9999
    return masses, table[:, 1]


def segment_coms(points: np.ndarray, sex: Sex = "male") -> np.ndarray:
    """COM of every segment, shape (..., 14, D).

    `points` is (..., 33, D): one frame (33, D) or a whole clip (n, 33, D);
    D = 2 for pixels, 3 for world coordinates.
    """
    points = np.asarray(points, dtype=float)
    _, ratios = segment_params(sex)
    segs = []
    for seg, r in zip(SEGMENTS, ratios, strict=True):
        p = _point(points, seg.proximal)
        d = _point(points, seg.distal)
        segs.append(p + r * (d - p))
    return np.stack(segs, axis=-2)


def whole_body_com(points: np.ndarray, sex: Sex = "male") -> np.ndarray:
    """Whole-body centre of mass, shape (..., D)."""
    masses, _ = segment_params(sex)
    return np.einsum("s,...sd->...d", masses, segment_coms(points, sex))


def com_confidence(visibility: np.ndarray, sex: Sex = "male") -> float:
    """Mass-weighted visibility of the landmarks the COM depends on, in [0, 1].

    A segment's confidence is the lowest visibility among its landmarks, so a
    hidden foot lowers confidence by the foot's (small) share of body mass.
    """
    visibility = np.asarray(visibility, dtype=float)
    masses, _ = segment_params(sex)
    seg_vis = np.array([visibility[list(s.proximal + s.distal)].min() for s in SEGMENTS])
    return float(masses @ seg_vis)
