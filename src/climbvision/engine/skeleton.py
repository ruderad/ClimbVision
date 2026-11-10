"""The engine's canonical skeleton.

Every pose backend is mapped onto these named joints (see `engine/backends/`), and
everything downstream uses the names, never a backend's own point numbers. Joints a
backend doesn't provide are NaN. No face points: for movement analysis the head is
one joint (`HEAD`, centre of the head), plus `HEAD_TOP` / `NECK` when a backend has them.

Left/right are the person's own sides.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class J(IntEnum):
    HEAD = 0          # centre of the head (MediaPipe: midpoint of the ears)
    HEAD_TOP = 1      # vertex (optional)
    NECK = 2          # base of the neck / C7 (optional)
    L_SHOULDER = 3
    R_SHOULDER = 4
    L_ELBOW = 5
    R_ELBOW = 6
    L_WRIST = 7
    R_WRIST = 8
    L_HAND = 9        # knuckles / middle of the hand
    R_HAND = 10
    L_HIP = 11
    R_HIP = 12
    L_KNEE = 13
    R_KNEE = 14
    L_ANKLE = 15
    R_ANKLE = 16
    L_HEEL = 17
    R_HEEL = 18
    L_TOE = 19        # big toe / tip of the foot
    R_TOE = 20
    L_SMALL_TOE = 21  # optional
    R_SMALL_TOE = 22


N_JOINTS = len(J)
SIDES = ("left", "right")


def side_joint(side: str, name: str) -> J:
    """side_joint('left', 'KNEE') -> J.L_KNEE"""
    return J[f"{'L' if side == 'left' else 'R'}_{name}"]


def both(name: str) -> tuple[J, J]:
    return side_joint("left", name), side_joint("right", name)


# drawn skeleton: (a, b) pairs; a pair with a missing joint is simply not drawn
BONES: list[tuple[J, J]] = [
    (J.L_SHOULDER, J.R_SHOULDER), (J.L_HIP, J.R_HIP),
    (J.L_SHOULDER, J.L_HIP), (J.R_SHOULDER, J.R_HIP),
    (J.L_SHOULDER, J.L_ELBOW), (J.L_ELBOW, J.L_WRIST), (J.L_WRIST, J.L_HAND),
    (J.R_SHOULDER, J.R_ELBOW), (J.R_ELBOW, J.R_WRIST), (J.R_WRIST, J.R_HAND),
    (J.L_HIP, J.L_KNEE), (J.L_KNEE, J.L_ANKLE), (J.L_ANKLE, J.L_HEEL),
    (J.L_HEEL, J.L_TOE), (J.L_ANKLE, J.L_TOE), (J.L_HEEL, J.L_SMALL_TOE),
    (J.R_HIP, J.R_KNEE), (J.R_KNEE, J.R_ANKLE), (J.R_ANKLE, J.R_HEEL),
    (J.R_HEEL, J.R_TOE), (J.R_ANKLE, J.R_TOE), (J.R_HEEL, J.R_SMALL_TOE),
    (J.HEAD_TOP, J.NECK),
]


@dataclass
class Skeleton:
    """One person's canonical pose in one frame.

    image: (N_JOINTS, 2) pixels, x right, y down.
    world: (N_JOINTS, 3) metres, origin at the hip centre (y down), NaN if unknown.
    visibility: (N_JOINTS,) 0-1, how sure the backend is the joint is visible.
    """

    image: np.ndarray
    world: np.ndarray
    visibility: np.ndarray

    def __post_init__(self) -> None:
        self.image = np.asarray(self.image, dtype=float).reshape(N_JOINTS, 2)
        self.world = np.asarray(self.world, dtype=float).reshape(N_JOINTS, 3)
        self.visibility = np.asarray(self.visibility, dtype=float).reshape(N_JOINTS)


@dataclass
class RawPose:
    """A backend's own output for one frame (its point order and names)."""

    image: np.ndarray  # (k, 2) pixels
    world: np.ndarray  # (k, 3) metres or NaN
    visibility: np.ndarray  # (k,)
