"""Backend-independent skeleton data model.

Everything downstream (COM, angles, drawing) works on `Skeleton`, never on
MediaPipe objects, so the pose backend can be swapped without touching it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class LM(IntEnum):
    """The 33 body landmarks (MediaPipe Pose / BlazePose topology)."""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


N_LANDMARKS = len(LM)

# Bones drawn for the skeleton overlay.
BONES: list[tuple[LM, LM]] = [
    (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER),
    (LM.LEFT_HIP, LM.RIGHT_HIP),
    (LM.LEFT_SHOULDER, LM.LEFT_HIP),
    (LM.RIGHT_SHOULDER, LM.RIGHT_HIP),
    (LM.LEFT_SHOULDER, LM.LEFT_ELBOW),
    (LM.LEFT_ELBOW, LM.LEFT_WRIST),
    (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW),
    (LM.RIGHT_ELBOW, LM.RIGHT_WRIST),
    (LM.LEFT_HIP, LM.LEFT_KNEE),
    (LM.LEFT_KNEE, LM.LEFT_ANKLE),
    (LM.RIGHT_HIP, LM.RIGHT_KNEE),
    (LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
    (LM.LEFT_ANKLE, LM.LEFT_HEEL),
    (LM.LEFT_HEEL, LM.LEFT_FOOT_INDEX),
    (LM.LEFT_ANKLE, LM.LEFT_FOOT_INDEX),
    (LM.RIGHT_ANKLE, LM.RIGHT_HEEL),
    (LM.RIGHT_HEEL, LM.RIGHT_FOOT_INDEX),
    (LM.RIGHT_ANKLE, LM.RIGHT_FOOT_INDEX),
    (LM.LEFT_WRIST, LM.LEFT_INDEX),
    (LM.LEFT_WRIST, LM.LEFT_PINKY),
    (LM.RIGHT_WRIST, LM.RIGHT_INDEX),
    (LM.RIGHT_WRIST, LM.RIGHT_PINKY),
    (LM.LEFT_EAR, LM.RIGHT_EAR),
]


@dataclass
class Skeleton:
    """One person's pose in one frame.

    Attributes:
        image: (33, 2) pixel coordinates (x right, y down).
        world: (33, 3) metric coordinates in metres, origin at the hip centre
            (MediaPipe convention: x right, y down, z towards camera is negative).
            May be all-NaN if the backend has no 3D estimate.
        visibility: (33,) confidence in [0, 1] that each landmark is visible.
    """

    image: np.ndarray
    world: np.ndarray
    visibility: np.ndarray

    def __post_init__(self) -> None:
        self.image = np.asarray(self.image, dtype=float).reshape(N_LANDMARKS, 2)
        self.world = np.asarray(self.world, dtype=float).reshape(N_LANDMARKS, 3)
        self.visibility = np.asarray(self.visibility, dtype=float).reshape(N_LANDMARKS)


@dataclass
class Frame:
    """A video frame's index, timestamp and detected skeleton (None if no person)."""

    index: int
    time_s: float
    skeleton: Skeleton | None
