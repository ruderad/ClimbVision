"""RTMPose with the Halpe-26 point set (body + head top, neck, toes and heels), ONNX.

Sizes: m (256x192, fast) or x (384x288, most accurate). 2D only: no 3D estimate, so
the scale comes from an assumed body size and front-view angles are 2D. No hand point:
HAND is extrapolated along the forearm.
"""

from __future__ import annotations

from typing import Literal

from climbvision.engine.backends._topdown import TopDownEstimator
from climbvision.engine.skeleton import J

NAME = "rtmpose_halpe26"
WORLD = False

POINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip",
    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "head", "neck",
    "hip", "left_big_toe", "right_big_toe", "left_small_toe", "right_small_toe",
    "left_heel", "right_heel",
]

MAPPING: dict[J, tuple[str, ...]] = {
    J.HEAD: ("left_ear", "right_ear"), J.HEAD_TOP: ("head",), J.NECK: ("neck",),
    **{J[f"{s[0].upper()}_{n.upper()}"]: (f"{s}_{n}",) for s in ("left", "right")
       for n in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle", "heel")},
    J.L_TOE: ("left_big_toe",), J.R_TOE: ("right_big_toe",),
    J.L_SMALL_TOE: ("left_small_toe",), J.R_SMALL_TOE: ("right_small_toe",),
}
# knuckles ~0.3 forearm lengths beyond the wrist
DERIVED = {J.L_HAND: (J.L_ELBOW, J.L_WRIST, 0.3), J.R_HAND: (J.R_ELBOW, J.R_WRIST, 0.3)}

MODELS = {"m": ("rtmpose-m_simcc-body7_pt-body7-halpe26", (192, 256)),
          "x": ("rtmpose-x_simcc-body7_pt-body7-halpe26", (288, 384))}


class Estimator(TopDownEstimator):
    backend = NAME

    def __init__(self, model_size: Literal["m", "x"] = "m", **kwargs) -> None:
        key, size = MODELS[model_size]
        super().__init__(key, size, len(POINTS), **kwargs)
