"""RTMW whole-body (COCO-WholeBody, 133 points: body, 6 foot points, face, 2 x 21 hand
points), ONNX. HAND = the middle-finger knuckle (exactly De Leva's hand end point).
The 68 face points are stored but not used. 2D only, like rtmpose_halpe26.
"""

from __future__ import annotations

from climbvision.engine.backends._topdown import TopDownEstimator
from climbvision.engine.skeleton import J

NAME = "rtmw_wholebody133"
WORLD = False

_BODY = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder",
         "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
         "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
_FEET = ["left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe",
         "right_heel"]
_FACE = [f"face_{i}" for i in range(68)]
_HAND = ["wrist", "thumb1", "thumb2", "thumb3", "thumb4", "index1", "index2", "index3",
         "index4", "middle1", "middle2", "middle3", "middle4", "ring1", "ring2", "ring3",
         "ring4", "pinky1", "pinky2", "pinky3", "pinky4"]
POINTS = (_BODY + _FEET + _FACE + [f"left_hand_{h}" for h in _HAND]
          + [f"right_hand_{h}" for h in _HAND])

MAPPING: dict[J, tuple[str, ...]] = {
    J.HEAD: ("left_ear", "right_ear"),
    **{J[f"{s[0].upper()}_{n.upper()}"]: (f"{s}_{n}",) for s in ("left", "right")
       for n in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle", "heel")},
    J.L_TOE: ("left_big_toe",), J.R_TOE: ("right_big_toe",),
    J.L_SMALL_TOE: ("left_small_toe",), J.R_SMALL_TOE: ("right_small_toe",),
    J.L_HAND: ("left_hand_middle1",), J.R_HAND: ("right_hand_middle1",),
}

MODELS = {"x": ("rtmw-dw-x-l_simcc-cocktail14_270e-384x288", (288, 384))}


class Estimator(TopDownEstimator):
    backend = NAME

    def __init__(self, model_size: str = "x", **kwargs) -> None:
        key, size = MODELS[model_size]
        super().__init__(key, size, len(POINTS), **kwargs)
