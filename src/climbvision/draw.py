"""Overlay drawing. Colours are BGR."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from climbvision.skeleton import BONES, Skeleton

BONE_COLOR = (230, 230, 230)
JOINT_COLOR = (0, 200, 255)
LOW_VIS_COLOR = (90, 90, 90)
COM_COLOR = (60, 60, 255)
TRAIL_COLOR = (60, 200, 60)


def _pt(p: np.ndarray) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def _scale(frame: np.ndarray) -> float:
    return max(frame.shape[:2]) / 1000.0


def draw_skeleton(frame: np.ndarray, sk: Skeleton, min_visibility: float = 0.5) -> None:
    s = _scale(frame)
    lw = max(1, int(2 * s))
    for a, b in BONES:
        ok = min(sk.visibility[a], sk.visibility[b]) >= min_visibility
        cv2.line(frame, _pt(sk.image[a]), _pt(sk.image[b]),
                 BONE_COLOR if ok else LOW_VIS_COLOR, lw, cv2.LINE_AA)
    r = max(2, int(3 * s))
    for p, v in zip(sk.image, sk.visibility, strict=False):
        cv2.circle(frame, _pt(p), r, JOINT_COLOR if v >= min_visibility else LOW_VIS_COLOR,
                   -1, cv2.LINE_AA)


def draw_com(frame: np.ndarray, com_xy: np.ndarray, trail: Sequence[np.ndarray] = ()) -> None:
    s = _scale(frame)
    pts = [p for p in trail if np.all(np.isfinite(p))]
    for i in range(1, len(pts)):
        cv2.line(frame, _pt(pts[i - 1]), _pt(pts[i]), TRAIL_COLOR, max(1, int(2 * s)),
                 cv2.LINE_AA)
    if np.all(np.isfinite(com_xy)):
        r = max(4, int(8 * s))
        cv2.circle(frame, _pt(com_xy), r, COM_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, _pt(com_xy), r + 2, (255, 255, 255), max(1, int(s)), cv2.LINE_AA)


def draw_label(frame: np.ndarray, text: str) -> None:
    s = _scale(frame)
    cv2.putText(frame, text, (int(15 * s) + 5, int(40 * s) + 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.8 * s, (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
