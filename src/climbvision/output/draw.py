"""Overlay drawing. Colours are BGR."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from climbvision.engine.skeleton import BONES, Skeleton

BONE_COLOR = (230, 230, 230)
JOINT_COLOR = (0, 200, 255)
LOW_VIS_COLOR = (90, 90, 90)
COM_COLOR = (200, 60, 255)  # pink
TRAIL_COLOR = (230, 150, 255)
FLOOR_COLOR = (200, 200, 200)
INSIDE_COLOR = (90, 210, 60)  # green
OUTSIDE_COLOR = (50, 50, 240)  # red
CONTACT_COLOR = (255, 220, 0)  # cyan
GRIP_COLOR = (255, 90, 160)  # purple
BASE_COLOR = (0, 230, 255)  # yellow: base of support on the wall


def _pt(p: np.ndarray) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def _scale(frame: np.ndarray) -> float:
    return max(frame.shape[:2]) / 1000.0


def draw_skeleton(frame: np.ndarray, sk: Skeleton, min_visibility: float = 0.5) -> None:
    s = _scale(frame)
    lw = max(1, int(2 * s))
    known = np.isfinite(sk.image).all(axis=1)
    for a, b in BONES:
        if not (known[a] and known[b]):
            continue  # joint not provided by this backend / not found
        ok = min(sk.visibility[a], sk.visibility[b]) >= min_visibility
        cv2.line(frame, _pt(sk.image[a]), _pt(sk.image[b]),
                 BONE_COLOR if ok else LOW_VIS_COLOR, lw, cv2.LINE_AA)
    r = max(2, int(3 * s))
    for p, v in zip(sk.image[known], sk.visibility[known], strict=True):
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


def draw_label(frame: np.ndarray, text: str | Sequence[str],
               colors: Sequence[tuple[int, int, int]] | None = None) -> None:
    """Text block in the top-left corner on a dark box; one entry per line."""
    lines = [text] if isinstance(text, str) else list(text)
    colors = list(colors) if colors else [(255, 255, 255)] * len(lines)
    s = _scale(frame)
    font, size, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, max(1, int(2 * s))
    sizes = [cv2.getTextSize(t, font, size, thick) for t in lines]
    lh = int(max(h + b for (_, h), b in sizes) * 1.35)
    x, y0, pad = int(15 * s) + 5, int(40 * s) + 10, int(8 * s)
    tw = max(w for (w, _), _ in sizes)
    th = sizes[0][0][1]
    cv2.rectangle(frame, (x - pad, y0 - th - pad), (x + tw + pad, y0 + lh * (len(lines) - 1)
                                                      + pad + sizes[-1][1]), (0, 0, 0), -1)
    for k, (t, c) in enumerate(zip(lines, colors, strict=True)):
        cv2.putText(frame, t, (x, y0 + k * lh), font, size, c, thick, cv2.LINE_AA)


def draw_floor(frame: np.ndarray, slope: float, intercept: float) -> None:
    w = frame.shape[1]
    p0 = (0, int(round(intercept)))
    p1 = (w, int(round(slope * w + intercept)))
    cv2.line(frame, p0, p1, FLOOR_COLOR, max(1, int(_scale(frame))), cv2.LINE_AA)


def draw_support(frame: np.ndarray, slope: float, intercept: float, lo: float, hi: float,
                 com_x: float, com_y: float, inside: bool) -> None:
    """Support span as a bar on the floor, and the COM's plumb line down to it."""
    s = _scale(frame)
    color = INSIDE_COLOR if inside else OUTSIDE_COLOR
    y_lo, y_hi = slope * lo + intercept, slope * hi + intercept
    cv2.line(frame, _pt((lo, y_lo)), _pt((hi, y_hi)), color, max(3, int(8 * s)), cv2.LINE_AA)
    y_floor = slope * com_x + intercept
    n = 14
    ys = np.linspace(com_y, y_floor, 2 * n)
    for k in range(0, 2 * n - 1, 2):  # dashed plumb line
        cv2.line(frame, _pt((com_x, ys[k])), _pt((com_x, ys[k + 1])), color,
                 max(1, int(2 * s)), cv2.LINE_AA)
    cv2.drawMarker(frame, _pt((com_x, y_floor)), color, cv2.MARKER_TRIANGLE_UP,
                   max(8, int(22 * s)), max(2, int(3 * s)), cv2.LINE_AA)


def draw_contacts(frame: np.ndarray, points: np.ndarray) -> None:
    s = _scale(frame)
    for p in points:
        if np.all(np.isfinite(p)):
            cv2.circle(frame, _pt(p), max(5, int(10 * s)), CONTACT_COLOR, max(2, int(3 * s)),
                       cv2.LINE_AA)


def draw_base(frame: np.ndarray, points: np.ndarray, plumb: tuple | None = None,
              inside: bool | None = None) -> None:
    """Base of support on the wall: the hull of the limbs on holds (translucent yellow,
    3+ limbs; a line for 2) and the COM's plumb line down to the lowest hold (green when
    the COM is over the feet, red when not)."""
    s = _scale(frame)
    pts = np.array([p for p in points if np.all(np.isfinite(p))], np.float32).reshape(-1, 2)
    if len(pts) >= 3:
        hull = cv2.convexHull(pts).astype(np.int32)
        over = frame.copy()
        cv2.fillPoly(over, [hull], BASE_COLOR, cv2.LINE_AA)
        cv2.addWeighted(over, 0.22, frame, 0.78, 0, frame)
        cv2.polylines(frame, [hull], True, BASE_COLOR, max(2, int(3 * s)), cv2.LINE_AA)
    elif len(pts) == 2:
        cv2.line(frame, _pt(pts[0]), _pt(pts[1]), BASE_COLOR, max(2, int(3 * s)), cv2.LINE_AA)
    if plumb is not None and all(np.all(np.isfinite(p)) for p in plumb):
        col = BONE_COLOR if inside is None else (INSIDE_COLOR if inside else OUTSIDE_COLOR)
        a, b = np.asarray(plumb[0], float), np.asarray(plumb[1], float)
        n = max(2, int(np.linalg.norm(b - a) / max(6, 12 * s)))
        for k in range(0, n, 2):  # dashed
            p0, p1 = a + (b - a) * k / n, a + (b - a) * (k + 1) / n
            cv2.line(frame, _pt(p0), _pt(p1), col, max(2, int(2 * s)), cv2.LINE_AA)


def draw_grips(frame: np.ndarray, points: Sequence[np.ndarray]) -> None:
    """Hands on a bar: purple squares, joined by the bar line when both hands hold."""
    s = _scale(frame)
    pts = [p for p in points if np.all(np.isfinite(p))]
    if len(pts) == 2:
        a, b = pts
        ext = (b - a) * 0.25
        cv2.line(frame, _pt(a - ext), _pt(b + ext), GRIP_COLOR, max(2, int(4 * s)), cv2.LINE_AA)
    r = max(5, int(10 * s))
    for p in pts:
        x, y = _pt(p)
        cv2.rectangle(frame, (x - r, y - r), (x + r, y + r), GRIP_COLOR, max(2, int(3 * s)),
                      cv2.LINE_AA)
