"""The ClimbVision watermark on output videos (annotated MP4s, viewer.mp4, wall_view.mp4).

On by default; `--no-watermark` turns it off. The mark is the identity's video
watermark (`visual-identity/CURRENT/colour/png/climbvision_video_watermark_snow_800.png`:
Snow #F4F8FA, monochrome — "chalk on footage", so it doesn't clash with the overlay
colours). It sits in the bottom-right corner of the video picture (never on the timeline
strip), 30 % of the frame's short side wide, at 75 % opacity, over a soft Alpine-night
shadow so it reads on a light wall too.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

ASSET = Path(__file__).parent / "assets" / "watermark.png"
WIDTH = 0.30  # of the frame's short side
MARGIN = 0.03  # of the frame's short side, from the right and bottom edges
OPACITY = 0.75
SHADOW = (44, 28, 11)  # Alpine night #0B1C2C, BGR
SHADOW_OPACITY = 0.5


@lru_cache(maxsize=1)
def _source() -> np.ndarray | None:
    img = cv2.imread(str(ASSET), cv2.IMREAD_UNCHANGED)
    if img is None or img.ndim != 3 or img.shape[2] != 4:
        return None
    return img


@lru_cache(maxsize=8)
def _prepared(w: int, h: int):
    """(x, y, colour (h, w, 3) float, alpha (h, w, 1), shadow alpha (h, w, 1)) for a
    w x h frame, or None if the asset is missing or the frame too small."""
    src = _source()
    short = min(w, h)
    if src is None or short < 120:
        return None
    ww = max(40, int(round(WIDTH * short)))
    wh = max(1, int(round(src.shape[0] * ww / src.shape[1])))
    mark = cv2.resize(src, (ww, wh), interpolation=cv2.INTER_AREA).astype(np.float32)
    pad = max(2, int(round(0.04 * wh)))
    a = np.zeros((wh + 2 * pad, ww + 2 * pad), np.float32)
    a[pad:pad + wh, pad:pad + ww] = mark[..., 3] / 255.0
    shadow = cv2.GaussianBlur(a, (0, 0), max(1.0, pad / 2.0)) * SHADOW_OPACITY
    colour = np.zeros((wh + 2 * pad, ww + 2 * pad, 3), np.float32)
    colour[pad:pad + wh, pad:pad + ww] = mark[..., :3]
    m = int(round(MARGIN * short))
    x, y = w - m - ww - pad, h - m - wh - pad
    if x < 0 or y < 0:
        return None
    return x, y, colour, (a * OPACITY)[..., None], shadow[..., None]


def apply_watermark(img: np.ndarray) -> np.ndarray:
    """Draw the watermark on a BGR frame in place (and return it)."""
    h, w = img.shape[:2]
    prep = _prepared(w, h)
    if prep is None:
        return img
    x, y, colour, alpha, shadow = prep
    hh, ww = alpha.shape[:2]
    roi = img[y:y + hh, x:x + ww].astype(np.float32)
    roi = roi * (1 - shadow) + np.array(SHADOW, np.float32) * shadow
    roi = roi * (1 - alpha) + colour * alpha
    img[y:y + hh, x:x + ww] = np.clip(roi, 0, 255).astype(np.uint8)
    return img
