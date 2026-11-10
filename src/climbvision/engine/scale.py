"""Scale layer: body size in pixels and pixels per metre, per frame.

Each body segment gives image length / true (3D) length. A segment pointing at the
camera looks short, but none looks *longer* than it is, so the upper end of the
per-segment ratios (80th percentile) is the scale; smoothed over ~1 s.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from climbvision.engine.skeleton import SIDES, J, side_joint
from climbvision.engine.track import Track

_SEGMENTS = [(J.L_SHOULDER, J.L_ELBOW), (J.L_ELBOW, J.L_WRIST),
             (J.R_SHOULDER, J.R_ELBOW), (J.R_ELBOW, J.R_WRIST),
             (J.L_HIP, J.L_KNEE), (J.L_KNEE, J.L_ANKLE),
             (J.R_HIP, J.R_KNEE), (J.R_KNEE, J.R_ANKLE),
             (J.L_SHOULDER, J.R_SHOULDER), (J.L_HIP, J.R_HIP),
             (J.L_SHOULDER, J.L_HIP), (J.R_SHOULDER, J.R_HIP)]


def rolling_median(x: np.ndarray, win: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan)
    h = win // 2
    for i in range(n):
        seg = x[max(0, i - h): i + h + 1]
        if np.isfinite(seg).any():
            out[i] = np.nanmedian(seg)
    ok = np.isfinite(out)
    if ok.any() and not ok.all():
        out = np.interp(np.arange(n), np.flatnonzero(ok), out[ok])
    return out


def per_frame_px_per_m(track: Track) -> np.ndarray:
    r = []
    for a, b in _SEGMENTS:
        l2 = np.linalg.norm(track.image[:, a] - track.image[:, b], axis=-1)
        l3 = np.linalg.norm(track.world[:, a] - track.world[:, b], axis=-1)
        r.append(np.where(l3 > 0.08, l2 / np.where(l3 > 0.08, l3, 1.0), np.nan))
    r = np.stack(r, axis=1)
    with np.errstate(all="ignore"):
        valid = np.isfinite(r).sum(axis=1) >= 3
        raw = np.full(track.n, np.nan)
        raw[valid] = np.nanpercentile(r[valid], 80, axis=1)
    return rolling_median(raw, max(3, int(track.fps)))


def chain_len(pts: np.ndarray) -> np.ndarray:
    """Trunk + longer leg (hip-knee-ankle) per frame, for (n, N_JOINTS, D) points."""
    mid_s = (pts[:, J.L_SHOULDER] + pts[:, J.R_SHOULDER]) / 2
    mid_h = (pts[:, J.L_HIP] + pts[:, J.R_HIP]) / 2
    trunk = np.linalg.norm(mid_s - mid_h, axis=-1)
    legs = []
    for s in SIDES:
        h, k, a = (pts[:, side_joint(s, n)] for n in ("HIP", "KNEE", "ANKLE"))
        legs.append(np.linalg.norm(k - h, axis=-1) + np.linalg.norm(a - k, axis=-1))
    return trunk + np.fmax(legs[0], legs[1])


@dataclass
class Scale:
    body_px: float  # trunk + leg length in pixels at the median distance (~0.55 stature)
    px_per_m: float  # clip median
    ppm: np.ndarray  # (n,) per-frame pixels per metre

    @property
    def rel(self) -> np.ndarray:
        """Per-frame size relative to the clip median (> 1 = nearer the camera)."""
        return self.ppm / self.px_per_m

    def ppm_at(self, i: int) -> float:
        v = self.ppm[i]
        return float(v) if np.isfinite(v) else self.px_per_m

    def constant(self) -> Scale:
        """One scale for the whole clip (no evidence the person changed distance)."""
        return Scale(self.body_px, self.px_per_m, np.full(len(self.ppm), self.px_per_m))


ASSUMED_BODY_M = 0.93  # trunk + leg of a 1.75 m adult (~0.53 x stature)


def estimate_scale(track: Track, body_m: float = ASSUMED_BODY_M) -> Scale:
    """From 2D/3D limb ratios. Backends without 3D (2D-only): body length in pixels
    from the image (90th percentile of trunk + leg: foreshortening only shortens) and
    an assumed body length in metres (`body_m`)."""
    ppm = per_frame_px_per_m(track)
    if not np.isfinite(ppm).any():
        v = chain_len(track.image)
        if not np.isfinite(v).any():
            return Scale(float("nan"), float("nan"), ppm)
        body_px = float(np.nanpercentile(v, 90))
        ref = body_px / body_m
        return Scale(body_px, ref, np.full(track.n, ref))
    ref = float(np.nanmedian(ppm))
    body_m = float(np.nanmedian(chain_len(track.world)))
    return Scale(ref * body_m, ref, ppm)
