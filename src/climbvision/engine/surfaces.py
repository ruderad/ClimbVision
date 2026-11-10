"""Surfaces layer: what the body can rest on.

Today: the floor, as a line in the image fitted to the lowest well-supported points
(RANSAC; "lowest" so resting hands at hip height don't win). Later: the wall plane
and holds (E2/E4), behind the same idea — a surface tells contacts how far a joint is
from it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from climbvision.engine.scale import Scale
from climbvision.engine.skeleton import J
from climbvision.engine.track import Track

# body part -> joints that can rest on the floor
FLOOR_PARTS: dict[str, tuple[J, ...]] = {
    "left_foot": (J.L_HEEL, J.L_TOE),
    "right_foot": (J.R_HEEL, J.R_TOE),
    "left_hand": (J.L_WRIST, J.L_HAND),
    "right_hand": (J.R_WRIST, J.R_HAND),
    "left_forearm": (J.L_ELBOW,),
    "right_forearm": (J.R_ELBOW,),
    "left_knee": (J.L_KNEE,),
    "right_knee": (J.R_KNEE,),
}


@dataclass
class Floor:
    """Floor height in the image: y = slope * x + y0[frame].

    y0 = a + depth_m * (ppm - ppm_ref). For a level camera at height H the floor
    under someone at distance d appears H * (f / d) pixels below the horizon, and
    f / d is the pixels-per-metre scale, so depth_m ~ the camera's height (m).
    This keeps the floor right when the person walks towards or away from the camera.
    """

    slope: float
    a: float
    depth_m: float
    ppm: np.ndarray
    ppm_ref: float
    inlier_frac: float

    @property
    def known(self) -> bool:
        return bool(np.isfinite(self.a))

    @property
    def y0(self) -> np.ndarray:
        return self.a + self.depth_m * np.nan_to_num(self.ppm - self.ppm_ref)

    def y(self, x: np.ndarray, i: int) -> np.ndarray:
        return self.slope * np.asarray(x) + self.y0[i]

    def height(self, pts: np.ndarray) -> np.ndarray:
        """Pixels above the floor for (n, ..., 2) points, frame axis first."""
        y0 = self.y0.reshape((-1,) + (1,) * (pts.ndim - 2))
        return self.slope * pts[..., 0] + y0 - pts[..., 1]

    @property
    def angle_deg(self) -> float:
        return float(np.degrees(np.arctan(self.slope)))


def unknown_floor(scale: Scale) -> Floor:
    """No floor visible/learnable: every height is NaN, so nothing touches it."""
    return Floor(0.0, float("nan"), 0.0, scale.ppm, scale.px_per_m, 0.0)


def estimate_floor(track: Track, scale: Scale, max_tilt_deg: float = 10.0,
                   max_cam_height_m: float = 3.0, n_samples: int = 900,
                   seed: int = 0, exclude: np.ndarray | None = None) -> Floor:
    """Fit the floor line. Frames in `exclude` (e.g. hanging from a bar) are ignored;
    with less than 1 s of usable frames the floor is unknown."""
    body_px = scale.body_px
    use = track.detected.copy() if exclude is None else track.detected & ~exclude
    if use.sum() < max(3, track.fps):
        return unknown_floor(scale)
    q_all = np.nan_to_num(scale.ppm - scale.px_per_m)
    rel_all = np.nan_to_num(scale.rel, nan=1.0)
    # candidate points: the lowest point of each body part in each frame
    xs, ys, qs, rs = [], [], [], []
    for joints in FLOOR_PARTS.values():
        pts = track.image[:, list(joints)]  # (n, k, 2)
        valid = np.isfinite(pts[..., 1]).all(axis=1) & use
        if not valid.any():
            continue
        j = np.argmax(np.where(np.isfinite(pts[..., 1]), pts[..., 1], -np.inf), axis=1)
        low = pts[np.arange(len(pts)), j]
        xs.append(low[valid, 0])
        ys.append(low[valid, 1])
        qs.append(q_all[valid])
        rs.append(rel_all[valid])
    if not ys:
        return unknown_floor(scale)
    X, Y, Q, R = (np.concatenate(v) for v in (xs, ys, qs, rs))
    tol = 0.03 * body_px * R  # tolerance shrinks for a person further away
    rng = np.random.default_rng(seed)
    max_slope = np.tan(np.radians(max_tilt_deg))
    q_span = 0.08 * scale.px_per_m

    models = []  # (slope, a, depth_m)
    for i in rng.integers(0, len(Y), n_samples // 3):
        models.append((0.0, Y[i], 0.0))
    for i, j in rng.integers(0, len(Y), (n_samples // 3, 2)):  # tilted floor
        dx = X[j] - X[i]
        if abs(dx) >= 0.2 * body_px:
            m = (Y[j] - Y[i]) / dx
            if abs(m) <= max_slope:
                models.append((m, Y[i] - m * X[i], 0.0))
    for i, j in rng.integers(0, len(Y), (n_samples // 3, 2)):  # person changes distance
        dq = Q[j] - Q[i]
        if abs(dq) >= q_span:
            d = (Y[j] - Y[i]) / dq
            if 0 <= d <= max_cam_height_m:
                models.append((0.0, Y[i] - d * Q[i], d))

    def resid(m):
        return Y - (m[0] * X + m[1] + m[2] * Q)  # > 0 = below the floor in the image

    # score = points on the floor minus a heavy penalty for points clearly *below* it
    # (nothing rests below the floor), so knees/hands at rest can't win over feet
    scores = [np.sum(np.abs(r) < tol) - 3 * np.sum(r > 2 * tol)
              for r in (resid(m) for m in models)]
    best = models[int(np.argmax(scores))]

    # refine on inliers with whichever terms the data can support
    inl = np.abs(resid(best)) < tol
    cols, names = [np.ones(inl.sum())], ["a"]
    if np.ptp(X[inl]) > 0.3 * body_px:
        cols.append(X[inl])
        names.append("slope")
    if np.ptp(Q[inl]) > 2 * q_span:
        cols.append(Q[inl])
        names.append("depth")
    coef = np.linalg.lstsq(np.column_stack(cols), Y[inl], rcond=None)[0]
    c = dict(zip(names, coef, strict=True))
    model = (float(np.clip(c.get("slope", 0.0), -max_slope, max_slope)), float(c["a"]),
             float(np.clip(c.get("depth", 0.0), 0.0, max_cam_height_m)))
    inl = np.abs(resid(model)) < tol
    return Floor(model[0], model[1], model[2], scale.ppm, scale.px_per_m, float(inl.mean()))
