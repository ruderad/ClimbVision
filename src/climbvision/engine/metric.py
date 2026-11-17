"""Metric wall: from wall coordinates (pixels of the first frame's view) to metres.

Wall coordinates are the first registered frame's view of the wall plane. That view
is in perspective (the camera looks up at the wall, or the wall leans towards it), so
one "pixels per metre" for the whole wall is wrong: on the handheld clips the climber's
trunk looks 1.5-1.6x bigger high on the wall than low. This layer corrects that with
the climber's own body as the ruler, then sets the scale from the climber's height.

**Self-calibration from the body.** A plane seen in perspective becomes true to shape
(up to scale, rotation and an aspect ratio) after a projective correction (2
parameters: `a`, `b`). The body's segments (trunk, thighs, shanks, upper arms,
forearms) have the same length wherever they are on the wall; in a picture they can only
look shorter (foreshortening), never longer. So:

1. `a`, `b`: chosen so that each segment's upper length (85th percentile) is the same in
   every region of the wall the climber visited. Kept only if the body sizes become
   clearly more uniform (spread down by a quarter) *and* the trunk — the most rigid
   ruler — agrees.
2. Scale: the body length (trunk + leg, 90th percentile) in the corrected plane divided
   by the climber's body length in metres (0.779 x height; an average 1.75 m climber if
   the height isn't known).

The aspect ratio (vertical vs horizontal, `k`) is *not* learned from the body: climbing
postures bias it (arms and legs are fully stretched up and down far more often than
sideways — tried on the test clips, it came out 0.8-0.9 where perspective says > 1).
`k` stays 1 until a known length on the wall or depth (LiDAR) sets it.

A correction is kept only if it makes the body sizes clearly more uniform; otherwise the
plain scale is used (`corrected: false`).

Metres: x to the right, y **up**; heights above the floor use the ground anchors (feet
standing on the floor) when there are any.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from climbvision.engine.skeleton import J

# segment -> (end a, end b); trunk uses midpoints
SEGMENTS = {
    "trunk": (None, None),
    "l_thigh": (J.L_HIP, J.L_KNEE), "r_thigh": (J.R_HIP, J.R_KNEE),
    "l_shank": (J.L_KNEE, J.L_ANKLE), "r_shank": (J.R_KNEE, J.R_ANKLE),
    "l_upper_arm": (J.L_SHOULDER, J.L_ELBOW), "r_upper_arm": (J.R_SHOULDER, J.R_ELBOW),
    "l_forearm": (J.L_ELBOW, J.L_WRIST), "r_forearm": (J.R_ELBOW, J.R_WRIST),
}
TYPES = {"trunk": ["trunk"], "thigh": ["l_thigh", "r_thigh"], "shank": ["l_shank", "r_shank"],
         "upper_arm": ["l_upper_arm", "r_upper_arm"], "forearm": ["l_forearm", "r_forearm"]}


def _ends(p: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for name, (a, b) in SEGMENTS.items():
        if name == "trunk":
            out[name] = ((p[:, J.L_SHOULDER] + p[:, J.R_SHOULDER]) / 2,
                         (p[:, J.L_HIP] + p[:, J.R_HIP]) / 2)
        else:
            out[name] = (p[:, a], p[:, b])
    return out


def _project(x: np.ndarray, a: float, b: float, k: float = 1.0) -> np.ndarray:
    """Normalised wall coords -> corrected plane: projective (a, b), then aspect k."""
    w = 1.0 + a * x[..., 0] + b * x[..., 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        q = x / w[..., None]
    q = q.copy()
    q[..., 1] *= k
    return q


@dataclass
class WallMetric:
    """Wall px -> metres. c, s: normalisation (centre, size in px); a, b, k: correction;
    px_per_unit: corrected-plane units per metre."""

    c: np.ndarray
    s: float
    a: float = 0.0
    b: float = 0.0
    k: float = 1.0
    per_m: float = 1.0  # corrected-plane units per metre
    ground_y_m: float | None = None  # floor height (y up) in metres, if known
    info: dict = field(default_factory=dict)

    @property
    def corrected(self) -> bool:
        return bool(self.a or self.b or self.k != 1.0)

    def to_m(self, p: np.ndarray) -> np.ndarray:
        """(..., 2) wall px -> (..., 2) metres, x right, y up."""
        q = _project((np.asarray(p, float) - self.c) / self.s, self.a, self.b, self.k)
        out = q / self.per_m
        out[..., 1] *= -1
        return out

    def dist_m(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        return np.linalg.norm(self.to_m(p) - self.to_m(q), axis=-1)

    def height_m(self, p: np.ndarray) -> np.ndarray:
        """Height above the floor (if known), else the plain y-up coordinate."""
        y = self.to_m(p)[..., 1]
        return y - self.ground_y_m if self.ground_y_m is not None else y

    def summary(self) -> dict:
        return {"perspective_corrected": self.corrected, **self.info}


def _upper(v: np.ndarray, q: float = 85) -> float:
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if len(v) >= 15 else float("nan")


def _bins(pos: np.ndarray, n_bins: int) -> list[np.ndarray]:
    """Frames grouped by where the body was on the wall (quantiles of y, then x)."""
    ok = np.isfinite(pos).all(1)
    idx = np.flatnonzero(ok)
    if len(idx) < 60:
        return []
    ys = np.array_split(idx[np.argsort(pos[idx, 1])], n_bins)
    out = []
    for g in ys:
        xs = np.array_split(g[np.argsort(pos[g, 0])], 2)
        out += [x for x in xs if len(x) >= 20]
    return out


def _spread(ends_n, bins, a, b, k=1.0, types=None) -> float:
    """Mean over segment types of the std of log(upper length) across wall regions."""
    costs = []
    for t, names in TYPES.items():
        if types is not None and t not in types:
            continue
        lens = []
        for nm in names:
            pa, pb = ends_n[nm]
            lens.append(np.linalg.norm(_project(pa, a, b, k) - _project(pb, a, b, k), axis=-1))
        L = np.concatenate(lens) if len(lens) > 1 else lens[0]
        reps = len(names)
        per = [_upper(np.concatenate([L[g + r * (len(L) // reps)] for r in range(reps)]))
               for g in bins]
        per = np.array([v for v in per if np.isfinite(v) and v > 0])
        if len(per) >= 3:
            costs.append(float(np.std(np.log(per))))
    return float(np.mean(costs)) if costs else float("nan")


def fit_wall_metric(wall_points: np.ndarray, use: np.ndarray, body_m: float,
                    ground_px: np.ndarray | None = None, correct: bool = True,
                    px_per_m: float | None = None, min_spread: float = 0.07,
                    max_ab: float = 0.3) -> WallMetric:
    """wall_points: (n, N_JOINTS, 2) wall px; use: frames to learn from (on the wall);
    ground_px: (k, 2) wall px of feet standing on the floor (for heights above it);
    px_per_m: the plain scale, used as is when no correction is needed (body sizes
    already uniform: spread of log sizes < `min_spread`, or the fit doesn't help).
    `max_ab`: the largest correction tried (scale change per body length on the wall;
    the test clips need up to 0.2)."""
    from climbvision.engine.scale import chain_len

    p = np.asarray(wall_points, float)
    use = np.asarray(use, bool) & np.isfinite(p[:, J.L_HIP]).all(1)
    if use.sum() < 30:
        use = np.isfinite(p[:, J.L_HIP]).all(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        hip = (p[:, J.L_HIP] + p[:, J.R_HIP]) / 2
        c = np.nanmedian(hip[use], axis=0) if use.any() else np.zeros(2)
        body_px = float(np.nanpercentile(chain_len(p[use]), 90)) if use.any() else float("nan")
    if not np.isfinite(body_px) or body_px <= 0:
        return WallMetric(np.zeros(2), 1.0, per_m=1.0, info={"scale": "unknown"})
    s = body_px
    pn = (p - c) / s
    ends_n = {k: (a[use], b[use]) for k, (a, b) in _ends(pn).items()}
    a = b = 0.0
    k = 1.0
    info: dict = {}
    bins = _bins(hip[use], 4) if correct else []
    if len(bins) >= 4:
        base = _spread(ends_n, bins, 0.0, 0.0)
        best = (base, 0.0, 0.0)
        for span, steps in ((max_ab, 13), (0.05, 11), (0.01, 11)):
            ca, cb = best[1], best[2]
            for ta in np.linspace(ca - span, ca + span, steps):
                for tb in np.linspace(cb - span, cb + span, steps):
                    if abs(ta) > max_ab or abs(tb) > max_ab:
                        continue
                    cost = _spread(ends_n, bins, ta, tb)
                    if np.isfinite(cost) and cost < best[0]:
                        best = (cost, ta, tb)
        # every point of the climb must stay on the same side of the horizon
        w = 1 + best[1] * pn[use][..., 0] + best[2] * pn[use][..., 1]
        ok_horizon = np.nanmin(w) > 0.3
        # the trunk is the most rigid ruler (no elbows or knees to bend): it must agree
        trunk0 = _spread(ends_n, bins, 0.0, 0.0, types={"trunk"})
        trunk1 = _spread(ends_n, bins, best[1], best[2], types={"trunk"})
        trunk_ok = np.isfinite(trunk0) and np.isfinite(trunk1) and trunk1 < trunk0 - 0.02
        if (np.isfinite(base) and base >= min_spread and best[0] < 0.75 * base and ok_horizon
                and trunk_ok):
            a, b = best[1], best[2]
        info = {"body_size_spread_before": round(base, 3),
                "body_size_spread_after": round(_spread(ends_n, bins, a, b, k), 3)}
    # scale: body length in the corrected plane
    if (a or b) or not px_per_m:
        q = _project(pn, a, b, k)
        per_m = float(np.nanpercentile(chain_len(q[use]), 90)) / body_m
    else:
        per_m = px_per_m / s  # normalised units per metre
    m = WallMetric(c, s, a, b, k, per_m, None, info)
    if ground_px is not None and len(ground_px):
        m.ground_y_m = float(np.median(m.to_m(np.asarray(ground_px, float))[:, 1]))
    return m
