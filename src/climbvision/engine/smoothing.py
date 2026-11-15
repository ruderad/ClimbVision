"""Smoothing layer: steady joints without lag, and hidden joints trusted less.

Pose models estimate every frame on its own, so a joint that isn't moving still shakes
by a few pixels from frame to frame, and a joint the model can't see (a leg behind the
body) is placed with low confidence, often wrongly. This layer fits each joint's path
with a **robust, confidence-weighted local quadratic** (LOESS, Cleveland 1979) — for
every frame, a parabola through the joint's positions in the surrounding `half_s`
seconds:

- **weights**: a tricube kernel in time (near frames count most) × the joint's
  confidence (visibility; repaired joints from the clean-up layer count little) ×
  a robustness weight;
- **robust**: a position far from the running median of its 5 frames (`outlier_k` ×
  the joint's typical distance) is a spike — a wrong frame, not motion: a median
  follows a fast smooth move exactly but ignores a 1–2 frame jump — and gets weight 0,
  so it doesn't drag its neighbours (it is replaced by what its neighbours say);
- **no lag**: the fit is centred, and a parabola follows accelerations (a hand
  starting or stopping a move), unlike a moving average which rounds them off; while a
  joint moves fast the window shortens (`half_fast_s`) so a quick move keeps its speed;
- **gaps**: a hidden joint (visibility < `min_visibility`: a guess) or a spike is
  ignored, and the stretch of ignored frames is filled from the good frames on both
  sides — a smooth curve for a short gap, a straight line for a longer one, up to
  `max_gap_s`; a longer gap keeps the model's guess (drawn grey: low visibility). The
  filled joints keep their low visibility, so they are drawn grey: inferred, not seen.

Runs in a stable frame (wall coordinates when the camera moves), after the clean-up
layer, so camera shake isn't mistaken for motion and isn't smoothed away from the image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SmoothParams:
    half_s: float = 0.12  # half window of the local fit (s) for a still or slow joint
    half_fast_s: float = 0.07  # ... for a joint moving faster than fast_body_s
    fast_body_s: float = 1.0  # body lengths / s: speed where the short window takes over
    min_visibility: float = 0.4  # below this a position is a guess and is ignored
    #                              (RTMPose score < 2.2: 1 in 5 of those are > 30 px off)
    outlier_k: float = 5.0  # spike: this many x the joint's typical distance from its
    #                         5-frame running median
    min_resid: float = 0.004  # floor of that typical distance, body lengths
    max_gap_s: float = 1.0  # longest stretch of ignored frames filled from both sides
    visibility_power: float = 2.0  # weight = visibility ** power


def _corr(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Centred correlation along axis 0 (zero padding): out[i] = sum_j k[j] x[i+j-h]."""
    h = len(k) // 2
    n = x.shape[0]
    xp = np.concatenate([np.zeros((h,) + x.shape[1:]), x, np.zeros((h,) + x.shape[1:])])
    out = np.zeros_like(x, dtype=float)
    for j, kj in enumerate(k):
        if kj != 0.0:
            out += kj * xp[j:j + n]
    return out


def _running_median(x: np.ndarray, half: int) -> np.ndarray:
    n = len(x)
    pad = np.concatenate([np.full((half,) + x.shape[1:], np.nan), x,
                          np.full((half,) + x.shape[1:], np.nan)])
    win = np.stack([pad[k:k + n] for k in range(2 * half + 1)])
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(win, axis=0)


def local_quadratic(x: np.ndarray, w: np.ndarray, half: int) -> tuple[np.ndarray,
                                                                      np.ndarray]:
    """Weighted local quadratic fit at every sample. x: (n, d) values (NaN allowed where
    w = 0), w: (n,) weights. Returns (fit (n, d), weight support (n,))."""
    tau = np.arange(-half, half + 1, dtype=float)
    kern = (1 - np.abs(tau / (half + 1)) ** 3) ** 3  # tricube
    x0 = np.where(np.isfinite(x), x, 0.0)
    w = np.where(np.isfinite(x).all(axis=1), w, 0.0)
    # sums S_m = sum k tau^m w, T_m = sum k tau^m w x  (tau measured from the centre)
    S = [_corr(w, kern * tau ** m) for m in range(5)]
    T = [_corr(w[:, None] * x0, kern * tau ** m) for m in range(3)]
    A = np.stack([np.stack([S[0], S[1], S[2]], -1),
                  np.stack([S[1], S[2], S[3]], -1),
                  np.stack([S[2], S[3], S[4]], -1)], -2)  # (n, 3, 3)
    B = np.stack(T, -2)  # (n, 3, d)
    fit = np.full(x.shape, np.nan)
    det = np.linalg.det(A)
    ok = np.abs(det) > 1e-9 * np.maximum(S[0], 1e-12) ** 3
    if ok.any():
        fit[ok] = np.linalg.solve(A[ok], B[ok])[:, 0]
    # fall back to a local linear / constant fit where the quadratic is ill-posed
    lin = ~ok & (S[0] * S[2] - S[1] ** 2 > 1e-9 * np.maximum(S[0], 1e-12) ** 2)
    if lin.any():
        d = S[0][lin] * S[2][lin] - S[1][lin] ** 2
        fit[lin] = ((S[2][lin] * T[0][lin].T - S[1][lin] * T[1][lin].T) / d).T
    const = ~ok & ~lin & (S[0] > 0)
    if const.any():
        fit[const] = T[0][const] / S[0][const, None]
    return fit, S[0] / kern.sum()


def _fill_gaps(res: np.ndarray, use: np.ndarray, fit: np.ndarray, max_gap: int,
               fps: float) -> int:
    """Fill runs of unusable frames (in place) between two usable ones, up to max_gap
    frames: a cubic (Hermite) from the end positions and speeds of the fit for a short
    gap (<= 0.3 s), a straight line for a longer one (a curve would overshoot)."""
    n = len(use)
    filled = 0
    idx = np.flatnonzero(use)
    for a, b in zip(idx[:-1], idx[1:], strict=True):
        g = b - a - 1
        if g <= 0 or g > max_gap:
            continue
        s = np.arange(1, g + 1) / (g + 1)
        p0, p1 = res[a], res[b]
        if g + 1 <= 0.3 * fps:
            v0 = (fit[a] - fit[max(a - 1, 0)]) * (g + 1)
            v1 = (fit[min(b + 1, n - 1)] - fit[b]) * (g + 1)
            span = np.linalg.norm(p1 - p0) + 1e-9
            for v in (v0, v1):  # no wild speeds out of a noisy edge
                m = np.linalg.norm(v)
                if not np.isfinite(m):
                    v[:] = p1 - p0
                elif m > 3 * span:
                    v *= 3 * span / m
            h00, h10 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s
            h01, h11 = -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
            seg = (h00[:, None] * p0 + h10[:, None] * v0 + h01[:, None] * p1
                   + h11[:, None] * v1)
        else:
            seg = p0 + s[:, None] * (p1 - p0)
        res[a + 1:b] = seg
        filled += g
    return filled


def smooth_points(p: np.ndarray, t: np.ndarray, conf: np.ndarray, body: float,
                  prm: SmoothParams | None = None) -> tuple[np.ndarray, dict]:
    """Smooth (n, J, 2) positions with (n, J) confidences. Returns (smoothed, report)."""
    prm = prm or SmoothParams()
    p = np.asarray(p, float)
    n, nj = p.shape[:2]
    if n < 5:
        return p.copy(), {}
    fps = 1.0 / float(np.median(np.diff(t)))
    half = max(2, int(round(prm.half_s * fps)))
    half_fast = max(1, int(round(prm.half_fast_s * fps)))
    max_gap = int(round(prm.max_gap_s * fps))
    out = np.full_like(p, np.nan)
    rejected = filled = used = 0
    for j in range(nj):
        x = p[:, j]
        have = np.isfinite(x).all(axis=1)
        if have.sum() < 5:
            out[:, j] = x
            continue
        w0 = np.clip(np.nan_to_num(conf[:, j]), 0, 1) ** prm.visibility_power
        w0 = np.where(have & (np.nan_to_num(conf[:, j]) >= prm.min_visibility), w0, 0.0)
        # spikes: far from the 5-frame running median of the joint's own path (a
        # median follows a fast, smooth move exactly; it ignores a 1-2 frame jump)
        med = _running_median(np.where(have[:, None], x, np.nan), 2)
        r = np.linalg.norm(x - med, axis=1)
        good = (w0 > 0) & np.isfinite(r)
        s = max(float(np.median(r[good])) * 1.4826 if good.any() else 0.0,
                prm.min_resid * body)
        rob = np.where(np.isfinite(r) & (r < prm.outlier_k * s), 1.0, 0.0)
        fit, _ = local_quadratic(x, w0 * rob, half)
        if half_fast < half:  # moving fast: the short window, which follows accelerations
            fast, _ = local_quadratic(x, w0 * rob, half_fast)
            v = np.linalg.norm(np.gradient(fast, axis=0), axis=1) * fps / body
            a = np.clip(np.nan_to_num(v) / prm.fast_body_s, 0, 1)[:, None]
            fit = np.where(np.isfinite(fast), a * fast + (1 - a) * fit, fit)
        use = ((w0 * rob) > 0) & np.isfinite(fit).all(axis=1)
        res = np.where(use[:, None], fit, np.nan)
        filled += int(_fill_gaps(res, use, fit, max_gap, fps))
        keep = ~np.isfinite(res).all(axis=1) & have  # gap too long: the model's own guess
        res[keep] = x[keep]
        rejected += int((have & (w0 > 0) & (rob == 0)).sum())
        used += int(have.sum())
        out[:, j] = res
    rep = {"outliers_replaced_pct": round(100 * rejected / max(used, 1), 2),
           "gaps_filled": filled, "window_s": round(2 * half / fps, 2)}
    return out, rep


def smooth_track(track, camera=None, prm: SmoothParams | None = None):
    """Smooth a Track's image positions (in wall coordinates when there's a camera
    layer, mapped back to each frame). Returns (smoothed Track, report)."""
    from dataclasses import replace

    from climbvision.engine.camera import LOST, apply_h
    from climbvision.engine.scale import chain_len

    img = track.image
    pts = camera.to_wall(img) if camera is not None else img.copy()
    v = chain_len(pts)
    if not np.isfinite(v).any() or track.n < 5:
        return track, {}
    body = float(np.nanpercentile(v, 90))
    out, rep = smooth_points(pts, track.time_s, track.visibility, body, prm)
    if camera is None:
        new = out
    else:
        new = img.copy()
        for i in range(min(track.n, camera.n)):
            if camera.status[i] != LOST and np.isfinite(out[i]).any():
                new[i] = apply_h(np.linalg.inv(camera.H[i]), out[i])
    new = np.where(np.isfinite(new), new, img)  # never lose a joint the model gave
    return replace(track, image=new), rep
