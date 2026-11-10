"""Clean-up layer: repair the pose where it is physically impossible.

Pose models are confident even when a limb is hidden. The typical failure on the wall is
a leg behind the body (a drop knee, a heel hook on the far side): for single frames or
short bursts the model puts the hidden knee or foot on whatever is visible — the other
leg, a hand — or swaps the two legs, and the limb flickers between the two places. This
layer finds those frames and repairs them with what we know about a body:

1. **Left/right swaps** of a whole leg (or arm): if exchanging the two limbs fits the
   previous frame much better than keeping them, they are exchanged back.
2. **Flicker**: each limb joint, relative to the limb's root (hip / shoulder), is compared
   with its running median over `window_s`; a frame further than `flicker_tol` from it
   is a glitch and takes the median (a limb doesn't jump away and back in 0.1 s).
3. **Impossible bones**, walking down the limb: a bone longer than `max_bone` x its
   75th-percentile length in the clip, or than the segment's share of the body length
   if that is longer (in a picture a bone can look shorter, never longer; the percentile
   stays put while up to a quarter of the clip is wrong), marks the joint at its far end
   as wrong. Repairs:
   - knee / elbow: rebuilt from hip and ankle (shoulder and wrist) with the limb's own
     bone lengths (two-bone inverse kinematics), bent to the same side as in the
     nearest good frame — this is what fixes a hidden drop knee;
   - ankle / wrist: interpolated over a gap up to `max_gap_s`;
   - heel, toes, hand: carried with the ankle / wrist, at their offset in the nearest
     good frame.
   Anything that can't be repaired is kept and flagged (visibility lowered), so it's drawn
   as uncertain and not trusted.
4. **Check**: a repair is kept only if it fits the limb's motion in the surrounding
   second (running median of the untouched frames) at least as well as the raw position
   did; otherwise the raw position is put back and flagged.

Works on positions in a stable frame (wall coordinates when the camera moves); the
caller maps the result back to the image. Nothing is invented where there's no
evidence: every repair is built from good frames of the same limb.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from climbvision.engine.events import runs
from climbvision.engine.skeleton import J

# limb chains: (root, middle, end, *extremities)
LEGS = {"l": (J.L_HIP, J.L_KNEE, J.L_ANKLE, J.L_HEEL, J.L_TOE, J.L_SMALL_TOE),
        "r": (J.R_HIP, J.R_KNEE, J.R_ANKLE, J.R_HEEL, J.R_TOE, J.R_SMALL_TOE)}
ARMS = {"l": (J.L_SHOULDER, J.L_ELBOW, J.L_WRIST, J.L_HAND),
        "r": (J.R_SHOULDER, J.R_ELBOW, J.R_WRIST, J.R_HAND)}
# segment lengths as a fraction of the body length (trunk + leg; Drillis & Contini):
# a bone's reference length is never below this, however foreshortened the clip is
MIN_REF = {J.L_KNEE: 0.31, J.R_KNEE: 0.31, J.L_ANKLE: 0.31, J.R_ANKLE: 0.31,
           J.L_ELBOW: 0.24, J.R_ELBOW: 0.24, J.L_WRIST: 0.20, J.R_WRIST: 0.20}


@dataclass
class CleanupParams:
    window_s: float = 0.6  # running median for flicker
    flicker_tol: float = 0.12  # body lengths away from the running median = glitch
    max_bone: float = 1.35  # x the bone's 75th-percentile length in the clip
    max_gap_s: float = 0.8  # longest gap an ankle / wrist is interpolated over
    check_s: float = 1.0  # window of the final check of every repair
    swap_ratio: float = 0.5  # swapped fits the previous frame this much better -> swap
    repaired_vis: float = 0.5  # visibility given to repaired joints
    bad_vis: float = 0.2  # ... and to joints found wrong but not repaired


@dataclass
class CleanupReport:
    swaps: dict = field(default_factory=dict)  # "legs"/"arms" -> frames swapped back
    flicker: dict = field(default_factory=dict)  # joint -> glitch frames replaced
    repaired: dict = field(default_factory=dict)  # joint -> impossible frames repaired
    left: dict = field(default_factory=dict)  # joint -> impossible frames not repaired
    reverted: dict = field(default_factory=dict)  # joint -> repairs that didn't fit, undone

    def as_dict(self) -> dict:
        return {"swaps": self.swaps, "flicker_frames": self.flicker,
                "repaired_frames": self.repaired, "not_repaired_frames": self.left,
                "reverted_frames": self.reverted}

    def total(self) -> int:
        return int(sum(self.swaps.values()) + sum(self.flicker.values())
                   + sum(self.repaired.values()))


def _bone(p, a, b):
    return np.linalg.norm(p[:, a] - p[:, b], axis=-1)


def _add(d: dict, j, k) -> None:
    if k:
        d[J(j).name.lower()] = d.get(J(j).name.lower(), 0) + int(k)


def fix_swaps(p: np.ndarray, chains: dict, prm: CleanupParams) -> tuple[np.ndarray, int]:
    """Exchange left and right limbs where that fits the previous frame much better."""
    p = p.copy()
    lj = list(chains["l"][1:])
    rj = list(chains["r"][1:])
    n_sw = 0
    for i in range(1, len(p)):
        a, b = p[i - 1][lj], p[i - 1][rj]
        c, d = p[i][lj], p[i][rj]
        ok = np.isfinite(a).all(1) & np.isfinite(b).all(1) & np.isfinite(c).all(1) & \
            np.isfinite(d).all(1)
        if ok.sum() < 2:
            continue
        same = np.linalg.norm(c[ok] - a[ok], axis=1).sum() + \
            np.linalg.norm(d[ok] - b[ok], axis=1).sum()
        swap = np.linalg.norm(c[ok] - b[ok], axis=1).sum() + \
            np.linalg.norm(d[ok] - a[ok], axis=1).sum()
        if swap < prm.swap_ratio * same:
            p[i, lj], p[i, rj] = d, c
            n_sw += 1
    return p, n_sw


def running_median(x: np.ndarray, half: int) -> np.ndarray:
    """NaN-aware centred running median of (n, d) along axis 0."""
    import warnings

    pad = np.full((half,) + x.shape[1:], np.nan)
    xp = np.concatenate([pad, x, pad])
    win = np.lib.stride_tricks.sliding_window_view(xp, 2 * half + 1, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(win, axis=-1)


def fix_flicker(p: np.ndarray, body: float, fps: float, prm: CleanupParams, chains: dict,
                report: CleanupReport) -> tuple[np.ndarray, np.ndarray]:
    p = p.copy()
    flag = np.zeros(p.shape[:2], bool)
    half = max(1, int(prm.window_s * fps / 2))
    for chain in chains.values():
        root = chain[0]
        for j in chain[1:]:
            q = p[:, j] - p[:, root]
            med = running_median(q, half)
            dev = np.linalg.norm(q - med, axis=1)
            f = np.nan_to_num(dev, nan=0) > prm.flicker_tol * body
            p[f, j] = p[f, root] + med[f]
            flag[f, j] = True
            _add(report.flicker, j, f.sum())
    return p, flag


def _side(p, i, root, mid, end) -> float:
    d = p[i, end] - p[i, root]
    m = p[i, mid] - p[i, root]
    return 1.0 if (d[0] * m[1] - d[1] * m[0]) >= 0 else -1.0


def _nearest(good: np.ndarray, i: int) -> int | None:
    idx = np.flatnonzero(good)
    if not len(idx):
        return None
    return int(idx[np.argmin(np.abs(idx - i))])


def _two_bone(root, end, l1, l2, side):
    """Middle joint of a two-bone chain (2D) with bone lengths l1, l2, bent to `side`
    (+1 / -1: which side of the root->end line)."""
    d = end - root
    dist = float(np.linalg.norm(d))
    if not np.isfinite(dist) or dist < 1e-6:
        return np.full(2, np.nan)
    u = d / dist
    if dist >= l1 + l2:  # out of reach: straight
        return root + u * l1
    a = (l1 ** 2 - l2 ** 2 + dist ** 2) / (2 * dist)
    h = np.sqrt(max(l1 ** 2 - a ** 2, 0.0))
    perp = np.array([-u[1], u[0]])
    return root + u * a + side * h * perp


def fix_bones(p: np.ndarray, body: float, fps: float, prm: CleanupParams, chains: dict,
              report: CleanupReport) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk down each limb; repair joints at the end of an impossibly long bone.
    Returns (positions, repaired mask, still-wrong mask)."""
    p = p.copy()
    n = len(p)
    fixed = np.zeros(p.shape[:2], bool)
    wrong = np.zeros(p.shape[:2], bool)
    gap = max(1, int(prm.max_gap_s * fps))

    def too_long(a, b):
        length = _bone(p, a, b)
        if not np.isfinite(length).any():
            return np.zeros(n, bool), np.nan
        ref = max(float(np.nanpercentile(length, 75)), MIN_REF.get(b, 0.0) * body)
        return np.nan_to_num(length, nan=0) > prm.max_bone * ref, ref

    for chain in chains.values():
        root, mid, end = chain[:3]
        lim1, ref1 = too_long(root, mid)
        lim2, ref2 = too_long(mid, end)
        # which joint is wrong: if root->end is out of reach of both bones, the end is
        reach = _bone(p, root, end) > prm.max_bone * (ref1 + ref2)
        bad_end = reach.copy()
        bad_mid = (lim1 | lim2) & ~bad_end
        # end (ankle / wrist): interpolate short gaps
        for a, b in runs(bad_end):
            if a > 0 and b < n and b - a <= gap and np.isfinite(p[[a - 1, b], end]).all():
                w = (np.arange(a, b) - (a - 1)) / (b - (a - 1))
                p[a:b, end] = p[a - 1, end] + w[:, None] * (p[b, end] - p[a - 1, end])
                fixed[a:b, end] = True
                _add(report.repaired, end, b - a)
                bad_mid[a:b] = True  # the knee goes with it
            else:
                wrong[a:b, end] = True
                _add(report.left, end, b - a)
        # middle (knee / elbow): two-bone IK
        good_mid = ~bad_mid & ~wrong[:, end] & np.isfinite(p[:, [root, mid, end]]).all((1, 2))
        l1_all, l2_all = _bone(p, root, mid), _bone(p, mid, end)
        for i in np.flatnonzero(bad_mid):
            k = _nearest(good_mid, i)
            if k is None or wrong[i, end] or not np.isfinite(p[i, [root, end]]).all():
                wrong[i, mid] = True
                _add(report.left, mid, 1)
                continue
            lo, hi = max(0, k - 15), min(n, k + 16)
            g = good_mid[lo:hi]
            l1 = float(np.nanmax(l1_all[lo:hi][g]))  # foreshortening only shortens
            l2 = float(np.nanmax(l2_all[lo:hi][g]))
            p[i, mid] = _two_bone(p[i, root], p[i, end], l1, l2, _side(p, k, root, mid, end))
            fixed[i, mid] = True
            _add(report.repaired, mid, 1)
        # extremities (heel, toes, hand): carried with the end
        for e in chain[3:]:
            lim, _ = too_long(end, e)
            good = ~lim & np.isfinite(p[:, [end, e]]).all((1, 2))
            for i in np.flatnonzero(lim):
                k = _nearest(good, i)
                if k is None or wrong[i, end]:
                    wrong[i, e] = True
                    _add(report.left, e, 1)
                    continue
                p[i, e] = p[i, end] + (p[k, e] - p[k, end])
                fixed[i, e] = True
                _add(report.repaired, e, 1)
    return p, fixed, wrong


def clean_points(p: np.ndarray, t: np.ndarray, body: float,
                 prm: CleanupParams | None = None, arms: bool = True
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, CleanupReport]:
    """Clean (n, N_JOINTS, 2) positions in a stable frame.
    Returns (cleaned, repaired mask, still-wrong mask, report)."""
    prm = prm or CleanupParams()
    report = CleanupReport()
    fps = 1.0 / float(np.median(np.diff(t))) if len(t) > 1 else 30.0
    out = np.asarray(p, float).copy()
    fixed = np.zeros(out.shape[:2], bool)
    wrong = np.zeros(out.shape[:2], bool)
    groups = {"legs": LEGS, **({"arms": ARMS} if arms else {})}
    for name, chains in groups.items():
        before = out.copy()
        out, n_sw = fix_swaps(out, chains, prm)
        report.swaps[name] = n_sw
        fixed |= ~np.isclose(np.nan_to_num(out, nan=-1e9),
                             np.nan_to_num(before, nan=-1e9)).all(-1)
        out, f = fix_flicker(out, body, fps, prm, chains, report)
        fixed |= f
        out, f, w = fix_bones(out, body, fps, prm, chains, report)
        fixed |= f
        wrong |= w
        out = _check(out, np.asarray(p, float), fixed, wrong, body, fps, prm, chains, report)
    return out, fixed & ~wrong, wrong, report


def _check(out, raw, fixed, wrong, body, fps, prm, chains, report):
    """Keep a repair only if it fits the limb's motion around it (the running median of
    the frames that weren't touched) at least as well as the raw position did;
    otherwise put the raw position back and flag the joint as wrong."""
    half = max(1, int(prm.check_s * fps / 2))
    for chain in chains.values():
        root = chain[0]
        for j in chain[1:]:
            rel = out[:, j] - out[:, root]
            ctx_in = np.where(fixed[:, [j]], np.nan, rel)
            ctx = running_median(ctx_in, half)
            idx = np.flatnonzero(fixed[:, j] & np.isfinite(ctx).all(1))
            if not len(idx):
                continue
            d_new = np.linalg.norm(rel[idx] - ctx[idx], axis=1)
            d_raw = np.linalg.norm(raw[idx, j] - out[idx, root] - ctx[idx], axis=1)
            worse = idx[np.nan_to_num(d_new, nan=np.inf)
                        > np.nan_to_num(d_raw, nan=np.inf) + prm.flicker_tol * body / 2]
            out[worse, j] = raw[worse, j]
            fixed[worse, j] = False
            wrong[worse, j] = True
            _add(report.reverted, j, len(worse))
    return out


def clean_track(track, camera=None, prm: CleanupParams | None = None):
    """Clean a Track's image positions. With a camera layer the work is done in wall
    coordinates (camera motion removed) and mapped back to the image. Returns
    (cleaned Track, report dict)."""
    from dataclasses import replace

    from climbvision.engine.camera import LOST, apply_h
    from climbvision.engine.scale import chain_len

    prm = prm or CleanupParams()
    img = track.image
    if camera is not None:
        pts = camera.to_wall(img)
    else:
        pts = img.copy()
    v = chain_len(track.image)
    if not np.isfinite(v).any() or track.n < 3:
        return track, {"frames_changed_pct": 0.0, "limb_points_changed_pct": 0.0}
    body = float(np.nanpercentile(v, 90))
    if camera is not None:  # body length in wall px
        w = chain_len(pts)
        if np.isfinite(w).any():
            body = float(np.nanpercentile(w, 90))
    out, fixed, wrong, rep = clean_points(pts, track.time_s, body, prm)
    new_img = img.copy()
    if camera is not None:
        for i in np.flatnonzero((fixed | wrong).any(axis=1)):
            if i < camera.n and camera.status[i] != LOST:
                back = apply_h(np.linalg.inv(camera.H[i]), out[i])
                new_img[i, fixed[i]] = back[fixed[i]]
    else:
        new_img[fixed] = out[fixed]
    vis = track.visibility.copy()
    vis[fixed] = np.minimum(vis[fixed], prm.repaired_vis)
    vis[wrong] = np.minimum(vis[wrong], prm.bad_vis)
    d = rep.as_dict()
    limb = sorted({j for c in (*LEGS.values(), *ARMS.values()) for j in c[1:]})
    d["frames_changed_pct"] = round(100 * float((fixed | wrong).any(axis=1).mean()), 1)
    d["limb_points_changed_pct"] = round(100 * float((fixed | wrong)[:, limb].mean()), 1)
    return replace(track, image=new_img, visibility=vis), d
