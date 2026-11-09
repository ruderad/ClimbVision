"""Exercise-specific metrics: squat reps and plank form.

Squat
  Reps are found on the knee angle with hysteresis: a rep starts when the knee
  bends more than `start_drop` degrees below the standing angle and ends when it
  straightens back to within `end_drop` degrees. Standing angle = 90th percentile.
  Depth = thigh angle vs horizontal at the bottom (below parallel when < 0); the
  3D hip-minus-knee height in cm is reported alongside.

Plank
  The hold is the longest stretch where the torso is roughly horizontal.
  Hip offset = signed distance of the hip from the shoulder-ankle line, as a % of
  that line's length. Positive = hips sag (below the line), negative = pike.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np

from climbvision.angles import Angles, Side
from climbvision.skeleton import LM
from climbvision.track import Track, smooth


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of consecutive True values."""
    m = np.concatenate([[False], mask.astype(bool), [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True))


# ---------------------------------------------------------------- cycles


def find_cycles(signal: np.ndarray, t: np.ndarray, start_drop: float, end_drop: float,
                min_s: float = 0.6, top: float | None = None) -> list[tuple[int, int, int]]:
    """Dips in `signal` below its resting level: [(start, bottom, end) indices].

    Resting level = 90th percentile (or `top`). A cycle starts when the signal falls
    more than `start_drop` below it (walking back to where the fall began) and ends
    when it comes back within `end_drop`. Cycles shorter than `min_s` are dropped.
    """
    top = float(np.nanpercentile(signal, 90)) if top is None else top
    out = []
    in_cycle, start = False, 0
    for i, v in enumerate(signal):
        if not np.isfinite(v):
            continue
        if not in_cycle and v < top - start_drop:
            in_cycle, start = True, i
            while start > 0 and np.isfinite(signal[start - 1]) and \
                    signal[start - 1] < top - end_drop:
                start -= 1
        elif in_cycle and v > top - end_drop:
            in_cycle = False
            if t[i] - t[start] >= min_s:
                b = start + int(np.nanargmin(signal[start:i + 1]))
                out.append((start, b, i))
    return out


def _stats(items: list, keys: tuple[str, ...]) -> dict:
    out = {}
    for key in keys:
        vals = np.array([getattr(r, key) for r in items], dtype=float)
        out[f"{key}_mean"] = round(float(np.nanmean(vals)), 2)
        out[f"{key}_sd"] = round(float(np.nanstd(vals)), 2)
    return out


# ---------------------------------------------------------------- squat


@dataclass
class SquatRep:
    rep: int
    leg: str  # "both", or the working leg for single-leg squats
    start_s: float
    bottom_s: float
    end_s: float
    descent_s: float
    ascent_s: float
    min_knee_deg: float
    min_hip_deg: float
    max_trunk_lean_deg: float
    min_thigh_deg: float  # thigh vs horizontal at the bottom; < 0 = hip below knee
    hip_below_knee_cm: float  # 3D estimate at the bottom; positive = hip below knee
    below_parallel: bool  # min_thigh_deg < 0


def squat_reps(track: Track, ang: Angles, single_leg: bool = False, start_drop: float = 35.0,
               end_drop: float = 15.0, min_rep_s: float = 0.6) -> tuple[list[SquatRep], dict]:
    """Squat reps. `single_leg` (pistol squats) follows whichever leg bends more."""
    fps = track.fps
    win = max(1, int(fps / 6))
    t = track.time_s

    def sm(x):
        return smooth(x, win)

    if single_leg:
        kl, kr = sm(ang.primary("knee", "left")), sm(ang.primary("knee", "right"))
        use_left = np.nan_to_num(kl, nan=999) <= np.nan_to_num(kr, nan=999)
        pick = {j: np.where(use_left, sm(ang.primary(j, "left")), sm(ang.primary(j, "right")))
                for j in ("knee", "hip", "thigh")}
    else:
        pick = {j: sm(ang.primary(j)) for j in ("knee", "hip", "thigh")}
        use_left = None
    knee, hip_ang, thigh = pick["knee"], pick["hip"], pick["thigh"]
    lean = sm(ang.primary("trunk_lean"))

    # hip below knee in world metres (y points down)
    w = track.world
    if single_leg:
        hip_y = np.where(use_left, w[:, LM.LEFT_HIP, 1], w[:, LM.RIGHT_HIP, 1])
        knee_y = np.where(use_left, w[:, LM.LEFT_KNEE, 1], w[:, LM.RIGHT_KNEE, 1])
    else:
        hip_y = (w[:, LM.LEFT_HIP, 1] + w[:, LM.RIGHT_HIP, 1]) / 2
        knee_y = (w[:, LM.LEFT_KNEE, 1] + w[:, LM.RIGHT_KNEE, 1]) / 2
    depth_cm = sm((hip_y - knee_y) * 100)

    top = float(np.nanpercentile(knee, 90))
    reps: list[SquatRep] = []
    for start, b, end in find_cycles(knee, t, start_drop, end_drop, min_rep_s, top):
        seg = slice(start, end + 1)
        if use_left is None:
            leg = "both"
        else:  # the leg that is more bent over the whole rep, not just one frame
            leg = "left" if np.nanmedian(kl[seg]) <= np.nanmedian(kr[seg]) else "right"
        reps.append(SquatRep(
            rep=len(reps) + 1,
            leg=leg,
            start_s=round(float(t[start]), 2),
            bottom_s=round(float(t[b]), 2),
            end_s=round(float(t[end]), 2),
            descent_s=round(float(t[b] - t[start]), 2),
            ascent_s=round(float(t[end] - t[b]), 2),
            min_knee_deg=round(float(knee[b]), 1),
            min_hip_deg=round(float(np.nanmin(hip_ang[seg])), 1),
            max_trunk_lean_deg=round(float(np.nanmax(lean[seg])), 1),
            min_thigh_deg=round(float(np.nanmin(thigh[seg])), 1),
            hip_below_knee_cm=round(float(depth_cm[b]), 1),
            below_parallel=bool(np.nanmin(thigh[seg]) < 0),
        ))
    summary: dict = {"reps": len(reps), "standing_knee_deg": round(top, 1)}
    if reps:
        summary.update(_stats(reps, ("min_knee_deg", "min_hip_deg", "max_trunk_lean_deg",
                                     "min_thigh_deg", "hip_below_knee_cm", "descent_s",
                                     "ascent_s")))
        summary["reps_below_parallel"] = sum(r.below_parallel for r in reps)
        if single_leg:
            summary["reps_by_leg"] = {leg: sum(r.leg == leg for r in reps)
                                      for leg in ("left", "right")}
    summary["series"] = {"knee": knee, "top": top, "cycles": [(r.start_s, r.end_s) for r in reps]}
    return reps, summary


# ---------------------------------------------------------------- plank


def hip_offset_pct(points: np.ndarray, side: Side | None = None) -> np.ndarray:
    """Signed hip distance from the shoulder-ankle line, % of line length.

    `points` is (n, 33, D) with y pointing down (image or MediaPipe world).
    If `side` is None both sides are averaged (mid-points).
    """
    def mid(name: str) -> np.ndarray:
        if side:
            return points[:, LM[f"{side.upper()}_{name}"]]
        return (points[:, LM[f"LEFT_{name}"]] + points[:, LM[f"RIGHT_{name}"]]) / 2

    a, p, b = mid("SHOULDER"), mid("HIP"), mid("ANKLE")
    ab = b - a
    length = np.linalg.norm(ab, axis=-1)
    tpar = np.sum((p - a) * ab, axis=-1) / np.maximum(length**2, 1e-12)
    off = p - (a + tpar[:, None] * ab)
    mag = np.linalg.norm(off, axis=-1)
    sign = np.sign(off[:, 1])  # +y = down = sag
    return 100 * sign * mag / np.maximum(length, 1e-12)


def plank_metrics(track: Track, ang: Angles, tolerance_pct: float = 5.0,
                  min_lean: float = 60.0, max_lean: float = 120.0) -> dict:
    fps = track.fps
    win = max(1, int(fps / 6))
    lean = smooth(ang.primary("trunk_lean"), win)
    if ang.view == "side":
        offset = hip_offset_pct(track.image, ang.side)
    else:
        offset = hip_offset_pct(track.world)
    offset = smooth(offset, win)
    line = smooth(ang.primary("body_line"), win)

    horizontal = np.isfinite(lean) & (lean > min_lean) & (lean < max_lean)
    runs = [r for r in _runs(horizontal) if r[1] - r[0] >= int(fps)]  # at least 1 s
    out: dict = {"view": ang.view, "hold_s": 0.0}
    if not runs:
        out["note"] = "no horizontal hold of >= 1 s found"
        out["series"] = {"offset": offset, "lean": lean, "hold": (0, 0)}
        return out
    s, e = max(runs, key=lambda r: r[1] - r[0])
    o, ln = offset[s:e], line[s:e]
    out.update({
        "hold_start_s": round(float(track.time_s[s]), 2),
        "hold_s": round(float(track.time_s[e - 1] - track.time_s[s]), 2),
        "body_line_deg_mean": round(float(np.nanmean(ln)), 1),
        "body_line_deg_min": round(float(np.nanmin(ln)), 1),
        "hip_offset_pct_mean": round(float(np.nanmean(o)), 2),
        "hip_offset_pct_sd": round(float(np.nanstd(o)), 2),
        "pct_time_straight": round(100 * float(np.mean(np.abs(o) <= tolerance_pct)), 1),
        "pct_time_sag": round(100 * float(np.mean(o > tolerance_pct)), 1),
        "pct_time_pike": round(100 * float(np.mean(o < -tolerance_pct)), 1),
        # drift: did the hips drop over the hold? (last third minus first third)
        "hip_offset_drift_pct": round(float(np.nanmean(o[-len(o) // 3:]) -
                                            np.nanmean(o[: len(o) // 3])), 2),
        "series": {"offset": offset, "lean": lean, "hold": (s, e)},
    })
    if ang.view != "side":
        out["note"] = "front view: hip offset from 3D estimate, less reliable than side view"
    return out


def reps_as_dicts(reps: list) -> list[dict]:
    return [asdict(r) for r in reps]


# ---------------------------------------------------------------- pike push-ups


@dataclass
class PikeRep:
    rep: int
    start_s: float
    bottom_s: float
    end_s: float
    min_elbow_deg: float  # push-up depth
    hip_deg_at_bottom: float  # how piked the hips were


def pike_pushup_reps(track: Track, ang: Angles, pike_hip: float = 120.0,
                     plank_hip: float = 145.0) -> tuple[list[PikeRep], dict]:
    """Push-ups done in a pike (hips folded, < `pike_hip` deg), counted on the elbow.

    Also reports time spent in a plank (hips open, torso horizontal) and in a pike
    with straight arms, since these clips usually move plank -> pike -> push-ups.
    """
    win = max(1, int(track.fps / 6))
    t = track.time_s
    hip = smooth(ang.primary("hip"), win)
    elbow = smooth(ang.primary("elbow"), win)
    lean = smooth(ang.primary("trunk_lean"), win)
    piked = np.nan_to_num(hip, nan=180) < pike_hip
    top = float(np.nanpercentile(elbow[piked], 90)) if piked.any() else 160.0
    reps = []
    for s, b, e in find_cycles(elbow, t, 45.0, 25.0, 0.5, top=top):
        if not piked[b]:
            continue
        reps.append(PikeRep(len(reps) + 1, round(float(t[s]), 2), round(float(t[b]), 2),
                            round(float(t[e]), 2), round(float(elbow[b]), 1),
                            round(float(hip[b]), 1)))
    dt = np.gradient(t) if len(t) > 1 else np.zeros(len(t))
    plank = (np.nan_to_num(hip) > plank_hip) & (np.abs(np.nan_to_num(lean) - 90) < 30)
    pike_hold = piked & (np.nan_to_num(elbow) > top - 25)
    summary: dict = {"reps": len(reps), "arms_straight_deg": round(top, 1),
                     "plank_s": round(float(dt[plank].sum()), 2),
                     "pike_straight_arms_s": round(float(dt[pike_hold].sum()), 2)}
    if reps:
        summary.update(_stats(reps, ("min_elbow_deg", "hip_deg_at_bottom")))
    summary["series"] = {"signal": elbow, "label": "elbow angle (deg)",
                         "cycles": [(r.start_s, r.end_s) for r in reps],
                         "extra": (hip, "hip angle (deg)")}
    return reps, summary


# ---------------------------------------------------------------- jumping jacks


@dataclass
class JackRep:
    rep: int
    start_s: float
    end_s: float
    max_arm_deg: float  # shoulder angle at the top (180 = arms straight overhead)
    max_feet_width: float  # ankle distance / shoulder width at the widest


def jumping_jack_reps(track: Track, ang: Angles) -> tuple[list[JackRep], dict]:
    """Counted on the arms: one rep = arms up and back down. Front view, 2D angles
    (the arms move in the image plane)."""
    win = max(1, int(track.fps / 15))
    t = track.time_s
    arms = smooth(np.nanmean([ang.values["shoulder_l_2d"], ang.values["shoulder_r_2d"]],
                             axis=0), win)
    img = track.image
    sw = np.linalg.norm(img[:, LM.LEFT_SHOULDER] - img[:, LM.RIGHT_SHOULDER], axis=-1)
    feet = smooth(np.linalg.norm(img[:, LM.LEFT_ANKLE] - img[:, LM.RIGHT_ANKLE], axis=-1)
                  / np.maximum(sw, 1e-6), win)
    reps = []
    for s, _b, e in find_cycles(180 - arms, t, 80.0, 40.0, 0.25):
        reps.append(JackRep(len(reps) + 1, round(float(t[s]), 2), round(float(t[e]), 2),
                            round(float(np.nanmax(arms[s:e + 1])), 1),
                            round(float(np.nanmax(feet[s:e + 1])), 2)))
    summary: dict = {"reps": len(reps)}
    if len(reps) >= 2:
        span = reps[-1].end_s - reps[0].start_s
        summary["reps_per_min"] = round(60 * len(reps) / span, 1)
    if reps:
        summary.update(_stats(reps, ("max_arm_deg", "max_feet_width")))
    summary["series"] = {"signal": arms, "label": "arm elevation (shoulder angle, deg)",
                         "cycles": [(r.start_s, r.end_s) for r in reps], "peaks": True}
    return reps, summary


# ---------------------------------------------------------------- bar exercises


def _grip_y(st) -> np.ndarray:
    """Height of the bar in the image per frame: mean y of the gripping hands."""
    ys = [np.where(st.contacts.grip[s], st.contacts.hand_px[s][:, 1], np.nan)
          for s in st.contacts.grip]
    if not ys:
        return np.full(len(st.com_px), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack(ys), axis=0)


def _cm(px: np.ndarray, st, i) -> np.ndarray:
    return px * 100.0 / np.where(np.isfinite(st.scale.ppm[i]), st.scale.ppm[i],
                                 st.scale.px_per_m)


def _arms(ang: Angles) -> np.ndarray:
    """Elbow angle for bar work: both arms in a front view (they move in the image
    plane there), the near arm in a side view."""
    if ang.view == "front":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean([ang.values["elbow_l_2d"], ang.values["elbow_r_2d"]], axis=0)
    return ang.primary("elbow")


@dataclass
class PullupRep:
    rep: int
    start_s: float
    top_s: float
    end_s: float
    up_s: float
    down_s: float
    pull_cm: float  # how far the shoulders rose towards the bar
    top_gap_cm: float  # shoulders below the bar at the top
    com_rise_cm: float  # how far the COM rose (less than pull_cm if the legs hang/kick)
    chin_clearance_cm: float  # estimated chin height above the bar at the top (+ = over)
    chin: str  # "over" / "at bar" (within +-chin_tol_cm, can't call it) / "under"
    top_elbow_deg: float  # elbow angle at the top
    start_elbow_deg: float  # elbow angle before pulling (dead hang ~170)
    swing_cm: float  # range of COM x relative to the grip during the rep
    hip_range_deg: float  # big values = kipping / leg drive
    kipping: bool


def _pull_elbow(ang: Angles) -> np.ndarray:
    """Elbow angle for pull-ups. Front view: 3D, because the elbows travel towards the
    camera (narrow/chin-up grips) and look folded flat in the image. Side view: near
    arm, 2D."""
    if ang.view == "front":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean([ang.values["elbow_l_3d"], ang.values["elbow_r_3d"]], axis=0)
    return ang.primary("elbow")


def pullup_reps(track: Track, ang: Angles, st, start_drop: float = 0.12,
                end_drop: float = 0.05, kip_deg: float = 40.0,
                chin_tol_cm: float = 5.0) -> tuple[list[PullupRep], dict]:
    """Pull-ups / chin-ups, counted on how far the shoulders hang below the bar.

    Signal = (shoulder height - grip height) / body length: ~0.4 in a dead hang, ~0.05
    at the top. A rep is a dip of more than `start_drop` that comes back within
    `end_drop` (body lengths). This works from any view and doesn't need the face.

    Chin over bar is estimated from the body, not the face (the face is often hidden by
    the bar or a door frame at the top): chin height = shoulder height + a fixed body
    proportion. Shoulder landmarks and proportions are each good to a few cm, so within
    `chin_tol_cm` of the bar the rep is reported as "at bar" rather than guessed.
    """
    win = max(1, int(track.fps / 6))
    t = track.time_s
    img = track.image
    bar_y = _grip_y(st)
    body = st.scale.body_px
    shoulder_y = smooth((img[:, LM.LEFT_SHOULDER, 1] + img[:, LM.RIGHT_SHOULDER, 1]) / 2, win)
    gap = smooth(shoulder_y - bar_y, win) / body
    elbow = smooth(_pull_elbow(ang), win)
    hip = smooth(ang.primary("hip"), win)
    com_y = smooth(st.com_px[:, 1], win)
    rest = float(np.nanpercentile(gap, 90)) if np.isfinite(gap).any() else np.nan
    # chin height above the shoulder line from body proportions (Drillis & Contini:
    # chin ~0.870 x stature, shoulder ~0.818 x stature; trunk + leg ~0.55 x stature),
    # i.e. ~0.095 body lengths. Not from the face: it is often hidden at the top, and
    # while hanging the shoulders shrug up towards the ears, which would bias it.
    chin_px = 0.095 * body
    reps = []
    for s, b, e in find_cycles(gap, t, start_drop, end_drop, 0.6, top=rest):
        if not st.on_bar[b]:
            continue
        seg = slice(s, e + 1)
        sw = st.swing_cm[seg]
        hr = float(np.nanmax(hip[seg]) - np.nanmin(hip[seg]))
        clearance_px = bar_y[b] - (shoulder_y[b] - chin_px)
        clearance = float(_cm(clearance_px, st, b))
        base = float(np.nanmax(gap[max(0, s - int(track.fps)):s + 1]))  # hang before the rep
        reps.append(PullupRep(
            rep=len(reps) + 1, start_s=round(float(t[s]), 2), top_s=round(float(t[b]), 2),
            end_s=round(float(t[e]), 2), up_s=round(float(t[b] - t[s]), 2),
            down_s=round(float(t[e] - t[b]), 2),
            pull_cm=round(float(_cm((base - gap[b]) * body, st, b)), 1),
            top_gap_cm=round(float(_cm(gap[b] * body, st, b)), 1),
            com_rise_cm=round(float(_cm(com_y[s] - np.nanmin(com_y[seg]), st, b)), 1),
            chin_clearance_cm=round(clearance, 1),
            chin=("over" if clearance > chin_tol_cm else "under" if clearance < -chin_tol_cm
                  else "at bar"),
            top_elbow_deg=round(float(np.nanmin(elbow[seg])), 1),
            start_elbow_deg=round(float(elbow[s]), 1),
            swing_cm=round(float(np.nanmax(sw) - np.nanmin(sw)), 1)
            if np.isfinite(sw).any() else float("nan"),
            hip_range_deg=round(hr, 1), kipping=hr > kip_deg))
    summary: dict = {"reps": len(reps), "chin_above_shoulders_cm":
                     round(float(_cm(chin_px, st, 0)), 1)}
    if reps:
        summary.update(_stats(reps, ("pull_cm", "top_gap_cm", "com_rise_cm",
                                     "chin_clearance_cm", "top_elbow_deg", "start_elbow_deg",
                                     "up_s", "down_s", "swing_cm", "hip_range_deg")))
        summary["reps_chin"] = {c: sum(r.chin == c for r in reps)
                                for c in ("over", "at bar", "under")}
        summary["reps_kipping"] = sum(r.kipping for r in reps)
    summary["series"] = {"signal": gap * 100, "label": "shoulders below bar (% body length)",
                         "cycles": [(r.start_s, r.end_s) for r in reps],
                         "extra": (elbow, "elbow angle (deg)")}
    return reps, summary


@dataclass
class LegRaiseRep:
    rep: int
    start_s: float
    end_s: float
    min_hip_deg: float  # 90 = legs horizontal, smaller = higher
    min_knee_deg: float  # ~180 straight-leg raise, small = knee raise
    toes_to_bar: bool  # feet reached the bar


def leg_raise_reps(track: Track, ang: Angles, st) -> tuple[list[LegRaiseRep], dict]:
    """Hanging leg raises / knee raises / toes-to-bar, counted on the hip angle."""
    win = max(1, int(track.fps / 6))
    t = track.time_s
    hip = smooth(ang.primary("hip"), win)
    knee = smooth(ang.primary("knee"), win)
    with warnings.catch_warnings():  # frames with no pose
        warnings.simplefilter("ignore", RuntimeWarning)
        feet_y = np.nanmin(track.image[:, [LM.LEFT_FOOT_INDEX, LM.RIGHT_FOOT_INDEX], 1], axis=1)
    bar_y = _grip_y(st)
    body = st.scale.body_px
    reps = []
    for s, b, e in find_cycles(hip, t, 40.0, 20.0, 0.6):
        if not st.on_bar[b]:
            continue
        seg = slice(s, e + 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            gap = np.nanmin(feet_y[seg] - bar_y[seg])
        reps.append(LegRaiseRep(len(reps) + 1, round(float(t[s]), 2), round(float(t[e]), 2),
                                round(float(hip[b]), 1), round(float(np.nanmin(knee[seg])), 1),
                                bool(np.isfinite(gap) and gap < 0.05 * body)))
    summary: dict = {"reps": len(reps)}
    if reps:
        summary.update(_stats(reps, ("min_hip_deg", "min_knee_deg")))
        summary["reps_toes_to_bar"] = sum(r.toes_to_bar for r in reps)
    # longest hold with the legs raised (hip < 120 deg) on the bar, e.g. a knee-raise hold
    raised = (np.nan_to_num(hip, nan=180) < 120) & st.on_bar
    runs = [r for r in _runs(raised) if t[r[1] - 1] - t[r[0]] >= 3.0]
    if runs:
        a, b = max(runs, key=lambda r: r[1] - r[0])
        k = max(1, int(track.fps))  # first / last second of the hold
        summary["raised_hold"] = {
            "start_s": round(float(t[a]), 2), "hold_s": round(float(t[b - 1] - t[a]), 2),
            "hip_deg_start": round(float(np.nanmedian(hip[a:a + k])), 1),
            "hip_deg_end": round(float(np.nanmedian(hip[b - k:b])), 1),
            "knee_deg_median": round(float(np.nanmedian(knee[a:b])), 1)}
        h = summary["raised_hold"]
        h["hip_drop_deg"] = round(h["hip_deg_end"] - h["hip_deg_start"], 1)
    summary["series"] = {"signal": hip, "label": "hip angle (deg)",
                         "cycles": [(r.start_s, r.end_s) for r in reps],
                         "extra": (knee, "knee angle (deg)")}
    return reps, summary


def dead_hang(track: Track, ang: Angles, st) -> dict:
    """Longest hang (hands on the bar, nothing on the floor): time, arm straightness,
    and swing of the COM relative to the grip."""
    t = track.time_s
    runs = _runs(st.hanging)
    if not runs:
        return {"note": "no hang found (hands on a bar with feet off the floor)"}
    a, b = max(runs, key=lambda r: r[1] - r[0])
    sw = st.swing_cm[a:b]
    sw = sw[np.isfinite(sw)]
    elbow = _arms(ang)[a:b]
    out = {"start_s": round(float(t[a]), 2), "hold_s": round(float(t[b - 1] - t[a]), 2),
           "elbow_deg_median": round(float(np.nanmedian(elbow)), 1)}
    if len(sw):
        out.update({"swing_sd_cm": round(float(np.std(sw)), 2),
                    "swing_range_cm": round(float(np.ptp(sw)), 2)})
    return out
