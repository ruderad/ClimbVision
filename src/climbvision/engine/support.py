"""Support layer: base of support, stability margin, sway, flight and bar hanging.

Base of support = the span of contact points along the image's horizontal axis
(side view -> front/back; front view -> left/right, contacts padded by half a
foot/hand width, since the image then shows their width). Stability margin = signed
distance from the COM's plumb line to the nearest support edge (+ inside), in cm.
A full 2D polygon needs reliable depth (MediaPipe's 3D was not good enough); on the
wall (E4) the polygon lives in the wall plane instead.

Bar scenes (`bar=True`): grips are detected, the floor is learned only from frames
without a grip, and the swing of the COM relative to the grip replaces the margin.
Assumes image "down" ~ gravity (camera roughly level).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from climbvision.engine.biomech import Sex, View, whole_body_com
from climbvision.engine.contacts import (
    FEET,
    Contacts,
    detect_contacts,
    detect_grips,
    radius_m,
    stance_labels,
)
from climbvision.engine.events import longest_run, runs
from climbvision.engine.scale import Scale, estimate_scale
from climbvision.engine.surfaces import FLOOR_PARTS, Floor, estimate_floor
from climbvision.engine.track import Track, smooth


@dataclass
class Stability:
    view: View
    scale: Scale
    floor: Floor
    contacts: Contacts
    com_px: np.ndarray  # (n, 2)
    support_lo_px: np.ndarray  # (n,) support span along image x
    support_hi_px: np.ndarray
    margin_cm: np.ndarray  # (n,) COM to nearest support edge; + inside / - outside
    com_x_cm: np.ndarray  # COM along image x in cm (smoothed), for sway
    feet_out_of_frame: np.ndarray  # (n,) bool, any heel/toe outside the image
    on_bar: np.ndarray  # (n,) bool, at least one hand gripping (bar mode only)
    hanging: np.ndarray  # (n,) bool, gripping with nothing on the floor
    swing_cm: np.ndarray  # (n,) COM x minus grip x, cm (NaN when not on the bar)
    labels: list[str]  # (n,) stance label per frame


def compute_stability(track: Track, view: View, sex: Sex = "male",
                      frame_size: tuple[int, int] | None = None,
                      bar: bool = False) -> Stability:
    """`bar=True` for bar scenes: detect grips and learn the floor without them."""
    n = track.n
    scale = estimate_scale(track)
    grips, hand_px = detect_grips(track, scale) if bar else ({}, {})
    on_bar = np.zeros(n, bool)
    for g in grips.values():
        on_bar |= g
    floor = estimate_floor(track, scale, exclude=on_bar if bar else None)
    if floor.depth_m == 0:
        # no evidence the person changed distance: per-frame scale changes are pose
        # noise (foreshortened limbs), so use one scale for the whole clip
        scale = scale.constant()
        floor.ppm = scale.ppm
    contacts = detect_contacts(track, floor, scale)
    contacts.grip, contacts.hand_px = grips, hand_px
    com_px = whole_body_com(track.image, sex)
    ppm = np.where(np.isfinite(scale.ppm), scale.ppm, scale.px_per_m)

    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    margin = np.full(n, np.nan)
    pad = view == "front"  # front view: image x spans foot *width*
    for i in range(n):
        xs = []
        for g, joints in FLOOR_PARTS.items():
            if not contacts.group[g][i]:
                continue
            r_px = radius_m(g) * ppm[i] if pad else 0.0
            for j in joints:
                if contacts.point[i, j]:
                    x = track.image[i, j, 0]
                    xs += [x - r_px, x + r_px]
        if xs and np.isfinite(com_px[i, 0]):
            lo[i], hi[i] = min(xs), max(xs)
            c = com_px[i, 0]
            margin[i] = min(c - lo[i], hi[i] - c) * 100.0 / ppm[i]

    feet = track.image[:, list(FEET)]
    if frame_size:
        w, h = frame_size
        out = (feet[..., 0] < 0) | (feet[..., 0] > w) | (feet[..., 1] > h)
        out_of_frame = out.any(axis=1) & track.detected
    else:
        out_of_frame = np.zeros(n, bool)
    # horizontal COM position in cm, about the image centre (perspective-correct
    # when the person changes distance; needs the frame size, else the image edge)
    win = max(1, int(track.fps / 10))
    cx = frame_size[0] / 2 if frame_size else 0.0
    com_x_cm = smooth((com_px[:, 0] - cx) * 100.0 / ppm, win)
    hanging = on_bar & ~contacts.on_floor()
    swing = np.full(n, np.nan)
    if bar:
        xs = [np.where(grips[sd], hand_px[sd][:, 0], np.nan) for sd in grips]
        with warnings.catch_warnings():  # frames with no grip are all-NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            grip_x = np.nanmean(np.stack(xs), axis=0)
        swing = smooth((com_px[:, 0] - grip_x) * 100.0 / ppm, win)
    return Stability(view, scale, floor, contacts, com_px, lo, hi, margin, com_x_cm,
                     out_of_frame, on_bar, hanging, swing, stance_labels(contacts))


def sway(x_cm: np.ndarray, t: np.ndarray) -> dict:
    """Postural sway of the COM along one axis: spread, range and mean speed."""
    ok = np.isfinite(x_cm)
    x, tt = x_cm[ok], t[ok]
    if len(x) < 3:
        return {}
    return {
        "sd_cm": round(float(np.std(x)), 2),
        "range_cm": round(float(np.ptp(x)), 2),
        "mean_speed_cm_s": round(float(np.sum(np.abs(np.diff(x))) / (tt[-1] - tt[0])), 2),
    }


def flight_phases(labels: list[str], t: np.ndarray, st: Stability | None = None,
                  min_s: float = 0.06, max_s: float = 1.2) -> list[tuple[float, float]]:
    """(takeoff, landing) times of airborne phases: no contact, contact on both sides,
    lasting min_s..max_s. With `st`, also require the feet to leave the floor and the
    COM to rise and fall like a jump (rules out contact flicker on a planted foot)."""
    none = np.array([lb == "none" for lb in labels])
    out = []
    for a, b in runs(none):
        if a == 0 or b >= len(t):
            continue
        dur = t[b] - t[a - 1]
        if not min_s <= dur <= max_s:
            continue
        if st is not None:
            body = st.scale.body_px * np.nan_to_num(st.scale.rel[a - 1], nan=1.0)
            feet = st.contacts.height_px[a:b][:, list(FEET)]
            low = np.min(np.where(np.isfinite(feet), feet, np.inf), axis=1)
            low = low[np.isfinite(low)]
            clearance = low.max() if len(low) else 0.0
            com_y = st.com_px[a - 1:b + 1, 1]
            apex = np.nanmin(com_y)
            rise = com_y[0] - apex  # COM goes up after takeoff...
            fall = com_y[-1] - apex  # ...and comes back down before landing (ballistic);
            # standing up out of a squat with a flickering foot contact only rises.
            # Short hops (<= 0.3 s, e.g. jumping jacks) land before the COM visibly
            # drops, so the fall test applies to longer flights only.
            ballistic = fall > 0.4 * rise or dur <= 0.3
            if not (clearance > 0 and rise > 0.015 * body and ballistic):
                continue
        out.append((float(t[a - 1]), float(t[b])))
    return out


def balance_hold(st: Stability, t: np.ndarray, min_s: float = 1.0) -> dict:
    """The longest unbroken hold on a reduced base (one foot, hands only, ...)."""
    labels = st.labels
    best = None
    for lab in sorted(set(labels) - {"none", "both feet"}):
        mask = np.array([lb == lab for lb in labels])
        for a, b in runs(mask):
            if t[b - 1] - t[a] >= min_s and (best is None or b - a > best[2] - best[1]):
                best = (lab, a, b)
    if best is None:
        return {}
    lab, a, b = best
    m = st.margin_cm[a:b]
    fm = m[np.isfinite(m)]
    out = {"stance": lab, "start_s": round(float(t[a]), 2),
           "hold_s": round(float(t[b - 1] - t[a]), 2), **sway(st.com_x_cm[a:b], t[a:b])}
    if len(fm):
        out.update({"margin_cm_median": round(float(np.median(fm)), 1),
                    "margin_cm_min": round(float(fm.min()), 1),
                    "pct_inside": round(100 * float(np.mean(fm > 0)), 1)})
    return out


def stability_summary(st: Stability, t: np.ndarray) -> dict:
    axis = "front-back" if st.view == "side" else "left-right"
    n = len(t)
    labels = st.labels
    dt = np.gradient(t) if n > 1 else np.ones(n)
    stance_time: dict[str, float] = {}
    for lab, d in zip(labels, dt, strict=True):
        stance_time[lab] = stance_time.get(lab, 0.0) + float(d)

    m = st.margin_cm
    out: dict = {
        "support_axis": axis,
        "scale_px_per_m": round(st.scale.px_per_m, 1),
        "floor_tilt_deg": round(st.floor.angle_deg, 2) if st.floor.known else None,
        "camera_height_m": round(st.floor.depth_m, 2) if st.floor.depth_m > 0 else None,
        "contact_pct": {g: round(100 * float(v.mean()), 1)
                        for g, v in st.contacts.group.items() if v.any()},
        "stance_time_s": {k: round(v, 2) for k, v in
                          sorted(stance_time.items(), key=lambda kv: -kv[1])},
    }
    if np.isfinite(m).any():
        fm = m[np.isfinite(m)]
        out["margin_cm"] = {"median": round(float(np.median(fm)), 1),
                            "min": round(float(fm.min()), 1),
                            "pct_inside": round(100 * float(np.mean(fm > 0)), 1)}
    out["com_sway"] = sway(st.com_x_cm, t)
    per = {}
    for lab in sorted(set(labels)):
        if lab == "none":
            continue
        r = longest_run(np.array([lb == lab for lb in labels]), t, min_s=2.0)
        if r:
            s, e = r
            per[lab] = {"longest_s": round(float(t[e - 1] - t[s]), 2),
                        **sway(st.com_x_cm[s:e], t[s:e])}
    if per:
        out["sway_by_stance"] = per
    if st.on_bar.any():
        hang = {"on_bar_s": round(float(dt[st.on_bar].sum()), 2),
                "hanging_s": round(float(dt[st.hanging].sum()), 2)}
        r = longest_run(st.hanging, t)
        if r:
            a, b = r
            hang["longest_hang_s"] = round(float(t[b - 1] - t[a]), 2)
            # swing: COM relative to the grip, over all hanging frames
            sw = sway(np.where(st.hanging, st.swing_cm, np.nan), t)
            hang["swing"] = {k: sw[k] for k in ("sd_cm", "range_cm") if k in sw}
            # a constant offset while hanging still = camera not level / at an angle,
            # not a real lean: judge swing by sd and range, not by this number
            if np.isfinite(st.swing_cm[st.hanging]).any():
                hang["swing"]["offset_median_cm"] = round(
                    float(np.nanmedian(st.swing_cm[st.hanging])), 1)
        out["bar"] = hang
    jumps = flight_phases(labels, t, st)
    if jumps:
        ft = np.array([b - a for a, b in jumps])
        out["jumps"] = {"count": len(jumps), "flight_s_mean": round(float(ft.mean()), 3),
                        "flight_s_max": round(float(ft.max()), 3),
                        # projectile: height = g * t^2 / 8
                        "height_cm_mean": round(float(100 * 9.81 * np.mean(ft**2) / 8), 1)}

    warn = []
    oof = 100 * float(st.feet_out_of_frame.mean())
    if oof > 10:
        warn.append(f"feet outside the frame in {oof:.0f}% of frames: foot contacts and "
                    "support are guesses there. Move the camera back.")
    # airborne time inside real jumps is expected, not a problem
    in_flight = sum(b - a for a, b in jumps) if jumps else 0.0
    none_s = max(stance_time.get("none", 0.0) - in_flight, 0.0)
    none_pct = 100 * none_s / max(t[-1] - t[0], 1e-9)
    if not st.floor.known:
        warn.append("floor not visible without the bar grip: floor contacts unknown")
    elif none_pct > 25:
        warn.append(f"no ground contact found in {none_pct:.0f}% of the clip")
    if warn:
        out["warnings"] = warn
    return out
