"""Ground contacts, base of support and stability (off the wall).

Pipeline per clip:
 1. Scale: body size in pixels and pixels-per-metre, from limb lengths in the image
    vs MediaPipe's metric 3D estimate.
 2. Floor: a line in the image fitted to the lowest well-supported points
    (RANSAC; "lowest" so resting hands at hip height don't win).
 3. Contacts: a landmark touches the floor when it is close to the floor line AND
    nearly still. Grouped into body parts (feet, hands, forearms, knees).
 4. Base of support:
      the span of contact points along the image's horizontal axis.
      Side view -> front/back; front view -> left/right (contacts padded by half
      a foot/hand width, since the image then shows their width, not length).
      A full 2D polygon would need reliable depth; MediaPipe's 3D estimate was too
      inconsistent for that on test clips (COM "outside" the feet while standing).
 5. Stability margin: signed distance from the COM's vertical projection to the
    edge of the support (positive = inside, negative = outside), in cm.

Bar exercises (pull-ups, hangs, leg raises) run with `bar=True`: hands that stay
put above the shoulders are "grips" on a bar, the floor is learned only from frames
without a grip (it may not be visible at all), and instead of a stability margin
the summary reports the swing of the COM relative to the grip.

Assumes a fixed, roughly level camera (image "down" ~ gravity).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from climbvision.angles import View
from climbvision.com import Sex, whole_body_com
from climbvision.skeleton import LM
from climbvision.track import Track, smooth

# body part -> landmarks that can touch the floor
CONTACT_GROUPS: dict[str, tuple[LM, ...]] = {
    "left_foot": (LM.LEFT_HEEL, LM.LEFT_FOOT_INDEX),
    "right_foot": (LM.RIGHT_HEEL, LM.RIGHT_FOOT_INDEX),
    "left_hand": (LM.LEFT_WRIST, LM.LEFT_INDEX, LM.LEFT_PINKY),
    "right_hand": (LM.RIGHT_WRIST, LM.RIGHT_INDEX, LM.RIGHT_PINKY),
    "left_forearm": (LM.LEFT_ELBOW,),
    "right_forearm": (LM.RIGHT_ELBOW,),
    "left_knee": (LM.LEFT_KNEE,),
    "right_knee": (LM.RIGHT_KNEE,),
}
# footprint radius (m) around each contact point: half foot width, hand, forearm, knee
CONTACT_RADIUS_M = {"foot": 0.045, "hand": 0.04, "forearm": 0.04, "knee": 0.05}


def _radius(group: str) -> float:
    return CONTACT_RADIUS_M[group.split("_", 1)[1]]


# ------------------------------------------------------------------ scale

_SEGMENTS = [(LM.LEFT_SHOULDER, LM.LEFT_ELBOW), (LM.LEFT_ELBOW, LM.LEFT_WRIST),
             (LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW), (LM.RIGHT_ELBOW, LM.RIGHT_WRIST),
             (LM.LEFT_HIP, LM.LEFT_KNEE), (LM.LEFT_KNEE, LM.LEFT_ANKLE),
             (LM.RIGHT_HIP, LM.RIGHT_KNEE), (LM.RIGHT_KNEE, LM.RIGHT_ANKLE),
             (LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER), (LM.LEFT_HIP, LM.RIGHT_HIP),
             (LM.LEFT_SHOULDER, LM.LEFT_HIP), (LM.RIGHT_SHOULDER, LM.RIGHT_HIP)]


def _rolling_median(x: np.ndarray, win: int) -> np.ndarray:
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
    """Pixels per metre in each frame (changes as the person moves nearer/farther).

    Each body segment gives image length / true (3D) length; a segment pointing at
    the camera looks short, but none looks *longer* than it is, so the upper end of
    the per-segment ratios (80th percentile) is the scale. Smoothed over ~1 s.
    """
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
    return _rolling_median(raw, max(3, int(track.fps)))


def _chain_len(pts: np.ndarray) -> np.ndarray:
    """Trunk + longer leg (hip-knee-ankle) length per frame, for (n, 33, D) points."""
    mid_s = (pts[:, LM.LEFT_SHOULDER] + pts[:, LM.RIGHT_SHOULDER]) / 2
    mid_h = (pts[:, LM.LEFT_HIP] + pts[:, LM.RIGHT_HIP]) / 2
    trunk = np.linalg.norm(mid_s - mid_h, axis=-1)
    legs = []
    for s in ("LEFT", "RIGHT"):
        h, k, a = (pts[:, LM[f"{s}_{n}"]] for n in ("HIP", "KNEE", "ANKLE"))
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


def estimate_scale(track: Track) -> Scale:
    ppm = per_frame_px_per_m(track)
    if not np.isfinite(ppm).any():
        return Scale(float("nan"), float("nan"), ppm)
    ref = float(np.nanmedian(ppm))
    body_m = float(np.nanmedian(_chain_len(track.world)))
    return Scale(ref * body_m, ref, ppm)


# ------------------------------------------------------------------ floor


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
    for lms in CONTACT_GROUPS.values():
        pts = track.image[:, list(lms)]  # (n, k, 2)
        valid = np.isfinite(pts[..., 1]).all(axis=1) & use
        if not valid.any():
            continue
        j = np.argmax(np.where(np.isfinite(pts[..., 1]), pts[..., 1], -np.inf), axis=1)
        low = pts[np.arange(len(pts)), j]
        xs.append(low[valid, 0])
        ys.append(low[valid, 1])
        qs.append(q_all[valid])
        rs.append(rel_all[valid])
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


# ------------------------------------------------------------------ contacts


def _clean_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Fill False gaps shorter than min_len, then drop True runs shorter than min_len."""
    out = mask.copy()
    for val in (False, True):
        m = np.concatenate([[not val], out == val, [not val]])
        d = np.diff(m.astype(int))
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True):
            interior = s > 0 and e < len(out)
            if e - s < min_len and (val or interior):
                out[s:e] = not val
    return out


@dataclass
class Contacts:
    point: np.ndarray  # (n, 33) bool: landmark touching the floor
    group: dict[str, np.ndarray]  # part -> (n,) bool
    height_px: np.ndarray  # (n, 33) smoothed height above the floor
    grip: dict[str, np.ndarray] = field(default_factory=dict)  # "left"/"right" -> (n,) bool
    hand_px: dict[str, np.ndarray] = field(default_factory=dict)  # side -> (n, 2) hand centre


def detect_contacts(track: Track, floor: Floor, scale: Scale, height_tol: float = 0.05,
                    speed_tol: float = 0.6, min_s: float = 0.15) -> Contacts:
    """height_tol in body lengths; speed_tol in body lengths per second (both scaled
    to the person's current distance from the camera)."""
    fps = track.fps
    win = max(1, int(fps / 10))
    img = np.stack([np.stack([smooth(track.image[:, i, d], win) for d in (0, 1)], -1)
                    for i in range(track.image.shape[1])], axis=1)
    h = floor.height(img)
    vel = np.gradient(img, track.time_s, axis=0)
    speed = np.linalg.norm(vel, axis=-1)
    body = (scale.body_px * np.nan_to_num(scale.rel, nan=1.0))[:, None]
    near = np.nan_to_num(h, nan=np.inf) < height_tol * body
    still = np.nan_to_num(speed, nan=np.inf) < speed_tol * body
    point = near & still
    min_len = max(1, int(min_s * fps))
    groups = {}
    for name, lms in CONTACT_GROUPS.items():
        g = point[:, list(lms)].any(axis=1)
        groups[name] = _clean_runs(g, min_len)
        point[:, list(lms)] &= groups[name][:, None]
    # a contacting group with no individual point (filled gap) keeps its lowest point
    for name, lms in CONTACT_GROUPS.items():
        gap = groups[name] & ~point[:, list(lms)].any(axis=1)
        if gap.any():
            hh = np.where(np.isfinite(h[:, list(lms)]), h[:, list(lms)], np.inf)
            j = np.array(lms)[np.argmin(hh, axis=1)]
            point[np.flatnonzero(gap), j[gap]] = True
    return Contacts(point, groups, h)


# ------------------------------------------------------------------ bar grips

HANDS = {"left": (LM.LEFT_WRIST, LM.LEFT_INDEX, LM.LEFT_PINKY),
         "right": (LM.RIGHT_WRIST, LM.RIGHT_INDEX, LM.RIGHT_PINKY)}


def detect_grips(track: Track, scale: Scale, speed_tol: float = 0.35, min_s: float = 0.3,
                 drift: float = 0.08) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Hands holding something fixed (a bar), per side.

    Seed: the hand is above its shoulder and nearly still (speed_tol body lengths/s)
    for at least `min_s`. Each seed then grows forwards and backwards for as long as
    the hand stays within `drift` body lengths of where it was holding, which keeps the
    grip through the top of a pull-up (when the shoulders reach the hands) and through
    tracking jitter. Only meaningful when the clip is known to involve a bar.
    """
    fps = track.fps
    n = track.n
    win = max(1, int(fps / 10))
    body = scale.body_px * np.nan_to_num(scale.rel, nan=1.0)
    min_len = max(1, int(min_s * fps))
    grips, pos = {}, {}
    for side, lms in HANDS.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            raw = np.nanmean(track.image[:, list(lms)], axis=1)
        c = np.column_stack([smooth(raw[:, 0], win), smooth(raw[:, 1], win)])
        speed = np.linalg.norm(np.gradient(c, track.time_s, axis=0), axis=-1)
        shoulder_y = track.image[:, LM[f"{side.upper()}_SHOULDER"], 1]
        seed = (np.nan_to_num(c[:, 1], nan=np.inf) < np.nan_to_num(shoulder_y, nan=-np.inf)) \
            & (np.nan_to_num(speed, nan=np.inf) < speed_tol * body)
        seed = _clean_runs(seed, min_len)
        mask = seed.copy()
        for a, b in _runs(seed):
            anchor = np.nanmedian(c[a:b], axis=0)
            near = np.nan_to_num(np.linalg.norm(c - anchor, axis=1), nan=np.inf) < drift * body
            i = a - 1
            while i >= 0 and near[i] and not mask[i]:
                mask[i] = True
                i -= 1
            i = b
            while i < n and near[i] and not mask[i]:
                mask[i] = True
                i += 1
        grips[side], pos[side] = mask, c
    return grips, pos


def stance_label(contacts: Contacts, i: int) -> str:
    """Short name for what's supporting the body, e.g. 'L foot', 'feet + forearms',
    'bar', 'both feet + bar'."""
    on = {g for g, m in contacts.group.items() if m[i]}
    gl = contacts.grip.get("left", np.zeros(i + 1, bool))[i]
    gr = contacts.grip.get("right", np.zeros(i + 1, bool))[i]
    bar = "bar" if gl and gr else "L hand on bar" if gl else "R hand on bar" if gr else ""
    if not on:
        return bar or "none"
    parts = []
    for part, plural in (("foot", "feet"), ("hand", "hands"), ("forearm", "forearms"),
                         ("knee", "knees")):
        left, right = f"left_{part}" in on, f"right_{part}" in on
        if left and right:
            parts.append(plural)
        elif left or right:
            parts.append(f"{'L' if left else 'R'} {part}")
    label = " + ".join(parts)
    label = "both feet" if label == "feet" else label
    return f"{label} + {bar}" if bar else label


# ------------------------------------------------------------------ stability


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


def compute_stability(track: Track, view: View, sex: Sex = "male",
                      frame_size: tuple[int, int] | None = None,
                      bar: bool = False) -> Stability:
    """`bar=True` for bar exercises: detect grips and learn the floor without them."""
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
        scale = Scale(scale.body_px, scale.px_per_m, np.full(track.n, scale.px_per_m))
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
        for g, lms in CONTACT_GROUPS.items():
            if not contacts.group[g][i]:
                continue
            r_px = _radius(g) * ppm[i] if pad else 0.0
            for lm in lms:
                if contacts.point[i, lm]:
                    x = track.image[i, lm, 0]
                    xs += [x - r_px, x + r_px]
        if xs and np.isfinite(com_px[i, 0]):
            lo[i], hi[i] = min(xs), max(xs)
            c = com_px[i, 0]
            margin[i] = min(c - lo[i], hi[i] - c) * 100.0 / ppm[i]

    feet = track.image[:, [LM.LEFT_HEEL, LM.LEFT_FOOT_INDEX, LM.RIGHT_HEEL,
                           LM.RIGHT_FOOT_INDEX]]
    if frame_size:
        w, h = frame_size
        out = (feet[..., 0] < 0) | (feet[..., 0] > w) | (feet[..., 1] > h)
        out_of_frame = out.any(axis=1) & track.detected
    else:
        out_of_frame = np.zeros(n, bool)
    # horizontal COM position in cm, about the image centre (perspective-correct
    # when the person changes distance; needs the frame size, else the image edge)
    cx = frame_size[0] / 2 if frame_size else 0.0
    com_x_cm = smooth((com_px[:, 0] - cx) * 100.0 / ppm, max(1, int(track.fps / 10)))
    on_floor = np.zeros(n, bool)
    for g in contacts.group.values():
        on_floor |= g
    hanging = on_bar & ~on_floor
    swing = np.full(n, np.nan)
    if bar:
        xs = [np.where(grips[sd], hand_px[sd][:, 0], np.nan) for sd in grips]
        with warnings.catch_warnings():  # frames with no grip are all-NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            grip_x = np.nanmean(np.stack(xs), axis=0)
        swing = smooth((com_px[:, 0] - grip_x) * 100.0 / ppm, max(1, int(track.fps / 10)))
    return Stability(view, scale, floor, contacts, com_px, lo, hi, margin, com_x_cm,
                     out_of_frame, on_bar, hanging, swing)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True))


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
    for a, b in _runs(none):
        if a == 0 or b >= len(t):
            continue
        dur = t[b] - t[a - 1]
        if not min_s <= dur <= max_s:
            continue
        if st is not None:
            body = st.scale.body_px * np.nan_to_num(st.scale.rel[a - 1], nan=1.0)
            feet = st.contacts.height_px[a:b][:, [LM.LEFT_HEEL, LM.LEFT_FOOT_INDEX,
                                                  LM.RIGHT_HEEL, LM.RIGHT_FOOT_INDEX]]
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
    labels = stance_labels(st.contacts)
    best = None
    for lab in set(labels) - {"none", "both feet"}:
        mask = np.array([lb == lab for lb in labels])
        for a, b in _runs(mask):
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


def stance_labels(contacts: Contacts) -> list[str]:
    n = len(next(iter(contacts.group.values())))
    return [stance_label(contacts, i) for i in range(n)]


def stability_summary(st: Stability, t: np.ndarray) -> dict:
    axis = "front-back" if st.view == "side" else "left-right"
    n = len(t)
    labels = stance_labels(st.contacts)
    dt = np.gradient(t) if n > 1 else np.ones(n)
    stance_time: dict[str, float] = {}
    for lab, d in zip(labels, dt, strict=True):
        stance_time[lab] = stance_time.get(lab, 0.0) + float(d)

    m = st.margin_cm
    out: dict = {
        "support_axis": axis,
        "scale_px_per_m": round(st.scale.px_per_m, 1),
        "floor_tilt_deg": round(st.floor.angle_deg, 2) if np.isfinite(st.floor.a) else None,
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
        mask = np.array([lb == lab for lb in labels])
        runs = [r for r in _runs(mask) if t[r[1] - 1] - t[r[0]] >= 2.0]
        if runs:
            s, e = max(runs, key=lambda r: r[1] - r[0])
            per[lab] = {"longest_s": round(float(t[e - 1] - t[s]), 2),
                        **sway(st.com_x_cm[s:e], t[s:e])}
    if per:
        out["sway_by_stance"] = per
    if st.on_bar.any():
        dtt = np.gradient(t) if n > 1 else np.zeros(n)
        hang = {"on_bar_s": round(float(dtt[st.on_bar].sum()), 2),
                "hanging_s": round(float(dtt[st.hanging].sum()), 2)}
        runs = _runs(st.hanging)
        if runs:
            a, b = max(runs, key=lambda r: r[1] - r[0])
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

    warnings = []
    oof = 100 * float(st.feet_out_of_frame.mean())
    if oof > 10:
        warnings.append(f"feet outside the frame in {oof:.0f}% of frames: foot contacts and "
                        "support are guesses there. Move the camera back.")
    # airborne time inside real jumps is expected, not a problem
    in_flight = sum(b - a for a, b in jumps) if jumps else 0.0
    none_s = max(stance_time.get("none", 0.0) - in_flight, 0.0)
    none_pct = 100 * none_s / max(t[-1] - t[0], 1e-9)
    if not np.isfinite(st.floor.a):
        warnings.append("floor not visible without the bar grip: floor contacts unknown")
    elif none_pct > 25:
        warnings.append(f"no ground contact found in {none_pct:.0f}% of the clip")
    if warnings:
        out["warnings"] = warnings
    return out
