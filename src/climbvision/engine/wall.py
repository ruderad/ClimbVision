"""Wall layer: anchors (hands/feet on holds), holds used, support on the wall.

Works in wall coordinates (camera layer), so a limb resting on a hold is still even
while a handheld camera moves.

Anchor rule ("still ~0.5 s = on a hold"): a hand or foot is anchored when, over a
0.5 s window, its positions stay within `still_tol` body lengths (~6 cm) of their median
(80th percentile of the distances, so a few jittery frames don't break it). The anchor
then grows forwards and backwards while the limb stays near that spot; brief jittery
excursions are bridged (see `detect_anchors`). Hand point = the grip point (2/3 from wrist
to knuckles); foot point = the toe (climbers stand on the front of the shoe).

Ground: a foot anchor is on the ground (not a foothold) when, in the wall image (the
wall without the climber, camera layer), the area just below the toe is floor. The floor
is found from the bottom of the wall image: its main colours (it may be concrete and a
mat) and every pixel of those colours connected to the bottom (`floor_mask`). Footholds
have wall below them. The climb is the stretch from the first to the last moment with a
foot on a hold and none on the ground; before/after it the climber is on the ground.
Without a wall image (no camera layer) or without floor in view, no ground is assumed.

Holds: anchors of any limb whose positions are within `hold_radius` of each other are
one hold (the system doesn't detect holds in the image yet; a hold is "a place a limb
rested"). Moves: a limb leaving one hold and anchoring on another at least `min_move`
away (smaller re-placements are adjustments on the same hold).

Support on the wall (gravity = "down" in the reference view):
  - contacts: how many limbs are on holds (4 = two hands, two feet)
  - COM over the feet: signed horizontal distance from the COM to the nearest edge of
    the span of the anchored feet (+ inside), cm — the climbing version of the floor
    stability margin
  - support polygon: convex hull of all anchored limbs; signed distance from the COM to
    its edge (+ inside), cm, when >= 3 limbs are anchored
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from climbvision.engine.biomech import whole_body_com
from climbvision.engine.events import runs
from climbvision.engine.kinematics import derivatives
from climbvision.engine.metric import WallMetric, fit_wall_metric
from climbvision.engine.scale import ASSUMED_BODY_M
from climbvision.engine.skeleton import J

LIMBS = ("l_hand", "r_hand", "l_foot", "r_foot")
LABEL = {"l_hand": "L hand", "r_hand": "R hand", "l_foot": "L foot", "r_foot": "R foot"}
GRIP_FRACTION = 2 / 3
FOOT_RADIUS_M = 0.045  # half a shoe width, pads the feet span


def limb_points(points: np.ndarray) -> dict[str, np.ndarray]:
    """(n, N_JOINTS, 2) -> {limb: (n, 2)}: grip point of each hand, toe of each foot."""
    out = {}
    for limb, (w, h) in (("l_hand", (J.L_WRIST, J.L_HAND)), ("r_hand", (J.R_WRIST, J.R_HAND))):
        pw, ph = points[:, w], points[:, h]
        out[limb] = np.where(np.isfinite(ph), pw + GRIP_FRACTION * (ph - pw), pw)
    for limb, (toe, ankle) in (("l_foot", (J.L_TOE, J.L_ANKLE)),
                               ("r_foot", (J.R_TOE, J.R_ANKLE))):
        out[limb] = np.where(np.isfinite(points[:, toe]), points[:, toe], points[:, ankle])
    return out


@dataclass
class Anchor:
    limb: str
    start: int  # frame index
    end: int  # exclusive
    pos: np.ndarray  # (2,) wall px
    spread_px: float  # 80th percentile distance of the limb from pos while anchored
    hold: int = -1  # hold id (-1 = none: ground, smear)
    ground: bool = False
    smear: bool = False  # on the wall with no detected hold under it, proven to touch
    box: int = -1  # the detected hold (holddetect.py) it rests on, if any
    flag: bool = False  # still, off any hold, not proven to touch: a flag (no contact)
    missed: bool = False  # on a hold the detector missed (a hand above the shoulder)

    @property
    def on_hold(self) -> bool:
        return not (self.ground or self.smear or self.flag)

    @property
    def on_wall_contact(self) -> bool:
        """On a hold or smearing: part of the base of support."""
        return not (self.ground or self.flag)


@dataclass
class Hold:
    id: int
    pos: np.ndarray  # (2,) wall px, mean of its anchors
    limbs: set = field(default_factory=set)
    anchors: list = field(default_factory=list)
    box: int = -1  # the detected hold, if holds were detected


@dataclass
class WallParams:
    window_s: float = 0.5  # how long a limb must be still to count as anchored
    median_s: float = 0.2  # running median on the limb positions first (pose jitter)
    still_tol: float = 0.05  # body lengths (~6 cm): spread allowed within the window
    hold_tol: float = 0.06  # body lengths: an anchored limb stays this close to the spot
    drift_tol: float = 0.08  # body lengths: brief excursions up to this far are bridged
    max_gap_s: float = 0.25  # missing data bridged inside an anchor
    merge_gap_s: float = 0.4  # same limb, same place, short break -> one anchor
    hold_radius: float = 0.09  # body lengths (~11 cm): anchors this close = one hold
    hold_margin: float = 0.05  # body lengths (~5 cm): a rest this close to a detected hold
    #                            is on it (the grip point / toe sits at the hold's edge)
    smear_proof_m: float = 0.10  # a still limb off any hold touches the wall (a smear) only
    #                              if the COM moved this far meanwhile; else it's a flag
    min_move: float = 0.1  # body lengths: shorter re-placements are adjustments, not moves
    floor_lab_tol: float = 25.0  # Lab colour distance to the floor colour
    floor_frac: float = 0.6  # share of floor-coloured pixels below the toe -> ground
    stand_edge_s: float = 2.0  # standing cue: within this long of the clip's start/end
    level_tol: float = 0.06  # body lengths: both feet this level = standing
    ground_near_s: float = 3.0  # with a climb window: ground anchors this close count
    hand_window_s: float = 0.2  # with detected holds: a hand still this long ON a
    #                             detected hold is anchored (fast climbing: brief grips);
    #                             off any hold the usual window_s applies
    frag_gap_s: float = 1.0  # same limb, same place, back within this long: one grip
    #                          (the hand was hidden for a moment, not let go)
    frag_dist: float = 0.06  # body lengths: "same place" (~7 cm; or the same detected hold)
    frag_dist_hidden: float = 0.25  # ... when either piece was hidden (visibility < 0.5)
    frag_aba_s: float = 2.5  # A, then a short B elsewhere, then A again within this long:
    frag_b_max_s: float = 1.5  # ... B (this short) was the pose losing the limb, if it sat
    #                            on the other hand (the model put both wrists on one) or
    #                            was hidden: one grip on A
    missed_hold_s: float = 1.0  # a hand still this long above the shoulder, off any
    #                             detected hold, the body moving: a missed hold
    ground_band: float = 0.3  # body lengths: a foot off any hold, no hands on holds, is on
    #                           the floor only this close to the lowest foot position


def median_filter(p: np.ndarray, k: int) -> np.ndarray:
    """Centred running median of (n, 2) positions over k frames, ignoring NaN (a frame
    stays NaN only if its whole window is). Removes single-frame pose spikes while
    keeping the moment a limb leaves a hold sharp."""
    if k <= 1:
        return p.copy()
    n, h = len(p), k // 2
    out = np.full_like(p, np.nan, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for i in range(n):
            out[i] = np.nanmedian(p[max(0, i - h): i + h + 1], axis=0)
    return out


def detect_anchors(p: np.ndarray, t: np.ndarray, body_px: float, limb: str,
                   prm: WallParams | None = None) -> list[Anchor]:
    """Anchors of one limb from its (n, 2) wall positions.

    1. Seed: frames whose centred 0.5 s window is still (80 % of the positions within
       `still_tol` of the window median). A seed run is split where the window median
       moves on by more than `still_tol` (a slow slide from one hold to the next).
    2. Grow each seed run while the limb stays within `hold_tol` of the anchor
       position; excursions (jitter, a missing frame) up to `max_gap_s` are bridged if
       the limb comes back and never went further than `drift_tol`.
    3. Merge anchors of the limb at the same place separated by <= `merge_gap_s`.
    """
    prm = prm or WallParams()
    n = len(p)
    fps = 1.0 / float(np.median(np.diff(t))) if n > 1 else 30.0
    p = median_filter(p, max(1, int(round(prm.median_s * fps))))
    k = max(2, int(round(prm.window_s * fps)))
    still, drift = prm.still_tol * body_px, prm.drift_tol * body_px
    stay = prm.hold_tol * body_px
    max_gap = max(1, int(prm.max_gap_s * fps))
    ok = np.isfinite(p).all(axis=1)
    disp = np.full(n, np.nan)
    med = np.full((n, 2), np.nan)
    h = k // 2
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i - h + k)
        m = ok[lo:hi]
        if m.sum() < 0.7 * k:
            continue
        w = p[lo:hi][m]
        med[i] = np.median(w, axis=0)
        disp[i] = np.percentile(np.linalg.norm(w - med[i], axis=1), 80)
    seed = np.nan_to_num(disp, nan=np.inf) < still
    segments = []
    for a, b in runs(seed):
        start, ref = a, med[a]
        for i in range(a + 1, b):
            if np.linalg.norm(med[i] - ref) > still:
                segments.append((start, i))
                start, ref = i, med[i]
        segments.append((start, b))

    taken = np.zeros(n, bool)
    anchors: list[Anchor] = []
    for a, b in sorted(segments, key=lambda r: r[0] - r[1]):  # longest first
        free = runs(~taken[a:b])
        if not free:
            continue
        x, y = max(free, key=lambda r: r[1] - r[0])
        a, b = a + x, a + y
        pos = np.median(p[a:b][ok[a:b]], axis=0)
        dist = np.where(ok, np.linalg.norm(p - pos, axis=1), np.nan)

        def grow(i, step, dist=dist):
            last, gap = i - step, 0
            while 0 <= i < n and not taken[i]:
                d = dist[i]
                if np.isfinite(d) and d < stay:
                    last, gap = i, 0
                else:
                    gap += 1
                    if gap > max_gap or (np.isfinite(d) and d > drift):
                        break
                i += step
            return last

        s_ = grow(a - 1, -1) if a > 0 else a
        e_ = grow(b, 1) + 1 if b < n else b
        s_, e_ = min(s_, a), max(e_, b)
        taken[s_:e_] = True
        anchors.append(_make(limb, p, s_, e_))
    anchors.sort(key=lambda x: x.start)
    merged: list[Anchor] = []
    for an in anchors:
        if merged:
            prev = merged[-1]
            if (t[an.start] - t[prev.end - 1] <= prm.merge_gap_s
                    and np.linalg.norm(an.pos - prev.pos) < still):
                merged[-1] = _make(limb, p, prev.start, an.end)
                continue
        merged.append(an)
    return [an for an in merged if t[an.end - 1] - t[an.start] >= prm.window_s * 0.9]


def _make(limb, p, s, e) -> Anchor:
    seg = p[s:e]
    seg = seg[np.isfinite(seg).all(axis=1)]
    pos = np.median(seg, axis=0)
    spread = float(np.percentile(np.linalg.norm(seg - pos, axis=1), 80)) if len(seg) else 0.0
    return Anchor(limb, s, e, pos, spread)


def _on_mask(anchors: list[Anchor], n: int) -> np.ndarray:
    m = np.zeros(n, bool)
    for an in anchors:
        m[an.start:an.end] = True
    return m


def floor_mask(wall_image, tol: float = 25.0, flat_sd: float = 5.0,
               max_objects: int = 8, object_frac: float = 1e-3, max_holds: int = 5,
               hold_boxes=None) -> np.ndarray | None:
    """Where the floor is in the wall image (bool, wall-image pixels), or None.

    The bottom band of the wall image (lowest 8 % of the rows seen) shows the floor
    when the camera sees it — possibly several surfaces (concrete, a mat, a towel). Its
    main colours (k-means, k = 3, clusters with >= 15 % of the band) are the floor
    colours; the floor is every pixel of those colours that is *connected* to the bottom
    band. Connectivity keeps a light wall panel higher up from counting as floor.

    A region that is really a wall panel with holds (the camera didn't see the floor)
    is dropped: it has `max_holds` or more detected holds in it (`hold_boxes`, when the
    hold detector ran), or `max_objects` or more separate hold-sized things inside it
    (each at least `object_frac` of the image). Only hold-sized things count: a textured
    floor (concrete, a speckled mat) has hundreds of small patches that fail the
    flatness test, and counting those turned real floors into "walls"."""
    img = wall_image.image
    seen = img.max(axis=2) > 0
    rows = np.flatnonzero(seen.any(axis=1))
    if len(rows) < 20:
        return None
    band = np.zeros_like(seen)
    band[rows[-max(5, len(rows) * 8 // 100):]] = True
    band &= seen
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    px = lab[band]
    if len(px) < 50:
        return None
    k = 3
    cv2.setRNGSeed(0)  # k-means starts at random: same wall image, same floor
    _, lbl, centres = cv2.kmeans(px, k, None,
                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0),
                                 3, cv2.KMEANS_PP_CENTERS)
    share = np.bincount(lbl.ravel(), minlength=k) / len(px)
    colours = centres[share >= 0.15]
    near = np.zeros_like(seen)
    for c in colours:
        near |= np.linalg.norm(lab - c, axis=2) < tol
    # floors and mats are flat; walls have texture (holds, bolt holes, grain), which
    # stops the floor from "leaking" up a wall panel of a similar colour
    L = lab[..., 0]
    m = cv2.blur(L, (9, 9))
    flat = np.sqrt(np.maximum(cv2.blur(L * L, (9, 9)) - m * m, 0)) < flat_sd
    near &= seen & flat
    n, comp = cv2.connectedComponents(near.astype(np.uint8), connectivity=8)
    keep = np.unique(comp[band & near])
    keep = keep[keep > 0]
    if not len(keep):
        return None
    mask = cv2.morphologyEx(np.isin(comp, keep).astype(np.uint8), cv2.MORPH_CLOSE,
                            np.ones((5, 5), np.uint8)) > 0
    # things lying on the floor (a towel, a chalk bag) are floor too: fill the holes,
    # i.e. non-floor regions that don't reach the top of what the camera saw
    top = np.zeros_like(seen)
    top[rows[: max(5, len(rows) * 8 // 100)]] = True

    def holes_of(mask):
        _, comp = cv2.connectedComponents((~mask & seen).astype(np.uint8), connectivity=4)
        open_ = np.unique(comp[top & ~mask & seen])
        return comp, (comp > 0) & ~np.isin(comp, open_)

    # ... but a region full of separate objects is a wall panel with holds, not a floor
    # (a camera close to the wall sees no floor: the bottom band is wall)
    comp, holes = holes_of(mask)
    n_m, mcomp = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    min_area = max(20, int(object_frac * seen.sum()))
    count = np.zeros(n_m, int)
    for h in np.unique(comp[holes]):
        hm = comp == h
        if hm.sum() < min_area:
            continue
        around = mcomp[cv2.dilate(hm.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0]
        around = around[around > 0]
        if len(around):
            count[np.bincount(around).argmax()] += 1
    wall_like = count >= max_objects
    if hold_boxes is not None and len(hold_boxes):  # detected holds: the surer test
        from climbvision.engine.camera import apply_h

        b = np.asarray(hold_boxes.boxes, float)
        c = np.round(apply_h(wall_image.T, np.c_[(b[:, 0] + b[:, 2]) / 2,
                                                   (b[:, 1] + b[:, 3]) / 2])).astype(int)
        hh, ww = mask.shape
        ok = (c[:, 0] >= 0) & (c[:, 0] < ww) & (c[:, 1] >= 0) & (c[:, 1] < hh)
        filled = cv2.connectedComponents((mask | holes).astype(np.uint8), connectivity=8)[1]
        # a hold sits in a hole of the region (or on its edge): count it for the region
        # around it
        at = filled[c[ok, 1], c[ok, 0]]
        for lab_ in np.unique(at[at > 0]):
            n_in = int((at == lab_).sum())
            if n_in >= max_holds:
                wall_like |= np.isin(np.arange(n_m),
                                     np.unique(mcomp[(filled == lab_) & mask]))
    wall_like = np.flatnonzero(wall_like & (np.arange(n_m) > 0))
    if len(wall_like):
        mask &= ~np.isin(mcomp, wall_like)
        if not mask.any():
            return None
        comp, holes = holes_of(mask)
    return mask | holes


def merge_fragments(anchors: list[Anchor], other: np.ndarray, t: np.ndarray, body_px: float,
                    prm: WallParams, vis: np.ndarray | None = None,
                    own: np.ndarray | None = None) -> list[Anchor]:
    """Join the pieces of one grip that the pose broke apart. A hand (or foot) on a hold
    is often hidden for a moment behind the body, the other arm or a volume: the pose
    then loses it, guesses it elsewhere or puts it on the other hand, and one grip reads
    as several short ones (annotated: most brief "touches" were exactly that).

    - *A, gap, A*: same limb back at the same place (`frag_dist` body lengths, or the
      same detected hold; `frag_dist_hidden` when either piece was hidden, visibility
      < 0.5) within `frag_gap_s`, and in the gap the limb was hidden or never got
      further than `drift_tol` from A (`own`: the limb's positions): one anchor. A limb
      seen lifting off and coming back (a re-grip, a foot re-placed, feet cut) stays two.
    - *A, B, A*: a short B (<= `frag_b_max_s`) between two pieces at the same place
      within `frag_aba_s`, where B sat on the other hand (both wrists put on one spot) or
      was hidden: B was the pose losing the limb; one anchor on A. A brief B (under
      `window_s`, kept only because it sat on a detected hold) that the hand left to go
      straight back is a tap, not two moves: also one anchor on A.

    The relaxed rules (hidden pieces, A-B-A) are for hands, which the annotations
    checked; feet are hidden far more often (drop knees, heel hooks) and step between
    footholds a few cm apart, so a foot is only joined back at the same detected hold or
    the same spot. Ground anchors are never joined. The joined anchor keeps the longer
    piece's place."""
    if len(anchors) < 2:
        return anchors
    seq = sorted(anchors, key=lambda a: a.start)
    hand = seq[0].limb.endswith("hand")

    def dur(a):
        return float(t[a.end - 1] - t[a.start])

    def hidden(a):
        if vis is None or not hand:
            return False
        v = vis[a.start:a.end]
        return bool(np.isfinite(v).any() and np.nanmean(v) < 0.5)

    def same_place(a, b):
        if a.box >= 0 and a.box == b.box:
            return True
        lim = prm.frag_dist_hidden if (hidden(a) or hidden(b)) else prm.frag_dist
        return float(np.linalg.norm(a.pos - b.pos)) <= lim * body_px

    def on_other(b):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            d = np.nanmedian(np.linalg.norm(other[b.start:b.end] - b.pos, axis=1))
        return bool(np.isfinite(d) and d <= prm.frag_dist * body_px)

    def stayed(a, b):
        """In the gap between a and b the limb was hidden, or never really left."""
        if own is None:
            return True
        g = slice(a.end, b.start)
        if b.start <= a.end:
            return True
        if vis is not None and hand:
            v = vis[g]
            if np.isfinite(v).any() and np.nanmean(v) < 0.5:
                return True
        d = np.linalg.norm(own[g] - a.pos, axis=1)
        d = d[np.isfinite(d)]
        return not len(d) or float(d.max()) <= prm.drift_tol * body_px

    def join(a, b):
        keep = a if dur(a) >= dur(b) else b
        return replace(keep, start=a.start, end=max(a.end, b.end),
                       spread_px=max(a.spread_px, b.spread_px))

    changed = True
    while changed:
        changed = False
        for i in range(len(seq) - 1):  # A, gap, A
            a, b = seq[i], seq[i + 1]
            if a.ground or b.ground:
                continue
            if t[b.start] - t[a.end - 1] <= prm.frag_gap_s and same_place(a, b) \
                    and stayed(a, b):
                seq[i:i + 2] = [join(a, b)]
                changed = True
                break
        if changed:
            continue
        for i in range(len(seq) - 2 if hand else 0):  # A, B, A (hands)
            a, b, c = seq[i:i + 3]
            if a.ground or b.ground or c.ground:
                continue
            if (t[c.start] - t[a.end - 1] <= prm.frag_aba_s and dur(b) <= prm.frag_b_max_s
                    and same_place(a, c)
                    and (on_other(b) or hidden(b) or dur(b) < prm.window_s)):
                seq[i:i + 3] = [join(a, c)]
                changed = True
                break
    return seq


def mark_standing(anchors: dict[str, list[Anchor]], t: np.ndarray, body_px: float,
                  prm: WallParams) -> None:
    """Second ground cue, from the pose: at the very start or end of the clip both feet
    anchored at the same moment, at the same height (within `level_tol`), and lower
    than every foot anchor of the climb in between = standing on the ground."""
    feet = [a for f in ("l_foot", "r_foot") for a in anchors[f]]
    if len(feet) < 3:
        return
    t0, t1 = t[0] + prm.stand_edge_s, t[-1] - prm.stand_edge_s
    for a in anchors["l_foot"]:
        for b in anchors["r_foot"]:
            together = min(a.end, b.end) - max(a.start, b.start) > 0
            at_edge = (min(a.start, b.start) <= np.searchsorted(t, t0)
                       or max(a.end, b.end) >= np.searchsorted(t, t1))
            level = abs(a.pos[1] - b.pos[1]) < prm.level_tol * body_px
            if not (together and at_edge and level):
                continue
            y = max(a.pos[1], b.pos[1])
            others = [c.pos[1] for c in feet if c is not a and c is not b]
            if not others or y >= max(others) - prm.level_tol * body_px:
                a.ground = b.ground = True


def mark_ground(anchors: dict[str, list[Anchor]], wall_image, body_px: float,
                prm: WallParams, t: np.ndarray | None = None, hold_boxes=None) -> bool:
    """Mark foot anchors on the ground (see module doc): the area just below the toe is
    floor (`floor_mask`), or (`mark_standing`) both feet stand level and lowest at the
    start or end. Returns whether a floor was found in the wall image."""
    from climbvision.engine.camera import apply_h

    mask = (floor_mask(wall_image, prm.floor_lab_tol, hold_boxes=hold_boxes)
            if wall_image is not None else None)
    if mask is None:
        return False
    b = body_px * wall_image.T[0, 0]  # body length in wall-image pixels
    h, w = mask.shape
    for a in (a for f in ("l_foot", "r_foot") for a in anchors[f]):
        x, y = apply_h(wall_image.T, a.pos)
        y0, y1 = int(np.clip(y + 0.02 * b, 0, h)), int(np.clip(y + 0.10 * b, 0, h))
        x0, x1 = int(np.clip(x - 0.05 * b, 0, w)), int(np.clip(x + 0.05 * b, 0, w))
        patch = mask[y0:y1, x0:x1]
        if patch.size >= 10:
            a.ground = float(patch.mean()) > prm.floor_frac
    if t is not None:
        mark_standing(anchors, t, body_px, prm)
    return True


def cluster_holds(anchors: list[Anchor], radius_px: float) -> list[Hold]:
    """Anchors -> holds, numbered in the order first used. With detected holds, one hold
    per detected box; without, greedy clustering of anchor positions (anchors closer
    than radius_px = one hold)."""
    holds: list[Hold] = []
    if anchors and any(a.box >= 0 for a in anchors):
        # detected holds: one hold per box; a hand hold the detector missed (box -1,
        # see compute_wall_support) joins the nearest hold within radius_px, else is new
        by_box: dict[int, Hold] = {}
        for an in sorted(anchors, key=lambda a: a.start):
            if an.box >= 0:
                h = by_box.get(an.box)
                if h is None:
                    h = Hold(len(holds), an.pos.copy(), box=an.box)
                    by_box[an.box] = h
                    holds.append(h)
            else:
                near = [(float(np.linalg.norm(x.pos - an.pos)), x) for x in holds]
                near = [nx for nx in near if nx[0] < radius_px]
                h = min(near, key=lambda nx: nx[0])[1] if near else None
                if h is None:
                    h = Hold(len(holds), an.pos.copy())
                    holds.append(h)
            h.anchors.append(an)
            h.limbs.add(an.limb)
            h.pos = np.mean([a.pos for a in h.anchors], axis=0)
            an.hold = h.id
        return holds
    for an in sorted(anchors, key=lambda a: a.start):
        best, dist = None, radius_px
        for h in holds:
            d = float(np.linalg.norm(h.pos - an.pos))
            if d < dist:
                best, dist = h, d
        if best is None:
            best = Hold(len(holds), an.pos.copy())
            holds.append(best)
        best.anchors.append(an)
        best.limbs.add(an.limb)
        best.pos = np.mean([a.pos for a in best.anchors], axis=0)
        an.hold = best.id
    return holds


@dataclass
class WallSupport:
    points: dict[str, np.ndarray]  # limb -> (n, 2) wall px
    anchors: dict[str, list[Anchor]]  # limb -> anchors (incl. ground)
    holds: list[Hold]
    on_hold: dict[str, np.ndarray]  # limb -> (n,) bool
    on_ground: np.ndarray  # (n,) any foot on the ground (or below it)
    on_wall: np.ndarray  # (n,) pose + camera ok, nothing on the ground
    ground_y: float | None  # wall y of the ground anchors (median), if any
    com: np.ndarray  # (n, 2) wall px
    n_contacts: np.ndarray  # (n,) limbs on holds
    com_over_feet_cm: np.ndarray  # (n,) + COM inside the anchored feet span (x)
    polygon_margin_cm: np.ndarray  # (n,) + COM inside the hull of anchored limbs
    px_per_m: float
    body_px: float
    labels: list[str]
    floor_seen: bool = False
    min_move_px: float = 0.0
    on_smear: dict | None = None  # limb -> (n,) bool: resting on the wall off any hold
    hold_boxes: object = None  # HoldBoxes from the detector, or None
    holds_source: str = "rests"  # "detected" (holds from the wall image) or "rests"
    metric: WallMetric | None = None  # wall px -> metres (perspective corrected)
    com_m: np.ndarray | None = None  # (n, 2) COM in metres, x right, y up
    com_v: np.ndarray | None = None  # (n, 2) COM velocity, m/s (x right, y up)
    com_a: np.ndarray | None = None  # (n, 2) COM acceleration, m/s^2
    window: np.ndarray | None = None  # (n,) the climb (engine/climb.py), or None

    @property
    def body_m(self) -> float:
        return self.body_px / self.px_per_m

    def to_m(self, p: np.ndarray) -> np.ndarray:
        """Wall px -> metres (x right, y up)."""
        if self.metric is not None:
            return self.metric.to_m(p)
        q = np.asarray(p, float) / self.px_per_m
        return q * np.array([1.0, -1.0])

    def height_m(self, p: np.ndarray) -> np.ndarray:
        """Height above the floor (when the climber stood on it in the clip), else above
        the lowest hold."""
        if self.metric is not None and self.metric.ground_y_m is not None:
            return self.metric.height_m(p)
        y = self.to_m(p)[..., 1]
        low = min((float(self.to_m(h.pos)[1]) for h in self.holds), default=0.0)
        return y - low

    @property
    def heights_above_floor(self) -> bool:
        return self.metric is not None and self.metric.ground_y_m is not None

    def hold_xy_m(self, h: Hold) -> tuple[float, float]:
        """Hold position in metres: x right from the first hold, y up from the ground
        (or from the lowest hold when there is no ground)."""
        x0 = float(self.to_m(self.holds[0].pos)[0]) if self.holds else 0.0
        return (float(self.to_m(h.pos)[0]) - x0, float(self.height_m(h.pos)))


def _signed_hull_distance(hull: np.ndarray, p: np.ndarray) -> float:
    return float(cv2.pointPolygonTest(hull.astype(np.float32).reshape(-1, 1, 2),
                                      (float(p[0]), float(p[1])), True))


def contact_label(on: dict[str, bool]) -> str:
    hands = [s for s in ("l_hand", "r_hand") if on[s]]
    feet = [s for s in ("l_foot", "r_foot") if on[s]]
    parts = []
    if len(hands) == 2:
        parts.append("2 hands")
    elif hands:
        parts.append(LABEL[hands[0]])
    if len(feet) == 2:
        parts.append("2 feet")
    elif feet:
        parts.append(LABEL[feet[0]])
    return " + ".join(parts) or "no holds"


def compute_wall_support(wall_points: np.ndarray, t: np.ndarray, body_px: float,
                         px_per_m: float, sex: str = "male", camera_ok: np.ndarray | None = None,
                         wall_image=None, prm: WallParams | None = None,
                         body_m: float = ASSUMED_BODY_M,
                         correct_perspective: bool = True, hold_boxes=None,
                         window: np.ndarray | None = None,
                         visibility: np.ndarray | None = None) -> WallSupport:
    """wall_points: (n, N_JOINTS, 2) joints in wall coordinates (Clip.wall).
    `window`: (n,) bool, the climb (engine/climb.py): frames outside it are not on the
    wall, and their anchors count for nothing (but standing on the ground just before or
    after the climb still sets the floor). `visibility`: (n, N_JOINTS) joint visibility
    (hidden limbs: their anchors are joined more readily, `merge_fragments`)."""
    prm = prm or WallParams()
    n = len(t)
    pts = limb_points(wall_points)
    detected_holds = hold_boxes is not None and len(hold_boxes) > 0
    hand_prm = replace(prm, window_s=prm.hand_window_s) if detected_holds else prm
    anchors = {limb: detect_anchors(pts[limb], t, body_px, limb,
                                    hand_prm if limb.endswith("hand") else prm)
               for limb in LIMBS}
    if detected_holds:
        margin = prm.hold_margin * body_px
        for limb in LIMBS:
            for an in anchors[limb]:
                an.box = hold_boxes.containing(an.pos, margin)
            if limb.endswith("hand"):  # brief stops count only on a detected hold
                anchors[limb] = [an for an in anchors[limb] if an.box >= 0 or
                                 t[an.end - 1] - t[an.start] >= prm.window_s]
    floor_seen = mark_ground(anchors, wall_image, body_px, prm, t, hold_boxes)
    if window is not None:
        window = np.asarray(window, bool)
        idx = np.flatnonzero(window)
        near = np.zeros(n, bool)  # the floor counts just before and after the climb
        if len(idx):
            near[(t >= t[idx[0]] - prm.ground_near_s) & (t <= t[idx[-1]] + prm.ground_near_s)] \
                = True
        for limb in LIMBS:
            anchors[limb] = [a for a in anchors[limb]
                             if (near if a.ground else window)[a.start:a.end].any()]
    vis = None
    if visibility is not None:
        vis = {"l_hand": visibility[:, J.L_WRIST], "r_hand": visibility[:, J.R_WRIST],
               "l_foot": np.fmax(visibility[:, J.L_ANKLE], visibility[:, J.L_TOE]),
               "r_foot": np.fmax(visibility[:, J.R_ANKLE], visibility[:, J.R_TOE])}
    other = {"l_hand": "r_hand", "r_hand": "l_hand", "l_foot": "r_foot", "r_foot": "l_foot"}
    for limb in LIMBS:
        anchors[limb] = merge_fragments(anchors[limb], pts[other[limb]], t, body_px, prm,
                                        None if vis is None else vis[limb], pts[limb])
    detected = np.isfinite(wall_points[:, J.L_SHOULDER, 0])
    ok = detected if camera_ok is None else detected & camera_ok
    foot_hold = np.zeros(n, bool)
    foot_ground = np.zeros(n, bool)
    for f in ("l_foot", "r_foot"):
        for an in anchors[f]:
            (foot_ground if an.ground else foot_hold)[an.start:an.end] = True
    # hands above the shoulders on a hold also mean climbing: hanging with the feet off
    # (campusing, a cut, kicking for a heel hook) is on the wall, not on the ground
    hand_hang = np.zeros(n, bool)
    for h, sh in (("l_hand", J.L_SHOULDER), ("r_hand", J.R_SHOULDER)):
        for an in anchors[h]:
            seg = slice(an.start, an.end)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                above = np.nanmedian(pts[h][seg, 1]) < np.nanmedian(wall_points[seg, sh, 1])
            if above:
                hand_hang[seg] = True
    climbing = (foot_hold | hand_hang) & ~foot_ground
    if window is not None:
        climbing &= window
    on_wall = np.zeros(n, bool)
    if climbing.any():
        a, b = np.flatnonzero(climbing)[[0, -1]]
        on_wall[a:b + 1] = True
    on_wall &= ok & ~foot_ground
    if window is not None:
        on_wall &= window
    on_ground = ~on_wall & detected if floor_seen else foot_ground
    # hands resting while on the ground are not on holds
    for h in ("l_hand", "r_hand"):
        sh = wall_points[:, J.L_SHOULDER if h == "l_hand" else J.R_SHOULDER, 1]
        for an in anchors[h]:
            seg = slice(an.start, an.end)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                low = np.nanmedian(pts[h][seg, 1]) > np.nanmedian(sh[seg])
            if on_ground[seg].mean() > 0.5 and low:
                an.ground = True
    # a rest counts as a hold only on a detected hold; elsewhere it is a smear if the
    # body moved over it (friction held it: it touches), else a flag (no contact)
    com = whole_body_com(wall_points, sex)  # type: ignore[arg-type]
    if detected_holds:
        proof = prm.smear_proof_m * body_px / body_m
        for limb in LIMBS:
            for an in anchors[limb]:
                if an.ground:
                    continue
                if an.box < 0:
                    moved = _com_moved(com[an.start:an.end])
                    an.smear = bool(moved >= proof)
                    an.flag = not an.smear
                    # a hand held still above the shoulder for a while, with the body
                    # moving under it, is holding something: a hold the detector missed
                    # (a volume, a big sloper, the top). Hands don't smear up there
                    if an.smear and limb.endswith("hand") and \
                            t[an.end - 1] - t[an.start] >= prm.missed_hold_s:
                        sh = J.L_SHOULDER if limb == "l_hand" else J.R_SHOULDER
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", RuntimeWarning)
                            above = np.nanmedian(pts[limb][an.start:an.end, 1]) < \
                                np.nanmedian(wall_points[an.start:an.end, sh, 1])
                        if above:
                            an.smear = False
                            an.missed = True
        # a foot "smearing" with no hand on a hold is standing on the floor (start or end of
        # a clip whose floor the camera doesn't show; a mat the floor test missed): nobody
        # climbs on a single smear with no hands
        hands_on = _on_mask([a for h in ("l_hand", "r_hand") for a in anchors[h]
                             if a.on_hold], n)
        # ... but only near the bottom: high on the wall it is a smear with the hands on
        # something the detector missed (a volume, the top), not the floor
        feet_y = [a.pos[1] for f in ("l_foot", "r_foot") for a in anchors[f]]
        low_y = (max(feet_y) - prm.ground_band * body_px) if feet_y else -np.inf
        for f in ("l_foot", "r_foot"):
            for an in anchors[f]:
                if (an.smear or an.flag) and hands_on[an.start:an.end].mean() < 0.4 \
                        and an.pos[1] >= low_y:
                    an.smear = an.flag = False
                    an.ground = True
                    on_wall[an.start:an.end] = False
                    on_ground[an.start:an.end] = True
    hold_anchors = [an for limb in LIMBS for an in anchors[limb] if an.on_hold]
    holds = cluster_holds(hold_anchors, prm.hold_radius * body_px)
    on_hold = {limb: _on_mask([a for a in anchors[limb] if a.on_hold], n) for limb in LIMBS}
    on_smear = {limb: _on_mask([a for a in anchors[limb] if a.smear], n) for limb in LIMBS}
    n_contacts = sum(on_hold[limb].astype(int) for limb in LIMBS)
    # metres on the wall: perspective corrected with the body as the ruler (metric.py)
    ground_px = _floor_points(anchors, window, t)
    metric = fit_wall_metric(wall_points, on_wall, body_m, ground_px,
                             correct=correct_perspective, px_per_m=px_per_m)
    com_m = metric.to_m(com)
    com_v, com_a = derivatives(com_m, t)
    feet_m = np.full(n, np.nan)
    poly_m = np.full(n, np.nan)
    pos_at = {limb: np.full((n, 2), np.nan) for limb in LIMBS}
    # the base of support: limbs on holds and smearing limbs
    contact = {limb: on_hold[limb] | on_smear[limb] for limb in LIMBS}
    for limb in LIMBS:
        for an in anchors[limb]:
            if an.on_wall_contact:
                pos_at[limb][an.start:an.end] = metric.to_m(an.pos)
    for i in range(n):
        if not np.isfinite(com_m[i]).all():
            continue
        fx = [pos_at[f][i, 0] for f in ("l_foot", "r_foot") if contact[f][i]]
        if fx:
            lo, hi = min(fx) - FOOT_RADIUS_M, max(fx) + FOOT_RADIUS_M
            feet_m[i] = min(com_m[i, 0] - lo, hi - com_m[i, 0]) * 100
        hull_pts = [pos_at[limb][i] for limb in LIMBS if contact[limb][i]]
        if len(hull_pts) >= 3:
            hull = cv2.convexHull(np.array(hull_pts, np.float32) * 1000)  # mm
            if cv2.contourArea(hull) > 100.0:
                poly_m[i] = _signed_hull_distance(hull, com_m[i] * 1000) / 10
    labels = [_label_with_smears(on_hold, on_smear, i)
              if on_wall[i] else ("on the ground" if on_ground[i] else
                                  "outside the climb" if window is not None else "–")
              for i in range(n)]
    ground_y = float(np.median(ground_px[:, 1])) if len(ground_px) else None
    return WallSupport(pts, anchors, holds, on_hold, on_ground, on_wall, ground_y, com,
                       n_contacts, feet_m, poly_m, metric.s * metric.per_m, body_px, labels,
                       floor_seen, prm.min_move * body_px, metric=metric, com_m=com_m,
                       com_v=com_v, com_a=com_a, on_smear=on_smear, hold_boxes=hold_boxes,
                       holds_source="detected" if detected_holds else "rests",
                       window=window)


def _floor_points(anchors: dict[str, list[Anchor]], window: np.ndarray | None,
                  t: np.ndarray, last_s: float = 0.5) -> np.ndarray:
    """Wall px of the feet that set the floor height. Everything is measured on the wall
    plane, and a foot standing on the mat in front of the wall projects below the real
    floor (nearer the camera; GUIDE §6.14b). The feet that push off for the climb stand
    closest to the wall: with a climb window, the floor comes from the ground contacts
    that end last before the climb (within `last_s` of the latest), else from the
    first ones after it (the landing, usually further out), else from all."""
    ground = [a for f in ("l_foot", "r_foot") for a in anchors[f] if a.ground]
    if window is not None and window.any() and ground:
        idx = np.flatnonzero(window)
        a0, a1 = idx[0], idx[-1]
        pre = [a for a in ground if a.end <= a0 + 1]
        post = [a for a in ground if a.start >= a1 - 1]
        if pre:
            e = max(t[a.end - 1] for a in pre)
            ground = [a for a in pre if t[a.end - 1] >= e - last_s]
        elif post:
            s = min(t[a.start] for a in post)
            ground = [a for a in post if t[a.start] <= s + last_s]
    return np.array([a.pos for a in ground]).reshape(-1, 2)


def _com_moved(com: np.ndarray) -> float:
    """How far the COM got from where it was when the limb settled (wall px): the
    largest distance from its first steady position (median of the first 3 frames)."""
    ok = np.isfinite(com).all(axis=1)
    if ok.sum() < 3:
        return 0.0
    c = com[ok]
    start = np.median(c[:3], axis=0)
    return float(np.max(np.linalg.norm(c - start, axis=1)))


def _label_with_smears(on_hold: dict, on_smear: dict, i: int) -> str:
    lab = contact_label({limb: bool(on_hold[limb][i]) for limb in LIMBS})
    sm = [LABEL[limb] for limb in LIMBS if on_smear[limb][i]]
    if sm:
        lab = ("" if lab == "no holds" else lab + " + ") + " + ".join(s + " smear" for s in sm)
    return lab


def moves(ws: WallSupport, t: np.ndarray) -> list[dict]:
    """A move = a limb leaving one hold and next anchoring on a different hold."""
    out = []
    for limb in LIMBS:
        seq = [a for a in ws.anchors[limb] if a.on_hold]
        for a, b in zip(seq, seq[1:], strict=False):
            if a.hold != b.hold and np.linalg.norm(b.pos - a.pos) >= ws.min_move_px:
                d = float(np.linalg.norm(ws.to_m(b.pos) - ws.to_m(a.pos)))
                out.append({"limb": limb, "from_hold": a.hold, "to_hold": b.hold,
                            "release_s": round(float(t[a.end - 1]), 2),
                            "catch_s": round(float(t[b.start]), 2),
                            "time_s": round(float(t[b.start] - t[a.end - 1]), 2),
                            "distance_cm": round(100 * d, 1)})
    return sorted(out, key=lambda m: m["catch_s"])


def wall_summary(ws: WallSupport, t: np.ndarray) -> dict:
    n = len(t)
    dt = np.gradient(t) if n > 1 else np.ones(n)
    on = ws.on_wall
    wall_s = float(dt[on].sum())
    out: dict = {"on_wall_s": round(wall_s, 2),
                 "holds_from": ("detected in the wall image" if ws.holds_source == "detected"
                                else "where limbs rested (no hold detector)"),
                 "ground": ("stood on" if ws.ground_y is not None else
                            "in view, not stood on" if ws.floor_seen else "not seen"),
                 "holds_used": len(ws.holds),
                 "hand_holds": sum(1 for h in ws.holds if h.limbs & {"l_hand", "r_hand"}),
                 "foot_holds": sum(1 for h in ws.holds if h.limbs & {"l_foot", "r_foot"})}
    if on.any():
        out["on_hold_pct"] = {LABEL[limb]: round(100 * float(ws.on_hold[limb][on].mean()), 1)
                              for limb in LIMBS}
        out["contacts_pct"] = {str(k): round(100 * float(np.mean(ws.n_contacts[on] == k)), 1)
                               for k in (4, 3, 2, 1, 0)}
        tc: dict[str, float] = {}
        for lab, d, o in zip(ws.labels, dt, on, strict=True):
            if o:
                tc[lab] = tc.get(lab, 0.0) + float(d)
        out["contact_time_s"] = {k: round(v, 2) for k, v in
                                 sorted(tc.items(), key=lambda kv: -kv[1])}
        cy = ws.com[on, 1]
        if np.isfinite(cy).any():
            ym = ws.com_m[on, 1] if ws.com_m is not None else -cy / ws.px_per_m
            out["com_height_gain_m"] = round(float(np.nanmax(ym) - np.nanmin(ym)), 2)
            if ws.heights_above_floor:
                out["com_top_above_floor_m"] = round(float(np.nanmax(ws.height_m(ws.com[on]))), 2)
        for key, arr in (("com_over_feet_cm", ws.com_over_feet_cm),
                         ("polygon_margin_cm", ws.polygon_margin_cm)):
            v = arr[on]
            v = v[np.isfinite(v)]
            if len(v):
                out[key] = {"median": round(float(np.median(v)), 1),
                            "min": round(float(v.min()), 1),
                            "pct_inside": round(100 * float(np.mean(v > 0)), 1)}
    mv = moves(ws, t)
    out["moves"] = {"hand": sum(m["limb"].endswith("hand") for m in mv),
                    "foot": sum(m["limb"].endswith("foot") for m in mv)}
    if mv:
        out["moves"]["time_s_median"] = round(float(np.median([m["time_s"] for m in mv])), 2)
    out["holds"] = []
    for h in ws.holds:
        x, y = ws.hold_xy_m(h)
        out["holds"].append({
            "id": h.id, "x_m": round(x, 2), "y_m": round(y, 2),
            "limbs": sorted(LABEL[limb] for limb in h.limbs),
            "first_s": round(float(t[min(a.start for a in h.anchors)]), 2),
            "time_s": round(float(sum(t[a.end - 1] - t[a.start] for a in h.anchors)), 2)})
    if ws.on_smear is not None and ws.holds_source == "detected":
        sm = {LABEL[limb]: round(float(dt[ws.on_smear[limb] & on].sum()), 1) for limb in LIMBS}
        out["smears"] = {"count": sum(a.smear for limb in LIMBS for a in ws.anchors[limb]),
                         "time_s": {k: v for k, v in sm.items() if v > 0}}
        out["flags"] = {"count": sum(a.flag for limb in LIMBS for a in ws.anchors[limb]),
                        "note": "still limbs off any hold that the body didn't move over: "
                                "not counted as contacts"}
        if ws.hold_boxes is not None:
            out["holds_detected"] = len(ws.hold_boxes)
        missed = sum(1 for h in ws.holds if h.box < 0)
        if missed:
            out["holds_missed_by_detector"] = missed
    warn = []
    if wall_s < 0.5 * (t[-1] - t[0]) if n > 1 else False:
        warn.append(f"on the wall only {wall_s:.1f} s of {t[-1] - t[0]:.1f} s")
    if warn:
        out["warnings"] = warn
    return out
