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
wall without the climber, camera layer), the area just below the toe looks like the floor
— the colour of the bottom of the wall image, where the floor is — and it is among the
lowest foot positions of the clip (within `ground_band` body lengths). Footholds have
wall below them. The climb is the stretch from the first to the last moment with a foot
on a hold and none on the ground; before/after it the climber is on the ground. Without a
wall image (no camera layer) or without floor in view, no ground is assumed.

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
from dataclasses import dataclass, field

import cv2
import numpy as np

from climbvision.engine.biomech import whole_body_com
from climbvision.engine.events import runs
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
    hold: int = -1  # hold id (-1 = none: ground or resting)
    ground: bool = False


@dataclass
class Hold:
    id: int
    pos: np.ndarray  # (2,) wall px, mean of its anchors
    limbs: set = field(default_factory=set)
    anchors: list = field(default_factory=list)


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
    min_move: float = 0.1  # body lengths: shorter re-placements are adjustments, not moves
    ground_band: float = 0.35  # body lengths: ground anchors are this close to the lowest foot
    floor_lab_tol: float = 25.0  # Lab colour distance to the floor colour
    floor_frac: float = 0.6  # share of floor-coloured pixels below the toe -> ground


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


def floor_colour(wall_image) -> np.ndarray | None:
    """Lab colour of the floor: median of the bottom 5 % of the wall image's seen rows,
    if that region is flat (little texture) — otherwise None (no floor in view)."""
    img = wall_image.image
    seen = img.max(axis=2) > 0
    rows = np.flatnonzero(seen.any(axis=1))
    if len(rows) < 20:
        return None
    bottom = rows[-max(5, len(rows) // 20):]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(float)
    px = lab[bottom][seen[bottom]]
    if len(px) < 50:
        return None
    med = np.median(px, axis=0)
    if np.mean(np.linalg.norm(px - med, axis=1) < 25) < 0.5:  # mixed/textured: not a floor
        return None
    return med


def mark_ground(anchors: dict[str, list[Anchor]], wall_image, body_px: float,
                prm: WallParams) -> bool:
    """Mark foot anchors on the ground (see module doc). Returns whether a floor was
    found in the wall image."""
    from climbvision.engine.camera import apply_h

    floor = floor_colour(wall_image) if wall_image is not None else None
    if floor is None:
        return False
    feet = [a for f in ("l_foot", "r_foot") for a in anchors[f]]
    if not feet:
        return True
    lowest = max(a.pos[1] for a in feet)  # wall y points down
    lab = cv2.cvtColor(wall_image.image, cv2.COLOR_BGR2LAB).astype(float)
    seen = wall_image.image.max(axis=2) > 0
    b = body_px * wall_image.T[0, 0]  # body length in wall-image pixels
    for a in feet:
        if a.pos[1] < lowest - prm.ground_band * body_px:
            continue
        x, y = apply_h(wall_image.T, a.pos)
        ys, xs = slice(max(0, int(y + 0.02 * b)), max(0, int(y + 0.10 * b))), \
            slice(max(0, int(x - 0.05 * b)), max(0, int(x + 0.05 * b)))
        m = seen[ys, xs]
        if m.sum() < 10:
            continue
        frac = float(np.mean(np.linalg.norm(lab[ys, xs][m] - floor, axis=1) < prm.floor_lab_tol))
        a.ground = frac > prm.floor_frac
    return True


def cluster_holds(anchors: list[Anchor], radius_px: float) -> list[Hold]:
    """Greedy clustering of anchor positions, in time order."""
    holds: list[Hold] = []
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

    def hold_xy_m(self, h: Hold) -> tuple[float, float]:
        """Hold position in metres: x right from the first hold, y up from the ground
        (or from the lowest hold when there is no ground)."""
        x0 = self.holds[0].pos[0] if self.holds else 0.0
        base = self.ground_y if self.ground_y is not None else max(
            (hh.pos[1] for hh in self.holds), default=0.0)
        return (float(h.pos[0] - x0) / self.px_per_m, float(base - h.pos[1]) / self.px_per_m)


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
                         wall_image=None, prm: WallParams | None = None) -> WallSupport:
    """wall_points: (n, N_JOINTS, 2) joints in wall coordinates (Clip.wall)."""
    prm = prm or WallParams()
    n = len(t)
    pts = limb_points(wall_points)
    anchors = {limb: detect_anchors(pts[limb], t, body_px, limb, prm) for limb in LIMBS}
    floor_seen = mark_ground(anchors, wall_image, body_px, prm)
    detected = np.isfinite(wall_points[:, J.L_SHOULDER, 0])
    ok = detected if camera_ok is None else detected & camera_ok
    foot_hold = np.zeros(n, bool)
    foot_ground = np.zeros(n, bool)
    for f in ("l_foot", "r_foot"):
        for an in anchors[f]:
            (foot_ground if an.ground else foot_hold)[an.start:an.end] = True
    climbing = foot_hold & ~foot_ground
    on_wall = np.zeros(n, bool)
    if climbing.any():
        a, b = np.flatnonzero(climbing)[[0, -1]]
        on_wall[a:b + 1] = True
    on_wall &= ok & ~foot_ground
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
    hold_anchors = [an for limb in LIMBS for an in anchors[limb] if not an.ground]
    holds = cluster_holds(hold_anchors, prm.hold_radius * body_px)
    on_hold = {limb: _on_mask([a for a in anchors[limb] if not a.ground], n) for limb in LIMBS}
    n_contacts = sum(on_hold[limb].astype(int) for limb in LIMBS)
    com = whole_body_com(wall_points, sex)  # type: ignore[arg-type]
    to_cm = 100.0 / px_per_m
    feet_m = np.full(n, np.nan)
    poly_m = np.full(n, np.nan)
    pad = FOOT_RADIUS_M * px_per_m
    pos_at = {limb: np.full((n, 2), np.nan) for limb in LIMBS}
    for limb in LIMBS:
        for an in anchors[limb]:
            if not an.ground:
                pos_at[limb][an.start:an.end] = an.pos
    for i in range(n):
        if not np.isfinite(com[i]).all():
            continue
        fx = [pos_at[f][i, 0] for f in ("l_foot", "r_foot") if on_hold[f][i]]
        if fx:
            lo, hi = min(fx) - pad, max(fx) + pad
            feet_m[i] = min(com[i, 0] - lo, hi - com[i, 0]) * to_cm
        hull_pts = [pos_at[limb][i] for limb in LIMBS if on_hold[limb][i]]
        if len(hull_pts) >= 3:
            hull = cv2.convexHull(np.array(hull_pts, np.float32))
            if cv2.contourArea(hull) > 1.0:
                poly_m[i] = _signed_hull_distance(hull, com[i]) * to_cm
    labels = [contact_label({limb: bool(on_hold[limb][i]) for limb in LIMBS})
              if on_wall[i] else ("on the ground" if on_ground[i] else "–") for i in range(n)]
    ground = [a.pos[1] for f in ("l_foot", "r_foot") for a in anchors[f] if a.ground]
    ground_y = float(np.median(ground)) if ground else None
    return WallSupport(pts, anchors, holds, on_hold, on_ground, on_wall, ground_y, com,
                       n_contacts, feet_m, poly_m, px_per_m, body_px, labels, floor_seen,
                       prm.min_move * body_px)


def moves(ws: WallSupport, t: np.ndarray) -> list[dict]:
    """A move = a limb leaving one hold and next anchoring on a different hold."""
    out = []
    for limb in LIMBS:
        seq = [a for a in ws.anchors[limb] if not a.ground]
        for a, b in zip(seq, seq[1:], strict=False):
            if a.hold != b.hold and np.linalg.norm(b.pos - a.pos) >= ws.min_move_px:
                d = float(np.linalg.norm(b.pos - a.pos)) / ws.px_per_m
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
            out["com_height_gain_m"] = round(float(np.nanmax(cy) - np.nanmin(cy)) / ws.px_per_m, 2)
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
    out["moves_detail"] = mv
    warn = []
    if wall_s < 0.5 * (t[-1] - t[0]) if n > 1 else False:
        warn.append(f"on the wall only {wall_s:.1f} s of {t[-1] - t[0]:.1f} s")
    if warn:
        out["warnings"] = warn
    return out
