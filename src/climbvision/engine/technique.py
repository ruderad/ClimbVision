"""Technique: how the climb was climbed, from the wall and route layers.

What coaches look at, measured from what the video shows reliably:

- **Quiet feet** — a foot placed once and left there. A *re-placement* is the same foot
  leaving a hold and coming back to the same hold (lifted and put back), or landing
  closer than a move to where it was, after really leaving it (>= 0.1 body lengths,
  ~9 cm, away in between). A foot sliding a few cm, or the pose wobbling, isn't a
  re-placement.
  `quiet_feet_pct` = foot placements that needed no re-placement.
- **Re-grips** — the same for hands: a hand letting go of a hold and taking it again.
- **Straight arms** — while a hand is on a hold, the share of time its arm is straight
  (elbow >= 150 deg in the picture). Climbing on straight arms saves the biceps; bent
  arms on easy ground cost strength. (2D: an arm pointing at the camera looks bent.)
- **Hesitation** — before each hand move, how long since the last limb landed
  (`pause_before_s`): reading, setting up, or hesitating.
- **COM path** — for each hand move, from the previous hand move's catch to this one
  (the move's cycle): straight-line distance of the COM / length of its path
  (`com_path_efficiency`, 1 = straight), and how many separate pushes it took
  (`com_pushes`: speed peaks; 1 = one smooth movement, 3+ = stop-and-go).
- **Feet stayed on** — share of hand moves with both feet on holds all the way.

Per move these go to `moves.csv`; the clip's summary to `technique` in metrics.json.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from climbvision.engine.biomech import angle_at
from climbvision.engine.skeleton import J

HANDS = ("l_hand", "r_hand")
FEET = ("l_foot", "r_foot")


@dataclass
class TechniqueParams:
    straight_deg: float = 150.0  # elbow at least this open = straight arm
    push_min_ms: float = 0.1  # a COM speed peak must reach this (m/s) ...
    push_drop: float = 0.5  # ... and the speed must fall below this share of it between
    lift: float = 0.1  # body lengths (~9 cm): a limb this far off its hold has left it


def _replacements(ws, limb: str, lift: float) -> tuple[int, int, list[int]]:
    """(placements, re-placements, frames where a re-placement landed) for one limb.

    A re-placement: the limb lands again on the same hold (or closer than a move) after
    really leaving it — at least `lift` body lengths away in between. An anchor that
    just restarts a frame later a few cm on (the foot sliding, or the pose wobbling)
    is not a re-placement."""
    seq = [a for a in ws.anchors[limb] if a.on_hold]
    placements = re = 0
    frames = []
    pts = ws.points[limb]
    for prev, a in zip([None, *seq], seq, strict=False):
        same = prev is not None and (a.hold == prev.hold or np.linalg.norm(a.pos - prev.pos)
                                     < ws.min_move_px)
        if not same:
            placements += 1
            continue
        gap = pts[prev.end:a.start]
        away = np.linalg.norm(gap - prev.pos, axis=1) if len(gap) else np.zeros(0)
        if np.isfinite(away).any() and np.nanmax(away) >= lift * ws.body_px:
            re += 1
            frames.append(a.start)
    return placements, re, frames


def _count_pushes(speed: np.ndarray, prm: TechniqueParams) -> int:
    """Separate COM speed peaks (>= push_min_ms). Two peaks are separate pushes when the
    speed between them dips below push_drop x the smaller one (stop-and-go)."""
    v = np.nan_to_num(np.asarray(speed, float), nan=0.0)
    if len(v) < 3:
        return 0
    idx = [i for i in range(1, len(v) - 1)
           if v[i] >= v[i - 1] and v[i] > v[i + 1] and v[i] >= prm.push_min_ms]
    kept: list[int] = []
    for i in idx:
        if kept:
            j = kept[-1]
            dip = float(v[j:i + 1].min())
            if dip >= prm.push_drop * min(v[j], v[i]):
                if v[i] > v[j]:
                    kept[-1] = i  # the same push
                continue
        kept.append(i)
    return len(kept)


def compute_technique(clip, prm: TechniqueParams | None = None) -> dict:
    """Clip-level technique summary; also fills per-move fields on clip.route.moves."""
    prm = prm or TechniqueParams()
    ws, route, t = clip.wall_support, clip.route, clip.t
    n = len(t)
    dt = np.gradient(t) if n > 1 else np.ones(n)
    on_wall = ws.on_wall
    speed = np.linalg.norm(ws.com_v, axis=1) if ws.com_v is not None else np.full(n, np.nan)
    out: dict = {}

    # quiet feet, re-grips
    fp = fr = hp = hr = 0
    for f in FEET:
        p, r, _ = _replacements(ws, f, prm.lift)
        fp, fr = fp + p, fr + r
    for h in HANDS:
        p, r, _ = _replacements(ws, h, prm.lift)
        hp, hr = hp + p, hr + r
    out["foot_placements"] = fp
    out["foot_replacements"] = fr
    out["quiet_feet_pct"] = round(100 * (1 - fr / (fp + fr)), 1) if fp + fr else None
    out["hand_regrips"] = hr

    # straight arms while holding (not moving that arm)
    img = clip.track.image
    straight = {}
    for h, (sh, el, wr) in (("l_hand", (J.L_SHOULDER, J.L_ELBOW, J.L_WRIST)),
                            ("r_hand", (J.R_SHOULDER, J.R_ELBOW, J.R_WRIST))):
        ang = angle_at(img[:, sh], img[:, el], img[:, wr])
        use = ws.on_hold[h] & on_wall & np.isfinite(ang)
        if use.sum():
            straight[h] = (ang[use] >= prm.straight_deg, dt[use])
    if straight:
        tot = sum(float(d.sum()) for _, d in straight.values())
        st = sum(float(d[s].sum()) for s, d in straight.values())
        out["straight_arms_pct"] = round(100 * st / tot, 1) if tot else None

    # per hand move: pause before, cycle, COM path efficiency, pushes, feet stayed on
    hand_moves = [m for m in route.moves if m.kind == "hand"]
    prev_catch = int(np.flatnonzero(on_wall)[0]) if on_wall.any() else 0
    feet_on = []
    for m in route.moves:
        earlier = [x.catch for x in route.moves if x.catch <= m.release and x is not m]
        last_land = max(earlier) if earlier else prev_catch
        m.pause_before_s = float(t[m.release] - t[last_land]) if m.release >= last_land \
            else 0.0
    prev = prev_catch
    for m in hand_moves:
        a, b = prev, m.catch
        prev = m.catch
        m.cycle_s = float(t[b] - t[a])
        seg = slice(a, b + 1)
        v = speed[seg]
        if b - a >= 3 and np.isfinite(v).sum() >= 0.8 * (b - a + 1):
            path = float(np.nansum(v * dt[seg]))
            disp = float(np.linalg.norm(ws.com_m[b] - ws.com_m[a])) \
                if np.isfinite(ws.com_m[[a, b]]).all() else float("nan")
            m.com_path_efficiency = disp / path if path > 0.02 else float("nan")
            m.com_pushes = _count_pushes(v, prm)
        span = slice(m.release + 1, max(m.catch, m.release + 1))
        both = ws.on_hold["l_foot"][span] & ws.on_hold["r_foot"][span]
        feet_on.append(bool(both.all()))

    def med(vals):
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return round(float(np.median(vals)), 2) if vals else None

    out["pause_before_hand_move_s_median"] = med([m.pause_before_s for m in hand_moves])
    out["time_per_hand_move_s"] = med([m.cycle_s for m in hand_moves])
    out["com_path_efficiency_median"] = med([m.com_path_efficiency for m in hand_moves])
    out["com_pushes_median"] = med([m.com_pushes for m in hand_moves])
    out["feet_stayed_on_pct"] = round(100 * float(np.mean(feet_on)), 1) if feet_on else None
    return out
