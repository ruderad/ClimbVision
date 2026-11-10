"""Route layer: the climb as a sequence of moves on holds.

Built on the wall layer (anchors = a hand or foot still ~0.5 s on the wall = on a hold;
holds numbered in the order they were first used). Here:

- **Moves**: a limb leaving one hold and next resting on a different one, at least
  `min_move` away (shorter re-placements are adjustments). Numbered in the order they
  land. Each move records what the other three limbs did meanwhile and the balance
  before and after, and gets tags the data can support:
  - `match`: a hand lands on the hold the other hand is on (or a foot on the other
    foot's hold);
  - `cross`: a hand lands on the far side of the other hand;
  - `dyno`: during a hand move both hands are off holds at some moment and the centre
    of mass moves at least `dyno_speed` (`jump` when nothing is on a hold at all);
  - `feet cut`: during a hand move both feet are off holds;
  - `high step`: a foot lands at or above the centre of mass;
  - `foot on handhold`: a foot lands on a hold a hand used before;
  - `after chalk`: the hand rested at the hip on the way (chalking / shaking out); the
    move is timed from when it left the hip.
  Moves from or to the ground (stepping on / off) are not counted.
- **Rests**: all four limbs on holds and the centre of mass still (< `rest_speed`) for
  at least `rest_min_s`.
- **Feet cut**: on the wall, a hand on a hold and both feet off holds for at least
  `cut_min_s` (also counts pose misses on the feet: watch the video).
- **Hold usage**: per hold, which limbs used it, when first, for how long, matched.

Pure functions of a WallSupport: nothing here touches the video.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from climbvision.engine.events import runs
from climbvision.engine.wall import LABEL, LIMBS, WallSupport, contact_label

HANDS = ("l_hand", "r_hand")
FEET = ("l_foot", "r_foot")
OTHER = {"l_hand": "r_hand", "r_hand": "l_hand", "l_foot": "r_foot", "r_foot": "l_foot"}


@dataclass
class RouteParams:
    rest_min_s: float = 1.0  # a rest lasts at least this long
    rest_speed: float = 0.12  # body lengths / s: COM slower than this = still
    cut_min_s: float = 0.3  # feet off for at least this long = feet cut
    cross_tol: float = 0.03  # body lengths: hands this far past each other = crossed
    dyno_speed: float = 0.8  # m/s: COM at least this fast during a dyno / jump
    dyno_max_s: float = 1.5  # a dyno is quick: hand off its hold at most this long
    label_s: float = 1.2  # a move's label stays on the video this long after it lands


@dataclass
class Move:
    n: int  # 1-based, in the order moves land
    limb: str
    from_hold: int
    to_hold: int
    release: int  # last frame on the old hold
    catch: int  # first frame on the new hold
    release_s: float
    catch_s: float
    time_s: float
    distance_cm: float
    direction_deg: float  # 0 = right, 90 = up, 180 = left, -90 = down (on the wall)
    others: str  # the other limbs on holds during the move (fewest), e.g. "2 hands + R foot"
    others_min: int
    com_over_feet_before_cm: float
    com_over_feet_after_cm: float
    com_speed_peak_ms: float
    tags: list[str] = field(default_factory=list)

    @property
    def kind(self) -> str:
        return "hand" if self.limb in HANDS else "foot"

    def label(self) -> str:
        tags = f" ({', '.join(self.tags)})" if self.tags else ""
        return f"move {self.n}: {LABEL[self.limb]} {self.from_hold} -> {self.to_hold}{tags}"

    def as_dict(self) -> dict:
        def r(v, k=1):
            return None if not np.isfinite(v) else round(float(v), k)

        return {"n": self.n, "limb": LABEL[self.limb], "kind": self.kind,
                "from_hold": self.from_hold, "to_hold": self.to_hold,
                "release_s": round(self.release_s, 2), "catch_s": round(self.catch_s, 2),
                "time_s": round(self.time_s, 2), "distance_cm": r(self.distance_cm),
                "direction_deg": r(self.direction_deg, 0), "others": self.others,
                "com_over_feet_before_cm": r(self.com_over_feet_before_cm),
                "com_over_feet_after_cm": r(self.com_over_feet_after_cm),
                "com_speed_peak_ms": r(self.com_speed_peak_ms, 2),
                "tags": list(self.tags)}


@dataclass
class Route:
    moves: list[Move]
    holds: list[dict]  # usage per hold
    rests: list[tuple[int, int]]  # [start, end) frames
    feet_cut: list[tuple[int, int]]
    resting: np.ndarray  # (n,) bool
    feet_off: np.ndarray  # (n,) bool: feet cut
    in_move: np.ndarray  # (n,) number of limbs between holds (moving)
    labels: list[str]  # per frame: the latest move / rest / feet cut, for the video


def _com_speed(ws: WallSupport, t: np.ndarray) -> np.ndarray:
    """COM speed, body lengths / s, smoothed over ~0.3 s."""
    c = ws.com
    n = len(t)
    if n < 3:
        return np.zeros(n)
    k = max(1, int(0.15 / max(float(np.median(np.diff(t))), 1e-3)))
    v = np.full(n, np.nan)
    for i in range(n):
        a, b = max(0, i - k), min(n - 1, i + k)
        if b > a and np.isfinite(c[[a, b]]).all():
            v[i] = np.linalg.norm(c[b] - c[a]) / (t[b] - t[a])
    return v / ws.body_px


def _hold_at(ws: WallSupport, limb: str, n: int) -> np.ndarray:
    out = np.full(n, -1)
    for a in ws.anchors[limb]:
        if a.on_hold:
            out[a.start:a.end] = a.hold
    return out


def compute_route(ws: WallSupport, t: np.ndarray, prm: RouteParams | None = None) -> Route:
    prm = prm or RouteParams()
    n = len(t)
    fps = 1.0 / float(np.median(np.diff(t))) if n > 1 else 30.0
    speed = _com_speed(ws, t)
    hold_at = {limb: _hold_at(ws, limb, n) for limb in LIMBS}
    on = ws.on_hold
    to_cm = 100.0 / ws.px_per_m
    hand_holds_before: dict[int, int] = {}  # hold -> first frame a hand used it
    for h in HANDS:
        for a in ws.anchors[h]:
            if a.on_hold:
                hand_holds_before[a.hold] = min(hand_holds_before.get(a.hold, n), a.start)

    raw = []
    for limb in LIMBS:
        seq = [a for a in ws.anchors[limb] if a.on_hold]
        for a, b in zip(seq, seq[1:], strict=False):
            if a.hold == b.hold or np.linalg.norm(b.pos - a.pos) < ws.min_move_px:
                continue
            if not (ws.on_wall[a.end - 1] or ws.on_wall[b.start]):
                continue  # stepping off / on from the ground
            # a hand that chalked or shook out on the way: the move starts when it
            # leaves the hip
            rest = [x for x in ws.anchors[limb] if x.body and a.end <= x.start and x.end <= b.start]
            raw.append((limb, a, b, rest[-1].end - 1 if rest else None))
    raw.sort(key=lambda x: x[2].start)

    moves: list[Move] = []
    in_move = np.zeros(n, int)
    for k, (limb, a, b, after_chalk) in enumerate(raw, start=1):
        rel, cat = (a.end - 1 if after_chalk is None else after_chalk), b.start
        span = slice(rel + 1, max(cat, rel + 1))
        in_move[span] += 1
        others = [x for x in LIMBS if x != limb]
        if cat > rel + 1:
            cnt = sum(on[x][span].astype(int) for x in others)
            j = int(np.argmin(cnt))
            i_min = rel + 1 + j
            others_min = int(cnt[j])
        else:
            i_min, others_min = rel, sum(int(on[x][rel]) for x in others)
        others_lab = contact_label({x: x != limb and bool(on[x][i_min]) for x in LIMBS})
        d = b.pos - a.pos
        ang = float(np.degrees(np.arctan2(-d[1], d[0])))  # wall y points down
        win = speed[rel:cat + 1]
        peak = float(np.nanmax(win)) * ws.body_px / ws.px_per_m if np.isfinite(win).any() \
            else float("nan")
        tags = [] if after_chalk is None else ["after chalk"]
        if limb in HANDS:
            o = OTHER[limb]
            if hold_at[o][cat] == b.hold or (
                    cat + int(0.5 * fps) < n and (hold_at[o][cat:cat + int(0.5 * fps)]
                                                  == b.hold).any()):
                tags.append("match")
            ox = ws.points[o][cat, 0]
            if on[o][cat] and np.isfinite(ox):
                past = (b.pos[0] - ox) * (1 if limb == "r_hand" else -1)
                if past < -prm.cross_tol * ws.body_px:
                    tags.append("cross")
            if cat > rel + 1:
                hands_off = ~on["l_hand"][span] & ~on["r_hand"][span] & ws.on_wall[span]
                if hands_off.sum() >= 2 and peak >= prm.dyno_speed and \
                        t[cat] - t[rel] <= prm.dyno_max_s:
                    nothing = sum(on[x][span].astype(int) for x in LIMBS) == 0
                    tags.append("jump" if (nothing & ws.on_wall[span]).sum() >= 2 else "dyno")
                feet_off = ~on["l_foot"][span] & ~on["r_foot"][span] & ws.on_wall[span]
                if feet_off.sum() >= max(2, int(prm.cut_min_s * fps)):
                    tags.append("feet cut")
        else:
            if hold_at[OTHER[limb]][cat] == b.hold:
                tags.append("match")
            if np.isfinite(ws.com[cat, 1]) and b.pos[1] <= ws.com[cat, 1]:
                tags.append("high step")
            if hand_holds_before.get(b.hold, n) < cat:
                tags.append("foot on handhold")
            if others_min == 0 and peak >= prm.dyno_speed:
                tags.append("jump")
        moves.append(Move(
            k, limb, a.hold, b.hold, rel, cat, float(t[rel]), float(t[cat]),
            float(t[cat] - t[rel]), float(np.linalg.norm(d)) * to_cm, ang, others_lab,
            others_min, float(ws.com_over_feet_cm[rel]), float(ws.com_over_feet_cm[cat]),
            peak, tags))

    # rests: four on holds and the COM still
    four = (ws.n_contacts == 4) & ws.on_wall
    still = np.nan_to_num(speed, nan=np.inf) < prm.rest_speed
    resting = np.zeros(n, bool)
    rests = []
    for a, b in runs(four & still):
        if t[b - 1] - t[a] >= prm.rest_min_s:
            rests.append((int(a), int(b)))
            resting[a:b] = True
    # feet cut: a hand on, both feet off, on the wall
    cut = ws.on_wall & (on["l_hand"] | on["r_hand"]) & ~on["l_foot"] & ~on["r_foot"]
    feet_off = np.zeros(n, bool)
    cuts = []
    for a, b in runs(cut):
        if t[b - 1] - t[a] >= prm.cut_min_s:
            cuts.append((int(a), int(b)))
            feet_off[a:b] = True

    labels = [""] * n
    for i in range(n):
        if resting[i]:
            a = next(r for r in rests if r[0] <= i < r[1])
            labels[i] = f"rest {t[i] - t[a[0]]:.1f} s"
        elif feet_off[i]:
            labels[i] = "feet cut"
    state = list(labels)
    for m in moves:  # the latest move landed within label_s
        last = int(np.searchsorted(t, m.catch_s + prm.label_s))
        for i in range(m.catch, min(n, last)):
            labels[i] = m.label() + (f" · {state[i]}" if state[i] else "")
    return Route(moves, hold_usage(ws, t, hold_at), rests, cuts, resting, feet_off,
                 in_move, labels)


def hold_usage(ws: WallSupport, t: np.ndarray, hold_at: dict | None = None) -> list[dict]:
    n = len(t)
    hold_at = hold_at or {limb: _hold_at(ws, limb, n) for limb in LIMBS}
    dt = np.gradient(t) if n > 1 else np.ones(n)
    out = []
    for h in ws.holds:
        x, y = ws.hold_xy_m(h)
        per = {LABEL[limb]: round(float(dt[hold_at[limb] == h.id].sum()), 2)
               for limb in LIMBS if (hold_at[limb] == h.id).any()}
        both_hands = (hold_at["l_hand"] == h.id) & (hold_at["r_hand"] == h.id)
        both_feet = (hold_at["l_foot"] == h.id) & (hold_at["r_foot"] == h.id)
        hand = bool(h.limbs & set(HANDS))
        foot = bool(h.limbs & set(FEET))
        out.append({"id": h.id, "x_m": round(x, 2), "y_m": round(y, 2),
                    "used_as": "hand + foot" if hand and foot else "hand" if hand else "foot",
                    "first_s": round(float(t[min(a.start for a in h.anchors)]), 2),
                    "time_s": round(sum(per.values()), 2), "time_by_limb_s": per,
                    "matched": bool(both_hands.any() or both_feet.any())})
    return out


def route_summary(route: Route, ws: WallSupport, t: np.ndarray) -> dict:
    mv = route.moves
    dt = np.gradient(t) if len(t) > 1 else np.ones(len(t))
    wall_s = float(dt[ws.on_wall].sum())
    tags: dict[str, int] = {}
    for m in mv:
        for g in m.tags:
            tags[g] = tags.get(g, 0) + 1
    hand = [m for m in mv if m.kind == "hand"]
    foot = [m for m in mv if m.kind == "foot"]
    out: dict = {"moves": len(mv), "hand_moves": len(hand), "foot_moves": len(foot),
                 "moves_per_min": round(60 * len(mv) / wall_s, 1) if wall_s > 0 else None,
                 "foot_moves_per_hand_move": round(len(foot) / len(hand), 2) if hand else None,
                 "tags": tags}
    for name, group in (("hand", hand), ("foot", foot)):
        if group:
            out[f"{name}_move_time_s_median"] = round(float(np.median(
                [m.time_s for m in group])), 2)
            out[f"{name}_move_cm_median"] = round(float(np.median(
                [m.distance_cm for m in group])), 1)
    if hand:
        big = max(hand, key=lambda m: m.distance_cm)
        out["longest_hand_move"] = {"n": big.n, "limb": LABEL[big.limb],
                                    "distance_cm": round(big.distance_cm, 1),
                                    "at_s": round(big.catch_s, 2)}
    rest_s = [float(t[b - 1] - t[a]) for a, b in route.rests]
    out["rests"] = {"count": len(rest_s), "total_s": round(sum(rest_s), 1),
                    "longest_s": round(max(rest_s), 1) if rest_s else 0.0}
    cut_s = [float(t[b - 1] - t[a]) for a, b in route.feet_cut]
    out["feet_cut"] = {"count": len(cut_s), "total_s": round(sum(cut_s), 1)}
    out["holds_matched"] = sum(h["matched"] for h in route.holds)
    out["feet_on_handholds"] = sum(h["used_as"] == "hand + foot" for h in route.holds)
    return out
