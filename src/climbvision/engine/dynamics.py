"""Dynamics on the wall: move phases, power, the deadpoint, swing, and joint effort.

Built on the COM's position, velocity and acceleration in metres (kinematics.py) and
the contact forces (forces.py). Everything is in the plane of the wall.

**Phases of a hand move** (from the COM's speed around the move):

- *push*: from the slowest moment before the hand lets go (within 1 s) to the COM's
  peak speed — the body is being driven;
- *reach*: from the peak speed to the catch — momentum carries the body, the hand
  travels;
- *catch*: from the catch until the COM is down to 40 % of its peak speed (at most
  0.4 s) — the hand absorbs the momentum;
- the rest of the time is setup / rest (no phase).

Per move: `push_s`, `push_power_wkg` (peak mechanical power of the push, W/kg:
v · (a + g)), `deadpoint_to_catch_s` (when the hand caught relative to the moment the
COM stopped rising: + after, − before; a perfect deadpoint catch is ~0).

Phases need a few frames each: at 30 fps a phase is resolved to 33 ms, and the
velocity smoothing (~0.2 s) blurs pushes shorter than ~0.1 s. Film fast moves at 60
fps or in slow motion (120 / 240 fps) for clean phases.

**Momentum arc** (dynamic moves: COM peak speed >= `dynamic_ms`): the path the COM would
follow from its peak speed with nothing but gravity — a free-flight parabola. On a dyno
that *is* the flight; on a deadpoint the feet still push, so it's where momentum alone
would carry the body. The apex is drawn where the real COM stopped rising.

**Swing** (feet off the holds, at least one hand on): how fast the body turns about the
hand(s), deg/s (the COM's angular velocity about the hand holds). Per feet-cut: the
peak swing and how long it took to settle (below `settle_deg_s` for 0.2 s).

**Joint effort**: the net moment each joint must produce to hold the contact force at
the end of its limb plus the weight of the segments beyond it (quasi-static inverse
dynamics in the wall plane), in Nm per kg of body mass — the standard proxy for how
hard the muscles around a joint work. Shoulders, elbows, hips, knees, and the *core*:
the moment the trunk must pass between the legs and the upper body (both legs' foot
forces and weights about the hip centre). Segment inertia is ignored (small next to
the contact forces except in hard swings); out-of-plane forces on steep walls are
missing like in forces.py. A model, not a measurement: compare moves and attempts,
don't read single values as truths.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from climbvision.engine.skeleton import J

G = 9.81
EFFORT_JOINTS = ("l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_hip", "r_hip",
                 "l_knee", "r_knee", "core")
PHASES = ("", "push", "reach", "catch")


@dataclass
class DynParams:
    push_back_s: float = 1.0
    catch_frac: float = 0.4  # catch ends when speed < this x the move's peak speed
    catch_max_s: float = 0.4
    dynamic_ms: float = 0.6  # COM peak speed for a move to get a momentum arc
    arc_s: float = 0.8  # longest arc drawn
    settle_deg_s: float = 30.0


@dataclass
class MoveDyn:
    n: int
    push: tuple[int, int]  # frames [a, b)
    reach: tuple[int, int]
    catch: tuple[int, int]
    push_s: float
    push_power_wkg: float
    peak_frame: int
    apex_frame: int | None  # the COM stopped rising (vy + -> -)
    deadpoint_to_catch_s: float
    dynamic: bool
    arc_m: np.ndarray | None = None  # (k, 2) metres: free-flight path from the peak


@dataclass
class Dynamics:
    phase: np.ndarray  # (n,) 0 none, 1 push, 2 reach, 3 catch
    power_wkg: np.ndarray  # (n,) COM mechanical power per kg
    moves: list[MoveDyn]
    swing_deg_s: np.ndarray  # (n,) NaN unless feet are off and a hand is on
    swing_pivot_m: np.ndarray  # (n, 2)
    swings: list[dict] = field(default_factory=list)  # per feet-cut: peak, settle
    effort: dict[str, np.ndarray] = field(default_factory=dict)  # joint -> (n,) Nm/kg


def compute_dynamics(clip, prm: DynParams | None = None) -> Dynamics:
    prm = prm or DynParams()
    ws = clip.wall_support
    t = clip.t
    n = len(t)
    fps = 1.0 / float(np.median(np.diff(t))) if n > 1 else 30.0
    v = ws.com_v if ws.com_v is not None else np.full((n, 2), np.nan)
    a = ws.com_a if ws.com_a is not None else np.full((n, 2), np.nan)
    speed = np.linalg.norm(v, axis=1)
    power = np.einsum("ij,ij->i", v, a + np.array([0.0, G]))
    phase = np.zeros(n, int)
    moves = []
    route = clip.route
    for m in route.moves:
        if m.kind != "hand":
            continue
        r, c = m.release, m.catch
        lo = max(0, r - int(prm.push_back_s * fps))
        seg = speed[lo:r + 1]
        if not np.isfinite(seg).any():
            continue
        start = lo + int(np.nanargmin(seg))
        win = speed[start:c + 1]
        if not np.isfinite(win).any():
            continue
        peak = start + int(np.nanargmax(win))
        vpk = speed[peak]
        end = c + 1
        stop = min(n, c + 1 + int(prm.catch_max_s * fps))
        while end < stop and np.isfinite(speed[end]) and speed[end] > prm.catch_frac * vpk:
            end += 1
        for k in range(start, peak):
            phase[k] = max(phase[k], 1)
        for k in range(peak, c + 1):
            phase[k] = max(phase[k], 2)
        for k in range(c + 1, end):
            phase[k] = max(phase[k], 3)
        pw = power[start:max(peak, start + 1)]
        # deadpoint: the COM stops rising (vy + -> -) after the peak
        apex = None
        for k in range(peak, min(n - 1, c + int(0.3 * fps))):
            if np.isfinite(v[k, 1]) and np.isfinite(v[k + 1, 1]) and v[k, 1] > 0 >= v[k + 1, 1]:
                apex = k + 1
                break
        dyn = bool(np.isfinite(vpk) and vpk >= prm.dynamic_ms)
        arc = None
        if dyn and np.isfinite(ws.com_m[peak]).all() and np.isfinite(v[peak]).all():
            ts = np.arange(0, prm.arc_s, 1 / fps)
            arc = ws.com_m[peak] + np.outer(ts, v[peak]) + 0.5 * np.outer(ts ** 2, [0, -G])
            below = np.flatnonzero(arc[:, 1] < ws.com_m[peak][1] - 0.05)
            if len(below):
                arc = arc[:below[0] + 1]
        md = MoveDyn(m.n, (start, peak), (peak, c + 1), (c + 1, end),
                     push_s=float(t[peak] - t[start]),
                     push_power_wkg=float(np.nanmax(pw)) if np.isfinite(pw).any() else np.nan,
                     peak_frame=peak, apex_frame=apex,
                     deadpoint_to_catch_s=float(t[c] - t[apex]) if apex is not None
                     else np.nan, dynamic=dyn, arc_m=arc)
        moves.append(md)
        m.push_s = md.push_s
        m.push_power_wkg = md.push_power_wkg
        m.deadpoint_to_catch_s = md.deadpoint_to_catch_s
    swing, pivot, swings = _swing(clip, prm, fps)
    return Dynamics(phase, power, moves, swing, pivot, swings, _effort(clip))


def _swing(clip, prm: DynParams, fps: float):
    """Angular velocity (deg/s) of the COM about the hand holds while the feet are off."""
    ws = clip.wall_support
    n = len(clip.t)
    hands = [ws.on_hold[h] for h in ("l_hand", "r_hand")]
    feet_off = ~(ws.on_hold["l_foot"] | ws.on_hold["r_foot"])
    if ws.on_smear is not None:
        feet_off &= ~(ws.on_smear["l_foot"] | ws.on_smear["r_foot"])
    pivot = np.full((n, 2), np.nan)
    swing = np.full(n, np.nan)
    for i in range(n):
        if not (feet_off[i] and ws.on_wall[i]):
            continue
        pts = [ws.points[h][i] for h, on in zip(("l_hand", "r_hand"), hands, strict=True)
               if on[i] and np.isfinite(ws.points[h][i]).all()]
        if not pts:
            continue
        pivot[i] = ws.to_m(np.mean(pts, axis=0))
        r = ws.com_m[i] - pivot[i]
        vv = ws.com_v[i] if ws.com_v is not None else np.full(2, np.nan)
        d2 = float(r @ r)
        if d2 > 1e-4 and np.isfinite(vv).all():
            swing[i] = np.degrees((r[0] * vv[1] - r[1] * vv[0]) / d2)
    swings = []
    route = clip.route
    calm = max(1, int(0.2 * fps))
    for a, b in route.feet_cut:
        s = np.abs(swing[a:b])
        if not np.isfinite(s).any():
            continue
        pk = a + int(np.nanargmax(s))
        settle = None
        for k in range(pk, b - calm + 1):
            w = np.abs(swing[k:k + calm])
            if np.isfinite(w).all() and (w < prm.settle_deg_s).all():
                settle = k
                break
        swings.append({"start_s": round(float(clip.t[a]), 2),
                       "peak_deg_s": round(float(np.nanmax(s))),
                       "settle_s": None if settle is None else
                       round(float(clip.t[settle] - clip.t[pk]), 2)})
    return swing, pivot, swings


def _effort(clip) -> dict[str, np.ndarray]:
    """Net joint moments (Nm/kg), quasi-static, wall plane."""
    from climbvision.engine.biomech import SEGMENTS, segment_params

    ws = clip.wall_support
    fz = clip.forces
    n = len(clip.t)
    P = clip.wall  # (n, J, 2) wall px
    pm = ws.to_m(P.reshape(-1, 2)).reshape(P.shape)  # metres, y up
    masses, ratios = segment_params(clip.sex)
    seg = {s.name: (s, mfrac, r) for s, mfrac, r in zip(SEGMENTS, masses, ratios, strict=True)}

    def seg_com(name):
        s, _, r = seg[name]
        p = pm[:, list(s.proximal)].mean(axis=1)
        d = pm[:, list(s.distal)].mean(axis=1)
        return p + r * (d - p)

    def moment(joint_xy, items):
        """Sum over (point (n,2), force (n,2) N/kg) of (point - joint) x force."""
        tot = np.zeros(n)
        ok = np.isfinite(joint_xy).all(axis=1)
        for pt, f in items:
            r = pt - joint_xy
            mz = r[:, 0] * f[:, 1] - r[:, 1] * f[:, 0]
            good = np.isfinite(mz)
            tot[good] += mz[good]
            ok &= np.isfinite(pt).all(axis=1) | ~np.isfinite(f).all(axis=1)
        return np.where(ok, np.abs(tot), np.nan)

    out: dict[str, np.ndarray] = {}
    contact = {}
    for limb in ("l_hand", "r_hand", "l_foot", "r_foot"):
        f = np.nan_to_num(fz.force[limb], nan=0.0) * G if fz is not None else np.zeros((n, 2))
        contact[limb] = (ws.to_m(ws.points[limb].reshape(-1, 2)).reshape(-1, 2), f)

    def weight(name):
        _, mfrac, _ = seg[name]
        return seg_com(name), np.tile([0.0, -mfrac * G], (n, 1))

    for side, s in (("l", "left"), ("r", "right")):
        hand = contact[f"{side}_hand"]
        foot = contact[f"{side}_foot"]
        fa, ua, hd = weight(f"{s}_forearm"), weight(f"{s}_upper_arm"), weight(f"{s}_hand")
        sh, th, ft = weight(f"{s}_shank"), weight(f"{s}_thigh"), weight(f"{s}_foot")
        jx = {nm: pm[:, getattr(J, f"{side.upper()}_{nm.upper()}")]
              for nm in ("elbow", "shoulder", "knee", "hip")}
        out[f"{side}_elbow"] = moment(jx["elbow"], [hand, fa, hd])
        out[f"{side}_shoulder"] = moment(jx["shoulder"], [hand, fa, hd, ua])
        out[f"{side}_knee"] = moment(jx["knee"], [foot, sh, ft])
        out[f"{side}_hip"] = moment(jx["hip"], [foot, sh, ft, th])
    mid_hip = (pm[:, J.L_HIP] + pm[:, J.R_HIP]) / 2
    legs = []
    for side, s in (("l", "left"), ("r", "right")):
        legs += [contact[f"{side}_foot"], weight(f"{s}_shank"), weight(f"{s}_thigh"),
                 weight(f"{s}_foot")]
    out["core"] = moment(mid_hip, legs)
    off = ~ws.on_wall
    for k in out:
        out[k][off] = np.nan
    return out


def dynamics_summary(d: Dynamics) -> dict:
    def r(v, k=2):
        return None if v is None or not np.isfinite(v) else round(float(v), k)

    dyn = [m for m in d.moves if m.dynamic]
    out = {
        "dynamic_moves": len(dyn),
        "push_s_median": r(np.nanmedian([m.push_s for m in d.moves])) if d.moves else None,
        "push_power_wkg_peak": r(np.nanmax([m.push_power_wkg for m in d.moves]), 1)
        if d.moves and np.isfinite([m.push_power_wkg for m in d.moves]).any() else None,
        "deadpoint_to_catch_s_median": r(np.nanmedian([m.deadpoint_to_catch_s for m in dyn]))
        if dyn and np.isfinite([m.deadpoint_to_catch_s for m in dyn]).any() else None,
        "swings": d.swings,
        # 95th percentile, not the maximum: single frames with a misplaced joint or a
        # contact force at the edge of its range make spikes
        "effort_nm_kg_p95": {k: r(np.nanpercentile(v, 95)) for k, v in d.effort.items()
                             if np.isfinite(v).any()},
        "effort_nm_kg_median": {k: r(np.nanmedian(v)) for k, v in d.effort.items()
                                if np.isfinite(v).any()},
        "model": "wall plane, quasi-static joint moments; see engine/dynamics.py",
    }
    return out
