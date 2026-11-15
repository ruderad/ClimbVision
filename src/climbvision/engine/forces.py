"""Forces at the hands and feet, in the plane of the wall.

**Physics.** The holds together must carry the body's weight and give it its
acceleration: the contact forces add up to m (a + g), with a the COM's acceleration
(engine/kinematics.py) and g pointing down. They must also not spin the body: their
moments about the COM add up to zero (point contacts: a hold doesn't hold a twist).
That's 3 equations in the plane of the wall.

**The split.** With 2+ limbs on holds there are more unknowns (2 per limb) than
equations: infinitely many splits balance. The moment balance is *soft*: holds close
together (a matched hold, two hands side by side) can only stop the body from turning
with huge opposing forces; real bodies swing a little instead (a barn door). So an
unbalanced moment costs as much as a force of moment / `swing_m` (5 cm): a lever of a
hand span or more balances it, two hands on one hold let the body swing. The model picks
the split that costs least —
the smallest sum of squared forces, with hand forces counting `hand_cost` times more
than foot forces (legs are stronger; climbers put weight on their feet) — under what
contacts can do:

- a foot can only push (the hold pushes the foot up, never pulls it down) and only
  within friction: sideways force <= `mu` x upward force;
- a hand can pull in any direction (a jug pulls up, an undercling pushes down on the
  body's hand, a sidepull sideways).

The constrained least-cost split is found exactly by trying each foot free, off, or on
either edge of its friction cone (at most 16 small linear solves).

With a single limb on a hold the moment can't be balanced (the body swings): only the
force sum is used. With none (a dyno in flight), all forces are zero.

**What it is and isn't.** Forces *in the wall plane*: up/down and sideways. On a steep
wall the hands also pull *outwards* (away from the wall) to keep the hips in; that needs
the wall angle and how far the hips are from the wall (depth, 1.6), so on overhangs the
hand loads here are lower than the real ones. Forces are in % of body weight (no mass
needed); in newtons when the climber's mass is known. Expect ballpark values (±20-30 %)
until checked against a scale.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from climbvision.engine.wall import LIMBS

G = 9.81
FEET = ("l_foot", "r_foot")


@dataclass
class ForceParams:
    hand_cost: float = 3.0  # hand force counts this much more than foot force
    mu: float = 0.8  # rubber on a hold: friction coefficient
    swing_m: float = 0.05  # a moment the holds don't balance costs like a force of
    #                        moment / swing_m: the body swings instead (barn door)


@dataclass
class Forces:
    force: dict  # limb -> (n, 2) force on the body, in body weights (x right, y up)
    load: dict  # limb -> (n,) size of the force, % body weight (NaN when not on a hold)
    hands_pct: np.ndarray  # (n,) both hands together, % of all the force from holds
    hands_bw: np.ndarray  # (n,) both hands together, % body weight
    feet_bw: np.ndarray
    mass_kg: float | None

    def newtons(self, pct_bw: np.ndarray) -> np.ndarray | None:
        return None if not self.mass_kg else np.asarray(pct_bw) / 100 * self.mass_kg * G


def _solve(points: list[np.ndarray], feet: list[bool], need: np.ndarray, com: np.ndarray,
           prm: ForceParams, moments: bool) -> np.ndarray | None:
    """Least-cost forces (k, 2) for contacts at `points` balancing `need` (force sum) and
    zero moment about `com`. None if nothing feasible."""
    k = len(points)
    best, best_cost = None, np.inf
    foot_idx = [i for i in range(k) if feet[i]]
    for modes in itertools.product(("free", "off", "left", "right"), repeat=len(foot_idx)):
        mode = dict(zip(foot_idx, modes, strict=True))
        cols, owners, weights = [], [], []
        for i in range(k):
            m = mode.get(i, "free")
            w = 1.0 if feet[i] else prm.hand_cost
            if m == "free":
                cols += [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
                owners += [i, i]
                weights += [w, w]
            elif m in ("left", "right"):
                d = np.array([-prm.mu if m == "left" else prm.mu, 1.0])
                cols.append(d)
                owners.append(i)
                weights.append(w * float(d @ d))
        if not cols:
            continue
        B = np.zeros((2 * k, len(cols)))
        for j, (c, i) in enumerate(zip(cols, owners, strict=True)):
            B[2 * i:2 * i + 2, j] = c
        # sum of forces = need (hard); moments about the COM ~ 0 (soft, see swing_m)
        A = np.tile(np.eye(2), k) @ B
        b = need
        Q = np.diag(np.array(weights))
        if moments and k >= 2:
            r = np.array(points) - com
            m_row = np.zeros(2 * k)
            m_row[0::2] = -r[:, 1]  # r x F = rx Fy - ry Fx
            m_row[1::2] = r[:, 0]
            mb = m_row @ B
            Q = Q + np.outer(mb, mb) / prm.swing_m ** 2
        try:
            Qi = np.linalg.inv(Q)
            z = Qi @ A.T @ np.linalg.solve(A @ Qi @ A.T, b)
        except np.linalg.LinAlgError:
            continue
        if not np.allclose(A @ z, b, atol=1e-6):
            continue
        f = (B @ z).reshape(k, 2)
        ok = True
        for i in foot_idx:
            m = mode[i]
            if m == "free":
                ok &= f[i, 1] >= -1e-9 and abs(f[i, 0]) <= prm.mu * f[i, 1] + 1e-9
            elif m in ("left", "right"):
                ok &= f[i, 1] >= -1e-9
        if not ok:
            continue
        cost = float(z @ Q @ z)
        if cost < best_cost - 1e-12:
            best, best_cost = f, cost
    return best


def compute_forces(ws, t: np.ndarray, mass_kg: float | None = None,
                   prm: ForceParams | None = None) -> Forces:
    prm = prm or ForceParams()
    n = len(t)
    acc = ws.com_a if ws.com_a is not None else np.zeros((n, 2))
    pos_at = {limb: np.full((n, 2), np.nan) for limb in LIMBS}
    for limb in LIMBS:
        for an in ws.anchors[limb]:
            if an.on_wall_contact:  # holds and smears both carry load
                pos_at[limb][an.start:an.end] = ws.to_m(an.pos)
    force = {limb: np.full((n, 2), np.nan) for limb in LIMBS}
    for i in range(n):
        if not ws.on_wall[i] or not np.isfinite(ws.com_m[i]).all():
            continue
        on = [limb for limb in LIMBS
              if (ws.on_hold[limb][i] or (ws.on_smear is not None and ws.on_smear[limb][i]))
              and np.isfinite(pos_at[limb][i]).all()]
        a = acc[i] if np.isfinite(acc[i]).all() else np.zeros(2)
        need = (a + np.array([0.0, G])) / G  # in body weights
        if not on:
            for limb in LIMBS:
                force[limb][i] = 0.0  # in the air
            continue
        f = _solve([pos_at[limb][i] for limb in on], [limb in FEET for limb in on], need,
                   ws.com_m[i], prm, moments=True)
        if f is None:  # e.g. feet only and the COM outside them: no static balance
            f = _solve([pos_at[limb][i] for limb in on], [limb in FEET for limb in on],
                       need, ws.com_m[i], prm, moments=False)
        if f is None:
            continue
        for limb in LIMBS:
            force[limb][i] = 0.0
        for limb, fi in zip(on, f, strict=True):
            force[limb][i] = fi
    load = {limb: 100 * np.linalg.norm(force[limb], axis=1) for limb in LIMBS}
    hands = load["l_hand"] + load["r_hand"]
    feet = load["l_foot"] + load["r_foot"]
    with np.errstate(invalid="ignore", divide="ignore"):
        share = 100 * hands / (hands + feet)
    return Forces(force, load, share, hands, feet, mass_kg)


def forces_summary(fz: Forces, ws, t: np.ndarray, route=None) -> dict:
    on = ws.on_wall & np.isfinite(fz.hands_bw)
    if not on.any():
        return {}
    dt = np.gradient(t) if len(t) > 1 else np.ones(len(t))
    hand_peak = {limb: float(np.nanmax(fz.load[limb][on])) for limb in ("l_hand", "r_hand")}
    out = {
        "hands_share_pct_median": round(float(np.nanmedian(fz.hands_pct[on])), 1),
        "hands_load_pct_bw_median": round(float(np.nanmedian(fz.hands_bw[on])), 1),
        "hands_load_pct_bw_peak": round(float(np.nanmax(fz.hands_bw[on])), 1),
        "one_hand_load_pct_bw_peak": {("L hand" if k == "l_hand" else "R hand"): round(v, 1)
                                      for k, v in hand_peak.items()},
        "time_hands_over_50pct_bw_s": round(float(dt[on & (fz.hands_bw > 50)].sum()), 1),
        "model": "in the wall plane only; out-of-plane pull (steep walls) not included",
    }
    if fz.mass_kg:
        out["mass_kg"] = fz.mass_kg
        out["hands_load_n_peak"] = round(float(fz.mass_kg * G * out["hands_load_pct_bw_peak"]
                                               / 100))
    if route is not None:
        for m in route.moves:
            if m.kind != "hand":
                continue
            j = slice(m.catch, min(len(t), m.catch + max(1, int(0.3 / float(np.median(
                np.diff(t)))))))
            v = fz.load[m.limb][j]
            m.catch_load_pct_bw = float(np.nanmax(v)) if np.isfinite(v).any() else np.nan
    return out
