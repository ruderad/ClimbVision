"""Wall layer on synthetic climbers: anchors (still ~0.5 s = on a hold), holds, moves,
ground, contacts. Positions are given directly in wall coordinates."""

import numpy as np

from climbvision.engine.camera import WallImage
from climbvision.engine.skeleton import N_JOINTS, J
from climbvision.engine.wall import (
    compute_wall_support,
    detect_anchors,
    moves,
    wall_summary,
)

FPS = 30.0
BODY = 600.0  # body length, px
PPM = 600.0  # px per metre


def body_at(lh, rh, lf, rf, rng):
    """A canonical skeleton (wall px, y down) with hands/feet at the given points."""
    p = np.full((N_JOINTS, 2), np.nan)
    mid = (np.array(lh) + np.array(rh) + np.array(lf) + np.array(rf)) / 4
    p[J.L_SHOULDER], p[J.R_SHOULDER] = mid + [-80, -150], mid + [80, -150]
    p[J.L_HIP], p[J.R_HIP] = mid + [-60, 120], mid + [60, 120]
    p[J.HEAD] = mid + [0, -260]
    for side, hand, foot in (("L", lh, lf), ("R", rh, rf)):
        sh, hip = p[J[f"{side}_SHOULDER"]], p[J[f"{side}_HIP"]]
        hand, foot = np.array(hand, float), np.array(foot, float)
        p[J[f"{side}_HAND"]] = hand
        p[J[f"{side}_WRIST"]] = hand + [0, 30]
        p[J[f"{side}_ELBOW"]] = (sh + hand) / 2 + [20, 0]
        p[J[f"{side}_TOE"]] = foot
        p[J[f"{side}_ANKLE"]] = foot + [-30, -20]
        p[J[f"{side}_HEEL"]] = foot + [-50, 0]
        p[J[f"{side}_KNEE"]] = (hip + foot) / 2 + [40, 0]
    return p + rng.normal(0, 2.0, p.shape)  # pose jitter (~0.3 % body)


def climb(plan, seconds=10.0, seed=0):
    """plan(t) -> (lh, rh, lf, rf). Returns (n, N_JOINTS, 2) and times."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FPS)) / FPS
    return np.stack([body_at(*plan(ti), rng) for ti in t]), t


def lerp(a, b, u):
    u = np.clip(u, 0, 1)
    return tuple(np.array(a) * (1 - u) + np.array(b) * u)


H = {"A": (300, 300), "B": (700, 250), "C": (420, 900), "D": (600, 920),
     "E": (800, 120), "F": (520, 780)}


def two_moves(ti):
    """4 limbs on holds; R hand B -> E at 3.0-3.5 s; L foot C -> F at 6.0-6.6 s."""
    rh = lerp(H["B"], H["E"], (ti - 3.0) / 0.5)
    lf = lerp(H["C"], H["F"], (ti - 6.0) / 0.6)
    return H["A"], rh, lf, H["D"]


def test_still_limb_is_anchored_moving_limb_is_not():
    t = np.arange(150) / FPS
    rng = np.random.default_rng(1)
    still = np.array([500.0, 500.0]) + rng.normal(0, 3, (150, 2))
    an = detect_anchors(still, t, BODY, "l_hand")
    assert len(an) == 1 and an[0].start == 0 and an[0].end == 150
    moving = np.column_stack([500 + 1.0 * BODY * t, np.full(150, 500.0)])  # 1 body/s
    assert detect_anchors(moving, t, BODY, "l_hand") == []
    # a short stop (0.3 s) is not a hold
    pause = moving.copy()
    pause[60:69] = pause[60]
    assert detect_anchors(pause, t, BODY, "l_hand") == []


def test_holds_moves_and_contacts():
    pts, t = climb(two_moves)
    ws = compute_wall_support(pts, t, BODY, PPM)
    assert len(ws.holds) == 6  # A, B, C, D + E, F
    assert {h.id for h in ws.holds if "r_hand" in h.limbs} == {
        ws.anchors["r_hand"][0].hold, ws.anchors["r_hand"][1].hold}
    mv = moves(ws, t)
    assert [(m["limb"]) for m in mv] == ["r_hand", "l_foot"]
    assert abs(mv[0]["distance_cm"] - 100 * np.hypot(100, 130) / PPM) < 3
    assert 0.2 < mv[0]["time_s"] < 0.6  # 0.5 s transit minus the few cm near each hold
    # 4 limbs on holds except while a limb travels
    assert ws.n_contacts[int(1.5 * FPS)] == 4 and ws.n_contacts[int(3.25 * FPS)] == 3
    assert np.mean(ws.n_contacts[ws.on_wall] == 4) > 0.8
    s = wall_summary(ws, t)
    assert s["holds_used"] == 6 and s["moves"] == {"hand": 1, "foot": 1,
                                                   "time_s_median": s["moves"]["time_s_median"]}
    assert s["ground"] == "not seen" and s["on_wall_s"] > 9


def test_com_over_feet_sign():
    def plan(ti):
        return H["A"], H["B"], H["C"], H["D"]  # COM between the feet (x 420..600)

    pts, t = climb(plan, seconds=2)
    ws = compute_wall_support(pts, t, BODY, PPM)
    assert np.nanmedian(ws.com_over_feet_cm) > 0
    assert np.nanmedian(ws.polygon_margin_cm) > 0

    def leaning(ti):  # both feet far to the right of the body
        return (200, 300), (350, 300), (900, 900), (980, 900)

    pts2, t2 = climb(leaning, seconds=2)
    ws2 = compute_wall_support(pts2, t2, BODY, PPM)
    assert np.nanmedian(ws2.com_over_feet_cm) < 0


def test_ground_from_the_wall_image():
    """Standing on the floor (bottom of the wall image, flat grey) for 2 s, then stepping
    onto footholds (on the wall: wood colour below them)."""
    wall = np.full((1300, 1200, 3), (60, 140, 200), np.uint8)  # wood
    wall[1000:] = (200, 200, 200)  # floor
    for c in (H["C"], H["D"], (430, 830), (610, 850)):  # holds
        cv2_circle(wall, c)
    img = WallImage(wall, np.ones(wall.shape[:2], np.int32), np.eye(3))

    def plan(ti):
        if ti < 2.0:  # standing on the floor, hands on the start holds
            return H["A"], H["B"], (430, 1000), (610, 1000)
        u = (ti - 2.0) / 0.6
        return H["A"], H["B"], lerp((430, 1000), H["C"], u), lerp((610, 1000), H["D"], u - 0.8)

    pts, t = climb(plan, seconds=8)
    ws = compute_wall_support(pts, t, BODY, PPM, wall_image=img)
    first = {f: ws.anchors[f][0] for f in ("l_foot", "r_foot")}
    assert first["l_foot"].ground and first["r_foot"].ground
    assert not ws.anchors["l_foot"][-1].ground and not ws.anchors["r_foot"][-1].ground
    assert not ws.on_wall[: int(1.5 * FPS)].any() and ws.on_wall[int(5 * FPS):].all()
    assert wall_summary(ws, t)["ground"] == "stood on"
    assert all(not ({"l_foot", "r_foot"} & h.limbs) or h.pos[1] < 1000 for h in ws.holds)


def cv2_circle(img, c):
    import cv2

    cv2.circle(img, (int(c[0]), int(c[1])), 15, (40, 40, 160), -1)
