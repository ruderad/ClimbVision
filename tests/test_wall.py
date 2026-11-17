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


def body_at(lh, rh, lf, rf, rng, shift=(0.0, 0.0)):
    """A canonical skeleton (wall px, y down) with hands/feet at the given points; the
    trunk and head moved by `shift` (the body moving between still limbs)."""
    p = np.full((N_JOINTS, 2), np.nan)
    mid = (np.array(lh) + np.array(rh) + np.array(lf) + np.array(rf)) / 4 + np.array(shift)
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
    """plan(t) -> (lh, rh, lf, rf[, body shift]). Returns (n, N_JOINTS, 2) and times."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FPS)) / FPS
    out = []
    for ti in t:
        r = plan(ti)
        out.append(body_at(*r[:4], rng, *r[4:]))
    return np.stack(out), t


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


def test_climb_recipe_outputs_plain_numbers():
    """Every value in the climb recipe's summary is a number (not an [expr, digits] list)."""
    from climbvision.engine.signals import Clip
    from climbvision.engine.track import Track
    from climbvision.recipes import load_recipes, run_recipe

    pts, t = climb(two_moves)
    n = len(t)
    tr = Track(t, pts, np.full((n, N_JOINTS, 3), np.nan), np.ones((n, N_JOINTS)))
    res = run_recipe(load_recipes()["climb"], Clip.build(tr))
    for k in ("climb_s", "pct_4_contacts", "feet_on_holds_pct", "height_gain_m"):
        assert isinstance(res.metrics[k], float), k


def test_floor_mask_ignores_a_textured_wall_of_the_floor_colour_and_fills_a_towel():
    import cv2

    from climbvision.engine.wall import floor_mask

    rng = np.random.default_rng(0)
    img = np.full((600, 400, 3), 200, np.uint8)  # light grey floor ...
    img[:380] = rng.integers(150, 250, (380, 400, 3)).astype(np.uint8)  # ... and a wall
    for _ in range(120):  # panel of about the same colour, with texture and holds
        c = (int(rng.integers(0, 400)), int(rng.integers(0, 370)))
        cv2.circle(img, c, int(rng.integers(4, 12)), (60, 60, 60), -1)
    img[480:510, 150:230] = (40, 60, 160)  # a towel lying on the floor
    m = floor_mask(WallImage(img, np.ones(img.shape[:2], np.int32), np.eye(3)))
    assert m[560, 200] and m[495, 190]  # floor, and the towel on it
    assert m[:360].mean() < 0.05  # the wall is not floor


def test_dark_wall_panel_with_holds_at_the_bottom_is_not_floor():
    """Camera close to the wall: the bottom of the wall image is a flat dark panel full
    of holds, no floor."""
    import cv2

    from climbvision.engine.wall import floor_mask

    rng = np.random.default_rng(1)
    img = np.full((600, 400, 3), (90, 170, 220), np.uint8)  # light wood panel
    img[350:] = (35, 35, 35)  # dark panel down to the bottom
    for x in range(20, 400, 40):  # a grid of holds, like a training wall
        for y in range(375, 600, 40):
            cv2.circle(img, (x + int(rng.integers(-5, 6)), y), 7, (150, 150, 200), -1)
    m = floor_mask(WallImage(img, np.ones(img.shape[:2], np.int32), np.eye(3)))
    assert m is None or m[360:].mean() < 0.05


def test_speckled_concrete_floor_is_still_floor():
    """Concrete and mats have small specks that fail the flatness test; they are not
    holds, so they must not turn the floor into a "wall panel"."""
    import cv2

    from climbvision.engine.wall import floor_mask

    rng = np.random.default_rng(2)
    img = np.full((600, 400, 3), (90, 170, 220), np.uint8)  # wood wall ...
    for _ in range(40):  # ... with holds
        c = (int(rng.integers(0, 400)), int(rng.integers(0, 330)))
        cv2.circle(img, c, int(rng.integers(8, 16)), (60, 60, 200), -1)
    img[400:] = (150, 150, 150)  # grey concrete floor ...
    for _ in range(150):  # ... full of small specks
        c = (int(rng.integers(0, 400)), int(rng.integers(405, 600)))
        cv2.circle(img, c, int(rng.integers(1, 3)), (125, 125, 125), -1)
    m = floor_mask(WallImage(img, np.ones(img.shape[:2], np.int32), np.eye(3)))
    assert m is not None and m[420:].mean() > 0.9
    assert m[:380].mean() < 0.05


def test_detected_holds_make_a_bottom_panel_a_wall():
    import cv2

    from climbvision.engine.holddetect import HoldBoxes
    from climbvision.engine.wall import floor_mask

    img = np.full((600, 400, 3), (90, 170, 220), np.uint8)
    img[350:] = (35, 35, 35)  # a dark panel with a few small holds at the bottom
    pts = [(60, 420), (160, 470), (260, 430), (340, 520), (110, 560), (220, 540)]
    for x, y in pts:
        cv2.circle(img, (x, y), 5, (150, 150, 200), -1)
    wi = WallImage(img, np.ones(img.shape[:2], np.int32), np.eye(3))
    assert floor_mask(wi) is not None  # too few hold-sized things to tell by itself ...
    boxes = np.array([[x - 6, y - 6, x + 6, y + 6] for x, y in pts], float)
    m = floor_mask(wi, hold_boxes=HoldBoxes(boxes, np.ones(len(pts)), "test"))
    assert m is None or m[360:].mean() < 0.05  # ... the detector tells


def _smear_plan(body_moves: bool):
    smear_spot = (380, 700)

    def plan(ti):
        lh, rh, lf, rf = two_moves(ti)
        if 7.0 <= ti < 9.0:
            lf = lerp(H["F"], smear_spot, (ti - 7.0) / 0.3)
        elif ti >= 9.0:
            lf = lerp(smear_spot, H["F"], (ti - 9.0) / 0.3)
        # the body rises 15 cm over the foot and back (friction held it) — or stays put
        rise = np.interp(ti, [7.4, 8.0, 8.6], [0, -90, 0]) if body_moves else 0.0
        return lh, rh, lf, rf, (0.0, rise)

    return plan


def test_detected_holds_split_holds_from_smears():
    """With holds found in the wall image, a rest on a hold is a hold; a rest elsewhere
    that the body moves over (the left foot pressed on the wall at 7-9 s) is a smear: no
    hold, no move, but part of the base of support."""
    from climbvision.engine.holddetect import HoldBoxes

    pts, t = climb(_smear_plan(True), seconds=11)
    boxes = np.array([[x - 25, y - 25, x + 25, y + 25] for x, y in H.values()], float)
    hb = HoldBoxes(boxes, np.ones(len(boxes)), "test")
    ws = compute_wall_support(pts, t, BODY, PPM, hold_boxes=hb)
    assert ws.holds_source == "detected"
    smears = [a for a in ws.anchors["l_foot"] if a.smear]
    assert len(smears) == 1 and 7.0 < t[smears[0].start] < 7.8
    assert len(ws.holds) == 6 and all(h.box >= 0 for h in ws.holds)
    i = int(8.0 * FPS)
    assert not ws.on_hold["l_foot"][i] and ws.on_smear["l_foot"][i]
    assert "L foot smear" in ws.labels[i]
    assert np.isfinite(ws.polygon_margin_cm[i])  # the smear is in the base of support
    assert [m["limb"] for m in moves(ws, t)] == ["r_hand", "l_foot"]  # F -> smear -> F: none
    s = wall_summary(ws, t)
    assert s["smears"]["count"] == 1 and s["holds_detected"] == 6


def test_still_foot_off_any_hold_is_a_flag():
    """The same foot off the holds while the body doesn't move: not proven to touch (a
    flag held still looks the same) — no contact, not in the base of support."""
    from climbvision.engine.holddetect import HoldBoxes

    pts, t = climb(_smear_plan(False), seconds=11)
    boxes = np.array([[x - 25, y - 25, x + 25, y + 25] for x, y in H.values()], float)
    ws = compute_wall_support(pts, t, BODY, PPM, hold_boxes=HoldBoxes(boxes, np.ones(6), "t"))
    flags = [a for a in ws.anchors["l_foot"] if a.flag]
    assert len(flags) == 1 and not any(a.smear for a in ws.anchors["l_foot"])
    i = int(8.0 * FPS)
    assert not ws.on_hold["l_foot"][i] and not ws.on_smear["l_foot"][i]
    assert wall_summary(ws, t)["flags"]["count"] == 1


def test_without_a_detector_every_rest_is_a_hold():
    pts, t = climb(two_moves)
    ws = compute_wall_support(pts, t, BODY, PPM)
    assert ws.holds_source == "rests" and not any(
        a.smear for limb in ws.anchors for a in ws.anchors[limb])


def test_hanging_with_the_feet_off_is_on_the_wall():
    """Both hands on holds above the shoulders, legs swinging (campusing, kicking for a
    heel hook): on the wall the whole time, even though no foot ever rests."""
    def plan(ti):
        swing = 150 * np.sin(2 * np.pi * 0.8 * ti)
        return H["A"], H["B"], (420 + swing, 950), (600 + swing, 960)

    pts, t = climb(plan, seconds=6)
    ws = compute_wall_support(pts, t, BODY, PPM)
    assert ws.on_wall.mean() > 0.95 and not ws.on_ground.any()
    assert ws.on_hold["l_hand"].mean() > 0.9 and not ws.on_hold["l_foot"].any()
