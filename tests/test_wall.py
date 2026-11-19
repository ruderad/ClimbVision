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


def test_floor_from_the_push_off_not_from_standing_back_or_landing():
    """Feet standing back from the wall (or landing) project below the real floor on
    the wall plane: the floor comes from the last ground contact before the climb."""
    from climbvision.engine.wall import Anchor, _floor_points

    t = np.arange(300) / 30.0
    window = np.zeros(300, bool)
    window[120:240] = True
    a = dict(limb="l_foot", spread_px=1.0, ground=True)
    anchors = {"l_foot": [Anchor(start=0, end=60, pos=np.array([0.0, 1100.0]), **a),  # back
                          Anchor(start=90, end=119, pos=np.array([0.0, 1000.0]), **a),  # push-off
                          Anchor(start=250, end=300, pos=np.array([0.0, 1150.0]), **a)],
               "r_foot": [], "l_hand": [], "r_hand": []}
    assert _floor_points(anchors, window, t)[:, 1].tolist() == [1000.0]
    assert sorted(_floor_points(anchors, None, t)[:, 1].tolist()) == [1000.0, 1100.0, 1150.0]
    anchors["l_foot"] = anchors["l_foot"][2:]  # only the landing: use it
    assert _floor_points(anchors, window, t)[:, 1].tolist() == [1150.0]


def test_a_hand_on_a_hold_the_detector_missed_joins_or_makes_a_hold():
    from climbvision.engine.wall import Anchor, cluster_holds

    mk = lambda limb, s, x, box: Anchor(limb, s, s + 10, np.array([x, 100.0]), 1.0, box=box)  # noqa: E731
    holds = cluster_holds([mk("l_hand", 0, 0.0, 3), mk("r_hand", 20, 5.0, -1),
                           mk("r_hand", 40, 300.0, -1), mk("l_foot", 60, 600.0, 7)], 20.0)
    assert len(holds) == 3
    assert holds[0].box == 3 and holds[0].limbs == {"l_hand", "r_hand"}  # joined
    assert holds[1].box == -1 and holds[2].box == 7


def _anc(limb, t0, t1, x, box=-1, fps=30):
    from climbvision.engine.wall import Anchor

    return Anchor(limb, int(t0 * fps), int(t1 * fps), np.array([x, 100.0]), 1.0, box=box)


def test_a_grip_broken_by_a_hidden_moment_is_one_grip():
    from climbvision.engine.wall import WallParams, merge_fragments

    t = np.arange(600) / 30.0
    other = np.full((600, 2), 900.0)  # the other hand, far away
    vis = np.ones(600)
    a, b = _anc("r_hand", 1, 4, 100, box=5), _anc("r_hand", 4.6, 7, 104, box=5)
    out = merge_fragments([a, b], other, t, 500, WallParams(), vis)
    assert len(out) == 1 and out[0].start == a.start and out[0].end == b.end
    # a real let-go (shaking out, 2 s away) stays two grips
    c = _anc("r_hand", 9, 12, 104, box=5)
    assert len(merge_fragments([a, c], other, t, 500, WallParams(), vis)) == 2


def test_the_pose_putting_both_wrists_on_one_hand_is_not_a_move():
    from climbvision.engine.wall import WallParams, merge_fragments

    t = np.arange(600) / 30.0
    other = np.full((600, 2), 100.0)
    other[:, 0] = 400.0  # the other hand sits at x = 400
    vis = np.ones(600)
    a = _anc("l_hand", 1, 4, 100, box=1)
    b = _anc("l_hand", 4.3, 5.3, 400, box=2)  # "on" the other hand's hold for 1 s
    c = _anc("l_hand", 5.6, 9, 102, box=1)
    out = merge_fragments([a, b, c], other, t, 500, WallParams(), vis)
    assert len(out) == 1 and out[0].box == 1
    # B on a hold of its own, visible and long enough: a real move and back
    other[:, 0] = 900.0
    assert len(merge_fragments([a, b, c], other, t, 500, WallParams(), vis)) == 3


def test_local_pace_in_body_lengths_per_second():
    from climbvision.engine.wall import _local_pace

    t = np.arange(90) / 30.0
    com = np.zeros((90, 2))
    com[:, 1] = -200.0 * t  # rising 200 px/s
    pace = _local_pace(com, t, 400.0, 0.5)
    assert abs(pace[45] - 0.5) < 1e-6  # 200 px/s on a 400 px body


def test_heel_hook_counts_as_a_hold():
    """A foot above the hip whose heel (not toe) is on a detected hold is hooking."""
    from climbvision.engine.holddetect import HoldBoxes
    from climbvision.engine.skeleton import N_JOINTS

    n = 120
    t = np.arange(n) / 30.0
    p = np.full((n, N_JOINTS, 2), np.nan)
    base = {J.L_SHOULDER: (380, 400), J.R_SHOULDER: (420, 400), J.L_HIP: (385, 560),
            J.R_HIP: (415, 560), J.L_KNEE: (385, 700), J.L_ANKLE: (385, 820),
            J.L_TOE: (385, 850), J.L_HEEL: (380, 830), J.L_WRIST: (370, 250),
            J.R_WRIST: (430, 250), J.L_ELBOW: (375, 330), J.R_ELBOW: (425, 330),
            J.R_KNEE: (560, 520), J.R_ANKLE: (640, 470), J.R_HEEL: (650, 480),
            J.R_TOE: (700, 470)}
    for j, xy in base.items():
        p[:, j] = xy
    p[:, :, 0] += np.linspace(0, 60, n)[:, None]  # the body moves over the feet ...
    for j in (J.R_ANKLE, J.R_HEEL, J.R_TOE, J.L_TOE, J.L_HEEL, J.L_ANKLE, J.L_WRIST,
              J.R_WRIST):
        p[:, j] = base[j]  # ... while the hands and feet stay put
    boxes = np.array([[640, 465, 665, 495],  # the hold under the right heel
                      [360, 235, 385, 265], [420, 235, 445, 265],  # hand holds
                      [370, 835, 400, 865]], float)  # left foot hold
    ws = compute_wall_support(p, t, 500, 370, hold_boxes=HoldBoxes(boxes, np.ones(4), "t"))
    hooks = [a for a in ws.anchors["r_foot"] if a.hook]
    assert hooks and hooks[0].on_hold
