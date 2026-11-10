"""Route layer on synthetic climbers: moves numbered and tagged, rests, feet cut."""

import numpy as np
from test_wall import BODY, FPS, PPM, H, climb, lerp, two_moves

from climbvision.engine.route import RouteParams, compute_route, route_summary
from climbvision.engine.wall import compute_wall_support

J_ = (700, 100)
K_ = (500, 80)
G_ = (900, 280)


def tricks(ti):
    """0-2 s rest; R hand B->A (match); L hand A->G (cross); both hands at once (dyno);
    both feet off 6.0-6.7 s (feet cut); R foot -> B, a handhold above the COM."""
    lh, rh, lf, rf = H["A"], H["B"], H["C"], H["D"]
    rh = lerp(H["B"], H["A"], (ti - 2.0) / 0.4)
    lh = lerp(H["A"], G_, (ti - 3.0) / 0.5)
    if ti >= 4.5:
        lh = lerp(G_, J_, (ti - 4.5) / 0.4)
        rh = lerp(H["A"], K_, (ti - 4.5) / 0.4)
    if 6.0 <= ti < 6.7:  # feet swinging off the wall
        sw = np.sin((ti - 6.0) * 12) * 60
        lf, rf = (H["C"][0] + sw, H["C"][1] + 40), (H["D"][0] + sw, H["D"][1] + 40)
    rf = lerp(rf, H["B"], (ti - 8.0) / 0.5) if ti >= 8.0 else rf
    return lh, rh, lf, rf


def test_two_moves_rest_and_others():
    pts, t = climb(two_moves)
    ws = compute_wall_support(pts, t, BODY, PPM)
    r = compute_route(ws, t)
    assert [(m.n, m.limb) for m in r.moves] == [(1, "r_hand"), (2, "l_foot")]
    assert r.moves[0].others == "L hand + 2 feet" and r.moves[0].tags == []
    assert 40 < r.moves[0].direction_deg < 65  # up and to the right
    assert r.rests and t[r.rests[0][0]] < 0.5 and t[r.rests[0][1] - 1] > 2.5
    assert not r.feet_cut
    s = route_summary(r, ws, t)
    assert s["hand_moves"] == 1 and s["foot_moves"] == 1 and s["rests"]["count"] >= 2
    assert any(lab.startswith("move 1: R hand") for lab in r.labels)


def test_tags_match_cross_dyno_feet_cut_high_step():
    pts, t = climb(tricks, seconds=10)
    ws = compute_wall_support(pts, t, BODY, PPM)
    r = compute_route(ws, t, RouteParams(dyno_speed=0.2))  # a slow synthetic dyno
    assert not any("dyno" in m.tags for m in compute_route(ws, t).moves)
    by = {(m.limb, round(m.catch_s)): m for m in r.moves}
    tags = {k: set(m.tags) for k, m in by.items()}
    assert "match" in tags[("r_hand", 2)]
    assert "cross" in tags[("l_hand", 4)] or "cross" in tags[("l_hand", 3)]
    dyno = [m for m in r.moves if 4.4 < m.catch_s < 5.5]
    assert len(dyno) == 2 and all("dyno" in m.tags for m in dyno)
    assert len(r.feet_cut) == 1
    a, b = r.feet_cut[0]
    assert 5.9 < t[a] < 6.4 and 6.4 < t[b - 1] < 7.2
    step = [m for m in r.moves if m.limb == "r_foot" and m.catch_s > 8]
    assert step and {"high step", "foot on handhold"} <= set(step[0].tags)
    assert [m.n for m in r.moves] == list(range(1, len(r.moves) + 1))
    assert FPS == 30.0


def test_route_map_and_tables(tmp_path):
    import cv2

    from climbvision.engine.signals import Clip
    from climbvision.engine.skeleton import N_JOINTS
    from climbvision.engine.track import Track
    from climbvision.output.route_map import draw_route_map
    from climbvision.output.tables import write_holds_csv, write_moves_csv

    pts, t = climb(tricks, seconds=10)
    n = len(t)
    clip = Clip.build(Track(t, pts, np.full((n, N_JOINTS, 3), np.nan), np.ones((n, N_JOINTS))))
    img = cv2.imread(str(draw_route_map(tmp_path / "route.png", clip)))
    assert img is not None and min(img.shape[:2]) > 200
    write_moves_csv(tmp_path / "moves.csv", clip.route)
    write_holds_csv(tmp_path / "holds.csv", clip.route)
    rows = (tmp_path / "moves.csv").read_text().splitlines()
    assert len(rows) == len(clip.route.moves) + 1 and "high step" in "".join(rows)
    assert clip.signal("resting").any() and len(clip.signal("feet_cut")) == n
