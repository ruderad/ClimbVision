"""Technique on synthetic climbers: quiet feet vs re-placements, re-grips, pushes."""

import numpy as np
from test_wall import H, climb, lerp, two_moves

from climbvision.engine.signals import Clip
from climbvision.engine.skeleton import N_JOINTS
from climbvision.engine.technique import TechniqueParams, _count_pushes
from climbvision.engine.track import Track


def clip_of(plan, seconds=10.0):
    pts, t = climb(plan, seconds)
    n = len(t)
    return Clip.build(Track(t, pts, np.full((n, N_JOINTS, 3), np.nan), np.ones((n, N_JOINTS))))


def fidgety(ti):
    """two_moves, plus: the right foot lifted off D and put back (1.5-2.5 s), the left
    hand letting go of A and taking it again (7.5-8.5 s)."""
    lh, rh, lf, rf = two_moves(ti)
    if 1.5 <= ti < 2.5:
        rf = lerp(H["D"], (H["D"][0] + 120, H["D"][1] + 60), 1 - abs(ti - 2.0) / 0.5)
    if 7.5 <= ti < 8.5:
        lh = lerp(H["A"], (H["A"][0] - 100, H["A"][1] + 120), 1 - abs(ti - 8.0) / 0.5)
    return lh, rh, lf, rf


def test_quiet_feet_and_regrips():
    calm = clip_of(two_moves).technique
    busy = clip_of(fidgety).technique
    assert calm["foot_replacements"] == 0 and calm["quiet_feet_pct"] == 100.0
    assert busy["foot_replacements"] == 1 and busy["quiet_feet_pct"] < 100
    assert calm["hand_regrips"] == 0 and busy["hand_regrips"] == 1


def test_per_move_fields_and_summary():
    c = clip_of(two_moves)
    tech = c.technique
    m = next(m for m in c.route.moves if m.kind == "hand")
    assert m.pause_before_s > 2.0  # nothing moved from the start until 3 s
    assert np.isfinite(m.cycle_s) and m.com_pushes >= 1
    assert 0 < m.com_path_efficiency <= 1.0
    assert tech["feet_stayed_on_pct"] == 100.0
    for k in ("straight_arms_pct", "pause_before_hand_move_s_median", "com_pushes_median"):
        assert k in tech


def test_pushes_count_stop_and_go():
    t = np.linspace(0, 2, 120)
    one = 0.6 * np.sin(np.pi * t / 2)  # one smooth push
    two = np.abs(0.6 * np.sin(np.pi * t))  # stops in the middle
    jitter = one + 0.01 * np.sin(40 * t)  # small wobbles are not pushes
    prm = TechniqueParams()
    assert _count_pushes(one, prm) == 1
    assert _count_pushes(two, prm) == 2
    assert _count_pushes(jitter, prm) == 1
