"""Smoothing layer, spine curve, flip permutation."""

import numpy as np

from climbvision.engine.backends._topdown import flip_permutation, steady_box
from climbvision.engine.skeleton import N_JOINTS, J, spine_curve
from climbvision.engine.smoothing import smooth_points

FPS = 30
BODY = 400.0


def _min_jerk(t, t0, dur, amp):
    s = np.clip((t - t0) / dur, 0, 1)
    return amp * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)


def _one_joint(y, noise=2.0, seed=0):
    rng = np.random.default_rng(seed)
    p = np.zeros((len(y), 1, 2))
    p[:, 0, 0] = rng.normal(0, noise, len(y))
    p[:, 0, 1] = y + rng.normal(0, noise, len(y))
    return p


def test_still_joint_gets_steadier():
    t = np.arange(0, 3, 1 / FPS)
    p = _one_joint(np.zeros_like(t))
    s, _ = smooth_points(p, t, np.ones((len(t), 1)), BODY)
    d2 = lambda x: np.abs(np.diff(x[:, 0, 1], 2)).mean()  # noqa: E731
    assert d2(s) < 0.3 * d2(p)


def test_fast_move_keeps_its_shape_and_speed():
    t = np.arange(0, 3, 1 / FPS)
    for dur in (0.4, 0.25):
        y = _min_jerk(t, 1.0, dur, BODY)  # one body length
        s, _ = smooth_points(_one_joint(y), t, np.ones((len(t), 1)), BODY)
        assert np.abs(s[:, 0, 1] - y).max() < 0.02 * BODY
        assert np.diff(s[:, 0, 1]).max() > 0.9 * np.diff(y).max()


def test_spike_and_hidden_stretch_are_replaced():
    t = np.arange(0, 3, 1 / FPS)
    y = np.linspace(0, 100, len(t))
    p = _one_joint(y)
    p[40:42, 0, 0] += 80  # a 2-frame jump: a wrong frame
    p[60:75, 0, 0] += 150  # 0.5 s hidden: the model guesses, with a low score
    conf = np.ones((len(t), 1))
    conf[60:75] = 0.1
    s, rep = smooth_points(p, t, conf, BODY)
    assert np.abs(s[38:44, 0, 0]).max() < 6
    assert np.abs(s[58:77, 0, 0]).max() < 6
    assert np.abs(s[58:77, 0, 1] - y[58:77]).max() < 6
    assert rep["gaps_filled"] >= 17


def test_long_hidden_stretch_keeps_the_guess():
    t = np.arange(0, 4, 1 / FPS)
    p = _one_joint(np.zeros_like(t))
    conf = np.ones((len(t), 1))
    conf[30:90] = 0.1  # 2 s: too long to bridge
    s, _ = smooth_points(p, t, conf, BODY)
    assert np.isfinite(s).all()
    np.testing.assert_allclose(s[40:80], p[40:80])


def _trunk(lh, rh, ls, rs):
    p = np.full((N_JOINTS, 2), np.nan)
    p[J.L_HIP], p[J.R_HIP], p[J.L_SHOULDER], p[J.R_SHOULDER] = lh, rh, ls, rs
    return p


def test_spine_straight_when_girdles_level():
    c = spine_curve(_trunk([-15, 0], [15, 0], [-20, -60], [20, -60]))
    np.testing.assert_allclose(c[:, 0], 0, atol=1e-9)
    np.testing.assert_allclose(c[[0, -1]], [[0, 0], [0, -60]])


def test_spine_bends_with_a_side_bend():
    # hips level, shoulders tilted: the trunk bends to one side (C shape)
    c = spine_curve(_trunk([-15, 0], [15, 0], [-20, -50], [20, -70]))
    chord = c[-1] - c[0]
    rel = c - c[0]
    off = (chord[0] * rel[:, 1] - chord[1] * rel[:, 0]) / np.linalg.norm(chord)
    assert np.abs(off).max() > 2
    assert (np.sign(off[1:-1]) == np.sign(off[len(off) // 2])).all()  # one side: a C


def test_flip_permutation_swaps_sides():
    pts = ["nose", "left_eye", "right_eye", "left_hand_wrist", "right_hand_wrist", "neck"]
    np.testing.assert_array_equal(flip_permutation(pts), [0, 2, 1, 4, 3, 5])


def test_steady_box_grows_fast_and_shrinks_slowly():
    b = steady_box(None, np.array([0, 0, 100, 100.0]))
    big = steady_box(b, np.array([0, 0, 200, 200.0]))
    small = steady_box(b, np.array([25, 25, 75, 75.0]))
    assert big[1][0] > 170
    assert small[1][0] > 90
