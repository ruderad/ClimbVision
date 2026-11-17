"""Forces at the hands and feet: balance, the least-cost split, what feet can do."""

import numpy as np
from test_wall import BODY, PPM, climb, two_moves

from climbvision.engine.forces import ForceParams, _solve, compute_forces, forces_summary
from climbvision.engine.wall import compute_wall_support

PRM = ForceParams()
UP = np.array([0.0, 1.0])  # standing still: the holds carry 1 body weight, upwards


def test_dead_hang_two_hands_share_equally():
    com = np.array([0.0, 1.0])
    f = _solve([np.array([-0.3, 2.0]), np.array([0.3, 2.0])], [False, False], UP, com, PRM,
               True)
    assert np.allclose(f.sum(0), UP) and np.allclose(f[0], f[1] * [-1, 1], atol=1e-9)
    assert np.allclose(f[:, 1], 0.5)


def test_one_hand_carries_everything():
    f = _solve([np.array([0.0, 2.0])], [False], UP, np.array([0.0, 1.0]), PRM, True)
    assert np.allclose(f, [UP])


def test_feet_take_most_when_the_com_is_over_them():
    pts = [np.array([-0.2, 1.8]), np.array([0.2, 1.8]),  # hands
           np.array([-0.15, 0.0]), np.array([0.15, 0.0])]  # feet
    f = _solve(pts, [False, False, True, True], UP, np.array([0.0, 1.0]), PRM, True)
    hands, feet = np.linalg.norm(f[:2], axis=1).sum(), np.linalg.norm(f[2:], axis=1).sum()
    assert np.allclose(f.sum(0), UP) and feet > 2 * hands
    assert (f[2:, 1] >= 0).all() and (np.abs(f[2:, 0]) <= PRM.mu * f[2:, 1] + 1e-9).all()


def test_feet_cannot_pull():
    """COM far out to the side of both feet, a hand above: the hand must take it."""
    pts = [np.array([1.0, 2.0]), np.array([-0.2, 0.0]), np.array([0.0, 0.0])]
    f = _solve(pts, [False, True, True], UP, np.array([1.0, 1.0]), PRM, True)
    assert f is not None and (f[1:, 1] >= -1e-9).all()
    assert np.linalg.norm(f[0]) > 0.3  # the hand holds the body from swinging out


def test_a_dyno_catch_loads_the_hand_more():
    catch = np.array([0.0, 2.0])  # decelerating upwards motion: need = a + g > g
    f = _solve([np.array([0.0, 2.0])], [False], catch, np.array([0.0, 1.0]), PRM, True)
    assert np.isclose(f[0, 1], 2.0)


def test_forces_on_a_synthetic_climb():
    pts, t = climb(two_moves)
    ws = compute_wall_support(pts, t, BODY, PPM)
    fz = compute_forces(ws, t, mass_kg=70)
    on = ws.on_wall & np.isfinite(fz.hands_bw)
    assert on.mean() > 0.8
    total = sum(fz.force[limb][on] for limb in fz.force)
    assert np.allclose(np.nanmedian(total[:, 1]), 1.0, atol=0.05)  # one body weight
    s = forces_summary(fz, ws, t)
    assert 0 < s["hands_share_pct_median"] < 100 and s["hands_load_n_peak"] > 0


def test_free_flight_arc_and_swing_sign():
    """The momentum arc is a parabola under g; a counter-clockwise swing is positive."""
    import numpy as np

    from climbvision.engine.dynamics import G

    p0, v0 = np.array([0.0, 1.0]), np.array([0.5, 2.0])
    ts = np.arange(0, 0.8, 1 / 30)
    arc = p0 + np.outer(ts, v0) + 0.5 * np.outer(ts ** 2, [0, -G])
    k = int(np.argmax(arc[:, 1]))
    assert abs(ts[k] - v0[1] / G) < 1 / 30  # apex when vy = 0
    r, v = np.array([0.0, -1.0]), np.array([1.0, 0.0])  # below the pivot, moving right
    assert np.degrees((r[0] * v[1] - r[1] * v[0]) / (r @ r)) > 0  # counter-clockwise


def test_a_smear_carries_less_than_a_foothold():
    """Same place, same need: as a smear (weak contact) the foot takes less of the load
    than as a foothold, and the hand more."""
    import numpy as np

    from climbvision.engine.forces import ForceParams, _solve

    prm = ForceParams()
    pts = [np.array([0.0, 0.9]), np.array([0.15, -0.8])]  # hand above, foot below
    feet = [False, True]
    need, com = np.array([0.0, 1.0]), np.array([0.05, 0.0])
    hold = _solve(pts, feet, need, com, prm, moments=True, costs=[prm.hand_cost, 1.0])
    smear = _solve(pts, feet, need, com, prm, moments=True,
                   costs=[prm.hand_cost, prm.smear_cost])
    assert np.linalg.norm(smear[1]) < np.linalg.norm(hold[1])
    assert np.linalg.norm(smear[0]) > np.linalg.norm(hold[0])
