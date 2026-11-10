"""Clean-up layer: glitches put into a synthetic climber are found and repaired; a clean
climb is left alone."""

import numpy as np
from test_wall import BODY, FPS, climb, two_moves

from climbvision.engine.cleanup import _two_bone, clean_points
from climbvision.engine.skeleton import J


def _err(a, b):
    return np.linalg.norm(a - b, axis=-1)


def test_clean_climb_is_left_alone():
    pts, t = climb(two_moves)
    out, fixed, wrong, rep = clean_points(pts, t, BODY)
    assert not wrong.any()
    assert np.nanmax(_err(out, pts)) < 0.05 * BODY  # nothing moved by more than noise
    assert fixed.mean() < 0.01


def test_glitches_are_repaired():
    truth, t = climb(two_moves)
    pts = truth.copy()
    # 1. hidden left knee lands on the right knee for one frame (drop knee)
    pts[100, J.L_KNEE] = truth[100, J.R_KNEE]
    # 2. left foot flickers onto the left hand every other frame for 0.3 s
    for i in range(150, 159, 2):
        for j in (J.L_ANKLE, J.L_HEEL, J.L_TOE):
            pts[i, j] = truth[i, J.L_HAND] + (truth[i, j] - truth[i, J.L_ANKLE]) * 0.1
    # 3. the toe alone sits far from its ankle for 1.5 s (longer than any median)
    pts[200:245, J.R_TOE] = truth[200:245, J.R_HAND]
    # 4. legs swapped for one frame
    for jl, jr in ((J.L_KNEE, J.R_KNEE), (J.L_ANKLE, J.R_ANKLE), (J.L_HEEL, J.R_HEEL),
                   (J.L_TOE, J.R_TOE), (J.L_SMALL_TOE, J.R_SMALL_TOE)):
        pts[60, [jl, jr]] = truth[60, [jr, jl]]
    out, fixed, wrong, rep = clean_points(pts, t, BODY)
    tol = 0.06 * BODY
    assert _err(out[100, J.L_KNEE], truth[100, J.L_KNEE]) < tol
    for i in range(150, 159, 2):
        assert _err(out[i, J.L_TOE], truth[i, J.L_TOE]) < tol
    assert np.max(_err(out[200:245, J.R_TOE], truth[200:245, J.R_TOE])) < tol
    assert np.max(_err(out[60, [J.L_KNEE, J.L_TOE, J.R_KNEE]],
                       truth[60, [J.L_KNEE, J.L_TOE, J.R_KNEE]])) < tol
    assert fixed[100, J.L_KNEE] and fixed[220, J.R_TOE]
    assert rep.total() > 0 and not wrong.any()


def test_knee_rebuilt_by_two_bone_ik_on_the_same_side():
    hip, ankle = np.array([0.0, 0.0]), np.array([0.0, 200.0])
    for side in (1.0, -1.0):
        k = _two_bone(hip, ankle, 150.0, 150.0, side)
        assert abs(np.linalg.norm(k - hip) - 150) < 1e-6
        assert abs(np.linalg.norm(ankle - k) - 150) < 1e-6
        d, m = ankle - hip, k - hip
        assert np.sign(d[0] * m[1] - d[1] * m[0]) == side
    # out of reach: straight, thigh-length from the hip
    k = _two_bone(hip, np.array([0.0, 400.0]), 150.0, 150.0, 1.0)
    assert np.allclose(k, [0, 150])


def test_impossible_thigh_rebuilds_the_knee():
    truth, t = climb(two_moves, seconds=4)
    pts = truth.copy()
    # knee far out for 0.5 s (too long to be flicker): thigh impossibly long
    pts[30:45, J.R_KNEE] = truth[30:45, J.R_HIP] + [0.6 * BODY, 0]
    out, fixed, wrong, _ = clean_points(pts, t, BODY)
    assert fixed[30:45, J.R_KNEE].all()
    assert np.max(_err(out[30:45, J.R_KNEE], truth[30:45, J.R_KNEE])) < 0.08 * BODY
    assert FPS == 30.0


def test_clean_track_repairs_a_glitch_and_flags_it():
    from climbvision.engine.cleanup import clean_track
    from climbvision.engine.skeleton import N_JOINTS
    from climbvision.engine.track import Track

    truth, t = climb(two_moves)
    pts = truth.copy()
    pts[50, J.L_KNEE] = truth[50, J.R_KNEE]
    n = len(t)
    tr = Track(t, pts, np.full((n, N_JOINTS, 3), np.nan), np.ones((n, N_JOINTS)))
    out, rep = clean_track(tr)
    assert _err(out.image[50, J.L_KNEE], truth[50, J.L_KNEE]) < 0.06 * BODY
    assert out.visibility[50, J.L_KNEE] == 0.5 and out.visibility[10, J.L_SHOULDER] == 1
    assert 0 < rep["frames_changed_pct"] < 2
