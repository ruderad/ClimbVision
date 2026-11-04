import numpy as np
import pytest

from climbvision.com import SEGMENTS, com_confidence, segment_params, whole_body_com
from climbvision.skeleton import LM, N_LANDMARKS


def standing_pose() -> np.ndarray:
    """Front view of a 1.8 m person standing upright, in metres, y up, origin between feet."""
    p = np.zeros((N_LANDMARKS, 2))
    pos = {
        "EAR": (0.07, 1.62), "SHOULDER": (0.19, 1.47), "ELBOW": (0.21, 1.15),
        "WRIST": (0.22, 0.88), "INDEX": (0.22, 0.80), "PINKY": (0.23, 0.80),
        "HIP": (0.09, 0.94), "KNEE": (0.10, 0.50), "ANKLE": (0.10, 0.08),
        "HEEL": (0.10, 0.02), "FOOT_INDEX": (0.10, 0.02),
    }
    for name, (x, y) in pos.items():
        p[LM[f"LEFT_{name}"]] = (-x, y)
        p[LM[f"RIGHT_{name}"]] = (x, y)
    p[LM.NOSE] = (0.0, 1.65)
    return p


@pytest.mark.parametrize("sex", ["male", "female"])
def test_mass_fractions_sum_to_one(sex):
    masses, ratios = segment_params(sex)
    assert len(masses) == len(SEGMENTS) == 14
    assert masses.sum() == pytest.approx(1.0)
    assert np.all((ratios >= 0) & (ratios <= 1))


def test_standing_com_is_centred_at_plausible_height():
    com = whole_body_com(standing_pose())
    assert com[0] == pytest.approx(0.0, abs=1e-9)  # symmetric pose
    # adult standing COM is ~55-57% of stature (1.8 m -> ~0.99-1.03 m)
    assert 0.95 < com[1] < 1.07


def test_translation_equivariance():
    p = standing_pose()
    shift = np.array([3.0, -2.0])
    np.testing.assert_allclose(whole_body_com(p + shift), whole_body_com(p) + shift)


def test_works_in_3d():
    p2 = standing_pose()
    p3 = np.column_stack([p2, np.full(N_LANDMARKS, 0.5)])
    com = whole_body_com(p3)
    assert com.shape == (3,)
    np.testing.assert_allclose(com[:2], whole_body_com(p2))
    assert com[2] == pytest.approx(0.5)


def test_squat_lowers_com():
    stand = standing_pose()
    squat = stand.copy()
    for side in ("LEFT", "RIGHT"):
        # hips drop and move back (−y shift), knees forward
        for name in ("EAR", "SHOULDER", "ELBOW", "WRIST", "INDEX", "PINKY", "HIP"):
            squat[LM[f"{side}_{name}"], 1] -= 0.40
    assert whole_body_com(squat)[1] < whole_body_com(stand)[1] - 0.25


def test_confidence():
    assert com_confidence(np.ones(N_LANDMARKS)) == pytest.approx(1.0)
    assert com_confidence(np.zeros(N_LANDMARKS)) == pytest.approx(0.0)
    vis = np.ones(N_LANDMARKS)
    vis[LM.LEFT_HEEL] = 0.0  # hiding one foot costs only that foot's mass share
    masses, _ = segment_params()
    foot = [s.name for s in SEGMENTS].index("left_foot")
    assert com_confidence(vis) == pytest.approx(1.0 - masses[foot])
