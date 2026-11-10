import numpy as np
import pytest
from synth import LM, N_LANDMARKS, analyse, mp_track

from climbvision.engine.biomech import angle_at, compute_angles
from climbvision.engine.events import find_cycles
from climbvision.engine.signals import hip_offset_pct
from climbvision.engine.track import Track, smooth

FPS = 30.0


def side_pose(shin_tilt: float, thigh_tilt: float, trunk_tilt: float) -> np.ndarray:
    """Side-view skeleton in pixels (y down), facing +x. Tilts in degrees from vertical."""
    p = np.zeros((N_LANDMARKS, 2))
    a, b, c = np.radians([shin_tilt, thigh_tilt, trunk_tilt])
    ankle = np.array([500.0, 1000.0])
    knee = ankle + 220 * np.array([np.sin(a), -np.cos(a)])
    hip = knee + 230 * np.array([-np.sin(b), -np.cos(b)])
    shoulder = hip + 300 * np.array([np.sin(c), -np.cos(c)])
    toe = ankle + np.array([80.0, 0.0])
    heel = ankle + np.array([-20.0, 0.0])
    for s in ("LEFT", "RIGHT"):
        for name, pt in (("ANKLE", ankle), ("KNEE", knee), ("HIP", hip), ("SHOULDER", shoulder),
                         ("FOOT_INDEX", toe), ("HEEL", heel), ("ELBOW", shoulder + [0, 150]),
                         ("WRIST", shoulder + [0, 280]), ("EAR", shoulder + [10, -90])):
            p[LM[f"{s}_{name}"]] = pt
    p[LM.NOSE] = shoulder + [30, -100]
    return p


def make_track(poses: list[np.ndarray]) -> Track:
    img = np.stack(poses)
    n = len(img)
    # side view: left and right sides at different depths (body width ~0.3 m)
    z = np.array([0.15 if lm.name.startswith("LEFT") else -0.15 if lm.name.startswith("RIGHT")
                  else 0.0 for lm in LM])
    world = np.concatenate([img / 1000.0, np.broadcast_to(z[None, :, None], (n, N_LANDMARKS, 1))],
                           axis=2)
    vis = np.full((n, N_LANDMARKS), 0.9)
    vis[:, [lm for lm in LM if lm.name.startswith("RIGHT")]] = 0.6  # left = near side
    return mp_track(np.arange(n) / FPS, img, world, vis)


def test_angle_at():
    v = np.zeros(2)
    assert angle_at(np.array([1.0, 0]), v, np.array([0, 1.0])) == pytest.approx(90)
    assert angle_at(np.array([1.0, 0]), v, np.array([-1.0, 0])) == pytest.approx(180)


def test_side_view_angles():
    tr = make_track([side_pose(20, 50, 30)])
    ang = compute_angles(tr)
    assert ang.view == "side" and ang.side == "left"
    assert ang.primary("knee")[0] == pytest.approx(180 - 20 - 50, abs=0.01)
    assert ang.primary("trunk_lean")[0] == pytest.approx(30, abs=0.01)
    assert ang.primary("thigh")[0] == pytest.approx(90 - 50, abs=0.01)


def test_squat_rep_counting_and_depth():
    poses = []
    depths = [60, 95, 80]  # thigh tilt at the bottom; 95 = slightly below parallel
    for d in depths:
        for _ in range(30):  # 1 s standing
            poses.append(side_pose(0, 0, 0))
        for x in np.sin(np.linspace(0, np.pi, 75)):  # 2.5 s down and up
            poses.append(side_pose(30 * x, d * x, 35 * x))
    poses += [side_pose(0, 0, 0)] * 30
    tr = make_track(poses)
    m, _ = analyse(tr, "squat")
    reps = m["reps_detail"]
    assert m["reps"] == 3
    assert [r["below_parallel"] for r in reps] == [False, True, False]
    assert reps[1]["min_thigh_deg"] == pytest.approx(-5, abs=1.5)
    assert reps[0]["max_trunk_lean_deg"] == pytest.approx(35, abs=2)


def plank_pose(hip_lift: float) -> np.ndarray:
    """Horizontal body, shoulders at x=200, ankles at x=1200; hip_lift px upwards."""
    p = np.zeros((N_LANDMARKS, 2))
    for s in ("LEFT", "RIGHT"):
        p[LM[f"{s}_SHOULDER"]] = (200, 700)
        p[LM[f"{s}_ELBOW"]] = (200, 850)
        p[LM[f"{s}_WRIST"]] = (330, 850)
        p[LM[f"{s}_HIP"]] = (650, 745 - hip_lift)  # on the shoulder-ankle line
        p[LM[f"{s}_KNEE"]] = (930, 760)
        p[LM[f"{s}_ANKLE"]] = (1200, 800)
        p[LM[f"{s}_HEEL"]] = (1220, 780)
        p[LM[f"{s}_FOOT_INDEX"]] = (1210, 850)
        p[LM[f"{s}_EAR"]] = (120, 680)
    p[LM.NOSE] = (90, 700)
    return p


def test_hip_offset_sign():
    def off(lift):
        return hip_offset_pct(make_track([plank_pose(lift)]).image, "left")[0]

    straight, pike, sag = off(0), off(80), off(-60)
    assert abs(straight) < 1
    assert pike < -5 and sag > 5


def test_plank_hold_and_drift():
    poses = [plank_pose(0)] * 150 + [plank_pose(-80)] * 150  # 5 s straight, 5 s sagging
    tr = make_track(poses)
    m, _ = analyse(tr, "plank")
    assert m["hold_s"] == pytest.approx(10, abs=0.1)
    assert 40 < m["pct_time_straight"] < 60 and 40 < m["pct_time_sag"] < 60
    assert m["hip_offset_drift_pct"] > 5


def test_smooth_keeps_nans():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    y = smooth(x, 3)
    assert np.isnan(y[2]) and y[1] == pytest.approx(1.5)


def test_find_cycles_counts_dips():
    t = np.arange(300) / FPS
    sig = 170 - 60 * np.clip(np.sin(2 * np.pi * t / 2.5), 0, None)  # 4 dips in 10 s
    cyc = find_cycles(sig, t, 35, 15)
    assert len(cyc) == 4
    assert all(sig[b] == pytest.approx(110, abs=1) for _, b, _ in cyc)


@pytest.mark.parametrize("name,expected", [
    ("squat_side", "squat"), ("pistol_squat_front", "pistol_squat"),
    ("plank_to_pike_pushup_side", "pike_pushup"), ("plank_front", "plank"),
    ("jumping_jacks_front", "jumping_jacks"), ("tree_pose_front", "balance"),
    ("crow_pose_side", "balance"), ("standing_onefoot_side", "balance"),
    ("lunge_side", "none"), ("hang_front", "dead_hang"), ("dead_hang", "dead_hang"),
    ("narrow_pullup_front", "pullup"), ("knee_raise_hold_front", "leg_raise"),
    ("overhang_traverse", "none"), ("coordination_failed", "none")])
def test_guess_recipe(name, expected):
    from synth import RECIPES

    from climbvision.recipes import guess_recipe

    assert guess_recipe(name, RECIPES).name == expected
