import numpy as np
import pytest

from climbvision.angles import compute_angles
from climbvision.skeleton import LM, N_LANDMARKS
from climbvision.stability import (
    compute_stability,
    estimate_floor,
    estimate_scale,
    stability_summary,
)
from climbvision.track import Track

FPS = 30.0
PX_PER_M = 700.0
FLOOR_Y = 1000.0


def standing(lean_px: float = 0.0, raise_right_px: float = 0.0) -> np.ndarray:
    """Side view, facing +x. Feet on the floor at y=1000; hands resting at hip height.

    lean_px shifts everything above the ankles forward; raise_right_px lifts the right foot.
    """
    p = np.zeros((N_LANDMARKS, 2))
    pos = {"ANKLE": (500, 990), "HEEL": (480, 1000), "FOOT_INDEX": (580, 1000),
           "KNEE": (500, 770), "HIP": (500, 540), "SHOULDER": (500, 240),
           "ELBOW": (500, 400), "WRIST": (505, 530), "INDEX": (510, 560),
           "PINKY": (505, 560), "EAR": (500, 150)}
    for s in ("LEFT", "RIGHT"):
        for name, (x, y) in pos.items():
            above = name not in ("ANKLE", "HEEL", "FOOT_INDEX")
            p[LM[f"{s}_{name}"]] = (x + (lean_px * (1000 - y) / 760 if above else 0), y)
    p[LM.NOSE] = p[LM.LEFT_EAR] + (30, 10)
    if raise_right_px:
        for name in ("ANKLE", "HEEL", "FOOT_INDEX", "KNEE"):
            p[LM[f"RIGHT_{name}"], 1] -= raise_right_px
    return p


def make_track(poses, noise: float = 0.3, seed: int = 0) -> Track:
    rng = np.random.default_rng(seed)
    img = np.stack(poses) + rng.normal(0, noise, (len(poses), N_LANDMARKS, 2))
    hip = (img[:, LM.LEFT_HIP] + img[:, LM.RIGHT_HIP]) / 2
    world = np.concatenate([(img - hip[:, None]) / PX_PER_M,
                            np.zeros((len(poses), N_LANDMARKS, 1))], axis=2)
    return Track(np.arange(len(poses)) / FPS, img, world, np.full((len(poses), N_LANDMARKS), 0.9))


def test_scale_and_floor_ignore_resting_hands():
    tr = make_track([standing()] * 60)
    sc = estimate_scale(tr)
    assert sc.px_per_m == pytest.approx(PX_PER_M, rel=0.02)
    fl = estimate_floor(tr, sc)
    assert fl.y(500, 0) == pytest.approx(FLOOR_Y, abs=3)


def test_contacts_and_one_foot_stance():
    poses = [standing()] * 30 + [standing(raise_right_px=150)] * 60 + [standing()] * 30
    tr = make_track(poses)
    st = compute_stability(tr, compute_angles(tr).view)
    left, right = st.contacts.group["left_foot"], st.contacts.group["right_foot"]
    assert left.all()
    assert right[:25].all() and not right[35:85].any() and right[-25:].all()
    for g in ("left_hand", "right_hand", "left_knee", "right_knee"):
        assert not st.contacts.group[g].any(), g
    summ = stability_summary(st, tr.time_s)
    assert summ["stance_time_s"]["L foot"] == pytest.approx(2.0, abs=0.3)


def test_margin_sign_and_size():
    tr = make_track([standing()] * 40)
    st = compute_stability(tr, "side")
    # COM straight above x~500; support heel 480 .. toe 580 -> nearest edge ~2-3 cm
    m = np.nanmedian(st.margin_cm)
    assert 1.0 < m < 6.0
    leaning = make_track([standing(lean_px=140)] * 40)
    st2 = compute_stability(leaning, "side")
    assert np.nanmedian(st2.margin_cm) < 0  # COM in front of the toes


def test_sway_measures_movement():
    poses = [standing(lean_px=20 * np.sin(2 * np.pi * k / 60)) for k in range(240)]
    tr = make_track(poses)
    summ = stability_summary(compute_stability(tr, "side"), tr.time_s)
    still = stability_summary(compute_stability(make_track([standing()] * 240), "side"),
                              np.arange(240) / FPS)
    assert summ["com_sway"]["range_cm"] > 3 * still["com_sway"]["range_cm"]


def test_feet_out_of_frame_warning():
    tr = make_track([standing()] * 40)
    st = compute_stability(tr, "side", frame_size=(1080, 990))
    assert st.feet_out_of_frame.all()
    assert "warnings" in stability_summary(st, tr.time_s)


def test_floor_follows_person_moving_away():
    """Pinhole camera at 0.857 m: walking away shrinks the person towards the horizon
    (y=400) and raises the floor in the image. Feet must stay 'on the floor'."""
    c = np.array([540.0, 400.0])
    ks = np.linspace(1.0, 0.6, 150)
    poses = [c + (standing() - c) * k for k in ks]
    tr = make_track(poses)
    tr.world[:] = make_track([standing()] * len(ks)).world  # true size never changes
    st = compute_stability(tr, "side")
    i, j = 20, len(ks) - 21  # away from the ends of the 1 s smoothing window
    assert st.scale.rel[i] / st.scale.rel[j] == pytest.approx(ks[i] / ks[j], rel=0.03)
    assert st.floor.depth_m == pytest.approx(600 / PX_PER_M, rel=0.1)
    assert st.contacts.group["left_foot"].mean() > 0.95
    assert st.contacts.group["right_foot"].mean() > 0.95


def test_jump_detected_and_planted_feet_are_not():
    from climbvision.stability import flight_phases, stance_labels

    rest = [standing()] * 30
    n_air = 9  # 0.3 s flight, parabolic up to 50 px
    air = [standing() - np.array([0, 200 * (k + 1) * (n_air - k) / (n_air + 1) ** 2])
           for k in range(n_air)]
    tr = make_track(rest + air + rest)
    st = compute_stability(tr, "side")
    jumps = flight_phases(stance_labels(st.contacts), tr.time_s, st)
    assert len(jumps) == 1
    assert stability_summary(st, tr.time_s)["jumps"]["count"] == 1

    still = make_track([standing()] * 70)
    st2 = compute_stability(still, "side")
    assert flight_phases(stance_labels(st2.contacts), still.time_s, st2) == []
