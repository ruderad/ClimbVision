"""Bar exercises on a synthetic front-view hanging skeleton (no video needed)."""

import numpy as np
import pytest

from climbvision.angles import compute_angles
from climbvision.exercises import dead_hang, leg_raise_reps, pullup_reps
from climbvision.skeleton import LM, N_LANDMARKS
from climbvision.stability import compute_stability, stability_summary
from climbvision.track import Track

FPS = 30.0
PPM = 700.0  # pixels per metre
BAR_Y = 200.0
UPPER, FORE = 210.0, 190.0  # arm segment lengths in px


def _elbow(shoulder, hand, outward):
    d = np.linalg.norm(hand - shoulder)
    alpha = np.arccos(np.clip((UPPER**2 + d**2 - FORE**2) / (2 * UPPER * d), -1, 1))
    u = (hand - shoulder) / d
    c, s = np.cos(alpha * outward), np.sin(alpha * outward)
    return shoulder + UPPER * np.array([c * u[0] - s * u[1], s * u[0] + c * u[1]])


def hanging(d: float = 395.0, leg_lift: float = 0.0) -> np.ndarray:
    """Front view, hands on a bar at y=200. d = hand-to-shoulder distance (395 = dead
    hang, smaller = pulled up). leg_lift in degrees raises straight legs forward."""
    p = np.zeros((N_LANDMARKS, 2))
    for side, sx, hx, out in (("LEFT", 610.0, 640.0, -1), ("RIGHT", 470.0, 440.0, 1)):
        hand = np.array([hx, BAR_Y])
        shoulder = np.array([sx, BAR_Y + np.sqrt(max(d**2 - (hx - sx) ** 2, 1.0))])
        hip = shoulder + [-(sx - 540) * 0.4, 320]
        # legs swing towards the camera in a front view: shown as shortening upwards
        lift = np.radians(leg_lift)
        knee = hip + [0, 240 * np.cos(lift)]
        ankle = knee + [0, 230 * np.cos(lift)]
        for name, pt in (("WRIST", hand + [0, 15]), ("INDEX", hand - [0, 10]),
                         ("PINKY", hand - [0, 5]), ("SHOULDER", shoulder),
                         ("ELBOW", _elbow(shoulder, hand, out)), ("HIP", hip),
                         ("KNEE", knee), ("ANKLE", ankle), ("HEEL", ankle + [0, 20]),
                         ("FOOT_INDEX", ankle + [0, 30]), ("EAR", shoulder + [-(sx - 540) * 0.5,
                                                                              -110])):
            p[LM[f"{side}_{name}"]] = pt
    mid = (p[LM.LEFT_SHOULDER] + p[LM.RIGHT_SHOULDER]) / 2
    p[LM.NOSE] = mid + [0, -100]  # chin estimate = mouth + 2 * (mouth - nose) = mid - 64
    p[LM.MOUTH_LEFT] = mid + [15, -88]
    p[LM.MOUTH_RIGHT] = mid + [-15, -88]
    return p


def make_track(poses, leg_lifts=None) -> Track:
    rng = np.random.default_rng(0)
    img = np.stack(poses) + rng.normal(0, 0.3, (len(poses), N_LANDMARKS, 2))
    hip = (img[:, LM.LEFT_HIP] + img[:, LM.RIGHT_HIP]) / 2
    world = np.concatenate([(img - hip[:, None]) / PPM, np.zeros((len(poses), N_LANDMARKS, 1))],
                           axis=2)
    if leg_lifts is not None:  # in 3D the legs come forward (towards the camera)
        for k, lift in enumerate(leg_lifts):
            a = np.radians(lift)
            for side in ("LEFT", "RIGHT"):
                h = world[k, LM[f"{side}_HIP"]]
                for name, r in (("KNEE", 240), ("ANKLE", 470), ("HEEL", 490),
                                ("FOOT_INDEX", 500)):
                    world[k, LM[f"{side}_{name}"]] = h + np.array([0, np.cos(a), -np.sin(a)]) \
                        * r / PPM
    return Track(np.arange(len(poses)) / FPS, img, world, np.full((len(poses), N_LANDMARKS), 0.9))


def pullup_clip(reps: int = 3, top_d: float = 32.0) -> Track:
    poses = [hanging()] * 30
    for _ in range(reps):
        for x in np.sin(np.linspace(0, np.pi, 60)):  # 2 s up and down
            poses.append(hanging(395 - (395 - top_d) * x))
        poses += [hanging()] * 15
    return make_track(poses + [hanging()] * 30)


def test_grips_and_hanging_without_floor():
    tr = pullup_clip()
    st = compute_stability(tr, "front", bar=True)
    assert st.contacts.grip["left"].mean() > 0.95 and st.contacts.grip["right"].mean() > 0.95
    assert st.hanging.mean() > 0.95  # nothing on the floor
    summ = stability_summary(st, tr.time_s)
    assert summ["bar"]["longest_hang_s"] > 8
    assert "floor not visible" in " ".join(summ.get("warnings", []))


def test_pullup_reps_rise_and_chin():
    tr = pullup_clip(reps=3, top_d=32)
    ang = compute_angles(tr)
    st = compute_stability(tr, ang.view, bar=True)
    reps, summ = pullup_reps(tr, ang, st)
    assert summ["reps"] == 3
    shoulder_rise = np.sqrt(395**2 - 30**2) - np.sqrt(32**2 - 30**2)
    assert summ["pull_cm_mean"] == pytest.approx(shoulder_rise / PPM * 100, rel=0.1)
    assert summ["reps_chin"]["over"] == 3
    assert summ["reps_kipping"] == 0

    shallow = pullup_clip(reps=2, top_d=220)
    ang2 = compute_angles(shallow)
    _, summ2 = pullup_reps(shallow, ang2, compute_stability(shallow, ang2.view, bar=True))
    assert summ2["reps"] == 2 and summ2["reps_chin"]["under"] == 2


def test_dead_hang_and_leg_raises():
    lifts = [0.0] * 30
    for _ in range(3):
        lifts += list(100 * np.sin(np.linspace(0, np.pi, 60))) + [0.0] * 15
    lifts += [0.0] * 30
    tr = make_track([hanging(leg_lift=a) for a in lifts], leg_lifts=lifts)
    ang = compute_angles(tr)
    st = compute_stability(tr, ang.view, bar=True)
    reps, summ = leg_raise_reps(tr, ang, st)
    assert summ["reps"] == 3
    assert summ["min_hip_deg_mean"] < 100
    hang = dead_hang(tr, ang, st)
    assert hang["hold_s"] > 8 and hang["elbow_deg_median"] > 150


def test_floor_learned_from_standing_frames_before_the_hang():
    """Standing under the bar first (feet on the floor at y=1500), then jumping up to hang."""
    stand = hanging()
    stand[:, 1] += 1500 - stand[LM.LEFT_FOOT_INDEX, 1]  # whole body down: feet on floor
    for side in ("LEFT", "RIGHT"):  # arms hanging by the sides
        sh = stand[LM[f"{side}_SHOULDER"]]
        stand[LM[f"{side}_ELBOW"]] = sh + [0, 210]
        for name in ("WRIST", "INDEX", "PINKY"):
            stand[LM[f"{side}_{name}"]] = sh + [0, 400]
    tr = make_track([stand] * 60 + [hanging()] * 90)
    st = compute_stability(tr, "front", bar=True)
    assert np.isfinite(st.floor.a)
    assert st.contacts.group["left_foot"][:50].all()
    assert st.hanging[70:].mean() > 0.9


def test_chin_over_bar_without_seeing_the_face():
    """Door frame hides the face at the top: the mouth landmarks there are garbage
    (pushed down to the chest). The body-based estimate must still say 'over'."""
    tr = pullup_clip(reps=2, top_d=32)
    shoulder_y = (tr.image[:, LM.LEFT_SHOULDER, 1] + tr.image[:, LM.RIGHT_SHOULDER, 1]) / 2
    top = shoulder_y < BAR_Y + 150
    for lm in (LM.MOUTH_LEFT, LM.MOUTH_RIGHT, LM.NOSE):
        tr.image[top, lm, 1] = shoulder_y[top] + 40
    ang = compute_angles(tr)
    _, summ = pullup_reps(tr, ang, compute_stability(tr, ang.view, bar=True))
    assert summ["reps"] == 2 and summ["reps_chin"]["over"] == 2
    assert summ["chin_above_shoulders_cm"] == pytest.approx(
        0.095 * compute_stability(tr, "front", bar=True).scale.body_px / PPM * 100, rel=0.05)
