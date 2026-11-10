"""Pose backends: mapping onto the canonical skeleton, 2D-only fallbacks, resumable pose
pass, and the default choice per scene. No model files needed."""

import json

import cv2
import numpy as np
import pytest

from climbvision.engine import backends
from climbvision.engine.backends import rtmpose_halpe26 as halpe
from climbvision.engine.backends import rtmw_wholebody as wb
from climbvision.engine.biomech import compute_angles
from climbvision.engine.pose import run_pose, run_pose_resumable
from climbvision.engine.scale import ASSUMED_BODY_M, estimate_scale
from climbvision.engine.skeleton import J, RawPose
from climbvision.engine.track import Track

STAND = {"nose": (500, 150), "left_ear": (540, 150), "right_ear": (460, 150),
         "head": (500, 100), "neck": (500, 220), "left_shoulder": (580, 250),
         "right_shoulder": (420, 250), "left_elbow": (600, 400), "right_elbow": (400, 400),
         "left_wrist": (610, 530), "right_wrist": (390, 530), "left_hip": (550, 550),
         "right_hip": (450, 550), "hip": (500, 550), "left_knee": (550, 780),
         "right_knee": (450, 780), "left_ankle": (550, 1000), "right_ankle": (450, 1000),
         "left_big_toe": (580, 1020), "right_big_toe": (420, 1020),
         "left_small_toe": (590, 1015), "right_small_toe": (410, 1015),
         "left_heel": (545, 1020), "right_heel": (455, 1020),
         "left_hand_middle1": (612, 600), "right_hand_middle1": (388, 600)}


def raw(points, n=1):
    img = np.array([STAND.get(p, (np.nan, np.nan)) for p in points], float)
    img[np.isnan(img).any(axis=1)] = (500, 500)  # face points etc.: anywhere
    return np.repeat(img[None], n, 0), np.full((n, len(points), 3), np.nan), \
        np.full((n, len(points)), 0.9)


def test_halpe26_mapping_and_derived_hand():
    img, wld, vis = raw(halpe.POINTS)
    c, w, v = backends.to_canonical(halpe.NAME, img, wld, vis)
    assert tuple(c[0, J.L_TOE]) == STAND["left_big_toe"]
    assert tuple(c[0, J.R_SMALL_TOE]) == STAND["right_small_toe"]
    assert tuple(c[0, J.HEAD_TOP]) == STAND["head"] and tuple(c[0, J.NECK]) == STAND["neck"]
    assert tuple(c[0, J.HEAD]) == (500, 150)  # mid-ears
    # no hand point in Halpe-26: 0.3 forearm lengths beyond the wrist
    wr, el = np.array(STAND["left_wrist"]), np.array(STAND["left_elbow"])
    np.testing.assert_allclose(c[0, J.L_HAND], wr + 0.3 * (wr - el))
    assert np.isnan(w).all() and v[0, J.L_HAND] == pytest.approx(0.9)


def test_wholebody_hand_is_the_middle_knuckle():
    assert len(wb.POINTS) == 133
    img, wld, vis = raw(wb.POINTS)
    c, _, _ = backends.to_canonical(wb.NAME, img, wld, vis)
    assert tuple(c[0, J.L_HAND]) == STAND["left_hand_middle1"]
    assert tuple(c[0, J.R_HEEL]) == STAND["right_heel"]
    assert np.isnan(c[0, J.NECK]).all()  # not in COCO-WholeBody


def test_2d_only_scale_and_angles_fall_back():
    img, wld, vis = raw(halpe.POINTS, n=30)
    tr = Track.from_raw(np.arange(30) / 30, img, wld, vis, halpe.NAME)
    sc = estimate_scale(tr)
    body = 300 + 450  # trunk (250 -> 550) + leg (550 -> 1000)
    assert sc.body_px == pytest.approx(body, rel=0.01)
    assert sc.px_per_m == pytest.approx(body / ASSUMED_BODY_M, rel=0.01)
    ang = compute_angles(tr)
    assert ang.view == "front"
    assert np.isfinite(ang.primary("knee")).all()  # 2D, since there is no 3D
    assert ang.primary("knee")[0] == pytest.approx(180, abs=1)


class FakeEstimator:
    backend = halpe.NAME

    def __init__(self):
        self.base = raw(halpe.POINTS)[0][0]

    def estimate(self, frame, time_ms):
        if frame.mean() > 200:  # a few "no person" frames
            return None
        shift = frame.mean()
        return RawPose(self.base + shift, np.full((26, 3), np.nan), np.full(26, 0.8))

    def close(self):
        pass


def make_video(path, n=40):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (64, 48))
    for i in range(n):
        vw.write(np.full((48, 64, 3), 250 if i in (10, 11) else 3 * i, np.uint8))
    vw.release()


def test_resumable_pose_pass_matches_one_go(tmp_path):
    video = tmp_path / "v.mp4"
    make_video(video)
    ref = run_pose(video, tmp_path / "one", FakeEstimator())
    out = tmp_path / "parts"
    rounds = 0
    while True:
        rounds += 1
        res = run_pose_resumable(video, out, FakeEstimator(), budget_s=0.0)
        if res["done"]:
            break
    assert rounds > 5  # it really stopped and resumed
    assert not (out / "pose_partial.npz").exists()
    a = (tmp_path / "one" / "landmarks.csv").read_text()
    b = (out / "landmarks.csv").read_text()
    assert a == b
    s = json.loads((out / "summary.json").read_text())
    assert s["frames_with_pose"] == ref["frames_with_pose"] == 38


def test_every_other_frame(tmp_path):
    video = tmp_path / "v.mp4"
    make_video(video)
    res = run_pose_resumable(video, tmp_path / "s2", FakeEstimator(), step=2)
    assert res["done"] and res["frames"] == 20 and res["frame_step"] == 2
    tr = Track.from_csv(tmp_path / "s2" / "landmarks.csv")
    assert tr.fps == pytest.approx(10, rel=0.05)


def test_default_backend_per_scene(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIMBVISION_MODELS", str(tmp_path))  # no model files here
    assert backends.choose("floor") == backends.DEFAULT
    assert backends.choose("wall") == backends.DEFAULT  # falls back
    assert not backends.available(wb.NAME)
