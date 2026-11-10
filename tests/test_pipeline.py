"""End-to-end pipeline test with a fake pose estimator (no MediaPipe needed)."""

import csv
import json

import cv2
import numpy as np
from synth import N_LANDMARKS

from climbvision.analysis import run_analysis
from climbvision.engine.pose import run_pose
from climbvision.engine.skeleton import RawPose
from climbvision.output.render import render


class FakeEstimator:
    """Returns a fixed skeleton, except every 5th frame where no person is found."""

    backend = "mediapipe_pose33"

    def __init__(self):
        rng = np.random.default_rng(0)
        self.img = rng.uniform(50, 150, size=(N_LANDMARKS, 2))
        self.world = rng.normal(size=(N_LANDMARKS, 3))
        self.last_t = -1

    def estimate(self, frame, time_ms):
        assert time_ms > self.last_t, "timestamps must increase"
        self.last_t = time_ms
        if (time_ms // 100) % 5 == 4:
            return None
        return RawPose(self.img, self.world, np.full(N_LANDMARKS, 0.9))

    def close(self):
        pass


def make_video(path, n=20, fps=10, size=(200, 160)):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n):
        w.write(np.full((size[1], size[0], 3), i * 10, np.uint8))
    w.release()


def test_analyze_writes_all_outputs(tmp_path):
    video = tmp_path / "clip.mp4"
    make_video(video)
    out = tmp_path / "out"
    summary = run_pose(video, out, FakeEstimator())

    for name in ("landmarks.csv", "com.csv", "summary.json"):
        assert (out / name).exists(), name

    assert summary["frames"] == 20
    assert summary["frames_with_pose"] == 16
    saved = json.loads((out / "summary.json").read_text())
    assert saved["frames"] == 20 and saved["backend"] == "mediapipe_pose33"

    with (out / "com.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 20
    assert sum(int(r["detected"]) for r in rows) == 16

    with (out / "landmarks.csv").open() as f:
        assert sum(1 for _ in f) == 1 + 16 * N_LANDMARKS

    # COM of a constant skeleton is constant
    assert summary["com_px"]["std"] == [0.0, 0.0]


def test_metrics_and_render_from_results(tmp_path):
    video = tmp_path / "clip.mp4"
    make_video(video)
    out = tmp_path / "out"
    run_pose(video, out, FakeEstimator())
    m = run_analysis(out, "none")
    assert "stability" in m and "angle_ranges" in m
    for name in ("angles.csv", "stability.csv", "metrics.json", "report.png"):
        assert (out / name).exists(), name
    path = render(video, out)
    cap = cv2.VideoCapture(str(path))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 20


def test_summary_headlines(tmp_path):
    from climbvision.output.summary import headline, markdown_table, summarize_results

    assert headline({"exercise": "squat", "reps": 1, "reps_below_parallel": 1,
                     "min_knee_deg_mean": 40.0}) == "1 rep, 1 below parallel, knee 40°"
    assert headline({"exercise": "balance", "balance": {"hold_s": 9.2, "stance": "R foot",
                                                       "sd_cm": 0.6}}).startswith("9.2 s on R")
    d = tmp_path / "clip"
    d.mkdir()
    (d / "metrics.json").write_text(json.dumps({"exercise": "pullup", "view": "front", "reps": 2,
                                                "pull_cm_mean": 45.0, "stability": {}}))
    rows = summarize_results([d, tmp_path / "missing"])
    assert len(rows) == 1 and rows[0]["result"].startswith("2 reps, pull 45 cm")
    assert "| clip |" in markdown_table(rows)
