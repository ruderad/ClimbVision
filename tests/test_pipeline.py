"""End-to-end pipeline test with a fake pose estimator (no MediaPipe needed)."""

import csv
import json

import cv2
import numpy as np

from climbvision.pipeline import analyze
from climbvision.skeleton import N_LANDMARKS, Skeleton


class FakeEstimator:
    """Returns a fixed skeleton, except every 5th frame where no person is found."""

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
        return Skeleton(self.img, self.world, np.full(N_LANDMARKS, 0.9))

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
    summary = analyze(video, out, FakeEstimator())

    for name in ("annotated.mp4", "landmarks.csv", "com.csv", "summary.json"):
        assert (out / name).exists(), name

    assert summary["frames"] == 20
    assert summary["frames_with_pose"] == 16
    assert json.loads((out / "summary.json").read_text())["frames"] == 20

    with (out / "com.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 20
    assert sum(int(r["detected"]) for r in rows) == 16

    with (out / "landmarks.csv").open() as f:
        assert sum(1 for _ in f) == 1 + 16 * N_LANDMARKS

    # COM of a constant skeleton is constant
    assert summary["com_px"]["std"] == [0.0, 0.0]
