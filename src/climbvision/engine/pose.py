"""Pose layer: video -> a backend's raw points per frame (the slow pass).

Writes the backend's *own* points to landmarks.csv (so nothing is lost if the
canonical mapping changes later), the canonical COM to com.csv, and summary.json
(video, frame size, fps, backend). Everything downstream re-reads these files.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Protocol

import numpy as np

from climbvision.engine import backends
from climbvision.engine.biomech import Sex, com_confidence, whole_body_com
from climbvision.engine.skeleton import RawPose
from climbvision.engine.video import VideoReader


class PoseEstimator(Protocol):
    backend: str  # name in engine.backends.BACKENDS

    def estimate(self, frame_bgr: np.ndarray, time_ms: int) -> RawPose | None:
        """Raw pose for one frame. `time_ms` must increase from call to call."""

    def close(self) -> None: ...


def make_estimator(backend: str = backends.DEFAULT, **kwargs) -> PoseEstimator:
    return backends.get(backend).Estimator(**kwargs)


def run_pose(video_path: str | Path, out_dir: str | Path, estimator: PoseEstimator,
             sex: Sex = "male", max_frames: int | None = None) -> dict:
    """Pose + COM pass over one video: writes landmarks.csv, com.csv, summary.json.
    Drawing happens later (`render`), once contacts and support are known."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = getattr(estimator, "backend", backends.DEFAULT)
    points = backends.get(name).POINTS
    frames: list[tuple[int, float, RawPose | None]] = []
    with VideoReader(video_path) as reader:
        info = reader.info
        for idx, t, img in reader.frames():
            if max_frames is not None and idx >= max_frames:
                break
            frames.append((idx, t, estimator.estimate(img, int(round(t * 1000)))))

    k = len(points)
    n = len(frames)
    raw_img = np.full((n, k, 2), np.nan)
    raw_wld = np.full((n, k, 3), np.nan)
    raw_vis = np.zeros((n, k))
    for r, (_, _, p) in enumerate(frames):
        if p is not None:
            raw_img[r], raw_wld[r], raw_vis[r] = p.image, p.world, p.visibility
    img, wld, vis = backends.to_canonical(name, raw_img, raw_wld, raw_vis)
    detected = np.array([p is not None for _, _, p in frames], bool)
    com_px = whole_body_com(img, sex)
    com_w = whole_body_com(wld, sex)
    conf = np.where(detected, com_confidence(vis, sex), 0.0)

    with (out_dir / "landmarks.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "landmark", "x_px", "y_px", "x_m", "y_m", "z_m",
                    "visibility"])
        for r, (idx, t, p) in enumerate(frames):
            if p is None:
                continue
            for j, pname in enumerate(points):
                x, y = raw_img[r, j]
                wx, wy, wz = raw_wld[r, j]
                w.writerow([idx, f"{t:.4f}", pname, f"{x:.2f}", f"{y:.2f}", f"{wx:.4f}",
                            f"{wy:.4f}", f"{wz:.4f}", f"{raw_vis[r, j]:.3f}"])
    with (out_dir / "com.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "detected", "com_x_px", "com_y_px", "com_x_m",
                    "com_y_m", "com_z_m", "confidence"])
        for r, (idx, t, _) in enumerate(frames):
            w.writerow([idx, f"{t:.4f}", int(detected[r]), *(f"{v:.2f}" for v in com_px[r]),
                        *(f"{v:.4f}" for v in com_w[r]), f"{conf[r]:.3f}"])

    summary: dict = {"video": str(video_path), "backend": name, "sex_model": sex,
                     "frame_size": [info.width, info.height], "frames": n, "fps": info.fps,
                     "duration_s": round(frames[-1][1], 3) if frames else 0.0,
                     "frames_with_pose": int(detected.sum()),
                     "detection_rate": float(detected.mean()) if n else 0.0}
    if detected.any():
        px = com_px[detected]
        summary["com_px"] = {"mean": px.mean(axis=0).round(1).tolist(),
                             "std": px.std(axis=0).round(2).tolist(),
                             "range": (px.max(axis=0) - px.min(axis=0)).round(1).tolist()}
        summary["mean_confidence"] = round(float(conf[detected].mean()), 3)
        wd = com_w[detected]
        wd = wd[np.all(np.isfinite(wd), axis=1)]
        if len(wd):
            summary["com_world_m"] = {"mean": wd.mean(axis=0).round(4).tolist(),
                                      "std": wd.std(axis=0).round(4).tolist()}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
