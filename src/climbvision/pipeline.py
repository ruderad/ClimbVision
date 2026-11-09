"""Video -> pose -> COM -> files (the slow pass; everything else reads its CSVs)."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from climbvision.com import Sex, com_confidence, whole_body_com
from climbvision.pose import PoseEstimator
from climbvision.skeleton import LM, Frame
from climbvision.video import VideoReader

NAN2 = np.full(2, np.nan)
NAN3 = np.full(3, np.nan)


@dataclass
class FrameResult:
    frame: Frame
    com_px: np.ndarray = field(default_factory=lambda: NAN2.copy())
    com_world: np.ndarray = field(default_factory=lambda: NAN3.copy())
    com_conf: float = 0.0


def process_frame(frame: Frame, sex: Sex = "male") -> FrameResult:
    res = FrameResult(frame)
    sk = frame.skeleton
    if sk is None:
        return res
    res.com_px = whole_body_com(sk.image, sex)
    if np.all(np.isfinite(sk.world)):
        res.com_world = whole_body_com(sk.world, sex)
    res.com_conf = com_confidence(sk.visibility, sex)
    return res


def _write_landmarks(path: Path, results: list[FrameResult]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "landmark", "x_px", "y_px",
                    "x_m", "y_m", "z_m", "visibility"])
        for r in results:
            sk = r.frame.skeleton
            if sk is None:
                continue
            for lm in LM:
                x, y = sk.image[lm]
                wx, wy, wz = sk.world[lm]
                w.writerow([r.frame.index, f"{r.frame.time_s:.4f}", lm.name.lower(),
                            f"{x:.2f}", f"{y:.2f}", f"{wx:.4f}", f"{wy:.4f}", f"{wz:.4f}",
                            f"{sk.visibility[lm]:.3f}"])


def _write_com(path: Path, results: list[FrameResult]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "detected", "com_x_px", "com_y_px",
                    "com_x_m", "com_y_m", "com_z_m", "confidence"])
        for r in results:
            w.writerow([r.frame.index, f"{r.frame.time_s:.4f}",
                        int(r.frame.skeleton is not None),
                        *(f"{v:.2f}" for v in r.com_px),
                        *(f"{v:.4f}" for v in r.com_world),
                        f"{r.com_conf:.3f}"])


def summarize(results: list[FrameResult], fps: float) -> dict:
    n = len(results)
    detected = [r for r in results if r.frame.skeleton is not None]
    out: dict = {
        "frames": n,
        "fps": fps,
        "duration_s": round(results[-1].frame.time_s, 3) if results else 0.0,
        "frames_with_pose": len(detected),
        "detection_rate": len(detected) / n if n else 0.0,
    }
    if detected:
        px = np.array([r.com_px for r in detected])
        conf = np.array([r.com_conf for r in detected])
        out["com_px"] = {
            "mean": px.mean(axis=0).round(1).tolist(),
            "std": px.std(axis=0).round(2).tolist(),
            "range": (px.max(axis=0) - px.min(axis=0)).round(1).tolist(),
        }
        out["mean_confidence"] = round(float(conf.mean()), 3)
        wd = np.array([r.com_world for r in detected])
        wd = wd[np.all(np.isfinite(wd), axis=1)]
        if len(wd):
            out["com_world_m"] = {
                "mean": wd.mean(axis=0).round(4).tolist(),
                "std": wd.std(axis=0).round(4).tolist(),
            }
    return out


def analyze(
    video_path: str | Path,
    out_dir: str | Path,
    estimator: PoseEstimator,
    sex: Sex = "male",
    max_frames: int | None = None,
) -> dict:
    """Pose + COM pass over one video: writes landmarks.csv, com.csv, summary.json.

    Drawing happens later in `render`, once metrics (contacts, support) are known.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[FrameResult] = []

    with VideoReader(video_path) as reader:
        info = reader.info
        for idx, t, img in reader.frames():
            if max_frames is not None and idx >= max_frames:
                break
            sk = estimator.estimate(img, int(round(t * 1000)))
            results.append(process_frame(Frame(idx, t, sk), sex))

    _write_landmarks(out_dir / "landmarks.csv", results)
    _write_com(out_dir / "com.csv", results)
    summary = {"video": str(video_path), "sex_model": sex,
               "frame_size": [info.width, info.height], **summarize(results, info.fps)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
