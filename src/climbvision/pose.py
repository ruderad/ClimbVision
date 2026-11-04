"""Pose estimation backends. Each returns a backend-independent `Skeleton`."""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from climbvision.skeleton import N_LANDMARKS, Skeleton

ModelSize = Literal["lite", "full", "heavy"]

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{size}/float16/latest/pose_landmarker_{size}.task"
)
MODEL_DIR = Path.home() / ".cache" / "climbvision" / "models"


class PoseEstimator(Protocol):
    def estimate(self, frame_bgr: np.ndarray, time_ms: int) -> Skeleton | None:
        """Pose for one frame. `time_ms` must increase from call to call."""

    def close(self) -> None: ...


def ensure_model(size: ModelSize = "full", model_dir: Path = MODEL_DIR) -> Path:
    """Download the MediaPipe pose model on first use and return its path."""
    path = model_dir / f"pose_landmarker_{size}.task"
    if not path.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        url = MODEL_URL.format(size=size)
        print(f"Downloading pose model ({size}) to {path} ...")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


class MediaPipePose:
    """MediaPipe Pose Landmarker in VIDEO mode (uses temporal tracking between frames)."""

    def __init__(
        self,
        model_size: ModelSize = "full",
        model_path: str | Path | None = None,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        path = Path(model_path) if model_path else ensure_model(model_size)
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def estimate(self, frame_bgr: np.ndarray, time_ms: int) -> Skeleton | None:
        h, w = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, int(time_ms))
        if not result.pose_landmarks:
            return None

        lms = result.pose_landmarks[0]
        img = np.array([[lm.x * w, lm.y * h] for lm in lms])
        vis = np.array([lm.visibility if lm.visibility is not None else 0.0 for lm in lms])
        if result.pose_world_landmarks:
            world = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_world_landmarks[0]])
        else:
            world = np.full((N_LANDMARKS, 3), np.nan)
        return Skeleton(image=img, world=world, visibility=vis)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> MediaPipePose:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
