"""MediaPipe Pose Landmarker (BlazePose, 33 points) backend."""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Literal

import numpy as np

from climbvision.engine.skeleton import J, RawPose

NAME = "mediapipe_pose33"

POINTS = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye",
    "right_eye_outer", "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
    "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb",
    "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle",
    "right_ankle", "left_heel", "right_heel", "left_foot_index", "right_foot_index",
]

# canonical joint <- MediaPipe point(s), averaged. Face points are deliberately unused
# (they are guessed with full confidence when the face is hidden); fingers are reduced
# to one hand point.
MAPPING: dict[J, tuple[str, ...]] = {
    J.HEAD: ("left_ear", "right_ear"),
    J.L_SHOULDER: ("left_shoulder",), J.R_SHOULDER: ("right_shoulder",),
    J.L_ELBOW: ("left_elbow",), J.R_ELBOW: ("right_elbow",),
    J.L_WRIST: ("left_wrist",), J.R_WRIST: ("right_wrist",),
    J.L_HAND: ("left_index", "left_pinky"), J.R_HAND: ("right_index", "right_pinky"),
    J.L_HIP: ("left_hip",), J.R_HIP: ("right_hip",),
    J.L_KNEE: ("left_knee",), J.R_KNEE: ("right_knee",),
    J.L_ANKLE: ("left_ankle",), J.R_ANKLE: ("right_ankle",),
    J.L_HEEL: ("left_heel",), J.R_HEEL: ("right_heel",),
    J.L_TOE: ("left_foot_index",), J.R_TOE: ("right_foot_index",),
}

ModelSize = Literal["lite", "full", "heavy"]
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{size}/float16/latest/pose_landmarker_{size}.task")
MODEL_DIR = Path.home() / ".cache" / "climbvision" / "models"


def ensure_model(size: ModelSize = "full", model_dir: Path = MODEL_DIR) -> Path:
    """Download the MediaPipe pose model on first use and return its path."""
    path = model_dir / f"pose_landmarker_{size}.task"
    if not path.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading pose model ({size}) to {path} ...")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL.format(size=size), tmp)
        tmp.rename(path)
    return path


class Estimator:
    """VIDEO running mode: tracks between frames; timestamps must increase."""

    backend = NAME

    def __init__(self, model_size: ModelSize = "full", model_path: str | Path | None = None,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        path = Path(model_path) if model_path else ensure_model(model_size)
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def estimate(self, frame_bgr: np.ndarray, time_ms: int) -> RawPose | None:
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
            world = np.full((len(POINTS), 3), np.nan)
        return RawPose(img, world, vis)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> Estimator:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
