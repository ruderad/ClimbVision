"""Pose backends.

A backend module provides:
  NAME      id stored in summary.json ("backend")
  POINTS    its own point names, in its output order (= landmark names in landmarks.csv)
  MAPPING   canonical joint -> tuple of its point names, averaged (visibility = lowest)
  DERIVED   (optional) joint -> (from, to, f): joints the backend lacks, extrapolated as
            to + f * (to - from), e.g. HAND from the forearm direction
  WORLD     (optional) False when the backend gives no 3D estimate (2D-only)
  Estimator class with  estimate(frame_bgr, time_ms) -> RawPose | None  and close()

`to_canonical` turns raw arrays of any backend into canonical arrays. Adding a
backend = one module + one line in BACKENDS.
"""

from __future__ import annotations

import warnings
from types import ModuleType

import numpy as np

from climbvision.engine.backends import mediapipe, rtmpose_halpe26, rtmw_wholebody
from climbvision.engine.skeleton import N_JOINTS, J

BACKENDS: dict[str, ModuleType] = {m.NAME: m for m in (mediapipe, rtmpose_halpe26,
                                                        rtmw_wholebody)}
DEFAULT = mediapipe.NAME


WALL_BACKEND = "rtmw_wholebody133"  # best hands/feet on the wall clips (E5)


def available(name: str) -> bool:
    """Whether a backend can run here (its model files are present)."""
    if name == mediapipe.NAME:
        return True
    from climbvision.engine.backends._topdown import DETECTOR, find_onnx

    try:
        find_onnx(DETECTOR)
        find_onnx(next(iter(get(name).MODELS.values()))[0])
        return True
    except (FileNotFoundError, StopIteration):
        return False


def choose(scene: str) -> str:
    """Default backend for a scene: on the wall the whole-body model (best hands and
    feet, 2D only); elsewhere MediaPipe (its 3D estimate is used for front-view angles
    and the metric scale). Falls back to MediaPipe when the model files are missing."""
    if scene == "wall" and available(WALL_BACKEND):
        return WALL_BACKEND
    return DEFAULT


def get(name: str) -> ModuleType:
    try:
        return BACKENDS[name]
    except KeyError:
        raise ValueError(f"unknown pose backend {name!r}; have {sorted(BACKENDS)}") from None


def to_canonical(backend: str, image: np.ndarray, world: np.ndarray,
                 visibility: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map raw (..., k, D) arrays of `backend` to canonical (..., N_JOINTS, D) arrays."""
    bk = get(backend)
    index = {n: i for i, n in enumerate(bk.POINTS)}
    lead = image.shape[:-2]
    img = np.full((*lead, N_JOINTS, 2), np.nan)
    wld = np.full((*lead, N_JOINTS, 3), np.nan)
    vis = np.zeros((*lead, N_JOINTS))
    with warnings.catch_warnings():  # frames without a pose are all-NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        for joint, names in bk.MAPPING.items():
            idx = [index[n] for n in names]
            img[..., joint, :] = image[..., idx, :].mean(axis=-2)
            wld[..., joint, :] = world[..., idx, :].mean(axis=-2)
            vis[..., joint] = visibility[..., idx].min(axis=-1)
        for joint, (a, b, f) in getattr(bk, "DERIVED", {}).items():
            img[..., joint, :] = img[..., b, :] + f * (img[..., b, :] - img[..., a, :])
            wld[..., joint, :] = wld[..., b, :] + f * (wld[..., b, :] - wld[..., a, :])
            vis[..., joint] = np.minimum(vis[..., a], vis[..., b])
    return img, wld, vis


__all__ = ["BACKENDS", "DEFAULT", "J", "get", "to_canonical"]
