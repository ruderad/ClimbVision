"""Pose backends.

A backend module provides:
  NAME      id stored in summary.json ("backend")
  POINTS    its own point names, in its output order (= landmark names in landmarks.csv)
  MAPPING   canonical joint -> tuple of its point names, averaged (visibility = lowest)
  Estimator class with  estimate(frame_bgr, time_ms) -> RawPose | None  and close()

`to_canonical` turns raw arrays of any backend into canonical arrays. Adding a
backend = one module + one line in BACKENDS.
"""

from __future__ import annotations

import warnings
from types import ModuleType

import numpy as np

from climbvision.engine.backends import mediapipe
from climbvision.engine.skeleton import N_JOINTS, J

BACKENDS: dict[str, ModuleType] = {mediapipe.NAME: mediapipe}
DEFAULT = mediapipe.NAME


def get(name: str) -> ModuleType:
    try:
        return BACKENDS[name]
    except KeyError:
        raise ValueError(f"unknown pose backend {name!r}; have {sorted(BACKENDS)}") from None


def to_canonical(backend: str, image: np.ndarray, world: np.ndarray,
                 visibility: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map raw (..., k, D) arrays of `backend` to canonical (..., N_JOINTS, D) arrays."""
    b = get(backend)
    index = {n: i for i, n in enumerate(b.POINTS)}
    lead = image.shape[:-2]
    img = np.full((*lead, N_JOINTS, 2), np.nan)
    wld = np.full((*lead, N_JOINTS, 3), np.nan)
    vis = np.zeros((*lead, N_JOINTS))
    with warnings.catch_warnings():  # frames without a pose are all-NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        for joint, names in b.MAPPING.items():
            idx = [index[n] for n in names]
            img[..., joint, :] = image[..., idx, :].mean(axis=-2)
            wld[..., joint, :] = world[..., idx, :].mean(axis=-2)
            vis[..., joint] = visibility[..., idx].min(axis=-1)
    return img, wld, vis


__all__ = ["BACKENDS", "DEFAULT", "J", "get", "to_canonical"]
