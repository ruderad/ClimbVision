"""A whole clip's landmarks as arrays, plus loading from landmarks.csv.

Analysis (angles, exercises) runs on a `Track`, so it can be re-run from saved
CSVs without running pose estimation again.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from climbvision.skeleton import LM, N_LANDMARKS, Frame


@dataclass
class Track:
    time_s: np.ndarray  # (n,)
    image: np.ndarray  # (n, 33, 2) pixels, NaN where no pose
    world: np.ndarray  # (n, 33, 3) metres, hip-centred, NaN where no pose
    visibility: np.ndarray  # (n, 33), 0 where no pose

    @property
    def n(self) -> int:
        return len(self.time_s)

    @property
    def detected(self) -> np.ndarray:
        return np.isfinite(self.image[:, 0, 0])

    @property
    def fps(self) -> float:
        dt = np.diff(self.time_s)
        return float(1.0 / np.median(dt)) if len(dt) else 30.0

    @classmethod
    def from_frames(cls, frames: list[Frame]) -> Track:
        n = len(frames)
        t = np.array([f.time_s for f in frames], dtype=float)
        img = np.full((n, N_LANDMARKS, 2), np.nan)
        wld = np.full((n, N_LANDMARKS, 3), np.nan)
        vis = np.zeros((n, N_LANDMARKS))
        for i, f in enumerate(frames):
            if f.skeleton is not None:
                img[i], wld[i], vis[i] = f.skeleton.image, f.skeleton.world, f.skeleton.visibility
        return cls(t, img, wld, vis)

    @classmethod
    def from_csv(cls, landmarks_csv: str | Path, com_csv: str | Path | None = None) -> Track:
        """Load landmarks.csv. com.csv (if given) supplies frames with no pose."""
        landmarks_csv = Path(landmarks_csv)
        if com_csv is None and (landmarks_csv.parent / "com.csv").exists():
            com_csv = landmarks_csv.parent / "com.csv"
        times: dict[int, float] = {}
        if com_csv:
            with open(com_csv, newline="") as f:
                for r in csv.DictReader(f):
                    times[int(r["frame"])] = float(r["time_s"])
        rows: dict[int, dict[str, list[float]]] = {}
        with landmarks_csv.open(newline="") as f:
            for r in csv.DictReader(f):
                i = int(r["frame"])
                times.setdefault(i, float(r["time_s"]))
                rows.setdefault(i, {})[r["landmark"]] = [
                    float(r[k]) for k in ("x_px", "y_px", "x_m", "y_m", "z_m", "visibility")
                ]
        idx = sorted(times)
        pos = {f: k for k, f in enumerate(idx)}
        n = len(idx)
        img = np.full((n, N_LANDMARKS, 2), np.nan)
        wld = np.full((n, N_LANDMARKS, 3), np.nan)
        vis = np.zeros((n, N_LANDMARKS))
        for fi, lms in rows.items():
            k = pos[fi]
            for lm in LM:
                v = lms.get(lm.name.lower())
                if v is not None:
                    img[k, lm] = v[0:2]
                    wld[k, lm] = v[2:5]
                    vis[k, lm] = v[5]
        return cls(np.array([times[f] for f in idx]), img, wld, vis)


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average that ignores NaNs (NaN stays NaN where there's no data)."""
    x = np.asarray(x, dtype=float)
    if window <= 1:
        return x.copy()
    ok = np.isfinite(x)
    k = np.ones(window)
    num = np.convolve(np.where(ok, x, 0.0), k, mode="same")
    den = np.convolve(ok.astype(float), k, mode="same")
    out = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    out[~ok] = np.nan
    return out
