"""A clip's canonical skeleton as arrays (frames first), loaded from the pose layer.

The pose layer (`landmarks.csv`) stores each backend's raw points; loading maps them
onto the canonical joints (engine/skeleton.py), so everything downstream is
backend-independent and can be re-run from saved files in seconds.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from climbvision.engine import backends
from climbvision.engine.skeleton import N_JOINTS


@dataclass
class Track:
    time_s: np.ndarray  # (n,)
    image: np.ndarray  # (n, N_JOINTS, 2) pixels, NaN where no pose
    world: np.ndarray  # (n, N_JOINTS, 3) metres, hip-centred, NaN where unknown
    visibility: np.ndarray  # (n, N_JOINTS), 0 where no pose

    @property
    def n(self) -> int:
        return len(self.time_s)

    @property
    def detected(self) -> np.ndarray:
        """Frames with a pose (judged on the shoulders, which every backend has)."""
        from climbvision.engine.skeleton import J

        return np.isfinite(self.image[:, J.L_SHOULDER, 0])

    @property
    def fps(self) -> float:
        dt = np.diff(self.time_s)
        return float(1.0 / np.median(dt)) if len(dt) else 30.0

    @classmethod
    def from_raw(cls, time_s: np.ndarray, image: np.ndarray, world: np.ndarray,
                 visibility: np.ndarray, backend: str = backends.DEFAULT) -> Track:
        """From a backend's raw arrays (n, k, ...), mapped to canonical joints."""
        img, wld, vis = backends.to_canonical(backend, np.asarray(image, float),
                                              np.asarray(world, float),
                                              np.asarray(visibility, float))
        return cls(np.asarray(time_s, float), img, wld, vis)

    @classmethod
    def from_csv(cls, landmarks_csv: str | Path, com_csv: str | Path | None = None,
                 backend: str | None = None) -> Track:
        """Load the pose layer. com.csv (if present) supplies frames with no pose;
        the backend is read from summary.json next to it (default: MediaPipe)."""
        landmarks_csv = Path(landmarks_csv)
        folder = landmarks_csv.parent
        if com_csv is None and (folder / "com.csv").exists():
            com_csv = folder / "com.csv"
        if backend is None:
            backend = backends.DEFAULT
            if (folder / "summary.json").exists():
                backend = json.loads((folder / "summary.json").read_text()).get(
                    "backend", backends.DEFAULT)
        points = backends.get(backend).POINTS
        pindex = {p: k for k, p in enumerate(points)}
        times: dict[int, float] = {}
        if com_csv:
            with open(com_csv, newline="") as f:
                for r in csv.DictReader(f):
                    times[int(r["frame"])] = float(r["time_s"])
        rows: dict[int, list[tuple[int, list[float]]]] = {}
        with landmarks_csv.open(newline="") as f:
            for r in csv.DictReader(f):
                i = int(r["frame"])
                times.setdefault(i, float(r["time_s"]))
                k = pindex.get(r["landmark"])
                if k is not None:
                    rows.setdefault(i, []).append((k, [
                        float(r[c]) for c in ("x_px", "y_px", "x_m", "y_m", "z_m", "visibility")]))
        idx = sorted(times)
        pos = {f: k for k, f in enumerate(idx)}
        n, kk = len(idx), len(points)
        img = np.full((n, kk, 2), np.nan)
        wld = np.full((n, kk, 3), np.nan)
        vis = np.zeros((n, kk))
        for fi, pts in rows.items():
            r = pos[fi]
            for k, v in pts:
                img[r, k], wld[r, k], vis[r, k] = v[0:2], v[2:5], v[5]
        return cls.from_raw(np.array([times[f] for f in idx]), img, wld, vis, backend)


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


def window(fps: float, seconds: float) -> int:
    """Smoothing window in frames for a duration (at least 1)."""
    return max(1, int(fps * seconds + 1e-6))


__all__ = ["N_JOINTS", "Track", "smooth", "window"]
