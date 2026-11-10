"""Video reading and writing (thin wrappers around OpenCV)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class VideoInfo:
    fps: float
    width: int
    height: int
    frame_count: int  # as reported by the container; may be approximate


class VideoReader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise OSError(f"Could not open video: {self.path}")
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.info = VideoInfo(
            fps=fps if fps > 0 else 30.0,
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def frames(self, start: int = 0, step: int = 1) -> Iterator[tuple[int, float, np.ndarray]]:
        """Yield (index, time in seconds, BGR frame), from frame `start`, every `step`-th.

        Times come from the container's presentation timestamps, so phone videos
        with a variable frame rate get real times. Falls back to index / fps when
        timestamps are missing or don't increase. Skipped frames are only grabbed, not
        decoded into images (cheaper than reading them).
        """
        i = 0
        last_t = -1.0
        while True:
            want = i >= start and (i - start) % step == 0
            if want:
                ok, frame = self._cap.read()
            else:
                ok, frame = self._cap.grab(), None
            if not ok:
                break
            t = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if not (t > last_t):
                t = max(i / self.info.fps, last_t + 1e-3)
            last_t = t
            if want:
                yield i, t, frame
            i += 1

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class VideoWriter:
    def __init__(self, path: str | Path, fps: float, size: tuple[int, int]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._w = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not self._w.isOpened():
            raise OSError(f"Could not open video writer: {self.path}")

    def write(self, frame: np.ndarray) -> None:
        self._w.write(frame)

    def close(self) -> None:
        self._w.release()

    def __enter__(self) -> VideoWriter:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
