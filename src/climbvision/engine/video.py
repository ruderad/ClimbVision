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


def fingerprint(path: str | Path) -> dict:
    """What identifies a video file: size on disk, frame count, fps, frame size.
    Stored in summary.json when the pose is computed."""
    path = Path(path)
    out: dict = {"size_bytes": path.stat().st_size}
    cap = cv2.VideoCapture(str(path))
    try:
        out.update(frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                   fps=round(float(cap.get(cv2.CAP_PROP_FPS)), 3),
                   width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                   height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        cap.release()
    return out


def video_changed(summary: dict, video: str | Path) -> str | None:
    """Why the results in a folder don't belong to this video file (None if they do).

    Results from a replaced or edited video can't be drawn on it: the skeleton no longer
    lines up. With a fingerprint (results from v1.0 on) the file size must match; older
    results are checked on fps and frame count (the container's count can be a few
    frames off, so small differences are allowed)."""
    video = Path(video)
    if not video.exists():
        return None
    now = fingerprint(video)
    fp = summary.get("video_fingerprint")
    if fp:
        if fp.get("size_bytes") != now["size_bytes"]:
            return (f"{video.name} has changed since its pose was computed "
                    f"({fp.get('size_bytes')} -> {now['size_bytes']} bytes)")
        return None
    fps, frames = summary.get("fps"), summary.get("frames")
    step = int(summary.get("frame_step", 1))
    if fps and now["fps"] and abs(fps - now["fps"]) > 0.01 * now["fps"]:
        return f"{video.name} is {now['fps']:.2f} fps, its results were made at {fps:.2f} fps"
    if frames and now["frame_count"] > 0:
        expect = now["frame_count"] / step
        if frames > expect + 5 or frames < 0.9 * expect:
            return (f"{video.name} has {now['frame_count']} frames, its results "
                    f"{frames * step}")
    return None


def warn_if_changed(result_dir: str | Path, video: str | Path) -> bool:
    """Print a warning when a results folder doesn't belong to the video any more.
    Returns True if it doesn't."""
    import json
    import sys

    p = Path(result_dir) / "summary.json"
    if not p.exists():
        return False
    why = video_changed(json.loads(p.read_text()), video)
    if why:
        print(f"[climbvision] WARNING {Path(result_dir).name}: {why}. The video seems to "
              f"have been replaced: re-run `climbvision analyze {video}`.", file=sys.stderr)
    return bool(why)
