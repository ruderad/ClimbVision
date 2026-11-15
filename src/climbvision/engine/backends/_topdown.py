"""Shared top-down estimator for the RTMPose family (ONNX, via rtmlib).

Top-down = find the person (box), then estimate the joints inside the box. To follow
*one* climber and save time, the box comes from the previous frame's joints; the person
detector (YOLOX) runs on the first frame, every `redetect_every` frames, and whenever the
pose looks lost, and the detected box overlapping the tracked person most is kept.

Accuracy (1.5):

- **Steady crop.** The pose model sees the person through the box, so a box that jumps
  from frame to frame makes the joints jump with it. The box is smoothed: its centre
  follows the body quickly, its size grows at once (a reach must stay in the crop) and
  shrinks slowly.
- **Flip averaging** (`flip=True`, ~2x time): the model also looks at the mirrored
  frame; the two estimates (left and right swapped back) are averaged, weighted by
  their scores. It halves the random error of a single look and often settles a hidden
  limb the two looks disagree on.
- **Confidence.** RTMPose scores aren't probabilities (visible joints score well above
  1); they used to be clipped to 1, so every joint looked certain. The score is now
  mapped to 0-1 between `SCORE_LO` and `SCORE_REF` (the score of a clearly visible
  joint), so a hidden or guessed joint (score 1-2.5: a quarter of those are off by
  > 30 px) has a low visibility, is drawn grey, and the smoothing layer trusts it less.

Model files: OpenMMLab's ONNX zips, put (zipped or unzipped) in a `models/` folder (the
project's, or $CLIMBVISION_MODELS); they are found by name.
"""

from __future__ import annotations

import os
import time
import zipfile
from pathlib import Path

import numpy as np

from climbvision.engine.skeleton import RawPose

DETECTOR = "yolox_m_8xb8-300e_humanart"
# RTMPose / RTMW SimCC scores -> visibility 0-1: (score - LO) / (REF - LO). Measured on
# coordination_failed (RTMW-x, 8 500 limb joints): score >= 4 -> 0.2 % of joints off by
# > 30 px from their 5-frame median; 3-4 -> 1.6 %; 2-3 -> 11 %; < 2 -> 26 %.
SCORE_LO, SCORE_REF = 1.0, 4.0


def flip_permutation(points: list[str]) -> np.ndarray:
    """Index of each point's mirror image (left <-> right by name; others stay)."""
    idx = {n: i for i, n in enumerate(points)}

    def mirror(n: str) -> str:
        return n.replace("left", "\0").replace("right", "left").replace("\0", "right")

    return np.array([idx.get(mirror(n), i) for i, n in enumerate(points)])


def steady_box(prev, box: np.ndarray, follow: float = 0.5, grow: float = 0.8,
               shrink: float = 0.15):
    """Smoothed box (centre, size) after a new box: the centre moves `follow` of the way,
    the size `grow` of the way when it gets bigger and `shrink` when it gets smaller."""
    c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    sz = np.array([box[2] - box[0], box[3] - box[1]])
    if prev is None:
        return c, sz
    pc, ps = prev
    k = np.where(sz > ps, grow, shrink)
    return pc + follow * (c - pc), ps + k * (sz - ps)


def models_dir() -> Path:
    env = os.environ.get("CLIMBVISION_MODELS")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:  # the repo's models/ folder
        if (parent / "pyproject.toml").exists():
            return parent / "models"
    return Path.home() / ".cache" / "climbvision" / "models"


def find_onnx(key: str, folder: Path | None = None) -> Path:
    """The ONNX file of a model whose zip/folder name contains `key` (unzips if needed)."""
    folder = folder or models_dir()
    if not folder.exists():
        raise FileNotFoundError(f"model folder {folder} not found")
    for z in sorted(folder.rglob("*.zip")):
        if key in z.name and not (z.with_suffix("")).exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(z.with_suffix(""))
    hits = [p for p in folder.rglob("*.onnx") if key in str(p.relative_to(folder))]
    if not hits:
        raise FileNotFoundError(f"no ONNX model matching {key!r} in {folder} (download the "
                                "OpenMMLab zip into it; see docs/GUIDE.md, pose backends)")
    return sorted(hits, key=lambda p: len(str(p)))[0]


def _iou(a, b) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


class TopDownEstimator:
    backend = ""  # set by subclasses

    def __init__(self, pose_key: str, input_size: tuple[int, int], n_points: int,
                 redetect_every: int = 15, min_score: float = 0.3,
                 models: str | Path | None = None, threads: int | None = None,
                 points: list[str] | None = None, flip: bool = True,
                 score_ref: float = SCORE_REF) -> None:
        from rtmlib import YOLOX, RTMPose

        folder = Path(models) if models else None
        self.det = YOLOX(str(find_onnx(DETECTOR, folder)), model_input_size=(640, 640),
                         backend="onnxruntime", device="cpu")
        self.pose = RTMPose(str(find_onnx(pose_key, folder)), model_input_size=input_size,
                            backend="onnxruntime", device="cpu")
        self.n_points = n_points
        self.flip = flip and points is not None
        self.score_ref = score_ref
        self.perm = flip_permutation(points) if points is not None else None
        self.sbox = None  # smoothed box: (centre, size)
        self.redetect_every = redetect_every
        self.min_score = min_score
        self.box = None
        self.since_detect = 0
        self.timing = {"detect_s": 0.0, "pose_s": 0.0, "frames": 0, "detections": 0}

    def _detect(self, frame) -> np.ndarray | None:
        t0 = time.perf_counter()
        boxes = np.asarray(self.det(frame)).reshape(-1, 4)
        self.timing["detect_s"] += time.perf_counter() - t0
        self.timing["detections"] += 1
        self.since_detect = 0
        if not len(boxes):
            return None
        if self.box is None:  # first lock: the largest person
            return boxes[np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))]
        ious = [_iou(self.box, b) for b in boxes]
        k = int(np.argmax(ious))
        return boxes[k] if ious[k] > 0.1 else None

    def estimate(self, frame_bgr: np.ndarray, time_ms: int) -> RawPose | None:
        self.timing["frames"] += 1
        box = self.box
        if box is None or self.since_detect >= self.redetect_every:
            det = self._detect(frame_bgr)
            box = det if det is not None else box
        if box is None:
            return None
        self.sbox = steady_box(self.sbox, np.asarray(box, float))
        c, sz = self.sbox
        h, w = frame_bgr.shape[:2]
        use = [max(0.0, c[0] - sz[0] / 2), max(0.0, c[1] - sz[1] / 2),
               min(w, c[0] + sz[0] / 2), min(h, c[1] + sz[1] / 2)]
        t0 = time.perf_counter()
        kpts, scores = self.pose(frame_bgr, bboxes=[use])
        kpts, scores = np.asarray(kpts[0], float), np.asarray(scores[0], float)
        if self.flip:
            fk, fs = self.pose(np.ascontiguousarray(frame_bgr[:, ::-1]),
                               bboxes=[[w - use[2], use[1], w - use[0], use[3]]])
            fk, fs = np.asarray(fk[0], float)[self.perm], np.asarray(fs[0], float)[self.perm]
            fk[:, 0] = w - 1 - fk[:, 0]
            a, b = np.clip(scores, 1e-3, None)[:, None], np.clip(fs, 1e-3, None)[:, None]
            kpts = (a * kpts + b * fk) / (a + b)
            scores = (scores + fs) / 2
        self.timing["pose_s"] += time.perf_counter() - t0
        self.since_detect += 1
        good = scores > self.min_score
        if good.sum() < 0.3 * len(scores):  # lost: look for the person again next frame
            self.box = None
            self.sbox = None
            return None
        # next box: around the confident joints, with a margin
        x0, y0 = kpts[good].min(axis=0)
        x1, y1 = kpts[good].max(axis=0)
        m = 0.12 * max(x1 - x0, y1 - y0)
        self.box = np.array([max(0, x0 - m), max(0, y0 - m), min(w, x1 + m), min(h, y1 + m)])
        return RawPose(kpts.astype(float), np.full((len(kpts), 3), np.nan),
                       np.clip((scores - SCORE_LO) / (self.score_ref - SCORE_LO), 0.0, 1.0))

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()
