"""Holds found in the wall image by a pretrained detector.

The wall layer finds where hands and feet *rested*; on its own that also counts a limb
paused in the air, or pressed flat on the wall, as "on a hold". With the holds found in
the picture of the wall (`wall.jpg`, the climber removed), a rest counts as a hold only
when it is on a detected hold; otherwise it is a **smear** (engine/wall.py).

Model: any single- or multi-class YOLOv8-style ONNX detector of climbing holds in
`models/` (file name containing "hold" or "climb"), e.g.
`yolov8n-freeclimbs-detect-2-fp32.onnx` (huggingface.co/jwlarocque/yolov8n-freeclimbs-
detect-2, AGPL-3.0). Run with onnxruntime (the `rtm` extra); no PyTorch needed.

The wall image is scanned at several sizes — the model's input size (2560 px for the
freeclimbs model) and smaller ones (1/2, 5/16) — and the boxes merged (non-maximum
suppression): big volumes and close-up walls are found at the small sizes, small
crimps and foot chips at the large one. Results are cached in the results folder
(`holds_detected.json`, boxes in wall coordinates) with a picture of them
(`holds_detected.jpg`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

CACHE = "holds_detected.json"
VERSION = "+ms"  # bump when detection changes, so caches are redone


def find_model(folder: Path | None = None) -> Path | None:
    from climbvision.engine.backends._topdown import models_dir

    folder = Path(folder) if folder else models_dir()
    if not folder.exists():
        return None
    cands = sorted(p for p in folder.rglob("*.onnx")
                   if any(k in p.name.lower() for k in ("hold", "climb")))
    fp32 = [p for p in cands if "fp32" in p.name]
    return (fp32 or cands or [None])[0]


@dataclass
class HoldBoxes:
    boxes: np.ndarray  # (k, 4) x0, y0, x1, y1 in wall coordinates
    scores: np.ndarray  # (k,)
    model: str

    def __len__(self) -> int:
        return len(self.boxes)

    def containing(self, p: np.ndarray, margin: float) -> int:
        """Index of the best box containing point p (grown by margin), or -1. Among
        several, the smallest (the most specific hold)."""
        if not len(self.boxes):
            return -1
        b = self.boxes
        inside = (p[0] >= b[:, 0] - margin) & (p[0] <= b[:, 2] + margin) & \
                 (p[1] >= b[:, 1] - margin) & (p[1] <= b[:, 3] + margin)
        idx = np.flatnonzero(inside)
        if not len(idx):
            return -1
        area = (b[idx, 2] - b[idx, 0]) * (b[idx, 3] - b[idx, 1])
        return int(idx[np.argmin(area)])

    def centre(self, i: int) -> np.ndarray:
        b = self.boxes[i]
        return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"model": self.model,
                                    "boxes": np.round(self.boxes, 1).tolist(),
                                    "scores": np.round(self.scores, 3).tolist()}))

    @classmethod
    def load(cls, path: Path) -> HoldBoxes:
        d = json.loads(path.read_text())
        return cls(np.array(d["boxes"], float).reshape(-1, 4), np.array(d["scores"], float),
                   d.get("model", "?"))


class HoldDetector:
    def __init__(self, model: Path, conf: float = 0.15, iou: float = 0.5):
        import onnxruntime as ort

        self.model = Path(model)
        self.sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.name = inp.name
        shape = inp.shape
        self.size = int(shape[2]) if isinstance(shape[2], int) else 640
        self.fp16 = "float16" in inp.type
        self.conf, self.iou = conf, iou

    def _run(self, tile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One square tile (size x size, BGR) -> boxes (x0, y0, x1, y1), scores."""
        x = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)[None]
        if self.fp16:
            x = x.astype(np.float16)
        out = self.sess.run(None, {self.name: x})[0][0].astype(np.float32)  # (4+nc, N)
        if out.shape[0] > out.shape[1]:
            out = out.T
        cx, cy, w, h = out[0], out[1], out[2], out[3]
        score = out[4:].max(axis=0)
        keep = score >= self.conf
        b = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)[keep]
        return b, score[keep]

    def detect(self, image: np.ndarray, work_side: int | None = None,
               overlap: int = 160) -> tuple[np.ndarray, np.ndarray]:
        """Boxes in `image` pixels. The image is scaled to `work_side` on its long side
        (default: the model's input size, i.e. one tile, like the photos it learned on)
        and scanned in overlapping tiles if larger."""
        H, W = image.shape[:2]
        k = (work_side or self.size) / max(H, W)
        img = cv2.resize(image, (int(round(W * k)), int(round(H * k))),
                         interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
        h, w = img.shape[:2]
        S = self.size
        pad = np.zeros((max(h, S), max(w, S), 3), np.uint8)
        pad[:h, :w] = img
        ys = list(range(0, max(1, pad.shape[0] - S + 1), S - overlap))
        xs = list(range(0, max(1, pad.shape[1] - S + 1), S - overlap))
        if ys[-1] + S < pad.shape[0]:
            ys.append(pad.shape[0] - S)
        if xs[-1] + S < pad.shape[1]:
            xs.append(pad.shape[1] - S)
        boxes, scores = [], []
        for y in ys:
            for x in xs:
                b, s = self._run(pad[y:y + S, x:x + S])
                if len(b):
                    boxes.append(b + [x, y, x, y])
                    scores.append(s)
        if not boxes:
            return np.zeros((0, 4)), np.zeros(0)
        B, Sc = np.vstack(boxes), np.concatenate(scores)
        keep = cv2.dnn.NMSBoxes([[float(a), float(b_), float(c - a), float(d - b_)]
                                 for a, b_, c, d in B], Sc.tolist(), self.conf, self.iou)
        keep = np.array(keep).reshape(-1)
        B = np.clip(B[keep] / k, 0, [W, H, W, H])
        return B, Sc[keep]


    def detect_multi(self, image: np.ndarray, sides: tuple[float, ...] = (1, 0.5, 0.3125),
                     iou: float = 0.4) -> tuple[np.ndarray, np.ndarray]:
        """`detect` at several sizes (fractions of the model's input size), merged."""
        bs, ss = [], []
        for f in sides:
            b, s = self.detect(image, work_side=int(self.size * f))
            bs.append(b)
            ss.append(s)
        B, S = np.vstack(bs), np.concatenate(ss)
        if not len(B):
            return B, S
        keep = cv2.dnn.NMSBoxes([[float(a), float(b_), float(c - a), float(d - b_)]
                                 for a, b_, c, d in B], S.tolist(), self.conf, iou)
        keep = np.array(keep).reshape(-1)
        return B[keep], S[keep]


def detect_wall_holds(wall_image, model: Path | None = None) -> HoldBoxes | None:
    """Holds in the wall image (WallImage), in wall coordinates. None without a model."""
    model = model or find_model()
    if model is None or wall_image is None:
        return None
    det = HoldDetector(model)
    seen = wall_image.image.max(axis=2) > 0
    b, s = det.detect_multi(wall_image.image)
    if len(b):  # drop boxes mostly outside what the camera saw
        cx = ((b[:, 0] + b[:, 2]) / 2).astype(int).clip(0, seen.shape[1] - 1)
        cy = ((b[:, 1] + b[:, 3]) / 2).astype(int).clip(0, seen.shape[0] - 1)
        ok = seen[cy, cx]
        b, s = b[ok], s[ok]
    Ti = np.linalg.inv(wall_image.T)  # image px -> wall coordinates
    corners = np.stack([b[:, [0, 1]], b[:, [2, 3]]], 1).reshape(-1, 2) if len(b) else \
        np.zeros((0, 2))
    if len(corners):
        q = np.c_[corners, np.ones(len(corners))] @ Ti.T
        q = (q[:, :2] / q[:, 2:]).reshape(-1, 2, 2)
        wb = np.c_[q.min(1), q.max(1)]
    else:
        wb = np.zeros((0, 4))
    return HoldBoxes(wb, s, model.name + VERSION)


def load_or_detect(result_dir: Path, wall_image) -> HoldBoxes | None:
    """Cached detections for a results folder (re-detected when the model or the wall
    image changed)."""
    result_dir = Path(result_dir)
    model = find_model()
    cache = result_dir / CACHE
    wall = result_dir / "wall.jpg"
    if cache.exists():
        hb = HoldBoxes.load(cache)
        fresh = (not wall.exists() or cache.stat().st_mtime >= wall.stat().st_mtime)
        if fresh and (model is None or hb.model == model.name + VERSION):
            return hb
    if model is None or wall_image is None:
        return None
    hb = detect_wall_holds(wall_image, model)
    if hb is not None:
        hb.save(cache)
        draw_detections(result_dir / "holds_detected.jpg", wall_image, hb)
    return hb


def draw_detections(path: Path, wall_image, hb: HoldBoxes) -> None:
    img = wall_image.image.copy()
    T = wall_image.T
    for (x0, y0, x1, y1), s in zip(hb.boxes, hb.scores, strict=True):
        p = np.array([[x0, y0, 1], [x1, y1, 1]]) @ T.T
        p = (p[:, :2] / p[:, 2:]).astype(int)
        cv2.rectangle(img, tuple(p[0]), tuple(p[1]), (0, 255, 255), 2)
        cv2.putText(img, f"{s:.2f}", (int(p[0][0]), int(p[0][1]) - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f"{len(hb)} holds detected ({hb.model})", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
