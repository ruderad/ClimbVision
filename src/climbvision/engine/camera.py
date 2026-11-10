"""Camera layer: where each video frame looks on the wall (handheld video).

For every frame we estimate a homography H[i] that maps image pixels of frame i to
*wall coordinates*: the pixel coordinates of the first registered frame (the reference
view), extended beyond its edges as the camera pans. A climbing wall is close to a plane
and a handheld camera mostly rotates, so one homography per frame describes the camera
motion well. With it, a hand or foot that rests on a hold stays put in wall coordinates
even while the camera moves — which is what the contacts layer needs (E4).

How (all at a reduced working resolution, ~960 px long side):
  1. Mask out the climber (capsules around the skeleton's bones, from the pose layer)
     and the image border: only the wall may vote.
  2. Track wall corners from frame to frame (pyramidal Lucas-Kanade, forward-backward
     checked). Each tracked point carries its wall coordinates, fixed when it was first
     seen, so H[i] is fitted directly frame -> wall (RANSAC), not chained frame to frame.
  3. When too few points survive, new corners are spawned and get wall coordinates
     through the current H. This is where drift could creep in, so:
  4. Keyframes: a frame is stored (ORB features + wall coordinates) whenever the view
     has moved on. Every few frames the current frame is matched against older
     keyframes that look at the same part of the wall; their wall coordinates join the
     fit, and the tracked points are re-anchored. Coming back to a part of the wall
     snaps the registration back onto it (loop closure).
  5. If tracking is lost (fast pan, motion blur, climber filling the frame), frames get
     no H (NaN) until the frame can be matched against the keyframes again.

A fixed camera (tripod) comes out as H ~ identity and is flagged `static`.
Output: camera.csv (+ camera.json) with H per frame and its quality, and wall.jpg: the wall
without the climber (per-pixel median of frames warped into wall coordinates, climber
masked out) — the background for subject separation (E3) and hold detection (E4).
"""

from __future__ import annotations

import csv
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from climbvision.engine.skeleton import BONES, J
from climbvision.engine.track import Track

LOST, TRACKED, ANCHORED = 0, 1, 2  # status per frame


# ------------------------------------------------------------------ geometry helpers

def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Map (..., 2) points with a 3x3 homography (NaN in -> NaN out)."""
    pts = np.asarray(pts, float)
    p = np.concatenate([pts, np.ones(pts.shape[:-1] + (1,))], axis=-1) @ np.asarray(H).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return p[..., :2] / p[..., 2:3]


def _scale_h(H: np.ndarray, s: float) -> np.ndarray:
    """H on working-resolution pixels -> H on full-resolution pixels (both sides x s)."""
    D = np.diag([s, s, 1.0])
    return np.linalg.inv(D) @ H @ D


@dataclass
class CameraTrack:
    time_s: np.ndarray  # (n,)
    H: np.ndarray  # (n, 3, 3) frame pixels -> wall pixels; NaN where lost
    status: np.ndarray  # (n,) LOST / TRACKED / ANCHORED
    inliers: np.ndarray  # (n,) points supporting H
    frame_size: tuple[int, int]
    static: bool = False
    info: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.time_s)

    @property
    def ok(self) -> np.ndarray:
        return self.status != LOST

    def to_wall(self, pts: np.ndarray) -> np.ndarray:
        """(n, ..., 2) per-frame image points -> wall coordinates (NaN where lost)."""
        pts = np.asarray(pts, float)
        out = np.full(pts.shape, np.nan)
        for i in range(min(self.n, len(pts))):
            if self.status[i] != LOST:
                out[i] = apply_h(self.H[i], pts[i])
        return out

    def centre_path(self) -> np.ndarray:
        """Wall position of the image centre per frame (where the camera points)."""
        w, h = self.frame_size
        c = np.array([w / 2, h / 2])
        return np.array([apply_h(self.H[i], c) if self.status[i] != LOST else [np.nan] * 2
                         for i in range(self.n)])

    # -------------------------------------------------------------- files
    def save(self, path: str | Path) -> None:
        with Path(path).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "time_s", "status", "inliers",
                        *(f"h{r}{c}" for r in range(3) for c in range(3))])
            for i in range(self.n):
                w.writerow([i, f"{self.time_s[i]:.4f}", int(self.status[i]),
                            int(self.inliers[i]), *(f"{v:.9g}" for v in self.H[i].ravel())])
        meta = {"frame_size": list(self.frame_size), "static": self.static, **self.info}
        Path(path).with_suffix(".json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> CameraTrack:
        rows = list(csv.DictReader(Path(path).open()))
        meta = json.loads(Path(path).with_suffix(".json").read_text())
        H = np.array([[float(r[f"h{a}{b}"]) for a in range(3) for b in range(3)]
                      for r in rows]).reshape(-1, 3, 3)
        info = {k: v for k, v in meta.items() if k not in ("frame_size", "static")}
        return cls(np.array([float(r["time_s"]) for r in rows]), H,
                   np.array([int(r["status"]) for r in rows]),
                   np.array([int(r["inliers"]) for r in rows]),
                   tuple(meta["frame_size"]), meta.get("static", False), info)

    @classmethod
    def identity(cls, time_s: np.ndarray, frame_size: tuple[int, int]) -> CameraTrack:
        n = len(time_s)
        return cls(np.asarray(time_s), np.tile(np.eye(3), (n, 1, 1)), np.full(n, ANCHORED),
                   np.zeros(n, int), frame_size, True, {"note": "assumed fixed camera"})


# ------------------------------------------------------------------ climber mask

def body_size_px(track: Track) -> float:
    from climbvision.engine.scale import chain_len

    v = chain_len(track.image)
    return float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")


def climber_mask(shape: tuple[int, int], joints: np.ndarray | None, body_px: float,
                 scale: float, border: float = 0.02) -> np.ndarray:
    """255 = wall (may be used), 0 = climber or image border. `joints` in full-res px."""
    h, w = shape
    m = np.full((h, w), 255, np.uint8)
    b = max(2, int(border * max(h, w)))
    m[:b], m[-b:], m[:, :b], m[:, -b:] = 0, 0, 0, 0
    if joints is None or not np.isfinite(body_px):
        return m
    p = joints * scale
    r = max(3, int(0.09 * body_px * scale))
    for a, c in BONES:
        if np.isfinite(p[a]).all() and np.isfinite(p[c]).all():
            cv2.line(m, tuple(int(v) for v in p[a]), tuple(int(v) for v in p[c]), 0, 2 * r)
    torso = [J.L_SHOULDER, J.R_SHOULDER, J.R_HIP, J.L_HIP]
    if np.isfinite(p[torso]).all():
        cv2.fillConvexPoly(m, p[torso].astype(np.int32), 0)
    for j in (J.HEAD, J.L_HAND, J.R_HAND, J.L_TOE, J.R_TOE, J.L_HEEL, J.R_HEEL):
        if np.isfinite(p[j]).all():
            cv2.circle(m, tuple(int(v) for v in p[j]), int(r * (1.8 if j == J.HEAD else 1.3)),
                       0, -1)
    return m


# ------------------------------------------------------------------ the tracker

@dataclass
class _Keyframe:
    frame: int
    kp_wall: np.ndarray  # (k, 2) wall coords (working scale)
    des: np.ndarray  # ORB descriptors
    footprint: np.ndarray  # (4, 2) wall coords of the frame corners


@dataclass
class CameraParams:
    work_long_side: int = 960  # working resolution
    max_points: int = 600  # tracked corners
    min_points: int = 250  # spawn new corners below this
    ransac_px: float = 2.5  # RANSAC threshold at working scale
    min_inliers: int = 25
    anchor_every: int = 5  # frames between keyframe matches
    keyframe_overlap: float = 0.75  # new keyframe when overlap with the last one is below
    anchor_min_age_s: float = 1.0  # only match keyframes at least this old (loop closure)
    orb_features: int = 2500
    static_px: float = 0.01  # camera centre moves < 1% of the frame diagonal -> static
    plate_every_s: float = 0.5  # sample a frame for the wall image this often...
    plate_max: int = 40  # ...at most this many (evenly spread)
    plate_scale: float = 1.0  # sample size relative to the working resolution
    plate_max_side: int = 1600  # wall image size limit (px)


def _overlap(fp_a: np.ndarray, fp_b: np.ndarray) -> float:
    """Area of intersection / area of a, for two convex quads."""
    a = fp_a.astype(np.float32)
    b = fp_b.astype(np.float32)
    area_a = cv2.contourArea(a)
    if area_a <= 0:
        return 0.0
    inter, _ = cv2.intersectConvexConvex(a, b)
    return float(inter) / area_a


class _Tracker:
    def __init__(self, p: CameraParams, size_small: tuple[int, int]):
        self.p = p
        self.w, self.h = size_small
        self.corners = np.array([[0, 0], [self.w, 0], [self.w, self.h], [0, self.h]], float)
        self.orb = cv2.ORB_create(p.orb_features, fastThreshold=10)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.prev = None  # previous gray frame
        self.pts = np.empty((0, 2), np.float32)  # tracked points, current frame
        self.wall = np.empty((0, 2))  # their wall coordinates
        self.H = None  # last good frame -> wall
        self.keyframes: list[_Keyframe] = []

    # ---- features
    def _orb(self, gray, mask):
        kp, des = self.orb.detectAndCompute(gray, mask)
        if des is None or len(kp) < 10:
            return None, None
        return np.float32([k.pt for k in kp]), des

    def _match(self, des_a, des_b, ratio=0.8):
        if des_a is None or des_b is None or len(des_b) < 2:
            return []
        out = []
        for m in self.bf.knnMatch(des_a, des_b, k=2):
            if len(m) == 2 and m[0].distance < ratio * m[1].distance:
                out.append((m[0].queryIdx, m[0].trainIdx))
        return out

    def _fit(self, src, dst):
        if len(src) < self.p.min_inliers:
            return None, None
        H, inl = cv2.findHomography(src.astype(np.float32), dst.astype(np.float32),
                                    cv2.USAC_MAGSAC, self.p.ransac_px, maxIters=2000,
                                    confidence=0.999)
        if H is None or inl is None or inl.sum() < self.p.min_inliers:
            return None, None
        return H, inl.ravel().astype(bool)

    def _spawn(self, gray, mask, H):
        """Add corners where there are none yet; their wall coords come from H."""
        m = mask.copy()
        for x, y in self.pts:
            cv2.circle(m, (int(x), int(y)), 8, 0, -1)
        need = self.p.max_points - len(self.pts)
        if need <= 0:
            return
        new = cv2.goodFeaturesToTrack(gray, need, 0.01, 8, mask=m, blockSize=7)
        if new is None:
            return
        new = new.reshape(-1, 2).astype(np.float32)
        self.pts = np.vstack([self.pts, new])
        self.wall = np.vstack([self.wall, apply_h(H, new)])

    def _add_keyframe(self, i, gray, mask, H):
        xy, des = self._orb(gray, mask)
        if xy is None:
            return
        self.keyframes.append(_Keyframe(i, apply_h(H, xy), des, apply_h(H, self.corners)))

    def _anchor(self, i, gray, mask, H, t, now_s):
        """Match against older keyframes that see the same wall; return extra
        (frame point, wall point) pairs."""
        fp = apply_h(H, self.corners) if H is not None else None
        cands = [k for k in self.keyframes if now_s - t[k.frame] >= self.p.anchor_min_age_s
                 and (fp is None or _overlap(fp, k.footprint) > 0.2)]
        if not cands:
            return None
        xy, des = self._orb(gray, mask)
        if xy is None:
            return None
        if fp is not None:  # closest views first
            cands.sort(key=lambda k: -_overlap(fp, k.footprint))
        src, dst = [], []
        for k in cands[:3]:
            for a, b in self._match(des, k.des):
                src.append(xy[a])
                dst.append(k.kp_wall[b])
        if len(src) < self.p.min_inliers:
            return None
        return np.array(src), np.array(dst)

    # ---- one frame
    def step(self, i, gray, mask, t) -> tuple[np.ndarray | None, int, int]:
        p = self.p
        H, status, n_in = None, LOST, 0
        if self.prev is not None and len(self.pts) >= p.min_inliers and self.H is not None:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, self.pts, None,
                                                  winSize=(21, 21), maxLevel=3)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev, nxt, None,
                                                    winSize=(21, 21), maxLevel=3)
            good = (st.ravel() == 1) & (st2.ravel() == 1) & \
                (np.linalg.norm(back - self.pts, axis=1) < 1.0)
            ix = np.clip(nxt[:, 0].astype(int), 0, self.w - 1)
            iy = np.clip(nxt[:, 1].astype(int), 0, self.h - 1)
            good &= mask[iy, ix] > 0  # the climber moved over it
            self.pts, self.wall = nxt[good], self.wall[good]
            Hf, inl = self._fit(self.pts, self.wall)
            if Hf is not None:
                self.pts, self.wall = self.pts[inl], self.wall[inl]
                H, status, n_in = Hf, TRACKED, int(inl.sum())

        anchor_due = (i % p.anchor_every == 0) or H is None
        if anchor_due and self.keyframes:
            pairs = self._anchor(i, gray, mask, H if H is not None else self.H, t, t[i])
            if pairs is not None:
                src, dst = pairs
                if H is not None and len(self.pts):  # joint fit: tracked + keyframe points
                    src = np.vstack([src, self.pts])
                    dst = np.vstack([dst, self.wall])
                Ha, inl = self._fit(src, dst)
                if Ha is not None and (H is None or inl.sum() >= p.min_inliers):
                    H, status, n_in = Ha, ANCHORED, int(inl.sum())
                    if len(self.pts):
                        # re-anchor the tracked points that agree with the anchored fit
                        # (removes the drift they carried); drop the rest, e.g. points
                        # sliding along the edge of a moving person
                        new_wall = apply_h(H, self.pts)
                        keep = np.linalg.norm(new_wall - self.wall, axis=1) < 2 * p.ransac_px
                        self.pts, self.wall = self.pts[keep], new_wall[keep]

        if H is None and not self.keyframes:  # very first frame: it *is* the reference
            H, status = np.eye(3), ANCHORED
        if H is not None:
            if len(self.pts) < p.min_points:
                self._spawn(gray, mask, H)
            fp = apply_h(H, self.corners)
            if not self.keyframes or _overlap(fp, self.keyframes[-1].footprint) \
                    < p.keyframe_overlap:
                self._add_keyframe(i, gray, mask, H)
            self.H = H
        else:
            self.pts = np.empty((0, 2), np.float32)
            self.wall = np.empty((0, 2))
        self.prev = gray
        return H, status, n_in


def estimate_camera(video_path: str | Path, track: Track | None = None,
                    params: CameraParams | None = None,
                    max_frames: int | None = None) -> tuple[CameraTrack, WallImage]:
    """Run the camera layer over a video. `track` (the pose layer) masks the climber.
    Returns the per-frame homographies and the wall image."""
    from climbvision.engine.video import VideoReader

    p = params or CameraParams()
    body = body_size_px(track) if track is not None else float("nan")
    if track is not None:  # cover the same frames as the pose layer
        max_frames = track.n if max_frames is None else min(max_frames, track.n)
    Hs, status, inliers, times = [], [], [], []
    with VideoReader(video_path) as reader:
        W, Hh = reader.info.width, reader.info.height
        s = min(1.0, p.work_long_side / max(W, Hh))
        size = (int(round(W * s)), int(round(Hh * s)))
        tr = _Tracker(p, size)
        last_joints = None  # frames without a pose keep the last known mask
        samples: list = []  # frames for the wall image
        next_sample = -np.inf
        for i, t, img in reader.frames():
            if max_frames is not None and i >= max_frames:
                break
            small = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            joints = track.image[i] if track is not None and i < track.n else None
            if joints is not None and np.isfinite(joints[J.L_SHOULDER]).all():
                last_joints = joints
            else:
                joints = last_joints
            mask = climber_mask(gray.shape, joints, body, s)
            times.append(t)
            H, st, n_in = tr.step(i, gray, mask, times)
            if H is not None and t >= next_sample:
                next_sample = t + p.plate_every_s
                hs = p.plate_scale
                samples.append((cv2.resize(small, None, fx=hs, fy=hs,
                                           interpolation=cv2.INTER_AREA),
                                cv2.resize(mask, None, fx=hs, fy=hs,
                                           interpolation=cv2.INTER_NEAREST),
                                _scale_h(H, s), s * hs))
            Hs.append(_scale_h(H, s) if H is not None else np.full((3, 3), np.nan))
            status.append(st)
            inliers.append(n_in)
    cam = CameraTrack(np.asarray(times), np.array(Hs), np.array(status), np.array(inliers),
                      (W, Hh))
    cam.info = {"work_scale": s, "keyframes": len(tr.keyframes),
                "registered_pct": round(100 * float(cam.ok.mean()), 1) if cam.n else 0.0}
    c = cam.centre_path()
    if np.isfinite(c).any():
        move = float(np.nanmax(np.linalg.norm(c - np.nanmedian(c, axis=0), axis=1)))
        cam.static = move < p.static_px * float(np.hypot(W, Hh))
        cam.info["camera_travel_px"] = round(move, 1)
    if len(samples) > p.plate_max:
        keep = np.linspace(0, len(samples) - 1, p.plate_max).round().astype(int)
        samples = [samples[k] for k in keep]
    return cam, build_wall_image(samples, (W, Hh), p.plate_max_side)


@dataclass
class WallImage:
    """The wall in wall coordinates, without the climber.

    image: (h, w, 3) BGR, black where never seen; coverage: (h, w) number of frames
    that saw each pixel unoccluded; T: 3x3 mapping wall coordinates (full-res px of the
    reference view) -> image pixels."""

    image: np.ndarray
    coverage: np.ndarray
    T: np.ndarray

    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        cv2.imwrite(str(folder / "wall.jpg"), self.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (folder / "wall.json").write_text(json.dumps({"T": self.T.tolist(),
                                                      "size": list(self.image.shape[1::-1])}))

    @classmethod
    def load(cls, folder: str | Path) -> WallImage:
        folder = Path(folder)
        img = cv2.imread(str(folder / "wall.jpg"))
        meta = json.loads((folder / "wall.json").read_text())
        return cls(img, np.ones(img.shape[:2], np.int32), np.array(meta["T"]))


def build_wall_image(samples: list, frame_size: tuple[int, int],
                     max_side: int = 1400) -> WallImage:
    """Per-pixel median of sample frames warped into wall coordinates, using only
    unmasked (wall) pixels. samples: (image, mask, H_full, image_scale)."""
    if not samples:
        return WallImage(np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10), np.int32),
                         np.eye(3))
    W, Hh = frame_size
    corners = np.array([[0, 0], [W, 0], [W, Hh], [0, Hh]], float)
    pts = np.vstack([apply_h(H, corners) for _, _, H, _ in samples])
    lo, hi = np.floor(pts.min(0)), np.ceil(pts.max(0))
    # never larger than the samples' own resolution, never larger than max_side
    zoom = min(samples[0][3], max_side / float(max(hi - lo)))
    T = np.array([[zoom, 0, -lo[0] * zoom], [0, zoom, -lo[1] * zoom], [0, 0, 1]])
    size = (int((hi - lo)[0] * zoom) + 1, int((hi - lo)[1] * zoom) + 1)
    imgs = np.zeros((len(samples), size[1], size[0], 3), np.uint8)
    masks = np.zeros((len(samples), size[1], size[0]), bool)
    for k, (img, m, H, sc) in enumerate(samples):
        M = T @ H @ np.diag([1 / sc, 1 / sc, 1.0])
        imgs[k] = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)
        masks[k] = cv2.warpPerspective(m, M, size, flags=cv2.INTER_NEAREST) > 0
    out = np.zeros((size[1], size[0], 3), np.uint8)
    for r0 in range(0, size[1], 64):  # median in row blocks (memory)
        block = imgs[:, r0:r0 + 64].astype(np.float32)
        block[~masks[:, r0:r0 + 64]] = np.nan
        with warnings.catch_warnings():  # pixels never seen are all-NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(block, axis=0)
        out[r0:r0 + 64] = np.nan_to_num(med, nan=0).astype(np.uint8)
    return WallImage(out, masks.sum(axis=0).astype(np.int32), T)


def run_camera(video_path: str | Path, result_dir: str | Path,
               params: CameraParams | None = None) -> CameraTrack:
    """Camera layer for a results folder: reads landmarks.csv (climber mask), writes
    camera.csv/.json and wall.jpg/.json."""
    result_dir = Path(result_dir)
    track = None
    if (result_dir / "landmarks.csv").exists():
        track = Track.from_csv(result_dir / "landmarks.csv")
    cam, wall = estimate_camera(video_path, track, params)
    cam.save(result_dir / "camera.csv")
    wall.save(result_dir)
    return cam


def camera_summary(cam: CameraTrack, px_per_m: float | None = None) -> dict:
    """Clip-level description of the camera motion for metrics.json."""
    c = cam.centre_path()
    out: dict = {"static": bool(cam.static),
                 "registered_pct": round(100 * float(cam.ok.mean()), 1) if cam.n else 0.0,
                 "keyframes": cam.info.get("keyframes")}
    if np.isfinite(c).any():
        span = np.nanmax(c, axis=0) - np.nanmin(c, axis=0)
        out["pan_px"] = [round(float(v), 1) for v in span]
        if px_per_m:
            out["pan_m"] = [round(float(v) / px_per_m, 2) for v in span]
    lost = cam.n - int(cam.ok.sum())
    if lost:
        out["lost_frames"] = lost
    return out
