"""wall_view.mp4: the video seen in wall coordinates (camera motion removed).

Every frame is warped onto the wall image (wall.jpg, dimmed where the current frame
doesn't reach), and the skeleton is drawn in wall coordinates, with 2-second trails for
hands and feet. If the camera layer works, the wall stands still while the camera moves,
and a hand or foot resting on a hold leaves a dot instead of a streak.
"""

from __future__ import annotations

import warnings
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from climbvision.engine.camera import LOST, CameraTrack, WallImage, apply_h
from climbvision.engine.skeleton import BONES, J
from climbvision.engine.track import Track
from climbvision.engine.video import VideoReader, VideoWriter
from climbvision.output.draw import draw_label

TRAIL = {"L hand": ((J.L_WRIST, J.L_HAND), (255, 90, 160)),
         "R hand": ((J.R_WRIST, J.R_HAND), (255, 170, 0)),
         "L foot": ((J.L_HEEL, J.L_TOE), (0, 200, 255)),
         "R foot": ((J.R_HEEL, J.R_TOE), (60, 220, 60))}


def render_wall_view(video_path: str | Path, result_dir: str | Path,
                     out_path: str | Path | None = None, max_side: int = 1280,
                     trail_seconds: float = 2.0) -> Path:
    result_dir = Path(result_dir)
    out_path = Path(out_path) if out_path else result_dir / "wall_view.mp4"
    cam = CameraTrack.load(result_dir / "camera.csv")
    wall = WallImage.load(result_dir)
    track = Track.from_csv(result_dir / "landmarks.csv")
    zoom = min(1.0, max_side / max(wall.image.shape[:2]))
    T = np.diag([zoom, zoom, 1.0]) @ wall.T  # wall coords -> output pixels
    size = (int(wall.image.shape[1] * zoom) // 2 * 2, int(wall.image.shape[0] * zoom) // 2 * 2)
    bg = (cv2.resize(wall.image, size, interpolation=cv2.INTER_AREA) * 0.4).astype(np.uint8)
    s = max(size) / 1000.0
    trails = {k: deque() for k in TRAIL}

    with VideoReader(video_path) as reader:
        fw, fh = cam.frame_size
        resize = (reader.info.width, reader.info.height) != (fw, fh)
        keep = max(1, int(trail_seconds * reader.info.fps))
        with VideoWriter(out_path, reader.info.fps, size) as writer:
            for i, t, img in reader.frames():
                if i >= cam.n:
                    break
                if resize:
                    img = cv2.resize(img, (fw, fh), interpolation=cv2.INTER_AREA)
                frame = bg.copy()
                lines = [f"t={t:5.2f}s  wall view"]
                if cam.status[i] != LOST:
                    M = T @ cam.H[i]
                    warped = cv2.warpPerspective(img, M, size)
                    m = cv2.warpPerspective(np.full((fh, fw), 255, np.uint8), M, size) > 0
                    frame[m] = warped[m]
                    if i < track.n and np.isfinite(track.image[i, J.L_SHOULDER]).all():
                        p = apply_h(M, track.image[i])
                        for a, b in BONES:
                            if np.isfinite(p[[a, b]]).all():
                                cv2.line(frame, tuple(int(v) for v in p[a]),
                                         tuple(int(v) for v in p[b]), (230, 230, 230),
                                         max(1, int(2 * s)), cv2.LINE_AA)
                        for k, (joints, _) in TRAIL.items():
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", RuntimeWarning)
                                q = np.nanmean(p[list(joints)], axis=0)
                            trails[k].append(q if np.isfinite(q).all() else None)
                    else:
                        for k in trails:
                            trails[k].append(None)
                else:
                    lines.append("camera lost")
                    for k in trails:
                        trails[k].append(None)
                for k, (_, col) in TRAIL.items():
                    while len(trails[k]) > keep:
                        trails[k].popleft()
                    pts = [q for q in trails[k] if q is not None]
                    for a, b in zip(pts, pts[1:], strict=False):
                        cv2.line(frame, tuple(int(v) for v in a), tuple(int(v) for v in b), col,
                                 max(2, int(3 * s)), cv2.LINE_AA)
                    if pts:
                        cv2.circle(frame, tuple(int(v) for v in pts[-1]), max(4, int(7 * s)),
                                   col, -1, cv2.LINE_AA)
                draw_label(frame, lines)
                writer.write(frame)
    return out_path
