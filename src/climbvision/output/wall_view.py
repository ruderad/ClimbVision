"""wall_view.mp4: the video seen in wall coordinates (camera motion removed).

Every frame is warped onto the wall image (wall.jpg, dimmed where the current frame
doesn't reach), and the skeleton is drawn in wall coordinates, with 2-second trails for
hands and feet. If the camera layer works, the wall stands still while the camera moves,
and a hand or foot resting on a hold leaves a dot instead of a streak.

From the wall layer: holds used so far (white rings with their id), limbs on holds
(filled discs), the hull of the limbs on holds (yellow) and the COM (green inside the
hull, red outside, pink with fewer than 3 limbs on holds). From the
route layer: an arrow for every move so far (the route builds up), and the latest move
in the label.
"""

from __future__ import annotations

import json
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
    ws = route = None
    try:
        from climbvision.analysis import load_clip

        clip = load_clip(result_dir)  # the cleaned pose
        track = clip.track
        ws = clip.wall_support
        route = clip.route
    except Exception:  # noqa: BLE001 - the view still works without the wall layer
        track = Track.from_csv(result_dir / "landmarks.csv")
        ws = route = None
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
        step = 1
        if (result_dir / "summary.json").exists():
            step = int(json.loads((result_dir / "summary.json").read_text()).get("frame_step", 1))
        keep = max(1, keep // step)
        with VideoWriter(out_path, reader.info.fps / step, size) as writer:
            for i, (_, t, img) in enumerate(reader.frames(step=step)):
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
                if ws is not None and i < len(ws.labels):
                    if route is not None:
                        _draw_route_so_far(frame, ws, route, i, T, s)
                    _draw_wall_layer(frame, ws, i, T, s)
                    lines.append(ws.labels[i])
                    if route is not None and route.labels[i]:
                        lines.append(route.labels[i])
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


LIMB_COLOUR = {"l_hand": TRAIL["L hand"][1], "r_hand": TRAIL["R hand"][1],
               "l_foot": TRAIL["L foot"][1], "r_foot": TRAIL["R foot"][1]}


def _draw_wall_layer(frame: np.ndarray, ws, i: int, T: np.ndarray, s: float) -> None:
    def pt(p):
        q = apply_h(T, p)
        return int(q[0]), int(q[1])

    r = max(6, int(14 * s))
    for h in ws.holds:  # holds used so far
        if min(a.start for a in h.anchors) <= i:
            cv2.circle(frame, pt(h.pos), r, (255, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
            x, y = pt(h.pos)
            cv2.putText(frame, str(h.id), (x + r, y - r), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * s + 0.2,
                        (255, 255, 255), 1, cv2.LINE_AA)
    on = []
    for limb, anchors in ws.anchors.items():
        for a in anchors:
            if a.start <= i < a.end and a.on_hold:
                on.append(pt(a.pos))
                cv2.circle(frame, pt(a.pos), int(r * 0.7), LIMB_COLOUR[limb], -1, cv2.LINE_AA)
    if len(on) >= 3:
        hull = cv2.convexHull(np.array(on, np.int32))
        cv2.polylines(frame, [hull], True, (0, 230, 255), max(1, int(2 * s)), cv2.LINE_AA)
    if np.isfinite(ws.com[i]).all():  # green inside the hull, red outside, pink: no hull
        pm = ws.polygon_margin_cm[i]
        col = (200, 60, 255) if not np.isfinite(pm) else (90, 210, 60) if pm > 0 else (
            50, 50, 240)
        cv2.circle(frame, pt(ws.com[i]), int(r * 0.6), col, -1, cv2.LINE_AA)


def _draw_route_so_far(frame: np.ndarray, ws, route, i: int, T: np.ndarray, s: float) -> None:
    """The route builds up: an arrow for every move that has landed, in the limb's
    colour (thin = feet), the latest ones brightest."""
    pos = {h.id: h.pos for h in ws.holds}
    done = [m for m in route.moves if m.catch <= i]
    for k, m in enumerate(done):
        a, b = apply_h(T, pos[m.from_hold]), apply_h(T, pos[m.to_hold])
        fade = 1.0 if k >= len(done) - 3 else 0.55
        col = tuple(int(c * fade) for c in LIMB_COLOUR[m.limb])
        w = max(1, int((3 if m.kind == "hand" else 2) * s))
        cv2.arrowedLine(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), col, w,
                        cv2.LINE_AA, tipLength=0.15)
