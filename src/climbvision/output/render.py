"""Draw the annotated video from saved results (no pose estimation needed).

Shows the skeleton, COM with a trail, the floor line, ground contacts, the support
span on the floor and the COM's plumb line (green = COM over the support, red = not).
On the wall: limbs on holds, the base of support (hull of the limbs on holds) and the
COM's plumb line down the wall (green = COM over the feet, red = not).
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from climbvision.analysis import clip_info, load_clip, resolve_recipe
from climbvision.engine.skeleton import Skeleton
from climbvision.engine.video import VideoReader, VideoWriter
from climbvision.output.draw import (
    INSIDE_COLOR,
    OUTSIDE_COLOR,
    draw_base,
    draw_com,
    draw_contacts,
    draw_floor,
    draw_grips,
    draw_label,
    draw_skeleton,
    draw_support,
)


def render(video_path: str | Path, result_dir: str | Path, out_path: str | Path | None = None,
           trail_seconds: float = 2.0) -> Path:
    result_dir = Path(result_dir)
    out_path = Path(out_path) if out_path else result_dir / "annotated.mp4"
    info = clip_info(result_dir)
    # same recipe (hence bar / no bar scene) as the analysis step used
    mpath = result_dir / "metrics.json"
    recipe = json.loads(mpath.read_text()).get("exercise", "auto") if mpath.exists() else "auto"
    try:
        rec = resolve_recipe(result_dir, recipe)
    except ValueError:  # a recipe from a folder that isn't loaded now
        rec = resolve_recipe(result_dir, "auto")
    clip = load_clip(result_dir, bar=rec.bar)
    track, st = clip.track, clip.stability
    wall_scene = rec.scene == "wall"
    if wall_scene:  # on the wall: holds instead of floor, support and grips
        ws = clip.wall_support
        labels = ws.labels
        route_labels = clip.route.labels
    else:
        labels = st.labels
    fl = st.floor

    with VideoReader(video_path) as reader:
        size = (reader.info.width, reader.info.height)
        # results may come from a resized copy of the video: draw at that size
        target = tuple(info["frame_size"]) if "frame_size" in info else size
        resize = target != size
        size = target
        trail: deque[np.ndarray] = deque(maxlen=max(1, int(trail_seconds * reader.info.fps)))
        step = int(info.get("frame_step", 1))
        with VideoWriter(out_path, reader.info.fps / step, size) as writer:
            for i, (_, t, img) in enumerate(reader.frames(step=step)):
                if i >= track.n:
                    break
                if resize:
                    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                com = st.com_px[i]
                trail.append(com)
                if np.isfinite(fl.y0[i]) and not wall_scene:
                    draw_floor(img, fl.slope, fl.y0[i])
                lines = [f"t={t:5.2f}s  {labels[i]}"]
                colors = [(255, 255, 255)]
                if wall_scene and route_labels[i]:
                    lines.append(route_labels[i])
                    colors.append((120, 230, 255))
                if track.detected[i]:
                    sk = Skeleton(track.image[i], track.world[i], track.visibility[i])
                    draw_skeleton(img, sk)
                    if wall_scene:
                        held = [ws.points[limb][i] for limb in ws.on_hold
                                if ws.on_hold[limb][i]]
                        img_pts = [p for p in _to_image(clip, i, held) if np.isfinite(p).all()]
                        plumb, inside = None, None
                        c = ws.com[i]
                        if held and ws.on_wall[i] and np.isfinite(c).all():  # straight down
                            # down to the lowest hold, at least a quarter body
                            low = max(max(float(p[1]) for p in held), c[1] + 0.25 * ws.body_px)
                            ends = _to_image(clip, i, [c, np.array([c[0], low])])
                            plumb = tuple(ends) if len(ends) == 2 else None
                            m = ws.com_over_feet_cm[i]
                            if np.isfinite(m):
                                inside = bool(m > 0)
                                lines.append(f"COM over feet {m:+.0f} cm")
                                colors.append(INSIDE_COLOR if inside else OUTSIDE_COLOR)
                        draw_base(img, np.array(img_pts).reshape(-1, 2), plumb, inside)
                        draw_contacts(img, np.array(img_pts).reshape(-1, 2))
                        draw_com(img, com, list(trail))
                        draw_label(img, lines, colors)
                        writer.write(img)
                        continue
                    m = st.margin_cm[i]
                    if np.isfinite(m):
                        draw_support(img, fl.slope, fl.y0[i], st.support_lo_px[i],
                                     st.support_hi_px[i], com[0], com[1], m > 0)
                        lines.append(f"margin {m:+.1f} cm")
                        colors.append(INSIDE_COLOR if m > 0 else OUTSIDE_COLOR)
                    draw_contacts(img, track.image[i, st.contacts.point[i]])
                    grips = [st.contacts.hand_px[sd][i] for sd, g in st.contacts.grip.items()
                             if g[i]]
                    if grips:
                        draw_grips(img, grips)
                        sw = st.swing_cm[i]
                        if np.isfinite(sw):
                            lines.append(f"swing {sw:+.1f} cm")
                            colors.append((255, 160, 210))
                    draw_com(img, com, list(trail))
                else:
                    lines[0] = f"t={t:5.2f}s  no pose"
                if st.feet_out_of_frame[i]:
                    lines.append("feet out of frame")
                    colors.append((0, 200, 255))
                draw_label(img, lines, colors)
                writer.write(img)
    return out_path


def _to_image(clip, i: int, wall_pts: list) -> list:
    """Wall coordinates -> image of frame i (inverse of the camera layer's H)."""
    if not wall_pts:
        return []
    if clip.camera is None:
        return list(wall_pts)
    from climbvision.engine.camera import LOST, apply_h

    if clip.camera.status[i] == LOST:
        return []
    return list(apply_h(np.linalg.inv(clip.camera.H[i]), np.array(wall_pts)))
