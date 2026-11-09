"""Draw the annotated video from saved results (no pose estimation needed).

Shows the skeleton, COM with a trail, the floor line, ground contacts, the support
span on the floor and the COM's plumb line (green = COM over the support, red = not).
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from climbvision.angles import compute_angles
from climbvision.draw import (
    INSIDE_COLOR,
    OUTSIDE_COLOR,
    draw_com,
    draw_contacts,
    draw_floor,
    draw_grips,
    draw_label,
    draw_skeleton,
    draw_support,
)
from climbvision.metrics import BAR_EXERCISES, guess_exercise
from climbvision.skeleton import Skeleton
from climbvision.stability import compute_stability, stance_labels
from climbvision.track import Track
from climbvision.video import VideoReader, VideoWriter


def render(video_path: str | Path, result_dir: str | Path, out_path: str | Path | None = None,
           trail_seconds: float = 2.0) -> Path:
    result_dir = Path(result_dir)
    out_path = Path(out_path) if out_path else result_dir / "annotated.mp4"
    info = json.loads((result_dir / "summary.json").read_text())
    track = Track.from_csv(result_dir / "landmarks.csv")
    ang = compute_angles(track)
    # same exercise (hence bar / no bar) as the metrics step used
    mpath = result_dir / "metrics.json"
    exercise = (json.loads(mpath.read_text()).get("exercise") if mpath.exists()
                else guess_exercise(result_dir.name))
    st = compute_stability(track, ang.view, info.get("sex_model", "male"),
                           tuple(info["frame_size"]) if "frame_size" in info else None,
                           bar=exercise in BAR_EXERCISES)
    labels = stance_labels(st.contacts)
    fl = st.floor

    with VideoReader(video_path) as reader:
        size = (reader.info.width, reader.info.height)
        # results may come from a resized copy of the video: draw at that size
        target = tuple(info["frame_size"]) if "frame_size" in info else size
        resize = target != size
        size = target
        trail: deque[np.ndarray] = deque(maxlen=max(1, int(trail_seconds * reader.info.fps)))
        with VideoWriter(out_path, reader.info.fps, size) as writer:
            for i, t, img in reader.frames():
                if i >= track.n:
                    break
                if resize:
                    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                com = st.com_px[i]
                trail.append(com)
                if np.isfinite(fl.y0[i]):
                    draw_floor(img, fl.slope, fl.y0[i])
                lines = [f"t={t:5.2f}s  {labels[i]}"]
                colors = [(255, 255, 255)]
                if track.detected[i]:
                    sk = Skeleton(track.image[i], track.world[i], track.visibility[i])
                    draw_skeleton(img, sk)
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
