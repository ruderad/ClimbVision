"""The annotated video (MP4), drawn from the scene layers (`scene.py`).

`render(video, results, layers=..., preset=...)` draws the chosen layers on every frame;
the default is the `balance` preset (skeleton, bent arms, holds, base of support, COM
and its trail, info text). With the `timeline` layer a strip is added under the video:
one row per limb (on a hold / smear / ground), moves, rests, feet cuts, and a cursor.

The interactive viewer (`viewer.py`) draws the very same shapes in the browser.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from climbvision.output.scene import Scene, build_scene, resolve_layers

# drawing order: things on the wall first, the body, then what sits on top
DRAW_ORDER = ("support", "trail", "contacts", "effort", "forces", "skeleton", "arms", "angles",
              "arc", "swing", "com", "speed", "velocity", "info")
PHASE_BGR = {"push": (0, 140, 255), "reach": (255, 210, 90), "catch": (80, 80, 255)}


def _bgr(col: str) -> tuple[int, int, int]:
    return int(col[5:7], 16), int(col[3:5], 16), int(col[1:3], 16)


def _pt(x, y) -> tuple[int, int]:
    return int(round(x)), int(round(y))


def draw_shapes(img: np.ndarray, shapes: list, unit: float, info: list | None = None) -> None:
    """Draw scene shapes (see scene.py) on a BGR frame; `unit` = px per size unit."""
    for s in shapes:
        k = s[0]
        if k == "l":
            _, x0, y0, x1, y1, col, w, dashed = s
            if dashed:
                _dashed(img, (x0, y0), (x1, y1), _bgr(col), max(1, int(w * unit)), unit)
            else:
                cv2.line(img, _pt(x0, y0), _pt(x1, y1), _bgr(col), max(1, int(w * unit)),
                         cv2.LINE_AA)
        elif k == "p":
            _, xy, col, w = s
            pts = np.round(np.array(xy, float).reshape(-1, 2)).astype(np.int32)
            cv2.polylines(img, [pts], False, _bgr(col), max(1, int(w * unit)), cv2.LINE_AA)
        elif k == "q":
            _, xy, cols, w = s
            pts = np.array(xy, float).reshape(-1, 2)
            for j, col in enumerate(cols):
                cv2.line(img, _pt(*pts[j]), _pt(*pts[j + 1]), _bgr(col),
                         max(1, int(w * unit)), cv2.LINE_AA)
        elif k == "g":
            _, xy, col, w, alpha = s
            pts = np.round(np.array(xy, float).reshape(-1, 2)).astype(np.int32)
            if alpha > 0:
                over = img.copy()
                cv2.fillPoly(over, [pts], _bgr(col), cv2.LINE_AA)
                cv2.addWeighted(over, alpha, img, 1 - alpha, 0, img)
            cv2.polylines(img, [pts], True, _bgr(col), max(1, int(w * unit)), cv2.LINE_AA)
        elif k == "c":
            _, x, y, r, col, w, fill = s
            rr = max(2, int(r * unit))
            if fill:
                if fill >= 1:
                    cv2.circle(img, _pt(x, y), rr, _bgr(col), -1, cv2.LINE_AA)
                else:
                    over = img.copy()
                    cv2.circle(over, _pt(x, y), rr, _bgr(col), -1, cv2.LINE_AA)
                    cv2.addWeighted(over, fill, img, 1 - fill, 0, img)
            if w:
                cv2.circle(img, _pt(x, y), rr, _bgr(col), max(1, int(w * unit)), cv2.LINE_AA)
        elif k == "d":
            _, x, y, r, col, w = s
            rr = max(2, int(r * unit))
            for a in range(0, 360, 45):
                cv2.ellipse(img, _pt(x, y), (rr, rr), 0, a, a + 25, _bgr(col),
                            max(1, int(w * unit)), cv2.LINE_AA)
        elif k == "a":
            _, x0, y0, x1, y1, col, w = s
            ln = float(np.hypot(x1 - x0, y1 - y0))
            if ln >= 2:
                cv2.arrowedLine(img, _pt(x0, y0), _pt(x1, y1), _bgr(col),
                                max(1, int(w * unit)), cv2.LINE_AA,
                                tipLength=min(0.4, 18 * unit / ln))
        elif k == "t":
            _, x, y, text, col, size = s
            _text(img, text, _pt(x, y), _bgr(col), 0.8 * size * unit)
        elif k == "T" and info is not None:
            info.append(s)


def _dashed(img, a, b, col, w, unit) -> None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(2, int(np.linalg.norm(b - a) / max(6, 12 * unit)))
    for k in range(0, n, 2):
        p0, p1 = a + (b - a) * k / n, a + (b - a) * (k + 1) / n
        cv2.line(img, _pt(*p0), _pt(*p1), col, w, cv2.LINE_AA)


def _text(img, text, org, col, scale) -> None:
    font, thick = cv2.FONT_HERSHEY_SIMPLEX, max(1, int(scale * 2.2))
    (w, h), b = cv2.getTextSize(text, font, scale, thick)
    x, y = org
    pad = max(2, int(h * 0.25))
    over = img.copy()
    cv2.rectangle(over, (x - pad, y - h - pad), (x + w + pad, y + b + pad), (0, 0, 0), -1)
    cv2.addWeighted(over, 0.55, img, 0.45, 0, img)
    cv2.putText(img, text, (x, y), font, scale, col, thick, cv2.LINE_AA)


def draw_info(img: np.ndarray, lines: list, unit: float) -> None:
    """The info box: text lines in the top-left corner on a dark box."""
    if not lines:
        return
    lines = sorted(lines, key=lambda s: s[1])
    font, size, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.8 * unit, max(1, int(2 * unit))
    sizes = [cv2.getTextSize(s[2], font, size, thick) for s in lines]
    lh = int(max(h + b for (_, h), b in sizes) * 1.35)
    x, y0, pad = int(15 * unit) + 5, int(40 * unit) + 10, int(8 * unit)
    tw = max(w for (w, _), _ in sizes)
    th = sizes[0][0][1]
    cv2.rectangle(img, (x - pad, y0 - th - pad),
                  (x + tw + pad, y0 + lh * (len(lines) - 1) + pad + sizes[-1][1]), (0, 0, 0), -1)
    for k, s in enumerate(lines):
        cv2.putText(img, s[2], (x, y0 + k * lh), font, size, _bgr(s[3]), thick, cv2.LINE_AA)


SPAN_ALPHA = {"hold": 1.0, "smear": 0.45, "ground": 0.0}


def draw_timeline(width: int, timeline: dict, t: float, unit: float) -> np.ndarray:
    """The strip under the video: limb rows, moves, rests / feet cuts and a cursor."""
    rows = timeline["rows"]
    dur = max(timeline["duration"], 1e-6)
    row_h = max(10, int(20 * unit))
    label_w = max(40, int(110 * unit))
    top = max(8, int(14 * unit))
    phases = timeline.get("phases") or []
    n_rows = len(rows) + (1 if phases else 0)
    h = top + row_h * n_rows + max(6, int(12 * unit))
    band = np.full((h, width, 3), 18, np.uint8)
    x0, x1 = label_w, width - max(6, int(10 * unit))

    def X(s):
        return int(x0 + (x1 - x0) * float(s) / dur)

    for a, b, kind in timeline.get("bands", []):
        col = (60, 140, 60) if kind == "rest" else (50, 50, 150)
        cv2.rectangle(band, (X(a), top), (X(b), top + row_h * len(rows)), col, -1)
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.45 * unit * 1.6, max(1, int(1.2 * unit))
    for r, row in enumerate(rows):
        y = top + r * row_h
        col = _bgr(row["colour"])
        cv2.putText(band, row["name"], (int(6 * unit), y + int(row_h * 0.72)), font, fs,
                    (200, 200, 200), th, cv2.LINE_AA)
        for a, b, kind, _hold in row["spans"]:
            ya, yb = y + int(row_h * 0.18), y + int(row_h * 0.82)
            if kind == "ground":
                cv2.rectangle(band, (X(a), ya), (max(X(a) + 1, X(b)), yb), (90, 90, 90), -1)
                continue
            alpha = SPAN_ALPHA[kind]
            over = band.copy()
            cv2.rectangle(over, (X(a), ya), (max(X(a) + 1, X(b)), yb), col, -1)
            cv2.addWeighted(over, alpha, band, 1 - alpha, 0, band)
    if phases:
        y = top + len(rows) * row_h
        cv2.putText(band, "phase", (int(6 * unit), y + int(row_h * 0.72)), font, fs,
                    (200, 200, 200), th, cv2.LINE_AA)
        for a, b, name in phases:
            cv2.rectangle(band, (X(a), y + int(row_h * 0.25)),
                          (max(X(a) + 1, X(b)), y + int(row_h * 0.75)),
                          PHASE_BGR.get(name, (150, 150, 150)), -1)
    idx = {row["limb"]: k for k, row in enumerate(rows)}
    for m in timeline.get("moves", []):
        k = idx.get(m["limb"])
        if k is None:
            continue
        y = top + k * row_h + int(row_h * 0.5)
        hot = {"dyno", "jump"} & set(m["tags"])
        col = (80, 80, 255) if hot else (235, 235, 235)
        cv2.line(band, (X(m["a"]), y), (max(X(m["a"]) + 1, X(m["b"])), y), col,
                 max(1, int(2 * unit)), cv2.LINE_AA)
    cx = X(t)
    cv2.line(band, (cx, 2), (cx, h - 2), (255, 255, 255), max(1, int(2 * unit)), cv2.LINE_AA)
    return band


def render(video_path: str | Path, result_dir: str | Path, out_path: str | Path | None = None,
           layers: str | list[str] | None = None, preset: str | None = None,
           scene: Scene | None = None) -> Path:
    """Draw the chosen layers on the video (default: the `balance` preset)."""
    from climbvision.engine.video import VideoReader, VideoWriter, warn_if_changed

    result_dir = Path(result_dir)
    out_path = Path(out_path) if out_path else result_dir / "annotated.mp4"
    warn_if_changed(result_dir, video_path)
    chosen = resolve_layers(layers, preset)
    scene = scene or build_scene(result_dir)
    n = len(scene.t)
    with VideoReader(video_path) as reader:
        size = tuple(scene.size) if scene.size and scene.size[0] else (
            reader.info.width, reader.info.height)
        resize = size != (reader.info.width, reader.info.height)
        unit = max(size) / 1000.0
        band_h = 0
        if "timeline" in chosen and scene.timeline["rows"]:
            band_h = draw_timeline(size[0], scene.timeline, 0.0, unit).shape[0]
        with VideoWriter(out_path, scene.fps, (size[0], size[1] + band_h)) as writer:
            for i, (_, _t, img) in enumerate(reader.frames(step=scene.step)):
                if i >= n:
                    break
                if resize:
                    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                info: list = []
                for name in DRAW_ORDER:
                    if name in chosen:
                        draw_shapes(img, scene.frames[name][i], unit, info)
                draw_info(img, info, unit)
                if band_h:
                    img = np.vstack([img, draw_timeline(size[0], scene.timeline,
                                                        scene.t[i], unit)])
                writer.write(img)
    return out_path
