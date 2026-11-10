"""Compare pose backends on the same clips (E5).

Layout: outputs/<clip>/ holds the camera layer (camera.csv, wall.jpg) of the clip;
outputs/_bench/<clip>/<variant>/ holds one pose run per variant (climbvision pose ...).

    python tools/compare_pose.py metrics  outputs/_bench   -> compare.json + table
    python tools/compare_pose.py sheets   outputs/_bench   -> blind side-by-side crops

Automatic measures (no ground truth needed):
  detected      frames with a pose
  ms_frame      pose time per frame on this computer
  on_body_*     % of hand / foot points that land on the climber's silhouette (the
                frame minus the climber-free wall image, in wall coordinates)
  swaps_min     left/right swaps of the feet per minute (the two feet exchange places
                from one frame to the next)
  jitter_feet   median frame-to-frame wobble of a foot while it is on a hold (% body)
  wall layer    feet / hands on holds while climbing, holds, moves
"""

from __future__ import annotations

import json
import random
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np

from climbvision.engine.camera import LOST, CameraTrack, WallImage, apply_h
from climbvision.engine.signals import Clip
from climbvision.engine.skeleton import BONES
from climbvision.engine.track import Track
from climbvision.engine.video import VideoReader
from climbvision.engine.wall import LIMBS, limb_points, moves

warnings.simplefilter("ignore")
VARIANTS = ["mp_full", "mp_heavy", "rtm_m", "rtm_x", "rtmw_x"]


def load(bench: Path, clip: str, variant: str):
    d = bench / clip / variant
    main = bench.parent / clip
    info = json.loads((d / "summary.json").read_text())
    tr = Track.from_csv(d / "landmarks.csv")
    cam = CameraTrack.load(main / "camera.csv")
    rows = np.clip(np.searchsorted(cam.time_s, tr.time_s - 1e-3), 0, cam.n - 1)
    sub = CameraTrack(cam.time_s[rows], cam.H[rows], cam.status[rows], cam.inliers[rows],
                      cam.frame_size, cam.static, cam.info)
    wall = WallImage.load(main)
    clip_ = Clip.build(tr, frame_size=tuple(info["frame_size"]), camera=sub, wall_image=wall)
    return clip_, info


def foreground_masks(video: Path, clip: Clip, every: int = 10) -> dict[int, np.ndarray]:
    """Climber silhouette (in wall-image pixels) for every `every`-th pose frame."""
    wall = clip.wall_image
    bg = cv2.cvtColor(wall.image, cv2.COLOR_BGR2LAB).astype(np.int16)
    seen = wall.image.max(axis=2) > 0
    t = clip.t
    want = {int(round(ti * 1000)): k for k, ti in enumerate(t) if k % every == 0}
    out = {}
    size = wall.image.shape[1::-1]
    with VideoReader(video) as r:
        for _i, ti, frame in r.frames():
            k = want.get(int(round(ti * 1000)))
            if k is None or clip.camera.status[k] == LOST:
                continue
            M = wall.T @ clip.camera.H[k]
            w = cv2.warpPerspective(frame, M, size)
            cover = cv2.warpPerspective(np.full(frame.shape[:2], 255, np.uint8), M, size) > 0
            lab = cv2.cvtColor(w, cv2.COLOR_BGR2LAB).astype(np.int16)
            d = np.linalg.norm(lab - bg, axis=2)
            fg = ((d > 30) & cover & seen).astype(np.uint8)
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            fg = cv2.dilate(fg, np.ones((7, 7), np.uint8))
            out[k] = fg > 0
    return out


def metrics(clip: Clip, info: dict, fg: dict[int, np.ndarray]) -> dict:
    ws = clip.wall_support
    t = clip.t
    body = clip.stability.scale.body_px
    pts = limb_points(clip.wall)
    T = clip.wall_image.T
    on = {"hands": [], "feet": []}
    for k, mask in fg.items():
        for limb in LIMBS:
            p = pts[limb][k]
            if not np.isfinite(p).all():
                continue
            x, y = apply_h(T, p)
            inside = 0 <= int(y) < mask.shape[0] and 0 <= int(x) < mask.shape[1] and \
                mask[int(y), int(x)]
            on["hands" if "hand" in limb else "feet"].append(bool(inside))
    # left/right swaps of the feet
    lf, rf = pts["l_foot"], pts["r_foot"]
    swaps = 0
    for i in range(1, len(t)):
        a, b, c, d = lf[i - 1], rf[i - 1], lf[i], rf[i]
        if not np.isfinite(np.r_[a, b, c, d]).all() or np.linalg.norm(a - b) < 0.15 * body:
            continue
        same = np.linalg.norm(c - a) + np.linalg.norm(d - b)
        cross = np.linalg.norm(c - b) + np.linalg.norm(d - a)
        if cross < 0.5 * same:
            swaps += 1
    jit = []
    for f in ("l_foot", "r_foot"):
        p = pts[f]
        acc = np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=1) / body
        m = ws.on_hold[f][1:-1] & np.isfinite(acc)
        jit += list(acc[m])
    onw = ws.on_wall
    mv = moves(ws, t)
    dur = (t[-1] - t[0]) / 60
    return {
        "detected": round(100 * float(clip.track.detected.mean()), 1),
        "ms_frame": info.get("pose_ms_per_frame"),
        "on_body_hands": round(100 * float(np.mean(on["hands"])), 1) if on["hands"] else None,
        "on_body_feet": round(100 * float(np.mean(on["feet"])), 1) if on["feet"] else None,
        "swaps_min": round(swaps / max(dur, 1e-9), 1),
        "jitter_feet_pct": round(100 * float(np.median(jit)), 2) if jit else None,
        "on_wall_s": round(float(np.gradient(t)[onw].sum()), 1),
        "feet_on_holds": round(100 * float(np.mean([ws.on_hold[f][onw].mean()
                                                    for f in ("l_foot", "r_foot")])), 1)
        if onw.any() else None,
        "hands_on_holds": round(100 * float(np.mean([ws.on_hold[h][onw].mean()
                                                     for h in ("l_hand", "r_hand")])), 1)
        if onw.any() else None,
        "holds": len(ws.holds),
        "moves_hand": sum(m["limb"].endswith("hand") for m in mv),
        "moves_foot": sum(m["limb"].endswith("foot") for m in mv),
    }


def run_metrics(bench: Path) -> dict:
    res: dict = {}
    for cdir in sorted(p for p in bench.iterdir() if p.is_dir()):
        clip = cdir.name
        variants = [v for v in VARIANTS if (cdir / v / "summary.json").exists()]
        if not variants:
            continue
        loaded = {v: load(bench, clip, v) for v in variants}
        video = Path(loaded[variants[0]][1]["video"])
        fg = foreground_masks(video, loaded[variants[0]][0])
        res[clip] = {v: metrics(c, info, fg) for v, (c, info) in loaded.items()}
        print(clip, json.dumps(res[clip]))
    (bench / "compare.json").write_text(json.dumps(res, indent=2))
    return res


COL = {"l_hand": (255, 90, 160), "r_hand": (255, 170, 0), "l_foot": (0, 200, 255),
       "r_foot": (60, 220, 60)}


def sheets(bench: Path, per_clip: int = 6, seed: int = 0) -> None:
    """Blind comparison: random on-wall frames; for each, one tile per variant in random
    order (A, B, ...). The key goes to sheets_key.json."""
    rng = random.Random(seed)
    key = {}
    rows = []
    for cdir in sorted(p for p in bench.iterdir() if p.is_dir()):
        clip = cdir.name
        variants = [v for v in VARIANTS if (cdir / v / "summary.json").exists()]
        if len(variants) < 2:
            continue
        loaded = {v: load(bench, clip, v) for v in variants}
        ref = loaded[variants[0]][0]
        onw = ref.wall_support.on_wall & ref.track.detected
        cand = list(np.flatnonzero(onw))
        picks = sorted(rng.sample(cand, min(per_clip, len(cand))))
        video = Path(loaded[variants[0]][1]["video"])
        frames = {}
        want = {int(round(ref.t[k] * 1000)): k for k in picks}
        with VideoReader(video) as r:
            for _i, ti, frame in r.frames():
                k = want.get(int(round(ti * 1000)))
                if k is not None:
                    frames[k] = frame
        for k in picks:
            frame = frames[k]
            allp = np.vstack([c.track.image[k] for c, _ in loaded.values()])
            allp = allp[np.isfinite(allp).all(axis=1)]
            x0, y0 = allp.min(axis=0)
            x1, y1 = allp.max(axis=0)
            m = 0.15 * max(x1 - x0, y1 - y0)
            x0, y0 = int(max(0, x0 - m)), int(max(0, y0 - m))
            x1, y1 = int(min(frame.shape[1], x1 + m)), int(min(frame.shape[0], y1 + m))
            order = variants[:]
            rng.shuffle(order)
            tiles = []
            for li, v in enumerate(order):
                c = loaded[v][0]
                im = frame.copy()
                p = c.track.image[k]
                s = max(1, int((y1 - y0) / 300))
                for a, b in BONES:
                    if np.isfinite(p[[a, b]]).all():
                        cv2.line(im, tuple(int(q) for q in p[a]), tuple(int(q) for q in p[b]),
                                 (255, 255, 255), s, cv2.LINE_AA)
                lp = limb_points(p[None])
                for limb, col in COL.items():
                    q = lp[limb][0]
                    if np.isfinite(q).all():
                        cv2.circle(im, (int(q[0]), int(q[1])), 4 * s, col, -1, cv2.LINE_AA)
                tile = cv2.resize(im[y0:y1, x0:x1], (300, int(300 * (y1 - y0) / max(1, x1 - x0))))
                tile = cv2.resize(tile, (300, 420))
                cv2.putText(tile, f"{clip[:6]} f{k} {'ABCDE'[li]}", (4, 16), 0, 0.5,
                            (0, 255, 255), 1, cv2.LINE_AA)
                tiles.append(tile)
            key[f"{clip} f{k}"] = dict(zip("ABCDE", order, strict=False))
            rows.append(np.hstack(tiles))
    out = bench / "sheets"
    out.mkdir(exist_ok=True)
    for n in range(0, len(rows), 3):
        cv2.imwrite(str(out / f"sheet_{n // 3:02d}.jpg"), np.vstack(rows[n:n + 3]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    (bench / "sheets_key.json").write_text(json.dumps(key, indent=1))


if __name__ == "__main__":
    cmd, bench = sys.argv[1], Path(sys.argv[2])
    if cmd == "metrics":
        run_metrics(bench)
    elif cmd == "sheets":
        sheets(bench)
