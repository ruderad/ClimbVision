"""route.png: the climb drawn on the wall image.

- holds numbered in the order they were first used: white ring = used by a hand,
  small green ring = by a foot only, both = double ring;
- every move as an arrow from the hold it left to the hold it reached, in the limb's
  colour (thick = hands, thin = feet), with the move number;
- dynos / jumps drawn with a white outline, matches with a small "m";
- the centre of mass path while on the wall (pink);
- start (first hand holds) and top (highest hand hold reached).

Without a wall image (no camera layer) the map is drawn on a grey background.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from climbvision.engine.camera import apply_h
from climbvision.engine.route import HANDS
from climbvision.engine.wall import LABEL

LIMB_COLOUR = {"l_hand": (255, 90, 160), "r_hand": (255, 170, 0),  # BGR, as in wall_view
               "l_foot": (0, 200, 255), "r_foot": (60, 220, 60)}
COM_COLOUR = (200, 60, 255)


def _arrow(img, a, b, col, w, r_end):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 1:
        return
    u = d / L
    a2, b2 = a + u * r_end, b - u * r_end  # stop at the hold rings
    if np.linalg.norm(b2 - a2) < 2:
        return
    cv2.arrowedLine(img, tuple(int(v) for v in a2), tuple(int(v) for v in b2), col, w,
                    cv2.LINE_AA, tipLength=min(0.35, 18 / max(L, 1)))


def draw_route_map(path: str | Path, clip, max_side: int = 1400) -> Path:
    path = Path(path)
    ws = clip.wall_support
    route = clip.route
    if clip.wall_image is not None:
        base = clip.wall_image.image
        T = clip.wall_image.T
    else:
        h, w = 1000, 1000
        base = np.full((h, w, 3), 60, np.uint8)
        T = np.eye(3)
    pts = [h.pos for h in ws.holds]
    com = ws.com[ws.on_wall]
    com = com[np.isfinite(com).all(1)]
    allp = np.array(pts + list(com)) if pts or len(com) else np.zeros((0, 2))
    if clip.wall_image is None and len(allp):  # fit the drawing into the grey canvas
        lo, hi = allp.min(0), allp.max(0)
        k = 900 / max(float((hi - lo).max()), 1.0)
        T = np.array([[k, 0, 50 - k * lo[0]], [0, k, 50 - k * lo[1]], [0, 0, 1]])
    # crop to what was used, with a margin
    H, W = base.shape[:2]
    x0, y0, x1, y1 = 0, 0, W, H
    if len(allp):
        q = apply_h(T, allp)
        m = 0.3 * ws.body_px * float(T[0, 0])
        x0, y0 = (max(0, int(v - m)) for v in q.min(0))
        x1, y1 = int(min(W, q[:, 0].max() + m)), int(min(H, q[:, 1].max() + m))
    img = base[y0:y1, x0:x1].astype(np.float32)
    grey = img.mean(axis=2, keepdims=True)
    img = (0.45 * (0.5 * img + 0.5 * grey)).astype(np.uint8)  # dim, desaturate
    z = min(1.0, max_side / max(img.shape[:2])) if max(img.shape[:2]) > max_side else \
        min(2.0, max_side / max(img.shape[:2]))
    img = cv2.resize(img, (int(img.shape[1] * z), int(img.shape[0] * z)),
                     interpolation=cv2.INTER_AREA)
    M = np.diag([z, z, 1.0]) @ np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]]) @ T

    def P(p):
        q = apply_h(M, np.asarray(p, float))
        return int(q[0]), int(q[1])

    s = max(img.shape[:2]) / 1000.0
    r = max(7, int(13 * s))
    font = cv2.FONT_HERSHEY_SIMPLEX
    # COM path
    if len(com) > 1:
        q = apply_h(M, com).astype(np.int32)
        cv2.polylines(img, [q], False, COM_COLOUR, max(1, int(1.5 * s)), cv2.LINE_AA)
    # holds
    for h in ws.holds:
        hand = bool(h.limbs & set(HANDS))
        c = P(h.pos)
        if hand:
            cv2.circle(img, c, r, (255, 255, 255), max(2, int(2.5 * s)), cv2.LINE_AA)
        if h.limbs - set(HANDS):
            cv2.circle(img, c, int(r * (0.6 if hand else 0.8)), (90, 220, 90),
                       max(1, int(2 * s)), cv2.LINE_AA)
        cv2.putText(img, str(h.id), (c[0] + r, c[1] - r // 2), font, 0.42 * s + 0.15,
                    (230, 230, 230), 1, cv2.LINE_AA)
    pos = {h.id: h.pos for h in ws.holds}
    # moves: feet first (thin), hands on top (thick)
    for m in sorted(route.moves, key=lambda m: m.kind == "hand"):
        hand = m.kind == "hand"
        w = max(2, int((4 if hand else 2) * s))
        a, b = P(pos[m.from_hold]), P(pos[m.to_hold])
        # left and right limbs side by side, so matched moves don't hide each other
        d = np.subtract(b, a, dtype=float)
        if np.linalg.norm(d) > 0:
            off = np.array([-d[1], d[0]]) / np.linalg.norm(d) * 0.35 * r * (
                -1 if m.limb.startswith("l_") else 1)
            a = (int(a[0] + off[0]), int(a[1] + off[1]))
            b = (int(b[0] + off[0]), int(b[1] + off[1]))
        if "dyno" in m.tags or "jump" in m.tags:
            _arrow(img, a, b, (255, 255, 255), w + max(2, int(2 * s)), r)
        _arrow(img, a, b, LIMB_COLOUR[m.limb], w, r)
        mid = ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)
        rr = max(6, int((9 if hand else 7) * s))
        cv2.circle(img, mid, rr, LIMB_COLOUR[m.limb], -1, cv2.LINE_AA)
        txt = str(m.n)
        sc = (0.33 if hand else 0.28) * s + 0.1
        (tw, th), _ = cv2.getTextSize(txt, font, sc, 1)
        cv2.putText(img, txt, (mid[0] - tw // 2, mid[1] + th // 2), font, sc, (0, 0, 0), 1,
                    cv2.LINE_AA)
        if "match" in m.tags:
            cv2.putText(img, "m", (b[0] - r, b[1] + 2 * r), font, 0.4 * s + 0.1,
                        LIMB_COLOUR[m.limb], 1, cv2.LINE_AA)
    # start and top
    hand_holds = [h for h in ws.holds if h.limbs & set(HANDS)]
    if hand_holds:
        first_t = min(min(a.start for a in h.anchors) for h in hand_holds)
        start = [h for h in hand_holds if min(a.start for a in h.anchors) <= first_t + 15]
        for h in start[:2]:
            _tag(img, "START", P(h.pos), r, s, below=True)
        top = min(hand_holds, key=lambda h: h.pos[1])
        _tag(img, "TOP", P(top.pos), r, s, below=False)
    img = np.vstack([img, _legend(img.shape[1], clip, s)])
    cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return path


def _tag(img, text, c, r, s, below):
    font = cv2.FONT_HERSHEY_SIMPLEX
    sc = 0.4 * s + 0.15
    (tw, th), _ = cv2.getTextSize(text, font, sc, 1)
    x = c[0] - tw // 2
    y = c[1] + r + th + 6 if below else c[1] - r - 6
    cv2.rectangle(img, (x - 3, y - th - 3), (x + tw + 3, y + 3), (255, 255, 255), -1)
    cv2.putText(img, text, (x, y), font, sc, (0, 0, 0), 1, cv2.LINE_AA)


def _legend(width, clip, s):
    """A strip under the map: counts, limb colours, tags."""
    route = clip.route
    hand = sum(m.kind == "hand" for m in route.moves)
    foot = len(route.moves) - hand
    tags: dict[str, int] = {}
    for m in route.moves:
        for g in m.tags:
            tags[g] = tags.get(g, 0) + 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    sc = 0.42 * s + 0.15
    lh = int(22 * s) + 8
    rows = [[(f"{len(clip.wall_support.holds)} holds, {hand} hand + {foot} foot moves, "
              f"{len(route.rests)} rest{'' if len(route.rests) == 1 else 's'}",
              (255, 255, 255))],
            [(LABEL[limb], col) for limb, col in LIMB_COLOUR.items()]
            + [("centre of mass", COM_COLOUR)],
            [("rings: white = hand hold, green = foot hold; thick arrows = hands, "
              "thin = feet; white edge = dyno", (200, 200, 200))]]
    if tags:
        rows.append([(", ".join(f"{k} {v}" for k, v in sorted(tags.items())),
                      (200, 200, 200))])
    strip = np.full((lh * len(rows) + 12, width, 3), 25, np.uint8)
    for k, row in enumerate(rows):
        x = 12
        for text, col in row:
            cv2.putText(strip, text, (x, lh * (k + 1)), font, sc, col, 1, cv2.LINE_AA)
            x += cv2.getTextSize(text, font, sc, 1)[0][0] + int(24 * s)
    return strip
