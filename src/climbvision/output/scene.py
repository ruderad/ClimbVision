"""What the annotated video and the viewer draw: layers of simple shapes, per frame.

The engine's results become *drawing instructions* here, once; two renderers draw the
same instructions — `render.py` into an MP4 (OpenCV) and `viewer.py` in the browser
(canvas), so the exported video and the viewer always show the same thing.

**Layers** (each can be switched on or off):

- skeleton: bones, spine, joints (grey = hidden / guessed)
- arms: holding arms orange when bent (< 150 deg)
- contacts: limbs on holds (rings), smears (dashed rings); floor: feet on the floor,
  hands on a bar
- support: base of support — hull of the limbs on holds + the COM's plumb line (wall);
  floor line + support span (floor)
- com: the centre of mass — green inside the base of support, red outside, pink none
- trail: the COM's path over the last 2 s, coloured the same way
- forces: load on each hand and foot — a disc (size and colour = % body weight) and an
  arrow (the force the hold gives the body)
- velocity: the COM's velocity — an arrow (where it will be in 0.25 s) and its speed
- info: text — time, what is on holds, the latest move, COM over the feet
- timeline: contacts per limb, moves, rests and feet cuts along the clip (below the video)

**Presets** pick layers for a purpose: `clean`, `balance` (the classic annotated
video), `load`, `dynamics`, `all`.

Shapes, in image pixels of the frame as analysed; widths and sizes are in *units* of
1/1000 of the frame's longer side, so they look the same at any resolution:

    ["l", x0, y0, x1, y1, colour, width, dashed]       line
    ["p", [x0, y0, x1, y1, ...], colour, width]         polyline (open)
    ["g", [x0, y0, ...], colour, width, fill_alpha]     polygon (closed, filled)
    ["c", x, y, r, colour, width, fill]                 circle (fill: 0 ring, 1 disc)
    ["d", x, y, r, colour, width]                       dashed ring
    ["a", x0, y0, x1, y1, colour, width]                arrow
    ["t", x, y, text, colour, size]                     text (with a dark backing)
    ["T", line, text, colour]                           a line of the info box (top left)
    ["q", [x0, y0, x1, y1, ...], [colour, ...], width]  polyline, one colour per segment

Colours are "#rrggbb" (alpha in the shape where it applies).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from climbvision.engine.skeleton import BONES, J, spine_curve

LAYERS = ("skeleton", "arms", "contacts", "support", "com", "trail", "forces", "velocity",
          "info", "timeline")
PRESETS = {
    "clean": ("skeleton",),
    "balance": ("skeleton", "arms", "contacts", "support", "com", "trail", "info"),
    "load": ("skeleton", "contacts", "forces", "timeline"),
    "dynamics": ("skeleton", "com", "trail", "velocity", "timeline"),
    "all": LAYERS,
}
DEFAULT_PRESET = "balance"

WHITE, GREY, ORANGE = "#e6e6e6", "#5a5a5a", "#ff8c00"
JOINT = "#ffc800"
COM, TRAIL = "#ff3cc8", "#ff96e6"
INSIDE, OUTSIDE = "#3cd25a", "#f03232"
CONTACT, SMEAR, BASE, GRIP, FLOOR = "#00dcff", "#ebebeb", "#ffe600", "#a05aff", "#c8c8c8"
VEL = "#5ad2ff"
LIMB_COLOURS = {"l_hand": "#b07aff", "r_hand": "#3c9bff", "l_foot": "#ffb000",
                "r_foot": "#3cd25a"}


def load_colour(pct: float) -> str:
    """% body weight -> green (light) .. yellow (half) .. red (all of it and more)."""
    x = float(np.clip(pct / 100.0, 0, 1))
    stops = [(0.0, (60, 210, 90)), (0.5, (255, 220, 0)), (1.0, (240, 50, 50))]
    for (a, ca), (b, cb) in zip(stops, stops[1:], strict=False):
        if x <= b:
            f = (x - a) / (b - a)
            c = [round(ca[k] + f * (cb[k] - ca[k])) for k in range(3)]
            return "#{:02x}{:02x}{:02x}".format(*c)
    return "#f03232"


def resolve_layers(layers: str | list[str] | None = None, preset: str | None = None
                   ) -> list[str]:
    """`--layers a,b` and/or `--preset name` -> layer names (unknown names raise)."""
    out: list[str] = []
    if preset:
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset!r} (known: {', '.join(PRESETS)})")
        out += PRESETS[preset]
    if layers:
        names = layers.split(",") if isinstance(layers, str) else list(layers)
        for n in (n.strip() for n in names):
            if n not in LAYERS:
                raise ValueError(f"unknown layer {n!r} (known: {', '.join(LAYERS)})")
            out.append(n)
    if not out:
        out = list(PRESETS[DEFAULT_PRESET])
    return [n for n in LAYERS if n in out]


@dataclass
class Scene:
    size: tuple[int, int]  # (width, height) of the frames drawn on
    fps: float  # of the drawn frames (video fps / frame step)
    step: int
    t: list[float]
    frames: dict[str, list[list]]  # layer -> per frame: shapes
    timeline: dict  # rows, spans, moves (see build_timeline)
    wall: bool
    name: str = ""
    layer_help: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"size": self.size, "fps": self.fps, "step": self.step,
                           "t": self.t, "frames": self.frames, "timeline": self.timeline,
                           "wall": self.wall, "name": self.name, "layers": list(LAYERS),
                           "presets": {k: list(v) for k, v in PRESETS.items()},
                           "default_preset": DEFAULT_PRESET},
                          separators=(",", ":"))


def _r(v: float) -> float:
    return round(float(v), 1)


def _ok(*ps) -> bool:
    return all(p is not None and np.isfinite(np.asarray(p, float)).all() for p in ps)


class _ToImage:
    """Wall coordinates -> image of frame i (inverse of the camera layer's homography)."""

    def __init__(self, clip):
        self.clip = clip
        self._inv: dict[int, np.ndarray | None] = {}

    def __call__(self, i: int, pts) -> np.ndarray | None:
        pts = np.asarray(pts, float).reshape(-1, 2)
        cam = self.clip.camera
        if cam is None:
            return pts
        from climbvision.engine.camera import LOST, apply_h

        if i >= cam.n or cam.status[i] == LOST:
            return None
        if i not in self._inv:
            self._inv[i] = np.linalg.inv(cam.H[i])
        return apply_h(self._inv[i], pts)


def build_scene(result_dir: str | Path, trail_seconds: float = 2.0,
                force_scale_body: float = 0.45, velocity_ahead_s: float = 0.25) -> Scene:
    """All layers for a results folder. `force_scale_body`: arrow length for a force of
    100 % body weight, in body lengths; `velocity_ahead_s`: the velocity arrow shows where
    the COM would be after this long at its current speed."""
    from climbvision.analysis import clip_info, load_clip, resolve_recipe

    result_dir = Path(result_dir)
    info = clip_info(result_dir)
    mpath = result_dir / "metrics.json"
    recipe = json.loads(mpath.read_text()).get("exercise", "auto") if mpath.exists() else "auto"
    try:
        rec = resolve_recipe(result_dir, recipe)
    except ValueError:
        rec = resolve_recipe(result_dir, "auto")
    clip = load_clip(result_dir, bar=rec.bar)
    wall = rec.scene == "wall"
    track = clip.track
    n = track.n
    t = [round(float(x), 4) for x in clip.t[:n]]
    size = tuple(info.get("frame_size") or (0, 0))
    step = int(info.get("frame_step", 1))
    fps = float(info.get("fps", 30.0)) / step
    F: dict[str, list[list]] = {name: [[] for _ in range(n)] for name in LAYERS
                               if name != "timeline"}
    to_img = _ToImage(clip)

    # skeleton (every scene)
    img = track.image
    vis = track.visibility
    for i in range(n):
        if not track.detected[i]:
            continue
        p = img[i]
        out = F["skeleton"][i]
        for a, b in BONES:
            if _ok(p[a], p[b]):
                c = WHITE if min(vis[i, a], vis[i, b]) >= 0.5 else GREY
                out.append(["l", _r(p[a][0]), _r(p[a][1]), _r(p[b][0]), _r(p[b][1]), c, 2, 0])
        sp = spine_curve(p)
        if sp is not None:
            trunk = [J.L_HIP, J.R_HIP, J.L_SHOULDER, J.R_SHOULDER]
            c = WHITE if vis[i, trunk].min() >= 0.5 else GREY
            out.append(["p", [_r(v) for v in sp.ravel()], c, 3])
            head = p[J.HEAD]
            if _ok(head):
                top = sp[-1]
                nk = top + (head - top) * 0.45
                out.append(["l", _r(top[0]), _r(top[1]), _r(nk[0]), _r(nk[1]), c, 2, 0])
        for j in range(len(p)):
            if _ok(p[j]):
                out.append(["c", _r(p[j][0]), _r(p[j][1]), 3,
                            JOINT if vis[i, j] >= 0.5 else GREY, 0, 1])

    if wall:
        _wall_layers(clip, F, to_img, trail_seconds, fps, force_scale_body,
                     velocity_ahead_s)
    else:
        _floor_layers(clip, F, trail_seconds, fps, size[0] or 4000)
    timeline = build_timeline(clip, wall)
    return Scene(size=size, fps=fps, step=step, t=t, frames=F, timeline=timeline, wall=wall,
                 name=result_dir.name)


def _wall_layers(clip, F, to_img, trail_seconds, fps, force_scale_body, ahead_s):
    from climbvision.engine.biomech import angle_at
    from climbvision.engine.wall import LIMBS

    ws = clip.wall_support
    track = clip.track
    n = track.n
    im = track.image
    _ = clip.technique  # adds technique to the move labels
    route = clip.route
    fz = clip.forces
    on_smear = ws.on_smear or {limb: np.zeros(n, bool) for limb in LIMBS}
    balance = [None if not np.isfinite(v) else bool(v > 0) for v in ws.polygon_margin_cm]
    keep = max(2, int(trail_seconds * fps))

    # local wall-px Jacobian of metres, for arrows given in metres (x right, y up)
    def m_to_px(p_px: np.ndarray, d_m: np.ndarray) -> np.ndarray:
        e = 1.0
        base = ws.to_m(p_px)
        jx = (ws.to_m(p_px + [e, 0]) - base) / e
        jy = (ws.to_m(p_px + [0, e]) - base) / e
        Jm = np.array([jx, jy]).T  # d(metres)/d(px)
        try:
            return np.linalg.solve(Jm, d_m)
        except np.linalg.LinAlgError:
            return np.full(2, np.nan)

    bent = {}
    for side, (sh, el, wr), limb in (("l", (J.L_SHOULDER, J.L_ELBOW, J.L_WRIST), "l_hand"),
                                     ("r", (J.R_SHOULDER, J.R_ELBOW, J.R_WRIST), "r_hand")):
        ang = angle_at(im[:, sh], im[:, el], im[:, wr])
        bent[side] = (ws.on_hold[limb] & np.nan_to_num(ang < 150.0, nan=False),
                      (sh, el, wr, J.L_HAND if side == "l" else J.R_HAND))

    for i in range(n):
        if not track.detected[i]:
            F["info"][i].append(["T", 0, f"t={clip.t[i]:5.2f}s  no pose", WHITE])
            continue
        p = im[i]
        # arms
        for _side, (b, chain) in bent.items():
            if b[i]:
                for a, c in zip(chain, chain[1:], strict=False):
                    if _ok(p[a], p[c]):
                        F["arms"][i].append(["l", _r(p[a][0]), _r(p[a][1]), _r(p[c][0]),
                                             _r(p[c][1]), ORANGE, 3, 0])
        # contacts and the base of support (wall px -> this frame's image)
        held = [limb for limb in LIMBS if ws.on_hold[limb][i]]
        smear = [limb for limb in LIMBS if on_smear[limb][i]]
        pts_w = {limb: ws.points[limb][i] for limb in held + smear}
        pts_i = {}
        for limb, q in pts_w.items():
            r = to_img(i, q)
            if r is not None and _ok(r[0]):
                pts_i[limb] = r[0]
        for limb in held:
            if limb in pts_i:
                q = pts_i[limb]
                F["contacts"][i].append(["c", _r(q[0]), _r(q[1]), 10, CONTACT, 3, 0])
        for limb in smear:
            if limb in pts_i:
                q = pts_i[limb]
                F["contacts"][i].append(["d", _r(q[0]), _r(q[1]), 10, SMEAR, 3])
        sup = [pts_i[limb] for limb in held + smear if limb in pts_i]
        if len(sup) >= 3:
            import cv2

            hull = cv2.convexHull(np.array(sup, np.float32)).reshape(-1, 2)
            F["support"][i].append(["g", [_r(v) for v in hull.ravel()], BASE, 3, 0.22])
        elif len(sup) == 2:
            F["support"][i].append(["l", _r(sup[0][0]), _r(sup[0][1]), _r(sup[1][0]),
                                    _r(sup[1][1]), BASE, 3, 0])
        c_w = ws.com[i]
        lines = [(f"t={clip.t[i]:5.2f}s  {ws.labels[i]}", WHITE)]
        if route.labels[i]:
            lines.append((route.labels[i], "#78e6ff"))
        all_w = [ws.points[limb][i] for limb in held + smear]
        if all_w and ws.on_wall[i] and _ok(c_w):
            low = max(max(float(q[1]) for q in all_w), c_w[1] + 0.25 * ws.body_px)
            ends = to_img(i, [c_w, [c_w[0], low]])
            m = ws.com_over_feet_cm[i]
            if ends is not None and _ok(ends):
                col = WHITE if not np.isfinite(m) else (INSIDE if m > 0 else OUTSIDE)
                F["support"][i].append(["l", _r(ends[0][0]), _r(ends[0][1]), _r(ends[1][0]),
                                        _r(ends[1][1]), col, 2, 1])
            if np.isfinite(m):
                lines.append((f"COM over feet {m:+.0f} cm", INSIDE if m > 0 else OUTSIDE))
        for k, (text, col) in enumerate(lines):
            F["info"][i].append(["T", k, text, col])
        # COM and its trail (kept in wall coordinates: it stays put when the camera pans)
        c_i = to_img(i, c_w) if _ok(c_w) else None
        if c_i is not None and _ok(c_i[0]):
            col = COM if balance[i] is None else (INSIDE if balance[i] else OUTSIDE)
            F["com"][i].append(["c", _r(c_i[0][0]), _r(c_i[0][1]), 8, col, 0, 1])
            F["com"][i].append(["c", _r(c_i[0][0]), _r(c_i[0][1]), 10, "#ffffff", 1, 0])
        a0 = max(0, i - keep)
        past = ws.com[a0:i + 1]
        tr = to_img(i, past)
        if tr is not None:
            _trail(F["trail"][i], tr, balance[a0:i + 1])
        # forces on the holds: disc + arrow (the force the hold gives the body)
        for limb in LIMBS:
            ld = fz.load[limb][i] if fz is not None else np.nan
            if not np.isfinite(ld) or ld < 1 or limb not in pts_i:
                continue
            q = pts_i[limb]
            col = load_colour(ld)
            r = 8 + 22 * np.sqrt(min(ld, 150) / 100)
            F["forces"][i].append(["c", _r(q[0]), _r(q[1]), _r(r), col, 0, 0.55])
            F["forces"][i].append(["c", _r(q[0]), _r(q[1]), _r(r), col, 2, 0])
            f = fz.force[limb][i]
            if _ok(f):
                d_px = m_to_px(ws.points[limb][i],
                               np.asarray(f, float) * force_scale_body * ws.body_m)
                tip_w = ws.points[limb][i] + d_px
                tip = to_img(i, tip_w)
                if tip is not None and _ok(tip[0]):
                    F["forces"][i].append(["a", _r(q[0]), _r(q[1]), _r(tip[0][0]),
                                           _r(tip[0][1]), col, 3])
            F["forces"][i].append(["t", _r(q[0] + r * 0.9), _r(q[1] - r * 0.9),
                                   f"{ld:.0f}%", col, 0.8])
        # velocity of the COM
        if ws.com_v is not None and _ok(ws.com_v[i], c_w) and c_i is not None:
            v = ws.com_v[i]
            spd = float(np.linalg.norm(v))
            d_px = m_to_px(c_w, v * ahead_s)
            tip = to_img(i, c_w + d_px) if _ok(d_px) else None
            if tip is not None and _ok(tip[0]) and spd > 0.05:
                F["velocity"][i].append(["a", _r(c_i[0][0]), _r(c_i[0][1]), _r(tip[0][0]),
                                         _r(tip[0][1]), VEL, 4])
            F["velocity"][i].append(["t", _r(c_i[0][0] + 14), _r(c_i[0][1] + 22),
                                     f"{spd:.1f} m/s", VEL, 0.8])


def _floor_layers(clip, F, trail_seconds, fps, w):
    st = clip.stability
    track = clip.track
    n = track.n
    fl = st.floor
    balance = [None if not np.isfinite(v) else bool(v > 0) for v in st.margin_cm]
    keep = max(2, int(trail_seconds * fps))
    labels = st.labels
    for i in range(n):
        if np.isfinite(fl.y0[i]):
            F["support"][i].append(["l", 0, _r(fl.y0[i]), w, _r(fl.slope * w + fl.y0[i]),
                                    FLOOR, 1, 0])
        lines = [(f"t={clip.t[i]:5.2f}s  {labels[i]}", WHITE)]
        if not track.detected[i]:
            lines[0] = (f"t={clip.t[i]:5.2f}s  no pose", WHITE)
            for k, (text, col) in enumerate(lines):
                F["info"][i].append(["T", k, text, col])
            continue
        com = st.com_px[i]
        m = st.margin_cm[i]
        if np.isfinite(m) and _ok(com):
            col = INSIDE if m > 0 else OUTSIDE
            lo, hi = st.support_lo_px[i], st.support_hi_px[i]
            F["support"][i].append(["l", _r(lo), _r(fl.slope * lo + fl.y0[i]), _r(hi),
                                    _r(fl.slope * hi + fl.y0[i]), col, 8, 0])
            yf = fl.slope * com[0] + fl.y0[i]
            F["support"][i].append(["l", _r(com[0]), _r(com[1]), _r(com[0]), _r(yf), col, 2, 1])
            lines.append((f"margin {m:+.1f} cm", col))
        for q in track.image[i, st.contacts.point[i]]:
            if _ok(q):
                F["contacts"][i].append(["c", _r(q[0]), _r(q[1]), 10, CONTACT, 3, 0])
        grips = [st.contacts.hand_px[sd][i] for sd, g in st.contacts.grip.items() if g[i]]
        for q in grips:
            if _ok(q):
                F["contacts"][i].append(["c", _r(q[0]), _r(q[1]), 10, GRIP, 3, 0])
        if grips and np.isfinite(st.swing_cm[i]):
            lines.append((f"swing {st.swing_cm[i]:+.1f} cm", "#ffa0d2"))
        if st.feet_out_of_frame[i]:
            lines.append(("feet out of frame", "#ffc800"))
        for k, (text, col) in enumerate(lines):
            F["info"][i].append(["T", k, text, col])
        if _ok(com):
            b = balance[i]
            col = COM if b is None else (INSIDE if b else OUTSIDE)
            F["com"][i].append(["c", _r(com[0]), _r(com[1]), 8, col, 0, 1])
            F["com"][i].append(["c", _r(com[0]), _r(com[1]), 10, "#ffffff", 1, 0])
        a0 = max(0, i - keep)
        _trail(F["trail"][i], st.com_px[a0:i + 1], balance[a0:i + 1])


def _trail(out: list, pts: np.ndarray, balance: list) -> None:
    """The COM's recent path as coloured polylines (older = dimmer), split at gaps."""
    m_ = len(pts)
    xy: list[float] = []
    cols: list[str] = []
    for k in range(m_):
        if not _ok(pts[k]):
            if len(cols):
                out.append(["q", xy, cols, 2.5])
            xy, cols = [], []
            continue
        if xy:
            b = balance[k]
            col = TRAIL if b is None else (INSIDE if b else OUTSIDE)
            cols.append(_fade(col, 0.35 + 0.65 * k / m_))
        xy += [_r(pts[k][0]), _r(pts[k][1])]
    if cols:
        out.append(["q", xy, cols, 2.5])


def _fade(col: str, f: float) -> str:
    c = [int(col[k:k + 2], 16) for k in (1, 3, 5)]
    return "#{:02x}{:02x}{:02x}".format(*(int(v * f) for v in c))


def build_timeline(clip, wall: bool) -> dict:
    """Rows of spans along the clip (seconds): one row per limb (on a hold / smear /
    ground), plus moves (a tick from release to catch, number), rests and feet cuts."""
    t = clip.t
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1 / 30
    end = float(t[-1]) + dt if len(t) else 0.0
    out: dict = {"duration": round(end, 3), "rows": [], "moves": [], "bands": []}
    if wall:
        from climbvision.engine.wall import LABEL, LIMBS

        ws = clip.wall_support
        for limb in LIMBS:
            spans = []
            for a in ws.anchors[limb]:
                kind = "ground" if a.ground else ("smear" if a.smear else "hold")
                spans.append([round(float(t[a.start]), 3), round(float(t[a.end - 1]) + dt, 3),
                              kind, int(a.hold)])
            out["rows"].append({"name": LABEL[limb], "limb": limb,
                                "colour": LIMB_COLOURS[limb], "spans": spans})
        route = clip.route
        for m in route.moves:
            out["moves"].append({"n": m.n, "limb": m.limb, "a": round(m.release_s, 3),
                                 "b": round(m.catch_s, 3), "tags": list(m.tags),
                                 "load": None if not np.isfinite(m.catch_load_pct_bw)
                                 else round(float(m.catch_load_pct_bw))})
        for a, b in route.rests:
            out["bands"].append([round(float(t[a]), 3), round(float(t[b - 1]) + dt, 3), "rest"])
        for a, b in route.feet_cut:
            out["bands"].append([round(float(t[a]), 3), round(float(t[b - 1]) + dt, 3),
                                 "feet cut"])
        fz = clip.forces
        if fz is not None:
            out["load"] = {"hands": [None if not np.isfinite(v) else round(float(v))
                                     for v in fz.hands_bw],
                           "feet": [None if not np.isfinite(v) else round(float(v))
                                    for v in fz.feet_bw]}
    else:
        st = clip.stability
        from climbvision.engine.events import runs

        for g, v in st.contacts.group.items():
            if not v.any():
                continue
            spans = [[round(float(t[a]), 3), round(float(t[min(b, len(t)) - 1]) + dt, 3),
                      "ground", -1] for a, b in runs(v)]
            out["rows"].append({"name": g.replace("_", " "), "limb": g,
                                "colour": "#3cd25a" if "foot" in g else "#ffb000",
                                "spans": spans})
    return out
