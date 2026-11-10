"""Contacts layer: which body parts are anchored, and on what.

An anchor is a hand/foot/etc. that is *still* (in the frame of what it rests on)
and *on* something:
  - floor contact: near the floor line and nearly still
  - grip: a hand above its shoulder, still, then held while it stays put (a bar)
  - (E4) hold: a hand/foot still on the wall for ~0.5 s

All thresholds are in body lengths (scaled to the person's current distance from
the camera), so they work at any resolution or distance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from climbvision.engine.events import clean_runs, runs
from climbvision.engine.scale import Scale
from climbvision.engine.skeleton import N_JOINTS, J, side_joint
from climbvision.engine.surfaces import FLOOR_PARTS, Floor
from climbvision.engine.track import Track, smooth

# footprint radius (m) around each contact point: half foot width, hand, forearm, knee
CONTACT_RADIUS_M = {"foot": 0.045, "hand": 0.04, "forearm": 0.04, "knee": 0.05}
HANDS = {"left": (J.L_WRIST, J.L_HAND), "right": (J.R_WRIST, J.R_HAND)}
FEET = (J.L_HEEL, J.L_TOE, J.R_HEEL, J.R_TOE)


def radius_m(part: str) -> float:
    return CONTACT_RADIUS_M[part.split("_", 1)[1]]


@dataclass
class Contacts:
    point: np.ndarray  # (n, N_JOINTS) bool: joint touching the floor
    group: dict[str, np.ndarray]  # part -> (n,) bool
    height_px: np.ndarray  # (n, N_JOINTS) smoothed height above the floor
    grip: dict[str, np.ndarray] = field(default_factory=dict)  # "left"/"right" -> (n,) bool
    hand_px: dict[str, np.ndarray] = field(default_factory=dict)  # side -> (n, 2)

    @property
    def n(self) -> int:
        return len(self.point)

    def on_floor(self) -> np.ndarray:
        out = np.zeros(self.n, bool)
        for g in self.group.values():
            out |= g
        return out

    def on_bar(self) -> np.ndarray:
        out = np.zeros(self.n, bool)
        for g in self.grip.values():
            out |= g
        return out


def body_px(scale: Scale) -> np.ndarray:
    """Body length in pixels per frame."""
    return scale.body_px * np.nan_to_num(scale.rel, nan=1.0)


def smoothed_points(track: Track, seconds: float = 0.1) -> np.ndarray:
    win = max(1, int(track.fps * seconds))
    return np.stack([np.stack([smooth(track.image[:, i, d], win) for d in (0, 1)], -1)
                     for i in range(N_JOINTS)], axis=1)


def detect_contacts(track: Track, floor: Floor, scale: Scale, height_tol: float = 0.05,
                    speed_tol: float = 0.6, min_s: float = 0.15) -> Contacts:
    """Floor contacts. height_tol in body lengths; speed_tol in body lengths per second."""
    fps = track.fps
    img = smoothed_points(track)
    h = floor.height(img)
    speed = np.linalg.norm(np.gradient(img, track.time_s, axis=0), axis=-1)
    body = body_px(scale)[:, None]
    near = np.nan_to_num(h, nan=np.inf) < height_tol * body
    still = np.nan_to_num(speed, nan=np.inf) < speed_tol * body
    point = near & still
    min_len = max(1, int(min_s * fps))
    groups = {}
    for name, joints in FLOOR_PARTS.items():
        g = point[:, list(joints)].any(axis=1)
        groups[name] = clean_runs(g, min_len)
        point[:, list(joints)] &= groups[name][:, None]
    # a contacting part with no individual point (filled gap) keeps its lowest point
    for name, joints in FLOOR_PARTS.items():
        gap = groups[name] & ~point[:, list(joints)].any(axis=1)
        if gap.any():
            hh = np.where(np.isfinite(h[:, list(joints)]), h[:, list(joints)], np.inf)
            j = np.array(joints)[np.argmin(hh, axis=1)]
            point[np.flatnonzero(gap), j[gap]] = True
    return Contacts(point, groups, h)


GRIP_FRACTION = 2 / 3  # a bar sits in the fingers: 2/3 of the way from wrist to knuckles


def hand_centre(track: Track, side: str, seconds: float = 0.1) -> np.ndarray:
    """Where a gripping hand holds: between WRIST and HAND (GRIP_FRACTION), or whichever
    of the two is known; smoothed."""
    w, h = (track.image[:, j] for j in HANDS[side])
    raw = np.where(np.isfinite(h), w + GRIP_FRACTION * (h - w), w)
    raw = np.where(np.isfinite(raw), raw, h)
    win = max(1, int(track.fps * seconds))
    return np.column_stack([smooth(raw[:, 0], win), smooth(raw[:, 1], win)])


def detect_grips(track: Track, scale: Scale, speed_tol: float = 0.35, min_s: float = 0.3,
                 drift: float = 0.08) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Hands holding something fixed (a bar), per side.

    Seed: the hand is above its shoulder and nearly still (speed_tol body lengths/s)
    for at least `min_s`. Each seed then grows forwards and backwards for as long as
    the hand stays within `drift` body lengths of where it was holding, which keeps the
    grip through the top of a pull-up (when the shoulders reach the hands) and through
    tracking jitter. Only meaningful when the clip is known to involve a bar.
    """
    n = track.n
    body = body_px(scale)
    min_len = max(1, int(min_s * track.fps))
    grips, pos = {}, {}
    for side in HANDS:
        c = hand_centre(track, side)
        speed = np.linalg.norm(np.gradient(c, track.time_s, axis=0), axis=-1)
        shoulder_y = track.image[:, side_joint(side, "SHOULDER"), 1]
        seed = (np.nan_to_num(c[:, 1], nan=np.inf) < np.nan_to_num(shoulder_y, nan=-np.inf)) \
            & (np.nan_to_num(speed, nan=np.inf) < speed_tol * body)
        seed = clean_runs(seed, min_len)
        mask = seed.copy()
        for a, b in runs(seed):
            anchor = np.nanmedian(c[a:b], axis=0)
            near = np.nan_to_num(np.linalg.norm(c - anchor, axis=1), nan=np.inf) < drift * body
            i = a - 1
            while i >= 0 and near[i] and not mask[i]:
                mask[i] = True
                i -= 1
            i = b
            while i < n and near[i] and not mask[i]:
                mask[i] = True
                i += 1
        grips[side], pos[side] = mask, c
    return grips, pos


_PARTS = (("foot", "feet"), ("hand", "hands"), ("forearm", "forearms"), ("knee", "knees"))


def stance_label(contacts: Contacts, i: int) -> str:
    """Short name for what's supporting the body, e.g. 'L foot', 'feet + forearms',
    'bar', 'both feet + bar'."""
    on = {g for g, m in contacts.group.items() if m[i]}
    gl = contacts.grip.get("left", np.zeros(i + 1, bool))[i]
    gr = contacts.grip.get("right", np.zeros(i + 1, bool))[i]
    bar = "bar" if gl and gr else "L hand on bar" if gl else "R hand on bar" if gr else ""
    if not on:
        return bar or "none"
    parts = []
    for part, plural in _PARTS:
        left, right = f"left_{part}" in on, f"right_{part}" in on
        if left and right:
            parts.append(plural)
        elif left or right:
            parts.append(f"{'L' if left else 'R'} {part}")
    label = " + ".join(parts)
    label = "both feet" if label == "feet" else label
    return f"{label} + {bar}" if bar else label


def stance_labels(contacts: Contacts) -> list[str]:
    return [stance_label(contacts, i) for i in range(contacts.n)]
