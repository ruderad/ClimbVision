"""Climb window: which part of the clip is the climb.

A clip usually holds more than the climb: walking up to the wall, standing below it,
the drop to the mat and, often, walking back to the camera or the camera turning to
the floor. Those frames must not shape the measurements: a person walking towards the
camera looks two or three times bigger than on the wall, and the scale (the body as
the ruler) then shrinks every height and distance on the wall by as much.

**The climb** runs from the moment both feet leave the ground to the touchdown on the
mat. In frames (per attempt):

- *start*: the first frame on the wall (the wall layer's `on_wall`: a foot on a hold or
  a hand on a hold above the shoulders, and no foot on the ground);
- *end*: the touchdown (the ground again), or, when the camera doesn't see it, the
  first sign that the climb is over:
    - the camera loses the wall for `break_s` or longer;
    - the pose is lost or unreliable (mean visibility < `min_visibility`);
    - the person's size changes beyond `size_range` x their size on the wall (walking
      towards or away from the camera, or the camera tracked someone else);
    - no limb on a hold for `max_air_s` (a dyno or a cut lasts well under a second;
      longer means the climber has left the wall).
  The end is then the last contact plus `tail_s` (the drop).
- A drop the camera sees but whose landing it doesn't (the floor out of view): a fall
  of the COM of `drop_bl` body lengths or more, faster than `drop_speed`, with no limb
  on a hold, ends the climb where the fall stops.

**Attempts.** A clip can hold several attempts (a fall, then a second go): each
stretch on the wall between touchdowns is one. The analysis measures one attempt, the
*main* one: the one with the largest rise of the centre of mass (the one that got
highest; the longer one on a tie). The others are listed in the metrics, with a
warning. Best practice is still one attempt per clip.

Frames outside the climb keep their video and pose, but count for nothing: not on the
wall, no holds, no part in the scale.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ClimbParams:
    break_s: float = 0.5  # a break (camera, pose, size) this long ends the climb
    min_visibility: float = 0.3  # mean joint visibility below this: pose unreliable
    size_range: tuple[float, float] = (0.55, 1.6)  # body size vs. its size on the wall
    max_air_s: float = 2.0  # no limb on a hold this long: the climber left the wall
    tail_s: float = 0.5  # after the last contact (the drop), when no touchdown is seen
    touchdown_s: float = 1.5  # a touchdown this soon after the last contact ends it there
    drop_bl: float = 0.6  # body lengths: the COM falling this far, fast (drop_speed
    drop_speed: float = 1.5  # body lengths/s), starting with no limb on a hold, is the
    #                          drop to the mat: the climb ends where the fall stops
    merge_s: float = 1.0  # two stretches on the wall this close, no ground in between: one
    min_climb_s: float = 2.0  # shorter stretches are not attempts


@dataclass
class Attempt:
    start: int  # first frame
    end: int  # one past the last frame
    end_reason: str
    rise_px: float  # COM rise on the wall (wall px)

    def span_s(self, t: np.ndarray) -> tuple[float, float]:
        return float(t[self.start]), float(t[self.end - 1])


@dataclass
class ClimbWindow:
    attempts: list[Attempt]
    main: int  # index into attempts
    mask: np.ndarray  # (n,) frames of the main attempt that count (breaks removed)
    size_px: float  # the climber's body size on the wall (image px), the reference
    notes: list[str] = field(default_factory=list)

    @property
    def attempt(self) -> Attempt:
        return self.attempts[self.main]

    def summary(self, t: np.ndarray) -> dict:
        a, b = self.attempt.span_s(t)
        out: dict = {"start_s": round(a, 2), "end_s": round(b, 2),
                     "duration_s": round(b - a, 2), "end": self.attempt.end_reason,
                     "frames_used_pct": round(100 * float(self.mask.mean()), 1)}
        if len(self.attempts) > 1:
            out["attempts"] = [{"start_s": round(x.span_s(t)[0], 2),
                                "end_s": round(x.span_s(t)[1], 2), "end": x.end_reason,
                                "measured": k == self.main}
                               for k, x in enumerate(self.attempts)]
        if self.notes:
            out["notes"] = self.notes
        return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of each run of True."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(int))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True))


def _persistent(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Only the runs of True at least min_len frames long."""
    out = np.zeros_like(mask, bool)
    for a, b in _runs(mask):
        if b - a >= min_len:
            out[a:b] = True
    return out


def _drop(com_y: np.ndarray, held: np.ndarray, a: int, b: int, fps: float, ref: float,
          prm: ClimbParams) -> int | None:
    """The frame where the drop to the mat ends, if the attempt [a, b) has one: a fall
    of the COM of `drop_bl` body lengths or more, faster than `drop_speed`, with no limb
    on a hold for most of its fast part. None if there is none."""
    y = np.asarray(com_y[a:b], float)
    if len(y) < 5 or not np.isfinite(y).any():
        return None
    ok = np.isfinite(y)
    y = np.interp(np.arange(len(y)), np.flatnonzero(ok), y[ok])
    k = max(1, int(round(0.1 * fps)))
    ys = np.convolve(y, np.ones(2 * k + 1) / (2 * k + 1), "same")
    v = np.gradient(ys) * fps / max(ref, 1e-6)  # body lengths/s, + = falling (y down)
    fast = v > prm.drop_speed
    for s0, s1 in _runs(fast):
        i = s0
        while i > 0 and v[i - 1] > 0.2:  # back to where the fall began
            i -= 1
        j = s1
        while j < len(v) and v[j] > 0.2:  # on to where it stops
            j += 1
        j = min(j, len(ys) - 1)
        after = ys[j: j + max(2, int(round(0.5 * fps)))]  # and it stays down (a pose
        stays = np.min(after) - ys[i] >= 0.7 * prm.drop_bl * ref  # glitch springs back)
        if ys[j] - ys[i] >= prm.drop_bl * ref and stays and held[a + s0: a + s1].mean() < 0.5:
            return a + j + 1
    return None


def find_climb(on_wall: np.ndarray, on_ground: np.ndarray, contact: np.ndarray,
               t: np.ndarray, size_px: np.ndarray, visibility: np.ndarray,
               camera_ok: np.ndarray | None, com_y: np.ndarray,
               prm: ClimbParams | None = None) -> ClimbWindow | None:
    """on_wall, on_ground: (n,) bool from a first pass of the wall layer; contact: (n,)
    limbs on holds or smearing; size_px: (n,) the body's size in the image (trunk +
    leg); visibility: (n,) mean joint visibility; com_y: (n,) COM height on the wall
    (wall px, y down). Returns None when nothing looks like a climb."""
    prm = prm or ClimbParams()
    n = len(t)
    if n < 5 or not on_wall.any():
        return None
    fps = 1.0 / float(np.median(np.diff(t)))

    def frames(s: float) -> int:
        return max(1, int(round(s * fps)))

    # the climber's size on the wall: frames with two or more limbs on holds
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref_frames = on_wall & np.isfinite(size_px)
        strong = ref_frames & (contact >= 2)
        ref = float(np.nanmedian(size_px[strong if strong.sum() > 30 else ref_frames]))
    size_bad = ~np.isfinite(size_px) | (size_px < prm.size_range[0] * ref) | (
        size_px > prm.size_range[1] * ref)
    vis_bad = ~np.isfinite(visibility) | (visibility < prm.min_visibility)
    cam_bad = np.zeros(n, bool) if camera_ok is None else ~np.asarray(camera_ok, bool)
    reasons = {"the camera lost the wall": cam_bad, "the pose was lost": vis_bad,
               "the person's size changed (walked off, or someone else)": size_bad}
    breaks = {k: _persistent(v, frames(prm.break_s)) for k, v in reasons.items()}
    broken = np.zeros(n, bool)
    for v in breaks.values():
        broken |= v
    held = contact > 0
    air = _persistent(~held, frames(prm.max_air_s))
    # stretches on the wall, cut at breaks and long stretches with no contact
    cand = on_wall & ~broken & ~air
    runs = _runs(cand)
    # merge stretches split by a short gap with no ground in it (a camera glitch)
    merged: list[list[int]] = []
    for a, b in runs:
        if merged:
            pa, pb = merged[-1]
            gap = slice(pb, a)
            if a - pb <= frames(prm.merge_s) and not on_ground[gap].any() \
                    and not air[gap].any():
                merged[-1][1] = b
                continue
        merged.append([a, b])
    attempts: list[Attempt] = []
    for a, b in merged:
        hold_idx = np.flatnonzero(held[a:b])
        if not len(hold_idx):
            continue
        last = a + int(hold_idx[-1])
        end, reason = b, "end of the clip" if b >= n else ""
        if b < n:
            # what ended it: the ground, a break, or leaving the wall
            nxt = slice(b, min(n, b + frames(prm.touchdown_s)))
            if on_ground[nxt].any():
                end = b + int(np.flatnonzero(on_ground[nxt])[0])
                reason = "touchdown"
            else:
                why = [k for k, v in breaks.items() if v[b:min(n, b + 2)].any()]
                reason = why[0] if why else ("left the wall" if air[b:min(n, b + 2)].any()
                                             else "touchdown")
                end = min(b, last + 1 + frames(prm.tail_s))
        end = max(end, last + 1)
        # the drop to the mat, when the camera doesn't see the touchdown (annotated:
        # arya1 landed at 20.6 s, then stood about; read as climbing until 23.6 s)
        drop = _drop(com_y, held, a, end, fps, ref, prm)
        if drop is not None:
            end, reason = drop, "touchdown (the drop)"
        if t[min(end, n) - 1] - t[a] < prm.min_climb_s:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            y = com_y[a:end]
            rise = float(np.nanmax(y) - np.nanmin(y)) if np.isfinite(y).any() else 0.0
        attempts.append(Attempt(a, int(end), reason, rise))
    if not attempts:
        return None
    main = max(range(len(attempts)),
               key=lambda k: (round(attempts[k].rise_px / max(ref, 1e-6), 1),
                              attempts[k].end - attempts[k].start))
    at = attempts[main]
    mask = np.zeros(n, bool)
    mask[at.start:at.end] = True
    mask &= ~broken
    notes = []
    if len(attempts) > 1:
        notes.append(f"{len(attempts)} attempts in the clip: measured attempt {main + 1} "
                     "(the one that got highest); film one attempt per clip for clean "
                     "results")
    return ClimbWindow(attempts, main, mask, ref, notes)
