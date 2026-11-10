"""Generic event detectors on per-frame signals and masks.

These know nothing about exercises or climbing: recipes (climbvision/recipes) and
other layers combine them.
"""

from __future__ import annotations

import numpy as np


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of consecutive True values."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True))


def clean_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Fill False gaps shorter than min_len, then drop True runs shorter than min_len."""
    out = np.asarray(mask, bool).copy()
    for val in (False, True):
        m = np.concatenate([[not val], out == val, [not val]])
        d = np.diff(m.astype(int))
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True):
            interior = s > 0 and e < len(out)
            if e - s < min_len and (val or interior):
                out[s:e] = not val
    return out


def longest_run(mask: np.ndarray, t: np.ndarray, min_s: float = 0.0,
                min_frames: int = 0) -> tuple[int, int] | None:
    """The longest run of True lasting at least min_s seconds (t[end-1] - t[start]) and
    min_frames frames, as [start, end) — or None."""
    ok = [r for r in runs(mask) if r[1] - r[0] >= max(min_frames, 1)
          and t[r[1] - 1] - t[r[0]] >= min_s]
    return max(ok, key=lambda r: r[1] - r[0]) if ok else None


def find_cycles(signal: np.ndarray, t: np.ndarray, start_drop: float, end_drop: float,
                min_s: float = 0.6, rest: float | None = None) -> list[tuple[int, int, int]]:
    """Dips of `signal` below its resting level: [(start, bottom, end) indices].

    Resting level = 90th percentile (or `rest`). A cycle starts when the signal falls more
    than `start_drop` below rest (walking back to where the fall began) and ends when it
    comes back within `end_drop`; two thresholds (hysteresis) stop noise from double
    counting. Cycles shorter than `min_s` are dropped. For peaks, pass a negated signal.
    """
    rest = float(np.nanpercentile(signal, 90)) if rest is None else rest
    out = []
    in_cycle, start = False, 0
    for i, v in enumerate(signal):
        if not np.isfinite(v):
            continue
        if not in_cycle and v < rest - start_drop:
            in_cycle, start = True, i
            while start > 0 and np.isfinite(signal[start - 1]) and \
                    signal[start - 1] < rest - end_drop:
                start -= 1
        elif in_cycle and v > rest - end_drop:
            in_cycle = False
            if t[i] - t[start] >= min_s:
                b = start + int(np.nanargmin(signal[start:i + 1]))
                out.append((start, b, i))
    return out
