"""Velocity and acceleration from positions (Savitzky-Golay: a local polynomial fit).

Differentiating pose positions amplifies their jitter, acceleration twice over. Fitting a
quadratic to a short window (default 0.2 s) around every frame and reading its slope and
curvature keeps real motion (a dyno lasts ~0.5 s) while averaging out frame-to-frame
noise. A window with a missing frame gives NaN rather than a guess.
"""

from __future__ import annotations

import numpy as np


def _coeffs(half: int, order: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(-half, half + 1) * dt
    A = np.vander(x, order + 1, increasing=True)  # 1, x, x^2
    P = np.linalg.pinv(A)  # rows: coefficient of 1, x, x^2
    return P[1], 2 * P[2]  # d/dt and d2/dt2 at x = 0


def derivatives(x: np.ndarray, t: np.ndarray, window_s: float = 0.2,
                order: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """(n, d) positions -> (velocity, acceleration), both (n, d), NaN near gaps."""
    x = np.asarray(x, float)
    n = len(x)
    vel = np.full(x.shape, np.nan)
    acc = np.full(x.shape, np.nan)
    if n < 5:
        return vel, acc
    dt = float(np.median(np.diff(t)))
    half = max(2, int(round(window_s / dt / 2)))
    if 2 * half + 1 > n:
        return vel, acc
    cv, ca = _coeffs(half, order, dt)
    win = np.lib.stride_tricks.sliding_window_view(x, 2 * half + 1, axis=0)  # (m, d, w)
    vel[half:n - half] = win @ cv
    acc[half:n - half] = win @ ca
    return vel, acc
