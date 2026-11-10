"""Per-frame CSV outputs (re-creatable from landmarks.csv at any time)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from climbvision.engine.biomech import Angles
from climbvision.engine.support import Stability


def write_angles_csv(path: Path, ang: Angles) -> None:
    cols = sorted(ang.values)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", *cols])
        for i, t in enumerate(ang.time_s):
            w.writerow([f"{t:.4f}", *(f"{ang.values[c][i]:.2f}" for c in cols)])


def write_stability_csv(path: Path, st: Stability, t: np.ndarray) -> None:
    groups = list(st.contacts.group)
    grips = list(st.contacts.grip)  # empty unless a bar scene
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "stance", *groups, *(f"{g}_hand_on_bar" for g in grips),
                    "com_x_px", "support_lo_px", "support_hi_px", "margin_cm", "swing_cm",
                    "feet_out_of_frame"])
        for i, ti in enumerate(t):
            w.writerow([f"{ti:.4f}", st.labels[i],
                        *(int(st.contacts.group[g][i]) for g in groups),
                        *(int(st.contacts.grip[g][i]) for g in grips),
                        f"{st.com_px[i, 0]:.1f}", f"{st.support_lo_px[i]:.1f}",
                        f"{st.support_hi_px[i]:.1f}", f"{st.margin_cm[i]:.2f}",
                        f"{st.swing_cm[i]:.2f}", int(st.feet_out_of_frame[i])])
