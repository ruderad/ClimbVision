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


def write_wall_csv(path: Path, ws, t: np.ndarray) -> None:
    """Per frame: contacts on the wall, limb positions (wall px), hold ids, COM, margins."""
    from climbvision.engine.wall import LIMBS

    hold_at = {limb: np.full(len(t), -1) for limb in LIMBS}
    for limb in LIMBS:
        for a in ws.anchors[limb]:
            if a.on_hold:
                hold_at[limb][a.start:a.end] = a.hold
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "contacts", "n_contacts", "on_wall", "on_ground",
                    *(f"{limb}_hold" for limb in LIMBS),
                    *(f"{limb}_{a}" for limb in LIMBS for a in ("x", "y")),
                    "com_x", "com_y", "com_over_feet_cm", "polygon_margin_cm"])
        for i, ti in enumerate(t):
            w.writerow([f"{ti:.4f}", ws.labels[i], int(ws.n_contacts[i]), int(ws.on_wall[i]),
                        int(ws.on_ground[i]), *(int(hold_at[limb][i]) for limb in LIMBS),
                        *(f"{ws.points[limb][i, k]:.1f}" for limb in LIMBS for k in (0, 1)),
                        f"{ws.com[i, 0]:.1f}", f"{ws.com[i, 1]:.1f}",
                        f"{ws.com_over_feet_cm[i]:.2f}", f"{ws.polygon_margin_cm[i]:.2f}"])


def write_moves_csv(path: Path, route) -> None:
    """One row per move (route layer)."""
    cols = ["n", "limb", "kind", "from_hold", "to_hold", "release_s", "catch_s", "time_s",
            "distance_cm", "direction_deg", "others", "com_over_feet_before_cm",
            "com_over_feet_after_cm", "com_speed_peak_ms", "tags"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for m in route.moves:
            d = m.as_dict()
            d["tags"] = "; ".join(d["tags"])
            w.writerow(["" if d[c] is None else d[c] for c in cols])


def write_holds_csv(path: Path, route) -> None:
    """One row per hold (route layer): position, first use, time per limb, matched."""
    from climbvision.engine.wall import LABEL, LIMBS

    limbs = [LABEL[x] for x in LIMBS]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "x_m", "y_m", "used_as", "first_s", "time_s",
                    *(f"{x.replace(' ', '_').lower()}_s" for x in limbs), "matched"])
        for h in route.holds:
            w.writerow([h["id"], h["x_m"], h["y_m"], h["used_as"], h["first_s"], h["time_s"],
                        *(h["time_by_limb_s"].get(x, 0.0) for x in limbs), int(h["matched"])])
