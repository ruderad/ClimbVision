"""report.png: angles, the recipe's own signal, stability margin and contacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from climbvision.engine.signals import Clip
from climbvision.recipes import Recipe, Result


def plot_report(path: Path, clip: Clip, recipe: Recipe, result: Result) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = clip.t
    ang, st = clip.angles, clip.stability
    fig, axes = plt.subplots(4, 1, figsize=(9, 9.5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 2.4, 1.6]})
    ax = axes[0]
    for j, c in (("knee", "C0"), ("hip", "C1"), ("trunk_lean", "C2"), ("body_line", "C3")):
        ax.plot(t, ang.primary(j), lw=1, color=c, label=j.replace("_", " "))
    ax.set_ylabel("degrees")
    ax.legend(loc="lower left", ncol=4, fontsize=8)
    ax.set_title(f"{path.parent.name} — {ang.view} view, {ang.side} side"
                 + (f" — {recipe.name}" if recipe.name != "none" else ""))

    # the recipe's signal, reps and holds
    ax = axes[1]
    spec = recipe.spec.get("plot") or {}
    p = result.plot
    if "signal" in p:
        ax.plot(t, p["signal"], color="C0", lw=1.2, label=spec.get("label", "signal"))
        if "extra" in p:
            ax.plot(t, p["extra"], color="C1", lw=1, alpha=0.8, label=spec["extra"][1])
        if "rest" in p:
            ax.axhline(p["rest"], color="0.6", lw=0.8, ls="--", label="resting level")
        if spec.get("band"):
            lo, hi, lab = spec["band"]
            ax.axhspan(lo, hi, color="C2", alpha=0.12, label=lab)
        for k, (a, b, _bottom) in enumerate(p.get("rep_spans", []), 1):
            ax.axvspan(a, b, color="C0", alpha=0.08)
            ax.annotate(str(k), ((a + b) / 2, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(0, -10), textcoords="offset points", ha="center", fontsize=8)
        for a, b, lab in p.get("hold_spans", []):
            ax.axvspan(a, b, color="C2", alpha=0.12,
                       label=f"hold: {lab}" if lab else "longest hold")
        ax.set_ylabel(spec.get("label", ""), fontsize=8)
        m = result.metrics
        if "reps" in m:
            ax.set_title(f"{m['reps']} reps", fontsize=10)
    else:
        for j in ("elbow", "shoulder", "ankle"):
            ax.plot(t, ang.primary(j), lw=1, label=j)
        ax.set_ylabel("degrees")
    ax.legend(loc="lower left", fontsize=8)

    # stability margin
    ax = axes[2]
    m = st.margin_cm
    ax.axhline(0, color="0.3", lw=0.8)
    ax.fill_between(t, 0, m, where=np.nan_to_num(m) >= 0, color="C2", alpha=0.25, lw=0)
    ax.fill_between(t, 0, m, where=np.nan_to_num(m) < 0, color="C3", alpha=0.35, lw=0)
    ax.plot(t, m, color="0.2", lw=1)
    axis = "front-back" if st.view == "side" else "left-right"
    ax.set_ylabel(f"stability margin\n{axis} (cm)")
    ax.set_title("COM distance to the edge of support (+ inside, − outside)", fontsize=10)
    if st.on_bar.any():
        ax.plot(t, st.swing_cm, color="C4", lw=1, label="on bar: COM − grip x (cm)")
        ax.set_title("floor: COM to support edge (+ inside) · bar: COM swing vs grip",
                     fontsize=10)
        ax.legend(loc="lower left", fontsize=8)
    oof = st.feet_out_of_frame
    if oof.any():
        ax.fill_between(t, 0, 1, where=oof, transform=ax.get_xaxis_transform(),
                        color="0.5", alpha=0.15, lw=0, label="feet out of frame")
        ax.legend(loc="lower left", fontsize=8)

    # contacts raster
    ax = axes[3]
    rows = {g: v for g, v in st.contacts.group.items() if v.any()}
    rows.update({f"{sd}_hand_on_bar": v for sd, v in st.contacts.grip.items() if v.any()})
    groups = list(rows)
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.03
    for k, g in enumerate(groups):
        v = rows[g]
        idx = np.flatnonzero(np.diff(np.concatenate([[0], v.astype(int), [0]])))
        bars = [(t[a], t[min(b, len(t) - 1)] - t[a] + dt)
                for a, b in zip(idx[::2], idx[1::2], strict=True)]
        ax.broken_barh(bars, (k - 0.4, 0.8),
                       color="C0" if "foot" in g else "C4" if "bar" in g else "C1")
    ax.set_yticks(range(len(groups)), [g.replace("_", " ") for g in groups], fontsize=8)
    ax.set_ylim(-0.6, max(len(groups), 1) - 0.4)
    ax.set_title("contacts", fontsize=10)
    ax.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
