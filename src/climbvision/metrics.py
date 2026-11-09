"""Turn a results folder (landmarks.csv) into angles.csv, stability.csv, metrics.json
and report.png."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Literal

import numpy as np

from climbvision.angles import Angles, compute_angles
from climbvision.exercises import (
    dead_hang,
    jumping_jack_reps,
    leg_raise_reps,
    pike_pushup_reps,
    plank_metrics,
    pullup_reps,
    reps_as_dicts,
    squat_reps,
)
from climbvision.stability import (
    Stability,
    balance_hold,
    compute_stability,
    stability_summary,
    stance_labels,
)
from climbvision.track import Track

Exercise = Literal["squat", "pistol_squat", "plank", "pike_pushup", "jumping_jacks",
                   "balance", "pullup", "dead_hang", "leg_raise", "none"]
EXERCISE_NAMES = ("squat", "pistol_squat", "plank", "pike_pushup", "jumping_jacks", "balance",
                  "pullup", "dead_hang", "leg_raise", "none")
BAR_EXERCISES = ("pullup", "dead_hang", "leg_raise")

# file-name patterns, checked in order (first match wins)
NAME_PATTERNS: list[tuple[str, str]] = [
    (r"toes.?to.?bar|t2b|leg.?raise|knee.?raise", "leg_raise"),
    (r"pull.?up|chin.?up|muscle.?up", "pullup"),
    (r"dead.?hang|hang", "dead_hang"),
    (r"pistol", "pistol_squat"),
    (r"squat", "squat"),
    (r"pike", "pike_pushup"),
    (r"plank", "plank"),
    (r"jumping.?jack|\bjacks?\b", "jumping_jacks"),
    (r"tree|crow|one.?foot|single.?leg|balance|stork", "balance"),
]


def guess_exercise(name: str) -> Exercise:
    for pattern, ex in NAME_PATTERNS:
        if re.search(pattern, name, re.IGNORECASE):
            return ex  # type: ignore[return-value]
    return "none"


def write_angles_csv(path: Path, ang: Angles) -> None:
    cols = sorted(ang.values)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", *cols])
        for i, t in enumerate(ang.time_s):
            w.writerow([f"{t:.4f}", *(f"{ang.values[c][i]:.2f}" for c in cols)])


def write_stability_csv(path: Path, st: Stability, t: np.ndarray) -> None:
    groups = list(st.contacts.group)
    grips = list(st.contacts.grip)  # empty unless a bar exercise
    labels = stance_labels(st.contacts)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "stance", *groups, *(f"{g}_hand_on_bar" for g in grips),
                    "com_x_px", "support_lo_px", "support_hi_px", "margin_cm", "swing_cm",
                    "feet_out_of_frame"])
        for i, ti in enumerate(t):
            w.writerow([f"{ti:.4f}", labels[i], *(int(st.contacts.group[g][i]) for g in groups),
                        *(int(st.contacts.grip[g][i]) for g in grips),
                        f"{st.com_px[i, 0]:.1f}", f"{st.support_lo_px[i]:.1f}",
                        f"{st.support_hi_px[i]:.1f}", f"{st.margin_cm[i]:.2f}",
                        f"{st.swing_cm[i]:.2f}", int(st.feet_out_of_frame[i])])


def _angle_ranges(ang: Angles) -> dict:
    out = {}
    for j in ("knee", "hip", "ankle", "thigh", "elbow", "shoulder", "body_line", "trunk_lean"):
        v = ang.primary(j)
        if np.isfinite(v).any():
            out[j] = {"min": round(float(np.nanmin(v)), 1),
                      "median": round(float(np.nanmedian(v)), 1),
                      "max": round(float(np.nanmax(v)), 1)}
    return out


def _plot(path: Path, track: Track, ang: Angles, exercise: str, result: dict,
          st: Stability) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = track.time_s
    fig, axes = plt.subplots(4, 1, figsize=(9, 9.5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 2.4, 1.6]})
    ax = axes[0]
    for j, c in (("knee", "C0"), ("hip", "C1"), ("trunk_lean", "C2"), ("body_line", "C3")):
        ax.plot(t, ang.primary(j), lw=1, color=c, label=j.replace("_", " "))
    ax.set_ylabel("degrees")
    ax.legend(loc="lower left", ncol=4, fontsize=8)
    ax.set_title(f"{path.parent.name} — {ang.view} view, {ang.side} side"
                 + (f" — {exercise}" if exercise != "none" else ""))

    ax = axes[1]
    if exercise in ("squat", "pistol_squat"):
        s = result["series"]
        ax.plot(t, s["knee"], color="C0", lw=1.2,
                label="knee (smoothed)" + (", bending leg" if exercise == "pistol_squat" else ""))
        ax.axhline(s["top"], color="0.6", lw=0.8, ls="--", label="standing")
        for r in result["reps_detail"]:
            ax.axvspan(r["start_s"], r["end_s"], color="C0", alpha=0.08)
            ax.plot(r["bottom_s"], r["min_knee_deg"], "v", color="C3")
            ax.annotate(str(r["rep"]), (r["bottom_s"], r["min_knee_deg"]),
                        textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8)
        ax.set_ylabel("knee angle (deg)")
        ax.set_title(f"{result['reps']} reps", fontsize=10)
    elif exercise == "plank":
        s = result["series"]
        ax.plot(t, s["offset"], color="C3", lw=1.2, label="hip offset")
        ax.axhspan(-5, 5, color="C2", alpha=0.12, label="straight (±5%)")
        a, b = s["hold"]
        if b > a:
            ax.axvspan(t[a], t[b - 1], color="0.5", alpha=0.08, label="hold")
        ax.set_ylabel("hip offset (% body line)\n+ sag / − pike")
        ax.set_title(f"hold {result.get('hold_s', 0)} s", fontsize=10)
    elif exercise in ("pike_pushup", "jumping_jacks", "pullup", "leg_raise"):
        s = result["series"]
        ax.plot(t, s["signal"], color="C0", lw=1.2, label=s["label"])
        if "extra" in s:
            ax.plot(t, s["extra"][0], color="C1", lw=1, alpha=0.8, label=s["extra"][1])
        for k, (a, b) in enumerate(s["cycles"], 1):
            ax.axvspan(a, b, color="C0", alpha=0.08)
            ax.annotate(str(k), ((a + b) / 2, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(0, -10), textcoords="offset points", ha="center", fontsize=8)
        ax.set_ylabel("degrees")
        ax.set_title(f"{result['reps']} reps", fontsize=10)
    elif exercise == "balance" and result:
        x = st.com_x_cm - np.nanmedian(st.com_x_cm)
        ax.plot(t, x, color="C4", lw=1.2, label="COM position (cm, + = image right)")
        a = result["start_s"]
        ax.axvspan(a, a + result["hold_s"], color="C2", alpha=0.12,
                   label=f"longest hold: {result['stance']}")
        ax.set_ylabel("COM sway (cm)")
        ax.set_title(f"hold {result['hold_s']} s on {result['stance']}, "
                     f"sway sd {result.get('sd_cm', float('nan'))} cm", fontsize=10)
    elif exercise == "dead_hang" and "hold_s" in result:
        ax.plot(t, st.swing_cm, color="C4", lw=1.2, label="COM relative to grip (cm)")
        a = result["start_s"]
        ax.axvspan(a, a + result["hold_s"], color="C2", alpha=0.12, label="longest hang")
        ax.set_ylabel("swing (cm)")
        ax.set_title(f"hang {result['hold_s']} s, swing sd {result.get('swing_sd_cm')} cm",
                     fontsize=10)
    else:
        ax.plot(t, ang.primary("elbow"), lw=1, label="elbow")
        ax.plot(t, ang.primary("shoulder"), lw=1, label="shoulder")
        ax.plot(t, ang.primary("ankle"), lw=1, label="ankle")
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
        bars = [(t[a], t[min(b, len(t) - 1)] - t[a] + dt) for a, b in zip(idx[::2], idx[1::2],
                                                                            strict=True)]
        ax.broken_barh(bars, (k - 0.4, 0.8),
                       color="C0" if "foot" in g else "C4" if "bar" in g else "C1")
    ax.set_yticks(range(len(groups)), [g.replace("_", " ") for g in groups], fontsize=8)
    ax.set_ylim(-0.6, max(len(groups), 1) - 0.4)
    ax.set_title("contacts", fontsize=10)
    ax.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_metrics(result_dir: str | Path, exercise: Exercise | Literal["auto"] = "auto",
                plot: bool = True) -> dict:
    result_dir = Path(result_dir)
    track = Track.from_csv(result_dir / "landmarks.csv")
    info = {}
    if (result_dir / "summary.json").exists():
        info = json.loads((result_dir / "summary.json").read_text())
    sex = info.get("sex_model", "male")
    frame_size = tuple(info["frame_size"]) if "frame_size" in info else None
    if exercise == "auto":
        exercise = guess_exercise(result_dir.name)
    ang = compute_angles(track)
    write_angles_csv(result_dir / "angles.csv", ang)

    out: dict = {"exercise": exercise, "view": ang.view, "body_yaw_deg": ang.yaw_deg,
                 "near_side": ang.side, "angle_ranges": _angle_ranges(ang)}
    st = compute_stability(track, ang.view, sex, frame_size, bar=exercise in BAR_EXERCISES)
    detail: dict = {}
    if exercise in ("squat", "pistol_squat"):
        reps, summary = squat_reps(track, ang, single_leg=exercise == "pistol_squat")
        detail = {**summary, "reps_detail": reps_as_dicts(reps)}
    elif exercise == "plank":
        detail = plank_metrics(track, ang)
    elif exercise == "pike_pushup":
        reps, summary = pike_pushup_reps(track, ang)
        detail = {**summary, "reps_detail": reps_as_dicts(reps)}
    elif exercise == "jumping_jacks":
        reps, summary = jumping_jack_reps(track, ang)
        detail = {**summary, "reps_detail": reps_as_dicts(reps)}
    elif exercise == "pullup":
        reps, summary = pullup_reps(track, ang, st)
        detail = {**summary, "reps_detail": reps_as_dicts(reps)}
    elif exercise == "leg_raise":
        reps, summary = leg_raise_reps(track, ang, st)
        detail = {**summary, "reps_detail": reps_as_dicts(reps)}
    elif exercise == "dead_hang":
        detail = dead_hang(track, ang, st)
    elif exercise == "balance":
        detail = balance_hold(st, track.time_s)
        if not detail:
            detail = {"note": "no hold of >= 1 s on one foot / hands found"}

    write_stability_csv(result_dir / "stability.csv", st, track.time_s)

    if plot:
        _plot(result_dir / "report.png", track, ang, exercise, detail, st)
    detail.pop("series", None)
    out.update(detail if exercise not in ("balance", "dead_hang") else {exercise: detail})
    out["stability"] = stability_summary(st, track.time_s)
    (result_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    return out
