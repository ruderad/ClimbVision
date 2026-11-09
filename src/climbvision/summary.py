"""One line per clip: `climbvision summary outputs/*` -> table + summary.csv."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def _f(v, fmt="{:.1f}", none="–") -> str:
    return none if v is None else fmt.format(v)


def _n(k: int, word: str) -> str:
    return f"{k} {word}" + ("" if k == 1 else "s")


def headline(m: dict) -> str:
    """The main result of a clip in a few words."""
    ex = m.get("exercise", "none")
    st = m.get("stability", {})
    if ex in ("squat", "pistol_squat"):
        s = f"{_n(m.get('reps', 0), 'rep')}, {m.get('reps_below_parallel', 0)} below parallel"
        if "min_knee_deg_mean" in m:
            s += f", knee {_f(m['min_knee_deg_mean'], '{:.0f}')}°"
        if "reps_by_leg" in m:
            s += f" (L {m['reps_by_leg']['left']} / R {m['reps_by_leg']['right']})"
        return s
    if ex == "plank":
        if not m.get("hold_s"):
            return "no plank hold found"
        return (f"hold {_f(m['hold_s'])} s, hips {_f(m.get('hip_offset_pct_mean'))}% "
                f"(straight {_f(m.get('pct_time_straight'), '{:.0f}')}%, "
                f"pike {_f(m.get('pct_time_pike'), '{:.0f}')}%, "
                f"sag {_f(m.get('pct_time_sag'), '{:.0f}')}%)")
    if ex == "pike_pushup":
        return (f"{_n(m.get('reps', 0), 'pike push-up')}, plank {_f(m.get('plank_s'))} s, "
                f"pike {_f(m.get('pike_straight_arms_s'))} s")
    if ex == "jumping_jacks":
        return (f"{_n(m.get('reps', 0), 'rep')}, {_f(m.get('reps_per_min'), '{:.0f}')}/min, "
                f"{(st.get('jumps') or {}).get('count', 0)} jumps")
    if ex == "balance":
        b = m.get("balance", {})
        if "hold_s" not in b:
            return "no balance hold found"
        return f"{b['hold_s']} s on {b['stance']}, sway sd {_f(b.get('sd_cm'))} cm"
    if ex == "pullup":
        c = m.get("reps_chin", {})
        return (f"{_n(m.get('reps', 0), 'rep')}, pull {_f(m.get('pull_cm_mean'), '{:.0f}')} cm, "
                f"chin over/at/under {c.get('over', 0)}/{c.get('at bar', 0)}/"
                f"{c.get('under', 0)}, kipping {m.get('reps_kipping', 0)}")
    if ex == "leg_raise":
        s = (f"{_n(m.get('reps', 0), 'rep')}, hip {_f(m.get('min_hip_deg_mean'), '{:.0f}')}°, "
             f"toes-to-bar {m.get('reps_toes_to_bar', 0)}")
        if "raised_hold" in m:
            h = m["raised_hold"]
            s += f"; hold {h['hold_s']} s, hips sank {h['hip_drop_deg']}°"
        return s
    if ex == "dead_hang":
        h = m.get("dead_hang", {})
        if "hold_s" not in h:
            return "no hang found"
        return f"hang {h['hold_s']} s, swing sd {_f(h.get('swing_sd_cm'))} cm"
    stances = list(st.get("stance_time_s", {}).items())[:2]
    s = ", ".join(f"{k} {v:.1f} s" for k, v in stances)
    if st.get("jumps"):
        s += f"; {st['jumps']['count']} jumps"
    return s or "–"


COLUMNS = ["clip", "exercise", "view", "result", "margin_median_cm", "sway_sd_cm",
           "warnings"]


def summarize_results(dirs: list[Path]) -> list[dict]:
    rows = []
    for d in dirs:
        p = d / "metrics.json"
        if not p.is_file():
            continue
        m = json.loads(p.read_text())
        st = m.get("stability", {})
        rows.append({
            "clip": d.name,
            "exercise": m.get("exercise", ""),
            "view": m.get("view", ""),
            "result": headline(m),
            "margin_median_cm": (st.get("margin_cm") or {}).get("median"),
            "sway_sd_cm": (st.get("com_sway") or {}).get("sd_cm"),
            "warnings": "; ".join(st.get("warnings", [])),
        })
    return rows


def write_summary(rows: list[dict], csv_path: Path) -> None:
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def markdown_table(rows: list[dict]) -> str:
    cols = ["clip", "exercise", "view", "result", "warnings"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        warn = "⚠ " + r["warnings"][:60] if r["warnings"] else ""
        lines.append("| " + " | ".join(str(x) for x in
                                       (r["clip"], r["exercise"], r["view"], r["result"],
                                        warn)) + " |")
    return "\n".join(lines)
