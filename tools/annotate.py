"""Annotation sheets: human feedback for what the pose can't settle on its own.

    python tools/annotate.py make  OUT_DIR results...   -> grips.csv, feet.csv, climbs.csv,
                                                         holds_tape.csv (pre-filled)
    python tools/annotate.py score grips.csv  results... -> how well the grips match
    python tools/annotate.py score feet.csv   results... -> how well the feet match

Each sheet lists what ClimbVision found, one row per item, with empty columns for you.
Fill what you can; leave the rest empty. Times are the viewer's clock (seconds from the
start of the video); `start_frame` helps when you step frame by frame.

grips.csv   every hand on a hold (and hands the system thinks are on a hold the detector
            missed). verdict: ok / partly (right hold, wrong times: give true times) /
            wrong (no grip there) / swap (it was the other hand). Add a row with verdict
            `missed` for a grip that isn't listed (clip, limb, true_start_s, true_end_s).
            true_end_s empty = the grip lasts to the end of the clip.
feet.csv    every foot that stayed still: on a hold / smear on the wall / in the air
            (a flag) / on the floor, as the system read it. truth: hold, smear, air,
            floor, pose error.
climbs.csv  one row per clip: where the climb starts and ends, attempts. Also the labels
            habit: grade, sent or fell.
holds_tape.csv  the holds used, with the height ClimbVision measured; with a tape, fill
            the true height above the mat (photo_no = your number on the wall photo).

`score` reads a filled grips.csv (rows with verdict ok / partly / missed are true grips,
`wrong` rows are places with no grip) and compares them with the analysis as it is now:
grips found, pieces per grip (1 = not broken up), share of each grip covered, and
grips reported where there was none.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import warnings
from pathlib import Path

import numpy as np

LIMB_NAME = {"l_hand": "L hand", "r_hand": "R hand", "l_foot": "L foot", "r_foot": "R foot"}
NAME_LIMB = {v.lower(): k for k, v in LIMB_NAME.items()}


def _clip(d: Path):
    from climbvision.analysis import load_clip

    warnings.filterwarnings("ignore")
    c = load_clip(d)
    _ = c.wall_support
    return c


def _kind(a) -> str:
    if a.ground:
        return "floor"
    if a.flag:
        return "air (flag)"
    if a.smear:
        return "smear"
    return "missed hold" if getattr(a, "missed", False) else "hold"


def _write(path: Path, cols: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, r in enumerate(rows, 1):
            w.writerow({"n": i, **r})
    print(f"wrote {path} ({len(rows)} rows)")


def make(out: Path, results: list[Path]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    grips, feet, climbs, holds = [], [], [], []
    for d in results:
        c = _clip(d)
        ws, t = c.wall_support, c.t
        fps = 1.0 / float(np.median(np.diff(t)))
        name = d.name
        for limb in LIMB_NAME:
            for a in ws.anchors[limb]:
                if ws.window is not None and not ws.window[a.start:a.end].any() and not a.ground:
                    continue
                row = {"clip": name, "limb": LIMB_NAME[limb], "start_s": f"{t[a.start]:.2f}",
                       "end_s": f"{t[a.end - 1]:.2f}", "start_frame": int(round(t[a.start] * fps)),
                       "system": _kind(a), "hold": a.hold if a.hold >= 0 else ""}
                if limb.endswith("hand"):
                    if a.on_hold:
                        grips.append({**row, "verdict": "", "true_start_s": "",
                                      "true_end_s": "", "note": ""})
                else:
                    feet.append({**row, "truth": "", "note": ""})
        cl = c.climb.summary(t) if c.climb is not None else {}
        climbs.append({"clip": name, "start_s": cl.get("start_s", ""), "end_s": cl.get("end_s", ""),
                       "end": cl.get("end", ""),
                       "attempts_found": len(cl.get("attempts", [])) or (1 if cl else 0),
                       "verdict": "", "true_start_s": "", "true_end_s": "", "attempts": "",
                       "climber": c.climber.name or "", "grade": "", "sent": "", "note": ""})
        for h in ws.holds:
            x, y = ws.hold_xy_m(h)
            holds.append({"clip": name, "hold": h.id,
                          "limbs": " + ".join(sorted(LIMB_NAME[x_] for x_ in h.limbs)),
                          "first_s": f"{t[min(a.start for a in h.anchors)]:.2f}",
                          "x_m": f"{x:.2f}", "height_m": f"{y:.2f}",
                          "height_from": "floor" if ws.heights_above_floor else "lowest hold",
                          "photo_no": "", "tape_height_m": "", "note": ""})
    _write(out / "grips.csv", ["n", "clip", "limb", "start_s", "end_s", "start_frame", "system",
                               "hold", "verdict", "true_start_s", "true_end_s", "note"], grips)
    _write(out / "feet.csv", ["n", "clip", "limb", "start_s", "end_s", "start_frame", "system",
                              "hold", "truth", "note"], feet)
    _write(out / "climbs.csv", ["n", "clip", "start_s", "end_s", "end", "attempts_found",
                                "verdict", "true_start_s", "true_end_s", "attempts", "climber",
                                "grade", "sent", "note"], climbs)
    _write(out / "holds_tape.csv", ["n", "clip", "hold", "limbs", "first_s", "x_m", "height_m",
                                    "height_from", "photo_no", "tape_height_m", "note"], holds)


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def score(sheet: Path, results: list[Path]) -> dict:
    """Compare the true grips of a filled grips.csv with the analysis as it is now."""
    rows = list(csv.DictReader(sheet.open()))
    by_clip: dict[str, list[dict]] = {}
    for r in rows:
        by_clip.setdefault(r["clip"], []).append(r)
    tot = {"grips": 0, "found": 0, "pieces": 0, "coverage": [], "no_grip": 0, "false": 0}
    for d in results:
        if d.name not in by_clip:
            continue
        c = _clip(d)
        ws, t = c.wall_support, c.t
        end = float(t[-1])
        for r in by_clip[d.name]:
            limb = NAME_LIMB.get(r["limb"].strip().lower())
            v = r.get("verdict", "").strip().lower()
            if limb is None or not v:
                continue
            if v == "swap":
                limb = {"l_hand": "r_hand", "r_hand": "l_hand"}.get(limb, limb)
            s = _f(r.get("true_start_s")) if r.get("true_start_s") else _f(r.get("start_s"))
            e = _f(r.get("true_end_s")) if r.get("true_end_s") else (
                end if v in ("missed",) or r.get("true_start_s") else _f(r.get("end_s")))
            if s is None or e is None or e <= s:
                continue
            held = [a for a in ws.anchors[limb] if a.on_hold]
            over = [a for a in held if t[a.start] < e and t[a.end - 1] > s]
            if v == "wrong":
                tot["no_grip"] += 1
                tot["false"] += bool(over)
                continue
            cov = sum(max(0.0, min(e, t[a.end - 1]) - max(s, t[a.start])) for a in over) / (e - s)
            tot["grips"] += 1
            tot["found"] += bool(over)
            tot["pieces"] += len(over)
            tot["coverage"].append(min(cov, 1.0))
    g = max(tot["grips"], 1)
    out = {"true_grips": tot["grips"], "found_pct": round(100 * tot["found"] / g, 1),
           "pieces_per_grip": round(tot["pieces"] / g, 2),
           "coverage_pct": round(100 * float(np.mean(tot["coverage"] or [0])), 1),
           "places_with_no_grip": tot["no_grip"], "reported_there": tot["false"]}
    for k, v in out.items():
        print(f"  {k:22s} {v}")
    return out


FOOT_TRUTH = (("brush", "smear"), ("smear", "smear"), ("flag", "air"), ("air", "air"),
              ("floor", "floor"), ("hold", "hold"))


def _norm_foot(v: str) -> str | None:
    v = v.strip().lower()
    if not v or " then " in v or " and " in v or "pose" in v:
        return None  # mixed rows (flag, then hold) and pose errors are not scored
    for key, cat in FOOT_TRUTH:
        if key in v:
            return cat
    return None


def _sys_foot(a) -> str:
    return "floor" if a.ground else "air" if a.flag else "smear" if a.smear else "hold"


def score_feet(sheet: Path, results: list[Path]) -> dict:
    """Compare a filled feet.csv (truth: hold / smear / air / floor) with the foot
    anchors as they are now (the anchor overlapping each row most)."""
    rows = list(csv.DictReader(sheet.open()))
    cats = ("hold", "smear", "air", "floor")
    conf = {(a, b): 0 for a in cats for b in cats + ("none",)}
    for d in results:
        rs = [r for r in rows if r["clip"] == d.name]
        if not rs:
            continue
        c = _clip(d)
        ws, t = c.wall_support, c.t
        for r in rs:
            truth = _norm_foot(r.get("truth", ""))
            limb = NAME_LIMB.get(r["limb"].strip().lower())
            s, e = _f(r.get("start_s")), _f(r.get("end_s"))
            if truth is None or limb is None or s is None or e is None:
                continue
            best, ov = None, 0.0
            for a in ws.anchors[limb]:
                o = min(e, t[a.end - 1]) - max(s, t[a.start])
                if o > ov:
                    best, ov = a, o
            conf[(truth, _sys_foot(best) if best is not None else "none")] += 1
    n = sum(conf.values())
    right = sum(conf[(a, a)] for a in cats)
    print(f"  feet rows scored       {n}")
    print(f"  system agrees          {right} ({100 * right / max(n, 1):.0f} %)")
    print("  truth -> system: " + ", ".join(f"{a}->{b} {v}" for (a, b), v in conf.items()
                                            if v and a != b))
    return {"rows": n, "agree": right, "confusion": {f"{a}->{b}": v for (a, b), v in
                                                       conf.items() if v}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make", help="pre-filled sheets for results folders")
    m.add_argument("out", type=Path)
    m.add_argument("results", nargs="+", type=Path)
    s = sub.add_parser("score", help="compare a filled grips.csv with the analysis")
    s.add_argument("sheet", type=Path)
    s.add_argument("results", nargs="+", type=Path)
    args = ap.parse_args(argv)
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    if args.cmd == "make":
        make(args.out, args.results)
    elif "truth" in (next(csv.reader(args.sheet.open())) or []):
        score_feet(args.sheet, args.results)
    else:
        score(args.sheet, args.results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
