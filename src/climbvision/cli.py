"""Command line.

    climbvision analyze clip.mp4 [--recipe squat]   pose, analysis and annotated video
    climbvision metrics outputs/clip [...]          re-run the analysis only (seconds)
    climbvision render outputs/clip [--video v.mp4] redraw annotated.mp4 from results
    climbvision summary outputs/*                   one line per clip + summary.csv
    climbvision recipes                             list recipes
    climbvision signals                             list the signals recipes can use

--recipe takes a recipe name, a path to a .yaml file, or `auto` (guess from the file
name). --recipes DIR adds your own recipes. --exercise is an old alias of --recipe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RECIPE_HELP = "recipe name or .yaml path; auto = guess from the file name (default)"


def _print_metrics(m: dict) -> None:
    keep = {k: v for k, v in m.items() if k not in ("angle_ranges", "reps_detail")}
    if "stability" in keep:
        keep["stability"] = {k: v for k, v in keep["stability"].items()
                             if k in ("margin_cm", "com_sway", "contact_pct", "warnings")}
    print(json.dumps(keep, indent=2))


def _add_recipe_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--recipe", "--exercise", dest="recipe", default="auto", help=RECIPE_HELP)
    p.add_argument("--recipes", type=Path, action="append", default=[],
                   help="extra folder of recipe .yaml files (repeatable)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="climbvision")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="pose, centre of mass and analysis for videos")
    a.add_argument("videos", nargs="+", type=Path)
    a.add_argument("--out", type=Path, default=Path("outputs"),
                   help="results go to OUT/<video name>/ (default: outputs)")
    _add_recipe_args(a)
    a.add_argument("--sex", choices=["male", "female"], default="male",
                   help="which De Leva segment table to use")
    a.add_argument("--backend", default=None, help="pose backend (default: mediapipe)")
    a.add_argument("--model", choices=["lite", "full", "heavy"], default="full",
                   help="MediaPipe model size: lite is fastest, heavy most accurate")
    a.add_argument("--no-video", action="store_true", help="skip writing annotated.mp4")
    a.add_argument("--max-frames", type=int, default=None)

    m = sub.add_parser("metrics", help="re-run the analysis from existing results")
    m.add_argument("results", nargs="+", type=Path, help="folders containing landmarks.csv")
    _add_recipe_args(m)

    r = sub.add_parser("render", help="redraw annotated.mp4 from existing results")
    r.add_argument("results", nargs="+", type=Path, help="folders containing landmarks.csv")
    r.add_argument("--video", type=Path, default=None,
                   help="source video (default: the path stored in summary.json)")

    sm = sub.add_parser("summary", help="one-line result per clip (table + CSV)")
    sm.add_argument("results", nargs="+", type=Path, help="result folders")
    sm.add_argument("--csv", type=Path, default=None,
                    help="where to write the CSV (default: summary.csv next to the folders)")
    sm.add_argument("--recipes", type=Path, action="append", default=[])

    rl = sub.add_parser("recipes", help="list the available recipes")
    rl.add_argument("--recipes", type=Path, action="append", default=[])
    sub.add_parser("signals", help="list the signals recipes can use")

    args = p.parse_args(argv)

    if args.cmd == "summary":
        from climbvision.output.summary import markdown_table, summarize_results, write_summary

        rows = summarize_results(sorted(args.results), args.recipes)
        out = args.csv or (args.results[0].parent / "summary.csv")
        write_summary(rows, out)
        print(markdown_table(rows))
        print(f"\n[climbvision] wrote {out}")
        return 0
    if args.cmd == "recipes":
        from climbvision.recipes import load_recipes

        for rec in sorted(load_recipes(args.recipes).values(), key=lambda r: r.priority):
            desc = " ".join(str(rec.spec.get("description", "")).split())
            print(f"{rec.name:15s} {rec.spec.get('scene', 'floor'):5s}  {desc[:90]}")
        return 0
    if args.cmd == "signals":
        from climbvision.engine.signals import describe

        for name, doc in describe().items():
            print(f"{name:28s} {doc}")
        return 0

    from climbvision.analysis import run_analysis

    if args.cmd == "analyze":
        from climbvision.engine import backends
        from climbvision.engine.pose import make_estimator, run_pose

        backend = args.backend or backends.DEFAULT
        for video in args.videos:
            out = args.out / video.stem
            print(f"[climbvision] {video} -> {out}/")
            # fresh estimator per video: VIDEO mode needs increasing timestamps
            kw = {"model_size": args.model} if backend == backends.DEFAULT else {}
            est = make_estimator(backend, **kw)
            try:
                summary = run_pose(video, out, est, sex=args.sex, max_frames=args.max_frames)
            finally:
                est.close()
            print(json.dumps(summary, indent=2))
            _print_metrics(run_analysis(out, args.recipe, recipe_dirs=args.recipes))
            if not args.no_video:
                from climbvision.output.render import render

                print(f"[climbvision] wrote {render(video, out)}")

    elif args.cmd == "metrics":
        for d in args.results:
            if not (d / "landmarks.csv").is_file():
                print(f"[climbvision] skipping {d} (no landmarks.csv)")
                continue
            print(f"[climbvision] {d}")
            _print_metrics(run_analysis(d, args.recipe, recipe_dirs=args.recipes))

    elif args.cmd == "render":
        from climbvision.output.render import render

        for d in args.results:
            summary = json.loads((d / "summary.json").read_text())
            video = args.video or Path(summary["video"])
            if not video.exists():
                print(f"[climbvision] {d}: source video {video} not found, use --video")
                continue
            print(f"[climbvision] wrote {render(video, d)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
