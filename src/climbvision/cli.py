"""Command line.

    climbvision analyze clip.mp4 [--exercise squat]   pose, metrics and annotated video
    climbvision metrics outputs/clip [...]            re-run angles/metrics only (fast)
    climbvision render outputs/clip [--video v.mp4]   redraw annotated.mp4 from results
    climbvision summary outputs/*                     one line per clip + summary.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXERCISES = ["auto", "squat", "pistol_squat", "plank", "pike_pushup", "jumping_jacks",
             "balance", "pullup", "dead_hang", "leg_raise", "none"]
EXERCISE_HELP = "exercise type, or auto = guess from the file name (default)"


def _print_metrics(m: dict) -> None:
    keep = {k: v for k, v in m.items() if k not in ("angle_ranges", "reps_detail")}
    if "stability" in keep:
        keep["stability"] = {k: v for k, v in keep["stability"].items()
                             if k in ("margin_cm", "com_sway", "contact_pct", "warnings")}
    print(json.dumps(keep, indent=2))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="climbvision")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="pose, centre of mass and metrics for videos")
    a.add_argument("videos", nargs="+", type=Path)
    a.add_argument("--out", type=Path, default=Path("outputs"),
                   help="results go to OUT/<video name>/ (default: outputs)")
    a.add_argument("--exercise", choices=EXERCISES, default="auto",
                   help=EXERCISE_HELP)
    a.add_argument("--sex", choices=["male", "female"], default="male",
                   help="which De Leva segment table to use")
    a.add_argument("--model", choices=["lite", "full", "heavy"], default="full",
                   help="MediaPipe model size: lite is fastest, heavy most accurate")
    a.add_argument("--no-video", action="store_true", help="skip writing annotated.mp4")
    a.add_argument("--max-frames", type=int, default=None)

    m = sub.add_parser("metrics", help="recompute angles/metrics from existing results")
    m.add_argument("results", nargs="+", type=Path, help="folders containing landmarks.csv")
    m.add_argument("--exercise", choices=EXERCISES, default="auto",
                   help=EXERCISE_HELP)

    r = sub.add_parser("render", help="redraw annotated.mp4 from existing results")
    r.add_argument("results", nargs="+", type=Path, help="folders containing landmarks.csv")
    r.add_argument("--video", type=Path, default=None,
                   help="source video (default: the path stored in summary.json)")

    sm = sub.add_parser("summary", help="one-line result per clip (table + CSV)")
    sm.add_argument("results", nargs="+", type=Path, help="result folders")
    sm.add_argument("--csv", type=Path, default=None,
                    help="where to write the CSV (default: summary.csv next to the folders)")

    args = p.parse_args(argv)

    if args.cmd == "summary":
        from climbvision.summary import markdown_table, summarize_results, write_summary

        rows = summarize_results(sorted(args.results))
        out = args.csv or (args.results[0].parent / "summary.csv")
        write_summary(rows, out)
        print(markdown_table(rows))
        print(f"\n[climbvision] wrote {out}")
        return 0

    from climbvision.metrics import run_metrics

    if args.cmd == "analyze":
        from climbvision.pipeline import analyze
        from climbvision.pose import MediaPipePose

        for video in args.videos:
            out = args.out / video.stem
            print(f"[climbvision] {video} -> {out}/")
            # fresh estimator per video: VIDEO mode needs increasing timestamps
            with MediaPipePose(model_size=args.model) as est:
                summary = analyze(video, out, est, sex=args.sex, max_frames=args.max_frames)
            print(json.dumps(summary, indent=2))
            _print_metrics(run_metrics(out, args.exercise))
            if not args.no_video:
                from climbvision.render import render

                print(f"[climbvision] wrote {render(video, out)}")

    elif args.cmd == "metrics":
        for d in args.results:
            if not (d / "landmarks.csv").is_file():
                print(f"[climbvision] skipping {d} (no landmarks.csv)")
                continue
            print(f"[climbvision] {d}")
            _print_metrics(run_metrics(d, args.exercise))

    elif args.cmd == "render":
        from climbvision.render import render

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
