"""Command line: `climbvision analyze clip.mp4`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="climbvision")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="pose + centre of mass for one or more videos")
    a.add_argument("videos", nargs="+", type=Path)
    a.add_argument("--out", type=Path, default=Path("outputs"),
                   help="results go to OUT/<video name>/ (default: outputs)")
    a.add_argument("--sex", choices=["male", "female"], default="male",
                   help="which De Leva segment table to use")
    a.add_argument("--model", choices=["lite", "full", "heavy"], default="full",
                   help="MediaPipe model size: lite is fastest, heavy most accurate")
    a.add_argument("--no-video", action="store_true", help="skip writing annotated.mp4")
    a.add_argument("--max-frames", type=int, default=None)

    args = p.parse_args(argv)

    if args.cmd == "analyze":
        from climbvision.pipeline import analyze
        from climbvision.pose import MediaPipePose

        for video in args.videos:
            out = args.out / video.stem
            print(f"[climbvision] {video} -> {out}/")
            # fresh estimator per video: VIDEO mode needs increasing timestamps
            with MediaPipePose(model_size=args.model) as est:
                summary = analyze(video, out, est, sex=args.sex,
                                  write_video=not args.no_video, max_frames=args.max_frames)
            print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
