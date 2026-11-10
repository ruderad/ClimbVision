"""Build / refresh the regression fixtures from analysed clips.

    python tests/regression/build_fixtures.py outputs/squat_side outputs/plank_side ...
        copies each clip's pose data (landmarks.csv, com.csv, summary.json) into
        tests/regression/clips/<clip>/ as .gz files. Pose data is the input of every
        analysis step, so the tests re-run all metrics without video or MediaPipe.

    python tests/regression/build_fixtures.py --update-snapshots
        re-runs the metrics on every fixture and rewrites the "approx" values in
        expected.json to the current results (tolerances are kept). Do this only after
        you have checked that a change in results is an improvement. "eq" checks are
        verified facts (rep counts seen in the video...) and are never auto-updated.
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).parent
CLIPS = HERE / "clips"
EXPECTED = HERE / "expected.json"
FILES = ("landmarks.csv", "com.csv", "summary.json")
# wall clips also need the camera layer (computed from the video, which isn't kept)
OPTIONAL = ("camera.csv", "camera.json", "wall.jpg", "wall.json")
GZ = (".csv",)


def add_clip(result_dir: Path) -> None:
    dest = CLIPS / result_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    for name in FILES + OPTIONAL:
        src = result_dir / name
        if name in OPTIONAL and not src.exists():
            continue
        if name.endswith(GZ):
            with src.open("rb") as fi, gzip.open(dest / f"{name}.gz", "wb", compresslevel=9) as fo:
                shutil.copyfileobj(fi, fo)
        else:
            shutil.copy(src, dest / name)
    print(f"fixture: {dest}")


def unpack(clip: str, workdir: Path) -> Path:
    """Decompress a fixture into workdir/<clip>/ and return that folder."""
    out = workdir / clip
    out.mkdir(parents=True, exist_ok=True)
    src = CLIPS / clip
    for name in FILES + OPTIONAL:
        if name.endswith(GZ):
            if not (src / f"{name}.gz").exists():
                continue
            with gzip.open(src / f"{name}.gz", "rb") as fi, (out / name).open("wb") as fo:
                shutil.copyfileobj(fi, fo)
        elif (src / name).exists():
            shutil.copy(src / name, out / name)
    return out


def get_path(d, path):
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


def update_snapshots() -> None:
    import tempfile

    from climbvision.analysis import run_analysis

    expected = json.loads(EXPECTED.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        for clip, spec in expected.items():
            m = run_analysis(unpack(clip, Path(tmp)), spec["exercise"], plot=False)
            for check in spec["checks"]:
                if "approx" in check:
                    v = get_path(m, check["path"])
                    print(f"{clip:28s} {'.'.join(check['path']):45s} {check['approx']} -> {v}")
                    check["approx"] = v
    EXPECTED.write_text(json.dumps(expected, indent=2, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("results", nargs="*", type=Path)
    p.add_argument("--update-snapshots", action="store_true")
    a = p.parse_args(argv)
    for r in a.results:
        add_clip(r)
    if a.update_snapshots:
        update_snapshots()
    return 0


if __name__ == "__main__":
    sys.exit(main())
