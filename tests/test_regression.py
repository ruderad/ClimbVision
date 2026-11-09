"""Regression tests on real clips.

Each fixture in tests/regression/clips/ is the saved pose data of a real video
(landmarks + COM + summary). The test re-runs every analysis step on it and compares
with tests/regression/expected.json:

  eq        a verified fact (e.g. rep count seen in the video) — must match exactly
  approx    a snapshot of the current result — must stay within `tol`
  contains  a warning / text that must be present

If a change makes an `approx` check fail, look at the clip: if the new value is better,
refresh snapshots with `python tests/regression/build_fixtures.py --update-snapshots`.
Never edit an `eq` value to make a test pass unless the video shows it was wrong.

Run only these:  pytest -m regression      Skip them:  pytest -m "not regression"
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "regression"))
from build_fixtures import EXPECTED, get_path, unpack  # noqa: E402

from climbvision.metrics import run_metrics  # noqa: E402

SPECS = json.loads(EXPECTED.read_text())


def check(metrics: dict, c: dict) -> str | None:
    v = get_path(metrics, c["path"])
    name = ".".join(c["path"])
    if "eq" in c and v != c["eq"]:
        return f"{name}: expected {c['eq']!r}, got {v!r}"
    if "approx" in c:
        if v is None or abs(v - c["approx"]) > c["tol"]:
            return f"{name}: expected {c['approx']} ± {c['tol']}, got {v}"
    if "contains" in c:
        text = " ".join(v) if isinstance(v, list) else str(v or "")
        if c["contains"] not in text:
            return f"{name}: expected to contain {c['contains']!r}, got {v!r}"
    return None


@pytest.mark.regression
@pytest.mark.parametrize("clip", sorted(SPECS))
def test_clip(clip, tmp_path):
    spec = SPECS[clip]
    m = run_metrics(unpack(clip, tmp_path), spec["exercise"], plot=False)
    problems = [p for c in spec["checks"] if (p := check(m, c))]
    assert not problems, f"{clip} ({spec['verified']}):\n  " + "\n  ".join(problems)
