"""Climber profiles: who is climbing, so measurements are in real units.

`data/climbers.yaml` (git-ignored: it holds personal data) lists climbers:

    default: alex             # used when a clip doesn't name a climber
    climbers:
      alex:
        height_m: 1.78
        mass_kg: 70           # for forces (1.3)
        arm_span_m: 1.82      # ape index = arm span - height
        sex: male             # which segment-mass table (De Leva)

A clip is tied to a climber with `--climber NAME` (`analyze` or `metrics`), stored in
its `summary.json`. Without a profile the engine assumes an average 1.75 m adult;
`--climber unknown` says so explicitly (e.g. someone else on the wall while your profile
is the default).
The file is looked up at $CLIMBVISION_CLIMBERS, else `data/climbers.yaml` in the
project folder.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from climbvision.engine.scale import ASSUMED_BODY_M

BODY_RATIO = 0.779  # trunk + leg / stature (Drillis & Contini; see scale.ASSUMED_BODY_M)
ASSUMED_HEIGHT_M = round(ASSUMED_BODY_M / BODY_RATIO, 2)
UNKNOWN = ("unknown", "none")  # --climber unknown: an average adult, whatever the default


@dataclass
class Climber:
    name: str = "unknown"
    height_m: float | None = None
    mass_kg: float | None = None
    arm_span_m: float | None = None
    sex: str | None = None

    @property
    def body_m(self) -> float:
        """Trunk + leg length (the engine's ruler), from the height."""
        return BODY_RATIO * self.height_m if self.height_m else ASSUMED_BODY_M

    @property
    def ape_index_cm(self) -> float | None:
        if self.height_m and self.arm_span_m:
            return round(100 * (self.arm_span_m - self.height_m), 1)
        return None

    def summary(self) -> dict:
        out: dict = {"climber": self.name,
                     "height_m": self.height_m or ASSUMED_HEIGHT_M,
                     "height_assumed": not self.height_m}
        if self.mass_kg:
            out["mass_kg"] = self.mass_kg
        if self.ape_index_cm is not None:
            out["ape_index_cm"] = self.ape_index_cm
        return out


def climbers_file() -> Path:
    env = os.environ.get("CLIMBVISION_CLIMBERS")
    if env:
        return Path(env)
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent / "data" / "climbers.yaml"
    return Path("data") / "climbers.yaml"


def load_climbers(path: str | Path | None = None) -> tuple[dict[str, Climber], str | None]:
    """All climbers in the file, and the default name (None if there's no file)."""
    import yaml

    path = Path(path) if path else climbers_file()
    if not path.exists():
        return {}, None
    spec = yaml.safe_load(path.read_text()) or {}
    out = {}
    for name, d in (spec.get("climbers") or {}).items():
        d = d or {}
        out[str(name)] = Climber(str(name), _num(d.get("height_m")), _num(d.get("mass_kg")),
                                 _num(d.get("arm_span_m")), d.get("sex"))
    return out, spec.get("default")


def _num(v) -> float | None:
    return float(v) if v is not None else None


def find_climber(name: str | None = None, path: str | Path | None = None) -> Climber:
    """The named climber (or the file's default); an unknown name raises."""
    if name in UNKNOWN:
        return Climber()
    climbers, default = load_climbers(path)
    name = name or default
    if not name:
        return Climber()
    if name not in climbers:
        raise ValueError(f"climber {name!r} is not in {path or climbers_file()} "
                         f"(known: {', '.join(sorted(climbers)) or 'none'})")
    return climbers[name]
