"""Run an analysis on a results folder: landmarks.csv -> angles.csv, stability.csv,
metrics.json, report.png. Fast (seconds): no video or pose model needed.

    clip = load_clip(folder, bar=...)      # engine layers
    result = run_recipe(recipe, clip)      # the recipe's measurements
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from climbvision.engine.signals import Clip
from climbvision.engine.support import stability_summary
from climbvision.engine.track import Track
from climbvision.recipes import Recipe, guess_recipe, load_recipes, run_recipe


def clip_info(result_dir: Path) -> dict:
    p = Path(result_dir) / "summary.json"
    return json.loads(p.read_text()) if p.exists() else {}


def load_clip(result_dir: str | Path, bar: bool = False) -> Clip:
    """All engine layers for a results folder. The camera layer (camera.csv) is used
    when present; otherwise the camera is assumed fixed."""
    from climbvision.engine.camera import CameraTrack, WallImage

    result_dir = Path(result_dir)
    info = clip_info(result_dir)
    track = Track.from_csv(result_dir / "landmarks.csv")
    cam = wall = None
    if (result_dir / "camera.csv").exists():
        cam = CameraTrack.load(result_dir / "camera.csv")
    if (result_dir / "wall.jpg").exists() and (result_dir / "wall.json").exists():
        wall = WallImage.load(result_dir)
    return Clip.build(track, info.get("sex_model", "male"),
                      tuple(info["frame_size"]) if "frame_size" in info else None, bar=bar,
                      camera=cam, wall_image=wall)


def resolve_recipe(result_dir: Path, recipe: str | Recipe = "auto",
                   recipe_dirs: list[Path] | None = None) -> Recipe:
    if isinstance(recipe, Recipe):
        return recipe
    recipes = load_recipes(recipe_dirs)
    if recipe == "auto":
        return guess_recipe(Path(result_dir).name, recipes)
    if Path(recipe).suffix in (".yaml", ".yml") and Path(recipe).exists():
        from climbvision.recipes import load_recipe

        return load_recipe(recipe)
    if recipe not in recipes:
        raise ValueError(f"unknown recipe {recipe!r}; available: {', '.join(sorted(recipes))}")
    return recipes[recipe]


def angle_ranges(clip: Clip) -> dict:
    out = {}
    for j in ("knee", "hip", "ankle", "thigh", "elbow", "shoulder", "body_line", "trunk_lean"):
        v = clip.angles.primary(j)
        if np.isfinite(v).any():
            out[j] = {"min": round(float(np.nanmin(v)), 1),
                      "median": round(float(np.nanmedian(v)), 1),
                      "max": round(float(np.nanmax(v)), 1)}
    return out


def run_analysis(result_dir: str | Path, recipe: str | Recipe = "auto", plot: bool = True,
                 recipe_dirs: list[Path] | None = None) -> dict:
    from climbvision.output.tables import write_angles_csv, write_stability_csv

    result_dir = Path(result_dir)
    rec = resolve_recipe(result_dir, recipe, recipe_dirs)
    clip = load_clip(result_dir, bar=rec.bar)
    write_angles_csv(result_dir / "angles.csv", clip.angles)
    write_stability_csv(result_dir / "stability.csv", clip.stability, clip.t)
    res = run_recipe(rec, clip)
    out: dict = {"exercise": rec.name, "view": clip.angles.view,
                 "body_yaw_deg": clip.angles.yaw_deg, "near_side": clip.angles.side,
                 "angle_ranges": angle_ranges(clip), **res.metrics}
    if rec.scene == "wall":
        from climbvision.engine.wall import wall_summary
        from climbvision.output.tables import write_wall_csv

        out["wall"] = wall_summary(clip.wall_support, clip.t)
        write_wall_csv(result_dir / "wall.csv", clip.wall_support, clip.t)
    else:
        out["stability"] = stability_summary(clip.stability, clip.t)
    if clip.camera is not None:
        from climbvision.engine.camera import camera_summary

        out["camera"] = camera_summary(clip.camera, clip.stability.scale.px_per_m)
    if plot:
        from climbvision.output.report import plot_report

        plot_report(result_dir / "report.png", clip, rec, res)
    (result_dir / "metrics.json").write_text(json.dumps(out, indent=2))
    return out
