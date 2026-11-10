"""Helpers for synthetic tests: build poses with MediaPipe's 33 point names (as a real
backend would output them), then map them onto the canonical skeleton."""

from enum import IntEnum

from climbvision.engine.backends import mediapipe
from climbvision.engine.signals import Clip
from climbvision.engine.track import Track
from climbvision.recipes import load_recipes, run_recipe

LM = IntEnum("LM", [p.upper() for p in mediapipe.POINTS], start=0)
N_LANDMARKS = len(LM)
RECIPES = load_recipes()


def mp_track(time_s, image, world, visibility) -> Track:
    """Canonical track from MediaPipe-ordered arrays (n, 33, ...)."""
    return Track.from_raw(time_s, image, world, visibility, mediapipe.NAME)


def analyse(track: Track, recipe: str, frame_size=None) -> tuple[dict, Clip]:
    rec = RECIPES[recipe]
    clip = Clip.build(track, frame_size=frame_size, bar=rec.bar)
    return run_recipe(rec, clip).metrics, clip
