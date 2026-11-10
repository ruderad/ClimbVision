"""Recipes: analyses described as data (YAML), run on the engine's signals.

A recipe says *what* to measure for one kind of movement; the engine says *how* to
measure the body. Adding an exercise or a climbing analysis should mean writing a
recipe, not engine code. Shipped recipes live in recipes/library/*.yaml; more can be
loaded from any folder (`--recipes DIR`).

Sections (all optional except name):

  name, description
  match:     [regex, ...]  file-name patterns for --recipe auto; `priority` orders
             recipes (lower first; first match wins)
  scene:     floor (default) | bar | wall   (bar: detect grips, learn the floor without
             them; wall: anchors on holds in wall coordinates, wall support)
  signals:   {name: expr}  per-frame series, in order; later ones can use earlier ones
  reps:      repeated movements = dips of a signal below its resting level
             (signal, start_drop, end_drop, min_s, rest, require, let, fields,
              stats, count, categories)
  summary:   {key: expr}   clip-level results (None = left out)
  holds:     [ {when | by, min_s, min_frames, nest_as, if_missing, let, fields, merge} ]
             the longest stretch where a condition holds (or one label stays the same)
  headline:  expr (usually an f-string) giving the one-line result for `summary`
  plot:      what to draw in the report's second panel

Field values are expressions (see recipes/expr.py), rounded to 1 decimal; write
`[expr, 2]` for 2 decimals or `[expr, null]` for no rounding. Booleans and text are
never rounded. In a rep or hold, aggregates (min, max, mean, median, sd, ptp, any,
all, pct, time, sway) of a full-length signal cover only that rep/hold.
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from climbvision.engine.events import find_cycles, longest_run, runs
from climbvision.engine.signals import Clip
from climbvision.engine.support import sway as _sway
from climbvision.engine.track import smooth as _smooth
from climbvision.engine.track import window as _window
from climbvision.recipes.expr import ExprError, evaluate

LIBRARY = Path(__file__).parent / "library"


@dataclass
class Recipe:
    name: str
    spec: dict
    path: Path | None = None

    @property
    def priority(self) -> int:
        return int(self.spec.get("priority", 100))

    @property
    def bar(self) -> bool:
        return self.spec.get("scene", "floor") == "bar"

    @property
    def scene(self) -> str:
        return str(self.spec.get("scene", "floor"))

    def matches(self, clip_name: str) -> bool:
        return any(re.search(p, clip_name, re.IGNORECASE) for p in self.spec.get("match", []))


# floor movements without a recipe of their own: no specific analysis
NONE = Recipe("none", {"name": "none", "description": "no movement-specific analysis",
                       "match": ["lunge|standing_still|walk|stretch"], "priority": 200})
DEFAULT_RECIPE = "climb"  # a clip whose name matches no recipe is a climb


def load_recipe(path: str | Path) -> Recipe:
    import yaml

    path = Path(path)
    spec = yaml.safe_load(path.read_text())
    if not isinstance(spec, dict) or "name" not in spec:
        raise ValueError(f"{path}: a recipe needs at least a `name`")
    return Recipe(spec["name"], spec, path)


def load_recipes(extra_dirs: list[Path] | None = None) -> dict[str, Recipe]:
    """Shipped recipes, then any in extra_dirs (same name = override)."""
    out = {"none": NONE}
    for d in [LIBRARY, *(extra_dirs or [])]:
        for p in sorted(Path(d).glob("*.y*ml")):
            r = load_recipe(p)
            out[r.name] = r
    return out


def guess_recipe(clip_name: str, recipes: dict[str, Recipe]) -> Recipe:
    """The recipe whose `match` fits the clip name (lowest priority number first);
    a name that matches nothing is a climb (DEFAULT_RECIPE)."""
    for r in sorted(recipes.values(), key=lambda r: (r.priority, r.name)):
        if r.matches(clip_name):
            return r
    return recipes.get(DEFAULT_RECIPE, NONE)


# ------------------------------------------------------------------ evaluation context

def _quiet(fn):
    def wrapped(*a, **k):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return fn(*a, **k)
    wrapped.__name__ = fn.__name__
    return wrapped


class Scope(Mapping):
    """Names visible to an expression: local values, then helpers, then signals."""

    def __init__(self, clip: Clip, values: dict, win: slice | None = None):
        self.clip = clip
        self.values = values
        self.win = win or slice(0, clip.track.n)
        self.helpers = _helpers(self)

    def child(self, win: slice, extra: dict) -> Scope:
        return Scope(self.clip, {**self.values, **extra}, win)

    def __getitem__(self, key: str) -> Any:
        if key in self.values:
            return self.values[key]
        if key in self.helpers:
            return self.helpers[key]
        try:
            return self.clip.signal(key)
        except KeyError:
            raise KeyError(key) from None

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return key in self.values or key in self.helpers or self.clip.has(key)

    def __iter__(self) -> Iterator[str]:
        return iter({**self.helpers, **self.values})

    def __len__(self) -> int:
        return len(self.values) + len(self.helpers)

    def ev(self, src: Any) -> Any:
        return evaluate(src, self)


def _helpers(sc: Scope) -> dict[str, Any]:
    clip, win = sc.clip, sc.win
    t = clip.t
    n = clip.track.n
    fps = clip.track.fps
    dt = np.gradient(t) if n > 1 else np.zeros(n)

    def w(x):
        """Restrict a full-length signal to the current window."""
        if isinstance(x, np.ndarray) and x.ndim >= 1 and len(x) == n:
            return x[win]
        return np.asarray(x) if isinstance(x, list | tuple) else x

    def agg(f, empty=float("nan")):
        @_quiet
        def g(x):
            v = np.asarray(w(x), dtype=float)
            return float(f(v)) if v.size else empty
        return g

    @_quiet
    def percentile(x, q, where=None):
        x = np.asarray(x, float)
        if where is not None:
            x = x[np.asarray(where, bool)]
        return float(np.nanpercentile(x, q)) if np.isfinite(x).any() else float("nan")

    @_quiet
    def avg(*xs):
        return np.nanmean(np.stack([np.asarray(x, float) for x in xs]), axis=0)

    def pct(mask, where=None):
        m = np.asarray(w(mask), bool)
        if where is not None:
            m = m[np.asarray(w(where), bool)]
        return 100.0 * float(np.mean(m)) if m.size else float("nan")

    def head(x, k):
        return w(x)[: int(k)]

    def tail(x, k):
        v = w(x)
        return v[len(v) - int(k):] if int(k) > 0 else v[:0]

    def before(x, seconds):
        s = win.start
        return np.asarray(x)[max(0, s - int(fps * seconds)): s + 1]

    def at(x, i):
        return np.asarray(x)[int(i)]

    def to_cm(px, i):
        return px * 100.0 / clip.signal("ppm")[int(i)]

    def finite(x):
        v = np.asarray(w(x), float)
        return v[np.isfinite(v)]

    return {
        "fps": fps, "n_frames": n, "nan": float("nan"), "t": t, "dt": dt, "view": clip.angles.view,
        "side": clip.angles.side, "yaw_deg": clip.angles.yaw_deg,
        "body_px": clip.stability.scale.body_px, "px_per_m": clip.stability.scale.px_per_m,
        # element-wise
        "smooth": lambda x, seconds: _smooth(x, _window(fps, seconds)),
        "avg": avg, "where": np.where, "isfinite": np.isfinite, "abs": np.abs,
        "nan_to_num": lambda x, v=0.0: np.nan_to_num(x, nan=v),
        "ceil": math.ceil, "floor": math.floor, "int": int, "float": float, "str": str,
        "bool": bool, "len": len, "frames": lambda s: max(1, int(fps * s)),
        "percentile": percentile, "to_cm": to_cm, "at": at,
        "plural": lambda k, word, words=None: f"{k} {word if k == 1 else (words or word + 's')}",
        # over the current window
        "min": agg(np.nanmin), "max": agg(np.nanmax), "mean": agg(np.nanmean),
        "median": agg(np.nanmedian), "sd": agg(np.nanstd),
        "ptp": agg(lambda v: np.nanmax(v) - np.nanmin(v)),
        "any": lambda x: bool(np.any(np.asarray(w(x), bool))),
        "all": lambda x: bool(np.all(np.asarray(w(x), bool))),
        "count": lambda x: int(np.sum(np.asarray(w(x), bool))),
        "pct": pct, "head": head, "tail": tail, "before": before, "finite": finite,
        "time": lambda mask: float(dt[win][np.asarray(w(mask), bool)].sum()),
        "sway": lambda x: _sway(w(x), t[win]),
    }


def _out(v: Any, digits: int | None = 1) -> Any:
    if v is None or isinstance(v, str | dict | list):
        return v
    if isinstance(v, bool | np.bool_):
        return bool(v)
    if isinstance(v, int | np.integer):
        return int(v)
    v = float(v)
    return v if digits is None else round(v, digits)


def _field(sc: Scope, spec: Any, default_digits: int = 1) -> Any:
    if isinstance(spec, list):
        if len(spec) != 2:
            raise ExprError(f"[expr, digits] expected, got {spec!r} (quote an expression "
                            "that contains commas)")
        expr, digits = spec
        return _out(sc.ev(expr), digits)
    return _out(sc.ev(spec), default_digits)


def _fields(sc: Scope, spec: dict) -> dict:
    """let (unrounded helpers) then fields, each visible to the next."""
    for k, e in (spec.get("let") or {}).items():
        sc.values[k] = sc.ev(e)
    out = {}
    for k, e in (spec.get("fields") or {}).items():
        v = _field(sc, e)
        sc.values[k] = v
        if v is not None:
            out[k] = v
    for e in spec.get("merge") or []:
        out.update(sc.ev(e) or {})
    return out


# ------------------------------------------------------------------ running a recipe

@dataclass
class Result:
    metrics: dict
    plot: dict = field(default_factory=dict)  # arrays and spans for the report


def run_recipe(recipe: Recipe, clip: Clip) -> Result:
    spec = recipe.spec
    sc = Scope(clip, {})
    try:
        for name, e in (spec.get("signals") or {}).items():
            sc.values[name] = sc.ev(e)
        out: dict = {}
        plot: dict = {"rep_spans": [], "hold_spans": []}
        if "reps" in spec:
            _run_reps(sc, spec["reps"], out, plot)
        for k, e in (spec.get("summary") or {}).items():
            v = _field(sc, e)
            if v is not None:
                out[k] = v
        for h in spec.get("holds") or []:
            _run_hold(sc, h, out, plot)
        if "reps" in spec:
            out["reps_detail"] = out.pop("reps_detail")
        if spec.get("plot"):
            p = spec["plot"]
            if p.get("signal"):
                plot["signal"] = np.asarray(sc.ev(p["signal"]), float)
            if p.get("extra"):
                plot["extra"] = np.asarray(sc.ev(p["extra"][0]), float)
            if p.get("rest_line") and "rest" in sc.values:
                plot["rest"] = sc.values["rest"]
        return Result(out, plot)
    except ExprError as e:
        raise ExprError(f"recipe {recipe.name!r}: {e}") from None


def _run_reps(sc: Scope, rs: dict, out: dict, plot: dict) -> None:
    clip = sc.clip
    t = clip.t
    signal = np.asarray(sc.ev(rs["signal"]), float)
    rest = sc.ev(rs["rest"]) if rs.get("rest") is not None else \
        sc.helpers["percentile"](signal, 90)
    sc.values["rest"] = rest
    reps = []
    for s, b, e in find_cycles(signal, t, float(rs["start_drop"]), float(rs["end_drop"]),
                               float(rs.get("min_s", 0.6)), rest):
        extra = {"i_start": s, "i_bottom": b, "i_end": e, "t_start": float(t[s]),
                 "t_bottom": float(t[b]), "t_end": float(t[e]), "rep": len(reps) + 1,
                 "at_start": lambda x, i=s: np.asarray(x)[i],
                 "at_bottom": lambda x, i=b: np.asarray(x)[i],
                 "at_end": lambda x, i=e: np.asarray(x)[i]}
        rsc = sc.child(slice(s, e + 1), extra)
        if rs.get("require") and not bool(rsc.ev(rs["require"])):
            continue
        reps.append({"rep": len(reps) + 1, **_fields(rsc, rs)})
        plot["rep_spans"].append((float(t[s]), float(t[e]), float(t[b])))
    out["reps"] = len(reps)
    sc.values["n"] = len(reps)
    sc.values["reps"] = reps
    sc.values["field"] = lambda k: np.array([r.get(k, np.nan) for r in reps], dtype=float)
    if reps:
        for k in rs.get("stats") or []:
            v = sc.values["field"](k)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out[f"{k}_mean"] = round(float(np.nanmean(v)), 2)
                out[f"{k}_sd"] = round(float(np.nanstd(v)), 2)
        for k, f in (rs.get("count") or {}).items():
            out[k] = int(sum(bool(r.get(f)) for r in reps))
        for k, c in (rs.get("categories") or {}).items():
            out[k] = {v: sum(r.get(c["field"]) == v for r in reps) for v in c["values"]}
    out["reps_detail"] = reps


def _run_hold(sc: Scope, h: dict, out: dict, plot: dict) -> None:
    clip = sc.clip
    t = clip.t
    min_s = float(sc.ev(h.get("min_s", 0.0)))
    min_frames = int(sc.ev(h.get("min_frames", 1)))
    best, label = None, None
    if "by" in h:  # longest run of one constant label
        labels = np.asarray(sc.ev(h["by"]), dtype=object)
        skip = set(h.get("exclude") or [])
        for lab in sorted(set(labels.tolist()) - skip):
            r = longest_run(labels == lab, t, min_s, min_frames)
            if r and (best is None or r[1] - r[0] > best[1] - best[0]):
                best, label = r, lab
    else:
        mask = np.asarray(sc.ev(h["when"]), bool)
        best = longest_run(mask, t, min_s, min_frames)
    if best is None:
        if "if_missing" not in h:
            return
        res = dict(h["if_missing"] or {})
    else:
        a, b = best
        extra = {"i_start": a, "i_end": b - 1, "t_start": float(t[a]),
                 "duration": float(t[b - 1] - t[a]), "n_win": b - a, "label": label}
        res = _fields(sc.child(slice(a, b), extra), h)
        plot["hold_spans"].append((float(t[a]), float(t[b - 1]), label))
    if h.get("nest_as"):
        out[h["nest_as"]] = res
    else:
        out.update(res)


__all__ = ["DEFAULT_RECIPE", "LIBRARY", "NONE", "Recipe", "Result", "guess_recipe", "load_recipe",
           "load_recipes", "run_recipe", "runs"]
