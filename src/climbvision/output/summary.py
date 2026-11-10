"""One line per clip: `climbvision summary outputs/*` -> table + summary.csv."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def headline(m: dict, recipes: dict | None = None) -> str:
    """The main result of a clip in a few words: the recipe's `headline` expression,
    evaluated on its metrics; otherwise the main stances (and jumps)."""
    from climbvision.recipes import load_recipes
    from climbvision.recipes.expr import evaluate

    recipes = recipes if recipes is not None else load_recipes()
    rec = recipes.get(m.get("exercise", "none"))
    st = m.get("stability", {})
    if rec is not None and rec.spec.get("headline"):
        names = {**m, "plural": lambda k, w, ws=None: f"{k} {w if k == 1 else (ws or w + 's')}"}
        try:
            return str(evaluate(rec.spec["headline"], names, missing_is_none=True))
        except Exception as e:  # noqa: BLE001 - a broken headline must not stop the table
            return f"(headline error: {e})"
    stances = list(st.get("stance_time_s", {}).items())[:2]
    s = ", ".join(f"{k} {v:.1f} s" for k, v in stances)
    if st.get("jumps"):
        s += f"; {st['jumps']['count']} jumps"
    return s or "–"


COLUMNS = ["clip", "exercise", "view", "result", "margin_median_cm", "sway_sd_cm",
           "warnings"]


def summarize_results(dirs: list[Path], recipe_dirs: list[Path] | None = None) -> list[dict]:
    from climbvision.recipes import load_recipes

    recipes = load_recipes(recipe_dirs)
    rows = []
    for d in dirs:
        p = d / "metrics.json"
        if not p.is_file():
            continue
        m = json.loads(p.read_text())
        st = m.get("stability", {})
        rows.append({
            "clip": d.name,
            "exercise": m.get("exercise", ""),
            "view": m.get("view", ""),
            "result": headline(m, recipes),
            "margin_median_cm": (st.get("margin_cm") or {}).get("median"),
            "sway_sd_cm": (st.get("com_sway") or {}).get("sd_cm"),
            "warnings": "; ".join(st.get("warnings", [])),
        })
    return rows


def write_summary(rows: list[dict], csv_path: Path) -> None:
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def markdown_table(rows: list[dict]) -> str:
    cols = ["clip", "exercise", "view", "result", "warnings"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        warn = "⚠ " + r["warnings"][:60] if r["warnings"] else ""
        lines.append("| " + " | ".join(str(x) for x in
                                       (r["clip"], r["exercise"], r["view"], r["result"],
                                        warn)) + " |")
    return "\n".join(lines)
