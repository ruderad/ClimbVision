# Recipes — reference

A recipe is a YAML file that describes one analysis: which signals to look at, what
counts as a rep or a hold, and what to report. The engine measures the body; the recipe
says what matters for one movement. The shipped recipes are in
`src/climbvision/recipes/library/` and are the best examples; this page is the complete
list of what a recipe can contain.

```bash
climbvision recipes                       # list recipes
climbvision signals                       # list the signals recipes can use
climbvision metrics outputs/clip --recipe squat
climbvision metrics outputs/clip --recipe ~/my_recipes/pushup.yaml
climbvision metrics outputs/* --recipes ~/my_recipes     # add a folder of recipes
```

## How a recipe runs

1. The engine builds the clip's layers (track, angles, contacts, support). `scene: bar`
   switches on grip detection first.
2. `signals:` are evaluated in order.
3. `reps:` finds repetitions and computes per-rep fields and statistics.
4. `summary:` adds clip-level results.
5. `holds:` finds the longest stretches where a condition holds.
6. The results go into `metrics.json` (together with the engine's `stability` block),
   `headline` gives the one-line result for `climbvision summary`, and `plot` draws the
   second panel of `report.png`.

## Top-level keys

| Key | Required | Meaning |
|---|---|---|
| `name` | yes | Recipe name (used by `--recipe` and stored as `exercise` in metrics.json) |
| `description` | | What it measures and how; shown by `climbvision recipes` |
| `match` | | List of regular expressions on the clip folder / video name (case-insensitive) |
| `priority` | | Order in which `match` is tried, lower first (default 100). First match wins |
| `scene` | | `floor` (default) or `bar`: detect hands gripping a bar; learn the floor without them |
| `signals` | | `{name: expression}`, evaluated in order |
| `reps` | | Repetition counting (below) |
| `summary` | | `{key: field}` clip-level results |
| `holds` | | List of hold detectors (below) |
| `headline` | | Expression (usually an f-string) for the one-line summary |
| `plot` | | `{signal, label, extra: [expr, label], rest_line: true, band: [lo, hi, label]}` |

## Expressions

Every value that computes something is an expression in a small, safe subset of Python:

- numbers, `'text'`, names (signals, helpers, earlier values), `None`/`none`, `True`/`False`
- `+ - * / // % **`, comparisons (also chained, and `is None`), `and or not`
- `&  |  ~` for element-wise *and / or / not* on per-frame boolean arrays
  (use them, not `and/or`, between arrays — and wrap comparisons in brackets:
  `(hip < 120) & on_bar`)
- `a if condition else b`
- calls to the helpers below; `x.key` or `x['key']` on dicts (None if missing)
- f-strings, e.g. `f"{reps} reps, knee {min_knee_deg_mean:.0f}°"` (None prints as `–`)

There are no imports, attribute access on objects, or loops: a recipe can only combine
what the engine provides.

Per-frame signals are NumPy arrays (one value per video frame, NaN where unknown).
Arithmetic on them is element-wise. Comparisons with NaN are False.

**YAML tips.** Quote an expression that starts with `[`, `{`, `'`, `"`, contains `: `,
or contains a comma inside `[expr, digits]`. For long expressions use `>-` and indent the
lines (folded into one line).

## Names available everywhere

| Name | Meaning |
|---|---|
| any signal | `knee`, `hip_l`, `elbow_r_3d`, `com_y_px`, `on_bar`, … (`climbvision signals`) |
| `t`, `dt` | time of each frame (s); frame duration (s) |
| `fps`, `n_frames` | frames per second; number of frames |
| `view`, `side`, `yaw_deg` | `'side'`/`'front'`; near side `'left'`/`'right'`; body yaw |
| `body_px`, `px_per_m` | body length (trunk + leg) in pixels; pixels per metre |
| `nan` | not-a-number |

### Element-wise helpers (return a per-frame array)

| Helper | Meaning |
|---|---|
| `smooth(x, seconds)` | centred moving average, NaN-aware (`smooth(knee, 1/6)`) |
| `avg(a, b, ...)` | NaN-aware mean of several signals, frame by frame |
| `where(cond, a, b)` | `a` where cond is true, else `b` |
| `abs(x)`, `isfinite(x)`, `nan_to_num(x, v)` | as in NumPy |

### Aggregates (one number)

Inside a rep or a hold, an aggregate of a full-length signal covers **only that rep or
hold**. Anywhere else it covers the whole clip. Aggregates ignore NaN.

| Helper | Meaning |
|---|---|
| `min(x)`, `max(x)`, `mean(x)`, `median(x)`, `sd(x)`, `ptp(x)` | statistics (`ptp` = max − min) |
| `any(mask)`, `all(mask)`, `count(mask)` | over a boolean signal |
| `pct(mask, where=None)` | % of frames where mask is true (optionally only among `where`) |
| `time(mask)` | seconds where mask is true |
| `sway(x)` | `{sd_cm, range_cm, mean_speed_cm_s}` of a cm signal (use with `merge`) |
| `percentile(x, q, where=None)` | q-th percentile of the whole signal |
| `finite(x)` | only the finite values |
| `head(x, k)`, `tail(x, k)` | first / last k frames of the window |
| `before(x, seconds)` | the signal from `seconds` before the rep start up to the start |
| `at(x, i)` | value at frame index i |
| `to_cm(px, i)` | pixels → cm with the scale at frame i |
| `frames(seconds)` | number of frames in a duration (at least 1) |
| `ceil`, `floor`, `int`, `float`, `str`, `bool`, `len` | as in Python |
| `plural(k, 'rep')` | `"1 rep"`, `"3 reps"` |

## Field values and rounding

`fields`, `summary` and hold fields map an output key to an expression. Numbers are
rounded to 1 decimal; write `[expression, 2]` for 2 decimals or `[expression, null]` for
none. True/False and text are not rounded. A value of `None` leaves the key out.

`let:` (in `reps` and holds) computes helper values *before* the fields, unrounded and
not written to the output. Each field is also visible to the fields after it, rounded.

## `reps:` — repetitions

A rep is a dip of a signal below its resting level (`engine/events.find_cycles`): it
starts when the signal falls more than `start_drop` below rest (walking back to where the
fall began) and ends when it comes back within `end_drop`. For a movement that *peaks*,
count dips of a flipped signal (`180 - arms`).

| Key | Meaning |
|---|---|
| `signal` | expression for the per-frame signal |
| `start_drop`, `end_drop` | thresholds, in the signal's units |
| `min_s` | shortest rep (default 0.6 s) |
| `rest` | resting level (expression); default = 90th percentile of the signal |
| `require` | condition a rep must meet, else it is ignored (`at_bottom(on_bar)`) |
| `let`, `fields` | per-rep values (see above) |
| `stats` | list of fields → `<field>_mean`, `<field>_sd` over reps (2 decimals) |
| `count` | `{key: boolean field}` → number of reps where it is true |
| `categories` | `{key: {field: f, values: [a, b]}}` → `{a: count, b: count}` |

Inside a rep: `t_start`, `t_bottom`, `t_end` (s), `i_start`, `i_bottom`, `i_end` (frame
indices), `rep` (number), `at_start(x)`, `at_bottom(x)`, `at_end(x)`, and the
aggregates over the rep. After the reps (in `summary`): `reps` (list of rep dicts), `n`
(count), `field('name')` (array of one field over the reps), `rest`.

Output: `reps` (count), the stats/counts/categories when there is at least one rep, and
`reps_detail` (list of every rep's fields).

## `summary:` — clip-level results

`{key: field}`, evaluated after `reps`. Example from jumping jacks:

```yaml
summary:
  reps_per_min: "60 * n / (field('end_s')[-1] - field('start_s')[0]) if n >= 2 else None"
```

## `holds:` — longest stretches

Each item finds one stretch:

| Key | Meaning |
|---|---|
| `when` | boolean per-frame expression: the longest run where it is true |
| `by` + `exclude` | instead of `when`: the longest run where this label signal (e.g. `stance`) keeps one value, ignoring the listed values |
| `min_s`, `min_frames` | the run must last at least this long (expressions allowed: `int(fps)`) |
| `let`, `fields` | values over the hold; also `merge: [expr, ...]` to add dicts (e.g. `sway(com_x_cm)`) |
| `nest_as` | put the results under this key instead of at the top level |
| `if_missing` | what to write when no run qualifies (omit to write nothing) |

Inside a hold: `t_start`, `duration` (s), `i_start`, `i_end`, `n_win` (frames), `label`
(with `by`), and the aggregates over the hold.

## Signals

`climbvision signals` prints the current list. The main groups:

| Group | Names |
|---|---|
| Angles | `knee hip ankle elbow shoulder body_line thigh trunk_lean` (best for the view); `_l`/`_r` one side; `_l_2d`, `_r_3d`, … raw |
| Body geometry | `hip_below_knee_cm` (`_l`, `_r`), `hip_offset_pct`, `feet_width`, `shoulder_y`, `toe_y` |
| Centre of mass | `com_x_px`, `com_y_px`, `com_x_cm` |
| Support | `margin_cm`, `on_floor`, `stance`, `ppm` |
| Bar | `on_bar`, `hanging`, `swing_cm`, `bar_y` |

Image y points **down**: a smaller y is higher.

A measurement that no signal provides is a few lines in `engine/signals.py` (GUIDE 8.4).
Keep signals generic ("hip below knee"), and put movement-specific logic in recipes.

## Worked examples

| Recipe | Shows |
|---|---|
| `squat.yaml` | signals with smoothing, reps, per-rep fields, stats, count, rest line |
| `pistol_squat.yaml` | choosing the working leg per frame with `where`, categories |
| `pike_pushup.yaml` | custom `rest`, `require`, `time(...)` in the summary |
| `jumping_jacks.yaml` | counting peaks via a flipped signal; summary from rep fields |
| `pullup.yaml` | `scene: bar`, `let`, `before`, `to_cm`, text categories |
| `leg_raise.yaml` | reps and a nested hold with `head`/`tail` |
| `plank.yaml` | a hold with `min_frames`, `pct`, `if_missing`, a band in the plot |
| `balance.yaml` | a hold `by: stance`, `merge: [sway(...)]` |
| `dead_hang.yaml` | a hold on an engine mask, conditional fields |
