# ClimbVision

Movement analysis from video: a generic engine (pose → canonical skeleton → centre of
mass, angles, contacts, support) plus **recipes** — small YAML files that say what to
measure for one movement. Off-the-wall movements work today; climbing is next.
See [PLAN.md](PLAN.md) for the milestones and **[docs/GUIDE.md](docs/GUIDE.md) for the
complete guide**: recording tips, every output explained, the science behind each
measurement, the architecture, and how to extend the package.

## Setup (once)

```bash
cd ClimbVision
conda create -n climbvision python=3.11 -y
conda activate climbvision
pip install -e ".[dev]"
```

## Run

```bash
climbvision analyze data/videos/squat_side.mp4
climbvision analyze data/videos/*.mp4 --sex female --model heavy
climbvision analyze data/videos/IMG_1234.MOV --recipe squat

# re-run the analysis on existing results (seconds), e.g. after changing a recipe
climbvision metrics outputs/*
# redraw the annotated video from existing results
climbvision render outputs/squat_side
```

The recipe is guessed from the file name, otherwise pass `--recipe` (a name or a
`.yaml` file; `climbvision recipes` lists them, `--recipes DIR` adds your own):

| `--recipe` | file names that match | what you get |
|---|---|---|
| `squat` | `squat` | reps, depth (thigh angle, below parallel), knee/hip angle, trunk lean, tempo |
| `pistol_squat` | `pistol` | the same, following the bending leg, plus reps per leg |
| `plank` | `plank` | hold time, hip sag / straight / pike, drift |
| `pike_pushup` | `pike` | push-ups done in a pike (elbow depth), time in plank and in pike |
| `jumping_jacks` | `jumping_jack`, `jacks` | reps (arms up and down), reps/min, how wide the feet go |
| `balance` | `tree`, `crow`, `onefoot`, `single_leg` | longest hold on a reduced base (one foot, hands), COM sway and margin during it |
| `pullup` | `pullup`, `chinup` | reps, COM rise (cm), chin over bar, tempo, swing, kipping |
| `leg_raise` | `leg_raise`, `toes_to_bar`, `knee_raise` | reps, hip/knee angle, toes to bar |
| `dead_hang` | `dead_hang`, `hang…` (not `overhang`) | hang time, arm straightness, swing |

Every clip also gets jumps (flight time, jump height from flight time), stances and
the stability margin. Judge squat, plank and pike form from **side-view** clips.

The pose model downloads automatically on the first run (~10–30 MB).

Each video gets a folder `outputs/<video name>/` with:

| File | What's in it |
|---|---|
| `annotated.mp4` | skeleton, COM (pink) with a 2-second trail, floor line, ground contacts (cyan rings), support span on the floor and the COM's plumb line — green when the COM is over the support, red when not |
| `landmarks.csv` | the pose backend's raw points per frame (MediaPipe: 33): pixel coords, 3D world coords (m), visibility |
| `com.csv` | whole-body COM per frame and a confidence score. `com_*_px` = position in the image; `com_*_m` = 3D position in metres **relative to the hip centre** (shows body shape, not where the body is in the room) |
| `summary.json` | detection rate, COM mean / spread / range |
| `angles.csv` | per-frame joint angles, both sides, 2D (image) and 3D versions |
| `metrics.json` | view (side/front, body yaw), near side, angle ranges, recipe results (table above) and `stability` |
| `stability.csv` | per frame: what's touching the floor, support span, COM-to-edge margin (cm) |
| `report.png` | angle plots, the recipe's signal with reps/holds, stability margin, contact timeline |

`metrics.json` → `stability` holds: contact time per body part, time in each stance
(both feet, L foot, feet + forearms, ...), the stability margin (median / min / % of
time inside), COM sway (spread, range, mean speed) overall and per stance, and jumps.
Stability is measured along the camera's horizontal axis: front–back in a side view,
left–right in a front view. **Feet must be in frame** — the report warns when they
aren't.

Options: `--no-video` (data only, faster), `--max-frames N`, `--out DIR`.

## Compare clips

```bash
climbvision summary outputs/*     # one line per clip + outputs/summary.csv
```

## Tests

```bash
pytest                      # synthetic tests + 20 real-clip regression tests (~5 s)
pytest -m "not regression"  # synthetic only
```

The regression clips live in `tests/regression/` (pose data only, no video); see
docs/GUIDE.md §8.7.

## Write your own analysis

```yaml
# pushup.yaml — then: climbvision metrics outputs/clip --recipe pushup.yaml
name: pushup
signals: {elbow: smooth(elbow, 1/6)}
reps:
  signal: elbow
  start_drop: 50
  end_drop: 25
  fields: {min_elbow_deg: at_bottom(elbow)}
  stats: [min_elbow_deg]
```

See [docs/RECIPES.md](docs/RECIPES.md) for everything a recipe can do.

## Layout

```
src/climbvision/
  engine/          the layers: video, pose + backends/, skeleton (canonical joints, no
                   face), track, biomech (COM, angles), scale, surfaces (floor),
                   contacts (floor, bar grips), support, events, signals
  recipes/         recipe runner + expression language; library/*.yaml = shipped recipes
  output/          report.png, CSV tables, drawing, annotated video, one-line summary
  analysis.py      results folder -> all analysis outputs
  cli.py           command line
tests/             synthetic + real-clip regression tests
docs/              GUIDE.md (complete guide), RECIPES.md (recipe reference)
data/videos/       your clips (git-ignored)
outputs/           results (git-ignored)
```
