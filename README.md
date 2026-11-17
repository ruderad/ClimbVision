# ClimbVision

Movement analysis from video: a generic engine (pose → canonical skeleton → centre of
mass, angles, contacts, support) plus **recipes** — small YAML files that say what to
measure for one movement. It analyses floor exercises and climbing: handheld wall
video, holds, moves, the route, rests and balance on the wall, technique (quiet feet,
straight arms, hesitation, smoothness) and the load on each hand and foot, in your real
units once it knows who is climbing. See [CHANGELOG.md](CHANGELOG.md).
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
climbvision analyze data/videos            # every video in the folder

# re-run the analysis on existing results (seconds), e.g. after changing a recipe
climbvision metrics outputs/*
# redraw the annotated video from existing results
climbvision render outputs/squat_side
climbvision render outputs                 # every results folder
# camera layer for clips analysed before it existed (handheld wall videos)
climbvision camera outputs/traverse
climbvision --version                      # version + where the code is loaded from
```

If a video file is replaced after it was analysed, every command warns: re-run
`analyze` on it (old results don't line up with the new video).

**Who is climbing.** Copy `climbers.example.yaml` to `data/climbers.yaml` (git-ignored:
it stays on your computer) and fill in height, mass and arm span. Then

```bash
climbvision analyze data/videos/boulder.mov --climber alex   # or: metrics ... --climber
climbvision metrics outputs/someone_elses_clip --climber unknown   # an average adult
```

The name is stored with the clip. Height sets every distance, height and speed; mass
gives forces in newtons.

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
| `climb` | `climb`, `traverse`, `boulder`, `moonboard`, `overhang`, `coordination`, `route`, `slab`, `crimp`, `dyno` — and **any name that matches nothing else** | hands/feet on holds (still ~0.5 s on the wall), holds used, the route move by move (`moves.csv`, `route.png`), rests, feet cuts, COM over the feet and inside the base of support, heights above the floor and COM speed, technique (quiet feet, straight arms, pause before moves, COM path), load on each hand and foot (`forces.csv`) |
| `none` | `lunge`, `standing_still`, `walk`, `stretch` | angles, stability and jumps only |

Every clip also gets jumps (flight time, jump height from flight time), stances and
the stability margin. Judge squat, plank and pike form from **side-view** clips.

The MediaPipe pose model downloads automatically on the first run (~10–30 MB). For
climbing videos install `pip install -e ".[dev,rtm]"` and put the RTMW-x and YOLOX-m
model zips in `models/` (links in docs/GUIDE.md §1.1): climbing recipes then use the
whole-body model, which finds hands and feet much better on the wall. Optional: a
climbing-hold detector (`yolov8n-freeclimbs-detect-2-fp32.onnx`, §1.1) in `models/` tells
holds from smears (a foot or palm pressed on the bare wall). `--fast` skips the
flip averaging of the climbing models (about 2x faster pose, a bit less steady).

Each video gets a folder `outputs/<video name>/` with:

| File | What's in it |
|---|---|
| `viewer.html` + `viewer.mp4` | the interactive viewer: the clip with overlays you switch on and off (presets clean / balance / load / dynamics / all: forces on each hand and foot, COM velocity, a timeline of contacts, moves and load), slow motion, frame stepping. Open in any browser; `climbvision view` redoes it |
| `annotated.mp4` | the `balance` preset burnt in (`climbvision render --preset dynamics` etc. for others): skeleton, COM with a 2-second trail (green = inside the base of support, red = outside, pink = no base), floor line, ground contacts (cyan rings), support span on the floor and the COM's plumb line. On the wall: limbs on holds, smears (dashed white rings), the base of support (yellow), the latest move. The trunk is the shoulder bar, hip bar and a curved spine; grey joints are hidden or guessed |
| `landmarks.csv` | the pose backend's raw points per frame (MediaPipe: 33): pixel coords, 3D world coords (m), visibility |
| `com.csv` | whole-body COM per frame and a confidence score. `com_*_px` = position in the image; `com_*_m` = 3D position in metres **relative to the hip centre** (shows body shape, not where the body is in the room) |
| `summary.json` | detection rate, COM mean / spread / range |
| `angles.csv` | per-frame joint angles, both sides, 2D (image) and 3D versions |
| `metrics.json` | view (side/front, body yaw), near side, angle ranges, recipe results (table above) and `stability` |
| `stability.csv` | per frame: what's touching the floor, support span, COM-to-edge margin (cm) |
| `camera.csv`, `wall.jpg`, `wall_view.mp4` | the camera layer: each frame lined up with the wall (handheld video), the wall without the climber, and the video in wall coordinates |
| `report.png` | angle plots, the recipe's signal with reps/holds, stability margin, contact timeline (on the wall: also the load on hands and feet, and holds, moves, rests, feet cuts) |
| `route.png`, `moves.csv`, `holds.csv`, `wall.csv` | wall clips: the route map (with a technique summary), every move, every hold, per-frame contacts on the wall |
| `forces.csv` | wall clips: per frame, the force on each hand and foot (% body weight; newtons with the mass) |

`metrics.json` → `stability` holds: contact time per body part, time in each stance
(both feet, L foot, feet + forearms, ...), the stability margin (median / min / % of
time inside), COM sway (spread, range, mean speed) overall and per stance, and jumps.
Stability is measured along the camera's horizontal axis: front–back in a side view,
left–right in a front view. **Feet must be in frame** — the report warns when they
aren't.

Options: `--no-video` (data only, faster), `--no-camera` (skip the camera layer),
`--max-frames N`, `--out DIR`.

## Compare clips

```bash
climbvision summary outputs/*     # one line per clip + outputs/summary.csv
```

## Tests

```bash
pytest                      # synthetic tests + 26 real-clip regression tests (~45 s)
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
  engine/          the layers: video, camera (wall registration), pose + backends/,
                   cleanup (pose repair), skeleton (canonical joints, no face), track,
                   biomech (COM, angles), scale, surfaces (floor), contacts (floor,
                   bar grips), support, wall (anchors, holds), route (moves, rests),
                   events, signals
  recipes/         recipe runner + expression language; library/*.yaml = shipped recipes
  output/          report.png, CSV tables, drawing, annotated video, one-line summary
  analysis.py      results folder -> all analysis outputs
  cli.py           command line
tests/             synthetic + real-clip regression tests
docs/              GUIDE.md (complete guide), RECIPES.md (recipe reference)
data/videos/       your clips (git-ignored)
outputs/           results (git-ignored)
```
