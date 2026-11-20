# ClimbVision

Movement analysis from video: a generic engine (pose → canonical skeleton → centre of
mass, angles, contacts, support) plus **recipes** — small YAML files that say what to
measure for one movement. It analyses floor exercises and climbing: handheld wall
video, holds and smears, moves, the route, rests and balance on the wall, technique
(quiet feet, straight arms, hesitation, smoothness), the load on each hand and foot,
move phases, the deadpoint, swing and joint effort, in your real units once it knows who
is climbing — in an interactive viewer where every overlay switches on and off and any
mix of them exports as video (with the clip's sound). See [CHANGELOG.md](CHANGELOG.md).
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
# the interactive viewer (also written by analyze); --open serves it and opens the browser
climbvision view outputs/traverse --open
climbvision view outputs --keep-video      # redo only the overlays, after `metrics`
# redraw the annotated video from existing results (default: the balance preset)
climbvision render outputs/squat_side
climbvision render outputs/iman3 --preset dynamics      # annotated_dynamics.mp4
climbvision render outputs/iman3 --layers skeleton,forces,timeline --out mine.mp4
climbvision render outputs                 # every results folder
climbvision render outputs/iman3 --no-watermark   # without the ClimbVision watermark
# camera layer for clips analysed before it existed (handheld wall videos)
climbvision camera outputs/traverse
# real 3D metres from a scan of the wall (once per wall, then per clip on it)
climbvision wall add steep-overhang scan.ply
climbvision wall mark steep-overhang outputs/radmehr4 --open      # pair 5-8 landmarks
climbvision wall fit  steep-overhang outputs/radmehr4 --marks ~/Downloads/wall_marks.csv
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
| `climb` | `climb`, `traverse`, `boulder`, `moonboard`, `overhang`, `coordination`, `route`, `slab`, `crimp`, `dyno` — and **any name that matches nothing else** | hands/feet on holds (still ~0.5 s on the wall), smears and flags, holds used, the route move by move (`moves.csv`, `route.png`), rests, feet cuts, COM over the feet and inside the base of support, heights above the floor and COM speed, technique (quiet feet, straight arms, pause before moves, COM path), load on each hand and foot (`forces.csv`), dynamics (move phases, push power, deadpoint timing, swing, joint effort) |
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
| `viewer.html` + `viewer.mp4` | the interactive viewer: the clip (with sound) and overlays you switch on and off — presets clean / balance / load / dynamics / form / all: forces on each hand and foot, joint effort, joint angles, COM speed and velocity, dyno arc, swing, a timeline of contacts, moves, phases and load — slow motion, frame stepping, and **Export MP4** of what you see. Open in any browser; `climbvision view` redoes it |
| `annotated.mp4` | the `balance` preset burnt in, with the clip's sound (`climbvision render --preset dynamics` etc. for others): skeleton, COM with a 2-second trail (green = inside the base of support, red = outside, pink = no base), floor line, ground contacts (cyan rings), support span on the floor and the COM's plumb line. On the wall: limbs on holds, smears (dashed white rings), the base of support (yellow), the latest move. The trunk is the shoulder bar, hip bar and a curved spine; grey joints are hidden or guessed |
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
pytest                      # synthetic tests + 26 real-clip regression tests (~1 min)
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
                   smoothing, biomech (COM, angles), scale, surfaces (floor), contacts
                   (floor, bar grips), support, holddetect, wall (anchors, holds, smears),
                   metric + profile + kinematics (metres), route (moves, rests),
                   technique, forces, dynamics, events, signals
  recipes/         recipe runner + expression language; library/*.yaml = shipped recipes
  output/          scene (overlay layers) -> render (MP4) and viewer (HTML); report.png,
                   route map, CSV tables, wall view, one-line summary
  analysis.py      results folder -> all analysis outputs
  cli.py           command line
tests/             synthetic + real-clip regression tests
docs/              GUIDE.md (complete guide), RECIPES.md (recipe reference)
data/videos/       your clips (git-ignored)
outputs/           results (git-ignored)
```
