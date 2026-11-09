# ClimbVision

Movement analysis from video: pose → centre of mass → stability.
Starting with off-the-wall movements (standing, squat, plank); climbing comes later.
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
climbvision analyze data/videos/IMG_1234.MOV --exercise squat

# re-run angles, exercise and stability metrics on existing results (seconds)
climbvision metrics outputs/*
# redraw the annotated video from existing results
climbvision render outputs/squat_side
```

The exercise is guessed from the file name, otherwise pass `--exercise`:

| `--exercise` | file names that match | what you get |
|---|---|---|
| `squat` | `squat` | reps, depth (thigh angle, below parallel), knee/hip angle, trunk lean, tempo |
| `pistol_squat` | `pistol` | the same, following the bending leg, plus reps per leg |
| `plank` | `plank` | hold time, hip sag / straight / pike, drift |
| `pike_pushup` | `pike` | push-ups done in a pike (elbow depth), time in plank and in pike |
| `jumping_jacks` | `jumping_jack`, `jacks` | reps (arms up and down), reps/min, how wide the feet go |
| `balance` | `tree`, `crow`, `onefoot`, `single_leg` | longest hold on a reduced base (one foot, hands), COM sway and margin during it |
| `pullup` | `pullup`, `chinup` | reps, COM rise (cm), chin over bar, tempo, swing, kipping |
| `leg_raise` | `leg_raise`, `toes_to_bar`, `knee_raise` | reps, hip/knee angle, toes to bar |
| `dead_hang` | `dead_hang`, `hang` | hang time, arm straightness, swing |

Every clip also gets jumps (flight time, jump height from flight time), stances and
the stability margin. Judge squat, plank and pike form from **side-view** clips.

The pose model downloads automatically on the first run (~10–30 MB).

Each video gets a folder `outputs/<video name>/` with:

| File | What's in it |
|---|---|
| `annotated.mp4` | skeleton, COM (pink) with a 2-second trail, floor line, ground contacts (cyan rings), support span on the floor and the COM's plumb line — green when the COM is over the support, red when not |
| `landmarks.csv` | 33 landmarks per frame: pixel coords, 3D world coords (m), visibility |
| `com.csv` | whole-body COM per frame and a confidence score. `com_*_px` = position in the image; `com_*_m` = 3D position in metres **relative to the hip centre** (shows body shape, not where the body is in the room) |
| `summary.json` | detection rate, COM mean / spread / range |
| `angles.csv` | per-frame joint angles, both sides, 2D (image) and 3D versions |
| `metrics.json` | view (side/front, body yaw), near side, angle ranges, exercise results (table above) and `stability` |
| `stability.csv` | per frame: what's touching the floor, support span, COM-to-edge margin (cm) |
| `report.png` | angle plots, squat reps or plank hip offset, stability margin, contact timeline |

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
pytest                      # synthetic tests + 20 real-clip regression tests (~10 s)
pytest -m "not regression"  # synthetic only
```

The regression clips live in `tests/regression/` (pose data only, no video); see
docs/GUIDE.md §8.6.

## Layout

```
src/climbvision/
  skeleton.py   landmark names + Skeleton/Frame data model (backend-independent)
  pose.py       MediaPipe pose backend (swappable)
  com.py        De Leva (1996) 14-segment centre of mass
  video.py      read/write video
  draw.py       overlays
  pipeline.py   video -> pose -> COM -> CSVs (the slow pass)
  track.py      a clip's landmarks as arrays; load from landmarks.csv
  angles.py     joint angles, view + near-side detection
  exercises.py  squats (incl. pistol), plank, pike push-ups, jumping jacks, bar work
  stability.py  scale, floor line, ground contacts, support, stability margin, sway
  metrics.py    angles.csv / stability.csv / metrics.json / report.png
  summary.py    one line per clip (climbvision summary)
  render.py     annotated.mp4 from saved results
  cli.py        command line
tests/
data/videos/    your clips (git-ignored)
outputs/        results (git-ignored)
```
