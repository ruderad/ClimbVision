# ClimbVision

Movement analysis from video: pose → centre of mass → stability.
Starting with off-the-wall movements (standing, squat, plank); climbing comes later.
See [PLAN.md](PLAN.md) for the milestones.

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
```

The pose model downloads automatically on the first run (~10–30 MB).

Each video gets a folder `outputs/<video name>/` with:

| File | What's in it |
|---|---|
| `annotated.mp4` | skeleton, COM dot and a 2-second COM trail |
| `landmarks.csv` | 33 landmarks per frame: pixel coords, 3D world coords (m), visibility |
| `com.csv` | whole-body COM per frame (pixels + metres) and a confidence score |
| `summary.json` | detection rate, COM mean / spread / range |

Options: `--no-video` (data only, faster), `--max-frames N`, `--out DIR`.

## Tests

```bash
pytest
```

## Layout

```
src/climbvision/
  skeleton.py   landmark names + Skeleton/Frame data model (backend-independent)
  pose.py       MediaPipe pose backend (swappable)
  com.py        De Leva (1996) 14-segment centre of mass
  video.py      read/write video
  draw.py       overlays
  pipeline.py   video -> pose -> COM -> files
  cli.py        command line
tests/
data/videos/    your clips (git-ignored)
outputs/        results (git-ignored)
```
