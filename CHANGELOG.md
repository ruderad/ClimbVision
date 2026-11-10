# Changelog

## 1.0.0 — 2026-11-11

First release: movement analysis from video, on the floor and on the climbing wall.

### Engine
- Layered engine with recipes (YAML) on top: video → camera → pose → clean-up →
  skeleton → biomechanics → wall → route → signals → recipes.
- **Pose**: MediaPipe (floor; 3D), RTMPose / RTMW whole-body (climbing; 2D, top-down
  tracking that locks onto the climber). `--backend auto` picks RTMW for climbs.
- **Clean-up**: left/right swaps, flicker and impossible bones (a hidden drop knee
  landing on the other leg) repaired from good frames of the same limb.
- **Camera**: handheld video registered to the wall frame by frame; the wall image
  without the climber; fixed cameras detected.
- **Wall**: a hand or foot still ~0.5 s on the wall = on a hold; holds, ground (floor
  in the wall image, standing at the start/end), contacts, COM over the feet, the base
  of support.
- **Route**: moves in order (from/to hold, time off holds, distance, direction, the
  other limbs, balance before/after), tags (match, cross, dyno, jump, feet cut, high
  step, foot on handhold), rests, feet cuts, hold usage.
- Floor recipes: squat, pistol squat, plank, pike push-up, jumping jacks, balance,
  pull-up, leg raise, dead hang; `climb` for wall clips (and for any clip name that
  matches no recipe).

### Outputs
- `annotated.mp4`: skeleton, base of support (hull of the limbs on holds), COM dot and
  trail coloured by balance (green inside the base, red outside), the latest move.
- `wall_view.mp4`: the climb in wall coordinates, the route building up move by move.
- `route.png` (route map), `moves.csv`, `holds.csv`, `wall.csv`, `report.png` with a
  move timeline, `metrics.json`, one-line `summary`.

### Safety nets
- Results remember which video they came from (size, frames, fps); every command warns
  when the video has been replaced since (re-run `analyze`).
- `climbvision --version` shows the version and where the code is loaded from.
- Commands take folders: `climbvision analyze data/videos`, `climbvision render outputs`.
- Regression tests on 26 real clips (20 floor, 6 climbing), pose data only.

### Known limits
See docs/GUIDE.md §9. The main ones: 2D only on the wall (centimetres from an assumed
body size; balance on steep overhangs means little); a hold is where a limb rested
(neighbouring holds within ~11 cm merge, touches under 0.5 s are skipped); move tags are
heuristics — watch the labelled moves in `annotated.mp4`; chalking is not detected.
