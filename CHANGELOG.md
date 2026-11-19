# Changelog

## Unreleased

### Floor detection (GUIDE §6.14)
- Textured floors are floors again: the wall-panel test counts only hold-sized objects
  (≥ 0.1 % of the image, 8 or more) instead of every patch that isn't flat, so concrete
  (steep-overhang side view: 0.1 % → 18 % of the image) and iman3's orange mat (1 % →
  17 %) are found; a foot on them is on the ground, not a smear.
- When the hold detector ran, 5 or more detected holds inside a region make it a wall
  panel (the surer test); the traverse's lower panel stays wall either way.

### Measuring the climb, not the clip (GUIDE §6.14a)
- The climb runs from both feet leaving the ground to the touchdown; when the camera
  doesn't see the touchdown, it ends at the first lasting sign that the climb is over
  (camera lost, pose lost, the person's size jumps, no contact for 2 s). The scale is
  learnt from the climb only, and frames outside it count for nothing. Fixes heights
  and distances squashed by people walking up to the camera after the climb (arya2:
  highest COM 1.08 -> 3.19 m above the floor; arya1, radmehr4) and rises inflated by a
  lost pose (gardash3: 4.28 -> 2.37 m).
- Several attempts in one clip: each is found, the one that got highest is measured,
  the others are listed (`climb.attempts`) with a warning.
- A foot off any hold with no hand on one counts as the floor only near the bottom
  (radmehr1's top section read as 8 s on the ground).

### Low-cost fixes from the limitations review
- The floor height comes from the feet that push off for the climb (closest to the
  wall), not from all standing: feet further out on the mat project below the floor and
  made heights read high. The standing COM check now centres on 0.56 x stature (true
  0.55; was 0.61); gardash2's highest point 4.18 -> 3.82 m, arch1 3.37 -> 3.02 m.
- A hand held above the shoulder for 1 s or more off any detected hold, with the body
  moving under it, is on a hold the detector missed (a volume, the top), not a smear:
  it counts for moves and forces (`wall.holds_missed_by_detector`; radmehr1 +3 hand
  moves, slab +2, iman2 +2, overhang +1).
- GUIDE §9: limitations brought up to date (floor, heights above the floor, attempts,
  perspective on some clips).

### Plausibility checks (GUIDE §6.14b)
- Every climbing clip checks what physics and anatomy tie together: the floor against
  the rise, the standing COM against stature, the top COM against the top hand hold,
  body sizes across the wall, pose visibility, attempts. In `metrics.json` (`checks`).
- Documented one camera's blind spot: the floor height comes from feet standing in
  front of the wall, so heights *above the floor* read high (0-0.8 m on the test
  clips); rises and distances don't depend on it.

### Regression fixtures
- The six climbing fixtures are rebuilt from the current pose (RTMW with flip
  averaging and calibrated visibility) and their snapshots updated. Fixtures are
  anonymous (`build_fixtures.py` drops the climber name), and `--update-snapshots` now
  runs without your climber profiles and models, exactly like the tests.

## 1.5.0 — 2026-11-17

Overlays, the interactive viewer and dynamics: what a climber sees, rebuilt around
dynamic climbing; plus the reliability fixes that came from looking at it.

### Sound
- `annotated.mp4`, `annotated_<preset>.mp4` and `viewer.mp4` keep the original clip's
  sound; the viewer's Export MP4 includes it for the chosen range (real time).

### Hanging counts as climbing
- A hand on a hold above the shoulders now starts / keeps the climb (`on_wall`) even
  with no foot on a hold: clips that start in a hang or campus long sections (iman5,
  iman6) had no forces or balance until a foot settled.

### Smears you can trust (GUIDE §6.20, §6.19)
- A still limb off any hold is a **smear** only if the body moved over it (≥ 10 cm);
  otherwise a **flag** (no contact: not in the base of support, no force), drawn grey
  "flag?". A foot "smearing" with no hand on a hold is standing on the floor.
- Smears are weak contacts in the force model (6x a foothold's cost); their loads are
  drawn hollow with "~"; `forces.smear_overloaded_s`. No forces in impacts (> 2.5 g)
  and dismounts.

### Export from the viewer
- **Export MP4** button: the clip with the layers you have on, a time range, size,
  slow motion, and the timeline strip; encoded in the browser (WebCodecs, H.264;
  `mp4-muxer`, MIT, bundled). `climbvision view --open` serves the viewer on localhost
  (byte ranges, so Safari plays it and export can read the video).

### Dynamics (`engine/dynamics.py`, GUIDE §6.23)
- Move phases (push / reach / catch) per hand move; push time and peak power (W/kg);
  when the hand caught relative to the deadpoint. In `moves.csv` and `dynamics` in
  metrics.json; a phase row on the timeline.
- Momentum arc of dynamic moves, swing about the hands with the feet off (peak and
  settling time per feet cut), joint effort (net joint moments, Nm/kg) at shoulders,
  elbows, hips, knees and the core.
- New layers: COM coloured by speed, joint angles, joint effort, dyno arc, swing.
  Presets: `dynamics` now uses the speed-coloured COM (balance colours mean nothing in
  a dyno); new `form` preset.

### Overlays and the interactive viewer
- Everything drawn on the video is a **layer** (skeleton, bent arms, holds, base of
  support, COM, trail, forces, velocity, info text, timeline), built once from the
  results (`output/scene.py`) and drawn by two renderers: the MP4 and the browser.
- **Viewer**: `climbvision view` (and `analyze`) writes `viewer.html` + `viewer.mp4`:
  layers on/off, presets (clean, balance, load, dynamics, all), slow motion, frame
  stepping, a clickable timeline with contacts, moves, rests, feet cuts and the load
  on hands and feet.
- **New layers**: forces (load on each hand and foot, % body weight, colour-coded, with
  the force direction), COM velocity, timeline.
- `climbvision render --preset NAME --layers a,b --out FILE`; default unchanged (balance).
- Videos are H.264 when ffmpeg is available (new dependency `imageio-ffmpeg`): they play
  in browsers and on phones.

## 1.4.1 — 2026-11-17

### Fixed
- **Every metre from the climbing models was 32 % short.** The body ruler (trunk +
  leg) is 0.779 x stature, not 0.531 (Drillis & Contini; 0.778 measured on a standing
  climber). Heights, reach, move lengths, COM speed and acceleration are now 1.47x what
  1.4.0 reported. Found by checking against tape-measured holds (GUIDE §6.22).
- Chin-over-bar estimate (pull-ups): chin 0.067 body lengths above the shoulders (was
  0.095, from the same wrong ratio).

## 1.4.0 — 2026-11-15

Hold detection (1.4) and pose accuracy (1.5).

### Steadier, more honest pose (climbing models)
- Real per-joint confidence from RTMPose/RTMW (scores were clipped to 1): hidden or
  guessed joints are drawn grey and trusted less.
- Steadier person crop and flip averaging (`--fast` turns flipping off).
- Smoothing layer (`engine/smoothing.py`): robust, confidence-weighted local parabola
  per joint — ~4x less shake, no lag on fast moves; spikes and hidden stretches up to
  1 s filled from both sides. `pose_smoothing` in metrics.json.
- The trunk is drawn as shoulder bar + hip bar + a curved spine instead of a box.

### Hold detection and smears
- Holds are found in `wall.jpg` by a pretrained detector (YOLOv8 ONNX in `models/`,
  optional; `engine/holddetect.py`), scanned at three sizes; cached in
  `holds_detected.json` with a picture `holds_detected.jpg`.
- A rest on the wall with no detected hold under it is a **smear**: it counts for the
  base of support, balance and forces, but is not a hold, a move or on the route map.
  Dashed ring on `annotated.mp4`, hatched bar in `report.png`, `smears` in metrics.json.
- One hold id per detected hold (instead of clustering rests).

## 1.3.0 — 2026-11-15

Milestones 1.1–1.3. Heights and forces are not yet checked against measurements (tape
heights, a bathroom-scale test): expect a 1.3.1 with adjusted constants after that.

### Forces (1.3)
- Load on each hand and foot in the wall plane (`engine/forces.py`): least-effort split
  under balance (soft moment balance: matched hands let the body swing), feet push only
  within friction. `forces` in metrics.json, `forces.csv`, a load panel in
  `report.png`, `catch_load_pct_bw` per hand move, signals `load_*_pct_bw`,
  `hands_load_pct_bw`, `feet_load_pct_bw`, `hands_share_pct`. Newtons with the mass.
- `--climber unknown`: an average adult, whatever the profile file's default.

### Technique (1.2)
- `technique` in metrics.json and on `route.png`: quiet feet (re-placements), hand
  re-grips, straight arms, pause before hand moves, time per hand move, COM path
  efficiency and pushes (stop-and-go), feet stayed on during hand moves. Per move in
  `moves.csv`: `pause_before_s`, `cycle_s`, `com_path_efficiency`, `com_pushes`.
- On `annotated.mp4`: holding arms turn orange when bent.

### Accurate scale (1.1)
- **Climber profiles** (`data/climbers.yaml`, git-ignored; template
  `climbers.example.yaml`): height (sets every metre), mass, arm span (ape index), sex.
  `--climber NAME` on `analyze` / `metrics`; `scale` in metrics.json says which height
  was used and whether it was assumed.
- **Perspective-corrected wall** (`engine/metric.py`): the body as a ruler across the
  wall; body sizes up to 1.6x apart on handheld clips are made uniform.
- **Heights above the floor** when the climber stood on it (`com_height_m`,
  `com_top_above_floor_m`, `com_top_m`); `height_gain_m` is now the gain on the wall.
- **Velocity and acceleration** of the COM (signals `com_speed_ms`, `com_vy_ms`,
  `com_vx_ms`, `com_acc_ms2`), per move: COM speed at the catch, peak speed of the moving
  limb, `deadpoint` tag.

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
