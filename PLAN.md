# ClimbVision — Action Plan

**Goal:** from a single video, measure how a body moves: pose → centre of mass (COM) → stability.
**Order:** get it right off the wall (standing, squat, plank) first, then move onto the wall.

Each milestone ends with something you can run on a real clip and look at.

---

## Assumptions (v1)

- One person in frame, fixed camera (tripod or propped phone) — handheld comes with E2.
- Single camera, so image measurements are 2D. MediaPipe also estimates rough 3D
  "world" coordinates in metres (hip-centred), which we store alongside.
- Python 3.10+, MediaPipe Pose Landmarker as the pose backend (swappable).

---

## M1 — Pose pipeline ✅ (built)
`climbvision analyze clip.mp4` → annotated video + per-frame data.

- Read video, run pose, draw skeleton, write `annotated.mp4`
- Save `landmarks.csv` (33 landmarks × frame, pixel + world coords + visibility)
- Pose backend behind an interface so MediaPipe can be replaced later

**Done when:** skeleton tracks you cleanly in a standing / squat / plank clip.

## M2 — Centre of mass ✅ (built, needs validation on your clips)
- Segment model from De Leva (1996): 14 segments, mass fractions + COM positions
- Whole-body COM per frame, in pixels and in 3D world coords; COM trail overlay
- `com.csv` + `summary.json`

**Validate:** standing still → COM sits roughly above the feet, near the navel/hip
level; squat → COM drops and moves back; plank → COM sits between hands and feet.

### First run on real clips (2026-11-04)
- 10 clips, 100% detection. Squat side view: 5 clean reps in the COM trace.
- Standing still: COM sits at 58–60% of nose-to-heel height (plausible); frame-to-frame
  jitter already < 1 px at 1080×1920, so smoothing is lower priority than expected.
- Problems: plank **front** view loses the legs (conf 0.62) → use side view for plank;
  one-foot **front** view mis-places the raised knee (foreshortening); side-view
  confidence ~0.76–0.78 because far-side limbs are hidden; squat side clip crops the feet.
- MediaPipe "world" coords are hip-centred, so 3D COM shows posture, not travel.
- Recording tips: camera at hip height (not floor level), whole body incl. feet in frame,
  2 s of standing still before moving.

## M3 — Smoothing & jitter
- One-Euro filter on landmarks (with a raw/filtered toggle)
- Jitter metric: COM standard deviation while standing still
- Velocity / acceleration of COM

**Done when:** a still-standing clip shows clearly less COM jitter after filtering, with
no visible lag during a normal squat.

## M4 — Joint angles & exercise metrics ✅ (built 2026-11-05)
- Knee, hip, ankle, thigh, elbow, shoulder, body-line angles (2D + 3D), trunk lean
- Auto-detects side vs front view and the near side; side view uses 2D, front uses 3D
- Squat: rep count, knee/hip angle at the bottom, thigh angle (parallel), trunk lean,
  descent/ascent time
- Plank: hold time, hip offset from the shoulder–ankle line (sag / straight / pike), drift
- `climbvision metrics outputs/*` recomputes all of this from saved landmarks in seconds

Results on your clips:
- squat_side: 5/5 reps; bottom knee 64–77°, trunk lean ~32°; reps 1–2 at/below
  parallel, 3–5 above (thigh +7 to +13°) — matches the video.
- squat_front: 5/5 reps, but front-view depth and lean are unreliable (3D straight
  knee reads ~163°, trunk lean ~20° when side view shows ~32°).
- plank_side: 16.6 s hold, hips ~9% above the line (mild pike) with a clear pike
  at 10–12 s — matches the video. plank_front: legs hidden, numbers not trustworthy.
- **Rule for now: judge squat and plank form from side views.** Front view is useful
  for left/right symmetry (later).

## M5 — Contacts, base of support, stability ✅ (built 2026-11-05)
- Pixel→metre scale from limb lengths (2D vs MediaPipe 3D)
- Floor line fitted to the lowest well-supported points (RANSAC, penalises points
  below the line so resting hands/knees can't win)
- Contacts = near the floor + nearly still, for feet, hands, forearms, knees
- Support span along the image axis; stability margin (cm) = COM plumb line to the
  nearest support edge; COM sway (sd, range, mean speed) overall and per stance
- Annotated video now drawn in a separate render pass (`climbvision render`)
- Dropped: a full 2D support polygon from MediaPipe's 3D estimate. On standing clips
  it put the COM *outside* both feet even after correcting for camera tilt, so the
  depth estimate isn't good enough for this. Revisit with two cameras or a depth phone.

Results on your clips:
- plank_side: feet + hands + forearms on the floor 90%+ of the hold; margin ~63 cm.
- lunge_side: jumps detected (no contact mid-air), knee touchdowns detected, margin
  goes negative at each push-off — as expected for a dynamic move.
- standing_still_side: margin +6 cm, sway sd 0.5 cm, 2.5 cm range over 12 s.
- **Most clips crop the feet** (heels/toes predicted below the frame in 65–100% of
  frames), so their contacts and margins are guesses. Re-record with feet visible:
  standing still, one foot, squat — side and front.

## Round 3: new clips (2026-11-05)
16 clips incl. crow pose, tree pose, jumping jacks, pistol squats, plank → pike push-ups.
Feet now in frame in all re-recorded clips.

Added:
- Exercises: pistol squats (follows the bending leg), pike push-ups, jumping jacks,
  balance holds (tree / crow / one foot)
- Jump detection for every clip: airborne phases checked for real flight (feet leave
  the floor, COM rises and falls); jump height from flight time
- View detection from the body's 3D yaw (front 3–11°, side 78–88° on all clips)
  instead of shoulder width — fixed tree pose being called a side view
- Floor model that follows the person moving towards/away from the camera
  (and the camera-height estimate that falls out of it); one scale per clip when
  there is no evidence of movement in depth
- `render` works from a resized copy of the video

Results:
- jumping_jacks_front: 6 reps, 57/min, 12 jumps (2 per rep), ~5 cm high.
- squat_side: 4 reps, all below parallel (thigh −36°), knee 36°, trunk lean 35°.
- squat_front: 4 reps, all below parallel; front view reads less depth/lean (as before).
- tree_pose_front: 10.6 s on the left foot, sway sd 0.9 cm.
- standing_onefoot_front/side: 9.2 / 7.6 s on the right foot, sway sd 0.6 / 1.3 cm.
- crow_pose_side: 10.8 s on the hands, COM within ~1 cm of the hands.
- plank_to_pike_pushup_side: 3.5 s plank, 5.8 s pike, 1 pike push-up (elbow 54°).
- pistol_squat_front: 6 reps, all below parallel. Weak spots: the person turns
  side-on mid-clip; at the bottom the standing foot is hidden behind the hips, so
  foot contact drops out there; the leg labels (5 left / 1 right) are not reliable
  when facing the camera.
- plank_front: legs hidden behind the body again — plank needs a side view.
- lunge_sinde (new lunge, side): 6 jumps detected; no lunge-specific metrics yet.

## Round 4: bar exercises + documentation (2026-11-05)
- Bar mode for `pullup`, `dead_hang`, `leg_raise` (set by the exercise name): grip
  detection (hand above shoulder, still, then held while it stays put), floor learned
  only from frames without a grip (unknown if the whole clip is on the bar),
  hanging time, swing of the COM vs the grip, purple grip/bar overlay
- Pull-ups: reps on the elbow, COM rise (cm), chin over bar, tempo, swing, kipping;
  leg raises: reps on the hip, toes to bar; dead hang: time, elbow, swing
- Tested on synthetic hanging skeletons only so far — **waiting for real clips**
- docs/GUIDE.md: full user + developer guide

## Round 5: real bar clips (2026-11-05)
4 front-view clips in a door frame, camera at an angle, face hidden by the frame at the
top of pull-ups: hang, knee-raise hold, leg raises, narrow-grip pull-ups.
- Grips found on both hands for ~100% of every clip; floor correctly "unknown" (on the
  bar the whole time).
- hang_front: 7.3 s, elbows 170°, swing sd 0.3 cm.
- narrow_pullup_front: 2 slow reps, shoulders rose ~45 cm to ~7 cm below the bar,
  elbows ~80° at the top, no kipping; chin: 1 "at bar", 1 just "over".
- leg_raise_front: 4 reps; reps 1–2 knee raises (hip ~60°), 3–4 higher (hip ~22°),
  1 toes-to-bar.
- knee_raise_hold_front: 9.0 s hold; hips sank from 70° to 102° (fatigue).
Fixes from these clips:
- Pull-up reps now counted on shoulder-to-bar distance (front-view 2D elbows read ~5°
  with a narrow grip); elbow reported from 3D in front views.
- Chin over bar from body proportions, not the (hidden) face; ±5 cm = "at bar".
- Leg raises report a raised-leg hold; swing reports the constant camera-angle offset
  separately.

## M6 — Reports & regression tests ✅ (built 2026-11-06)
- `climbvision summary outputs/*`: one line per clip + summary.csv
- 20 real clips as regression fixtures (pose data only, 6 MB) in tests/regression/,
  with 75 checks: verified facts (rep counts, stances, warnings) must match exactly,
  snapshots of continuous results must stay within a tolerance
- Whole suite (38 synthetic + 20 real) runs in ~5–10 s without video or MediaPipe
- Checked the net works: tightening the floor-contact height threshold (5% → 2% of body length) makes 2 real clips fail
- Also: the "no ground contact" warning no longer counts time spent in real jumps

## Round 6: climbing clips (2026-11-07)
First wall clips exposed the limits of the off-the-wall design: handheld camera (the
floor/stillness logic assumes a still image), no footholds found, base of support
meaningless on the wall, the skeleton jumping to another person or onto holds
(`coordination_failed`), and face/finger points that are wrong when the climber's back
is to the camera. Decision: rebuild as a generic engine + recipes, then add the wall
layers one at a time.

## Engine roadmap
| Step | What | Done when |
|---|---|---|
| **E1 ✅** | Layered engine: canonical skeleton (no face), backend mapping, layers re-run from saved data, recipe system, today's exercises as YAML recipes | the 20 reference clips give the same results |
| **E2 ✅** | Camera layer: register every frame to a reference view / mosaic of the wall (handheld video) | holds stay put in wall coordinates on the handheld clips |
| E3 | Subject lock: track + segment the climber, reject poses on holds/other people | no skeleton jumps on `coordination`, `coordination_failed` |
| **E4 ✅** | Anchors on the wall: hand/foot still ~0.5 s in wall coordinates = on a hold; support polygon on the wall | footholds found on the traverse clips |
| E5 | Pose backend benchmark on your Mac (RTMPose Halpe26 with feet, RTMW, ViTPose), switch if better | hand/foot accuracy measured on the wall clips |
| E6 | Climbing recipes: moves, hold usage, route map | first per-move report |

## E1 — engine restructure ✅ (2026-11-10)
- `src/climbvision/engine/`: video, pose + backends, skeleton (23 canonical joints:
  head, neck/head-top, shoulders…hands, hips…toes; no face, fingers → one HAND point),
  track, biomech, scale, surfaces, contacts, support, events, signals
- `landmarks.csv` now keeps the backend's raw points + `summary.json` records the
  backend, so mappings can change without re-running pose
- Recipes (`recipes/library/*.yaml`) replace exercises.py/metrics.py: squat,
  pistol_squat, plank, pike_pushup, jumping_jacks, balance, pullup, leg_raise, dead_hang.
  A small safe expression language; `--recipe` (name or file), `--recipes DIR`,
  `climbvision recipes`, `climbvision signals`; headlines come from the recipes
- Fixed: "overhang" no longer selects the dead-hang recipe
- Equivalence: all 20 regression clips pass. 15 of 20 give identical metrics.json; the
  other 5 (crow pose, planks, pike push-ups, pistol front) differ slightly in floor
  contacts of the hands only, because a hand is now one HAND point instead of three
  finger points (e.g. crow balance hold 10.81 → 10.74 s, pistol front floor tilt
  0.85° → −1.33°). Every rep count, hold and exercise result is unchanged except the
  crow hold time.
- Off-the-wall recipes stay in this repo until the engine settles; they can then move
  to a separate package that depends on the engine.

## E2 — camera layer ✅ (2026-11-10)
- `engine/camera.py`: one homography per frame (frame → wall coordinates). Wall corners
  tracked with Lucas–Kanade, each carrying its wall position, so every frame is fitted
  straight to the wall (MAGSAC++), not chained; keyframes (ORB) re-anchor every 5th frame
  and close loops when the camera comes back. The climber is masked out using the pose.
- `wall.jpg`: the wall without the climber (median of frames in wall coordinates) — the
  background for E3 (subject separation) and E4 (holds).
- `wall_view.mp4`: the video in wall coordinates, hand/foot trails.
- `climbvision camera` (also run by `analyze`; `--no-camera` to skip); `metrics.json`
  gets a `camera` block; signals `wall_speed_*`, `image_speed_*`, `camera_ok`,
  `camera_speed`; `Clip.wall` = joints in wall coordinates.
- Fixed cameras are detected (`static: true`; standing_onefoot_front: 0.3 px travel).

Results on the 5 handheld wall clips (720p copies). Error = offset of wall patches
(holds) re-found in every 3rd frame after warping it into wall coordinates:

| clip | registered | camera pan (x, y) | checks | median | 90 % | 99 % | > 2 cm |
|---|---|---|---|---|---|---|---|
| traverse | 100 % | 26 × 120 cm | 24 376 | 0.06 cm | 0.14 cm | 0.29 cm | 0.06 % |
| coordination | 100 % | 44 × 56 cm | 52 038 | 0.12 cm | 0.26 cm | 0.50 cm | 0.12 % |
| coordination_failed | 100 % | 44 × 39 cm | 28 252 | 0.15 cm | 0.47 cm | 3.77 cm | 1.44 % |
| moonboard | 100 % | 4 × 14 cm | 45 175 | 0.07 cm | 0.11 cm | 0.23 cm | 0 % |
| overhang | 100 % | 23 × 25 cm | 44 043 | 0.10 cm | 0.19 cm | 0.32 cm | 0.14 % |

- This check measures whether every frame agrees with the wall map; absolute accuracy
  of the map itself comes from the synthetic tests (known camera motion: corners within
  ~1–2 px, no drift after panning away and back).
- coordination_failed is the weakest because the skeleton sometimes sits on the orange
  volume, so the real climber is not masked out (fixed by E3), and the volume is not
  flat.
- Finding: in these clips the camera moves slowly (pans of 0.1–1.2 m over 30–60 s,
  mostly between moves). That is enough to put the same hold up to a metre apart in
  image coordinates at different times — fixed now — but hand/foot *stillness* was
  barely affected by it. The missing footholds are therefore mainly E4 (contacts are
  still defined by a floor, which the wall doesn't have) and pose noise on the feet
  (E5): feet are "still" only 27–63 % of the time even in wall coordinates, and the
  feet appear to swap left/right at times (traverse wall view).

---

## E4 — anchors on the wall ✅ (2026-11-10)
- `engine/wall.py`: a hand (grip point) or foot (toe) still ~0.5 s in wall coordinates
  is anchored on a hold (window 0.5 s, 80 % within ~6 cm; grows while it stays put,
  jitter bridged; 0.2 s running median first). Holds = places limbs rested (anchors
  within ~11 cm), moves = re-anchoring ≥ 12 cm away, contacts (4/3/2…), COM over the
  feet and inside the hull of limbs on holds.
- Ground: foot anchors with floor colour (bottom of `wall.jpg`) just below the toe and
  among the lowest foot positions; `on_wall` = first to last foot on a hold, minus ground.
- `climb` recipe (`scene: wall`): metrics.json gets a `wall` block (holds, moves,
  contacts) instead of the floor `stability`; `wall.csv`; report panels and
  `wall_view.mp4` show holds, limbs on holds, the support hull and the COM;
  `annotated.mp4` rings the limbs on holds.
- Recipe matching: traverse, moonboard, overhang, coordination, boulder, climb, route →
  `climb`.

Results (720p copies; the full-resolution runs are in the clip folders):

| clip | on the wall | holds (hand / foot) | moves (hand / foot) | 4 limbs on holds | feet on holds |
|---|---|---|---|---|---|
| traverse | 36.3 s | 28 (14 / 15) | 14 / 16 | 38 % | 78 % |
| moonboard | 41.9 s (stood on the floor at start and end) | 37 (14 / 25) | 18 / 31 | 43 % | 71 % |
| overhang | 44.1 s (floor at the start) | 40 (17 / 27) | 18 / 32 | 31 % | 59 % |
| coordination | 46.9 s (floor at start, stepped down at 13–16 s) | 30 (14 / 18) | 14 / 20 | 15 % | 39 % |
| coordination_failed | 11.4 s of 28 s (pose lost on the volume) | 10 (7 / 3) | 6 / 1 | 18 % | 31 % |

- Footholds found (before: none). Checked by eye on the traverse: of 30 foot anchors
  ~7 in 10 sit on the shoe on a hold; the others are pose errors (the foot point on the
  shin/hip while the body is still). Of the sampled moments on the wall without a foot
  anchor, about half are the foot moving and half pose errors.
- Hand anchors look right on the traverse (checked by eye; hands are tracked far better
  than feet).
- Foot counts are inflated by pose errors (false anchors become extra "holds" and
  "moves"); E5 (pose backend) is now the main lever. coordination_failed needs E3.
- Thresholds were chosen on these clips without ground truth; a few labelled clips
  (which hold, when) would let us tune and score them properly.

## Test clips to record (for M1–M5)

Fixed camera, whole body in frame, 10–20 s each. One side-on and one front-on of each:

1. Standing still (feet together) — jitter baseline
2. Standing on one foot
3. Slow squats ×5
4. Plank (forearm) — 15 s
5. Lunge or step forward/back

Put them in `data/videos/` (git-ignored).
