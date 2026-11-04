# ClimbVision — Action Plan

**Goal:** from a single video, measure how a body moves: pose → centre of mass (COM) → stability.
**Order:** get it right off the wall (standing, squat, plank) first, then move onto the wall.

Each milestone ends with something you can run on a real clip and look at.

---

## Assumptions (v1)

- One person in frame, fixed camera (tripod or propped phone).
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

## M3 — Smoothing & jitter
- One-Euro filter on landmarks (with a raw/filtered toggle)
- Jitter metric: COM standard deviation while standing still
- Velocity / acceleration of COM

**Done when:** a still-standing clip shows clearly less COM jitter after filtering, with
no visible lag during a normal squat.

## M4 — Joint angles & exercise metrics
- Knee, hip, ankle, elbow, shoulder angles; trunk lean
- Squat: depth (hip vs knee height), knee/hip angle at the bottom, rep count
- Plank: body-line straightness (shoulder–hip–ankle angle), hip sag over time

## M5 — Contacts, base of support, stability
- Ground contact detection by geometry (no ML): low velocity + near the floor line
- Support polygon (convex hull of contact points), COM projection
- Stability margin = distance from COM projection to the polygon edge
- Note: with one side-on camera this is 1D (front–back); a front-on view gives
  side-to-side. Use the 3D world estimate where possible.

## M6 — Reports & regression tests
- Per-clip plots (COM path, angles, stability margin over time) + summary
- A small set of reference clips with expected values → regression tests

## M7 — On the wall
- Wall-plane calibration (mark the wall corners once per camera setup)
- Hand + foot contacts on holds, support polygon on the wall plane
- Move-by-move segmentation, compare attempts

---

## Test clips to record (for M1–M5)

Fixed camera, whole body in frame, 10–20 s each. One side-on and one front-on of each:

1. Standing still (feet together) — jitter baseline
2. Standing on one foot
3. Slow squats ×5
4. Plank (forearm) — 15 s
5. Lunge or step forward/back

Put them in `data/videos/` (git-ignored).
