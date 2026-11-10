"""The movement-analysis engine, as layers. Each layer only reads the ones above it,
and everything below the pose layer re-runs from saved files in seconds.

  video      engine/video.py      frames + timestamps
  (camera)   -                    E2: register frames to the wall (handheld video)
  (subject)  -                    E3: lock onto the climber, separate from the wall
  pose       engine/pose.py       a backend's raw points -> landmarks.csv
             engine/backends/     MediaPipe today; others plug in via a mapping
  skeleton   engine/skeleton.py   canonical joints (no face), Track = arrays per clip
  biomech    engine/biomech.py    centre of mass (De Leva), joint angles, view
  scale      engine/scale.py      pixels per metre, body size
  surfaces   engine/surfaces.py   floor line (later: wall plane, holds)
  contacts   engine/contacts.py   anchors: floor contacts, bar grips (later: holds)
  support    engine/support.py    base of support, margin, sway, flight, hanging
  events     engine/events.py     runs, cycles (reps), longest holds
  signals    engine/signals.py    named per-frame series: the recipes' vocabulary

Analyses are recipes (climbvision/recipes), not engine code.
"""
