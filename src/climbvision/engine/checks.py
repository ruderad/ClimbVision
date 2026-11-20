"""Plausibility checks: is this clip's measurement believable?

No ground truth is needed: each check compares two things the system measured that
physics or anatomy ties together. A failed check doesn't say which number is wrong,
but that the clip's metres (heights, distances, forces) shouldn't be trusted as they
are. The checks run on every climbing clip and land in `metrics.json` (`checks`).

- **floor_vs_rise**: the COM's highest point above the floor can't be below its rise
  during the climb (the climb doesn't start underground). Fails when the floor
  reference is wrong.
- **standing_com**: standing upright on the floor (both knees > 150 deg), the COM is at
  ~0.55 x stature (0.53-0.57 in adults; Drillis & Contini 1966). Tests the scale and the
  floor together. Accepted: 0.45-0.75, because standing *in front of* the wall reads
  high: everything is measured on the wall plane, and a body 1 m nearer the camera
  than the wall projects onto it ~1.25x taller, its feet lower than the real floor
  (a 5 m camera distance; this is one camera's depth blind spot). With the
  floor from the push-off feet (closest to the wall) the test clips centre on 0.56
  (0.44-0.79); heights above the floor may still read up to ~0.5 m high (GUIDE §6.14b).
- **com_vs_top_hold**: hanging from the highest hand hold with straight arms, the COM is
  ~0.65 x stature below the hands; bent arms bring it to ~0.3 x. A COM much higher
  than the top hand hold (more than 0.3 x stature: unless the climber topped out) or
  much lower (more than 1 x stature) means the heights are off.
- **body_size**: the body's segments, measured in metres in every region of the wall,
  should agree (perspective corrected). Their spread (log std of the 85th-percentile
  lengths across regions) beyond 0.10 means distances differ by ~10 % or more across
  the wall.
- **pose_quality**: mean joint visibility during the climb; below 0.6 the pose model
  was guessing much of the time.
- **attempts**: more than one attempt in the clip (only one is measured).
"""

from __future__ import annotations

import warnings

import numpy as np

from climbvision.engine.profile import ASSUMED_HEIGHT_M


def _item(name: str, ok: bool | None, value, expected: str, note: str) -> dict:
    return {"name": name, "ok": ok, "value": value, "expected": expected, "note": note}


def wall_checks(clip, wall_out: dict) -> dict:
    """Plausibility checks for a climbing clip (Clip with a wall layer) and its
    `wall_summary`. Returns {"passed", "failed", "items"}; a check that can't run (no
    floor seen...) has ok None and is left out of the counts."""
    ws = clip.wall_support
    t = clip.t
    stature = clip.climber.height_m or ASSUMED_HEIGHT_M
    items = []
    top = wall_out.get("com_top_above_floor_m")
    rise = wall_out.get("com_height_gain_m")
    if top is not None and rise is not None:
        items.append(_item("floor_vs_rise", top >= rise - 0.1, {"top_m": top, "rise_m": rise},
                           "top >= rise", "the floor reference is wrong" if top < rise - 0.1
                           else "consistent"))
    # standing on the floor next to the climb
    if ws.heights_above_floor:
        stand = ws.on_ground & ~ws.on_wall & np.isfinite(ws.com).all(axis=1)
        with warnings.catch_warnings():  # standing upright: both knees nearly straight
            warnings.simplefilter("ignore", RuntimeWarning)
            knees = np.fmin(clip.signal("knee_l"), clip.signal("knee_r"))
        stand &= np.nan_to_num(knees) >= 150
        if ws.window is not None and ws.window.any():
            idx = np.flatnonzero(ws.window)
            stand &= (t >= t[idx[0]] - 3.0) & (t <= t[idx[-1]] + 3.0)
        if stand.sum() >= 5:
            hs = ws.height_m(ws.com[stand])
            hs = hs[np.isfinite(hs)]
            h = float(np.median(hs)) if len(hs) >= 5 else float("nan")
            r = h / stature
            ok = 0.45 <= r <= 0.75
            if not np.isfinite(r):
                ok = None  # off the scan / no reading: not judged
            items.append(_item("standing_com", ok,
                               {"com_m": round(h, 2), "x_stature": round(r, 2)},
                               "0.45-0.75 x stature (~0.55 on the wall plane)",
                               "consistent" if ok else
                               "no reading (off the wall scan)" if ok is None else
                               "scale or floor off: standing COM "
                               f"{'too low' if r < 0.45 else 'too high'}"
                               + ("" if r < 0.45 else " (standing far from the wall: "
                                  "heights above the floor read high)")))
    # COM top vs the highest hand hold
    hand_h = [ws.height_m(h.pos) if ws.heights_above_floor else ws.to_m(h.pos)[1]
              for h in ws.holds if h.limbs & {"l_hand", "r_hand"}]
    on = ws.on_wall & np.isfinite(ws.com).all(axis=1)
    if hand_h and on.any():
        com_top = float(np.max(ws.height_m(ws.com[on]) if ws.heights_above_floor
                               else ws.to_m(ws.com[on])[:, 1]))
        below = float(np.max(hand_h)) - com_top
        r = below / stature
        ok = -0.3 <= r <= 1.0
        items.append(_item("com_vs_top_hold", ok,
                           {"top_hand_hold_minus_com_m": round(below, 2),
                            "x_stature": round(r, 2)},
                           "-0.3 to 1.0 x stature (hanging straight-armed: ~0.65)",
                           "consistent" if ok else
                           ("COM far above the top hand hold: heights off, unless the "
                            "climber topped out" if r < -0.3 else
                            "COM far below the top hand hold: heights off")))
    info = ws.metric.info if ws.metric is not None else {}
    spread = info.get("body_size_spread_after")
    if spread is not None and np.isfinite(spread):
        items.append(_item("body_size", spread <= 0.10, round(float(spread), 3), "<= 0.10",
                           "consistent" if spread <= 0.10 else
                           f"body sizes differ by ~{100 * (np.exp(spread) - 1):.0f} % across "
                           "the wall: distances and heights uncertain by as much"))
    win = ws.window if ws.window is not None else ws.on_wall
    if win.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            vis = float(np.nanmean(clip.track.visibility[win]))
        if np.isfinite(vis):
            items.append(_item("pose_quality", vis >= 0.6, round(vis, 2), ">= 0.6",
                               "consistent" if vis >= 0.6 else
                               "the pose model was guessing much of the time (hidden "
                               "limbs, blur, far away)"))
    if clip.climb is not None:
        n_att = len(clip.climb.attempts)
        items.append(_item("attempts", n_att == 1, n_att, "1",
                           "one attempt" if n_att == 1 else
                           f"{n_att} attempts in the clip: only one is measured"))
    counted = [i for i in items if i["ok"] is not None]
    return {"passed": sum(bool(i["ok"]) for i in counted),
            "failed": [i["name"] for i in counted if not i["ok"]],
            "items": items}
