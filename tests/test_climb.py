"""Climb window (engine/climb.py): the climb, not the whole clip."""

import numpy as np

from climbvision.engine.climb import find_climb


def _clip(n_s=30.0, fps=30):
    t = np.arange(int(n_s * fps)) / fps
    n = len(t)
    return t, {"on_wall": np.zeros(n, bool), "on_ground": np.zeros(n, bool),
               "contact": np.zeros(n, int), "size": np.full(n, 400.0),
               "vis": np.full(n, 0.9), "cam": np.ones(n, bool),
               "com_y": np.full(n, 1000.0)}


def _climb(t, c, a, b, rise=600.0):
    s = (t >= a) & (t < b)
    c["on_wall"][s] = True
    c["contact"][s] = 3
    c["com_y"][s] = 1000 - rise * (t[s] - a) / (b - a)


def _run(t, c):
    return find_climb(c["on_wall"], c["on_ground"], c["contact"], t, c["size"], c["vis"],
                      c["cam"], c["com_y"])


def test_walk_off_towards_the_camera_ends_the_climb():
    t, c = _clip()
    c["on_ground"][t < 3] = True
    _climb(t, c, 3, 20)
    # after the drop (unseen), the wall layer still says "on the wall" (stray anchors)
    # while the climber walks up to the camera and grows
    walk = t >= 20.5
    c["on_wall"][walk] = True
    c["contact"][walk & (t > 25)] = 1
    c["size"][walk] = np.linspace(500, 1400, walk.sum())
    cw = _run(t, c)
    a, b = cw.attempt.span_s(t)
    assert abs(a - 3) < 0.1 and 19.9 <= b <= 21.0
    assert not cw.mask[t > 22].any()


def test_touchdown_ends_the_climb_and_the_higher_attempt_is_measured():
    t, c = _clip(40)
    c["on_ground"][:] = True
    _climb(t, c, 2, 7, rise=200)  # a fall low on the route
    _climb(t, c, 10, 30, rise=900)  # the second go gets to the top
    c["on_ground"][c["on_wall"]] = False
    cw = _run(t, c)
    assert len(cw.attempts) == 2 and cw.main == 1
    a, b = cw.attempt.span_s(t)
    assert abs(a - 10) < 0.1 and abs(b - 30) < 0.1
    assert cw.attempt.end_reason == "touchdown"
    assert "attempts" in cw.summary(t)


def test_a_short_camera_glitch_doesnt_split_the_climb_but_its_frames_dont_count():
    t, c = _clip()
    _climb(t, c, 0, 25)
    c["cam"][(t > 10) & (t < 10.8)] = False  # 0.8 s: a break
    c["on_wall"][(t > 10) & (t < 10.8)] = False  # (the wall layer drops lost frames)
    cw = _run(t, c)
    assert len(cw.attempts) == 1
    a, b = cw.attempt.span_s(t)
    assert a < 0.1 and b > 24.5
    assert not cw.mask[(t > 10.1) & (t < 10.7)].any()


def test_a_long_pose_loss_ends_the_climb():
    t, c = _clip()
    _climb(t, c, 0, 30)
    c["vis"][t > 18] = 0.1  # the pose is lost: whatever follows is guessing
    cw = _run(t, c)
    assert cw.attempt.span_s(t)[1] < 18.6
    assert "pose" in cw.attempt.end_reason


def test_scale_learnt_from_the_climb_only():
    from climbvision.engine.scale import chain_len, estimate_scale
    from climbvision.engine.skeleton import N_JOINTS, J
    from climbvision.engine.track import Track

    n = 100
    img = np.full((n, N_JOINTS, 2), np.nan)
    for i in range(n):
        k = 1.0 if i < 70 else 3.0  # last 30 frames: walking up to the camera
        img[i, J.L_SHOULDER] = img[i, J.R_SHOULDER] = (0, 0)
        img[i, J.L_HIP] = img[i, J.R_HIP] = (0, 100 * k)
        img[i, J.L_KNEE] = img[i, J.R_KNEE] = (0, 200 * k)
        img[i, J.L_ANKLE] = img[i, J.R_ANKLE] = (0, 300 * k)
    tr = Track(np.arange(n) / 30.0, img, np.full((n, N_JOINTS, 3), np.nan),
               np.ones((n, N_JOINTS)))
    assert np.nanpercentile(chain_len(img), 90) == 900
    frames = np.arange(n) < 70
    assert estimate_scale(tr, 1.36, frames=frames).body_px == 300
