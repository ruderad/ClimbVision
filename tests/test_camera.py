"""Camera layer on synthetic video: a textured wall filmed by a moving camera whose
motion is known exactly."""

import cv2
import numpy as np
import pytest

from climbvision.engine.camera import (
    LOST,
    CameraTrack,
    apply_h,
    camera_summary,
    estimate_camera,
)
from climbvision.engine.skeleton import N_JOINTS, J
from climbvision.engine.track import Track

W, H = 320, 480  # frame size (portrait, like a phone)


def climber_xy(i):
    return int(W / 2 + 60 * np.sin(i / 9)), H // 2 + 40


def climber_track(n):
    """A pose-layer track for the synthetic 'climber' (so the camera layer can mask it)."""
    img = np.full((n, N_JOINTS, 2), np.nan)
    for i in range(n):
        cx, cy = climber_xy(i)
        for side, sx in (("L", -1), ("R", 1)):
            for name, (dx, dy) in (("SHOULDER", (40, -90)), ("HIP", (30, 20)),
                                   ("KNEE", (30, 90)), ("ANKLE", (30, 150))):
                img[i, J[f"{side}_{name}"]] = (cx + sx * dx, cy + dy)
    return Track(np.arange(n) / 30, img, np.full((n, N_JOINTS, 3), np.nan),
                 np.ones((n, N_JOINTS)))


def wall_texture(seed=0, size=(1400, 1000)):
    """A 'climbing wall': speckled panel with many coloured holds."""
    rng = np.random.default_rng(seed)
    h, w = size
    img = rng.integers(90, 140, (h, w, 3)).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 2)
    for _ in range(500):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        p = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        r = int(rng.integers(5, 16))
        cv2.ellipse(img, p, (r, int(r * rng.uniform(0.5, 1.2))), float(rng.uniform(0, 180)),
                    0, 360, c, -1, cv2.LINE_AA)
        cv2.circle(img, p, max(2, r // 4), (30, 30, 30), -1)
    return img


def camera_motion(n, kind):
    """G[i]: frame pixels -> texture pixels."""
    G = []
    for i in range(n):
        u = i / max(n - 1, 1)
        if kind == "static":
            tx, ty, rot, z = 400, 300, 0.0, 1.0
        elif kind == "pan":  # pan up the wall and back down, with shake and roll
            ph = np.sin(np.pi * u)
            tx = 380 + 60 * ph + 3 * np.sin(17 * u)
            ty = 500 - 380 * ph + 3 * np.cos(13 * u)
            rot, z = np.radians(4 * np.sin(5 * u)), 1.0 + 0.1 * ph
        A = np.array([[z * np.cos(rot), -z * np.sin(rot), tx],
                      [z * np.sin(rot), z * np.cos(rot), ty], [0, 0, 1]])
        if kind != "static":  # a little perspective change
            A[2, :2] = [1e-5 * np.sin(3 * u), 2e-5 * np.cos(2 * u)]
        G.append(A)
    return np.array(G)


def make_video(path, G, climber=True):
    tex = wall_texture()
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    for i, g in enumerate(G):
        f = cv2.warpPerspective(tex, np.linalg.inv(g), (W, H), flags=cv2.INTER_LINEAR)
        if climber:  # a large moving dark shape that is not part of the wall
            cv2.ellipse(f, climber_xy(i), (55, 120), 0, 0, 360, (40, 35, 30), -1)
        vw.write(f)
    vw.release()


def expected_h(G, i):
    """Wall coordinates are frame-0 pixels: frame i -> texture -> frame 0."""
    return np.linalg.inv(G[0]) @ G[i]


def corner_error(Hest, Hexp):
    """Worst error over a grid covering the frame (10-90 % of width and height), px."""
    gx, gy = np.meshgrid(np.linspace(0.1, 0.9, 5) * W, np.linspace(0.1, 0.9, 5) * H)
    c = np.column_stack([gx.ravel(), gy.ravel()])
    return float(np.max(np.linalg.norm(apply_h(Hest, c) - apply_h(Hexp, c), axis=1)))


@pytest.fixture(scope="module")
def pan_clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("cam") / "pan.mp4"
    G = camera_motion(120, "pan")
    make_video(path, G)
    cam, wall = estimate_camera(path, climber_track(len(G)))
    return G, cam, wall


def test_moving_camera_registered_to_the_wall(pan_clip):
    G, cam, _ = pan_clip
    assert cam.n == len(G) and cam.ok.all() and not cam.static
    err = [corner_error(cam.H[i], expected_h(G, i)) for i in range(cam.n)]
    # the camera travels ~400 px; the frame corners (the worst place, extrapolated)
    # must land within a few px of the truth, also at the end, after panning away
    # and coming back (no drift)
    assert np.median(err) < 2.0
    assert max(err) < 6.0
    assert err[-1] < 2.0


def test_wall_image_has_no_climber(pan_clip):
    _, _, wall = pan_clip
    # the dark climber covered the middle of every frame, but the wall image comes
    # from the frames where each spot was uncovered: (almost) no dark pixels left
    seen = wall.coverage > 0
    dark = (wall.image.max(axis=2) < 50) & seen
    assert dark.mean() < 0.01
    assert seen.mean() > 0.3


def test_points_on_the_wall_stay_put(pan_clip):
    G, cam, _ = pan_clip
    # a fixed wall point, seen in every frame at a different image position
    tex_pt = apply_h(G[60], np.array([W / 2, H / 3]))
    img_pts = np.array([apply_h(np.linalg.inv(g), tex_pt) for g in G])
    wall_pts = cam.to_wall(img_pts[:, None, :])[:, 0]
    assert np.ptp(img_pts[:, 1]) > 200  # moves a lot in the image...
    assert np.max(np.std(wall_pts, axis=0)) < 1.0  # ...but not on the wall


def test_static_camera(tmp_path):
    path = tmp_path / "static.mp4"
    make_video(path, camera_motion(40, "static"))
    cam, _ = estimate_camera(path, climber_track(40))
    assert cam.static and cam.ok.all()
    assert corner_error(cam.H[-1], np.eye(3)) < 2.0


def test_unmasked_person_does_not_drag_the_wall(tmp_path):
    """Without a pose (no mask) a large moving person costs accuracy, but the
    registration must not follow the person (it moves ~120 px)."""
    path = tmp_path / "nomask.mp4"
    G = camera_motion(120, "pan")
    make_video(path, G)
    cam, _ = estimate_camera(path)
    err = [corner_error(cam.H[i], expected_h(G, i)) for i in range(cam.n)]
    assert np.median(err) < 4.0 and max(err) < 20.0


def test_lost_frames_and_recovery(tmp_path):
    """Blank frames (camera covered) are LOST; registration resumes afterwards."""
    path = tmp_path / "blank.mp4"
    G = camera_motion(90, "pan")
    make_video(path, G, climber=False)
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    for k in range(40, 50):
        frames[k][:] = 0
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    for f in frames:
        vw.write(f)
    vw.release()
    cam, _ = estimate_camera(path)
    assert (cam.status[41:49] == LOST).all()
    assert cam.ok[55:].all()
    assert corner_error(cam.H[-1], expected_h(G, len(G) - 1)) < 3.0


def test_save_load_and_summary(pan_clip, tmp_path):
    _, cam, _ = pan_clip
    cam.save(tmp_path / "camera.csv")
    back = CameraTrack.load(tmp_path / "camera.csv")
    np.testing.assert_allclose(back.H, cam.H, rtol=1e-6, atol=1e-6)
    assert back.frame_size == cam.frame_size and back.static == cam.static
    s = camera_summary(back, px_per_m=100.0)
    assert s["registered_pct"] == 100.0 and s["pan_px"][1] > 300
