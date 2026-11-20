"""Wall scans (engine/wallmodel.py): reading a scan, the standard frame, fitting a camera."""

import json

import cv2
import numpy as np

from climbvision.engine.wallmodel import (
    ScanMetric,
    WallModel,
    fit_clip,
    fit_from_marks,
    load_ply,
    load_scan_metric,
    render,
)


def _scan(rng):
    """A 4 m x 4 m wall leaning 20 deg over a floor, with coloured holds; z-up, the wall
    along x, the floor in front at -y; then rotated so z is no longer up (like a raw scan)."""
    pts, cols = [], []
    xs, hs = np.meshgrid(np.linspace(-2, 2, 160), np.linspace(0, 4, 160))
    lean = np.tan(np.radians(20))
    wall = np.c_[xs.ravel(), -hs.ravel() * lean, hs.ravel()]  # overhang: top leans to -y
    pts.append(wall)
    cols.append(np.tile([210, 190, 160], (len(wall), 1)))
    fx, fy = np.meshgrid(np.linspace(-2, 2, 120), np.linspace(-3, 0.3, 100))
    floor = np.c_[fx.ravel(), fy.ravel(), np.zeros(fx.size)]
    pts.append(floor)
    cols.append(np.tile([120, 120, 120], (len(floor), 1)))
    holds = [(-1.5, 0.5), (1.4, 0.8), (0.0, 1.6), (-1.0, 2.4), (1.0, 2.9), (-0.3, 3.5), (0.6, 1.0)]
    for x, h in holds:
        b = rng.normal(0, 0.05, (400, 3)) + [x, -h * lean - 0.05, h]
        pts.append(b)
        cols.append(np.tile([200, 20, 30], (len(b), 1)))
    P = np.vstack(pts)
    C = np.vstack(cols).astype(np.uint8)
    Rz = cv2.Rodrigues(np.array([np.pi / 2, 0, 0]))[0]  # a raw scan: y up (like Stray)
    return P @ Rz.T, C, holds


def _write_ply(path, P, C):
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"),
                   ("blue", "u1")])
    a = np.zeros(len(P), dt)
    a["x"], a["y"], a["z"] = P.T
    a["red"], a["green"], a["blue"] = C.T
    head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\n"
            "property float x\nproperty float y\nproperty float z\nproperty uchar red\n"
            "property uchar green\nproperty uchar blue\nend_header\n").encode()
    path.write_bytes(head + a.tobytes())


def test_scan_to_the_standard_frame_and_landmarks(tmp_path):
    rng = np.random.default_rng(0)
    P, C, holds = _scan(rng)
    _write_ply(tmp_path / "s.ply", P, C)
    P2, C2 = load_ply(tmp_path / "s.ply")
    assert len(P2) == len(P) and (C2 == C).all()
    m = WallModel.from_scan("t", tmp_path / "s.ply")
    assert abs(np.percentile(m.P[:, 2], 1)) < 0.05  # the floor at z = 0
    assert 3.9 < m.info["height_m"] < 4.3
    hz = sorted(d["xyz"][2] for d in m.landmarks)
    assert len(m.landmarks) == len(holds)
    assert np.allclose(hz, sorted(h for _, h in holds), atol=0.05)
    # the overhang leans towards the climber: the top holds are nearer (smaller y)
    top = max(m.landmarks, key=lambda d: d["xyz"][2])
    low = min(m.landmarks, key=lambda d: d["xyz"][2])
    assert top["xyz"][1] < low["xyz"][1]
    # x to the right when facing the wall: the left-most hold is at x -1.5
    lm = min(m.landmarks, key=lambda d: d["xyz"][0])
    assert abs(lm["xyz"][2] - 0.5) < 0.05


def test_camera_from_marked_pairs_and_metres_from_the_scan(tmp_path):
    rng = np.random.default_rng(1)
    P, C, _ = _scan(rng)
    _write_ply(tmp_path / "s.ply", P, C)
    m = WallModel.from_scan("t", tmp_path / "s.ply")
    frame = (1080, 1920)
    f = 0.75 * 1920
    K = np.array([[f, 0, 540], [0, f, 960], [0, 0, 1.0]])
    eye = np.array([0.3, -5.0, 1.5])
    fwd = np.array([0, 1.0, 0.25])
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    t = -R @ eye
    P3 = np.array([d["xyz"] for d in m.landmarks])
    p = (P3 @ R.T + t) @ K.T
    p2 = p[:, :2] / p[:, 2:]
    fit = fit_from_marks(P3, p2, frame)
    assert abs(fit.K[0, 0] - f) / f < 0.03
    assert np.linalg.norm(fit.camera_m - eye) < 0.1
    # a clip folder: the wall picture is the frame at half size, offset
    T = np.array([[0.5, 0, 10.0], [0, 0.5, 20.0], [0, 0, 1.0]])
    img, _ = render(m, T @ K, R, t, 560, 1000)
    d = tmp_path / "clip"
    d.mkdir()
    cv2.imwrite(str(d / "wall.jpg"), img)
    (d / "wall.json").write_text(json.dumps({"T": T.tolist(), "size": [560, 1000]}))
    (d / "summary.json").write_text(json.dumps({"frame_size": list(frame)}))
    q = np.c_[p2, np.ones(len(p2))] @ T.T
    marks = [(dd["id"], float(a), float(b)) for dd, (a, b, _) in zip(m.landmarks, q, strict=True)]
    fit_clip(m, d, marks, do_refine=False)
    met = load_scan_metric(d)
    assert isinstance(met, ScanMetric)
    for dd, (a, b) in zip(m.landmarks, p2, strict=True):  # heights of the holds, wall coords
        hz = met.xyz_at(np.array([a, b]))[2]
        assert abs(hz - dd["xyz"][2]) < 0.12
    assert (d / "wall_fit.jpg").exists()
