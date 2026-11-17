"""Metric wall and climber profiles: perspective corrected with the body as the ruler,
heights from the climber's height, velocities."""

import numpy as np

from climbvision.engine.kinematics import derivatives
from climbvision.engine.metric import fit_wall_metric
from climbvision.engine.profile import Climber, find_climber, load_climbers
from climbvision.engine.skeleton import N_JOINTS, J

BODY_M = 1.40  # trunk (0.5) + leg (0.45 + 0.45) of the synthetic climber


def rigid_body(hip, rng):
    """A skeleton with fixed segment lengths (m), limbs at random angles; y up."""
    p = np.full((N_JOINTS, 2), np.nan)
    hip = np.asarray(hip, float)
    p[J.L_HIP], p[J.R_HIP] = hip + [-0.1, 0], hip + [0.1, 0]
    p[J.L_SHOULDER], p[J.R_SHOULDER] = hip + [-0.18, 0.5], hip + [0.18, 0.5]
    for side, sx in (("L", -1), ("R", 1)):
        a1, a2 = rng.uniform(-2.6, -0.5), rng.uniform(-0.4, 0.4)  # thigh, shank directions
        knee = p[J[f"{side}_HIP"]] + 0.45 * np.array([np.cos(a1) * sx * 0.6, np.sin(a1)])
        knee = p[J[f"{side}_HIP"]] + 0.45 * (knee - p[J[f"{side}_HIP"]]) / np.linalg.norm(
            knee - p[J[f"{side}_HIP"]])
        ank = knee + 0.45 * np.array([np.sin(a2), -np.cos(a2)])
        p[J[f"{side}_KNEE"]], p[J[f"{side}_ANKLE"]] = knee, ank
        p[J[f"{side}_TOE"]] = ank + [0.12 * sx, -0.03]
        p[J[f"{side}_HEEL"]] = ank + [-0.04 * sx, -0.03]
        b1 = rng.uniform(0.3, 2.8)
        elb = p[J[f"{side}_SHOULDER"]] + 0.3 * np.array([np.cos(b1) * sx, np.sin(b1)])
        b2 = b1 + rng.uniform(-0.6, 0.6)
        wr = elb + 0.27 * np.array([np.cos(b2) * sx, np.sin(b2)])
        p[J[f"{side}_ELBOW"]], p[J[f"{side}_WRIST"]] = elb, wr
        p[J[f"{side}_HAND"]] = wr + 0.08 * (wr - elb) / 0.27
    p[J.HEAD] = hip + [0, 0.75]
    return p


def climb_on_wall(n=600, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 30.0
    hips = np.column_stack([1.0 * np.sin(t / 4), 1.0 + 2.2 * t / t[-1]])  # 1 m -> 3.2 m up
    return np.stack([rigid_body(h, rng) for h in hips]), t


def to_picture(m, a, b, ppm=300.0):
    """Metres (y up) -> a picture of the wall in perspective (wall px, y down)."""
    x, y = m[..., 0], -m[..., 1]
    w = 1 + a * (x - 0.0) + b * (y + 2.0)
    return np.stack([x / w, y / w], -1) * ppm


def test_perspective_corrected_with_the_body():
    true, t = climb_on_wall()
    pic = to_picture(true, 0.0, 0.12)  # the top of the wall looks 1.3x bigger (overhang)
    on = np.ones(len(t), bool)
    flat = fit_wall_metric(pic, on, BODY_M, correct=False)
    fixed = fit_wall_metric(pic, on, BODY_M)
    assert fixed.corrected and not flat.corrected
    hip = (true[:, J.L_HIP] + true[:, J.R_HIP]) / 2
    true_gain = hip[-1, 1] - hip[0, 1]
    for m, tol in ((fixed, 0.06), (flat, None)):
        h = (m.to_m(pic[:, J.L_HIP]) + m.to_m(pic[:, J.R_HIP])) / 2
        gain = h[-1, 1] - h[0, 1]
        if tol:
            assert abs(gain - true_gain) / true_gain < tol, gain
        else:
            assert abs(gain - true_gain) / true_gain > 0.08  # one scale gets it wrong
    assert fixed.info["body_size_spread_after"] < fixed.info["body_size_spread_before"] / 2


def test_no_correction_when_the_view_is_square_on():
    true, t = climb_on_wall(seed=1)
    pic = to_picture(true, 0.0, 0.0)
    m = fit_wall_metric(pic, np.ones(len(t), bool), BODY_M, px_per_m=300.0)
    assert not m.corrected
    assert abs(m.dist_m(pic[0, J.L_HIP], pic[-1, J.L_HIP])
               - np.linalg.norm(true[0, J.L_HIP] - true[-1, J.L_HIP])) < 0.01


def test_heights_above_the_floor():
    true, t = climb_on_wall(seed=2)
    pic = to_picture(true, 0.0, 0.0)
    floor = to_picture(np.array([[0.0, 0.0], [0.3, 0.0]]), 0.0, 0.0)
    m = fit_wall_metric(pic, np.ones(len(t), bool), BODY_M, floor, px_per_m=300.0)
    assert abs(m.height_m(pic[-1, J.L_HIP]) - true[-1, J.L_HIP, 1]) < 0.02


def test_velocity_and_acceleration():
    t = np.arange(90) / 30.0
    x = np.column_stack([0.5 * t, 0.2 * t ** 2])  # 0.5 m/s across, accelerating 0.4 m/s^2 up
    x += np.random.default_rng(0).normal(0, 0.003, x.shape)  # pose jitter (3 mm)
    v, a = derivatives(x, t)
    mid = slice(10, 80)
    assert np.allclose(v[mid, 0], 0.5, atol=0.05)
    assert np.allclose(v[mid, 1], 0.4 * t[mid], atol=0.05)
    assert abs(np.nanmedian(a[mid, 1]) - 0.4) < 0.15
    x[40, 0] = np.nan
    v, _ = derivatives(x, t)
    assert np.isnan(v[40, 0]) and np.isfinite(v[20, 0])


def test_climber_profiles(tmp_path, monkeypatch):
    f = tmp_path / "climbers.yaml"
    f.write_text("default: alex\nclimbers:\n  alex: {height_m: 1.80, mass_kg: 70, "
                 "arm_span_m: 1.86, sex: male}\n  sam: {height_m: 1.62}\n")
    monkeypatch.setenv("CLIMBVISION_CLIMBERS", str(f))
    climbers, default = load_climbers()
    assert default == "alex" and set(climbers) == {"alex", "sam"}
    alex = find_climber()
    assert abs(alex.body_m - 0.779 * 1.80) < 1e-9 and alex.ape_index_cm == 6.0
    assert find_climber("sam").mass_kg is None
    assert Climber().summary()["height_assumed"]
    import pytest

    with pytest.raises(ValueError):
        find_climber("nobody")
