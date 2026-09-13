"""
tests/test_dispersion.py — Tests for module1_ocean/dispersion.py

Run with:  pytest tests/test_dispersion.py -v
"""

import numpy as np
import pytest

from module1_ocean.dispersion import (
    solve_wave_number,
    dispersion_residual,
    phase_velocity,
    group_velocity,
    wavelength_from_k,
    deep_water_wave_number,
)

G = 9.81

# Representative frequency array used across most tests
F = np.linspace(0.05, 0.5, 50)   # Hz


# ---------------------------------------------------------------------------
# 1. Positive frequencies return positive wave numbers
# ---------------------------------------------------------------------------

def test_positive_frequencies_give_positive_k():
    k = solve_wave_number(F, depth_m=50.0)
    assert np.all(k > 0.0)


# ---------------------------------------------------------------------------
# 2. f = 0 returns k = 0
# ---------------------------------------------------------------------------

def test_zero_frequency_returns_zero_k_scalar():
    k = solve_wave_number(0.0, depth_m=50.0)
    assert k == 0.0


def test_zero_frequency_in_array():
    f = np.array([0.0, 0.1, 0.2])
    k = solve_wave_number(f, depth_m=50.0)
    assert k[0] == 0.0
    assert np.all(k[1:] > 0.0)


# ---------------------------------------------------------------------------
# 3. Scalar input works
# ---------------------------------------------------------------------------

def test_scalar_input_returns_float():
    result = solve_wave_number(0.1, depth_m=30.0)
    assert isinstance(result, float)
    assert result > 0.0


# ---------------------------------------------------------------------------
# 4. Array input preserves shape
# ---------------------------------------------------------------------------

def test_array_input_preserves_shape():
    f = np.linspace(0.05, 0.5, 40)
    k = solve_wave_number(f, depth_m=50.0)
    assert k.shape == f.shape


# ---------------------------------------------------------------------------
# 5. depth_m <= 0 raises ValueError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("depth", [0.0, -1.0, -100.0])
def test_invalid_depth_raises(depth):
    with pytest.raises(ValueError):
        solve_wave_number(0.1, depth_m=depth)


def test_invalid_depth_group_velocity_raises():
    with pytest.raises(ValueError):
        group_velocity(0.1, 0.04, depth_m=0.0)


# ---------------------------------------------------------------------------
# 6. Negative frequencies raise ValueError
# ---------------------------------------------------------------------------

def test_negative_frequency_raises():
    with pytest.raises(ValueError):
        solve_wave_number(-0.1, depth_m=50.0)


def test_negative_frequency_in_array_raises():
    with pytest.raises(ValueError):
        solve_wave_number(np.array([-0.1, 0.1, 0.2]), depth_m=50.0)


# ---------------------------------------------------------------------------
# 7. Dispersion residual is close to zero for solved k
# ---------------------------------------------------------------------------

def test_dispersion_residual_near_zero():
    """
    For each solved k, |R| / ω² must be < 1e-10.
    Brent's method with xtol=rtol=1e-12 guarantees this.
    """
    depth = 30.0
    k = solve_wave_number(F, depth_m=depth)
    R = dispersion_residual(F, k, depth_m=depth)
    omega_sq = (2.0 * np.pi * F) ** 2
    rel_residual = np.abs(R) / omega_sq
    assert np.all(rel_residual < 1e-10), (
        f"Max relative residual = {rel_residual.max():.2e}"
    )


# ---------------------------------------------------------------------------
# 8. Increasing frequency increases k
# ---------------------------------------------------------------------------

def test_k_increases_with_frequency():
    """Higher frequency → shorter wavelength → larger k."""
    f = np.linspace(0.05, 0.5, 20)
    k = solve_wave_number(f, depth_m=50.0)
    assert np.all(np.diff(k) > 0.0)


# ---------------------------------------------------------------------------
# 9. Increasing depth changes k appropriately
# ---------------------------------------------------------------------------

def test_k_decreases_with_depth():
    """
    For a fixed frequency, deeper water → k approaches the deep-water limit
    from above (k decreases as depth increases toward the deep-water value).
    """
    f0 = 0.1
    depths = [5.0, 20.0, 50.0, 200.0]
    k_vals = [solve_wave_number(f0, depth_m=d) for d in depths]
    # k should decrease (or stay flat) as depth increases toward deep-water
    assert k_vals[0] > k_vals[1] > k_vals[2] > k_vals[3]


# ---------------------------------------------------------------------------
# 10. Deep-water limit: finite-depth k ≈ ω²/g for large kd
# ---------------------------------------------------------------------------

def test_deep_water_limit():
    """
    At d = 200 m and f = 0.1 Hz, kd ≈ 0.04 * 200 = 8 >> 1.
    The finite-depth k must agree with k_deep = ω²/g to within 0.1%.
    """
    f0 = 0.1
    d = 200.0
    k_fd = solve_wave_number(f0, depth_m=d)
    k_dw = deep_water_wave_number(f0)
    rel_err = abs(k_fd - k_dw) / k_dw
    assert rel_err < 1e-3, (
        f"k_finite={k_fd:.6f}, k_deep={k_dw:.6f}, rel_err={rel_err:.2e}"
    )


# ---------------------------------------------------------------------------
# 11. Shallow-water behaviour is physically sensible
# ---------------------------------------------------------------------------

def test_shallow_water_k_larger_than_deep():
    """
    In shallow water, tanh(kd) < 1, so k must be larger than k_deep = ω²/g
    to satisfy ω² = g k tanh(kd).
    """
    f0 = 0.1
    k_shallow = solve_wave_number(f0, depth_m=2.0)
    k_deep = deep_water_wave_number(f0)
    assert k_shallow > k_deep


def test_shallow_water_approaches_linear_limit():
    """
    In very shallow water (kd << 1), tanh(kd) ≈ kd, so:
        ω² ≈ g k² d  =>  k ≈ ω / sqrt(g d)
    Check within 1% for d = 0.5 m, f = 0.05 Hz.
    """
    f0 = 0.05
    d = 0.5
    k_fd = solve_wave_number(f0, depth_m=d)
    omega = 2.0 * np.pi * f0
    k_sw = omega / np.sqrt(G * d)
    rel_err = abs(k_fd - k_sw) / k_sw
    assert rel_err < 0.01, (
        f"k_finite={k_fd:.6f}, k_shallow={k_sw:.6f}, rel_err={rel_err:.2e}"
    )


# ---------------------------------------------------------------------------
# 12. Wavelength is λ = 2π / k
# ---------------------------------------------------------------------------

def test_wavelength_formula():
    k = solve_wave_number(F, depth_m=50.0)
    lam = wavelength_from_k(k)
    expected = 2.0 * np.pi / k
    np.testing.assert_allclose(lam, expected, rtol=1e-12)


def test_wavelength_k_zero_is_inf():
    lam = wavelength_from_k(np.array([0.0]))
    assert np.isinf(lam[0])


# ---------------------------------------------------------------------------
# 13. Phase velocity is c = ω / k
# ---------------------------------------------------------------------------

def test_phase_velocity_formula():
    depth = 50.0
    k = solve_wave_number(F, depth_m=depth)
    c = phase_velocity(F, k)
    omega = 2.0 * np.pi * F
    expected = omega / k
    np.testing.assert_allclose(c, expected, rtol=1e-12)


def test_phase_velocity_zero_frequency():
    c = phase_velocity(np.array([0.0]), np.array([0.0]))
    assert c[0] == 0.0


# ---------------------------------------------------------------------------
# 14. Group velocity is positive for positive frequencies
# ---------------------------------------------------------------------------

def test_group_velocity_positive():
    depth = 50.0
    k = solve_wave_number(F, depth_m=depth)
    cg = group_velocity(F, k, depth_m=depth)
    assert np.all(cg > 0.0)


# ---------------------------------------------------------------------------
# 15. Deep water: Cg ≈ c / 2
# ---------------------------------------------------------------------------

def test_deep_water_group_velocity():
    """
    At d = 200 m, f = 0.1 Hz: kd ≈ 8, deep-water limit applies.
    Cg / c must be within 0.1% of 0.5.
    """
    f0 = 0.1
    d = 200.0
    k = solve_wave_number(f0, depth_m=d)
    c = phase_velocity(f0, k)
    cg = group_velocity(f0, k, depth_m=d)
    ratio = cg / c
    assert abs(ratio - 0.5) < 1e-3, (
        f"Cg/c = {ratio:.6f}, expected ~0.5 in deep water"
    )


# ---------------------------------------------------------------------------
# 16. Shallow water: Cg approaches sqrt(g d)
# ---------------------------------------------------------------------------

def test_shallow_water_group_velocity():
    """
    In shallow water (kd << 1), n → 1 and Cg → c → sqrt(g d).
    Check within 1% for d = 0.5 m, f = 0.05 Hz.
    """
    f0 = 0.05
    d = 0.5
    k = solve_wave_number(f0, depth_m=d)
    cg = group_velocity(f0, k, depth_m=d)
    c_sw = np.sqrt(G * d)
    rel_err = abs(cg - c_sw) / c_sw
    assert rel_err < 0.01, (
        f"Cg={cg:.4f}, sqrt(gd)={c_sw:.4f}, rel_err={rel_err:.2e}"
    )


# ---------------------------------------------------------------------------
# 17. No NaN or infinite values for valid positive frequencies/depths
# ---------------------------------------------------------------------------

def test_no_nan_or_inf_in_outputs():
    """All dispersion outputs must be finite for valid inputs."""
    depths = [5.0, 20.0, 50.0, 200.0]
    for d in depths:
        k = solve_wave_number(F, depth_m=d)
        assert np.all(np.isfinite(k)), f"k has non-finite values at d={d}"

        c = phase_velocity(F, k)
        assert np.all(np.isfinite(c)), f"c has non-finite values at d={d}"

        cg = group_velocity(F, k, depth_m=d)
        assert np.all(np.isfinite(cg)), f"Cg has non-finite values at d={d}"

        lam = wavelength_from_k(k)
        assert np.all(np.isfinite(lam)), f"λ has non-finite values at d={d}"
