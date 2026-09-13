"""
tests/test_direction.py — Tests for module1_ocean/direction.py

Run with:  pytest tests/test_direction.py -v
"""

import numpy as np
import pytest

from module1_ocean.direction import (
    cosine_squared_spreading,
    directional_grid,
    normalize_directional_distribution,
    directional_weights,
    directional_spectrum,
)

N_DIR = 360   # fine grid for integration tests
THETA = directional_grid(N_DIR)
DTHETA = 2.0 * np.pi / N_DIR
MEAN_DIR = np.pi / 4.0   # 45° = NE


# ---------------------------------------------------------------------------
# 1. Directional grid is uniform
# ---------------------------------------------------------------------------

def test_directional_grid_uniform():
    theta = directional_grid(72)
    diffs = np.diff(theta)
    assert np.allclose(diffs, diffs[0], rtol=1e-12)


# ---------------------------------------------------------------------------
# 2. Grid covers [0, 2π) without duplicate endpoint
# ---------------------------------------------------------------------------

def test_directional_grid_range():
    theta = directional_grid(72)
    assert theta[0] == pytest.approx(0.0)
    assert theta[-1] < 2.0 * np.pi
    assert theta[-1] == pytest.approx(2.0 * np.pi - 2.0 * np.pi / 72)


def test_directional_grid_no_duplicate_2pi():
    theta = directional_grid(36)
    assert not np.any(np.isclose(theta, 2.0 * np.pi))


def test_directional_grid_invalid_raises():
    with pytest.raises(ValueError):
        directional_grid(1)


# ---------------------------------------------------------------------------
# 3. cos² spreading is non-negative
# ---------------------------------------------------------------------------

def test_spreading_non_negative():
    D = cosine_squared_spreading(THETA, MEAN_DIR)
    assert np.all(D >= 0.0)


# ---------------------------------------------------------------------------
# 4. Spreading is periodic (D(θ) = D(θ + 2π))
# ---------------------------------------------------------------------------

def test_spreading_periodic():
    theta = np.linspace(0, 2.0 * np.pi, 200, endpoint=False)
    D1 = cosine_squared_spreading(theta, MEAN_DIR)
    D2 = cosine_squared_spreading(theta + 2.0 * np.pi, MEAN_DIR)
    np.testing.assert_allclose(D1, D2, atol=1e-14)


# ---------------------------------------------------------------------------
# 5. Spreading peaks at mean direction
# ---------------------------------------------------------------------------

def test_spreading_peaks_at_mean_direction():
    D = cosine_squared_spreading(THETA, MEAN_DIR)
    peak_idx = np.argmax(D)
    assert abs(THETA[peak_idx] - MEAN_DIR) < DTHETA


def test_spreading_peak_value():
    """Peak value of (2/π)cos²(0) = 2/π."""
    D_peak = cosine_squared_spreading(np.array([MEAN_DIR]), MEAN_DIR)
    assert D_peak[0] == pytest.approx(2.0 / np.pi, rel=1e-12)


# ---------------------------------------------------------------------------
# 6. Normalized integral is approximately 1
# ---------------------------------------------------------------------------

def test_spreading_integral_unity():
    """∫D(θ)dθ ≈ 1 on a fine uniform grid."""
    D = cosine_squared_spreading(THETA, MEAN_DIR)
    integral = np.sum(D) * DTHETA
    assert abs(integral - 1.0) < 1e-4


def test_spreading_integral_unity_various_directions():
    for mean_dir in [0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 0.1, 5.9]:
        D = cosine_squared_spreading(THETA, mean_dir)
        integral = np.sum(D) * DTHETA
        assert abs(integral - 1.0) < 1e-4, f"Failed at mean_dir={mean_dir:.2f}"


# ---------------------------------------------------------------------------
# 7. Directional weights sum to 1
# ---------------------------------------------------------------------------

def test_directional_weights_sum_to_one():
    w = directional_weights(THETA, MEAN_DIR)
    assert abs(w.sum() - 1.0) < 1e-12


def test_directional_weights_non_negative():
    w = directional_weights(THETA, MEAN_DIR)
    assert np.all(w >= 0.0)


# ---------------------------------------------------------------------------
# 8. Directional spectrum has correct shape
# ---------------------------------------------------------------------------

def test_directional_spectrum_shape():
    f = np.linspace(0.05, 0.5, 30)
    S_f = np.ones(30)
    S2d = directional_spectrum(f, S_f, THETA, MEAN_DIR)
    assert S2d.shape == (30, N_DIR)


# ---------------------------------------------------------------------------
# 9. Integrating directional spectrum over θ reconstructs S(f)
# ---------------------------------------------------------------------------

def test_directional_spectrum_energy_conservation():
    """
    Σⱼ S(fᵢ, θⱼ) · Δθ must equal S(fᵢ) for each frequency.
    Tolerance: 0.1% (limited by directional grid resolution).
    """
    f = np.linspace(0.05, 0.5, 20)
    S_f = np.random.default_rng(0).uniform(0.1, 2.0, 20)
    S2d = directional_spectrum(f, S_f, THETA, MEAN_DIR)
    S_reconstructed = np.sum(S2d, axis=1) * DTHETA
    np.testing.assert_allclose(S_reconstructed, S_f, rtol=1e-3)


# ---------------------------------------------------------------------------
# 10. Changing mean direction rotates the distribution
# ---------------------------------------------------------------------------

def test_changing_mean_direction_rotates():
    D_north = cosine_squared_spreading(THETA, 0.0)
    D_east  = cosine_squared_spreading(THETA, np.pi / 2.0)
    # Peak of D_north should be near θ=0, peak of D_east near θ=π/2
    assert abs(THETA[np.argmax(D_north)] - 0.0) < DTHETA
    assert abs(THETA[np.argmax(D_east)] - np.pi / 2.0) < DTHETA


# ---------------------------------------------------------------------------
# 11. Wrap-around works near 0°/360°
# ---------------------------------------------------------------------------

def test_wraparound_near_zero():
    """
    Mean direction near 0 (North) should produce the same distribution
    as mean direction near 2π (also North).
    """
    D_0   = cosine_squared_spreading(THETA, 0.01)
    D_2pi = cosine_squared_spreading(THETA, 2.0 * np.pi + 0.01)
    np.testing.assert_allclose(D_0, D_2pi, atol=1e-14)


def test_wraparound_mean_direction_near_2pi():
    """Mean direction = 2π − 0.1 should peak near θ = 2π − 0.1 (≈ 0)."""
    mean = 2.0 * np.pi - 0.1
    D = cosine_squared_spreading(THETA, mean)
    peak_theta = THETA[np.argmax(D)]
    # Peak should be within one grid step of mean (mod 2π)
    diff = abs(((peak_theta - mean) + np.pi) % (2 * np.pi) - np.pi)
    assert diff < DTHETA


# ---------------------------------------------------------------------------
# 12. Negative distributions are rejected
# ---------------------------------------------------------------------------

def test_normalize_rejects_negative():
    D_bad = np.array([1.0, -0.1, 0.5])
    theta_small = np.linspace(0, 2 * np.pi, 3, endpoint=False)
    with pytest.raises(ValueError, match="non-negative"):
        normalize_directional_distribution(theta_small, D_bad)
