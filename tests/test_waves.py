"""
tests/test_waves.py — Tests for module1_ocean/waves.py

Run with:  pytest tests/test_waves.py -v
"""

import numpy as np
import pytest

from module1_ocean.waves import (
    frequency_grid,
    component_amplitudes,
    random_phases,
    synthesize_surface_elevation,
    significant_wave_height_from_timeseries,
    surface_statistics,
)
from module1_ocean.spectrum import jonswap, significant_wave_height_from_spectrum

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Standard development sea state
HS = 1.0    # m
TP = 10.0   # s

# Frequency grid used across most tests
N_COMP = 512
F_MIN  = 0.01   # Hz
F_MAX  = 2.0    # Hz
F      = frequency_grid(F_MIN, F_MAX, N_COMP)

# JONSWAP spectrum on that grid
S_JONSWAP = jonswap(F, HS, TP, gamma=3.3)

# Time array: 1 hour at 0.5 s resolution — long enough for Hs convergence
DT   = 0.5    # s
T    = np.arange(0.0, 3600.0, DT)

FIXED_SEED = 42


# ---------------------------------------------------------------------------
# 1. frequency_grid returns 1-D increasing frequencies
# ---------------------------------------------------------------------------

def test_frequency_grid_shape():
    """frequency_grid must return a 1-D array of the requested length."""
    f = frequency_grid(0.01, 2.0, 100)
    assert f.ndim == 1
    assert len(f) == 100


def test_frequency_grid_strictly_increasing():
    """All consecutive differences must be positive."""
    f = frequency_grid(0.01, 2.0, 100)
    assert np.all(np.diff(f) > 0)


def test_frequency_grid_bounds():
    """First and last values must match f_min and f_max."""
    f = frequency_grid(0.05, 1.5, 200)
    assert f[0] == pytest.approx(0.05)
    assert f[-1] == pytest.approx(1.5)


def test_frequency_grid_all_positive():
    """All frequencies must be positive."""
    f = frequency_grid(0.01, 2.0, 50)
    assert np.all(f > 0.0)


# ---------------------------------------------------------------------------
# 2. Invalid frequency_grid arguments raise exceptions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("f_min, f_max, n", [
    (0.0,  2.0,  100),   # f_min = 0
    (-0.1, 2.0,  100),   # f_min < 0
    (1.0,  0.5,  100),   # f_max < f_min
    (1.0,  1.0,  100),   # f_max == f_min
    (0.01, 2.0,  1),     # n_components < 2
    (0.01, 2.0,  0),     # n_components = 0
])
def test_frequency_grid_invalid_raises(f_min, f_max, n):
    """frequency_grid must raise ValueError for invalid arguments."""
    with pytest.raises(ValueError):
        frequency_grid(f_min, f_max, n)


# ---------------------------------------------------------------------------
# 3. component_amplitudes returns expected shape
# ---------------------------------------------------------------------------

def test_component_amplitudes_shape():
    """component_amplitudes must return an array with the same shape as f."""
    a = component_amplitudes(F, S_JONSWAP)
    assert a.shape == F.shape


# ---------------------------------------------------------------------------
# 4. Component amplitudes are non-negative
# ---------------------------------------------------------------------------

def test_component_amplitudes_non_negative():
    """All component amplitudes must be >= 0."""
    a = component_amplitudes(F, S_JONSWAP)
    assert np.all(a >= 0.0)


# ---------------------------------------------------------------------------
# 5. Zero spectrum produces zero amplitudes
# ---------------------------------------------------------------------------

def test_component_amplitudes_zero_spectrum():
    """A zero spectrum must produce zero amplitudes everywhere."""
    S_zero = np.zeros_like(F)
    a = component_amplitudes(F, S_zero)
    np.testing.assert_array_equal(a, np.zeros_like(F))


# ---------------------------------------------------------------------------
# 6. Non-uniform frequency spacing is rejected
# ---------------------------------------------------------------------------

def test_component_amplitudes_nonuniform_raises():
    """component_amplitudes must raise ValueError for non-uniform spacing."""
    f_nonuniform = np.array([0.01, 0.02, 0.05, 0.10, 0.20])
    S_dummy = np.ones(5)
    with pytest.raises(ValueError, match="uniform"):
        component_amplitudes(f_nonuniform, S_dummy)


def test_component_amplitudes_non_increasing_raises():
    """component_amplitudes must raise ValueError if f is not strictly increasing."""
    f_bad = np.array([0.1, 0.2, 0.15, 0.3])
    S_dummy = np.ones(4)
    with pytest.raises(ValueError):
        component_amplitudes(f_bad, S_dummy)


# ---------------------------------------------------------------------------
# 7. Negative spectral density is rejected
# ---------------------------------------------------------------------------

def test_component_amplitudes_negative_S_raises():
    """component_amplitudes must raise ValueError if any S value is negative."""
    S_bad = S_JONSWAP.copy()
    S_bad[10] = -0.001
    with pytest.raises(ValueError, match="non-negative"):
        component_amplitudes(F, S_bad)


# ---------------------------------------------------------------------------
# 8. random_phases returns correct number of phases
# ---------------------------------------------------------------------------

def test_random_phases_length():
    """random_phases must return an array of the requested length."""
    phases = random_phases(100, seed=0)
    assert len(phases) == 100


def test_random_phases_shape():
    """random_phases must return a 1-D array."""
    phases = random_phases(50, seed=1)
    assert phases.ndim == 1


# ---------------------------------------------------------------------------
# 9. Phases are in [0, 2π)
# ---------------------------------------------------------------------------

def test_random_phases_range():
    """All phases must be in [0, 2π)."""
    phases = random_phases(10_000, seed=7)
    assert np.all(phases >= 0.0)
    assert np.all(phases < 2.0 * np.pi)


# ---------------------------------------------------------------------------
# 10. Same seed produces identical phases
# ---------------------------------------------------------------------------

def test_random_phases_reproducible():
    """The same seed must produce identical phase arrays."""
    p1 = random_phases(200, seed=42)
    p2 = random_phases(200, seed=42)
    np.testing.assert_array_equal(p1, p2)


# ---------------------------------------------------------------------------
# 11. Different seeds normally produce different phases
# ---------------------------------------------------------------------------

def test_random_phases_different_seeds():
    """Different seeds must produce different phase arrays."""
    p1 = random_phases(200, seed=1)
    p2 = random_phases(200, seed=2)
    assert not np.array_equal(p1, p2)


# ---------------------------------------------------------------------------
# 12. synthesize_surface_elevation returns same length as t
# ---------------------------------------------------------------------------

def test_synthesis_output_length():
    """synthesize_surface_elevation must return an array of length len(t)."""
    eta = synthesize_surface_elevation(T, F, S_JONSWAP, seed=FIXED_SEED)
    assert len(eta) == len(T)


def test_synthesis_output_1d():
    """synthesize_surface_elevation must return a 1-D array."""
    eta = synthesize_surface_elevation(T, F, S_JONSWAP, seed=FIXED_SEED)
    assert eta.ndim == 1


# ---------------------------------------------------------------------------
# 13. A zero spectrum produces η(t) = 0
# ---------------------------------------------------------------------------

def test_synthesis_zero_spectrum():
    """A zero spectrum must produce η(t) = 0 for all t."""
    S_zero = np.zeros_like(F)
    eta = synthesize_surface_elevation(T, F, S_zero, seed=FIXED_SEED)
    np.testing.assert_array_equal(eta, np.zeros(len(T)))


# ---------------------------------------------------------------------------
# 14. Supplied phases make the result deterministic
# ---------------------------------------------------------------------------

def test_synthesis_supplied_phases_deterministic():
    """Supplying the same phases twice must produce identical η(t)."""
    phases = random_phases(N_COMP, seed=99)
    eta1 = synthesize_surface_elevation(T, F, S_JONSWAP, phases=phases)
    eta2 = synthesize_surface_elevation(T, F, S_JONSWAP, phases=phases)
    np.testing.assert_array_equal(eta1, eta2)


# ---------------------------------------------------------------------------
# 15. Same seed produces identical η(t)
# ---------------------------------------------------------------------------

def test_synthesis_same_seed_identical():
    """The same seed must produce identical η(t) arrays."""
    eta1 = synthesize_surface_elevation(T, F, S_JONSWAP, seed=FIXED_SEED)
    eta2 = synthesize_surface_elevation(T, F, S_JONSWAP, seed=FIXED_SEED)
    np.testing.assert_array_equal(eta1, eta2)


# ---------------------------------------------------------------------------
# 16. Different seeds normally produce different η(t)
# ---------------------------------------------------------------------------

def test_synthesis_different_seeds_differ():
    """Different seeds must produce different η(t) arrays."""
    eta1 = synthesize_surface_elevation(T, F, S_JONSWAP, seed=1)
    eta2 = synthesize_surface_elevation(T, F, S_JONSWAP, seed=2)
    assert not np.array_equal(eta1, eta2)


# ---------------------------------------------------------------------------
# 17. Inputs are not mutated
# ---------------------------------------------------------------------------

def test_synthesis_does_not_mutate_inputs():
    """synthesize_surface_elevation must not modify t, f, or S."""
    t_copy = T.copy()
    f_copy = F.copy()
    S_copy = S_JONSWAP.copy()
    synthesize_surface_elevation(T, F, S_JONSWAP, seed=FIXED_SEED)
    np.testing.assert_array_equal(T, t_copy)
    np.testing.assert_array_equal(F, f_copy)
    np.testing.assert_array_equal(S_JONSWAP, S_copy)


# ---------------------------------------------------------------------------
# 18. Single-frequency spectrum produces expected sinusoidal behaviour
# ---------------------------------------------------------------------------

def test_synthesis_single_frequency_cosine():
    """
    A single-frequency spectrum with a known phase must produce an exact cosine.

    Setup:
        f = [0.1] Hz,  S = [2.0] m²/Hz,  Δf = 0.01 Hz,  φ = 0
        a = √(2 · 2.0 · 0.01) = 0.2 m
        η(t) = 0.2 · cos(2π · 0.1 · t)

    Verified analytically:
        η(0)   = 0.2 · cos(0)    =  0.2 m
        η(2.5) = 0.2 · cos(π/2)  =  0.0 m
        η(5.0) = 0.2 · cos(π)    = -0.2 m
        η(10.) = 0.2 · cos(2π)   =  0.2 m
    """
    f_single = frequency_grid(0.095, 0.105, 2)  # two-point grid, Δf = 0.01 Hz
    # Use only the first component by zeroing the second
    S_single = np.array([2.0, 0.0])
    phi_zero = np.array([0.0, 0.0])

    t_test = np.array([0.0, 2.5, 5.0, 10.0])
    eta = synthesize_surface_elevation(t_test, f_single, S_single, phases=phi_zero)

    df = f_single[1] - f_single[0]
    a_expected = np.sqrt(2.0 * 2.0 * df)
    f0 = f_single[0]

    expected = a_expected * np.cos(2.0 * np.pi * f0 * t_test)
    np.testing.assert_allclose(eta, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# 19. Multi-component synthesis produces finite η(t)
# ---------------------------------------------------------------------------

def test_synthesis_finite_output():
    """synthesize_surface_elevation must produce finite values for a valid spectrum."""
    eta = synthesize_surface_elevation(T, F, S_JONSWAP, seed=FIXED_SEED)
    assert np.all(np.isfinite(eta)), "η(t) contains non-finite values"


# ---------------------------------------------------------------------------
# 20. Time-series Hs is reasonably close to spectral Hs
# ---------------------------------------------------------------------------

def test_synthesis_hs_convergence():
    """
    Hs estimated from a 1-hour JONSWAP realization must be within 10% of
    the spectral Hs.

    Justification for 10% tolerance:
        - Numerical experiments over 20 random seeds show max error < 0.2%
          for T=3600 s, N=512 components (see pre-implementation verification).
        - The 10% tolerance is deliberately generous to remain valid across
          any seed and any future changes to the frequency grid, while still
          catching genuine implementation errors (e.g. wrong Δf, wrong factor).
        - The spectral Hs itself is within 0.01% of the target (1.0 m) due
          to the rescaling in jonswap().

    Sea state: Hs=1.0 m, Tp=10.0 s, JONSWAP γ=3.3, T=3600 s, N=512.
    """
    Hs_spectral = significant_wave_height_from_spectrum(F, S_JONSWAP)
    eta = synthesize_surface_elevation(T, F, S_JONSWAP, seed=FIXED_SEED)
    Hs_ts = significant_wave_height_from_timeseries(eta)

    rel_err = abs(Hs_ts - Hs_spectral) / Hs_spectral
    assert rel_err < 0.10, (
        f"Time-series Hs={Hs_ts:.4f} m deviates {rel_err:.1%} from "
        f"spectral Hs={Hs_spectral:.4f} m (tolerance: 10%)"
    )


# ---------------------------------------------------------------------------
# surface_statistics tests
# ---------------------------------------------------------------------------

def test_surface_statistics_keys():
    """surface_statistics must return all required keys."""
    eta = synthesize_surface_elevation(T, F, S_JONSWAP, seed=FIXED_SEED)
    stats = surface_statistics(eta)
    for key in ("mean_m", "std_m", "min_m", "max_m", "Hs_est_m"):
        assert key in stats, f"Missing key: {key}"


def test_surface_statistics_with_time_includes_duration():
    """surface_statistics with t provided must include 'duration_s'."""
    eta = synthesize_surface_elevation(T, F, S_JONSWAP, seed=FIXED_SEED)
    stats = surface_statistics(eta, t=T)
    assert "duration_s" in stats
    assert stats["duration_s"] == pytest.approx(T[-1] - T[0])


def test_surface_statistics_does_not_mutate_eta():
    """surface_statistics must not modify the input array."""
    eta = synthesize_surface_elevation(T, F, S_JONSWAP, seed=FIXED_SEED)
    eta_copy = eta.copy()
    surface_statistics(eta, t=T)
    np.testing.assert_array_equal(eta, eta_copy)


def test_surface_statistics_hs_consistent():
    """Hs_est_m in surface_statistics must equal 4 * std_m."""
    eta = synthesize_surface_elevation(T, F, S_JONSWAP, seed=FIXED_SEED)
    stats = surface_statistics(eta)
    assert stats["Hs_est_m"] == pytest.approx(4.0 * stats["std_m"])
