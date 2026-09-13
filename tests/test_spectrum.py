"""
tests/test_spectrum.py — Tests for module1_ocean/spectrum.py

All tests use a fine frequency grid (20 000 points) to ensure numerical
integration errors are negligible and do not cause false failures.

Run with:  pytest tests/test_spectrum.py -v
"""

import numpy as np
import pytest

from module1_ocean.spectrum import (
    pierson_moskowitz,
    jonswap,
    spectral_moments,
    significant_wave_height_from_spectrum,
    validate_spectrum,
    peak_frequency,
    peak_period,
)

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

# Fine grid: resolves the PM/JONSWAP peak well and keeps integration error small
F = np.linspace(0.001, 2.0, 20_000)

# Representative development sea state
HS = 1.0   # m
TP = 10.0  # s


@pytest.fixture(scope="module")
def pm_spectrum():
    return pierson_moskowitz(F, HS, TP)


@pytest.fixture(scope="module")
def jonswap_spectrum():
    return jonswap(F, HS, TP, gamma=3.3)


# ---------------------------------------------------------------------------
# 1. PM output shape matches input
# ---------------------------------------------------------------------------

def test_pm_output_shape(pm_spectrum):
    """PM spectrum array must have the same shape as the input frequency array."""
    assert pm_spectrum.shape == F.shape


def test_pm_output_shape_scalar_f():
    """PM must handle a single-element frequency array."""
    f_single = np.array([0.1])
    S = pierson_moskowitz(f_single, HS, TP)
    assert S.shape == (1,)


# ---------------------------------------------------------------------------
# 2. PM spectrum is non-negative
# ---------------------------------------------------------------------------

def test_pm_non_negative(pm_spectrum):
    """PM spectrum must be non-negative at all frequencies."""
    assert np.all(pm_spectrum >= 0.0)


# ---------------------------------------------------------------------------
# 3. PM at f=0 is finite and equals zero
# ---------------------------------------------------------------------------

def test_pm_at_zero_frequency():
    """PM spectrum must return exactly 0.0 at f=0 (not NaN or inf)."""
    f_with_zero = np.concatenate([[0.0], F])
    S = pierson_moskowitz(f_with_zero, HS, TP)
    assert np.isfinite(S[0]), "S(f=0) must be finite"
    assert S[0] == 0.0, "S(f=0) must be zero"


# ---------------------------------------------------------------------------
# 4. Invalid Hs/Tp inputs raise ValueError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("Hs, Tp", [
    (0.0,  10.0),   # Hs = 0
    (-1.0, 10.0),   # Hs < 0
    (1.0,  0.0),    # Tp = 0
    (1.0,  -5.0),   # Tp < 0
])
def test_pm_invalid_hs_tp_raises(Hs, Tp):
    """PM must raise ValueError for non-physical Hs or Tp."""
    with pytest.raises(ValueError):
        pierson_moskowitz(F, Hs, Tp)


def test_pm_negative_frequency_raises():
    """PM must raise ValueError if any frequency is negative."""
    f_bad = np.array([-0.1, 0.1, 0.2])
    with pytest.raises(ValueError, match="non-negative"):
        pierson_moskowitz(f_bad, HS, TP)


# ---------------------------------------------------------------------------
# 5. JONSWAP output shape matches input
# ---------------------------------------------------------------------------

def test_jonswap_output_shape(jonswap_spectrum):
    """JONSWAP spectrum array must have the same shape as the input frequency array."""
    assert jonswap_spectrum.shape == F.shape


# ---------------------------------------------------------------------------
# 6. JONSWAP spectrum is non-negative
# ---------------------------------------------------------------------------

def test_jonswap_non_negative(jonswap_spectrum):
    """JONSWAP spectrum must be non-negative at all frequencies."""
    assert np.all(jonswap_spectrum >= 0.0)


# ---------------------------------------------------------------------------
# 7. JONSWAP at f=0 is finite and equals zero
# ---------------------------------------------------------------------------

def test_jonswap_at_zero_frequency():
    """JONSWAP spectrum must return exactly 0.0 at f=0 (not NaN or inf)."""
    f_with_zero = np.concatenate([[0.0], F])
    S = jonswap(f_with_zero, HS, TP, gamma=3.3)
    assert np.isfinite(S[0]), "S_J(f=0) must be finite"
    assert S[0] == 0.0, "S_J(f=0) must be zero"


# ---------------------------------------------------------------------------
# 8. gamma <= 0 raises ValueError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gamma", [0.0, -1.0, -100.0])
def test_jonswap_invalid_gamma_raises(gamma):
    """JONSWAP must raise ValueError for gamma <= 0."""
    with pytest.raises(ValueError, match="gamma"):
        jonswap(F, HS, TP, gamma=gamma)


# ---------------------------------------------------------------------------
# 9. gamma=1 reduces to the underlying PM spectrum
# ---------------------------------------------------------------------------

def test_jonswap_gamma1_equals_pm():
    """
    JONSWAP with gamma=1 must equal the PM spectrum exactly.

    When gamma=1, gamma^r(f) = 1 for all f, so the enhancement factor
    is identically 1.  The rescaling constant C = m0_PM / m0_raw = 1
    trivially, so S_J = S_PM exactly.
    """
    S_pm = pierson_moskowitz(F, HS, TP)
    S_j1 = jonswap(F, HS, TP, gamma=1.0)
    np.testing.assert_array_equal(
        S_j1, S_pm,
        err_msg="JONSWAP(gamma=1) must equal PM spectrum exactly",
    )


# ---------------------------------------------------------------------------
# 10. spectral_moments returns finite values
# ---------------------------------------------------------------------------

def test_spectral_moments_finite(pm_spectrum):
    """All spectral moments must be finite for a valid PM spectrum."""
    moments = spectral_moments(F, pm_spectrum)
    for key, val in moments.items():
        assert np.isfinite(val), f"Moment {key} is not finite: {val}"


def test_spectral_moments_keys(pm_spectrum):
    """spectral_moments must return m0, m1, and m2."""
    moments = spectral_moments(F, pm_spectrum)
    assert "m0" in moments
    assert "m1" in moments
    assert "m2" in moments


def test_spectral_moments_m0_positive(pm_spectrum):
    """m0 must be positive for a non-trivial spectrum."""
    assert spectral_moments(F, pm_spectrum)["m0"] > 0.0


# ---------------------------------------------------------------------------
# 11. significant_wave_height_from_spectrum reconstructs Hs accurately
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("Hs_target, Tp_target", [
    (1.0, 10.0),
    (0.5, 8.0),
    (2.0, 12.0),
    (1.5, 6.0),
])
def test_pm_hs_reconstruction(Hs_target, Tp_target):
    """
    Hs reconstructed from PM spectrum must match target Hs within 0.1%.

    Uses a fine grid to keep numerical integration error negligible.
    """
    S = pierson_moskowitz(F, Hs_target, Tp_target)
    Hs_recon = significant_wave_height_from_spectrum(F, S)
    rel_err = abs(Hs_recon - Hs_target) / Hs_target
    assert rel_err < 1e-3, (
        f"PM Hs reconstruction error {rel_err:.2e} exceeds 0.1% "
        f"for Hs={Hs_target}, Tp={Tp_target}"
    )


@pytest.mark.parametrize("Hs_target, Tp_target", [
    (1.0, 10.0),
    (0.5, 8.0),
    (2.0, 12.0),
])
def test_jonswap_hs_reconstruction(Hs_target, Tp_target):
    """
    Hs reconstructed from JONSWAP spectrum must match target Hs within 0.1%.

    The rescaling inside jonswap() ensures this holds for any gamma > 0.
    """
    S = jonswap(F, Hs_target, Tp_target, gamma=3.3)
    Hs_recon = significant_wave_height_from_spectrum(F, S)
    rel_err = abs(Hs_recon - Hs_target) / Hs_target
    assert rel_err < 1e-3, (
        f"JONSWAP Hs reconstruction error {rel_err:.2e} exceeds 0.1% "
        f"for Hs={Hs_target}, Tp={Tp_target}"
    )


# ---------------------------------------------------------------------------
# 12. validate_spectrum reports small relative error on a fine grid
# ---------------------------------------------------------------------------

def test_validate_spectrum_small_error_pm():
    """validate_spectrum must report < 0.1% relative error for PM on fine grid."""
    S = pierson_moskowitz(F, HS, TP)
    result = validate_spectrum(F, S, HS)
    assert result["rel_error"] < 1e-3, (
        f"PM validation relative error {result['rel_error']:.2e} exceeds 0.1%"
    )


def test_validate_spectrum_small_error_jonswap():
    """validate_spectrum must report < 0.1% relative error for JONSWAP on fine grid."""
    S = jonswap(F, HS, TP, gamma=3.3)
    result = validate_spectrum(F, S, HS)
    assert result["rel_error"] < 1e-3, (
        f"JONSWAP validation relative error {result['rel_error']:.2e} exceeds 0.1%"
    )


def test_validate_spectrum_keys():
    """validate_spectrum must return all required keys."""
    S = pierson_moskowitz(F, HS, TP)
    result = validate_spectrum(F, S, HS)
    for key in ("m0", "Hs_reconstructed", "target_Hs", "abs_error_m", "rel_error"):
        assert key in result, f"Missing key: {key}"


def test_validate_spectrum_does_not_modify_spectrum():
    """validate_spectrum must not modify the input spectrum array."""
    S = pierson_moskowitz(F, HS, TP)
    S_copy = S.copy()
    validate_spectrum(F, S, HS)
    np.testing.assert_array_equal(S, S_copy, err_msg="validate_spectrum modified S")


# ---------------------------------------------------------------------------
# 13. peak_frequency returns a positive frequency
# ---------------------------------------------------------------------------

def test_peak_frequency_positive_pm(pm_spectrum):
    """peak_frequency must return a positive value for PM spectrum."""
    fp = peak_frequency(F, pm_spectrum)
    assert fp > 0.0


def test_peak_frequency_positive_jonswap(jonswap_spectrum):
    """peak_frequency must return a positive value for JONSWAP spectrum."""
    fp = peak_frequency(F, jonswap_spectrum)
    assert fp > 0.0


def test_peak_frequency_near_fp():
    """
    Numerical peak frequency must be close to fp = 1/Tp.

    For the f-domain PM formulation used here:
        S(f) ∝ f⁻⁵ · exp(−1.25·(fp/f)⁴)

    Setting dS/df = 0 (equivalently dS/dx = 0 with x = fp/f) gives x = 1,
    i.e. f_peak = fp = 1/Tp exactly.

    Note: the (5/4)^(1/4) shift appears in the omega-domain PM formulation
    and does NOT apply to this f-domain form.

    We allow 0.1% tolerance for grid discretisation.
    """
    fp_param = 1.0 / TP
    S = pierson_moskowitz(F, HS, TP)
    f_peak_numerical = peak_frequency(F, S)
    rel_err = abs(f_peak_numerical - fp_param) / fp_param
    assert rel_err < 0.001, (
        f"Numerical peak {f_peak_numerical:.5f} Hz deviates {rel_err:.2%} "
        f"from fp = {fp_param:.5f} Hz"
    )


# ---------------------------------------------------------------------------
# 14. peak_period returns a positive period
# ---------------------------------------------------------------------------

def test_peak_period_positive_pm(pm_spectrum):
    """peak_period must return a positive value for PM spectrum."""
    Tp_spec = peak_period(F, pm_spectrum)
    assert Tp_spec > 0.0


def test_peak_period_positive_jonswap(jonswap_spectrum):
    """peak_period must return a positive value for JONSWAP spectrum."""
    Tp_spec = peak_period(F, jonswap_spectrum)
    assert Tp_spec > 0.0


def test_peak_period_consistent_with_peak_frequency(pm_spectrum):
    """peak_period must equal 1 / peak_frequency."""
    fp = peak_frequency(F, pm_spectrum)
    Tp_spec = peak_period(F, pm_spectrum)
    assert abs(Tp_spec - 1.0 / fp) < 1e-10


# ---------------------------------------------------------------------------
# 15. JONSWAP has enhanced peak energy relative to PM when gamma > 1
# ---------------------------------------------------------------------------

def test_jonswap_enhanced_peak_energy():
    """
    JONSWAP (gamma=3.3) must have greater spectral density than PM near fp.

    The peak enhancement factor gamma^r(fp) = gamma^1 = gamma > 1 at f=fp.
    After rescaling, the peak of S_J must still exceed the peak of S_PM
    because the energy is redistributed toward the peak.
    """
    S_pm = pierson_moskowitz(F, HS, TP)
    S_j = jonswap(F, HS, TP, gamma=3.3)
    fp_param = 1.0 / TP
    # Find the index closest to fp
    idx_fp = np.argmin(np.abs(F - fp_param))
    assert S_j[idx_fp] > S_pm[idx_fp], (
        "JONSWAP spectral density at fp must exceed PM when gamma > 1"
    )


# ---------------------------------------------------------------------------
# 16. Increasing gamma changes spectral shape, not just array dimensions
# ---------------------------------------------------------------------------

def test_jonswap_gamma_changes_shape():
    """
    Different gamma values must produce different spectral shapes.

    Specifically, a higher gamma must produce a sharper (more peaked) spectrum
    with greater energy density at fp and less energy in the tails.
    """
    S_low  = jonswap(F, HS, TP, gamma=1.5)
    S_high = jonswap(F, HS, TP, gamma=5.0)

    # Both must have the same shape (array dimensions unchanged)
    assert S_low.shape == S_high.shape

    # Higher gamma → more energy at peak
    fp_param = 1.0 / TP
    idx_fp = np.argmin(np.abs(F - fp_param))
    assert S_high[idx_fp] > S_low[idx_fp], (
        "Higher gamma must produce greater spectral density at fp"
    )

    # Higher gamma → less energy far from peak (tails suppressed)
    # Check at f = 3 * fp (well into the tail)
    f_tail = 3.0 * fp_param
    idx_tail = np.argmin(np.abs(F - f_tail))
    assert S_high[idx_tail] < S_low[idx_tail], (
        "Higher gamma must produce less spectral density in the tail"
    )
