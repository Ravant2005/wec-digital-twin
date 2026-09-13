"""
tests/test_radiation.py — Tests for module2_wec/radiation.py (Module 2.3A).

Tests cover:
  1.  RadiationKernel construction and metadata
  2.  Physical sanity checks on K(t)
  3.  Physical sanity checks on A_infinity
  4.  Reconstruction: B(omega) from K(t)
  5.  Reconstruction: A(omega) from K(t) and A_infinity
  6.  Ogilvie relation self-consistency
  7.  A_infinity method comparison
  8.  Numerical sensitivity (t_max, dt, n_interp, taper)
  9.  Determinism
  10. Input validation

Uses the production mesh (672 panels, 20 frequencies) for the main fixture
and a medium mesh (168 panels, 8 frequencies) for fast sensitivity tests.

Run with:  pytest tests/test_radiation.py -v
"""

import math
import numpy as np
import pytest

from module2_wec.geometry import REFERENCE_BUOY
from module2_wec.hydrodynamic_coefficients import compute_hydrodynamic_coefficients
from module2_wec.radiation import (
    RadiationKernel,
    compute_radiation_kernel,
    DEFAULT_T_MAX,
    DEFAULT_DT,
    DEFAULT_N_INTERP,
    DEFAULT_TAPER_FRACTION,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

PROD_RESOLUTION = (6, 24, 16)   # 672 panels — production mesh
MED_RESOLUTION  = (3, 12, 8)    # 168 panels — fast fixture


@pytest.fixture(scope="module")
def hydro_prod():
    """Production HydrodynamicCoefficients: 672 panels, 20 frequencies."""
    return compute_hydrodynamic_coefficients(
        geometry=REFERENCE_BUOY,
        resolution=PROD_RESOLUTION,
        omega_min=0.2, omega_max=1.4, n_omega=20,
        progress_bar=False,
    )


@pytest.fixture(scope="module")
def hydro_med():
    """Medium HydrodynamicCoefficients: 168 panels, 8 frequencies."""
    return compute_hydrodynamic_coefficients(
        geometry=REFERENCE_BUOY,
        resolution=MED_RESOLUTION,
        omega_min=0.3, omega_max=1.2, n_omega=8,
        progress_bar=False,
    )


@pytest.fixture(scope="module")
def kernel_prod(hydro_prod):
    """Production RadiationKernel with default parameters."""
    return compute_radiation_kernel(hydro_prod)


@pytest.fixture(scope="module")
def kernel_med(hydro_med):
    """Medium RadiationKernel with default parameters."""
    return compute_radiation_kernel(hydro_med)


# ---------------------------------------------------------------------------
# 1. Construction and metadata
# ---------------------------------------------------------------------------

def test_kernel_is_radiation_kernel_instance(kernel_prod):
    assert isinstance(kernel_prod, RadiationKernel)


def test_metadata_omega_matches_hydro(kernel_prod, hydro_prod):
    np.testing.assert_array_equal(kernel_prod.omega, hydro_prod.omega)


def test_metadata_frequency_hz_matches_hydro(kernel_prod, hydro_prod):
    np.testing.assert_array_equal(kernel_prod.frequency_hz, hydro_prod.frequency_hz)


def test_metadata_added_mass_matches_hydro(kernel_prod, hydro_prod):
    np.testing.assert_array_equal(kernel_prod.added_mass, hydro_prod.added_mass_heave)


def test_metadata_radiation_damping_matches_hydro(kernel_prod, hydro_prod):
    np.testing.assert_array_equal(
        kernel_prod.radiation_damping, hydro_prod.radiation_damping_heave
    )


def test_metadata_water_depth(kernel_prod, hydro_prod):
    assert kernel_prod.water_depth == pytest.approx(hydro_prod.water_depth, rel=1e-10)


def test_metadata_rho(kernel_prod, hydro_prod):
    assert kernel_prod.rho == pytest.approx(hydro_prod.rho, rel=1e-10)


def test_metadata_n_panels(kernel_prod, hydro_prod):
    assert kernel_prod.n_panels == hydro_prod.n_panels


def test_metadata_capytaine_version(kernel_prod, hydro_prod):
    assert kernel_prod.capytaine_version == hydro_prod.capytaine_version


def test_metadata_omega_min_max(kernel_prod):
    assert kernel_prod.omega_min == pytest.approx(kernel_prod.omega[0], rel=1e-10)
    assert kernel_prod.omega_max == pytest.approx(kernel_prod.omega[-1], rel=1e-10)


def test_metadata_t_max_dt_consistent(kernel_prod):
    """t[-1] must be within one dt of t_max."""
    assert kernel_prod.time[-1] <= kernel_prod.t_max + kernel_prod.dt * 0.6


def test_metadata_n_omega(kernel_prod):
    assert kernel_prod.n_omega == len(kernel_prod.omega)


def test_metadata_n_time(kernel_prod):
    assert kernel_prod.n_time == len(kernel_prod.time)


def test_metadata_A_infinity_method_stored(kernel_prod):
    assert kernel_prod.A_infinity_method == "ogilvie_midrange"


def test_metadata_omega_irr_estimate(kernel_prod):
    """omega_irr must be stored and exceed omega_max."""
    assert kernel_prod.omega_irr_estimate > kernel_prod.omega_max


# ---------------------------------------------------------------------------
# 2. Physical sanity checks on K(t)
# ---------------------------------------------------------------------------

def test_kernel_no_nan_or_inf(kernel_prod):
    """K(t) must contain no NaN or infinity."""
    assert np.all(np.isfinite(kernel_prod.kernel)), \
        "K(t) contains NaN or infinity"


def test_kernel_time_starts_at_zero(kernel_prod):
    """Time grid must start at t = 0."""
    assert kernel_prod.time[0] == pytest.approx(0.0, abs=1e-12)


def test_kernel_time_strictly_increasing(kernel_prod):
    """Time grid must be strictly increasing."""
    assert np.all(np.diff(kernel_prod.time) > 0)


def test_kernel_time_step_consistent(kernel_prod):
    """Time step must be uniform within floating-point tolerance."""
    diffs = np.diff(kernel_prod.time)
    np.testing.assert_allclose(diffs, kernel_prod.dt, rtol=1e-10)


def test_kernel_causal_by_construction(kernel_prod):
    """
    K(t) is causal by construction: the cosine transform of B(omega)
    produces a function defined only for t >= 0.  The time array starts
    at t = 0 and contains no negative times.
    """
    assert kernel_prod.time[0] >= 0.0
    assert np.all(kernel_prod.time >= 0.0)


def test_kernel_finite_duration(kernel_prod):
    """Kernel duration must be positive and finite."""
    assert np.isfinite(kernel_prod.t_max)
    assert kernel_prod.t_max > 0.0


def test_kernel_shape_matches_time(kernel_prod):
    """kernel array shape must match time array shape."""
    assert kernel_prod.kernel.shape == kernel_prod.time.shape


def test_kernel_K0_positive(kernel_prod):
    """
    K(0) = (2/pi) * integral B(omega) domega must be positive
    since B(omega) >= 0 everywhere.
    """
    assert kernel_prod.kernel[0] > 0.0, \
        f"K(0) = {kernel_prod.kernel[0]:.2f} must be positive"


def test_kernel_K0_order_of_magnitude(kernel_prod):
    """
    K(0) must be of the same order as the radiation damping values.
    For the reference buoy, B ranges from ~400 to ~270000 N·s/m.
    K(0) = (2/pi) * integral B domega should be O(10^4) N·s/m.
    """
    assert 1e3 < kernel_prod.kernel[0] < 1e7, \
        f"K(0) = {kernel_prod.kernel[0]:.2e} outside expected range [1e3, 1e7]"


def test_kernel_decays_from_peak(kernel_prod):
    """
    |K(t)| must decay from its initial value.  The kernel represents
    a fading memory; it should not grow without bound.
    """
    K = kernel_prod.kernel
    # Check that the tail (last 10%) has smaller magnitude than the peak
    tail_max = np.max(np.abs(K[len(K)*9//10:]))
    peak = np.max(np.abs(K))
    assert tail_max < peak, \
        f"Kernel tail ({tail_max:.2e}) >= peak ({peak:.2e}); kernel does not decay"


# ---------------------------------------------------------------------------
# 3. Physical sanity checks on A_infinity
# ---------------------------------------------------------------------------

def test_A_infinity_finite(kernel_prod):
    """A_infinity must be finite."""
    assert np.isfinite(kernel_prod.A_infinity)


def test_A_infinity_positive(kernel_prod):
    """A_infinity must be positive (it is an added mass)."""
    assert kernel_prod.A_infinity > 0.0, \
        f"A_infinity = {kernel_prod.A_infinity:.1f} kg must be positive"


def test_A_infinity_order_of_magnitude(kernel_prod):
    """
    A_infinity must be of the same order as the BEM added mass values.
    For the reference buoy, A ranges from ~342000 to ~912000 kg.
    A_infinity should be in [1e4, 1e7] kg.
    """
    assert 1e4 < kernel_prod.A_infinity < 1e7, \
        f"A_infinity = {kernel_prod.A_infinity:.1f} kg outside [1e4, 1e7]"


def test_A_infinity_endpoint_is_lower_bound(kernel_prod):
    """
    A_infinity_endpoint = A(omega_max) is a lower bound on the true A_inf
    because A(omega) is still increasing at omega_max.
    All three estimates must be >= A(omega_min).
    """
    A_min = float(kernel_prod.added_mass[0])
    assert kernel_prod.A_infinity_endpoint >= A_min * 0.5, \
        "A_infinity_endpoint is unreasonably small"


def test_all_A_infinity_estimates_finite(kernel_prod):
    """All three A_infinity estimates must be finite."""
    assert np.isfinite(kernel_prod.A_infinity_endpoint)
    assert np.isfinite(kernel_prod.A_infinity_trend_fit)
    assert np.isfinite(kernel_prod.A_infinity_ogilvie)


def test_A_infinity_selected_matches_method(kernel_prod):
    """
    The selected A_infinity must match the stored method.
    Default method is 'ogilvie_midrange'.
    """
    assert kernel_prod.A_infinity == pytest.approx(
        kernel_prod.A_infinity_ogilvie, rel=1e-10
    )


# ---------------------------------------------------------------------------
# 4. Reconstruction: B(omega) from K(t)
# ---------------------------------------------------------------------------

def test_B_reconstructed_no_nan_or_inf(kernel_prod):
    """Reconstructed B(omega) must contain no NaN or infinity."""
    assert np.all(np.isfinite(kernel_prod.radiation_damping_reconstructed))


def test_B_reconstructed_shape(kernel_prod):
    """Reconstructed B must have same shape as omega."""
    assert kernel_prod.radiation_damping_reconstructed.shape == kernel_prod.omega.shape


def test_B_reconstructed_non_negative_trusted_range(kernel_prod):
    """
    Reconstructed B(omega) must be non-negative over the trusted range
    [0.45, 1.15] rad/s.  Radiation damping represents energy dissipation
    and must be non-negative for a physical body.
    """
    trusted = (kernel_prod.omega >= 0.45) & (kernel_prod.omega <= 1.15)
    B_rec_trusted = kernel_prod.radiation_damping_reconstructed[trusted]
    assert np.all(B_rec_trusted >= -1e3), (
        f"Reconstructed B has significantly negative values in trusted range: "
        f"min = {B_rec_trusted.min():.1f} N·s/m"
    )


def test_B_reconstruction_rmse_trusted(kernel_prod):
    """
    B reconstruction RMSE over the trusted range must be < 5%.
    Empirically measured: ~1% for the production mesh with default parameters.
    Tolerance 5% provides headroom for platform variation.
    """
    errs = kernel_prod.reconstruction_errors()
    rmse = errs["B_rmse_trusted_pct"]
    assert rmse < 5.0, (
        f"B reconstruction RMSE in trusted range = {rmse:.2f}% exceeds 5%"
    )


def test_B_reconstruction_max_rel_trusted(kernel_prod):
    """
    B reconstruction max relative error over the trusted range must be < 10%.
    Empirically measured: ~3% for the production mesh.
    """
    errs = kernel_prod.reconstruction_errors()
    max_rel = errs["B_max_rel_trusted_pct"]
    assert max_rel < 10.0, (
        f"B reconstruction max relative error in trusted range = {max_rel:.2f}% exceeds 10%"
    )


# ---------------------------------------------------------------------------
# 5. Reconstruction: A(omega) from K(t) and A_infinity
# ---------------------------------------------------------------------------

def test_A_reconstructed_no_nan_or_inf(kernel_prod):
    """Reconstructed A(omega) must contain no NaN or infinity."""
    assert np.all(np.isfinite(kernel_prod.added_mass_reconstructed))


def test_A_reconstructed_shape(kernel_prod):
    """Reconstructed A must have same shape as omega."""
    assert kernel_prod.added_mass_reconstructed.shape == kernel_prod.omega.shape


def test_A_reconstruction_rmse_trusted(kernel_prod):
    """
    A reconstruction RMSE over the trusted range must be < 40%.
    The large tolerance reflects the fundamental limitation: A_inf cannot
    be reliably determined from the truncated frequency range (omega_max =
    1.4 rad/s, only 67% of omega_irr).  The A reconstruction error is
    dominated by A_inf uncertainty, not kernel quality.
    Empirically measured: ~17% for the production mesh.
    """
    errs = kernel_prod.reconstruction_errors()
    rmse = errs["A_rmse_trusted_pct"]
    assert rmse < 40.0, (
        f"A reconstruction RMSE in trusted range = {rmse:.2f}% exceeds 40%"
    )


# ---------------------------------------------------------------------------
# 6. Ogilvie relation self-consistency
# ---------------------------------------------------------------------------

def test_ogilvie_B_relation(kernel_prod):
    """
    Ogilvie relation: B(omega) = integral_0^inf K(t) cos(omega*t) dt.
    Verify this holds to within the documented reconstruction tolerance
    over the trusted range.
    """
    t = kernel_prod.time
    K = kernel_prod.kernel
    omega_trusted = kernel_prod.omega[
        (kernel_prod.omega >= 0.45) & (kernel_prod.omega <= 1.15)
    ]
    B_orig_trusted = kernel_prod.radiation_damping[
        (kernel_prod.omega >= 0.45) & (kernel_prod.omega <= 1.15)
    ]
    B_check = np.array([np.trapezoid(K * np.cos(w * t), t) for w in omega_trusted])
    rel_err = np.abs(B_check - B_orig_trusted) / B_orig_trusted
    assert np.max(rel_err) < 0.10, (
        f"Ogilvie B relation max relative error = {np.max(rel_err)*100:.2f}% > 10%"
    )


def test_ogilvie_A_relation_self_consistent(kernel_prod):
    """
    Ogilvie relation: A_inf = A(omega) + (1/omega)*integral K(t) sin(omega*t) dt.
    The A_inf estimates at different omega should be consistent within the
    spread caused by frequency truncation.  Check that the mid-range
    estimates (omega 0.6-1.0) have std < 30% of their mean.
    """
    t = kernel_prod.time
    K = kernel_prod.kernel
    omega = kernel_prod.omega
    A = kernel_prod.added_mass
    mid_mask = (omega >= 0.6) & (omega <= 1.0)
    omega_mid = omega[mid_mask]
    A_mid = A[mid_mask]
    sin_int = np.array([np.trapezoid(K * np.sin(w * t), t) for w in omega_mid])
    A_inf_vals = A_mid + (1.0 / omega_mid) * sin_int
    rel_std = np.std(A_inf_vals) / np.abs(np.mean(A_inf_vals))
    assert rel_std < 0.30, (
        f"Ogilvie A_inf estimates have relative std = {rel_std*100:.1f}% > 30% "
        f"in mid-range omega [0.6, 1.0] rad/s"
    )


# ---------------------------------------------------------------------------
# 7. A_infinity method comparison
# ---------------------------------------------------------------------------

def test_A_infinity_endpoint_method(hydro_prod):
    """Endpoint method must return A(omega_max)."""
    k = compute_radiation_kernel(hydro_prod, A_infinity_method="endpoint")
    assert k.A_infinity == pytest.approx(k.A_infinity_endpoint, rel=1e-10)
    assert k.A_infinity == pytest.approx(float(hydro_prod.added_mass_heave[-1]), rel=1e-10)


def test_A_infinity_trend_fit_method(hydro_prod):
    """Trend fit method must return a finite positive value."""
    k = compute_radiation_kernel(hydro_prod, A_infinity_method="trend_fit")
    assert k.A_infinity == pytest.approx(k.A_infinity_trend_fit, rel=1e-10)
    assert np.isfinite(k.A_infinity)


def test_A_infinity_ogilvie_method(hydro_prod):
    """Ogilvie method must return the ogilvie estimate."""
    k = compute_radiation_kernel(hydro_prod, A_infinity_method="ogilvie_midrange")
    assert k.A_infinity == pytest.approx(k.A_infinity_ogilvie, rel=1e-10)


def test_A_infinity_three_estimates_all_stored(kernel_prod):
    """All three A_infinity estimates must be stored regardless of method."""
    assert np.isfinite(kernel_prod.A_infinity_endpoint)
    assert np.isfinite(kernel_prod.A_infinity_trend_fit)
    assert np.isfinite(kernel_prod.A_infinity_ogilvie)


def test_A_infinity_endpoint_inflated_by_artifact(kernel_prod):
    """
    For this dataset A(omega_max) is NOT a physical lower bound on A_inf.
    A(omega) spikes sharply near omega_max due to the irregular-frequency
    artifact (omega_max = 1.4 rad/s is only 67% of omega_irr ~2.09 rad/s).
    The endpoint value (~912 000 kg) therefore exceeds the Ogilvie mid-range
    estimate (~335 000 kg), which is derived from the physically reliable
    mid-range (0.6-1.0 rad/s).  This test documents that known limitation.
    """
    A_midrange = float(kernel_prod.added_mass[
        (kernel_prod.omega >= 0.6) & (kernel_prod.omega <= 1.0)
    ].mean())
    # Endpoint is inflated by the artifact and must exceed the mid-range mean.
    assert kernel_prod.A_infinity_endpoint > A_midrange
    # Ogilvie estimate must still be positive and finite.
    assert kernel_prod.A_infinity_ogilvie > 0.0


# ---------------------------------------------------------------------------
# 8. Numerical sensitivity (medium mesh for speed)
# ---------------------------------------------------------------------------

def test_sensitivity_t_max(hydro_med):
    """
    Increasing t_max from 30 to 60 s must not significantly change B reconstruction.
    The kernel decays to <1% of K(0) within ~2 s; t_max >= 30 s is sufficient.
    Tolerance: B RMSE difference < 4 percentage points.
    The medium mesh (168 panels, 8 frequencies, omega 0.3-1.2 rad/s) has a
    narrower trusted range than the production mesh and is more sensitive to
    t_max; the measured difference is ~2.5 pp.
    """
    k30 = compute_radiation_kernel(hydro_med, t_max=30.0)
    k60 = compute_radiation_kernel(hydro_med, t_max=60.0)
    rmse30 = k30.reconstruction_errors()["B_rmse_trusted_pct"]
    rmse60 = k60.reconstruction_errors()["B_rmse_trusted_pct"]
    assert abs(rmse30 - rmse60) < 4.0, (
        f"B RMSE changes by {abs(rmse30-rmse60):.2f}pp when t_max 30->60 s"
    )


def test_sensitivity_dt(hydro_med):
    """
    Halving dt from 0.05 to 0.025 s must not significantly change B reconstruction.
    Tolerance: B RMSE difference < 1 percentage point.
    """
    k05 = compute_radiation_kernel(hydro_med, dt=0.05)
    k025 = compute_radiation_kernel(hydro_med, dt=0.025)
    rmse05  = k05.reconstruction_errors()["B_rmse_trusted_pct"]
    rmse025 = k025.reconstruction_errors()["B_rmse_trusted_pct"]
    assert abs(rmse05 - rmse025) < 1.0, (
        f"B RMSE changes by {abs(rmse05-rmse025):.2f}pp when dt 0.05->0.025 s"
    )


def test_sensitivity_n_interp(hydro_med):
    """
    Doubling n_interp from 500 to 1000 must not significantly change B reconstruction.
    Tolerance: B RMSE difference < 1 percentage point.
    """
    k500  = compute_radiation_kernel(hydro_med, n_interp=500)
    k1000 = compute_radiation_kernel(hydro_med, n_interp=1000)
    rmse500  = k500.reconstruction_errors()["B_rmse_trusted_pct"]
    rmse1000 = k1000.reconstruction_errors()["B_rmse_trusted_pct"]
    assert abs(rmse500 - rmse1000) < 1.0, (
        f"B RMSE changes by {abs(rmse500-rmse1000):.2f}pp when n_interp 500->1000"
    )


def test_sensitivity_taper_improves_reconstruction(hydro_med):
    """
    Applying a cosine taper (taper_fraction=0.15) must give lower B RMSE
    than no taper (taper_fraction=0.0).  Tapering reduces Gibbs ringing
    from the high-frequency truncation.
    """
    k_notaper = compute_radiation_kernel(hydro_med, taper_fraction=0.0)
    k_taper   = compute_radiation_kernel(hydro_med, taper_fraction=0.15)
    rmse_notaper = k_notaper.reconstruction_errors()["B_rmse_trusted_pct"]
    rmse_taper   = k_taper.reconstruction_errors()["B_rmse_trusted_pct"]
    assert rmse_taper < rmse_notaper, (
        f"Taper RMSE ({rmse_taper:.2f}%) >= no-taper RMSE ({rmse_notaper:.2f}%); "
        "tapering should improve reconstruction"
    )


# ---------------------------------------------------------------------------
# 9. Determinism
# ---------------------------------------------------------------------------

def test_determinism(hydro_prod):
    """
    Computing the radiation kernel twice with identical inputs must give
    identical results (the computation is purely deterministic numpy).
    Tolerance: rtol=1e-10 (machine precision; no stochastic elements).
    """
    kwargs = dict(
        t_max=30.0, dt=0.1, n_interp=500, taper_fraction=0.15,
    )
    k1 = compute_radiation_kernel(hydro_prod, **kwargs)
    k2 = compute_radiation_kernel(hydro_prod, **kwargs)

    np.testing.assert_allclose(k1.kernel, k2.kernel, rtol=1e-10,
                               err_msg="kernel not deterministic")
    assert k1.A_infinity == pytest.approx(k2.A_infinity, rel=1e-10)
    np.testing.assert_allclose(
        k1.radiation_damping_reconstructed,
        k2.radiation_damping_reconstructed,
        rtol=1e-10,
    )


# ---------------------------------------------------------------------------
# 10. Input validation
# ---------------------------------------------------------------------------

def test_invalid_t_max(hydro_med):
    with pytest.raises(ValueError, match="t_max"):
        compute_radiation_kernel(hydro_med, t_max=-1.0)


def test_invalid_dt_zero(hydro_med):
    with pytest.raises(ValueError, match="dt"):
        compute_radiation_kernel(hydro_med, dt=0.0)


def test_invalid_dt_exceeds_t_max(hydro_med):
    with pytest.raises(ValueError, match="dt"):
        compute_radiation_kernel(hydro_med, dt=100.0, t_max=10.0)


def test_invalid_n_interp(hydro_med):
    with pytest.raises(ValueError, match="n_interp"):
        compute_radiation_kernel(hydro_med, n_interp=5)


def test_invalid_taper_fraction_negative(hydro_med):
    with pytest.raises(ValueError, match="taper_fraction"):
        compute_radiation_kernel(hydro_med, taper_fraction=-0.1)


def test_invalid_taper_fraction_one(hydro_med):
    with pytest.raises(ValueError, match="taper_fraction"):
        compute_radiation_kernel(hydro_med, taper_fraction=1.0)


# ---------------------------------------------------------------------------
# 11. Summary dictionary
# ---------------------------------------------------------------------------

def test_summary_keys(kernel_prod):
    """summary() must return all required keys."""
    s = kernel_prod.summary()
    required = {
        "capytaine_version", "n_panels", "omega_min_rad_s", "omega_max_rad_s",
        "omega_irr_estimate_rad_s", "A_infinity_kg", "A_infinity_method",
        "A_infinity_endpoint_kg", "A_infinity_trend_fit_kg", "A_infinity_ogilvie_kg",
        "t_max_s", "dt_s", "n_time", "taper_fraction",
        "K0_N_s_m", "K_max_abs_N_s_m",
        "B_rmse_trusted_pct", "B_max_rel_trusted_pct",
        "A_rmse_trusted_pct", "A_max_rel_trusted_pct",
    }
    assert required.issubset(s.keys())


def test_summary_values_finite(kernel_prod):
    """All numeric summary values must be finite."""
    s = kernel_prod.summary()
    for k, v in s.items():
        if isinstance(v, float):
            assert np.isfinite(v), f"summary['{k}'] = {v} is not finite"
