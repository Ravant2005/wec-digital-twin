"""
tests/test_excitation.py — Tests for module2_wec/excitation.py (Module 2.3B).

Validations:
  1. Analytical single-component test
  2. Linear amplitude scaling
  3. Phase shift propagation
  4. Linear superposition
  5. Module 1 integration
  6. Frequency-domain amplitude check
  7. Frequency-response check across omega grid
  8. Input validation (out-of-range, bad shapes)
  9. ExcitationSignal metadata
"""

import numpy as np
import pytest

from module2_wec.geometry import REFERENCE_BUOY
from module2_wec.hydrodynamic_coefficients import compute_hydrodynamic_coefficients
from module2_wec.excitation import (
    ExcitationSignal,
    compute_excitation_force,
    interpolate_excitation,
)
from module1_ocean.waves import frequency_grid, component_amplitudes, random_phases


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

PROD_RESOLUTION = (6, 24, 16)   # 672 panels


@pytest.fixture(scope="module")
def hydro():
    return compute_hydrodynamic_coefficients(
        geometry=REFERENCE_BUOY,
        resolution=PROD_RESOLUTION,
        omega_min=0.2, omega_max=1.4, n_omega=20,
        progress_bar=False,
    )


@pytest.fixture(scope="module")
def t_fine():
    """Dense time grid for amplitude measurement."""
    return np.linspace(0.0, 200.0, 40001)


# ---------------------------------------------------------------------------
# 1. Analytical single-component test
# ---------------------------------------------------------------------------

def test_single_component_analytical(hydro, t_fine):
    """
    F(t) must equal a*|F_exc|*cos(omega*t + phi + angle(F_exc)) to machine precision.
    """
    omega_q = np.array([0.7])
    a = np.array([1.5])
    phi = np.array([0.4])

    sig = compute_excitation_force(hydro, t_fine, omega_q, a, phi)

    F_amp = sig.excitation_amplitude[0]
    F_ang = sig.excitation_phase[0]
    F_expected = a[0] * F_amp * np.cos(omega_q[0] * t_fine + phi[0] + F_ang)

    np.testing.assert_allclose(sig.force, F_expected, rtol=1e-12,
                               err_msg="Single-component force does not match analytical formula")


def test_single_component_wave_elevation(hydro, t_fine):
    """Wave elevation must equal a*cos(omega*t + phi)."""
    omega_q = np.array([0.8])
    a = np.array([2.0])
    phi = np.array([1.1])

    sig = compute_excitation_force(hydro, t_fine, omega_q, a, phi)
    eta_expected = a[0] * np.cos(omega_q[0] * t_fine + phi[0])
    np.testing.assert_allclose(sig.wave_elevation, eta_expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# 2. Linear amplitude scaling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a_scale", [0.5, 1.0, 2.0, 3.0])
def test_amplitude_scaling(hydro, t_fine, a_scale):
    """F(a*scale) = scale * F(a=1) to machine precision."""
    omega_q = np.array([0.6])
    phi = np.array([0.0])

    sig1 = compute_excitation_force(hydro, t_fine, omega_q, np.array([1.0]), phi)
    sig_s = compute_excitation_force(hydro, t_fine, omega_q, np.array([a_scale]), phi)

    np.testing.assert_allclose(sig_s.force, a_scale * sig1.force, rtol=1e-12,
                               err_msg=f"Amplitude scaling failed for a={a_scale}")


def test_amplitude_doubling(hydro, t_fine):
    """F(2a) = 2*F(a) exactly."""
    omega_q = np.array([0.9])
    phi = np.array([0.3])
    sig1 = compute_excitation_force(hydro, t_fine, omega_q, np.array([1.0]), phi)
    sig2 = compute_excitation_force(hydro, t_fine, omega_q, np.array([2.0]), phi)
    np.testing.assert_allclose(sig2.force, 2.0 * sig1.force, rtol=1e-12)


# ---------------------------------------------------------------------------
# 3. Phase shift propagation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dphi", [0.0, np.pi/4, np.pi/2, np.pi])
def test_phase_shift(hydro, t_fine, dphi):
    """
    Shifting the input wave phase by dphi must shift the force by exactly dphi.
    Direct check: F(phi+dphi, t) = a*|F|*cos(omega*t + phi+dphi + angle_F).
    Tolerance atol=1e-10 N accounts for floating-point associativity differences
    between the two computation paths (~2e-14 absolute at unit amplitude).
    """
    omega_q = np.array([0.75])
    a = np.array([1.0])
    phi0 = np.array([0.2])

    sig1 = compute_excitation_force(hydro, t_fine, omega_q, a, phi0 + dphi)

    F_ang = sig1.excitation_phase[0]
    F_amp = sig1.excitation_amplitude[0]
    F_expected = a[0] * F_amp * np.cos(omega_q[0] * t_fine + phi0[0] + dphi + F_ang)

    np.testing.assert_allclose(sig1.force, F_expected, atol=1e-10,
                               err_msg=f"Phase shift dphi={dphi:.4f} not propagated correctly")


# ---------------------------------------------------------------------------
# 4. Linear superposition
# ---------------------------------------------------------------------------

def test_superposition_two_components(hydro, t_fine):
    """F(eta1+eta2) = F(eta1) + F(eta2) to machine precision."""
    omega1 = np.array([0.5])
    omega2 = np.array([1.0])
    a1 = np.array([1.2])
    a2 = np.array([0.8])
    phi1 = np.array([0.1])
    phi2 = np.array([1.3])

    sig1 = compute_excitation_force(hydro, t_fine, omega1, a1, phi1)
    sig2 = compute_excitation_force(hydro, t_fine, omega2, a2, phi2)
    sig_both = compute_excitation_force(
        hydro, t_fine,
        np.concatenate([omega1, omega2]),
        np.concatenate([a1, a2]),
        np.concatenate([phi1, phi2]),
    )

    np.testing.assert_allclose(
        sig_both.force, sig1.force + sig2.force, rtol=1e-12,
        err_msg="Superposition: F(1+2) != F(1) + F(2)"
    )
    np.testing.assert_allclose(
        sig_both.wave_elevation, sig1.wave_elevation + sig2.wave_elevation, rtol=1e-12,
        err_msg="Superposition: eta(1+2) != eta(1) + eta(2)"
    )


def test_superposition_five_components(hydro, t_fine):
    """Superposition holds for 5 components."""
    omegas = np.array([0.3, 0.5, 0.7, 0.9, 1.1])
    amps   = np.array([0.5, 1.0, 0.8, 0.6, 0.4])
    phases = np.array([0.1, 0.5, 1.0, 1.5, 2.0])

    sig_all = compute_excitation_force(hydro, t_fine, omegas, amps, phases)
    F_sum = np.zeros(len(t_fine))
    for j in range(5):
        s = compute_excitation_force(
            hydro, t_fine, omegas[j:j+1], amps[j:j+1], phases[j:j+1]
        )
        F_sum += s.force

    np.testing.assert_allclose(sig_all.force, F_sum, rtol=1e-12)


# ---------------------------------------------------------------------------
# 5. Module 1 integration
# ---------------------------------------------------------------------------

def test_module1_integration_deterministic(hydro):
    """
    Using Module 1 wave components with a fixed seed must give deterministic results.
    """
    f_min = 0.25 / (2 * np.pi)   # omega=0.25 rad/s — safely inside [0.2, 1.4]
    f_max = 1.2 / (2 * np.pi)
    f = frequency_grid(f_min, f_max, n_components=8)
    from module1_ocean.spectrum import jonswap
    S = jonswap(f, Hs=1.5, Tp=8.0)
    a = component_amplitudes(f, S)
    omega_j = 2.0 * np.pi * f

    t = np.linspace(0.0, 100.0, 2001)

    phi1 = random_phases(len(f), seed=42)
    phi2 = random_phases(len(f), seed=42)

    sig1 = compute_excitation_force(hydro, t, omega_j, a, phi1)
    sig2 = compute_excitation_force(hydro, t, omega_j, a, phi2)

    np.testing.assert_array_equal(sig1.force, sig2.force,
                                  err_msg="Same seed must give identical force")


def test_module1_integration_seed_changes_force(hydro):
    """Different seeds must give different forces."""
    f_min = 0.25 / (2 * np.pi)
    f_max = 1.2 / (2 * np.pi)
    f = frequency_grid(f_min, f_max, n_components=8)
    from module1_ocean.spectrum import jonswap
    S = jonswap(f, Hs=1.5, Tp=8.0)
    a = component_amplitudes(f, S)
    omega_j = 2.0 * np.pi * f
    t = np.linspace(0.0, 100.0, 2001)

    phi1 = random_phases(len(f), seed=42)
    phi2 = random_phases(len(f), seed=99)

    sig1 = compute_excitation_force(hydro, t, omega_j, a, phi1)
    sig2 = compute_excitation_force(hydro, t, omega_j, a, phi2)

    assert not np.allclose(sig1.force, sig2.force), \
        "Different seeds must produce different forces"


def test_module1_integration_no_nan_inf(hydro):
    """Force and elevation must be finite for a Module 1 sea state."""
    f_min = 0.25 / (2 * np.pi)
    f_max = 1.2 / (2 * np.pi)
    f = frequency_grid(f_min, f_max, n_components=10)
    from module1_ocean.spectrum import jonswap
    S = jonswap(f, Hs=2.0, Tp=9.0)
    a = component_amplitudes(f, S)
    omega_j = 2.0 * np.pi * f
    t = np.linspace(0.0, 200.0, 4001)
    phi = random_phases(len(f), seed=7)

    sig = compute_excitation_force(hydro, t, omega_j, a, phi)

    assert np.all(np.isfinite(sig.force)), "Force contains NaN or inf"
    assert np.all(np.isfinite(sig.wave_elevation)), "Elevation contains NaN or inf"


def test_module1_integration_component_count(hydro):
    """ExcitationSignal must store the same number of components as input."""
    f_min = 0.25 / (2 * np.pi)
    f_max = 1.0 / (2 * np.pi)
    n = 6
    f = frequency_grid(f_min, f_max, n_components=n)
    from module1_ocean.spectrum import jonswap
    S = jonswap(f, Hs=1.0, Tp=7.0)
    a = component_amplitudes(f, S)
    omega_j = 2.0 * np.pi * f
    t = np.linspace(0.0, 50.0, 1001)
    phi = random_phases(n, seed=1)

    sig = compute_excitation_force(hydro, t, omega_j, a, phi)

    assert sig.n_components == n
    np.testing.assert_array_equal(sig.omega, omega_j)


def test_module1_integration_phases_consistent(hydro):
    """
    Wave elevation and force must use the same component phases.
    Verify: the stored wave_phase matches the input phi exactly.
    """
    f_min = 0.3 / (2 * np.pi)
    f_max = 1.1 / (2 * np.pi)
    f = frequency_grid(f_min, f_max, n_components=5)
    from module1_ocean.spectrum import jonswap
    S = jonswap(f, Hs=1.0, Tp=8.0)
    a = component_amplitudes(f, S)
    omega_j = 2.0 * np.pi * f
    t = np.linspace(0.0, 50.0, 501)
    phi = random_phases(5, seed=13)

    sig = compute_excitation_force(hydro, t, omega_j, a, phi)

    np.testing.assert_array_equal(sig.wave_phase, phi)


# ---------------------------------------------------------------------------
# 6. Frequency-domain amplitude check (regular wave)
# ---------------------------------------------------------------------------

def _measure_amplitude(signal: np.ndarray, omega: float, t: np.ndarray) -> float:
    """
    Measure the amplitude of a single-frequency sinusoid via matched filter.

    A_measured = (2/N) * |sum_i signal[i] * exp(-i*omega*t[i])|
    """
    N = len(t)
    phasor = np.sum(signal * np.exp(-1j * omega * t)) * 2.0 / N
    return float(np.abs(phasor))


@pytest.mark.parametrize("omega_q", [0.4, 0.6, 0.8, 1.0, 1.2])
def test_force_amplitude_regular_wave(hydro, omega_q):
    """
    For eta(t) = a*cos(omega*t), the force amplitude must equal a*|F_exc(omega)|.
    Tolerance: 0.5% (limited by DFT spectral leakage over finite time window).
    """
    a = 1.0
    t = np.linspace(0.0, 400.0, 80001)   # long record, fine resolution

    sig = compute_excitation_force(
        hydro, t, np.array([omega_q]), np.array([a]), np.array([0.0])
    )

    F_amp_theoretical = a * sig.excitation_amplitude[0]
    F_amp_measured = _measure_amplitude(sig.force, omega_q, t)

    rel_err = abs(F_amp_measured - F_amp_theoretical) / F_amp_theoretical
    assert rel_err < 0.005, (
        f"omega={omega_q:.2f}: measured amplitude {F_amp_measured:.2f} N, "
        f"theoretical {F_amp_theoretical:.2f} N, rel error {rel_err*100:.3f}%"
    )


# ---------------------------------------------------------------------------
# 7. Frequency-response check
# ---------------------------------------------------------------------------

def test_frequency_response_all_bem_points(hydro):
    """
    At each BEM omega grid point, the measured force amplitude / wave amplitude
    must match |F_exc(omega)| to within 0.5%.
    """
    t = np.linspace(0.0, 400.0, 80001)
    a = 1.0
    errors = []

    for omega_q in hydro.omega:
        sig = compute_excitation_force(
            hydro, t, np.array([omega_q]), np.array([a]), np.array([0.0])
        )
        F_amp_theoretical = a * sig.excitation_amplitude[0]
        F_amp_measured = _measure_amplitude(sig.force, omega_q, t)
        rel_err = abs(F_amp_measured - F_amp_theoretical) / F_amp_theoretical
        errors.append(rel_err)

    max_err = max(errors)
    assert max_err < 0.005, (
        f"Max frequency-response relative error = {max_err*100:.3f}% > 0.5%"
    )


# ---------------------------------------------------------------------------
# 8. Input validation
# ---------------------------------------------------------------------------

def test_omega_above_range_raises(hydro):
    """omega > omega_max must raise ValueError."""
    t = np.linspace(0.0, 10.0, 101)
    with pytest.raises(ValueError, match="outside the BEM frequency range"):
        compute_excitation_force(hydro, t, np.array([2.0]), np.array([1.0]), np.array([0.0]))


def test_omega_below_range_raises(hydro):
    """omega < omega_min must raise ValueError."""
    t = np.linspace(0.0, 10.0, 101)
    with pytest.raises(ValueError, match="outside the BEM frequency range"):
        compute_excitation_force(hydro, t, np.array([0.1]), np.array([1.0]), np.array([0.0]))


def test_negative_amplitude_raises(hydro):
    """Negative wave amplitude must raise ValueError."""
    t = np.linspace(0.0, 10.0, 101)
    with pytest.raises(ValueError, match="non-negative"):
        compute_excitation_force(hydro, t, np.array([0.7]), np.array([-1.0]), np.array([0.0]))


def test_shape_mismatch_raises(hydro):
    """Mismatched omega/amplitude shapes must raise ValueError."""
    t = np.linspace(0.0, 10.0, 101)
    with pytest.raises(ValueError):
        compute_excitation_force(
            hydro, t, np.array([0.5, 0.7]), np.array([1.0]), np.array([0.0, 0.0])
        )


def test_interpolate_excitation_out_of_range(hydro):
    """interpolate_excitation must raise for out-of-range omega."""
    with pytest.raises(ValueError, match="outside the BEM frequency range"):
        interpolate_excitation(hydro, np.array([0.05, 0.7]))


# ---------------------------------------------------------------------------
# 9. ExcitationSignal metadata
# ---------------------------------------------------------------------------

def test_signal_metadata(hydro):
    """ExcitationSignal must store correct metadata."""
    t = np.linspace(0.0, 10.0, 201)
    omega_q = np.array([0.6, 0.9])
    a = np.array([1.0, 0.5])
    phi = np.array([0.0, 1.0])

    sig = compute_excitation_force(hydro, t, omega_q, a, phi)

    assert isinstance(sig, ExcitationSignal)
    assert sig.n_components == 2
    assert sig.n_time == 201
    assert sig.source_omega_min == pytest.approx(hydro.omega_min)
    assert sig.source_omega_max == pytest.approx(hydro.omega_max)
    assert sig.wave_direction == pytest.approx(0.0)
    assert sig.n_panels == hydro.n_panels
    assert sig.capytaine_version == hydro.capytaine_version
    assert "linear" in sig.interpolation_method.lower()
    np.testing.assert_array_equal(sig.omega, omega_q)
    np.testing.assert_array_equal(sig.wave_amplitude, a)
    np.testing.assert_array_equal(sig.wave_phase, phi)
    assert np.all(np.isfinite(sig.excitation_amplitude))
    assert np.all(np.isfinite(sig.excitation_phase))


def test_signal_shapes(hydro):
    """All array fields must have consistent shapes."""
    t = np.linspace(0.0, 20.0, 401)
    omega_q = np.array([0.5, 0.7, 1.0])
    a = np.ones(3)
    phi = np.zeros(3)

    sig = compute_excitation_force(hydro, t, omega_q, a, phi)

    assert sig.time.shape == (401,)
    assert sig.force.shape == (401,)
    assert sig.wave_elevation.shape == (401,)
    assert sig.omega.shape == (3,)
    assert sig.wave_amplitude.shape == (3,)
    assert sig.wave_phase.shape == (3,)
    assert sig.excitation_amplitude.shape == (3,)
    assert sig.excitation_phase.shape == (3,)


def test_excitation_amplitude_positive(hydro):
    """Interpolated |F_exc| must be positive at all BEM grid points."""
    F_complex, F_amp, F_phase = interpolate_excitation(hydro, hydro.omega)
    assert np.all(F_amp > 0.0)


def test_interpolation_at_grid_points(hydro):
    """
    Interpolating at the exact BEM grid points must recover the stored values
    to machine precision.
    """
    F_complex, F_amp, F_phase = interpolate_excitation(hydro, hydro.omega)
    np.testing.assert_allclose(
        F_amp, hydro.excitation_force_heave_amplitude, rtol=1e-12,
        err_msg="Interpolation at grid points does not recover stored amplitude"
    )
    np.testing.assert_allclose(
        np.real(F_complex), hydro.excitation_force_heave_real, rtol=1e-12,
    )
    np.testing.assert_allclose(
        np.imag(F_complex), hydro.excitation_force_heave_imag, rtol=1e-12,
    )
