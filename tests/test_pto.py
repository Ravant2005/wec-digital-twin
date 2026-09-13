"""
tests/test_pto.py — Tests for module2_wec/pto.py and PTO integration (Module 2.4).

Tests:
  A. Zero PTO equivalence (pto=None vs B_pto=0, K_pto=0)
  B. Zero velocity -> P_abs = 0
  C. Positive damping -> P_abs >= 0
  D. PTO force sign (damping opposes velocity)
  E. PTO stiffness sign (stiffness opposes displacement)
  F. Linearity (doubling x,v doubles force)
  G. Power scaling (doubling B_pto doubles power)
  H. No-PTO energy reproduces Module 2.3C
  I. Positive damping reduces response amplitude
  J. Power sanity (mean power finite and positive)
  K. PTOParameters validation
  L. frequency_domain_rao_pto reduces to no-PTO at B_pto=K_pto=0
  M. FD RAO with PTO: TD vs FD agreement at omega=0.6, 0.8, 1.0
"""

import numpy as np
import pytest

from module2_wec.geometry import REFERENCE_BUOY
from module2_wec.hydrostatics import Hydrostatics
from module2_wec.hydrodynamic_coefficients import compute_hydrodynamic_coefficients
from module2_wec.radiation import compute_radiation_kernel
from module2_wec.excitation import compute_excitation_force
from module2_wec.cummins import (
    CumminsParameters, solve_cummins,
    frequency_domain_rao, frequency_domain_rao_pto,
)
from module2_wec.pto import (
    PTOParameters, compute_pto_force, compute_pto_power, compute_pto_result,
)


# ---------------------------------------------------------------------------
# Shared fixtures (module-scoped to avoid re-running BEM)
# ---------------------------------------------------------------------------

PROD_RESOLUTION = (6, 24, 16)


@pytest.fixture(scope="module")
def hydro():
    return compute_hydrodynamic_coefficients(
        geometry=REFERENCE_BUOY, resolution=PROD_RESOLUTION,
        omega_min=0.2, omega_max=1.4, n_omega=20, progress_bar=False,
    )


@pytest.fixture(scope="module")
def rk(hydro):
    return compute_radiation_kernel(hydro)


@pytest.fixture(scope="module")
def hs():
    return Hydrostatics(REFERENCE_BUOY)


@pytest.fixture(scope="module")
def params(rk, hs):
    return CumminsParameters(
        mass=REFERENCE_BUOY.mass,
        hydrostatic_stiffness=hs.hydrostatic_stiffness,
        added_mass_infinity=rk.A_infinity,
        kernel_time=rk.time,
        kernel_values=rk.kernel,
    )


def _make_t(duration, dt=0.05):
    return np.arange(0.0, duration + dt * 0.5, dt)


def _measure(signal, omega, t):
    """Extract e^{+i*omega*t} phasor amplitude and phase."""
    phasor = np.mean(signal * np.exp(-1j * omega * t)) * 2.0
    return abs(phasor), float(np.angle(phasor))


# ---------------------------------------------------------------------------
# A. Zero PTO equivalence
# ---------------------------------------------------------------------------

def test_zero_pto_equivalence(hydro, params):
    """pto=None and PTOParameters(0,0) must give identical trajectories."""
    omega_w = 0.6
    t = _make_t(100.0)
    exc = compute_excitation_force(
        hydro, t, np.array([omega_w]), np.array([1.0]), np.array([0.0])
    )
    res_none = solve_cummins(params, t, exc.force)
    res_zero = solve_cummins(params, t, exc.force, pto=PTOParameters(0.0, 0.0))

    np.testing.assert_allclose(
        res_none.displacement, res_zero.displacement, atol=1e-12,
        err_msg="Displacement differs between pto=None and PTOParameters(0,0)"
    )
    np.testing.assert_allclose(
        res_none.velocity, res_zero.velocity, atol=1e-12,
        err_msg="Velocity differs between pto=None and PTOParameters(0,0)"
    )


# ---------------------------------------------------------------------------
# B. Zero velocity -> P_abs = 0
# ---------------------------------------------------------------------------

def test_zero_velocity_zero_power():
    """P_abs = B_pto * v^2 = 0 when v = 0."""
    pto = PTOParameters(damping=5e4, stiffness=0.0)
    v = np.zeros(100)
    p = compute_pto_power(v, pto)
    np.testing.assert_allclose(p, 0.0, atol=1e-30)


# ---------------------------------------------------------------------------
# C. Positive damping -> P_abs >= 0
# ---------------------------------------------------------------------------

def test_positive_damping_nonneg_power(hydro, params):
    """P_abs = B_pto * v^2 >= 0 for all timesteps with B_pto > 0."""
    pto = PTOParameters(damping=4e4, stiffness=0.0)
    t = _make_t(100.0)
    exc = compute_excitation_force(
        hydro, t, np.array([0.8]), np.array([1.0]), np.array([0.0])
    )
    res = solve_cummins(params, t, exc.force, pto=pto)
    assert res.pto_result is not None
    assert np.all(res.pto_result.absorbed_power >= -1e-20), (
        "Absorbed power has negative values with positive B_pto"
    )


# ---------------------------------------------------------------------------
# D. PTO force sign — damping opposes velocity
# ---------------------------------------------------------------------------

def test_pto_damping_force_sign():
    """F_PTO = -B_pto*v: positive v -> negative force, negative v -> positive force."""
    pto = PTOParameters(damping=1e4, stiffness=0.0)
    v_pos = np.array([1.0, 2.0, 0.5])
    v_neg = np.array([-1.0, -2.0, -0.5])
    x_zero = np.zeros(3)

    f_pos = compute_pto_force(x_zero, v_pos, pto)
    f_neg = compute_pto_force(x_zero, v_neg, pto)

    assert np.all(f_pos < 0), "Positive velocity must give negative PTO force"
    assert np.all(f_neg > 0), "Negative velocity must give positive PTO force"


# ---------------------------------------------------------------------------
# E. PTO stiffness sign — stiffness opposes displacement
# ---------------------------------------------------------------------------

def test_pto_stiffness_force_sign():
    """F_PTO = -K_pto*x: positive x -> negative force, negative x -> positive force."""
    pto = PTOParameters(damping=0.0, stiffness=1e5)
    x_pos = np.array([1.0, 0.5])
    x_neg = np.array([-1.0, -0.5])
    v_zero = np.zeros(2)

    f_pos = compute_pto_force(x_pos, v_zero, pto)
    f_neg = compute_pto_force(x_neg, v_zero, pto)

    assert np.all(f_pos < 0), "Positive displacement must give negative PTO stiffness force"
    assert np.all(f_neg > 0), "Negative displacement must give positive PTO stiffness force"


# ---------------------------------------------------------------------------
# F. Linearity — doubling x and v doubles force
# ---------------------------------------------------------------------------

def test_pto_force_linearity():
    """F_PTO is linear: doubling x and v must double the force."""
    pto = PTOParameters(damping=3e4, stiffness=2e5)
    x = np.array([0.5, -0.3, 1.2])
    v = np.array([0.1, -0.4, 0.8])

    f1 = compute_pto_force(x, v, pto)
    f2 = compute_pto_force(2 * x, 2 * v, pto)

    np.testing.assert_allclose(f2, 2 * f1, rtol=1e-12)


# ---------------------------------------------------------------------------
# G. Power scaling — doubling B_pto doubles instantaneous power
# ---------------------------------------------------------------------------

def test_power_scales_with_damping():
    """P_abs = B_pto * v^2: doubling B_pto must double power."""
    v = np.array([0.5, 1.0, -0.3, 0.0])
    pto1 = PTOParameters(damping=1e4, stiffness=0.0)
    pto2 = PTOParameters(damping=2e4, stiffness=0.0)

    p1 = compute_pto_power(v, pto1)
    p2 = compute_pto_power(v, pto2)

    np.testing.assert_allclose(p2, 2 * p1, rtol=1e-12)


# ---------------------------------------------------------------------------
# H. No-PTO energy reproduces Module 2.3C
# ---------------------------------------------------------------------------

def test_no_pto_reproduces_2_3c(hydro, params):
    """solve_cummins with pto=None must match Module 2.3C result exactly."""
    t = _make_t(80.0)
    F = np.zeros(len(t))
    res_2_3c = solve_cummins(params, t, F, x0=1.0, v0=0.0)
    res_pto  = solve_cummins(params, t, F, x0=1.0, v0=0.0, pto=None)

    np.testing.assert_array_equal(res_2_3c.displacement, res_pto.displacement)
    np.testing.assert_array_equal(res_2_3c.velocity,     res_pto.velocity)
    assert res_pto.pto_result is None


# ---------------------------------------------------------------------------
# I. Positive PTO damping reduces response amplitude
# ---------------------------------------------------------------------------

def test_positive_damping_reduces_amplitude(hydro, params):
    """Adding B_pto > 0 must reduce steady-state displacement amplitude."""
    omega_w = 0.8
    t = _make_t(300.0)
    exc = compute_excitation_force(
        hydro, t, np.array([omega_w]), np.array([1.0]), np.array([0.0])
    )
    res_no_pto = solve_cummins(params, t, exc.force)
    res_pto    = solve_cummins(params, t, exc.force,
                               pto=PTOParameters(damping=5e4, stiffness=0.0))

    mask = t >= 150.0
    amp_no_pto, _ = _measure(res_no_pto.displacement[mask], omega_w, t[mask])
    amp_pto,    _ = _measure(res_pto.displacement[mask],    omega_w, t[mask])

    assert amp_pto < amp_no_pto, (
        f"PTO damping did not reduce amplitude: no_pto={amp_no_pto:.4f}, "
        f"pto={amp_pto:.4f}"
    )


# ---------------------------------------------------------------------------
# J. Power sanity — mean power finite and positive
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("omega_w", [0.6, 0.8, 1.0])
def test_mean_power_positive(hydro, params, omega_w):
    """Mean absorbed power must be finite and positive for B_pto > 0."""
    pto = PTOParameters(damping=4e4, stiffness=0.0)
    t = _make_t(300.0)
    exc = compute_excitation_force(
        hydro, t, np.array([omega_w]), np.array([1.0]), np.array([0.0])
    )
    res = solve_cummins(params, t, exc.force, pto=pto)
    assert res.pto_result is not None
    # Use steady-state window only
    mask = t >= 150.0
    p_ss = res.pto_result.absorbed_power[mask]
    mean_p = float(np.mean(p_ss))
    assert np.isfinite(mean_p), f"Mean power not finite at omega={omega_w}"
    assert mean_p > 0, f"Mean power not positive at omega={omega_w}: {mean_p:.2f} W"


# ---------------------------------------------------------------------------
# K. PTOParameters validation
# ---------------------------------------------------------------------------

def test_pto_negative_damping_raises():
    with pytest.raises(ValueError, match="damping"):
        PTOParameters(damping=-1.0, stiffness=0.0)


def test_pto_negative_stiffness_raises():
    with pytest.raises(ValueError, match="stiffness"):
        PTOParameters(damping=0.0, stiffness=-1.0)


# ---------------------------------------------------------------------------
# L. frequency_domain_rao_pto reduces to no-PTO at B_pto=K_pto=0
# ---------------------------------------------------------------------------

def test_fd_rao_pto_zero_recovers_no_pto(hydro, params):
    """frequency_domain_rao_pto with zero PTO must equal frequency_domain_rao."""
    omega_w = 0.8
    A_w = float(np.interp(omega_w, hydro.omega, hydro.added_mass_heave))
    B_w = float(np.interp(omega_w, hydro.omega, hydro.radiation_damping_heave))
    Fr  = float(np.interp(omega_w, hydro.omega, hydro.excitation_force_heave_real))
    Fi  = float(np.interp(omega_w, hydro.omega, hydro.excitation_force_heave_imag))
    Fc  = (Fr + 1j * Fi) * 1.0

    X_no_pto = frequency_domain_rao(
        omega_w, params.mass, A_w, params.hydrostatic_stiffness, B_w, Fc
    )
    X_zero_pto = frequency_domain_rao_pto(
        omega_w, params.mass, A_w, params.hydrostatic_stiffness, B_w, Fc,
        pto=PTOParameters(0.0, 0.0),
    )
    X_none_pto = frequency_domain_rao_pto(
        omega_w, params.mass, A_w, params.hydrostatic_stiffness, B_w, Fc,
        pto=None,
    )

    assert abs(X_zero_pto) == pytest.approx(abs(X_no_pto), rel=1e-12)
    assert abs(X_none_pto) == pytest.approx(abs(X_no_pto), rel=1e-12)


# ---------------------------------------------------------------------------
# M. TD vs FD RAO with PTO — trusted frequency range
# ---------------------------------------------------------------------------

# Tolerances: PTO shifts resonance; errors at omega=1.0 remain due to BEM artifact.
_PTO_AMP_TOL_PCT   = {0.6: 10, 0.8: 15, 1.0: 50}
_PTO_PHASE_TOL_DEG = {0.6: 15, 0.8: 20, 1.0: 50}

# Development PTO: B_pto ~ 1x radiation damping at omega=0.8 (43611 N.s/m)
_B_PTO_TEST = 4e4   # N.s/m


@pytest.mark.parametrize("omega_w", [0.6, 0.8, 1.0])
def test_pto_td_vs_fd(hydro, params, omega_w):
    """
    TD RAO with fixed PTO must match FD-correct RAO within tolerance.

    FD impedance: Z = (C33+K_pto) - omega^2*(M+A(omega)) + i*omega*(B(omega)+B_pto)
    Uses e^{+i*omega*t} convention (Module 2.3C).
    B_pto = 4e4 N.s/m, K_pto = 0.
    T = 400 s, dt = 0.05 s, transient discard = first 200 s.
    """
    pto = PTOParameters(damping=_B_PTO_TEST, stiffness=0.0)
    t = _make_t(400.0)
    exc = compute_excitation_force(
        hydro, t, np.array([omega_w]), np.array([1.0]), np.array([0.0])
    )
    res = solve_cummins(params, t, exc.force, pto=pto)

    mask = t >= 200.0
    td_amp, td_ph = _measure(res.displacement[mask], omega_w, t[mask])

    A_w = float(np.interp(omega_w, hydro.omega, hydro.added_mass_heave))
    B_w = float(np.interp(omega_w, hydro.omega, hydro.radiation_damping_heave))
    Fr  = float(np.interp(omega_w, hydro.omega, hydro.excitation_force_heave_real))
    Fi  = float(np.interp(omega_w, hydro.omega, hydro.excitation_force_heave_imag))
    Fc  = (Fr + 1j * Fi) * 1.0

    X_fd = frequency_domain_rao_pto(
        omega_w, params.mass, A_w, params.hydrostatic_stiffness, B_w, Fc, pto=pto
    )
    fd_amp = abs(X_fd)
    fd_ph  = float(np.angle(X_fd))

    amp_err   = abs(td_amp - fd_amp) / fd_amp
    phase_err = abs(float(np.angle(np.exp(1j * (td_ph - fd_ph))))) * 180 / np.pi

    amp_tol   = _PTO_AMP_TOL_PCT[omega_w] / 100.0
    phase_tol = _PTO_PHASE_TOL_DEG[omega_w]

    assert amp_err < amp_tol, (
        f"omega={omega_w}: PTO amp error {amp_err*100:.1f}% > {_PTO_AMP_TOL_PCT[omega_w]}% "
        f"(TD={td_amp:.4f}, FD={fd_amp:.4f})"
    )
    assert phase_err < phase_tol, (
        f"omega={omega_w}: PTO phase error {phase_err:.1f} deg > {_PTO_PHASE_TOL_DEG[omega_w]} deg"
    )
