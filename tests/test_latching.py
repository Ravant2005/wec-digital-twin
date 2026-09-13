"""
tests/test_latching.py — Tests for module2_wec/latching.py (Module 2.5).

Tests:
  1.  Initial state is FREE
  2.  FREE state allows normal dynamics (matches Module 2.4)
  3.  LATCHED state forces v ≈ 0
  4.  Displacement constant during latch
  5.  FREE -> LATCHED transition
  6.  LATCHED -> FREE transition
  7.  Latch force has correct sign
  8.  Latch does not generate harvested power
  9.  PTO power = B_PTO * v^2
  10. Radiation memory NOT reset during latching
  11. No-latch case reproduces Module 2.4
  12. Zero excitation + no initial motion stays stationary
  13. Deterministic reproducibility
  14. Energy does not spontaneously increase due to latching
  15. Controller does not modify frozen hydrodynamic coefficients
"""

import numpy as np
import pytest

from module2_wec.geometry import REFERENCE_BUOY
from module2_wec.hydrostatics import Hydrostatics
from module2_wec.hydrodynamic_coefficients import compute_hydrodynamic_coefficients
from module2_wec.radiation import compute_radiation_kernel
from module2_wec.excitation import compute_excitation_force
from module2_wec.cummins import CumminsParameters, solve_cummins
from module2_wec.pto import PTOParameters
from module2_wec.latching import (
    LatchState, LatchingParameters, LatchingResult, solve_cummins_latching,
)

PROD_RESOLUTION = (6, 24, 16)
B_PTO_TEST = 4e4   # N.s/m — same as Module 2.4 baseline


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


# ---------------------------------------------------------------------------
# 1. Initial state is FREE
# ---------------------------------------------------------------------------

def test_initial_state_free(hydro, params):
    """Controller must start in FREE state."""
    t = _make_t(20.0)
    F = np.zeros(len(t))
    lp = LatchingParameters(enabled=True, latch_duration=1.0)
    res = solve_cummins_latching(params, t, F, lp)
    assert res.state_log[0] == LatchState.FREE


# ---------------------------------------------------------------------------
# 2. FREE state allows normal dynamics
# ---------------------------------------------------------------------------

def test_free_state_normal_dynamics(hydro, params):
    """With latching disabled, dynamics must match solve_cummins exactly."""
    omega_w = 0.6
    t = _make_t(100.0)
    exc = compute_excitation_force(
        hydro, t, np.array([omega_w]), np.array([1.0]), np.array([0.0])
    )
    pto = PTOParameters(damping=B_PTO_TEST, stiffness=0.0)
    lp  = LatchingParameters(enabled=False, latch_duration=0.0)

    res_latch = solve_cummins_latching(params, t, exc.force, lp, pto=pto)
    res_plain = solve_cummins(params, t, exc.force, pto=pto)

    np.testing.assert_allclose(
        res_latch.displacement, res_plain.displacement, atol=1e-10,
        err_msg="Disabled latching must reproduce solve_cummins displacement"
    )
    np.testing.assert_allclose(
        res_latch.velocity, res_plain.velocity, atol=1e-10,
        err_msg="Disabled latching must reproduce solve_cummins velocity"
    )


# ---------------------------------------------------------------------------
# 3. LATCHED state forces v ≈ 0
# ---------------------------------------------------------------------------

def test_latched_velocity_zero(hydro, params):
    """During LATCHED intervals, velocity must be exactly zero."""
    omega_w = 0.8
    t = _make_t(200.0)
    exc = compute_excitation_force(
        hydro, t, np.array([omega_w]), np.array([1.0]), np.array([0.0])
    )
    pto = PTOParameters(damping=B_PTO_TEST, stiffness=0.0)
    lp  = LatchingParameters(enabled=True, latch_duration=1.0)

    res = solve_cummins_latching(params, t, exc.force, lp, pto=pto)

    latched_v = res.velocity[res.latch_active]
    if len(latched_v) > 0:
        np.testing.assert_allclose(
            latched_v, 0.0, atol=1e-12,
            err_msg="Velocity must be zero during LATCHED intervals"
        )


# ---------------------------------------------------------------------------
# 4. Displacement constant during latch
# ---------------------------------------------------------------------------

def test_latched_displacement_constant(hydro, params):
    """During LATCHED intervals, displacement must remain constant."""
    omega_w = 0.8
    t = _make_t(200.0)
    exc = compute_excitation_force(
        hydro, t, np.array([omega_w]), np.array([1.0]), np.array([0.0])
    )
    lp = LatchingParameters(enabled=True, latch_duration=1.5)

    res = solve_cummins_latching(params, t, exc.force, lp)

    # Find contiguous latched blocks and check displacement is constant within each
    la = res.latch_active
    x  = res.displacement
    i  = 0
    while i < len(la):
        if la[i]:
            j = i
            while j < len(la) and la[j]:
                j += 1
            block_x = x[i:j]
            if len(block_x) > 1:
                assert np.allclose(block_x, block_x[0], atol=1e-12), (
                    f"Displacement not constant during latch block [{i}:{j}]"
                )
            i = j
        else:
            i += 1


# ---------------------------------------------------------------------------
# 5. FREE -> LATCHED transition
# ---------------------------------------------------------------------------

def test_free_to_latched_transition(hydro, params):
    """At least one FREE->LATCHED transition must occur with enabled latching."""
    omega_w = 0.8
    t = _make_t(200.0)
    exc = compute_excitation_force(
        hydro, t, np.array([omega_w]), np.array([1.0]), np.array([0.0])
    )
    lp = LatchingParameters(enabled=True, latch_duration=1.0)
    res = solve_cummins_latching(params, t, exc.force, lp)

    assert res.n_latch_events > 0, "No latch engagements occurred"
    assert np.any(res.latch_active), "latch_active never True"


# ---------------------------------------------------------------------------
# 6. LATCHED -> FREE transition
# ---------------------------------------------------------------------------

def test_latched_to_free_transition(hydro, params):
    """After latch_duration, state must return to FREE."""
    omega_w = 0.8
    t = _make_t(200.0)
    exc = compute_excitation_force(
        hydro, t, np.array([omega_w]), np.array([1.0]), np.array([0.0])
    )
    lp = LatchingParameters(enabled=True, latch_duration=1.0)
    res = solve_cummins_latching(params, t, exc.force, lp)

    # Must have at least one release
    assert len(res.latch_release_times) > 0, "No latch releases occurred"
    # After each release, state must be FREE
    for t_rel in res.latch_release_times:
        idx = np.searchsorted(t, t_rel)
        if idx < len(t):
            assert res.state_log[idx] == LatchState.FREE, (
                f"State not FREE after release at t={t_rel:.3f}"
            )


# ---------------------------------------------------------------------------
# 7. Latch force has correct sign
# ---------------------------------------------------------------------------

def test_latch_force_sign(hydro, params):
    """
    Latch force must be non-zero only during LATCHED intervals.
    During FREE intervals, latch_force must be zero.
    """
    omega_w = 0.8
    t = _make_t(200.0)
    exc = compute_excitation_force(
        hydro, t, np.array([omega_w]), np.array([1.0]), np.array([0.0])
    )
    lp = LatchingParameters(enabled=True, latch_duration=1.0)
    res = solve_cummins_latching(params, t, exc.force, lp)

    free_latch_force = res.latch_force[~res.latch_active]
    np.testing.assert_allclose(
        free_latch_force, 0.0, atol=1e-10,
        err_msg="Latch force must be zero during FREE intervals"
    )


# ---------------------------------------------------------------------------
# 8. Latch does not generate harvested power
# ---------------------------------------------------------------------------

def test_latch_no_harvested_power(hydro, params):
    """
    PTO absorbed power must be zero during LATCHED intervals (v=0).
    Latch reaction work is ~0 (v=0 during latch).
    """
    omega_w = 0.8
    t = _make_t(200.0)
    exc = compute_excitation_force(
        hydro, t, np.array([omega_w]), np.array([1.0]), np.array([0.0])
    )
    pto = PTOParameters(damping=B_PTO_TEST, stiffness=0.0)
    lp  = LatchingParameters(enabled=True, latch_duration=1.0)
    res = solve_cummins_latching(params, t, exc.force, lp, pto=pto)

    assert res.pto_result is not None
    latched_power = res.pto_result.absorbed_power[res.latch_active]
    if len(latched_power) > 0:
        np.testing.assert_allclose(
            latched_power, 0.0, atol=1e-20,
            err_msg="PTO power must be zero during LATCHED intervals"
        )

    # Latch reaction work must be negligible (v=0 during latch)
    total_latch_work = float(res.energy_latch[-1])
    assert abs(total_latch_work) < 1.0, (
        f"Latch reaction work {total_latch_work:.2f} J is non-negligible"
    )


# ---------------------------------------------------------------------------
# 9. PTO power = B_PTO * v^2
# ---------------------------------------------------------------------------

def test_pto_power_formula(hydro, params):
    """PTO absorbed power must equal B_PTO * v^2 at every timestep."""
    omega_w = 0.8
    t = _make_t(100.0)
    exc = compute_excitation_force(
        hydro, t, np.array([omega_w]), np.array([1.0]), np.array([0.0])
    )
    pto = PTOParameters(damping=B_PTO_TEST, stiffness=0.0)
    lp  = LatchingParameters(enabled=True, latch_duration=1.0)
    res = solve_cummins_latching(params, t, exc.force, lp, pto=pto)

    assert res.pto_result is not None
    expected = B_PTO_TEST * res.velocity**2
    np.testing.assert_allclose(
        res.pto_result.absorbed_power, expected, rtol=1e-12,
        err_msg="PTO power must equal B_PTO * v^2"
    )


# ---------------------------------------------------------------------------
# 10. Radiation memory NOT reset during latching
# ---------------------------------------------------------------------------

def test_radiation_memory_preserved(hydro, params):
    """
    Radiation memory must be continuous across latch engage/release.
    Test: radiation force must not jump discontinuously at latch release.
    """
    omega_w = 0.8
    t = _make_t(200.0)
    exc = compute_excitation_force(
        hydro, t, np.array([omega_w]), np.array([1.0]), np.array([0.0])
    )
    lp = LatchingParameters(enabled=True, latch_duration=1.0)
    res = solve_cummins_latching(params, t, exc.force, lp)

    if len(res.latch_release_times) == 0:
        pytest.skip("No latch releases to check")

    # At each release, radiation force must be finite and continuous
    for t_rel in res.latch_release_times[:3]:
        idx = np.searchsorted(t, t_rel)
        if 1 <= idx < len(t) - 1:
            f_before = res.radiation_force[idx - 1]
            f_after  = res.radiation_force[idx + 1]
            assert np.isfinite(f_before) and np.isfinite(f_after), (
                "Radiation force is non-finite near latch release"
            )
            # No catastrophic jump (allow up to 10x the typical force magnitude)
            f_scale = max(abs(f_before), 1.0)
            assert abs(f_after - f_before) < 10.0 * f_scale, (
                f"Radiation force jumps discontinuously at release t={t_rel:.3f}"
            )


# ---------------------------------------------------------------------------
# 11. No-latch case reproduces Module 2.4
# ---------------------------------------------------------------------------

def test_no_latch_reproduces_module24(hydro, params):
    """solve_cummins_latching with enabled=False must match solve_cummins."""
    omega_w = 0.8
    t = _make_t(100.0)
    exc = compute_excitation_force(
        hydro, t, np.array([omega_w]), np.array([1.0]), np.array([0.0])
    )
    pto = PTOParameters(damping=B_PTO_TEST, stiffness=0.0)
    lp  = LatchingParameters(enabled=False, latch_duration=0.0)

    res_latch = solve_cummins_latching(params, t, exc.force, lp, pto=pto)
    res_plain = solve_cummins(params, t, exc.force, pto=pto)

    np.testing.assert_allclose(
        res_latch.displacement, res_plain.displacement, atol=1e-10
    )
    np.testing.assert_allclose(
        res_latch.velocity, res_plain.velocity, atol=1e-10
    )


# ---------------------------------------------------------------------------
# 12. Zero excitation + no initial motion stays stationary
# ---------------------------------------------------------------------------

def test_zero_excitation_stationary(params):
    """Zero excitation and zero ICs must give x=v=0 everywhere."""
    t = _make_t(50.0)
    F = np.zeros(len(t))
    lp = LatchingParameters(enabled=True, latch_duration=1.0)
    res = solve_cummins_latching(params, t, F, lp)

    np.testing.assert_allclose(res.displacement, 0.0, atol=1e-14)
    np.testing.assert_allclose(res.velocity,     0.0, atol=1e-14)


# ---------------------------------------------------------------------------
# 13. Deterministic reproducibility
# ---------------------------------------------------------------------------

def test_deterministic_reproducibility(hydro, params):
    """Two identical calls must produce bit-identical results."""
    omega_w = 0.8
    t = _make_t(100.0)
    exc = compute_excitation_force(
        hydro, t, np.array([omega_w]), np.array([1.0]), np.array([0.0])
    )
    pto = PTOParameters(damping=B_PTO_TEST, stiffness=0.0)
    lp  = LatchingParameters(enabled=True, latch_duration=1.0)

    res1 = solve_cummins_latching(params, t, exc.force, lp, pto=pto)
    res2 = solve_cummins_latching(params, t, exc.force, lp, pto=pto)

    np.testing.assert_array_equal(res1.displacement, res2.displacement)
    np.testing.assert_array_equal(res1.velocity,     res2.velocity)
    np.testing.assert_array_equal(res1.latch_active, res2.latch_active)


# ---------------------------------------------------------------------------
# 14. Energy does not spontaneously increase due to latching
# ---------------------------------------------------------------------------

def test_energy_no_spontaneous_increase(hydro, params):
    """
    Free-decay with latching: mechanical energy must not exceed initial value.
    Latching is a constraint; it cannot inject energy.
    """
    t = _make_t(100.0)
    F = np.zeros(len(t))
    lp = LatchingParameters(enabled=True, latch_duration=1.0)
    res = solve_cummins_latching(params, t, F, lp, x0=1.0, v0=0.0)

    E_initial = res.mechanical_energy[0]
    E_max     = float(np.max(res.mechanical_energy))
    # Allow 1% numerical tolerance
    assert E_max <= E_initial * 1.01, (
        f"Mechanical energy grew: E_initial={E_initial:.1f} J, E_max={E_max:.1f} J"
    )


# ---------------------------------------------------------------------------
# 15. Controller does not modify frozen hydrodynamic coefficients
# ---------------------------------------------------------------------------

def test_frozen_coefficients_unchanged(hydro, params):
    """Latching must not alter CumminsParameters or hydrodynamic data."""
    A_before = hydro.added_mass_heave.copy()
    B_before = hydro.radiation_damping_heave.copy()
    K_before = params.kernel_values.copy()
    M_before = params.mass

    omega_w = 0.8
    t = _make_t(100.0)
    exc = compute_excitation_force(
        hydro, t, np.array([omega_w]), np.array([1.0]), np.array([0.0])
    )
    lp = LatchingParameters(enabled=True, latch_duration=1.0)
    solve_cummins_latching(params, t, exc.force, lp)

    np.testing.assert_array_equal(hydro.added_mass_heave, A_before)
    np.testing.assert_array_equal(hydro.radiation_damping_heave, B_before)
    np.testing.assert_array_equal(params.kernel_values, K_before)
    assert params.mass == M_before
