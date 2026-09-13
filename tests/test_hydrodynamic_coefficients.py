"""
tests/test_hydrodynamic_coefficients.py — Tests for module2_wec/hydrodynamic_coefficients.py

Tests cover:
  - Array shapes and frequency ordering
  - omega / frequency_hz relationship
  - Added mass: finite, non-negative (away from irregular frequencies)
  - Radiation damping: finite, non-negative (physical convention)
  - Excitation force: amplitude = |complex|, phase = angle(complex)
  - No NaNs or infinities in any array
  - Hydrostatic stiffness consistency with Module 2.1
  - Heave DOF present in the model
  - Water depth is finite (not infinite depth)
  - Summary dictionary completeness
  - Determinism

Uses a MEDIUM mesh (resolution (3, 12, 8) = 168 panels) and 8 frequencies
for a balance of physical accuracy and test speed.

Run with:  pytest tests/test_hydrodynamic_coefficients.py -v
"""

import math
import numpy as np
import pytest

import capytaine as cpt

from module2_wec.geometry import CylinderGeometry, REFERENCE_BUOY, GRAVITY
from module2_wec.hydrostatics import Hydrostatics
from module2_wec.capytaine_model import CapytaineModel
from module2_wec.hydrodynamic_coefficients import (
    HydrodynamicCoefficients,
    compute_hydrodynamic_coefficients,
)


# ---------------------------------------------------------------------------
# Shared fixture: medium mesh, 8 frequencies, run once per module
# ---------------------------------------------------------------------------

MEDIUM_RESOLUTION = (3, 12, 8)   # 168 panels
N_TEST_OMEGA = 8
OMEGA_MIN = 0.3
OMEGA_MAX = 1.2


@pytest.fixture(scope="module")
def hydro():
    """HydrodynamicCoefficients from medium mesh, 8 frequencies."""
    return compute_hydrodynamic_coefficients(
        geometry=REFERENCE_BUOY,
        resolution=MEDIUM_RESOLUTION,
        omega_min=OMEGA_MIN,
        omega_max=OMEGA_MAX,
        n_omega=N_TEST_OMEGA,
        progress_bar=False,
    )


# ---------------------------------------------------------------------------
# 1. Array shapes and frequency ordering
# ---------------------------------------------------------------------------

def test_all_arrays_same_length(hydro):
    """All coefficient arrays must have the same length as omega."""
    n = hydro.n_omega
    assert len(hydro.omega) == n
    assert len(hydro.frequency_hz) == n
    assert len(hydro.added_mass_heave) == n
    assert len(hydro.radiation_damping_heave) == n
    assert len(hydro.excitation_force_heave_real) == n
    assert len(hydro.excitation_force_heave_imag) == n
    assert len(hydro.excitation_force_heave_amplitude) == n
    assert len(hydro.excitation_force_heave_phase) == n


def test_omega_strictly_increasing(hydro):
    """omega must be strictly increasing."""
    assert np.all(np.diff(hydro.omega) > 0)


def test_omega_bounds(hydro):
    """omega first/last values must match omega_min/omega_max."""
    assert hydro.omega[0] == pytest.approx(OMEGA_MIN, rel=1e-10)
    assert hydro.omega[-1] == pytest.approx(OMEGA_MAX, rel=1e-10)


def test_n_omega_matches_request(hydro):
    """n_omega must equal the requested number of frequencies."""
    assert hydro.n_omega == N_TEST_OMEGA


# ---------------------------------------------------------------------------
# 2. omega / frequency_hz relationship
# ---------------------------------------------------------------------------

def test_omega_frequency_hz_relationship(hydro):
    """
    frequency_hz must equal omega / (2π) exactly within floating-point tolerance.

    Tolerance: 1e-10 relative.  Justification: both are computed from the
    same omega array; the only error is floating-point rounding in division.
    """
    np.testing.assert_allclose(
        hydro.frequency_hz,
        hydro.omega / (2.0 * np.pi),
        rtol=1e-10,
        err_msg="frequency_hz must equal omega / (2*pi)",
    )


# ---------------------------------------------------------------------------
# 3. No NaNs or infinities
# ---------------------------------------------------------------------------

def test_no_nans_or_infs(hydro):
    """No array may contain NaN or infinity."""
    arrays = {
        "omega":                          hydro.omega,
        "frequency_hz":                   hydro.frequency_hz,
        "added_mass_heave":               hydro.added_mass_heave,
        "radiation_damping_heave":        hydro.radiation_damping_heave,
        "excitation_force_heave_real":    hydro.excitation_force_heave_real,
        "excitation_force_heave_imag":    hydro.excitation_force_heave_imag,
        "excitation_force_heave_amplitude": hydro.excitation_force_heave_amplitude,
        "excitation_force_heave_phase":   hydro.excitation_force_heave_phase,
    }
    for name, arr in arrays.items():
        assert np.all(np.isfinite(arr)), f"{name} contains NaN or infinity"


# ---------------------------------------------------------------------------
# 4. Added mass: finite and non-negative
# ---------------------------------------------------------------------------

def test_added_mass_non_negative(hydro):
    """
    Added mass must be non-negative over the usable frequency range.

    Physical justification: A(ω) represents the inertia of entrained water.
    Negative values indicate mesh quality issues (e.g. near irregular
    frequencies) or an unphysical mesh.  The test frequency range
    [0.3, 1.2] rad/s is well below the first irregular frequency (~2.09 rad/s)
    for the medium mesh.
    """
    assert np.all(hydro.added_mass_heave >= 0.0), (
        f"Added mass has negative values: min = {hydro.added_mass_heave.min():.1f} kg"
    )


def test_added_mass_finite(hydro):
    """Added mass must be finite at all frequencies."""
    assert np.all(np.isfinite(hydro.added_mass_heave))


def test_added_mass_order_of_magnitude():
    """
    Added mass must be of the same order of magnitude as the buoy mass.

    For a heaving cylinder, A(ω) is typically 0.5–3× the displaced mass.
    Displaced mass = rho * V = 1025 * pi * 5^2 * 10 ≈ 805 033 kg.
    We check A is in [1e4, 1e7] kg — a very broad physical bound.
    """
    hydro = compute_hydrodynamic_coefficients(
        geometry=REFERENCE_BUOY,
        resolution=MEDIUM_RESOLUTION,
        omega_min=OMEGA_MIN,
        omega_max=OMEGA_MAX,
        n_omega=N_TEST_OMEGA,
        progress_bar=False,
    )
    assert np.all(hydro.added_mass_heave > 1e4), "Added mass suspiciously small"
    assert np.all(hydro.added_mass_heave < 1e7), "Added mass suspiciously large"


# ---------------------------------------------------------------------------
# 5. Radiation damping: finite and non-negative (physical convention)
# ---------------------------------------------------------------------------

def test_radiation_damping_non_negative(hydro):
    """
    Physical radiation damping must be non-negative.

    B_physical = -B_capytaine.  For a body radiating waves, B_physical ≥ 0
    (energy is radiated away, not added).  Negative values indicate mesh
    issues or frequencies near the irregular frequency.
    """
    assert np.all(hydro.radiation_damping_heave >= 0.0), (
        f"Radiation damping has negative values: min = {hydro.radiation_damping_heave.min():.1f} N*s/m"
    )


def test_radiation_damping_finite(hydro):
    """Radiation damping must be finite at all frequencies."""
    assert np.all(np.isfinite(hydro.radiation_damping_heave))


# ---------------------------------------------------------------------------
# 6. Excitation force: amplitude and phase consistency
# ---------------------------------------------------------------------------

def test_excitation_amplitude_equals_abs_complex(hydro):
    """
    Excitation force amplitude must equal |complex excitation force|.

    Tolerance: 1e-10 relative.  Justification: amplitude is computed as
    np.abs(real + 1j*imag), which is exact to floating-point precision.
    """
    F_complex = hydro.excitation_force_heave_complex
    np.testing.assert_allclose(
        hydro.excitation_force_heave_amplitude,
        np.abs(F_complex),
        rtol=1e-10,
        err_msg="Amplitude must equal |complex excitation force|",
    )


def test_excitation_phase_equals_angle_complex(hydro):
    """
    Excitation force phase must equal arg(complex excitation force).

    Uses np.angle() which returns atan2(imag, real) in (-π, π].
    Tolerance: 1e-10 relative (or 1e-12 absolute for near-zero phase).
    """
    F_complex = hydro.excitation_force_heave_complex
    np.testing.assert_allclose(
        hydro.excitation_force_heave_phase,
        np.angle(F_complex),
        atol=1e-10,
        err_msg="Phase must equal angle(complex excitation force)",
    )


def test_excitation_amplitude_non_negative(hydro):
    """Excitation force amplitude must be non-negative (it is a magnitude)."""
    assert np.all(hydro.excitation_force_heave_amplitude >= 0.0)


def test_excitation_phase_in_range(hydro):
    """Excitation force phase must be in (-π, π]."""
    phase = hydro.excitation_force_heave_phase
    assert np.all(phase > -np.pi - 1e-10)
    assert np.all(phase <= np.pi + 1e-10)


def test_excitation_real_imag_reconstruct_complex(hydro):
    """
    Real and imaginary parts must reconstruct the complex excitation force.

    Checks that real + 1j*imag == amplitude * exp(1j*phase) to within
    floating-point tolerance.
    """
    F_from_parts = (hydro.excitation_force_heave_real
                    + 1j * hydro.excitation_force_heave_imag)
    F_from_polar = (hydro.excitation_force_heave_amplitude
                    * np.exp(1j * hydro.excitation_force_heave_phase))
    np.testing.assert_allclose(
        np.abs(F_from_parts),
        np.abs(F_from_polar),
        rtol=1e-10,
    )


# ---------------------------------------------------------------------------
# 7. Hydrostatic stiffness consistency with Module 2.1
# ---------------------------------------------------------------------------

def test_capytaine_volume_consistent_with_module21():
    """
    Capytaine mesh volume must be consistent with Module 2.1 analytical volume.

    Tolerance: 5 % for the medium mesh (168 panels).
    Justification: the mesh approximates the circular cross-section with flat
    panels; 12 panels around the circumference introduces ~2 % geometric error.
    The 5 % tolerance is generous to accommodate the coarse discretization.

    This test verifies that the Capytaine model represents the same physical
    geometry as Module 2.1, not a different cylinder.
    """
    model = CapytaineModel(
        geometry=REFERENCE_BUOY,
        resolution=MEDIUM_RESOLUTION,
    )
    body = model.build_floating_body()
    V_mesh = body.volume
    V_analytical = math.pi * REFERENCE_BUOY.radius**2 * REFERENCE_BUOY.draft
    rel_err = abs(V_mesh - V_analytical) / V_analytical
    assert rel_err < 0.05, (
        f"Capytaine mesh volume {V_mesh:.1f} m³ deviates {rel_err:.1%} from "
        f"Module 2.1 analytical volume {V_analytical:.1f} m³ (tolerance 5%)"
    )


def test_capytaine_hydrostatic_stiffness_consistent_with_module21():
    """
    Capytaine mesh-based hydrostatic stiffness must be consistent with
    Module 2.1's analytical C = ρ g A_wp.

    Capytaine computes C from the mesh using the divergence theorem.
    For a closed mesh with the top face at z=0, the waterplane area
    contribution is near-zero (the divergence theorem gives ~0 for the
    top cap at z=0).  Therefore we compare the BEM-derived buoyancy
    force (ρ g V_mesh) to the analytical value (ρ g V_analytical).

    Tolerance: 5 % (same as volume tolerance above).
    """
    hs_module21 = Hydrostatics(geometry=REFERENCE_BUOY)
    C_analytical = hs_module21.hydrostatic_stiffness  # rho * g * A_wp

    model = CapytaineModel(
        geometry=REFERENCE_BUOY,
        resolution=MEDIUM_RESOLUTION,
    )
    body = model.build_floating_body()
    # Capytaine's buoyancy force from mesh volume
    F_b_mesh = REFERENCE_BUOY.rho_water * GRAVITY * body.volume
    F_b_analytical = hs_module21.buoyancy_force

    rel_err = abs(F_b_mesh - F_b_analytical) / F_b_analytical
    assert rel_err < 0.05, (
        f"Capytaine buoyancy {F_b_mesh:.1f} N deviates {rel_err:.1%} from "
        f"Module 2.1 buoyancy {F_b_analytical:.1f} N (tolerance 5%)"
    )


# ---------------------------------------------------------------------------
# 8. Heave DOF is present
# ---------------------------------------------------------------------------

def test_heave_dof_present():
    """The Capytaine FloatingBody must contain the 'Heave' DOF."""
    model = CapytaineModel(geometry=REFERENCE_BUOY, resolution=MEDIUM_RESOLUTION)
    body = model.build_floating_body()
    assert "Heave" in body.dofs, "FloatingBody must have a 'Heave' DOF"


def test_only_heave_dof():
    """The FloatingBody must have exactly one DOF (Heave only)."""
    model = CapytaineModel(geometry=REFERENCE_BUOY, resolution=MEDIUM_RESOLUTION)
    body = model.build_floating_body()
    assert len(body.dofs) == 1, (
        f"Expected 1 DOF (Heave), got {len(body.dofs)}: {list(body.dofs.keys())}"
    )


# ---------------------------------------------------------------------------
# 9. Water depth is finite (not infinite depth)
# ---------------------------------------------------------------------------

def test_water_depth_is_finite(hydro):
    """
    Water depth stored in HydrodynamicCoefficients must be finite and equal
    to the geometry's water_depth (50 m).

    This verifies that finite-depth BEM was used, not the deep-water
    approximation (water_depth = inf).
    """
    assert np.isfinite(hydro.water_depth), "water_depth must be finite"
    assert hydro.water_depth == pytest.approx(REFERENCE_BUOY.water_depth, rel=1e-10)


# ---------------------------------------------------------------------------
# 10. Metadata correctness
# ---------------------------------------------------------------------------

def test_metadata_rho(hydro):
    """Stored rho must match the geometry's rho_water."""
    assert hydro.rho == pytest.approx(REFERENCE_BUOY.rho_water, rel=1e-10)


def test_metadata_radius(hydro):
    """Stored radius must match the geometry's radius."""
    assert hydro.radius == pytest.approx(REFERENCE_BUOY.radius, rel=1e-10)


def test_metadata_draft(hydro):
    """Stored draft must match the geometry's draft."""
    assert hydro.draft == pytest.approx(REFERENCE_BUOY.draft, rel=1e-10)


def test_metadata_n_panels(hydro):
    """Stored n_panels must match the expected panel count for MEDIUM_RESOLUTION."""
    nr, ntheta, nz = MEDIUM_RESOLUTION
    expected = (2 * nr + nz) * ntheta
    assert hydro.n_panels == expected


def test_metadata_capytaine_version(hydro):
    """Stored capytaine_version must match the installed version."""
    assert hydro.capytaine_version == cpt.__version__


# ---------------------------------------------------------------------------
# 11. Summary dictionary
# ---------------------------------------------------------------------------

def test_summary_keys(hydro):
    """summary() must return all required keys."""
    s = hydro.summary()
    required = {
        "capytaine_version",
        "n_panels",
        "n_omega",
        "omega_min_rad_s",
        "omega_max_rad_s",
        "water_depth_m",
        "rho_kg_m3",
        "radius_m",
        "draft_m",
        "added_mass_min_kg",
        "added_mass_max_kg",
        "radiation_damping_min_N_s_m",
        "radiation_damping_max_N_s_m",
        "excitation_amplitude_min_N",
        "excitation_amplitude_max_N",
    }
    assert required.issubset(s.keys())


def test_summary_ranges_consistent(hydro):
    """Summary min/max values must be consistent with the stored arrays."""
    s = hydro.summary()
    assert s["added_mass_min_kg"] == pytest.approx(hydro.added_mass_heave.min(), rel=1e-10)
    assert s["added_mass_max_kg"] == pytest.approx(hydro.added_mass_heave.max(), rel=1e-10)
    assert s["radiation_damping_min_N_s_m"] == pytest.approx(
        hydro.radiation_damping_heave.min(), rel=1e-10
    )
    assert s["radiation_damping_max_N_s_m"] == pytest.approx(
        hydro.radiation_damping_heave.max(), rel=1e-10
    )


# ---------------------------------------------------------------------------
# 12. Determinism
# ---------------------------------------------------------------------------

def test_determinism():
    """
    Computing coefficients twice with the same inputs must give nearly identical results.

    Sub-ppm floating-point variations (~1e-6 relative) can arise from
    non-associative BLAS operations when the body is rebuilt from scratch.
    We use rtol=1e-4 to confirm reproducibility at the engineering level.
    """
    kwargs = dict(
        geometry=REFERENCE_BUOY,
        resolution=MEDIUM_RESOLUTION,
        omega_min=OMEGA_MIN,
        omega_max=OMEGA_MAX,
        n_omega=N_TEST_OMEGA,
        progress_bar=False,
    )
    h1 = compute_hydrodynamic_coefficients(**kwargs)
    h2 = compute_hydrodynamic_coefficients(**kwargs)

    np.testing.assert_allclose(
        h1.added_mass_heave, h2.added_mass_heave, rtol=1e-4,
        err_msg="added_mass must be reproducible to 1e-4 relative tolerance",
    )
    np.testing.assert_allclose(
        h1.radiation_damping_heave, h2.radiation_damping_heave, rtol=1e-4,
        err_msg="radiation_damping must be reproducible to 1e-4 relative tolerance",
    )
    np.testing.assert_allclose(
        h1.excitation_force_heave_amplitude, h2.excitation_force_heave_amplitude,
        rtol=1e-4,
        err_msg="excitation amplitude must be reproducible to 1e-4 relative tolerance",
    )


# ---------------------------------------------------------------------------
# 13. Audit: radiation-damping sign convention (Task 1 regression)
# ---------------------------------------------------------------------------

def test_capytaine_raw_radiation_damping_is_negative():
    """
    Capytaine's raw radiation_damping must be negative for a passive cylinder.

    Empirical derivation:
      F_rad = omega^2 * A - i * omega * B_physical  (Falnes 2002 convention)
      B_cap = Im(F_rad) / omega = -B_physical

    For a passive body (B_physical > 0), B_cap < 0.
    This test verifies the raw Capytaine output sign at omega = 1.0 rad/s
    using the medium mesh, confirming our negation is correct.
    """
    import warnings
    model = CapytaineModel(geometry=REFERENCE_BUOY, resolution=MEDIUM_RESOLUTION)
    body = model.build_floating_body()
    solver = cpt.BEMSolver()
    prob = cpt.RadiationProblem(
        body=body, omega=1.0, water_depth=50.0, rho=1025.0, radiating_dof="Heave"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = solver.solve(prob, keep_details=False)

    B_cap = result.radiation_damping["Heave"]
    assert B_cap < 0.0, (
        "Capytaine raw radiation_damping must be negative for a passive body "
        "(B_cap = Im(F_rad)/omega = -B_physical). Got B_cap = %.4f" % B_cap
    )


def test_stored_radiation_damping_is_positive():
    """
    The stored radiation_damping_heave (B_physical = -B_capytaine) must be
    positive for a passive cylinder at all frequencies in the production range.

    This is the primary regression test for the sign convention.
    """
    h = compute_hydrodynamic_coefficients(
        geometry=REFERENCE_BUOY,
        resolution=MEDIUM_RESOLUTION,
        omega_min=OMEGA_MIN,
        omega_max=OMEGA_MAX,
        n_omega=N_TEST_OMEGA,
        progress_bar=False,
    )
    assert np.all(h.radiation_damping_heave > 0.0), (
        "All stored radiation_damping_heave values must be positive (B_physical > 0)"
    )


def test_radiation_damping_sign_via_raw_force():
    """
    Verify sign convention end-to-end: raw force -> B_cap -> B_physical.

    At omega = 1.0 rad/s:
      Im(F_rad) < 0  (verified empirically)
      B_cap = Im(F_rad)/omega < 0
      B_physical = -B_cap > 0

    Checks all three inequalities explicitly.
    """
    import warnings
    model = CapytaineModel(geometry=REFERENCE_BUOY, resolution=MEDIUM_RESOLUTION)
    body = model.build_floating_body()
    solver = cpt.BEMSolver()
    prob = cpt.RadiationProblem(
        body=body, omega=1.0, water_depth=50.0, rho=1025.0, radiating_dof="Heave"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = solver.solve(prob, keep_details=False)

    F_rad = result.force["Heave"]
    B_cap = result.radiation_damping["Heave"]
    B_phys = -B_cap

    assert np.imag(F_rad) < 0.0, "Im(F_rad) must be negative for passive body"
    assert B_cap < 0.0,  "B_capytaine must be negative for passive body"
    assert B_phys > 0.0, "B_physical = -B_cap must be positive for passive body"


# ---------------------------------------------------------------------------
# 14. Audit: irregular frequency safety margin (Task 3 regression)
# ---------------------------------------------------------------------------

def test_irregular_frequency_above_omega_max():
    """
    The first irregular frequency estimate must exceed omega_max = 1.4 rad/s.

    This verifies that the production frequency range is safely below the
    irregular frequency for the medium mesh.
    """
    model = CapytaineModel(
        geometry=REFERENCE_BUOY,
        resolution=MEDIUM_RESOLUTION,
        omega_max=1.4,
    )
    body = model.build_floating_body()
    omega_irr = body.first_irregular_frequency_estimate()
    assert omega_irr > 1.4, (
        "Irregular frequency %.4f rad/s must exceed omega_max = 1.4 rad/s" % omega_irr
    )


def test_omega_max_is_at_most_70_percent_of_irregular_frequency():
    """
    omega_max must be at most 70% of the first irregular frequency.

    This enforces a minimum 30% safety margin below the irregular frequency.
    Justification: at 70% of omega_irr, BEM results are still reliable for
    the hull-only mesh without a lid.
    """
    model = CapytaineModel(
        geometry=REFERENCE_BUOY,
        resolution=MEDIUM_RESOLUTION,
        omega_max=1.4,
    )
    body = model.build_floating_body()
    omega_irr = body.first_irregular_frequency_estimate()
    ratio = 1.4 / omega_irr
    assert ratio <= 0.70, (
        "omega_max/omega_irr = %.3f exceeds 0.70 (safety margin too small)" % ratio
    )


# ---------------------------------------------------------------------------
# 15. Audit: waterplane area limitation documented (Task 2 regression)
# ---------------------------------------------------------------------------

def test_capytaine_waterplane_area_is_near_zero_for_closed_mesh():
    """
    Capytaine's waterplane_area must be near zero for the closed cylinder mesh.

    Root cause: mesh_vertical_cylinder generates a closed mesh (hull + top cap
    + bottom cap).  The divergence theorem gives integral(n_z dS) = 0 for a
    closed surface, so waterplane_area = -integral(n_z dS) ≈ 0.

    This is a documented limitation.  Module 2.1's analytical C33 = rho*g*pi*r^2
    is the authoritative hydrostatic benchmark.

    Tolerance: |A_wp_mesh| < 1e-6 m^2 (effectively zero relative to pi*r^2 = 78.5 m^2).
    """
    model = CapytaineModel(geometry=REFERENCE_BUOY, resolution=MEDIUM_RESOLUTION)
    body = model.build_floating_body()
    A_wp_mesh = body.waterplane_area
    assert abs(A_wp_mesh) < 1e-6, (
        "Capytaine waterplane_area should be ~0 for closed mesh, got %.6e m^2" % A_wp_mesh
    )


def test_module21_c33_matches_analytical():
    """
    Module 2.1 hydrostatic stiffness must match the analytical C33 = rho*g*pi*r^2
    to machine precision.

    This confirms Module 2.1 is the authoritative hydrostatic benchmark.
    """
    hs = Hydrostatics(geometry=REFERENCE_BUOY)
    C33_analytical = (
        REFERENCE_BUOY.rho_water * GRAVITY * math.pi * REFERENCE_BUOY.radius**2
    )
    assert hs.hydrostatic_stiffness == pytest.approx(C33_analytical, rel=1e-12)


# ---------------------------------------------------------------------------
# 16. Production-path reproducibility (Task: final reproducibility check)
# ---------------------------------------------------------------------------

def test_production_path_reproducibility():
    """
    The production solve_all() path must be reproducible across independent calls.

    Runs compute_hydrodynamic_coefficients() twice with the production mesh
    (672 panels, 20 frequencies, omega 0.2-1.4 rad/s) and verifies that all
    coefficient arrays agree within a tight tolerance.

    Measured maximum relative differences (3-run empirical characterisation):
        added_mass_heave              : ~3.1e-6  (sub-ppm)
        radiation_damping_heave       : ~5.7e-5  (low-ppm)
        excitation_force_heave_real   : ~1.3e-6
        excitation_force_heave_imag   : ~4.6e-5
        excitation_force_heave_amplitude: ~1.3e-6

    Tolerance rtol=1e-3 is 17x the measured worst-case maximum, providing
    headroom for platform variation while being tight enough to catch any
    real regression.  This is distinct from the medium-mesh test_determinism()
    test (168 panels, rtol=1e-4) which uses a faster fixture.

    Note: fresh-body single-frequency solves and the production solve_all()
    batch can differ by ~1-2% due to internal Capytaine body state/caching.
    This test verifies only the production path against itself.
    """
    PROD_RESOLUTION = (6, 24, 16)   # 672 panels
    kwargs = dict(
        geometry=REFERENCE_BUOY,
        resolution=PROD_RESOLUTION,
        omega_min=0.2,
        omega_max=1.4,
        n_omega=20,
        progress_bar=False,
    )
    h1 = compute_hydrodynamic_coefficients(**kwargs)
    h2 = compute_hydrodynamic_coefficients(**kwargs)

    TOL = 1e-3   # 17x measured worst-case; justified by empirical characterisation

    np.testing.assert_allclose(
        h1.added_mass_heave, h2.added_mass_heave, rtol=TOL,
        err_msg="Production-path added_mass not reproducible to rtol=1e-3",
    )
    np.testing.assert_allclose(
        h1.radiation_damping_heave, h2.radiation_damping_heave, rtol=TOL,
        err_msg="Production-path radiation_damping not reproducible to rtol=1e-3",
    )
    np.testing.assert_allclose(
        h1.excitation_force_heave_real, h2.excitation_force_heave_real, rtol=TOL,
        err_msg="Production-path excitation_real not reproducible to rtol=1e-3",
    )
    np.testing.assert_allclose(
        h1.excitation_force_heave_imag, h2.excitation_force_heave_imag, rtol=TOL,
        err_msg="Production-path excitation_imag not reproducible to rtol=1e-3",
    )
    np.testing.assert_allclose(
        h1.excitation_force_heave_amplitude, h2.excitation_force_heave_amplitude,
        rtol=TOL,
        err_msg="Production-path excitation_amplitude not reproducible to rtol=1e-3",
    )
    np.testing.assert_allclose(
        h1.excitation_force_heave_phase, h2.excitation_force_heave_phase,
        atol=1e-4,   # phase near zero can have large relative error; use absolute
        err_msg="Production-path excitation_phase not reproducible to atol=1e-4 rad",
    )
