"""
tests/test_capytaine_model.py — Tests for module2_wec/capytaine_model.py

Tests cover:
  - FloatingBody construction (mesh, DOF, panels)
  - Frequency grid properties
  - BEM dataset structure and coordinate consistency
  - Water depth is passed explicitly (not infinite depth)
  - Heave DOF is present
  - Determinism: same inputs → same outputs

Uses a COARSE mesh (resolution (2, 8, 5) = 72 panels) and a small frequency
grid (5 points) to keep test runtime short.  Physical accuracy is tested in
test_hydrodynamic_coefficients.py with a finer mesh.

Run with:  pytest tests/test_capytaine_model.py -v
"""

import math
import numpy as np
import pytest

import capytaine as cpt

from module2_wec.geometry import CylinderGeometry, REFERENCE_BUOY
from module2_wec.capytaine_model import CapytaineModel, DEFAULT_RESOLUTION


# ---------------------------------------------------------------------------
# Shared fixture: coarse model run once per module
# ---------------------------------------------------------------------------

COARSE_RESOLUTION = (2, 8, 5)   # 72 panels — fast for tests
TEST_OMEGAS = np.linspace(0.3, 1.0, 5)


@pytest.fixture(scope="module")
def coarse_model():
    """CapytaineModel with coarse mesh and 5 frequencies."""
    return CapytaineModel(
        geometry=REFERENCE_BUOY,
        resolution=COARSE_RESOLUTION,
        omega_min=float(TEST_OMEGAS[0]),
        omega_max=float(TEST_OMEGAS[-1]),
        n_omega=len(TEST_OMEGAS),
    )


@pytest.fixture(scope="module")
def coarse_dataset(coarse_model):
    """Raw Capytaine dataset from the coarse model."""
    return coarse_model.run(progress_bar=False)


# ---------------------------------------------------------------------------
# 1. CapytaineModel construction and validation
# ---------------------------------------------------------------------------

def test_model_omega_grid_length(coarse_model):
    """omega_grid must have n_omega elements."""
    assert len(coarse_model.omega_grid) == coarse_model.n_omega


def test_model_omega_grid_strictly_increasing(coarse_model):
    """omega_grid must be strictly increasing."""
    omega = coarse_model.omega_grid
    assert np.all(np.diff(omega) > 0)


def test_model_omega_grid_bounds(coarse_model):
    """omega_grid first/last values must match omega_min/omega_max."""
    omega = coarse_model.omega_grid
    assert omega[0] == pytest.approx(coarse_model.omega_min, rel=1e-12)
    assert omega[-1] == pytest.approx(coarse_model.omega_max, rel=1e-12)


def test_model_n_panels_formula():
    """n_panels must equal (2*nr + nz) * ntheta."""
    for res in [(2, 8, 5), (3, 12, 8), (4, 16, 12)]:
        nr, ntheta, nz = res
        model = CapytaineModel(geometry=REFERENCE_BUOY, resolution=res)
        expected = (2 * nr + nz) * ntheta
        assert model.n_panels() == expected


@pytest.mark.parametrize("kwargs, exc_match", [
    ({"omega_min": 0.0},  "omega_min"),
    ({"omega_min": -0.1}, "omega_min"),
    ({"omega_max": 0.1, "omega_min": 0.2}, "omega_max"),
    ({"n_omega": 1},      "n_omega"),
])
def test_model_invalid_params_raise(kwargs, exc_match):
    """Invalid CapytaineModel parameters must raise ValueError."""
    with pytest.raises(ValueError, match=exc_match):
        CapytaineModel(geometry=REFERENCE_BUOY, **kwargs)


# ---------------------------------------------------------------------------
# 2. FloatingBody construction
# ---------------------------------------------------------------------------

def test_floating_body_has_heave_dof(coarse_model):
    """The built FloatingBody must have a 'Heave' DOF."""
    body = coarse_model.build_floating_body()
    assert "Heave" in body.dofs


def test_floating_body_panel_count(coarse_model):
    """The built FloatingBody must have the expected number of panels."""
    body = coarse_model.build_floating_body()
    assert body.mesh.nb_faces == coarse_model.n_panels()


def test_floating_body_volume_approx():
    """
    Immersed volume must be close to π r² d (within 12 % for coarse mesh).

    Tolerance justification: the mesh_vertical_cylinder approximates the
    circular cross-section with flat panels.  With only 8 panels around
    the circumference (COARSE_RESOLUTION), the inscribed polygon area is
    (8/2π) * sin(2π/8) ≈ 90.5 % of the circle area, giving ~10 % volume
    error.  The 12 % tolerance provides a small margin above this geometric
    error without being so loose as to miss gross implementation mistakes.
    """
    model = CapytaineModel(
        geometry=REFERENCE_BUOY,
        resolution=COARSE_RESOLUTION,
    )
    body = model.build_floating_body()
    V_mesh = body.volume
    V_analytical = math.pi * REFERENCE_BUOY.radius**2 * REFERENCE_BUOY.draft
    rel_err = abs(V_mesh - V_analytical) / V_analytical
    assert rel_err < 0.12, (
        f"Mesh volume {V_mesh:.1f} m³ deviates {rel_err:.1%} from "
        f"analytical {V_analytical:.1f} m³ (tolerance 12% for coarse mesh)"
    )


# ---------------------------------------------------------------------------
# 3. Dataset structure
# ---------------------------------------------------------------------------

def test_dataset_has_required_variables(coarse_dataset):
    """Dataset must contain added_mass, radiation_damping, excitation_force."""
    for var in ("added_mass", "radiation_damping", "excitation_force"):
        assert var in coarse_dataset.data_vars, f"Missing variable: {var}"


def test_dataset_omega_length(coarse_model, coarse_dataset):
    """Dataset omega coordinate must have n_omega elements."""
    assert len(coarse_dataset.coords["omega"]) == coarse_model.n_omega


def test_dataset_omega_matches_grid(coarse_model, coarse_dataset):
    """Dataset omega values must match the model's omega_grid."""
    np.testing.assert_allclose(
        coarse_dataset.coords["omega"].values,
        coarse_model.omega_grid,
        rtol=1e-10,
    )


def test_dataset_water_depth_is_finite_depth(coarse_dataset):
    """
    Water depth in dataset must equal the geometry's water_depth (50 m),
    not np.inf (infinite depth).

    This verifies that finite-depth BEM was used, not the deep-water
    approximation.
    """
    depth = float(coarse_dataset.coords["water_depth"].values)
    assert np.isfinite(depth), "water_depth must be finite (not infinite depth)"
    assert depth == pytest.approx(REFERENCE_BUOY.water_depth, rel=1e-10)


def test_dataset_rho_matches_geometry(coarse_dataset):
    """Dataset rho must match the geometry's rho_water."""
    rho = float(coarse_dataset.coords["rho"].values)
    assert rho == pytest.approx(REFERENCE_BUOY.rho_water, rel=1e-10)


def test_dataset_freq_omega_relationship(coarse_dataset):
    """
    Dataset freq coordinate must satisfy freq = omega / (2π) exactly
    within floating-point tolerance.
    """
    omega = coarse_dataset.coords["omega"].values
    freq = coarse_dataset.coords["freq"].values
    np.testing.assert_allclose(freq, omega / (2.0 * np.pi), rtol=1e-10)


# ---------------------------------------------------------------------------
# 4. Determinism
# ---------------------------------------------------------------------------

def test_determinism(coarse_model):
    """
    Running the same model twice must produce nearly identical datasets.

    Capytaine BEM is deterministic for a fixed mesh and frequency grid when
    the same body object is reused.  When the body is rebuilt from scratch,
    sub-ppm floating-point variations (~1e-6 relative) can arise from
    non-associative BLAS operations.  We use rtol=1e-4 to confirm
    reproducibility at the engineering level while tolerating BLAS rounding.
    """
    ds1 = coarse_model.run(progress_bar=False)
    ds2 = coarse_model.run(progress_bar=False)
    np.testing.assert_allclose(
        ds1["added_mass"].values,
        ds2["added_mass"].values,
        rtol=1e-4,
        err_msg="added_mass must be reproducible to 1e-4 relative tolerance",
    )
    np.testing.assert_allclose(
        ds1["radiation_damping"].values,
        ds2["radiation_damping"].values,
        rtol=1e-4,
        err_msg="radiation_damping must be reproducible to 1e-4 relative tolerance",
    )
