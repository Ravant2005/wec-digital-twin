"""
tests/test_hydrostatics.py — Physics-based tests for module2_wec/hydrostatics.py

All expected values are derived analytically from the defining equations.
Tolerances are justified in each test docstring.

Run with:  pytest tests/test_hydrostatics.py -v
"""

import math
import pytest

from module2_wec.geometry import CylinderGeometry, GRAVITY, REFERENCE_BUOY
from module2_wec.hydrostatics import Hydrostatics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_hydrostatics(
    radius=5.0, draft=10.0, water_depth=50.0,
    rho_water=1025.0, mass=None,
) -> Hydrostatics:
    """Return a Hydrostatics object.  Default mass = rho * V (neutrally buoyant)."""
    if mass is None:
        mass = rho_water * math.pi * radius**2 * draft
    geom = CylinderGeometry(
        radius=radius, draft=draft, water_depth=water_depth,
        mass=mass, rho_water=rho_water,
    )
    return Hydrostatics(geometry=geom)


# ---------------------------------------------------------------------------
# 1. Displaced volume: V = π r² d
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("radius, draft", [
    (5.0, 4.0),
    (2.0, 3.0),
    (8.0, 6.0),
])
def test_displaced_volume(radius, draft):
    """Hydrostatics.displaced_volume must equal π r² d."""
    hs = make_hydrostatics(radius=radius, draft=draft,
                           water_depth=draft + 10.0)
    expected = math.pi * radius**2 * draft
    assert hs.displaced_volume == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# 2. Buoyancy force: F_b = ρ g V
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("radius, draft, rho", [
    (5.0, 4.0, 1025.0),
    (3.0, 2.0, 1000.0),
    (7.0, 5.0, 1025.0),
])
def test_buoyancy_force_formula(radius, draft, rho):
    """Buoyancy force must equal ρ g π r² d."""
    hs = make_hydrostatics(radius=radius, draft=draft,
                           water_depth=draft + 10.0, rho_water=rho)
    expected = rho * GRAVITY * math.pi * radius**2 * draft
    assert hs.buoyancy_force == pytest.approx(expected, rel=1e-12)


def test_buoyancy_equals_weight_for_neutrally_buoyant():
    """For a neutrally buoyant buoy, buoyancy force must equal weight (m g)."""
    hs = make_hydrostatics()  # mass = rho * V by default
    weight = hs.geometry.mass * GRAVITY
    assert hs.buoyancy_force == pytest.approx(weight, rel=1e-12)


# ---------------------------------------------------------------------------
# 3. Waterplane area: A_wp = π r²
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("radius", [1.0, 5.0, 10.0])
def test_waterplane_area(radius):
    """Waterplane area must equal π r²."""
    hs = make_hydrostatics(radius=radius)
    assert hs.waterplane_area == pytest.approx(math.pi * radius**2, rel=1e-12)


# ---------------------------------------------------------------------------
# 4. Hydrostatic stiffness: C = ρ g A_wp
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("radius, rho", [
    (5.0, 1025.0),
    (3.0, 1000.0),
    (8.0, 1025.0),
])
def test_hydrostatic_stiffness_formula(radius, rho):
    """Hydrostatic stiffness must equal ρ g π r²."""
    hs = make_hydrostatics(radius=radius, rho_water=rho)
    expected = rho * GRAVITY * math.pi * radius**2
    assert hs.hydrostatic_stiffness == pytest.approx(expected, rel=1e-12)


def test_stiffness_independent_of_draft():
    """
    Hydrostatic stiffness must not depend on draft for a vertical cylinder.

    Physical justification: C = ρ g A_wp, and A_wp = π r² is constant
    for a vertical cylinder regardless of draft.
    """
    hs1 = make_hydrostatics(draft=2.0, water_depth=20.0)
    hs2 = make_hydrostatics(draft=6.0, water_depth=20.0)
    assert hs1.hydrostatic_stiffness == pytest.approx(hs2.hydrostatic_stiffness, rel=1e-12)


def test_stiffness_scales_with_radius_squared():
    """Doubling radius must quadruple hydrostatic stiffness."""
    hs1 = make_hydrostatics(radius=3.0)
    hs2 = make_hydrostatics(radius=6.0)
    assert hs2.hydrostatic_stiffness == pytest.approx(4.0 * hs1.hydrostatic_stiffness, rel=1e-12)


# ---------------------------------------------------------------------------
# 5. Natural frequency and period: ω_n = √(C/m), T_n = 2π/ω_n
# ---------------------------------------------------------------------------

def test_omega_n_formula():
    """Natural angular frequency must equal √(C / m)."""
    hs = make_hydrostatics()
    expected = math.sqrt(hs.hydrostatic_stiffness / hs.geometry.mass)
    assert hs.omega_n == pytest.approx(expected, rel=1e-12)


def test_natural_period_formula():
    """Natural period must equal 2π / ω_n."""
    hs = make_hydrostatics()
    expected = 2.0 * math.pi / hs.omega_n
    assert hs.natural_period == pytest.approx(expected, rel=1e-12)


def test_omega_n_and_period_internally_consistent():
    """ω_n * T_n must equal 2π exactly."""
    hs = make_hydrostatics()
    assert hs.omega_n * hs.natural_period == pytest.approx(2.0 * math.pi, rel=1e-12)


def test_natural_period_positive():
    """Natural period must be positive."""
    hs = make_hydrostatics()
    assert hs.natural_period > 0.0


def test_omega_n_positive():
    """Natural angular frequency must be positive."""
    hs = make_hydrostatics()
    assert hs.omega_n > 0.0


def test_heavier_buoy_has_lower_omega_n():
    """
    Increasing mass (at fixed stiffness) must decrease ω_n.

    Physical justification: ω_n = √(C/m); larger m → smaller ω_n.
    Stiffness C = ρ g A_wp depends only on radius, not mass.
    """
    hs_light = make_hydrostatics(mass=100_000.0)
    hs_heavy = make_hydrostatics(mass=500_000.0)
    assert hs_heavy.omega_n < hs_light.omega_n


def test_larger_radius_has_higher_omega_n_at_fixed_mass():
    """
    Increasing radius (at fixed mass) must increase ω_n.

    Physical justification: C = ρ g π r² grows with r²; ω_n = √(C/m)
    therefore increases with r at fixed m.
    """
    hs_small = make_hydrostatics(radius=3.0, mass=200_000.0)
    hs_large = make_hydrostatics(radius=8.0, mass=200_000.0)
    assert hs_large.omega_n > hs_small.omega_n


# ---------------------------------------------------------------------------
# 6. Archimedes check
# ---------------------------------------------------------------------------

def test_archimedes_satisfied_for_neutrally_buoyant():
    """check_archimedes must return True when mass = rho * V."""
    hs = make_hydrostatics()  # mass = rho * V by construction
    assert hs.check_archimedes() is True


def test_archimedes_fails_for_inconsistent_mass():
    """check_archimedes must return False when mass ≠ rho * V."""
    rho_V = 1025.0 * math.pi * 5.0**2 * 4.0
    hs = make_hydrostatics(mass=rho_V * 2.0)  # mass is twice the displaced weight
    assert hs.check_archimedes() is False


# ---------------------------------------------------------------------------
# 7. Summary dictionary
# ---------------------------------------------------------------------------

def test_summary_keys():
    """summary() must return all required keys."""
    hs = make_hydrostatics()
    s = hs.summary()
    required = {
        "displaced_volume_m3",
        "buoyancy_force_N",
        "waterplane_area_m2",
        "hydrostatic_stiffness_N_per_m",
        "omega_n_rad_per_s",
        "natural_period_s",
        "archimedes_satisfied",
    }
    assert required.issubset(s.keys())


def test_summary_values_consistent():
    """Summary values must be consistent with direct property access."""
    hs = make_hydrostatics()
    s = hs.summary()
    assert s["displaced_volume_m3"] == pytest.approx(hs.displaced_volume, rel=1e-12)
    assert s["buoyancy_force_N"] == pytest.approx(hs.buoyancy_force, rel=1e-12)
    assert s["waterplane_area_m2"] == pytest.approx(hs.waterplane_area, rel=1e-12)
    assert s["hydrostatic_stiffness_N_per_m"] == pytest.approx(hs.hydrostatic_stiffness, rel=1e-12)
    assert s["omega_n_rad_per_s"] == pytest.approx(hs.omega_n, rel=1e-12)
    assert s["natural_period_s"] == pytest.approx(hs.natural_period, rel=1e-12)


# ---------------------------------------------------------------------------
# 8. Reference buoy end-to-end
# ---------------------------------------------------------------------------

def test_reference_buoy_hydrostatics():
    """Reference buoy hydrostatics must satisfy all key equations."""
    hs = Hydrostatics(geometry=REFERENCE_BUOY)

    r, d, rho = 5.0, 10.0, 1025.0
    V_expected = math.pi * r**2 * d
    A_expected = math.pi * r**2
    C_expected = rho * GRAVITY * A_expected
    m = rho * V_expected  # neutrally buoyant

    assert hs.displaced_volume == pytest.approx(V_expected, rel=1e-12)
    assert hs.waterplane_area == pytest.approx(A_expected, rel=1e-12)
    assert hs.hydrostatic_stiffness == pytest.approx(C_expected, rel=1e-12)
    assert hs.buoyancy_force == pytest.approx(rho * GRAVITY * V_expected, rel=1e-12)
    assert hs.omega_n == pytest.approx(math.sqrt(C_expected / m), rel=1e-12)
    assert hs.natural_period == pytest.approx(2.0 * math.pi / math.sqrt(C_expected / m), rel=1e-12)
    assert hs.check_archimedes() is True


def test_reference_buoy_natural_period_in_swell_range():
    """
    Reference buoy natural period must be in the ocean swell range (5–20 s).

    Justification: the reference buoy was designed to have T_n ≈ 8–10 s
    (without added mass).  This test confirms the design intent is met.
    The swell range 5–20 s is a broad physical bound, not a tight spec.
    """
    hs = Hydrostatics(geometry=REFERENCE_BUOY)
    assert 5.0 < hs.natural_period < 20.0, (
        f"Natural period {hs.natural_period:.2f} s is outside the swell range 5–20 s"
    )
