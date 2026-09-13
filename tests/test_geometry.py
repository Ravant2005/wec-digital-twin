"""
tests/test_geometry.py — Physics-based tests for module2_wec/geometry.py

All expected values are derived analytically from the defining equations;
no arbitrary magic numbers are used.

Run with:  pytest tests/test_geometry.py -v
"""

import math
import pytest

from module2_wec.geometry import (
    CylinderGeometry,
    REFERENCE_BUOY,
    GRAVITY,
    RHO_SEAWATER,
    _RHO_MIN,
    _RHO_MAX,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_buoy(**kwargs) -> CylinderGeometry:
    """Return a valid CylinderGeometry, overriding defaults with kwargs."""
    defaults = dict(radius=5.0, draft=10.0, water_depth=50.0,
                    mass=1025.0 * math.pi * 5.0**2 * 10.0)
    defaults.update(kwargs)
    return CylinderGeometry(**defaults)


# ---------------------------------------------------------------------------
# 1. Displaced volume: V = π r² d
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("radius, draft", [
    (5.0, 4.0),
    (1.0, 1.0),
    (3.0, 2.5),
    (10.0, 8.0),
])
def test_displaced_volume_formula(radius, draft):
    """Displaced volume must equal π r² d exactly."""
    geom = make_buoy(radius=radius, draft=draft,
                     water_depth=draft + 10.0,
                     mass=1025.0 * math.pi * radius**2 * draft)
    expected = math.pi * radius**2 * draft
    assert geom.displaced_volume == pytest.approx(expected, rel=1e-12)


# ---------------------------------------------------------------------------
# 2. Waterplane area: A_wp = π r²
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("radius", [1.0, 3.0, 5.0, 10.0])
def test_waterplane_area_formula(radius):
    """Waterplane area must equal π r² exactly."""
    geom = make_buoy(radius=radius)
    expected = math.pi * radius**2
    assert geom.waterplane_area == pytest.approx(expected, rel=1e-12)


def test_waterplane_area_independent_of_draft():
    """For a vertical cylinder, waterplane area must not depend on draft."""
    geom1 = make_buoy(draft=2.0, water_depth=20.0,
                      mass=1025.0 * math.pi * 5.0**2 * 2.0)
    geom2 = make_buoy(draft=6.0, water_depth=20.0,
                      mass=1025.0 * math.pi * 5.0**2 * 6.0)
    assert geom1.waterplane_area == pytest.approx(geom2.waterplane_area, rel=1e-12)


# ---------------------------------------------------------------------------
# 3. Volume scales with radius squared
# ---------------------------------------------------------------------------

def test_volume_scales_with_radius_squared():
    """Doubling radius must quadruple displaced volume (at same draft)."""
    geom1 = make_buoy(radius=3.0, draft=4.0, water_depth=20.0,
                      mass=1025.0 * math.pi * 3.0**2 * 4.0)
    geom2 = make_buoy(radius=6.0, draft=4.0, water_depth=20.0,
                      mass=1025.0 * math.pi * 6.0**2 * 4.0)
    assert geom2.displaced_volume == pytest.approx(4.0 * geom1.displaced_volume, rel=1e-12)


# ---------------------------------------------------------------------------
# 4. Volume scales linearly with draft
# ---------------------------------------------------------------------------

def test_volume_scales_linearly_with_draft():
    """Doubling draft must double displaced volume (at same radius)."""
    geom1 = make_buoy(draft=3.0, water_depth=20.0,
                      mass=1025.0 * math.pi * 5.0**2 * 3.0)
    geom2 = make_buoy(draft=6.0, water_depth=20.0,
                      mass=1025.0 * math.pi * 5.0**2 * 6.0)
    assert geom2.displaced_volume == pytest.approx(2.0 * geom1.displaced_volume, rel=1e-12)


# ---------------------------------------------------------------------------
# 5. Reference buoy sanity checks
# ---------------------------------------------------------------------------

def test_reference_buoy_volume():
    """Reference buoy displaced volume must equal π * 5² * 10."""
    expected = math.pi * 5.0**2 * 10.0
    assert REFERENCE_BUOY.displaced_volume == pytest.approx(expected, rel=1e-12)


def test_reference_buoy_waterplane_area():
    """Reference buoy waterplane area must equal π * 5²."""
    expected = math.pi * 5.0**2
    assert REFERENCE_BUOY.waterplane_area == pytest.approx(expected, rel=1e-12)


def test_reference_buoy_is_neutrally_buoyant():
    """Reference buoy mass must equal rho * V (neutrally buoyant by construction)."""
    rho_V = REFERENCE_BUOY.rho_water * REFERENCE_BUOY.displaced_volume
    assert REFERENCE_BUOY.mass == pytest.approx(rho_V, rel=1e-12)


def test_reference_buoy_draft():
    """Reference buoy draft must be 10.0 m."""
    assert REFERENCE_BUOY.draft == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 6. Input validation — invalid geometry must raise ValueError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs, match", [
    ({"radius": 0.0},                   "radius"),
    ({"radius": -1.0},                  "radius"),
    ({"draft": 0.0},                    "draft"),
    ({"draft": -2.0},                   "draft"),
    ({"water_depth": 0.0},              "water_depth"),
    ({"water_depth": -5.0},             "water_depth"),
    ({"water_depth": 3.0},              "water_depth"),   # water_depth <= draft
    ({"mass": 0.0},                     "mass"),
    ({"mass": -100.0},                  "mass"),
    ({"rho_water": _RHO_MIN - 1.0},     "rho_water"),
    ({"rho_water": _RHO_MAX + 1.0},     "rho_water"),
])
def test_invalid_geometry_raises(kwargs, match):
    """Invalid geometry parameters must raise ValueError with a descriptive message."""
    with pytest.raises(ValueError, match=match):
        make_buoy(**kwargs)


def test_water_depth_equal_to_draft_raises():
    """water_depth == draft must raise ValueError (buoy touches seabed)."""
    with pytest.raises(ValueError, match="water_depth"):
        make_buoy(draft=4.0, water_depth=4.0)


# ---------------------------------------------------------------------------
# 7. Valid boundary values are accepted
# ---------------------------------------------------------------------------

def test_rho_water_at_lower_bound_accepted():
    """rho_water = _RHO_MIN must be accepted."""
    geom = make_buoy(rho_water=_RHO_MIN)
    assert geom.rho_water == _RHO_MIN


def test_rho_water_at_upper_bound_accepted():
    """rho_water = _RHO_MAX must be accepted."""
    geom = make_buoy(rho_water=_RHO_MAX)
    assert geom.rho_water == _RHO_MAX


def test_very_small_radius_accepted():
    """A small but positive radius must be accepted."""
    geom = make_buoy(radius=0.01, draft=0.01, water_depth=1.0,
                     mass=1025.0 * math.pi * 0.01**2 * 0.01)
    assert geom.radius == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# 8. Immutability: dataclass fields are accessible
# ---------------------------------------------------------------------------

def test_geometry_fields_accessible():
    """All geometry fields must be accessible as attributes."""
    geom = make_buoy()
    assert geom.radius == pytest.approx(5.0)
    assert geom.draft == pytest.approx(10.0)
    assert geom.water_depth == pytest.approx(50.0)
    assert geom.mass > 0.0
    assert geom.rho_water == pytest.approx(RHO_SEAWATER)
