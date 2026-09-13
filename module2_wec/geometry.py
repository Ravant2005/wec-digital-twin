"""
geometry.py — Buoy geometry for the WEC Digital Twin (Module 2).

Defines the physical geometry of a vertical circular cylinder heaving
point-absorber.  This is the foundational layer; hydrodynamic coefficients
(added mass, radiation damping, excitation force) are NOT computed here —
they will be obtained from Capytaine BEM in a future milestone.

Temporary reference buoy
------------------------
The parameters below define a TEMPORARY DEVELOPMENT BUOY used only for
testing and algorithm development.  They are NOT the final Goa deployment
buoy geometry.  All parameters are explicit and configurable.

Coordinate system
-----------------
    z = 0   : undisturbed free surface
    z < 0   : submerged (positive draft means the keel is at z = -draft)
    z > 0   : above waterplane

Degree of freedom
-----------------
    Only heave (vertical translation, z-axis) is modelled in Module 2.
    Surge, sway, roll, pitch, yaw are not included at this stage.

Units (SI throughout)
---------------------
    radius      : m
    draft       : m
    water_depth : m
    mass        : kg
    rho_water   : kg/m³
    volume      : m³
    area        : m²

References
----------
- Falnes (2002), Ocean Waves and Oscillating Systems, Cambridge University Press.
- Newman (1977), Marine Hydrodynamics, MIT Press.
"""

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

#: Standard gravitational acceleration [m/s²]
GRAVITY: float = 9.81

#: Default seawater density [kg/m³]
#: Typical open-ocean value at ~20 °C and 35 ppt salinity.
#: The Goa nearshore value may differ slightly; use rho_water parameter to override.
RHO_SEAWATER: float = 1025.0

#: Valid range for water density [kg/m³].
#: Fresh water ≈ 998 kg/m³; dense brine ≈ 1100 kg/m³.
_RHO_MIN: float = 900.0
_RHO_MAX: float = 1100.0


# ---------------------------------------------------------------------------
# CylinderGeometry
# ---------------------------------------------------------------------------

@dataclass
class CylinderGeometry:
    """
    Geometry of a vertical circular cylinder heaving point-absorber.

    The cylinder is assumed to be:
    - Vertical axis aligned with z (heave direction).
    - Uniform cross-section (constant radius from keel to deck).
    - Freely floating at the specified draft in static equilibrium.

    This is a TEMPORARY DEVELOPMENT GEOMETRY for algorithm testing.
    It is NOT the final Goa deployment buoy.

    Parameters
    ----------
    radius : float
        Cylinder radius [m].  Must be > 0.
    draft : float
        Submerged depth at static equilibrium [m].  Must be > 0.
        The keel is at z = -draft; the waterplane is at z = 0.
    water_depth : float
        Total water column depth [m].  Must be > draft (buoy must not touch
        the seabed) and > 0.
    mass : float
        Total buoy mass [kg].  Must be > 0.
        At static equilibrium, mass = rho_water * displaced_volume
        (Archimedes).  The constructor does NOT enforce this automatically
        because the user may specify mass independently; use
        Hydrostatics.check_archimedes() to verify equilibrium.
    rho_water : float, optional
        Water density [kg/m³].  Default 1025.0 (open-ocean seawater).
        Must be in [{_rho_min}, {_rho_max}] kg/m³.

    Attributes (derived, read-only)
    --------------------------------
    displaced_volume : float
        Volume of water displaced at the given draft [m³].
        V = π r² d
    waterplane_area : float
        Area of the waterplane cross-section [m²].
        A_wp = π r²
        For a vertical cylinder, this equals the cross-sectional area
        at any depth.

    Raises
    ------
    ValueError
        If any parameter is outside its valid range.

    Examples
    --------
    >>> geom = CylinderGeometry(radius=5.0, draft=4.0, water_depth=50.0, mass=314_159.0)
    >>> geom.displaced_volume
    314.1592653589793
    >>> geom.waterplane_area
    78.53981633974483
    """

    radius: float
    draft: float
    water_depth: float
    mass: float
    rho_water: float = RHO_SEAWATER

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.radius <= 0.0:
            raise ValueError(f"radius must be positive, got {self.radius} m")
        if self.draft <= 0.0:
            raise ValueError(f"draft must be positive, got {self.draft} m")
        if self.water_depth <= 0.0:
            raise ValueError(f"water_depth must be positive, got {self.water_depth} m")
        if self.water_depth <= self.draft:
            raise ValueError(
                f"water_depth ({self.water_depth} m) must exceed draft ({self.draft} m); "
                "the buoy must not touch the seabed"
            )
        if self.mass <= 0.0:
            raise ValueError(f"mass must be positive, got {self.mass} kg")
        if not (_RHO_MIN <= self.rho_water <= _RHO_MAX):
            raise ValueError(
                f"rho_water must be in [{_RHO_MIN}, {_RHO_MAX}] kg/m³, "
                f"got {self.rho_water} kg/m³"
            )

    @property
    def displaced_volume(self) -> float:
        """
        Volume of water displaced at the given draft [m³].

        V = π r² d

        This is the volume of the submerged portion of the cylinder
        (from keel at z = -draft to waterplane at z = 0).
        """
        return math.pi * self.radius**2 * self.draft

    @property
    def waterplane_area(self) -> float:
        """
        Area of the waterplane cross-section [m²].

        A_wp = π r²

        For a vertical cylinder, the waterplane area is constant and
        independent of draft.  This quantity enters the hydrostatic
        stiffness: C = ρ g A_wp.
        """
        return math.pi * self.radius**2


# ---------------------------------------------------------------------------
# Default temporary development buoy
# ---------------------------------------------------------------------------

#: Temporary reference buoy for development and testing.
#:
#: Parameters chosen to give a heave natural period of ~6 s (without added
#: mass), which falls within the ocean swell range (5–20 s).
#:
#: For a neutrally buoyant vertical cylinder:
#:   T_n = 2π √(d / g)   (added mass = 0)
#: With draft = 10 m:  T_n = 2π √(10/9.81) ≈ 6.34 s
#:
#: Geometry:
#:   radius      = 5.0 m   → waterplane area ≈ 78.5 m²
#:   draft       = 10.0 m  → displaced volume ≈ 785.4 m³
#:   water_depth = 50.0 m  → finite-depth effects are moderate (kd ≈ 3 at fp)
#:   mass        = rho * V = 1025 * 785.4 ≈ 805 033 kg  (neutrally buoyant)
#:
#: This is NOT the Goa deployment buoy.  All parameters are placeholders.
REFERENCE_BUOY: CylinderGeometry = CylinderGeometry(
    radius=5.0,
    draft=10.0,
    water_depth=50.0,
    mass=1025.0 * math.pi * 5.0**2 * 10.0,  # neutrally buoyant: m = rho * V
    rho_water=1025.0,
)
