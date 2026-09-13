"""
hydrostatics.py — Hydrostatic equilibrium quantities for a heaving WEC (Module 2).

Computes the static hydrostatic properties of a floating cylinder from its
geometry.  These quantities are inputs to the Cummins equation (future
milestone) and define the restoring force in heave.

Physical model
--------------
For a freely floating body in static equilibrium, Archimedes' principle gives:

    F_buoyancy = ρ g V_displaced = m g

where V_displaced is the submerged volume at the equilibrium draft.

When the body is displaced vertically by a small heave z (positive upward),
the change in displaced volume is A_wp * z, giving a restoring force:

    F_restoring = -C * z,   C = ρ g A_wp   [N/m]

This is the linear hydrostatic stiffness.  It is valid for small heave
amplitudes (|z| ≪ draft) where the waterplane area remains approximately
constant.

Heave natural frequency (undamped, no PTO)
------------------------------------------
In the absence of radiation damping and PTO, the heave equation of motion
reduces to:

    (m + m_a) * z̈ + C * z = 0

where m_a is the added mass at the natural frequency.  Since m_a is not
yet available (requires Capytaine BEM), the undamped natural frequency
computed here uses m_a = 0:

    ω_n = √(C / m)   [rad/s]
    T_n = 2π / ω_n   [s]

This is a LOWER BOUND on the true natural period (added mass increases the
effective inertia, lengthening the period).  The true value will be
computed in the hydrodynamics milestone.

Units (SI throughout)
---------------------
    C       : N/m   (hydrostatic stiffness)
    F_b     : N     (buoyancy force)
    ω_n     : rad/s (natural angular frequency)
    T_n     : s     (natural period)
    m       : kg
    ρ       : kg/m³
    g       : m/s²
    V       : m³
    A_wp    : m²

References
----------
- Falnes (2002), Ocean Waves and Oscillating Systems, §2.2.
- Newman (1977), Marine Hydrodynamics, §6.
- Faltinsen (1990), Sea Loads on Ships and Offshore Structures, §3.2.
"""

import math
from dataclasses import dataclass

from module2_wec.geometry import CylinderGeometry, GRAVITY


@dataclass(frozen=True)
class Hydrostatics:
    """
    Hydrostatic equilibrium quantities for a floating cylinder.

    All quantities are computed analytically from the geometry at
    construction time.  The object is immutable (frozen dataclass).

    Parameters
    ----------
    geometry : CylinderGeometry
        Validated buoy geometry.

    Attributes
    ----------
    displaced_volume : float
        Volume of water displaced at equilibrium draft [m³].
        V = π r² d
    buoyancy_force : float
        Upward buoyancy force at equilibrium [N].
        F_b = ρ g V
    waterplane_area : float
        Waterplane cross-sectional area [m²].
        A_wp = π r²
    hydrostatic_stiffness : float
        Linear hydrostatic restoring stiffness in heave [N/m].
        C = ρ g A_wp
        Restoring force: F_restoring = -C * z  (z positive upward)
    omega_n : float
        Undamped heave natural angular frequency [rad/s], with added mass = 0.
        ω_n = √(C / m)
        NOTE: This is a lower bound; true ω_n is smaller due to added mass.
    natural_period : float
        Undamped heave natural period [s], with added mass = 0.
        T_n = 2π / ω_n
        NOTE: This is an upper bound on the true natural period.

    Raises
    ------
    ValueError
        If the geometry is invalid (propagated from CylinderGeometry).
    """

    geometry: CylinderGeometry

    @property
    def displaced_volume(self) -> float:
        """Displaced volume at equilibrium draft [m³].  V = π r² d."""
        return self.geometry.displaced_volume

    @property
    def buoyancy_force(self) -> float:
        """
        Upward buoyancy force at equilibrium [N].

        F_b = ρ g V = ρ g π r² d

        At static equilibrium this equals the weight: F_b = m g.
        """
        return self.geometry.rho_water * GRAVITY * self.displaced_volume

    @property
    def waterplane_area(self) -> float:
        """Waterplane area [m²].  A_wp = π r²."""
        return self.geometry.waterplane_area

    @property
    def hydrostatic_stiffness(self) -> float:
        """
        Linear hydrostatic restoring stiffness in heave [N/m].

        C = ρ g A_wp

        Derived from the change in buoyancy force per unit heave displacement:
            dF_b/dz = ρ g A_wp  (for small z, constant waterplane area)

        The restoring force is F_restoring = -C * z (negative for upward z,
        restoring toward equilibrium).
        """
        return self.geometry.rho_water * GRAVITY * self.waterplane_area

    @property
    def omega_n(self) -> float:
        """
        Undamped heave natural angular frequency [rad/s], added mass = 0.

        ω_n = √(C / m)

        This uses only the buoy mass m (no added mass).  The true natural
        frequency is lower because added mass m_a > 0 increases effective
        inertia:  ω_n_true = √(C / (m + m_a)).

        Capytaine BEM will provide m_a in the next milestone.
        """
        return math.sqrt(self.hydrostatic_stiffness / self.geometry.mass)

    @property
    def natural_period(self) -> float:
        """
        Undamped heave natural period [s], added mass = 0.

        T_n = 2π / ω_n

        See omega_n docstring for the added-mass caveat.
        """
        return 2.0 * math.pi / self.omega_n

    def check_archimedes(self, rtol: float = 1e-6) -> bool:
        """
        Check whether the buoy mass satisfies Archimedes' principle.

        At static equilibrium: m = ρ V_displaced, i.e. weight = buoyancy.

        Parameters
        ----------
        rtol : float
            Relative tolerance for the check.  Default 1e-6.
            Justification: floating-point arithmetic on double-precision
            values introduces errors of order 1e-15; 1e-6 is generous
            enough to pass for analytically consistent inputs while
            catching genuine physical inconsistencies.

        Returns
        -------
        bool
            True if |m - ρV| / (ρV) ≤ rtol.
        """
        m = self.geometry.mass
        rho_V = self.geometry.rho_water * self.displaced_volume
        return abs(m - rho_V) / rho_V <= rtol

    def summary(self) -> dict:
        """
        Return all hydrostatic quantities as a dictionary.

        Keys and units
        --------------
        displaced_volume_m3     : m³
        buoyancy_force_N        : N
        waterplane_area_m2      : m²
        hydrostatic_stiffness_N_per_m : N/m
        omega_n_rad_per_s       : rad/s
        natural_period_s        : s
        archimedes_satisfied    : bool
        """
        return {
            "displaced_volume_m3":          self.displaced_volume,
            "buoyancy_force_N":             self.buoyancy_force,
            "waterplane_area_m2":           self.waterplane_area,
            "hydrostatic_stiffness_N_per_m": self.hydrostatic_stiffness,
            "omega_n_rad_per_s":            self.omega_n,
            "natural_period_s":             self.natural_period,
            "archimedes_satisfied":         self.check_archimedes(),
        }
