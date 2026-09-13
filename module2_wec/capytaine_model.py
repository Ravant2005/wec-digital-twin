"""
capytaine_model.py — Capytaine floating-body model for the WEC Digital Twin (Module 2).

Builds the Capytaine FloatingBody from a CylinderGeometry and runs the
frequency-domain BEM (radiation + diffraction) to produce hydrodynamic
coefficients.

Physical model
--------------
A vertical circular cylinder is meshed and clipped to its immersed part
(z ≤ 0).  Only the HEAVE degree of freedom is activated.  The BEM solves:

    Radiation problem  : body oscillates in heave, no incident wave.
    Diffraction problem: body fixed, incident Airy wave from wave_direction.

The excitation force is assembled as:
    F_exc = F_diffraction + F_Froude-Krylov

Sign convention (Capytaine 3.x) — empirically verified
-------------------------------------------------------
Capytaine 3.x computes the radiation force as:

    F_rad = integral(p * n_z dS),   p = i ω ρ φ_rad

For a body oscillating with unit heave amplitude, the standard decomposition
(Falnes 2002, §5.2) defines A and B via:

    F_rad = ω² A - i ω B_physical

Capytaine extracts:
    A_cap     = Re(F_rad) / ω²
    B_cap     = Im(F_rad) / ω

Empirical check at ω = 1.0 rad/s (672-panel mesh, depth = 50 m,
fresh body per solve — matches production code path):
    F_rad = 445761 - 74076j  N
    A_cap = Re(F_rad)/ω² = +445761 kg   (positive ✓)
    B_cap = Im(F_rad)/ω  = -74076 N·s/m (negative)

Note on body reuse: reusing a single Capytaine FloatingBody across multiple
sequential solves (different ω) can produce ~1% different results compared
to building a fresh body per solve.  The production code path (CapytaineModel.run)
builds one body and passes it to solve_all() in a single batch; the values
above are from a single-frequency fresh-body solve (audit Task 5 path).  Both
paths confirm B_cap < 0; the sign convention conclusion is unaffected.

Production-path reproducibility (verified, 672-panel mesh, 20 frequencies):
    Repeated calls to compute_hydrodynamic_coefficients() with identical inputs
    agree to within 6×10⁻⁵ relative (worst case: radiation damping).  Added
    mass and excitation amplitude agree to within 4×10⁻⁶ relative.  The
    production solve_all() path is deterministic at the engineering level.
    Fresh-body and reused-body paths can differ by approximately 1–2% because
    of internal body state/caching, but the production solve_all() path is
    used consistently for all production coefficients.

Comparing with the standard decomposition:
    Re(ω²A - iωB) = ω²A  →  A_cap = A  ✓
    Im(ω²A - iωB) = -ωB  →  B_cap = -B_physical

Therefore:  B_physical = -B_capytaine > 0  (positive for energy dissipation)

This is consistent with Capytaine's own RAO transfer function source:
    H = -ω²(M+A) - iω B_cap + C
    Im(-iω B_cap) = -ω B_cap = +ω B_physical > 0  (passive damping ✓)

Conclusion: our negation  B_physical = -B_capytaine  is CORRECT.

Waterplane area from Capytaine mesh
------------------------------------
mesh_vertical_cylinder generates a CLOSED mesh (hull + top cap + bottom cap).
Capytaine's waterplane_area uses the divergence theorem:
    A_wp = -∫ n_z dS
For a closed mesh, ∫ n_z dS = 0 (top cap cancels bottom cap), so A_wp ≈ 0.
This is a known limitation of the closed-mesh approach.
Module 2.1's analytical value C33 = ρ g π r² is the authoritative benchmark.
The BEM force integration is NOT affected: Capytaine uses hull_mask = all faces
for a hull-only body, which correctly includes the top-cap pressure.

Irregular frequencies
---------------------
Capytaine's panel BEM has irregular (spurious) frequencies where the interior
Dirichlet problem has a non-trivial solution.  For this cylinder geometry
(r = 5 m, draft = 10 m) the first irregular frequency is estimated at
ω_irr ≈ 2.09 rad/s (Capytaine's parallelepiped formula, independent of
mesh resolution for this geometry).  The default frequency grid is capped
at ω_max = 1.4 rad/s (67 % of ω_irr) to provide a 33 % safety margin.

Lid mesh note
-------------
A lid mesh suppresses irregular frequencies but also changes the BEM
formulation at ALL frequencies, not just near ω_irr.  Audit results
(hull-only vs hull+lid, 672-panel mesh):

    ω = 0.8 rad/s:  ΔA =  5.5 %,  ΔB =  23.5 %
    ω = 1.0 rad/s:  ΔA = 12.7 %,  ΔB =  34.2 %
    ω = 1.2 rad/s:  ΔA = 29.1 %,  ΔB =  60.0 %
    ω = 1.4 rad/s:  ΔA = 90.4 %,  ΔB = 201.2 %

The lid is NOT a drop-in correction; it substantially changes the BEM
formulation across the entire production range.  Hull-only is retained
for production (ω ≤ 1.4 rad/s).  Extension toward or above the
irregular-frequency region requires independent validation against an
analytical or published BEM reference before a lid can be adopted.

Mesh resolution and convergence
--------------------------------
Default resolution (6, 24, 16) → 672 wetted panels.
Panel sizes: ~1.31 m (circumferential), ~0.62 m (vertical).
At ω = 1.4 rad/s, λ_deep ≈ 31.8 m → panel size ≈ λ/24 (well resolved).
Richardson extrapolation (672 vs 1152 panels, assuming first-order convergence
p = 1) gives conditional estimates of remaining errors at ω = 1.0 rad/s:
    Added mass A:        ~3 % (estimated relative to Richardson extrapolant)
    Radiation damping B: ~20 % (estimated relative to Richardson extrapolant)
These are NOT measured ground-truth errors; they are first-order Richardson
estimates and depend on the assumed convergence order p = 1.
The 672-panel mesh is NOT fully converged for radiation damping.
Further refinement is recommended before production deployment.

Units (SI throughout)
---------------------
    omega           : rad/s
    added_mass      : kg
    radiation_damping: N·s/m
    excitation_force: N  (complex, per unit wave amplitude [m])
    water_depth     : m
    rho             : kg/m³

References
----------
- Falnes (2002), Ocean Waves and Oscillating Systems, Cambridge University Press.
- Capytaine documentation: https://capytaine.github.io/
- Ancellin & Dias (2019), JOSS, doi:10.21105/joss.01341.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import capytaine as cpt
import numpy as np

from module2_wec.geometry import CylinderGeometry, GRAVITY

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default BEM parameters
# ---------------------------------------------------------------------------

#: Default mesh resolution (nr, ntheta, nz).
#: Total panels = (2*nr + nz) * ntheta = (12 + 16) * 24 = 672.
DEFAULT_RESOLUTION: tuple = (6, 24, 16)

#: Default frequency grid [rad/s]: 20 points from 0.2 to 1.4 rad/s.
#: Upper limit is 67 % of the first irregular frequency (~2.09 rad/s)
#: for the default mesh, providing a safety margin.
DEFAULT_OMEGA_MIN: float = 0.2   # rad/s
DEFAULT_OMEGA_MAX: float = 1.4   # rad/s
DEFAULT_N_OMEGA: int = 20

#: Default incident wave direction [rad].  0 = waves propagating toward North.
DEFAULT_WAVE_DIRECTION: float = 0.0


# ---------------------------------------------------------------------------
# CapytaineModel
# ---------------------------------------------------------------------------

@dataclass
class CapytaineModel:
    """
    Capytaine BEM model for a heaving cylindrical WEC.

    Parameters
    ----------
    geometry : CylinderGeometry
        Validated buoy geometry (from Module 2.1).
    resolution : tuple of 3 ints, optional
        Mesh resolution (nr, ntheta, nz) for mesh_vertical_cylinder.
        Total panels = (2*nr + nz) * ntheta.
        Default (6, 24, 16) → 672 panels.
    omega_min : float, optional
        Minimum angular frequency [rad/s].  Default 0.2.
    omega_max : float, optional
        Maximum angular frequency [rad/s].  Default 1.4.
        Must be below the first irregular frequency of the mesh.
    n_omega : int, optional
        Number of frequency points.  Default 20.
    wave_direction : float, optional
        Incident wave direction [rad].  Default 0.0 (toward North).

    Raises
    ------
    ValueError
        If omega_min <= 0, omega_max <= omega_min, or n_omega < 2.
    """

    geometry: CylinderGeometry
    resolution: tuple = DEFAULT_RESOLUTION
    omega_min: float = DEFAULT_OMEGA_MIN
    omega_max: float = DEFAULT_OMEGA_MAX
    n_omega: int = DEFAULT_N_OMEGA
    wave_direction: float = DEFAULT_WAVE_DIRECTION

    def __post_init__(self) -> None:
        if self.omega_min <= 0.0:
            raise ValueError(f"omega_min must be positive, got {self.omega_min}")
        if self.omega_max <= self.omega_min:
            raise ValueError(
                f"omega_max ({self.omega_max}) must exceed omega_min ({self.omega_min})"
            )
        if self.n_omega < 2:
            raise ValueError(f"n_omega must be >= 2, got {self.n_omega}")

    @property
    def omega_grid(self) -> np.ndarray:
        """Angular frequency grid [rad/s], shape (n_omega,)."""
        return np.linspace(self.omega_min, self.omega_max, self.n_omega)

    def build_floating_body(self) -> cpt.FloatingBody:
        """
        Build and return the immersed Capytaine FloatingBody.

        The cylinder is meshed with the specified resolution, clipped to
        its immersed part (z ≤ 0), and given a single Heave DOF.

        The mesh center is placed at (0, 0, -draft/2) so that the top face
        is at z = 0 (waterplane) and the bottom face is at z = -draft.

        Returns
        -------
        cpt.FloatingBody
            Immersed body with Heave DOF.  Ready for BEM problems.
        """
        g = self.geometry
        center_z = -g.draft / 2.0

        mesh = cpt.mesh_vertical_cylinder(
            length=g.draft,
            radius=g.radius,
            center=(0.0, 0.0, center_z),
            resolution=self.resolution,
        )
        body = cpt.FloatingBody(mesh=mesh, name="reference_cylinder")
        body.add_translation_dof(name="Heave")
        immersed = body.immersed_part()
        logger.debug(
            "Built FloatingBody: %d panels, draft=%.1f m, radius=%.1f m",
            immersed.mesh.nb_faces,
            g.draft,
            g.radius,
        )
        return immersed

    def n_panels(self) -> int:
        """Return the number of wetted panels for the current resolution."""
        nr, ntheta, nz = self.resolution
        return (2 * nr + nz) * ntheta

    def irregular_frequency_estimate(self) -> float:
        """
        Estimate the first irregular frequency [rad/s] for the current mesh.

        Irregular frequencies are spurious resonances of the interior
        Dirichlet problem.  Results near this frequency are unreliable.
        """
        body = self.build_floating_body()
        return body.first_irregular_frequency_estimate()

    def run(self, progress_bar: bool = False) -> "cpt.Dataset":
        """
        Run the BEM and return an xarray Dataset of hydrodynamic coefficients.

        Solves radiation and diffraction problems at all frequencies in
        omega_grid.  Returns the raw Capytaine dataset (use
        HydrodynamicCoefficients.from_capytaine_dataset() to get the
        processed result).

        Parameters
        ----------
        progress_bar : bool, optional
            Show Capytaine progress bar.  Default False.

        Returns
        -------
        xarray.Dataset
            Capytaine dataset with variables: added_mass, radiation_damping,
            excitation_force, diffraction_force, Froude_Krylov_force.
            Coordinates include omega, freq, water_depth, rho.
        """
        body = self.build_floating_body()
        solver = cpt.BEMSolver()
        omegas = self.omega_grid
        g = self.geometry

        problems = []
        for w in omegas:
            problems.append(cpt.RadiationProblem(
                body=body,
                omega=float(w),
                water_depth=g.water_depth,
                rho=g.rho_water,
                radiating_dof="Heave",
            ))
            problems.append(cpt.DiffractionProblem(
                body=body,
                omega=float(w),
                water_depth=g.water_depth,
                rho=g.rho_water,
                wave_direction=self.wave_direction,
            ))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = solver.solve_all(problems, progress_bar=progress_bar)

        ds = cpt.assemble_dataset(results, hydrostatics=False)
        logger.info(
            "BEM complete: %d frequencies, %d panels, depth=%.1f m",
            len(omegas),
            self.n_panels(),
            g.water_depth,
        )
        return ds
