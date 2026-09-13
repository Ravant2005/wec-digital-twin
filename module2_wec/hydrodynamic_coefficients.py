"""
hydrodynamic_coefficients.py — Processed hydrodynamic coefficients for the WEC (Module 2).

Extracts, validates, and stores the frequency-domain hydrodynamic coefficients
produced by Capytaine BEM.  Provides a clean, typed interface for downstream
modules (Cummins equation, control, etc.).

Physical quantities stored
--------------------------
All quantities are for the HEAVE degree of freedom only.

    omega [rad/s]
        Angular frequency grid.  Strictly increasing.

    frequency_hz [Hz]
        f = omega / (2π).  Derived from omega.

    added_mass_heave [kg]
        Frequency-dependent added mass A(ω).
        Physical interpretation: inertia of entrained water.
        A(ω) ≥ 0 for a physically valid mesh away from irregular frequencies.

    radiation_damping_heave [N·s/m]
        Physical radiation damping B(ω) = -B_capytaine(ω).
        B(ω) ≥ 0 for a physically valid mesh away from irregular frequencies.
        Sign convention: B_physical > 0 means energy is radiated away.
        NOTE: Capytaine stores -B_physical internally (see capytaine_model.py).

    excitation_force_heave_real [N/m]
        Real part of the complex excitation force per unit wave amplitude.

    excitation_force_heave_imag [N/m]
        Imaginary part of the complex excitation force per unit wave amplitude.

    excitation_force_heave_amplitude [N/m]
        |F_exc(ω)| = sqrt(real² + imag²).

    excitation_force_heave_phase [rad]
        arg(F_exc(ω)) = atan2(imag, real).  In (-π, π].

Metadata stored
---------------
    water_depth [m], rho [kg/m³], radius [m], draft [m],
    n_panels, capytaine_version, omega_min, omega_max, n_omega.

Units
-----
    All SI.  Forces are per unit wave amplitude [m], so units are N/m → N
    when multiplied by wave amplitude.  Stored as N for unit-amplitude waves.

References
----------
- Falnes (2002), Ocean Waves and Oscillating Systems, §5.
- Newman (1977), Marine Hydrodynamics, §6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import capytaine as cpt
import numpy as np

from module2_wec.geometry import CylinderGeometry
from module2_wec.capytaine_model import CapytaineModel


# ---------------------------------------------------------------------------
# HydrodynamicCoefficients
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HydrodynamicCoefficients:
    """
    Frequency-domain hydrodynamic coefficients for a heaving cylinder.

    Immutable (frozen dataclass).  All arrays have shape (n_omega,).

    Parameters
    ----------
    omega : np.ndarray
        Angular frequency grid [rad/s].  Strictly increasing.
    frequency_hz : np.ndarray
        Frequency [Hz] = omega / (2π).
    added_mass_heave : np.ndarray
        Added mass A(ω) [kg].
    radiation_damping_heave : np.ndarray
        Physical radiation damping B(ω) [N·s/m].  Positive for dissipation.
    excitation_force_heave_real : np.ndarray
        Re(F_exc(ω)) [N/m].
    excitation_force_heave_imag : np.ndarray
        Im(F_exc(ω)) [N/m].
    excitation_force_heave_amplitude : np.ndarray
        |F_exc(ω)| [N/m].
    excitation_force_heave_phase : np.ndarray
        arg(F_exc(ω)) [rad].  In (-π, π].
    water_depth : float
        Water depth used in BEM [m].
    rho : float
        Water density used in BEM [kg/m³].
    radius : float
        Cylinder radius [m].
    draft : float
        Cylinder draft [m].
    n_panels : int
        Number of wetted BEM panels.
    capytaine_version : str
        Capytaine version string.
    omega_min : float
        Minimum angular frequency [rad/s].
    omega_max : float
        Maximum angular frequency [rad/s].
    """

    omega: np.ndarray
    frequency_hz: np.ndarray
    added_mass_heave: np.ndarray
    radiation_damping_heave: np.ndarray
    excitation_force_heave_real: np.ndarray
    excitation_force_heave_imag: np.ndarray
    excitation_force_heave_amplitude: np.ndarray
    excitation_force_heave_phase: np.ndarray
    water_depth: float
    rho: float
    radius: float
    draft: float
    n_panels: int
    capytaine_version: str
    omega_min: float
    omega_max: float

    @property
    def n_omega(self) -> int:
        """Number of frequency points."""
        return len(self.omega)

    @property
    def excitation_force_heave_complex(self) -> np.ndarray:
        """
        Complex excitation force F_exc(ω) [N/m].

        Reconstructed from stored real and imaginary parts.
        Equivalent to excitation_force_heave_real + 1j * excitation_force_heave_imag.
        """
        return self.excitation_force_heave_real + 1j * self.excitation_force_heave_imag

    @classmethod
    def from_capytaine_dataset(
        cls,
        ds,
        geometry: CylinderGeometry,
        n_panels: int,
    ) -> "HydrodynamicCoefficients":
        """
        Construct from a raw Capytaine xarray Dataset.

        Extracts the Heave-Heave components, applies the sign correction
        for radiation damping (B_physical = -B_capytaine), and computes
        amplitude and phase of the excitation force.

        Parameters
        ----------
        ds : xarray.Dataset
            Output of cpt.assemble_dataset().
        geometry : CylinderGeometry
            The geometry used to run the BEM.
        n_panels : int
            Number of wetted panels used.

        Returns
        -------
        HydrodynamicCoefficients
        """
        omega = ds.coords["omega"].values.astype(float)

        # Added mass [kg]: Re(F_rad) / omega^2.  Positive for physical mesh.
        A = ds["added_mass"].sel(
            influenced_dof="Heave", radiating_dof="Heave"
        ).values.astype(float)

        # Radiation damping sign convention (Capytaine 3.x, empirically verified):
        #
        #   Capytaine computes: B_cap = Im(F_rad) / omega
        #   Standard decomposition (Falnes 2002): F_rad = omega^2*A - i*omega*B_physical
        #   => Im(F_rad) = -omega * B_physical
        #   => B_cap = Im(F_rad)/omega = -B_physical
        #
        #   Empirical check at omega=1.0 rad/s (672-panel mesh):
        #     F_rad = 445761 - 74076j N
        #     B_cap = Im(F_rad)/omega = -74076 N*s/m  (negative)
        #     B_physical = -B_cap = +74076 N*s/m      (positive, correct)
        #
        #   Consistent with Capytaine's RAO: H = -omega^2*(M+A) - i*omega*B_cap + C
        #     Im(-i*omega*B_cap) = -omega*B_cap = +omega*B_physical > 0  (passive damping)
        #
        B_cap = ds["radiation_damping"].sel(
            influenced_dof="Heave", radiating_dof="Heave"
        ).values.astype(float)
        B_phys = -B_cap   # B_physical > 0 for energy dissipation

        # Excitation force: complex, shape (n_omega, n_wave_direction, n_influenced_dof)
        # Select wave_direction=0, influenced_dof=Heave
        F_exc_complex = ds["excitation_force"].sel(
            influenced_dof="Heave", wave_direction=0.0
        ).values.astype(complex)

        F_real = np.real(F_exc_complex)
        F_imag = np.imag(F_exc_complex)
        F_amp = np.abs(F_exc_complex)
        F_phase = np.angle(F_exc_complex)  # atan2(imag, real), in (-pi, pi]

        freq_hz = omega / (2.0 * np.pi)

        return cls(
            omega=omega,
            frequency_hz=freq_hz,
            added_mass_heave=A,
            radiation_damping_heave=B_phys,
            excitation_force_heave_real=F_real,
            excitation_force_heave_imag=F_imag,
            excitation_force_heave_amplitude=F_amp,
            excitation_force_heave_phase=F_phase,
            water_depth=float(ds.coords["water_depth"].values),
            rho=float(ds.coords["rho"].values),
            radius=geometry.radius,
            draft=geometry.draft,
            n_panels=n_panels,
            capytaine_version=cpt.__version__,
            omega_min=float(omega.min()),
            omega_max=float(omega.max()),
        )

    def summary(self) -> dict:
        """
        Return a text-friendly summary dictionary of key metadata and ranges.

        Returns
        -------
        dict
            Keys: capytaine_version, n_panels, n_omega, omega_min_rad_s,
            omega_max_rad_s, water_depth_m, rho_kg_m3, radius_m, draft_m,
            added_mass_min_kg, added_mass_max_kg,
            radiation_damping_min_N_s_m, radiation_damping_max_N_s_m,
            excitation_amplitude_min_N, excitation_amplitude_max_N.
        """
        return {
            "capytaine_version":             self.capytaine_version,
            "n_panels":                      self.n_panels,
            "n_omega":                       self.n_omega,
            "omega_min_rad_s":               self.omega_min,
            "omega_max_rad_s":               self.omega_max,
            "water_depth_m":                 self.water_depth,
            "rho_kg_m3":                     self.rho,
            "radius_m":                      self.radius,
            "draft_m":                       self.draft,
            "added_mass_min_kg":             float(self.added_mass_heave.min()),
            "added_mass_max_kg":             float(self.added_mass_heave.max()),
            "radiation_damping_min_N_s_m":   float(self.radiation_damping_heave.min()),
            "radiation_damping_max_N_s_m":   float(self.radiation_damping_heave.max()),
            "excitation_amplitude_min_N":    float(self.excitation_force_heave_amplitude.min()),
            "excitation_amplitude_max_N":    float(self.excitation_force_heave_amplitude.max()),
        }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def compute_hydrodynamic_coefficients(
    geometry: CylinderGeometry,
    resolution: tuple = (6, 24, 16),
    omega_min: float = 0.2,
    omega_max: float = 1.4,
    n_omega: int = 20,
    wave_direction: float = 0.0,
    progress_bar: bool = False,
) -> HydrodynamicCoefficients:
    """
    Build the Capytaine model, run BEM, and return processed coefficients.

    This is the primary entry point for computing hydrodynamic coefficients.

    Parameters
    ----------
    geometry : CylinderGeometry
        Validated buoy geometry.
    resolution : tuple of 3 ints, optional
        Mesh resolution (nr, ntheta, nz).  Default (6, 24, 16) → 672 panels.
    omega_min : float, optional
        Minimum angular frequency [rad/s].  Default 0.2.
    omega_max : float, optional
        Maximum angular frequency [rad/s].  Default 1.4.
    n_omega : int, optional
        Number of frequency points.  Default 20.
    wave_direction : float, optional
        Incident wave direction [rad].  Default 0.0.
    progress_bar : bool, optional
        Show Capytaine progress bar.  Default False.

    Returns
    -------
    HydrodynamicCoefficients
        Processed, validated hydrodynamic coefficients.
    """
    model = CapytaineModel(
        geometry=geometry,
        resolution=resolution,
        omega_min=omega_min,
        omega_max=omega_max,
        n_omega=n_omega,
        wave_direction=wave_direction,
    )
    ds = model.run(progress_bar=progress_bar)
    return HydrodynamicCoefficients.from_capytaine_dataset(
        ds, geometry, model.n_panels()
    )
