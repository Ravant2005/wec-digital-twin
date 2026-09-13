"""
excitation.py — Time-domain wave excitation force (Module 2.3B).

Converts Module 2.2 frequency-domain excitation coefficients and Module 1
wave components into a time-domain heave excitation force using linear
superposition.

============================================================
PHYSICS
============================================================

For a regular wave component:

    eta_j(t) = a_j * cos(omega_j * t + phi_j)

and Capytaine's complex excitation coefficient (per unit wave amplitude):

    F_exc(omega_j) = F_real(omega_j) + i * F_imag(omega_j)   [N/m]

the time-domain excitation force contribution is:

    F_j(t) = a_j * |F_exc(omega_j)| * cos(omega_j*t + phi_j + angle(F_exc(omega_j)))

This follows directly from linear wave theory (Falnes 2002, §5):

    F_j(t) = Re[ a_j * F_exc(omega_j) * exp(i*(omega_j*t + phi_j)) ]
           = a_j * Re[ F_exc(omega_j) * exp(i*(omega_j*t + phi_j)) ]
           = a_j * |F_exc| * cos(omega_j*t + phi_j + angle(F_exc))

For N wave components, linear superposition gives:

    F_exc(t) = sum_{j=1}^{N} F_j(t)

============================================================
UNITS
============================================================

    Capytaine F_exc(omega)  : N/m   (force per unit wave amplitude)
    Wave amplitude a_j      : m
    Product a_j * F_exc     : N     (physical force)

    omega                   : rad/s
    t                       : s
    phi_j                   : rad
    angle(F_exc)            : rad

============================================================
INTERPOLATION
============================================================

Module 2.2 provides F_exc at a discrete omega grid (20 points, 0.2–1.4 rad/s).
For arbitrary omega_j within this range, the complex excitation coefficient is
interpolated by separately interpolating the real and imaginary parts using
linear interpolation (numpy.interp).

Rationale for interpolating real/imag rather than amplitude/phase:
- The real and imaginary parts are smooth, monotone functions of omega for
  this dataset.  Linear interpolation introduces negligible error.
- Phase interpolation requires unwrapping to avoid discontinuities at ±pi.
  Interpolating real/imag avoids this entirely.
- Amplitude and phase are derived from the interpolated complex value:
    |F_interp| = abs(F_real_interp + i*F_imag_interp)
    angle_interp = angle(F_real_interp + i*F_imag_interp)

Frequency range enforcement:
- Requesting omega outside [omega_min, omega_max] raises ValueError.
- No silent extrapolation is performed.

============================================================
REFERENCES
============================================================
- Falnes (2002), Ocean Waves and Oscillating Systems, Cambridge Univ. Press.
- Cummins (1962), The impulse response function and ship motions.
- Newman (1977), Marine Hydrodynamics, MIT Press.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from module2_wec.hydrodynamic_coefficients import HydrodynamicCoefficients


# ---------------------------------------------------------------------------
# ExcitationSignal dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ExcitationSignal:
    """
    Time-domain wave excitation force for a heaving cylinder.

    Produced by compute_excitation_force().  Immutable (frozen dataclass).

    Parameters
    ----------
    time : np.ndarray
        Time grid [s], shape (n_time,).
    force : np.ndarray
        Heave excitation force F_exc(t) [N], shape (n_time,).
        F_exc(t) = sum_j a_j * |F_exc(omega_j)| * cos(omega_j*t + phi_j + angle_j)
    wave_elevation : np.ndarray
        Wave surface elevation eta(t) [m], shape (n_time,).
        eta(t) = sum_j a_j * cos(omega_j*t + phi_j)
    omega : np.ndarray
        Angular frequencies of wave components [rad/s], shape (n_components,).
    wave_amplitude : np.ndarray
        Wave component amplitudes a_j [m], shape (n_components,).
    wave_phase : np.ndarray
        Wave component phases phi_j [rad], shape (n_components,).
    excitation_amplitude : np.ndarray
        |F_exc(omega_j)| interpolated at each component [N/m],
        shape (n_components,).
    excitation_phase : np.ndarray
        angle(F_exc(omega_j)) interpolated at each component [rad],
        shape (n_components,).
    wave_direction : float
        Wave propagation direction [deg].  0 = head-on.
    source_omega_min : float
        Minimum omega of the BEM frequency grid [rad/s].
    source_omega_max : float
        Maximum omega of the BEM frequency grid [rad/s].
    interpolation_method : str
        Description of the interpolation method used.
    n_panels : int
        Number of BEM panels used to compute the coefficients.
    capytaine_version : str
        Capytaine version string.
    """

    time: np.ndarray
    force: np.ndarray
    wave_elevation: np.ndarray
    omega: np.ndarray
    wave_amplitude: np.ndarray
    wave_phase: np.ndarray
    excitation_amplitude: np.ndarray
    excitation_phase: np.ndarray
    wave_direction: float
    source_omega_min: float
    source_omega_max: float
    interpolation_method: str
    n_panels: int
    capytaine_version: str

    @property
    def n_components(self) -> int:
        """Number of wave frequency components."""
        return len(self.omega)

    @property
    def n_time(self) -> int:
        """Number of time steps."""
        return len(self.time)


# ---------------------------------------------------------------------------
# Interpolation helper
# ---------------------------------------------------------------------------

def interpolate_excitation(
    hydro: HydrodynamicCoefficients,
    omega_query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate the complex excitation coefficient at requested frequencies.

    Interpolates real and imaginary parts separately using linear
    interpolation (numpy.interp).  Amplitude and phase are derived from
    the interpolated complex value.

    Parameters
    ----------
    hydro : HydrodynamicCoefficients
        BEM coefficients from Module 2.2.
    omega_query : np.ndarray
        Angular frequencies at which to evaluate F_exc [rad/s].
        All values must lie within [hydro.omega_min, hydro.omega_max].

    Returns
    -------
    F_complex : np.ndarray, shape (n_query,), complex
        Interpolated complex excitation coefficient [N/m].
    F_amplitude : np.ndarray, shape (n_query,)
        |F_exc(omega)| [N/m].
    F_phase : np.ndarray, shape (n_query,)
        angle(F_exc(omega)) [rad], in (-pi, pi].

    Raises
    ------
    ValueError
        If any omega_query value is outside [omega_min, omega_max].
    """
    omega_query = np.asarray(omega_query, dtype=float)
    lo, hi = hydro.omega_min, hydro.omega_max

    # Allow a small relative tolerance for floating-point round-trips
    # (e.g. 2*pi * (omega/(2*pi)) may differ from omega by ~1 ULP).
    tol = 1e-10 * max(abs(lo), abs(hi))
    out_of_range = (omega_query < lo - tol) | (omega_query > hi + tol)
    if np.any(out_of_range):
        bad = omega_query[out_of_range]
        raise ValueError(
            f"omega values {bad} rad/s are outside the BEM frequency range "
            f"[{lo:.4f}, {hi:.4f}] rad/s.  No extrapolation is performed."
        )

    F_real = np.interp(omega_query, hydro.omega, hydro.excitation_force_heave_real)
    F_imag = np.interp(omega_query, hydro.omega, hydro.excitation_force_heave_imag)
    F_complex = F_real + 1j * F_imag
    return F_complex, np.abs(F_complex), np.angle(F_complex)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def compute_excitation_force(
    hydro: HydrodynamicCoefficients,
    t: np.ndarray,
    omega: np.ndarray,
    amplitude: np.ndarray,
    phase: np.ndarray,
    wave_direction: float = 0.0,
) -> ExcitationSignal:
    """
    Compute the time-domain heave excitation force from wave components.

    For each wave component j:

        F_j(t) = a_j * |F_exc(omega_j)| * cos(omega_j*t + phi_j + angle(F_exc(omega_j)))

    Total force:

        F_exc(t) = sum_j F_j(t)

    Wave elevation (for reference):

        eta(t) = sum_j a_j * cos(omega_j*t + phi_j)

    Parameters
    ----------
    hydro : HydrodynamicCoefficients
        BEM coefficients from Module 2.2.  Provides F_exc(omega) [N/m].
    t : np.ndarray
        Time grid [s], shape (n_time,).
    omega : np.ndarray
        Angular frequencies of wave components [rad/s], shape (n_components,).
        All values must be within [hydro.omega_min, hydro.omega_max].
    amplitude : np.ndarray
        Wave component amplitudes a_j [m], shape (n_components,).  Must be >= 0.
    phase : np.ndarray
        Wave component phases phi_j [rad], shape (n_components,).
    wave_direction : float, optional
        Wave direction [deg].  Default 0.0.  Currently only 0 deg is supported
        (head-on waves, consistent with Module 2.2 BEM run at direction=0).

    Returns
    -------
    ExcitationSignal
        Frozen dataclass containing force, wave elevation, and metadata.

    Raises
    ------
    ValueError
        If array shapes are inconsistent, amplitudes are negative, or any
        omega is outside the BEM frequency range.

    Notes
    -----
    Units:
        omega      : rad/s
        amplitude  : m
        phase      : rad
        F_exc(t)   : N
        eta(t)     : m

    The computation is fully vectorized:
        theta[i, j] = omega[j]*t[i] + phase[j]
        F_j[i, j]   = amplitude[j] * |F_exc_j| * cos(theta[i,j] + angle_j)
        F_exc[i]    = sum_j F_j[i, j]
    """
    t = np.asarray(t, dtype=float)
    omega = np.asarray(omega, dtype=float)
    amplitude = np.asarray(amplitude, dtype=float)
    phase = np.asarray(phase, dtype=float)

    if t.ndim != 1:
        raise ValueError("t must be 1-D")
    if omega.ndim != 1:
        raise ValueError("omega must be 1-D")
    if amplitude.shape != omega.shape:
        raise ValueError(f"amplitude shape {amplitude.shape} != omega shape {omega.shape}")
    if phase.shape != omega.shape:
        raise ValueError(f"phase shape {phase.shape} != omega shape {omega.shape}")
    if np.any(amplitude < 0.0):
        raise ValueError("All wave amplitudes must be non-negative")

    # Interpolate excitation coefficients at requested frequencies
    F_complex, F_amp, F_ang = interpolate_excitation(hydro, omega)

    # Vectorized synthesis: shapes (n_time, n_components)
    # theta[i,j] = omega[j]*t[i] + phase[j]
    theta = omega[np.newaxis, :] * t[:, np.newaxis] + phase[np.newaxis, :]

    # Wave elevation: eta[i] = sum_j a_j * cos(theta[i,j])
    eta = np.sum(amplitude[np.newaxis, :] * np.cos(theta), axis=1)

    # Excitation force: F[i] = sum_j a_j * |F_exc_j| * cos(theta[i,j] + angle_j)
    total_phase = theta + F_ang[np.newaxis, :]
    force = np.sum(amplitude[np.newaxis, :] * F_amp[np.newaxis, :] * np.cos(total_phase), axis=1)

    return ExcitationSignal(
        time=t,
        force=force,
        wave_elevation=eta,
        omega=omega,
        wave_amplitude=amplitude,
        wave_phase=phase,
        excitation_amplitude=F_amp,
        excitation_phase=F_ang,
        wave_direction=float(wave_direction),
        source_omega_min=hydro.omega_min,
        source_omega_max=hydro.omega_max,
        interpolation_method="linear interpolation of Re(F_exc) and Im(F_exc) separately",
        n_panels=hydro.n_panels,
        capytaine_version=hydro.capytaine_version,
    )
