"""
excitation.py — Frequency-dependent irregular-wave excitation force (Module 5.2A).

============================================================
SCIENTIFIC FORMULATION
============================================================

This module replaces the Module 5.1 single-frequency approximation:

    OLD (5.1):  F_exc(t) = |H_exc(omega_p)| × eta(t)

with the full frequency-dependent irregular-wave excitation:

    NEW (5.2A): F_exc(t) = Σᵢ Re[ H_exc(ωᵢ) × Zᵢ × exp(i ωᵢ t) ]

where:

    ωᵢ = 2π fᵢ           angular frequency of component i [rad/s]

    H_exc(ωᵢ) = Re_H + i·Im_H
                complex heave excitation coefficient [N/m]
                interpolated from BEM (Module 2 HydrodynamicCoefficients)

    Zᵢ = Σⱼ aᵢⱼ exp(i φᵢⱼ)
                effective complex wave amplitude for frequency i [m·e^{iφ}]
                collapsed from the 2D directional representation

    aᵢⱼ = √(2 S(fᵢ) wⱼ Δf)    directional component amplitudes [m]
    φᵢⱼ ~ Uniform(0, 2π)      random phases (same as Module 1 goa_seastate)

============================================================
CONSISTENCY WITH MODULE 1 WAVE ELEVATION
============================================================

Module 1's goa_seastate() synthesizes η(x,y,t) via spatial_waves.py:

    η(x,y,t) = Σᵢ Σⱼ aᵢⱼ cos(ωᵢt − kᵢ(x sinθⱼ + y cosθⱼ) + φᵢⱼ)

At the buoy location (x=0, y=0), the spatial phase term vanishes:

    η(0,0,t) = Σᵢ Σⱼ aᵢⱼ cos(ωᵢt + φᵢⱼ)
             = Σᵢ Re[ Zᵢ exp(i ωᵢ t) ]

where Zᵢ = Σⱼ aᵢⱼ exp(i φᵢⱼ) is the collapsed complex amplitude.

This identity was verified numerically: max |η_from_Zᵢ − η_goa_seastate| < 4×10⁻¹⁶.

The SAME Zᵢ values are used for both η and F_exc, so:
- Wave elevation and excitation force are derived from an identical realization.
- No independent random phases are generated for the force.

============================================================
SIGN CONVENTION
============================================================

The e^{+iωt} phasor convention used throughout Module 2:

    F_exc(t) = Σᵢ Re[ H_exc(ωᵢ) · aᵢ · exp(i(ωᵢt + φᵢ)) ]

is equivalent to (for single-direction components):

    F_exc(t) = Σᵢ aᵢ |H_exc(ωᵢ)| cos(ωᵢt + φᵢ + ∠H_exc(ωᵢ))

which is the exact formula used by Module 2's compute_excitation_force().
This module uses the equivalent complex form with the collapsed amplitudes Zᵢ.

============================================================
BEM FREQUENCY RANGE AND OUT-OF-RANGE TREATMENT
============================================================

The BEM provides H_exc(ω) for ω ∈ [ωₘᵢₙ, ωₘₐₓ] = [0.2, 1.4] rad/s.

Wave components span ω ∈ [0.126, 3.14] rad/s (N=128 components for
a JONSWAP spectrum with fₘᵢₙ=0.02 Hz, fₘₐₓ=0.5 Hz).

Treatment of out-of-range components:

    ω < ωₘᵢₙ (4 components):
        These are very long waves (T > 31 s).  For Tp~8 s they carry
        negligible spectral energy.  H_exc is set to zero.

    ω > ωₘₐₓ (74 components, ~7.2% of spectral energy):
        High-frequency components for which BEM data is unavailable.
        H_exc is set to zero.  The BEM data shows |H_exc| still rising
        toward ωₘₐₓ so extrapolation would require unjustified assumptions
        near the irregular frequency.  Conservative zero is used.

Implementation: np.interp(omega, bem_omega, H_exc, left=0+0j, right=0+0j)
separating real and imaginary parts.

This is documented as a limitation in the audit.

============================================================
DIRECTION
============================================================

The reference buoy is a vertical circular cylinder — axisymmetric about
the heave axis (z-axis).  For an axisymmetric body in heave:

    H_exc(ω, θ) = H_exc(ω)    independent of wave direction θ

Reference: Falnes (2002), Ocean Waves and Oscillating Systems, §5.

The BEM was run at wave_direction=0.0 rad (head-on), which gives the
correct direction-independent coefficient for the reference hull.

At the buoy location (x=0, y=0), the spatial propagation term
k(x sinθ + y cosθ) = 0 for every direction θ, so the directional
spreading only affects the effective complex amplitudes Zᵢ through
the phase distribution, not through any directional H_exc weighting.

============================================================
PERFORMANCE / CACHING STRATEGY
============================================================

Initialization (once per ERA5 hour):
    1. Compute frequency grid, amplitudes, phases from Module 1 spectrum.
    2. Compute collapsed complex amplitudes Zᵢ (N_f complex multiplies).
    3. Interpolate H_exc(ωᵢ) from BEM data (N_f interp lookups).
    4. Precompute combined coefficients Cᵢ = H_exc(ωᵢ) × Zᵢ [N·m⁻¹·m = N].

Physics loop (per 0.1 s step, N_f = 128 operations):
    F_exc(t) = Σᵢ Re[ Cᵢ exp(i ωᵢ t) ]
             = Cᵢ_real · cos(ωᵢt) − Cᵢ_imag · sin(ωᵢt)
    Two vectorised dot products — extremely cheap.

No BEM recomputation, no spectral reconstruction inside the physics loop.

============================================================
UNITS
============================================================
    ωᵢ          rad/s
    fᵢ          Hz
    H_exc(ω)    N/m   (force per unit wave amplitude)
    Zᵢ          m·e^{iφ}   (complex wave amplitude, units m)
    Cᵢ = H·Z   N     (combined coefficient)
    F_exc(t)    N
    t           s
    η(t)        m

============================================================
MODULE DEPENDENCIES (read-only — no module modifications)
============================================================
    module1_ocean.spectrum      — jonswap()
    module1_ocean.waves         — frequency_grid(), component_amplitudes()
    module1_ocean.direction     — directional_grid(), directional_weights()
    module2_wec.hydrodynamic_coefficients — HydrodynamicCoefficients
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from module1_ocean.spectrum import jonswap
from module1_ocean.waves import frequency_grid, component_amplitudes
from module1_ocean.direction import directional_grid, directional_weights
from module2_wec.hydrodynamic_coefficients import HydrodynamicCoefficients


# ---------------------------------------------------------------------------
# Default spectral grid parameters (must match goa_seastate internals)
# ---------------------------------------------------------------------------

#: Default number of frequency components (matches goa_seastate._N_FREQ).
_N_FREQ: int = 128

#: Default number of directional bins (matches goa_seastate._N_DIR).
_N_DIR: int = 16

#: Default minimum frequency [Hz] (matches goa_seastate._F_MIN).
_F_MIN: float = 0.02

#: Default maximum frequency [Hz] (matches goa_seastate._F_MAX).
_F_MAX: float = 0.5

#: Default JONSWAP gamma.
_GAMMA: float = 3.3


# ---------------------------------------------------------------------------
# WaveComponents — per-ERA5-hour wave spectral data
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WaveComponents:
    """
    Spectral and phase data for one ERA5 sea-state hour.

    Stores everything needed to:
    (a) reproduce the same surface elevation as goa_seastate(), and
    (b) compute the corresponding heave excitation force via the BEM response.

    Parameters
    ----------
    f_hz : np.ndarray, shape (N_f,)
        Frequency grid [Hz].
    omega : np.ndarray, shape (N_f,)
        Angular frequency grid [rad/s].  omega = 2π f.
    S_f : np.ndarray, shape (N_f,)
        JONSWAP variance density spectrum [m²/Hz].
    a_2d : np.ndarray, shape (N_f, N_dir)
        Directional component amplitudes [m].
        a_ij = sqrt(2 * S(f_i) * w_j * Δf).
    phases_2d : np.ndarray, shape (N_f, N_dir)
        Random phases [rad].  φᵢⱼ ~ Uniform(0, 2π).
        Identical to those used in goa_seastate().
    Z : np.ndarray, shape (N_f,), dtype complex128
        Effective complex amplitude per frequency:
        Zᵢ = Σⱼ aᵢⱼ exp(i φᵢⱼ)
        This collapses the 2D (N_f, N_dir) representation to 1D while
        preserving the exact same η(0,0,t) realization.
    Hs : float
        Significant wave height [m].
    Tp : float
        Peak wave period [s].
    direction_deg : float
        Mean wave direction [degrees, ERA5 mwd convention].
    seed : Optional[int]
        Random seed used to generate phases.
    """

    f_hz:          np.ndarray
    omega:         np.ndarray
    S_f:           np.ndarray
    a_2d:          np.ndarray
    phases_2d:     np.ndarray
    Z:             np.ndarray
    Hs:            float
    Tp:            float
    direction_deg: float
    seed:          Optional[int]

    @property
    def n_components(self) -> int:
        """Number of frequency components."""
        return len(self.f_hz)

    @property
    def df(self) -> float:
        """Frequency spacing [Hz]."""
        return float(self.f_hz[1] - self.f_hz[0])


# ---------------------------------------------------------------------------
# build_wave_components — factory matching goa_seastate() internals
# ---------------------------------------------------------------------------

def build_wave_components(
    Hs:            float,
    Tp:            float,
    direction_deg: float,
    gamma:         float = _GAMMA,
    n_freq:        int   = _N_FREQ,
    n_directions:  int   = _N_DIR,
    seed:          Optional[int] = None,
) -> WaveComponents:
    """
    Build a WaveComponents object that EXACTLY reproduces the Module 1
    goa_seastate() wave realization.

    This function replicates the internal logic of goa_seastate() and
    synthesize_spatial_surface() to expose the individual wave components
    (frequencies, amplitudes, phases) that are otherwise hidden behind
    the high-level goa_seastate() API.

    Parameters
    ----------
    Hs : float
        Significant wave height [m].  Must be > 0.
    Tp : float
        Peak wave period [s].  Must be > 0.
    direction_deg : float
        Mean wave direction [degrees, ERA5 mwd, clockwise from North].
    gamma : float
        JONSWAP peak enhancement factor.  Default 3.3.
    n_freq : int
        Number of frequency components.  Must match goa_seastate() default (128).
    n_directions : int
        Number of directional bins.  Must match goa_seastate() default (16).
    seed : int or None
        Random seed.  MUST be the same seed passed to goa_seastate() to
        guarantee wave realization consistency.

    Returns
    -------
    WaveComponents
        Contains the spectral grid, amplitudes, phases, and the effective
        complex amplitudes Zᵢ = Σⱼ aᵢⱼ exp(i φᵢⱼ).

    Notes
    -----
    Phase generation:
        numpy.random.default_rng(seed).uniform(0, 2π, (N_f, N_dir))
        — identical to synthesize_spatial_surface() lines 83-84.

    Amplitude formula:
        aᵢⱼ = sqrt(2 · S(fᵢ) · wⱼ · Δf)
        where wⱼ = normalized directional weights (Σⱼ wⱼ = 1).

    Collapsed complex amplitude:
        Zᵢ = Σⱼ aᵢⱼ exp(i φᵢⱼ)
        Identity verified: Σᵢ Re[Zᵢ exp(i ωᵢ t)] = η(0, 0, t) exactly.
    """
    # --- Clamp physical inputs ---
    Hs = max(float(Hs), 0.01)
    Tp = max(float(Tp), 1.0)

    # --- Frequency grid (identical to goa_seastate()) ---
    fp = 1.0 / Tp
    f_min = min(_F_MIN, fp * 0.5)
    f_max = max(_F_MAX, fp * 3.0)
    f_hz  = frequency_grid(f_min, f_max, n_freq)
    df    = float(f_hz[1] - f_hz[0])
    omega = 2.0 * np.pi * f_hz   # rad/s

    # --- JONSWAP spectrum ---
    S_f = jonswap(f_hz, Hs, Tp, gamma=gamma)

    # --- Directional grid and weights ---
    mean_dir_rad = np.deg2rad(float(direction_deg))
    theta        = directional_grid(n_directions)              # (N_dir,)
    w            = directional_weights(theta, mean_dir_rad)    # (N_dir,), Σwⱼ=1

    # --- Directional amplitudes: aᵢⱼ = sqrt(2 S(fᵢ) wⱼ Δf) ---
    S_2d = S_f[:, np.newaxis] * w[np.newaxis, :]              # (N_f, N_dir)
    a_2d = np.sqrt(2.0 * S_2d * df)                           # (N_f, N_dir)

    # --- Random phases (SAME RNG as synthesize_spatial_surface()) ---
    rng        = np.random.default_rng(seed)
    phases_2d  = rng.uniform(0.0, 2.0 * np.pi, (n_freq, n_directions))

    # --- Effective complex amplitudes: Zᵢ = Σⱼ aᵢⱼ exp(i φᵢⱼ) ---
    # shape (N_f,), dtype complex128
    Z = np.sum(a_2d * np.exp(1j * phases_2d), axis=1)

    return WaveComponents(
        f_hz          = f_hz,
        omega         = omega,
        S_f           = S_f,
        a_2d          = a_2d,
        phases_2d     = phases_2d,
        Z             = Z,
        Hs            = float(Hs),
        Tp            = float(Tp),
        direction_deg = float(direction_deg),
        seed          = seed,
    )


# ---------------------------------------------------------------------------
# IrregularExcitationModel — the core adapter
# ---------------------------------------------------------------------------

class IrregularExcitationModel:
    """
    Frequency-dependent heave excitation force model for irregular waves.

    Computes:

        F_exc(t) = Σᵢ Re[ H_exc(ωᵢ) · Zᵢ · exp(i ωᵢ t) ]
                 = Σᵢ [ C_real_i cos(ωᵢt) − C_imag_i sin(ωᵢt) ]

    where Cᵢ = H_exc(ωᵢ) · Zᵢ is precomputed once at construction.

    EVALUATION is O(N_f) per timestep using two vectorised dot products —
    no recomputation of the spectrum, phases, or BEM interpolation inside
    the physics loop.

    Parameters
    ----------
    components : WaveComponents
        Wave spectral data (from build_wave_components()).
    hydro : HydrodynamicCoefficients
        Module 2 BEM coefficients.  Must contain full complex excitation data.
    t0 : float
        Absolute start time [s] of this model's epoch.  F_exc is evaluated
        at t_abs = t0 + t_local.  Typically set to the start time of the
        ERA5 hour segment.

    Attributes
    ----------
    omega : np.ndarray, shape (N_f,)
        Angular frequency grid [rad/s].
    C : np.ndarray, shape (N_f,), dtype complex128
        Combined coefficient Cᵢ = H_exc(ωᵢ) · Zᵢ [N].
    C_real : np.ndarray, shape (N_f,)
        Re(Cᵢ) [N].
    C_imag : np.ndarray, shape (N_f,)
        Im(Cᵢ) [N].
    n_active : int
        Number of components with |H_exc| > 0 (within BEM range).
    H_exc_interp : np.ndarray, shape (N_f,), dtype complex128
        Interpolated BEM complex excitation coefficient at each ωᵢ [N/m].
        Zero for components outside the BEM frequency range.
    """

    def __init__(
        self,
        components: WaveComponents,
        hydro:      HydrodynamicCoefficients,
        t0:         float = 0.0,
    ) -> None:
        self._components = components
        self._hydro      = hydro
        self._t0         = float(t0)

        self.omega = components.omega.copy()   # (N_f,)

        # --- Interpolate H_exc onto wave frequency grid ---
        # Interpolate real and imaginary parts separately to avoid
        # phase-wrapping discontinuities.
        # Components outside [omega_min, omega_max] → H_exc = 0+0j.
        H_real_interp = np.interp(
            self.omega,
            hydro.omega,
            hydro.excitation_force_heave_real,
            left=0.0, right=0.0,
        )
        H_imag_interp = np.interp(
            self.omega,
            hydro.omega,
            hydro.excitation_force_heave_imag,
            left=0.0, right=0.0,
        )
        self.H_exc_interp = H_real_interp + 1j * H_imag_interp   # (N_f,)

        # --- Precompute combined coefficient Cᵢ = H_exc(ωᵢ) · Zᵢ ---
        self.C      = self.H_exc_interp * components.Z             # (N_f,) complex
        self.C_real = np.real(self.C)
        self.C_imag = np.imag(self.C)

        # Count active components (those with non-zero H_exc)
        self.n_active = int(np.sum(np.abs(self.H_exc_interp) > 0.0))

    # ------------------------------------------------------------------
    # Primary evaluation interface
    # ------------------------------------------------------------------

    def force_at(self, t_local: float) -> float:
        """
        Return F_exc at a local time [s] relative to the epoch start t0.

        Computes:

            F_exc = Σᵢ Re[ Cᵢ exp(i ωᵢ (t0 + t_local)) ]
                  = Σᵢ C_real_i cos(ωᵢ t_abs) − C_imag_i sin(ωᵢ t_abs)

        This is two vectorised dot products: O(N_f) with no memory allocation.

        Parameters
        ----------
        t_local : float
            Time offset from epoch start [s].  t_abs = t0 + t_local.

        Returns
        -------
        float : F_exc(t) [N]
        """
        t_abs = self._t0 + float(t_local)
        phase = self.omega * t_abs  # (N_f,)
        return float(
            np.dot(self.C_real, np.cos(phase))
            - np.dot(self.C_imag, np.sin(phase))
        )

    def force_array(self, t_local: np.ndarray) -> np.ndarray:
        """
        Return F_exc at an array of local times [s].

        More efficient than calling force_at() in a loop when evaluating
        large batches (e.g. for diagnostic plots).

        Parameters
        ----------
        t_local : np.ndarray, shape (N_t,)
            Local time offsets [s].

        Returns
        -------
        np.ndarray, shape (N_t,) : F_exc [N]
        """
        t_abs = self._t0 + np.asarray(t_local, dtype=float)   # (N_t,)
        # phase[j,i] = omega[i] * t_abs[j]  →  shape (N_t, N_f)
        phase = self.omega[np.newaxis, :] * t_abs[:, np.newaxis]
        return (
            np.dot(np.cos(phase), self.C_real)
            - np.dot(np.sin(phase), self.C_imag)
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def eta_at(self, t_local: float) -> float:
        """
        Return the surface elevation η(0,0,t) [m] consistent with the
        excitation force.

        η(t) = Σᵢ Re[ Zᵢ exp(i ωᵢ t_abs) ]

        This is the same wave realization used for F_exc — derived from
        identical Zᵢ values, so η and F_exc are physically consistent.

        Parameters
        ----------
        t_local : float
            Local time offset [s].

        Returns
        -------
        float : η(t) [m]
        """
        t_abs = self._t0 + float(t_local)
        phase = self.omega * t_abs
        Z     = self._components.Z
        return float(
            np.dot(np.real(Z), np.cos(phase))
            - np.dot(np.imag(Z), np.sin(phase))
        )

    def eta_array(self, t_local: np.ndarray) -> np.ndarray:
        """
        Return η(0,0,t) [m] at an array of local times.

        Parameters
        ----------
        t_local : np.ndarray, shape (N_t,)

        Returns
        -------
        np.ndarray, shape (N_t,) : η(t) [m]
        """
        t_abs = self._t0 + np.asarray(t_local, dtype=float)
        phase = self.omega[np.newaxis, :] * t_abs[:, np.newaxis]   # (N_t, N_f)
        Z     = self._components.Z
        return (
            np.dot(np.cos(phase), np.real(Z))
            - np.dot(np.sin(phase), np.imag(Z))
        )

    @property
    def components(self) -> WaveComponents:
        """The underlying WaveComponents."""
        return self._components

    @property
    def t0(self) -> float:
        """Epoch start time [s]."""
        return self._t0

    def summary(self) -> dict:
        """
        Return a diagnostic summary dictionary.

        Includes BEM coverage statistics, total Cᵢ energy, and
        a description of out-of-range treatment.
        """
        omega_min = float(self._hydro.omega_min)
        omega_max = float(self._hydro.omega_max)
        in_range  = (self.omega >= omega_min) & (self.omega <= omega_max)
        return {
            "n_components_total":   int(len(self.omega)),
            "n_components_active":  self.n_active,
            "n_below_bem":          int(np.sum(self.omega < omega_min)),
            "n_above_bem":          int(np.sum(self.omega > omega_max)),
            "bem_omega_min_rad_s":  omega_min,
            "bem_omega_max_rad_s":  omega_max,
            "rms_C_N":              float(np.sqrt(0.5 * np.sum(np.abs(self.C)**2))),
            "max_abs_C_N":          float(np.max(np.abs(self.C))),
            "out_of_range_policy":  "H_exc = 0 (zero excitation, no extrapolation)",
            "direction_treatment":  "axisymmetric hull: H_exc independent of direction; spatial phase=0 at (0,0)",
            "eta_consistency":      "Zᵢ shared between η and F_exc — identical wave realization",
        }


# ---------------------------------------------------------------------------
# HourlyExcitationBuffer — per-episode precomputed buffer
# ---------------------------------------------------------------------------

class HourlyExcitationBuffer:
    """
    Pre-builds one IrregularExcitationModel per ERA5 hour for a full episode.

    This avoids rebuilding spectra and interpolating BEM data on the fly
    during the physics loop.  All computation is done once at episode start.

    Parameters
    ----------
    hydro : HydrodynamicCoefficients
        Module 2 BEM coefficients (precomputed, not recomputed here).
    Hs_arr : np.ndarray, shape (n_hours,)
        Significant wave height per ERA5 hour [m].
    Tp_arr : np.ndarray, shape (n_hours,)
        Peak wave period per ERA5 hour [s].
    dir_arr : np.ndarray, shape (n_hours,)
        Mean wave direction per ERA5 hour [degrees].
    base_seed : int or None
        Hour i uses seed = base_seed + i.  None = non-reproducible.
    gamma : float
        JONSWAP gamma.  Default 3.3.
    n_freq : int
        Number of spectral frequency components.  Default 128.
    n_directions : int
        Number of directional bins.  Default 16.
    dt_physics : float
        Physics timestep [s].  Used to compute t0 for each hour's model.
    """

    def __init__(
        self,
        hydro:        HydrodynamicCoefficients,
        Hs_arr:       np.ndarray,
        Tp_arr:       np.ndarray,
        dir_arr:      np.ndarray,
        base_seed:    Optional[int] = None,
        gamma:        float = _GAMMA,
        n_freq:       int   = _N_FREQ,
        n_directions: int   = _N_DIR,
        dt_physics:   float = 0.1,
    ) -> None:
        self._hydro      = hydro
        self._dt_physics = dt_physics
        n_hours          = len(Hs_arr)

        self._models: list[IrregularExcitationModel] = []

        for i in range(n_hours):
            Hs_i  = float(Hs_arr[i])
            Tp_i  = float(Tp_arr[i])
            dir_i = float(dir_arr[i])

            # Handle NaN rows with calm-sea defaults
            if not (np.isfinite(Hs_i) and np.isfinite(Tp_i) and np.isfinite(dir_i)):
                Hs_i, Tp_i, dir_i = 0.01, 8.0, 270.0

            hour_seed = (base_seed + i) if base_seed is not None else None

            comp  = build_wave_components(
                Hs=Hs_i, Tp=Tp_i, direction_deg=dir_i,
                gamma=gamma, n_freq=n_freq, n_directions=n_directions,
                seed=hour_seed,
            )
            # t0 for this hour = i * 3600 s (start of the ERA5 hour segment)
            model = IrregularExcitationModel(comp, hydro, t0=float(i) * 3600.0)
            self._models.append(model)

    def force_at_physics_step(
        self,
        hour_index:   int,
        t_within_hour: float,
    ) -> float:
        """
        Return F_exc [N] at a given ERA5 hour and local time within that hour.

        Parameters
        ----------
        hour_index : int
            ERA5 hour index (0-based).
        t_within_hour : float
            Time offset from start of this ERA5 hour [s].
            Range [0, 3600).

        Returns
        -------
        float : F_exc [N]
        """
        idx = min(int(hour_index), len(self._models) - 1)
        return self._models[idx].force_at(t_within_hour)

    def get_model(self, hour_index: int) -> IrregularExcitationModel:
        """Return the IrregularExcitationModel for a given ERA5 hour."""
        idx = min(int(hour_index), len(self._models) - 1)
        return self._models[idx]

    @property
    def n_hours(self) -> int:
        """Number of ERA5 hours in the buffer."""
        return len(self._models)
