"""
radiation.py — Radiation memory model for the WEC Digital Twin (Module 2.3A).

Converts frequency-domain hydrodynamic radiation data A(ω), B(ω) from
Module 2.2 (Capytaine BEM) into the time-domain Cummins radiation model:

    F_rad(t) = -A_inf * x_ddot(t)
               - integral_0^t K(t-tau) * x_dot(tau) dtau

where K(t) is the radiation impulse-response function (radiation kernel).

============================================================
TRANSFORM CONVENTION — DERIVATION
============================================================

The Cummins (1962) equation of motion for a heaving body is:

    (M + A_inf) x_ddot(t) + integral_0^t K(t-tau) x_dot(tau) dtau + C x(t)
        = F_exc(t)

where the radiation force is split into:
    - instantaneous inertia term:  -A_inf * x_ddot
    - memory (convolution) term:   -integral K(t-tau) x_dot(tau) dtau

Taking the Fourier transform of the convolution term and comparing with
the frequency-domain radiation force:

    F_rad(omega) = [-omega^2 A(omega) + i*omega*B(omega)] * X(omega)

where X(omega) is the Fourier transform of x(t), and using:

    F_rad(omega) = [-omega^2 A_inf + i*omega * hat_K(omega)] * X(omega)

we identify:

    hat_K(omega) = B(omega) + i*omega*(A(omega) - A_inf)

where hat_K(omega) is the one-sided Fourier transform of K(t):

    hat_K(omega) = integral_0^inf K(t) e^{-i*omega*t} dt

Since K(t) is causal (K(t) = 0 for t < 0) and real, the real and imaginary
parts of hat_K(omega) are related by:

    Re[hat_K(omega)] = B(omega)
    Im[hat_K(omega)] = omega * (A(omega) - A_inf)

These are the Ogilvie (1964) relations.

From Re[hat_K(omega)] = B(omega) and the cosine-transform pair for a
causal real function:

    hat_K(omega) = integral_0^inf K(t) cos(omega*t) dt
                   - i * integral_0^inf K(t) sin(omega*t) dt

Therefore:

    B(omega) = integral_0^inf K(t) cos(omega*t) dt          ... (1)

    omega*(A(omega) - A_inf) = -integral_0^inf K(t) sin(omega*t) dt  ... (2)

Inverting (1) using the cosine transform:

    K(t) = (2/pi) * integral_0^inf B(omega) cos(omega*t) domega   ... (3)

This is the EXACT formula used in this module.

Inverting (2) gives the Ogilvie relation for A_inf:

    A_inf = A(omega) + (1/omega) * integral_0^inf K(t) sin(omega*t) dt  ... (4)

============================================================
A_INFINITY — ESTIMATION AND LIMITATIONS
============================================================

A_inf is the added mass at infinite frequency (high-frequency limit).
For a heaving cylinder, A(omega) increases monotonically from A(0) toward
A_inf as omega increases.

CRITICAL LIMITATION for this dataset:
    omega_max = 1.4 rad/s is only 67% of omega_irr ≈ 2.088 rad/s.
    A(omega) is still rising steeply at omega_max (dA/domega > 0).
    The data does NOT reach the high-frequency asymptote.
    A_inf CANNOT be reliably determined from this frequency range alone.

Three estimation methods are implemented and compared:

Method 1 — Endpoint estimate:
    A_inf_endpoint = A(omega_max) = A[-1]
    This is a LOWER BOUND. A(omega) is still rising at omega_max,
    so the true A_inf > A[-1]. Underestimates A_inf significantly.

Method 2 — Power-law trend fit:
    Fit A(omega) ~ A_inf + c / omega^2 over the upper portion of the
    frequency range. The 1/omega^2 decay is the expected high-frequency
    behaviour for a heaving body (Ogilvie 1964).
    Quality is poor because the data is far from the asymptote.

Method 3 — Ogilvie self-consistency:
    Use equation (4) with the computed K(t) to estimate A_inf at each
    omega. In the mid-range (omega ~ 0.6–1.0 rad/s), the estimate is
    most stable. The median of this range is used.
    This is the SELECTED METHOD for the current development value.
    It is self-consistent with the kernel but still affected by the
    truncation of B(omega) at omega_max.

SELECTED VALUE: Ogilvie mid-range median (omega 0.6–1.0 rad/s).
This is a development estimate only. It will change when the frequency
range is extended or a higher-quality mesh is used.

============================================================
LOW-FREQUENCY TREATMENT (omega < omega_min = 0.2 rad/s)
============================================================

The integration domain for the forward kernel is [omega_min, omega_max],
not [0, omega_max].  The region 0 <= omega < omega_min is ABSENT.

Exact mechanism:
    omega_dense = np.linspace(omega_min, omega_max, n_interp)
    B_dense     = np.interp(omega_dense, omega, B)

    np.interp does NOT extrapolate below omega_min.  The dense grid
    starts exactly at omega_min = 0.2 rad/s.  No point below 0.2 rad/s
    is ever evaluated or included in the trapezoid integral.

This is equivalent to assuming B(omega) = 0 for omega < omega_min,
but the zero region is never explicitly constructed — it is simply
not part of the integration domain.

Effect on K(t):
    The missing contribution is bounded above by:
        (2/pi) * B(omega_min) * omega_min
        = (2/pi) * 399.6 * 0.2 ≈ 50.9 N·s/m
    This is 0.15% of K(0) ≈ 33 737 N·s/m.
    The low-frequency truncation has negligible effect on K(t).

Effect on B_rec(omega):
    The reconstruction integral is over TIME:
        B_rec(omega) = integral_0^{t_max} K(t) cos(omega*t) dt
    K(t) was built from B on [omega_min, omega_max] only, so B_rec
    inherits the same low-frequency deficit at all omega.  However,
    because the missing contribution to K(t) is only 0.15% of K(0),
    the effect on B_rec is negligible across the entire omega grid.

Effect on the trusted reconstruction range [0.45, 1.15] rad/s:
    None measurable.  The trusted range starts at 0.45 rad/s, well
    above omega_min = 0.2 rad/s.  The 59% relative error at
    omega = 0.2 rad/s is caused by the finite kernel duration
    (t_max = 60 s) and the high-frequency truncation artifact, not
    by the low-frequency gap.

No extrapolation of B below omega_min is performed or needed.

============================================================
RECONSTRUCTION QUALITY
============================================================

The reconstruction quality is limited by two factors:

1. Truncation of B(omega) at omega_max = 1.4 rad/s:
   B(omega) is still rising at omega_max. The missing high-frequency
   content causes:
   - Gibbs-like ringing in K(t) (mitigated by cosine tapering)
   - Systematic underestimation of B_rec at omega near omega_max
   - Large errors in A_rec because A_inf is uncertain

2. Finite frequency range (omega_min = 0.2 rad/s):
   The low-frequency truncation causes errors in B_rec at low omega.

TRUSTED RECONSTRUCTION RANGE: omega in [0.45, 1.15] rad/s
   B reconstruction: RMSE ~ 1%, max relative error ~ 3%
   A reconstruction: RMSE ~ 17%, max relative error ~ 37%
   (A errors dominated entirely by A_inf uncertainty, not kernel quality)

Outside the trusted range:
   omega < 0.45 rad/s: B is small; absolute errors are small but
                        relative errors are large.
   omega > 1.15 rad/s: truncation artifact dominates; B_rec degrades
                        rapidly toward omega_max.

============================================================
NUMERICAL PARAMETERS
============================================================

Default parameters (selected after sensitivity study):
    t_max          = 60.0 s   — kernel duration
    dt             = 0.05 s   — time step
    n_interp       = 2000     — interpolation points on [omega_min, omega_max]
    taper_fraction = 0.15     — fraction of omega range tapered at high end

Sensitivity findings:
    - t_max: 30 s gives similar B reconstruction to 60 s; 60 s used for
      safety (kernel has decayed to <1% of K(0) by ~1.5 s).
    - dt: 0.05 s and 0.02 s give identical results; 0.05 s used.
    - n_interp: 1000 and 2000 give identical results; 2000 used.
    - taper_fraction: 0.15 gives best B RMSE; 0.25 over-smooths.
    - No taper: B RMSE ~36%, max ~112%. Taper is essential.

============================================================
UNITS (SI throughout)
============================================================
    omega       : rad/s
    t           : s
    K(t)        : N·s/m  (radiation damping kernel)
    A_inf       : kg     (infinite-frequency added mass)
    B(omega)    : N·s/m  (radiation damping)
    A(omega)    : kg     (added mass)

============================================================
REFERENCES
============================================================
- Cummins (1962), The impulse response function and ship motions,
  Schiffstechnik 9, 101–109.
- Ogilvie (1964), Recent progress toward the understanding and prediction
  of ship motions, 5th Symp. Naval Hydrodynamics.
- Falnes (2002), Ocean Waves and Oscillating Systems, Cambridge Univ. Press.
- Yu & Falnes (1995), State-space modelling of a vertical cylinder in heave,
  Applied Ocean Research 17, 265–275.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import capytaine as cpt
import numpy as np

from module2_wec.hydrodynamic_coefficients import HydrodynamicCoefficients


# ---------------------------------------------------------------------------
# Default numerical parameters
# ---------------------------------------------------------------------------

#: Default kernel duration [s].
DEFAULT_T_MAX: float = 60.0

#: Default time step [s].
DEFAULT_DT: float = 0.05

#: Default number of interpolation points on [omega_min, omega_max].
DEFAULT_N_INTERP: int = 2000

#: Default cosine-taper fraction at the high-frequency end.
#: Fraction of the interpolated omega range that is tapered to zero.
DEFAULT_TAPER_FRACTION: float = 0.15

#: Omega range used for Ogilvie A_inf mid-range estimate [rad/s].
_OGILVIE_OMEGA_LO: float = 0.6
_OGILVIE_OMEGA_HI: float = 1.0


# ---------------------------------------------------------------------------
# RadiationKernel dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RadiationKernel:
    """
    Radiation memory model: A_infinity and K(t) for a heaving cylinder.

    Immutable (frozen dataclass).  Produced by compute_radiation_kernel().

    Parameters
    ----------
    omega : np.ndarray
        Angular frequency grid from Module 2.2 BEM [rad/s], shape (n_omega,).
    frequency_hz : np.ndarray
        Frequency [Hz] = omega / (2*pi), shape (n_omega,).
    added_mass : np.ndarray
        BEM added mass A(omega) [kg], shape (n_omega,).
    radiation_damping : np.ndarray
        BEM radiation damping B(omega) [N·s/m], shape (n_omega,).
    A_infinity : float
        Infinite-frequency added mass [kg].  See module docstring for
        estimation method and limitations.
    A_infinity_method : str
        Method used to estimate A_infinity.  One of:
        'endpoint', 'trend_fit', 'ogilvie_midrange'.
    A_infinity_endpoint : float
        Endpoint estimate A(omega_max) [kg].  Lower bound.
    A_infinity_trend_fit : float
        Power-law trend fit estimate [kg].  Unreliable for this dataset.
    A_infinity_ogilvie : float
        Ogilvie mid-range median estimate [kg].  Selected method.
    time : np.ndarray
        Time grid [s], shape (n_time,).  Starts at 0, step = dt.
    kernel : np.ndarray
        Radiation impulse-response function K(t) [N·s/m], shape (n_time,).
        K(t) = (2/pi) * integral_{omega_min}^{omega_max} B_tapered(omega) cos(omega*t) domega.
        The region [0, omega_min) is absent from the integral (see module
        docstring, LOW-FREQUENCY TREATMENT section).
    added_mass_reconstructed : np.ndarray
        A(omega) reconstructed from K(t) via Ogilvie relation [kg].
    radiation_damping_reconstructed : np.ndarray
        B(omega) reconstructed from K(t) [N·s/m].
    omega_min : float
        Minimum BEM frequency [rad/s].
    omega_max : float
        Maximum BEM frequency [rad/s].
    omega_irr_estimate : float
        First irregular frequency estimate [rad/s].
    t_max : float
        Kernel duration [s].
    dt : float
        Kernel time step [s].
    n_interp : int
        Number of interpolation points used on [omega_min, omega_max].
    taper_fraction : float
        Fraction of omega range tapered at high end.
    water_depth : float
        Water depth used in BEM [m].
    rho : float
        Water density used in BEM [kg/m³].
    radius : float
        Cylinder radius [m].
    draft : float
        Cylinder draft [m].
    n_panels : int
        Number of BEM panels.
    capytaine_version : str
        Capytaine version string.
    """

    # Frequency-domain inputs (from Module 2.2)
    omega: np.ndarray
    frequency_hz: np.ndarray
    added_mass: np.ndarray
    radiation_damping: np.ndarray

    # A_infinity estimates
    A_infinity: float
    A_infinity_method: str
    A_infinity_endpoint: float
    A_infinity_trend_fit: float
    A_infinity_ogilvie: float

    # Time-domain kernel
    time: np.ndarray
    kernel: np.ndarray

    # Reconstruction
    added_mass_reconstructed: np.ndarray
    radiation_damping_reconstructed: np.ndarray

    # Metadata
    omega_min: float
    omega_max: float
    omega_irr_estimate: float
    t_max: float
    dt: float
    n_interp: int
    taper_fraction: float
    water_depth: float
    rho: float
    radius: float
    draft: float
    n_panels: int
    capytaine_version: str

    @property
    def n_omega(self) -> int:
        """Number of BEM frequency points."""
        return len(self.omega)

    @property
    def n_time(self) -> int:
        """Number of time steps in the kernel."""
        return len(self.time)

    def reconstruction_errors(self) -> dict:
        """
        Compute reconstruction errors for B(omega) and A(omega).

        Returns absolute error, relative error, RMSE, and max relative
        error over the full omega grid and the trusted sub-range.

        The trusted range [0.45, 1.15] rad/s excludes:
        - Low omega where B is small and relative errors are large.
        - High omega where truncation artifact dominates.

        Returns
        -------
        dict with keys:
            B_abs_error, B_rel_error, B_rmse_pct, B_max_rel_pct,
            A_abs_error, A_rel_error, A_rmse_pct, A_max_rel_pct,
            B_rmse_trusted_pct, B_max_rel_trusted_pct,
            A_rmse_trusted_pct, A_max_rel_trusted_pct,
            trusted_omega_min, trusted_omega_max.
        """
        B = self.radiation_damping
        B_rec = self.radiation_damping_reconstructed
        A = self.added_mass
        A_rec = self.added_mass_reconstructed

        B_abs = np.abs(B_rec - B)
        B_rel = np.where(B > 1.0, B_abs / B, np.nan)
        A_abs = np.abs(A_rec - A)
        A_rel = A_abs / np.abs(A)

        # Trusted range
        trusted = (self.omega >= 0.45) & (self.omega <= 1.15)
        B_rel_t = B_rel[trusted]
        A_rel_t = A_rel[trusted]

        def _rmse(arr):
            valid = arr[np.isfinite(arr)]
            return float(np.sqrt(np.mean(valid**2)) * 100) if len(valid) > 0 else float('nan')

        def _maxrel(arr):
            valid = arr[np.isfinite(arr)]
            return float(np.max(valid) * 100) if len(valid) > 0 else float('nan')

        return {
            "B_abs_error":           B_abs,
            "B_rel_error":           B_rel,
            "B_rmse_pct":            _rmse(B_rel),
            "B_max_rel_pct":         _maxrel(B_rel),
            "A_abs_error":           A_abs,
            "A_rel_error":           A_rel,
            "A_rmse_pct":            _rmse(A_rel),
            "A_max_rel_pct":         _maxrel(A_rel),
            "B_rmse_trusted_pct":    _rmse(B_rel_t),
            "B_max_rel_trusted_pct": _maxrel(B_rel_t),
            "A_rmse_trusted_pct":    _rmse(A_rel_t),
            "A_max_rel_trusted_pct": _maxrel(A_rel_t),
            "trusted_omega_min":     float(self.omega[trusted][0]) if trusted.any() else float('nan'),
            "trusted_omega_max":     float(self.omega[trusted][-1]) if trusted.any() else float('nan'),
        }

    def summary(self) -> dict:
        """
        Return a text-friendly summary of the radiation kernel.

        Returns
        -------
        dict
        """
        errs = self.reconstruction_errors()
        return {
            "capytaine_version":        self.capytaine_version,
            "n_panels":                 self.n_panels,
            "omega_min_rad_s":          self.omega_min,
            "omega_max_rad_s":          self.omega_max,
            "omega_irr_estimate_rad_s": self.omega_irr_estimate,
            "A_infinity_kg":            self.A_infinity,
            "A_infinity_method":        self.A_infinity_method,
            "A_infinity_endpoint_kg":   self.A_infinity_endpoint,
            "A_infinity_trend_fit_kg":  self.A_infinity_trend_fit,
            "A_infinity_ogilvie_kg":    self.A_infinity_ogilvie,
            "t_max_s":                  self.t_max,
            "dt_s":                     self.dt,
            "n_time":                   self.n_time,
            "taper_fraction":           self.taper_fraction,
            "K0_N_s_m":                 float(self.kernel[0]),
            "K_max_abs_N_s_m":          float(np.max(np.abs(self.kernel))),
            "B_rmse_trusted_pct":       errs["B_rmse_trusted_pct"],
            "B_max_rel_trusted_pct":    errs["B_max_rel_trusted_pct"],
            "A_rmse_trusted_pct":       errs["A_rmse_trusted_pct"],
            "A_max_rel_trusted_pct":    errs["A_max_rel_trusted_pct"],
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cosine_taper(n: int, taper_fraction: float) -> np.ndarray:
    """
    Return a window that is 1 for the first (1-taper_fraction) of n points,
    then tapers smoothly to 0 using a raised cosine.
    """
    w = np.ones(n)
    n_taper = int(taper_fraction * n)
    if n_taper > 0:
        w[-n_taper:] = 0.5 * (1.0 + np.cos(np.pi * np.arange(n_taper) / n_taper))
    return w


def _compute_kernel_array(
    omega_dense: np.ndarray,
    B_tapered: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """
    Compute K(t) = (2/pi) * integral_{omega_min}^{omega_max} B_tapered(omega) cos(omega*t) domega.

    Uses numpy.trapezoid for numerical integration.

    LOW-FREQUENCY NOTE: omega_dense starts at omega_min (not 0).  The region
    [0, omega_min) is absent from the integral.  This is equivalent to
    assuming B = 0 there, but that region is never explicitly constructed.
    The missing contribution is < 0.2% of K(0) for the production dataset.

    Parameters
    ----------
    omega_dense : np.ndarray, shape (n_interp,)
        Dense frequency grid [rad/s], spanning [omega_min, omega_max].
    B_tapered : np.ndarray, shape (n_interp,)
        Tapered radiation damping [N·s/m].
    t : np.ndarray, shape (n_time,)
        Time grid [s].

    Returns
    -------
    np.ndarray, shape (n_time,)
        Radiation kernel K(t) [N·s/m].
    """
    K = np.empty(len(t))
    for i, ti in enumerate(t):
        K[i] = (2.0 / np.pi) * np.trapezoid(
            B_tapered * np.cos(omega_dense * ti), omega_dense
        )
    return K


def _estimate_A_infinity(
    omega: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
) -> tuple[float, float, float]:
    """
    Estimate A_infinity using three methods.

    Returns
    -------
    (A_inf_endpoint, A_inf_trend_fit, A_inf_ogilvie) : tuple of float
    """
    # Method 1: endpoint
    A_inf_endpoint = float(A[-1])

    # Method 2: power-law trend fit A ~ A_inf + c/omega^2 (last 8 points)
    n_fit = min(8, len(omega))
    omega_fit = omega[-n_fit:]
    A_fit = A[-n_fit:]
    X = np.column_stack([np.ones(n_fit), 1.0 / omega_fit**2])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X, A_fit, rcond=None)
        A_inf_trend_fit = float(coeffs[0])
    except np.linalg.LinAlgError:
        A_inf_trend_fit = float(A[-1])

    # Method 3: Ogilvie self-consistency — use mid-range omega
    # A_inf = A(omega) + (1/omega) * integral_0^inf K(t) sin(omega*t) dt
    mid_mask = (omega >= _OGILVIE_OMEGA_LO) & (omega <= _OGILVIE_OMEGA_HI)
    if not mid_mask.any():
        # Fall back to all omega if mid-range not available
        mid_mask = np.ones(len(omega), dtype=bool)
    omega_mid = omega[mid_mask]
    A_mid = A[mid_mask]
    sin_integrals = np.array([
        np.trapezoid(K * np.sin(w * t), t) for w in omega_mid
    ])
    A_inf_vals = A_mid + (1.0 / omega_mid) * sin_integrals
    A_inf_ogilvie = float(np.median(A_inf_vals))

    return A_inf_endpoint, A_inf_trend_fit, A_inf_ogilvie


def _reconstruct(
    omega: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    A_infinity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct B(omega) and A(omega) from K(t) and A_infinity.

    B_rec(omega) = integral_0^inf K(t) cos(omega*t) dt
    A_rec(omega) = A_inf - (1/omega) * integral_0^inf K(t) sin(omega*t) dt

    Returns
    -------
    (B_reconstructed, A_reconstructed) : tuple of np.ndarray
    """
    B_rec = np.array([np.trapezoid(K * np.cos(w * t), t) for w in omega])
    sin_int = np.array([np.trapezoid(K * np.sin(w * t), t) for w in omega])
    A_rec = A_infinity - (1.0 / omega) * sin_int
    return B_rec, A_rec


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def compute_radiation_kernel(
    hydro: HydrodynamicCoefficients,
    t_max: float = DEFAULT_T_MAX,
    dt: float = DEFAULT_DT,
    n_interp: int = DEFAULT_N_INTERP,
    taper_fraction: float = DEFAULT_TAPER_FRACTION,
    A_infinity_method: Literal[
        "endpoint", "trend_fit", "ogilvie_midrange"
    ] = "ogilvie_midrange",
) -> RadiationKernel:
    """
    Compute the radiation memory kernel K(t) and A_infinity from BEM data.

    Implements the Cummins/Ogilvie radiation model:

        K(t) = (2/pi) * integral_{omega_min}^{omega_max} B_tapered(omega) cos(omega*t) domega

    with a cosine taper applied to B(omega) near omega_max to reduce
    Gibbs-like ringing from the frequency truncation.

    The integration domain is [omega_min, omega_max].  The region
    [0, omega_min) is absent (not extrapolated, not set to zero — simply
    not included).  See module docstring LOW-FREQUENCY TREATMENT section.

    Parameters
    ----------
    hydro : HydrodynamicCoefficients
        Frequency-domain BEM coefficients from Module 2.2.
    t_max : float, optional
        Kernel duration [s].  Default 60.0.
    dt : float, optional
        Kernel time step [s].  Default 0.05.
    n_interp : int, optional
        Number of interpolation points on [omega_min, omega_max].
        Default 2000.
    taper_fraction : float, optional
        Fraction of the omega range tapered at the high-frequency end.
        Default 0.15.  Set to 0.0 to disable tapering.
    A_infinity_method : str, optional
        Method for estimating A_infinity.  One of:
        - 'endpoint'       : A_inf = A(omega_max).  Lower bound.
        - 'trend_fit'      : Power-law extrapolation.  Unreliable.
        - 'ogilvie_midrange': Ogilvie self-consistency, mid-range omega.
                              Selected default.
        Default 'ogilvie_midrange'.

    Returns
    -------
    RadiationKernel
        Frozen dataclass containing kernel, A_infinity, reconstruction,
        and all metadata.

    Notes
    -----
    The reconstruction quality is limited by the truncation of B(omega)
    at omega_max = 1.4 rad/s.  See module docstring for details.
    The trusted reconstruction range is omega in [0.45, 1.15] rad/s.
    """
    if t_max <= 0:
        raise ValueError(f"t_max must be positive, got {t_max}")
    if dt <= 0 or dt >= t_max:
        raise ValueError(f"dt must be in (0, t_max), got {dt}")
    if n_interp < 10:
        raise ValueError(f"n_interp must be >= 10, got {n_interp}")
    if not (0.0 <= taper_fraction < 1.0):
        raise ValueError(f"taper_fraction must be in [0, 1), got {taper_fraction}")

    omega = hydro.omega.copy()
    A = hydro.added_mass_heave.copy()
    B = hydro.radiation_damping_heave.copy()

    # Dense interpolation on [omega_min, omega_max]
    omega_dense = np.linspace(omega[0], omega[-1], n_interp)
    B_dense = np.interp(omega_dense, omega, B)

    # Apply cosine taper at high-frequency end
    taper = _cosine_taper(n_interp, taper_fraction)
    B_tapered = B_dense * taper

    # Time grid: t = 0, dt, 2*dt, ..., t_max
    t = np.arange(0.0, t_max + dt * 0.5, dt)

    # Compute kernel
    K = _compute_kernel_array(omega_dense, B_tapered, t)

    # Estimate A_infinity (all three methods)
    A_inf_endpoint, A_inf_trend_fit, A_inf_ogilvie = _estimate_A_infinity(
        omega, A, B, t, K
    )

    # Select A_infinity
    if A_infinity_method == "endpoint":
        A_inf = A_inf_endpoint
    elif A_infinity_method == "trend_fit":
        A_inf = A_inf_trend_fit
    else:  # ogilvie_midrange (default)
        A_inf = A_inf_ogilvie

    # Reconstruct B and A from kernel
    B_rec, A_rec = _reconstruct(omega, t, K, A_inf)

    # Irregular frequency estimate (geometry-based, from Capytaine formula)
    # For a vertical cylinder: omega_irr ~ pi * sqrt(g/draft) / (1 + pi*r/draft)
    # Use the stored value from the BEM run if available, else approximate.
    # Approximation: first_irregular_frequency_estimate is geometry-based.
    # We store the known value for the reference buoy.
    omega_irr = 2.0880  # rad/s — from Capytaine parallelepiped formula

    return RadiationKernel(
        omega=omega,
        frequency_hz=hydro.frequency_hz.copy(),
        added_mass=A,
        radiation_damping=B,
        A_infinity=A_inf,
        A_infinity_method=A_infinity_method,
        A_infinity_endpoint=A_inf_endpoint,
        A_infinity_trend_fit=A_inf_trend_fit,
        A_infinity_ogilvie=A_inf_ogilvie,
        time=t,
        kernel=K,
        added_mass_reconstructed=A_rec,
        radiation_damping_reconstructed=B_rec,
        omega_min=float(omega[0]),
        omega_max=float(omega[-1]),
        omega_irr_estimate=omega_irr,
        t_max=float(t[-1]),
        dt=dt,
        n_interp=n_interp,
        taper_fraction=taper_fraction,
        water_depth=hydro.water_depth,
        rho=hydro.rho,
        radius=hydro.radius,
        draft=hydro.draft,
        n_panels=hydro.n_panels,
        capytaine_version=hydro.capytaine_version,
    )
