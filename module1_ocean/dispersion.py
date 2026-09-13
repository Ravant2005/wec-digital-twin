"""
dispersion.py — Finite-depth linear wave dispersion relation.

Solves the full linear dispersion relation:

    ω² = g k tanh(k d)

where:
    ω = 2π f   [rad/s]  — angular frequency
    k          [rad/m]  — wave number
    d          [m]      — water depth
    g          [m/s²]   — gravitational acceleration

Units throughout:
    f   : Hz
    ω   : rad/s
    k   : rad/m
    d   : m
    c   : m/s  (phase velocity)
    Cg  : m/s  (group velocity)
    λ   : m    (wavelength)

The deep-water approximation k ≈ ω²/g is provided separately for
comparison only.  All public functions use the full relation by default.

References
----------
- Dean & Dalrymple (1991), Water Wave Mechanics for Engineers and Scientists.
- Holthuijsen (2007), Waves in Oceanic and Coastal Waters.
"""

import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_array(f) -> np.ndarray:
    """Return f as a float64 ndarray, preserving scalar vs array distinction."""
    return np.asarray(f, dtype=float)


def _validate_depth(depth_m: float) -> None:
    if depth_m <= 0.0:
        raise ValueError(f"depth_m must be positive, got {depth_m}")


def _validate_frequencies(f: np.ndarray) -> None:
    if np.any(f < 0.0):
        raise ValueError("All frequencies must be non-negative (f >= 0).")


def _solve_single(omega: float, d: float, g: float) -> float:
    """
    Solve ω² = g k tanh(k d) for a single positive ω using Brent's method.

    Bracket strategy
    ----------------
    The deep-water approximation k_deep = ω²/g is a strict lower bound on
    the true k, because tanh(kd) ≤ 1 implies:

        g k_deep · tanh(k_deep · d) ≤ g k_deep = ω²

    so F(k_deep) = g k_deep tanh(k_deep d) − ω² ≤ 0.

    The upper bracket is 2 · k_deep, which satisfies F > 0 for all depths
    because at k = 2k_deep the factor of 2 more than compensates for
    tanh ≤ 1.  In very shallow water this may not hold, so we double the
    upper bound until F > 0.
    """
    k_deep = omega ** 2 / g  # lower bound

    def F(k):
        return g * k * np.tanh(k * d) - omega ** 2

    k_lo = 1e-12  # avoid exactly zero to keep F well-defined
    k_hi = 2.0 * k_deep

    # Expand upper bracket if needed (can happen in very shallow water)
    while F(k_hi) <= 0.0:
        k_hi *= 2.0

    return brentq(F, k_lo, k_hi, xtol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# 1. solve_wave_number
# ---------------------------------------------------------------------------

def solve_wave_number(
    f,
    depth_m: float,
    g: float = 9.81,
) -> np.ndarray:
    """
    Solve the full linear dispersion relation for wave number k [rad/m].

    Solves ω² = g k tanh(k d) for each frequency using Brent's method,
    which is guaranteed to converge to the unique non-negative root.

    Parameters
    ----------
    f : float or array_like
        Wave frequency [Hz].  Must be non-negative.  f = 0 returns k = 0.
    depth_m : float
        Water depth [m].  Must be positive.
    g : float, optional
        Gravitational acceleration [m/s²].  Default 9.81.

    Returns
    -------
    np.ndarray
        Wave number k [rad/m].  Same shape as f.  Non-negative.

    Raises
    ------
    ValueError
        If depth_m <= 0 or any frequency is negative.
    """
    _validate_depth(depth_m)
    scalar_input = np.ndim(f) == 0
    f_arr = np.atleast_1d(_to_array(f))
    _validate_frequencies(f_arr)

    k = np.zeros_like(f_arr)
    for i, fi in enumerate(f_arr.flat):
        if fi == 0.0:
            k.flat[i] = 0.0
        else:
            omega = 2.0 * np.pi * fi
            k.flat[i] = _solve_single(omega, depth_m, g)

    return float(k[0]) if scalar_input else k


# ---------------------------------------------------------------------------
# 2. dispersion_residual
# ---------------------------------------------------------------------------

def dispersion_residual(
    f,
    k,
    depth_m: float,
    g: float = 9.81,
) -> np.ndarray:
    """
    Compute the dispersion residual R = g k tanh(k d) − ω².

    For a correctly solved k, R should be close to zero relative to ω².
    Used to verify numerical solutions.

    Parameters
    ----------
    f : float or array_like
        Frequency [Hz].
    k : float or array_like
        Wave number [rad/m].
    depth_m : float
        Water depth [m].
    g : float, optional
        Gravitational acceleration [m/s²].  Default 9.81.

    Returns
    -------
    np.ndarray
        Residual R = g k tanh(k d) − ω².  Same shape as f.
        Units: rad²/s².
    """
    f_arr = _to_array(f)
    k_arr = _to_array(k)
    omega = 2.0 * np.pi * f_arr
    return g * k_arr * np.tanh(k_arr * depth_m) - omega ** 2


# ---------------------------------------------------------------------------
# 3. phase_velocity
# ---------------------------------------------------------------------------

def phase_velocity(f, k) -> np.ndarray:
    """
    Compute linear wave phase velocity c = ω / k [m/s].

    For f = 0 (or k = 0), returns 0.0 rather than NaN.

    Parameters
    ----------
    f : float or array_like
        Frequency [Hz].
    k : float or array_like
        Wave number [rad/m].

    Returns
    -------
    np.ndarray
        Phase velocity [m/s].  Same shape as f.
    """
    f_arr = _to_array(f)
    k_arr = _to_array(k)
    omega = 2.0 * np.pi * f_arr
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.where(k_arr > 0.0, omega / k_arr, 0.0)
    return c


# ---------------------------------------------------------------------------
# 4. group_velocity
# ---------------------------------------------------------------------------

def group_velocity(f, k, depth_m: float, g: float = 9.81) -> np.ndarray:
    """
    Compute linear wave group velocity Cg = n · c [m/s].

    Uses the exact finite-depth expression:

        n  = ½ [1 + 2kd / sinh(2kd)]
        Cg = n · c = n · ω/k

    Limits:
        Deep water  (kd → ∞):  sinh(2kd) → ∞,  n → ½,  Cg → c/2
        Shallow water (kd → 0): sinh(2kd) → 2kd, n → 1,  Cg → c = √(gd)

    Numerical note
    --------------
    sinh(2kd) overflows for 2kd ≳ 710.  For kd > 350 the deep-water limit
    n = 0.5 is accurate to machine precision and is used directly.

    Parameters
    ----------
    f : float or array_like
        Frequency [Hz].
    k : float or array_like
        Wave number [rad/m].
    depth_m : float
        Water depth [m].
    g : float, optional
        Gravitational acceleration [m/s²].  Default 9.81.

    Returns
    -------
    np.ndarray
        Group velocity [m/s].  Same shape as f.  Non-negative.
    """
    _validate_depth(depth_m)
    f_arr = _to_array(f)
    k_arr = _to_array(k)
    omega = 2.0 * np.pi * f_arr

    c = np.where(k_arr > 0.0, omega / k_arr, 0.0)

    kd = k_arr * depth_m
    two_kd = 2.0 * kd

    # n = 0.5 * (1 + 2kd/sinh(2kd))
    # For large kd use deep-water limit n=0.5 to avoid overflow
    with np.errstate(over="ignore", invalid="ignore"):
        sinh_val = np.sinh(two_kd)
        ratio = np.where(two_kd < 700.0, two_kd / sinh_val, 0.0)

    n = 0.5 * (1.0 + ratio)
    # At k=0 (f=0), Cg=0
    cg = np.where(k_arr > 0.0, n * c, 0.0)
    return cg


# ---------------------------------------------------------------------------
# 5. wavelength_from_k
# ---------------------------------------------------------------------------

def wavelength_from_k(k) -> np.ndarray:
    """
    Compute wavelength λ = 2π / k [m].

    For k = 0 (zero frequency), returns np.inf, which is physically correct
    (infinite wavelength at zero frequency).

    Parameters
    ----------
    k : float or array_like
        Wave number [rad/m].  Must be non-negative.

    Returns
    -------
    np.ndarray
        Wavelength [m].  Same shape as k.
    """
    k_arr = _to_array(k)
    with np.errstate(divide="ignore"):
        return np.where(k_arr > 0.0, 2.0 * np.pi / k_arr, np.inf)


# ---------------------------------------------------------------------------
# 6. deep_water_wave_number
# ---------------------------------------------------------------------------

def deep_water_wave_number(f, g: float = 9.81) -> np.ndarray:
    """
    Analytical deep-water approximation: k_deep = ω² / g [rad/m].

    Valid when kd ≫ 1 (tanh(kd) ≈ 1).  As a rule of thumb, the deep-water
    approximation is accurate to within ~0.4% when d > λ/2.

    This function is provided for comparison and testing ONLY.
    Use solve_wave_number() for the general finite-depth solution.

    Parameters
    ----------
    f : float or array_like
        Frequency [Hz].  Must be non-negative.

    Returns
    -------
    np.ndarray
        Deep-water wave number [rad/m].  Same shape as f.
    """
    f_arr = _to_array(f)
    omega = 2.0 * np.pi * f_arr
    return omega ** 2 / g
