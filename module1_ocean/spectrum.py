"""
spectrum.py — Wave energy spectrum models for the WEC Digital Twin project.

Implements the Pierson-Moskowitz (PM) and JONSWAP one-dimensional variance
density spectra, parameterised directly from significant wave height (Hs) and
peak period (Tp).

All spectra are expressed as one-sided variance density spectra S(f) in the
frequency domain:

    S(f)  [m²/Hz],   f ≥ 0

The zeroth spectral moment m0 is related to Hs by:

    m0 = ∫₀^∞ S(f) df = Hs² / 16
    Hs = 4 √m0

Units throughout:
    f   : Hz  (cycles per second)
    S   : m²/Hz
    mₙ  : m² · Hzⁿ⁻¹  (m² for n=0)
    Hs  : m
    Tp  : s

IMPORTANT — fp and the spectral peak
--------------------------------------
For this f-domain formulation, the spectral peak of S(f) is at f = fp = 1/Tp
exactly.  Setting dS/df = 0 with the substitution x = fp/f reduces to x = 1,
confirming f_peak = fp.

Note: the omega-domain PM formulation (written in rad/s) has its peak at
omega_peak = (5/4)^(1/4) · omega_p.  That factor does NOT apply here and
must not be mixed with this f-domain form.

References
----------
- Pierson & Moskowitz (1964), J. Geophys. Res., 69(24), 5181–5190.
- Hasselmann et al. (1973), Dtsch. Hydrogr. Z. Suppl. A, 8(12).
- DNV-RP-C205 (2021), Environmental Conditions and Environmental Loads.
- ITTC Recommended Procedures 7.5-02-07-03.1.
"""

import numpy as np
from scipy import integrate


# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------

def _validate_hs_tp(Hs: float, Tp: float) -> None:
    """Raise ValueError for non-physical Hs or Tp."""
    if Hs <= 0:
        raise ValueError(f"Hs must be positive, got Hs={Hs}")
    if Tp <= 0:
        raise ValueError(f"Tp must be positive, got Tp={Tp}")


def _validate_frequencies(f: np.ndarray) -> np.ndarray:
    """
    Validate and return f as a float64 array.

    Raises ValueError if any frequency is negative.
    f = 0 is allowed; the spectrum returns zero there.
    """
    f = np.asarray(f, dtype=float)
    if np.any(f < 0):
        raise ValueError("All frequencies must be non-negative (f ≥ 0).")
    return f


# ---------------------------------------------------------------------------
# Pierson-Moskowitz spectrum
# ---------------------------------------------------------------------------

def pierson_moskowitz(f: np.ndarray, Hs: float, Tp: float) -> np.ndarray:
    """
    One-sided Pierson-Moskowitz variance density spectrum S(f) [m²/Hz].

    Formulation
    -----------
    The PM spectrum is parameterised directly from Hs and Tp using the
    ITTC / DNV convention:

        S(f) = (5/16) · Hs² · fp⁴ · f⁻⁵ · exp(−5/4 · (fp/f)⁴)

    where fp = 1/Tp is the peak period parameter.

    This prefactor (5/16) is derived by requiring the integral constraint:

        m0 = ∫₀^∞ S(f) df = Hs² / 16

    to be satisfied exactly, so that Hs = 4√m0 holds by construction.

    Derivation sketch:
        Let u = (fp/f)⁴.  Then the integral reduces to
        (5/16) · Hs² · (1/4) · ∫₀^∞ exp(−5u/4) du
        = (5/16) · Hs² · (1/4) · (4/5) = Hs²/16  ✓

    Note on fp vs. spectral peak
    ----------------------------
    For this f-domain formulation, the spectral peak is at f_peak = fp = 1/Tp
    exactly.  Setting dS/df = 0 with x = fp/f gives x = 1, i.e. f_peak = fp.
    The numerical peak_period(f, S) will therefore return a value close to Tp
    (within grid-discretisation error).

    Parameters
    ----------
    f : array_like
        Frequencies [Hz].  Must be non-negative.  S(f=0) = 0.
    Hs : float
        Significant wave height [m].  Must be > 0.
    Tp : float
        Peak period [s].  Must be > 0.  fp = 1/Tp.

    Returns
    -------
    np.ndarray
        Variance density spectrum S(f) [m²/Hz], same shape as f.
        Non-negative everywhere.  Zero at f = 0.
    """
    _validate_hs_tp(Hs, Tp)
    f = _validate_frequencies(f)

    fp = 1.0 / Tp
    S = np.zeros_like(f, dtype=float)
    mask = f > 0.0
    S[mask] = (
        (5.0 / 16.0)
        * Hs**2
        * fp**4
        * f[mask] ** (-5)
        * np.exp(-1.25 * (fp / f[mask]) ** 4)
    )
    return S


# ---------------------------------------------------------------------------
# JONSWAP spectrum
# ---------------------------------------------------------------------------

def jonswap(
    f: np.ndarray,
    Hs: float,
    Tp: float,
    gamma: float = 3.3,
) -> np.ndarray:
    """
    One-sided JONSWAP variance density spectrum S(f) [m²/Hz].

    Formulation
    -----------
    JONSWAP enhances the PM spectrum with a frequency-dependent peak
    amplification factor:

        S_J(f) = C · S_PM(f) · γ^r(f)

    where:
        S_PM(f)  — PM spectrum (this module, same Hs and Tp)
        γ        — peak enhancement factor (dimensionless, > 0)
        r(f)     = exp(−(f − fp)² / (2 σ² fp²))   [Gaussian bell]
        σ        = 0.07  for f ≤ fp
                 = 0.09  for f > fp
        fp       = 1/Tp

    The scaling constant C is chosen so that the integral constraint
    m0 = Hs²/16 is preserved:

        C = m0_PM / m0_J_raw

    where m0_J_raw = ∫ S_PM(f) · γ^r(f) df.

    This rescaling is necessary because the γ enhancement increases the
    total variance.  Without it, the reconstructed Hs would be larger than
    the target Hs.

    Note on γ = 1
    -------------
    When γ = 1, γ^r(f) = 1 for all f, so S_J(f) = S_PM(f) exactly
    (C = 1 trivially).  This is the correct limiting behaviour.

    Note on γ = 3.3
    ---------------
    γ = 3.3 is the conventional mean value from the original JONSWAP field
    campaign (North Sea, 1973).  It is used here as a development default
    only.  It has NOT been calibrated for the Goa coast.  Future work will
    estimate γ from the ERA5 Goa dataset.

    Parameters
    ----------
    f : array_like
        Frequencies [Hz].  Must be non-negative.  S(f=0) = 0.
    Hs : float
        Significant wave height [m].  Must be > 0.
    Tp : float
        Peak period [s].  Must be > 0.  fp = 1/Tp.
    gamma : float
        Peak enhancement factor.  Must be > 0.
        γ = 1  → reduces to PM spectrum.
        γ = 3.3 → conventional JONSWAP default (not Goa-calibrated).

    Returns
    -------
    np.ndarray
        Variance density spectrum S(f) [m²/Hz], same shape as f.
        Non-negative everywhere.  Zero at f = 0.
        Integral gives m0 = Hs²/16 (Hs preserved by construction).

    Raises
    ------
    ValueError
        If Hs ≤ 0, Tp ≤ 0, gamma ≤ 0, or any frequency is negative.
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got gamma={gamma}")
    _validate_hs_tp(Hs, Tp)
    f = _validate_frequencies(f)

    fp = 1.0 / Tp
    S_pm = pierson_moskowitz(f, Hs, Tp)

    # Gaussian peak enhancement exponent r(f)
    sigma = np.where(f <= fp, 0.07, 0.09)
    r = np.exp(-0.5 * ((f - fp) / (sigma * fp)) ** 2)

    S_raw = S_pm * gamma**r

    # Rescale to preserve Hs (i.e. m0 = Hs²/16)
    m0_pm = np.trapezoid(S_pm, f)
    m0_raw = np.trapezoid(S_raw, f)

    # If m0_raw is effectively zero (degenerate grid), skip rescaling
    if m0_raw > 0.0:
        S_raw *= m0_pm / m0_raw

    return S_raw


# ---------------------------------------------------------------------------
# Spectral moments
# ---------------------------------------------------------------------------

def spectral_moments(f: np.ndarray, S: np.ndarray) -> dict:
    """
    Compute the zeroth, first, and second spectral moments by numerical
    integration using the trapezoidal rule.

    The n-th spectral moment is defined as:

        mₙ = ∫₀^∞ fⁿ · S(f) df

    Units:
        m0 : m²          (variance)
        m1 : m² · Hz     (mean frequency weighted by energy)
        m2 : m² · Hz²    (mean square frequency weighted by energy)

    Parameters
    ----------
    f : array_like
        Frequencies [Hz].  Must be non-negative and monotonically increasing.
    S : array_like
        Variance density spectrum [m²/Hz].  Same length as f.

    Returns
    -------
    dict
        Keys: 'm0', 'm1', 'm2'.  All values are floats.
    """
    f = np.asarray(f, dtype=float)
    S = np.asarray(S, dtype=float)
    return {
        "m0": float(np.trapezoid(S, f)),
        "m1": float(np.trapezoid(f * S, f)),
        "m2": float(np.trapezoid(f**2 * S, f)),
    }


def significant_wave_height_from_spectrum(
    f: np.ndarray, S: np.ndarray
) -> float:
    """
    Reconstruct significant wave height from a variance density spectrum.

    Uses the standard definition:

        Hs = 4 · √m0 = 4 · √(∫₀^∞ S(f) df)

    Parameters
    ----------
    f : array_like
        Frequencies [Hz].
    S : array_like
        Variance density spectrum [m²/Hz].

    Returns
    -------
    float
        Reconstructed Hs [m].
    """
    m0 = spectral_moments(f, S)["m0"]
    return 4.0 * float(np.sqrt(m0))


# ---------------------------------------------------------------------------
# Spectrum validation
# ---------------------------------------------------------------------------

def validate_spectrum(
    f: np.ndarray, S: np.ndarray, target_Hs: float
) -> dict:
    """
    Validate a computed spectrum against a target significant wave height.

    Computes the numerical m0, reconstructs Hs = 4√m0, and reports the
    absolute and relative errors against target_Hs.

    The spectrum is NOT modified or renormalised inside this function.
    It is a read-only diagnostic.

    Parameters
    ----------
    f : array_like
        Frequencies [Hz].
    S : array_like
        Variance density spectrum [m²/Hz].
    target_Hs : float
        The Hs value used to construct the spectrum [m].

    Returns
    -------
    dict
        Keys:
          'm0'           — numerical zeroth moment [m²]
          'Hs_reconstructed' — 4√m0 [m]
          'target_Hs'    — the supplied target [m]
          'abs_error_m'  — |Hs_reconstructed − target_Hs| [m]
          'rel_error'    — abs_error / target_Hs  (dimensionless)
    """
    m0 = spectral_moments(f, S)["m0"]
    Hs_recon = 4.0 * float(np.sqrt(m0))
    abs_err = abs(Hs_recon - target_Hs)
    rel_err = abs_err / target_Hs if target_Hs > 0 else float("inf")
    return {
        "m0":               m0,
        "Hs_reconstructed": Hs_recon,
        "target_Hs":        target_Hs,
        "abs_error_m":      abs_err,
        "rel_error":        rel_err,
    }


# ---------------------------------------------------------------------------
# Peak frequency and period
# ---------------------------------------------------------------------------

def peak_frequency(f: np.ndarray, S: np.ndarray) -> float:
    """
    Return the frequency of maximum spectral density.

    This is the numerically observed spectral peak f_peak = f[argmax(S)].

    For this f-domain PM/JONSWAP formulation, f_peak = fp = 1/Tp exactly
    (analytically).  The numerical result will be within grid-discretisation
    error of fp.  See module docstring.

    Parameters
    ----------
    f : array_like
        Frequencies [Hz].  Must contain at least one positive value.
    S : array_like
        Variance density spectrum [m²/Hz].

    Returns
    -------
    float
        f_peak [Hz].  Always positive.

    Raises
    ------
    ValueError
        If no positive frequency exists in f.
    """
    f = np.asarray(f, dtype=float)
    S = np.asarray(S, dtype=float)

    # Restrict to f > 0 to avoid the trivially zero f=0 bin
    pos = f > 0.0
    if not np.any(pos):
        raise ValueError("No positive frequencies found in f.")

    idx = np.argmax(S[pos])
    return float(f[pos][idx])


def peak_period(f: np.ndarray, S: np.ndarray) -> float:
    """
    Return the period corresponding to the spectral peak frequency.

        Tp_spectral = 1 / f_peak

    For this f-domain formulation, the spectral peak is at fp = 1/Tp, so
    the returned value will be close to the input Tp (within grid-discretisation
    error).  See pierson_moskowitz() docstring for details.

    Parameters
    ----------
    f : array_like
        Frequencies [Hz].
    S : array_like
        Variance density spectrum [m²/Hz].

    Returns
    -------
    float
        Spectral peak period [s].  Always positive.

    Raises
    ------
    ValueError
        If f_peak ≤ 0 (should not occur for a valid spectrum).
    """
    fp_val = peak_frequency(f, S)
    if fp_val <= 0.0:
        raise ValueError(f"Peak frequency is non-positive: {fp_val}")
    return 1.0 / fp_val
