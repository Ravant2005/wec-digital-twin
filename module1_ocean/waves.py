"""
waves.py — Irregular wave time-series synthesis for the WEC Digital Twin.

Implements the linear random-phase superposition model for a 1-D surface
elevation time series at a single spatial point:

    η(t) = Σᵢ aᵢ · cos(2π fᵢ t + φᵢ)

where:
    fᵢ   — discrete frequency components [Hz]
    aᵢ   — component amplitudes [m],  aᵢ = √(2 S(fᵢ) Δf)
    φᵢ   — independent random phases, φᵢ ~ Uniform(0, 2π)
    S(f) — one-sided variance density spectrum [m²/Hz]
    Δf   — uniform frequency spacing [Hz]

This is the standard linear irregular-wave model (Dean & Dalrymple, 1991;
Tucker & Pitt, 2001).  It assumes:
    - deep water (no dispersion relation needed at this stage)
    - linear superposition (small-amplitude waves)
    - stationary sea state
    - ergodic process (time average ≈ ensemble average for long records)

Units throughout:
    f    : Hz  (cycles per second, NOT rad/s)
    Δf   : Hz
    S(f) : m²/Hz
    aᵢ   : m
    t    : s
    η(t) : m
    Hs   : m

IMPORTANT — angular frequency convention:
    The synthesis uses 2π fᵢ t, NOT ωᵢ t, because fᵢ is in Hz.
    Do not substitute ω = 2π f without adjusting the spectrum units.

References
----------
- Dean & Dalrymple (1991), Water Wave Mechanics for Engineers and Scientists.
- Tucker & Pitt (2001), Waves in Ocean Engineering.
- ITTC Recommended Procedures 7.5-02-07-03.1.
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Frequency grid
# ---------------------------------------------------------------------------

def frequency_grid(
    f_min: float,
    f_max: float,
    n_components: int,
) -> np.ndarray:
    """
    Return a uniformly spaced 1-D frequency array [Hz].

    Parameters
    ----------
    f_min : float
        Lowest frequency [Hz].  Must be > 0.
    f_max : float
        Highest frequency [Hz].  Must be > f_min.
    n_components : int
        Number of frequency components.  Must be >= 2.

    Returns
    -------
    np.ndarray
        Shape (n_components,), dtype float64.
        Strictly increasing, uniformly spaced, all values in (0, ∞).

    Raises
    ------
    ValueError
        If f_min <= 0, f_max <= f_min, or n_components < 2.
    """
    if f_min <= 0.0:
        raise ValueError(f"f_min must be positive, got f_min={f_min}")
    if f_max <= f_min:
        raise ValueError(
            f"f_max must be greater than f_min, got f_min={f_min}, f_max={f_max}"
        )
    if n_components < 2:
        raise ValueError(
            f"n_components must be >= 2, got n_components={n_components}"
        )
    return np.linspace(f_min, f_max, n_components)


# ---------------------------------------------------------------------------
# 2. Component amplitudes
# ---------------------------------------------------------------------------

def component_amplitudes(f: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    Compute discrete wave component amplitudes from a variance density spectrum.

    Uses the standard discrete-spectrum relationship:

        aᵢ = √(2 · S(fᵢ) · Δf)

    where Δf is the uniform frequency spacing.  This ensures that the total
    variance of the synthesized signal equals the spectral variance:

        Σ aᵢ²/2 = Σ S(fᵢ) Δf ≈ ∫ S(f) df = m0 = Hs²/16

    Parameters
    ----------
    f : array_like
        Frequency array [Hz].  Must be strictly increasing and uniformly
        spaced.  All values must be positive.
    S : array_like
        Variance density spectrum [m²/Hz].  Must be non-negative.
        Same length as f.

    Returns
    -------
    np.ndarray
        Component amplitudes [m].  Shape matches f.  Non-negative.

    Raises
    ------
    ValueError
        If f and S have different lengths, f is not strictly increasing,
        spacing is not uniform (relative variation > 1e-6), or S contains
        negative values.
    """
    f = np.asarray(f, dtype=float)
    S = np.asarray(S, dtype=float)

    if f.ndim != 1 or S.ndim != 1:
        raise ValueError("f and S must be 1-D arrays.")
    if len(f) != len(S):
        raise ValueError(
            f"f and S must have the same length, got {len(f)} and {len(S)}."
        )
    if len(f) < 2:
        raise ValueError("f must contain at least 2 elements.")

    diffs = np.diff(f)
    if np.any(diffs <= 0.0):
        raise ValueError("f must be strictly increasing.")

    # Check uniformity: relative variation in spacing must be negligible
    df_mean = diffs.mean()
    rel_variation = (diffs.max() - diffs.min()) / df_mean
    if rel_variation > 1e-6:
        raise ValueError(
            f"Frequency spacing must be uniform. "
            f"Relative spacing variation = {rel_variation:.2e} (tolerance: 1e-6). "
            f"Use frequency_grid() to generate a uniform grid."
        )

    if np.any(S < 0.0):
        raise ValueError("Spectral density S must be non-negative everywhere.")

    df = df_mean
    return np.sqrt(2.0 * S * df)


# ---------------------------------------------------------------------------
# 3. Random phases
# ---------------------------------------------------------------------------

def random_phases(n_components: int, seed=None) -> np.ndarray:
    """
    Generate independent random phases φᵢ ~ Uniform(0, 2π) [radians].

    Parameters
    ----------
    n_components : int
        Number of phase values to generate.  Must be >= 1.
    seed : int or None, optional
        Seed for the random number generator.  When provided, the output is
        fully deterministic and reproducible.  When None, a random seed is
        used (non-reproducible).

    Returns
    -------
    np.ndarray
        Shape (n_components,), dtype float64.
        Values in [0, 2π).

    Raises
    ------
    ValueError
        If n_components < 1.

    Notes
    -----
    Uses numpy.random.default_rng (Generator API) rather than the legacy
    global random state, so calls to this function do not affect or depend
    on any external random state.
    """
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}.")
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 2.0 * np.pi, n_components)


# ---------------------------------------------------------------------------
# 4. Surface elevation synthesis
# ---------------------------------------------------------------------------

def synthesize_surface_elevation(
    t: np.ndarray,
    f: np.ndarray,
    S: np.ndarray,
    phases: np.ndarray = None,
    seed=None,
) -> np.ndarray:
    """
    Synthesize an irregular wave surface elevation time series η(t) [m].

    Implements the linear random-phase superposition model:

        η(t) = Σᵢ aᵢ · cos(2π fᵢ t + φᵢ)

    where aᵢ = √(2 S(fᵢ) Δf) and φᵢ are independent uniform random phases.

    The synthesis is fully vectorized: the phase matrix
    θ[j, i] = 2π fᵢ tⱼ + φᵢ  has shape (N_t, N_f), and η = sum over i.
    No Python loop over time steps is used.

    Parameters
    ----------
    t : array_like
        Time array [s].  1-D, any length.  Not required to start at zero.
    f : array_like
        Frequency array [Hz].  Must be strictly increasing and uniformly
        spaced.  All values must be positive.
    S : array_like
        Variance density spectrum [m²/Hz].  Must be non-negative.
        Same length as f.
    phases : array_like or None, optional
        Pre-computed phase array [radians].  If supplied, must have the same
        length as f.  If None, phases are generated internally using seed.
    seed : int or None, optional
        Seed passed to random_phases() when phases=None.  Ignored if phases
        is supplied.

    Returns
    -------
    np.ndarray
        Surface elevation η(t) [m].  Shape (len(t),).

    Raises
    ------
    ValueError
        If array dimensions are inconsistent, S contains negative values,
        or the frequency grid is not uniformly spaced.

    Notes
    -----
    Memory: the intermediate phase matrix has shape (len(t), len(f)).
    For len(t) = 7200 and len(f) = 512 this is ~29 MB — acceptable for
    typical simulation durations.  For very long simulations, consider
    chunking over time.
    """
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    S = np.asarray(S, dtype=float)

    if t.ndim != 1:
        raise ValueError("t must be a 1-D array.")

    # component_amplitudes validates f and S (shape, uniformity, non-negativity)
    a = component_amplitudes(f, S)

    if phases is not None:
        phases = np.asarray(phases, dtype=float)
        if phases.shape != f.shape:
            raise ValueError(
                f"phases must have the same shape as f: "
                f"got phases.shape={phases.shape}, f.shape={f.shape}."
            )
    else:
        phases = random_phases(len(f), seed=seed)

    # Vectorized synthesis:
    #   theta[j, i] = 2π * f[i] * t[j] + phases[i]
    #   eta[j]      = Σᵢ a[i] * cos(theta[j, i])
    #
    # Broadcasting: t[:, None] has shape (N_t, 1), f[None, :] has shape (1, N_f)
    theta = 2.0 * np.pi * f[np.newaxis, :] * t[:, np.newaxis] + phases[np.newaxis, :]
    eta = np.sum(a[np.newaxis, :] * np.cos(theta), axis=1)

    return eta


# ---------------------------------------------------------------------------
# 5. Significant wave height from time series
# ---------------------------------------------------------------------------

def significant_wave_height_from_timeseries(eta: np.ndarray) -> float:
    """
    Estimate significant wave height from a surface elevation time series.

    Uses the standard deviation estimator:

        Hs_est ≈ 4 · std(η)

    This follows from the linear random-phase model: for a stationary
    Gaussian sea state with many independent components, the variance of η
    equals m0 = Hs²/16, so std(η) = √m0 and Hs = 4√m0 = 4·std(η).

    Statistical caveats
    -------------------
    This is a statistical estimate.  Its accuracy depends on:
    - Simulation duration: longer records reduce sampling variance.
      A minimum of ~100 peak periods (≥ 1000 s for Tp = 10 s) is recommended.
    - Number of frequency components: more components improve the Gaussian
      approximation of the sea surface.
    - Stationarity: the sea state must not change during the record.
    - Random phases: a single realization may deviate from the ensemble mean.
      The deviation is typically < 5% for T ≥ 3600 s and N ≥ 256 components.

    The input η is not detrended, smoothed, or normalized.  If η has a
    non-zero mean (e.g. due to a DC offset), std() naturally removes it.

    Parameters
    ----------
    eta : array_like
        Surface elevation time series [m].  Must contain at least 2 values.

    Returns
    -------
    float
        Estimated significant wave height [m].

    Raises
    ------
    ValueError
        If eta has fewer than 2 elements.
    """
    eta = np.asarray(eta, dtype=float)
    if eta.size < 2:
        raise ValueError("eta must contain at least 2 values.")
    return 4.0 * float(np.std(eta, ddof=0))


# ---------------------------------------------------------------------------
# 6. Surface statistics
# ---------------------------------------------------------------------------

def surface_statistics(eta: np.ndarray, t: np.ndarray = None) -> dict:
    """
    Compute basic descriptive statistics of a surface elevation time series.

    The input array is not modified.

    Parameters
    ----------
    eta : array_like
        Surface elevation time series [m].
    t : array_like or None, optional
        Time array [s].  If provided, the simulation duration is included
        in the output.

    Returns
    -------
    dict
        Keys:
          'mean_m'   — mean surface elevation [m]
          'std_m'    — standard deviation [m]
          'min_m'    — minimum elevation [m]
          'max_m'    — maximum elevation [m]
          'Hs_est_m' — estimated significant wave height = 4·std [m]
          'duration_s' — simulation duration [s]  (only if t is provided)
    """
    eta = np.asarray(eta, dtype=float)
    result = {
        "mean_m":   float(np.mean(eta)),
        "std_m":    float(np.std(eta, ddof=0)),
        "min_m":    float(np.min(eta)),
        "max_m":    float(np.max(eta)),
        "Hs_est_m": significant_wave_height_from_timeseries(eta),
    }
    if t is not None:
        t = np.asarray(t, dtype=float)
        result["duration_s"] = float(t[-1] - t[0])
    return result
