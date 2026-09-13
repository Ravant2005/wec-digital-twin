"""
direction.py — Directional spreading functions for the WEC Digital Twin.

Implements a cos²-type directional distribution D(θ) that spreads wave
energy around a mean propagation direction.

Coordinate and direction convention
-------------------------------------
    x = East,  y = North
    θ = 0      → propagation toward North
    θ = π/2    → propagation toward East
    θ = π      → propagation toward South
    θ = 3π/2   → propagation toward West

    ERA5 mwd is the direction waves are propagating TOWARD (not from).
    Do NOT add π to ERA5 mwd values.

All angles are in radians internally.  Public functions that accept
directional inputs use radians unless the name explicitly says "_deg".

Directional spreading function
--------------------------------
The cos²-type distribution used here is:

    D(θ) = (2/π) · cos²(θ − θ_mean)    for |θ − θ_mean| ≤ π/2
    D(θ) = 0                             otherwise

This is the simplest normalized spreading function.  It integrates to 1
over [0, 2π) and is non-negative everywhere.

For a discrete directional grid with uniform spacing Δθ = 2π/N, the
discrete weights are:

    w_j = D(θ_j) · Δθ / Σ_j D(θ_j) · Δθ

which ensures Σ w_j = 1 exactly (numerical normalization).

References
----------
- Holthuijsen (2007), Waves in Oceanic and Coastal Waters, §6.3.
- ITTC Recommended Procedures 7.5-02-07-03.1.
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. cosine_squared_spreading
# ---------------------------------------------------------------------------

def cosine_squared_spreading(
    theta: np.ndarray,
    mean_direction: float,
) -> np.ndarray:
    """
    Evaluate the cos²-type directional spreading function D(θ) [rad⁻¹].

    Definition
    ----------
        D(θ) = (2/π) · cos²(θ − θ_mean)   for |Δθ| ≤ π/2  (mod 2π)
        D(θ) = 0                            otherwise

    where Δθ = (θ − θ_mean) wrapped to (−π, π].

    The function is normalized so that ∫₀²π D(θ) dθ = 1.

    Parameters
    ----------
    theta : array_like
        Evaluation angles [radians].  Any shape.
    mean_direction : float
        Mean propagation direction [radians].
        θ = 0 → North, θ = π/2 → East (x-East, y-North convention).

    Returns
    -------
    np.ndarray
        D(θ) [rad⁻¹].  Non-negative.  Same shape as theta.
    """
    theta = np.asarray(theta, dtype=float)
    # Wrap angular difference to (−π, π]
    delta = _wrap(theta - mean_direction)
    D = np.where(np.abs(delta) <= np.pi / 2.0, (2.0 / np.pi) * np.cos(delta) ** 2, 0.0)
    return D


# ---------------------------------------------------------------------------
# 2. directional_grid
# ---------------------------------------------------------------------------

def directional_grid(n_directions: int) -> np.ndarray:
    """
    Create a uniform directional grid over [0, 2π) [radians].

    The grid has no duplicate endpoint: the last point is
    2π − Δθ, not 2π.

    Parameters
    ----------
    n_directions : int
        Number of directions.  Must be >= 2.

    Returns
    -------
    np.ndarray
        Shape (n_directions,).  Uniformly spaced in [0, 2π).

    Raises
    ------
    ValueError
        If n_directions < 2.
    """
    if n_directions < 2:
        raise ValueError(f"n_directions must be >= 2, got {n_directions}.")
    return np.linspace(0.0, 2.0 * np.pi, n_directions, endpoint=False)


# ---------------------------------------------------------------------------
# 3. normalize_directional_distribution
# ---------------------------------------------------------------------------

def normalize_directional_distribution(
    theta: np.ndarray,
    D: np.ndarray,
) -> np.ndarray:
    """
    Numerically normalize a directional distribution so that ∫D(θ)dθ = 1.

    Uses the trapezoidal rule on a uniform periodic grid.  For a grid
    produced by directional_grid(), the spacing is Δθ = 2π/N and the
    integral is approximated as Σ D(θ_j) · Δθ.

    Parameters
    ----------
    theta : array_like
        Directional grid [radians].  Must be uniformly spaced.
    D : array_like
        Directional distribution values.  Must be non-negative.
        Same length as theta.

    Returns
    -------
    np.ndarray
        Normalized distribution D_norm such that Σ D_norm · Δθ ≈ 1.

    Raises
    ------
    ValueError
        If D contains negative values or integrates to zero.
    """
    theta = np.asarray(theta, dtype=float)
    D = np.asarray(D, dtype=float)

    if np.any(D < 0.0):
        raise ValueError("Directional distribution D must be non-negative.")

    dtheta = theta[1] - theta[0]
    integral = np.sum(D) * dtheta

    if integral <= 0.0:
        raise ValueError("Directional distribution integrates to zero; cannot normalize.")

    return D / integral


# ---------------------------------------------------------------------------
# 4. directional_weights
# ---------------------------------------------------------------------------

def directional_weights(
    theta: np.ndarray,
    mean_direction: float,
) -> np.ndarray:
    """
    Return normalized directional weights w_j such that Σ w_j = 1.

    The weights represent the fraction of spectral energy assigned to each
    direction.  They are computed from the cos²-type spreading function and
    normalized numerically so that the discrete sum is exactly 1.

    Parameters
    ----------
    theta : array_like
        Directional grid [radians].  Produced by directional_grid().
    mean_direction : float
        Mean propagation direction [radians].

    Returns
    -------
    np.ndarray
        Weights w_j ≥ 0, Σ w_j = 1.  Same shape as theta.
    """
    theta = np.asarray(theta, dtype=float)
    D = cosine_squared_spreading(theta, mean_direction)
    dtheta = 2.0 * np.pi / len(theta)   # uniform spacing for grid from directional_grid()
    weights = D * dtheta
    total = weights.sum()
    if total <= 0.0:
        # Fallback: uniform weights (should not happen for valid mean_direction)
        return np.ones(len(theta)) / len(theta)
    return weights / total


# ---------------------------------------------------------------------------
# 5. directional_spectrum
# ---------------------------------------------------------------------------

def directional_spectrum(
    f: np.ndarray,
    S_f: np.ndarray,
    theta: np.ndarray,
    mean_direction: float,
) -> np.ndarray:
    """
    Construct the 2-D directional variance density spectrum S(f, θ) [m²/Hz/rad].

    Definition
    ----------
        S(f, θ) = S(f) · D(θ)

    where D(θ) is the normalized cos²-type spreading function [rad⁻¹].

    Energy conservation
    -------------------
    For each frequency:

        ∫₀²π S(f, θ) dθ = S(f) · ∫₀²π D(θ) dθ = S(f) · 1 = S(f)

    The discrete equivalent uses normalized weights w_j (Σ w_j = 1):

        Σ_j S(f_i, θ_j) · Δθ = S(f_i)   (approximately)

    Discretization note
    -------------------
    The returned array S[i, j] = S_f[i] · D(θ_j).  To recover the 1-D
    spectrum from the 2-D array, integrate over θ:

        S_f_reconstructed[i] = Σ_j S[i, j] · Δθ

    This is tested explicitly in tests/test_direction.py.

    Parameters
    ----------
    f : array_like
        Frequency array [Hz].  Shape (N_f,).
    S_f : array_like
        1-D variance density spectrum [m²/Hz].  Shape (N_f,).
    theta : array_like
        Directional grid [radians].  Shape (N_θ,).  From directional_grid().
    mean_direction : float
        Mean propagation direction [radians].

    Returns
    -------
    np.ndarray
        S(f, θ) [m²/Hz/rad].  Shape (N_f, N_θ).
    """
    f = np.asarray(f, dtype=float)
    S_f = np.asarray(S_f, dtype=float)
    theta = np.asarray(theta, dtype=float)

    D = cosine_squared_spreading(theta, mean_direction)   # shape (N_θ,)
    dtheta = 2.0 * np.pi / len(theta)
    D_norm = normalize_directional_distribution(theta, D)  # ∫D dθ = 1

    # S[i, j] = S_f[i] * D_norm[j]
    # Broadcasting: S_f[:, None] * D_norm[None, :]
    return S_f[:, np.newaxis] * D_norm[np.newaxis, :]


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _wrap(angle: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to (−π, π]."""
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi
