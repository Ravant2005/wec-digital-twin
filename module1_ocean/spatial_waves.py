"""
spatial_waves.py — Directional irregular wave field synthesis.

Implements the linear random-phase directional wave model:

    η(x, y, t) = Σᵢ Σⱼ aᵢⱼ · cos(2π fᵢ t − kᵢ [x sin θⱼ + y cos θⱼ] + φᵢⱼ)

where:
    fᵢ        — frequency components [Hz]
    θⱼ        — propagation directions [rad]
    kᵢ        — wave number for frequency fᵢ [rad/m], from dispersion relation
    aᵢⱼ       — component amplitude [m]
    φᵢⱼ       — independent random phases [rad]

Coordinate convention
----------------------
    x = East,  y = North
    θ = 0      → propagation toward North
    θ = π/2    → propagation toward East

    The spatial phase term is:  −kᵢ (x sin θⱼ + y cos θⱼ)

    This follows from the wave vector:
        k⃗ = kᵢ [sin θⱼ, cos θⱼ]   (East, North components)
    and the dot product k⃗ · r⃗ = kᵢ (x sin θⱼ + y cos θⱼ).

Direction convention
---------------------
    ERA5 mwd is the direction waves propagate TOWARD.
    Do NOT add π to ERA5 mwd values.

Amplitude formula
-----------------
    aᵢⱼ = √(2 · S(fᵢ) · wⱼ · Δf)

where wⱼ are the normalized directional weights (Σⱼ wⱼ = 1) and Δf is
the uniform frequency spacing.  This ensures:

    Σᵢ Σⱼ aᵢⱼ²/2 = Σᵢ S(fᵢ) Δf · Σⱼ wⱼ = Σᵢ S(fᵢ) Δf ≈ m0 = Hs²/16

so the total variance — and therefore Hs — is preserved.

References
----------
- Dean & Dalrymple (1991), Water Wave Mechanics for Engineers and Scientists.
- Holthuijsen (2007), Waves in Oceanic and Coastal Waters.
"""

import numpy as np

from module1_ocean.waves import component_amplitudes, random_phases
from module1_ocean.dispersion import solve_wave_number
from module1_ocean.direction import directional_grid, directional_weights


def synthesize_spatial_surface(
    x: float,
    y: float,
    t: np.ndarray,
    f: np.ndarray,
    S_f: np.ndarray,
    mean_direction: float,
    depth_m: float,
    n_directions: int = 16,
    phases: np.ndarray = None,
    seed=None,
) -> np.ndarray:
    """
    Synthesize a directional irregular wave surface elevation η(x, y, t) [m].

    Implements the linear random-phase directional model:

        η(x,y,t) = Σᵢ Σⱼ aᵢⱼ · cos(2π fᵢ t − kᵢ[x sinθⱼ + y cosθⱼ] + φᵢⱼ)

    Parameters
    ----------
    x : float
        East coordinate of the evaluation point [m].
    y : float
        North coordinate of the evaluation point [m].
    t : array_like
        Time array [s].  1-D.
    f : array_like
        Frequency array [Hz].  Uniformly spaced, strictly increasing, > 0.
    S_f : array_like
        1-D variance density spectrum [m²/Hz].  Non-negative.  Same length as f.
    mean_direction : float
        Mean wave propagation direction [radians].
        θ = 0 → North, θ = π/2 → East.
        Use ERA5 mwd converted to radians directly (do NOT add π).
    depth_m : float
        Water depth [m].  Must be positive.
    n_directions : int, optional
        Number of directional bins.  Default 16.  Must be >= 2.
    phases : array_like or None, optional
        Pre-computed phase array [radians].  Shape (N_f, N_directions).
        If None, phases are generated internally using seed.
    seed : int or None, optional
        Seed for random phase generation.  Ignored if phases is supplied.

    Returns
    -------
    np.ndarray
        Surface elevation η(x, y, t) [m].  Shape (len(t),).

    Notes
    -----
    Memory: the computation uses arrays of shape (N_t, N_f, N_dir).
    For N_t=7200, N_f=128, N_dir=16 this is ~118 MB.  For larger grids,
    consider reducing N_f or N_dir, or chunking over time.

    Energy conservation
    -------------------
    The directional weights wⱼ satisfy Σⱼ wⱼ = 1, so:

        Σᵢ Σⱼ aᵢⱼ²/2 = Σᵢ S(fᵢ) Δf ≈ m0

    and Hs = 4√m0 is preserved at any fixed point for a long realization.
    """
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)
    S_f = np.asarray(S_f, dtype=float)

    if t.ndim != 1:
        raise ValueError("t must be a 1-D array.")

    # --- Frequency spacing (validated by component_amplitudes below) ---
    df = f[1] - f[0]

    # --- Directional grid and weights ---
    theta = directional_grid(n_directions)          # shape (N_dir,)
    w = directional_weights(theta, mean_direction)  # shape (N_dir,), sum=1

    # --- Wave numbers via finite-depth dispersion ---
    k = solve_wave_number(f, depth_m=depth_m)       # shape (N_f,)

    # --- Amplitudes: aᵢⱼ = sqrt(2 * S(fᵢ) * wⱼ * Δf) ---
    # S_f[i] * w[j] is the energy fraction for component (i,j).
    # Shape: (N_f, N_dir)
    S_2d = S_f[:, np.newaxis] * w[np.newaxis, :]   # energy per (freq, dir) bin
    # Validate S_f via component_amplitudes (checks uniformity, non-negativity)
    _ = component_amplitudes(f, S_f)               # raises if invalid
    a = np.sqrt(2.0 * S_2d * df)                   # shape (N_f, N_dir)

    # --- Random phases ---
    N_f, N_dir = len(f), n_directions
    if phases is not None:
        phases = np.asarray(phases, dtype=float)
        if phases.shape != (N_f, N_dir):
            raise ValueError(
                f"phases must have shape (N_f, N_dir) = ({N_f}, {N_dir}), "
                f"got {phases.shape}."
            )
    else:
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0.0, 2.0 * np.pi, (N_f, N_dir))

    # --- Spatial phase term: kᵢ (x sinθⱼ + y cosθⱼ) ---
    # shape (N_f, N_dir)
    spatial = k[:, np.newaxis] * (x * np.sin(theta)[np.newaxis, :]
                                  + y * np.cos(theta)[np.newaxis, :])

    # --- Synthesis ---
    # argument[t_idx, i, j] = 2π fᵢ t − spatial[i,j] + phases[i,j]
    # η[t_idx] = Σᵢ Σⱼ a[i,j] * cos(argument[t_idx, i, j])
    #
    # Shapes:
    #   2π f t  : (N_t, N_f, 1)  via t[:,None,None] * f[None,:,None]
    #   spatial : (1,   N_f, N_dir)
    #   phases  : (1,   N_f, N_dir)
    omega_t = 2.0 * np.pi * f[np.newaxis, :, np.newaxis] * t[:, np.newaxis, np.newaxis]
    arg = omega_t - spatial[np.newaxis, :, :] + phases[np.newaxis, :, :]
    eta = np.sum(a[np.newaxis, :, :] * np.cos(arg), axis=(1, 2))

    return eta
