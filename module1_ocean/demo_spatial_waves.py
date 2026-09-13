"""
demo_spatial_waves.py — Directional irregular wave field demonstration.

Demonstrates a directional JONSWAP wave field at a fixed point and over
a small spatial domain.

NOTE: depth = 50 m is a demonstration value only.
      It does NOT represent the actual Goa deployment depth.

Run from the project root:
    python -m module1_ocean.demo_spatial_waves
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from module1_ocean.spectrum import jonswap, significant_wave_height_from_spectrum
from module1_ocean.waves import frequency_grid, significant_wave_height_from_timeseries
from module1_ocean.direction import (
    directional_grid,
    cosine_squared_spreading,
    directional_weights,
)
from module1_ocean.spatial_waves import synthesize_spatial_surface

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
HS         = 1.0
TP         = 10.0
GAMMA      = 3.3
DEPTH      = 50.0          # demonstration depth only
MEAN_DIR_DEG = 270.0       # westerly — waves propagating toward West
MEAN_DIR   = np.deg2rad(MEAN_DIR_DEG)
N_DIR      = 24
SEED       = 42

N_F   = 128
F     = frequency_grid(0.02, 0.5, N_F)
S_F   = jonswap(F, HS, TP, gamma=GAMMA)

DT    = 0.5
T     = np.arange(0.0, 3600.0, DT)

OUTPUT_DIR = "outputs"


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    Hs_spectral = significant_wave_height_from_spectrum(F, S_F)

    # --- Fixed-point time series at origin ---
    eta_origin = synthesize_spatial_surface(
        0.0, 0.0, T, F, S_F, MEAN_DIR, DEPTH, N_DIR, seed=SEED
    )
    Hs_ts = significant_wave_height_from_timeseries(eta_origin)

    print()
    print("=" * 62)
    print("  Directional Wave Field — Development Sea State")
    print("  NOTE: depth=50 m is a demonstration value only")
    print("=" * 62)
    print(f"  Target Hs          = {HS:.3f} m")
    print(f"  Spectral Hs        = {Hs_spectral:.6f} m")
    print(f"  Fixed-point Hs     = {Hs_ts:.6f} m")
    print(f"  Mean elevation     = {np.mean(eta_origin):+.6f} m")
    print(f"  Std(η)             = {np.std(eta_origin):.6f} m")
    print(f"  Min η              = {np.min(eta_origin):.4f} m")
    print(f"  Max η              = {np.max(eta_origin):.4f} m")
    print(f"  Mean direction     = {MEAN_DIR_DEG:.1f}° (toward West)")
    print(f"  Depth              = {DEPTH} m (demonstration only)")
    print(f"  N_directions       = {N_DIR}")
    print(f"  N_frequencies      = {N_F}")
    print(f"  Duration           = {T[-1]:.0f} s")

    # Directional normalization check
    theta = directional_grid(360)
    D = cosine_squared_spreading(theta, MEAN_DIR)
    dtheta = 2.0 * np.pi / 360
    norm_err = abs(np.sum(D) * dtheta - 1.0)
    print(f"  Directional norm error = {norm_err:.2e}")

    # Energy conservation check
    w = directional_weights(directional_grid(N_DIR), MEAN_DIR)
    energy_err = abs(w.sum() - 1.0)
    print(f"  Directional weight sum error = {energy_err:.2e}")
    print("=" * 62)
    print()

    # --- Plot A: Directional spreading ---
    theta_plot = directional_grid(360)
    D_plot = cosine_squared_spreading(theta_plot, MEAN_DIR)

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(5, 5))
    # Project convention: θ=0 → North, clockwise positive.
    # set_theta_zero_location("N") + set_theta_direction(-1) puts matplotlib
    # into exactly this convention, so theta_plot is used directly.
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)   # clockwise
    ax.plot(theta_plot, D_plot, color="#1f77b4", linewidth=1.5)
    ax.fill(theta_plot, D_plot, alpha=0.2, color="#1f77b4")
    # Explicit compass labels: N=0°, E=90°, S=180°, W=270° (clockwise from North)
    ax.set_thetagrids([0, 90, 180, 270], labels=["N\n0°", "E\n90°", "S\n180°", "W\n270°"])
    ax.set_title(
        f"cos² Directional Spreading\nMean direction = {MEAN_DIR_DEG:.0f}° (toward West)",
        pad=15
    )
    fig.tight_layout()
    path_a = os.path.join(OUTPUT_DIR, "spatial_directional_spreading.png")
    fig.savefig(path_a, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {path_a}")

    # --- Plot B: Fixed-point time series (first 300 s) ---
    window = T <= 300.0
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(T[window], eta_origin[window], color="#1f77b4", linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Time  t  [s]")
    ax.set_ylabel("η(0,0,t)  [m]")
    ax.set_title(
        f"Fixed-point surface elevation  (Hs={HS} m, Tp={TP} s, "
        f"dir={MEAN_DIR_DEG:.0f}°, d={DEPTH} m)\n"
        f"Hs_est = {Hs_ts:.3f} m  |  spectral Hs = {Hs_spectral:.3f} m"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path_b = os.path.join(OUTPUT_DIR, "spatial_timeseries_origin.png")
    fig.savefig(path_b, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {path_b}")

    # --- Plot C: Instantaneous 2-D surface snapshot ---
    # Small domain: ±2 wavelengths around origin
    # Wavelength at fp: λ ≈ 2π/k(fp)
    from module1_ocean.dispersion import solve_wave_number, wavelength_from_k
    k_fp = float(solve_wave_number(1.0 / TP, depth_m=DEPTH))
    lam_fp = float(wavelength_from_k(k_fp))
    domain = 2.0 * lam_fp

    nx, ny = 60, 60
    x_arr = np.linspace(-domain, domain, nx)
    y_arr = np.linspace(-domain, domain, ny)

    t_snap = np.array([0.0])
    rng = np.random.default_rng(SEED)
    phases_snap = rng.uniform(0.0, 2.0 * np.pi, (N_F, N_DIR))

    eta_2d = np.zeros((ny, nx))
    for iy, yv in enumerate(y_arr):
        for ix, xv in enumerate(x_arr):
            eta_2d[iy, ix] = synthesize_spatial_surface(
                xv, yv, t_snap, F, S_F, MEAN_DIR, DEPTH, N_DIR, phases=phases_snap
            )[0]

    fig, ax = plt.subplots(figsize=(7, 6))
    vmax = max(abs(eta_2d.min()), abs(eta_2d.max()))
    im = ax.pcolormesh(
        x_arr, y_arr, eta_2d,
        cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto"
    )
    plt.colorbar(im, ax=ax, label="η  [m]")
    ax.set_xlabel("x (East)  [m]")
    ax.set_ylabel("y (North)  [m]")
    ax.set_title(
        f"Instantaneous surface  t=0 s\n"
        f"Hs={HS} m, Tp={TP} s, dir={MEAN_DIR_DEG:.0f}°, d={DEPTH} m (demo)"
    )
    ax.set_aspect("equal")
    path_c = os.path.join(OUTPUT_DIR, "spatial_snapshot_2d.png")
    fig.savefig(path_c, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {path_c}")


if __name__ == "__main__":
    main()
