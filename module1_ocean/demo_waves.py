"""
demo_waves.py — Irregular wave surface elevation synthesis demonstration.

Generates a JONSWAP-based irregular wave time series for a representative
Goa development sea state, prints statistics, and saves plots.

Run from the project root:
    python -m module1_ocean.demo_waves
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from module1_ocean.spectrum import jonswap, significant_wave_height_from_spectrum
from module1_ocean.waves import (
    frequency_grid,
    synthesize_surface_elevation,
    surface_statistics,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
HS   = 1.0    # m   — target significant wave height
TP   = 10.0   # s   — peak period
GAMMA = 3.3   # —   — JONSWAP peak enhancement (conventional default, not Goa-calibrated)
SEED = 42     # —   — fixed seed for reproducibility

# Frequency grid: 512 components from 0.01 to 2.0 Hz
N_COMP = 512
F_MIN  = 0.01   # Hz
F_MAX  = 2.0    # Hz

# Time array: 1 hour at 0.5 s resolution
DT     = 0.5    # s
T_END  = 3600.0 # s

OUTPUT_DIR = "outputs"
PLOT_TIMESERIES = os.path.join(OUTPUT_DIR, "waves_surface_elevation.png")
PLOT_HISTOGRAM  = os.path.join(OUTPUT_DIR, "waves_elevation_histogram.png")


def main() -> None:
    # --- Frequency grid and spectrum ---
    f = frequency_grid(F_MIN, F_MAX, N_COMP)
    S = jonswap(f, HS, TP, gamma=GAMMA)
    Hs_spectral = significant_wave_height_from_spectrum(f, S)

    # --- Time array ---
    t = np.arange(0.0, T_END, DT)

    # --- Synthesize surface elevation ---
    eta = synthesize_surface_elevation(t, f, S, seed=SEED)

    # --- Statistics ---
    stats = surface_statistics(eta, t=t)

    # --- Print report ---
    print()
    print("=" * 60)
    print("  Irregular Wave Synthesis — Development Sea State")
    print("=" * 60)
    print(f"  Sea state parameters:")
    print(f"    Target Hs        = {HS:.3f} m")
    print(f"    Peak period Tp   = {TP:.3f} s")
    print(f"    JONSWAP gamma    = {GAMMA}  (conventional default)")
    print()
    print(f"  Frequency grid:")
    print(f"    Components       = {N_COMP}")
    print(f"    Range            = {F_MIN:.3f} – {F_MAX:.1f} Hz")
    print(f"    Spacing Δf       = {f[1]-f[0]:.5f} Hz")
    print()
    print(f"  Spectral Hs (from spectrum integral):")
    print(f"    Hs_spectral      = {Hs_spectral:.6f} m")
    print()
    print(f"  Simulation:")
    print(f"    Duration         = {stats['duration_s']:.0f} s  ({stats['duration_s']/3600:.2f} h)")
    print(f"    Time step        = {DT} s")
    print(f"    Samples          = {len(t)}")
    print(f"    Seed             = {SEED}")
    print()
    print(f"  Time-series statistics:")
    print(f"    Mean η           = {stats['mean_m']:+.6f} m")
    print(f"    Std(η)           = {stats['std_m']:.6f} m")
    print(f"    Min η            = {stats['min_m']:.4f} m")
    print(f"    Max η            = {stats['max_m']:.4f} m")
    print(f"    Hs_est = 4·std   = {stats['Hs_est_m']:.6f} m")
    print()
    rel_err = abs(stats['Hs_est_m'] - Hs_spectral) / Hs_spectral * 100
    print(f"  Hs agreement: spectral={Hs_spectral:.4f} m, "
          f"time-series={stats['Hs_est_m']:.4f} m, "
          f"error={rel_err:.3f}%")
    print("=" * 60)
    print()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Plot 1: Surface elevation time series ---
    # Show first 300 s for readability (30 peak periods)
    window = t <= 300.0
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t[window], eta[window], color="#1f77b4", linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Time  t  [s]")
    ax.set_ylabel("Surface elevation  η(t)  [m]")
    ax.set_title(
        f"Irregular Wave Surface Elevation  (Hs={HS} m, Tp={TP} s, "
        f"JONSWAP γ={GAMMA})\n"
        f"First 300 s shown  |  Hs_est = {stats['Hs_est_m']:.3f} m  "
        f"(spectral Hs = {Hs_spectral:.3f} m)"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_TIMESERIES, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {PLOT_TIMESERIES}")

    # --- Plot 2: Histogram of η(t) ---
    # For a linear Gaussian sea state, η should be approximately normally
    # distributed with mean ≈ 0 and std ≈ Hs/4.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(eta, bins=80, density=True, color="#1f77b4", alpha=0.7,
            label="η(t) histogram")

    # Overlay theoretical Gaussian N(0, std²)
    x = np.linspace(eta.min(), eta.max(), 500)
    sigma = stats["std_m"]
    gaussian = np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    ax.plot(x, gaussian, color="#d62728", linewidth=1.5,
            label=f"N(0, σ²)  σ={sigma:.3f} m")

    ax.set_xlabel("Surface elevation  η  [m]")
    ax.set_ylabel("Probability density  [m⁻¹]")
    ax.set_title(
        f"Distribution of η(t)  (Hs={HS} m, Tp={TP} s)\n"
        f"Linear sea state → approximately Gaussian"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_HISTOGRAM, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {PLOT_HISTOGRAM}")


if __name__ == "__main__":
    main()
