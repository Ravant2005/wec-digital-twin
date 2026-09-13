"""
demo_dispersion.py — Finite-depth wave dispersion demonstration.

Compares wave numbers, wavelengths, phase velocities, and group velocities
across several water depths for a representative frequency range.

NOTE: The depths used here (5, 20, 50, 200 m) are numerical demonstration
cases only.  They do NOT represent the actual Goa deployment depth, which
has not yet been determined.

Run from the project root:
    python -m module1_ocean.demo_dispersion
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from module1_ocean.dispersion import (
    solve_wave_number,
    dispersion_residual,
    phase_velocity,
    group_velocity,
    wavelength_from_k,
    deep_water_wave_number,
)

DEPTHS   = [5.0, 20.0, 50.0, 200.0]   # m — demonstration depths only
COLORS   = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
F        = np.linspace(0.02, 0.5, 500)  # Hz
TP       = 10.0                          # s — peak period for reference
FP       = 1.0 / TP                      # Hz

OUTPUT_DIR = "outputs"


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Compute for all depths ---
    results = {}
    for d in DEPTHS:
        k   = solve_wave_number(F, depth_m=d)
        lam = wavelength_from_k(k)
        c   = phase_velocity(F, k)
        cg  = group_velocity(F, k, depth_m=d)
        R   = dispersion_residual(F, k, depth_m=d)
        results[d] = dict(k=k, lam=lam, c=c, cg=cg, R=R)

    k_deep = deep_water_wave_number(F)

    # --- Print table at fp = 0.1 Hz ---
    print()
    print("=" * 72)
    print("  Finite-Depth Dispersion — f = 0.1 Hz (Tp = 10 s)")
    print("  NOTE: depths are demonstration values, not the Goa deployment depth")
    print("=" * 72)
    print(f"  {'depth(m)':>10}  {'k(rad/m)':>10}  {'λ(m)':>10}  "
          f"{'c(m/s)':>8}  {'Cg(m/s)':>8}  {'Cg/c':>6}  {'|R|/ω²':>10}")
    print("  " + "-" * 70)

    omega_fp = 2.0 * np.pi * FP
    for d, col in zip(DEPTHS, COLORS):
        k_fp  = float(solve_wave_number(FP, depth_m=d))
        lam_fp = float(wavelength_from_k(k_fp))
        c_fp  = float(phase_velocity(FP, k_fp))
        cg_fp = float(group_velocity(FP, k_fp, depth_m=d))
        R_fp  = float(dispersion_residual(FP, k_fp, depth_m=d))
        rel_R = abs(R_fp) / omega_fp**2
        print(f"  {d:>10.1f}  {k_fp:>10.6f}  {lam_fp:>10.3f}  "
              f"{c_fp:>8.3f}  {cg_fp:>8.3f}  {cg_fp/c_fp:>6.4f}  {rel_R:>10.2e}")

    k_deep_fp = float(deep_water_wave_number(FP))
    lam_deep  = float(wavelength_from_k(k_deep_fp))
    print(f"  {'deep-water':>10}  {k_deep_fp:>10.6f}  {lam_deep:>10.3f}  "
          f"{'(approx)':>8}  {'':>8}  {'':>6}  {'':>10}")
    print("=" * 72)

    # --- Max residual across all depths ---
    max_rel_R = max(
        np.max(np.abs(results[d]["R"]) / (2.0 * np.pi * F) ** 2)
        for d in DEPTHS
    )
    print(f"\n  Max relative dispersion residual across all depths: {max_rel_R:.2e}")
    print()

    # --- Plot 1: k vs frequency ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for d, col in zip(DEPTHS, COLORS):
        ax.plot(F, results[d]["k"], color=col, linewidth=1.4,
                label=f"d = {d:.0f} m")
    ax.plot(F, k_deep, color="black", linewidth=1.0, linestyle="--",
            label="deep-water approx")
    ax.axvline(FP, color="gray", linewidth=0.8, linestyle=":",
               label=f"fp = {FP:.2f} Hz (Tp={TP} s)")
    ax.set_xlabel("Frequency  f  [Hz]")
    ax.set_ylabel("Wave number  k  [rad/m]")
    ax.set_title("Wave Number vs Frequency\n(demonstration depths)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Plot 2: wavelength vs frequency ---
    ax = axes[1]
    for d, col in zip(DEPTHS, COLORS):
        ax.plot(F, results[d]["lam"], color=col, linewidth=1.4,
                label=f"d = {d:.0f} m")
    lam_deep = wavelength_from_k(k_deep)
    ax.plot(F, lam_deep, color="black", linewidth=1.0, linestyle="--",
            label="deep-water approx")
    ax.axvline(FP, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Frequency  f  [Hz]")
    ax.set_ylabel("Wavelength  λ  [m]")
    ax.set_title("Wavelength vs Frequency\n(demonstration depths)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, "dispersion_k_wavelength.png")
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {path1}")

    # --- Plot 3: phase and group velocity ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    for d, col in zip(DEPTHS, COLORS):
        ax.plot(F, results[d]["c"], color=col, linewidth=1.4,
                label=f"d = {d:.0f} m")
    ax.set_xlabel("Frequency  f  [Hz]")
    ax.set_ylabel("Phase velocity  c  [m/s]")
    ax.set_title("Phase Velocity vs Frequency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for d, col in zip(DEPTHS, COLORS):
        ax.plot(F, results[d]["cg"], color=col, linewidth=1.4,
                label=f"d = {d:.0f} m")
    ax.set_xlabel("Frequency  f  [Hz]")
    ax.set_ylabel("Group velocity  Cg  [m/s]")
    ax.set_title("Group Velocity vs Frequency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, "dispersion_velocities.png")
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {path2}")


if __name__ == "__main__":
    main()
