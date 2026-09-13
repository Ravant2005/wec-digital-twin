"""
demo_spectrum.py — Demonstration of PM and JONSWAP wave spectra.

Defines a representative development sea state, computes both spectra,
validates Hs reconstruction, and saves a comparison plot.

Run from the project root:
    python -m module1_ocean.demo_spectrum
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from module1_ocean.spectrum import (
    pierson_moskowitz,
    jonswap,
    validate_spectrum,
    peak_period,
)

# ---------------------------------------------------------------------------
# Sea state parameters
# ---------------------------------------------------------------------------
HS = 1.0   # m  — representative Goa development sea state
TP = 10.0  # s
FP = 1.0 / TP  # Hz — peak period parameter (shape parameter, not spectral peak)

# Fine frequency grid: resolves the spectral peak and keeps integration error < 0.1%
F = np.linspace(0.001, 2.0, 20_000)

OUTPUT_DIR = "outputs"
PLOT_FILE = os.path.join(OUTPUT_DIR, "spectrum_pm_vs_jonswap.png")


def main() -> None:
    # --- Compute spectra ---
    S_pm = pierson_moskowitz(F, HS, TP)
    S_j  = jonswap(F, HS, TP, gamma=3.3)

    # --- Validate ---
    val_pm = validate_spectrum(F, S_pm, HS)
    val_j  = validate_spectrum(F, S_j,  HS)

    Tp_pm = peak_period(F, S_pm)
    Tp_j  = peak_period(F, S_j)

    # --- Print report ---
    print()
    print("=" * 60)
    print("  Wave Spectrum Demo — Development Sea State")
    print("=" * 60)
    print(f"  Input parameters:")
    print(f"    Hs (target)  = {HS:.3f} m")
    print(f"    Tp (input)   = {TP:.3f} s   (fp = {FP:.4f} Hz)")
    print()
    print(f"  Pierson-Moskowitz (PM):")
    print(f"    Reconstructed Hs = {val_pm['Hs_reconstructed']:.6f} m")
    print(f"    Relative error   = {val_pm['rel_error']*100:.4f}%")
    print(f"    Spectral peak Tp = {Tp_pm:.4f} s")
    print(f"    (f-domain PM peak is at fp = 1/Tp = {FP:.4f} Hz by construction)")
    print()
    print(f"  JONSWAP (γ = 3.3, conventional default — not Goa-calibrated):")
    print(f"    Reconstructed Hs = {val_j['Hs_reconstructed']:.6f} m")
    print(f"    Relative error   = {val_j['rel_error']*100:.4f}%")
    print(f"    Spectral peak Tp = {Tp_j:.4f} s")
    print()
    print(f"  Frequency grid: {len(F)} points, "
          f"{F[0]:.3f}–{F[-1]:.1f} Hz")
    print("=" * 60)
    print()

    # --- Plot ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(F, S_pm, color="#1f77b4", linewidth=1.5,
            label=f"Pierson-Moskowitz  (Hs={HS} m, Tp={TP} s)")
    ax.plot(F, S_j,  color="#d62728", linewidth=1.5, linestyle="--",
            label=f"JONSWAP  (γ=3.3, conventional default)")

    # Mark fp = 1/Tp (the shape parameter; also the spectral peak for f-domain PM)
    ax.axvline(FP, color="gray", linewidth=0.9, linestyle=":",
               label=f"fp = 1/Tp = {FP:.3f} Hz  (shape param = spectral peak)")

    ax.set_xlim(0.0, 0.5)
    ax.set_xlabel("Frequency  f  [Hz]")
    ax.set_ylabel("Variance density  S(f)  [m²/Hz]")
    ax.set_title(
        f"PM vs JONSWAP Wave Spectra\n"
        f"Hs = {HS} m,  Tp = {TP} s  (development sea state)"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {PLOT_FILE}")


if __name__ == "__main__":
    main()
