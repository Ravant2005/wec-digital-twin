"""
demo_radiation.py — Radiation memory model demonstration (Module 2.3A).

Produces:
  F1. A_infinity comparison: all three estimates vs A(omega) BEM data
  F2. Radiation damping B(omega): original vs reconstructed
  F3. Added mass A(omega): original vs reconstructed
  F4. Radiation kernel K(t)
  F5. Reconstruction relative error (B and A)
  F6. Numerical sensitivity: B RMSE vs t_max, taper_fraction

Output files (outputs/module2_hydro/):
  F1_A_infinity_comparison.png
  F2_B_reconstruction.png
  F3_A_reconstruction.png
  F4_radiation_kernel.png
  F5_reconstruction_error.png
  F6_sensitivity.png
  module2_radiation_summary.txt

Usage:
    python -m module2_wec.demo_radiation

This uses the TEMPORARY REFERENCE BUOY (r=5m, draft=10m, depth=50m).
It is NOT the final Goa deployment analysis.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from module2_wec.geometry import REFERENCE_BUOY
from module2_wec.hydrodynamic_coefficients import compute_hydrodynamic_coefficients
from module2_wec.radiation import compute_radiation_kernel, DEFAULT_TAPER_FRACTION

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "module2_hydro",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROD_RESOLUTION = (6, 24, 16)   # 672 panels

print("=" * 60)
print("Module 2.3A — Radiation Memory Model Demonstration")
print("=" * 60)
print(f"Geometry: TEMPORARY REFERENCE BUOY (not Goa deployment)")
print(f"  radius={REFERENCE_BUOY.radius} m, draft={REFERENCE_BUOY.draft} m, "
      f"depth={REFERENCE_BUOY.water_depth} m")
print()

# ---------------------------------------------------------------------------
# Compute BEM coefficients (production path)
# ---------------------------------------------------------------------------
print("Computing BEM coefficients (672 panels, 20 frequencies)...")
hydro = compute_hydrodynamic_coefficients(
    geometry=REFERENCE_BUOY,
    resolution=PROD_RESOLUTION,
    omega_min=0.2, omega_max=1.4, n_omega=20,
    progress_bar=False,
)
print("BEM complete.")

# ---------------------------------------------------------------------------
# Compute radiation kernel (default parameters)
# ---------------------------------------------------------------------------
print("Computing radiation kernel...")
kernel = compute_radiation_kernel(hydro)
print("Kernel complete.")
print()

omega = kernel.omega
A = kernel.added_mass
B = kernel.radiation_damping
t = kernel.time
K = kernel.kernel
A_rec = kernel.added_mass_reconstructed
B_rec = kernel.radiation_damping_reconstructed
errs = kernel.reconstruction_errors()

# Trusted range mask
trusted = (omega >= 0.45) & (omega <= 1.15)

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------
s = kernel.summary()
print(f"A_infinity estimates:")
print(f"  Endpoint (A[omega_max]):     {kernel.A_infinity_endpoint:>12.1f} kg  [lower bound]")
print(f"  Trend fit (1/omega^2):       {kernel.A_infinity_trend_fit:>12.1f} kg  [unreliable]")
print(f"  Ogilvie mid-range (SELECTED):{kernel.A_infinity_ogilvie:>12.1f} kg")
print()
print(f"Kernel: K(0) = {K[0]:.1f} N·s/m, duration = {t[-1]:.1f} s, dt = {kernel.dt} s")
print()
print(f"Reconstruction errors (trusted range {errs['trusted_omega_min']:.3f}–"
      f"{errs['trusted_omega_max']:.3f} rad/s):")
print(f"  B RMSE = {errs['B_rmse_trusted_pct']:.2f}%,  "
      f"B max rel = {errs['B_max_rel_trusted_pct']:.2f}%")
print(f"  A RMSE = {errs['A_rmse_trusted_pct']:.2f}%,  "
      f"A max rel = {errs['A_max_rel_trusted_pct']:.2f}%")
print()
print("NOTE: A reconstruction errors are dominated by A_inf uncertainty,")
print("      not by kernel quality.  B reconstruction is the primary metric.")

# ---------------------------------------------------------------------------
# Plot F1: A_infinity comparison
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(omega, A / 1e3, "b-o", markersize=5, linewidth=1.5, label="A(ω) BEM")
ax.axhline(kernel.A_infinity_endpoint / 1e3, color="orange", linestyle="--",
           linewidth=1.5, label=f"Endpoint: {kernel.A_infinity_endpoint/1e3:.1f}×10³ kg")
ax.axhline(kernel.A_infinity_trend_fit / 1e3, color="red", linestyle=":",
           linewidth=1.5, label=f"Trend fit: {kernel.A_infinity_trend_fit/1e3:.1f}×10³ kg")
ax.axhline(kernel.A_infinity_ogilvie / 1e3, color="green", linestyle="-.",
           linewidth=2.0, label=f"Ogilvie (selected): {kernel.A_infinity_ogilvie/1e3:.1f}×10³ kg")
ax.axvline(kernel.omega_irr_estimate, color="gray", linestyle="--", linewidth=1,
           alpha=0.6, label=f"ω_irr ≈ {kernel.omega_irr_estimate:.3f} rad/s")
ax.set_xlabel("Angular frequency ω [rad/s]")
ax.set_ylabel("Added mass [×10³ kg]")
ax.set_title("A_infinity estimates vs BEM added mass\n"
             "TEMPORARY reference buoy — A(ω) has NOT reached asymptote at ω_max")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3)
fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "F1_A_infinity_comparison.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved: {path}")

# ---------------------------------------------------------------------------
# Plot F2: B(omega) original vs reconstructed
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(omega, B / 1e3, "b-o", markersize=5, linewidth=1.5, label="B(ω) BEM (original)")
ax.plot(omega, B_rec / 1e3, "r--s", markersize=5, linewidth=1.5,
        label="B_rec(ω) from K(t)")
ax.axvspan(errs["trusted_omega_min"], errs["trusted_omega_max"],
           alpha=0.08, color="green", label="Trusted range")
ax.axvline(kernel.omega_irr_estimate, color="gray", linestyle="--", linewidth=1,
           alpha=0.6, label=f"ω_irr ≈ {kernel.omega_irr_estimate:.3f} rad/s")
ax.set_xlabel("Angular frequency ω [rad/s]")
ax.set_ylabel("Radiation damping [×10³ N·s/m]")
ax.set_title(f"B(ω) reconstruction from K(t)\n"
             f"Trusted range RMSE = {errs['B_rmse_trusted_pct']:.2f}%, "
             f"max rel = {errs['B_max_rel_trusted_pct']:.2f}%")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "F2_B_reconstruction.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved: {path}")

# ---------------------------------------------------------------------------
# Plot F3: A(omega) original vs reconstructed
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(omega, A / 1e3, "b-o", markersize=5, linewidth=1.5, label="A(ω) BEM (original)")
ax.plot(omega, A_rec / 1e3, "r--s", markersize=5, linewidth=1.5,
        label="A_rec(ω) from K(t) + A_inf")
ax.axhline(kernel.A_infinity / 1e3, color="green", linestyle="-.", linewidth=1.5,
           label=f"A_inf (Ogilvie) = {kernel.A_infinity/1e3:.1f}×10³ kg")
ax.axvspan(errs["trusted_omega_min"], errs["trusted_omega_max"],
           alpha=0.08, color="green", label="Trusted range")
ax.set_xlabel("Angular frequency ω [rad/s]")
ax.set_ylabel("Added mass [×10³ kg]")
ax.set_title(f"A(ω) reconstruction from K(t) and A_inf\n"
             f"Trusted range RMSE = {errs['A_rmse_trusted_pct']:.2f}% "
             f"(dominated by A_inf uncertainty)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "F3_A_reconstruction.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved: {path}")

# ---------------------------------------------------------------------------
# Plot F4: Radiation kernel K(t)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Full duration
axes[0].plot(t, K / 1e3, "b-", linewidth=1.2)
axes[0].axhline(0, color="k", linewidth=0.5)
axes[0].set_xlabel("Time t [s]")
axes[0].set_ylabel("K(t) [×10³ N·s/m]")
axes[0].set_title(f"Radiation kernel K(t) — full duration ({t[-1]:.0f} s)")
axes[0].grid(True, alpha=0.3)

# First 10 s (where kernel is significant)
mask10 = t <= 10.0
axes[1].plot(t[mask10], K[mask10] / 1e3, "b-", linewidth=1.5)
axes[1].axhline(0, color="k", linewidth=0.5)
axes[1].set_xlabel("Time t [s]")
axes[1].set_ylabel("K(t) [×10³ N·s/m]")
axes[1].set_title(f"Radiation kernel K(t) — first 10 s\nK(0) = {K[0]/1e3:.2f}×10³ N·s/m")
axes[1].grid(True, alpha=0.3)

fig.suptitle("Radiation impulse-response function K(t)\n"
             "K(t) = (2/π) ∫ B_tapered(ω) cos(ωt) dω", fontsize=10)
fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "F4_radiation_kernel.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved: {path}")

# ---------------------------------------------------------------------------
# Plot F5: Reconstruction relative error
# ---------------------------------------------------------------------------
B_rel = errs["B_rel_error"] * 100   # percent, NaN where B is tiny
A_rel = errs["A_rel_error"] * 100

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(omega, B_rel, "r-o", markersize=5, linewidth=1.5)
axes[0].axhline(0, color="k", linewidth=0.5)
axes[0].axvspan(errs["trusted_omega_min"], errs["trusted_omega_max"],
                alpha=0.1, color="green", label="Trusted range")
axes[0].set_xlabel("Angular frequency ω [rad/s]")
axes[0].set_ylabel("Relative error [%]")
axes[0].set_title(f"B(ω) reconstruction relative error\n"
                  f"Trusted RMSE = {errs['B_rmse_trusted_pct']:.2f}%")
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].plot(omega, A_rel, "b-o", markersize=5, linewidth=1.5)
axes[1].axhline(0, color="k", linewidth=0.5)
axes[1].axvspan(errs["trusted_omega_min"], errs["trusted_omega_max"],
                alpha=0.1, color="green", label="Trusted range")
axes[1].set_xlabel("Angular frequency ω [rad/s]")
axes[1].set_ylabel("Relative error [%]")
axes[1].set_title(f"A(ω) reconstruction relative error\n"
                  f"Trusted RMSE = {errs['A_rmse_trusted_pct']:.2f}% "
                  f"(A_inf uncertainty dominates)")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

fig.suptitle("Reconstruction errors — limited by B(ω) truncation at ω_max = 1.4 rad/s",
             fontsize=10)
fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "F5_reconstruction_error.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved: {path}")

# ---------------------------------------------------------------------------
# Plot F6: Sensitivity study
# ---------------------------------------------------------------------------
print("\nRunning sensitivity study (medium mesh for speed)...")
from module2_wec.hydrodynamic_coefficients import compute_hydrodynamic_coefficients as chc
hydro_med = chc(
    geometry=REFERENCE_BUOY, resolution=(3, 12, 8),
    omega_min=0.3, omega_max=1.2, n_omega=8, progress_bar=False,
)

t_maxes = [10, 20, 30, 45, 60]
tapers  = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

B_rmse_tmax  = []
B_rmse_taper = []

for tm in t_maxes:
    k = compute_radiation_kernel(hydro_med, t_max=float(tm), dt=0.05)
    B_rmse_tmax.append(k.reconstruction_errors()["B_rmse_trusted_pct"])

for tf in tapers:
    k = compute_radiation_kernel(hydro_med, taper_fraction=tf)
    B_rmse_taper.append(k.reconstruction_errors()["B_rmse_trusted_pct"])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].plot(t_maxes, B_rmse_tmax, "b-o", markersize=6, linewidth=1.5)
axes[0].set_xlabel("Kernel duration t_max [s]")
axes[0].set_ylabel("B RMSE in trusted range [%]")
axes[0].set_title("Sensitivity to kernel duration")
axes[0].grid(True, alpha=0.3)

axes[1].plot([t*100 for t in tapers], B_rmse_taper, "r-o", markersize=6, linewidth=1.5)
axes[1].axvline(DEFAULT_TAPER_FRACTION * 100, color="green", linestyle="--",
                linewidth=1.5, label=f"Default ({DEFAULT_TAPER_FRACTION*100:.0f}%)")
axes[1].set_xlabel("Taper fraction [%]")
axes[1].set_ylabel("B RMSE in trusted range [%]")
axes[1].set_title("Sensitivity to cosine taper fraction")
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

fig.suptitle("Numerical sensitivity — medium mesh (168 panels, 8 frequencies)", fontsize=10)
fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "F6_sensitivity.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved: {path}")

# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------
report_lines = [
    "=" * 65,
    "Module 2.3A — Radiation Memory Model Summary",
    "=" * 65,
    "",
    "IMPORTANT: TEMPORARY DEVELOPMENT BUOY — not the Goa deployment buoy.",
    "",
    "1. BEM input (Module 2.2, FROZEN)",
    f"   Capytaine version : {hydro.capytaine_version}",
    f"   Mesh resolution   : {PROD_RESOLUTION} = {hydro.n_panels} panels",
    f"   Frequency range   : {hydro.omega_min:.2f}–{hydro.omega_max:.2f} rad/s "
    f"({hydro.n_omega} points)",
    f"   omega_irr estimate: {kernel.omega_irr_estimate:.4f} rad/s",
    f"   omega_max/omega_irr: {hydro.omega_max/kernel.omega_irr_estimate*100:.1f}%",
    "",
    "2. Transform convention",
    "   K(t) = (2/pi) * integral_0^{omega_max} B_tapered(omega) cos(omega*t) domega",
    "   B(omega) = integral_0^inf K(t) cos(omega*t) dt  [Ogilvie 1964]",
    "   A(omega) - A_inf = -(1/omega) * integral_0^inf K(t) sin(omega*t) dt",
    "",
    "3. Kernel parameters",
    f"   t_max          = {kernel.t_max:.1f} s",
    f"   dt             = {kernel.dt} s",
    f"   n_time         = {kernel.n_time}",
    f"   n_interp       = {kernel.n_interp}",
    f"   taper_fraction = {kernel.taper_fraction}",
    f"   K(0)           = {K[0]:.1f} N·s/m",
    "",
    "4. A_infinity estimates",
    f"   Endpoint A(omega_max)  = {kernel.A_infinity_endpoint:.1f} kg  [lower bound]",
    f"   Trend fit (1/omega^2)  = {kernel.A_infinity_trend_fit:.1f} kg  [unreliable]",
    f"   Ogilvie mid-range      = {kernel.A_infinity_ogilvie:.1f} kg  [SELECTED]",
    "",
    "   LIMITATION: omega_max = 1.4 rad/s is only 67% of omega_irr.",
    "   A(omega) has NOT reached its high-frequency asymptote.",
    "   A_inf cannot be reliably determined from this frequency range.",
    "   The Ogilvie estimate is self-consistent with the kernel but",
    "   is still affected by the truncation of B(omega) at omega_max.",
    "",
    "5. Reconstruction quality",
    f"   Trusted range: {errs['trusted_omega_min']:.3f}–{errs['trusted_omega_max']:.3f} rad/s",
    f"   B RMSE (trusted)   = {errs['B_rmse_trusted_pct']:.2f}%",
    f"   B max rel (trusted)= {errs['B_max_rel_trusted_pct']:.2f}%",
    f"   A RMSE (trusted)   = {errs['A_rmse_trusted_pct']:.2f}%",
    f"   A max rel (trusted)= {errs['A_max_rel_trusted_pct']:.2f}%",
    "",
    "   B reconstruction is the primary quality metric.",
    "   A reconstruction errors are dominated by A_inf uncertainty.",
    "",
    "6. Numerical limitations",
    "   - B(omega) is truncated at omega_max = 1.4 rad/s where it is",
    "     still rising steeply (near irregular frequency artifact).",
    "   - The cosine taper (15%) reduces Gibbs ringing but does not",
    "     eliminate the truncation error near omega_max.",
    "   - The reconstruction degrades rapidly for omega > 1.15 rad/s.",
    "   - A_inf is the dominant source of A reconstruction error.",
    "   - Mesh convergence: 672-panel mesh has ~20% B error (Module 2.2).",
    "     This propagates into K(t) and the reconstruction.",
    "",
    "7. Recommendations before Cummins deployment",
    "   - Extend frequency range to at least 3*omega_irr with a finer",
    "     mesh to obtain a reliable A_inf estimate.",
    "   - Refine mesh to reduce the ~20% B(omega) error (Module 2.2).",
    "   - Consider state-space approximation of K(t) for efficiency.",
    "=" * 65,
]

report_text = "\n".join(report_lines)
print()
print(report_text)

report_path = os.path.join(OUTPUT_DIR, "module2_radiation_summary.txt")
with open(report_path, "w") as f:
    f.write(report_text + "\n")
print(f"\nSaved: {report_path}")
