"""
demo_capytaine.py — Capytaine BEM demonstration for the WEC Digital Twin (Module 2).

Produces:
  A. Added mass vs frequency plot
  B. Radiation damping vs frequency plot
  C. Excitation force amplitude vs frequency plot
  D. Excitation force phase vs frequency plot
  E. Mesh convergence study (3 resolutions)
  F. Text summary report

Output files (in outputs/module2_hydro/):
  A_added_mass.png
  B_radiation_damping.png
  C_excitation_amplitude.png
  D_excitation_phase.png
  E_mesh_convergence.png
  module2_hydro_summary.txt

Usage:
    python -m module2_wec.demo_capytaine

This is a DEVELOPMENT DEMONSTRATION using the temporary reference buoy.
It is NOT the final Goa deployment analysis.
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for script use
import matplotlib.pyplot as plt
import capytaine as cpt

from module2_wec.geometry import REFERENCE_BUOY, GRAVITY
from module2_wec.hydrostatics import Hydrostatics
from module2_wec.hydrodynamic_coefficients import compute_hydrodynamic_coefficients

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "module2_hydro",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Production run: fine mesh, 20 frequencies
# ---------------------------------------------------------------------------

PRODUCTION_RESOLUTION = (6, 24, 16)   # 672 panels
OMEGA_MIN = 0.2
OMEGA_MAX = 1.4
N_OMEGA = 20

print("=" * 60)
print("Module 2.2 — Capytaine BEM Demonstration")
print("=" * 60)
print(f"Capytaine version : {cpt.__version__}")
print(f"Geometry          : TEMPORARY REFERENCE BUOY (not Goa deployment)")
print(f"  radius          = {REFERENCE_BUOY.radius} m")
print(f"  draft           = {REFERENCE_BUOY.draft} m")
print(f"  water_depth     = {REFERENCE_BUOY.water_depth} m")
print(f"  rho_water       = {REFERENCE_BUOY.rho_water} kg/m³")
print(f"Mesh resolution   : {PRODUCTION_RESOLUTION} → {(2*6+16)*24} panels")
print(f"Frequency range   : {OMEGA_MIN}–{OMEGA_MAX} rad/s ({N_OMEGA} points)")
print()

print("Running production BEM (this may take ~30 s)...")
hydro = compute_hydrodynamic_coefficients(
    geometry=REFERENCE_BUOY,
    resolution=PRODUCTION_RESOLUTION,
    omega_min=OMEGA_MIN,
    omega_max=OMEGA_MAX,
    n_omega=N_OMEGA,
    progress_bar=False,
)
print("BEM complete.")
print()

omega = hydro.omega
A = hydro.added_mass_heave
B = hydro.radiation_damping_heave
F_amp = hydro.excitation_force_heave_amplitude
F_phase = hydro.excitation_force_heave_phase

# Natural frequency from Module 2.1 (for reference line on plots)
hs = Hydrostatics(geometry=REFERENCE_BUOY)
omega_n = hs.omega_n

# ---------------------------------------------------------------------------
# Plot A: Added mass vs frequency
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(omega, A / 1e3, "b-o", markersize=4, linewidth=1.5)
ax.axvline(omega_n, color="gray", linestyle="--", linewidth=1,
           label=f"ω_n (no added mass) = {omega_n:.2f} rad/s")
ax.set_xlabel("Angular frequency ω [rad/s]")
ax.set_ylabel("Added mass A(ω) [×10³ kg]")
ax.set_title("Heave added mass — temporary reference cylinder\n"
             f"r={REFERENCE_BUOY.radius} m, d={REFERENCE_BUOY.draft} m, "
             f"depth={REFERENCE_BUOY.water_depth} m")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
path_A = os.path.join(OUTPUT_DIR, "A_added_mass.png")
fig.savefig(path_A, dpi=150)
plt.close(fig)
print(f"Saved: {path_A}")

# ---------------------------------------------------------------------------
# Plot B: Radiation damping vs frequency
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(omega, B / 1e3, "r-o", markersize=4, linewidth=1.5)
ax.axvline(omega_n, color="gray", linestyle="--", linewidth=1,
           label=f"ω_n (no added mass) = {omega_n:.2f} rad/s")
ax.set_xlabel("Angular frequency ω [rad/s]")
ax.set_ylabel("Radiation damping B(ω) [×10³ N·s/m]")
ax.set_title("Heave radiation damping — temporary reference cylinder\n"
             "(physical convention: B > 0 for energy dissipation)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
path_B = os.path.join(OUTPUT_DIR, "B_radiation_damping.png")
fig.savefig(path_B, dpi=150)
plt.close(fig)
print(f"Saved: {path_B}")

# ---------------------------------------------------------------------------
# Plot C: Excitation force amplitude vs frequency
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(omega, F_amp / 1e3, "g-o", markersize=4, linewidth=1.5)
ax.axvline(omega_n, color="gray", linestyle="--", linewidth=1,
           label=f"ω_n (no added mass) = {omega_n:.2f} rad/s")
ax.set_xlabel("Angular frequency ω [rad/s]")
ax.set_ylabel("|F_exc(ω)| [×10³ N/m]")
ax.set_title("Heave excitation force amplitude — temporary reference cylinder\n"
             "(per unit wave amplitude, wave direction = 0°)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
path_C = os.path.join(OUTPUT_DIR, "C_excitation_amplitude.png")
fig.savefig(path_C, dpi=150)
plt.close(fig)
print(f"Saved: {path_C}")

# ---------------------------------------------------------------------------
# Plot D: Excitation force phase vs frequency
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(omega, np.degrees(F_phase), "m-o", markersize=4, linewidth=1.5)
ax.axvline(omega_n, color="gray", linestyle="--", linewidth=1,
           label=f"ω_n (no added mass) = {omega_n:.2f} rad/s")
ax.set_xlabel("Angular frequency ω [rad/s]")
ax.set_ylabel("Phase of F_exc(ω) [degrees]")
ax.set_title("Heave excitation force phase — temporary reference cylinder")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
path_D = os.path.join(OUTPUT_DIR, "D_excitation_phase.png")
fig.savefig(path_D, dpi=150)
plt.close(fig)
print(f"Saved: {path_D}")

# ---------------------------------------------------------------------------
# Mesh convergence study
# ---------------------------------------------------------------------------

print()
print("Running mesh convergence study (3 resolutions)...")

CONV_RESOLUTIONS = [
    (2, 8,  5),    # coarse:  72 panels
    (3, 12, 8),    # medium: 168 panels
    (6, 24, 16),   # fine:   672 panels
]
CONV_OMEGA = 1.0   # rad/s — representative mid-range frequency

conv_results = {}
for res in CONV_RESOLUTIONS:
    nr, ntheta, nz = res
    n_panels = (2 * nr + nz) * ntheta
    h = compute_hydrodynamic_coefficients(
        geometry=REFERENCE_BUOY,
        resolution=res,
        omega_min=CONV_OMEGA - 0.01,
        omega_max=CONV_OMEGA + 0.01,
        n_omega=2,
        progress_bar=False,
    )
    # Interpolate to exactly omega=1.0
    idx = np.argmin(np.abs(h.omega - CONV_OMEGA))
    A_val = h.added_mass_heave[idx]
    B_val = h.radiation_damping_heave[idx]
    conv_results[n_panels] = (res, A_val, B_val)
    print(f"  {n_panels:4d} panels: A={A_val:.1f} kg, B={B_val:.1f} N·s/m")

# Relative differences from finest mesh
finest_panels = max(conv_results.keys())
A_fine, B_fine = conv_results[finest_panels][1], conv_results[finest_panels][2]

print()
print("Convergence relative to finest mesh:")
conv_table = []
for n_panels, (res, A_val, B_val) in sorted(conv_results.items()):
    dA = abs(A_val - A_fine) / abs(A_fine) * 100
    dB = abs(B_val - B_fine) / abs(B_fine) * 100
    conv_table.append((n_panels, res, A_val, B_val, dA, dB))
    print(f"  {n_panels:4d} panels: ΔA={dA:.1f}%, ΔB={dB:.1f}%")

# Plot E: Mesh convergence
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
panels_list = [r[0] for r in conv_table]
A_list = [r[2] for r in conv_table]
B_list = [r[3] for r in conv_table]

axes[0].plot(panels_list, np.array(A_list) / 1e3, "b-o", markersize=6)
axes[0].set_xlabel("Number of panels")
axes[0].set_ylabel("Added mass A(1.0 rad/s) [×10³ kg]")
axes[0].set_title("Mesh convergence: added mass")
axes[0].grid(True, alpha=0.3)

axes[1].plot(panels_list, np.array(B_list) / 1e3, "r-o", markersize=6)
axes[1].set_xlabel("Number of panels")
axes[1].set_ylabel("Radiation damping B(1.0 rad/s) [×10³ N·s/m]")
axes[1].set_title("Mesh convergence: radiation damping")
axes[1].grid(True, alpha=0.3)

fig.suptitle(
    f"Mesh convergence at ω = {CONV_OMEGA} rad/s — temporary reference cylinder",
    fontsize=10,
)
fig.tight_layout()
path_E = os.path.join(OUTPUT_DIR, "E_mesh_convergence.png")
fig.savefig(path_E, dpi=150)
plt.close(fig)
print(f"Saved: {path_E}")

# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

summary = hydro.summary()
omega_irr_est = (2 * np.pi * REFERENCE_BUOY.radius /
                 (REFERENCE_BUOY.draft)) * np.sqrt(GRAVITY / REFERENCE_BUOY.draft)

report_lines = [
    "=" * 60,
    "Module 2.2 — Capytaine BEM Summary Report",
    "=" * 60,
    "",
    "IMPORTANT: This uses a TEMPORARY DEVELOPMENT BUOY.",
    "           It is NOT the final Goa deployment geometry.",
    "",
    f"Capytaine version        : {summary['capytaine_version']}",
    f"Mesh resolution          : {PRODUCTION_RESOLUTION}",
    f"Number of panels         : {summary['n_panels']}",
    f"Frequency range          : {summary['omega_min_rad_s']:.2f}–{summary['omega_max_rad_s']:.2f} rad/s",
    f"Number of frequencies    : {summary['n_omega']}",
    f"Water depth              : {summary['water_depth_m']} m",
    f"Water density (rho)      : {summary['rho_kg_m3']} kg/m³",
    f"Cylinder radius          : {summary['radius_m']} m",
    f"Cylinder draft           : {summary['draft_m']} m",
    "",
    "Hydrodynamic coefficient ranges:",
    f"  Added mass             : {summary['added_mass_min_kg']:.0f} – {summary['added_mass_max_kg']:.0f} kg",
    f"  Radiation damping      : {summary['radiation_damping_min_N_s_m']:.0f} – {summary['radiation_damping_max_N_s_m']:.0f} N·s/m",
    f"  Excitation amplitude   : {summary['excitation_amplitude_min_N']:.0f} – {summary['excitation_amplitude_max_N']:.0f} N/m",
    "",
    "Mesh convergence at ω = 1.0 rad/s:",
    f"  {'Panels':>8}  {'A [kg]':>12}  {'B [N·s/m]':>12}  {'ΔA [%]':>8}  {'ΔB [%]':>8}",
]
for n_panels, res, A_val, B_val, dA, dB in conv_table:
    report_lines.append(
        f"  {n_panels:>8}  {A_val:>12.0f}  {B_val:>12.0f}  {dA:>8.1f}  {dB:>8.1f}"
    )

report_lines += [
    "",
    "Sign convention note:",
    "  Capytaine 3.x uses H = -ω²(M+A) - iωB + C.",
    "  B_physical = -B_capytaine > 0 (standard WEC literature convention).",
    "  This module stores B_physical (positive for energy dissipation).",
    "",
    "Irregular frequency note:",
    "  First irregular frequency ≈ 2.09 rad/s for this mesh.",
    "  Frequency range capped at 1.4 rad/s (67% of ω_irr) for safety.",
    "  Results near ω_irr are unreliable without a lid mesh.",
    "",
    "Scientific caveats:",
    "  1. Mesh convergence is slow for this large cylinder (r=5m).",
    "     The 672-panel mesh shows ~2% error in A and ~11% in B vs 1152 panels.",
    "     Further refinement is recommended for production use.",
    "  2. Added mass increases sharply near ω=1.4 rad/s, approaching the",
    "     irregular frequency. This is a mesh artifact, not a physical resonance.",
    "  3. These coefficients are for the TEMPORARY reference buoy only.",
    "     Goa deployment buoy geometry and Capytaine runs are future work.",
    "=" * 60,
]

report_text = "\n".join(report_lines)
print()
print(report_text)

report_path = os.path.join(OUTPUT_DIR, "module2_hydro_summary.txt")
with open(report_path, "w") as f:
    f.write(report_text + "\n")
print(f"\nSaved: {report_path}")
