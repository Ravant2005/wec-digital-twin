"""
demo_excitation.py — Wave excitation force demonstration (Module 2.3B).

Produces:
  P1. Regular wave elevation and excitation force vs time
  P2. Two-component wave elevation and force
  P3. Excitation amplitude frequency response (Capytaine vs time-domain)
  P4. Measured vs theoretical force amplitude
  P5. Phase comparison

Output files (outputs/module2_hydro/):
  P1_regular_wave_excitation.png
  P2_two_component_excitation.png
  P3_frequency_response.png
  P4_amplitude_validation.png
  P5_phase_comparison.png
  module2_excitation_summary.txt

Usage:
    python -m module2_wec.demo_excitation

TEMPORARY REFERENCE BUOY — not the Goa deployment buoy.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from module2_wec.geometry import REFERENCE_BUOY
from module2_wec.hydrodynamic_coefficients import compute_hydrodynamic_coefficients
from module2_wec.excitation import compute_excitation_force, interpolate_excitation
from module1_ocean.waves import frequency_grid, component_amplitudes, random_phases
from module1_ocean.spectrum import jonswap

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "module2_hydro",
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("Module 2.3B — Wave Excitation Force Demonstration")
print("=" * 60)

# ---------------------------------------------------------------------------
# BEM coefficients (production path, frozen)
# ---------------------------------------------------------------------------
print("Computing BEM coefficients (672 panels, 20 frequencies)...")
hydro = compute_hydrodynamic_coefficients(
    geometry=REFERENCE_BUOY, resolution=(6, 24, 16),
    omega_min=0.2, omega_max=1.4, n_omega=20, progress_bar=False,
)
print("BEM complete.\n")

# ---------------------------------------------------------------------------
# P1: Regular wave — elevation and force
# ---------------------------------------------------------------------------
t = np.linspace(0.0, 60.0, 6001)
omega_reg = 0.7   # rad/s — well within trusted range
a_reg = 1.0       # m
phi_reg = 0.0     # rad

sig_reg = compute_excitation_force(
    hydro, t, np.array([omega_reg]), np.array([a_reg]), np.array([phi_reg])
)

F_amp_theory = a_reg * sig_reg.excitation_amplitude[0]
F_phase_theory = sig_reg.excitation_phase[0]

print(f"Regular wave: omega={omega_reg} rad/s, a={a_reg} m")
print(f"  |F_exc(omega)| = {sig_reg.excitation_amplitude[0]/1e3:.3f} kN/m")
print(f"  angle(F_exc)   = {np.degrees(F_phase_theory):.2f} deg")
print(f"  Force amplitude = {F_amp_theory/1e3:.3f} kN")
print()

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(t, sig_reg.wave_elevation, "b-", linewidth=1.2)
axes[0].set_ylabel("η(t) [m]")
axes[0].set_title(f"Regular wave: ω={omega_reg} rad/s, a={a_reg} m")
axes[0].grid(True, alpha=0.3)
axes[1].plot(t, sig_reg.force / 1e3, "r-", linewidth=1.2)
axes[1].set_ylabel("F_exc(t) [kN]")
axes[1].set_xlabel("Time [s]")
axes[1].set_title(
    f"Heave excitation force: amplitude = {F_amp_theory/1e3:.2f} kN, "
    f"phase = {np.degrees(F_phase_theory):.1f}°"
)
axes[1].grid(True, alpha=0.3)
fig.suptitle("Module 2.3B — Regular wave excitation force\n"
             "TEMPORARY reference buoy", fontsize=10)
fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "P1_regular_wave_excitation.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved: {path}")

# ---------------------------------------------------------------------------
# P2: Two-component wave
# ---------------------------------------------------------------------------
omega2 = np.array([0.5, 1.0])
a2 = np.array([1.2, 0.8])
phi2 = np.array([0.0, np.pi / 4])

sig2 = compute_excitation_force(hydro, t, omega2, a2, phi2)

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(t, sig2.wave_elevation, "b-", linewidth=1.2)
axes[0].set_ylabel("η(t) [m]")
axes[0].set_title(
    f"Two-component wave: ω=[{omega2[0]}, {omega2[1]}] rad/s, "
    f"a=[{a2[0]}, {a2[1]}] m"
)
axes[0].grid(True, alpha=0.3)
axes[1].plot(t, sig2.force / 1e3, "r-", linewidth=1.2)
axes[1].set_ylabel("F_exc(t) [kN]")
axes[1].set_xlabel("Time [s]")
axes[1].set_title("Heave excitation force (linear superposition)")
axes[1].grid(True, alpha=0.3)
fig.suptitle("Module 2.3B — Two-component wave excitation\n"
             "TEMPORARY reference buoy", fontsize=10)
fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "P2_two_component_excitation.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved: {path}")

# ---------------------------------------------------------------------------
# P3 & P4: Frequency response — measured vs theoretical
# ---------------------------------------------------------------------------
t_long = np.linspace(0.0, 400.0, 80001)
a_test = 1.0

omega_test = hydro.omega   # all 20 BEM grid points
F_amp_theory_arr = hydro.excitation_force_heave_amplitude * a_test
F_amp_measured_arr = np.empty(len(omega_test))

for i, w in enumerate(omega_test):
    sig_i = compute_excitation_force(
        hydro, t_long, np.array([w]), np.array([a_test]), np.array([0.0])
    )
    N = len(t_long)
    phasor = np.sum(sig_i.force * np.exp(-1j * w * t_long)) * 2.0 / N
    F_amp_measured_arr[i] = abs(phasor)

rel_errors = np.abs(F_amp_measured_arr - F_amp_theory_arr) / F_amp_theory_arr

print("Frequency-response validation (all 20 BEM grid points):")
print(f"  Max relative error = {rel_errors.max()*100:.4f}%")
print(f"  Mean relative error = {rel_errors.mean()*100:.4f}%")
print()

# P3: Frequency response comparison
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(omega_test, F_amp_theory_arr / 1e3, "b-o", markersize=5,
        linewidth=1.5, label="|F_exc(ω)| Capytaine (a=1 m)")
ax.plot(omega_test, F_amp_measured_arr / 1e3, "r--s", markersize=5,
        linewidth=1.5, label="Measured from time series")
ax.set_xlabel("Angular frequency ω [rad/s]")
ax.set_ylabel("Force amplitude [kN]  (per 1 m wave)")
ax.set_title("Excitation force frequency response\n"
             "Capytaine |F_exc(ω)| vs measured time-domain amplitude")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "P3_frequency_response.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved: {path}")

# P4: Amplitude validation error
fig, ax = plt.subplots(figsize=(9, 4))
ax.semilogy(omega_test, rel_errors * 100, "g-o", markersize=5, linewidth=1.5)
ax.axhline(0.5, color="r", linestyle="--", linewidth=1, label="0.5% threshold")
ax.set_xlabel("Angular frequency ω [rad/s]")
ax.set_ylabel("Relative error [%]")
ax.set_title(f"Force amplitude relative error\n"
             f"Max = {rel_errors.max()*100:.4f}%, Mean = {rel_errors.mean()*100:.4f}%")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "P4_amplitude_validation.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved: {path}")

# ---------------------------------------------------------------------------
# P5: Phase comparison
# ---------------------------------------------------------------------------
F_complex_interp, _, F_phase_interp = interpolate_excitation(hydro, omega_test)
F_phase_stored = hydro.excitation_force_heave_phase

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(omega_test, np.degrees(F_phase_stored), "b-o", markersize=5,
        linewidth=1.5, label="Capytaine phase (stored)")
ax.plot(omega_test, np.degrees(F_phase_interp), "r--s", markersize=5,
        linewidth=1.5, label="Interpolated phase")
ax.set_xlabel("Angular frequency ω [rad/s]")
ax.set_ylabel("Phase [deg]")
ax.set_title("Excitation force phase angle(F_exc(ω))\n"
             "Stored Capytaine values vs interpolated at grid points")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
path = os.path.join(OUTPUT_DIR, "P5_phase_comparison.png")
fig.savefig(path, dpi=150)
plt.close(fig)
print(f"Saved: {path}")

# ---------------------------------------------------------------------------
# Module 1 integration demo
# ---------------------------------------------------------------------------
print("\nModule 1 integration demo:")
f_min = 0.25 / (2 * np.pi)
f_max = 1.2 / (2 * np.pi)
f_grid = frequency_grid(f_min, f_max, n_components=12)
S = jonswap(f_grid, Hs=1.5, Tp=8.0)
a_m1 = component_amplitudes(f_grid, S)
omega_m1 = 2.0 * np.pi * f_grid
phi_m1 = random_phases(len(f_grid), seed=42)

t_m1 = np.linspace(0.0, 300.0, 6001)
sig_m1 = compute_excitation_force(hydro, t_m1, omega_m1, a_m1, phi_m1)

print(f"  Components: {sig_m1.n_components}")
print(f"  omega range: [{omega_m1.min():.3f}, {omega_m1.max():.3f}] rad/s")
print(f"  Force RMS: {np.sqrt(np.mean(sig_m1.force**2))/1e3:.3f} kN")
print(f"  Force peak: {np.max(np.abs(sig_m1.force))/1e3:.3f} kN")
print(f"  All finite: {np.all(np.isfinite(sig_m1.force))}")

# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------
report_lines = [
    "=" * 65,
    "Module 2.3B — Wave Excitation Force Summary",
    "=" * 65,
    "",
    "IMPORTANT: TEMPORARY DEVELOPMENT BUOY — not the Goa deployment buoy.",
    "",
    "1. Physics",
    "   F_j(t) = a_j * |F_exc(omega_j)| * cos(omega_j*t + phi_j + angle(F_exc))",
    "   F_exc(t) = sum_j F_j(t)   [linear superposition]",
    "   Units: a_j [m], |F_exc| [N/m], F_j(t) [N]",
    "",
    "2. BEM input (Module 2.2, FROZEN)",
    f"   Capytaine version : {hydro.capytaine_version}",
    f"   Mesh resolution   : (6, 24, 16) = {hydro.n_panels} panels",
    f"   Frequency range   : {hydro.omega_min:.2f}–{hydro.omega_max:.2f} rad/s "
    f"({hydro.n_omega} points)",
    "",
    "3. Interpolation method",
    "   Linear interpolation of Re(F_exc) and Im(F_exc) separately.",
    "   Amplitude and phase derived from interpolated complex value.",
    "   No extrapolation: ValueError raised outside [0.2, 1.4] rad/s.",
    "",
    "4. Frequency-response validation (all 20 BEM grid points, a=1 m)",
    f"   Max relative error  = {rel_errors.max()*100:.4f}%",
    f"   Mean relative error = {rel_errors.mean()*100:.4f}%",
    "   (Limited by DFT spectral leakage over 400 s window)",
    "",
    "5. Module 1 integration",
    f"   Components: {sig_m1.n_components}",
    f"   omega range: [{omega_m1.min():.3f}, {omega_m1.max():.3f}] rad/s",
    f"   Force RMS: {np.sqrt(np.mean(sig_m1.force**2))/1e3:.3f} kN",
    f"   All finite: {np.all(np.isfinite(sig_m1.force))}",
    "",
    "6. Numerical limitations",
    "   - Frequency range limited to [0.2, 1.4] rad/s (Module 2.2 grid).",
    "   - Linear interpolation between 20 BEM points; error is negligible",
    "     for smooth F_exc(omega) but increases near omega_max where the",
    "     irregular-frequency artifact causes rapid variation.",
    "   - Wave direction fixed at 0 deg (head-on); directional spreading",
    "     not yet implemented.",
    "   - No buoy motion, radiation, or PTO included.",
    "=" * 65,
]

report_text = "\n".join(report_lines)
print()
print(report_text)

report_path = os.path.join(OUTPUT_DIR, "module2_excitation_summary.txt")
with open(report_path, "w") as f:
    f.write(report_text + "\n")
print(f"\nSaved: {report_path}")
