"""
demo_latching.py — Module 2.5 oracle/reference latching control baseline.

Generates:
  L1  — excitation force + latch state
  L2  — displacement: no-PTO vs fixed-PTO vs latching
  L3  — velocity
  L4  — PTO force
  L5  — instantaneous PTO power
  L6  — cumulative PTO energy
  L7  — latch-duration sweep vs mean power
  L8  — latch-duration sweep vs displacement amplitude
  L9  — phase sensitivity (omega=0.8 rad/s, 7 initial phases)
  L10 — energy balance

Summary written to outputs/module2_hydro/module2_latching_summary.txt

ORACLE/REFERENCE LATCHING BASELINE — uses true excitation signal.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from module2_wec.geometry import REFERENCE_BUOY
from module2_wec.hydrostatics import Hydrostatics
from module2_wec.hydrodynamic_coefficients import compute_hydrodynamic_coefficients
from module2_wec.radiation import compute_radiation_kernel
from module2_wec.excitation import compute_excitation_force
from module2_wec.cummins import CumminsParameters, solve_cummins
from module2_wec.pto import PTOParameters
from module2_wec.latching import LatchingParameters, solve_cummins_latching

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

OUT = "outputs/module2_hydro"
os.makedirs(OUT, exist_ok=True)

RESOLUTION  = (6, 24, 16)
B_PTO       = 4e4          # N·s/m — Module 2.4 baseline
DT          = 0.05         # s
T_SIM       = 300.0        # s  (long enough for steady state)
OMEGA_CASES = [0.6, 0.8, 1.0]
LATCH_DUR   = 1.0          # s  — default latch duration

print("Building hydrodynamic model …")
hydro = compute_hydrodynamic_coefficients(
    geometry=REFERENCE_BUOY, resolution=RESOLUTION,
    omega_min=0.2, omega_max=1.4, n_omega=20, progress_bar=False,
)
rk = compute_radiation_kernel(hydro)
hs = Hydrostatics(REFERENCE_BUOY)

params = CumminsParameters(
    mass=REFERENCE_BUOY.mass,
    hydrostatic_stiffness=hs.hydrostatic_stiffness,
    added_mass_infinity=rk.A_infinity,
    kernel_time=rk.time,
    kernel_values=rk.kernel,
)

pto = PTOParameters(damping=B_PTO, stiffness=0.0)
lp  = LatchingParameters(enabled=True, latch_duration=LATCH_DUR)
lp0 = LatchingParameters(enabled=False, latch_duration=0.0)

t = np.arange(0.0, T_SIM + DT * 0.5, DT)


def _exc(omega_w, phase=0.0):
    return compute_excitation_force(
        hydro, t, np.array([omega_w]), np.array([1.0]), np.array([phase])
    )


def _ss_half(arr):
    """Mean over second half (steady-state estimate)."""
    return float(np.mean(arr[len(arr) // 2:]))


def _amp(arr):
    """Peak-to-peak amplitude / 2 over second half."""
    h = arr[len(arr) // 2:]
    return 0.5 * (h.max() - h.min())


# ---------------------------------------------------------------------------
# Three-case comparison at omega=0.8 rad/s
# ---------------------------------------------------------------------------

print("Running three-case comparison (omega=0.8 rad/s) …")
exc08 = _exc(0.8)

res_A = solve_cummins(params, t, exc08.force)                          # no PTO, no latch
res_B = solve_cummins(params, t, exc08.force, pto=pto)                 # fixed PTO
res_C = solve_cummins_latching(params, t, exc08.force, lp, pto=pto)   # latching

# ---------------------------------------------------------------------------
# L1 — excitation force + latch state
# ---------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
t_plot = t[t <= 100]
ax1.plot(t_plot, exc08.force[t <= 100] / 1e3, "b")
ax1.set_ylabel("F_exc [kN]")
ax1.set_title("L1 — Excitation force and latch state (ω=0.8 rad/s)")
ax2.fill_between(t_plot, res_C.latch_active[t <= 100].astype(float),
                 alpha=0.5, color="orange", label="LATCHED")
ax2.set_ylabel("Latch active")
ax2.set_xlabel("Time [s]")
ax2.set_ylim(-0.1, 1.3)
ax2.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/L1_latch_state.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# L2 — displacement comparison
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t_plot, res_A.displacement[t <= 100], label="No PTO")
ax.plot(t_plot, res_B.displacement[t <= 100], label="Fixed PTO")
ax.plot(t_plot, res_C.displacement[t <= 100], label="Latching")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Displacement [m]")
ax.set_title("L2 — Displacement comparison (ω=0.8 rad/s)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/L2_displacement.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# L3 — velocity
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t_plot, res_A.velocity[t <= 100], label="No PTO")
ax.plot(t_plot, res_B.velocity[t <= 100], label="Fixed PTO")
ax.plot(t_plot, res_C.velocity[t <= 100], label="Latching")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Velocity [m/s]")
ax.set_title("L3 — Velocity comparison (ω=0.8 rad/s)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/L3_velocity.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# L4 — PTO force
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t_plot, res_B.pto_result.force[t <= 100] / 1e3, label="Fixed PTO")
ax.plot(t_plot, res_C.pto_result.force[t <= 100] / 1e3, label="Latching")
ax.set_xlabel("Time [s]")
ax.set_ylabel("PTO force [kN]")
ax.set_title("L4 — PTO force (ω=0.8 rad/s)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/L4_pto_force.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# L5 — instantaneous PTO power
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t_plot, res_B.pto_result.absorbed_power[t <= 100] / 1e3, label="Fixed PTO")
ax.plot(t_plot, res_C.pto_result.absorbed_power[t <= 100] / 1e3, label="Latching")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Power [kW]")
ax.set_title("L5 — Instantaneous PTO power (ω=0.8 rad/s)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/L5_pto_power.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# L6 — cumulative PTO energy
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t, res_B.pto_result.cumulative_energy / 1e3, label="Fixed PTO")
ax.plot(t, res_C.pto_result.cumulative_energy / 1e3, label="Latching")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Cumulative energy [kJ]")
ax.set_title("L6 — Cumulative PTO energy (ω=0.8 rad/s)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/L6_pto_energy.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# L7 & L8 — latch-duration sweep
# ---------------------------------------------------------------------------

print("Running latch-duration sweep …")
durations = [0.0, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]
mean_powers = []
disp_amps   = []

for d in durations:
    lp_d = LatchingParameters(enabled=(d > 0), latch_duration=d)
    r = solve_cummins_latching(params, t, exc08.force, lp_d, pto=pto)
    mean_powers.append(_ss_half(r.pto_result.absorbed_power))
    disp_amps.append(_amp(r.displacement))

P_fixed_ss = _ss_half(res_B.pto_result.absorbed_power)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(durations, [p / 1e3 for p in mean_powers], "o-")
ax.axhline(P_fixed_ss / 1e3, color="gray", linestyle="--", label="Fixed PTO baseline")
ax.set_xlabel("Latch duration [s]")
ax.set_ylabel("Mean absorbed power [kW]")
ax.set_title("L7 — Latch duration vs mean power (ω=0.8 rad/s)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/L7_latch_duration_power.png", dpi=120)
plt.close()

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(durations, disp_amps, "o-")
ax.set_xlabel("Latch duration [s]")
ax.set_ylabel("Displacement amplitude [m]")
ax.set_title("L8 — Latch duration vs displacement amplitude (ω=0.8 rad/s)")
plt.tight_layout()
plt.savefig(f"{OUT}/L8_latch_duration_disp.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# L9 — phase sensitivity
# ---------------------------------------------------------------------------

print("Running phase sensitivity study …")
phases_deg = [0, 30, 60, 90, 120, 150, 180]
phase_powers_B = []
phase_powers_C = []

for ph_deg in phases_deg:
    ph = np.deg2rad(ph_deg)
    exc_ph = compute_excitation_force(
        hydro, t, np.array([0.8]), np.array([1.0]), np.array([ph])
    )
    rB = solve_cummins(params, t, exc_ph.force, pto=pto)
    rC = solve_cummins_latching(params, t, exc_ph.force, lp, pto=pto)
    phase_powers_B.append(_ss_half(rB.pto_result.absorbed_power))
    phase_powers_C.append(_ss_half(rC.pto_result.absorbed_power))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(phases_deg, [p / 1e3 for p in phase_powers_B], "o-", label="Fixed PTO")
ax.plot(phases_deg, [p / 1e3 for p in phase_powers_C], "s-", label="Latching")
ax.set_xlabel("Initial wave phase [deg]")
ax.set_ylabel("Mean absorbed power [kW]")
ax.set_title("L9 — Phase sensitivity (ω=0.8 rad/s)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/L9_phase_sensitivity.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# L10 — energy balance
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(t, res_C.energy_excitation / 1e3, label="E_exc (excitation work)")
ax.plot(t, res_C.energy_radiation / 1e3,  label="E_rad (radiation memory work)")
ax.plot(t, res_C.pto_result.cumulative_energy / 1e3, label="E_PTO (extracted)")
ax.plot(t, res_C.mechanical_energy / 1e3, label="E_mech (KE+PE)")
ax.plot(t, res_C.energy_latch / 1e3,      label="E_latch (constraint work ≈0)")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Energy [kJ]")
ax.set_title("L10 — Energy balance (latching, ω=0.8 rad/s)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/L10_energy_balance.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# Regular-wave validation table
# ---------------------------------------------------------------------------

print("Running regular-wave validation …")
reg_results = {}
for omega_w in OMEGA_CASES:
    exc = _exc(omega_w)
    rA = solve_cummins(params, t, exc.force)
    rB = solve_cummins(params, t, exc.force, pto=pto)
    rC = solve_cummins_latching(params, t, exc.force, lp, pto=pto)
    reg_results[omega_w] = (rA, rB, rC, exc)

# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

lines = []
lines.append("=" * 70)
lines.append("MODULE 2.5 — DETERMINISTIC LATCHING CONTROL BASELINE")
lines.append("ORACLE/REFERENCE LATCHING BASELINE — uses true excitation signal")
lines.append("=" * 70)
lines.append("")
lines.append("CONTROLLER: Phase-based latching (downward F_exc zero-crossing trigger)")
lines.append(f"Default latch duration: {LATCH_DUR} s")
lines.append(f"PTO damping: {B_PTO:.0f} N·s/m")
lines.append(f"Simulation duration: {T_SIM} s, dt={DT} s")
lines.append("")
lines.append("-" * 70)
lines.append("REGULAR-WAVE VALIDATION")
lines.append("-" * 70)
lines.append(f"{'omega':>8} {'case':>10} {'x_amp[m]':>10} {'v_amp[m/s]':>12} "
             f"{'P_mean[W]':>12} {'E_cum[kJ]':>12}")

for omega_w, (rA, rB, rC, exc) in reg_results.items():
    for label, r in [("No PTO", rA), ("Fixed PTO", rB), ("Latching", rC)]:
        x_amp = _amp(r.displacement)
        v_amp = _amp(r.velocity)
        if r.pto_result is not None:
            p_mean = _ss_half(r.pto_result.absorbed_power)
            e_cum  = r.pto_result.cumulative_energy[-1] / 1e3
        else:
            p_mean = 0.0
            e_cum  = 0.0
        lines.append(f"{omega_w:>8.1f} {label:>10} {x_amp:>10.4f} {v_amp:>12.4f} "
                     f"{p_mean:>12.1f} {e_cum:>12.2f}")
    lines.append("")

lines.append("-" * 70)
lines.append("FIXED PTO vs LATCHING COMPARISON (omega=0.8 rad/s, steady-state half)")
lines.append("-" * 70)
P_B = _ss_half(reg_results[0.8][1].pto_result.absorbed_power)
P_C = _ss_half(reg_results[0.8][2].pto_result.absorbed_power)
pct = (P_C - P_B) / P_B * 100.0
lines.append(f"  Fixed PTO mean power:   {P_B:.1f} W")
lines.append(f"  Latching mean power:    {P_C:.1f} W")
lines.append(f"  Improvement:            {pct:+.1f}%")
lines.append("")

lines.append("-" * 70)
lines.append("LATCH-DURATION SWEEP (omega=0.8 rad/s)")
lines.append("-" * 70)
lines.append(f"{'Duration[s]':>12} {'P_mean[W]':>12} {'vs_fixed[%]':>12} {'x_amp[m]':>10}")
for d, p, xa in zip(durations, mean_powers, disp_amps):
    pct_d = (p - P_fixed_ss) / P_fixed_ss * 100.0
    lines.append(f"{d:>12.2f} {p:>12.1f} {pct_d:>+12.1f} {xa:>10.4f}")
lines.append("")

lines.append("-" * 70)
lines.append("PHASE SENSITIVITY (omega=0.8 rad/s, latch_duration=1.0 s)")
lines.append("-" * 70)
lines.append(f"{'Phase[deg]':>12} {'P_fixed[W]':>12} {'P_latch[W]':>12} {'improvement[%]':>16}")
for ph, pB, pC in zip(phases_deg, phase_powers_B, phase_powers_C):
    pct_ph = (pC - pB) / pB * 100.0 if pB > 0 else float("nan")
    lines.append(f"{ph:>12} {pB:>12.1f} {pC:>12.1f} {pct_ph:>+16.1f}")
lines.append("")

lines.append("-" * 70)
lines.append("ENERGY ACCOUNTING (omega=0.8 rad/s, latching)")
lines.append("-" * 70)
rC08 = reg_results[0.8][2]
lines.append(f"  Total excitation work:  {rC08.energy_excitation[-1]/1e3:>10.2f} kJ")
lines.append(f"  Radiation memory work:  {rC08.energy_radiation[-1]/1e3:>10.2f} kJ")
lines.append(f"  PTO extracted energy:   {rC08.pto_result.cumulative_energy[-1]/1e3:>10.2f} kJ")
lines.append(f"  Latch reaction work:    {rC08.energy_latch[-1]:>10.4f} J  (should be ~0)")
lines.append(f"  Final mechanical energy:{rC08.mechanical_energy[-1]:>10.2f} J")
lines.append(f"  Latch fraction:         {rC08.latch_fraction*100:.1f}%")
lines.append(f"  Latch events:           {rC08.n_latch_events}")
lines.append("")

lines.append("-" * 70)
lines.append("LIMITATIONS")
lines.append("-" * 70)
lines.append("  1. ORACLE controller — uses true future excitation (not deployable).")
lines.append("  2. Ideal kinematic latch — no mechanical latch model.")
lines.append("  3. Trigger rule is heuristic (downward F_exc zero-crossing).")
lines.append("  4. Euler-Cromer integration; accuracy limited by dt=0.05 s.")
lines.append("  5. Single-DOF heave only; surge/pitch coupling ignored.")
lines.append("")

lines.append("=" * 70)
lines.append("MODULE 2.5 STATUS: READY TO FREEZE")
lines.append("=" * 70)

summary = "\n".join(lines)
print(summary)

with open(f"{OUT}/module2_latching_summary.txt", "w") as f:
    f.write(summary + "\n")

print(f"\nPlots saved to {OUT}/")
print(f"Summary saved to {OUT}/module2_latching_summary.txt")
