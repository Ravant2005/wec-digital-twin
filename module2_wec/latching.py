"""
latching.py — Deterministic latching control baseline (Module 2.5).

============================================================
PURPOSE
============================================================

Implements an oracle/reference latching controller for the heaving
point-absorber.  This is a deterministic baseline — NOT a realistic
deployable controller.  It has access to the true/reference wave
excitation signal, which is not available in practice.

Its purpose is to establish what deterministic latching can achieve
when timing information is perfectly known, before forecasting and
RL are introduced.

ORACLE/REFERENCE LATCHING BASELINE — uses true future excitation.

============================================================
PHYSICAL MODEL
============================================================

Latching temporarily constrains the buoy:

    v(t) = 0,  x(t) = x_latch   during LATCHED intervals

The latch is an ideal kinematic constraint.  It does NO positive
net energy injection.  The latch reaction force is the force
required to hold the buoy stationary; it is NOT counted as
harvested power.

During FREE intervals the full Cummins + PTO dynamics apply:

    (M + A_inf) x_ddot + (C33 + K_pto) x + convolution + B_pto v = F_exc

During LATCHED intervals:

    v = 0,  x = x_latch  (enforced kinematically)

The radiation-memory convolution is updated at every timestep
using the actual velocity history (v=0 during latch).  The
hydrodynamic memory is NEVER reset.

============================================================
LATCH FORCE
============================================================

The constraint force required to hold the buoy stationary is:

    F_latch = F_exc + F_rad + F_restore - M * a_free

where a_free is the acceleration that would occur without the
constraint.  In the ideal kinematic model:

    F_latch = F_exc(t) + F_mem(t) - C33 * x_latch

(the force that the latch must exert to prevent motion).

F_latch is recorded but NOT counted as harvested power.

============================================================
CONTROLLER STRATEGY — PHASE-BASED LATCHING
============================================================

The controller uses the excitation force signal to determine
latch/release timing.

RULE:
  1. Detect zero-crossings of F_exc(t) (excitation changes sign).
  2. At each downward zero-crossing (F_exc goes from + to -),
     engage the latch for a configurable duration latch_duration.
  3. Release the latch after latch_duration seconds.
  4. The buoy is then FREE until the next trigger.

RATIONALE:
  For a heaving buoy near resonance, maximum velocity occurs near
  the zero-crossing of the excitation force (90 deg phase lag).
  Latching at the downward zero-crossing and releasing before the
  next upward excitation peak allows the buoy to build up velocity
  in phase with the excitation, improving energy capture.

  This is a standard phase-based latching heuristic (Falnes 2002,
  Ringwood et al. 2014).  The latch_duration is a configurable
  parameter; its optimal value depends on wave period and system
  dynamics.

ORACLE NOTE:
  The zero-crossing detection uses the pre-computed excitation
  force array (true future information).  This is intentional for
  the baseline.  A realistic controller would need a wave forecast.

============================================================
STATE MACHINE
============================================================

States:
    FREE    — normal Cummins + PTO dynamics
    LATCHED — v=0, x=x_latch enforced

Transitions:
    FREE -> LATCHED : at a trigger event (downward F_exc zero-crossing)
                      if latch_duration > 0
    LATCHED -> FREE : after latch_duration seconds have elapsed

============================================================
ENERGY ACCOUNTING
============================================================

Tracked quantities:
    E_exc      = integral F_exc * v dt       [J]  excitation work
    E_rad      = integral F_mem * v dt       [J]  radiation memory work (negative = loss)
    E_restore  = integral (-C33*x) * v dt   [J]  hydrostatic work
    E_pto      = integral B_pto * v^2 dt    [J]  PTO extracted energy
    E_latch    = integral F_latch * v dt    [J]  latch reaction work (= 0 ideally)
    E_mech     = 0.5*M*v^2 + 0.5*C33*x^2  [J]  mechanical energy

Energy balance check:
    E_exc + E_rad + E_restore - E_pto - E_latch ≈ delta(E_mech)

During ideal latch v=0, so all power terms are zero.  Any numerical
residual in E_latch is a discretization artifact and is reported.

============================================================
UNITS (SI throughout)
============================================================
    x           : m
    v           : m/s
    F_latch     : N
    E_*         : J
    P_abs       : W
    latch_duration : s

============================================================
REFERENCES
============================================================
- Falnes (2002), Ocean Waves and Oscillating Systems, §6.
- Ringwood et al. (2014), Energy maximisation for wave energy converters.
- Babarit & Clement (2006), Optimal latching control of a wave energy device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np

from module2_wec.cummins import CumminsParameters
from module2_wec.pto import PTOParameters, PTOResult, compute_pto_result


# ---------------------------------------------------------------------------
# LatchState
# ---------------------------------------------------------------------------

class LatchState(Enum):
    """Controller state."""
    FREE    = "FREE"
    LATCHED = "LATCHED"


# ---------------------------------------------------------------------------
# LatchingParameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LatchingParameters:
    """
    Parameters for the deterministic phase-based latching controller.

    Parameters
    ----------
    enabled : bool
        If False, latching is disabled (equivalent to fixed PTO only).
    latch_duration : float
        Duration to hold the latch after each trigger event [s].
        Must be >= 0.
    min_latch_duration : float
        Minimum latch duration enforced [s].  Triggers shorter than this
        are ignored.  Default 0.0.
    dead_time : float
        Minimum time between consecutive latch engagements [s].
        Prevents rapid re-latching.  Default 0.0.
    """

    enabled: bool
    latch_duration: float
    min_latch_duration: float = 0.0
    dead_time: float = 0.0

    def __post_init__(self) -> None:
        if self.latch_duration < 0:
            raise ValueError(f"latch_duration must be >= 0, got {self.latch_duration}")
        if self.min_latch_duration < 0:
            raise ValueError(f"min_latch_duration must be >= 0, got {self.min_latch_duration}")
        if self.dead_time < 0:
            raise ValueError(f"dead_time must be >= 0, got {self.dead_time}")


# ---------------------------------------------------------------------------
# LatchingResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LatchingResult:
    """
    Full time-domain solution with latching control.

    All array fields have shape (n_time,).

    Parameters
    ----------
    time : np.ndarray
        Time grid [s].
    displacement : np.ndarray
        Heave displacement x(t) [m].
    velocity : np.ndarray
        Heave velocity v(t) [m/s].  Zero during LATCHED intervals.
    acceleration : np.ndarray
        Heave acceleration [m/s^2].  Zero during LATCHED intervals.
    excitation_force : np.ndarray
        Wave excitation force F_exc(t) [N].
    radiation_force : np.ndarray
        Total radiation force [N].
    restoring_force : np.ndarray
        Hydrostatic restoring force -C33*x [N].
    latch_force : np.ndarray
        Constraint force required to hold the latch [N].
        Zero during FREE intervals.
    latch_active : np.ndarray
        Boolean array: True when LATCHED.
    state_log : List[LatchState]
        Controller state at each timestep.
    latch_engage_times : np.ndarray
        Times at which latch was engaged [s].
    latch_release_times : np.ndarray
        Times at which latch was released [s].
    pto_result : Optional[PTOResult]
        PTO metrics (None if no PTO).
    params : CumminsParameters
        Cummins parameters used.
    latching_params : LatchingParameters
        Latching parameters used.
    pto_params : Optional[PTOParameters]
        PTO parameters used.
    energy_excitation : np.ndarray
        Cumulative excitation work integral F_exc*v dt [J].
    energy_radiation : np.ndarray
        Cumulative radiation memory work integral F_mem*v dt [J].
    energy_latch : np.ndarray
        Cumulative latch reaction work integral F_latch*v dt [J].
        Should be ~0 for ideal latch (v=0 during latch).
    """

    time: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    excitation_force: np.ndarray
    radiation_force: np.ndarray
    restoring_force: np.ndarray
    latch_force: np.ndarray
    latch_active: np.ndarray
    state_log: list
    latch_engage_times: np.ndarray
    latch_release_times: np.ndarray
    pto_result: Optional[PTOResult]
    params: CumminsParameters
    latching_params: LatchingParameters
    pto_params: Optional[PTOParameters]
    energy_excitation: np.ndarray
    energy_radiation: np.ndarray
    energy_latch: np.ndarray

    @property
    def kinetic_energy(self) -> np.ndarray:
        """0.5 * M * v^2 [J]."""
        return 0.5 * self.params.mass * self.velocity**2

    @property
    def potential_energy(self) -> np.ndarray:
        """0.5 * C33 * x^2 [J]."""
        return 0.5 * self.params.hydrostatic_stiffness * self.displacement**2

    @property
    def mechanical_energy(self) -> np.ndarray:
        """Kinetic + potential energy [J]."""
        return self.kinetic_energy + self.potential_energy

    @property
    def n_latch_events(self) -> int:
        """Number of latch engagements."""
        return len(self.latch_engage_times)

    @property
    def latch_fraction(self) -> float:
        """Fraction of simulation time spent latched."""
        return float(np.mean(self.latch_active))

    @property
    def mean_pto_power_ss(self) -> float:
        """Mean PTO power over second half of simulation [W]."""
        if self.pto_result is None:
            return 0.0
        n_half = len(self.time) // 2
        return float(np.mean(self.pto_result.absorbed_power[n_half:]))


# ---------------------------------------------------------------------------
# Trigger detection
# ---------------------------------------------------------------------------

def _find_downward_zero_crossings(f: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Find indices where f crosses zero from positive to negative.

    Returns indices i such that f[i] >= 0 and f[i+1] < 0.
    """
    crossings = []
    for i in range(len(f) - 1):
        if f[i] >= 0.0 and f[i + 1] < 0.0:
            crossings.append(i)
    return np.array(crossings, dtype=int)


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def solve_cummins_latching(
    params: CumminsParameters,
    time: np.ndarray,
    excitation_force: np.ndarray,
    latching: LatchingParameters,
    pto: Optional[PTOParameters] = None,
    x0: float = 0.0,
    v0: float = 0.0,
) -> LatchingResult:
    """
    Solve the Cummins equation with deterministic phase-based latching.

    ORACLE/REFERENCE LATCHING BASELINE — uses true excitation signal.

    During FREE intervals: full Cummins + PTO Euler-Cromer dynamics.
    During LATCHED intervals: v=0, x=x_latch enforced kinematically.
    Radiation memory is updated at every step (v=0 during latch).
    Latch reaction force is computed but NOT counted as harvested power.

    Parameters
    ----------
    params : CumminsParameters
        Frozen Cummins parameters (Module 2.3C).
    time : np.ndarray
        Uniform time grid [s], shape (n_time,).
    excitation_force : np.ndarray
        Wave excitation force F_exc(t) [N], shape (n_time,).
        Used both as physical input and for trigger detection (oracle).
    latching : LatchingParameters
        Latching controller parameters.
    pto : PTOParameters or None, optional
        PTO parameters.  None = no PTO.
    x0 : float, optional
        Initial displacement [m].
    v0 : float, optional
        Initial velocity [m/s].

    Returns
    -------
    LatchingResult
    """
    time = np.asarray(time, dtype=float)
    excitation_force = np.asarray(excitation_force, dtype=float)

    dts = np.diff(time)
    dt  = float(dts[0])
    if not np.allclose(dts, dt, rtol=1e-6):
        raise ValueError("time grid must be uniformly spaced")
    if excitation_force.shape != time.shape:
        raise ValueError("excitation_force shape must match time shape")

    n     = len(time)
    M_eff = params.effective_mass
    C33   = params.hydrostatic_stiffness
    B_pto = pto.damping   if pto is not None else 0.0
    K_pto = pto.stiffness if pto is not None else 0.0

    # Pre-interpolate kernel
    K_t       = params.kernel_time
    K_v       = params.kernel_values
    n_K_max   = len(K_t)
    n_lag_max = min(n, n_K_max)
    lags      = np.arange(1, n_lag_max + 1) * dt
    K_interp  = np.interp(lags, K_t, K_v, left=0.0, right=0.0)

    # Pre-compute trigger times from oracle excitation signal
    # Downward zero-crossings of F_exc
    trigger_indices: np.ndarray = np.array([], dtype=int)
    if latching.enabled and latching.latch_duration > latching.min_latch_duration:
        trigger_indices = _find_downward_zero_crossings(excitation_force, time)

    # Output arrays
    x          = np.zeros(n)
    v          = np.zeros(n)
    a          = np.zeros(n)
    F_mem_arr  = np.zeros(n)
    F_rad_arr  = np.zeros(n)
    F_res_arr  = np.zeros(n)
    F_latch_arr = np.zeros(n)
    latch_active = np.zeros(n, dtype=bool)
    state_log  = [LatchState.FREE] * n

    # Energy tracking (cumulative, trapezoidal)
    E_exc_arr  = np.zeros(n)
    E_rad_arr  = np.zeros(n)
    E_lat_arr  = np.zeros(n)

    x[0] = x0
    v[0] = v0

    # Controller state
    state          = LatchState.FREE
    x_latch        = 0.0
    latch_end_time = -np.inf
    last_latch_end = -np.inf

    # Convert trigger indices to a set for O(1) lookup
    trigger_set = set(trigger_indices.tolist())

    engage_times  = []
    release_times = []

    for i in range(n):
        t_now = time[i]

        # --- Radiation memory convolution (always, regardless of latch state) ---
        # v[j] = 0 during latch, which is physically correct.
        if i > 0:
            n_lag  = min(i, n_lag_max)
            v_hist = v[i - n_lag:i][::-1]
            F_mem  = -dt * np.dot(K_interp[:n_lag], v_hist)
        else:
            F_mem = 0.0

        # --- Controller state machine ---
        # Check release first (LATCHED -> FREE)
        if state == LatchState.LATCHED and t_now >= latch_end_time:
            state          = LatchState.FREE
            last_latch_end = t_now
            release_times.append(t_now)

        # Check engage (FREE -> LATCHED)
        if (state == LatchState.FREE
                and latching.enabled
                and latching.latch_duration > latching.min_latch_duration
                and i in trigger_set
                and (t_now - last_latch_end) >= latching.dead_time):
            state          = LatchState.LATCHED
            x_latch        = x[i]
            latch_end_time = t_now + latching.latch_duration
            engage_times.append(t_now)

        state_log[i]    = state
        # latch_active[i] is set by the previous step's update (or False for i=0)

        # --- Dynamics ---
        if state == LatchState.FREE:
            # Normal Cummins + PTO
            a[i] = (excitation_force[i]
                    - (C33 + K_pto) * x[i]
                    - B_pto * v[i]
                    + F_mem) / M_eff
            # Latch force is zero when FREE (even at the engage step,
            # because the constraint only takes effect at i+1)
            F_latch_arr[i] = 0.0

        else:
            # LATCHED: v[i]=0 and x[i]=x_latch were enforced by previous step.
            # Compute latch reaction force only when the constraint is active.
            # latch_active[i] is True iff v[i] was zeroed by the constraint.
            if latch_active[i]:
                F_latch_arr[i] = -(excitation_force[i] + F_mem - C33 * x_latch)
            else:
                # First step of latch: v[i] is still the pre-latch velocity.
                # Constraint takes effect at i+1; no latch force at this step.
                F_latch_arr[i] = 0.0
            a[i] = 0.0

        F_rad = -params.added_mass_infinity * a[i] + F_mem
        F_res = -C33 * x[i]

        F_mem_arr[i] = F_mem
        F_rad_arr[i] = F_rad
        F_res_arr[i] = F_res

        # --- Energy tracking (trapezoidal increment) ---
        if i > 0:
            v_mid = 0.5 * (v[i - 1] + v[i])
            E_exc_arr[i] = E_exc_arr[i-1] + excitation_force[i-1] * v_mid * dt
            E_rad_arr[i] = E_rad_arr[i-1] + F_mem_arr[i-1] * v_mid * dt
            E_lat_arr[i] = E_lat_arr[i-1] + F_latch_arr[i-1] * v_mid * dt

        # --- Euler-Cromer update ---
        if i < n - 1:
            if state == LatchState.FREE:
                v[i + 1] = v[i] + dt * a[i]
                x[i + 1] = x[i] + dt * v[i + 1]
                latch_active[i + 1] = False
            else:
                # Kinematic constraint
                v[i + 1] = 0.0
                x[i + 1] = x_latch
                latch_active[i + 1] = True

    pto_result = compute_pto_result(time, x, v, pto) if pto is not None else None

    return LatchingResult(
        time=time,
        displacement=x,
        velocity=v,
        acceleration=a,
        excitation_force=excitation_force,
        radiation_force=F_rad_arr,
        restoring_force=F_res_arr,
        latch_force=F_latch_arr,
        latch_active=latch_active,
        state_log=state_log,
        latch_engage_times=np.array(engage_times),
        latch_release_times=np.array(release_times),
        pto_result=pto_result,
        params=params,
        latching_params=latching,
        pto_params=pto,
        energy_excitation=E_exc_arr,
        energy_radiation=E_rad_arr,
        energy_latch=E_lat_arr,
    )
