"""
pto.py — Fixed linear PTO model for the WEC Digital Twin (Module 2.4).

Models a linear, passive power take-off (PTO) device attached to the heaving
buoy.  The PTO exerts a force on the buoy and extracts mechanical energy.

============================================================
EQUATION OF MOTION WITH PTO
============================================================

The frozen Cummins equation (Module 2.3C) is extended as:

    (M + A_inf) x_ddot
    + C33 x
    + integral_0^t K(t-tau) v(tau) dtau
    + B_PTO * v
    + K_PTO * x
    = F_exc(t)

where:
    B_PTO >= 0   PTO damping coefficient [N·s/m]
    K_PTO >= 0   PTO stiffness coefficient [N/m]
    v = x_dot    heave velocity [m/s]

The force exerted BY the PTO ON THE BUOY is:

    F_PTO = -B_PTO * v - K_PTO * x

Sign convention: positive heave is upward.  The PTO damping force opposes
velocity (dissipative); the PTO stiffness force opposes displacement
(spring-like, shifts resonance).

============================================================
POWER DEFINITION
============================================================

Instantaneous absorbed mechanical power:

    P_abs(t) = B_PTO * v(t)^2   [W]

This is the power dissipated by the damping element.  It is always >= 0
for B_PTO >= 0.

The reactive stiffness term K_PTO * x * v does NOT represent net absorbed
energy over a periodic steady state (it averages to zero over a full cycle),
so it is NOT included in P_abs.

Cumulative extracted energy:

    E_PTO(t) = integral_0^t P_abs(tau) dtau   [J]

============================================================
FREQUENCY-DOMAIN IMPEDANCE WITH PTO
============================================================

Under the project's e^{+i*omega*t} convention (established in Module 2.3C):

    x(t) = Re[X * e^{+i*omega*t}]
    v(t) = Re[i*omega * X * e^{+i*omega*t}]

The PTO force on the buoy transforms as:
    F_PTO = -B_PTO * v - K_PTO * x
          -> -(B_PTO * i*omega + K_PTO) * X

Adding to the left-hand side of the Cummins FD equation:

    [C33 + K_PTO - omega^2*(M + A(omega)) + i*omega*(B(omega) + B_PTO)] * X
    = F_exc

So the FD impedance with PTO is:

    Z_PTO(omega) = (C33 + K_PTO)
                  - omega^2 * (M + A(omega))
                  + i*omega * (B(omega) + B_PTO)

This is consistent with the no-PTO case (B_PTO=0, K_PTO=0 recovers Z from
Module 2.3C).

============================================================
UNITS (SI throughout)
============================================================
    B_PTO   : N·s/m
    K_PTO   : N/m
    F_PTO   : N
    P_abs   : W
    E_PTO   : J
    v       : m/s
    x       : m

============================================================
REFERENCES
============================================================
- Falnes (2002), Ocean Waves and Oscillating Systems, §6.
- Ringwood et al. (2014), Energy maximisation for wave energy converters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# PTOParameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PTOParameters:
    """
    Parameters for the fixed linear PTO model.

    Parameters
    ----------
    damping : float
        PTO damping coefficient B_PTO [N·s/m].  Must be >= 0.
        Dissipates energy; force opposes velocity.
    stiffness : float
        PTO stiffness coefficient K_PTO [N/m].  Must be >= 0.
        Spring-like; shifts resonance frequency.  Does not contribute
        to net absorbed energy over a periodic steady state.
    """

    damping: float
    stiffness: float

    def __post_init__(self) -> None:
        if self.damping < 0:
            raise ValueError(f"PTO damping must be >= 0, got {self.damping}")
        if self.stiffness < 0:
            raise ValueError(f"PTO stiffness must be >= 0, got {self.stiffness}")


# ---------------------------------------------------------------------------
# PTOResult
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PTOResult:
    """
    PTO force and power metrics computed from a time-domain simulation.

    All array fields have shape (n_time,).

    Parameters
    ----------
    time : np.ndarray
        Time grid [s].
    force : np.ndarray
        PTO force on the buoy F_PTO(t) = -B_PTO*v - K_PTO*x [N].
    absorbed_power : np.ndarray
        Instantaneous absorbed power P_abs(t) = B_PTO * v^2 [W].
        Always >= 0 for B_PTO >= 0.
    cumulative_energy : np.ndarray
        Cumulative extracted energy E_PTO(t) = integral P_abs dt [J].
        Computed via trapezoidal rule.
    mean_power : float
        Time-mean absorbed power over the full simulation [W].
    rms_force : float
        RMS PTO force [N].
    rms_velocity : float
        RMS velocity [m/s].
    max_force : float
        Maximum absolute PTO force [N].
    max_velocity : float
        Maximum absolute velocity [m/s].
    params : PTOParameters
        PTO parameters used.
    """

    time: np.ndarray
    force: np.ndarray
    absorbed_power: np.ndarray
    cumulative_energy: np.ndarray
    mean_power: float
    rms_force: float
    rms_velocity: float
    max_force: float
    max_velocity: float
    params: PTOParameters


# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------

def compute_pto_force(
    x: np.ndarray,
    v: np.ndarray,
    parameters: PTOParameters,
) -> np.ndarray:
    """
    Compute the PTO force exerted ON THE BUOY.

    F_PTO = -B_PTO * v - K_PTO * x

    The damping term opposes velocity; the stiffness term opposes displacement.

    Parameters
    ----------
    x : np.ndarray
        Heave displacement [m], shape (n,).
    v : np.ndarray
        Heave velocity [m/s], shape (n,).
    parameters : PTOParameters
        PTO damping and stiffness.

    Returns
    -------
    np.ndarray
        PTO force on buoy [N], shape (n,).
    """
    x = np.asarray(x, dtype=float)
    v = np.asarray(v, dtype=float)
    return -parameters.damping * v - parameters.stiffness * x


def compute_pto_power(
    v: np.ndarray,
    parameters: PTOParameters,
) -> np.ndarray:
    """
    Compute instantaneous absorbed PTO power.

    P_abs(t) = B_PTO * v(t)^2

    Always >= 0 for B_PTO >= 0.  The reactive stiffness term is excluded
    because K_PTO * x * v averages to zero over a periodic steady state
    and does not represent net harvested energy.

    Parameters
    ----------
    v : np.ndarray
        Heave velocity [m/s], shape (n,).
    parameters : PTOParameters
        PTO parameters.

    Returns
    -------
    np.ndarray
        Instantaneous absorbed power [W], shape (n,).
    """
    v = np.asarray(v, dtype=float)
    return parameters.damping * v**2


def compute_pto_result(
    time: np.ndarray,
    x: np.ndarray,
    v: np.ndarray,
    parameters: PTOParameters,
) -> PTOResult:
    """
    Compute full PTO result from displacement and velocity time series.

    Parameters
    ----------
    time : np.ndarray
        Time grid [s], shape (n_time,).
    x : np.ndarray
        Heave displacement [m], shape (n_time,).
    v : np.ndarray
        Heave velocity [m/s], shape (n_time,).
    parameters : PTOParameters
        PTO parameters.

    Returns
    -------
    PTOResult
    """
    time = np.asarray(time, dtype=float)
    x    = np.asarray(x,    dtype=float)
    v    = np.asarray(v,    dtype=float)

    force   = compute_pto_force(x, v, parameters)
    p_abs   = compute_pto_power(v, parameters)
    # Build cumulative integral via trapezoidal running sum
    dt = np.diff(time)
    e_running = np.zeros(len(time))
    e_running[1:] = np.cumsum(0.5 * (p_abs[:-1] + p_abs[1:]) * dt)

    return PTOResult(
        time=time,
        force=force,
        absorbed_power=p_abs,
        cumulative_energy=e_running,
        mean_power=float(np.mean(p_abs)),
        rms_force=float(np.sqrt(np.mean(force**2))),
        rms_velocity=float(np.sqrt(np.mean(v**2))),
        max_force=float(np.max(np.abs(force))),
        max_velocity=float(np.max(np.abs(v))),
        params=parameters,
    )
