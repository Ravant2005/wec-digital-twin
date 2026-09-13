"""
rewards.py — Reward configuration and calculation (Module 5.1).

============================================================
REWARD DESIGN
============================================================

Tier-1 reward is dominated by harvested power with small penalties
for switching and end-stop violations.

    r(t) = P_norm(t) - λ_switch · I_switch(t) - λ_end_stop · I_end_stop(t)

where:

    P_norm(t)       Normalised mean absorbed PTO power over the control
                    interval [dimensionless, ≈ O(1) for typical sea states].
                    = mean(P_abs) / P_ref
                    P_abs = B_PTO × v² [W]  (always ≥ 0)
                    P_ref = reference power scale [W]

    λ_switch        Switching penalty coefficient [dimensionless].
                    Applied once each time the latch state changes.
                    Prevents pathological rapid switching.

    I_switch(t)     1 if latch state changed this control step, else 0.

    λ_end_stop      End-stop penalty coefficient [dimensionless].
                    Applied when |x| > x_end_stop_m.
                    Module 2 does NOT currently implement a hard end-stop
                    constraint; this provides a clean hook for the future.

    I_end_stop(t)   1 if |x| > x_end_stop_m at any physics step in the
                    control interval, else 0.

============================================================
SCALING RATIONALE
============================================================

P_ref is the reference power scale used for normalisation.  A sensible
default is the mean PTO power for a passive (non-latching) WEC in a
moderate sea state.  Empirically for the reference buoy with B_PTO=200 kN·s/m
in a 1.5 m, 8 s sea state, the mean absorbed power is approximately 1–5 kW.

Setting P_ref = 1000 W (1 kW) gives rewards of order 1–5 for typical
moderate sea states, which is a numerically stable range for PPO/DQN.

The switching penalty λ_switch = 0.01 (1% of P_ref) discourages rapid
oscillation without strongly limiting the agent's latching strategy.

The end-stop coefficient λ_end_stop = 0.1 (10% of P_ref per step) is a
moderate deterrent.  It is a hook — the end-stop is not enforced by Module 2.

All coefficients are exposed through RewardConfig and must be tuned as
part of the RL training process.  Do not hard-code large penalty values.

============================================================
ENERGY TRACKING
============================================================

The environment tracks:
    instantaneous_power_W     : P_abs at the current physics step [W]
    cumulative_energy_J       : Σ P_abs × dt_physics [J]
    cumulative_energy_Wh      : cumulative_energy_J / 3600 [Wh]
    episode_energy_J          : total energy for the episode [J]

Energy integration uses the actual physics timestep (DT_PHYSICS = 0.1 s),
not the RL action interval (DT_CONTROL = 1.0 s).  This ensures that
energy is correctly integrated over the 10 physics sub-steps per action.

============================================================
UNITS
============================================================
    P_abs       : W
    P_ref       : W
    energy      : J  (internally), Wh (reported)
    λ_switch    : dimensionless
    λ_end_stop  : dimensionless
    x_end_stop  : m
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# RewardConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RewardConfig:
    """
    Configuration for the Tier-1 reward function.

    All coefficients are dimensionless (relative to P_ref).

    Parameters
    ----------
    p_ref_W : float
        Reference power for normalisation [W].
        Default 1000 W (1 kW) — approximate mean PTO power for the
        reference buoy in a 1.5 m / 8 s sea state with B_PTO = 200 kN·s/m.
        Rationale: keeps reward ≈ O(1) for typical operating conditions,
        which is numerically stable for PPO / DQN value function learning.
    lambda_switch : float
        Switching penalty coefficient [dimensionless, ≥ 0].
        Applied once per control step when latch state changes.
        Default 0.01: 1% of P_ref — discourages rapid switching while
        still allowing frequent latch/release cycles.
    lambda_end_stop : float
        End-stop penalty coefficient [dimensionless, ≥ 0].
        Applied per control step when |x| exceeds x_end_stop_m.
        Default 0.1: 10% of P_ref — provides a deterrent without
        dominating the reward signal.
        NOTE: This is a future hook.  Module 2 does not currently enforce
        a hard end-stop constraint.  Set to 0 to disable.
    x_end_stop_m : float
        End-stop threshold [m].  Default 3.0 m (conservative for a
        5 m radius, 10 m draft cylinder).
        Set to a large value (e.g. 1e9) to disable end-stop penalties.
    """

    p_ref_W:          float = 1_000.0  # 1 kW
    lambda_switch:    float = 0.01
    lambda_end_stop:  float = 0.1
    x_end_stop_m:     float = 3.0

    def __post_init__(self) -> None:
        if self.p_ref_W <= 0:
            raise ValueError(f"p_ref_W must be > 0, got {self.p_ref_W}")
        if self.lambda_switch < 0:
            raise ValueError(f"lambda_switch must be >= 0, got {self.lambda_switch}")
        if self.lambda_end_stop < 0:
            raise ValueError(f"lambda_end_stop must be >= 0, got {self.lambda_end_stop}")
        if self.x_end_stop_m <= 0:
            raise ValueError(f"x_end_stop_m must be > 0, got {self.x_end_stop_m}")


# ---------------------------------------------------------------------------
# compute_step_reward
# ---------------------------------------------------------------------------

def compute_step_reward(
    mean_power_W:       float,
    latch_switched:     bool,
    end_stop_violated:  bool,
    config:             RewardConfig,
) -> float:
    """
    Compute the scalar reward for one RL control step.

    Parameters
    ----------
    mean_power_W : float
        Mean PTO absorbed power over the control interval [W].
        = mean(B_PTO × v²) over the physics sub-steps.
        Always >= 0 for B_PTO >= 0.
    latch_switched : bool
        True if the latch state changed at the start of this step.
    end_stop_violated : bool
        True if |x| > x_end_stop_m at any physics sub-step.
    config : RewardConfig

    Returns
    -------
    float
        Scalar reward value.
        Positive values indicate energy harvesting.
        Negative values are possible only if switching/end-stop penalties
        exceed the power term (rare in normal operation).

    Notes
    -----
    The reward is dimensionless (normalised by P_ref).
    A passive (no-latch) WEC in a 1.5 m/8 s sea state should yield
    r ≈ 1.0–5.0 per step; an ideal latch controller should yield more.
    """
    # Normalised power term (always >= 0)
    power_term = float(mean_power_W) / config.p_ref_W

    # Switching penalty: applies once per step if state changed
    switch_term = config.lambda_switch * float(latch_switched)

    # End-stop penalty: applies if displacement exceeded threshold
    end_stop_term = config.lambda_end_stop * float(end_stop_violated)

    reward = power_term - switch_term - end_stop_term

    # Finite-value guard
    if not np.isfinite(reward):
        reward = 0.0

    return reward


# ---------------------------------------------------------------------------
# PowerAccumulator
# ---------------------------------------------------------------------------

class PowerAccumulator:
    """
    Accumulates PTO power and energy across physics sub-steps.

    Used by WECControlEnv to track energy at the correct physics timestep,
    independent of the RL action interval.

    Parameters
    ----------
    dt_physics : float
        Physics integration timestep [s].  Default 0.1 s.
    """

    def __init__(self, dt_physics: float = 0.1) -> None:
        self.dt_physics = dt_physics
        self.reset()

    def reset(self) -> None:
        """Reset all accumulators to zero."""
        self._instantaneous_power_W: float = 0.0
        self._cumulative_energy_J:   float = 0.0
        self._step_power_sum:        float = 0.0
        self._step_count:            int   = 0

    def update(self, power_W: float) -> None:
        """
        Record the instantaneous power at one physics sub-step.

        Parameters
        ----------
        power_W : float
            PTO absorbed power at this physics step [W].  Must be >= 0.
        """
        p = max(float(power_W), 0.0)  # enforce non-negativity
        self._instantaneous_power_W  = p
        self._cumulative_energy_J   += p * self.dt_physics
        self._step_power_sum        += p
        self._step_count            += 1

    def mean_step_power_W(self) -> float:
        """
        Mean power over the physics sub-steps accumulated since last
        begin_step() call [W].
        """
        if self._step_count == 0:
            return 0.0
        return self._step_power_sum / self._step_count

    def begin_step(self) -> None:
        """Reset per-control-step accumulators (call at each RL action)."""
        self._step_power_sum = 0.0
        self._step_count     = 0

    @property
    def instantaneous_power_W(self) -> float:
        """Most recent instantaneous PTO power [W]."""
        return self._instantaneous_power_W

    @property
    def cumulative_energy_J(self) -> float:
        """Total cumulative absorbed energy since reset() [J]."""
        return self._cumulative_energy_J

    @property
    def cumulative_energy_Wh(self) -> float:
        """Total cumulative absorbed energy since reset() [Wh]."""
        return self._cumulative_energy_J / 3600.0
