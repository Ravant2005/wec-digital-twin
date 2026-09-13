"""
controllers.py — Deterministic baseline controllers for WEC latching (Module 5.2C).

============================================================
PURPOSE
============================================================

Provides the controller hierarchy used for benchmarking before any RL
training begins.  All controllers share a common BaseController interface
and are designed to be used with WECControlEnv.

The hierarchy is:

    BaseController
    ├── PassiveController           — always RELEASE (never latches)
    ├── FixedThresholdController    — LATCH when |v| < threshold
    ├── ReactiveController          — same as FixedThreshold, forced reactive mode
    ├── OracleThresholdController   — same rule, but runs in perfect_forecast env
    └── RealisticThresholdController — same rule, but runs in realistic_forecast env

============================================================
CONTROLLER INTERFACE
============================================================

All controllers implement:

    reset()                       → None
        Called once before each episode.  Resets any internal state.
        Does NOT interact with the environment.

    act(obs, info) → int
        Called once per RL control step (1 Hz = DT_CONTROL = 1.0 s).
        Returns action: 0 = RELEASE, 1 = LATCH.

        Parameters:
            obs  : np.ndarray, shape (obs_dim,), dtype float32
                   Flat observation vector from env.step() or env.reset().
            info : dict
                   Info dict from env.step() or env.reset().

        Safe info keys (no future truth):
            x, v, latch_status, latch_switched,
            instantaneous_power_W, mean_step_power_W,
            cumulative_energy_J, cumulative_energy_Wh,
            control_step, physics_step, era5_hour_index, action, reward, mode.

        FORBIDDEN for realistic/reactive controllers:
            Hs_true, Tp_true, direction_true — these are present-hour oracle values
            replay.Hs_true[future]           — future truth from the replay buffer

============================================================
PASSIVE CONTROLLER (Baseline 1)
============================================================

Definition: always returns ACTION_RELEASE = 0.

The buoy undergoes free heave motion under the Cummins equation with:
    - full PTO damping (B_PTO = 200 kN·s/m as configured in the environment)
    - no latching

Physical interpretation:
    The PTO is a passive damper.  No active control is applied.
    Energy extraction = B_PTO × v² integrated over time.
    This is the simplest possible WEC operating mode.

PTO damping: 200,000 N·s/m (200 kN·s/m).
Source: PTOParameters(damping=200_000.0, stiffness=0.0) as used throughout Module 5.

============================================================
FIXED-THRESHOLD LATCHING CONTROLLER (Baseline 2)
============================================================

Definition:

    LATCH   when |v| < v_latch_threshold   (buoy near velocity zero-crossing)
    RELEASE when |v| >= v_latch_threshold  (buoy has built sufficient velocity)

where:
    v = heave velocity from info["v"]  [m/s]
    v_latch_threshold = configurable [m/s]

Default development baseline: v_latch_threshold = 0.05 m/s.

Rationale:
    Classical reactive latching engages the latch near velocity zero-crossings
    to hold the buoy while the incoming wave builds up excitation force, then
    releases at a specified condition.  Falnes (2002) §6 describes this as
    the foundational latching concept.

    Using |v| < threshold as the LATCH condition is one of the simplest
    realisations.  When velocity is near zero, the buoy is at a turning point;
    holding it there allows radiation memory to build and the next wave to
    apply force.  Release at |v| >= threshold resumes free motion.

Scientific note:
    The threshold value of 0.05 m/s is a DEVELOPMENT BASELINE.
    It has NOT been optimised against any replay segment.
    It was selected before any benchmark results were computed.
    The natural velocity scale for the reference buoy in a 1.5 m/8 s sea
    state is O(0.1–0.2 m/s) based on the physics sanity check in Module 5.2A
    (max |v| ~ 1.82 m/s in a 1-hour run); 0.05 m/s represents ~25% of that
    scale, placing the latch trigger well before zero without chasing noise.

    A minimum hold time (min_latch_steps) prevents immediate re-latching
    after release, avoiding high-frequency toggling at the DT_CONTROL = 1 s
    resolution.

INFORMATION BOUNDARY:
    Uses only info["v"] — the current heave velocity, which is a physically
    measured quantity at the buoy location.  No future truth accessed.

Units:
    v_latch_threshold : m/s
    min_latch_steps   : control steps (each = DT_CONTROL = 1.0 s)
    min_release_steps : control steps

============================================================
REACTIVE CONTROLLER (Baseline 3)
============================================================

Definition: identical physics to FixedThresholdController.

The INFORMATION difference is enforced at the environment level:
    mode = "reactive"
    Block B of the observation = all zeros (forecast withheld)

The controller itself may use Block C (WEC state) and Block A (sensor
history), but must NOT use Block B.  Since ReactiveController uses only
info["v"] (from Block C / the info dict), it naturally satisfies this
requirement.

This establishes the baseline: "what can a threshold controller achieve
without any forecast information?"

============================================================
ORACLE / PERFECT-FORECAST PATHWAY (Baseline 4)
============================================================

The environment provides the oracle pathway automatically when
mode = "perfect_forecast" is set in EnvironmentConfig.

OracleThresholdController applies the SAME FixedThreshold rule
but runs in a perfect_forecast environment.  At this stage, before PPO
training, we do not have a trained policy that exploits the forecast.

This establishes: "what does the threshold rule achieve when the
environment happens to have a true future forecast in Block B?"

IMPORTANT: At this stage OracleThresholdController is NOT a trained
RL policy.  It is the same threshold rule running in the oracle environment.
The "ceiling" for a forecast-exploiting policy cannot be claimed until
a policy that actually uses Block B is trained.  This pathway is
labelled ORACLE_INTERFACE_ONLY — not yet a true oracle comparison.

============================================================
REALISTIC FORECAST PATHWAY (Baseline 5)
============================================================

RealisticThresholdController applies the SAME FixedThreshold rule
but runs in a realistic_forecast environment with ForecastErrorSampler.

Like the oracle case, this is NOT yet a trained RL policy exploiting
the forecast.  It is the threshold rule running under realistic forecast
conditions.  Labelled REALISTIC_INTERFACE_ONLY.

============================================================
WHY NO TRAINED RL YET
============================================================

Controllers 4 and 5 require an RL policy that actually exploits Block B
(the 48-hour forecast window) to decide latching timing.  Such a policy
can only be obtained after PPO/DQN training on the environment, which is
explicitly deferred to Module 5.3.

The current controllers confirm that:
1. The environment infrastructure supports all three information modes.
2. The same threshold rule can be applied in any mode.
3. The benchmark framework is ready to receive a trained policy.

============================================================
REFERENCES
============================================================
- Falnes (2002), Ocean Waves and Oscillating Systems, §6 — latching control.
- Babarit & Clement (2006), Optimal latching control of a wave energy device.
- Ringwood et al. (2014), Energy maximisation for wave energy converters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np

from module5_control.observations import _BLOCK_B_START, _BLOCK_B_END

# ---------------------------------------------------------------------------
# Action constants (re-exported for convenience)
# ---------------------------------------------------------------------------
ACTION_RELEASE: int = 0
ACTION_LATCH:   int = 1


# ---------------------------------------------------------------------------
# BaseController
# ---------------------------------------------------------------------------

class BaseController(ABC):
    """
    Abstract base class for all WEC latching controllers.

    All controllers implement:
        reset()             → None
        act(obs, info)      → int  (0=RELEASE, 1=LATCH)

    The act() method is called once per RL control step at 1 Hz
    (DT_CONTROL = 1.0 s).  It receives the Gymnasium observation
    vector and the info dict from env.step() / env.reset().

    Subclasses must NOT access future truth (Hs_true, Tp_true, etc.)
    unless they are explicitly oracle controllers.
    """

    @property
    def name(self) -> str:
        """Human-readable controller name."""
        return self.__class__.__name__

    @property
    def label(self) -> str:
        """
        Short machine-readable label used in benchmark CSV columns.
        Override in subclasses for custom labels.
        """
        return self.name.lower()

    @abstractmethod
    def reset(self) -> None:
        """
        Reset any internal controller state.

        Called once before each episode, after env.reset().
        Does NOT call env.reset() itself.
        """
        ...

    @abstractmethod
    def act(self, obs: np.ndarray, info: dict) -> int:
        """
        Decide the next action.

        Parameters
        ----------
        obs : np.ndarray, shape (obs_dim,), dtype float32
            Flattened observation from env.step() or env.reset().
        info : dict
            Info dict from env.step() or env.reset().

        Returns
        -------
        int : 0 = RELEASE, 1 = LATCH
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ---------------------------------------------------------------------------
# PassiveController
# ---------------------------------------------------------------------------

class PassiveController(BaseController):
    """
    Passive WEC controller — always RELEASE, never LATCH.

    The buoy undergoes free heave motion under Cummins + PTO dynamics.
    Energy extraction = B_PTO × v² integrated over time.
    No active control is applied.

    Passive PTO: B_PTO = 200,000 N·s/m (set in environment PTOParameters).
    This value is NOT set here — it is a property of the environment
    configuration.  PassiveController simply never issues a latch command.

    This is Baseline 1 in the Module 5.2C benchmark.
    """

    @property
    def label(self) -> str:
        return "passive"

    def reset(self) -> None:
        """No internal state to reset."""
        pass

    def act(self, obs: np.ndarray, info: dict) -> int:
        """Always return RELEASE."""
        return ACTION_RELEASE

    def __repr__(self) -> str:
        return "PassiveController(B_PTO=200 kN·s/m, always RELEASE)"


# ---------------------------------------------------------------------------
# FixedThresholdConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FixedThresholdConfig:
    """
    Configuration for the FixedThresholdController.

    Parameters
    ----------
    v_latch_threshold : float
        Velocity magnitude threshold [m/s] for LATCH engagement.
        LATCH  when |v| <  v_latch_threshold  (velocity near zero-crossing)
        RELEASE when |v| >= v_latch_threshold  (velocity has grown enough)
        Default 0.05 m/s — DEVELOPMENT BASELINE, not tuned.
        Selected BEFORE any benchmark evaluation.
    min_latch_steps : int
        Minimum number of control steps (each = 1.0 s) to remain LATCHED
        before release is permitted.  Default 3 (3 s minimum hold time).
        Prevents immediate re-latching at DT_CONTROL = 1 s resolution.
    min_release_steps : int
        Minimum number of control steps to remain RELEASED before the
        next LATCH can be engaged.  Default 3 (3 s dead time).
        Prevents high-frequency latch cycling.
    """

    v_latch_threshold: float = 0.05   # [m/s] — DEVELOPMENT BASELINE, not tuned
    min_latch_steps:   int   = 4      # [control steps = s] ≈ T_n/2 ≈ 3.78 s rounded up
    min_release_steps: int   = 3      # [control steps = seconds]

    def __post_init__(self) -> None:
        if self.v_latch_threshold < 0:
            raise ValueError(f"v_latch_threshold must be >= 0, got {self.v_latch_threshold}")
        if self.min_latch_steps < 0:
            raise ValueError(f"min_latch_steps must be >= 0, got {self.min_latch_steps}")
        if self.min_release_steps < 0:
            raise ValueError(f"min_release_steps must be >= 0, got {self.min_release_steps}")


# ---------------------------------------------------------------------------
# FixedThresholdController
# ---------------------------------------------------------------------------

class FixedThresholdController(BaseController):
    """
    Fixed-threshold latching controller — Baseline 2.

    LATCH   when |v| <  v_latch_threshold
    RELEASE when |v| >= v_latch_threshold

    Velocity v is read from info["v"] (the current heave velocity [m/s]).
    No future truth is accessed.

    This is a deterministic, reactive, state-based controller.
    It uses only the current WEC velocity — not the forecast (Block B),
    not the sensor history (Block A), not any future information.

    Threshold: 0.05 m/s (development baseline, selected before evaluation).

    Physical rationale (Falnes 2002, §6):
        Classical latching engages near velocity zero-crossings to allow
        the WEC to build phase coherence with the incoming wave excitation.
        Threshold |v| < 0.05 m/s ~ 25% of typical peak velocity in a
        1.5 m / 8 s sea state, placing the latch trigger near — but not
        exactly at — the turning point.

    Parameters
    ----------
    config : FixedThresholdConfig
        Threshold and timing parameters.
    """

    def __init__(self, config: Optional[FixedThresholdConfig] = None) -> None:
        self.config = config or FixedThresholdConfig()
        self._latch_steps:   int = 0   # steps spent in current LATCH interval
        self._release_steps: int = 0   # steps spent in current RELEASE interval
        self._current_action: int = ACTION_RELEASE

    @property
    def label(self) -> str:
        return "fixed_threshold"

    def reset(self) -> None:
        """Reset internal timing counters."""
        self._latch_steps    = 0
        self._release_steps  = 0
        self._current_action = ACTION_RELEASE

    def act(self, obs: np.ndarray, info: dict) -> int:
        """
        Decide action based on current velocity magnitude and timer.

        Logic
        -----
        If currently LATCHED:
            - Increment latch counter.
            - TIMER-BASED release: unconditionally release after min_latch_steps.
              (v = 0 while latched by kinematic constraint; velocity-based
               release is physically impossible while the latch is engaged.)

        If currently RELEASED:
            - Increment release counter.
            - Latch if |v| < threshold AND min_release_steps have elapsed.

        Information boundary: uses only info["v"] — no future truth.

        Parameters
        ----------
        obs : np.ndarray  (not used — velocity comes from info dict)
        info : dict

        Returns
        -------
        int : 0 = RELEASE, 1 = LATCH
        """
        v     = float(info["v"])
        v_abs = abs(v)
        cfg   = self.config

        if self._current_action == ACTION_LATCH:
            self._latch_steps += 1
            # TIMER-BASED release after min_latch_steps.
            # Note: velocity-based release (|v| >= threshold) can never
            # trigger while latched because v = 0 by kinematic constraint.
            if self._latch_steps >= cfg.min_latch_steps:
                self._current_action = ACTION_RELEASE
                self._latch_steps    = 0
                self._release_steps  = 0

        else:  # currently RELEASED
            self._release_steps += 1
            # Latch when velocity drops below threshold AND minimum release elapsed.
            if (v_abs < cfg.v_latch_threshold
                    and self._release_steps >= cfg.min_release_steps):
                self._current_action = ACTION_LATCH
                self._release_steps  = 0
                self._latch_steps    = 0

        return self._current_action

    def __repr__(self) -> str:
        return (
            f"FixedThresholdController("
            f"v_threshold={self.config.v_latch_threshold} m/s, "
            f"min_latch={self.config.min_latch_steps}s, "
            f"min_release={self.config.min_release_steps}s)"
        )


# ---------------------------------------------------------------------------
# ReactiveController
# ---------------------------------------------------------------------------

class ReactiveController(FixedThresholdController):
    """
    Reactive controller — Baseline 3.

    Identical physics rule to FixedThresholdController.

    The key difference is at the ENVIRONMENT level:
        The environment must be constructed with mode='reactive',
        which sets Block B of the observation to all zeros (no forecast).

    ReactiveController enforces an additional check in act(): it verifies
    that the Block B slice is all zeros when assertions are enabled, to
    guard against accidental information leakage.  In production (no
    assertions), behavior is identical to FixedThresholdController.

    This establishes Baseline 3: "what does this rule achieve without any
    forecast information?"
    """

    @property
    def label(self) -> str:
        return "reactive"

    def act(self, obs: np.ndarray, info: dict) -> int:
        """
        Act using only WEC state (info["v"]).

        Verifies Block B = zeros (reactive mode enforcement).
        Falls back silently in production if assertion is disabled.
        """
        # Reactive mode guard: Block B must be zeros (forecast withheld)
        # This is a debug-time check; it is NOT the information boundary itself —
        # the boundary is enforced by the environment's mode='reactive'.
        assert np.allclose(obs[_BLOCK_B_START:_BLOCK_B_END], 0.0), (
            "ReactiveController: Block B is not zero. "
            "Ensure environment is constructed with mode='reactive'."
        )
        return super().act(obs, info)

    def __repr__(self) -> str:
        return (
            f"ReactiveController("
            f"v_threshold={self.config.v_latch_threshold} m/s, "
            f"env_mode=reactive)"
        )


# ---------------------------------------------------------------------------
# OracleThresholdController
# ---------------------------------------------------------------------------

class OracleThresholdController(FixedThresholdController):
    """
    Oracle / Perfect-Forecast pathway — Baseline 4.

    Applies the SAME FixedThreshold rule but runs in a
    perfect_forecast environment (mode='perfect_forecast').

    IMPORTANT SCIENTIFIC NOTE:
        This is NOT a trained RL policy that exploits the forecast.
        It is the same deterministic threshold rule applied in an
        environment where Block B contains the true future 48h sea state.

        At this stage, the controller does NOT read Block B or make
        decisions based on it.  The oracle pathway is INTERFACE_ONLY:
        it confirms the environment provides true future information,
        but no controller yet exploits it.

        The true oracle comparison (how much energy a policy that
        actually uses the perfect forecast can achieve) requires a
        trained PPO/DQN policy — deferred to Module 5.3.

        Label: ORACLE_INTERFACE_ONLY

    Latch rule: same as FixedThresholdController.
    Information used: info["v"] only.
    """

    @property
    def label(self) -> str:
        return "oracle_interface"

    @property
    def benchmark_status(self) -> str:
        return "ORACLE_INTERFACE_ONLY — threshold rule only; no trained policy uses the forecast"

    def __repr__(self) -> str:
        return (
            f"OracleThresholdController("
            f"v_threshold={self.config.v_latch_threshold} m/s, "
            f"env_mode=perfect_forecast, "
            f"status=INTERFACE_ONLY)"
        )


# ---------------------------------------------------------------------------
# RealisticThresholdController
# ---------------------------------------------------------------------------

class RealisticThresholdController(FixedThresholdController):
    """
    Realistic Forecast pathway — Baseline 5.

    Applies the SAME FixedThreshold rule but runs in a
    realistic_forecast environment (mode='realistic_forecast' +
    ForecastErrorSampler from Module 5.2B).

    IMPORTANT SCIENTIFIC NOTE:
        This is NOT a trained RL policy that exploits the GRU forecast.
        It is the same deterministic threshold rule applied in an
        environment where Block B contains the GRU+wind forecast with
        empirical error injection.

        The controller does NOT read Block B.  This is INTERFACE_ONLY:
        it confirms that the realistic forecast pathway works end-to-end,
        but no controller yet makes decisions based on the forecast content.

        The true realistic forecast comparison requires a trained
        PPO/DQN policy — deferred to Module 5.3.

        Label: REALISTIC_INTERFACE_ONLY

    No future truth is accessed by this controller (same guarantee
    as FixedThresholdController).
    """

    @property
    def label(self) -> str:
        return "realistic_interface"

    @property
    def benchmark_status(self) -> str:
        return "REALISTIC_INTERFACE_ONLY — threshold rule only; no trained policy uses the forecast"

    def __repr__(self) -> str:
        return (
            f"RealisticThresholdController("
            f"v_threshold={self.config.v_latch_threshold} m/s, "
            f"env_mode=realistic_forecast+5.2B, "
            f"status=INTERFACE_ONLY)"
        )


# ---------------------------------------------------------------------------
# Factory utility
# ---------------------------------------------------------------------------

def make_controller(
    controller_type: str,
    config: Optional[FixedThresholdConfig] = None,
) -> BaseController:
    """
    Instantiate a named controller.

    Parameters
    ----------
    controller_type : str
        One of: "passive", "fixed_threshold", "reactive",
                "oracle_interface", "realistic_interface".
    config : FixedThresholdConfig or None
        Threshold config for latching controllers.  Defaults to
        FixedThresholdConfig() (v_threshold=0.05 m/s).

    Returns
    -------
    BaseController
    """
    cfg = config or FixedThresholdConfig()
    mapping = {
        "passive":              PassiveController(),
        "fixed_threshold":      FixedThresholdController(cfg),
        "reactive":             ReactiveController(cfg),
        "oracle_interface":     OracleThresholdController(cfg),
        "realistic_interface":  RealisticThresholdController(cfg),
    }
    key = controller_type.lower().strip()
    if key not in mapping:
        raise ValueError(
            f"Unknown controller_type '{controller_type}'. "
            f"Valid options: {sorted(mapping.keys())}"
        )
    return mapping[key]
