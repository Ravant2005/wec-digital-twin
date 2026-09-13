"""
environment.py — Gymnasium WEC RL Control Environment (Module 5.1).

============================================================
OVERVIEW
============================================================

WECControlEnv is a Gymnasium-compatible environment wrapping:

    Module 1  — ERA5-conditioned wave replay (goa_seastate)
    Module 2  — Cummins Euler-Cromer WEC dynamics (step-by-step)
    Module 3  — WaveSensor for realistic sensor observations
    Module 4  — GRU+wind forecaster for 48h ahead prediction

The environment implements the standard Gymnasium API:

    reset(seed, options) → (observation, info)
    step(action)         → (observation, reward, terminated, truncated, info)

============================================================
PHYSICS vs CONTROL TIMESTEP
============================================================

    DT_PHYSICS  = 0.1 s    Cummins Euler-Cromer integrator step
    DT_CONTROL  = 1.0 s    RL action interval

One RL step advances N_PHYSICS_PER_CONTROL = 10 physics sub-steps.
The latch/release action is applied at the start of the control step
and held constant for all 10 sub-steps.

This timestep separation is mandatory: running the Cummins solver at
DT_CONTROL = 1 s would violate the Nyquist criterion (omega_max = 1.4
rad/s → dt < 2.24 s is satisfied, but accuracy would be very poor).
DT_PHYSICS = 0.1 s gives ~75 steps per natural period — conservative
and accurate.

============================================================
STEP-WISE CUMMINS INTEGRATOR
============================================================

Module 2's solve_cummins() is batch-only (solves the entire trajectory
at once).  For RL, we need a step-by-step interface compatible with
interleaved action decisions.

CumminsStepIntegrator (defined below) wraps the Module 2 physics by:
  1. Pre-interpolating the radiation kernel onto DT_PHYSICS.
  2. Maintaining a velocity history ring buffer for the convolution.
  3. Exposing a step(F_exc) → (x, v, a) method.

The integrator uses the IDENTICAL Euler-Cromer scheme as solve_cummins()
and the IDENTICAL convolution formula.  No physics is re-derived.

============================================================
ACTION-GATED LATCHING
============================================================

Module 2's latching (latching.py) is oracle-based (uses true future
excitation zero-crossings).  For the RL environment we need
*action-gated* latching: the agent decides when to latch/release.

The kinematic constraint is identical to solve_cummins_latching():
    LATCHED: v = 0, x = x_latch (enforced by setting v_next = 0)
    FREE:    normal Cummins + PTO dynamics

Crucially:
  - Radiation memory history is updated at EVERY physics step (v=0 during latch)
  - x_latch is captured the moment the latch is engaged
  - The latch reaction force is computed but not counted as harvested power
  - No radiation-memory erasure occurs on latch/release

============================================================
EXCITATION FORCE SCALING
============================================================

Module 1's goa_seastate() produces surface elevation eta(t) [m], not
the excitation force F_exc(t) [N] directly.  F_exc requires the BEM
excitation coefficient |F_exc(omega)| from Module 2.

For the RL environment, we compute F_exc = eta * F_exc_scale where
F_exc_scale [N/m] is the BEM excitation coefficient at the peak
frequency omega_p = 2π/Tp.  This is the linear single-frequency
approximation — consistent with the linear wave theory used throughout.

F_exc_scale is interpolated from the Module 2 hydrodynamic coefficients
at each ERA5 hour's peak frequency.

============================================================
INFORMATION MODES
============================================================

    "perfect_forecast"
        Block B of the observation is filled with TRUE future ERA5
        values (Hs_true, Tp_true, direction_true) for the next H_FORE
        hours.  This is the upper bound — oracle future information.

    "realistic_forecast"
        Block B is filled with the deterministic GRU+wind forecast
        output.  The GRU is conditioned on the Module 3 sensor
        observations (not the true sea state).
        *** MODULE 5.1 LIMITATION ***:
        Stochastic error injection from forecast_errors.csv.gz is NOT
        yet implemented.  The deterministic GRU prediction is used.
        Future work: sample from the empirical error distribution by
        lead time and sea-state class.

    "reactive"
        Block B is all zeros.  The controller sees only the sensor
        history — no predictive information.

TRUE wave values in the info dict are available at all times for
evaluation and debugging.  They NEVER appear in the observation vector
in realistic/reactive modes.

============================================================
DETERMINISM
============================================================

Given:
    - same EpisodeReplay (same df slice + same seed)
    - same EnvironmentConfig
    - same reset(seed=...)
    - same action sequence

the following are bit-for-bit identical across runs:
    - observations
    - WEC trajectory (x, v)
    - PTO power
    - reward sequence
    - cumulative energy

Local numpy RNGs are used.  Global state is not modified.

============================================================
MODULE DEPENDENCIES (read-only)
============================================================
    module2_wec.cummins          — CumminsParameters
    module2_wec.pto              — PTOParameters, compute_pto_power, compute_pto_force
    module2_wec.hydrodynamic_coefficients — HydrodynamicCoefficients
    module3_sensor.sensor        — WaveSensor, SensorParameters
    module4_forecasting.training — load_checkpoint
    module5_control.replay       — EpisodeReplay, DT_PHYSICS, DT_CONTROL,
                                   N_PHYSICS_PER_CONTROL
    module5_control.observations — ObservationConfig, ObservationBuilder,
                                   build_observation_space, H_OBS, H_FORE
    module5_control.rewards      — RewardConfig, compute_step_reward,
                                   PowerAccumulator
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from module2_wec.cummins import CumminsParameters
from module2_wec.pto import PTOParameters, compute_pto_power, compute_pto_force
from module3_sensor.sensor import WaveSensor, SensorParameters
from module4_forecasting.dataset import direction_to_sincos, sincos_to_direction

from module5_control.replay import (
    EpisodeReplay,
    DT_PHYSICS,
    DT_CONTROL,
    N_PHYSICS_PER_CONTROL,
    N_CONTROL_PER_HOUR,
    N_PHYSICS_PER_HOUR,
)
from module5_control.excitation import HourlyExcitationBuffer
from module5_control.forecast_uncertainty import ForecastErrorSampler
from module5_control.observations import (
    ObservationConfig,
    ObservationBuilder,
    build_observation_space,
    H_OBS,
    H_FORE,
    F_FORE,
    F_OBS_WAVE_WIND,
)
from module5_control.rewards import (
    RewardConfig,
    PowerAccumulator,
    compute_step_reward,
)

# ---------------------------------------------------------------------------
# Action constants
# ---------------------------------------------------------------------------

ACTION_RELEASE: int = 0
ACTION_LATCH:   int = 1


# ---------------------------------------------------------------------------
# EnvironmentConfig
# ---------------------------------------------------------------------------

@dataclass
class EnvironmentConfig:
    """
    Full configuration for WECControlEnv.

    Parameters
    ----------
    cummins_params : CumminsParameters
        Module 2 Cummins equation parameters (mass, stiffness, kernel).
    pto_params : PTOParameters
        Module 2 PTO parameters (damping, stiffness).
    hydro_excitation_amplitude : np.ndarray
        BEM excitation force amplitude |F_exc(omega)| [N/m], shape (n_omega,).
        Used to scale Module 1 eta → F_exc.
    hydro_omega : np.ndarray
        Angular frequency grid [rad/s] matching hydro_excitation_amplitude.
    sensor_params : SensorParameters
        Module 3 sensor configuration.
    obs_config : ObservationConfig
        Observation space configuration.
    reward_config : RewardConfig
        Reward function configuration.
    mode : str
        Information mode: "perfect_forecast", "realistic_forecast",
        or "reactive".
    gru_checkpoint_path : str or None
        Path to the canonical GRU+wind checkpoint
        (results/forecasting/exp_gru_wind/best_model.pt).
        Required for mode="realistic_forecast".
        Ignored for other modes.
    max_episode_steps : int or None
        Maximum RL control steps per episode before truncation.
        None = run until replay is exhausted.
    dt_physics : float
        Physics timestep [s].  Default DT_PHYSICS = 0.1 s.
    dt_control : float
        Control timestep [s].  Default DT_CONTROL = 1.0 s.
    """

    cummins_params:              CumminsParameters
    pto_params:                  PTOParameters
    hydro_excitation_amplitude:  np.ndarray
    hydro_omega:                 np.ndarray
    sensor_params:               SensorParameters
    obs_config:                  ObservationConfig       = field(default_factory=ObservationConfig)
    reward_config:               RewardConfig            = field(default_factory=RewardConfig)
    mode:                        str                     = "reactive"
    gru_checkpoint_path:         Optional[str]           = None
    max_episode_steps:           Optional[int]           = None
    dt_physics:                  float                   = DT_PHYSICS
    dt_control:                  float                   = DT_CONTROL
    # Module 5.2A: full HydrodynamicCoefficients for frequency-dependent excitation.
    # If None, falls back to the Module 5.1 single-frequency approximation.
    hydro:                       Optional[Any]           = None  # HydrodynamicCoefficients
    # Module 5.2B: empirical forecast-error sampler for realistic_forecast mode.
    # If None, deterministic GRU output is used (Module 5.1 behaviour).
    forecast_error_sampler:      Optional[Any]           = None  # ForecastErrorSampler

    def __post_init__(self) -> None:
        valid_modes = ("perfect_forecast", "realistic_forecast", "reactive")
        if self.mode not in valid_modes:
            raise ValueError(
                f"mode must be one of {valid_modes}, got '{self.mode}'"
            )
        if self.mode == "realistic_forecast" and self.gru_checkpoint_path is None:
            raise ValueError(
                "gru_checkpoint_path must be provided for mode='realistic_forecast'."
            )
        if self.dt_physics <= 0:
            raise ValueError(f"dt_physics must be > 0, got {self.dt_physics}")
        if self.dt_control <= 0:
            raise ValueError(f"dt_control must be > 0, got {self.dt_control}")


# ---------------------------------------------------------------------------
# CumminsStepIntegrator
# ---------------------------------------------------------------------------

class CumminsStepIntegrator:
    """
    Step-by-step Cummins Euler-Cromer integrator.

    Implements the IDENTICAL physics as module2_wec.cummins.solve_cummins()
    but exposes a step() method rather than a batch interface.  This is
    the adapter that makes Module 2's validated physics usable by an RL loop.

    The radiation memory convolution is:
        F_mem[n] = -dt * sum_{j=0}^{n-1} K((n-j)*dt) * v[j]

    using a rolling ring buffer of past velocities.  The kernel is
    pre-interpolated onto DT_PHYSICS once at construction time.

    During LATCHED steps the caller should pass v=0.0 to update() after
    obtaining the free-motion acceleration — or call latch_step() which
    enforces the kinematic constraint directly.

    Parameters
    ----------
    params : CumminsParameters
        Module 2 Cummins parameters (must have kernel_time, kernel_values).
    pto : PTOParameters or None
        PTO parameters.
    dt : float
        Physics timestep [s].  Default DT_PHYSICS = 0.1 s.
    """

    def __init__(
        self,
        params: CumminsParameters,
        pto:    Optional[PTOParameters] = None,
        dt:     float = DT_PHYSICS,
    ) -> None:
        self.params = params
        self.pto    = pto
        self.dt     = dt

        # Pre-interpolate kernel onto solver dt grid
        K_t   = params.kernel_time
        K_v   = params.kernel_values
        n_lag_max = len(K_t)

        lags        = np.arange(1, n_lag_max + 1) * dt
        self._K_interp  = np.interp(lags, K_t, K_v, left=0.0, right=0.0)
        self._n_lag_max = n_lag_max

        # Derived physical constants
        self._M_eff  = params.effective_mass
        self._C33    = params.hydrostatic_stiffness
        self._B_pto  = pto.damping   if pto is not None else 0.0
        self._K_pto  = pto.stiffness if pto is not None else 0.0

        # State
        self._step_count: int           = 0
        self._x:          float         = 0.0
        self._v:          float         = 0.0
        self._a:          float         = 0.0
        # Rolling velocity history for convolution — NEVER reset on latch
        self._v_history: list[float]    = []

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, x0: float = 0.0, v0: float = 0.0) -> None:
        """Reset integrator to initial conditions."""
        self._step_count = 0
        self._x          = x0
        self._v          = v0
        self._a          = 0.0
        self._v_history  = []

    # ------------------------------------------------------------------
    # Step (FREE motion)
    # ------------------------------------------------------------------

    def step(self, F_exc: float) -> Tuple[float, float, float]:
        """
        Advance one physics step under free (non-latched) Cummins dynamics.

        Implements the Euler-Cromer scheme from cummins.py:
            F_mem = -dt * sum K((n-j)*dt) * v[j]    (radiation memory)
            a     = (F_exc - C33*x - B_pto*v + F_mem) / M_eff
            v_new = v + dt * a
            x_new = x + dt * v_new    (Euler-Cromer: uses updated v)

        The PTO force (-B_pto*v - K_pto*x) is included in the dynamics
        consistent with solve_cummins() when pto is not None.

        Parameters
        ----------
        F_exc : float
            Wave excitation force [N] at this timestep.

        Returns
        -------
        x : float  heave displacement [m] at NEW state
        v : float  heave velocity [m/s] at NEW state
        a : float  heave acceleration [m/s²] at CURRENT state
        """
        # Radiation memory convolution (causal, left-endpoint quadrature)
        # Same formula as cummins.py Step 1
        n = self._step_count
        if n > 0:
            n_lag   = min(n, self._n_lag_max)
            # v_hist: v[n-1], v[n-2], ..., v[n-n_lag]
            v_hist  = self._v_history[-n_lag:][::-1]
            F_mem   = -self.dt * float(
                np.dot(self._K_interp[:n_lag], v_hist)
            )
        else:
            F_mem = 0.0

        x = self._x
        v = self._v

        # Acceleration (same as cummins.py Step 2)
        a = (
            float(F_exc)
            - (self._C33 + self._K_pto) * x
            - self._B_pto * v
            + F_mem
        ) / self._M_eff

        # Euler-Cromer update (same as cummins.py Steps 3-4)
        v_new = v + self.dt * a
        x_new = x + self.dt * v_new   # uses updated v (Euler-Cromer)

        # Store current velocity in history BEFORE updating state
        self._v_history.append(v)
        self._step_count += 1

        # Update state
        self._x = x_new
        self._v = v_new
        self._a = a

        return x_new, v_new, a

    # ------------------------------------------------------------------
    # Latch step (LATCHED: kinematic constraint)
    # ------------------------------------------------------------------

    def latch_step(self, F_exc: float, x_latch: float) -> Tuple[float, float, float]:
        """
        Advance one physics step under LATCHED constraint (v=0, x=x_latch).

        The radiation memory history is updated with v=0 at this step,
        exactly as in solve_cummins_latching().  This preserves the
        hydrodynamic memory across latch/release transitions.

        The latch reaction force is returned for diagnostic purposes but
        is NOT counted as harvested power.

        Parameters
        ----------
        F_exc : float
            Wave excitation force [N] at this timestep.
        x_latch : float
            The locked heave displacement [m].

        Returns
        -------
        x : float  = x_latch (unchanged)
        v : float  = 0.0     (kinematic constraint)
        F_latch : float  latch reaction force [N] (diagnostic only)
        """
        # Radiation memory with v=0 at this step
        n = self._step_count
        if n > 0:
            n_lag  = min(n, self._n_lag_max)
            v_hist = self._v_history[-n_lag:][::-1]
            F_mem  = -self.dt * float(
                np.dot(self._K_interp[:n_lag], v_hist)
            )
        else:
            F_mem = 0.0

        # Latch reaction force: force needed to hold v=0, x=x_latch
        # From solve_cummins_latching: F_latch = -(F_exc + F_mem - C33 * x_latch)
        F_latch = -(float(F_exc) + F_mem - self._C33 * x_latch)

        # Append v=0 to velocity history (physically correct — buoy is held still)
        self._v_history.append(0.0)
        self._step_count += 1

        # State remains at latch position
        self._x = x_latch
        self._v = 0.0
        self._a = 0.0

        return x_latch, 0.0, F_latch

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    @property
    def x(self) -> float:
        """Current heave displacement [m]."""
        return self._x

    @property
    def v(self) -> float:
        """Current heave velocity [m/s]."""
        return self._v

    @property
    def a(self) -> float:
        """Most recent heave acceleration [m/s²]."""
        return self._a

    @property
    def step_count(self) -> int:
        """Total number of physics steps taken."""
        return self._step_count

    def get_state_snapshot(self) -> dict:
        """
        Return a copy of the full integrator state for checkpointing.

        The returned dict contains all state needed to restore the
        integrator exactly, including the velocity history ring buffer.
        """
        return {
            "step_count": self._step_count,
            "x":          self._x,
            "v":          self._v,
            "a":          self._a,
            "v_history":  list(self._v_history),
        }

    def set_state_snapshot(self, snap: dict) -> None:
        """Restore integrator state from a snapshot dict."""
        self._step_count = snap["step_count"]
        self._x          = snap["x"]
        self._v          = snap["v"]
        self._a          = snap["a"]
        self._v_history  = list(snap["v_history"])


# ---------------------------------------------------------------------------
# WECControlEnv
# ---------------------------------------------------------------------------

class WECControlEnv(gym.Env):
    """
    Gymnasium WEC Latching Control Environment — Module 5.1.

    Wraps ERA5-replayed sea states, Module 2 Cummins dynamics, Module 3
    sensor model, and (optionally) the Module 4 GRU+wind forecaster into
    a standard Gymnasium interface.

    Action space
    ------------
    Discrete(2):
        0 = RELEASE  — allow free heave motion under Cummins + PTO dynamics
        1 = LATCH    — hold heave velocity at zero (kinematic constraint)

    Observation space
    -----------------
    Box(obs_dim,), dtype=float32, bounds (-inf, inf).
    obs_dim = 723 in wave_wind mode.
    See observations.py for the full schema.

    Parameters
    ----------
    replay : EpisodeReplay
        Pre-built ERA5 replay buffer (from replay.py).
    config : EnvironmentConfig
        Full environment configuration.

    Attributes
    ----------
    action_space    : spaces.Discrete(2)
    observation_space : spaces.Box

    Notes
    -----
    The ``render`` method is not implemented (no visual output).
    The environment is not thread-safe.  Use a separate instance per
    parallel RL worker.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        replay: EpisodeReplay,
        config: EnvironmentConfig,
    ) -> None:
        super().__init__()

        self._replay = replay
        self._config = config

        # --- Gymnasium spaces ---
        self.action_space      = spaces.Discrete(2)
        self.observation_space = build_observation_space(config.obs_config)

        # --- Step-wise Cummins integrator ---
        self._integrator = CumminsStepIntegrator(
            params=config.cummins_params,
            pto=config.pto_params,
            dt=config.dt_physics,
        )

        # --- Frequency-dependent excitation buffer (Module 5.2A) ---
        # Pre-builds one IrregularExcitationModel per ERA5 hour.
        # Falls back gracefully if wave_components_list is empty (legacy replay).
        self._excitation_buffer: Optional[HourlyExcitationBuffer] = None
        self._build_excitation_buffer()

        # --- Power accumulator ---
        self._power_acc = PowerAccumulator(dt_physics=config.dt_physics)

        # --- Module 3 sensor ---
        self._sensor = WaveSensor(config.sensor_params)

        # --- GRU forecast model (loaded once) ---
        self._gru_model  = None
        self._gru_config = None
        self._gru_scaler = None
        if config.mode == "realistic_forecast":
            self._load_gru_checkpoint(config.gru_checkpoint_path)

        # --- Module 5.2B: forecast-error sampler ---
        # Attached here; reset() is called in env.reset() to draw a new
        # trajectory for each episode.  If None, deterministic GRU is used.
        self._error_sampler: Optional[ForecastErrorSampler] = (
            config.forecast_error_sampler
        )

        # --- Observation builder ---
        scaler = self._gru_scaler  # may be None in reactive/perfect modes
        self._obs_builder = ObservationBuilder(config.obs_config, scaler=scaler)

        # --- Episode state (initialised by reset()) ---
        self._control_step:     int             = 0
        self._physics_step:     int             = 0
        self._latch_status:     int             = ACTION_RELEASE
        self._x_latch:          float           = 0.0
        self._prev_latch:       int             = ACTION_RELEASE

        # Observation history rolling buffer — shape (H_OBS, F_OBS)
        self._obs_history = np.zeros(
            (H_OBS, config.obs_config.n_obs_features), dtype=np.float64
        )

        # Sensor observation buffer (hourly, aligned with ERA5 hours)
        # Used to construct the H_OBS rolling window
        self._sensor_obs_buffer: list[np.ndarray] = []

        # Local RNG for environment-level randomness (not physics, which
        # is deterministic given the replay seed)
        self._rng: Optional[np.random.Generator] = None

        # Episode terminated / truncated flags
        self._terminated = False
        self._truncated  = False

    # ------------------------------------------------------------------
    # Gymnasium API: reset
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed:    Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment to the start of the replay episode.

        Parameters
        ----------
        seed : int or None
            Seeds the environment's local RNG.  Does NOT re-seed the
            replay (the replay's determinism is fixed at build time).
        options : dict or None
            Reserved for future use (start_hour, etc.).

        Returns
        -------
        observation : np.ndarray, shape (obs_dim,), dtype float32
        info : dict
            Contains the initial true sea-state diagnostics.
        """
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)

        # Reset episode counters
        self._control_step = 0
        self._physics_step = 0
        self._terminated   = False
        self._truncated    = False

        # Reset latch state
        self._latch_status = ACTION_RELEASE
        self._prev_latch   = ACTION_RELEASE
        self._x_latch      = 0.0

        # Reset physics integrator
        self._integrator.reset(x0=0.0, v0=0.0)

        # Reset power accumulator
        self._power_acc.reset()

        # Reset sensor — re-seed the sensor RNG for determinism
        sensor_seed = int(self._rng.integers(0, 2**31)) if seed is not None else None
        self._sensor = WaveSensor(
            SensorParameters(
                **{k: v for k, v in vars(self._config.sensor_params).items()
                   if k != "random_seed"},
                random_seed=sensor_seed,
            )
        )

        # Module 5.2B: draw a new error trajectory for this episode.
        # The sampler's RNG advances naturally — no re-seeding here —
        # so each episode gets a different trajectory while the full
        # sequence remains reproducible given the same initial seed.
        if self._error_sampler is not None:
            self._error_sampler.reset()

        # Build initial observation history by running the first H_OBS
        # ERA5 hours through the sensor model.  This gives the agent a
        # full 48h lookback from the very first action.
        self._build_initial_obs_history()

        # Initial observation
        forecast   = self._get_forecast()
        obs        = self._obs_builder.build(
            obs_history  = self._obs_history,
            forecast     = forecast,
            x            = self._integrator.x,
            v            = self._integrator.v,
            latch_status = self._latch_status,
        )

        info = self._make_info(
            action=None,
            reward=0.0,
            mean_power_W=0.0,
            latch_switched=False,
        )
        return obs, info

    # ------------------------------------------------------------------
    # Gymnasium API: step
    # ------------------------------------------------------------------

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Advance the environment by one RL control step.

        Parameters
        ----------
        action : int
            0 = RELEASE, 1 = LATCH.

        Returns
        -------
        observation : np.ndarray, shape (obs_dim,), dtype float32
        reward : float
        terminated : bool   — episode ended naturally (replay exhausted)
        truncated : bool    — episode cut short by max_episode_steps
        info : dict         — diagnostic information (true sea state etc.)
        """
        if self._terminated or self._truncated:
            warnings.warn(
                "step() called on a terminated/truncated environment. "
                "Call reset() first.",
                RuntimeWarning,
                stacklevel=2,
            )

        # --- Apply action ---
        latch_switched = (action != self._prev_latch)
        self._latch_status = int(action)
        if action == ACTION_LATCH and self._prev_latch == ACTION_RELEASE:
            # Capture latch position at the moment of engagement
            self._x_latch = self._integrator.x
        self._prev_latch = self._latch_status

        # --- Advance N_PHYSICS_PER_CONTROL physics sub-steps ---
        self._power_acc.begin_step()
        end_stop_violated = False

        for sub in range(N_PHYSICS_PER_CONTROL):
            global_phys = self._physics_step
            F_exc = self._get_excitation_force(global_phys)

            if self._latch_status == ACTION_LATCH:
                x_new, v_new, _ = self._integrator.latch_step(F_exc, self._x_latch)
            else:
                x_new, v_new, _ = self._integrator.step(F_exc)

            # PTO power at this sub-step (uses v AFTER the step)
            p_abs = float(compute_pto_power(
                np.array([v_new]), self._config.pto_params
            )[0])
            self._power_acc.update(p_abs)

            # End-stop check
            if abs(x_new) > self._config.reward_config.x_end_stop_m:
                end_stop_violated = True

            self._physics_step += 1

        # --- Advance ERA5 / sensor buffers every N_CONTROL_PER_HOUR steps ---
        # (i.e. every 3600 control steps = 1 ERA5 hour)
        self._control_step += 1
        self._maybe_advance_obs_buffer()

        # --- Compute reward ---
        mean_power_W = self._power_acc.mean_step_power_W()
        reward = compute_step_reward(
            mean_power_W      = mean_power_W,
            latch_switched    = latch_switched,
            end_stop_violated = end_stop_violated,
            config            = self._config.reward_config,
        )

        # --- Check termination / truncation ---
        max_steps     = (
            self._config.max_episode_steps
            if self._config.max_episode_steps is not None
            else self._replay.n_control_steps
        )
        terminated = self._physics_step >= self._replay.n_physics_steps
        truncated  = (self._control_step >= max_steps) and not terminated

        self._terminated = terminated
        self._truncated  = truncated

        # --- Build observation ---
        forecast = self._get_forecast()
        obs      = self._obs_builder.build(
            obs_history  = self._obs_history,
            forecast     = forecast,
            x            = self._integrator.x,
            v            = self._integrator.v,
            latch_status = self._latch_status,
        )

        info = self._make_info(
            action        = action,
            reward        = reward,
            mean_power_W  = mean_power_W,
            latch_switched= latch_switched,
        )

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Info dict
    # ------------------------------------------------------------------

    def _make_info(
        self,
        action:        Optional[int],
        reward:        float,
        mean_power_W:  float,
        latch_switched: bool,
    ) -> dict:
        """
        Build the info dict with true diagnostics.

        True sea-state values are ALWAYS included here for evaluation
        and debug purposes.  They must NEVER be copied into the
        observation vector in realistic/reactive modes.
        """
        era5_idx  = self._replay.get_control_step_hour(
            max(self._control_step - 1, 0)
        )
        Hs_true   = float(self._replay.Hs_true[era5_idx])
        Tp_true   = float(self._replay.Tp_true[era5_idx])
        dir_true  = float(self._replay.direction_true[era5_idx])
        ts        = (self._replay.timestamps[era5_idx]
                     if era5_idx < len(self._replay.timestamps)
                     else None)

        return {
            # --- True sea state (for evaluation/debug ONLY) ---
            "timestamp":         ts,
            "Hs_true":           Hs_true,
            "Tp_true":           Tp_true,
            "direction_true":    dir_true,
            # --- WEC state ---
            "x":                 self._integrator.x,
            "v":                 self._integrator.v,
            "latch_status":      self._latch_status,
            "latch_switched":    latch_switched,
            # --- Power / energy ---
            "instantaneous_power_W":  self._power_acc.instantaneous_power_W,
            "mean_step_power_W":      mean_power_W,
            "cumulative_energy_J":    self._power_acc.cumulative_energy_J,
            "cumulative_energy_Wh":   self._power_acc.cumulative_energy_Wh,
            # --- RL step ---
            "action":            action,
            "reward":            reward,
            "control_step":      self._control_step,
            "physics_step":      self._physics_step,
            "era5_hour_index":   era5_idx,
            # --- Mode ---
            "mode":              self._config.mode,
        }

    # ------------------------------------------------------------------
    # Excitation force (Module 5.2A — frequency-dependent)
    # ------------------------------------------------------------------

    def _build_excitation_buffer(self) -> None:
        """
        Pre-build the HourlyExcitationBuffer for this episode (Module 5.2A).

        Requires:
        - replay.wave_components_list to be non-empty (set by build_episode_replay)
        - config.hydro to be a HydrodynamicCoefficients object

        If either is missing, falls back to the Module 5.1 single-frequency
        approximation in _get_excitation_force().
        """
        if not self._replay.wave_components_list:
            self._excitation_buffer = None
            return

        if self._config.hydro is None:
            self._excitation_buffer = None
            return

        self._excitation_buffer = HourlyExcitationBuffer(
            hydro      = self._config.hydro,
            Hs_arr     = self._replay.Hs_true,
            Tp_arr     = self._replay.Tp_true,
            dir_arr    = self._replay.direction_true,
            base_seed  = self._replay.seed,
            dt_physics = self._config.dt_physics,
        )

    def _get_excitation_force(self, physics_step: int) -> float:
        """
        Return the wave excitation force [N] at a global physics step.

        Module 5.2A path (preferred):
            Uses IrregularExcitationModel — frequency-dependent irregular
            excitation from the same wave realization as eta(t).

            F_exc(t) = Σᵢ Re[ H_exc(ωᵢ) · Zᵢ · exp(i ωᵢ t) ]

        Module 5.1 fallback path:
            F_exc(t) = |H_exc(ω_p)| × eta(t)
            (single-frequency approximation using peak-period coefficient)
        """
        # --- Module 5.2A path ---
        if self._excitation_buffer is not None:
            era5_idx       = self._replay.get_era5_hour_index(physics_step)
            t_within_hour  = (physics_step % N_PHYSICS_PER_HOUR) * self._config.dt_physics
            return self._excitation_buffer.force_at_physics_step(
                hour_index    = era5_idx,
                t_within_hour = t_within_hour,
            )

        # --- Module 5.1 fallback path ---
        eta_val = self._replay.get_excitation_at_physics_step(physics_step)
        era5_idx = self._replay.get_era5_hour_index(physics_step)
        era5_idx = min(era5_idx, self._replay.n_hours - 1)

        Tp = float(self._replay.Tp_true[era5_idx])
        if not np.isfinite(Tp) or Tp <= 0:
            Tp = 8.0

        omega_p = 2.0 * np.pi / Tp
        omega_p = float(np.clip(
            omega_p,
            float(self._config.hydro_omega[0]),
            float(self._config.hydro_omega[-1]),
        ))
        F_exc_scale = float(np.interp(
            omega_p,
            self._config.hydro_omega,
            self._config.hydro_excitation_amplitude,
        ))
        return F_exc_scale * eta_val

    # ------------------------------------------------------------------
    # Observation history management
    # ------------------------------------------------------------------

    def _build_initial_obs_history(self) -> None:
        """
        Populate the obs_history buffer with the first H_OBS ERA5 hours.

        Runs all H_OBS true sea-state values through the Module 3 sensor
        model to produce noisy/delayed observations.  The resulting
        feature array fills the rolling history buffer so the agent has
        a full 48-hour lookback from the very first action.

        Wind is taken directly from ERA5 (not sensor-modelled).
        """
        n_avail = min(H_OBS, self._replay.n_hours)

        # Run sensor on all available hours at once
        hours = np.arange(n_avail, dtype=float)
        Hs_t  = self._replay.Hs_true[:n_avail]
        Tp_t  = self._replay.Tp_true[:n_avail]
        dir_t = self._replay.direction_true[:n_avail]

        # Clamp NaN to neutral defaults before passing to sensor
        Hs_t_safe  = np.where(np.isfinite(Hs_t),  Hs_t,  0.5)
        Tp_t_safe  = np.where(np.isfinite(Tp_t),  Tp_t,  8.0)
        dir_t_safe = np.where(np.isfinite(dir_t), dir_t, 270.0)

        sens_obs = self._sensor.observe(
            time           = hours,
            Hs_true        = Hs_t_safe,
            Tp_true        = Tp_t_safe,
            direction_true = dir_t_safe,
        )

        # Build feature matrix
        F = self._config.obs_config.n_obs_features
        buffer = np.zeros((n_avail, F), dtype=np.float64)

        # Wave features (indices 0–6)
        Hs_obs  = np.where(sens_obs.valid_Hs,  sens_obs.Hs_obs,  np.nan)
        Tp_obs  = np.where(sens_obs.valid_Tp,  sens_obs.Tp_obs,  np.nan)
        dir_obs = np.where(sens_obs.valid_direction, sens_obs.direction_obs, np.nan)

        sin_obs, cos_obs = direction_to_sincos(dir_obs)

        buffer[:, 0] = Hs_obs
        buffer[:, 1] = Tp_obs
        buffer[:, 2] = sin_obs
        buffer[:, 3] = cos_obs
        buffer[:, 4] = sens_obs.valid_Hs.astype(float)
        buffer[:, 5] = sens_obs.valid_Tp.astype(float)
        buffer[:, 6] = sens_obs.valid_direction.astype(float)

        # Wind features (indices 7–10), if wave_wind mode
        if self._config.obs_config.use_wind and F >= 11:
            u10 = self._replay.u10[:n_avail]
            v10 = self._replay.v10[:n_avail]
            u10_fin = np.where(np.isfinite(u10), u10, np.nan)
            v10_fin = np.where(np.isfinite(v10), v10, np.nan)
            buffer[:, 7]  = u10_fin
            buffer[:, 8]  = v10_fin
            buffer[:, 9]  = np.isfinite(u10).astype(float)
            buffer[:, 10] = np.isfinite(v10).astype(float)

        # Initialise obs_history (pad front with zeros if n_avail < H_OBS)
        self._obs_history = np.zeros((H_OBS, F), dtype=np.float64)
        self._obs_history[H_OBS - n_avail:] = buffer

        # Store in sensor buffer for the rolling-update logic
        self._sensor_obs_buffer = [buffer[i] for i in range(n_avail)]

    def _maybe_advance_obs_buffer(self) -> None:
        """
        Slide the observation history window by one hour if a new ERA5
        hour has elapsed.

        Called once per control step.  Every N_CONTROL_PER_HOUR steps
        (= 3600 steps = 1 ERA5 hour) we generate a new sensor observation
        for the current ERA5 hour and append it to the rolling history.
        """
        # Check if we've completed a full ERA5 hour
        if self._control_step % N_CONTROL_PER_HOUR != 0:
            return

        era5_idx = self._replay.get_control_step_hour(self._control_step)
        if era5_idx >= self._replay.n_hours:
            return

        F = self._config.obs_config.n_obs_features

        # Run sensor for this single hour
        Hs_t  = float(self._replay.Hs_true[era5_idx])
        Tp_t  = float(self._replay.Tp_true[era5_idx])
        dir_t = float(self._replay.direction_true[era5_idx])

        Hs_t_s  = Hs_t  if np.isfinite(Hs_t)  else 0.5
        Tp_t_s  = Tp_t  if np.isfinite(Tp_t)  else 8.0
        dir_t_s = dir_t if np.isfinite(dir_t) else 270.0

        sens_obs = self._sensor.observe(
            time           = np.array([float(era5_idx)]),
            Hs_true        = np.array([Hs_t_s]),
            Tp_true        = np.array([Tp_t_s]),
            direction_true = np.array([dir_t_s]),
        )

        new_row = np.zeros(F, dtype=np.float64)

        Hs_obs  = sens_obs.Hs_obs[0]  if sens_obs.valid_Hs[0]        else np.nan
        Tp_obs  = sens_obs.Tp_obs[0]  if sens_obs.valid_Tp[0]        else np.nan
        dir_obs = sens_obs.direction_obs[0] if sens_obs.valid_direction[0] else np.nan

        sin_obs, cos_obs = direction_to_sincos(np.array([dir_obs]))
        new_row[0] = Hs_obs
        new_row[1] = Tp_obs
        new_row[2] = float(sin_obs[0])
        new_row[3] = float(cos_obs[0])
        new_row[4] = float(sens_obs.valid_Hs[0])
        new_row[5] = float(sens_obs.valid_Tp[0])
        new_row[6] = float(sens_obs.valid_direction[0])

        if self._config.obs_config.use_wind and F >= 11:
            u10 = float(self._replay.u10[era5_idx])
            v10 = float(self._replay.v10[era5_idx])
            new_row[7]  = u10 if np.isfinite(u10) else np.nan
            new_row[8]  = v10 if np.isfinite(v10) else np.nan
            new_row[9]  = float(np.isfinite(u10))
            new_row[10] = float(np.isfinite(v10))

        # Slide rolling window: drop oldest, append newest
        self._obs_history = np.roll(self._obs_history, -1, axis=0)
        self._obs_history[-1] = new_row

    # ------------------------------------------------------------------
    # Forecast
    # ------------------------------------------------------------------

    def _get_forecast(self) -> np.ndarray:
        """
        Return the forecast block B of shape (H_FORE, 4).

        Mode dispatch:
            "perfect_forecast"   — true future ERA5 values
            "realistic_forecast" — GRU+wind deterministic prediction
            "reactive"           — zeros
        """
        mode = self._config.mode

        if mode == "reactive":
            return np.zeros((H_FORE, F_FORE), dtype=np.float64)

        if mode == "perfect_forecast":
            return self._perfect_forecast()

        # mode == "realistic_forecast"
        return self._gru_forecast()

    def _perfect_forecast(self) -> np.ndarray:
        """
        Fill Block B with TRUE future ERA5 values.

        Reads Hs_true, Tp_true, direction_true for the next H_FORE hours
        from the replay buffer.  This is oracle information — the upper
        bound for any forecaster.
        """
        era5_now = self._replay.get_control_step_hour(self._control_step)
        out      = np.zeros((H_FORE, F_FORE), dtype=np.float64)

        for h in range(H_FORE):
            idx = era5_now + h + 1   # +1: next hour onwards
            if idx >= self._replay.n_hours:
                break   # zero-pad remaining horizon
            Hs_h  = float(self._replay.Hs_true[idx])
            Tp_h  = float(self._replay.Tp_true[idx])
            dir_h = float(self._replay.direction_true[idx])

            if not np.isfinite(Hs_h):   Hs_h  = 0.0
            if not np.isfinite(Tp_h):   Tp_h  = 0.0
            if not np.isfinite(dir_h):  dir_h = 0.0

            sin_h, cos_h = direction_to_sincos(np.array([dir_h]))
            out[h, 0] = Hs_h
            out[h, 1] = Tp_h
            out[h, 2] = float(sin_h[0])
            out[h, 3] = float(cos_h[0])

        return out

    def _gru_forecast(self) -> np.ndarray:
        """
        Run the GRU+wind model on the current sensor history to produce
        the 48h forecast, then optionally inject empirical forecast errors
        (Module 5.2B).

        INPUT to GRU: the current obs_history (H_OBS × 11) scaled by the
        Module 4 scaler.  This is EXACTLY the same feature matrix the GRU
        was trained on — sensor observations passed through Module 3, then
        Module 4 preprocessing.  No true values leak in.

        OUTPUT (deterministic): shape (H_FORE, 4) in normalised space, then
        inverse-transformed to physical units before returning.

        Module 5.2B — error injection:
        If self._error_sampler is set, empirical forecast errors drawn from
        the Module 4.2 held-out test distribution are added to the
        deterministic GRU output.  The errors come from historical GRU
        mistakes on real data, NOT from the current episode's future truth.
        See forecast_uncertainty.py for full documentation.

        Returns
        -------
        np.ndarray, shape (H_FORE, 4)
            [Hs, Tp, sin_dir, cos_dir] in physical units (Hs [m], Tp [s]).
            If error_sampler is active: realistic forecast with sampled errors.
            If error_sampler is None: deterministic GRU forecast.
        """
        import torch

        if self._gru_model is None:
            # Fallback to zeros if model not loaded
            return np.zeros((H_FORE, F_FORE), dtype=np.float64)

        scaler = self._gru_scaler
        model  = self._gru_model

        # Scale the history window
        X_raw = self._obs_history.copy()   # (H_OBS, F_OBS)
        if scaler is not None and scaler.is_fitted:
            X_scaled = scaler.transform(X_raw)
        else:
            X_scaled = X_raw

        # Replace NaN with 0.0 (model convention)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        # Run GRU
        X_tensor = torch.from_numpy(
            X_scaled.astype(np.float32)
        ).unsqueeze(0)  # (1, H_OBS, F_OBS)

        model.eval()
        with torch.no_grad():
            pred = model(X_tensor)   # (1, H_FORE, 4) in normalised space

        pred_np = pred.squeeze(0).cpu().numpy()   # (H_FORE, 4)

        # Inverse-transform to physical units
        if scaler is not None and scaler.is_fitted:
            pred_np = scaler.inverse_transform_target(pred_np)

        det_forecast = pred_np.astype(np.float64)

        # Module 5.2B: apply empirical forecast errors
        if self._error_sampler is not None:
            return self._error_sampler.apply(det_forecast)

        return det_forecast

    # ------------------------------------------------------------------
    # GRU checkpoint loading
    # ------------------------------------------------------------------

    def _load_gru_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load the canonical GRU+wind checkpoint.

        Uses module4_forecasting.training.load_checkpoint() without
        modification.  The checkpoint is NOT re-trained or fine-tuned.
        """
        from module4_forecasting.training import load_checkpoint
        import torch

        device = torch.device("cpu")  # inference only; no GPU needed
        model, train_cfg, scaler = load_checkpoint(checkpoint_path, device=device)
        model.eval()

        self._gru_model  = model
        self._gru_config = train_cfg
        self._gru_scaler = scaler

        # Also update the obs_builder scaler so history is normalised
        # on the same scale the GRU was trained on.
        self._obs_builder = ObservationBuilder(
            self._config.obs_config,
            scaler=scaler,
        )

    # ------------------------------------------------------------------
    # Gymnasium utilities
    # ------------------------------------------------------------------

    def render(self) -> None:
        """Not implemented — no visual output in Module 5.1."""
        pass

    def close(self) -> None:
        """Clean up resources."""
        pass

    # ------------------------------------------------------------------
    # Convenience properties for tests
    # ------------------------------------------------------------------

    @property
    def replay(self) -> EpisodeReplay:
        """The underlying EpisodeReplay buffer."""
        return self._replay

    @property
    def config(self) -> EnvironmentConfig:
        """The environment configuration."""
        return self._config

    @property
    def integrator(self) -> CumminsStepIntegrator:
        """The step-wise Cummins integrator (for introspection/tests)."""
        return self._integrator

    @property
    def power_accumulator(self) -> PowerAccumulator:
        """The power accumulator (for introspection/tests)."""
        return self._power_acc

    @property
    def latch_status(self) -> int:
        """Current latch status: 0=FREE, 1=LATCHED."""
        return self._latch_status


# ---------------------------------------------------------------------------
# Factory: build_env_from_checkpoint
# ---------------------------------------------------------------------------

def build_env_from_checkpoint(
    replay:              EpisodeReplay,
    cummins_params:      CumminsParameters,
    pto_params:          PTOParameters,
    hydro_excitation_amplitude: np.ndarray,
    hydro_omega:         np.ndarray,
    mode:                str = "reactive",
    gru_checkpoint_path: Optional[str] = None,
    sensor_params:       Optional[SensorParameters] = None,
    obs_config:          Optional[ObservationConfig] = None,
    reward_config:       Optional[RewardConfig] = None,
    max_episode_steps:   Optional[int] = None,
    hydro:               Optional[Any] = None,
    forecast_error_sampler: Optional[Any] = None,
) -> WECControlEnv:
    """
    Convenience factory that builds a WECControlEnv with sensible defaults.

    Parameters
    ----------
    replay : EpisodeReplay
    cummins_params : CumminsParameters
    pto_params : PTOParameters
    hydro_excitation_amplitude : np.ndarray
        BEM |F_exc(omega)| [N/m], shape (n_omega,).
    hydro_omega : np.ndarray
        BEM frequency grid [rad/s], shape (n_omega,).
    mode : str
        "perfect_forecast", "realistic_forecast", or "reactive".
    gru_checkpoint_path : str or None
        Required for mode="realistic_forecast".
    sensor_params : SensorParameters or None
        Default: moderate noise, 1-step delay, 5% missing rate.
    obs_config : ObservationConfig or None
        Default: ObservationConfig().
    reward_config : RewardConfig or None
        Default: RewardConfig().
    max_episode_steps : int or None
    hydro : HydrodynamicCoefficients or None
        Full BEM coefficients for Module 5.2A frequency-dependent excitation.
        If None, falls back to the Module 5.1 single-frequency approximation.
    forecast_error_sampler : ForecastErrorSampler or None
        Module 5.2B empirical forecast-error sampler.
        If None, deterministic GRU output is used (no error injection).
        Only active in mode="realistic_forecast".

    Returns
    -------
    WECControlEnv
    """
    if sensor_params is None:
        sensor_params = SensorParameters(
            sampling_interval   = 3600.0,
            hs_noise_std        = 0.1,    # ~5–10% of typical Hs=1.5m
            tp_noise_std        = 0.5,    # ~6% of typical Tp=8s
            direction_noise_std = 10.0,   # degrees
            delay_steps         = 1,      # 1 hour sensor delay
            missing_probability = 0.05,   # 5% missing rate
            random_seed         = 42,
        )

    if obs_config is None:
        obs_config = ObservationConfig()

    if reward_config is None:
        reward_config = RewardConfig()

    cfg = EnvironmentConfig(
        cummins_params              = cummins_params,
        pto_params                  = pto_params,
        hydro_excitation_amplitude  = hydro_excitation_amplitude,
        hydro_omega                 = hydro_omega,
        sensor_params               = sensor_params,
        obs_config                  = obs_config,
        reward_config               = reward_config,
        mode                        = mode,
        gru_checkpoint_path         = gru_checkpoint_path,
        max_episode_steps           = max_episode_steps,
        hydro                       = hydro,
        forecast_error_sampler      = forecast_error_sampler,
    )

    return WECControlEnv(replay=replay, config=cfg)
