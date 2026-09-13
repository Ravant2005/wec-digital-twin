"""
module5_control — Gymnasium-compatible WEC RL control environment (Module 5.1).

Module structure
----------------
replay.py        — ERA5 sea-state replay: loads a DataFrame slice and produces
                   per-timestep (Hs, Tp, direction, u10, v10) sequences.
observations.py  — Observation space definition, builder, and normaliser.
rewards.py       — Reward configuration dataclass and calculation functions.
environment.py   — WECControlEnv: the Gymnasium Env implementing reset/step.

Information modes
-----------------
Three modes are supported:

    "perfect_forecast"   — controller receives true future sea-state values
                           drawn directly from the replay buffer.
    "realistic_forecast" — controller receives the canonical GRU+wind
                           deterministic forecast (no stochastic error
                           injection yet — Module 5.1 limitation).
    "reactive"           — future forecast is withheld; controller sees only
                           the sensor-observed history.

Physics / control timestep distinction
---------------------------------------
    DT_PHYSICS  = 0.1 s  — Cummins Euler-Cromer integrator step
    DT_CONTROL  = 1.0 s  — RL action interval (10 physics steps per action)

See environment.py for full documentation.

Public exports
--------------
    WECControlEnv
    EnvironmentConfig
    RewardConfig
    ObservationConfig
    EpisodeReplay
"""

from module5_control.replay import EpisodeReplay, build_episode_replay
from module5_control.observations import ObservationConfig, build_observation_space
from module5_control.rewards import RewardConfig
from module5_control.environment import WECControlEnv, EnvironmentConfig
from module5_control.forecast_uncertainty import ForecastErrorSampler, load_error_bank

__all__ = [
    "WECControlEnv",
    "EnvironmentConfig",
    "RewardConfig",
    "ObservationConfig",
    "EpisodeReplay",
    "build_episode_replay",
    "build_observation_space",
    "ForecastErrorSampler",
    "load_error_bank",
]
