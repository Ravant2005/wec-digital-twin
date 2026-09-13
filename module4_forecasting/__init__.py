"""
module4_forecasting — Wave forecasting dataset and baselines (Module 4.1).

Provides the supervised learning dataset pipeline, non-ML baselines,
and evaluation metrics for wave forecasting.

NO neural network is implemented in this module.

Exports
-------
ForecastConfig, ForecastSample, ForecastDataset
TimeSeriesSplit, make_split, build_datasets
FEATURE_NAMES, TARGET_NAMES, N_FEATURES, N_TARGETS
direction_to_sincos, sincos_to_direction

WaveScaler

PersistenceForecaster, ClimatologyForecaster

ForecastMetrics, HorizonMetrics
evaluate_forecast, evaluate_by_horizon, circular_error, skill_score
"""

from module4_forecasting.dataset import (
    ForecastConfig,
    ForecastSample,
    ForecastDataset,
    TimeSeriesSplit,
    make_split,
    build_datasets,
    FEATURE_NAMES,
    TARGET_NAMES,
    N_FEATURES,
    N_TARGETS,
    direction_to_sincos,
    sincos_to_direction,
)
from module4_forecasting.preprocessing import WaveScaler
from module4_forecasting.baselines import PersistenceForecaster, ClimatologyForecaster
from module4_forecasting.metrics import (
    ForecastMetrics,
    HorizonMetrics,
    evaluate_forecast,
    evaluate_by_horizon,
    circular_error,
    skill_score,
)

__all__ = [
    "ForecastConfig", "ForecastSample", "ForecastDataset",
    "TimeSeriesSplit", "make_split", "build_datasets",
    "FEATURE_NAMES", "TARGET_NAMES", "N_FEATURES", "N_TARGETS",
    "direction_to_sincos", "sincos_to_direction",
    "WaveScaler",
    "PersistenceForecaster", "ClimatologyForecaster",
    "ForecastMetrics", "HorizonMetrics",
    "evaluate_forecast", "evaluate_by_horizon", "circular_error", "skill_score",
]
