# Module Map

| Module | Key files | Inputs | Outputs / consumers | Status evidence |
| --- | --- | --- | --- | --- |
| 1 — Ocean | `spectrum.py`, `dispersion.py`, `direction.py`, `waves.py`, `spatial_waves.py`, `goa_seastate.py`, `era5.py` | Wave parameters and ERA5 records | Spectra, wave numbers, synthetic surface/replay inputs for forcing | `outputs/module1_validation/`, `tests/test_module1.py` and ocean tests |
| 2 — WEC physics | `geometry.py`, `hydrostatics.py`, `capytaine_model.py`, `hydrodynamic_coefficients.py`, `radiation.py`, `excitation.py`, `cummins.py`, `pto.py`, `latching.py` | Geometry, BEM data, waves/excitation, PTO settings | Hydrodynamic coefficients, radiation kernel, heave response, PTO quantities | `outputs/module2_hydro/`, physics tests |
| 3 — Sensor | `sensor.py` | True Hs/Tp/direction and noise/delay/missingness configuration | `SensorObservation` and observable model input | `outputs/module3_sensor/`, `tests/test_sensor.py` |
| 4 — Forecasting | `era5_archive.py`, `dataset_v2.py`, `preprocessing.py`, `models.py`, `training.py`, `evaluation.py`, `baselines.py` | Production ERA5 waves/wind archive | Datasets, scalers, forecasts, metrics, error banks | `results/forecasting/`, forecasting tests |
| 5 — Control | `replay.py`, `excitation.py`, `observations.py`, `environment.py`, `controllers.py`, `forecast_uncertainty.py`, `env_factory.py`, `train_ppo.py`, `evaluate_ppo.py` | Module 2 parameters; Module 3/4 information; ERA5 replay | Gymnasium environment, controller and PPO artefacts | `results/control_benchmarks/`, `results/rl/`, Module 5 tests |

## Interface checkpoints

1. **ERA5 archive → forecasting/replay.** `load_era5_archive()` aligns hourly
   wave and wind streams.  Chronological splits are 2010–2021/2022–2023/
   2024–2025; window construction must not cross a split.
2. **Module 1 → Module 2.** Spectral and directional conventions determine the
   forcing used by hydrodynamics.  Check frequency units (Hz vs rad/s), water
   depth, and wave-direction convention at every boundary.
3. **Module 2 → Module 5.** `CumminsParameters`, radiation kernels, and
   excitation are used in repeated 0.1 s physics steps.  Preserve the upward
   positive-heave sign and PTO force/power signs.
4. **Module 3/4 → Module 5 observations.** Sensor history is causal; forecast
   content differs by mode.  Only the oracle mode may contain future truth.
5. **Module 5 → evaluation.** PPO training normalises rewards through
   VecNormalize; evaluation loads it with reward normalisation disabled so
   reported metrics retain raw meaning.

## Tests by change area

- Wave spectrum/dispersion/direction: `test_spectrum.py`, `test_dispersion.py`,
  `test_direction.py`, `test_waves.py`, `test_spatial_waves.py`.
- Hydrodynamics/Cummins/PTO/latching: `test_capytaine_model.py`,
  `test_hydrodynamic_coefficients.py`, `test_radiation.py`, `test_cummins.py`,
  `test_pto.py`, `test_latching.py`, `test_excitation.py`.
- Sensing and forecasting: `test_sensor.py`, `test_pipeline_v2.py`,
  `test_forecasting_*.py`, `test_persistence_comparison.py`.
- RL/control interfaces: `test_module5_*.py`, `test_module5_3b_smoke.py`,
  `test_module5_environment.py`, `test_module5_ppo_infra.py`.

When an edit crosses one of the checkpoints above, test both sides of the
boundary rather than only the file containing the edit.
