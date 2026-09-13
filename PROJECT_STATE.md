# Project State

## Objective

Develop and validate a physics-based digital twin of a heaving point-absorber
WEC off the Goa coast, using ERA5-conditioned waves, a synthetic sensor model,
wave forecasting, and RL latching control.  The intended Tier 1 comparison is
reactive PPO versus a perfect-forecast oracle and a realistic-forecast PPO.

## Current stage

- Modules 1–4 are implemented with validation outputs and tests.
- Module 5.3A (RL experiment design) and 5.3B (PPO infrastructure, safe error
  bank, VecNormalize integration, and smoke tests) are documented as complete.
- The next planned research work is Module 5.3C: train the three PPO policies,
  validate on 2022–2023, and reserve 2024–2025 for final frozen-policy
  evaluation.
- `results/rl/` contains smoke and short trial runs (32–1,024 timesteps), not
  research-grade trained-policy evidence.

## Completed module chain

| Module | Responsibility | Main package |
| --- | --- | --- |
| 1 | Ocean waves, spectra, dispersion, direction, ERA5-conditioned synthesis | `module1_ocean/` |
| 2 | Cylinder hydrostatics, Capytaine/BEM coefficients, radiation, Cummins/PTO/latching dynamics | `module2_wec/` |
| 3 | Synthetic wave-sensor uncertainty, delay, missingness, direction errors | `module3_sensor/` |
| 4 | ERA5 archive, chronological datasets, baselines, LSTM/GRU forecasting and evaluation | `module4_forecasting/` |
| 5 | Replay, observations, Gymnasium WEC environment, controllers, uncertainty and PPO tooling | `module5_control/` |

See [MODULE_MAP.md](MODULE_MAP.md) for interfaces and boundaries.

## Data and experiment contracts

- Production ERA5 wave and wind archive: `data/raw/era5/production/`, expected
  coverage 2010–2025, loaded by `module4_forecasting.era5_archive`.
- The production manifest records the selected grid point at 15.5° N, 73.5° E.
- Forecasting and RL splits are chronological: training 2010–2021, validation
  2022–2023, final holdout 2024–2025.
- `results/forecasting/exp_gru_wind/best_model.pt` is the configured realistic
  forecast checkpoint.  The only permitted PPO training/validation error bank
  is `forecast_errors_val2022_2023.csv.gz`; the general `forecast_errors.csv.gz`
  is final-test material.
- `data/raw/era5_goa_jan2024.nc` is a smaller raw ERA5 input used by Module 1
  workflows.  Synthetic sea surfaces are statistically conditioned on ERA5;
  they are not historical surface-elevation measurements.

## Important interfaces

- Module 1 produces sea-state/spectrum quantities and synthetic wave fields;
  directions enter as ERA5 degrees (clockwise from North, propagation toward)
  and are converted to radians at the documented API boundary.
- Module 2 supplies `CumminsParameters` and excitation/hydrodynamic quantities
  to Module 5.  Positive heave is upward.
- Module 3 `WaveSensor` supplies observable sensor fields only; truth fields
  must not enter ML or reactive-control observations.
- Module 4 uses 48-hour lookback and 48-hour forecast horizons.  The GRU+wind
  output feeds Module 5 realistic forecasts.
- Module 5 uses a 723-value observation: 48×11 sensor/wind history, 48×4
  forecast features, and three WEC-state values.  Control is 1 Hz; physics is
  integrated at 0.1 s.

## Validation status

- 1,174 pytest tests collected successfully on 2026-09-13 with the project
  Python 3.11 environment.
- The repository contains module validation reports and visual outputs under
  `outputs/`, plus forecasts, benchmarks, and smoke-run artefacts under
  `results/`.
- Existing smoke summaries show finite PPO observations and rewards for the
  three information modes; they do not validate PPO performance or energy
  improvement.

## Known limitations and unresolved work

The authoritative tracker is [docs/research_limitations.md](docs/research_limitations.md).
Important open issues include development cylinder geometry, finite BEM/radiation
coverage, assumed 30 m water depth, uncalibrated JONSWAP gamma, synthetic sensor
noise, forecast errors not conditioned on sea state, PPO training still pending,
and fairness of the threshold versus unrestricted RL latch timing.

## Environment and test commands

- Python environment: `.venv/` (Python 3.11.15; pytest, NumPy, SciPy, PyTorch,
  Gymnasium, Stable-Baselines3, xarray, pandas, and Matplotlib are installed).
- No dependency lockfile, `pyproject.toml`, or requirements file was present at
  setup time; do not infer a reproducible dependency specification from this
  note.  Create one only with user approval.
- Run targeted tests before the full suite:

```bash
MPLCONFIGDIR=/private/tmp/wec-mpl .venv/bin/python -m pytest tests/test_module5_ppo_infra.py -q
MPLCONFIGDIR=/private/tmp/wec-mpl .venv/bin/python -m pytest tests/test_pipeline_v2.py -q
MPLCONFIGDIR=/private/tmp/wec-mpl .venv/bin/python -m pytest -q
```
