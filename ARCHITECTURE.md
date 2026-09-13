# Architecture

## System flow

```text
ERA5 wave + wind archive (hourly, Goa)
        |
        +--> Module 1: spectra, finite-depth dispersion, directional/synthetic waves
        |                 |
        |                 +--> Module 2: BEM-derived hydrodynamics + Cummins/PTO dynamics
        |
        +--> Module 3: noisy, delayed, possibly missing sensor observations
        |
        +--> Module 4: chronological datasets -> baselines / LSTM / GRU forecasts
                                      |
                                      v
Module 5: EpisodeReplay + observations + forecast uncertainty + Gymnasium WECControlEnv
                                      |
                                      v
                         controllers / PPO training and evaluation
```

## Layer responsibilities

### Ocean and forcing — Module 1

`module1_ocean` implements spectral wave modelling (Pierson–Moskowitz and
JONSWAP), directional spreading, finite-depth dispersion, spatial synthesis,
and ERA5-conditioned sea-state replay.  ERA5 values are hourly bulk
statistics; generated surface elevations are synthetic realisations.

### WEC physics — Module 2

`module2_wec` models a vertical cylinder in heave.  It provides hydrostatics,
Capytaine BEM coefficients, radiation kernels, excitation force, the
single-DOF Cummins solver, linear PTO, and latching.  The documented Cummins
equation uses effective inertia `(M + A_inf)`, hydrostatic stiffness, and a
causal radiation-memory convolution.  `A_inf` must not be double counted.

### Sensing — Module 3

`module3_sensor` is a synthetic camera/sensor error model, not a computer
vision implementation.  It adds documented noise, bias, delay, and missingness
while preserving a strict distinction between observed and true state.

### Forecasting — Module 4

`module4_forecasting` loads the 2010–2025 archive, builds time-respecting
windows, applies training-only scaling, and evaluates persistence, climatology,
LSTM, and GRU forecasters.  Direction is represented through sine/cosine where
needed to avoid circular discontinuity.

### Control — Module 5

`module5_control` joins replayed sea states, Module 2 forcing/dynamics, Module
3 sensor history, Module 4 forecasts, and an RL-compatible environment.
`WECControlEnv` accepts `RELEASE`/`LATCH` actions at 1 s intervals and advances
the physical integrator in ten 0.1 s substeps.  Forecast modes are `reactive`,
`perfect_forecast` (oracle only), and `realistic_forecast` (GRU plus empirical
error trajectories).

## Design boundaries that must remain explicit

- Reanalysis inputs are not direct field measurements.
- The sensor model produces synthetic observations; it is not a validated
  camera pipeline.
- BEM outputs and Cummins dynamics are a model with documented approximations,
  not a deployment-certified device model.
- The perfect-forecast mode is an upper bound, never a deployable controller.
- PPO is a policy-learning layer over the physics model; it does not validate
  its physical assumptions.

## Repository layout

| Path | Purpose |
| --- | --- |
| `module1_ocean/` … `module5_control/` | Source packages ordered by system flow |
| `tests/` | Pytest unit, regression, interface, and smoke tests |
| `data/raw/` | ERA5 NetCDF/JSON inputs and archive manifest |
| `scripts/` | ERA5 acquisition, auditing, and validation utilities |
| `results/` | Forecasting, benchmark, and PPO experiment artefacts |
| `outputs/` | Generated visualisations and validation reports kept as evidence |
| `docs/` | RL design and limitations tracker |

No build/packaging manifest was present when this document was created; use the
existing `.venv` rather than assuming `pip install -e .` is supported.
