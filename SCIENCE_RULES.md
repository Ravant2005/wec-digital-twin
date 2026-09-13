# Scientific Rules and Invariants

## Evidence labels

Use these labels precisely in code, documentation, plots, and reports:

| Label | Meaning in this project |
| --- | --- |
| ERA5 reanalysis | Hourly bulk wave/wind values from the archived ERA5 source; not an in-situ measurement |
| Synthetic sea state | A JONSWAP/directional/dispersion realisation conditioned on ERA5 statistics; not historical surface elevation |
| BEM/hydrodynamic result | A Capytaine/model-derived quantity for the documented development geometry |
| Sensor observation | Synthetic noisy/delayed/missing proxy from Module 3; not camera-validated field data |
| Forecast | Baseline or neural model prediction; distinguish perfect oracle from realistic GRU+error output |
| RL result | Policy behaviour within this simulator and protocol; not a validated physical deployment result |

## Units, conventions, and time bases

- Use SI units unless a public API or external input explicitly documents an
  exception.  Document each exception at the boundary.
- Module 2 positive heave displacement is **upward**.  Force signs must remain
  consistent with the Cummins equation and PTO convention.
- ERA5 mean-wave direction is degrees clockwise from North, propagation
  **toward**.  Convert degrees to radians once at the documented internal
  boundary; do not add an unexplained π offset.
- Frequencies require explicit Hz versus rad/s labels.  Do not compare or add
  them without conversion.
- Module 5 uses a 1.0 s control step and 0.1 s physics step.  A one-hour ERA5
  state is not a second-by-second measurement.
- The present synthetic Goa default uses `gamma=3.3` and an assumed 30 m water
  depth in some workflows; neither is Goa-calibrated.  Treat them as explicit
  development assumptions, never hidden constants.

## Numerical-method rules

- Preserve documented schemes unless deliberately changing and revalidating
  them.  The Cummins solver uses an explicit Euler-Cromer update and causal,
  trapezoidal/left-endpoint radiation-memory treatment as documented in
  `module2_wec/cummins.py`.
- Check input shape, monotonicity, finite values, units, and sampling interval
  at numerical boundaries.  Numerical stability, convergence, conservation,
  and a passing unit test answer different questions.
- Record seeds for stochastic synthesis, sensor noise, replay sampling, and
  RL experiments.  Do not silently make a deterministic routine stochastic or
  vice versa.

## Data leakage and evaluation rules

- Fit scalers and select/tune models on training data only.  Use 2022–2023 for
  validation and retain 2024–2025 for a frozen final evaluation.
- Reactive observations have no future forecast values.  Perfect forecast is
  explicitly an oracle upper bound.  Realistic forecast must use the configured
  GRU+wind prediction and safe empirical error bank, not current-episode truth.
- Do not use the 2024–2025 error bank while training or validating PPO.
- A benchmark or short PPO smoke run establishes execution health, not energy
  superiority, generalisation, deployment readiness, or a publishable claim.

## Change checklist for scientific code

For each physics, numerical, data, ML, or control change, record:

1. equation or transformation and existing documented source;
2. assumptions, units, coordinate/sign convention, and initial/boundary
   conditions;
3. numerical method, resolution, and stability/convergence concern;
4. affected data split and whether any future-information pathway changes;
5. validation tests and an independent physical/numerical sanity check;
6. limitations that remain after the test passes.

If one of these is unknown, stop and ask rather than inventing an answer.
