# WEC Digital Twin — Agent Operating Rules

## Mission and scope

This repository is a physics-based WEC digital twin for an AI-forecasted,
RL-controlled heaving point absorber off the Goa coast.  Treat it as a
scientific research codebase, not as a generic application.  A passing test
does **not** establish physical validity.

Read [PROJECT_STATE.md](PROJECT_STATE.md), [ARCHITECTURE.md](ARCHITECTURE.md),
[MODULE_MAP.md](MODULE_MAP.md), and [SCIENCE_RULES.md](SCIENCE_RULES.md) before
planning a cross-module or scientific change.

## Required working method

1. Inspect the relevant implementation, tests, call sites, and documentation
   before proposing a change.  Do not infer an interface from its name.
2. State the intended change, assumptions, affected interfaces, and relevant
   validation before editing.
3. Make the smallest coherent change.  Preserve public interfaces and
   backwards-compatible behaviour unless the user explicitly authorises an
   interface change.
4. Run the narrowest relevant tests after each change.  Run cross-module or
   integration tests when an interface, data contract, unit, or time-base
   changes.
5. Report what changed, tests run, results, assumptions, and unresolved risks
   honestly.  Never claim a result is validated solely because code executes.

## Scientific integrity rules

- Do not silently change units, coordinate frames, sign conventions,
  normalisation, time steps, random seeds, train/validation/test splits, or
  default physical parameters.
- Do not invent equations, constants, references, validation results, data,
  observations, or model performance.  Use only sources already documented in
  the repository unless the user asks for research.
- Clearly distinguish: measured data, ERA5 reanalysis, synthetic sea-surface
  realisations, BEM/hydrodynamic simulation output, and ML/RL predictions.
- Preserve the distinction between physics, numerical method, and ML/control
  policy.  Do not represent a learned or synthetic result as a physical
  measurement.
- Do not leak future truth into reactive or realistic forecast observations.
  The 2024–2025 final test period is holdout data; do not use it for fitting,
  tuning, error-bank construction, or model selection.
- Never delete, weaken, skip, or loosen a test merely to make a failure pass.
  Diagnose the cause and correct the implementation or explicitly document a
  justified test correction.

## Extra requirements for physics-related changes

Before implementation, identify the equation(s), assumptions, quantities and
SI units, numerical scheme, time step, boundary/initial conditions, and the
existing source/reference if one is documented.  After implementation, state
the validation method and its limits.  Stop and ask the user when a scientific
assumption is genuinely ambiguous or lacks a documented basis.

## Data, results, and safety

- Raw data in `data/` and experiment artefacts in `results/` are inputs or
  evidence, not disposable cache.  Do not modify, regenerate, or delete them
  unless the user explicitly requests that operation.
- Do not read, print, add, commit, or transmit `.env` files, credentials,
  tokens, private keys, SSH material, or personal files.  Keep credentials in
  the tool's secure local store, never source code.
- Inspect `git status` and `git diff` before meaningful changes.  Preserve
  unrelated user work.  Do not use `git reset --hard`, force push, destructive
  cleanup, or history rewriting.  Do not commit or push without explicit
  approval.
- Prefer the project interpreter: `.venv/bin/python`.  Do not install or
  upgrade project dependencies without approval.  Set a writable
  `MPLCONFIGDIR` when running Matplotlib-dependent commands in restricted
  environments.

## Useful validation commands

```bash
MPLCONFIGDIR=/private/tmp/wec-mpl .venv/bin/python -m pytest --collect-only -q
MPLCONFIGDIR=/private/tmp/wec-mpl .venv/bin/python -m pytest tests/test_module5_ppo_infra.py -q
MPLCONFIGDIR=/private/tmp/wec-mpl .venv/bin/python -m pytest tests/test_pipeline_v2.py -q
MPLCONFIGDIR=/private/tmp/wec-mpl .venv/bin/python -m pytest -q
```

Use a targeted test file first; the full suite is appropriate for integration
changes or before a user-requested milestone handoff.
