# WEC PROJECT STATE

## 1. PURPOSE

This file is the current operational state of the WEC research software project.

Unlike `ARCHITECTURE.md` and `MODULE_MAP.md`, this document is expected to change frequently.

The purpose is to tell both the researcher and AI coding agents:

* where the project currently stands
* what has already been completed
* what is currently being developed
* what remains
* what is known to work
* what is not yet validated
* what must not be changed casually
* what should be done next

This document must remain factual.

Do not mark work as complete merely because files exist.

---

# 2. PROJECT IDENTITY

## Project

AI-Forecasted, Confidence-Gated, RL-Corrected Control of a Point-Absorber Wave Energy Converter

## Geographic focus

Goa coastal waters, Arabian Sea.

The research plan uses approximately:

* latitude: 15.3–15.6° N
* longitude: 73.6–73.9° E

Exact simulation coordinates must come from the active site configuration rather than being duplicated throughout code.

## Core project type

Physics-based, simulation-first WEC digital twin.

## Core objective

Build a reproducible closed-loop system:

```
real / reanalysis wave data
        ↓
  ocean environment
        ↓
  WEC dynamics
        ↑
   controller
        ↑
   forecasting
        ↑
  sensing proxy
        ↓
    evaluation
```

The core research goal is to determine whether forecast information can improve control of a point-absorber WEC, and ultimately whether calibrated forecast confidence can be used to control how aggressively the WEC trusts predictive information.

---

# 3. RESEARCH ROADMAP

## Tier 1 — GUARANTEED CORE

Tier 1 is the mandatory foundation.

It consists of:

1. Ocean / Wave Environment
2. Buoy / WEC Dynamics
3. Synthetic Camera Sensing
4. Wave Forecasting
5. Forecast-informed RL / control
6. Evaluation and robustness harness

The intended Tier-1 result is a complete, reproducible closed-loop simulation.

Tier 1 must remain independently runnable even if all advanced tiers are abandoned.

---

## Tier 2 — FLAGSHIP EXTENSION

Confidence-Gated Predictive Control.

Key ingredients:

* probabilistic forecast
* calibrated uncertainty
* predictive/MPC layer
* confidence gate
* reactive fallback

Conceptual idea:

```
high forecast confidence
        ↓
more predictive control

low forecast confidence
        ↓
more reactive control
```

The central scientific question is not merely whether forecasting helps, but whether the controller should dynamically determine how much it trusts its own forecast.

---

## Tier 3

Residual RL correction on top of physics-based predictive control.

---

## Tier 4

Robustness hardening:

* measured forecast-error distributions
* sensor dropout
* distribution shift
* bounded forecast perturbation
* lower-tail / risk-aware objectives

---

## Tier 5

Advanced forecasting/generalization:

* spatial upstream context
* advection-aware features
* probabilistic forecasting
* physics-informed consistency
* compact Transformer/attention architecture
* location conditioning
* second-site validation

Tier 5 is explicitly deferred until the lower tiers are solid.

---

# 4. CURRENT IMPLEMENTATION STATUS

## Important rule

The research plan describes what should exist.

Repository inspection determines what actually exists.

Therefore this section must be updated whenever major implementation work is completed.

Do not infer implementation completion solely from the roadmap.

---

# 5. CURRENTLY VERIFIED DEVELOPMENT STATE

## AI Coding Infrastructure

### Status

READY / OPERATIONAL

The development environment currently uses:

* Ollama as local inference runtime
* local Qwen 3.5 9B Q4_K_M-based coding model
* `wec-coder:9b-16k` local alias
* OpenCode as the coding-agent frontend
* OpenRouter as an optional cloud reasoning fallback

The intended operating model is:

```
local model
    ↓
normal engineering work

OpenRouter
    ↓
difficult reasoning / review / research / complex debugging
```

The local model is the default implementation engine.

---

# 6. PROJECT CONTEXT FILES

The repository should contain:

* `AGENTS.md`
* `ARCHITECTURE.md`
* `MODULE_MAP.md`
* `PROJECT_STATE.md`
* `SCIENCE_RULES.md`

Additional AI workflow/routing documentation may be added.

These files collectively define:

### AGENTS.md

How the AI agent must behave.

### ARCHITECTURE.md

How the whole research system is structured.

### MODULE_MAP.md

What each module is responsible for.

### PROJECT_STATE.md

Where the implementation currently stands.

### SCIENCE_RULES.md

Scientific/data/validation constraints.

---

# 7. MODULE STATUS BOARD

Status meanings:

* NOT_STARTED
* IN_PROGRESS
* SMOKE_TESTED
* VALIDATED
* FROZEN
* EXPERIMENTAL

The status below must be updated from repository evidence.

---

## Module 1 — Ocean / Wave Environment

### Research-plan status

MANDATORY TIER-1 MODULE

### Intended responsibility

Produce:

* sea states
* surface elevation
* wave excitation
* replayable real/reanalysis conditions

using real Goa-region data and controlled spectral generation.

### Planned technologies

* NumPy
* SciPy
* wavespectra where useful
* ERA5/CDS
* INCOIS/WAMAN/OMNI where available
* JONSWAP
* Pierson–Moskowitz

### Current implementation status

VERIFY FROM REPOSITORY.

### Required evidence before marking VALIDATED

* real/reanalysis data ingestion
* synthetic spectral generation
* Hs/Tp/direction validation
* reproducible replay
* data provenance
* Module-2 integration

---

## Module 2 — Buoy / WEC Dynamics

### Research-plan status

MANDATORY TIER-1 MODULE

### Intended responsibility

Implement the physics-based heaving point absorber.

### Planned technologies

* Capytaine
* SciPy
* Cummins-equation time-domain dynamics

### Current implementation status

VERIFY FROM REPOSITORY.

### Required evidence before marking VALIDATED

* hydrodynamic coefficients connected
* dynamics propagate correctly
* passive baseline
* classical threshold-latching baseline
* power calculation
* numerical stability
* representative validation

---

## Module 3 — Synthetic Camera / Sensing

### Research-plan status

MANDATORY TIER-1 MODULE

### Intended responsibility

Convert true simulated ocean state into noisy/delayed observations.

### Planned components

* noise model
* delay model
* update rate
* observation buffer
* optional dropout

### Current implementation status

VERIFY FROM REPOSITORY.

### Required evidence before marking VALIDATED

* noisy observations
* delay behaviour
* deterministic seeded behaviour
* no hidden truth leakage
* Module-4 integration

---

## Module 4 — Wave Forecasting

### Research-plan status

MANDATORY TIER-1 MODULE

### Tier-1 baseline

LSTM/GRU sequence model.

### Intended configuration

Historical observation context → future wave-state forecast.

The research plan describes approximately:

* two-day historical context
* two-day forecast horizon
* 1–3 hour output resolution

Final values must be taken from active experiment configuration.

### Required outputs

* future Hs
* future Tp
* future direction
* lead-time metrics
* forecast-error distribution

### Current implementation status

VERIFY FROM REPOSITORY.

### Required evidence before marking VALIDATED

* chronological data split
* train/validation/test pipeline
* no target leakage
* reproducible preprocessing
* model save/load
* forecast metrics
* error-distribution export
* integration with Module 5

---

# 8. MODULE 5 — CONTROLLER / RL

## Research-plan status

MANDATORY TIER-1 MODULE

## Tier-1 goal

Forecast-informed RL latching control.

## Candidate framework

* Gymnasium
* Stable-Baselines3

## Candidate algorithms

* PPO
* SAC
* DQN where appropriate

Algorithm choice must follow the actual action-space/training evidence rather than the appearance of sophistication.

---

# 9. CURRENT RL INFRASTRUCTURE STATE

## Known development direction

The PPO training infrastructure has been implemented and smoke-tested for the project's three intended information modes:

1. Reactive
2. Perfect forecast
3. Realistic forecast

The implementation is intended to use the real Capytaine-based WEC physics model.

No simplified physics replacement should be introduced merely to make RL training easier.

## Important constraint

Large-scale PPO training must not be assumed to have been completed merely because the infrastructure smoke test passes.

The following must be tracked separately:

* infrastructure works
* smoke test works
* short training works
* convergence demonstrated
* final training complete
* benchmark validated

---

# 10. RL EXPERIMENT MODES

The controller/evaluation framework should preserve at least:

## Mode 1 — Reactive

No future forecast.

Purpose:

establish a reactive baseline.

## Mode 2 — Perfect Forecast

Future information available without forecast error.

Purpose:

estimate the value of future information.

## Mode 3 — Realistic Forecast

Trained forecast including realistic forecast error.

Purpose:

represent the deployable predictive condition.

## Mode 4 — Forecast Withheld

Optional explicit fallback/robustness condition.

---

# 11. CURRENT RL VALIDATION RULE

Never interpret:

```
smoke-test passed
```

as:

```
controller scientifically validated
```

The progression is:

```
infrastructure
    ↓
smoke test
    ↓
short run
    ↓
training stability
    ↓
held-out evaluation
    ↓
baseline comparison
    ↓
robustness analysis
    ↓
scientific conclusion
```

---

# 12. MODULE 6 — EVALUATION HARNESS

## Research-plan status

MANDATORY TIER-1 MODULE

## Intended responsibility

Provide reproducible benchmarking and ablation.

Must ultimately compare:

* passive
* classical threshold latching
* reactive RL
* forecast-informed control
* perfect forecast
* realistic forecast

and later:

* confidence-gated control
* residual RL
* robustness variants

### Current implementation status

VERIFY FROM REPOSITORY.

---

# 13. CURRENT SCIENTIFIC PRIORITY

The current priority is NOT:

* Transformer first
* largest RL algorithm
* most sophisticated forecasting architecture
* full CFD
* photorealistic camera simulation

The priority is:

```
working physics
      ↓
real wave data
      ↓
forecast
      ↓
controller
      ↓
integrated benchmark
      ↓
robustness
      ↓
advanced contribution
```

---

# 14. IMMEDIATE DEVELOPMENT PRIORITIES

When asked "what should I build next?", use this ordering unless the user explicitly changes it.

## Priority 1

Protect the existing working implementation.

## Priority 2

Complete/validate any unfinished Tier-1 module.

## Priority 3

Integrate the full closed loop.

## Priority 4

Run baseline comparisons.

## Priority 5

Validate realistic forecast-error robustness.

## Priority 6

Only after Tier 1 is stable, implement Tier 2 confidence gating.

---

# 15. CURRENT NEXT-TASK RULE

The AI agent must NOT automatically choose the most technically interesting task.

It must choose the highest-value incomplete task that:

1. is consistent with the research roadmap
2. depends on already validated work
3. does not unnecessarily destabilize completed modules
4. moves the project toward an end-to-end Tier-1 result
5. has a clear validation criterion

If two tasks are equal in importance, prefer the one with:

* smaller scope
* lower risk
* faster validation
* stronger dependency leverage

---

# 16. KNOWN SCIENTIFIC BOUNDARIES

The current project is simulation-first.

It does NOT yet prove:

* hardware performance
* real camera hardware performance
* full CFD fidelity
* universal geographical generalization
* patentability
* commercial readiness

The research plan explicitly identifies these as limitations/future directions.

---

# 17. CURRENT DATA PRINCIPLES

### ERA5

Used as the main historical/reanalysis training backbone where applicable.

### INCOIS / WAMAN / OMNI

Used as in-situ reference/validation where available.

### INCOIS OSF

May be used as an auxiliary benchmark/sanity-check source.

### Synthetic spectra

Used for:

* physics validation
* controlled experiments
* stress testing

They must not become the sole evidence supporting a real-data claim.

---

# 18. CURRENT MODELING PRINCIPLE

The system must preserve the distinction:

```
PHYSICS MODEL
    ≠
FORECAST MODEL
    ≠
CONTROLLER
    ≠
EVALUATION MODEL
    ≠
AI CODING MODEL
```

Specifically:

### Physics model

Simulates the WEC.

### Forecast model

Predicts the future ocean state.

### Control model

Chooses actions.

### Evaluation

Determines whether those actions actually improve performance.

### Coding model

Helps build all of the above.

The coding model must never be treated as scientific evidence.

---

# 19. ADVANCED ROADMAP

Only after Tier 1 is robust:

## Tier 2

Probabilistic forecast + conformal calibration + confidence-gated predictive control.

## Tier 3

Residual RL on top of predictive control.

## Tier 4

Measured-error robustness + risk-aware objectives + expanded stress matrix.

## Tier 5

Spatial/advection features + compact Transformer/attention + physics-informed consistency + location conditioning + second-site validation.

---

# 20. FALLBACK POLICY

If an advanced feature fails:

### Tier 5 failure

Return to Tier 4.

### Tier 4 failure

Return to Tier 3.

### Tier 3 failure

Return to Tier 2.

### Tier 2 failure

Return to Tier 1.

### Advanced forecaster fails

Keep LSTM/GRU baseline.

### Complex RL becomes unstable

Use simpler action space / simpler algorithm.

### Probabilistic calibration fails

Keep deterministic forecast.

The project must always retain its last validated state.

---

# 21. EXPERIMENT STATUS TEMPLATE

Every significant experiment should be tracked using:

## Experiment ID

Unique identifier.

## Date

Date run.

## Git commit

Code version.

## Data version

Exact dataset/data configuration.

## Configuration

Experiment parameters.

## Model

Architecture and version.

## Training

Epochs/timesteps/seeds/etc.

## Evaluation mode

Reactive / perfect / realistic / withheld / other.

## Metrics

Recorded results.

## Artifacts

Model/checkpoint/plots/logs.

## Status

PLANNED / RUNNING / COMPLETE / FAILED / INVALID

## Scientific interpretation

What the experiment does and does not establish.

---

# 22. BUG / ISSUE STATUS

Issues should be classified as:

### BLOCKING

Prevents progress or invalidates scientific results.

### HIGH

Major implementation or validation problem.

### MEDIUM

Important but does not block current milestone.

### LOW

Quality/cleanup/documentation issue.

When reporting a problem, include:

* symptom
* likely cause
* evidence
* affected modules
* proposed fix
* validation plan

---

# 23. CURRENT KNOWN LIMITATIONS

This section should always reflect the latest actual state.

Examples of items that require periodic updating:

* data availability
* hydrodynamic validation depth
* forecast dataset length
* camera-model validation
* RL convergence
* integration completeness
* compute limitations
* missing real-world hardware validation

Do not leave old limitations here after they have been experimentally resolved.

---

# 24. CURRENT PROJECT HEALTH

Use this compact health summary:

| Area                        | Status                     | Evidence                       | Risk   |
| --------------------------- | -------------------------- | ------------------------------ | ------ |
| AI coding environment       | Operational                | OpenCode + Ollama + OpenRouter | Low    |
| Project context             | Operational                | AI context documents           | Low    |
| Module 1                    | Verify                     | Repository evidence required   | Medium |
| Module 2                    | Verify                     | Repository evidence required   | High   |
| Module 3                    | Verify                     | Repository evidence required   | Medium |
| Module 4                    | Verify                     | Repository evidence required   | Medium |
| RL/PPO infrastructure       | Smoke-tested               | Training infrastructure test   | Medium |
| Tier-1 integration          | Verify                     | Full pipeline test required    | High   |
| Benchmark harness           | Verify                     | End-to-end evidence required   | High   |
| Tier 2 confidence gating    | Not started / experimental | Must follow Tier 1             | High   |
| Tier 3 residual RL          | Not started                | Depends on Tier 2              | High   |
| Tier 4 robustness hardening | Not started / partial      | Depends on measured errors     | Medium |
| Tier 5 generalization       | Deferred                   | Requires Tier 1–4              | High   |

IMPORTANT:

"Verify" does not mean "failed".

It means this document should be updated using actual repository/test evidence before an AI agent treats the module as validated.

---

# 25. CURRENT DEFINITION OF "DONE"

The project is NOT done when:

* code compiles
* one simulation runs
* a model trains once
* a loss curve decreases
* an RL reward increases

Tier-1 is done when the complete research loop is reproducibly demonstrated:

```
real/reanalysis wave data
        ↓
    ocean model
        ↓
    WEC physics
        ↑
    controller
        ↑
    forecast
        ↑
    sensing
        ↓
    evaluation
```

and the system can support meaningful comparisons between:

* passive
* classical control
* reactive RL
* forecast-informed control

under explicit information conditions.

---

# 26. STATE UPDATE POLICY

This file must be updated whenever:

* a module becomes validated
* a module changes status
* a major experiment completes
* a dataset is added/removed
* a model changes
* a major bug is discovered/fixed
* an interface changes
* a new tier begins
* a tier is abandoned/falls back
* a major research conclusion is supported or rejected

Do not update this file with optimistic language.

Use evidence.

---

# 27. AI AGENT RULE FOR THIS FILE

Before beginning a substantial task:

1. read `PROJECT_STATE.md`
2. inspect the repository
3. determine whether this file is stale
4. avoid trusting status labels without supporting evidence

After a substantial task:

1. update relevant status
2. record what was actually validated
3. record unresolved issues
4. record next logical task
5. preserve historical context where useful

The agent must never rewrite the entire project history merely to make the state look clean.

---

# 28. CURRENT NEXT-STEP TEMPLATE

At the bottom of this document, maintain the current immediate action.

Use exactly this structure:

## NEXT ACTION

### Objective

<one sentence>

### Why now

<why this is higher priority than other tasks>

### Files/modules

<affected areas>

### Acceptance criteria

<observable pass conditions>

### Validation command(s)

<exact commands>

### Risks

<known risks>

### Fallback

<what happens if this fails>

This section should always describe the CURRENT immediate engineering step, not the entire roadmap.

---

# 29. GOLDEN RULE

The purpose of this file is not to make the project look advanced.

Its purpose is to prevent the AI agent from:

* rebuilding completed work
* skipping dependencies
* confusing planned work with completed work
* prematurely jumping to advanced tiers
* forgetting known failures
* claiming validation without evidence

The project state must reflect reality, not aspiration.
