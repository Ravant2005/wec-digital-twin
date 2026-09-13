# WEC MODULE MAP

## 1. PURPOSE

This document maps the WEC research system into independently testable software modules.

It answers:

* What does each module do?
* What does it consume?
* What does it produce?
* Where should its implementation live?
* What does it depend on?
* What must be tested?
* What counts as completion?
* What must NOT be changed casually?

The six Tier-1 modules are:

1. Ocean / Wave Environment
2. Buoy / WEC Dynamics
3. Synthetic Camera Sensing
4. Wave Forecasting
5. Controller
6. Evaluation Harness

The project must preserve these boundaries even when advanced Tier-2–5 research features are added.

---

# 2. MODULE STATUS VOCABULARY

Every module should have one of these states:

### NOT_STARTED

Required implementation has not begun.

### IN_PROGRESS

Implementation exists but the module is not yet considered reliable.

### SMOKE_TESTED

Basic execution works, but scientific validation is incomplete.

### VALIDATED

Required unit/integration/physics/data validation has passed.

### FROZEN

The module is considered stable for the current project tier.

Changes to a frozen module require explicit justification and regression tests.

### EXPERIMENTAL

An optional research extension is being evaluated and must not silently replace the validated baseline.

---

# 3. MODULE 1 — OCEAN / WAVE ENVIRONMENT

## Purpose

Represent the wave environment experienced by the WEC at the selected site.

The module converts sea-state information into reproducible wave/elevation/excitation inputs for the WEC dynamics model.

---

## Primary scientific responsibilities

* represent real/reanalysis sea states
* support spectral descriptions
* generate irregular waves
* support directional information
* support replay of real sea-state segments
* produce environmental input for the WEC model

---

## Inputs

Potential inputs include:

### Sea-state parameters

* significant wave height Hs
* peak period Tp
* mean/dominant direction
* spectral peakedness γ where appropriate
* duration
* timestep

### Site parameters

* latitude
* longitude
* local water depth
* frequency range
* directional conventions

### Data

* ERA5
* INCOIS / WAMAN / OMNI where available
* synthetic spectral realizations

---

## Expected processing

Potential processing components:

* spectrum construction
* JONSWAP
* Pierson–Moskowitz
* directional spreading
* frequency discretization
* phase generation
* irregular-wave synthesis
* dispersion relation
* replay of archived sea states

---

## Outputs

At minimum:

* η(t): surface elevation time series
* time vector
* wave-component metadata
* wave excitation information required by Module 2
* source/provenance metadata

---

## Required guarantees

The module must:

* produce deterministic output when a seed is specified
* preserve units
* record source information
* distinguish real/reanalysis/synthetic data
* allow reproducible replay
* validate generated sea states against requested sea-state statistics

---

## Important validation

At minimum test:

### Hs reconstruction

Generate a sea state using requested Hs and verify that recovered significant wave height is within an appropriate tolerance.

### Tp consistency

Verify that the generated spectrum/time series has the intended peak period within the expected numerical tolerance.

### Dispersion relation

Verify that wave number/frequency relationships are physically consistent with the configured depth and wave theory.

### Direction

Verify directional conventions and wrapping.

### Reproducibility

Same configuration + same seed → same generated result.

---

## Definition of Done

Module 1 is complete for Tier 1 when:

* real Goa-region wave data can enter the system
* synthetic spectral generation works
* sea-state replay works
* Hs/Tp/direction are validated
* outputs are accepted by Module 2
* provenance is recorded
* deterministic tests pass
* no hidden data leakage is introduced

---

## Do not

* silently substitute synthetic data for real-data evaluation
* invent sea-state parameters
* mix units
* change coordinate/direction conventions without documentation
* hard-code Goa-specific values inside reusable algorithms

---

# 4. MODULE 2 — BUOY / WEC DYNAMICS

## Purpose

Represent the physical WEC plant.

The Tier-1 plant is a heaving point absorber.

The model should implement the established hydrodynamic/time-domain formulation rather than replacing the physics with a black-box neural model.

---

## Primary scientific responsibilities

* represent buoy mass/inertia
* represent hydrostatic restoring force
* incorporate hydrodynamic coefficients
* propagate heave motion
* apply PTO/control force
* compute extracted power
* expose physically meaningful state

---

## Inputs

### From Module 1

* wave excitation
* time vector
* environmental metadata

### Physical parameters

* buoy mass
* geometry
* water properties
* hydrostatic stiffness
* hydrodynamic coefficients
* added-mass representation
* radiation-memory representation

### Control

* PTO force
* latch/release action
* damping command when applicable

---

## Expected processing

Potential components:

* Cummins-equation dynamics
* hydrodynamic force evaluation
* radiation-memory convolution or equivalent
* hydrostatic restoring force
* PTO force
* numerical time integration

Preferred tools:

* Capytaine for BEM/hydrodynamic coefficients
* SciPy for time-domain integration

---

## Outputs

At minimum:

* displacement z(t)
* velocity ż(t)
* acceleration when required
* PTO force
* instantaneous power
* cumulative energy
* constraint state
* diagnostic information

---

## Required baselines

The module must support evaluation under:

### Passive operation

No active latching control.

### Classical threshold latching

Reference control policy against which learned methods can be compared.

---

## Validation

### Physical sanity tests

Test:

* zero excitation
* zero control
* restoring behaviour
* sign conventions
* boundedness

### Timestep/convergence

Verify that numerical results are not artifacts of an inappropriate timestep.

### Energy/power consistency

Verify correct sign and units for:

```
P(t) = F_PTO(t) × v(t)
```

subject to the established project sign convention.

### Hydrodynamic coefficient sanity

Check that imported BEM outputs are physically plausible.

### Reference validation

Where available, compare against validated reference geometry/data such as the selected published/reference buoy configuration.

---

## Definition of Done

Module 2 is complete for Tier 1 when:

* the point absorber integrates successfully
* hydrodynamic coefficients are connected
* passive operation works
* threshold latching works
* PTO power is calculated correctly
* representative sea states produce plausible motion
* tests pass
* numerical stability is demonstrated
* Module 1 and Module 5 interfaces are stable

---

## Do not

* invent hydrodynamic coefficients
* silently change physical sign conventions
* replace the physics model with ML
* optimize controller performance inside the plant model
* mix plant dynamics with evaluation logic

---

# 5. MODULE 3 — SYNTHETIC CAMERA / SENSING LAYER

## Purpose

Approximate the observations that an onboard wave-sensing/camera system would provide.

The Tier-1 implementation is a statistical sensing proxy, not a claim of a fully deployed computer-vision system.

---

## Inputs

* true wave state from Module 1
* sensing-noise parameters
* delay/latency
* sampling/update interval
* optional dropout configuration

---

## Outputs

Examples:

* observed Hs
* observed Tp
* observed direction
* timestamps
* observation availability
* observation latency metadata

---

## Processing

Conceptually:

```
TRUE STATE
   |
   v
SENSOR MODEL
   |
   +--> noise
   +--> delay
   +--> sampling
   +--> dropout
   |
   v
OBSERVED STATE
```

---

## Required behaviour

The module must prevent the forecasting/control pipeline from accidentally receiving hidden perfect information.

The standard pipeline should be:

```
true state
   ↓
sensing layer
   ↓
observed state
   ↓
forecasting/control
```

The true state may still be retained internally for ground-truth evaluation.

---

## Validation

Test:

* zero-noise condition
* expected statistical noise
* delay correctness
* timestamp correctness
* missing-data behaviour
* deterministic output under seed
* correct unit preservation

---

## Definition of Done

Module 3 is complete when:

* observed wave variables are generated
* noise and delay are configurable
* observations can be streamed to Module 4
* ground truth remains available only for evaluation
* dropout conditions can be simulated
* tests pass

---

## Do not

* pass true future values directly to the forecaster
* silently remove observation noise for headline experiments
* call the proxy a real camera system
* hard-code unexplained sensor accuracy

---

# 6. MODULE 4 — WAVE FORECASTING

## Purpose

Predict future wave state from observations available to the WEC.

Tier 1 uses an LSTM/GRU baseline.

Advanced forecasting models belong to later tiers and must remain compatible with the Tier-1 interface.

---

## Inputs

Primary:

* rolling observed Hs
* rolling observed Tp
* rolling observed direction

Optional:

* wind variables
* additional environmental variables

Configured:

* historical context length
* prediction horizon
* sampling frequency

---

## Output interface

The forecaster should expose a standard prediction interface.

Conceptually:

```
observations
    ↓
forecast(observation_history)
    ↓
future wave state
```

The exact implementation may change without changing the logical contract.

---

## Tier-1 baseline

Expected model family:

* LSTM and/or GRU

The core research plan describes approximately:

* 2-day historical context
* 2-day forecast
* 1–3 hour output resolution

Final values must come from the experiment configuration rather than being hard-coded.

---

## Data splitting

Mandatory:

* chronological train split
* chronological validation split
* chronological test split

Never perform random future/past mixing for headline forecasting evaluation.

---

## Required outputs

The module should expose:

* point forecast
* lead-time information
* preprocessing metadata
* model metadata
* uncertainty/error information when available

Tier 1 primarily requires the forecast-error distribution for robustness experiments.

Tier 2 adds calibrated probabilistic uncertainty.

---

## Evaluation

At minimum:

* RMSE
* MAE
* error by forecast horizon

Where Tier 2 is implemented:

* prediction interval coverage
* probabilistic calibration
* CRPS or appropriate equivalent

---

## Validation

Test:

* input shape
* output shape
* temporal alignment
* no future leakage
* deterministic inference where configured
* preprocessing reproducibility
* model save/load
* inverse scaling

---

## Definition of Done — Tier 1

Module 4 is complete for Tier 1 when:

* real/reanalysis historical data enters correctly
* preprocessing is reproducible
* the LSTM/GRU trains
* held-out future testing works
* forecast metrics are logged
* forecast-error distribution is saved
* forecast output connects to Module 5

---

## Definition of Done — Tier 2 extension

Tier 2 forecasting is complete when:

* probabilistic output is implemented
* calibration is performed correctly
* uncertainty is quantified
* confidence can be consumed by the controller
* calibration is evaluated separately from point accuracy

---

## Do not

* leak future information
* randomly shuffle time-series observations for final evaluation
* call better RMSE "better control" without downstream evidence
* claim calibrated uncertainty without calibration testing
* replace Tier-1 baseline until the new model has demonstrated value

---

# 7. MODULE 5 — CONTROLLER

## Purpose

Determine WEC control actions using the available state, forecast information and control objectives.

The controller is explicitly tiered.

---

# 7.1 Tier-1 Forecast-Informed RL

## Inputs

Potential observation components:

* rolling observed wave history
* future forecast
* current heave displacement
* current heave velocity
* latch status
* relevant physical/control state

---

## Actions

Primary Tier-1 action space:

* latch
* release

Continuous PTO control is optional and must not block the Tier-1 result.

---

## Reward

Primary objective:

* extracted energy / power

Potential secondary penalties:

* excessive switching
* end-stop/constraint violations

The exact reward must be version-controlled and documented.

---

## RL framework

Preferred tools include:

* Gymnasium
* Stable-Baselines3
* PPO/SAC/DQN as justified by the current experiment

Do not change algorithms solely because another algorithm sounds more advanced.

Choose based on:

* stability
* action space
* reproducibility
* training cost
* evidence for the project

---

## Required controller variants

The evaluation architecture should support:

1. passive
2. classical threshold
3. reactive RL
4. forecast-informed RL

These variants are necessary for causal comparison.

---

# 7.2 Tier-2 Confidence-Gated Predictive Control

Tier 2 adds:

* probabilistic forecast
* calibrated confidence
* predictive control
* confidence gate

Conceptually:

```
forecast
   +
confidence
   |
   v
predictive action
   |
   +------ confidence gate ------+
                                  |
                          reactive fallback
                                  |
                                  v
                           final control
```

The core research hypothesis is that the controller should become more predictive when forecast trustworthiness is high and more reactive when trustworthiness is low.

---

# 7.3 Tier-3 Residual RL

Tier 3 changes the learning target from:

```
learn complete controller
```

to:

```
physics-based action
      +
learned residual
      =
final action
```

RL should learn corrections rather than replace the entire physics-based decision.

---

# Controller validation

Test:

* observation construction
* action validity
* reward correctness
* environment reset
* episode termination
* constraint handling
* deterministic evaluation
* model save/load
* seeded reproducibility

---

## Definition of Done — Tier 1

Controller is complete when:

* Gymnasium environment runs
* baseline controller runs
* reactive RL can train/evaluate
* forecast-informed RL can train/evaluate
* observations are verified
* reward is verified
* evaluation is reproducible
* Module 2 integration works

---

# 8. MODULE 6 — EVALUATION HARNESS

## Purpose

Provide an independent and reproducible framework for comparing forecasting/control configurations.

The evaluation harness is the project's main defence against cherry-picked results.

---

## Inputs

From other modules:

* wave environment outputs
* WEC dynamics outputs
* forecast outputs
* controller outputs
* metadata

---

## Primary experiment modes

### A. Reactive

No future forecast.

### B. Perfect forecast

Future state is supplied without forecast error.

Used as an information upper-bound experiment.

### C. Realistic forecast

Controller receives the actual forecaster's imperfect output.

This is the headline deployable predictive condition.

### D. Forecast withheld

Forecast is unavailable or intentionally removed.

Used for fallback/robustness analysis.

---

## Required baseline ladder

At minimum:

### Experiment A

Passive

### Experiment B

Classical threshold latching

### Experiment C

Reactive RL

### Experiment D

Predictive control / forecast-informed controller as implemented

### Experiment E

Perfect forecast condition

### Experiment F

Realistic forecast condition

Additional experiments are added as higher tiers mature.

---

## Metrics

### Forecasting

* RMSE
* MAE
* error by lead time
* calibration metrics when applicable

### WEC/control

* instantaneous power
* cumulative energy
* capture-width ratio
* baseline-relative improvement
* reference/theoretical optimum when justified

### Robustness

* perfect forecast
* realistic forecast error
* withheld forecast
* sensor dropout
* distribution shift
* bounded perturbations

---

## Required plots

The evaluation harness should support at least:

* forecast error vs lead time
* energy/power comparison
* baseline comparison
* robustness vs forecast error
* controller performance under different information conditions

Later-tier headline plots may include:

* energy vs forecast confidence
* energy vs forecast error
* predictive vs reactive behaviour across confidence regimes

---

## Definition of Done

Module 6 is complete for Tier 1 when:

* all baseline controllers can be evaluated consistently
* held-out test sea states are used
* forecast conditions are explicit
* metrics are reproducible
* benchmark tables are generated
* plots are generated
* results can be traced back to configuration/model/data version

---

# 9. MODULE INTERFACE CONTRACTS

The following logical contracts must remain stable.

## Module 1 → Module 2

Provides:

* time
* wave elevation/excitation
* environmental metadata

---

## Module 1 → Module 3

Provides:

* true wave state

---

## Module 3 → Module 4

Provides:

* observed wave history
* timestamps
* observation metadata

The data must represent sensing output rather than hidden true future state.

---

## Module 4 → Module 5

Provides:

* forecast
* lead-time information
* optional uncertainty/confidence in Tier 2+

---

## Module 5 → Module 2

Provides:

* control action
* PTO command
* latch/release state

---

## Modules 2 + 4 + 5 → Module 6

Provides:

* plant trajectory
* forecast quality
* controller actions
* power/energy
* experiment metadata

---

# 10. SHARED DATA CONTRACT

Shared data structures should clearly distinguish:

### State

What is true inside the simulation.

### Observation

What the simulated sensing system makes available.

### Forecast

What the forecasting model predicts.

### Control

What the controller commands.

### Evaluation

What is measured afterward.

Never use one generic object for all five concepts if doing so makes information leakage possible.

---

# 11. CROSS-MODULE RULES

## Rule 1

Lower modules must not depend on higher-level AI logic.

Example:

Module 2 must not import Module 5 merely to define physical dynamics.

---

## Rule 2

Physics must not depend on the forecasting model.

The WEC plant must remain usable with:

* no controller
* classical controller
* reactive controller
* predictive controller

---

## Rule 3

The forecasting model must not directly alter the physical simulator.

Its role is to estimate future information.

---

## Rule 4

The evaluation system must not secretly improve/retrain a model during testing.

Evaluation is evaluation.

---

## Rule 5

Training and evaluation datasets must be explicit.

---

## Rule 6

Experimental extensions should be additive.

A new research tier should not destroy the validated baseline.

---

# 12. CURRENT PRIORITY ORDER

Unless the user explicitly changes the research plan, development priority is:

## Priority 1

Make Tier-1 modules stable.

## Priority 2

Complete the full Tier-1 closed loop.

## Priority 3

Complete benchmark/ablation evaluation.

## Priority 4

Only then implement Tier 2 confidence-gated control.

## Priority 5

Tier 3 residual RL.

## Priority 6

Tier 4 robustness hardening.

## Priority 7

Tier 5 advanced/geospatial forecasting and cross-location validation.

---

# 13. DEFINITION OF SYSTEM-LEVEL COMPLETION

The Tier-1 project is complete when:

```
Real/reanalysis wave data
         ↓
   Module 1
         ↓
   Module 2
         ↑
   Module 5
         ↑
   Module 4
         ↑
   Module 3
         ↓
   Module 6
```

can be executed as a reproducible closed-loop experiment on held-out sea states.

The system must support comparison between:

* passive operation
* threshold latching
* reactive RL
* forecast-informed control

and must report results under explicit forecast-information conditions.

---

# 14. FAILURE / FALLBACK POLICY

If a higher-tier module fails:

### Forecasting failure

Fallback:

* simpler LSTM/GRU baseline

### Advanced control failure

Fallback:

* Tier-1 RL

### RL instability

Fallback:

* simpler action space
* simpler algorithm
* classical baseline

### Probabilistic calibration failure

Fallback:

* deterministic Tier-1 forecast

### Advanced forecasting failure

Fallback:

* Tier-1 LSTM

### Advanced robustness failure

Fallback:

* Tier-1 three-condition test

A failed advanced experiment must never erase a working lower-tier result.

---

# 15. MODULE CHANGE POLICY

Before modifying a module, identify:

1. current status
2. reason for change
3. dependent modules
4. tests affected
5. scientific assumptions affected
6. rollback/fallback

Before changing a shared interface:

* search all callers
* update tests
* run integration tests
* inspect downstream impact

---

# 16. MODULE CHANGE REPORT

When an AI agent modifies a module, it should report:

## Module

Name and tier.

## Change

What changed.

## Reason

Why.

## Inputs/outputs

Whether the interface changed.

## Scientific impact

Equations/data/assumptions affected.

## Validation

Tests and experiments run.

## Regression risk

Potential effects on downstream modules.

## Fallback

Previous stable version or tier.

---

# 17. GOLDEN MODULE PRINCIPLE

Every module should answer five questions:

1. What goes in?
2. What happens inside?
3. What comes out?
4. How do we know it is correct?
5. What is the fallback if it fails?

If any module cannot answer all five, it is not yet sufficiently specified for autonomous implementation.

---

# 18. FINAL MODULE MAP

```
┌──────────────────────────────┐
│ MODULE 1                     │
│ Ocean / Wave Environment     │
└──────────────┬───────────────┘
               │
      true wave environment
               │
      ┌────────┴─────────┐
      │                  │
      v                  v
MODULE 2             MODULE 3
WEC Dynamics         Sensing Proxy
      ^                  │
      │                  v
      │             MODULE 4
      │             Forecaster
      │                  │
      │             forecast/
      │             uncertainty
      │                  │
      └──────────┬───────┘
                 v
             MODULE 5
             Controller
                 │
            control action
                 │
                 v
             MODULE 2
                 │
                 v
             MODULE 6
             Evaluation
```

The architecture is complete when this graph can execute without hidden dependencies, information leakage, or scientifically unexplained shortcuts.
