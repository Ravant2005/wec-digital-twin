# WEC PROJECT ARCHITECTURE

## 1. PURPOSE

This document defines the software and scientific architecture of the Wave Energy Converter (WEC) project.

The system is a simulation-first, physics-based digital twin of a heaving point-absorber WEC operating under real-world Goa-region Arabian Sea wave conditions.

The architecture combines:

* real/reanalysis wave data
* physics-based ocean/wave modelling
* hydrodynamic WEC dynamics
* simulated sensing
* deep-learning wave forecasting
* predictive/RL control
* systematic evaluation and robustness testing

The architecture is intentionally modular so that:

1. every module can be tested independently
2. modules can be replaced without rewriting the entire system
3. Tier-1 remains runnable even when advanced research features fail
4. later research tiers can be added incrementally
5. scientific assumptions remain visible and auditable

---

# 2. ARCHITECTURAL PRINCIPLE

The system follows:

```
REAL / REANALYSIS OCEAN DATA
            |
            v
    MODULE 1: OCEAN
            |
            v
    MODULE 2: WEC DYNAMICS
            ^
            |
    CONTROL ACTION
            ^
            |
    MODULE 5: CONTROLLER
            ^
            |
    +-------+--------+
    |                |
    |                v
    |        MODULE 4: FORECASTER
    |                ^
    |                |
    |        MODULE 3: SENSING
    |                ^
    |                |
    +------- OCEAN OBSERVATION

            |
            v
    MODULE 6: EVALUATION
```

The controller does not directly bypass the sensing/forecasting architecture unless an explicit experiment requires it.

The physics model remains the plant/environment being controlled.

Machine learning does not replace the underlying WEC physics model.

---

# 3. CORE SYSTEM MODULES

The Tier-1 system consists of six modules.

## Module 1 — Ocean / Wave Environment

### Purpose

Represent the sea state at the WEC location and generate the wave excitation/environmental inputs required by the WEC dynamics model.

### Inputs

Primary data/configuration may include:

* significant wave height Hs
* peak period Tp
* wave direction
* spectral parameters
* water depth
* geographic/site configuration
* historical/reanalysis wave observations

Potential data sources include:

* ERA5 reanalysis
* INCOIS / WAMAN / OMNI observations where available
* controlled synthetic spectral realizations

### Processing

The module may implement:

* irregular wave generation
* JONSWAP spectrum
* Pierson–Moskowitz spectrum
* directional spreading
* frequency-domain representation
* wave-component synthesis
* dispersion relation
* transformation from sea-state parameters to time-domain surface elevation
* wave excitation representation

### Outputs

At minimum:

* sea-surface elevation η(t)
* wave excitation information at the WEC
* replayable sea-state segments
* associated metadata

### Scientific role

This module defines the environmental forcing of the digital twin.

It must distinguish:

* observed/in-situ conditions
* ERA5 reanalysis
* synthetic validation/stress-test conditions

Synthetic waves must never silently replace the real-data basis of Tier-1 evaluation.

---

# 4. MODULE 2 — BUOY / WEC DYNAMICS

## Purpose

Represent the physical WEC plant being controlled.

The Tier-1 WEC is a heaving point absorber with a dominant heave degree of freedom.

### Inputs

* Module 1 wave excitation
* buoy/hull parameters
* hydrodynamic coefficients
* mass
* hydrostatic stiffness
* control action from Module 5

### Physics

The time-domain dynamics are based on the Cummins equation.

Conceptually:

```
(m + A∞) z¨(t)
+ radiation-memory contribution
+ C z(t)
+ F_PTO(t)
=
F_exc(t)
```

The actual implementation must respect the project's established mathematical formulation and sign conventions.

### Hydrodynamic information

The preferred hydrodynamic pipeline uses BEM-derived coefficients such as:

* added mass
* radiation damping / radiation response
* excitation forces
* retardation kernel or equivalent time-domain representation

Capytaine is the primary BEM tool.

SciPy numerical integration is used for time-domain propagation where appropriate.

### Outputs

* heave displacement
* heave velocity
* acceleration where required
* PTO force
* instantaneous extracted power
* control/constraint state
* diagnostic quantities

### Baselines

This module must support at least:

* passive damping / passive operation
* fixed-threshold latching

so subsequent AI controllers can be evaluated against classical references.

---

# 5. MODULE 3 — SYNTHETIC CAMERA / SENSING LAYER

## Purpose

Approximate what an onboard sensing system would provide to the forecasting/controller stack.

The system does not claim to have a real physical camera.

Instead, this module creates a statistically realistic observation process from the simulated true ocean state.

### Inputs

* Module 1 true sea state
* camera/sensor noise model
* observation latency
* sampling/update rate

### Observation model

Conceptually:

```
observed_Hs  = true_Hs  + measurement_noise
observed_Tp  = true_Tp  + measurement_noise
observed_dir = true_dir + measurement_noise
```

The exact distributions and parameter values must be documented rather than silently guessed.

### Important characteristics

The sensing layer may include:

* measurement noise
* delayed observation
* rolling observation buffers
* missing/dropout conditions for robustness experiments

### Outputs

The controller/forecaster-facing observation is not the perfect simulated truth.

It is:

```
TRUE OCEAN STATE
      |
      v
  SENSOR MODEL
      |
   noise + delay
      |
      v
OBSERVED STATE
```

This separation is essential for evaluating realistic controller behaviour.

---

# 6. MODULE 4 — WAVE FORECASTING

## Purpose

Predict future wave-state variables from the observation history available to the WEC system.

The Tier-1 forecasting model is a recurrent baseline.

### Tier-1 baseline

Primary model family:

* LSTM and/or GRU sequence model

### Inputs

The forecasting system may consume:

* rolling observed Hs
* rolling observed Tp
* rolling observed direction
* optional environmental variables such as wind
* historical context window

The Tier-1 research plan uses a two-day historical context with a two-day forecast horizon, subject to the final implementation configuration.

### Outputs

Forecasts may include:

* future Hs
* future Tp
* future direction

at the configured prediction resolution.

### Evaluation

Forecast quality must be evaluated using time-aware splits.

Core metrics may include:

* RMSE
* MAE
* error by lead time
* forecast-error distribution

The forecast-error distribution is particularly important because it becomes part of the controller-robustness methodology.

### Data discipline

Never:

* randomly shuffle future data into training
* use test information for preprocessing
* leak future forecast targets into past observations
* treat ERA5 as equivalent to direct buoy measurement

---

# 7. MODULE 5 — CONTROLLER

Module 5 is the control intelligence layer.

It is intentionally tiered.

---

## 7.1 Tier-1 Controller

The Tier-1 controller is a forecast-informed RL controller.

### Observation may include

* historical observed wave state
* future forecast
* current heave position
* current heave velocity
* latch status
* other explicitly configured control state

### Action

The Tier-1 implementation uses a simple action space appropriate for latching control, such as:

* latch
* release

Continuous PTO control can be added later if justified and stable.

### Reward

The primary physical objective is energy capture.

Conceptually:

```
power ≈ PTO force × velocity
```

The exact reward function must remain explicit and version-controlled.

Potential penalties may include:

* excessive switching
* constraint/end-stop violations

but they must never be introduced without documenting their purpose and weighting.

---

# 8. TIER-2+ CONTROL ARCHITECTURE

After Tier 1 works, the controller can evolve into a confidence-aware predictive architecture.

The intended architecture is:

```
OCEAN
  |
  v
WAVE FORECASTER
  |
  v
FORECAST + CALIBRATED UNCERTAINTY
  |
  v
CONFIDENCE / RISK ESTIMATION
  |
  +----------------------+
  |                      |
HIGH                    LOW
  |                      |
  v                      v
```

PREDICTIVE MPC        REACTIVE / THRESHOLD
|                      |
+----------+-----------+
|
v
RESIDUAL RL CORRECTION
|
v
PTO
|
v
WEC

The defining idea of the advanced architecture is not simply "use a larger neural network."

The central research question is:

```
How aggressively should the controller
trust its forecast given the forecast's
current calibrated confidence?
```

---

# 9. TIER-2 CONFIDENCE GATING

Tier 2 introduces calibrated forecast uncertainty.

Candidate methods include:

* quantile prediction heads
* pinball loss
* deep ensembles
* conformal calibration

The uncertainty signal must be calibrated rather than manually invented.

Conceptually:

```
forecast
   +
uncertainty
   |
   v
confidence
   |
   v
control aggressiveness
```

A conceptual control law is:

```
u_t =
    α(U_t) * u_predictive
    +
    (1 - α(U_t)) * u_reactive
```

where:

* U_t represents calibrated forecast confidence
* α(U_t) determines predictive-control aggressiveness
* high confidence allows more predictive behaviour
* low confidence increases reliance on reactive behaviour

This equation is a conceptual architecture definition.

Do not treat it as a validated final control law until experiments establish the implementation and evidence.

---

# 10. TIER-3 RESIDUAL RL

Tier 3 places RL above a physics-based predictive controller.

Instead of learning the complete controller from scratch:

```
MPC / physics-based action
          |
          v
   residual RL correction
          |
          v
      final action
```

The purpose is to allow learning to correct:

* nonlinear behaviour
* plant/model mismatch
* end-stop effects
* unmodelled dynamics

The Tier-1 RL controller remains the fallback if Tier 3 proves unstable.

---

# 11. TIER-4 ROBUSTNESS ARCHITECTURE

Tier 4 strengthens robustness.

Candidate stressors include:

* measured forecast-error distributions
* sensor/camera dropout
* storm/distribution shift
* bounded forecast perturbations
* lower-tail performance objectives
* CVaR/distributional objectives where justified

The baseline robustness comparison remains:

1. perfect forecast
2. realistic forecast error
3. forecast withheld

Higher-order stress testing must not replace these basic conditions.

---

# 12. MODULE 6 — EVALUATION HARNESS

## Purpose

Provide an independent evaluation layer for the entire system.

Module 6 must prevent the controller from evaluating itself using only its own preferred metrics.

### Inputs

Outputs from:

* Module 2
* Module 4
* Module 5

across defined test sea states.

### Core evaluation families

#### Forecasting

* RMSE
* MAE
* lead-time performance
* probabilistic calibration metrics when applicable

#### Energy/control

* instantaneous power
* accumulated energy
* capture-width ratio
* baseline-relative gain
* theoretical/reference optimum comparisons where scientifically justified

#### Robustness

* perfect forecast
* realistic forecast error
* withheld forecast
* sensor dropout
* distribution shift
* bounded perturbation

#### Ablation

The evaluation harness should support a ladder capable of answering:

```
Does forecasting help?
      ↓
Does RL help?
      ↓
Does predictive information help?
      ↓
Does uncertainty help?
      ↓
Does confidence gating help?
      ↓
Does residual RL help?
      ↓
Does robustness training help?
```

---

# 13. DATA FLOW

The primary Tier-1 closed loop is:

```
Real / reanalysis wave data
          |
          v
Module 1 — Ocean Environment
          |
          +--------------------+
          |                    |
          v                    |
  True simulated ocean        |
          |                    |
          v                    |
Module 3 — Sensor Proxy        |
          |                    |
          v                    |
   Observed wave state         |
          |                    |
          v                    |
Module 4 — Forecasting         |
          |                    |
          v                    |
    Future forecast            |
          |                    |
          +---------+----------+
                    |
                    v
          Module 5 — Controller
                    |
                    v
           Control action
                    |
                    v
          Module 2 — WEC Dynamics
                    |
                    v
          Heave / PTO / Power
                    |
                    v
          Module 6 — Evaluation
```

The true simulated state may be retained for ground-truth evaluation, but the controller must not receive hidden perfect information unless the experiment explicitly defines a perfect-information condition.

---

# 14. EXPERIMENTAL INFORMATION MODES

The system should support distinct information conditions so conclusions are causal rather than ambiguous.

At minimum, the architecture should support:

## Mode A — Reactive

Controller does not use future forecast information.

Purpose:

* establishes reactive-control baseline

## Mode B — Perfect Forecast

Controller receives future information without forecast error.

Purpose:

* estimates an upper-bound information condition
* separates the value of future information from forecast quality

## Mode C — Realistic Forecast

Controller receives the trained forecast with realistic forecast-error characteristics.

Purpose:

* represents the deployable predictive-control condition

## Mode D — Forecast Withheld

Optional explicit condition in which a forecast-trained controller loses forecast input or falls back to reactive behaviour.

Purpose:

* robustness/fallback evaluation

Each mode must be clearly recorded in experiment metadata.

---

# 15. TIER-1 INTEGRATED PIPELINE

The Tier-1 final demonstration should be executable as one coherent pipeline:

```
held-out Goa sea state
         |
         v
    sensing proxy
         |
         v
    wave forecaster
         |
         v
      RL policy
         |
         v
    WEC dynamics
         |
         v
   power/energy log
         |
         v
    evaluation plots
```

The pipeline should be reproducible from a documented configuration.

The integration script must not contain duplicated implementations of physics, forecasting or control logic.

It should call the modules.

---

# 16. SITE CONFIGURATION

Goa-specific parameters must not be scattered throughout the codebase.

Use a site-configuration abstraction for quantities such as:

* latitude
* longitude
* water depth
* ERA5 grid location
* wave-climate configuration
* seasonal settings
* buoy/hull configuration

The architecture should allow future site adaptation without rewriting the entire system.

However, cross-location generalization is an experimental claim, not an assumption.

---

# 17. CONFIGURATION SEPARATION

Separate:

### Code

Algorithms and reusable implementations.

### Configuration

Experiment parameters and site settings.

### Data

Raw and processed observations.

### Models

Trained model artifacts.

### Results

Evaluation outputs.

### Documentation

Architecture, scientific assumptions and experiment descriptions.

Do not hard-code site-specific or experiment-specific values inside core algorithms unless they are truly invariant constants.

---

# 18. REPRODUCIBILITY

Every major experiment should be associated with enough information to reconstruct:

* code version
* configuration
* model version
* random seed where applicable
* dataset/version
* train/validation/test split
* evaluation mode
* controller configuration
* forecasting configuration
* hardware/software environment where relevant

The result should be traceable from:

```
experiment result
      ↓
   config
      ↓
    model
      ↓
    data
      ↓
   code/Git
```

---

# 19. FAILURE CONTAINMENT

Higher-level research features must not become hard dependencies of lower-level validated functionality.

The intended hierarchy is:

```
Tier 1
  |
  +--> Tier 2
         |
         +--> Tier 3
                |
                +--> Tier 4
                       |
                       +--> Tier 5
```

If Tier 5 fails:

```
fall back to Tier 4
```

If Tier 4 fails:

```
fall back to Tier 3
```

If Tier 3 fails:

```
fall back to Tier 2
```

If Tier 2 fails:

```
fall back to Tier 1
```

Tier 1 must remain independently useful.

---

# 20. SOFTWARE DEPENDENCY DIRECTION

Preferred dependency structure:

```
CONFIG
  ↓
DATA LAYER
  ↓
OCEAN MODEL
  ↓
WEC DYNAMICS
  ↓
SENSOR / FORECAST / CONTROL
  ↓
EVALUATION
```

Cross-module dependencies should be explicit.

Avoid circular dependencies.

The evaluation layer should consume outputs from other modules rather than embedding alternative implementations of them.

---

# 21. RESEARCH EXPERIMENT BOUNDARY

Research experiments must not silently change production implementations.

Prefer:

```
src/
  stable implementation

experiments/
  experimental configuration / evaluation

tests/
  verification

results/
  outputs
```

A research hypothesis should be testable without permanently changing the core architecture.

---

# 22. COMPUTATIONAL ARCHITECTURE

The local development system is designed around a lightweight local AI coding model.

This AI layer is separate from the scientific simulation/ML execution environment.

AI coding assistance:

```
OpenCode
    |
    +--> local coding model
    |
    +--> optional cloud reasoning
```

Scientific computation:

```
Python
  |
  +--> NumPy
  +--> SciPy
  +--> Capytaine
  +--> PyTorch
  +--> Gymnasium
  +--> Stable-Baselines3
  +--> evaluation tooling
```

Do not confuse:

* the model used to write code
  with
* the models being developed/trained by the WEC project.

---

# 23. AI-ASSISTED DEVELOPMENT PRINCIPLE

The AI coding agent should accelerate implementation, not determine scientific truth.

Preferred workflow:

```
user requirement
      |
      v
AI inspects repository
      |
      v
   plan
      |
      v
local implementation
      |
      v
    tests
      |
      v
  diagnostics
      |
      +---- if difficult ----> stronger reasoning model
      |                         |
      |                         v
      |                   design/review
      |                         |
      +<------------------------+
      |
      v
   implementation
      |
      v
  validation
      |
      v
 user scientific review
```

The stronger cloud model should be treated as a reasoning/review resource, not as a replacement for repository-local engineering.

---

# 24. CURRENT TARGET ARCHITECTURE

The immediate target is not the full Tier-5 system.

The immediate target is:

```
Tier-1 complete closed-loop WEC system
```

with:

* real/reanalysis Goa-region wave data
* ocean/wave environment
* physics-based point-absorber dynamics
* sensing proxy
* baseline wave forecast
* forecast-aware RL controller
* robustness comparison
* evaluation harness

Once that is stable, move upward through the tier ladder.

Do not make advanced forecasting architecture, complex RL algorithms or full location generalization prerequisites for a valid Tier-1 result.

---

# 25. ARCHITECTURAL SUCCESS CRITERION

The architecture is successful when:

1. each module can be tested independently
2. the complete Tier-1 loop can execute
3. data provenance is explicit
4. physical assumptions are traceable
5. controller information conditions can be separated
6. forecast quality and control quality can be evaluated independently
7. robustness experiments can be reproduced
8. advanced tiers can be added without destroying Tier 1
9. experiments remain reproducible
10. AI assistance accelerates engineering without becoming the source of scientific authority

The ultimate architectural principle is:

```
PHYSICS defines the plant.
DATA defines the environment.
FORECASTING estimates the future.
CONTROL decides the action.
RL improves where justified.
UNCERTAINTY determines how much the system should trust prediction.
EVALUATION determines whether the claimed improvement is real.
HUMAN SCIENTIFIC REVIEW determines whether the result is defensible.
```
