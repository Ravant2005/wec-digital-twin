# SCIENCE RULES

## 1. PURPOSE

This document defines the scientific rules that must govern all implementation, experiments, analysis and AI-assisted development in the WEC project.

These rules exist to prevent:

* invented physics
* hidden assumptions
* data leakage
* invalid experimental comparisons
* misleading performance claims
* accidental information leakage
* reward hacking
* unsupported generalization claims
* confusing simulation results with hardware results
* confusing successful software execution with scientific validation

These rules take priority over convenience.

---

# 2. CORE SCIENTIFIC PRINCIPLE

The project is a:

```
PHYSICS-BASED
SIMULATION-FIRST
WEC DIGITAL TWIN
```

The system combines:

* ocean/wave data
* physics-based wave/environment modelling
* physics-based WEC dynamics
* simulated sensing
* machine-learning forecasting
* control/RL
* evaluation

Machine learning is an additional layer.

It must not silently replace established physical modelling unless that replacement is an explicitly defined experiment.

---

# 3. NEVER INVENT SCIENTIFIC FACTS

The AI agent must never fabricate:

* equations
* hydrodynamic coefficients
* buoy properties
* empirical constants
* datasets
* physical measurements
* literature results
* sensor accuracies
* forecast accuracies
* control gains
* energy improvements
* validation results

If a required quantity is unavailable:

1. search the existing project/data/documentation
2. identify an authoritative source if the task permits external research
3. state the missing quantity explicitly
4. use a documented assumption only when appropriate
5. label the assumption clearly

Never silently guess.

---

# 4. PHYSICS MUST REMAIN TRACEABLE

Every important physical computation should be traceable to:

* governing equation
* parameter
* unit
* numerical method
* boundary/initial condition where relevant
* data/source
* assumption

A developer should be able to answer:

```
What equation produced this number?
What parameters were used?
What units are they in?
Where did those parameters come from?
```

---

# 5. UNIT DISCIPLINE

Units are mandatory for physical quantities.

Examples:

| Quantity          | Expected unit |
| ----------------- | ------------- |
| Time              | s             |
| Wave height       | m             |
| Wave period       | s             |
| Frequency         | Hz            |
| Angular frequency | rad/s         |
| Displacement      | m             |
| Velocity          | m/s           |
| Acceleration      | m/s²          |
| Force             | N             |
| Power             | W             |
| Energy            | J             |

Do not silently mix:

* Hz and rad/s
* degrees and radians
* seconds and hours
* metres and kilometres
* geographic coordinates and Cartesian coordinates
* dimensional and normalized variables

When normalization/standardization is used, record:

* original units
* transformation
* fitted statistics
* inverse transformation

---

# 6. WAVE REPRESENTATION

The project may represent irregular sea states through spectral methods.

Relevant planned models include:

* JONSWAP
* Pierson–Moskowitz
* directional spreading

The wave-generation implementation must preserve the definitions and conventions actually selected by the project.

Where frequency-domain quantities are converted into time-domain waves:

* frequency convention must be explicit
* phase convention must be explicit
* timestep must be explicit
* duration must be explicit
* direction convention must be explicit

---

# 7. DISPERSION RELATION

Where linear wave theory is used, the project plan identifies:

```
ω² = g k tanh(kd)
```

where:

* ω = angular frequency
* k = wave number
* g = gravitational acceleration
* d = water depth

Implementations must maintain dimensional consistency.

Do not silently substitute a deep-water approximation for finite-depth calculations without documenting the approximation.

If a simplification is used, state:

* what was simplified
* why
* where it is valid
* what effect it may have

---

# 8. SEA-STATE VALIDATION

When a synthetic or reconstructed wave series is generated from target sea-state parameters, verify that its resulting characteristics are consistent with the intended inputs.

At minimum, where applicable:

* Hs consistency
* Tp consistency
* directional consistency
* spectral consistency

A generated signal that looks visually plausible is not sufficient scientific validation.

---

# 9. DATA PROVENANCE

The project must distinguish four categories.

## 9.1 In-situ observations

Examples include:

* INCOIS
* WAMAN
* OMNI / NIOT buoy observations

These represent observational data.

---

## 9.2 Reanalysis

Example:

* ERA5

ERA5 is reanalysis/model-derived information and must not be described as direct in-situ measurement.

The distinction must remain visible in:

* code
* metadata
* experiments
* reports
* figures

---

## 9.3 Operational forecast products

Example:

* INCOIS Ocean State Forecast

These may be used as auxiliary information or sanity checks when appropriate.

They must not accidentally leak into the target-definition/evaluation procedure.

---

## 9.4 Synthetic data

Examples:

* JONSWAP realizations
* Pierson–Moskowitz realizations
* simulated camera noise
* synthetic forecast perturbation
* controlled stress-test conditions

Synthetic data are appropriate for:

* physics validation
* algorithm development
* controlled experiments
* stress testing

They are not sufficient by themselves to support the project's real-data headline claim.

---

# 10. DATA LEAKAGE RULE

No future information may enter a forecasting model through:

* feature construction
* preprocessing
* normalization
* interpolation
* window construction
* target engineering
* dataset shuffling
* caching
* metadata

For time-series forecasting:

```
PAST → TRAIN
LATER PERIOD → VALIDATION
FUTURE HELD-OUT PERIOD → TEST
```

Do not randomly shuffle the complete time series for headline forecasting results.

---

# 11. PREPROCESSING RULE

Anything learned from data must be fit using the appropriate training data.

Examples:

* normalization mean/std
* scalers
* imputation parameters
* feature-selection statistics
* calibration parameters
* learned transforms

Do not fit preprocessing using the complete dataset before splitting.

The transformation applied to validation/test data must come from the training pipeline.

---

# 12. FORECASTING RULES

The Tier-1 baseline is an LSTM/GRU sequence forecaster.

The architecture may evolve later.

However:

* baseline performance must remain reproducible
* advanced models must be compared against the baseline
* output alignment must be verified
* lead-time indexing must be correct
* target timestamps must correspond exactly to prediction timestamps

A better forecasting score does not automatically mean better WEC control.

The causal chain must be demonstrated:

```
better forecast
      ↓
better decision
      ↓
better control
      ↓
more energy / better objective
```

---

# 13. FORECAST ERROR IS A SCIENTIFIC OBJECT

Forecast error must be treated as measurable information.

Where possible, record:

* error by lead time
* error distribution
* error by sea-state regime
* error by target variable

Do not simply inject an arbitrary Gaussian noise level and call it "realistic" unless the assumption is explicitly documented and justified.

The research plan specifically calls for using measured forecast-error behaviour for robustness work as the project advances.

---

# 14. WEC PHYSICS MODEL

The Tier-1 WEC is a heaving point absorber.

The project uses a time-domain hydrodynamic model based on the Cummins-equation formulation.

The implementation must preserve the intended physical structure:

* body mass
* infinite-frequency added mass representation where used
* radiation-memory contribution
* hydrostatic restoring force
* wave excitation force
* PTO/control force

Capytaine is the planned BEM source for hydrodynamic coefficients.

SciPy is used for numerical time integration where appropriate.

---

# 15. HYDRODYNAMIC COEFFICIENT RULE

Never fabricate:

* added mass
* damping
* excitation coefficients
* radiation kernel
* hull properties

If coefficients look implausible:

1. inspect mesh/geometry
2. inspect solver configuration
3. inspect units
4. inspect frequency convention
5. inspect coordinate/sign conventions
6. compare against a documented/reference configuration where available

Do not "fix" suspicious coefficients by manually changing values until they produce a desirable control result.

---

# 16. NUMERICAL-INTEGRATION RULES

For time-domain WEC simulation, investigate:

* timestep sensitivity
* numerical stability
* initial conditions
* transient effects
* integration tolerance
* memory/convolution behaviour
* solver warnings

If changing a solver setting materially changes the physical result:

* document the change
* perform a convergence/sensitivity check where necessary

A single simulation run is not evidence of numerical correctness.

---

# 17. POWER CALCULATION

Extracted power must follow the project's defined PTO sign convention.

The conceptual relation is:

```
P(t) = F_PTO(t) × v(t)
```

The implementation must explicitly establish which force direction corresponds to positive extracted power.

Never change the sign merely to make plots look positive.

Validate power using simple known cases.

---

# 18. ENERGY CALCULATION

Energy should be calculated consistently from power over time.

The implementation must document:

* integration timestep
* interval definition
* units
* treatment of initial/final samples

Do not mix:

* instantaneous power
* average power
* cumulative energy

when reporting results.

---

# 19. SIMULATED CAMERA / SENSOR RULES

The camera layer is a simulation proxy.

It must not be described as a validated hardware camera implementation unless separate evidence exists.

The sensing layer may introduce:

* noise
* latency
* sampling limitations
* missing observations

The exact distributions and values must be documented.

The controller/forecasting stack should receive the simulated observation, not hidden ground-truth future information, except in explicitly defined perfect-information experiments.

---

# 20. INFORMATION-ACCESS RULE

The experiment must explicitly specify what information the controller receives.

At minimum distinguish:

### Reactive

Current/past observed information only.

### Perfect forecast

Future information provided without prediction error.

### Realistic forecast

Actual model forecast including realistic prediction error.

### Forecast withheld

Forecast information unavailable or removed.

Do not accidentally expose true future wave values to a "realistic forecast" controller.

---

# 21. CONTROLLER SCIENCE RULE

The controller must be evaluated against meaningful baselines.

At minimum, the intended ladder includes:

1. Passive
2. Classical threshold control
3. Reactive RL
4. Forecast-informed predictive control/RL

Later tiers add:

5. probabilistic forecast
6. predictive MPC
7. confidence gating
8. residual RL
9. robustness-aware control

A new controller must never be evaluated only against a weak or poorly tuned baseline chosen to make the new method look good.

---

# 22. RL REWARD INTEGRITY

The reward function is part of the scientific method.

Do not change it casually.

Every reward change must document:

* physical objective
* term added/removed
* mathematical form
* weight
* purpose
* possible side effects

Never modify the reward merely because:

```
"training wasn't working."
```

First diagnose:

* environment
* observation
* action space
* normalization
* episode setup
* reward scale
* algorithm choice
* hyperparameters
* simulator stability

A reward that produces higher reward but worse physical performance must be investigated, not celebrated.

---

# 23. RL REPRODUCIBILITY

Record where appropriate:

* random seed
* environment configuration
* observation definition
* action definition
* reward definition
* algorithm
* hyperparameters
* training duration
* evaluation protocol
* checkpoint/model version

A single lucky run must not be presented as proof.

---

# 24. RL TRAINING VS SCIENTIFIC VALIDATION

These are separate states:

### Infrastructure works

Environment and training pipeline execute.

### Smoke test works

Small controlled run behaves correctly.

### Training works

Policy can actually learn under defined conditions.

### Convergence demonstrated

Performance is stable enough to evaluate.

### Held-out evaluation

Policy tested on data/sequences not used for training.

### Scientific comparison

Policy compared against baselines under controlled conditions.

Do not collapse these into one "trained" label.

---

# 25. FORECAST ROBUSTNESS

The Tier-1 robustness logic requires distinguishing at least:

1. perfect forecast
2. realistic forecast error
3. forecast withheld

The experiment should answer:

* how much does prediction help?
* how badly does the controller degrade when the forecast is imperfect?
* does the controller remain useful without prediction?

A predictive controller that performs well only with perfect future knowledge must not be presented as robust.

---

# 26. CAUSAL ABLATION RULE

The project's results should support a meaningful ladder.

The analysis should be capable of distinguishing:

```
no forecast
   ↓
forecast
   ↓
probabilistic forecast
   ↓
confidence-aware control
   ↓
residual RL
   ↓
robustness training
```

When a change improves energy capture, investigate which mechanism caused the improvement.

Do not attribute an improvement to "AI" as a single undifferentiated factor.

---

# 27. BASELINE INTEGRITY

Baselines must be:

* implemented correctly
* documented
* reasonably configured
* evaluated on the same relevant test conditions

Do not intentionally cripple the baseline.

Do not tune the new method on the test set while leaving the baseline untouched.

---

# 28. TRAIN / VALIDATE / TEST SEPARATION

Keep separate:

### Training

Used to learn model parameters.

### Validation

Used for model/configuration selection.

### Test

Used for final unbiased evaluation.

Do not repeatedly tune the model against the final test set and then continue calling it an unbiased test.

---

# 29. HELD-OUT SEA STATES

For headline results, use sea-state segments not used to train the relevant forecasting/control system.

Record:

* source
* time period
* site
* sea-state characteristics
* preprocessing
* model version
* experiment configuration

---

# 30. METRIC INTEGRITY

Forecast metrics include:

* RMSE
* MAE
* lead-time error

Probabilistic forecasting may additionally use:

* CRPS
* prediction interval coverage
* calibration measures

Control metrics may include:

* power
* energy
* capture-width ratio
* baseline-relative gain
* percentage of reference/theoretical optimum where scientifically justified

Never report only the metric that makes the method look best.

---

# 31. RESULT ROUNDING

Do not manufacture precision.

If the data support only an approximate conclusion, do not report excessive decimal places.

Example:

Bad:

```
12.483729% improvement
```

when the uncertainty/variation does not justify that precision.

Prefer an appropriately supported precision.

---

# 32. STATISTICAL CLAIMS

Do not call a difference "significant" merely because:

* one mean is larger
* one reward curve looks higher
* one run performed better

When statistical claims matter, use an appropriate analysis and document:

* number of runs
* sampling unit
* variability
* statistical test where appropriate
* effect size where useful

---

# 33. GENERALIZATION CLAIMS

The system is initially designed around Goa.

Do not claim:

* universal geographic performance
* "works anywhere"
* global generalization

unless multiple geographically distinct validation sites support such a claim.

A location-conditioned architecture is a design strategy, not evidence of generalization.

---

# 34. SIMULATION VS HARDWARE

The project is simulation-first.

Therefore never write:

* "the physical buoy achieved..."
* "the real camera achieved..."
* "the hardware system produced..."

unless actual hardware evidence exists.

Use:

* simulation
* digital twin
* simulated sensing
* numerical experiment
* reanalysis-driven simulation

where appropriate.

---

# 35. CFD LIMITATION

The Tier-1 architecture intentionally does not require a full 3D CFD wave flume.

Do not claim:

* CFD-level validation
* full fluid-structure interaction
* hardware-equivalent fidelity

from the lower-cost time-domain hydrodynamic model alone.

Where possible, cross-check against appropriate reference data.

---

# 36. NEGATIVE RESULTS

Negative results are scientifically valid.

If:

* the advanced forecaster does not beat the LSTM
* confidence gating does not improve energy capture
* residual RL becomes unstable
* robustness training does not help
* a second site fails to generalize

record it honestly.

Do not modify the experiment merely to remove an inconvenient result.

---

# 37. NO POST-HOC CLAIM DRIFT

The claim must match the evidence.

Do not write:

```
"Our method is the most accurate"
```

when only one baseline was tested.

Do not write:

```
"works for any geographical location"
```

when only Goa was evaluated.

Do not write:

```
"patentable"
```

without an appropriate prior-art/legal process.

Use evidence-grounded language.

---

# 38. FORECAST IMPROVEMENT ≠ CONTROL IMPROVEMENT

This is a central scientific rule.

An improvement in:

```
RMSE / MAE
```

does not automatically imply an improvement in:

```
harvested energy
```

The evaluation must test the downstream causal relationship.

The headline scientific result is about useful control performance, not forecasting accuracy alone.

---

# 39. UNCERTAINTY RULES FOR TIER 2

If probabilistic forecasting is introduced:

* distinguish uncertainty from ordinary forecast error
* define exactly how confidence is derived
* calibrate confidence
* evaluate coverage/calibration
* avoid hand-picked "confidence" values

A confidence value is not scientifically meaningful merely because the model outputs a number between 0 and 1.

---

# 40. CONFIDENCE-GATING RULE

The Tier-2 hypothesis is:

```
more trustworthy forecast
        →
more aggressive predictive control
```

and:

```
less trustworthy forecast
        →
safer/reactive behaviour
```

The gate must be experimentally evaluated.

Do not hard-code a threshold and call the resulting system "confidence-aware" without testing whether the confidence signal actually carries useful information.

---

# 41. THEORETICAL OPTIMUM

Where energy is compared to a theoretical/non-causal or reference optimum:

* define the reference precisely
* distinguish it from measured baseline performance
* ensure assumptions match the plant/model
* do not compare incompatible physical formulations

Never call the reference a measured achievable hardware result.

---

# 42. COMPUTATIONAL REPRODUCIBILITY

Record enough information to reproduce important experiments:

* Git commit
* configuration
* dataset/version
* model/checkpoint
* seed where applicable
* software versions where relevant
* evaluation conditions

Generated plots/results without provenance should not be treated as final research evidence.

---

# 43. AI CODING AGENT SCIENCE RULE

The AI agent may propose:

* equations
* algorithms
* parameter values
* experiments

but these are proposals until verified against:

* project documentation
* authoritative sources
* mathematical derivation
* tests
* physical sanity checks
* experiments

A model-generated explanation is not evidence.

---

# 44. WHEN THE AGENT MUST STOP

The AI agent must stop and ask for direction when:

1. two physically incompatible interpretations are plausible
2. a missing parameter materially changes the result
3. data provenance is unclear
4. changing an equation could invalidate previous experiments
5. an evaluation protocol would change the research claim
6. a result suggests a surprising scientific conclusion that needs verification
7. an implementation decision could introduce hidden future information
8. an irreversible scientific-methodology change is proposed

---

# 45. SCIENCE REVIEW CHECKLIST

Before accepting a meaningful scientific change, check:

### Physics

* Are equations correct?
* Are units correct?
* Are signs/conventions correct?
* Are assumptions documented?

### Data

* Is provenance known?
* Is the data type correctly labelled?
* Is leakage prevented?

### ML

* Are train/validation/test sets separated?
* Is preprocessing leakage-free?
* Is the metric appropriate?

### RL/control

* Is the reward physically meaningful?
* Is the baseline fair?
* Is the evaluation held out?
* Are information conditions explicit?

### Results

* Are the claims supported?
* Is the uncertainty reported?
* Are negative results preserved?

---

# 46. FINAL PRINCIPLE

The WEC system must never become:

```
"a large AI model that generates impressive code."
```

It must become:

```
a reproducible scientific system
in which AI accelerates engineering
while physics, data provenance,
experiment design and scientific claims
remain evidence-controlled.
```

The hierarchy is:

```
PHYSICS
   ↓
DATA
   ↓
MODEL
   ↓
CONTROL
   ↓
EXPERIMENT
   ↓
EVIDENCE
   ↓
CLAIM
```

Never reverse this order.

Do not choose the physics because the AI implementation is convenient.

Do not choose the experiment because it produces a nicer graph.

Do not choose the claim before measuring the result.

Scientific correctness takes priority over implementation speed.
