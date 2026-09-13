# AI WORKFLOW

## 1. PURPOSE

This document defines how the AI coding agent should perform engineering work on the WEC project.

The objective is not to maximize autonomous activity.

The objective is to maximize:

* useful engineering output
* scientific correctness
* reproducibility
* validation
* project progress
* efficient use of local and cloud AI resources

The local model should perform the majority of implementation work.

Stronger cloud reasoning should be used selectively for difficult tasks.

---

# 2. DEFAULT OPERATING PRINCIPLE

The default workflow is:

```
UNDERSTAND
    ↓
INSPECT
    ↓
PLAN
    ↓
IMPLEMENT
    ↓
TEST
    ↓
DEBUG
    ↓
VALIDATE
    ↓
REVIEW
    ↓
DOCUMENT
    ↓
UPDATE PROJECT STATE
```

Never skip directly from:

```
USER REQUEST
```

to:

```
CODE
```

without understanding the repository context.

---

# 3. PHASE 1 — UNDERSTAND THE REQUEST

Before changing code, determine:

* what the user actually wants
* which module is involved
* which project tier it belongs to
* whether the task is implementation, debugging, research, experiment or documentation
* whether the task affects scientific methodology
* whether the task affects existing interfaces

Classify the task.

### A. Coding

Example:

"Implement a function to compute wave energy."

### B. Debugging

Example:

"PPO crashes after 40k steps."

### C. Scientific reasoning

Example:

"Is our forecast horizon physically appropriate?"

### D. Architecture

Example:

"How should confidence gating interact with MPC?"

### E. Research

Example:

"Find literature supporting this calibration method."

### F. Experiment

Example:

"Run the perfect vs realistic forecast comparison."

### G. Documentation

Example:

"Update the methodology."

The classification determines the appropriate workflow and model.

---

# 4. PHASE 2 — INSPECT

The agent should inspect only the repository context needed for the task, while expanding context when dependencies require it.

Recommended order:

1. `AGENTS.md`
2. `PROJECT_STATE.md`
3. relevant section of `ARCHITECTURE.md`
4. relevant section of `MODULE_MAP.md`
5. relevant section of `SCIENCE_RULES.md`
6. target source files
7. related tests
8. callers/dependents
9. configuration
10. recent Git changes when relevant

Do not load the entire repository unnecessarily.

---

# 5. PHASE 3 — LOCATE THE SOURCE OF TRUTH

Before implementing a feature:

Search for:

* existing function
* existing class
* existing configuration
* existing test
* related interface
* existing utility
* existing experiment
* documentation describing intended behaviour

Prefer modifying existing functionality over creating duplicate implementations.

---

# 6. PHASE 4 — FORM A PLAN

For a normal coding task:

Provide a concise plan.

Example:

```
1. Inspect current forecast interface.
2. Add uncertainty output without breaking point forecast.
3. Add calibration test.
4. Update controller input adapter.
5. Run forecasting tests.
6. Run controller integration test.
```

For a scientific/architectural task:

Include:

* assumptions
* affected modules
* alternatives considered
* scientific consequences
* validation strategy

---

# 7. PHASE 5 — DETERMINE MODEL ROUTING

Before beginning difficult reasoning, determine whether the task belongs to:

LOCAL

or:

CLOUD REVIEW / ESCALATION

Use `MODEL_ROUTING.md`.

Default to the local model.

Escalate only when the additional reasoning quality is likely to materially improve the result.

---

# 8. PHASE 6 — IMPLEMENT MINIMALLY

Prefer the smallest correct change.

Rules:

* do not rewrite unrelated modules
* do not introduce unnecessary dependencies
* do not duplicate logic
* preserve interfaces
* keep scientific parameters configurable
* preserve existing tests
* add tests with new functionality

Avoid speculative abstractions.

---

# 9. PHASE 7 — RUN THE CHEAPEST VALIDATION FIRST

Before expensive training or long simulation:

### Level 1

Import/syntax check.

### Level 2

Unit tests.

### Level 3

Smoke test.

### Level 4

Short integration run.

### Level 5

Small training run.

### Level 6

Full experiment.

Do not start Level 6 if Level 1–5 are failing.

---

# 10. DEBUGGING LOOP

When something fails:

```
FAILURE
   ↓
READ ERROR
   ↓
IDENTIFY FIRST REAL FAILURE
   ↓
TRACE DEPENDENCY
   ↓
FORM HYPOTHESIS
   ↓
MAKE MINIMAL FIX
   ↓
RE-RUN TARGETED TEST
   ↓
RUN REGRESSION TESTS
```

Do not repeatedly change unrelated code until the error disappears.

---

# 11. ERROR CLASSIFICATION

Classify failures before fixing them.

### Syntax

Python/import error.

### Configuration

Wrong parameter/path/environment.

### Interface

Input/output mismatch.

### Numerical

NaN, instability, convergence, overflow.

### Physics

Wrong equation/sign/unit/parameter.

### Data

Missing values, misalignment, leakage, provenance issue.

### ML

Model, tensor shape, loss, training instability.

### RL

Observation/action/reward/environment/training problem.

### Infrastructure

Ollama/OpenCode/GPU/memory/package problem.

The category determines where the investigation starts.

---

# 12. SCIENTIFIC DEBUGGING

When a numerical or physics-related test fails:

Do not immediately increase tolerances.

Investigate:

1. units
2. sign conventions
3. dimensions/shapes
4. initial conditions
5. boundary conditions
6. timestep
7. input data
8. numerical solver
9. physical parameters
10. algorithm implementation

Only relax a tolerance when there is a documented numerical reason.

---

# 13. MACHINE-LEARNING DEBUGGING

For ML problems inspect:

1. data shape
2. timestamps
3. train/validation/test split
4. normalization
5. target alignment
6. masking
7. model dimensions
8. loss
9. optimizer
10. learning rate
11. seed
12. evaluation code

Never conclude that "the model is bad" before checking the data pipeline.

---

# 14. RL DEBUGGING

For RL problems inspect in order:

1. environment reset
2. observation values
3. observation scaling
4. action validity
5. action interpretation by plant
6. reward values
7. reward scale
8. termination
9. constraints
10. episode length
11. environment determinism
12. algorithm compatibility
13. hyperparameters

Inspect actual trajectories.

Do not debug PPO/SAC/DQN only by looking at final reward.

---

# 15. EXPERIMENT WORKFLOW

Every meaningful experiment should have:

## Hypothesis

What are we testing?

## Variables

What changes?

## Controls

What stays fixed?

## Dataset

What data are used?

## Evaluation split

What is held out?

## Metric

What determines success?

## Expected interpretation

What would each outcome mean?

## Reproducibility

What configuration/version/seed is required?

---

# 16. ABLATION WORKFLOW

When testing a new method:

Do not compare only:

```
old system
vs.
new system
```

Prefer progressive ablations.

Example:

```
baseline
   ↓
+ forecast
   ↓
+ probabilistic forecast
   ↓
+ confidence
   ↓
+ confidence gating
   ↓
+ residual RL
```

The purpose is to identify which mechanism caused the improvement.

---

# 17. TRAINING WORKFLOW

For expensive model training:

### Stage 1 — Sanity

Tiny dataset / tiny environment / very short run.

### Stage 2 — Smoke

Small realistic configuration.

### Stage 3 — Stability

Long enough to detect divergence/instability.

### Stage 4 — Pilot

Representative training configuration.

### Stage 5 — Final

Full training configuration.

### Stage 6 — Held-out evaluation

Never use training data as final evidence.

---

# 18. RESOURCE-AWARE WORKFLOW

The local machine has 16 GB unified memory.

Therefore:

* avoid unnecessary simultaneous heavy workloads
* avoid loading huge AI models
* monitor memory during training
* unload OpenCode/local inference when large training is running
* use short tests before expensive runs

The goal is:

```
cheap local iteration
    +
expensive computation only when justified
```

---

# 19. LOCAL-FIRST DEVELOPMENT LOOP

For ordinary implementation:

```
USER REQUEST
      ↓
LOCAL MODEL
      ↓
   inspect
      ↓
    code
      ↓
    test
      ↓
   debug
      ↓
local validation
      ↓
    DONE
```

Do not send routine implementation tasks to the cloud without a reason.

---

# 20. CLOUD-ASSISTED DEVELOPMENT LOOP

For difficult reasoning:

```
USER REQUEST
      ↓
LOCAL MODEL
      ↓
detects difficulty
      ↓
OPENROUTER
      ↓
stronger reasoning/review
      ↓
actionable solution
      ↓
LOCAL MODEL
      ↓
implements
      ↓
tests
      ↓
validates
```

The cloud response should become an engineering input.

Do not permanently outsource the whole project to the cloud.

---

# 21. SENIOR-REVIEW WORKFLOW

A useful pattern is:

### Pass 1 — Local

Ask the local agent to solve the task.

### Pass 2 — Cloud review

Send only the relevant:

* problem statement
* architecture
* code
* error
* proposed solution

to a stronger model.

Ask:

"Review this implementation for correctness, hidden assumptions, scientific issues and failure modes."

### Pass 3 — Local

Give the local model the review.

Ask it to:

* evaluate the recommendations
* implement justified corrections
* run tests

### Pass 4 — User

For major scientific changes, present the final decision and evidence to the researcher.

---

# 22. DON'T USE CLOUD JUST BECAUSE THE TASK IS LONG

Length does not automatically mean difficulty.

A 2,000-line mechanical refactor may still be a local-model task.

A five-line equation change may require cloud reasoning.

Classify by:

```
reasoning difficulty
```

not:

```
code length.
```

---

# 23. CONTEXT MANAGEMENT

Give the model only the context it needs.

For ordinary tasks:

* relevant module
* relevant test
* project state
* architecture section

For architecture tasks:

* relevant modules
* interfaces
* current state

For scientific reasoning:

* equation
* assumptions
* relevant data
* experiment setup

Avoid unnecessary repository-wide context.

---

# 24. WHEN TO SEARCH THE WEB / LITERATURE

External research is appropriate when:

* a scientific claim needs verification
* a method is unfamiliar
* a parameter requires an authoritative source
* current documentation is required
* a paper/prior-art question exists

Do not use general model memory as the sole evidence for important scientific claims.

Record relevant sources.

---

# 25. WHEN TO TRUST LOCAL KNOWLEDGE

Local repository knowledge should dominate when the question is:

* where code lives
* what interface exists
* how the project currently works
* what tests expect
* what configuration is currently used
* what has already been implemented

Do not replace project-specific facts with generic best practices.

---

# 26. COMPLETION CHECK

Before declaring success, ask:

### Code

Does the implementation exist?

### Integration

Does it connect correctly?

### Tests

Did relevant tests pass?

### Science

Are assumptions valid/documented?

### Data

Is provenance/leakage correct?

### Reproducibility

Can this be rerun?

### Regression

Did existing behaviour remain intact?

### State

Was `PROJECT_STATE.md` updated?

---

# 27. FINAL RESPONSE FROM AGENT

For non-trivial work report:

## Summary

What was done.

## Files changed

Exact paths.

## Tests

Commands + results.

## Scientific implications

Relevant physical/data/ML considerations.

## Remaining risks

Anything unresolved.

## Recommended next step

One clear next action.

Do not use vague statements such as:

"Everything looks good."

Use evidence.

---

# 28. STOP CONDITIONS

Stop and escalate when:

* scientific assumptions conflict
* source data are ambiguous
* hidden future information may have leaked
* a physical equation must be changed
* a reward definition materially changes the experiment
* evaluation methodology changes
* evidence contradicts the current hypothesis
* destructive operations are necessary

Do not stop merely because the task is difficult.

Investigate first.

---

# 29. GOLDEN WORKFLOW

The preferred WEC AI engineering loop is:

```
REQUEST
   ↓
CLASSIFY
   ↓
READ PROJECT STATE
   ↓
INSPECT REPOSITORY
   ↓
UNDERSTAND INTERFACES
   ↓
PLAN
   ↓
ROUTE MODEL
   ↓
IMPLEMENT
   ↓
TEST
   ↓
DEBUG
   ↓
VALIDATE
   ↓
REVIEW
   ↓
UPDATE STATE
   ↓
REPORT
```

This workflow should become habitual.

---

# 30. FINAL PRINCIPLE

Use AI to increase engineering throughput.

Do not use AI to bypass scientific reasoning.

The best workflow is:

```
AI proposes
↓
AI implements
↓
tests check software
↓
experiments check behaviour
↓
evidence checks claims
↓
researcher makes the scientific decision
```
