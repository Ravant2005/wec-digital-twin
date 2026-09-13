# MODEL ROUTING

## 1. PURPOSE

This document defines how AI model resources should be allocated during WEC development.

The system uses two classes of AI:

### Local

Primary engineering model running through Ollama.

### Cloud

Optional stronger models accessed through OpenRouter.

The objective is to maximize project capability while minimizing cloud usage.

---

# 2. CORE POLICY

The local model is the DEFAULT.

OpenRouter is an ESCALATION RESOURCE.

Do not route every request to the strongest cloud model.

The intended architecture is:

```
LOCAL MODEL
    ↓
primary builder
    ↓
tests / validation
    ↓
cloud only when justified
```

The local model should perform most repository work.

---

# 3. WHAT THE LOCAL MODEL SHOULD HANDLE

Use the local model by default for:

## Repository work

* file discovery
* repository navigation
* code search
* dependency tracing
* reading project context
* inspecting Git diffs

## Standard coding

* Python
* NumPy
* SciPy
* pandas/xarray
* PyTorch
* Gymnasium
* Stable-Baselines3
* utility functions
* configuration
* logging
* test creation

## Testing

* pytest
* smoke tests
* regression tests
* unit tests
* simple integration tests

## Debugging

* traceback diagnosis
* shape mismatch
* missing imports
* file path errors
* ordinary API mistakes
* configuration errors
* straightforward numerical bugs

## Documentation

* docstrings
* README updates
* experiment notes
* module documentation
* state updates

## Refactoring

* localized refactoring
* duplication removal
* naming cleanup
* test-driven changes
* interface-preserving improvements

---

# 4. WHAT SHOULD TRIGGER CLOUD CONSIDERATION

Consider OpenRouter when the task involves unusually difficult:

### Scientific reasoning

* complex physical interpretation
* competing physical formulations
* non-obvious numerical modelling decisions

### Mathematical reasoning

* difficult derivations
* advanced optimization
* control-theoretic derivation
* uncertainty/calibration theory

### Architecture

* major cross-module redesign
* uncertain interface design
* architecture tradeoffs with long-term consequences

### Debugging

* persistent failure after reasonable local investigation
* non-obvious numerical instability
* complex multi-module failure
* contradictory test/evaluation evidence

### Research

* deep literature synthesis
* comparison of advanced methods
* difficult prior-art reasoning
* research design requiring broad synthesis

### Review

* important scientific design review
* reviewer-style critique
* second opinion before a major change

---

# 5. AUTOMATIC ESCALATION SIGNALS

The agent should consider escalation when one or more of these occur:

1. local model has attempted a reasonable solution and tests still fail
2. the task requires multiple competing scientific interpretations
3. a mathematical derivation is central to the decision
4. the task could materially affect a research claim
5. the local model repeatedly proposes inconsistent solutions
6. the task requires broad literature synthesis
7. the task involves a major architectural decision
8. the same issue remains unresolved after two or three focused iterations
9. the model cannot confidently identify the source of a numerical failure
10. a second independent review would materially reduce risk

Escalation is a judgment call.

Do not escalate merely because a task is long.

---

# 6. DO NOT ESCALATE FOR

Do not waste cloud requests on:

* simple CRUD/file changes
* basic Python functions
* straightforward tests
* formatting
* ordinary refactoring
* reading repository files
* running commands
* simple tracebacks
* project-state maintenance
* obvious dependency fixes
* repetitive implementation

---

# 7. THREE-LEVEL ROUTING

Use three levels.

## LEVEL 1 — LOCAL

Default.

```
Ollama
   ↓
local coding model
```

Use for normal implementation.

---

## LEVEL 2 — LOCAL + CLOUD REVIEW

Use when local implementation is possible but confidence is low.

Workflow:

```
local model
   ↓
draft solution
   ↓
OpenRouter review
   ↓
local model evaluates review
   ↓
local implementation/test
```

This is the preferred escalation mode.

---

## LEVEL 3 — CLOUD-FIRST REASONING

Use only for high-complexity reasoning where the local model is unlikely to produce a reliable plan.

Examples:

* advanced control architecture
* difficult mathematical derivation
* major scientific redesign
* complex research synthesis

Workflow:

```
problem
  ↓
OpenRouter reasoning
  ↓
proposed architecture
  ↓
local implementation
  ↓
tests
  ↓
validation
  ↓
user scientific review
```

Cloud generates reasoning.

Local still performs repository-native implementation wherever practical.

---

# 8. TASK ROUTING TABLE

| Task                                       | Default                         |
| ------------------------------------------ | ------------------------------- |
| Find a function                            | Local                           |
| Write pytest                               | Local                           |
| Fix import error                           | Local                           |
| Refactor a function                        | Local                           |
| Build a data pipeline                      | Local                           |
| Implement a PyTorch module                 | Local                           |
| Debug tensor shape                         | Local                           |
| Run experiment                             | Local                           |
| Analyze simple test failure                | Local                           |
| Write documentation                        | Local                           |
| Design new reward                          | Local → Cloud review if complex |
| Diagnose numerical instability             | Local → Cloud if unresolved     |
| Modify physical equation                   | Cloud review recommended        |
| Design confidence gate                     | Cloud review recommended        |
| Derive advanced MPC formulation            | Cloud                           |
| Compare advanced forecasting architectures | Cloud review                    |
| Major architecture redesign                | Cloud review                    |
| Deep literature synthesis                  | Cloud                           |
| Prior-art analysis                         | Cloud + external research       |
| Final scientific interpretation            | Human review                    |

---

# 9. TOKEN / REQUEST ECONOMY

Cloud requests are a scarce resource.

Therefore:

### Do not send unnecessary context.

Send:

* exact problem
* relevant code
* relevant architecture
* relevant error
* relevant science rules
* proposed local solution

Do not send:

* entire unrelated repository
* irrelevant logs
* duplicate files
* large raw datasets when summaries suffice

---

# 10. CLOUD REQUEST TEMPLATE

When escalating, formulate the task clearly.

Use:

## Problem

What is wrong?

## Context

Which module and tier?

## Current implementation

Relevant code/design.

## Evidence

Tests, traceback, metrics, plots or observations.

## Local attempt

What has already been tried?

## Scientific constraints

What cannot be changed?

## Question

What exact reasoning is required?

Example:

```
We are implementing Tier-2 confidence-gated control.

The local implementation produces a confidence signal
but the gate becomes unstable when confidence changes rapidly.

Current behaviour:
...

Constraints:
- preserve Tier-1 fallback
- do not modify WEC physics
- confidence must remain calibrated
- controller must remain causal

Review the architecture and propose the smallest
scientifically defensible stabilisation strategy.
```

---

# 11. CLOUD RESPONSE HANDLING

Never blindly copy a cloud response into the repository.

The local agent must:

1. understand the recommendation
2. compare it against project architecture
3. verify assumptions
4. identify conflicts
5. implement only justified changes
6. run tests
7. validate experimentally

The cloud model is a consultant, not the final authority.

---

# 12. TWO-MODEL CRITIQUE

For important changes, use:

### Builder

Local model proposes and implements.

### Reviewer

Cloud model critiques.

The reviewer should inspect:

* correctness
* assumptions
* edge cases
* scientific risks
* maintainability
* test coverage

Then the local model incorporates justified feedback.

---

# 13. MODEL DISAGREEMENT

If local and cloud models disagree:

Do NOT automatically choose the larger model.

Instead:

1. identify the exact disagreement
2. state each interpretation
3. inspect project constraints
4. check authoritative scientific sources where required
5. build the smallest discriminating test
6. measure
7. choose based on evidence

---

# 14. SCIENTIFIC DECISION ROUTING

For a scientifically consequential decision:

```
LOCAL
  ↓
INITIAL ANALYSIS
  ↓
CLOUD REVIEW
  ↓
SOURCE / LITERATURE CHECK
  ↓
EXPERIMENT
  ↓
HUMAN DECISION
```

Do not allow a model's confidence to substitute for evidence.

---

# 15. EXPERIMENT ROUTING

Routine experiment execution:

LOCAL.

Experiment design involving a major hypothesis:

LOCAL + CLOUD REVIEW.

Final interpretation:

HUMAN REVIEW.

For example:

### Routine

"Run the three PPO information modes."

→ Local.

### Difficult

"Design the most defensible experiment to prove confidence gating itself—not merely probabilistic forecasting—causes an improvement."

→ Cloud review recommended.

### Final

"Does this result establish our research claim?"

→ Human scientific decision.

---

# 16. DATA ROUTING

Local model may handle:

* data loading code
* transformations
* cleaning
* feature engineering
* standard evaluation
* plotting

Cloud review may help with:

* unusual data-quality problems
* missing-data methodology
* leakage risk analysis
* statistical methodology
* sophisticated uncertainty interpretation

Large raw datasets should generally remain local or in the project's data environment.

Do not send entire raw datasets to the cloud just because the cloud model is stronger.

---

# 17. MODEL ROUTING FOR THE WEC TIERS

## Tier 1

Primarily local.

Cloud only for difficult blockers.

## Tier 2

Local implementation + regular cloud architectural review.

Because confidence calibration and predictive control are scientifically consequential.

## Tier 3

Local implementation + cloud review.

## Tier 4

Local experimentation + cloud review of robustness methodology.

## Tier 5

More cloud reasoning may be justified because the architecture becomes substantially more complex.

---

# 18. DO NOT MAKE CLOUD A HARD DEPENDENCY

The project must remain functional if:

* OpenRouter is unavailable
* free cloud quota is exhausted
* network is unavailable
* API configuration breaks

The local workflow must remain usable.

Cloud is an enhancement, not infrastructure required for basic development.

---

# 19. MODEL SWITCHING SAFETY

When changing models:

* preserve project context
* preserve task specification
* preserve constraints
* preserve test expectations
* do not assume another model understands previous hidden reasoning

The new model must receive explicit context.

---

# 20. API SECURITY

OpenRouter credentials must:

* remain outside Git
* remain outside source code
* remain outside project documentation
* be supplied through secure environment/configuration mechanisms

Never print API keys into:

* terminal logs
* model prompts
* code
* commits
* documentation
* issue trackers

---

# 21. MODEL CONTEXT PRIORITY

When a model receives context, prioritize:

1. user request
2. active project state
3. science rules
4. architecture
5. module contract
6. relevant implementation
7. relevant tests
8. evidence/logs
9. optional general context

Do not overwhelm the model with irrelevant information.

---

# 22. MODEL ROUTING DECISION TREE

Use this decision process:

```
Is the task routine implementation?
         |
        YES
         ↓
       LOCAL

NO
 |
 v
Is the task scientifically consequential?
         |
        NO
         ↓
       LOCAL

YES
 |
 v
Can the local model solve it reliably?
         |
     YES | NO
         | 
         v
  LOCAL + REVIEW
         |
         v
   Need deeper reasoning?
         |
        YES
         ↓
      OPENROUTER
```

After cloud reasoning:

```
         ↓
  LOCAL IMPLEMENTATION
         ↓
       TEST
         ↓
     VALIDATE
         ↓
   HUMAN REVIEW
when scientifically important
```

---

# 23. DEFAULT BUDGET STRATEGY

Think of cloud usage as a limited research resource.

Spend it on:

```
architecture
hard debugging
science
literature
review
```

Not on:

```
typing code
ordinary tests
simple refactoring
repetitive changes
```

The local model should absorb the bulk of token-equivalent work through free local inference.

---

# 24. SUCCESS CONDITION

The routing system is successful when:

* most routine engineering occurs locally
* difficult reasoning gets stronger review
* cloud requests remain limited
* local implementation remains the main workflow
* the project does not depend on cloud availability
* scientific decisions receive appropriate review
* model choice does not compromise reproducibility

---

# 25. GOLDEN RULE

Use:

```
LOCAL AI for WORK
```

and:

```
CLOUD AI for THINKING WHEN NEEDED
```

but never interpret that literally as "local cannot think" or "cloud must implement."

The real principle is:

```
Local = default engineering engine

Cloud = scarce high-reasoning consultant

Tests = software judge

Experiments = scientific judge

Human researcher = final authority
```
