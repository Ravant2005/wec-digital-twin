# Module 5.3A — RL Experiment Design Document

**WEC Digital Twin: AI-Forecasted, RL-Controlled Point-Absorber Buoy off the Goa Coast**

Status: DESIGN COMPLETE — DO NOT TRAIN UNTIL THIS DOCUMENT IS REVIEWED

---

## 1. Objective

Train three PPO policies that learn optimal WEC latching timing under three information conditions:

| Experiment | Forecast condition | Scientific purpose |
|-----------|-------------------|--------------------|
| Reactive PPO | No forecast (Block B = 0) | Baseline: what RL can do without prediction |
| Perfect-forecast PPO *(oracle)* | True future ERA5 values | Upper bound: theoretical max with perfect knowledge |
| Realistic-forecast PPO | GRU+wind + empirical errors | Real operating condition |

**Do not claim energy improvement until all three conditions are evaluated on held-out test data.**

---

## 2. RL State / Observation Vector

Verified against actual implementation (`module5_control/observations.py`).

### 2.1 Block A — Sensor history (obs[0:528])

48 hourly steps × 11 features = **528 values**

| Idx | Feature | Raw range | Preprocessing | RL range | Additional norm? |
|-----|---------|-----------|---------------|----------|-----------------|
| 0 | Hs_obs [m] | 0–5 m | z-score (µ=1.266, σ=0.766) | ~[-2, 5] | No — already scaled |
| 1 | Tp_obs [s] | 3–18 s | z-score (µ=9.631, σ=2.785) | ~[-3, 3] | No |
| 2 | sin(dir_obs) | [-1, 1] | Identity (σ=1.0) | [-1, 1] | No |
| 3 | cos(dir_obs) | [-1, 1] | Identity (σ=1.0) | [-1, 1] | No |
| 4 | Hs_valid | {0, 1} | Identity | {0, 1} | No |
| 5 | Tp_valid | {0, 1} | Identity | {0, 1} | No |
| 6 | dir_valid | {0, 1} | Identity | {0, 1} | No |
| 7 | u10 [m/s] | -15–20 m/s | z-score (µ=2.713, σ=3.277) | ~[-3, 5] | No |
| 8 | v10 [m/s] | -15–15 m/s | z-score (µ=-1.491, σ=3.088) | ~[-3, 5] | No |
| 9 | u10_valid | {0, 1} | Identity | {0, 1} | No |
| 10 | v10_valid | {0, 1} | Identity | {0, 1} | No |

NaN values: replaced by 0.0 via `np.nan_to_num` before entering observation vector. ✓

### 2.2 Block B — Forecast (obs[528:720])

48 forecast steps × 4 features = **192 values**

| Idx | Feature | Preprocessing | Mode="reactive" | Mode="perfect_forecast" | Mode="realistic_forecast" |
|-----|---------|---------------|----------------|------------------------|--------------------------|
| 0 | Hs_forecast [m] | z-score same as Block A | 0.0 | true future ERA5 | GRU+wind ± empirical error |
| 1 | Tp_forecast [s] | z-score same as Block A | 0.0 | true future ERA5 | GRU+wind ± empirical error |
| 2 | sin(dir_forecast) | Identity | 0.0 | true future ERA5 | GRU+wind ± empirical error |
| 3 | cos(dir_forecast) | Identity | 0.0 | true future ERA5 | GRU+wind ± empirical error |

**No future truth leakage in reactive and realistic modes — confirmed by code inspection.**

### 2.3 Block C — WEC state (obs[720:723])

| Idx | Feature | Raw range | Preprocessing | RL range |
|-----|---------|-----------|---------------|----------|
| 720 | x / 2.0 | [-6, 6] m | divide by X_SCALE=2.0 | ~[-3, 3] |
| 721 | v / 1.0 | [-2, 2] m/s | divide by V_SCALE=1.0 | ~[-2, 2] |
| 722 | latch_status | {0, 1} | float cast | {0.0, 1.0} |

No clipping applied. x >2 m possible in energetic sea states.

### 2.4 Summary

```
Total: 528 + 192 + 3 = 723 float32 values
Space: Box(shape=(723,), dtype=float32, low=-inf, high=+inf)
```

**VecNormalize obs normalisation must be OFF.** Features are already appropriately
scaled by the Module 4 scaler (Hs, Tp, u10, v10 z-scored) and physical scale factors.
Applying VecNormalize obs normalisation would double-normalise the already-scaled features,
corrupting the GRU-trained representation.

---

## 3. Action Space

```
spaces.Discrete(2)
  0 = RELEASE  — free Cummins + PTO dynamics
  1 = LATCH    — kinematic constraint: v=0, x=x_latch
```

### 3.1 Semantics (verified from environment.py)

| Scenario | Behaviour | Switch penalty? |
|----------|-----------|-----------------|
| RELEASE → LATCH | x_latch captured; v=0 for 10 physics sub-steps | Yes (once) |
| LATCH → RELEASE | Free dynamics resume from x_latch, v=0 | Yes (once) |
| LATCH → LATCH (hold) | v=0 maintained; x_latch unchanged | No |
| RELEASE → RELEASE (hold) | Free dynamics continue | No |
| LATCH then immediate RELEASE | Valid; radiation memory preserved | Yes (both transitions) |

**No action masking needed.** Redundant actions are no-ops with no switch penalty.

### 3.2 Physical constraints (no artificial enforcement)

- No minimum latch duration enforced by the environment.
- No dead time between actions.
- The switch penalty (λ=0.01) provides a soft cost against excessive switching.
- The agent can freely latch/release at any 1 Hz step.
- Compare with FixedThresholdController: timer holds latch for min_latch_steps=4 s.

---

## 4. Reward Formulation

### 4.1 Current implementation

```
r(t) = P_mean(t) / P_ref  −  λ_switch · I_switch  −  λ_end_stop · I_end_stop
```

| Parameter | Value | Notes |
|-----------|-------|-------|
| P_ref | 1,000 W | Reference power for normalisation |
| λ_switch | 0.01 | Per switch event penalty |
| λ_end_stop | 0.1 | Per step penalty when \|x\| > 3 m |
| x_end_stop | 3.0 m | End-stop threshold |

`P_mean` = mean of `B_PTO × v²` over 10 physics sub-steps per RL step. Always ≥ 0. ✓

### 4.2 Reward magnitude issue (CRITICAL for PPO)

From Module 5.2C benchmark (Hs=1.5 m, Tp=8 s, B_PTO=200 kN·s/m):

| Controller | Mean P (W) | Per-step reward (P_ref=1000) |
|-----------|-----------|------------------------------|
| Passive | 67,220 | **67.2** |
| Threshold | 46,524 | **46.5** |

Per-step rewards of O(50–70) are too large for PPO gradient stability.
Episode total reward over 7,200 steps ≈ 480,000 — far outside the range where
PPO's normalised advantage estimates work reliably.

**Recommended fix: use `VecNormalize(norm_reward=True, clip_reward=10.0)`**

This normalises rewards online without modifying frozen `RewardConfig`.
With VecNormalize active, the per-step reward is rescaled to approximately unit variance.
The value function learns on normalised rewards; the policy gradient uses the same scale.

Alternatively: set `P_ref = 50_000 W` in a training-specific `RewardConfig`.
Per-step passive reward → ~1.34 (within [0, 10] range suitable for PPO).

**The frozen RewardConfig (P_ref=1000) is correct as an energy unit; it just needs
VecNormalize wrapping for PPO training stability.**

---

## 5. Temporal Information Flow

```
ERA5 true sea state (1-hour grid)
    ↓  Hs_true, Tp_true, dir_true, u10, v10 per hour
Module 3 sensor model (adds noise, delay, missingness)
    ↓  once per ERA5 hour (every 3600 control steps)
obs_history rolling buffer: H_OBS=48 hourly rows × 11 features
    ↓  held constant between hourly sensor updates
GRU+wind model (mode=realistic/perfect only)
    ↓  once per ERA5 hour
Block B forecast: 48×4 — stays constant for 3600 control steps
    ↓
RL observation (723 floats): updated every control step
    ↓  Block A same for 3600 steps between updates; Block C updates each step
RL action at 1 Hz (DT_CONTROL = 1.0 s)
    ↓
WEC physics (10 sub-steps at DT_PHYSICS = 0.1 s)
    ↓
PTO power = B_PTO × v²  →  reward = P_mean / P_ref − penalties
```

### 5.1 Important timing properties

- **Block A and Block B are updated hourly** (every 3600 control steps).
- **Block C is updated every step** (x, v after each 10-sub-step Euler-Cromer advance).
- **The forecast is stale for 3600 steps** between updates. This is by design (ERA5 resolution).
- **Sensor delay** (SensorParameters.delay_steps=1) means the most recent ERA5 hour
  is not available until the next hour. The agent always sees lagged observations.

### 5.2 Leakage verification

| Data path | Future truth? | Verified clean? |
|-----------|--------------|-----------------|
| Block A (sensor history) | No — past sensor obs only | ✓ |
| Block B (reactive mode) | No — all zeros | ✓ |
| Block B (perfect_forecast) | **Yes — intentionally** | ✓ (oracle only) |
| Block B (realistic_forecast) | No — GRU+empirical errors | ✓ |
| info dict Hs_true | Yes — but NEVER put in obs | ✓ |
| Wind u10/v10 in Block A | No — ERA5 at current hour | ✓ |

**No unintentional future leakage detected in reactive or realistic modes.**

---

## 6. Three Forecast Conditions

### Experiment 1: Reactive PPO

- `mode="reactive"`
- Block B = zeros (48×4 zeros)
- GRU model not loaded
- ForecastErrorSampler not used
- Controller sees: 48h sensor history + WEC state only
- Scientific question: "What can RL achieve without any forecast?"

### Experiment 2: Perfect-Forecast PPO *(ORACLE UPPER BOUND)*

- `mode="perfect_forecast"`
- Block B = true future ERA5 Hs/Tp/sin(dir)/cos(dir) for next 48h
- GRU model not needed
- **Future truth is explicitly used — this is the oracle condition**
- **Must NOT claim this as a deployable result**
- Requires replay with ≥ 98 hours (48 lookback + 2 episode + 48 forecast)
- Scientific question: "How much energy could a perfect predictor unlock?"

### Experiment 3: Realistic-Forecast PPO

- `mode="realistic_forecast"`
- Block B = GRU+wind prediction + Module 5.2B empirical error trajectory
- GRU checkpoint: `results/forecasting/exp_gru_wind/best_model.pt` (frozen)
- ForecastErrorSampler: **must use training-period error bank (see leakage section)**
- No current-episode future truth enters the observation
- Scientific question: "What does RL achieve under realistic forecast uncertainty?"

---

## 7. Data Split

```
RL TRAINING episodes:   ERA5 2010–2021 (12 years, ~105,120 hourly records)
RL VALIDATION episodes: ERA5 2022–2023 (2 years, ~17,520 hourly records)
RL FINAL TEST:          ERA5 2024–2025 — HOLD OUT — touch only for paper results
```

### 7.1 Episode sampling strategy for training

1. Load ERA5 archive for years 2010–2021 into a DataFrame.
2. At each training episode, sample a **random start index** (uniformly from valid
   positions leaving at least `min_replay_hours` of headroom).
3. Build `EpisodeReplay(n_hours=98)` from that start index.
4. Run the episode for 2 hours from the 48th hour (giving GRU a full 48h lookback).
5. Reset with a new seed for the next episode.

Stochastic start index gives the agent diverse sea-state exposure across the training period.

### 7.2 Episode sampling for evaluation

Use a fixed set of evaluation episodes drawn from 2022–2023 validation data
with deterministic seeds. Do not evaluate on 2024–2025 until the policy is frozen.

---

## 8. Forecast Error Leakage Audit

### CURRENT ISSUE

`results/forecasting/exp_gru_wind/forecast_errors.csv.gz` contains GRU forecast
errors evaluated on the **2024–2025 test period** (origin_idx 122,759–140,207,
confirmed by timestamp calculation).

### WHY IT MATTERS

If `ForecastErrorSampler` uses this bank during RL **training**, we are injecting
test-period error statistics into the training process:

1. The error distribution was computed on held-out 2024–2025 data that the GRU
   never saw during training (correct for evaluating GRU quality).
2. But the RL agent's training distribution would then be shaped by knowledge of
   how the GRU performs on future (2024–2025) data.
3. This does not directly expose future sea-state values to the agent — the errors
   are statistics, not truth values. However, it contaminates the training distribution
   with information derived from the test period.
4. Concretely: if the 2024–2025 period has different sea-state characteristics from
   2010–2021 (e.g., a different storm season), the error statistics may reflect that,
   biasing the training uncertainty profile.
5. For the Tier 1 claim ("robust under realistic forecast uncertainty"), this is a
   material methodological concern.

### RECOMMENDED SOLUTION

**Derive a separate RL training error bank from the 2022–2023 validation period:**

1. Run the frozen GRU+wind model on the 2022–2023 ERA5 data.
2. Collect (Hs_pred, Hs_true, Hs_error, Tp_error, dir_error) for all forecast origins.
3. Store in the same format as `forecast_errors.csv.gz`.
4. Use this bank for `ForecastErrorSampler` during RL training.
5. Retain the existing 2024–2025 bank **exclusively** for final evaluation.

This matches the Module 4 data split: the validation period evaluates the forecaster's
performance; the same period can provide the uncertainty model for RL training without
touching test data.

### WHEN TO IMPLEMENT

**Before Module 5.3B (actual PPO training begins).**

The Module 5.2B `ForecastErrorSampler` implementation requires no changes —
only the CSV path needs to point to the new bank.

`ForecastErrorSampler.from_csv(csv_path="<new_bank_path>")` — that is the entire change.

### IMPACT ON CURRENT MODULES

- Module 5.2B code: **unchanged** — no modification to frozen implementation.
- `rl_config.py`: documents the pending bank and the correct path.
- Module 5.3B (future): will use the new bank path for training.

---

## 9. Episode and Training Design

### 9.1 Episode length analysis

| Duration | Steps | Notes |
|---------|-------|-------|
| 1 hour | 3600 | Too short: only 1 GRU forecast update (at hour boundary) |
| **2 hours** | **7200** | **Current benchmark. 1 forecast update mid-episode. Practical.** |
| 4 hours | 14,400 | More forecast updates; slower episode turnover |
| 24 hours | 86,400 | Full-day; very slow training; impractical for early experiments |

**Recommendation: keep 2-hour episodes for Tier 1 training.**

Rationale:
- Consistent with Module 5.2C benchmark — direct comparison possible.
- Fast enough for efficient PPO rollout collection.
- Long enough to include 10–20 latch/release cycles per episode.
- The 48h GRU forecast is available from the first step; the agent can plan ahead.
- Extend to 4 or 6 hours for later experiments if longer planning horizon matters.

### 9.2 PPO rollout design

```
n_steps      = 3600  (one ERA5 hour per rollout buffer)
n_envs       = 4     (parallel environments, different sea-state seeds)
batch_size   = 256
Rollout size = n_steps × n_envs = 14,400 transitions per update
n_minibatches = 14400 / 256 = 56.25 → round to 56
```

With 5M total timesteps: 5,000,000 / 14,400 ≈ 347 gradient updates.

### 9.3 Training curriculum (optional, Tier 2)

Not required for Tier 1. If policy diverges:
1. Start with reactive mode (simpler problem).
2. Pre-train reactive policy.
3. Fine-tune with realistic forecast by loading reactive weights and continuing.

---

## 10. PPO Architecture Recommendation

```python
model = PPO(
    policy          = "MlpPolicy",
    env             = VecNormalize(make_vec_env(..., n_envs=4), norm_obs=False, norm_reward=True, clip_reward=10.0),
    learning_rate   = linear_schedule(3e-4),   # decay to 0 over training
    n_steps         = 3600,
    batch_size      = 256,
    n_epochs        = 10,
    gamma           = 0.995,
    gae_lambda      = 0.95,
    clip_range      = 0.2,
    clip_range_vf   = None,
    ent_coef        = 0.01,
    vf_coef         = 0.5,
    max_grad_norm   = 0.5,
    policy_kwargs   = dict(net_arch=[256, 256, 128], activation_fn=nn.Tanh),
    seed            = 42,
    verbose         = 1,
)
```

### 10.1 Parameter classification

| Parameter | Value | Category | Notes |
|-----------|-------|----------|-------|
| action_space | Discrete(2) | **P** (physics) | Binary latch/release |
| obs_space | Box(723,) | **P** (physics) | Fixed by Module 5.1 |
| n_steps | 3600 | **D** | = 1 ERA5 hour; sensible default |
| batch_size | 256 | **D** | Standard PPO |
| gamma | 0.995 | **D** | Long episodes; high discount |
| gae_lambda | 0.95 | **D** | Standard |
| clip_range | 0.2 | **D** | Standard PPO |
| learning_rate | 3e-4 | **T** (tunable) | Linear decay recommended |
| ent_coef | 0.01 | **T** | Explore latch timing |
| net_arch | [256,256,128] | **T** | Moderate depth |
| activation | Tanh | **D** | Better than ReLU for PPO |
| total_timesteps | 5M | **T** | Starting budget |

**P = fixed by physics/problem; D = reasonable default; T = tuning candidate**

---

## 11. Normalisation Analysis

### 11.1 VecNormalize recommendation

Use `VecNormalize(norm_obs=False, norm_reward=True, clip_reward=10.0)`.

**Observation normalisation: OFF**
Features are already appropriately scaled:
- Block A Hs, Tp, u10, v10: z-scored by Module 4 scaler → approximately N(0,1)
- Block A sin/cos direction, validity: in [-1, 1] or {0, 1}
- Block B: same scaling as Block A (via `_normalise_forecast`)
- Block C x: scaled by 1/X_SCALE=0.5 → typically [-1, 1]
- Block C v: scaled by 1/V_SCALE=1.0 → typically [-2, 2]
- Block C latch: {0.0, 1.0}

Applying additional VecNormalize obs normalisation would corrupt the z-scored features.

**Reward normalisation: ON**
With `P_ref=1000 W`, per-step rewards are O(50–70) — too large for PPO.
`VecNormalize(norm_reward=True, clip_reward=10.0)` normalises adaptively.
Clip at 10.0 prevents outlier steps from dominating the value estimate.

### 11.2 Feature range table (post-preprocessing)

| Block | Feature | RL range | Double-norm risk? |
|-------|---------|---------|------------------|
| A[0] | Hs_obs (z-scored) | ~[-2, 5] | No — scaler applied |
| A[1] | Tp_obs (z-scored) | ~[-3, 3] | No |
| A[2,3] | sin/cos dir | [-1, 1] | No |
| A[4,5,6] | validity | {0, 1} | No |
| A[7,8] | u10/v10 (z-scored) | ~[-3, 5] | No |
| A[9,10] | wind validity | {0, 1} | No |
| B[0] | Hs forecast (z-scored) | ~[-2, 5] | No |
| B[1] | Tp forecast (z-scored) | ~[-3, 3] | No |
| B[2,3] | sin/cos dir forecast | [-1, 1] | No |
| C[0] | x/2.0 | ~[-3, 3] | No |
| C[1] | v/1.0 | ~[-2, 2] | No |
| C[2] | latch_status | {0, 1} | No |

---

## 12. Experiment Matrix

```
                        ┌────────────────────────────────────────────┐
                        │           FORECAST CONDITION               │
                        ├──────────┬─────────────────┬──────────────┤
CONTROLLER              │ Reactive │ Perfect (oracle) │  Realistic   │
────────────────────────┼──────────┼─────────────────┼──────────────┤
Passive (Baseline 1)    │    ✓     │        ✓        │      ✓       │
Fixed-threshold (B2)    │    ✓     │        ✓        │      ✓       │
Reactive RL (Exp 1)     │    ✓     │        -        │      -       │
Perfect-forecast RL(Exp2│    -     │        ✓        │      -       │ (oracle)
Realistic RL (Exp 3)    │    ✓     │        -        │      ✓       │
────────────────────────┴──────────┴─────────────────┴──────────────┘
```

Primary results: reactive RL vs passive vs threshold (reactive column)
Oracle ceiling: perfect-forecast RL vs passive (perfect column)
Deployable result: realistic RL vs passive vs threshold (realistic column)

**The three-condition matrix from the research plan (Experiments C, E, H from ablation ladder).**

---

## 13. Evaluation Metrics

Per episode, record:

| Metric | Unit | Source |
|--------|------|--------|
| total_energy_Wh | Wh | PowerAccumulator |
| mean_power_W | W | total_energy_Wh / episode_hours |
| n_latch_events | count | BenchmarkHarness |
| n_release_events | count | BenchmarkHarness |
| latch_fraction | — | latch_steps / n_steps |
| max_abs_displacement_m | m | from info["x"] |
| max_abs_velocity_ms | m/s | from info["v"] |
| max_abs_acceleration_ms2 | m/s² | |∆v| / DT_CONTROL (coarse) |
| total_reward | — | sum of step rewards |
| n_end_stop_violations | count | \|x\| > 3 m |
| is_finite | bool | no NaN/Inf |

Aggregate over 7 benchmark segments and 5 evaluation seeds:
mean, median, std, min, max energy.

---

## 14. Deterministic Evaluation Protocol

1. Freeze the policy (no more gradient updates).
2. For each benchmark segment and each forecast condition:
   a. Build `EpisodeReplay` with seed=eval_seed.
   b. Reset env with seed=eval_seed.
   c. Run policy in deterministic mode (set `deterministic=True` in SB3).
   d. Record all metrics.
3. Repeat for all 5 evaluation seeds.
4. Report mean ± std across seeds.
5. Final test results (2024–2025 data): computed once, after policy is frozen.

---

## 15. Computational Considerations

- Cummins BEM computation: ~15 s (one-time cost; cache results).
- Episode simulation at 7200 control steps: ~4 s wall clock (from benchmark).
- With n_envs=4 parallel: ~1 s per rollout collection (4 envs × 3600 steps ÷ ~4 s/ep).
- PPO update: < 1 s for 14,400 transitions.
- 5M timestep training: ~5M / (4 × 3600) ≈ 347 updates × ~5 s/update = ~29 minutes.
- realistic_forecast with GRU inference: +1–2 s per episode (GRU forward pass per hour).
  With n_envs=4: dominated by parallelism, approximately 2× slower than reactive.
- Total estimated training time (all three experiments): ~2–3 hours on CPU.

**No GPU required for Tier 1 PPO training.** The bottleneck is WEC physics simulation,
not neural network training. The GRU is run in inference mode (no gradient).

---

## 16. Known Limitations (Tier 1 scope)

See `docs/research_limitations.md` for the full tracker.

Critical items for PPO training:

- **Reward scaling**: P_ref=1000 W gives O(50) rewards. Use VecNormalize. *(ORANGE)*
- **Forecast error bank leakage**: current bank from 2024–2025 test period. *(RED — resolve before training)*
- **Development BEM geometry**: cylinder, not the actual Goa deployment buoy. *(ORANGE)*
- **Synthetic sea states for benchmark**: threshold comparisons use constant Hs/Tp. *(YELLOW)*
- **No trained policy exploiting Block B**: oracle/realistic interfaces use threshold rule. *(This is what Exp 2/3 will fix)*

---

## 17. Files Created

| File | Purpose |
|------|---------|
| `module5_control/rl_config.py` | Declarative PPO experiment configurations |
| `docs/module5_3a_rl_design.md` | This design document |
| `docs/research_limitations.md` | Full limitation tracker |
| `tests/test_module5_rl_design.py` | Design validation tests (no training) |

---

*Document prepared: Module 5.3A design-only phase. No training has occurred.*
*Next step: Module 5.3B — generate RL training error bank, implement training loop, train.*
