# Research Limitations Tracker

**WEC Digital Twin: AI-Forecasted, RL-Controlled Point-Absorber Buoy — Goa Coast**

Last updated: Module 5.3B — PPO training infrastructure & smoke test.

Priority levels:
- 🔴 RED: Must solve before research-grade claims.
- 🟠 ORANGE: Should solve before final paper/report.
- 🟡 YELLOW: Robustness/improvement.
- 🟢 GREEN: Future work / Tier 5.

---

## Physics and Hydrodynamics

### L-PHY-01: Development cylinder geometry
- **Current state**: Reference hull is a vertical cylinder (radius=5 m, draft=10 m). Not the actual Goa deployment buoy.
- **Scientific impact**: All energy numbers are geometry-specific. Cannot claim results apply to a different hull without re-running BEM (Capytaine) with the real buoy mesh.
- **Priority**: 🟠 ORANGE
- **Resolution**: Replace with validated buoy mesh once hardware specifications are confirmed. Re-run `compute_hydrodynamic_coefficients()`.
- **Module/stage**: Module 2 upgrade, before final paper.
- **Blocks Tier 1?**: No (Tier 1 claims are geometry-conditional).
- **Blocks final claims?**: Yes — must state geometry clearly.

### L-PHY-02: BEM radiation kernel uncertainty
- **Current state**: `A_infinity` estimated via Ogilvie mid-range method; BEM frequency range [0.2, 1.4] rad/s does not reach the high-frequency asymptote.
- **Scientific impact**: Radiation memory kernel is approximate. Affects transient energy and oscillation decay rate.
- **Priority**: 🟠 ORANGE
- **Resolution**: Extend BEM frequency range to [0.2, 3.0] rad/s; use Capytaine mesh with higher panel count.
- **Module/stage**: Module 2 upgrade.
- **Blocks Tier 1?**: No.
- **Blocks final claims?**: Should be stated as a limitation.

### L-PHY-03: A_infinity is a development estimate
- **Current state**: `added_mass_infinity` is estimated from Ogilvie relation at mid-range BEM frequencies. Documented as "development estimate" in CumminsParameters.
- **Scientific impact**: Affects effective inertia and natural frequency. Ogilvie estimate may be off by 10–20% due to BEM frequency truncation.
- **Priority**: 🟡 YELLOW
- **Resolution**: Extend BEM frequency range (see L-PHY-02).
- **Blocks Tier 1?**: No.

### L-PHY-04: Water depth 30 m is a development assumption
- **Current state**: `DEPTH_GOA_M = 30.0 m` assumed for the Goa shelf. Not validated against bathymetry.
- **Scientific impact**: Affects wave dispersion relation and wave number at each frequency component.
- **Priority**: 🟡 YELLOW
- **Resolution**: Use INCOIS/GEBCO bathymetry for the exact deployment coordinates.
- **Blocks Tier 1?**: No.

### L-PHY-05: JONSWAP gamma = 3.3 (not Goa-calibrated)
- **Current state**: Default North Sea gamma used throughout wave synthesis.
- **Scientific impact**: Spectral shape and peak enhancement affect excitation force distribution.
- **Priority**: 🟡 YELLOW
- **Resolution**: Estimate gamma from ERA5 Goa data using spectral fitting.
- **Blocks Tier 1?**: No (stated as development default throughout documentation).

### L-PHY-06: BEM excitation force coverage only 92.3% of spectral energy
- **Current state**: BEM covers omega=[0.2, 1.4] rad/s. Wave components above 1.4 rad/s receive H_exc=0 (conservative). ~7.2% of spectral energy (Tp=8s) excluded.
- **Scientific impact**: Excitation force slightly underestimated for high-frequency components. Conservative (never overestimates).
- **Priority**: 🟡 YELLOW
- **Resolution**: Extend BEM frequency range (see L-PHY-02).
- **Blocks Tier 1?**: No.

---

## Sensing and Observation

### L-SEN-01: Synthetic camera noise (not validated against real imagery)
- **Current state**: Module 3 uses Gaussian noise model (σ_Hs=0.1 m, σ_Tp=0.5 s, σ_dir=10°) from published stereo-vision accuracy estimates.
- **Scientific impact**: Noise model is i.i.d. Gaussian; real camera errors are correlated and environment-dependent.
- **Priority**: 🟠 ORANGE
- **Resolution**: Calibrate against real buoy-camera measurements when available.
- **Blocks Tier 1?**: No — stated as synthetic proxy.

### L-SEN-02: 1-hour sensor update (ERA5 resolution, not sub-hourly)
- **Current state**: The sensor observation buffer updates once per ERA5 hour. A real camera system would update at sub-second rates.
- **Scientific impact**: The RL agent sees stale Hs/Tp/direction for 3,600 control steps (1 hour). Within-hour wave variability is not captured.
- **Priority**: 🟡 YELLOW
- **Resolution**: Sub-hourly ERA5 (30 min) or in-situ buoy data if available.
- **Blocks Tier 1?**: No — consistent with ERA5 temporal resolution.

---

## Forecasting

### L-FCT-01: Forecast error bank from 2024–2025 test period (LEAKAGE RISK)
- **Current state**: ✅ RESOLVED for training. `forecast_errors_val2022_2023.csv.gz` (2022–2023 validation-period bank) now exists and is the ONLY bank permitted during PPO training and validation. `assert_safe_training_error_bank()` in `rl_config.py` hard-guards against accidental use of the 2024–2025 final-test bank (`forecast_errors.csv.gz`). Final-test bank remains reserved exclusively for frozen-policy final evaluation.
- **Scientific impact**: Methodological leakage avoided. Realistic-forecast RL experiment is now publication-quality.
- **Priority**: ✅ RESOLVED (was 🔴 RED)
- **Resolution**: Generated training-safe error bank from 2022–2023 validation period. Added `assert_safe_training_error_bank()` safety guard. `env_factory.build_forecast_error_sampler()` defaults to the 2022–2023 bank and enforces the guard unless `allow_final_test_bank=True` (final eval only).
- **Module/stage**: Module 5.3B complete.
- **Blocks Tier 1?**: ✅ No — resolved.

### L-FCT-02: Forecast error bank not conditioned on sea-state severity
- **Current state**: `ForecastErrorSampler` draws uniformly from all trajectories regardless of current Hs/Tp conditions.
- **Scientific impact**: Error magnitudes should vary with sea-state severity (larger Hs → larger absolute errors). Uniform draw may under-represent or over-represent errors for extreme conditions.
- **Priority**: 🟠 ORANGE
- **Resolution**: Bin trajectories by Hs class and condition sampler on current sea state.
- **Module/stage**: Module 5.2B upgrade or Module 5.3 extension.
- **Blocks Tier 1?**: No.

### L-FCT-03: Deterministic GRU in realistic_forecast (no conformal calibration)
- **Current state**: Module 5.2B uses empirical error resampling (trajectory draw). No conformal prediction intervals or quantile heads on the GRU.
- **Scientific impact**: The agent receives a point estimate + error sample, not a calibrated probability distribution.
- **Priority**: 🟠 ORANGE (required for Tier 2 confidence-gated control)
- **Resolution**: Add quantile/pinball heads or conformal calibration (Module 4.3 / Tier 2).
- **Blocks Tier 1?**: No.

### L-FCT-04: Single-site forecaster (Goa only)
- **Current state**: GRU trained on Goa ERA5 data only.
- **Priority**: 🟢 GREEN (Tier 5 generalization)

### L-FCT-05: No spatial advection in forecaster
- **Priority**: 🟢 GREEN (Tier 5)

### L-FCT-06: No physics-informed forecast loss
- **Priority**: 🟢 GREEN (Tier 5)

### L-FCT-07: No Transformer decoder (LSTM baseline only)
- **Priority**: 🟢 GREEN (Tier 5)

---

## RL and Control

### L-RL-01: Reward scaling (P_ref=1000 W gives O(50) per-step rewards)
- **Current state**: ✅ RESOLVED for PPO training. Per-step rewards ~50–70 for the reference buoy/PTO configuration are normalised via `VecNormalize(norm_reward=True, clip_reward=10.0)` in the SB3 training wrapper. Frozen `RewardConfig` is unchanged (P_ref=1000 W preserved). During evaluation, VecNormalize is loaded with `training=False, norm_reward=False` so validation/test metrics use the original raw reward scale.
- **Scientific impact**: PPO gradient stability guaranteed — value function no longer sees O(50) TD targets.
- **Priority**: ✅ RESOLVED (was 🔴 RED)
- **Resolution**: `env_factory.build_training_env()` wraps `VecNormalize(norm_obs=False, norm_reward=True, clip_reward=10.0)`. `build_eval_env()` loads saved statistics with `training=False, norm_reward=False`. No changes to frozen RewardConfig.
- **Module/stage**: Module 5.3B complete.
- **Blocks Tier 1?**: ✅ No — resolved.

### L-RL-02: No trained policy yet exploits Block B (forecast)
- **Current state**: `OracleThresholdController` and `RealisticThresholdController` use the same threshold rule regardless of forecast content. The energy advantage of forecasting is unmeasured.
- **Scientific impact**: The oracle and realistic energy numbers in the Module 5.2C benchmark are IDENTICAL to the threshold baseline. This is expected — training has not occurred.
- **Priority**: 🔴 RED — this is the main Tier 1 research result.
- **Resolution**: Train PPO with Block B in the observation. This is the purpose of Module 5.3.
- **Blocks Tier 1?**: Yes — Tier 1 result requires a trained policy.

### L-RL-03: Threshold/RL action fairness (min latch duration comparison)
- **Current state**: 🔴 AUDIT COMPLETE — experimental design issue documented. **`FixedThresholdController`** enforces: `min_latch_steps=4` (4 s minimum hold time when latched) + `min_release_steps=3` (3 s dead time before re-latching when released). **RL environment (`WECControlEnv.step()`)** enforces NO minimum latch duration or dead time — the RL agent can switch between LATCH (1) and RELEASE (0) every single control step (1 s). This is an UNFAIR apples-to-oranges comparison when evaluating RL vs. the threshold baseline.
- **Scientific impact**: Two competing effects: (1) RL has more degrees of freedom (finer timing) which may give it an artificial engineering advantage over a mechanically-realistic latch controller that would have minimum dwell times; (2) the λ_switch=0.01 switching penalty in the reward function partially compensates by discouraging rapid toggling, but does not enforce a hard minimum. For scientifically-fair comparison in the final paper, either (a) add a configurable minimum-latch/dead-time wrapper to the RL environment (not modifying physics or WECControlEnv core), or (b) document the asymmetry explicitly and justify why the comparison still holds (e.g., "RL explores the full digital-control design space, while the threshold controller's min_latch represents a specific mechanical realisation").
- **Priority**: 🟠 ORANGE — document now; resolve before final comparison table.
- **Resolution**: Implement a post-step action wrapper `MinLatchWrapper(gym.Wrapper)` with configurable `min_latch_steps` and `min_release_steps` that enforces the same timing constraints as FixedThresholdConfig. Apply it ONLY when a fair head-to-head comparison is needed. Do NOT apply during the initial PPO training in Module 5.3C (training the RL with full action freedom first is the correct exploratory strategy; restrict later if comparison is desired). **Do NOT modify the existing `FixedThresholdController` or frozen `WECControlEnv` core.**
- **Module/stage**: Module 5.3B (documented now). Resolution deferred to pre-paper Module 5.4 analysis.
- **Blocks Tier 1?**: No — does not block initial PPO training. But final published comparison table requires this issue to be addressed (either by wrapper or by explicit disclosure).

### L-RL-04: Manually selected threshold (v_threshold=0.05 m/s not tuned)
- **Current state**: `FixedThresholdController` uses a development baseline threshold not optimised against the replay.
- **Scientific impact**: The threshold baseline may underperform an optimised fixed-threshold controller. Comparison with RL may be artificially favourable to RL.
- **Priority**: 🟠 ORANGE
- **Resolution**: Sweep threshold values on validation data and use best-performing value for the final comparison. Report both.
- **Module/stage**: Module 5.3 or 5.4 pre-paper analysis.
- **Blocks Tier 1?**: No — Tier 1 only requires a threshold baseline exists. But a poorly tuned threshold weakens the comparison.

### L-RL-05: WEC simulation is computationally expensive for RL
- **Current state**: Each episode requires Cummins integration at dt=0.1 s (O(36,000) physics steps per hour). BEM (Capytaine) computation ~15–20 s on first call per session; cached thereafter for all subsequent env instances. Single env step latency to be measured in Module 5.3B smoke test.
- **Scientific impact**: 5M timesteps with n_envs=4 expected ~20–40 minutes on CPU based on initial estimates. Acceptable for Tier 1. Grid hyperparameter search over many experiments is slow.
- **Priority**: 🟡 YELLOW
- **Resolution**: Pre-compute excitation force arrays; vectorise Python loops in CumminsStepIntegrator using numpy; or use GPU acceleration. BEM params already cached via `build_wec_params()` global cache — avoids repeated Capytaine runs.
- **Blocks Tier 1?**: No — 30 minutes is acceptable.

### L-RL-06: No confidence-gated MPC (Tier 2)
- **Priority**: 🟢 GREEN (Tier 2)

### L-RL-07: No residual RL correction layer (Tier 3)
- **Priority**: 🟢 GREEN (Tier 3)

### L-RL-08: No CVaR/distributional robustness (Tier 4)
- **Priority**: 🟢 GREEN (Tier 4)

### L-RL-09: No 5-condition stress matrix (Tier 4)
- **Priority**: 🟢 GREEN (Tier 4)

---

## Benchmark and Evaluation

### L-BEN-01: Benchmark uses synthetic constant sea states
- **Current state**: Module 5.2C benchmark uses 7 constant Hs/Tp conditions.
- **Scientific impact**: Does not reflect real sea-state variability. Energy results may not generalise.
- **Priority**: 🟠 ORANGE
- **Resolution**: Add real ERA5 sea-state benchmark segments from 2022–2023 validation period.
- **Blocks Tier 1?**: No — but should be done before paper submission.

### L-BEN-02: No second-site validation
- **Priority**: 🟢 GREEN (Tier 5)

### L-BEN-03: stable-baselines3 not yet installed
- **Current state**: ✅ RESOLVED. `stable_baselines3==2.9.0` installed in project venv. Confirmed working with Gymnasium 1.3.0 + PyTorch 2.14.0.
- **Scientific impact**: PPO training unblocked.
- **Priority**: ✅ RESOLVED (was 🔴 RED)
- **Resolution**: `pip install stable-baselines3==2.9.0` done. All SB3 VecNormalize wrappers, PPO class, CheckpointCallback, EvalCallback verified in tests.
- **Module/stage**: Module 5.3B complete.
- **Blocks Tier 1?**: ✅ No — resolved.

---

## Summary: Tier 1 Blockers

| ID | Issue | Required action | Status |
|----|-------|----------------|--------|
| L-FCT-01 | Error bank leakage | Derive 2022–2023 error bank before PPO training | ✅ RESOLVED (5.3B) |
| L-RL-01 | Reward scaling | Add VecNormalize wrapper in training script | ✅ RESOLVED (5.3B) |
| L-RL-02 | No trained policy | Train PPO (Module 5.3C) | 🔴 PENDING — pending 5.3C approval |
| L-RL-03 | Action fairness | Document + optional MinLatchWrapper before final comparison | 🟠 Documented (5.3B), wrapper TBD |
| L-BEN-03 | SB3 not installed | Install stable-baselines3 | ✅ RESOLVED (5.3B) |

Remaining Tier 1 blockers before research claims:
- **L-RL-02**: Train PPO policies (Module 5.3C: reactive → perfect → realistic) and validate on 2022–2023 data.

All other limitations are ORANGE or lower and do not block the Tier 1 result.
