"""
benchmark.py — Fair baseline benchmark harness for WEC control (Module 5.2C).

============================================================
SCIENTIFIC FAIRNESS GUARANTEE
============================================================

All controllers are evaluated on EXACTLY the same:
    - EpisodeReplay (sea-state sequence, excitation realisation, seed)
    - WEC physical parameters (mass, stiffness, BEM coefficients)
    - PTO parameters (B_PTO = 200 kN·s/m)
    - Radiation kernel
    - Physics timestep (DT_PHYSICS = 0.1 s)
    - Control timestep (DT_CONTROL = 1.0 s)
    - Initial conditions (x₀ = 0 m, v₀ = 0 m/s)
    - Episode duration

The ONLY difference between controller evaluations is the action sequence
returned by the controller.  The benchmark harness enforces this by
re-running the same environment instance (with the same reset seed) for
each controller, using the same replay object.

============================================================
BENCHMARK SEGMENTS
============================================================

The benchmark uses a fixed set of synthetic sea-state segments.
These were defined BEFORE any controller results were computed.
They are NOT selected based on which controller performs best.

Segment definitions:

    seg_0:  Hs=0.5 m,  Tp=8.0 s  — calm conditions
    seg_1:  Hs=1.0 m,  Tp=8.0 s  — moderate
    seg_2:  Hs=1.5 m,  Tp=8.0 s  — reference sea state (used in Module 5.2A audit)
    seg_3:  Hs=2.0 m,  Tp=8.0 s  — energetic
    seg_4:  Hs=2.5 m,  Tp=8.0 s  — high energy
    seg_5:  Hs=1.5 m,  Tp=6.0 s  — same Hs, shorter period
    seg_6:  Hs=1.5 m,  Tp=12.0 s — same Hs, longer period

Each segment: 2 ERA5 hours = 7200 control steps = 7200 s per run.
Base random seed: 42 for all segments.

============================================================
ENERGY CALCULATION
============================================================

Energy comes ONLY from the physical PTO power:

    P_PTO(t) = B_PTO × v(t)²   [W]    (Module 2 definition)
    E         = Σ P_PTO(t) × dt_physics   [J]
    E_Wh      = E_J / 3600               [Wh]

Source: info["cumulative_energy_Wh"] from PowerAccumulator, which
integrates at dt_physics = 0.1 s.

This is NOT derived from reward.

============================================================
OUTPUTS
============================================================

Saved to results/control_benchmarks/:
    benchmark_results.csv      — one row per (controller, segment)
    benchmark_summary.json     — aggregate statistics per controller

============================================================
MODULE DEPENDENCIES (read-only)
============================================================
    module5_control.environment   — WECControlEnv, build_env_from_checkpoint
    module5_control.replay        — build_synthetic_replay, EpisodeReplay
    module5_control.controllers   — BaseController and subclasses
    module5_control.forecast_uncertainty — ForecastErrorSampler (Baseline 5)
    module2_wec.cummins           — CumminsParameters
    module2_wec.pto               — PTOParameters
    module2_wec.hydrodynamic_coefficients — HydrodynamicCoefficients
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from module5_control.environment import (
    WECControlEnv,
    EnvironmentConfig,
    build_env_from_checkpoint,
    ACTION_RELEASE,
    ACTION_LATCH,
)
from module5_control.replay import (
    EpisodeReplay,
    build_synthetic_replay,
    DT_PHYSICS,
    DT_CONTROL,
    N_CONTROL_PER_HOUR,
)
from module5_control.controllers import (
    BaseController,
    PassiveController,
    FixedThresholdController,
    FixedThresholdConfig,
    ReactiveController,
    OracleThresholdController,
    RealisticThresholdController,
)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

RESULTS_DIR: str = os.path.join("results", "control_benchmarks")

# ---------------------------------------------------------------------------
# Benchmark segment definitions
# ---------------------------------------------------------------------------

#: Fixed benchmark sea-state segments — selected BEFORE evaluation.
#: Format: (segment_id, Hs_m, Tp_s, description)
BENCHMARK_SEGMENTS: List[Tuple[str, float, float, str]] = [
    ("seg_calm",      0.5,  8.0,  "calm sea (Hs=0.5m, Tp=8s)"),
    ("seg_moderate",  1.0,  8.0,  "moderate sea (Hs=1.0m, Tp=8s)"),
    ("seg_reference", 1.5,  8.0,  "reference sea (Hs=1.5m, Tp=8s) — Module 5.2A audit condition"),
    ("seg_energetic", 2.0,  8.0,  "energetic sea (Hs=2.0m, Tp=8s)"),
    ("seg_high",      2.5,  8.0,  "high energy (Hs=2.5m, Tp=8s)"),
    ("seg_short_T",   1.5,  6.0,  "shorter period (Hs=1.5m, Tp=6s)"),
    ("seg_long_T",    1.5,  12.0, "longer period (Hs=1.5m, Tp=12s)"),
]

#: Number of ERA5 hours per benchmark segment.
BENCHMARK_N_HOURS: int = 2   # 2 hours = 7200 control steps

#: Base random seed for all benchmark replays.
BENCHMARK_BASE_SEED: int = 42

#: Reset seed for environment (sensor RNG, sampler).
BENCHMARK_RESET_SEED: int = 0


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """
    Results for one controller on one benchmark segment.

    All energy / power quantities are derived from the Module 2 PTO model:
        P_abs = B_PTO × v²   [W]
        E     = Σ P_abs × dt_physics   [J → Wh]

    NOT derived from reward.
    """

    # Identification
    controller_label:  str
    segment_id:        str
    segment_desc:      str
    env_mode:          str

    # Sea state
    Hs_m:              float
    Tp_s:              float

    # Configuration
    pto_damping_Ns_m:  float
    pto_stiffness_N_m: float
    replay_seed:       int
    episode_hours:     float

    # Energy (from PowerAccumulator — Module 2 PTO physics)
    total_energy_Wh:   float
    total_energy_kWh:  float
    mean_power_W:      float

    # Power statistics (computed from step-by-step info)
    max_step_power_W:  float

    # Episode statistics
    total_reward:      float
    n_steps:           int

    # Latching statistics
    n_latch_events:    int
    n_release_events:  int
    latch_fraction:    float   # fraction of control steps in LATCH state

    # Displacement / velocity / acceleration statistics
    max_abs_displacement_m:  float
    max_abs_velocity_ms:     float
    max_abs_acceleration_ms2: float   # estimated as max |Δv| / DT_PHYSICS

    # Numerical health
    has_nan:           bool
    has_inf:           bool
    is_finite:         bool

    # Timing
    wall_time_s:       float

    # Energy cross-check: E_Wh ≈ mean_power_W × duration_hours
    energy_check_rel_err: float   # |E_Wh - mean_power × T_h| / E_Wh

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# BenchmarkHarness
# ---------------------------------------------------------------------------

class BenchmarkHarness:
    """
    Executes controlled benchmark comparisons across WEC controllers.

    All controllers run on the same EpisodeReplay with the same WEC
    physical parameters, the same reset seed, and the same episode duration.
    Only the controller's action sequence differs.

    Parameters
    ----------
    cummins_params : CumminsParameters
        Module 2 Cummins equation parameters.
    pto_params : PTOParameters
        Module 2 PTO parameters.  B_PTO = 200,000 N·s/m in the reference.
    hydro_excitation_amplitude : np.ndarray
        BEM |F_exc(omega)| [N/m].
    hydro_omega : np.ndarray
        BEM angular frequency grid [rad/s].
    hydro : optional HydrodynamicCoefficients
        For Module 5.2A frequency-dependent excitation.  If None,
        falls back to Module 5.1 single-frequency approximation.
    gru_checkpoint_path : str or None
        Path to canonical GRU+wind checkpoint.  Required for
        realistic/oracle benchmarks.
    max_episode_steps : int or None
        Cap on control steps.  None = run full replay.
    verbose : bool
        Print progress to stdout.
    """

    def __init__(
        self,
        cummins_params:              Any,    # CumminsParameters
        pto_params:                  Any,    # PTOParameters
        hydro_excitation_amplitude:  np.ndarray,
        hydro_omega:                 np.ndarray,
        hydro:                       Optional[Any] = None,
        gru_checkpoint_path:         Optional[str] = None,
        max_episode_steps:           Optional[int] = None,
        verbose:                     bool = True,
    ) -> None:
        self.cummins_params             = cummins_params
        self.pto_params                 = pto_params
        self.hydro_excitation_amplitude = hydro_excitation_amplitude
        self.hydro_omega                = hydro_omega
        self.hydro                      = hydro
        self.gru_checkpoint_path        = gru_checkpoint_path
        self.max_episode_steps          = max_episode_steps
        self.verbose                    = verbose

    # ------------------------------------------------------------------
    # Core: run one controller on one replay
    # ------------------------------------------------------------------

    def run_episode(
        self,
        controller:  BaseController,
        replay:      EpisodeReplay,
        env_mode:    str = "reactive",
        reset_seed:  int = BENCHMARK_RESET_SEED,
        segment_id:  str = "unknown",
        segment_desc: str = "",
        Hs_m:        float = float("nan"),
        Tp_s:        float = float("nan"),
    ) -> BenchmarkResult:
        """
        Run one complete episode with the given controller and return results.

        Parameters
        ----------
        controller : BaseController
        replay : EpisodeReplay
            Pre-built deterministic replay.  NOT re-created here.
        env_mode : str
            "passive", "reactive", "perfect_forecast", or "realistic_forecast".
            Note: "passive" is not a valid env mode — use "reactive" for the
            environment; the PassiveController handles action=RELEASE.
        reset_seed : int
        segment_id, segment_desc, Hs_m, Tp_s : metadata

        Returns
        -------
        BenchmarkResult
        """
        # Determine the environment information mode
        # "passive" is a controller property, not an env mode.
        # PassiveController always issues ACTION_RELEASE in any env mode.
        _env_mode_map = {
            "passive":             "reactive",
            "reactive":            "reactive",
            "fixed_threshold":     "reactive",
            "oracle_interface":    "perfect_forecast",
            "realistic_interface": "realistic_forecast",
        }
        actual_env_mode = _env_mode_map.get(env_mode, env_mode)

        # Build forecast-error sampler for realistic mode
        error_sampler = None
        if actual_env_mode == "realistic_forecast":
            from module5_control.forecast_uncertainty import ForecastErrorSampler
            error_sampler = ForecastErrorSampler.from_csv(seed=reset_seed)

        # Construct environment
        env = build_env_from_checkpoint(
            replay                    = replay,
            cummins_params            = self.cummins_params,
            pto_params                = self.pto_params,
            hydro_excitation_amplitude= self.hydro_excitation_amplitude,
            hydro_omega               = self.hydro_omega,
            mode                      = actual_env_mode,
            gru_checkpoint_path       = self.gru_checkpoint_path,
            max_episode_steps         = self.max_episode_steps,
            hydro                     = self.hydro,
            forecast_error_sampler    = error_sampler,
        )

        # Reset controller and environment
        controller.reset()
        obs, info = env.reset(seed=reset_seed)

        # Episode loop
        t0 = time.perf_counter()

        total_reward        = 0.0
        n_steps             = 0
        n_latch_events      = 0
        n_release_events    = 0
        latch_step_count    = 0
        prev_latch          = 0
        max_step_power      = 0.0
        max_abs_x           = 0.0
        max_abs_v           = 0.0
        prev_v              = 0.0
        max_abs_dv          = 0.0   # for acceleration estimate
        has_nan             = False
        has_inf             = False

        terminated = False
        truncated  = False

        while not (terminated or truncated):
            action = controller.act(obs, info)

            obs, reward, terminated, truncated, info = env.step(action)

            # Accumulate statistics
            total_reward += reward
            n_steps      += 1

            curr_latch = int(info["latch_status"])
            if curr_latch == 1 and prev_latch == 0:
                n_latch_events += 1
            if curr_latch == 0 and prev_latch == 1:
                n_release_events += 1
            if curr_latch == 1:
                latch_step_count += 1
            prev_latch = curr_latch

            p = float(info["mean_step_power_W"])
            if p > max_step_power:
                max_step_power = p

            x = float(info["x"])
            v = float(info["v"])
            if abs(x) > max_abs_x:
                max_abs_x = abs(x)
            if abs(v) > max_abs_v:
                max_abs_v = abs(v)

            # Velocity change as acceleration proxy (v changes each 1.0 s step)
            dv = abs(v - prev_v)
            if dv > max_abs_dv:
                max_abs_dv = dv
            prev_v = v

            if not np.isfinite(float(info["cumulative_energy_J"])):
                has_nan = True if np.isnan(float(info["cumulative_energy_J"])) else has_nan
                has_inf = True if np.isinf(float(info["cumulative_energy_J"])) else has_inf

            if np.any(~np.isfinite(obs)):
                has_nan = True if np.any(np.isnan(obs)) else has_nan
                has_inf = True if np.any(np.isinf(obs)) else has_inf

        wall_time = time.perf_counter() - t0

        # Final energy (from PowerAccumulator via info)
        total_energy_Wh  = float(info["cumulative_energy_Wh"])
        total_energy_kWh = total_energy_Wh / 1000.0
        episode_hours    = replay.episode_duration_s / 3600.0
        mean_power_W     = (
            total_energy_Wh * 3600.0 / replay.episode_duration_s
            if replay.episode_duration_s > 0 else 0.0
        )

        latch_fraction = latch_step_count / n_steps if n_steps > 0 else 0.0

        # Estimated max acceleration = max |Δv| / DT_CONTROL
        # (Δv measured at 1 Hz control steps, not physics sub-steps)
        max_abs_a_est = max_abs_dv / DT_CONTROL

        # Energy cross-check: |E_Wh - mean_P × T_h| / E_Wh
        e_check = mean_power_W * episode_hours
        energy_check_rel_err = (
            abs(total_energy_Wh - e_check) / max(total_energy_Wh, 1e-12)
        )

        return BenchmarkResult(
            controller_label         = controller.label,
            segment_id               = segment_id,
            segment_desc             = segment_desc,
            env_mode                 = actual_env_mode,
            Hs_m                     = Hs_m,
            Tp_s                     = Tp_s,
            pto_damping_Ns_m         = float(self.pto_params.damping),
            pto_stiffness_N_m        = float(self.pto_params.stiffness),
            replay_seed              = int(replay.seed) if replay.seed is not None else -1,
            episode_hours            = episode_hours,
            total_energy_Wh          = total_energy_Wh,
            total_energy_kWh         = total_energy_kWh,
            mean_power_W             = mean_power_W,
            max_step_power_W         = max_step_power,
            total_reward             = total_reward,
            n_steps                  = n_steps,
            n_latch_events           = n_latch_events,
            n_release_events         = n_release_events,
            latch_fraction           = latch_fraction,
            max_abs_displacement_m   = max_abs_x,
            max_abs_velocity_ms      = max_abs_v,
            max_abs_acceleration_ms2 = max_abs_a_est,
            has_nan                  = has_nan,
            has_inf                  = has_inf,
            is_finite                = not (has_nan or has_inf),
            wall_time_s              = wall_time,
            energy_check_rel_err     = energy_check_rel_err,
        )

    # ------------------------------------------------------------------
    # High-level: run all controllers on all segments
    # ------------------------------------------------------------------

    def run_benchmark(
        self,
        controllers:          List[BaseController],
        segments:             Optional[List[Tuple[str, float, float, str]]] = None,
        n_hours:              int  = BENCHMARK_N_HOURS,
        base_seed:            int  = BENCHMARK_BASE_SEED,
        reset_seed:           int  = BENCHMARK_RESET_SEED,
        save_results:         bool = True,
        results_dir:          str  = RESULTS_DIR,
        include_oracle:       bool = True,
        include_realistic:    bool = True,
    ) -> List[BenchmarkResult]:
        """
        Run all controllers on all benchmark segments.

        The same EpisodeReplay (same sea state, same seed) is used for
        EVERY controller evaluation within a segment.  This guarantees
        identical physics conditions.

        Parameters
        ----------
        controllers : list of BaseController
            Controllers to evaluate.  Should include at minimum:
            PassiveController, FixedThresholdController, ReactiveController.
        segments : list of (seg_id, Hs, Tp, desc) or None
            Segment definitions.  Defaults to BENCHMARK_SEGMENTS.
        n_hours : int
            Hours per segment.  Default BENCHMARK_N_HOURS = 2.
        base_seed : int
            Base seed for replay generation.  Same for all controllers.
        reset_seed : int
            Seed for env.reset().  Same for all controllers.
        save_results : bool
            Whether to save CSV/JSON to results_dir.
        results_dir : str
            Output directory.
        include_oracle : bool
            Whether to include oracle (perfect_forecast) pathway.
        include_realistic : bool
            Whether to include realistic_forecast pathway.

        Returns
        -------
        list of BenchmarkResult
        """
        if segments is None:
            segments = BENCHMARK_SEGMENTS

        all_results: List[BenchmarkResult] = []
        total_runs = len(segments) * len(controllers)
        run_idx    = 0

        for seg_id, Hs, Tp, seg_desc in segments:
            # Build ONE replay per segment — same for ALL controllers
            replay = build_synthetic_replay(
                n_hours       = n_hours,
                Hs            = Hs,
                Tp            = Tp,
                direction_deg = 270.0,
                u10           = 5.0,
                v10           = 2.0,
                seed          = base_seed,
            )

            if self.verbose:
                print(f"\n[Segment {seg_id}] Hs={Hs}m Tp={Tp}s — "
                      f"{n_hours}h, {replay.n_control_steps} steps")

            for controller in controllers:
                run_idx += 1
                label = controller.label

                # Determine env_mode from controller type
                env_mode = label  # passed to run_episode which maps it

                if self.verbose:
                    print(f"  [{run_idx}/{total_runs}] {label} ... ", end="", flush=True)

                result = self.run_episode(
                    controller   = controller,
                    replay       = replay,
                    env_mode     = env_mode,
                    reset_seed   = reset_seed,
                    segment_id   = seg_id,
                    segment_desc = seg_desc,
                    Hs_m         = Hs,
                    Tp_s         = Tp,
                )
                all_results.append(result)

                if self.verbose:
                    print(
                        f"E={result.total_energy_Wh:.4f} Wh  "
                        f"P_mean={result.mean_power_W:.1f} W  "
                        f"latches={result.n_latch_events}  "
                        f"finite={result.is_finite}  "
                        f"t={result.wall_time_s:.1f}s"
                    )

        if save_results:
            self._save_results(all_results, results_dir)

        return all_results

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def _save_results(
        self,
        results:     List[BenchmarkResult],
        results_dir: str,
    ) -> None:
        """Save benchmark results to CSV and JSON."""
        os.makedirs(results_dir, exist_ok=True)

        # --- CSV ---
        rows = [r.to_dict() for r in results]
        df   = pd.DataFrame(rows)
        csv_path = os.path.join(results_dir, "benchmark_results.csv")
        df.to_csv(csv_path, index=False)
        if self.verbose:
            print(f"\nResults saved: {csv_path}")

        # --- JSON summary ---
        summary = _compute_summary(results)
        json_path = os.path.join(results_dir, "benchmark_summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        if self.verbose:
            print(f"Summary saved:  {json_path}")


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------

def _compute_summary(results: List[BenchmarkResult]) -> dict:
    """
    Compute per-controller aggregate statistics from benchmark results.

    Returns a dict suitable for JSON serialisation.
    """
    from collections import defaultdict

    by_ctrl: Dict[str, List[float]] = defaultdict(list)
    ctrl_meta: Dict[str, dict] = {}

    for r in results:
        lbl = r.controller_label
        by_ctrl[lbl].append(r.total_energy_Wh)
        ctrl_meta[lbl] = {
            "env_mode":            r.env_mode,
            "pto_damping_Ns_m":    r.pto_damping_Ns_m,
            "pto_stiffness_N_m":   r.pto_stiffness_N_m,
            "all_finite":          all(
                rr.is_finite for rr in results if rr.controller_label == lbl
            ),
        }

    summary = {
        "metadata": {
            "n_segments":          len({r.segment_id for r in results}),
            "n_controllers":       len(ctrl_meta),
            "benchmark_seed":      BENCHMARK_BASE_SEED,
            "reset_seed":          BENCHMARK_RESET_SEED,
            "episode_hours":       results[0].episode_hours if results else 0,
            "scientific_note":     (
                "Energy derived from B_PTO*v^2 integrated at dt_physics=0.1s. "
                "All controllers evaluated on identical EpisodeReplay. "
                "oracle_interface and realistic_interface use FixedThreshold rule "
                "only — no trained RL policy. PPO training deferred to Module 5.3."
            ),
        },
        "controllers": {},
    }

    for lbl, energies in by_ctrl.items():
        arr = np.array(energies)
        summary["controllers"][lbl] = {
            **ctrl_meta.get(lbl, {}),
            "n_segments":    int(len(arr)),
            "mean_Wh":       float(np.mean(arr)),
            "median_Wh":     float(np.median(arr)),
            "std_Wh":        float(np.std(arr)),
            "min_Wh":        float(np.min(arr)),
            "max_Wh":        float(np.max(arr)),
            "mean_kWh":      float(np.mean(arr) / 1000),
        }

    return summary


# ---------------------------------------------------------------------------
# Convenience: standard benchmark setup
# ---------------------------------------------------------------------------

def build_standard_benchmark_controllers(
    threshold_config: Optional[FixedThresholdConfig] = None,
) -> List[BaseController]:
    """
    Return the standard set of Module 5.2C benchmark controllers.

    Includes:
        1. PassiveController
        2. FixedThresholdController
        3. ReactiveController
        4. OracleThresholdController    (INTERFACE_ONLY)
        5. RealisticThresholdController  (INTERFACE_ONLY)

    All latching controllers use the same threshold configuration.
    """
    cfg = threshold_config or FixedThresholdConfig()
    return [
        PassiveController(),
        FixedThresholdController(cfg),
        ReactiveController(cfg),
        OracleThresholdController(cfg),
        RealisticThresholdController(cfg),
    ]


def build_wec_params() -> dict:
    """
    Build and return the standard WEC physical parameters for benchmarking.

    Computes BEM coefficients, radiation kernel, and CumminsParameters.
    This is expensive (~15 s) — call once and cache.

    Returns
    -------
    dict with keys: cummins, pto, hydro_amp, hydro_omega, hydro
    """
    from module2_wec.geometry import REFERENCE_BUOY
    from module2_wec.hydrostatics import Hydrostatics
    from module2_wec.hydrodynamic_coefficients import compute_hydrodynamic_coefficients
    from module2_wec.radiation import compute_radiation_kernel
    from module2_wec.cummins import CumminsParameters
    from module2_wec.pto import PTOParameters

    hydro = compute_hydrodynamic_coefficients(
        geometry  = REFERENCE_BUOY,
        resolution= (6, 24, 16),
        omega_min = 0.2,
        omega_max = 1.4,
        n_omega   = 20,
        progress_bar = False,
    )
    hs_obj = Hydrostatics(REFERENCE_BUOY)
    rk     = compute_radiation_kernel(hydro)
    cp     = CumminsParameters(
        mass                  = REFERENCE_BUOY.mass,
        hydrostatic_stiffness = hs_obj.hydrostatic_stiffness,
        added_mass_infinity   = rk.A_infinity,
        kernel_time           = rk.time,
        kernel_values         = rk.kernel,
    )
    pto = PTOParameters(damping=200_000.0, stiffness=0.0)

    return {
        "cummins":    cp,
        "pto":        pto,
        "hydro_amp":  hydro.excitation_force_heave_amplitude,
        "hydro_omega":hydro.omega,
        "hydro":      hydro,
    }


def print_benchmark_table(results: List[BenchmarkResult]) -> None:
    """Print a compact benchmark table to stdout."""
    labels = sorted({r.controller_label for r in results})
    segs   = sorted({r.segment_id for r in results})

    # Header
    col_w = 22
    hdr   = f"{'Controller':<{col_w}}"
    for s in segs:
        hdr += f"  {s[:12]:>12}"
    hdr += f"  {'Mean Wh':>10}"
    print(hdr)
    print("-" * len(hdr))

    for lbl in labels:
        row = f"{lbl:<{col_w}}"
        vals = []
        for s in segs:
            matches = [r for r in results if r.controller_label == lbl and r.segment_id == s]
            if matches:
                e = matches[0].total_energy_Wh
                row += f"  {e:>12.4f}"
                vals.append(e)
            else:
                row += f"  {'N/A':>12}"
        if vals:
            row += f"  {np.mean(vals):>10.4f}"
        print(row)
    print()
    print("(all values in Wh; energy from B_PTO*v^2 integrated at dt=0.1s)")
