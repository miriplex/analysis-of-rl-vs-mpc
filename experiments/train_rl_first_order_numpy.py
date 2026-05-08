from __future__ import annotations

import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import os
from pathlib import Path
import sys
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from control_bench.config import (
    FIRST_ORDER_COST_WEIGHTS,
    FIRST_ORDER_FAMILY as family,
    FIRST_ORDER_PLANT_IDS,
)
from control_bench.controllers.rl_numpy import (
    PlantParams,
    RLBPTTConfig,
    LinearPIDFeaturePolicy,
    MLPPolicy,
    MLPPolicySpec,
)
from control_bench.controllers.rl_numpy.feature_sets import first_order_mlp_feature_input_dim
from control_bench.experiment_manifest import first_order_default_mlp_variant, first_order_mlp_variants
from control_bench.controllers.rl_numpy.training import load_policy_npz, train_one_policy_with_validation
from first_order_rl_helpers import (
    build_first_order_robust_validation_rollouts,
    build_first_order_universal_robust_validation_rollouts,
    build_first_order_validation_rollouts as build_validation_rollouts,
    extract_first_order_plant_params as extract_plant_params,
    sample_first_order_robust_training_rollout,
    sample_first_order_universal_robust_training_rollout,
    sample_first_order_training_rollout as sample_training_rollout,
)


UNSTABLE_PLANT_IDS = {
    "unstable__no_zero",
    "unstable__lhp_zero",
    "unstable__rhp_zero",
}
UNSTABLE_REG_TAIL_GATE = 1.0
RICH_ROBUST_SELECTION_WEIGHTS = {
    "track_mean": 0.25,
    "reg_mean": 0.15,
    "dist_mean": 0.20,
    "dist_tail_mean": 0.10,
    "lag_mean": 0.20,
    "lag_tail_mean": 0.10,
}


def _save_params_npz_quiet(path: str, *, params: dict, meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {**params}
    payload["_meta"] = np.array([repr(meta)], dtype=object)
    np.savez(path, **payload)


def _save_policy_npz_quiet(path: str, *, policy, meta: dict) -> None:
    _save_params_npz_quiet(path, params=policy.params, meta=meta)


def _parse_meta_repr(meta: dict) -> dict:
    meta_repr = meta.get("_meta_repr")
    if not meta_repr:
        return {}
    try:
        parsed = ast.literal_eval(meta_repr)
    except (SyntaxError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _seeded_save_name(prefix: str, plant_id: str, seed_index: int) -> str:
    return f"{prefix}__{plant_id}__seed{seed_index}.npz"


def _seeded_pidfeat_save_name(plant_id: str, seed_index: int) -> str:
    return f"rl_pidfeat__{plant_id}__seed{seed_index}.npz"


def _train_and_save_job(
    *,
    plant_id: str,
    plant: PlantParams,
    cfg: RLBPTTConfig,
    steps: int,
    lr: float,
    kind: str,
    out_dir: str,
    seed_index: int = 0,
    meta_extra: Optional[dict] = None,
    mlp_spec: Optional[MLPPolicySpec] = None,
    eval_every: int = 100,
    early_stop_patience_evals: int = 10,
    early_stop_rel_improve: float = 0.01,
    early_stop_best_val_threshold: float = 1.0,
    grad_clip_norm: Optional[float] = None,
    save_name: Optional[str] = None,
    rich_feature_set: str = "rich11",
    rich_tracking_probability: float = 0.45,
    rich_regulation_probability: float = 0.20,
    rich_measurement_noise_std: float = 0.01,
    rich_disturbance_noise_std: float = 0.01,
    rich_pole_rel_range: float = 0.05,
    rich_zero_rel_range: float = 0.05,
    rich_gain_rel_range: float = 0.05,
    rich_hidden_pole_enabled: bool = False,
    rich_hidden_pole_low: float = 8.0,
    rich_hidden_pole_high: float = 20.0,
    rich_validation_measurement_noise_std: float = 0.01,
    rich_validation_disturbance_noise_std: float = 0.02,
    rich_sampler_kind: str = "current_robust",
    rich_measurement_noise_std_low: float = 0.01,
    rich_measurement_noise_std_high: float = 0.02,
    rich_disturbance_noise_std_low: float = 0.01,
    rich_disturbance_noise_std_high: float = 0.02,
    rich_disturbance_step_time_seconds_low: float = 2.0,
    rich_disturbance_step_time_seconds_high: float = 8.0,
    rich_disturbance_step_magnitude_abs_low: float = 0.03,
    rich_disturbance_step_magnitude_abs_high: float = 0.20,
    rich_disturbance_both_signs: bool = True,
    rich_validation_sampler_kind: str = "current_robust",
    rich_validation_disturbance_cases: Optional[tuple[tuple[float, float], ...]] = None,
    rich_validation_hidden_poles: Optional[tuple[float, ...]] = None,
) -> dict:
    import zlib

    job_seed = (int(seed_index) + int(zlib.crc32(f"{kind}:{plant_id}".encode("utf-8")))) % (2**32)
    rng = np.random.default_rng(job_seed)

    if kind == "pidfeat":
        policy = LinearPIDFeaturePolicy(seed=job_seed)
        episode_sampler = sample_training_rollout
        validation_rollouts = build_validation_rollouts(cfg)
    elif kind == "rich":
        if mlp_spec is None:
            raise ValueError("mlp_spec must be provided for kind='rich'")
        policy = MLPPolicy(mlp_spec, seed=job_seed)
        if rich_sampler_kind == "current_robust":
            episode_sampler = lambda local_rng, local_cfg: sample_first_order_robust_training_rollout(
                local_rng,
                local_cfg,
                plant_id=plant_id,
                tracking_probability=rich_tracking_probability,
                regulation_probability=rich_regulation_probability,
                measurement_noise_std=rich_measurement_noise_std,
                disturbance_noise_std=rich_disturbance_noise_std,
                pole_rel_range=rich_pole_rel_range,
                zero_rel_range=rich_zero_rel_range,
                gain_rel_range=rich_gain_rel_range,
                hidden_pole_low=rich_hidden_pole_low if rich_hidden_pole_enabled else None,
                hidden_pole_high=rich_hidden_pole_high if rich_hidden_pole_enabled else None,
            )
        elif rich_sampler_kind == "universal_robust":
            episode_sampler = lambda local_rng, local_cfg: sample_first_order_universal_robust_training_rollout(
                local_rng,
                local_cfg,
                plant_id=plant_id,
                tracking_probability=rich_tracking_probability,
                regulation_probability=rich_regulation_probability,
                measurement_noise_std_low=rich_measurement_noise_std_low,
                measurement_noise_std_high=rich_measurement_noise_std_high,
                disturbance_noise_std_low=rich_disturbance_noise_std_low,
                disturbance_noise_std_high=rich_disturbance_noise_std_high,
                disturbance_step_time_seconds_low=rich_disturbance_step_time_seconds_low,
                disturbance_step_time_seconds_high=rich_disturbance_step_time_seconds_high,
                disturbance_step_magnitude_abs_low=rich_disturbance_step_magnitude_abs_low,
                disturbance_step_magnitude_abs_high=rich_disturbance_step_magnitude_abs_high,
                disturbance_both_signs=rich_disturbance_both_signs,
                pole_rel_range=rich_pole_rel_range,
                zero_rel_range=rich_zero_rel_range,
                gain_rel_range=rich_gain_rel_range,
            )
        else:
            raise ValueError(f"Unknown rich_sampler_kind: {rich_sampler_kind}")

        if rich_validation_sampler_kind == "current_robust":
            validation_rollouts = build_first_order_robust_validation_rollouts(
                cfg,
                plant_id=plant_id,
                measurement_noise_std=rich_validation_measurement_noise_std,
                disturbance_noise_std=rich_validation_disturbance_noise_std,
                hidden_pole_cases=rich_validation_hidden_poles or (),
            )
        elif rich_validation_sampler_kind == "universal_robust":
            validation_rollouts = build_first_order_universal_robust_validation_rollouts(
                cfg,
                measurement_noise_std=rich_validation_measurement_noise_std,
                disturbance_noise_std=rich_validation_disturbance_noise_std,
                disturbance_cases=(
                    rich_validation_disturbance_cases
                    if rich_validation_disturbance_cases is not None
                    else ((2.5, 0.08), (5.0, -0.12), (7.5, 0.15))
                ),
            )
        else:
            raise ValueError(f"Unknown rich_validation_sampler_kind: {rich_validation_sampler_kind}")
    else:
        raise ValueError(f"Unknown kind: {kind}")

    summary = train_one_policy_with_validation(
        policy=policy,
        plant=plant,
        cfg=cfg,
        steps=steps,
        lr=lr,
        kind=kind,
        episode_sampler=episode_sampler,
        validation_rollouts=validation_rollouts,
        eval_every=eval_every,
        early_stop_patience_evals=early_stop_patience_evals,
        early_stop_rel_improve=early_stop_rel_improve,
        early_stop_best_val_threshold=early_stop_best_val_threshold,
        print_every=0,
        rng=rng,
        grad_clip_norm=grad_clip_norm,
        rich_feature_set=rich_feature_set,
    )

    if save_name is None:
        save_name = f"rl_{kind}__{plant_id}.npz"
    save_path = str(Path(out_dir) / save_name)
    extra = {} if meta_extra is None else dict(meta_extra)
    _save_policy_npz_quiet(
        save_path,
        policy=policy,
        meta={
            "kind": kind,
            "plant_id": plant_id,
            "final_loss": summary["final_loss"],
            "best_val_loss": summary["best_val_loss"],
            "steps_completed": summary["steps_completed"],
            "stopped_early": summary["stopped_early"],
            "cfg": cfg.__dict__,
            "steps_cap": steps,
            "lr": lr,
            "seed_index": seed_index,
            "job_seed": job_seed,
            "eval_every": eval_every,
            "early_stop_patience_evals": early_stop_patience_evals,
            "early_stop_rel_improve": early_stop_rel_improve,
            "early_stop_best_val_threshold": early_stop_best_val_threshold,
            "grad_clip_norm": grad_clip_norm,
            "meta_extra": extra,
            "rich_feature_set": rich_feature_set if kind == "rich" else None,
        },
    )

    return {
        "kind": kind,
        "plant_id": plant_id,
        "final_loss": float(summary["final_loss"]),
        "best_val_loss": float(summary["best_val_loss"]),
        "final_val_loss": float(summary["val_loss"][-1] if len(summary["val_loss"]) else np.inf),
        "validation_metrics": dict(summary["validation_metrics"]),
        "steps_completed": int(summary["steps_completed"]),
        "stopped_early": bool(summary["stopped_early"]),
        "updates": np.asarray(summary["updates"], dtype=np.int64),
        "train_loss": np.asarray(summary["train_loss"], dtype=np.float64),
        "val_loss": np.asarray(summary["val_loss"], dtype=np.float64),
        "seed_index": int(seed_index),
        "job_seed": int(job_seed),
        "save_path": save_path,
        "rich_feature_set": rich_feature_set if kind == "rich" else None,
        "selection_weight_overrides": (
            dict(
                extra.get("robust_seed_selection", {}).get("weights", {})
                or extra.get("default_mlp_variant", {}).get("selection_weights", {})
            )
            if kind == "rich"
            else {}
        ),
        "selection_cfg_overrides": (
            dict(extra.get("robust_seed_selection", {}))
            if kind == "rich"
            else {}
        ),
    }


def _default_worker_count(n_jobs: int) -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(n_jobs, cpu_count - 1 if cpu_count > 1 else 1))


def plot_loss_histories(
    results: Dict[str, Dict[str, dict]],
    *,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    eps = 1e-12

    figure_specs = [
        ("pidfeat", "PID-features", out_dir / "rl_pidfeat_loss_curves.png"),
        ("rich", "MLP rich", out_dir / "rl_rich_loss_curves.png"),
    ]

    for kind, title_tag, save_path in figure_specs:
        fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=False, sharey=False)
        axes = axes.reshape(-1)

        for ax, plant_id in zip(axes, FIRST_ORDER_PLANT_IDS):
            result = results[kind][plant_id]
            train_loss = np.maximum(result["train_loss"], eps)
            val_loss = np.maximum(result["val_loss"], eps)
            ax.plot(result["updates"], train_loss, linewidth=1.8, label="train")
            ax.plot(result["updates"], val_loss, linewidth=1.8, label="validation")
            ax.set_yscale("log")
            status = "early stop" if result["stopped_early"] else "cap"
            ax.set_title(f"{plant_id}\n{status} @ {result['steps_completed']}")
            ax.set_xlabel("Update")
            ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2)
        fig.suptitle(f"RL loss curves | {title_tag}")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(save_path, dpi=160)

    plt.show()


def plot_seed_selection_histories(
    seed_results_by_plant: Dict[str, Dict[int, dict]],
    *,
    selected_seed_by_plant: Dict[str, int],
    out_dir: Path,
    title_tag: str,
    file_prefix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    eps = 1e-12

    for plant_id in FIRST_ORDER_PLANT_IDS:
        seed_map = seed_results_by_plant.get(plant_id)
        if not seed_map:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        chosen_seed = selected_seed_by_plant[plant_id]

        for seed_index in sorted(seed_map):
            result = seed_map[seed_index]
            val_loss = np.maximum(result["val_loss"], eps)
            label = f"seed {seed_index}"
            linewidth = 1.8
            alpha = 0.75
            if seed_index == chosen_seed:
                label += " (chosen)"
                linewidth = 2.8
                alpha = 1.0
            ax.plot(result["updates"], val_loss, linewidth=linewidth, alpha=alpha, label=label)

        best_result = seed_map[chosen_seed]
        best_update = int(best_result["updates"][-1]) if len(best_result["updates"]) else 0
        selection_metrics = best_result.get("selection_metrics", {})
        ax.set_yscale("log")
        title = (
            f"{plant_id}\nchosen seed={chosen_seed} | "
            f"best_val={best_result['best_val_loss']:.6f} | stop @ {best_update}"
        )
        if plant_id in UNSTABLE_PLANT_IDS:
            title += (
                f"\nreg_tail={selection_metrics.get('reg_tail_mean', np.nan):.6f} | "
                f"score={selection_metrics.get('selection_score', np.nan):.6f}"
            )
        ax.set_title(title)
        ax.set_xlabel("Update")
        ax.set_ylabel("Validation loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.suptitle(f"RL seed selection | {title_tag}", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out_dir / f"{file_prefix}__{plant_id}.png", dpi=160)
        plt.close(fig)


def _select_best_seed_results(kind: str, plant_id: str, seed_results: Dict[int, dict]) -> tuple[int, dict]:
    if not seed_results:
        raise ValueError("seed_results must be non-empty")

    def default_rank_key(item: tuple[int, dict]) -> tuple[float, float, int]:
        seed_index, result = item
        return (float(result["best_val_loss"]), float(result["final_val_loss"]), int(seed_index))

    if kind == "rich":
        weights = None
        for result in seed_results.values():
            weights = dict(result.get("selection_weight_overrides", {}))
            if weights:
                break
        if not weights:
            weights = dict(RICH_ROBUST_SELECTION_WEIGHTS)

        selection_cfg = {}
        for result in seed_results.values():
            selection_cfg = dict(result.get("selection_cfg_overrides", {}))
            if selection_cfg:
                break

        def robust_score(metrics: dict) -> float:
            return float(
                float(weights.get("track_mean", 0.0)) * float(metrics.get("track_mean", np.inf))
                + float(weights.get("reg_mean", 0.0)) * float(metrics.get("reg_mean", np.inf))
                + float(weights.get("dist_mean", 0.0)) * float(metrics.get("dist_mean", np.inf))
                + float(weights.get("dist_tail_mean", 0.0)) * float(metrics.get("dist_tail_mean", np.inf))
                + float(weights.get("lag_mean", 0.0)) * float(metrics.get("lag_mean", np.inf))
                + float(weights.get("lag_tail_mean", 0.0)) * float(metrics.get("lag_tail_mean", np.inf))
            )

        reg_tail_gate = float(selection_cfg.get("nominal_gate_reg_tail_max", np.inf))
        track_mean_gate = float(selection_cfg.get("nominal_gate_track_mean_max", np.inf))

        gated_items: list[tuple[int, dict]] = []
        for seed_index, result in seed_results.items():
            metrics = dict(result["validation_metrics"])
            score = robust_score(metrics)
            gate_passed = bool(
                float(metrics.get("reg_tail_mean", np.inf)) <= reg_tail_gate
                and float(metrics.get("track_mean", np.inf)) <= track_mean_gate
            )
            result["selection_metrics"] = {
                "selection_strategy": str(selection_cfg.get("strategy", "robust_validation_score")),
                "selection_score": score,
                "gate_passed": gate_passed,
                **metrics,
            }
            if gate_passed:
                gated_items.append((seed_index, result))

        candidate_items = gated_items if gated_items else list(seed_results.items())

        def rich_rank_key(item: tuple[int, dict]) -> tuple[float, float, float, float, int]:
            seed_index, result = item
            metrics = dict(result["selection_metrics"])
            return (
                float(metrics["selection_score"]),
                float(metrics.get("lag_tail_mean", np.inf)),
                float(result["best_val_loss"]),
                float(result["final_val_loss"]),
                int(seed_index),
            )

        best_seed, best_result = min(candidate_items, key=rich_rank_key)
        return best_seed, best_result

    if plant_id not in UNSTABLE_PLANT_IDS:
        best_seed, best_result = min(seed_results.items(), key=default_rank_key)
        selected_metrics = dict(best_result["validation_metrics"])
        best_result["selection_metrics"] = {
            "selection_strategy": "lowest_best_val_loss",
            "selection_score": float(best_result["best_val_loss"]),
            "gate_passed": True,
            **selected_metrics,
        }
        return best_seed, best_result

    gate_passed_items = []
    for seed_index, result in seed_results.items():
        metrics = dict(result["validation_metrics"])
        metrics["selection_score"] = float(result["best_val_loss"])
        metrics["gate_passed"] = bool(metrics["reg_tail_mean"] <= UNSTABLE_REG_TAIL_GATE)
        result["selection_metrics"] = {
            "selection_strategy": "reg_tail_gate_then_lowest_best_val_loss",
            **metrics,
        }
        if metrics["gate_passed"]:
            gate_passed_items.append((seed_index, result))

    if gate_passed_items:
        def unstable_rank_key(item: tuple[int, dict]) -> tuple[float, float, float, float, int]:
            seed_index, result = item
            metrics = result["selection_metrics"]
            return (
                float(result["best_val_loss"]),
                float(metrics["reg_tail_mean"]),
                float(metrics["reg_mean"]),
                float(result["final_val_loss"]),
                int(seed_index),
            )

        best_seed, best_result = min(gate_passed_items, key=unstable_rank_key)
        return best_seed, best_result

    def fallback_rank_key(item: tuple[int, dict]) -> tuple[float, float, float, float, int]:
        seed_index, result = item
        metrics = result["selection_metrics"]
        return (
            float(metrics["reg_tail_mean"]),
            float(metrics["reg_mean"]),
            float(result["best_val_loss"]),
            float(result["final_val_loss"]),
            int(seed_index),
        )

    best_seed, best_result = min(seed_results.items(), key=fallback_rank_key)
    best_result["selection_metrics"]["selection_strategy"] = "fallback_lowest_reg_tail_mean"
    return best_seed, best_result


def _write_canonical_seeded_policy(
    *,
    kind: str,
    plant_id: str,
    out_dir: Path,
    selected_seed: int,
    selected_result: dict,
    all_seed_results: Dict[int, dict],
    canonical_filename_prefix: Optional[str] = None,
) -> None:
    params, loaded_meta = load_policy_npz(selected_result["save_path"])
    selected_meta = _parse_meta_repr(loaded_meta)
    canonical_meta = dict(selected_meta)
    canonical_meta.update(
        {
            "kind": kind,
            "selected_seed": int(selected_seed),
            "selected_best_val_loss": float(selected_result["best_val_loss"]),
            "selected_final_val_loss": float(selected_result["final_val_loss"]),
            "training_horizon_steps": int(selected_meta.get("cfg", {}).get("horizon_steps", 0)),
            "training_horizon_seconds": float(
                selected_meta.get("cfg", {}).get("horizon_steps", 0)
                * selected_meta.get("cfg", {}).get("dt", 0.0)
            ),
            "lr": float(selected_meta.get("lr", 0.0)),
            "steps_cap": int(selected_meta.get("steps_cap", 0)),
            "candidate_seeds": {
                int(seed): {
                    "best_val_loss": float(result["best_val_loss"]),
                    "final_val_loss": float(result["final_val_loss"]),
                    "steps_completed": int(result["steps_completed"]),
                    "stopped_early": bool(result["stopped_early"]),
                    "job_seed": int(result["job_seed"]),
                    "seed_file": Path(result["save_path"]).name,
                    "validation_metrics": dict(result.get("validation_metrics", {})),
                    "selection_metrics": dict(result.get("selection_metrics", {})),
                }
                for seed, result in sorted(all_seed_results.items())
            },
            "selection_metrics": dict(selected_result.get("selection_metrics", {})),
        }
    )
    prefix = canonical_filename_prefix or f"rl_{kind}"
    canonical_path = out_dir / f"{prefix}__{plant_id}.npz"
    _save_params_npz_quiet(str(canonical_path), params=params, meta=canonical_meta)


if __name__ == "__main__":
    default_mlp_variant = first_order_default_mlp_variant()
    first_order_mlp_variants()  # validate manifest registry at startup
    plants = extract_plant_params()

    episode_seconds = 20.0
    cfg = RLBPTTConfig(
        horizon_steps=int(round(episode_seconds / family.dt)),
        qy=FIRST_ORDER_COST_WEIGHTS.qy,
        ru=FIRST_ORDER_COST_WEIGHTS.ru,
        qu=FIRST_ORDER_COST_WEIGHTS.qu,
        r0=1.0,
        u_min=family.u_min,
        u_max=family.u_max,
    )

    meta_extra = {
        "training_mix": {
            "tracking": 0.50,
            "regulation": 0.25,
            "disturbance_rejection": 0.25,
        },
        "disturbance_rejection": {
            "input_step_amp_uniform": [-0.25, 0.25],
            "input_step_amp_exclude_radius": 0.05,
            "step_start_fraction_range": [0.20, 0.70],
        },
        "default_mlp_variant": dict(default_mlp_variant),
    }

    rich_rollout_mix = dict(default_mlp_variant["rollout_mix"])
    rich_training_noise = dict(default_mlp_variant["training_noise"])
    rich_disturbance_distribution = dict(default_mlp_variant.get("disturbance_distribution", {}))
    rich_domain_randomization = dict(default_mlp_variant["domain_randomization"])
    rich_hidden_pole_randomization = dict(default_mlp_variant.get("hidden_pole_randomization", {}))
    rich_validation_noise = dict(default_mlp_variant["robust_validation"])
    rich_validation_disturbance_cases = tuple(
        (float(case[0]), float(case[1]))
        for case in rich_validation_noise.get("disturbance_cases", ())
    )
    rich_validation_hidden_poles = tuple(float(v) for v in rich_validation_noise.get("hidden_pole_cases", ()))
    rich_selection_cfg = dict(default_mlp_variant["robust_seed_selection"])
    meta_extra["default_mlp_variant"]["selection_weights"] = dict(rich_selection_cfg.get("weights", {}))

    out_dir = Path(__file__).resolve().parent / "results" / "rl_numpy" / "per_plant"
    out_dir_str = str(out_dir)
    plot_dir = Path(__file__).resolve().parent / "results" / "figures" / "rl_numpy"
    rich_feature_set = str(default_mlp_variant["feature_set"])
    spec = MLPPolicySpec(
        input_dim=first_order_mlp_feature_input_dim(rich_feature_set),
        hidden_layers=tuple(default_mlp_variant["hidden_layers"]),
        activation=str(default_mlp_variant["activation"]),
    )
    rich_cfg = replace(cfg, ru=float(default_mlp_variant["cost_overrides"]["ru"]))
    pidfeat_seed_indices = tuple(range(1))
    rich_seed_indices = tuple(range(int(default_mlp_variant["seeds"])))
    rich_grad_clip_norm = default_mlp_variant["grad_clip_norm"]

    jobs = []
    for plant_id, plant in plants.items():
        for seed_index in pidfeat_seed_indices:
            jobs.append(
                {
                    "plant_id": plant_id,
                    "plant": plant,
                    "cfg": cfg,
                    "steps": 100000,
                    "lr": 3e-3,
                    "kind": "pidfeat",
                    "out_dir": out_dir_str,
                    "seed_index": seed_index,
                    "meta_extra": meta_extra,
                    "mlp_spec": None,
                    "eval_every": 100,
                    "early_stop_patience_evals": 10,
                    "early_stop_rel_improve": 0.01,
                    "early_stop_best_val_threshold": 1.0,
                    "grad_clip_norm": None,
                    "save_name": _seeded_pidfeat_save_name(plant_id, seed_index),
                }
            )
        for seed_index in rich_seed_indices:
            jobs.append(
                {
                    "plant_id": plant_id,
                    "plant": plant,
                    "cfg": rich_cfg,
                    "steps": int(default_mlp_variant["steps_cap"]),
                    "lr": float(default_mlp_variant["learning_rate"]),
                    "kind": "rich",
                    "out_dir": out_dir_str,
                    "seed_index": seed_index,
                    "meta_extra": meta_extra,
                    "mlp_spec": spec,
                    "eval_every": 100,
                    "early_stop_patience_evals": 10,
                    "early_stop_rel_improve": 0.01,
                    "early_stop_best_val_threshold": 1.0,
                    "grad_clip_norm": rich_grad_clip_norm,
                    "save_name": _seeded_save_name(
                        str(default_mlp_variant["seed_filename_prefix"]),
                        plant_id,
                        seed_index,
                    ),
                    "rich_feature_set": rich_feature_set,
                    "rich_sampler_kind": str(default_mlp_variant.get("sampler_kind", "current_robust")),
                    "rich_tracking_probability": float(rich_rollout_mix["tracking"]),
                    "rich_regulation_probability": float(rich_rollout_mix["regulation"]),
                    "rich_measurement_noise_std": float(rich_training_noise.get("measurement_noise_std", rich_training_noise.get("measurement_noise_std_low", 0.01))),
                    "rich_disturbance_noise_std": float(rich_training_noise.get("disturbance_noise_std", rich_training_noise.get("disturbance_noise_std_low", 0.01))),
                    "rich_measurement_noise_std_low": float(rich_training_noise.get("measurement_noise_std_low", rich_training_noise.get("measurement_noise_std", 0.01))),
                    "rich_measurement_noise_std_high": float(rich_training_noise.get("measurement_noise_std_high", rich_training_noise.get("measurement_noise_std", 0.01))),
                    "rich_disturbance_noise_std_low": float(rich_training_noise.get("disturbance_noise_std_low", rich_training_noise.get("disturbance_noise_std", 0.01))),
                    "rich_disturbance_noise_std_high": float(rich_training_noise.get("disturbance_noise_std_high", rich_training_noise.get("disturbance_noise_std", 0.01))),
                    "rich_disturbance_step_time_seconds_low": float(rich_disturbance_distribution.get("step_time_seconds_low", 2.0)),
                    "rich_disturbance_step_time_seconds_high": float(rich_disturbance_distribution.get("step_time_seconds_high", 8.0)),
                    "rich_disturbance_step_magnitude_abs_low": float(rich_disturbance_distribution.get("step_magnitude_abs_low", 0.03)),
                    "rich_disturbance_step_magnitude_abs_high": float(rich_disturbance_distribution.get("step_magnitude_abs_high", 0.20)),
                    "rich_disturbance_both_signs": bool(rich_disturbance_distribution.get("both_signs", True)),
                    "rich_pole_rel_range": float(rich_domain_randomization["pole_relative_range"]),
                    "rich_zero_rel_range": float(rich_domain_randomization["zero_relative_range"]),
                    "rich_gain_rel_range": float(rich_domain_randomization["gain_relative_range"]),
                    "rich_hidden_pole_enabled": bool(rich_hidden_pole_randomization.get("enabled", False)),
                    "rich_hidden_pole_low": float(rich_hidden_pole_randomization.get("location_low", 8.0)),
                    "rich_hidden_pole_high": float(rich_hidden_pole_randomization.get("location_high", 20.0)),
                    "rich_validation_sampler_kind": str(rich_validation_noise.get("sampler_kind", "current_robust")),
                    "rich_validation_measurement_noise_std": float(rich_validation_noise["measurement_noise_std"]),
                    "rich_validation_disturbance_noise_std": float(rich_validation_noise["disturbance_noise_std"]),
                    "rich_validation_disturbance_cases": rich_validation_disturbance_cases if rich_validation_disturbance_cases else None,
                    "rich_validation_hidden_poles": rich_validation_hidden_poles if rich_validation_hidden_poles else None,
                }
            )

    max_workers = _default_worker_count(len(jobs))
    max_workers = int(os.environ.get("RL_TRAIN_WORKERS", max_workers))

    print(f"Training {len(jobs)} RL jobs with {max_workers} worker processes...")
    print(f"Training rollout: {cfg.horizon_steps} steps @ dt={family.dt:g} ({cfg.horizon_steps * family.dt:g} s)")

    results: Dict[str, Dict[str, dict]] = {"pidfeat": {}, "rich": {}}
    pidfeat_seed_results: Dict[str, Dict[int, dict]] = {plant_id: {} for plant_id in FIRST_ORDER_PLANT_IDS}
    rich_seed_results: Dict[str, Dict[int, dict]] = {plant_id: {} for plant_id in FIRST_ORDER_PLANT_IDS}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_train_and_save_job, **job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            if result["kind"] == "pidfeat":
                pidfeat_seed_results[result["plant_id"]][result["seed_index"]] = result
                print(
                    "Finished RL "
                    f"({result['kind']}) for plant: {result['plant_id']} | "
                    f"seed={result['seed_index']} | "
                    f"best_val={result['best_val_loss']:.6f} | "
                    f"steps={result['steps_completed']}"
                )
            else:
                rich_seed_results[result["plant_id"]][result["seed_index"]] = result
                print(
                    "Finished RL "
                    f"({result['kind']}) for plant: {result['plant_id']} | "
                    f"seed={result['seed_index']} | "
                    f"best_val={result['best_val_loss']:.6f} | "
                    f"steps={result['steps_completed']}"
                )

    selected_pidfeat_seed_by_plant: Dict[str, int] = {}
    for plant_id in FIRST_ORDER_PLANT_IDS:
        best_seed, best_result = _select_best_seed_results("pidfeat", plant_id, pidfeat_seed_results[plant_id])
        selected_pidfeat_seed_by_plant[plant_id] = best_seed
        results["pidfeat"][plant_id] = best_result
        _write_canonical_seeded_policy(
            kind="pidfeat",
            plant_id=plant_id,
            out_dir=out_dir,
            selected_seed=best_seed,
            selected_result=best_result,
            all_seed_results=pidfeat_seed_results[plant_id],
        )
        print(
            "Selected RL (pidfeat) seed "
            f"{best_seed} for plant: {plant_id} | "
            f"best_val={best_result['best_val_loss']:.6f} | "
            f"reg_tail={best_result['selection_metrics'].get('reg_tail_mean', np.nan):.6f} | "
            f"score={best_result['selection_metrics'].get('selection_score', np.nan):.6f}"
        )

    selected_rich_seed_by_plant: Dict[str, int] = {}
    for plant_id in FIRST_ORDER_PLANT_IDS:
        best_seed, best_result = _select_best_seed_results("rich", plant_id, rich_seed_results[plant_id])
        selected_rich_seed_by_plant[plant_id] = best_seed
        results["rich"][plant_id] = best_result
        _write_canonical_seeded_policy(
            kind="rich",
            plant_id=plant_id,
            out_dir=out_dir,
            selected_seed=best_seed,
            selected_result=best_result,
            all_seed_results=rich_seed_results[plant_id],
            canonical_filename_prefix=str(default_mlp_variant["canonical_filename_prefix"]),
        )
        print(
            "Selected RL (rich) seed "
            f"{best_seed} for plant: {plant_id} | "
            f"best_val={best_result['best_val_loss']:.6f} | "
            f"reg_tail={best_result['selection_metrics'].get('reg_tail_mean', np.nan):.6f} | "
            f"score={best_result['selection_metrics'].get('selection_score', np.nan):.6f}"
        )

    plot_loss_histories(results, out_dir=plot_dir)
    plot_seed_selection_histories(
        pidfeat_seed_results,
        selected_seed_by_plant=selected_pidfeat_seed_by_plant,
        out_dir=plot_dir / "seed_selection",
        title_tag="PID-features",
        file_prefix="rl_pidfeat_seed_selection",
    )
    plot_seed_selection_histories(
        rich_seed_results,
        selected_seed_by_plant=selected_rich_seed_by_plant,
        out_dir=plot_dir / "seed_selection",
        title_tag="MLP rich",
        file_prefix="rl_rich_seed_selection",
    )

    print("\nDone. Saved per-plant policies to:")
    print(out_dir)
    print("Saved loss figures to:")
    print(plot_dir)
    print("Saved seed-selection figures to:")
    print(plot_dir / "seed_selection")
