from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from control_bench.config import FIRST_ORDER_COST_WEIGHTS, FIRST_ORDER_FAMILY as family, FIRST_ORDER_PLANT_IDS
from control_bench.controllers.rl_numpy import MLPPolicySpec, RLBPTTConfig
from control_bench.controllers.rl_numpy.feature_sets import first_order_mlp_feature_input_dim
from control_bench.experiment_manifest import first_order_default_mlp_variant, first_order_mlp_variants
from train_rl_first_order_numpy import (
    _default_worker_count,
    _seeded_pidfeat_save_name,
    _seeded_save_name,
    _select_best_seed_results,
    _train_and_save_job,
    _write_canonical_seeded_policy,
    plot_seed_selection_histories,
)
from first_order_rl_helpers import extract_first_order_plant_params


def _plot_variant_loss_histories(
    *,
    results_by_plant: Dict[str, dict],
    plant_ids: List[str],
    title_tag: str,
    save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    eps = 1e-12
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=False, sharey=False)
    axes = axes.reshape(-1)

    for ax, plant_id in zip(axes, plant_ids):
        result = results_by_plant[plant_id]
        train_loss = np.maximum(np.asarray(result["train_loss"], dtype=float), eps)
        val_loss = np.maximum(np.asarray(result["val_loss"], dtype=float), eps)
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
    plt.close(fig)


def _pidfeat_meta_extra(default_mlp_variant: dict) -> dict:
    return {
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


def _mlp_variant_meta_extra(variant: dict) -> dict:
    return {
        "training_mix": dict(variant["rollout_mix"]),
        "training_noise": dict(variant["training_noise"]),
        "domain_randomization": dict(variant["domain_randomization"]),
        "robust_validation": dict(variant["robust_validation"]),
        "robust_seed_selection": dict(variant["robust_seed_selection"]),
        "mlp_variant": dict(variant),
    }


def main() -> None:
    plants = extract_first_order_plant_params()
    plant_ids = list(FIRST_ORDER_PLANT_IDS)
    default_mlp_variant = first_order_default_mlp_variant()
    mlp_variants = first_order_mlp_variants()

    episode_seconds = 20.0
    base_cfg = RLBPTTConfig(
        horizon_steps=int(round(episode_seconds / family.dt)),
        qy=FIRST_ORDER_COST_WEIGHTS.qy,
        ru=FIRST_ORDER_COST_WEIGHTS.ru,
        qu=FIRST_ORDER_COST_WEIGHTS.qu,
        r0=1.0,
        u_min=family.u_min,
        u_max=family.u_max,
    )

    out_dir = EXPERIMENTS_ROOT / "results" / "rl_numpy" / "per_plant"
    plot_dir = EXPERIMENTS_ROOT / "results" / "figures" / "rl_numpy" / "manifest_variants"
    out_dir_str = str(out_dir)

    jobs = []
    pid_meta_extra = _pidfeat_meta_extra(default_mlp_variant)
    for plant_id, plant in plants.items():
        jobs.append(
            {
                "variant_key": "pidfeat",
                "label": "RL (PID-features)",
                "job": {
                    "plant_id": plant_id,
                    "plant": plant,
                    "cfg": base_cfg,
                    "steps": 100000,
                    "lr": 3e-3,
                    "kind": "pidfeat",
                    "out_dir": out_dir_str,
                    "seed_index": 0,
                    "meta_extra": pid_meta_extra,
                    "mlp_spec": None,
                    "eval_every": 100,
                    "early_stop_patience_evals": 10,
                    "early_stop_rel_improve": 0.01,
                    "early_stop_best_val_threshold": 1.0,
                    "grad_clip_norm": None,
                    "save_name": _seeded_pidfeat_save_name(plant_id, 0),
                },
            }
        )

        for variant in mlp_variants:
            feature_set = str(variant["feature_set"])
            spec = MLPPolicySpec(
                input_dim=first_order_mlp_feature_input_dim(feature_set),
                hidden_layers=tuple(int(v) for v in variant["hidden_layers"]),
                activation=str(variant["activation"]),
            )
            variant_cfg = RLBPTTConfig(
                horizon_steps=base_cfg.horizon_steps,
                qy=base_cfg.qy,
                ru=float(variant["cost_overrides"]["ru"]),
                qu=base_cfg.qu,
                r0=base_cfg.r0,
                u_min=base_cfg.u_min,
                u_max=base_cfg.u_max,
            )
            variant_meta_extra = _mlp_variant_meta_extra(variant)
            rollout_mix = dict(variant["rollout_mix"])
            training_noise = dict(variant["training_noise"])
            disturbance_distribution = dict(variant.get("disturbance_distribution", {}))
            domain_randomization = dict(variant["domain_randomization"])
            hidden_pole_randomization = dict(variant.get("hidden_pole_randomization", {}))
            validation_noise = dict(variant["robust_validation"])
            validation_disturbance_cases = tuple(
                (float(case[0]), float(case[1]))
                for case in validation_noise.get("disturbance_cases", ())
            )
            validation_hidden_poles = tuple(float(v) for v in validation_noise.get("hidden_pole_cases", ()))

            for seed_index in range(int(variant["seeds"])):
                jobs.append(
                    {
                        "variant_key": str(variant["id"]),
                        "label": str(variant["label"]),
                        "job": {
                            "plant_id": plant_id,
                            "plant": plant,
                            "cfg": variant_cfg,
                            "steps": int(variant["steps_cap"]),
                            "lr": float(variant["learning_rate"]),
                            "kind": "rich",
                            "out_dir": out_dir_str,
                            "seed_index": seed_index,
                            "meta_extra": variant_meta_extra,
                            "mlp_spec": spec,
                            "eval_every": 100,
                            "early_stop_patience_evals": 10,
                            "early_stop_rel_improve": 0.01,
                            "early_stop_best_val_threshold": 1.0,
                            "grad_clip_norm": variant["grad_clip_norm"],
                            "save_name": _seeded_save_name(str(variant["seed_filename_prefix"]), plant_id, seed_index),
                            "rich_feature_set": feature_set,
                            "rich_sampler_kind": str(variant.get("sampler_kind", "current_robust")),
                            "rich_tracking_probability": float(rollout_mix["tracking"]),
                            "rich_regulation_probability": float(rollout_mix["regulation"]),
                            "rich_measurement_noise_std": float(training_noise.get("measurement_noise_std", training_noise.get("measurement_noise_std_low", 0.01))),
                            "rich_disturbance_noise_std": float(training_noise.get("disturbance_noise_std", training_noise.get("disturbance_noise_std_low", 0.01))),
                            "rich_measurement_noise_std_low": float(training_noise.get("measurement_noise_std_low", training_noise.get("measurement_noise_std", 0.01))),
                            "rich_measurement_noise_std_high": float(training_noise.get("measurement_noise_std_high", training_noise.get("measurement_noise_std", 0.01))),
                            "rich_disturbance_noise_std_low": float(training_noise.get("disturbance_noise_std_low", training_noise.get("disturbance_noise_std", 0.01))),
                            "rich_disturbance_noise_std_high": float(training_noise.get("disturbance_noise_std_high", training_noise.get("disturbance_noise_std", 0.01))),
                            "rich_disturbance_step_time_seconds_low": float(disturbance_distribution.get("step_time_seconds_low", 2.0)),
                            "rich_disturbance_step_time_seconds_high": float(disturbance_distribution.get("step_time_seconds_high", 8.0)),
                            "rich_disturbance_step_magnitude_abs_low": float(disturbance_distribution.get("step_magnitude_abs_low", 0.03)),
                            "rich_disturbance_step_magnitude_abs_high": float(disturbance_distribution.get("step_magnitude_abs_high", 0.20)),
                            "rich_disturbance_both_signs": bool(disturbance_distribution.get("both_signs", True)),
                            "rich_pole_rel_range": float(domain_randomization["pole_relative_range"]),
                            "rich_zero_rel_range": float(domain_randomization["zero_relative_range"]),
                            "rich_gain_rel_range": float(domain_randomization["gain_relative_range"]),
                            "rich_hidden_pole_enabled": bool(hidden_pole_randomization.get("enabled", False)),
                            "rich_hidden_pole_low": float(hidden_pole_randomization.get("location_low", 8.0)),
                            "rich_hidden_pole_high": float(hidden_pole_randomization.get("location_high", 20.0)),
                            "rich_validation_sampler_kind": str(validation_noise.get("sampler_kind", "current_robust")),
                            "rich_validation_measurement_noise_std": float(validation_noise["measurement_noise_std"]),
                            "rich_validation_disturbance_noise_std": float(validation_noise["disturbance_noise_std"]),
                            "rich_validation_disturbance_cases": validation_disturbance_cases if validation_disturbance_cases else None,
                            "rich_validation_hidden_poles": validation_hidden_poles if validation_hidden_poles else None,
                        },
                    }
                )

    max_workers = _default_worker_count(len(jobs))
    max_workers = int(os.environ.get("RL_TRAIN_WORKERS", max_workers))

    print(f"Training {len(jobs)} manifest-backed first-order RL jobs with {max_workers} worker processes...")
    print(f"Training rollout: {base_cfg.horizon_steps} steps @ dt={family.dt:g} ({base_cfg.horizon_steps * family.dt:g} s)")

    results_by_variant: Dict[str, Dict[str, dict]] = {"pidfeat": {}}
    seed_results_by_variant: Dict[str, Dict[str, Dict[int, dict]]] = {"pidfeat": {plant_id: {} for plant_id in plant_ids}}
    variant_meta: Dict[str, dict] = {"pidfeat": {"label": "RL (PID-features)", "canonical_filename_prefix": "rl_pidfeat"}}
    for variant in mlp_variants:
        variant_key = str(variant["id"])
        results_by_variant[variant_key] = {}
        seed_results_by_variant[variant_key] = {plant_id: {} for plant_id in plant_ids}
        variant_meta[variant_key] = dict(variant)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_train_and_save_job, **entry["job"]) for entry in jobs]
        for future, entry in zip(futures, jobs):
            future.variant_key = entry["variant_key"]  # type: ignore[attr-defined]
            future.variant_label = entry["label"]  # type: ignore[attr-defined]

        for future in as_completed(futures):
            result = future.result()
            variant_key = future.variant_key  # type: ignore[attr-defined]
            variant_label = future.variant_label  # type: ignore[attr-defined]
            seed_results_by_variant[variant_key][result["plant_id"]][result["seed_index"]] = result
            print(
                "Finished RL "
                f"({variant_label}) for plant: {result['plant_id']} | "
                f"seed={result['seed_index']} | "
                f"best_val={result['best_val_loss']:.6f} | "
                f"steps={result['steps_completed']}"
            )

    for variant_key, plant_seed_map in seed_results_by_variant.items():
        meta = variant_meta[variant_key]
        label = str(meta["label"])
        canonical_prefix = str(meta["canonical_filename_prefix"])
        selected_seed_by_plant: Dict[str, int] = {}

        for plant_id in plant_ids:
            kind = "pidfeat" if variant_key == "pidfeat" else "rich"
            best_seed, best_result = _select_best_seed_results(kind, plant_id, plant_seed_map[plant_id])
            selected_seed_by_plant[plant_id] = best_seed
            results_by_variant[variant_key][plant_id] = best_result
            _write_canonical_seeded_policy(
                kind=kind,
                plant_id=plant_id,
                out_dir=out_dir,
                selected_seed=best_seed,
                selected_result=best_result,
                all_seed_results=plant_seed_map[plant_id],
                canonical_filename_prefix=canonical_prefix,
            )
            print(
                f"Selected {label} seed {best_seed} for plant: {plant_id} | "
                f"best_val={best_result['best_val_loss']:.6f}"
            )

        file_prefix = canonical_prefix
        _plot_variant_loss_histories(
            results_by_plant=results_by_variant[variant_key],
            plant_ids=plant_ids,
            title_tag=label,
            save_path=plot_dir / f"{file_prefix}_loss_curves.png",
        )
        plot_seed_selection_histories(
            plant_seed_map,
            selected_seed_by_plant=selected_seed_by_plant,
            out_dir=plot_dir / "seed_selection",
            title_tag=label,
            file_prefix=f"{file_prefix}_seed_selection",
        )

    print("\nDone. Saved per-plant policies to:")
    print(out_dir)
    print("Saved manifest-backed loss figures to:")
    print(plot_dir)
    print("Saved manifest-backed seed-selection figures to:")
    print(plot_dir / "seed_selection")


if __name__ == "__main__":
    main()
