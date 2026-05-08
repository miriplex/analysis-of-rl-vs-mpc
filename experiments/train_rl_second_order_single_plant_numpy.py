from __future__ import annotations

import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import sys
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from control_bench.config import SECOND_ORDER_FAMILY as family
from control_bench.config import SECOND_ORDER_SINGLE_LIGHTLY_DAMPED as custom_plant_cfg
from control_bench.plants.second_order_family import (
    build_second_order_single_plant,
    build_second_order_suite_plant,
    tf_coeffs_for_plant_id,
)
from control_bench.controllers.rl_numpy import (
    PlantParams,
    RLBPTTConfig,
    LinearPIDFeaturePolicy,
    MLPPolicy,
    MLPPolicySpec,
)
from control_bench.controllers.rl_numpy.training import load_policy_npz, train_one_policy_with_validation
from control_bench.controllers.rl_numpy.types import RolloutSpec


PLANT_IDS = (
    custom_plant_cfg.plant_id,
    "boss_unstable_osc_rhp_zero",
)


def build_experiment_plants() -> Dict[str, object]:
    return {
        custom_plant_cfg.plant_id: build_second_order_single_plant(custom_plant_cfg),
        "boss_unstable_osc_rhp_zero": build_second_order_suite_plant("boss_unstable_osc_rhp_zero", family),
    }


def plant_tf_coeffs(plant_id: str) -> tuple[float, float, float, float]:
    if plant_id == custom_plant_cfg.plant_id:
        return (
            float(custom_plant_cfg.b1),
            float(custom_plant_cfg.b0),
            float(custom_plant_cfg.a1),
            float(custom_plant_cfg.a0),
        )
    return tf_coeffs_for_plant_id(plant_id, family)


def plant_bounds(plant_id: str) -> tuple[Optional[float], Optional[float]]:
    if plant_id == custom_plant_cfg.plant_id:
        return custom_plant_cfg.u_min, custom_plant_cfg.u_max
    return family.u_min, family.u_max


def extract_plant_params() -> Dict[str, PlantParams]:
    plants = build_experiment_plants()
    return {
        plant_id: PlantParams(
            A=plant.A,
            B=plant.B,
            C=plant.C,
            D=plant.D,
            dt=plant.dt,
        )
        for plant_id, plant in plants.items()
    }


def _sample_nonzero_uniform(
    rng: np.random.Generator,
    low: float,
    high: float,
    *,
    exclude_center: Optional[float] = None,
    exclude_radius: float = 0.0,
) -> float:
    while True:
        value = float(rng.uniform(low, high))
        if exclude_center is None:
            if abs(value) > exclude_radius:
                return value
        elif abs(value - exclude_center) > exclude_radius:
            return value


def _sample_nonzero_vector_uniform(
    rng: np.random.Generator,
    low: float,
    high: float,
    *,
    dim: int,
    exclude_norm_radius: float,
) -> np.ndarray:
    while True:
        value = rng.uniform(low, high, size=(dim,)).astype(np.float64)
        if float(np.linalg.norm(value)) > float(exclude_norm_radius):
            return value


def sample_training_rollout(rng: np.random.Generator, cfg: RLBPTTConfig) -> RolloutSpec:
    T = int(cfg.horizon_steps)
    dice = float(rng.random())
    state_dim = 2

    if dice < 0.50:
        r0 = _sample_nonzero_uniform(
            rng,
            -1.8,
            1.8,
            exclude_center=-1.0,
            exclude_radius=0.05,
        )
        x0 = _sample_nonzero_vector_uniform(
            rng,
            -0.5,
            0.5,
            dim=state_dim,
            exclude_norm_radius=0.02,
        )
        return RolloutSpec(
            r=np.full(T, r0, dtype=np.float64),
            du=np.zeros(T, dtype=np.float64),
            x0=x0,
            mode="tracking",
        )

    if dice < 0.75:
        if float(rng.random()) < 0.35:
            x0 = np.zeros((state_dim,), dtype=np.float64)
        else:
            x0 = _sample_nonzero_vector_uniform(
                rng,
                -0.5,
                0.5,
                dim=state_dim,
                exclude_norm_radius=0.02,
            )
        return RolloutSpec(
            r=np.zeros(T, dtype=np.float64),
            du=np.zeros(T, dtype=np.float64),
            x0=x0,
            mode="regulation",
        )

    du = np.zeros(T, dtype=np.float64)
    step_start = int(rng.integers(max(1, int(0.2 * T)), max(2, int(0.7 * T))))
    amp = _sample_nonzero_uniform(rng, -0.25, 0.25, exclude_radius=0.05)
    du[step_start:] = amp
    if float(rng.random()) < 0.80:
        x0 = np.zeros((state_dim,), dtype=np.float64)
    else:
        x0 = _sample_nonzero_vector_uniform(
            rng,
            -0.1,
            0.1,
            dim=state_dim,
            exclude_norm_radius=0.01,
        )
    return RolloutSpec(
        r=np.zeros(T, dtype=np.float64),
        du=du,
        x0=x0,
        mode="disturbance_rejection",
    )


def build_validation_rollouts(cfg: RLBPTTConfig) -> tuple[RolloutSpec, ...]:
    T = int(cfg.horizon_steps)
    zero_state = np.zeros((2,), dtype=np.float64)

    def const_rollout(r0: float, x0: np.ndarray, mode: str) -> RolloutSpec:
        return RolloutSpec(
            r=np.full(T, r0, dtype=np.float64),
            du=np.zeros(T, dtype=np.float64),
            x0=np.asarray(x0, dtype=np.float64),
            mode=mode,
        )

    def disturbance_rollout(start_frac: float, amp: float) -> RolloutSpec:
        du = np.zeros(T, dtype=np.float64)
        start_idx = min(T - 1, max(1, int(round(start_frac * T))))
        du[start_idx:] = amp
        return RolloutSpec(
            r=np.zeros(T, dtype=np.float64),
            du=du,
            x0=zero_state,
            mode="disturbance_rejection",
        )

    return (
        const_rollout(-1.5, zero_state, "tracking"),
        const_rollout(-0.5, zero_state, "tracking"),
        const_rollout(0.5, zero_state, "tracking"),
        const_rollout(1.5, zero_state, "tracking"),
        const_rollout(0.0, zero_state, "regulation"),
        const_rollout(0.0, np.array([0.4, 0.0], dtype=np.float64), "regulation"),
        const_rollout(0.0, np.array([0.0, 0.4], dtype=np.float64), "regulation"),
        disturbance_rollout(0.30, 0.15),
        disturbance_rollout(0.60, -0.15),
    )


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


def _seeded_save_name(kind: str, plant_id: str, seed_index: int) -> str:
    return f"rl_{kind}__{plant_id}__seed{seed_index}.npz"


def _train_and_save_job(
    *,
    plant_id: str,
    plant: PlantParams,
    cfg: RLBPTTConfig,
    steps: int,
    lr: float,
    kind: str,
    out_dir: str,
    seed_index: int,
    meta_extra: Optional[dict],
    mlp_spec: Optional[MLPPolicySpec],
    eval_every: int,
    early_stop_patience_evals: int,
    early_stop_rel_improve: float,
    early_stop_best_val_threshold: float,
    grad_clip_norm: Optional[float],
    save_name: str,
) -> dict:
    import zlib

    job_seed = (int(seed_index) + int(zlib.crc32(f"{kind}:{plant_id}".encode("utf-8")))) % (2**32)
    rng = np.random.default_rng(job_seed)

    if kind == "pidfeat":
        policy = LinearPIDFeaturePolicy(seed=job_seed)
    elif kind == "rich":
        if mlp_spec is None:
            raise ValueError("mlp_spec must be provided for kind='rich'")
        policy = MLPPolicy(mlp_spec, seed=job_seed)
    else:
        raise ValueError(f"Unknown kind: {kind}")

    summary = train_one_policy_with_validation(
        policy=policy,
        plant=plant,
        cfg=cfg,
        steps=steps,
        lr=lr,
        kind=kind,
        episode_sampler=sample_training_rollout,
        validation_rollouts=build_validation_rollouts(cfg),
        eval_every=eval_every,
        early_stop_patience_evals=early_stop_patience_evals,
        early_stop_rel_improve=early_stop_rel_improve,
        early_stop_best_val_threshold=early_stop_best_val_threshold,
        print_every=0,
        rng=rng,
        grad_clip_norm=grad_clip_norm,
    )

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
            "plant_dt": plant.dt,
            "eval_every": eval_every,
            "early_stop_patience_evals": early_stop_patience_evals,
            "early_stop_rel_improve": early_stop_rel_improve,
            "early_stop_best_val_threshold": early_stop_best_val_threshold,
            "grad_clip_norm": grad_clip_norm,
            "meta_extra": extra,
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
    }


def _default_worker_count(n_jobs: int) -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(n_jobs, cpu_count - 1 if cpu_count > 1 else 1))


def plot_loss_histories(results: Dict[str, Dict[str, dict]], *, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    eps = 1e-12

    figure_specs = [
        ("pidfeat", "PID-features", out_dir / "rl_pidfeat_loss_curves.png"),
        ("rich", "MLP rich", out_dir / "rl_rich_loss_curves.png"),
    ]

    for kind, title_tag, save_path in figure_specs:
        fig, axes = plt.subplots(1, len(PLANT_IDS), figsize=(7 * len(PLANT_IDS), 5), sharex=False, sharey=False)
        if len(PLANT_IDS) == 1:
            axes = [axes]

        for ax, plant_id in zip(axes, PLANT_IDS):
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
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(save_path, dpi=160)
        plt.close(fig)


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

    for plant_id in PLANT_IDS:
        seed_results = seed_results_by_plant.get(plant_id, {})
        if not seed_results:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        chosen_seed = selected_seed_by_plant[plant_id]

        for seed_index in sorted(seed_results):
            result = seed_results[seed_index]
            val_loss = np.maximum(result["val_loss"], eps)
            label = f"seed {seed_index}"
            linewidth = 1.8
            alpha = 0.75
            if seed_index == chosen_seed:
                label += " (chosen)"
                linewidth = 2.8
                alpha = 1.0
            ax.plot(result["updates"], val_loss, linewidth=linewidth, alpha=alpha, label=label)

        best_result = seed_results[chosen_seed]
        best_update = int(best_result["updates"][-1]) if len(best_result["updates"]) else 0
        ax.set_yscale("log")
        ax.set_title(
            f"{plant_id}\nchosen seed={chosen_seed} | best_val={best_result['best_val_loss']:.6f} | stop @ {best_update}"
        )
        ax.set_xlabel("Update")
        ax.set_ylabel("Validation loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.suptitle(f"RL seed selection | {title_tag}", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(out_dir / f"{file_prefix}__{plant_id}.png", dpi=160)
        plt.close(fig)


def _select_best_seed_results(seed_results: Dict[int, dict]) -> tuple[int, dict]:
    if not seed_results:
        raise ValueError("seed_results must be non-empty")
    return min(
        seed_results.items(),
        key=lambda item: (
            float(item[1]["best_val_loss"]),
            float(item[1]["final_val_loss"]),
            int(item[0]),
        ),
    )


def _write_canonical_seeded_policy(
    *,
    kind: str,
    plant_id: str,
    out_dir: Path,
    selected_seed: int,
    selected_result: dict,
    all_seed_results: Dict[int, dict],
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
                selected_meta.get("cfg", {}).get("horizon_steps", 0) * selected_meta.get("plant_dt", 0.0)
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
                }
                for seed, result in sorted(all_seed_results.items())
            },
        }
    )
    canonical_path = out_dir / f"rl_{kind}__{plant_id}.npz"
    _save_params_npz_quiet(str(canonical_path), params=params, meta=canonical_meta)


if __name__ == "__main__":
    plants = extract_plant_params()

    out_dir = Path(__file__).resolve().parent / "results" / "rl_numpy" / "second_order_single" / "per_plant"
    out_dir_str = str(out_dir)
    plot_dir = Path(__file__).resolve().parent / "results" / "figures" / "rl_numpy" / "second_order_single"
    spec = MLPPolicySpec(input_dim=11, hidden_layers=(16, 16), activation="tanh")
    pidfeat_seed_indices = tuple(range(1))
    rich_seed_indices = tuple(range(5))
    rich_grad_clip_norm = 1.0

    jobs = []
    for plant_id in PLANT_IDS:
        plant = plants[plant_id]
        u_min, u_max = plant_bounds(plant_id)
        b1, b0, a1, a0 = plant_tf_coeffs(plant_id)
        cfg = RLBPTTConfig(
            horizon_steps=int(round(20.0 / plant.dt)),
            qy=1.0,
            ru=1.0,
            qu=0.01,
            r0=1.0,
            u_min=u_min,
            u_max=u_max,
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
            "plant_tf": {
                "numerator": [b1, b0],
                "denominator": [1.0, a1, a0],
            },
        }

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
                    "save_name": _seeded_save_name("pidfeat", plant_id, seed_index),
                }
            )
        for seed_index in rich_seed_indices:
            jobs.append(
                {
                    "plant_id": plant_id,
                    "plant": plant,
                    "cfg": cfg,
                    "steps": 100000,
                    "lr": 5e-4,
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
                    "save_name": _seeded_save_name("rich", plant_id, seed_index),
                }
            )

    max_workers = _default_worker_count(len(jobs))
    max_workers = int(os.environ.get("RL_TRAIN_WORKERS", max_workers))
    print(f"Training {len(jobs)} RL jobs with {max_workers} worker processes...")

    results: Dict[str, Dict[str, dict]] = {"pidfeat": {}, "rich": {}}
    pidfeat_seed_results: Dict[str, Dict[int, dict]] = {plant_id: {} for plant_id in PLANT_IDS}
    rich_seed_results: Dict[str, Dict[int, dict]] = {plant_id: {} for plant_id in PLANT_IDS}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_train_and_save_job, **job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            if result["kind"] == "pidfeat":
                pidfeat_seed_results[result["plant_id"]][result["seed_index"]] = result
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
    for plant_id in PLANT_IDS:
        best_seed, best_result = _select_best_seed_results(pidfeat_seed_results[plant_id])
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
            f"best_val={best_result['best_val_loss']:.6f}"
        )

    selected_rich_seed_by_plant: Dict[str, int] = {}
    for plant_id in PLANT_IDS:
        best_seed, best_result = _select_best_seed_results(rich_seed_results[plant_id])
        selected_rich_seed_by_plant[plant_id] = best_seed
        results["rich"][plant_id] = best_result
        _write_canonical_seeded_policy(
            kind="rich",
            plant_id=plant_id,
            out_dir=out_dir,
            selected_seed=best_seed,
            selected_result=best_result,
            all_seed_results=rich_seed_results[plant_id],
        )
        print(
            "Selected RL (rich) seed "
            f"{best_seed} for plant: {plant_id} | "
            f"best_val={best_result['best_val_loss']:.6f}"
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

    print("\nDone. Saved second-order per-plant policies to:")
    print(out_dir)
    print("Saved second-order loss figures to:")
    print(plot_dir)
    print("Saved second-order seed-selection figures to:")
    print(plot_dir / "seed_selection")
