from __future__ import annotations

import argparse
import ast
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import sys
import zlib
from typing import Dict, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from control_bench.config import INVERTED_PENDULUM_LINEARIZED, INVERTED_PENDULUM_RL_TRAINING
from control_bench.controllers.inverted_pendulum_rl import (
    InvertedPendulumRLBPTTConfig,
    LinearFeaturePolicy,
    MLPPolicy,
    PENDULUM_PID_INPUT_DIM,
    build_terminal_cost_matrix,
    train_one_policy_with_validation,
)
from control_bench.controllers.rl_numpy.training import load_policy_npz
from inverted_pendulum_rl_helpers import (
    PENDULUM_RL_PLANT_ID,
    build_inverted_pendulum_training_plant,
    build_inverted_pendulum_validation_rollouts,
    sample_inverted_pendulum_training_rollout,
)
from inverted_pendulum_rl_variants import (
    PLANT_ID,
    VARIANT_METRICS_DIR,
    VARIANT_WEIGHTS_DIR,
    PendulumRLVariant,
    resolve_variants,
    variant_by_id,
)


def _save_params_npz_quiet(path: str, *, params: dict, meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {**params}
    payload["_meta"] = np.array([repr(meta)], dtype=object)
    np.savez(path, **payload)


def _parse_meta_repr(meta: dict) -> dict:
    meta_repr = meta.get("_meta_repr")
    if not meta_repr:
        return {}
    try:
        parsed = ast.literal_eval(meta_repr)
    except (SyntaxError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _default_worker_count(n_jobs: int) -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(n_jobs, cpu_count - 1 if cpu_count > 1 else 1))


def _seeded_job_name(variant_id: str, seed_index: int) -> str:
    return f"rl_{variant_id}__{PLANT_ID}__seed{seed_index}.npz"


def _train_variant_seed_job(
    *,
    variant_id: str,
    seed_index: int,
    steps_override: Optional[int],
) -> dict:
    variant = variant_by_id(variant_id)
    plant = build_inverted_pendulum_training_plant()
    A = np.asarray(plant.A, dtype=np.float64).reshape(4, 4)
    B = np.asarray(plant.B, dtype=np.float64).reshape(4)
    cfg = InvertedPendulumRLBPTTConfig(
        horizon_steps=int(round(INVERTED_PENDULUM_RL_TRAINING.horizon_seconds / float(plant.dt))),
        q_state_diag=INVERTED_PENDULUM_RL_TRAINING.q_state_diag,
        ru=INVERTED_PENDULUM_RL_TRAINING.ru,
        qu=INVERTED_PENDULUM_RL_TRAINING.qu,
        u_min=INVERTED_PENDULUM_LINEARIZED.u_min,
        u_max=INVERTED_PENDULUM_LINEARIZED.u_max,
        use_terminal_cost=bool(INVERTED_PENDULUM_RL_TRAINING.use_terminal_cost),
    )
    terminal_P_aug = build_terminal_cost_matrix(A=A, B=B.reshape(4, 1), cfg=cfg)
    cfg = InvertedPendulumRLBPTTConfig(
        horizon_steps=cfg.horizon_steps,
        q_state_diag=cfg.q_state_diag,
        ru=cfg.ru,
        qu=cfg.qu,
        u_min=cfg.u_min,
        u_max=cfg.u_max,
        use_terminal_cost=cfg.use_terminal_cost,
        terminal_riccati_iters=cfg.terminal_riccati_iters,
        terminal_riccati_tol=cfg.terminal_riccati_tol,
        terminal_P_aug=terminal_P_aug,
    )

    job_seed = (int(seed_index) + int(zlib.crc32(f"{variant.variant_id}:{PENDULUM_RL_PLANT_ID}".encode("utf-8")))) % (
        2**32
    )
    steps = int(steps_override if steps_override is not None else variant.steps)
    rng = np.random.default_rng(job_seed)

    if variant.is_pid:
        policy = LinearFeaturePolicy(PENDULUM_PID_INPUT_DIM, seed=job_seed)
    else:
        mlp_spec = variant.mlp_spec()
        if mlp_spec is None:
            raise ValueError(f"MLP variant {variant.variant_id} is missing mlp_spec")
        policy = MLPPolicy(mlp_spec, seed=job_seed)

    summary = train_one_policy_with_validation(
        policy=policy,
        A=A,
        B=B,
        dt=float(plant.dt),
        cfg=cfg,
        steps=steps,
        lr=variant.lr,
        kind=variant.kind,
        episode_sampler=sample_inverted_pendulum_training_rollout,
        validation_rollouts=build_inverted_pendulum_validation_rollouts(cfg),
        eval_every=INVERTED_PENDULUM_RL_TRAINING.eval_every,
        early_stop_patience_evals=INVERTED_PENDULUM_RL_TRAINING.early_stop_patience_evals,
        early_stop_rel_improve=INVERTED_PENDULUM_RL_TRAINING.early_stop_rel_improve,
        early_stop_best_val_threshold=INVERTED_PENDULUM_RL_TRAINING.early_stop_best_val_threshold,
        rng=rng,
        grad_clip_norm=variant.grad_clip_norm,
    )

    save_path = variant.seeded_weight_path(seed_index)
    _save_params_npz_quiet(
        str(save_path),
        params=policy.params,
        meta={
            "variant_id": variant.variant_id,
            "display_name": variant.display_name,
            "kind": variant.kind,
            "hidden_layers": variant.hidden_layers,
            "activation": variant.activation,
            "plant_id": PENDULUM_RL_PLANT_ID,
            "final_loss": summary["final_loss"],
            "best_val_loss": summary["best_val_loss"],
            "final_val_loss": summary["final_val_loss"],
            "steps_completed": summary["steps_completed"],
            "stopped_early": summary["stopped_early"],
            "cfg": cfg.__dict__,
            "steps_cap": steps,
            "lr": variant.lr,
            "seed_index": seed_index,
            "job_seed": job_seed,
            "plant_dt": float(plant.dt),
            "eval_every": INVERTED_PENDULUM_RL_TRAINING.eval_every,
            "early_stop_patience_evals": INVERTED_PENDULUM_RL_TRAINING.early_stop_patience_evals,
            "early_stop_rel_improve": INVERTED_PENDULUM_RL_TRAINING.early_stop_rel_improve,
            "early_stop_best_val_threshold": INVERTED_PENDULUM_RL_TRAINING.early_stop_best_val_threshold,
            "grad_clip_norm": variant.grad_clip_norm,
            "validation_metrics": dict(summary["validation_metrics"]),
        },
    )

    return {
        "variant_id": variant.variant_id,
        "display_name": variant.display_name,
        "kind": variant.kind,
        "seed_index": int(seed_index),
        "job_seed": int(job_seed),
        "steps_cap": int(steps),
        "final_loss": float(summary["final_loss"]),
        "best_val_loss": float(summary["best_val_loss"]),
        "final_val_loss": float(summary["final_val_loss"]),
        "validation_metrics": dict(summary["validation_metrics"]),
        "steps_completed": int(summary["steps_completed"]),
        "stopped_early": bool(summary["stopped_early"]),
        "save_path": str(save_path),
    }


def _select_best_seed_results(seed_results: Dict[int, dict]) -> tuple[int, dict]:
    return min(
        seed_results.items(),
        key=lambda item: (
            float(item[1]["best_val_loss"]),
            float(item[1]["final_val_loss"]),
            int(item[0]),
        ),
    )


def _write_canonical_policy(
    *,
    variant: PendulumRLVariant,
    selected_seed: int,
    selected_result: dict,
    all_seed_results: Dict[int, dict],
) -> None:
    params, loaded_meta = load_policy_npz(selected_result["save_path"])
    selected_meta = _parse_meta_repr(loaded_meta)
    canonical_meta = dict(selected_meta)
    canonical_meta.update(
        {
            "variant_id": variant.variant_id,
            "display_name": variant.display_name,
            "kind": variant.kind,
            "hidden_layers": variant.hidden_layers,
            "activation": variant.activation,
            "selected_seed": int(selected_seed),
            "selected_best_val_loss": float(selected_result["best_val_loss"]),
            "selected_final_val_loss": float(selected_result["final_val_loss"]),
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
    _save_params_npz_quiet(str(variant.canonical_weight_path), params=params, meta=canonical_meta)


def _write_summary_files(summary_rows: list[dict]) -> None:
    VARIANT_METRICS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = VARIANT_METRICS_DIR / "training_summary.csv"
    md_path = VARIANT_METRICS_DIR / "training_summary.md"

    fieldnames = [
        "variant_id",
        "display_name",
        "kind",
        "hidden_layers",
        "selected_seed",
        "best_val_loss",
        "final_val_loss",
        "steps_completed",
        "stopped_early",
        "weight_path",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    lines = [
        "# Pendulum RL Variant Training Summary",
        "",
        "| Variant | Kind | Hidden | Best val | Final val | Steps | Early stop |",
        "| --- | --- | --- | ---: | ---: | ---: | :---: |",
    ]
    for row in summary_rows:
        hidden = row["hidden_layers"] if row["hidden_layers"] else "--"
        lines.append(
            f"| {row['display_name']} | {row['kind']} | {hidden} | "
            f"{row['best_val_loss']:.6f} | {row['final_val_loss']:.6f} | "
            f"{row['steps_completed']} | {'yes' if row['stopped_early'] else 'no'} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train inverted-pendulum RL variant sweep.")
    parser.add_argument(
        "--variants",
        type=str,
        default="all",
        help="Comma-separated variant ids to train, or 'all'.",
    )
    parser.add_argument(
        "--steps-override",
        type=int,
        default=None,
        help="Override training steps for every variant. Useful for smoke tests.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker process count. Defaults to a CPU-aware choice.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if the canonical variant weight file already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    selected_ids = None if args.variants == "all" else tuple(v.strip() for v in args.variants.split(",") if v.strip())
    variants = resolve_variants(selected_ids)

    jobs: list[dict] = []
    for variant in variants:
        if variant.canonical_weight_path.exists() and not args.force:
            print(f"Skipping {variant.variant_id}: canonical weights already exist at {variant.canonical_weight_path}")
            continue
        for seed_index in range(variant.seeds):
            jobs.append(
                {
                    "variant_id": variant.variant_id,
                    "seed_index": seed_index,
                    "steps_override": args.steps_override,
                }
            )

    if not jobs:
        print("No training jobs to run.")
        return

    VARIANT_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    max_workers = _default_worker_count(len(jobs)) if args.workers is None else max(1, int(args.workers))
    print(f"Training {len(jobs)} pendulum RL jobs with {max_workers} worker processes...")

    results_by_variant: Dict[str, Dict[int, dict]] = {variant.variant_id: {} for variant in variants}
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_train_variant_seed_job, **job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results_by_variant[result["variant_id"]][result["seed_index"]] = result
            print(
                f"Finished {result['variant_id']} | seed={result['seed_index']} | "
                f"best_val={result['best_val_loss']:.6f} | steps={result['steps_completed']}"
            )

    summary_rows: list[dict] = []
    for variant in variants:
        seed_results = results_by_variant.get(variant.variant_id, {})
        if not seed_results:
            if variant.canonical_weight_path.exists():
                summary_rows.append(
                    {
                        "variant_id": variant.variant_id,
                        "display_name": variant.display_name,
                        "kind": variant.kind,
                        "hidden_layers": variant.hidden_layers,
                        "selected_seed": "existing",
                        "best_val_loss": float("nan"),
                        "final_val_loss": float("nan"),
                        "steps_completed": 0,
                        "stopped_early": False,
                        "weight_path": str(variant.canonical_weight_path),
                    }
                )
            continue

        best_seed, best_result = _select_best_seed_results(seed_results)
        _write_canonical_policy(
            variant=variant,
            selected_seed=best_seed,
            selected_result=best_result,
            all_seed_results=seed_results,
        )
        summary_rows.append(
            {
                "variant_id": variant.variant_id,
                "display_name": variant.display_name,
                "kind": variant.kind,
                "hidden_layers": variant.hidden_layers,
                "selected_seed": best_seed,
                "best_val_loss": best_result["best_val_loss"],
                "final_val_loss": best_result["final_val_loss"],
                "steps_completed": best_result["steps_completed"],
                "stopped_early": best_result["stopped_early"],
                "weight_path": str(variant.canonical_weight_path),
            }
        )
        print(
            f"Selected {variant.variant_id} seed {best_seed} | "
            f"best_val={best_result['best_val_loss']:.6f} | saved {variant.canonical_weight_path.name}"
        )

    summary_rows.sort(key=lambda row: row["variant_id"])
    _write_summary_files(summary_rows)
    print(f"Saved variant weights to: {VARIANT_WEIGHTS_DIR}")
    print(f"Saved training summary to: {VARIANT_METRICS_DIR}")


if __name__ == "__main__":
    main()
