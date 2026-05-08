from __future__ import annotations

import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import sys
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from control_bench.config import (
    INVERTED_PENDULUM_LINEARIZED,
    INVERTED_PENDULUM_RL_TRAINING,
)
from control_bench.controllers.inverted_pendulum_rl import (
    PENDULUM_PID_INPUT_DIM,
    InvertedPendulumRLBPTTConfig,
    LinearFeaturePolicy,
    MLPPolicy,
    MLPPolicySpec,
    train_one_policy_with_validation,
)
from control_bench.controllers.rl_numpy.training import load_policy_npz
from inverted_pendulum_rl_helpers import (
    PENDULUM_RL_PLANT_ID,
    build_inverted_pendulum_training_plant,
    build_inverted_pendulum_validation_rollouts,
    sample_inverted_pendulum_training_rollout,
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


def _seeded_save_name(kind: str, seed_index: int) -> str:
    return f"rl_{kind}__{PENDULUM_RL_PLANT_ID}__seed{seed_index}.npz"


def _default_worker_count(n_jobs: int) -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(n_jobs, cpu_count - 1 if cpu_count > 1 else 1))


def _train_and_save_job(
    *,
    A: np.ndarray,
    B: np.ndarray,
    dt: float,
    cfg: InvertedPendulumRLBPTTConfig,
    steps: int,
    lr: float,
    kind: str,
    out_dir: str,
    seed_index: int,
    mlp_spec: Optional[MLPPolicySpec],
    eval_every: int,
    early_stop_patience_evals: int,
    early_stop_rel_improve: float,
    early_stop_best_val_threshold: float,
    grad_clip_norm: Optional[float],
    save_name: str,
) -> dict:
    import zlib

    job_seed = (int(seed_index) + int(zlib.crc32(f"{kind}:{PENDULUM_RL_PLANT_ID}".encode("utf-8")))) % (2**32)
    rng = np.random.default_rng(job_seed)

    if kind == "pidfeat":
        policy = LinearFeaturePolicy(PENDULUM_PID_INPUT_DIM, seed=job_seed)
    elif kind == "rich":
        if mlp_spec is None:
            raise ValueError("mlp_spec must be provided for kind='rich'")
        policy = MLPPolicy(mlp_spec, seed=job_seed)
    else:
        raise ValueError(f"Unknown kind: {kind}")

    summary = train_one_policy_with_validation(
        policy=policy,
        A=A,
        B=B,
        dt=dt,
        cfg=cfg,
        steps=steps,
        lr=lr,
        kind=kind,
        episode_sampler=sample_inverted_pendulum_training_rollout,
        validation_rollouts=build_inverted_pendulum_validation_rollouts(cfg),
        eval_every=eval_every,
        early_stop_patience_evals=early_stop_patience_evals,
        early_stop_rel_improve=early_stop_rel_improve,
        early_stop_best_val_threshold=early_stop_best_val_threshold,
        rng=rng,
        grad_clip_norm=grad_clip_norm,
    )

    save_path = str(Path(out_dir) / save_name)
    _save_params_npz_quiet(
        save_path,
        params=policy.params,
        meta={
            "kind": kind,
            "plant_id": PENDULUM_RL_PLANT_ID,
            "final_loss": summary["final_loss"],
            "best_val_loss": summary["best_val_loss"],
            "steps_completed": summary["steps_completed"],
            "stopped_early": summary["stopped_early"],
            "cfg": cfg.__dict__,
            "steps_cap": steps,
            "lr": lr,
            "seed_index": seed_index,
            "job_seed": job_seed,
            "plant_dt": dt,
            "eval_every": eval_every,
            "early_stop_patience_evals": early_stop_patience_evals,
            "early_stop_rel_improve": early_stop_rel_improve,
            "early_stop_best_val_threshold": early_stop_best_val_threshold,
            "grad_clip_norm": grad_clip_norm,
        },
    )

    return {
        "kind": kind,
        "plant_id": PENDULUM_RL_PLANT_ID,
        "final_loss": float(summary["final_loss"]),
        "best_val_loss": float(summary["best_val_loss"]),
        "final_val_loss": float(summary["final_val_loss"]),
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


def plot_loss_histories(results: Dict[str, dict], *, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    eps = 1e-12
    figure_specs = [
        ("pidfeat", "PID-features", out_dir / "rl_pidfeat_loss_curve.png"),
        ("rich", "MLP rich", out_dir / "rl_rich_loss_curve.png"),
    ]

    for kind, title_tag, save_path in figure_specs:
        result = results[kind]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(result["updates"], np.maximum(result["train_loss"], eps), linewidth=1.8, label="train")
        ax.plot(result["updates"], np.maximum(result["val_loss"], eps), linewidth=1.8, label="validation")
        ax.set_yscale("log")
        status = "early stop" if result["stopped_early"] else "cap"
        ax.set_title(f"{PENDULUM_RL_PLANT_ID}\n{status} @ {result['steps_completed']}")
        ax.set_xlabel("Update")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.suptitle(f"Pendulum RL loss curve | {title_tag}")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(save_path, dpi=160)
        plt.close(fig)


def plot_seed_selection_histories(
    seed_results: Dict[int, dict],
    *,
    selected_seed: int,
    out_dir: Path,
    title_tag: str,
    file_prefix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    eps = 1e-12
    fig, ax = plt.subplots(figsize=(8, 5))

    for seed_index in sorted(seed_results):
        result = seed_results[seed_index]
        linewidth = 2.8 if seed_index == selected_seed else 1.8
        alpha = 1.0 if seed_index == selected_seed else 0.75
        label = f"seed {seed_index}" + (" (chosen)" if seed_index == selected_seed else "")
        ax.plot(result["updates"], np.maximum(result["val_loss"], eps), linewidth=linewidth, alpha=alpha, label=label)

    best_result = seed_results[selected_seed]
    ax.set_yscale("log")
    ax.set_title(
        f"{PENDULUM_RL_PLANT_ID}\nchosen seed={selected_seed} | "
        f"best_val={best_result['best_val_loss']:.6f} | stop @ {best_result['steps_completed']}"
    )
    ax.set_xlabel("Update")
    ax.set_ylabel("Validation loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.suptitle(f"Pendulum RL seed selection | {title_tag}", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / f"{file_prefix}__{PENDULUM_RL_PLANT_ID}.png", dpi=160)
    plt.close(fig)


def _select_best_seed_results(seed_results: Dict[int, dict]) -> tuple[int, dict]:
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
    canonical_path = out_dir / f"rl_{kind}__{PENDULUM_RL_PLANT_ID}.npz"
    _save_params_npz_quiet(str(canonical_path), params=params, meta=canonical_meta)


if __name__ == "__main__":
    plant = build_inverted_pendulum_training_plant()
    A = np.asarray(plant.A, dtype=np.float64)
    B = np.asarray(plant.B, dtype=np.float64).reshape(4)

    cfg = InvertedPendulumRLBPTTConfig(
        horizon_steps=int(round(INVERTED_PENDULUM_RL_TRAINING.horizon_seconds / float(plant.dt))),
        q_state_diag=INVERTED_PENDULUM_RL_TRAINING.q_state_diag,
        ru=INVERTED_PENDULUM_RL_TRAINING.ru,
        qu=INVERTED_PENDULUM_RL_TRAINING.qu,
        u_min=INVERTED_PENDULUM_LINEARIZED.u_min,
        u_max=INVERTED_PENDULUM_LINEARIZED.u_max,
    )

    pid_steps = int(os.environ.get("INVERTED_PENDULUM_PID_STEPS", INVERTED_PENDULUM_RL_TRAINING.pid_steps))
    rich_steps = int(os.environ.get("INVERTED_PENDULUM_RICH_STEPS", INVERTED_PENDULUM_RL_TRAINING.rich_steps))
    pid_seed_indices = tuple(range(INVERTED_PENDULUM_RL_TRAINING.pid_seeds))
    rich_seed_indices = tuple(range(INVERTED_PENDULUM_RL_TRAINING.rich_seeds))
    rich_spec = MLPPolicySpec(
        input_dim=6,
        hidden_layers=tuple(INVERTED_PENDULUM_RL_TRAINING.rich_hidden_layers),
        activation=INVERTED_PENDULUM_RL_TRAINING.rich_activation,
    )

    out_dir = EXPERIMENTS_ROOT / "results" / "rl_numpy" / "inverted_pendulum_linearized" / "per_plant"
    plot_dir = EXPERIMENTS_ROOT / "results" / "figures" / "rl_numpy" / "inverted_pendulum_linearized"
    out_dir_str = str(out_dir)

    jobs = []
    for seed_index in pid_seed_indices:
        jobs.append(
            {
                "A": A,
                "B": B,
                "dt": float(plant.dt),
                "cfg": cfg,
                "steps": pid_steps,
                "lr": INVERTED_PENDULUM_RL_TRAINING.pid_lr,
                "kind": "pidfeat",
                "out_dir": out_dir_str,
                "seed_index": seed_index,
                "mlp_spec": None,
                "eval_every": INVERTED_PENDULUM_RL_TRAINING.eval_every,
                "early_stop_patience_evals": INVERTED_PENDULUM_RL_TRAINING.early_stop_patience_evals,
                "early_stop_rel_improve": INVERTED_PENDULUM_RL_TRAINING.early_stop_rel_improve,
                "early_stop_best_val_threshold": INVERTED_PENDULUM_RL_TRAINING.early_stop_best_val_threshold,
                "grad_clip_norm": None,
                "save_name": _seeded_save_name("pidfeat", seed_index),
            }
        )

    for seed_index in rich_seed_indices:
        jobs.append(
            {
                "A": A,
                "B": B,
                "dt": float(plant.dt),
                "cfg": cfg,
                "steps": rich_steps,
                "lr": INVERTED_PENDULUM_RL_TRAINING.rich_lr,
                "kind": "rich",
                "out_dir": out_dir_str,
                "seed_index": seed_index,
                "mlp_spec": rich_spec,
                "eval_every": INVERTED_PENDULUM_RL_TRAINING.eval_every,
                "early_stop_patience_evals": INVERTED_PENDULUM_RL_TRAINING.early_stop_patience_evals,
                "early_stop_rel_improve": INVERTED_PENDULUM_RL_TRAINING.early_stop_rel_improve,
                "early_stop_best_val_threshold": INVERTED_PENDULUM_RL_TRAINING.early_stop_best_val_threshold,
                "grad_clip_norm": INVERTED_PENDULUM_RL_TRAINING.rich_grad_clip_norm,
                "save_name": _seeded_save_name("rich", seed_index),
            }
        )

    max_workers = _default_worker_count(len(jobs))
    max_workers = int(os.environ.get("RL_TRAIN_WORKERS", max_workers))
    print(f"Training {len(jobs)} pendulum RL jobs with {max_workers} worker processes...")
    print(f"Training rollout: {cfg.horizon_steps} steps @ dt={plant.dt:g} ({cfg.horizon_steps * plant.dt:g} s)")

    pidfeat_seed_results: Dict[int, dict] = {}
    rich_seed_results: Dict[int, dict] = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_train_and_save_job, **job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            if result["kind"] == "pidfeat":
                pidfeat_seed_results[result["seed_index"]] = result
            else:
                rich_seed_results[result["seed_index"]] = result
            print(
                f"Finished RL ({result['kind']}) | seed={result['seed_index']} | "
                f"best_val={result['best_val_loss']:.6f} | steps={result['steps_completed']}"
            )

    results: Dict[str, dict] = {}

    pid_best_seed, pid_best_result = _select_best_seed_results(pidfeat_seed_results)
    _write_canonical_seeded_policy(
        kind="pidfeat",
        out_dir=out_dir,
        selected_seed=pid_best_seed,
        selected_result=pid_best_result,
        all_seed_results=pidfeat_seed_results,
    )
    results["pidfeat"] = pid_best_result
    print(f"Selected RL (pidfeat) seed {pid_best_seed} | best_val={pid_best_result['best_val_loss']:.6f}")

    rich_best_seed, rich_best_result = _select_best_seed_results(rich_seed_results)
    _write_canonical_seeded_policy(
        kind="rich",
        out_dir=out_dir,
        selected_seed=rich_best_seed,
        selected_result=rich_best_result,
        all_seed_results=rich_seed_results,
    )
    results["rich"] = rich_best_result
    print(f"Selected RL (rich) seed {rich_best_seed} | best_val={rich_best_result['best_val_loss']:.6f}")

    plot_loss_histories(results, out_dir=plot_dir)
    plot_seed_selection_histories(
        pidfeat_seed_results,
        selected_seed=pid_best_seed,
        out_dir=plot_dir / "seed_selection",
        title_tag="PID-features",
        file_prefix="rl_pidfeat_seed_selection",
    )
    plot_seed_selection_histories(
        rich_seed_results,
        selected_seed=rich_best_seed,
        out_dir=plot_dir / "seed_selection",
        title_tag="MLP rich",
        file_prefix="rl_rich_seed_selection",
    )
    print(f"Saved pendulum RL artifacts to: {out_dir.parent}")
