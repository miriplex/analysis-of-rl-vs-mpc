from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import zlib

import numpy as np

from .bptt import (
    bptt_rollout_and_grads_pidfeat,
    bptt_rollout_and_grads_rich,
    rollout_loss_pidfeat,
    rollout_loss_rich,
    rollout_metrics_pidfeat,
    rollout_metrics_rich,
)
from .optim import Adam
from .types import PlantParams, RLBPTTConfig, RolloutSpec, Params, Grads


def save_policy_npz(path: str, *, policy, meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {**policy.params}
    payload["_meta"] = np.array([repr(meta)], dtype=object)
    np.savez(path, **payload)
    print(f"Saved: {path}")


def load_policy_npz(path: str) -> Tuple[Dict[str, np.ndarray], dict]:
    data = np.load(path, allow_pickle=True)
    params = {k: data[k] for k in data.files if k != "_meta"}
    meta = {}
    if "_meta" in data.files:
        meta_str = str(data["_meta"][0])
        meta = {"_meta_repr": meta_str}
    return params, meta


def _clip_gradients(grads: Grads, max_norm: Optional[float]) -> Grads:
    if max_norm is None:
        return grads
    if max_norm <= 0:
        raise ValueError("max_norm must be > 0 when gradient clipping is enabled")

    total_sq_norm = 0.0
    for grad in grads.values():
        total_sq_norm += float(np.sum(np.square(grad)))

    total_norm = float(np.sqrt(total_sq_norm))
    if total_norm <= max_norm:
        return grads

    scale = float(max_norm / (total_norm + 1e-12))
    return {key: grad * scale for key, grad in grads.items()}


def train_one_policy(
    *,
    policy,
    plant: PlantParams,
    cfg: RLBPTTConfig,
    steps: int,
    lr: float,
    kind: str,  # "pidfeat" or "rich"
    print_every: int = 0,
    rng: Optional[np.random.Generator] = None,
    r0_sampler: Optional[Callable[[np.random.Generator], float]] = None,
    x0_sampler: Optional[Callable[[np.random.Generator], Any]] = None,
    grad_clip_norm: Optional[float] = None,
) -> float:
    """
    Train a single policy on a single plant using full BPTT gradients.
    Returns final loss.
    """
    opt = Adam(lr=lr)
    last_loss = 0.0

    if (r0_sampler is not None) or (x0_sampler is not None):
        rng = np.random.default_rng(0) if rng is None else rng

    for it in range(steps):
        if r0_sampler is None:
            cfg_it = cfg
        else:
            r0 = float(r0_sampler(rng))
            cfg_it = replace(cfg, r0=r0)

        if x0_sampler is None:
            x0 = np.zeros((plant.state_dim,), dtype=np.float64)
        else:
            x0 = np.asarray(x0_sampler(rng), dtype=np.float64).reshape(-1)

        if kind == "pidfeat":
            loss, grads = bptt_rollout_and_grads_pidfeat(policy=policy, plant=plant, cfg=cfg_it, x0=x0)
        elif kind == "rich":
            loss, grads = bptt_rollout_and_grads_rich(policy=policy, plant=plant, cfg=cfg_it, x0=x0)
        else:
            raise ValueError(f"Unknown kind: {kind}")

        grads = _clip_gradients(grads, grad_clip_norm)
        opt.step(policy.params, grads)
        last_loss = loss

        if print_every and ((it + 1) % print_every == 0):
            print(f"  [{it+1:5d}/{steps}] loss={loss:.6f}")

    return float(last_loss)


def evaluate_policy(
    *,
    policy,
    plant: PlantParams,
    cfg: RLBPTTConfig,
    kind: str,
    rollout: RolloutSpec,
    rich_feature_set: str = "rich11",
) -> float:
    plant_eval = rollout.plant_override if rollout.plant_override is not None else plant
    if kind == "pidfeat":
        return rollout_loss_pidfeat(
            policy=policy,
            plant=plant_eval,
            cfg=cfg,
            x0=rollout.x0,
            r_seq=rollout.r,
            du_seq=rollout.du,
            noise_y_seq=rollout.noise_y,
        )
    if kind == "rich":
        return rollout_loss_rich(
            policy=policy,
            plant=plant_eval,
            cfg=cfg,
            x0=rollout.x0,
            r_seq=rollout.r,
            du_seq=rollout.du,
            noise_y_seq=rollout.noise_y,
            feature_set=rich_feature_set,
        )
    raise ValueError(f"Unknown kind: {kind}")


def evaluate_policy_metrics(
    *,
    policy,
    plant: PlantParams,
    cfg: RLBPTTConfig,
    kind: str,
    rollout: RolloutSpec,
    tail_fraction: float = 0.25,
    rich_feature_set: str = "rich11",
) -> dict:
    plant_eval = rollout.plant_override if rollout.plant_override is not None else plant
    if kind == "pidfeat":
        return rollout_metrics_pidfeat(
            policy=policy,
            plant=plant_eval,
            cfg=cfg,
            x0=rollout.x0,
            r_seq=rollout.r,
            du_seq=rollout.du,
            noise_y_seq=rollout.noise_y,
            tail_fraction=tail_fraction,
        )
    if kind == "rich":
        return rollout_metrics_rich(
            policy=policy,
            plant=plant_eval,
            cfg=cfg,
            x0=rollout.x0,
            r_seq=rollout.r,
            du_seq=rollout.du,
            noise_y_seq=rollout.noise_y,
            tail_fraction=tail_fraction,
            feature_set=rich_feature_set,
        )
    raise ValueError(f"Unknown kind: {kind}")


def summarize_validation_metrics(
    *,
    policy,
    plant: PlantParams,
    cfg: RLBPTTConfig,
    kind: str,
    validation_rollouts: Sequence[RolloutSpec],
    tail_fraction: float = 0.25,
    rich_feature_set: str = "rich11",
) -> dict:
    grouped_losses = {
        "tracking": [],
        "regulation": [],
        "disturbance_rejection": [],
        "lag_mismatch": [],
    }
    regulation_tail = []
    disturbance_tail = []
    lag_tail = []
    rollout_metrics = []

    for rollout in validation_rollouts:
        metrics = evaluate_policy_metrics(
            policy=policy,
            plant=plant,
            cfg=cfg,
            kind=kind,
            rollout=rollout,
            tail_fraction=tail_fraction,
            rich_feature_set=rich_feature_set,
        )
        grouped_losses.setdefault(rollout.mode, []).append(float(metrics["mean_loss"]))
        if rollout.mode == "regulation":
            regulation_tail.append(float(metrics["tail_mse"]))
        if rollout.mode == "disturbance_rejection":
            disturbance_tail.append(float(metrics["tail_mse"]))
        if rollout.mode == "lag_mismatch":
            lag_tail.append(float(metrics["tail_mse"]))
        rollout_metrics.append(
            {
                "mode": rollout.mode,
                "mean_loss": float(metrics["mean_loss"]),
                "tail_mse": float(metrics["tail_mse"]),
                "final_error": float(metrics["final_error"]),
            }
        )

    def _mean_or_inf(values):
        return float(np.mean(values)) if values else float(np.inf)

    all_losses = [entry["mean_loss"] for entry in rollout_metrics]
    return {
        "overall_mean": _mean_or_inf(all_losses),
        "track_mean": _mean_or_inf(grouped_losses.get("tracking", [])),
        "reg_mean": _mean_or_inf(grouped_losses.get("regulation", [])),
        "dist_mean": _mean_or_inf(grouped_losses.get("disturbance_rejection", [])),
        "lag_mean": _mean_or_inf(grouped_losses.get("lag_mismatch", [])),
        "reg_tail_mean": _mean_or_inf(regulation_tail),
        "dist_tail_mean": _mean_or_inf(disturbance_tail),
        "lag_tail_mean": _mean_or_inf(lag_tail),
        "rollout_metrics": rollout_metrics,
        "tail_fraction": float(tail_fraction),
    }


def train_one_policy_with_validation(
    *,
    policy,
    plant: PlantParams,
    cfg: RLBPTTConfig,
    steps: int,
    lr: float,
    kind: str,
    episode_sampler: Callable[[np.random.Generator, RLBPTTConfig], RolloutSpec],
    validation_rollouts: Sequence[RolloutSpec],
    eval_every: int = 100,
    early_stop_patience_evals: int = 10,
    early_stop_rel_improve: float = 0.01,
    early_stop_best_val_threshold: float = np.inf,
    print_every: int = 0,
    rng: Optional[np.random.Generator] = None,
    grad_clip_norm: Optional[float] = None,
    rich_feature_set: str = "rich11",
) -> dict:
    """
    Train one policy with a sampled rollout distribution and fixed validation suite.

    Early stopping uses validation loss:
      stop if no validation improvement larger than `early_stop_rel_improve`
      is observed for `early_stop_patience_evals` consecutive evaluations,
      but only after the best validation loss is below
      `early_stop_best_val_threshold`.
    """
    if eval_every <= 0:
        raise ValueError("eval_every must be > 0")
    if early_stop_patience_evals <= 0:
        raise ValueError("early_stop_patience_evals must be > 0")
    if not validation_rollouts:
        raise ValueError("validation_rollouts must be non-empty")
    if early_stop_best_val_threshold <= 0:
        raise ValueError("early_stop_best_val_threshold must be > 0")

    opt = Adam(lr=lr)
    rng = np.random.default_rng(0) if rng is None else rng

    best_val_loss = np.inf
    best_params = {k: v.copy() for k, v in policy.params.items()}
    no_improve_evals = 0
    steps_completed = 0

    updates = []
    train_hist = []
    val_hist = []
    train_window = []

    for it in range(steps):
        rollout = episode_sampler(rng, cfg)

        if kind == "pidfeat":
            loss, grads = bptt_rollout_and_grads_pidfeat(
                policy=policy,
                plant=rollout.plant_override if rollout.plant_override is not None else plant,
                cfg=cfg,
                x0=rollout.x0,
                r_seq=rollout.r,
                du_seq=rollout.du,
                noise_y_seq=rollout.noise_y,
            )
        elif kind == "rich":
            loss, grads = bptt_rollout_and_grads_rich(
                policy=policy,
                plant=rollout.plant_override if rollout.plant_override is not None else plant,
                cfg=cfg,
                x0=rollout.x0,
                r_seq=rollout.r,
                du_seq=rollout.du,
                noise_y_seq=rollout.noise_y,
                feature_set=rich_feature_set,
            )
        else:
            raise ValueError(f"Unknown kind: {kind}")

        grads = _clip_gradients(grads, grad_clip_norm)
        opt.step(policy.params, grads)
        train_window.append(float(loss))
        steps_completed = it + 1

        if print_every and ((it + 1) % print_every == 0):
            print(f"  [{it+1:5d}/{steps}] train_loss={loss:.6f}")

        should_eval = ((it + 1) % eval_every == 0) or ((it + 1) == steps)
        if not should_eval:
            continue

        mean_train_loss = float(np.mean(train_window))
        train_window.clear()
        mean_val_loss = float(
            np.mean([
                evaluate_policy(policy=policy, plant=plant, cfg=cfg, kind=kind, rollout=rollout_val)
                if kind != "rich"
                else evaluate_policy(
                    policy=policy,
                    plant=plant,
                    cfg=cfg,
                    kind=kind,
                    rollout=rollout_val,
                    rich_feature_set=rich_feature_set,
                )
                for rollout_val in validation_rollouts
            ])
        )

        updates.append(steps_completed)
        train_hist.append(mean_train_loss)
        val_hist.append(mean_val_loss)

        if mean_val_loss < best_val_loss * (1.0 - early_stop_rel_improve):
            best_val_loss = mean_val_loss
            best_params = {k: v.copy() for k, v in policy.params.items()}
            no_improve_evals = 0
        else:
            no_improve_evals += 1

        if (
            best_val_loss < early_stop_best_val_threshold
            and no_improve_evals >= early_stop_patience_evals
        ):
            break

    for key, value in best_params.items():
        policy.params[key] = value

    validation_metrics = summarize_validation_metrics(
        policy=policy,
        plant=plant,
        cfg=cfg,
        kind=kind,
        validation_rollouts=validation_rollouts,
        tail_fraction=0.25,
        rich_feature_set=rich_feature_set,
    )

    return {
        "final_loss": float(train_hist[-1] if train_hist else np.nan),
        "best_val_loss": float(best_val_loss),
        "steps_completed": int(steps_completed),
        "stopped_early": bool(steps_completed < steps),
        "updates": np.asarray(updates, dtype=np.int64),
        "train_loss": np.asarray(train_hist, dtype=np.float64),
        "val_loss": np.asarray(val_hist, dtype=np.float64),
        "validation_metrics": validation_metrics,
    }


def train_per_plant_and_save(
    *,
    plants: Dict[str, PlantParams],
    make_policy: Callable[[], object],
    cfg: RLBPTTConfig,
    steps: int,
    lr: float,
    kind: str,  # "pidfeat" or "rich"
    out_dir: str,
    print_every: int = 0,
    rng_seed: int = 0,
    r0_sampler: Optional[Callable[[np.random.Generator], float]] = None,
    x0_sampler: Optional[Callable[[np.random.Generator], Any]] = None,
    meta_extra: Optional[dict] = None,
    grad_clip_norm: Optional[float] = None,
) -> None:
    """
    Fix B: trains a separate policy for each plant_id and saves it to:
      {out_dir}/rl_{kind}__{plant_id}.npz
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for plant_id, plant in plants.items():
        print(f"\nTraining RL ({kind}) for plant: {plant_id}")
        seed_pid = (int(rng_seed) + int(zlib.crc32(plant_id.encode("utf-8")))) % (2**32)
        rng = np.random.default_rng(seed_pid)
        policy = make_policy()
        final_loss = train_one_policy(
            policy=policy,
            plant=plant,
            cfg=cfg,
            steps=steps,
            lr=lr,
            kind=kind,
            print_every=print_every,
            rng=rng,
            r0_sampler=r0_sampler,
            x0_sampler=x0_sampler,
            grad_clip_norm=grad_clip_norm,
        )

        save_path = str(Path(out_dir) / f"rl_{kind}__{plant_id}.npz")
        extra = {} if meta_extra is None else dict(meta_extra)
        save_policy_npz(
            save_path,
            policy=policy,
            meta={
                "kind": kind,
                "plant_id": plant_id,
                "final_loss": final_loss,
                "cfg": cfg.__dict__,
                "steps": steps,
                "lr": lr,
                "rng_seed": rng_seed,
                "grad_clip_norm": grad_clip_norm,
                "meta_extra": extra,
            },
        )
