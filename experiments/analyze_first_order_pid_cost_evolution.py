from __future__ import annotations

import warnings
from pathlib import Path
import sys
from typing import Dict, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from control_bench.config import FIRST_ORDER_COST_WEIGHTS, FIRST_ORDER_FAMILY as family
from control_bench.controllers.rl_numpy import LinearPIDFeaturePolicy, RLBPTTConfig
from control_bench.controllers.rl_numpy.bptt import bptt_rollout_and_grads_pidfeat
from control_bench.controllers.rl_numpy.optim import Adam
from control_bench.controllers.rl_numpy.training import evaluate_policy
from first_order_rl_helpers import (
    build_first_order_validation_rollouts,
    extract_first_order_plant_params,
    sample_first_order_training_rollout,
)


PLANT_ID = "stable__no_zero"
QY = FIRST_ORDER_COST_WEIGHTS.qy
RU = FIRST_ORDER_COST_WEIGHTS.ru
QU = FIRST_ORDER_COST_WEIGHTS.qu
LR = 3e-3
STEPS = 15000
SEED = 0
CHECKPOINT_EVERY = 100
EPISODE_SECONDS = 20.0
N_FREQ = 300
OMEGA_MIN_RAD_PER_S = 1e-3
OMEGA_MAX_NYQUIST_FRACTION = 0.98
SNAPSHOT_COUNT = 6


def _format_tag_value(value: object) -> str:
    text = str(value)
    return text.replace("-", "m").replace(".", "p")


def make_run_tag(*, plant_id: str, qy: float, ru: float, qu: float, lr: float, seed: int, steps: int) -> str:
    return (
        f"{plant_id}"
        f"__qy{_format_tag_value(qy)}"
        f"__ru{_format_tag_value(ru)}"
        f"__qu{_format_tag_value(qu)}"
        f"__lr{_format_tag_value(lr)}"
        f"__seed{seed}"
        f"__steps{steps}"
    )


def build_cfg(*, qy: float, ru: float, qu: float) -> RLBPTTConfig:
    return RLBPTTConfig(
        horizon_steps=int(round(EPISODE_SECONDS / family.dt)),
        qy=qy,
        ru=ru,
        qu=qu,
        r0=1.0,
        u_min=family.u_min,
        u_max=family.u_max,
    )


def extract_pid_gains(policy: LinearPIDFeaturePolicy) -> tuple[float, float, float]:
    kp, ki, kd = np.asarray(policy.params["W"], dtype=np.float64).reshape(3)
    return float(kp), float(ki), float(kd)


def build_omega_grid(dt: float, *, n_freq: int, omega_min_rad_per_s: float, max_nyquist_fraction: float) -> np.ndarray:
    if n_freq < 2:
        raise ValueError("n_freq must be at least 2")
    if not (0.0 < max_nyquist_fraction < 1.0):
        raise ValueError("max_nyquist_fraction must be in (0, 1)")
    nyquist = np.pi / dt
    omega_max = max_nyquist_fraction * nyquist
    if omega_min_rad_per_s <= 0.0 or omega_min_rad_per_s >= omega_max:
        raise ValueError("omega_min_rad_per_s must be in (0, omega_max)")
    return np.logspace(
        np.log10(omega_min_rad_per_s),
        np.log10(omega_max),
        n_freq,
        dtype=np.float64,
    )


def pid_closed_loop_reference_to_output_ss(
    plant,
    *,
    kp: float,
    ki: float,
    kd: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    A = np.asarray(plant.A, dtype=np.float64)
    B = np.asarray(plant.B, dtype=np.float64).reshape(-1, 1)
    C = np.asarray(plant.C, dtype=np.float64).reshape(1, -1)
    D = float(np.asarray(plant.D, dtype=np.float64).reshape(1, 1).item())
    dt = float(plant.dt)

    n = int(A.shape[0])
    alpha = float(kp + ki * dt + kd / dt)

    acl = np.zeros((n + 3, n + 3), dtype=np.float64)
    bcl = np.zeros((n + 3, 1), dtype=np.float64)
    ccl = np.zeros((1, n + 3), dtype=np.float64)
    dcl = np.zeros((1, 1), dtype=np.float64)

    acl[:n, :n] = A - alpha * (B @ C)
    acl[:n, n] = (-alpha * D) * B[:, 0]
    acl[:n, n + 1] = ki * B[:, 0]
    acl[:n, n + 2] = -(kd / dt) * B[:, 0]

    acl[n, :n] = -alpha * C.reshape(-1)
    acl[n, n] = -alpha * D
    acl[n, n + 1] = ki
    acl[n, n + 2] = -(kd / dt)

    acl[n + 1, :n] = -dt * C.reshape(-1)
    acl[n + 1, n] = -dt * D
    acl[n + 1, n + 1] = 1.0

    acl[n + 2, :n] = -C.reshape(-1)
    acl[n + 2, n] = -D

    bcl[:n, 0] = alpha * B[:, 0]
    bcl[n, 0] = alpha
    bcl[n + 1, 0] = dt
    bcl[n + 2, 0] = 1.0

    ccl[0, :n] = C.reshape(-1)
    ccl[0, n] = D

    return acl, bcl, ccl, dcl


def complementary_sensitivity_magnitude(
    plant,
    *,
    kp: float,
    ki: float,
    kd: float,
    omega_grid: np.ndarray,
) -> np.ndarray:
    acl, bcl, ccl, dcl = pid_closed_loop_reference_to_output_ss(
        plant,
        kp=kp,
        ki=ki,
        kd=kd,
    )
    eye = np.eye(acl.shape[0], dtype=np.complex128)
    out = np.zeros_like(omega_grid, dtype=np.float64)

    for idx, omega in enumerate(np.asarray(omega_grid, dtype=np.float64)):
        z_eval = np.exp(1j * omega * float(plant.dt))
        resp = ccl @ np.linalg.solve(z_eval * eye - acl, bcl) + dcl
        out[idx] = float(np.abs(resp.item()))

    return out


def mean_validation_loss(policy, plant, cfg: RLBPTTConfig, validation_rollouts: Sequence) -> float:
    return float(
        np.mean(
            [
                evaluate_policy(
                    policy=policy,
                    plant=plant,
                    cfg=cfg,
                    kind="pidfeat",
                    rollout=rollout,
                )
                for rollout in validation_rollouts
            ]
        )
    )


def select_snapshot_indices(val_loss_history: np.ndarray, *, snapshot_count: int) -> np.ndarray:
    total = int(val_loss_history.size)
    if total == 0:
        raise ValueError("val_loss_history must be non-empty")
    base = np.linspace(0, total - 1, num=min(snapshot_count, total), dtype=int)
    best_idx = int(np.nanargmin(val_loss_history))
    indices = sorted({0, total - 1, best_idx, *base.tolist()})
    return np.asarray(indices, dtype=np.int64)


def save_analysis_npz(path: Path, *, payload: Dict[str, np.ndarray], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload, _meta=np.array([repr(meta)], dtype=object))


def plot_loss_figure(*, updates: np.ndarray, train_loss: np.ndarray, val_loss: np.ndarray, out_path: Path, title_tag: str) -> None:
    eps = 1e-12
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    train_safe = np.where(np.isnan(train_loss), np.nan, np.maximum(train_loss, eps))
    val_safe = np.maximum(val_loss, eps)
    ax.plot(updates, train_safe, linewidth=2.0, label="train")
    ax.plot(updates, val_safe, linewidth=2.0, label="validation")
    ax.set_yscale("log")
    ax.set_xlabel("Update")
    ax.set_ylabel("Loss")
    ax.set_title(f"PID-feature loss evolution | {title_tag}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_gain_figure(*, updates: np.ndarray, kp: np.ndarray, ki: np.ndarray, kd: np.ndarray, out_path: Path, title_tag: str) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(updates, kp, linewidth=2.0, label="Kp")
    ax.plot(updates, ki, linewidth=2.0, label="Ki")
    ax.plot(updates, kd, linewidth=2.0, label="Kd")
    ax.axhline(0.0, color="0.7", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Update")
    ax.set_ylabel("Weight")
    ax.set_title(f"PID-feature weight evolution | {title_tag}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_t_heatmap(*, updates: np.ndarray, omega_grid: np.ndarray, t_mag: np.ndarray, out_path: Path, title_tag: str) -> None:
    eps = 1e-12
    t_db = 20.0 * np.log10(np.maximum(t_mag, eps))
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    mesh = ax.pcolormesh(updates, omega_grid, t_db.T, shading="nearest", cmap="viridis")
    ax.set_yscale("log")
    ax.set_xlabel("Update")
    ax.set_ylabel("Frequency [rad/s]")
    ax.set_title(f"Complementary sensitivity |T| [dB] | {title_tag}")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("|T| [dB]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_t_snapshots(
    *,
    updates: np.ndarray,
    omega_grid: np.ndarray,
    t_mag: np.ndarray,
    val_loss: np.ndarray,
    snapshot_indices: Iterable[int],
    out_path: Path,
    title_tag: str,
) -> None:
    eps = 1e-12
    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    for idx in snapshot_indices:
        idx_int = int(idx)
        label = f"update {int(updates[idx_int])} | val={float(val_loss[idx_int]):.4g}"
        curve_db = 20.0 * np.log10(np.maximum(t_mag[idx_int], eps))
        ax.plot(omega_grid, curve_db, linewidth=2.0, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Frequency [rad/s]")
    ax.set_ylabel("|T| [dB]")
    ax.set_title(f"Complementary sensitivity snapshots | {title_tag}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_analysis(
    *,
    plant_id: str = PLANT_ID,
    qy: float = QY,
    ru: float = RU,
    qu: float = QU,
    lr: float = LR,
    steps: int = STEPS,
    seed: int = SEED,
    checkpoint_every: int = CHECKPOINT_EVERY,
    n_freq: int = N_FREQ,
    omega_min_rad_per_s: float = OMEGA_MIN_RAD_PER_S,
    omega_max_nyquist_fraction: float = OMEGA_MAX_NYQUIST_FRACTION,
    snapshot_count: int = SNAPSHOT_COUNT,
) -> dict:
    plants = extract_first_order_plant_params()
    if plant_id not in plants:
        raise KeyError(f"Unknown plant_id: {plant_id}")
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be > 0")
    if steps <= 0:
        raise ValueError("steps must be > 0")

    plant = plants[plant_id]
    cfg = build_cfg(qy=qy, ru=ru, qu=qu)
    validation_rollouts = build_first_order_validation_rollouts(cfg)
    omega_grid = build_omega_grid(
        plant.dt,
        n_freq=n_freq,
        omega_min_rad_per_s=omega_min_rad_per_s,
        max_nyquist_fraction=omega_max_nyquist_fraction,
    )

    if (cfg.u_min is not None) or (cfg.u_max is not None):
        warnings.warn(
            "Action bounds are active in the training config; the complementary sensitivity "
            "analysis ignores tanh saturation and should be interpreted as local small-signal behavior.",
            stacklevel=2,
        )

    policy = LinearPIDFeaturePolicy(seed=seed)
    optimizer = Adam(lr=lr)
    rng = np.random.default_rng(seed)

    updates = []
    train_loss_history = []
    val_loss_history = []
    kp_history = []
    ki_history = []
    kd_history = []
    t_mag_history = []
    train_window = []

    def record_checkpoint(update: int, mean_train_loss: float) -> None:
        kp, ki, kd = extract_pid_gains(policy)
        val_loss = mean_validation_loss(policy, plant, cfg, validation_rollouts)
        t_mag = complementary_sensitivity_magnitude(
            plant,
            kp=kp,
            ki=ki,
            kd=kd,
            omega_grid=omega_grid,
        )
        updates.append(int(update))
        train_loss_history.append(float(mean_train_loss))
        val_loss_history.append(float(val_loss))
        kp_history.append(float(kp))
        ki_history.append(float(ki))
        kd_history.append(float(kd))
        t_mag_history.append(t_mag.astype(np.float64))

    record_checkpoint(update=0, mean_train_loss=np.nan)

    for step_index in range(steps):
        rollout = sample_first_order_training_rollout(rng, cfg)
        loss, grads = bptt_rollout_and_grads_pidfeat(
            policy=policy,
            plant=plant,
            cfg=cfg,
            x0=rollout.x0,
            r_seq=rollout.r,
            du_seq=rollout.du,
        )
        optimizer.step(policy.params, grads)
        train_window.append(float(loss))

        is_checkpoint = ((step_index + 1) % checkpoint_every == 0) or ((step_index + 1) == steps)
        if not is_checkpoint:
            continue

        mean_train_loss = float(np.mean(train_window)) if train_window else np.nan
        train_window.clear()
        record_checkpoint(update=step_index + 1, mean_train_loss=mean_train_loss)
        print(
            f"[{step_index + 1:6d}/{steps}] "
            f"train={mean_train_loss:.6f} "
            f"val={val_loss_history[-1]:.6f} "
            f"K=({kp_history[-1]:.5f}, {ki_history[-1]:.5f}, {kd_history[-1]:.5f})"
        )

    updates_arr = np.asarray(updates, dtype=np.int64)
    train_loss_arr = np.asarray(train_loss_history, dtype=np.float64)
    val_loss_arr = np.asarray(val_loss_history, dtype=np.float64)
    kp_arr = np.asarray(kp_history, dtype=np.float64)
    ki_arr = np.asarray(ki_history, dtype=np.float64)
    kd_arr = np.asarray(kd_history, dtype=np.float64)
    t_mag_arr = np.asarray(t_mag_history, dtype=np.float64)
    snapshot_indices = select_snapshot_indices(val_loss_arr, snapshot_count=snapshot_count)

    run_tag = make_run_tag(
        plant_id=plant_id,
        qy=qy,
        ru=ru,
        qu=qu,
        lr=lr,
        seed=seed,
        steps=steps,
    )
    title_tag = (
        f"{plant_id} | qy={qy:g}, ru={ru:g}, qu={qu:g}, "
        f"lr={lr:g}, seed={seed}, steps={steps}"
    )

    raw_dir = Path(__file__).resolve().parent / "results" / "rl_numpy" / "pid_cost_dive"
    fig_dir = Path(__file__).resolve().parent / "results" / "figures" / "pid_cost_dive"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    save_analysis_npz(
        raw_dir / f"{run_tag}.npz",
        payload={
            "updates": updates_arr,
            "train_loss": train_loss_arr,
            "val_loss": val_loss_arr,
            "kp": kp_arr,
            "ki": ki_arr,
            "kd": kd_arr,
            "omega_grid_rad_per_s": omega_grid,
            "t_mag": t_mag_arr,
            "snapshot_indices": snapshot_indices,
        },
        meta={
            "plant_id": plant_id,
            "qy": qy,
            "ru": ru,
            "qu": qu,
            "lr": lr,
            "seed": seed,
            "steps": steps,
            "checkpoint_every": checkpoint_every,
            "episode_seconds": EPISODE_SECONDS,
            "dt": float(plant.dt),
            "u_min": cfg.u_min,
            "u_max": cfg.u_max,
            "n_freq": n_freq,
            "omega_min_rad_per_s": omega_min_rad_per_s,
            "omega_max_nyquist_fraction": omega_max_nyquist_fraction,
            "snapshot_count": snapshot_count,
        },
    )

    plot_loss_figure(
        updates=updates_arr,
        train_loss=train_loss_arr,
        val_loss=val_loss_arr,
        out_path=fig_dir / f"{run_tag}__losses.png",
        title_tag=title_tag,
    )
    plot_gain_figure(
        updates=updates_arr,
        kp=kp_arr,
        ki=ki_arr,
        kd=kd_arr,
        out_path=fig_dir / f"{run_tag}__gains.png",
        title_tag=title_tag,
    )
    plot_t_heatmap(
        updates=updates_arr,
        omega_grid=omega_grid,
        t_mag=t_mag_arr,
        out_path=fig_dir / f"{run_tag}__t_heatmap.png",
        title_tag=title_tag,
    )
    plot_t_snapshots(
        updates=updates_arr,
        omega_grid=omega_grid,
        t_mag=t_mag_arr,
        val_loss=val_loss_arr,
        snapshot_indices=snapshot_indices,
        out_path=fig_dir / f"{run_tag}__t_snapshots.png",
        title_tag=title_tag,
    )

    return {
        "run_tag": run_tag,
        "raw_path": raw_dir / f"{run_tag}.npz",
        "fig_dir": fig_dir,
        "updates": updates_arr,
        "train_loss": train_loss_arr,
        "val_loss": val_loss_arr,
        "kp": kp_arr,
        "ki": ki_arr,
        "kd": kd_arr,
        "omega_grid_rad_per_s": omega_grid,
        "t_mag": t_mag_arr,
        "snapshot_indices": snapshot_indices,
    }


if __name__ == "__main__":
    result = run_analysis()
    print(f"Saved analysis data to: {result['raw_path']}")
    print(f"Saved figures to: {result['fig_dir']}")
