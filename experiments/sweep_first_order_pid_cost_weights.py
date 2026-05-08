from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from analyze_first_order_pid_cost_evolution import (
    CHECKPOINT_EVERY,
    LR,
    N_FREQ,
    OMEGA_MAX_NYQUIST_FRACTION,
    OMEGA_MIN_RAD_PER_S,
    PLANT_ID,
    QY,
    QU,
    SEED,
    SNAPSHOT_COUNT,
    STEPS,
    run_analysis,
)


RU_VALUES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def _format_tag_value(value: object) -> str:
    text = str(value)
    return text.replace("-", "m").replace(".", "p")


def _parse_ru_values(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("At least one ru value must be provided")
    return values


def _approx_bandwidth_rad_per_s(
    omega_grid: np.ndarray,
    t_mag_db: np.ndarray,
    *,
    threshold_db: float = -3.0,
) -> float:
    below = np.flatnonzero(np.asarray(t_mag_db, dtype=np.float64) <= float(threshold_db))
    if below.size == 0:
        return float("nan")

    idx = int(below[0])
    if idx == 0:
        return float(omega_grid[0])

    omega_lo = float(omega_grid[idx - 1])
    omega_hi = float(omega_grid[idx])
    db_lo = float(t_mag_db[idx - 1])
    db_hi = float(t_mag_db[idx])

    if abs(db_hi - db_lo) < 1e-12:
        return omega_hi

    x_lo = float(np.log10(omega_lo))
    x_hi = float(np.log10(omega_hi))
    frac = (float(threshold_db) - db_lo) / (db_hi - db_lo)
    return float(10.0 ** (x_lo + frac * (x_hi - x_lo)))


def _plot_gains_vs_ru(rows: Sequence[dict], *, out_path: Path, title_tag: str) -> None:
    ru = np.asarray([row["ru"] for row in rows], dtype=np.float64)
    kp = np.asarray([row["kp"] for row in rows], dtype=np.float64)
    ki = np.asarray([row["ki"] for row in rows], dtype=np.float64)
    kd = np.asarray([row["kd"] for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(ru, kp, marker="o", linewidth=2.0, label="Kp")
    ax.plot(ru, ki, marker="o", linewidth=2.0, label="Ki")
    ax.plot(ru, kd, marker="o", linewidth=2.0, label="Kd")
    ax.axhline(0.0, color="0.7", linewidth=1.0, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("ru")
    ax.set_ylabel("Final learned weight")
    ax.set_title(f"PID-feature gains vs control-move penalty | {title_tag}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_validation_vs_ru(rows: Sequence[dict], *, out_path: Path, title_tag: str) -> None:
    ru = np.asarray([row["ru"] for row in rows], dtype=np.float64)
    final_val = np.asarray([row["final_val_loss"] for row in rows], dtype=np.float64)
    best_val = np.asarray([row["best_val_loss"] for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.plot(ru, final_val, marker="o", linewidth=2.0, label="final validation loss")
    ax.plot(ru, best_val, marker="o", linewidth=2.0, label="best validation loss")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ru")
    ax.set_ylabel("Validation loss")
    ax.set_title(f"Validation loss vs control-move penalty | {title_tag}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_t_overlay(
    rows: Sequence[dict],
    *,
    omega_grid: np.ndarray,
    out_path: Path,
    title_tag: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    for row in rows:
        ax.plot(
            omega_grid,
            row["t_mag_db"],
            linewidth=2.0,
            label=f"ru={row['ru']:g} | K=({row['kp']:.2f}, {row['ki']:.2f}, {row['kd']:.2f})",
        )
    ax.axhline(-3.0, color="0.65", linewidth=1.0, linestyle="--", label="-3 dB")
    ax.set_xscale("log")
    ax.set_xlabel("Frequency [rad/s]")
    ax.set_ylabel("|T| [dB]")
    ax.set_title(f"Final complementary sensitivity vs control-move penalty | {title_tag}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_t_heatmap(
    rows: Sequence[dict],
    *,
    omega_grid: np.ndarray,
    out_path: Path,
    title_tag: str,
) -> None:
    t_db = np.asarray([row["t_mag_db"] for row in rows], dtype=np.float64)
    ru_labels = [f"{row['ru']:g}" for row in rows]
    y = np.arange(len(rows) + 1, dtype=np.float64)
    x = np.arange(omega_grid.size + 1, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    mesh = ax.pcolormesh(x, y, t_db, shading="auto", cmap="viridis")
    ax.set_xticks(np.linspace(0, omega_grid.size - 1, num=7))
    ax.set_xticklabels([f"{omega_grid[int(i)]:.3g}" for i in np.linspace(0, omega_grid.size - 1, num=7)])
    ax.set_yticks(np.arange(len(rows), dtype=np.float64) + 0.5)
    ax.set_yticklabels(ru_labels)
    ax.set_xlabel("Frequency [rad/s] (log-sampled grid)")
    ax.set_ylabel("ru")
    ax.set_title(f"Final complementary sensitivity heatmap | {title_tag}")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("|T| [dB]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_bandwidth_peak_vs_ru(rows: Sequence[dict], *, out_path: Path, title_tag: str) -> None:
    ru = np.asarray([row["ru"] for row in rows], dtype=np.float64)
    peak_db = np.asarray([row["peak_t_db"] for row in rows], dtype=np.float64)
    bandwidth = np.asarray([row["bandwidth_3db_rad_per_s"] for row in rows], dtype=np.float64)

    fig, (ax_peak, ax_bw) = plt.subplots(2, 1, figsize=(9.5, 8.0), sharex=True)

    ax_peak.plot(ru, peak_db, marker="o", linewidth=2.0, color="tab:red")
    ax_peak.axhline(0.0, color="0.7", linewidth=1.0, linestyle="--")
    ax_peak.set_xscale("log")
    ax_peak.set_ylabel("Peak |T| [dB]")
    ax_peak.set_title(f"Loop-shape summary vs control-move penalty | {title_tag}")
    ax_peak.grid(True, alpha=0.3)

    ax_bw.plot(ru, bandwidth, marker="o", linewidth=2.0, color="tab:blue")
    ax_bw.set_xscale("log")
    ax_bw.set_xlabel("ru")
    ax_bw.set_ylabel("-3 dB bandwidth [rad/s]")
    ax_bw.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _write_summary_csv(rows: Sequence[dict], *, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ru",
        "final_train_loss",
        "final_val_loss",
        "best_val_loss",
        "best_val_update",
        "kp",
        "ki",
        "kd",
        "peak_t_db",
        "peak_t_omega_rad_per_s",
        "bandwidth_3db_rad_per_s",
        "raw_path",
        "run_tag",
    ]
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def _write_summary_markdown(rows: Sequence[dict], *, out_path: Path, title_tag: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PID-feature cost-weight sweep",
        "",
        f"Study: {title_tag}",
        "",
        "| ru | final val loss | best val loss | Kp | Ki | Kd | peak |T| [dB] | -3 dB bw [rad/s] |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['ru']:g} | "
            f"{row['final_val_loss']:.6g} | "
            f"{row['best_val_loss']:.6g} | "
            f"{row['kp']:.6g} | "
            f"{row['ki']:.6g} | "
            f"{row['kd']:.6g} | "
            f"{row['peak_t_db']:.6g} | "
            f"{row['bandwidth_3db_rad_per_s']:.6g} |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def run_sweep(
    *,
    plant_id: str = PLANT_ID,
    qy: float = 1.0,
    qu: float = 0.0,
    ru_values: Sequence[float] = RU_VALUES,
    lr: float = LR,
    steps: int = STEPS,
    seed: int = SEED,
    checkpoint_every: int = CHECKPOINT_EVERY,
    n_freq: int = N_FREQ,
    omega_min_rad_per_s: float = OMEGA_MIN_RAD_PER_S,
    omega_max_nyquist_fraction: float = OMEGA_MAX_NYQUIST_FRACTION,
    snapshot_count: int = SNAPSHOT_COUNT,
) -> dict:
    ordered_ru_values = tuple(float(value) for value in ru_values)
    if not ordered_ru_values:
        raise ValueError("ru_values must be non-empty")

    rows = []
    omega_grid_ref = None

    for ru in ordered_ru_values:
        result = run_analysis(
            plant_id=plant_id,
            qy=float(qy),
            ru=float(ru),
            qu=float(qu),
            lr=float(lr),
            steps=int(steps),
            seed=int(seed),
            checkpoint_every=int(checkpoint_every),
            n_freq=int(n_freq),
            omega_min_rad_per_s=float(omega_min_rad_per_s),
            omega_max_nyquist_fraction=float(omega_max_nyquist_fraction),
            snapshot_count=int(snapshot_count),
        )

        omega_grid = np.asarray(result["omega_grid_rad_per_s"], dtype=np.float64)
        t_mag_final = np.asarray(result["t_mag"][-1], dtype=np.float64)
        t_mag_db = 20.0 * np.log10(np.maximum(t_mag_final, 1e-12))
        peak_idx = int(np.argmax(t_mag_db))
        val_loss = np.asarray(result["val_loss"], dtype=np.float64)
        updates = np.asarray(result["updates"], dtype=np.int64)
        best_idx = int(np.argmin(val_loss))

        if omega_grid_ref is None:
            omega_grid_ref = omega_grid
        elif not np.allclose(omega_grid_ref, omega_grid):
            raise ValueError("Inconsistent omega grids across sweep runs")

        rows.append(
            {
                "ru": float(ru),
                "final_train_loss": float(np.asarray(result["train_loss"], dtype=np.float64)[-1]),
                "final_val_loss": float(val_loss[-1]),
                "best_val_loss": float(val_loss[best_idx]),
                "best_val_update": int(updates[best_idx]),
                "kp": float(np.asarray(result["kp"], dtype=np.float64)[-1]),
                "ki": float(np.asarray(result["ki"], dtype=np.float64)[-1]),
                "kd": float(np.asarray(result["kd"], dtype=np.float64)[-1]),
                "peak_t_db": float(t_mag_db[peak_idx]),
                "peak_t_omega_rad_per_s": float(omega_grid[peak_idx]),
                "bandwidth_3db_rad_per_s": _approx_bandwidth_rad_per_s(omega_grid, t_mag_db),
                "t_mag_db": t_mag_db,
                "raw_path": str(result["raw_path"]),
                "run_tag": str(result["run_tag"]),
            }
        )

    title_tag = (
        f"{plant_id} | qy={qy:g}, qu={qu:g}, "
        f"ru sweep={','.join(f'{value:g}' for value in ordered_ru_values)}, "
        f"lr={lr:g}, seed={seed}, steps={steps}"
    )
    sweep_tag = (
        f"{plant_id}"
        f"__qy{_format_tag_value(qy)}"
        f"__qu{_format_tag_value(qu)}"
        f"__rusweep_{'_'.join(_format_tag_value(value) for value in ordered_ru_values)}"
        f"__lr{_format_tag_value(lr)}"
        f"__seed{seed}"
        f"__steps{steps}"
    )

    figures_dir = Path(__file__).resolve().parent / "results" / "figures" / "pid_cost_weight_sweep"
    metrics_dir = Path(__file__).resolve().parent / "results" / "metrics" / "pid_cost_weight_sweep"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    _write_summary_csv(rows, out_path=metrics_dir / f"{sweep_tag}__summary.csv")
    _write_summary_markdown(rows, out_path=metrics_dir / f"{sweep_tag}__summary.md", title_tag=title_tag)
    _plot_gains_vs_ru(rows, out_path=figures_dir / f"{sweep_tag}__gains_vs_ru.png", title_tag=title_tag)
    _plot_validation_vs_ru(rows, out_path=figures_dir / f"{sweep_tag}__validation_vs_ru.png", title_tag=title_tag)
    _plot_t_overlay(
        rows,
        omega_grid=np.asarray(omega_grid_ref, dtype=np.float64),
        out_path=figures_dir / f"{sweep_tag}__t_overlay.png",
        title_tag=title_tag,
    )
    _plot_t_heatmap(
        rows,
        omega_grid=np.asarray(omega_grid_ref, dtype=np.float64),
        out_path=figures_dir / f"{sweep_tag}__t_heatmap.png",
        title_tag=title_tag,
    )
    _plot_bandwidth_peak_vs_ru(
        rows,
        out_path=figures_dir / f"{sweep_tag}__loop_shape_summary.png",
        title_tag=title_tag,
    )

    return {
        "rows": rows,
        "figures_dir": figures_dir,
        "metrics_dir": metrics_dir,
        "sweep_tag": sweep_tag,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep ru for first-order PID-feature RL training analysis.")
    parser.add_argument("--plant-id", default=PLANT_ID)
    parser.add_argument("--qy", type=float, default=1.0)
    parser.add_argument("--qu", type=float, default=0.0)
    parser.add_argument(
        "--ru-values",
        type=str,
        default=",".join(str(value) for value in RU_VALUES),
        help="Comma-separated ru values to sweep.",
    )
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_EVERY)
    parser.add_argument("--n-freq", type=int, default=N_FREQ)
    parser.add_argument("--omega-min-rad-per-s", type=float, default=OMEGA_MIN_RAD_PER_S)
    parser.add_argument("--omega-max-nyquist-fraction", type=float, default=OMEGA_MAX_NYQUIST_FRACTION)
    parser.add_argument("--snapshot-count", type=int, default=SNAPSHOT_COUNT)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    ru_values = _parse_ru_values(args.ru_values)
    result = run_sweep(
        plant_id=args.plant_id,
        qy=args.qy,
        qu=args.qu,
        ru_values=ru_values,
        lr=args.lr,
        steps=args.steps,
        seed=args.seed,
        checkpoint_every=args.checkpoint_every,
        n_freq=args.n_freq,
        omega_min_rad_per_s=args.omega_min_rad_per_s,
        omega_max_nyquist_fraction=args.omega_max_nyquist_fraction,
        snapshot_count=args.snapshot_count,
    )
    print(f"Saved sweep figures to: {result['figures_dir']}")
    print(f"Saved sweep metrics to: {result['metrics_dir']}")


if __name__ == "__main__":
    main()
