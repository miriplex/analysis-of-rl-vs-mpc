from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
SRC_ROOT = PROJECT_ROOT / "src"
FIRST_ORDER_ROOT = EXPERIMENTS_ROOT / "first_order"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(FIRST_ORDER_ROOT) not in sys.path:
    sys.path.insert(0, str(FIRST_ORDER_ROOT))

from benchmark_first_order_robustness_metrics import (
    TRADEOFF_PLANT_ID,
    _load_csv_rows,
    _measurement_noise_survival_for_mpc_horizons,
)


HORIZONS = tuple(range(2, 51))
RESULTS_DIR = EXPERIMENTS_ROOT / "results" / "metrics" / "first_order_robustness"
FIGURES_DIR = EXPERIMENTS_ROOT / "results" / "figures" / "first_order_robustness"
FOCAL_RUNTIME_CSV = RESULTS_DIR / f"mpc_horizon_sweep_nominal__{TRADEOFF_PLANT_ID}.csv"
PNG_PATH = FIGURES_DIR / "runtime_vs_measurement_noise__mpc_horizons_2_to_50.png"
PDF_PATH = FIGURES_DIR / "runtime_vs_measurement_noise__mpc_horizons_2_to_50.pdf"

MPC_COLOR = "#991b1b"
MPC_MARKER = "D"


def _runtime_rows_by_horizon() -> dict[int, dict]:
    rows = _load_csv_rows(FOCAL_RUNTIME_CSV)
    by_horizon: dict[int, dict] = {}
    for row in rows:
        horizon_n = int(row["N"])
        if horizon_n in HORIZONS:
            by_horizon[horizon_n] = dict(row)
    missing = [n for n in HORIZONS if n not in by_horizon]
    if missing:
        raise RuntimeError(
            "Missing focal MPC runtime rows for horizons: "
            + ", ".join(str(n) for n in missing)
            + f". Re-run {PROJECT_ROOT / 'experiments' / 'first_order' / 'benchmark_first_order_robustness_metrics.py'}."
        )
    return by_horizon


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    runtime_rows = _runtime_rows_by_horizon()
    measurement_survival = _measurement_noise_survival_for_mpc_horizons(horizons=HORIZONS)

    x_vals = np.array([float(runtime_rows[n]["mean_step_ms"]) for n in HORIZONS], dtype=float)
    y_vals = np.array([float(measurement_survival[n]) for n in HORIZONS], dtype=float)
    reference_y = float(measurement_survival[20])
    last_zero_horizon = max(n for n in HORIZONS if np.isclose(float(measurement_survival[n]), 0.0))
    first_reference_horizon = min(n for n in HORIZONS if np.isclose(float(measurement_survival[n]), reference_y))

    fig, ax = plt.subplots(1, 1, figsize=(8.4, 5.4))
    ax.plot(
        x_vals,
        y_vals,
        color=MPC_COLOR,
        linewidth=1.2,
        alpha=0.85,
        zorder=3,
    )
    ax.scatter(
        x_vals,
        y_vals,
        s=92,
        color=MPC_COLOR,
        marker=MPC_MARKER,
        edgecolors="white",
        linewidths=0.9,
        alpha=0.9,
        label="MPC horizons",
        zorder=4,
    )

    highlight_horizons = (2, last_zero_horizon, first_reference_horizon, 50)
    highlight_label_text = {
        2: "N=2",
        last_zero_horizon: f"last zero: N={last_zero_horizon}",
        first_reference_horizon: f"first plateau: N={first_reference_horizon}",
        50: "N=50",
    }
    label_offsets = {
        2: (-8, 18),
        last_zero_horizon: (24, 18),
        first_reference_horizon: (18, -28),
        50: (18, 18),
    }

    for horizon_n in highlight_horizons:
        idx = HORIZONS.index(int(horizon_n))
        x_val = float(x_vals[idx])
        y_val = float(y_vals[idx])
        dx, dy = label_offsets[int(horizon_n)]
        ax.annotate(
            highlight_label_text[int(horizon_n)],
            (x_val, y_val),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8.4,
            color=MPC_COLOR,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.18",
                "fc": "white",
                "ec": MPC_COLOR,
                "alpha": 0.95,
                "lw": 0.8,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": MPC_COLOR,
                "lw": 0.8,
                "shrinkA": 4,
                "shrinkB": 4,
                "alpha": 0.85,
            },
            annotation_clip=False,
            zorder=6,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Mean controller step time [ms]")
    ax.set_ylabel("Max survived measurement-noise scale")
    ax.set_title("MPC runtime vs measurement-noise robustness")
    ax.grid(True, which="both", alpha=0.3)
    ax.margins(x=0.12, y=0.18)
    ax.legend(loc="upper left", frameon=True)

    fig.subplots_adjust(top=0.90, bottom=0.14, left=0.13, right=0.98)

    fig.savefig(PNG_PATH, dpi=160)
    fig.savefig(PDF_PATH)

    backend = plt.get_backend().lower()
    if "agg" not in backend:
        plt.show()
    else:
        plt.close(fig)

    print(f"Saved PNG: {PNG_PATH}")
    print(f"Saved PDF: {PDF_PATH}")


if __name__ == "__main__":
    main()
