from __future__ import annotations

import csv
from pathlib import Path
import sys
from typing import Dict, List

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

from benchmark_first_order_robustness_metrics import R_STEP, SEED, USE_BOUNDS, step_reference
from robustness_tests.hidden_pole_stress_test import (
    HIDDEN_POLE,
    METRICS_DIR,
    build_controller_factories,
    build_hidden_pole_chain_mismatch_plant,
)
from control_bench.config import FIRST_ORDER_FAMILY as family
from control_bench.core.scenario_impl import BasicScenario
from control_bench.core.sim import simulate_closed_loop
from control_bench.core.types import SimConfig
from control_bench.plants.first_order_family import build_first_order_grid


PLANT_ID = "unstable__rhp_zero"
SUMMARY_PATH = METRICS_DIR / "controller_survival_summary.csv"
OUT_DIR = EXPERIMENTS_ROOT / "results" / "figures" / "first_order_hidden_pole_focus"
Y_MIN = -12.0
Y_MAX = 4.0
FIGURE_T_FINAL = 15.0

DISPLAY_NAME_MAP = {
    "MPC": "MPC",
    "RL (PID-features)": "RL (PID-features)",
    "16_16 (rich11)": r"MLP$_r$(16,16)",
    "32_32 (rich11)": r"MLP$_r$(32,32)",
    "32_32_32 (rich11)": r"MLP$_r$(32,32,32)",
}

BASE_STYLE_MAP = {
    "MPC": {"color": "#111111", "linestyle": "-", "linewidth": 2.4},
    "RL (PID-features)": {"color": "#b22222", "linestyle": "-", "linewidth": 2.1},
    "16_16 (rich11)": {"color": "#1f77b4", "linestyle": "-", "linewidth": 1.8},
    "32_32 (rich11)": {"color": "#1f77b4", "linestyle": "--", "linewidth": 1.8},
    "32_32_32 (rich11)": {"color": "#1f77b4", "linestyle": ":", "linewidth": 2.2},
}


def _load_breakpoints() -> List[dict]:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing hidden-pole summary: {SUMMARY_PATH}")

    with SUMMARY_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    eligible = [row for row in rows if row["eligible_after_nominal"] == "True"]
    fail_orders = sorted(
        {
            int(row["first_failure_order"])
            for row in eligible
            if str(row["first_failure_order"]).strip()
        }
    )

    breakpoints: List[dict] = []
    for order in fail_orders:
        newly_failing = [
            str(row["controller"])
            for row in eligible
            if str(row["first_failure_order"]).strip() and int(row["first_failure_order"]) == order
        ]
        still_passing = [
            str(row["controller"])
            for row in eligible
            if int(row["max_pass_all_order"]) >= order
        ]
        breakpoints.append(
            {
                "hidden_order": int(order),
                "newly_failing": newly_failing,
                "still_passing": still_passing,
            }
        )
    return breakpoints


def _controller_order(name: str) -> tuple[int, str]:
    priority = {
        "MPC": 0,
        "RL (PID-features)": 1,
        "16_16 (rich11)": 2,
        "32_32 (rich11)": 3,
        "32_32_32 (rich11)": 4,
    }
    return (priority.get(name, 99), name)


def _simulate_stage(hidden_order: int, controller_names: List[str]) -> Dict[str, object]:
    nominal = build_first_order_grid(family)[PLANT_ID]
    scenario = BasicScenario(io=nominal.io, reference_fn=step_reference(R_STEP))
    cfg = SimConfig(t_final=FIGURE_T_FINAL, seed=SEED)
    factories = build_controller_factories(PLANT_ID, nominal=nominal, use_bounds=USE_BOUNDS)

    trajectories: Dict[str, object] = {}
    for name in controller_names:
        controller = factories[name]()
        plant = build_hidden_pole_chain_mismatch_plant(
            PLANT_ID,
            hidden_pole=HIDDEN_POLE,
            hidden_order=hidden_order,
            use_bounds=USE_BOUNDS,
        )
        result = simulate_closed_loop(plant=plant, controller=controller, scenario=scenario, cfg=cfg)
        trajectories[name] = result.traj
    return trajectories


def _plot_stage(ax, *, hidden_order: int, newly_failing: List[str], still_passing: List[str]) -> None:
    names = sorted(set(newly_failing + still_passing), key=_controller_order)
    trajectories = _simulate_stage(hidden_order, names)
    first_traj = next(iter(trajectories.values()))
    reference = np.asarray(first_traj.r[:, 0], dtype=float)

    ax.plot(
        first_traj.t,
        reference,
        color="#7f7f7f",
        linestyle="--",
        linewidth=1.5,
        label="reference",
        zorder=1,
    )

    for name in names:
        style = dict(BASE_STYLE_MAP[name])
        if name in newly_failing:
            style["linewidth"] = max(style["linewidth"], 2.8)
            style["alpha"] = 1.0
            zorder = 5
        else:
            style["alpha"] = 0.95
            zorder = 3
        y = np.asarray(trajectories[name].y[:, 0], dtype=float)
        y_plot = np.clip(y, Y_MIN, Y_MAX)
        ax.plot(
            trajectories[name].t,
            y_plot,
            label=DISPLAY_NAME_MAP[name],
            zorder=zorder,
            **style,
        )

    failing_text = ", ".join(DISPLAY_NAME_MAP[name] for name in newly_failing)
    ax.set_title(f"Hidden order $n={hidden_order}$\nnewly failing: {failing_text}", fontsize=13, pad=10)
    ax.grid(True, alpha=0.24)
    ax.set_xlim(float(first_traj.t[0]), float(first_traj.t[-1]))
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.axhline(-2.0, color="#bdbdbd", linewidth=0.9, linestyle=":")
    ax.text(
        0.98,
        0.03,
        "y clipped to [-12, 4]",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#666666",
    )


def run() -> None:
    breakpoints = _load_breakpoints()
    if not breakpoints:
        raise RuntimeError("No hidden-pole failure breakpoints found.")

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), sharex=True, sharey=True)
    axes = axes.reshape(-1)

    for ax, stage in zip(axes, breakpoints):
        _plot_stage(
            ax,
            hidden_order=int(stage["hidden_order"]),
            newly_failing=list(stage["newly_failing"]),
            still_passing=list(stage["still_passing"]),
        )

    for idx, ax in enumerate(axes):
        if idx >= 2:
            ax.set_xlabel("Time [s]", fontsize=12)
        if idx % 2 == 0:
            ax.set_ylabel("Output y", fontsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    seen = set()
    dedup_handles = []
    dedup_labels = []
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen.add(label)
            dedup_handles.append(handle)
            dedup_labels.append(label)
    fig.legend(
        dedup_handles,
        dedup_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
        fontsize=11,
        columnspacing=1.4,
        handlelength=2.8,
    )
    fig.suptitle(
        "Hidden-pole stress on the unstable + RHP-zero plant:\nfailure-onset responses for nominal-screened controllers",
        fontsize=18,
        y=0.995,
    )
    fig.tight_layout(rect=[0.03, 0.07, 1.0, 0.95], h_pad=2.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "unstable_rhp_zero__hidden_pole_failure_stages.png"
    pdf_path = OUT_DIR / "unstable_rhp_zero__hidden_pole_failure_stages.pdf"
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    run()
