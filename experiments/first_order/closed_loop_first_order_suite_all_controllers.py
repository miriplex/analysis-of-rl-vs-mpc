from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark_first_order_robustness_metrics import (
    R_STEP,
    SEED,
    T_FINAL,
    USE_BOUNDS,
    build_controllers,
    step_reference,
)
from control_bench.config import FIRST_ORDER_FAMILY, FIRST_ORDER_PLANT_IDS
from control_bench.core.scenario_impl import BasicScenario
from control_bench.core.sim import simulate_closed_loop
from control_bench.core.types import SimConfig
from control_bench.plants.first_order_family import build_first_order_grid


POLE_ORDER = ("stable", "integrator", "unstable")
ZERO_ORDER = ("no_zero", "lhp_zero", "rhp_zero")

DISPLAY_NAME_MAP = {
    "MPC": "MPC",
    "RL (PID-features)": "RL (PID-features)",
    "16_16 (rich11)": r"MLP$_r$(16,16)",
    "16_16 (compact6)": r"MLP$_c$(16,16)",
    "32_32 (rich11)": r"MLP$_r$(32,32)",
    "32_32 (compact6)": r"MLP$_c$(32,32)",
    "16_16_16 (rich11)": r"MLP$_r$(16,16,16)",
    "16_16_16 (compact6)": r"MLP$_c$(16,16,16)",
    "32_32_32 (rich11)": r"MLP$_r$(32,32,32)",
    "32_32_32 (compact6)": r"MLP$_c$(32,32,32)",
}

STYLE_MAP = {
    "MPC": {"color": "#111111", "linestyle": "-", "linewidth": 2.6, "zorder": 5},
    "RL (PID-features)": {"color": "#b22222", "linestyle": "-", "linewidth": 1.9, "zorder": 4},
    "16_16 (rich11)": {"color": "#1f77b4", "linestyle": "-", "linewidth": 1.6, "zorder": 3},
    "32_32 (rich11)": {"color": "#1f77b4", "linestyle": "--", "linewidth": 1.6, "zorder": 3},
    "16_16_16 (rich11)": {"color": "#1f77b4", "linestyle": "-.", "linewidth": 1.6, "zorder": 3},
    "32_32_32 (rich11)": {"color": "#1f77b4", "linestyle": ":", "linewidth": 2.0, "zorder": 3},
    "16_16 (compact6)": {"color": "#ff7f0e", "linestyle": "-", "linewidth": 1.6, "zorder": 2},
    "32_32 (compact6)": {"color": "#ff7f0e", "linestyle": "--", "linewidth": 1.6, "zorder": 2},
    "16_16_16 (compact6)": {"color": "#ff7f0e", "linestyle": "-.", "linewidth": 1.6, "zorder": 2},
    "32_32_32 (compact6)": {"color": "#ff7f0e", "linestyle": ":", "linewidth": 2.0, "zorder": 2},
}


def _humanize_pole(pole_tag: str) -> str:
    if pole_tag == "stable":
        return "Stable"
    if pole_tag == "integrator":
        return "Integrator"
    if pole_tag == "unstable":
        return "Unstable"
    raise ValueError(f"Unknown pole tag: {pole_tag}")


def _humanize_zero(zero_tag: str) -> str:
    if zero_tag == "no_zero":
        return "No zero"
    if zero_tag == "lhp_zero":
        return "LHP zero"
    if zero_tag == "rhp_zero":
        return "RHP zero"
    raise ValueError(f"Unknown zero tag: {zero_tag}")


def _plant_pz_from_id(plant_id: str) -> tuple[float, str]:
    pole_tag, zero_tag = plant_id.split("__", 1)
    if pole_tag == "stable":
        p = float(FIRST_ORDER_FAMILY.stable_p)
    elif pole_tag == "integrator":
        p = 0.0
    elif pole_tag == "unstable":
        p = float(FIRST_ORDER_FAMILY.unstable_p)
    else:
        raise ValueError(f"Unknown pole tag: {pole_tag}")

    if zero_tag == "no_zero":
        z_display = "none"
    elif zero_tag == "lhp_zero":
        z_display = f"{float(FIRST_ORDER_FAMILY.lhp_z):g}"
    elif zero_tag == "rhp_zero":
        z_display = f"{float(FIRST_ORDER_FAMILY.rhp_z):g}"
    else:
        raise ValueError(f"Unknown zero tag: {zero_tag}")
    return p, z_display


def _simulate_nominal_runs(plant_id: str) -> Dict[str, object]:
    nominal = build_first_order_grid(FIRST_ORDER_FAMILY)[plant_id]
    scenario = BasicScenario(io=nominal.io, reference_fn=step_reference(R_STEP))
    cfg = SimConfig(t_final=T_FINAL, seed=SEED)

    trajectories: Dict[str, object] = {}
    for controller in build_controllers(plant_id, nominal=nominal, use_bounds=USE_BOUNDS):
        plant = build_first_order_grid(FIRST_ORDER_FAMILY)[plant_id]
        plant.u_bounds = nominal.u_bounds if USE_BOUNDS else None
        result = simulate_closed_loop(plant=plant, controller=controller, scenario=scenario, cfg=cfg)
        trajectories[controller.name] = result.traj
    return trajectories


def run() -> None:
    if set(build_first_order_grid(FIRST_ORDER_FAMILY).keys()) != set(FIRST_ORDER_PLANT_IDS):
        raise RuntimeError("Configured first-order plant ids do not match the plant family.")

    all_runs: Dict[str, Dict[str, object]] = {}
    for plant_id in FIRST_ORDER_PLANT_IDS:
        all_runs[plant_id] = _simulate_nominal_runs(plant_id)

    fig, axes = plt.subplots(3, 3, figsize=(16.5, 10.6), sharex=True)
    legend_handles: List[object] = []
    legend_labels: List[str] = []

    y_min_common = -10.5
    y_max_common = 1.5

    for row_idx, pole_tag in enumerate(POLE_ORDER):
        for col_idx, zero_tag in enumerate(ZERO_ORDER):
            plant_id = f"{pole_tag}__{zero_tag}"
            ax = axes[row_idx, col_idx]
            runs = all_runs[plant_id]
            first_traj = next(iter(runs.values()))
            reference = np.asarray(first_traj.r[:, 0], dtype=float)
            ax.plot(
                first_traj.t,
                reference,
                color="#777777",
                linestyle="--",
                linewidth=1.6,
                label="reference",
                zorder=1,
            )

            for raw_name, traj in runs.items():
                style = dict(STYLE_MAP[raw_name])
                display_name = DISPLAY_NAME_MAP[raw_name]
                line, = ax.plot(traj.t, traj.y[:, 0], label=display_name, **style)
                if plant_id == FIRST_ORDER_PLANT_IDS[0]:
                    legend_handles.append(line)
                    legend_labels.append(display_name)

            ax.grid(True, alpha=0.24)
            ax.set_xlim(float(first_traj.t[0]), float(first_traj.t[-1]))
            ax.set_ylim(y_min_common, y_max_common)

            p, z_display = _plant_pz_from_id(plant_id)
            if row_idx == 0:
                ax.set_title(f"{_humanize_zero(zero_tag)}\n$z={z_display}$", fontsize=15, pad=12)
            if col_idx == 0:
                ax.set_ylabel(f"{_humanize_pole(pole_tag)}\n$p={p:g}$\nOutput y", fontsize=12)
            if row_idx == len(POLE_ORDER) - 1:
                ax.set_xlabel("Time [s]", fontsize=12)

    ref_handle = plt.Line2D([], [], color="#777777", linestyle="--", linewidth=1.6)
    fig.legend(
        [ref_handle] + legend_handles,
        ["reference"] + legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
        fontsize=11,
        columnspacing=1.3,
        handlelength=2.8,
    )
    fig.suptitle(
        f"Nominal closed-loop step responses of the 9 first-order benchmark plants ($r={R_STEP:g}$)",
        fontsize=20,
        y=0.995,
    )
    fig.tight_layout(rect=[0.03, 0.09, 1.0, 0.95], h_pad=2.0)

    out_dir = EXPERIMENTS_ROOT / "results" / "figures" / "first_order"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "closed_loop_first_order_suite__all_controllers.png"
    pdf_path = out_dir / "closed_loop_first_order_suite__all_controllers.pdf"
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    run()
