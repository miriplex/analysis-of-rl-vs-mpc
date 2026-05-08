from __future__ import annotations

import csv
from pathlib import Path
import sys
from typing import Dict, List
import zlib

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
    SEED,
    USE_BOUNDS,
    make_disturbance_u_fn_from_trace,
    make_measurement_fn_from_trace,
    n_steps_for,
    step_reference,
)
from robustness_tests.disturbance_stress_common import build_controller_factories
from control_bench.config import FIRST_ORDER_FAMILY as family
from control_bench.core.scenario_impl import BasicScenario
from control_bench.core.sim import simulate_closed_loop
from control_bench.core.types import SimConfig
from control_bench.experiment_manifest import first_order_measurement_noise_stress_test
from control_bench.plants.first_order_family import build_first_order_grid


PLANT_ID = "unstable__rhp_zero"
STRESS_CFG = first_order_measurement_noise_stress_test()
SUMMARY_PATH = (
    EXPERIMENTS_ROOT
    / "results"
    / "metrics"
    / "robustness_tests"
    / str(STRESS_CFG["output_metrics_namespace"])
    / "controller_survival_summary.csv"
)
OUT_DIR = EXPERIMENTS_ROOT / "results" / "figures" / "first_order_hidden_pole_focus"
BASE_SIGMA_Y = float(STRESS_CFG["base_measurement_noise_std"])
R_STEP = float(STRESS_CFG["reference_step"])
FIGURE_T_FINAL = float(STRESS_CFG["t_final_seconds"])
Y_MIN = -12.0
Y_MAX = 4.0

DISPLAY_NAME_MAP = {
    "MPC": "MPC",
    "RL (PID-features)": "RL (PID-features)",
    "16_16 (rich11)": r"MLP$_r$(16,16)",
    "32_32 (rich11)": r"MLP$_r$(32,32)",
    "32_32_32 (rich11)": r"MLP$_r$(32,32,32)",
    "32_32 (compact6)": r"MLP$_c$(32,32)",
}

BASE_STYLE_MAP = {
    "MPC": {"color": "#111111", "linestyle": "-", "linewidth": 2.4},
    "RL (PID-features)": {"color": "#b22222", "linestyle": "-", "linewidth": 2.1},
    "16_16 (rich11)": {"color": "#1f77b4", "linestyle": "-", "linewidth": 1.8},
    "32_32 (rich11)": {"color": "#1f77b4", "linestyle": "--", "linewidth": 1.8},
    "32_32_32 (rich11)": {"color": "#1f77b4", "linestyle": ":", "linewidth": 2.2},
    "32_32 (compact6)": {"color": "#d97706", "linestyle": "-.", "linewidth": 1.9},
}


def _load_breakpoints() -> List[dict]:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Missing measurement-noise summary: {SUMMARY_PATH}")

    with SUMMARY_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    eligible = [row for row in rows if row["eligible_after_nominal"] == "True"]
    fail_scales = sorted(
        {
            float(row["first_failure_scale"])
            for row in eligible
            if str(row["first_failure_scale"]).strip()
        }
    )

    breakpoints: List[dict] = []
    for scale in fail_scales:
        newly_failing = [
            str(row["controller"])
            for row in eligible
            if str(row["first_failure_scale"]).strip() and abs(float(row["first_failure_scale"]) - scale) < 1e-12
        ]
        still_passing = [
            str(row["controller"])
            for row in eligible
            if float(row["max_pass_all_scale"]) >= scale
        ]
        breakpoints.append(
            {
                "stress_scale": float(scale),
                "newly_failing": newly_failing,
                "still_passing": still_passing,
            }
        )
    return breakpoints


def _eligible_controllers() -> List[str]:
    with SUMMARY_PATH.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return sorted(
        [str(row["controller"]) for row in rows if row["eligible_after_nominal"] == "True"],
        key=_controller_order,
    )


def _controller_order(name: str) -> tuple[int, str]:
    priority = {
        "MPC": 0,
        "RL (PID-features)": 1,
        "16_16 (rich11)": 2,
        "32_32 (rich11)": 3,
        "32_32_32 (rich11)": 4,
        "32_32 (compact6)": 5,
    }
    return (priority.get(name, 99), name)


def _measurement_noise_trace(*, plant_id: str, n_steps: int, stress_scale: float) -> np.ndarray:
    seed_pid = (int(SEED) + int(zlib.crc32(f"meas::{plant_id}".encode("utf-8")))) % (2**32)
    rng = np.random.default_rng(seed_pid)
    base_eps_y = rng.normal(loc=0.0, scale=1.0, size=(n_steps,))
    return float(stress_scale) * float(BASE_SIGMA_Y) * base_eps_y


def _simulate_stage(stress_scale: float, controller_names: List[str]) -> Dict[str, object]:
    nominal = build_first_order_grid(family)[PLANT_ID]
    n_steps = n_steps_for(t_final=FIGURE_T_FINAL, dt=nominal.dt)
    noise_y = _measurement_noise_trace(plant_id=PLANT_ID, n_steps=n_steps, stress_scale=stress_scale)
    du = np.zeros((n_steps,), dtype=float)

    scenario = BasicScenario(
        io=nominal.io,
        reference_fn=step_reference(R_STEP),
        measurement_fn=make_measurement_fn_from_trace(noise_y=noise_y, dt=nominal.dt),
        disturbance_u_fn=make_disturbance_u_fn_from_trace(du=du, dt=nominal.dt),
    )
    cfg = SimConfig(t_final=FIGURE_T_FINAL, seed=SEED)
    factories = build_controller_factories(PLANT_ID, nominal=nominal, use_bounds=USE_BOUNDS)

    trajectories: Dict[str, object] = {}
    for name in controller_names:
        controller = factories[name]()
        plant = build_first_order_grid(family)[PLANT_ID]
        result = simulate_closed_loop(plant=plant, controller=controller, scenario=scenario, cfg=cfg)
        trajectories[name] = result.traj
    return trajectories


def _format_scale(scale: float) -> str:
    if abs(scale - round(scale)) < 1e-12:
        return str(int(round(scale)))
    return f"{scale:g}"


def _plot_stage(
    ax,
    *,
    stress_scale: float,
    controller_names: List[str],
    title_line_2: str,
    highlight_names: List[str],
) -> None:
    names = sorted(set(controller_names), key=_controller_order)
    trajectories = _simulate_stage(stress_scale, names)
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
        if name in highlight_names:
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

    scale_text = _format_scale(stress_scale)
    ax.set_title(f"Noise scale $x={scale_text}$\n{title_line_2}", fontsize=13, pad=10)
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
    breakpoints = [bp for bp in _load_breakpoints() if "RL (PID-features)" not in bp["newly_failing"]]
    eligible = _eligible_controllers()
    if not breakpoints:
        raise RuntimeError("No measurement-noise failure breakpoints found.")

    panel_specs = []
    for stage in breakpoints:
        failing_text = ", ".join(DISPLAY_NAME_MAP[name] for name in stage["newly_failing"])
        panel_specs.append(
            {
                "stress_scale": float(stage["stress_scale"]),
                "title_line_2": f"newly failing: {failing_text}",
                "highlight_names": list(stage["newly_failing"]),
            }
        )
    panel_specs.append(
        {
            "stress_scale": 20.0,
            "title_line_2": "highest tested scale; MPC still passes all plants",
            "highlight_names": ["MPC"],
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, stage in zip(axes, panel_specs):
        _plot_stage(
            ax,
            stress_scale=float(stage["stress_scale"]),
            controller_names=eligible,
            title_line_2=str(stage["title_line_2"]),
            highlight_names=list(stage["highlight_names"]),
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
        ncol=4,
        frameon=False,
        fontsize=11,
        columnspacing=1.4,
        handlelength=2.8,
    )
    fig.suptitle(
        "Measurement-noise stress on the unstable + RHP-zero plant:\nnominal-screened controllers from first failure onset to the highest tested scale",
        fontsize=18,
        y=0.995,
    )
    fig.tight_layout(rect=[0.03, 0.07, 1.0, 0.95], h_pad=2.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "unstable_rhp_zero__measurement_noise_failure_stages.png"
    pdf_path = OUT_DIR / "unstable_rhp_zero__measurement_noise_failure_stages.pdf"
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    run()
