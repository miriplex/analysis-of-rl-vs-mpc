from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark_first_order_robustness_metrics import build_controllers, compute_metrics
from control_bench.config import FIRST_ORDER_FAMILY as family
from control_bench.core.scenario_impl import BasicScenario
from control_bench.core.sim import simulate_closed_loop
from control_bench.core.types import SimConfig, Trajectory
from control_bench.experiment_manifest import first_order_hidden_pole_stress_test
from control_bench.plants.first_order_family import build_first_order_grid


PLANT_ID = "unstable__rhp_zero"
SEED = 0
USE_BOUNDS = False
R_STEP = -2.0
T_FINAL = 7.0


def step_reference(level: float):
    def _r(t: float) -> np.ndarray:
        return np.array([level], dtype=float)

    return _r


def _controller_sort_key(name: str) -> Tuple[int, str]:
    if name == "MPC":
        return (0, name)
    if name == "RL (PID-features)":
        return (1, name)
    return (2, name)


def _plot_group(
    ax,
    *,
    title: str,
    r_trace: np.ndarray,
    trajectories: Dict[str, Trajectory],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> None:
    if trajectories:
        t = next(iter(trajectories.values())).t
        ax.plot(t, r_trace, "k--", linewidth=2.0, label="reference")

        for controller_name in sorted(trajectories, key=_controller_sort_key):
            traj = trajectories[controller_name]
            ax.plot(traj.t, traj.y[:, 0], linewidth=2.0, label=controller_name)
        ax.legend(loc="best", fontsize=10)
    else:
        t = np.linspace(xlim[0], xlim[1], 2)
        ax.plot(t, np.full_like(t, r_trace[0]), "k--", linewidth=2.0, label="reference")
        ax.text(
            0.5,
            0.5,
            "No failing controllers\nunder current failure rule",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            bbox={"facecolor": "white", "edgecolor": "#999999", "boxstyle": "round,pad=0.35"},
        )

    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Output y")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.35)


def _make_output_plot(
    *,
    r_trace: np.ndarray,
    successful: Dict[str, Trajectory],
    failing: Dict[str, Trajectory],
    save_path: Path,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.8), sharex=True, sharey=True)

    _plot_group(
        axes[0],
        title="Successful controllers",
        r_trace=r_trace,
        trajectories=successful,
        xlim=xlim,
        ylim=ylim,
    )
    _plot_group(
        axes[1],
        title="Failing controllers",
        r_trace=r_trace,
        trajectories=failing,
        xlim=xlim,
        ylim=ylim,
    )

    fig.suptitle(f"{PLANT_ID} | nominal plant | reference step = {R_STEP:g}")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def main() -> None:
    nominal = build_first_order_grid(family)[PLANT_ID]
    scenario = BasicScenario(io=nominal.io, reference_fn=step_reference(R_STEP))
    sim_cfg = SimConfig(t_final=T_FINAL, seed=SEED)

    trajectories: Dict[str, Trajectory] = {}
    metrics_by_controller: Dict[str, dict] = {}

    for controller in build_controllers(PLANT_ID, nominal=nominal, use_bounds=USE_BOUNDS):
        plant = build_first_order_grid(family)[PLANT_ID]
        plant.u_bounds = nominal.u_bounds if USE_BOUNDS else None
        result = simulate_closed_loop(plant=plant, controller=controller, scenario=scenario, cfg=sim_cfg)
        trajectories[controller.name] = result.traj
        metrics_by_controller[controller.name] = compute_metrics(result.traj)

    successful = {
        controller_name: traj
        for controller_name, traj in trajectories.items()
        if not bool(metrics_by_controller[controller_name]["failure"])
    }
    failing = {
        controller_name: traj
        for controller_name, traj in trajectories.items()
        if bool(metrics_by_controller[controller_name]["failure"])
    }

    if not successful:
        raise RuntimeError("No successful controllers found for unstable__rhp_zero.")

    t = next(iter(trajectories.values())).t
    r_trace = np.array([scenario.reference(float(ti))[0] for ti in t], dtype=float)
    all_y = np.concatenate([traj.y[:, 0] for traj in trajectories.values()])
    y_min = float(min(np.min(all_y), np.min(r_trace)))
    y_max = float(max(np.max(all_y), np.max(r_trace)))
    y_pad = 0.05 * max(1.0, y_max - y_min)
    xlim = (float(t[0]), float(t[-1]))
    ylim = (y_min - y_pad, y_max + y_pad)

    figures_dir = EXPERIMENTS_ROOT / "results" / "figures" / "first_order_nominal_screening"
    combined_path = figures_dir / f"{PLANT_ID}__success_vs_failure__rstep_{str(R_STEP).replace('.', 'p')}.pdf"

    _make_output_plot(
        r_trace=r_trace,
        successful=successful,
        failing=failing,
        save_path=combined_path,
        xlim=xlim,
        ylim=ylim,
    )

    print("Successful controllers:")
    for controller_name in sorted(successful, key=_controller_sort_key):
        metrics = metrics_by_controller[controller_name]
        print(
            f"- {controller_name} | mean_cost={metrics['mean_cost']:.6g} | "
            f"peak_error={metrics['peak_error']:.6g} | tail_rms={metrics['tail_rms_error']:.6g}"
        )

    print("\nFailing controllers:")
    if failing:
        for controller_name in sorted(failing, key=_controller_sort_key):
            metrics = metrics_by_controller[controller_name]
            print(
                f"- {controller_name} | reason={metrics['failure_reason'] or 'unknown'} | "
                f"detail={metrics['failure_detail'] or 'n/a'}"
            )
    else:
        print("- none")

    print(f"\nSaved: {combined_path}")


if __name__ == "__main__":
    main()
