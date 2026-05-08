from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from control_bench.config import SECOND_ORDER_FAMILY as family
from control_bench.config import SECOND_ORDER_SINGLE_LIGHTLY_DAMPED as custom_plant_cfg
from control_bench.core.scenario_impl import BasicScenario
from control_bench.core.sim import simulate_closed_loop
from control_bench.core.types import SimConfig
from control_bench.controllers.mpc import MPCConfig, LinearMPCController
from control_bench.controllers.rl_numpy.controller import BackpropRLController
from control_bench.plants.second_order_family import (
    build_second_order_single_plant,
    build_second_order_suite_plant,
    tf_coeffs_for_plant_id,
)


PLANT_IDS = (
    custom_plant_cfg.plant_id,
    "boss_unstable_osc_rhp_zero",
)


def step_reference(level: float = 1.0):
    def _r(t: float) -> np.ndarray:
        return np.array([level], dtype=float)

    return _r


def build_experiment_plants():
    return {
        custom_plant_cfg.plant_id: build_second_order_single_plant(custom_plant_cfg),
        "boss_unstable_osc_rhp_zero": build_second_order_suite_plant("boss_unstable_osc_rhp_zero", family),
    }


def build_experiment_plant(plant_id: str):
    return build_experiment_plants()[plant_id]


def plant_tf_coeffs(plant_id: str) -> tuple[float, float, float, float]:
    if plant_id == custom_plant_cfg.plant_id:
        return (
            float(custom_plant_cfg.b1),
            float(custom_plant_cfg.b0),
            float(custom_plant_cfg.a1),
            float(custom_plant_cfg.a0),
        )
    return tf_coeffs_for_plant_id(plant_id, family)


def plant_bounds(plant_id: str) -> tuple[float | None, float | None]:
    if plant_id == custom_plant_cfg.plant_id:
        return custom_plant_cfg.u_min, custom_plant_cfg.u_max
    return family.u_min, family.u_max


def tf_string(plant_id: str) -> str:
    b1, b0, a1, a0 = plant_tf_coeffs(plant_id)
    if abs(float(b1)) < 1e-12:
        num = f"{b0:g}"
    elif abs(float(b0)) < 1e-12:
        num = f"{b1:g}s"
    elif float(b0) >= 0.0:
        num = f"{b1:g}s + {b0:g}"
    else:
        num = f"{b1:g}s - {abs(b0):g}"
    return f"G(s) = ({num}) / (s^2 + {a1:g}s + {a0:g})"


def require_file(path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            f"Run: python experiments/train_rl_second_order_single_plant_numpy.py"
        )


def simulate_open_loop(*, plant_id: str, t_final: float, u_step: float) -> tuple[np.ndarray, np.ndarray]:
    plant = build_experiment_plant(plant_id)
    dt = float(plant.dt)
    n_steps = int(np.floor(float(t_final) / dt)) + 1

    t = np.arange(n_steps, dtype=float) * dt
    y = np.zeros((n_steps,), dtype=float)
    u = np.array([float(u_step)], dtype=float)

    plant.reset(x0=np.zeros((2,), dtype=float))
    for k in range(n_steps):
        y[k] = float(plant.output()[0])
        plant.step(u)

    return t, y


def run(*, use_bounds: bool = False, show_bounds_on_plot: bool = False, save_control_plots: bool = False) -> None:
    figures_dir = EXPERIMENTS_ROOT / "results" / "figures" / "second_order_single"
    figures_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = (
        EXPERIMENTS_ROOT / "results" / "rl_numpy" / "second_order_single" / "per_plant"
    )

    t_final = 20.0
    seed = 0
    r_step = -2.0

    for plant_id in PLANT_IDS:
        nominal = build_experiment_plant(plant_id)
        u_bounds = nominal.u_bounds if use_bounds else None
        u_min, u_max = plant_bounds(plant_id)

        scenario = BasicScenario(
            io=nominal.io,
            reference_fn=step_reference(r_step),
        )
        cfg = SimConfig(t_final=t_final, seed=seed)

        mpc = LinearMPCController(
            MPCConfig(N=50, qy=1.0, ru=1.0, pgd_iters=80),
            dt=nominal.dt,
            A=nominal.A,
            B=nominal.B,
            C=nominal.C,
            D=nominal.D,
            u_bounds=u_bounds,
            name="MPC",
        )

        rl_simple_path = str(weights_dir / f"rl_pidfeat__{plant_id}.npz")
        rl_rich_path = str(weights_dir / f"rl_rich__{plant_id}.npz")
        require_file(rl_simple_path)
        require_file(rl_rich_path)

        rl_simple = BackpropRLController.load_npz(
            rl_simple_path,
            kind="pidfeat",
            dt=nominal.dt,
            u_bounds=u_bounds,
            u_min=u_min,
            u_max=u_max,
            name="RL (PID-features)",
        )

        rl_rich = BackpropRLController.load_npz(
            rl_rich_path,
            kind="rich",
            dt=nominal.dt,
            u_bounds=u_bounds,
            u_min=u_min,
            u_max=u_max,
            mlp_hidden=(16, 16),
            mlp_activation="tanh",
            name="RL (MLP rich)",
        )

        controllers = [mpc, rl_simple, rl_rich]

        runs = {}
        for controller in controllers:
            plant = build_experiment_plant(plant_id)
            plant.u_bounds = u_bounds
            res = simulate_closed_loop(plant=plant, controller=controller, scenario=scenario, cfg=cfg)
            runs[controller.name] = res.traj

        t = next(iter(runs.values())).t
        r_trace = np.array([scenario.reference(float(ti))[0] for ti in t], dtype=float)
        t_open, y_open = simulate_open_loop(plant_id=plant_id, t_final=t_final, u_step=r_step)
        subtitle = tf_string(plant_id)

        plt.figure()
        plt.plot(t, r_trace, "k--", linewidth=2.0, label="reference")
        plt.plot(t_open, y_open, "--", color="0.5", linewidth=2.0, label="open-loop")
        for name, traj in runs.items():
            plt.plot(traj.t, traj.y[:, 0], label=name)

        plt.title(
            f"Output y(t) | {plant_id}\n{subtitle} | bounds={'on' if use_bounds else 'off'}"
        )
        plt.xlabel("Time [s]")
        plt.ylabel("Output y")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / f"{plant_id}__output__bounds_{'on' if use_bounds else 'off'}.png", dpi=160)
        plt.close()

        if save_control_plots:
            plt.figure()
            for name, traj in runs.items():
                plt.plot(traj.t, traj.u[:, 0], label=name)

            if show_bounds_on_plot and (u_bounds is not None):
                umin = float(u_bounds.low[0])
                umax = float(u_bounds.high[0])
                plt.axhline(umin, linestyle="--", linewidth=1.5, label="u_min")
                plt.axhline(umax, linestyle="--", linewidth=1.5, label="u_max")

            plt.title(
                f"Control u(t) | {plant_id}\n{subtitle} | bounds={'on' if use_bounds else 'off'}"
            )
            plt.xlabel("Time [s]")
            plt.ylabel("Control u")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(figures_dir / f"{plant_id}__control__bounds_{'on' if use_bounds else 'off'}.png", dpi=160)
            plt.close()

        print(f"Saved plots for {plant_id} (bounds={'on' if use_bounds else 'off'})")

    print(f"\nAll figures saved to: {figures_dir}")


if __name__ == "__main__":
    run(use_bounds=False, show_bounds_on_plot=False, save_control_plots=False)
