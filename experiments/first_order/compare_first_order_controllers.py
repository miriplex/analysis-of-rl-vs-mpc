from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from control_bench.config import FIRST_ORDER_COST_WEIGHTS, FIRST_ORDER_FAMILY as family
from control_bench.core.sim import simulate_closed_loop
from control_bench.core.types import SimConfig
from control_bench.core.scenario_impl import BasicScenario

from control_bench.plants.first_order_family import build_first_order_grid
from control_bench.controllers.pid import PIDConfig, PIDController
from control_bench.controllers.mpc import MPCConfig, LinearMPCController
from control_bench.controllers.rl_numpy.controller import BackpropRLController
from control_bench.experiment_manifest import first_order_default_mlp_variant


def step_reference(level: float = 1.0):
    def _r(t: float) -> np.ndarray:
        return np.array([level], dtype=float)
    return _r


def require_file(path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Missing file: {path}\nRun: python experiments/train_rl_first_order_numpy.py"
        )


def run_compare(
    *,
    plant_id: str = "stable__no_zero",
    t_final: float = 8.0,
    seed: int = 0,
    r_step: float = 1.0,
) -> None:
    default_mlp_variant = first_order_default_mlp_variant()
    dummy = build_first_order_grid(family)[plant_id]

    scenario = BasicScenario(
        io=dummy.io,
        reference_fn=step_reference(r_step),
    )
    cfg = SimConfig(t_final=t_final, seed=seed)

    pid = PIDController(
        PIDConfig(kp=2.0, ki=0.8, kd=0.1, derivative_filter_tau=0.02, anti_windup="conditional"),
        dt=family.dt,
        u_bounds=dummy.u_bounds,
        name="PID",
    )

    mpc = LinearMPCController(
        MPCConfig(
            N=15,
            qy=FIRST_ORDER_COST_WEIGHTS.qy,
            ru=FIRST_ORDER_COST_WEIGHTS.ru,
            qu=FIRST_ORDER_COST_WEIGHTS.qu,
            pgd_iters=80,
        ),
        dt=dummy.dt,
        A=dummy.A, B=dummy.B, C=dummy.C, D=dummy.D,
        u_bounds=dummy.u_bounds,
        name="MPC",
    )

    weights_dir = EXPERIMENTS_ROOT / "results" / "rl_numpy" / "per_plant"
    rl_simple_path = str(weights_dir / f"rl_pidfeat__{plant_id}.npz")
    rl_rich_path = str(weights_dir / f"{default_mlp_variant['canonical_filename_prefix']}__{plant_id}.npz")
    require_file(rl_simple_path)
    require_file(rl_rich_path)

    rl_simple = BackpropRLController.load_npz(
        rl_simple_path,
        kind="pidfeat",
        dt=family.dt,
        u_bounds=dummy.u_bounds,
        u_min=family.u_min,
        u_max=family.u_max,
        name="RL (PID-features)",
    )

    rl_rich = BackpropRLController.load_npz(
        rl_rich_path,
        kind="rich",
        dt=family.dt,
        u_bounds=dummy.u_bounds,
        u_min=family.u_min,
        u_max=family.u_max,
        mlp_hidden=tuple(default_mlp_variant["hidden_layers"]),
        mlp_activation=str(default_mlp_variant["activation"]),
        rich_feature_set=str(default_mlp_variant["feature_set"]),
        name=str(default_mlp_variant["label"]),
    )

    controllers = [pid, mpc, rl_simple, rl_rich]
    results = {}

    for c in controllers:
        plant = build_first_order_grid(family)[plant_id]
        res = simulate_closed_loop(plant=plant, controller=c, scenario=scenario, cfg=cfg)
        results[c.name] = res.traj

    t = next(iter(results.values())).t
    r_trace = np.array([scenario.reference(float(ti))[0] for ti in t], dtype=float)

    plt.figure()
    plt.plot(t, r_trace, "k--", linewidth=2.0, label="reference")
    for name, traj in results.items():
        plt.plot(traj.t, traj.y[:, 0], label=name)
    plt.title(f"{plant_id} | Output y(t)")
    plt.xlabel("Time [s]")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.figure()
    for name, traj in results.items():
        plt.plot(traj.t, traj.u[:, 0], label=name)
    if dummy.u_bounds is not None:
        plt.axhline(float(dummy.u_bounds.low[0]), linestyle="--", linewidth=1.5, label="u_min")
        plt.axhline(float(dummy.u_bounds.high[0]), linestyle="--", linewidth=1.5, label="u_max")
    plt.title(f"{plant_id} | Control u(t)")
    plt.xlabel("Time [s]")
    plt.ylabel("u")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    run_compare(plant_id="stable__no_zero")
