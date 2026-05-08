from __future__ import annotations

from pathlib import Path
import sys
from typing import Callable, Dict, Optional, Tuple
import zlib

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from control_bench.config import (
    FIRST_ORDER_COST_WEIGHTS,
    FIRST_ORDER_FAMILY as family,
    FIRST_ORDER_PLANT_IDS,
)
from control_bench.core.sim import simulate_closed_loop
from control_bench.core.types import SimConfig
from control_bench.core.scenario_impl import BasicScenario
from control_bench.plants.first_order_family import build_first_order_grid
from control_bench.controllers.mpc import MPCConfig, LinearMPCController
from control_bench.controllers.rl_numpy.controller import BackpropRLController
from control_bench.experiment_manifest import first_order_default_mlp_variant


def step_reference(level: float = 1.0):
    def _r(t: float) -> np.ndarray:
        return np.array([level], dtype=float)

    return _r


def plant_pz_from_id(plant_id: str) -> Tuple[float, Optional[float]]:
    pole_tag, zero_tag = plant_id.split("__", 1)

    if pole_tag == "stable":
        p = float(family.stable_p)
    elif pole_tag == "integrator":
        p = 0.0
    elif pole_tag == "unstable":
        p = float(family.unstable_p)
    else:
        raise ValueError(f"Unknown pole tag: {pole_tag}")

    if zero_tag == "no_zero":
        z = None
    elif zero_tag == "lhp_zero":
        z = float(family.lhp_z)
    elif zero_tag == "rhp_zero":
        z = float(family.rhp_z)
    else:
        raise ValueError(f"Unknown zero tag: {zero_tag}")

    return p, z


def tf_string(k: float, p: float, z: Optional[float]) -> str:
    if z is None:
        return f"G(s) = {k:g} / (s - ({p:g}))"
    return f"G(s) = {k:g}(s - ({z:g})) / (s - ({p:g}))"


def require_file(path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing file: {path}\nRun: python experiments/train_rl_first_order_numpy.py")


def n_steps_for(*, t_final: float, dt: float) -> int:
    # Match simulate_closed_loop(): floor(t_final/dt) + 1
    return int(np.floor(float(t_final) / float(dt))) + 1


def step_index(*, t: float, dt: float) -> int:
    # t is generated as k*dt in the simulator; round is robust to fp noise.
    return int(np.round(float(t) / float(dt)))


def make_measurement_fn_from_trace(*, noise_y: np.ndarray, dt: float) -> Callable[[np.ndarray, float], np.ndarray]:
    noise_y = np.asarray(noise_y, dtype=float).reshape(-1)
    dt = float(dt)

    def _measurement(y: np.ndarray, t: float) -> np.ndarray:
        k = step_index(t=t, dt=dt)
        if k < 0 or k >= noise_y.size:
            return y
        return y + np.array([float(noise_y[k])], dtype=float)

    return _measurement


def make_disturbance_u_fn_from_trace(*, du: np.ndarray, dt: float) -> Callable[[float], np.ndarray]:
    du = np.asarray(du, dtype=float).reshape(-1)
    dt = float(dt)

    def _du(t: float) -> np.ndarray:
        k = step_index(t=t, dt=dt)
        if k < 0 or k >= du.size:
            return np.zeros((1,), dtype=float)
        return np.array([float(du[k])], dtype=float)

    return _du


def run_all(*, use_bounds: bool = False, show_bounds_on_plot: bool = False, save_control_plots: bool = False) -> None:
    default_mlp_variant = first_order_default_mlp_variant()
    # Noise / disturbance configuration
    sigma_y = 0.02        # measurement noise std (obs = y + noise)
    sigma_du = 0.02       # input disturbance std (u_actual = u + du)
    du_step_time = 5.0    # optional step disturbance time [s]
    du_step_mag = 0.10    # optional step disturbance magnitude (set to 0 to disable)

    tag = f"meas{sigma_y:g}_du{sigma_du:g}_step{du_step_mag:g}_t{du_step_time:g}".replace(".", "p").replace("-", "m")

    figures_dir = EXPERIMENTS_ROOT / "results" / "figures" / "first_order_disturbed" / tag
    figures_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = EXPERIMENTS_ROOT / "results" / "rl_numpy" / "per_plant"

    t_final = 10.0
    seed = 0
    r_step = -2.0

    for plant_id in FIRST_ORDER_PLANT_IDS:
        dummy = build_first_order_grid(family)[plant_id]

        # Pre-sample disturbance/noise traces ONCE per plant_id so every controller
        # sees exactly the same realization (fair side-by-side comparison).
        dt = float(dummy.dt)
        n_steps = n_steps_for(t_final=t_final, dt=dt)
        seed_pid = (int(seed) + int(zlib.crc32(plant_id.encode("utf-8")))) % (2**32)
        rng = np.random.default_rng(seed_pid)

        noise_y = rng.normal(loc=0.0, scale=float(sigma_y), size=(n_steps,))
        noise_du = rng.normal(loc=0.0, scale=float(sigma_du), size=(n_steps,))
        step_on = (np.arange(n_steps, dtype=float) * dt) >= float(du_step_time)
        du = noise_du + (float(du_step_mag) * step_on.astype(float))

        # Decide whether to enforce actuator bounds
        u_bounds = dummy.u_bounds if use_bounds else None
        u_min = family.u_min if use_bounds else None
        u_max = family.u_max if use_bounds else None

        scenario = BasicScenario(
            io=dummy.io,
            reference_fn=step_reference(r_step),
            disturbance_u_fn=make_disturbance_u_fn_from_trace(du=du, dt=dt),
            measurement_fn=make_measurement_fn_from_trace(noise_y=noise_y, dt=dt),
        )
        cfg = SimConfig(t_final=t_final, seed=seed)

        # MPC (model from this plant; disturbance/noise acts on the simulated "real" loop)
        mpc = LinearMPCController(
            MPCConfig(
                N=50,
                qy=FIRST_ORDER_COST_WEIGHTS.qy,
                ru=FIRST_ORDER_COST_WEIGHTS.ru,
                qu=FIRST_ORDER_COST_WEIGHTS.qu,
                pgd_iters=80,
            ),
            dt=dummy.dt,
            A=dummy.A,
            B=dummy.B,
            C=dummy.C,
            D=dummy.D,
            u_bounds=u_bounds,
            name="MPC",
        )

        # RL weights (per-plant)
        rl_simple_path = str(weights_dir / f"rl_pidfeat__{plant_id}.npz")
        rl_rich_path = str(weights_dir / f"{default_mlp_variant['canonical_filename_prefix']}__{plant_id}.npz")
        require_file(rl_simple_path)
        require_file(rl_rich_path)

        rl_simple = BackpropRLController.load_npz(
            rl_simple_path,
            kind="pidfeat",
            dt=family.dt,
            u_bounds=u_bounds,
            u_min=u_min,
            u_max=u_max,
            name="RL (PID-features)",
        )

        rl_rich = BackpropRLController.load_npz(
            rl_rich_path,
            kind="rich",
            dt=family.dt,
            u_bounds=u_bounds,
            u_min=u_min,
            u_max=u_max,
            mlp_hidden=tuple(default_mlp_variant["hidden_layers"]),
            mlp_activation=str(default_mlp_variant["activation"]),
            rich_feature_set=str(default_mlp_variant["feature_set"]),
            name=str(default_mlp_variant["label"]),
        )

        controllers = [mpc, rl_simple, rl_rich]

        runs: Dict[str, object] = {}
        for c in controllers:
            plant = build_first_order_grid(family)[plant_id]
            # disable bounds in the plant too (so simulator doesn't clip)
            plant.u_bounds = u_bounds
            res = simulate_closed_loop(plant=plant, controller=c, scenario=scenario, cfg=cfg)
            runs[c.name] = res.traj

        p, z = plant_pz_from_id(plant_id)
        title = f"{plant_id} | p={p:g}, z={'None' if z is None else f'{z:g}'}"
        subtitle = tf_string(family.k, p, z)
        noise_title = f"meas σ={sigma_y:g}, du σ={sigma_du:g}, du_step={du_step_mag:g}@{du_step_time:g}s"

        t = next(iter(runs.values())).t
        r_trace = np.array([scenario.reference(float(ti))[0] for ti in t], dtype=float)

        # ----- Output plot -----
        plt.figure()
        plt.plot(t, r_trace, "k--", linewidth=2.0, label="reference")
        for name, traj in runs.items():
            plt.plot(traj.t, traj.y[:, 0], label=name)

        plt.title(
            f"Output y(t) | {title}\n{subtitle} | bounds={'on' if use_bounds else 'off'} | {noise_title}"
        )
        plt.xlabel("Time [s]")
        plt.ylabel("Output y")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / f"{plant_id}__output__bounds_{'on' if use_bounds else 'off'}.png", dpi=160)
        plt.close()

        if save_control_plots:
            # ----- Control plot -----
            plt.figure()
            for name, traj in runs.items():
                plt.plot(traj.t, traj.u[:, 0], label=name)

            # Only draw bound lines if requested and bounds are enabled
            if show_bounds_on_plot and (u_bounds is not None):
                umin = float(u_bounds.low[0])
                umax = float(u_bounds.high[0])
                plt.axhline(umin, linestyle="--", linewidth=1.5, label="u_min")
                plt.axhline(umax, linestyle="--", linewidth=1.5, label="u_max")

            plt.title(
                f"Control u(t) | {title} | bounds={'on' if use_bounds else 'off'} | {noise_title}"
            )
            plt.xlabel("Time [s]")
            plt.ylabel("Control u")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(figures_dir / f"{plant_id}__control__bounds_{'on' if use_bounds else 'off'}.png", dpi=160)
            plt.close()

        print(f"Saved disturbed plots for {plant_id} (bounds={'on' if use_bounds else 'off'})")

    print(f"\nAll figures saved to: {figures_dir}")


if __name__ == "__main__":
    run_all(use_bounds=False, show_bounds_on_plot=False, save_control_plots=False)
