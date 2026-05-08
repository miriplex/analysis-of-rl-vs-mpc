from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

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
from control_bench.core.scenario_impl import BasicScenario
from control_bench.core.sim import simulate_closed_loop
from control_bench.core.types import SimConfig
from control_bench.controllers.mpc import MPCConfig, LinearMPCController
from control_bench.controllers.rl_numpy.controller import BackpropRLController
from control_bench.experiment_manifest import first_order_default_mlp_variant
from control_bench.plants.first_order_family import build_first_order_grid
from control_bench.plants.lti_state_space import DiscreteLTIPlant


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
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            f"Run: python experiments/train_rl_first_order_numpy.py"
        )


def build_gain_mismatch_plant(
    plant_id: str,
    *,
    gain_scale: float = 2.0,
    use_bounds: bool = False,
) -> DiscreteLTIPlant:
    """
    Real plant used in the gain-mismatch experiment.

    Controllers are designed/trained against the nominal first-order plant G_nom(s).
    The actual simulated plant is:

        G_real(s) = gain_scale * G_nom(s)
    """
    nominal = build_first_order_grid(family)[plant_id]
    if gain_scale <= 0:
        raise ValueError("gain_scale must be > 0")

    u_bounds = nominal.u_bounds if use_bounds else None
    A = np.asarray(nominal.A, dtype=float)
    if A.ndim == 0:
        A = A.reshape(1, 1)

    B = np.asarray(nominal.B, dtype=float)
    if B.ndim == 0:
        B = B.reshape(1, 1)

    C = np.asarray(nominal.C, dtype=float)
    if C.ndim == 0:
        C = C.reshape(1, 1)

    D = np.asarray(nominal.D, dtype=float)
    return DiscreteLTIPlant(
        dt=nominal.dt,
        A=A.copy(),
        B=B.copy(),
        C=(float(gain_scale) * C.copy()),
        D=(float(gain_scale) * D.copy()),
        u_bounds=u_bounds,
    )


def actual_tf_string(k: float, p: float, z: Optional[float], gain_scale: float) -> str:
    base = tf_string(k, p, z).replace("G(s) = ", "")
    return f"G_real(s) = {gain_scale:g} * ({base})"


def run_all(
    *,
    gain_scale: float = 2.0,
    plant_ids: Optional[Sequence[str]] = None,
    use_bounds: bool = False,
    show_bounds_on_plot: bool = False,
    save_control_plots: bool = False,
) -> None:
    default_mlp_variant = first_order_default_mlp_variant()
    tag = f"gain_x{gain_scale:g}".replace(".", "p").replace("-", "m")
    figures_dir = EXPERIMENTS_ROOT / "results" / "figures" / "first_order_mismatch" / tag
    figures_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = EXPERIMENTS_ROOT / "results" / "rl_numpy" / "per_plant"

    t_final = 20.0
    seed = 0
    r_step = -2.0
    plant_ids = tuple(FIRST_ORDER_PLANT_IDS if plant_ids is None else plant_ids)

    for plant_id in plant_ids:
        nominal = build_first_order_grid(family)[plant_id]

        u_bounds = nominal.u_bounds if use_bounds else None
        u_min = family.u_min if use_bounds else None
        u_max = family.u_max if use_bounds else None

        scenario = BasicScenario(
            io=nominal.io,
            reference_fn=step_reference(r_step),
        )
        cfg = SimConfig(t_final=t_final, seed=seed)

        # MPC intentionally keeps the nominal model. Only the real plant is mismatched.
        mpc = LinearMPCController(
            MPCConfig(
                N=50,
                qy=FIRST_ORDER_COST_WEIGHTS.qy,
                ru=FIRST_ORDER_COST_WEIGHTS.ru,
                qu=FIRST_ORDER_COST_WEIGHTS.qu,
                pgd_iters=80,
            ),
            dt=nominal.dt,
            A=nominal.A,
            B=nominal.B,
            C=nominal.C,
            D=nominal.D,
            u_bounds=u_bounds,
            name="MPC (nominal model)",
        )

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
        for controller in controllers:
            plant = build_gain_mismatch_plant(
                plant_id,
                gain_scale=gain_scale,
                use_bounds=use_bounds,
            )
            res = simulate_closed_loop(plant=plant, controller=controller, scenario=scenario, cfg=cfg)
            runs[controller.name] = res.traj

        p, z = plant_pz_from_id(plant_id)
        title = f"{plant_id} | gain scale={gain_scale:g}"
        subtitle = f"nominal: {tf_string(family.k, p, z)}"
        actual_subtitle = actual_tf_string(family.k, p, z, gain_scale)

        t = next(iter(runs.values())).t
        r_trace = np.array([scenario.reference(float(ti))[0] for ti in t], dtype=float)

        plt.figure()
        plt.plot(t, r_trace, "k--", linewidth=2.0, label="reference")
        for name, traj in runs.items():
            plt.plot(traj.t, traj.y[:, 0], label=name)

        plt.title(
            f"Output y(t) | {title}\n{subtitle}\n{actual_subtitle} | bounds={'on' if use_bounds else 'off'}"
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
                f"Control u(t) | {title}\nnominal controller model retained | bounds={'on' if use_bounds else 'off'}"
            )
            plt.xlabel("Time [s]")
            plt.ylabel("Control u")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(figures_dir / f"{plant_id}__control__bounds_{'on' if use_bounds else 'off'}.png", dpi=160)
            plt.close()

        print(f"Saved gain-mismatch plots for {plant_id} (bounds={'on' if use_bounds else 'off'})")

    print(f"\nAll figures saved to: {figures_dir}")


if __name__ == "__main__":
    run_all(gain_scale=2.0, use_bounds=False, show_bounds_on_plot=False)
