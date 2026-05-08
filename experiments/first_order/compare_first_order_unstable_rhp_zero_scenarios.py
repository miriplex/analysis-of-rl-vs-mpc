from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import sys
from typing import Callable, Dict, Optional, Tuple
import zlib

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from control_bench.config import FIRST_ORDER_COST_WEIGHTS, FIRST_ORDER_FAMILY as family
from control_bench.core.scenario_impl import BasicScenario
from control_bench.core.sim import simulate_closed_loop
from control_bench.core.types import SimConfig
from control_bench.controllers.mpc import MPCConfig, LinearMPCController
from control_bench.controllers.rl_numpy.controller import BackpropRLController
from control_bench.experiment_manifest import first_order_default_mlp_variant
from control_bench.plants.first_order_family import build_first_order_grid
from control_bench.plants.lti_state_space import DiscreteLTIPlant
from control_bench.plants.second_order_family import expm_small


PLANT_ID = "unstable__rhp_zero"


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


def n_steps_for(*, t_final: float, dt: float) -> int:
    return int(np.floor(float(t_final) / float(dt))) + 1


def step_index(*, t: float, dt: float) -> int:
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


def _discretize_exact(A_c: np.ndarray, B_c: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    A_c = np.asarray(A_c, dtype=float)
    B_c = np.asarray(B_c, dtype=float)
    n = int(A_c.shape[0])

    M = np.zeros((n + 1, n + 1), dtype=float)
    M[:n, :n] = A_c
    M[:n, n:] = B_c

    Md = np.asarray(expm_small(M * float(dt)), dtype=float)
    return Md[:n, :n], Md[:n, n:]


def build_secret_pole_mismatch_plant(
    plant_id: str,
    *,
    hidden_pole: float = 10.0,
    use_bounds: bool = False,
) -> DiscreteLTIPlant:
    if hidden_pole <= 0:
        raise ValueError("hidden_pole must be > 0")

    nominal = build_first_order_grid(family)[plant_id]
    p, z = plant_pz_from_id(plant_id)

    if z is None:
        c1 = float(family.k)
        d1 = 0.0
    else:
        c1 = float(family.k * (p - z))
        d1 = float(family.k)

    a = float(hidden_pole)

    A_c = np.array(
        [
            [float(p), 0.0],
            [a * c1, -a],
        ],
        dtype=float,
    )
    B_c = np.array(
        [
            [1.0],
            [a * d1],
        ],
        dtype=float,
    )
    A_d, B_d = _discretize_exact(A_c, B_c, dt=family.dt)

    u_bounds = nominal.u_bounds if use_bounds else None
    return DiscreteLTIPlant(
        dt=family.dt,
        A=A_d,
        B=B_d,
        C=np.array([[0.0, 1.0]], dtype=float),
        D=np.array([[0.0]], dtype=float),
        u_bounds=u_bounds,
    )


def make_nominal_plant(*, plant_id: str, use_bounds: bool) -> object:
    plant = build_first_order_grid(family)[plant_id]
    plant.u_bounds = plant.u_bounds if use_bounds else None
    return plant


def load_controller_factories(*, plant_id: str, use_bounds: bool) -> "OrderedDict[str, Callable[[], object]]":
    default_mlp_variant = first_order_default_mlp_variant()
    nominal = build_first_order_grid(family)[plant_id]
    u_bounds = nominal.u_bounds if use_bounds else None
    u_min = family.u_min if use_bounds else None
    u_max = family.u_max if use_bounds else None

    weights_dir = EXPERIMENTS_ROOT / "results" / "rl_numpy" / "per_plant"
    rl_pid_path = str(weights_dir / f"rl_pidfeat__{plant_id}.npz")
    rl_rich_path = str(weights_dir / f"{default_mlp_variant['canonical_filename_prefix']}__{plant_id}.npz")
    require_file(rl_pid_path)
    require_file(rl_rich_path)

    factories: "OrderedDict[str, Callable[[], object]]" = OrderedDict()
    factories["MPC"] = lambda: LinearMPCController(
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
        name="MPC",
    )
    factories["RL (PID-features)"] = lambda: BackpropRLController.load_npz(
        rl_pid_path,
        kind="pidfeat",
        dt=family.dt,
        u_bounds=u_bounds,
        u_min=u_min,
        u_max=u_max,
        name="RL (PID-features)",
    )
    factories[str(default_mlp_variant["label"])] = lambda: BackpropRLController.load_npz(
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
    return factories


def make_disturbed_scenario(
    *,
    plant_id: str,
    io: object,
    t_final: float,
    seed: int,
    r_step: float,
    sigma_y: float,
    sigma_du: float,
    du_step_time: float,
    du_step_mag: float,
) -> BasicScenario:
    dt = float(family.dt)
    n_steps = n_steps_for(t_final=t_final, dt=dt)
    seed_pid = (int(seed) + int(zlib.crc32(plant_id.encode("utf-8")))) % (2**32)
    rng = np.random.default_rng(seed_pid)

    noise_y = rng.normal(loc=0.0, scale=float(sigma_y), size=(n_steps,))
    noise_du = rng.normal(loc=0.0, scale=float(sigma_du), size=(n_steps,))
    step_on = (np.arange(n_steps, dtype=float) * dt) >= float(du_step_time)
    du = noise_du + (float(du_step_mag) * step_on.astype(float))

    return BasicScenario(
        io=io,
        reference_fn=step_reference(r_step),
        disturbance_u_fn=make_disturbance_u_fn_from_trace(du=du, dt=dt),
        measurement_fn=make_measurement_fn_from_trace(noise_y=noise_y, dt=dt),
    )


def run(*, use_bounds: bool = False, hidden_pole: float = 10.0) -> None:
    figures_dir = (
        EXPERIMENTS_ROOT / "results" / "figures" / "first_order_unstable_rhp_zero_scenarios"
    )
    figures_dir.mkdir(parents=True, exist_ok=True)

    plant_id = PLANT_ID
    t_final = 20.0
    seed = 0
    r_step = -2.0

    sigma_y = 0.02
    sigma_du = 0.02
    du_step_time = 5.0
    du_step_mag = 0.10

    nominal = build_first_order_grid(family)[plant_id]
    cfg = SimConfig(t_final=t_final, seed=seed)

    scenarios: "OrderedDict[str, Tuple[BasicScenario, Callable[[], object]]]" = OrderedDict()
    scenarios["Vanilla"] = (
        BasicScenario(io=nominal.io, reference_fn=step_reference(r_step)),
        lambda: make_nominal_plant(plant_id=plant_id, use_bounds=use_bounds),
    )
    scenarios["Disturbance"] = (
        make_disturbed_scenario(
            plant_id=plant_id,
            io=nominal.io,
            t_final=t_final,
            seed=seed,
            r_step=r_step,
            sigma_y=sigma_y,
            sigma_du=sigma_du,
            du_step_time=du_step_time,
            du_step_mag=du_step_mag,
        ),
        lambda: make_nominal_plant(plant_id=plant_id, use_bounds=use_bounds),
    )
    scenarios["Plant mismatch"] = (
        BasicScenario(io=nominal.io, reference_fn=step_reference(r_step)),
        lambda: build_secret_pole_mismatch_plant(
            plant_id,
            hidden_pole=hidden_pole,
            use_bounds=use_bounds,
        ),
    )

    controller_factories = load_controller_factories(plant_id=plant_id, use_bounds=use_bounds)

    runs: Dict[str, Dict[str, object]] = OrderedDict()
    for controller_name, controller_factory in controller_factories.items():
        controller_runs: Dict[str, object] = OrderedDict()
        for scenario_name, (scenario, plant_factory) in scenarios.items():
            controller = controller_factory()
            plant = plant_factory()
            res = simulate_closed_loop(plant=plant, controller=controller, scenario=scenario, cfg=cfg)
            controller_runs[scenario_name] = res.traj
        runs[controller_name] = controller_runs

    p, z = plant_pz_from_id(plant_id)
    subtitle = tf_string(family.k, p, z)

    reference_scenario = scenarios["Vanilla"][0]
    t = next(iter(next(iter(runs.values())).values())).t
    r_trace = np.array([reference_scenario.reference(float(ti))[0] for ti in t], dtype=float)

    scenario_colors = {
        "Vanilla": "#1f77b4",
        "Disturbance": "#d62728",
        "Plant mismatch": "#2ca02c",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    for ax, (controller_name, controller_runs) in zip(axes, runs.items()):
        ax.plot(t, r_trace, "k--", linewidth=2.0, label="reference")
        for scenario_name, traj in controller_runs.items():
            ax.plot(
                traj.t,
                traj.y[:, 0],
                label=scenario_name,
                color=scenario_colors[scenario_name],
                linewidth=2.0,
            )
        ax.set_title(controller_name)
        ax.set_xlabel("Time [s]")
        ax.grid(True)

    axes[0].set_ylabel("Output y")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True)
    fig.suptitle(
        "unstable__rhp_zero scenario comparison by controller\n"
        f"{subtitle} | disturbance: meas σ={sigma_y:g}, du σ={sigma_du:g}, "
        f"du_step={du_step_mag:g}@{du_step_time:g}s | pole mismatch: x 10/(s+10)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))

    out_path = figures_dir / f"{plant_id}__scenario_comparison__bounds_{'on' if use_bounds else 'off'}.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    print(f"Saved scenario comparison to: {out_path}")


if __name__ == "__main__":
    run(use_bounds=False, hidden_pole=10.0)
