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

from control_bench.config import (
    INVERTED_PENDULUM_LINEARIZED,
    INVERTED_PENDULUM_MPC_EXPERIMENT,
    INVERTED_PENDULUM_RL_TRAINING,
)
from control_bench.controllers.inverted_pendulum_mpc import InvertedPendulumMPCConfig, InvertedPendulumMPCController
from control_bench.controllers.inverted_pendulum_rl import InvertedPendulumRLController
from control_bench.plants.inverted_pendulum import build_linearized_inverted_pendulum


def zero_reference() -> np.ndarray:
    return np.asarray(INVERTED_PENDULUM_MPC_EXPERIMENT.reference, dtype=float).reshape(4)


def require_file(path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            f"Run: python {PROJECT_ROOT / 'experiments' / 'inverted_pendulum' / 'train_rl_inverted_pendulum_numpy.py'}"
        )


def simulate_controller(*, controller, t_final: float, x0: np.ndarray) -> dict:
    plant = build_linearized_inverted_pendulum(INVERTED_PENDULUM_LINEARIZED)
    dt = float(plant.dt)
    n_steps = int(np.floor(float(t_final) / dt)) + 1
    t = np.arange(n_steps, dtype=float) * dt

    x = np.zeros((n_steps, 4), dtype=float)
    u = np.zeros((n_steps,), dtype=float)
    r = zero_reference()

    plant.reset(x0=x0)
    controller.reset()

    for k, tk in enumerate(t):
        obs = plant.observe()
        u_cmd = controller.step(r=r, obs=obs, t=tk)
        x[k] = plant.state
        u[k] = float(u_cmd[0])
        if k < n_steps - 1:
            plant.step(u_cmd)

    return {"t": t, "x": x, "u": u}


def run() -> None:
    figures_dir = EXPERIMENTS_ROOT / "results" / "figures" / "inverted_pendulum"
    figures_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = EXPERIMENTS_ROOT / "results" / "rl_numpy" / "inverted_pendulum_linearized" / "per_plant"
    pid_path = str(weights_dir / f"rl_pidfeat__{INVERTED_PENDULUM_RL_TRAINING.plant_id}.npz")
    rich_path = str(weights_dir / f"rl_rich__{INVERTED_PENDULUM_RL_TRAINING.plant_id}.npz")
    require_file(pid_path)
    require_file(rich_path)

    plant = build_linearized_inverted_pendulum(INVERTED_PENDULUM_LINEARIZED)
    u_bounds = plant.u_bounds
    u_min = INVERTED_PENDULUM_LINEARIZED.u_min
    u_max = INVERTED_PENDULUM_LINEARIZED.u_max

    mpc = InvertedPendulumMPCController(
        InvertedPendulumMPCConfig(
            N=INVERTED_PENDULUM_MPC_EXPERIMENT.N,
            q_state_diag=INVERTED_PENDULUM_MPC_EXPERIMENT.q_state_diag,
            ru=INVERTED_PENDULUM_MPC_EXPERIMENT.ru,
            qu=INVERTED_PENDULUM_MPC_EXPERIMENT.qu,
            pgd_iters=INVERTED_PENDULUM_MPC_EXPERIMENT.pgd_iters,
            pgd_step_size=INVERTED_PENDULUM_MPC_EXPERIMENT.pgd_step_size,
        ),
        dt=plant.dt,
        A=plant.A,
        B=plant.B,
        u_bounds=u_bounds,
        name="MPC",
    )
    rl_pid = InvertedPendulumRLController.load_npz(
        pid_path,
        kind="pidfeat",
        dt=plant.dt,
        u_bounds=u_bounds,
        u_min=u_min,
        u_max=u_max,
        name="RL (PID-features)",
    )
    rl_rich = InvertedPendulumRLController.load_npz(
        rich_path,
        kind="rich",
        dt=plant.dt,
        u_bounds=u_bounds,
        u_min=u_min,
        u_max=u_max,
        mlp_hidden=tuple(INVERTED_PENDULUM_RL_TRAINING.rich_hidden_layers),
        mlp_activation=INVERTED_PENDULUM_RL_TRAINING.rich_activation,
        name="RL (MLP rich)",
    )

    x0 = np.asarray(INVERTED_PENDULUM_MPC_EXPERIMENT.x0, dtype=float)
    t_final = float(INVERTED_PENDULUM_MPC_EXPERIMENT.t_final)
    runs = {
        "MPC": simulate_controller(controller=mpc, t_final=t_final, x0=x0),
        "RL (PID-features)": simulate_controller(controller=rl_pid, t_final=t_final, x0=x0),
        "RL (MLP rich)": simulate_controller(controller=rl_rich, t_final=t_final, x0=x0),
    }

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.5), sharex=True)

    for name, run_data in runs.items():
        axes[0].plot(run_data["t"], run_data["x"][:, 0], linewidth=2.2, label=f"{name} | x [m]")
        axes[0].plot(run_data["t"], run_data["x"][:, 2], linewidth=2.2, linestyle="--", label=f"{name} | theta [rad]")
        axes[1].plot(run_data["t"], run_data["x"][:, 1], linewidth=2.2, label=f"{name} | x_dot [m/s]")
        axes[1].plot(run_data["t"], run_data["x"][:, 3], linewidth=2.2, linestyle="--", label=f"{name} | theta_dot [rad/s]")
        axes[2].plot(run_data["t"], run_data["u"], linewidth=2.2, label=name)

    axes[0].axhline(0.0, color="0.7", linewidth=1.2, linestyle="--")
    axes[1].axhline(0.0, color="0.7", linewidth=1.2, linestyle="--")
    axes[2].axhline(0.0, color="0.7", linewidth=1.2, linestyle="--")
    axes[0].set_ylabel("Position / angle")
    axes[1].set_ylabel("Velocity / rate")
    axes[2].set_ylabel("Force [N]")
    axes[2].set_xlabel("Time [s]")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[2].grid(True, alpha=0.3)
    axes[0].set_title("Position and angle")
    axes[1].set_title("Velocity and angular rate")
    axes[2].set_title("Control force")
    axes[0].legend(loc="upper right", ncol=2)
    axes[1].legend(loc="upper right", ncol=2)
    axes[2].legend(loc="upper right")

    fig.suptitle(
        "Linearized Inverted Pendulum | MPC vs RL controllers\n"
        f"x0 = [{x0[0]:.3f}, {x0[1]:.3f}, {x0[2]:.3f}, {x0[3]:.3f}] | "
        f"Qx diag = {INVERTED_PENDULUM_RL_TRAINING.q_state_diag} | "
        f"ru = {INVERTED_PENDULUM_RL_TRAINING.ru:g}, qu = {INVERTED_PENDULUM_RL_TRAINING.qu:g}"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_path = figures_dir / "compare_linearized_inverted_pendulum_controllers.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    run()
