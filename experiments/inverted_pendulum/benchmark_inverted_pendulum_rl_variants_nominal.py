from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from control_bench.config import (
    INVERTED_PENDULUM_LINEARIZED,
    INVERTED_PENDULUM_MPC_EXPERIMENT,
    INVERTED_PENDULUM_NONLINEAR,
    INVERTED_PENDULUM_NONLINEAR_MPC_EXPERIMENT,
)
from control_bench.controllers.inverted_pendulum_mpc import InvertedPendulumMPCConfig, InvertedPendulumMPCController
from control_bench.controllers.inverted_pendulum_rl import InvertedPendulumRLController
from control_bench.plants.inverted_pendulum import build_linearized_inverted_pendulum
from control_bench.plants.inverted_pendulum_nonlinear import build_nonlinear_inverted_pendulum
from inverted_pendulum_rl_variants import PLANT_ID, VARIANT_METRICS_DIR, resolve_variants


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "Run the pendulum RL variant training script first."
        )


def _zero_reference(reference_cfg: tuple[float, float, float, float]) -> np.ndarray:
    return np.asarray(reference_cfg, dtype=float).reshape(4)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nominal screening for pendulum controllers.")
    parser.add_argument(
        "--plant",
        choices=("linearized", "nonlinear"),
        default="nonlinear",
        help="Plant to evaluate on. Default is the actual nonlinear pendulum.",
    )
    return parser.parse_args()


def _experiment_cfg(plant_kind: str):
    if plant_kind == "linearized":
        return INVERTED_PENDULUM_LINEARIZED, INVERTED_PENDULUM_MPC_EXPERIMENT
    if plant_kind == "nonlinear":
        return INVERTED_PENDULUM_NONLINEAR, INVERTED_PENDULUM_NONLINEAR_MPC_EXPERIMENT
    raise ValueError(f"Unknown plant kind: {plant_kind}")


def _build_eval_plant(plant_kind: str):
    params, _ = _experiment_cfg(plant_kind)
    if plant_kind == "linearized":
        return build_linearized_inverted_pendulum(params)
    return build_nonlinear_inverted_pendulum(params)


def _build_nominal_controller_model():
    # Both MPC and RL policies are deployed using the upright linearized model.
    return build_linearized_inverted_pendulum(INVERTED_PENDULUM_LINEARIZED)


def _load_rl_controller(variant) -> InvertedPendulumRLController:
    nominal_model = _build_nominal_controller_model()
    _require_file(variant.canonical_weight_path)
    if variant.is_pid:
        return InvertedPendulumRLController.load_npz(
            str(variant.canonical_weight_path),
            kind=variant.kind,
            dt=nominal_model.dt,
            u_bounds=nominal_model.u_bounds,
            u_min=INVERTED_PENDULUM_LINEARIZED.u_min,
            u_max=INVERTED_PENDULUM_LINEARIZED.u_max,
            name=variant.display_name,
        )
    return InvertedPendulumRLController.load_npz(
        str(variant.canonical_weight_path),
        kind=variant.kind,
        dt=nominal_model.dt,
        u_bounds=nominal_model.u_bounds,
        u_min=INVERTED_PENDULUM_LINEARIZED.u_min,
        u_max=INVERTED_PENDULUM_LINEARIZED.u_max,
        mlp_hidden=variant.hidden_layers,
        mlp_activation=variant.activation,
        name=variant.display_name,
    )


def _load_mpc_controller(plant_kind: str) -> InvertedPendulumMPCController:
    nominal_model = _build_nominal_controller_model()
    _, exp_cfg = _experiment_cfg(plant_kind)
    mpc_horizon = int(exp_cfg.N)
    use_terminal_cost = False
    if plant_kind == "nonlinear":
        # Empirically, the nonlinear local stabilization region is reliable for
        # the exact same stage cost once an exact terminal penalty is added and
        # the horizon stays in the short/moderate regime.
        mpc_horizon = 20
        use_terminal_cost = True
    return InvertedPendulumMPCController(
        InvertedPendulumMPCConfig(
            N=mpc_horizon,
            q_state_diag=exp_cfg.q_state_diag,
            ru=exp_cfg.ru,
            qu=exp_cfg.qu,
            use_terminal_cost=use_terminal_cost,
            pgd_iters=exp_cfg.pgd_iters,
            pgd_step_size=exp_cfg.pgd_step_size,
        ),
        dt=nominal_model.dt,
        A=nominal_model.A,
        B=nominal_model.B,
        u_bounds=nominal_model.u_bounds,
        name="MPC",
    )


def _simulate_nominal(
    *,
    controller,
    variant_id: str,
    display_name: str,
    kind: str,
    hidden_layers: tuple[int, ...],
    plant_kind: str,
) -> dict:
    plant = _build_eval_plant(plant_kind)
    _, exp_cfg = _experiment_cfg(plant_kind)
    dt = float(plant.dt)
    t_final = float(exp_cfg.t_final)
    n_steps = int(np.floor(t_final / dt)) + 1
    t = np.arange(n_steps, dtype=float) * dt
    x0 = np.asarray(exp_cfg.x0, dtype=float)
    r = _zero_reference(exp_cfg.reference)
    angle_limit = float(exp_cfg.angle_limit_rad)
    cart_limit = float(exp_cfg.cart_limit_m)

    x = np.zeros((n_steps, 4), dtype=float)
    u = np.zeros((n_steps,), dtype=float)
    plant.reset(x0=x0)
    controller.reset()

    pass_flag = True
    failure_reason = ""
    failure_time = np.nan
    steps_completed = 0

    for k, tk in enumerate(t):
        obs = plant.observe()
        if not np.all(np.isfinite(obs)):
            pass_flag = False
            failure_reason = "nonfinite_observation"
            failure_time = float(tk)
            break

        u_cmd = controller.step(r=r, obs=obs, t=tk)
        if not np.all(np.isfinite(u_cmd)):
            pass_flag = False
            failure_reason = "nonfinite_control"
            failure_time = float(tk)
            break

        x[k] = plant.state
        u[k] = float(u_cmd[0])
        steps_completed = k + 1

        theta = float(x[k, 2])
        cart_x = float(x[k, 0])
        if not np.isfinite(theta) or not np.isfinite(cart_x):
            pass_flag = False
            failure_reason = "nonfinite_state"
            failure_time = float(tk)
            break
        if abs(theta) > angle_limit:
            pass_flag = False
            failure_reason = "angle_limit"
            failure_time = float(tk)
            break
        if np.isfinite(cart_limit) and abs(cart_x) > cart_limit:
            pass_flag = False
            failure_reason = "cart_limit"
            failure_time = float(tk)
            break

        if k < n_steps - 1:
            plant.step(u_cmd)
            if not np.all(np.isfinite(plant.state)):
                pass_flag = False
                failure_reason = "nonfinite_next_state"
                failure_time = float(tk + dt)
                steps_completed = k + 1
                break

    used_x = x[: max(steps_completed, 1)]
    used_u = u[: max(steps_completed, 1)]
    final_state = used_x[min(len(used_x) - 1, max(0, steps_completed - 1))]
    return {
        "plant_kind": plant_kind,
        "variant_id": variant_id,
        "display_name": display_name,
        "kind": kind,
        "hidden_layers": hidden_layers,
        "pass_nominal": bool(pass_flag and steps_completed == n_steps),
        "failure_reason": failure_reason if failure_reason else "--",
        "failure_time_s": float(failure_time),
        "steps_completed": int(steps_completed),
        "max_abs_theta_deg": float(np.rad2deg(np.max(np.abs(used_x[:, 2])))),
        "max_abs_x_m": float(np.max(np.abs(used_x[:, 0]))),
        "max_abs_u_n": float(np.max(np.abs(used_u))),
        "final_state_norm": float(np.linalg.norm(final_state)),
    }


def _write_summary_files(rows: list[dict], plant_kind: str) -> tuple[Path, Path]:
    nominal_dir = VARIANT_METRICS_DIR / "nominal_screen"
    nominal_dir.mkdir(parents=True, exist_ok=True)
    csv_path = nominal_dir / f"nominal_summary__{PLANT_ID}__{plant_kind}.csv"
    md_path = nominal_dir / f"nominal_summary__{PLANT_ID}__{plant_kind}.md"

    fieldnames = [
        "plant_kind",
        "variant_id",
        "display_name",
        "kind",
        "hidden_layers",
        "pass_nominal",
        "failure_reason",
        "failure_time_s",
        "steps_completed",
        "max_abs_theta_deg",
        "max_abs_x_m",
        "max_abs_u_n",
        "final_state_norm",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})

    lines = [
        f"# Inverted Pendulum Nominal Screening ({plant_kind})",
        "",
        f"Training plant id: `{PLANT_ID}`",
        "",
        "| Variant | Pass? | Failure | Max |theta| [deg] | Max |x| [m] | Max |u| [N] | Final state norm |",
        "| --- | :---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['display_name']} | {'yes' if row['pass_nominal'] else 'no'} | {row['failure_reason']} | "
            f"{row['max_abs_theta_deg']:.3f} | {row['max_abs_x_m']:.3f} | {row['max_abs_u_n']:.3f} | "
            f"{row['final_state_norm']:.6f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def main() -> None:
    args = _parse_args()
    plant_kind = args.plant
    variants = resolve_variants()
    rows = [
        _simulate_nominal(
            controller=_load_mpc_controller(plant_kind),
            variant_id="mpc",
            display_name="MPC",
            kind="mpc",
            hidden_layers=(),
            plant_kind=plant_kind,
        )
    ]
    rows.extend(
        _simulate_nominal(
            controller=_load_rl_controller(variant),
            variant_id=variant.variant_id,
            display_name=variant.display_name,
            kind=variant.kind,
            hidden_layers=variant.hidden_layers,
            plant_kind=plant_kind,
        )
        for variant in variants
    )
    rows.sort(key=lambda row: (not row["pass_nominal"], row["final_state_norm"], row["variant_id"]))
    csv_path, _ = _write_summary_files(rows, plant_kind)

    print(f"Nominal screening written to: {csv_path}")
    print("")
    for row in rows:
        status = "PASS" if row["pass_nominal"] else "FAIL"
        print(
            f"{status:4s} | {row['display_name']:18s} | "
            f"reason={row['failure_reason']:>18s} | "
            f"max|theta|={row['max_abs_theta_deg']:.3f} deg | "
            f"final_norm={row['final_state_norm']:.6f}"
        )


if __name__ == "__main__":
    main()
