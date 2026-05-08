from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from control_bench.config import (
    INVERTED_PENDULUM_LINEARIZED,
    INVERTED_PENDULUM_NONLINEAR,
    INVERTED_PENDULUM_NONLINEAR_MPC_EXPERIMENT,
)
from control_bench.controllers.inverted_pendulum_mpc import InvertedPendulumMPCConfig, InvertedPendulumMPCController
from control_bench.plants.inverted_pendulum import build_linearized_inverted_pendulum
from control_bench.plants.inverted_pendulum_nonlinear import build_nonlinear_inverted_pendulum


LOCAL_FORCE_LIMIT_N = 10.0
LOCAL_ANGLE_LIMIT_DEG = 20.0
LOCAL_X0_GRID_M = (-0.05, 0.0, 0.05)
LOCAL_THETA0_GRID_DEG = (-8.0, -5.0, -3.0, 3.0, 5.0, 8.0)
LOCAL_SETTLE_THETA_DEG = 2.0
LOCAL_SETTLE_X_M = 0.10
LOCAL_SETTLE_RATE_NORM = 0.50
HORIZON_SWEEP = (20, 30, 40, 60)
TERMINAL_COST_MODES = (False, True)


def _bounded_params():
    return (
        replace(INVERTED_PENDULUM_LINEARIZED, u_min=-LOCAL_FORCE_LIMIT_N, u_max=LOCAL_FORCE_LIMIT_N),
        replace(INVERTED_PENDULUM_NONLINEAR, u_min=-LOCAL_FORCE_LIMIT_N, u_max=LOCAL_FORCE_LIMIT_N),
    )


def _reference() -> np.ndarray:
    return np.asarray(INVERTED_PENDULUM_NONLINEAR_MPC_EXPERIMENT.reference, dtype=float).reshape(4)


def _case_grid() -> list[np.ndarray]:
    cases: list[np.ndarray] = []
    for x0 in LOCAL_X0_GRID_M:
        for theta0_deg in LOCAL_THETA0_GRID_DEG:
            cases.append(np.array([x0, 0.0, np.deg2rad(theta0_deg), 0.0], dtype=float))
    return cases


def _simulate_case(N: int, x0: np.ndarray, *, use_terminal_cost: bool) -> dict:
    lin_params, nonlin_params = _bounded_params()
    nominal = build_linearized_inverted_pendulum(lin_params)
    plant = build_nonlinear_inverted_pendulum(nonlin_params)
    exp = INVERTED_PENDULUM_NONLINEAR_MPC_EXPERIMENT
    controller = InvertedPendulumMPCController(
        InvertedPendulumMPCConfig(
            N=N,
            q_state_diag=exp.q_state_diag,
            ru=exp.ru,
            qu=exp.qu,
            use_terminal_cost=use_terminal_cost,
            pgd_iters=exp.pgd_iters,
            pgd_step_size=exp.pgd_step_size,
        ),
        dt=nominal.dt,
        A=nominal.A,
        B=nominal.B,
        u_bounds=nominal.u_bounds,
        name=f"MPC N={N}",
    )

    dt = float(plant.dt)
    n_steps = int(np.floor(float(exp.t_final) / dt)) + 1
    angle_limit_rad = np.deg2rad(LOCAL_ANGLE_LIMIT_DEG)
    plant.reset(x0=x0)
    controller.reset()

    x_hist = np.zeros((n_steps, 4), dtype=float)
    u_hist = np.zeros((n_steps,), dtype=float)
    failure_reason = "--"
    failure_time_s = np.nan
    steps_completed = 0

    for k in range(n_steps):
        tk = k * dt
        obs = plant.observe()
        if not np.all(np.isfinite(obs)):
            failure_reason = "nonfinite_observation"
            failure_time_s = float(tk)
            break

        u_cmd = controller.step(r=_reference(), obs=obs, t=tk)
        if not np.all(np.isfinite(u_cmd)):
            failure_reason = "nonfinite_control"
            failure_time_s = float(tk)
            break

        x_hist[k] = plant.state
        u_hist[k] = float(u_cmd[0])
        steps_completed = k + 1

        if abs(float(x_hist[k, 2])) > angle_limit_rad:
            failure_reason = "angle_limit"
            failure_time_s = float(tk)
            break

        if k < n_steps - 1:
            plant.step(u_cmd)
            if not np.all(np.isfinite(plant.state)):
                failure_reason = "nonfinite_next_state"
                failure_time_s = float(tk + dt)
                break

    used_x = x_hist[: max(steps_completed, 1)]
    used_u = u_hist[: max(steps_completed, 1)]
    final_state = used_x[min(len(used_x) - 1, max(0, steps_completed - 1))]
    settled = (
        abs(float(final_state[2])) <= np.deg2rad(LOCAL_SETTLE_THETA_DEG)
        and abs(float(final_state[0])) <= LOCAL_SETTLE_X_M
        and float(np.linalg.norm(final_state[[1, 3]])) <= LOCAL_SETTLE_RATE_NORM
    )
    survived = failure_reason == "--" and steps_completed == n_steps
    passed = survived and settled

    return {
        "N": int(N),
        "use_terminal_cost": bool(use_terminal_cost),
        "x0": float(x0[0]),
        "theta0_deg": float(np.rad2deg(x0[2])),
        "pass_case": bool(passed),
        "survived_horizon": bool(survived),
        "settled": bool(settled),
        "failure_reason": failure_reason,
        "failure_time_s": float(failure_time_s),
        "steps_completed": int(steps_completed),
        "max_abs_theta_deg": float(np.rad2deg(np.max(np.abs(used_x[:, 2])))),
        "max_abs_x_m": float(np.max(np.abs(used_x[:, 0]))),
        "max_abs_u_n": float(np.max(np.abs(used_u))),
        "final_x_m": float(final_state[0]),
        "final_theta_deg": float(np.rad2deg(final_state[2])),
        "final_rate_norm": float(np.linalg.norm(final_state[[1, 3]])),
    }


def _write_outputs(case_rows: list[dict], summary_rows: list[dict]) -> tuple[Path, Path]:
    out_dir = EXPERIMENTS_ROOT / "results" / "metrics" / "inverted_pendulum_mpc_local"
    out_dir.mkdir(parents=True, exist_ok=True)
    case_csv = out_dir / "nonlinear_local_case_results.csv"
    summary_csv = out_dir / "nonlinear_local_horizon_summary.csv"
    summary_md = out_dir / "nonlinear_local_horizon_summary.md"

    case_fieldnames = [
        "use_terminal_cost",
        "N",
        "x0",
        "theta0_deg",
        "pass_case",
        "survived_horizon",
        "settled",
        "failure_reason",
        "failure_time_s",
        "steps_completed",
        "max_abs_theta_deg",
        "max_abs_x_m",
        "max_abs_u_n",
        "final_x_m",
        "final_theta_deg",
        "final_rate_norm",
    ]
    with case_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=case_fieldnames)
        writer.writeheader()
        for row in case_rows:
            writer.writerow({name: row.get(name, "") for name in case_fieldnames})

    summary_fieldnames = [
        "use_terminal_cost",
        "N",
        "cases_total",
        "cases_passed",
        "cases_survived",
        "pass_fraction",
        "survival_fraction",
        "max_final_abs_x_m",
        "max_final_abs_theta_deg",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=summary_fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({name: row.get(name, "") for name in summary_fieldnames})

    lines = [
        "# Nonlinear Local MPC Horizon Sweep",
        "",
        f"Force bounds: `[-{LOCAL_FORCE_LIMIT_N:.1f}, +{LOCAL_FORCE_LIMIT_N:.1f}] N`",
        f"Angle survival limit: `{LOCAL_ANGLE_LIMIT_DEG:.1f} deg`",
        f"Settlement rule: `|theta(T)| <= {LOCAL_SETTLE_THETA_DEG:.1f} deg`, `|x(T)| <= {LOCAL_SETTLE_X_M:.2f} m`, "
        f"`||(x_dot, theta_dot)|| <= {LOCAL_SETTLE_RATE_NORM:.2f}`",
        "",
        "| Terminal cost | N | Passed / total | Survived / total | Pass frac | Survival frac | Max final |x| [m] | Max final |theta| [deg] |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {'on' if row['use_terminal_cost'] else 'off'} | {row['N']} | "
            f"{row['cases_passed']}/{row['cases_total']} | {row['cases_survived']}/{row['cases_total']} | "
            f"{row['pass_fraction']:.3f} | {row['survival_fraction']:.3f} | {row['max_final_abs_x_m']:.3f} | "
            f"{row['max_final_abs_theta_deg']:.3f} |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_csv, summary_md


def main() -> None:
    cases = _case_grid()
    case_rows: list[dict] = []
    summary_rows: list[dict] = []

    for use_terminal_cost in TERMINAL_COST_MODES:
        for N in HORIZON_SWEEP:
            per_horizon = [_simulate_case(N, x0, use_terminal_cost=use_terminal_cost) for x0 in cases]
            case_rows.extend(per_horizon)
            passed = sum(1 for row in per_horizon if row["pass_case"])
            survived = sum(1 for row in per_horizon if row["survived_horizon"])
            summary_rows.append(
                {
                    "use_terminal_cost": bool(use_terminal_cost),
                    "N": int(N),
                    "cases_total": len(per_horizon),
                    "cases_passed": passed,
                    "cases_survived": survived,
                    "pass_fraction": float(passed / len(per_horizon)),
                    "survival_fraction": float(survived / len(per_horizon)),
                    "max_final_abs_x_m": float(max(abs(row["final_x_m"]) for row in per_horizon)),
                    "max_final_abs_theta_deg": float(max(abs(row["final_theta_deg"]) for row in per_horizon)),
                }
            )

    summary_rows.sort(key=lambda row: (row["use_terminal_cost"], row["N"]))
    summary_csv, _ = _write_outputs(case_rows, summary_rows)
    print(f"Saved nonlinear local MPC benchmark to: {summary_csv}")
    for row in summary_rows:
        print(
            f"terminal={'on' if row['use_terminal_cost'] else 'off'} | N={row['N']:2d} | "
            f"pass={row['cases_passed']}/{row['cases_total']} | "
            f"survive={row['cases_survived']}/{row['cases_total']} | "
            f"max_final_|x|={row['max_final_abs_x_m']:.3f} m"
        )


if __name__ == "__main__":
    main()
