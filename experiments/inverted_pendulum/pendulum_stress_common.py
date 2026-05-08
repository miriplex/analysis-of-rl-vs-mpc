from __future__ import annotations

import csv
import zlib
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
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
    INVERTED_PENDULUM_NONLINEAR,
    INVERTED_PENDULUM_NONLINEAR_MPC_EXPERIMENT,
)
from control_bench.controllers.inverted_pendulum_mpc import (
    InvertedPendulumMPCConfig,
    InvertedPendulumMPCController,
)
from control_bench.controllers.inverted_pendulum_rl import InvertedPendulumRLController
from control_bench.plants.inverted_pendulum import build_linearized_inverted_pendulum
from control_bench.plants.inverted_pendulum_nonlinear import build_nonlinear_inverted_pendulum
from control_bench.plants.second_order_family import expm_small
from inverted_pendulum_rl_variants import PLANT_ID, VARIANT_METRICS_DIR, resolve_variants


SEED = 20260424
LOCAL_X0_GRID_M = (-0.05, 0.0, 0.05)
LOCAL_THETA0_GRID_DEG = (-8.0, -5.0, -3.0, 3.0, 5.0, 8.0)
LOCAL_SETTLE_THETA_DEG = 2.0
LOCAL_SETTLE_X_M = 0.10
LOCAL_SETTLE_RATE_NORM = 0.50
ANGLE_LIMIT_DEG = 20.0
T_FINAL = float(INVERTED_PENDULUM_NONLINEAR_MPC_EXPERIMENT.t_final)
REFERENCE = np.asarray(INVERTED_PENDULUM_NONLINEAR_MPC_EXPERIMENT.reference, dtype=float).reshape(4)
PASSING_NOMINAL_SCREEN_CSV = (
    VARIANT_METRICS_DIR / "nominal_screen" / f"nominal_summary__{PLANT_ID}__nonlinear.csv"
)


@dataclass(frozen=True)
class PendulumControllerSpec:
    variant_id: str
    display_name: str
    kind: str
    hidden_layers: tuple[int, ...] = ()
    activation: str = "tanh"


@dataclass
class HiddenPoleActuatorChain:
    A_d: np.ndarray
    B_d: np.ndarray
    state: np.ndarray

    def reset(self) -> None:
        self.state[:] = 0.0

    def step(self, u_cmd: float) -> float:
        self.state = (self.A_d @ self.state) + (self.B_d[:, 0] * float(u_cmd))
        return float(self.state[-1])


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def local_case_grid() -> list[tuple[str, np.ndarray]]:
    cases: list[tuple[str, np.ndarray]] = []
    for x0 in LOCAL_X0_GRID_M:
        for theta0_deg in LOCAL_THETA0_GRID_DEG:
            label = f"x{x0:+.2f}__theta{theta0_deg:+.1f}"
            cases.append((label, np.array([x0, 0.0, np.deg2rad(theta0_deg), 0.0], dtype=float)))
    return cases


def _nominal_controller_model():
    return build_linearized_inverted_pendulum(INVERTED_PENDULUM_LINEARIZED)


def _load_rl_controller(spec: PendulumControllerSpec) -> InvertedPendulumRLController:
    nominal_model = _nominal_controller_model()
    variant_lookup = {variant.variant_id: variant for variant in resolve_variants()}
    variant = variant_lookup[spec.variant_id]
    require_file(variant.canonical_weight_path)
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


def _load_mpc_controller() -> InvertedPendulumMPCController:
    nominal_model = _nominal_controller_model()
    exp_cfg = INVERTED_PENDULUM_NONLINEAR_MPC_EXPERIMENT
    return InvertedPendulumMPCController(
        InvertedPendulumMPCConfig(
            N=20,
            q_state_diag=exp_cfg.q_state_diag,
            ru=exp_cfg.ru,
            qu=exp_cfg.qu,
            use_terminal_cost=True,
            pgd_iters=exp_cfg.pgd_iters,
            pgd_step_size=exp_cfg.pgd_step_size,
        ),
        dt=nominal_model.dt,
        A=nominal_model.A,
        B=nominal_model.B,
        u_bounds=nominal_model.u_bounds,
        name="MPC",
    )


def load_controller(spec: PendulumControllerSpec):
    if spec.kind == "mpc":
        return _load_mpc_controller()
    return _load_rl_controller(spec)


def passing_controller_specs() -> tuple[PendulumControllerSpec, ...]:
    require_file(PASSING_NOMINAL_SCREEN_CSV)
    variant_lookup = {variant.variant_id: variant for variant in resolve_variants()}
    specs: list[PendulumControllerSpec] = []
    with PASSING_NOMINAL_SCREEN_CSV.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if str(row.get("pass_nominal", "")).strip().lower() != "true":
                continue
            variant_id = str(row["variant_id"])
            if variant_id == "mpc":
                specs.append(PendulumControllerSpec(variant_id="mpc", display_name="MPC", kind="mpc"))
                continue
            variant = variant_lookup[variant_id]
            specs.append(
                PendulumControllerSpec(
                    variant_id=variant.variant_id,
                    display_name=variant.display_name,
                    kind=variant.kind,
                    hidden_layers=variant.hidden_layers,
                    activation=variant.activation,
                )
            )
    if not specs:
        raise RuntimeError(f"No passing controllers found in nominal screen: {PASSING_NOMINAL_SCREEN_CSV}")
    return tuple(specs)


def _discretize_hidden_chain_exact(hidden_pole: float, hidden_order: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    if hidden_pole <= 0.0:
        raise ValueError("hidden_pole must be > 0")
    if hidden_order < 1:
        raise ValueError("hidden_order must be >= 1")
    A_c = np.zeros((hidden_order, hidden_order), dtype=float)
    B_c = np.zeros((hidden_order, 1), dtype=float)
    A_c[0, 0] = -float(hidden_pole)
    B_c[0, 0] = float(hidden_pole)
    for idx in range(1, hidden_order):
        A_c[idx, idx] = -float(hidden_pole)
        A_c[idx, idx - 1] = float(hidden_pole)

    aug = np.zeros((hidden_order + 1, hidden_order + 1), dtype=float)
    aug[:hidden_order, :hidden_order] = A_c
    aug[:hidden_order, hidden_order:] = B_c
    exp_aug = np.asarray(expm_small(aug * float(dt)), dtype=float)
    return exp_aug[:hidden_order, :hidden_order], exp_aug[:hidden_order, hidden_order:]


def build_hidden_pole_chain(*, hidden_pole: float, hidden_order: int, dt: float) -> HiddenPoleActuatorChain:
    A_d, B_d = _discretize_hidden_chain_exact(hidden_pole, hidden_order, dt)
    return HiddenPoleActuatorChain(A_d=A_d, B_d=B_d, state=np.zeros((hidden_order,), dtype=float))


def measurement_noise_trace(
    *,
    case_label: str,
    n_steps: int,
    stress_scale: float,
    base_std: np.ndarray,
) -> np.ndarray:
    base_std = np.asarray(base_std, dtype=float).reshape(4)
    seed = (int(SEED) + int(zlib.crc32(f"meas::{case_label}::{stress_scale:g}".encode("utf-8")))) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.normal(loc=0.0, scale=1.0, size=(n_steps, 4)) * (float(stress_scale) * base_std.reshape(1, 4))


def simulate_case(
    *,
    controller,
    x0: np.ndarray,
    measurement_noise: Optional[np.ndarray] = None,
    hidden_chain: Optional[HiddenPoleActuatorChain] = None,
) -> dict:
    plant = build_nonlinear_inverted_pendulum(INVERTED_PENDULUM_NONLINEAR)
    dt = float(plant.dt)
    n_steps = int(np.floor(T_FINAL / dt)) + 1
    t = np.arange(n_steps, dtype=float) * dt
    angle_limit_rad = np.deg2rad(ANGLE_LIMIT_DEG)

    plant.reset(x0=x0)
    controller.reset()
    if hidden_chain is not None:
        hidden_chain.reset()

    x_hist = np.zeros((n_steps, 4), dtype=float)
    u_cmd_hist = np.zeros((n_steps,), dtype=float)
    u_applied_hist = np.zeros((n_steps,), dtype=float)
    failure_reason = "--"
    failure_time_s = np.nan
    steps_completed = 0

    for k, tk in enumerate(t):
        obs = plant.observe()
        if not np.all(np.isfinite(obs)):
            failure_reason = "nonfinite_observation"
            failure_time_s = float(tk)
            break
        obs_ctrl = obs if measurement_noise is None else obs + measurement_noise[k]
        if not np.all(np.isfinite(obs_ctrl)):
            failure_reason = "nonfinite_noisy_observation"
            failure_time_s = float(tk)
            break

        u_cmd = controller.step(r=REFERENCE, obs=obs_ctrl, t=float(tk))
        if not np.all(np.isfinite(u_cmd)):
            failure_reason = "nonfinite_control"
            failure_time_s = float(tk)
            break

        u_applied = float(u_cmd[0]) if hidden_chain is None else hidden_chain.step(float(u_cmd[0]))
        if not np.isfinite(u_applied):
            failure_reason = "nonfinite_applied_control"
            failure_time_s = float(tk)
            break

        x_hist[k] = plant.state
        u_cmd_hist[k] = float(u_cmd[0])
        u_applied_hist[k] = float(u_applied)
        steps_completed = k + 1

        theta = float(x_hist[k, 2])
        if abs(theta) > angle_limit_rad:
            failure_reason = "angle_limit"
            failure_time_s = float(tk)
            break

        if k < n_steps - 1:
            plant.step(np.array([u_applied], dtype=float))
            if not np.all(np.isfinite(plant.state)):
                failure_reason = "nonfinite_next_state"
                failure_time_s = float(tk + dt)
                break

    used_x = x_hist[: max(steps_completed, 1)]
    used_u_cmd = u_cmd_hist[: max(steps_completed, 1)]
    used_u_applied = u_applied_hist[: max(steps_completed, 1)]
    final_state = used_x[min(len(used_x) - 1, max(0, steps_completed - 1))]

    survive_horizon = failure_reason == "--" and steps_completed == n_steps
    settle_failures: list[str] = []
    if abs(float(final_state[2])) > np.deg2rad(LOCAL_SETTLE_THETA_DEG):
        settle_failures.append("final_theta_not_settled")
    if abs(float(final_state[0])) > LOCAL_SETTLE_X_M:
        settle_failures.append("final_x_not_settled")
    if float(np.linalg.norm(final_state[[1, 3]])) > LOCAL_SETTLE_RATE_NORM:
        settle_failures.append("final_rate_not_settled")

    pass_case = bool(survive_horizon and not settle_failures)
    if survive_horizon and settle_failures:
        failure_reason = ",".join(settle_failures)

    return {
        "pass_case": bool(pass_case),
        "survived_horizon": bool(survive_horizon),
        "failure": bool(not pass_case),
        "failure_reason": failure_reason,
        "failure_time_s": float(failure_time_s),
        "steps_completed": int(steps_completed),
        "max_abs_theta_deg": float(np.rad2deg(np.max(np.abs(used_x[:, 2])))),
        "max_abs_x_m": float(np.max(np.abs(used_x[:, 0]))),
        "max_abs_u_cmd_n": float(np.max(np.abs(used_u_cmd))),
        "max_abs_u_applied_n": float(np.max(np.abs(used_u_applied))),
        "final_x_m": float(final_state[0]),
        "final_theta_deg": float(np.rad2deg(final_state[2])),
        "final_rate_norm": float(np.linalg.norm(final_state[[1, 3]])),
        "final_state_norm": float(np.linalg.norm(final_state)),
    }


def summarize_stage_rows(rows: Sequence[dict], *, group_keys: Sequence[str], progression_key: str) -> List[dict]:
    grouped: Dict[Tuple[object, ...], List[dict]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in group_keys), []).append(row)

    summary_rows: List[dict] = []
    for key in sorted(grouped):
        group = grouped[key]
        summary = {group_key: key[idx] for idx, group_key in enumerate(group_keys)}
        summary["num_cases"] = int(len(group))
        summary["failure_count"] = int(sum(bool(row["failure"]) for row in group))
        summary["pass_count"] = int(sum(bool(row["pass_case"]) for row in group))
        summary["pass_fraction"] = float(summary["pass_count"] / summary["num_cases"]) if summary["num_cases"] else 0.0
        summary["pass_all_cases"] = bool(summary["failure_count"] == 0)
        summary["failed_cases"] = ",".join(sorted(str(row["case_label"]) for row in group if bool(row["failure"])))
        summary["failure_reasons"] = ",".join(
            sorted(
                {
                    reason
                    for row in group
                    if bool(row["failure"])
                    for reason in str(row.get("failure_reason", "")).split(",")
                    if reason
                }
            )
        )
        summary["mean_final_state_norm"] = float(np.mean([float(row["final_state_norm"]) for row in group]))
        summary["worst_final_state_norm"] = float(np.max([float(row["final_state_norm"]) for row in group]))
        summary["worst_max_abs_theta_deg"] = float(np.max([float(row["max_abs_theta_deg"]) for row in group]))
        summary["worst_max_abs_x_m"] = float(np.max([float(row["max_abs_x_m"]) for row in group]))
        summary["worst_final_abs_x_m"] = float(np.max([abs(float(row["final_x_m"])) for row in group]))
        summary["worst_final_abs_theta_deg"] = float(np.max([abs(float(row["final_theta_deg"])) for row in group]))
        summary_rows.append(summary)
    summary_rows.sort(
        key=lambda row: (
            int(not bool(row["pass_all_cases"])),
            -float(row["pass_fraction"]),
            float(row["worst_final_state_norm"]),
            str(row["controller"]),
            float(row.get(progression_key, 0.0)),
        )
    )
    return summary_rows


def controller_survival_summary(
    nominal_rows: Sequence[dict],
    stress_rows: Sequence[dict],
    *,
    progression_key: str,
    max_pass_key: str,
) -> List[dict]:
    nominal_by_controller = {str(row["controller"]): row for row in nominal_rows}
    grouped: Dict[str, List[dict]] = {}
    for row in stress_rows:
        grouped.setdefault(str(row["controller"]), []).append(row)

    summary_rows: List[dict] = []
    for controller in sorted(nominal_by_controller):
        nominal = nominal_by_controller[controller]
        controller_rows = sorted(grouped.get(controller, []), key=lambda row: float(row[progression_key]))
        pass_values = [float(row[progression_key]) for row in controller_rows if bool(row["pass_all_cases"])]
        fail_rows = [row for row in controller_rows if not bool(row["pass_all_cases"])]
        first_fail_row = fail_rows[0] if fail_rows else None
        summary_rows.append(
            {
                "controller": controller,
                "eligible_after_nominal": bool(nominal["pass_all_cases"]),
                "nominal_failure_count": int(nominal["failure_count"]),
                "nominal_failed_cases": str(nominal["failed_cases"]),
                max_pass_key: float(max(pass_values)) if pass_values else 0.0,
                f"first_failure_{progression_key}": (
                    float(first_fail_row[progression_key]) if first_fail_row is not None else ""
                ),
                "failed_cases_at_first_failure": (
                    str(first_fail_row["failed_cases"]) if first_fail_row is not None else ""
                ),
                "failure_reasons_at_first_failure": (
                    str(first_fail_row["failure_reasons"]) if first_fail_row is not None else ""
                ),
                "worst_final_state_norm_at_first_failure": (
                    float(first_fail_row["worst_final_state_norm"]) if first_fail_row is not None else 0.0
                ),
                f"last_tested_{progression_key}": (
                    float(controller_rows[-1][progression_key]) if controller_rows else 0.0
                ),
                "num_levels_tested": int(len(controller_rows)),
            }
        )

    summary_rows.sort(
        key=lambda row: (
            int(not bool(row["eligible_after_nominal"])),
            -float(row[max_pass_key]),
            float(row[f"first_failure_{progression_key}"])
            if str(row[f"first_failure_{progression_key}"]).strip()
            else 10**9,
            float(row["worst_final_state_norm_at_first_failure"]),
            str(row["controller"]),
        )
    )
    return summary_rows


def plot_pass_fraction_heatmap(
    *,
    nominal_rows: Sequence[dict],
    stress_rows: Sequence[dict],
    tested_levels: Sequence[float | int],
    progression_key: str,
    nominal_label: str,
    level_label_fn: Callable[[float | int], str],
    xlabel: str,
    title: str,
    save_path: Path,
) -> None:
    controller_order = [str(row["controller"]) for row in nominal_rows]
    values = np.zeros((len(controller_order), len(tested_levels) + 1), dtype=float)
    annotations = [["" for _ in range(len(tested_levels) + 1)] for _ in controller_order]

    nominal_by_controller = {str(row["controller"]): row for row in nominal_rows}
    stress_lookup = {(str(row["controller"]), float(row[progression_key])): row for row in stress_rows}

    for row_idx, controller in enumerate(controller_order):
        nominal = nominal_by_controller[controller]
        values[row_idx, 0] = float(nominal["pass_fraction"])
        annotations[row_idx][0] = f"{int(nominal['pass_count'])}/{int(nominal['num_cases'])}"
        eliminated = False
        for col_idx, level in enumerate(tested_levels, start=1):
            row = stress_lookup.get((controller, float(level)))
            if row is None:
                if eliminated:
                    values[row_idx, col_idx] = 0.0
                    annotations[row_idx][col_idx] = "elim"
                else:
                    values[row_idx, col_idx] = np.nan
                    annotations[row_idx][col_idx] = "-"
            else:
                values[row_idx, col_idx] = float(row["pass_fraction"])
                annotations[row_idx][col_idx] = f"{int(row['pass_count'])}/{int(row['num_cases'])}"
                if not bool(row["pass_all_cases"]):
                    eliminated = True

    fig_h = max(4.8, 0.6 * len(controller_order) + 2.0)
    fig_w = max(8.0, 1.0 * (len(tested_levels) + 1) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    masked = np.ma.masked_invalid(values)
    im = ax.imshow(masked, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fraction of local nonlinear cases passed")

    ax.set_xticks(np.arange(len(tested_levels) + 1))
    ax.set_xticklabels([nominal_label] + [level_label_fn(level) for level in tested_levels], rotation=0)
    ax.set_yticks(np.arange(len(controller_order)))
    ax.set_yticklabels(controller_order)
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            text = annotations[row_idx][col_idx]
            if not text:
                continue
            display_value = values[row_idx, col_idx]
            color = "black" if not np.isfinite(display_value) or display_value >= 0.55 else "white"
            ax.text(col_idx, row_idx, text, ha="center", va="center", fontsize=8, color=color)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def _write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _write_summary_markdown(
    *,
    path: Path,
    title: str,
    summary_lines: Sequence[str],
    nominal_rows: Sequence[dict],
    survival_rows: Sequence[dict],
    progression_key: str,
    max_pass_key: str,
) -> None:
    lines = [f"# {title}", ""]
    lines.extend(summary_lines)
    lines.append("")
    lines.append("## Nominal Local Screen")
    lines.append("")
    lines.append("| Controller | Passed / total | Pass all? | Failed cases | Worst final state norm |")
    lines.append("| --- | ---: | :---: | --- | ---: |")
    for row in nominal_rows:
        lines.append(
            f"| {row['controller']} | {row['pass_count']}/{row['num_cases']} | "
            f"{'yes' if row['pass_all_cases'] else 'no'} | {row['failed_cases'] or '--'} | "
            f"{row['worst_final_state_norm']:.6f} |"
        )
    lines.append("")
    lines.append("## Controller Survival Summary")
    lines.append("")
    lines.append(
        f"| Controller | Max fully passed {progression_key} | First failure {progression_key} | "
        "Failure reasons at first failure | Failed cases at first failure |"
    )
    lines.append("| --- | ---: | ---: | --- | --- |")
    for row in survival_rows:
        lines.append(
            f"| {row['controller']} | {row[max_pass_key]} | {row[f'first_failure_{progression_key}'] or '--'} | "
            f"{row['failure_reasons_at_first_failure'] or '--'} | {row['failed_cases_at_first_failure'] or '--'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_stress_test(
    *,
    title: str,
    figures_dir: Path,
    metrics_dir: Path,
    summary_lines: Sequence[str],
    progression_key: str,
    tested_levels: Sequence[float | int],
    nominal_label: str,
    level_label_fn: Callable[[float | int], str],
    xlabel: str,
    heatmap_title: str,
    simulate_level: Callable[[object, str, np.ndarray, float | int], dict],
) -> None:
    specs = passing_controller_specs()
    cases = local_case_grid()
    all_rows: list[dict] = []
    nominal_case_rows: list[dict] = []
    stress_case_rows: list[dict] = []
    nominal_summary_rows: list[dict] = []
    stress_summary_rows: list[dict] = []

    for spec in specs:
        controller = load_controller(spec)
        controller_nominal_rows: list[dict] = []
        for case_label, x0 in cases:
            result = simulate_level(controller, case_label, x0, 0)
            result.update(
                {
                    "controller": spec.display_name,
                    "variant_id": spec.variant_id,
                    "case_label": case_label,
                    progression_key: 0.0,
                    "stage": "nominal",
                }
            )
            nominal_case_rows.append(result)
            controller_nominal_rows.append(result)
        nominal_summary = summarize_stage_rows(
            controller_nominal_rows,
            group_keys=("controller",),
            progression_key=progression_key,
        )[0]
        nominal_summary_rows.append(nominal_summary)

    nominal_summary_rows.sort(
        key=lambda row: (
            int(not bool(row["pass_all_cases"])),
            -float(row["pass_fraction"]),
            float(row["worst_final_state_norm"]),
            str(row["controller"]),
        )
    )

    for spec in specs:
        controller = load_controller(spec)
        nominal_summary = next(row for row in nominal_summary_rows if row["controller"] == spec.display_name)
        if not bool(nominal_summary["pass_all_cases"]):
            continue
        eliminated = False
        for level in tested_levels:
            if eliminated:
                break
            per_level_rows: list[dict] = []
            for case_label, x0 in cases:
                result = simulate_level(controller, case_label, x0, level)
                result.update(
                    {
                        "controller": spec.display_name,
                        "variant_id": spec.variant_id,
                        "case_label": case_label,
                        progression_key: float(level),
                        "stage": "stress",
                    }
                )
                stress_case_rows.append(result)
                per_level_rows.append(result)
            level_summary = summarize_stage_rows(
                per_level_rows,
                group_keys=("controller", progression_key),
                progression_key=progression_key,
            )[0]
            stress_summary_rows.append(level_summary)
            if not bool(level_summary["pass_all_cases"]):
                eliminated = True

    survival_rows = controller_survival_summary(
        nominal_summary_rows,
        stress_summary_rows,
        progression_key=progression_key,
        max_pass_key=f"max_pass_all_{progression_key}",
    )

    all_rows.extend(nominal_case_rows)
    all_rows.extend(stress_case_rows)

    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(
        metrics_dir / "case_results.csv",
        all_rows,
        fieldnames=(
            "stage",
            "controller",
            "variant_id",
            "case_label",
            progression_key,
            "pass_case",
            "survived_horizon",
            "failure",
            "failure_reason",
            "failure_time_s",
            "steps_completed",
            "max_abs_theta_deg",
            "max_abs_x_m",
            "max_abs_u_cmd_n",
            "max_abs_u_applied_n",
            "final_x_m",
            "final_theta_deg",
            "final_rate_norm",
            "final_state_norm",
        ),
    )
    _write_csv(
        metrics_dir / "nominal_screening.csv",
        nominal_summary_rows,
        fieldnames=(
            "controller",
            "num_cases",
            "failure_count",
            "pass_count",
            "pass_fraction",
            "pass_all_cases",
            "failed_cases",
            "failure_reasons",
            "mean_final_state_norm",
            "worst_final_state_norm",
            "worst_max_abs_theta_deg",
            "worst_max_abs_x_m",
            "worst_final_abs_x_m",
            "worst_final_abs_theta_deg",
        ),
    )
    _write_csv(
        metrics_dir / "progression_summary.csv",
        stress_summary_rows,
        fieldnames=(
            "controller",
            progression_key,
            "num_cases",
            "failure_count",
            "pass_count",
            "pass_fraction",
            "pass_all_cases",
            "failed_cases",
            "failure_reasons",
            "mean_final_state_norm",
            "worst_final_state_norm",
            "worst_max_abs_theta_deg",
            "worst_max_abs_x_m",
            "worst_final_abs_x_m",
            "worst_final_abs_theta_deg",
        ),
    )
    _write_csv(
        metrics_dir / "controller_survival_summary.csv",
        survival_rows,
        fieldnames=tuple(survival_rows[0].keys()) if survival_rows else ("controller",),
    )

    plot_pass_fraction_heatmap(
        nominal_rows=nominal_summary_rows,
        stress_rows=stress_summary_rows,
        tested_levels=tested_levels,
        progression_key=progression_key,
        nominal_label=nominal_label,
        level_label_fn=level_label_fn,
        xlabel=xlabel,
        title=heatmap_title,
        save_path=figures_dir / "pass_fraction_heatmap.png",
    )
    _write_summary_markdown(
        path=metrics_dir / "summary.md",
        title=title,
        summary_lines=summary_lines,
        nominal_rows=nominal_summary_rows,
        survival_rows=survival_rows,
        progression_key=progression_key,
        max_pass_key=f"max_pass_all_{progression_key}",
    )
