from __future__ import annotations

import csv
from pathlib import Path
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import zlib

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
from control_bench.controllers.mpc import MPCConfig, LinearMPCController
from control_bench.controllers.rl_numpy.controller import BackpropRLController
from control_bench.experiment_manifest import (
    first_order_hidden_pole_stress_test,
    first_order_measurement_noise_stress_test,
    first_order_mlp_variants,
)
from control_bench.core.scenario_impl import BasicScenario
from control_bench.core.sim import simulate_closed_loop
from control_bench.core.types import SimConfig, Trajectory
from control_bench.plants.first_order_family import build_first_order_grid
from control_bench.plants.lti_state_space import DiscreteLTIPlant
from control_bench.plants.second_order_family import expm_small


T_FINAL = 10.0
SEED = 0
R_STEP = -2.0
USE_BOUNDS = False
TRADEOFF_PLANT_ID = "unstable__rhp_zero"

SIGMA_Y = 0.02
SIGMA_DU = 0.02
DU_STEP_TIME = 5.0
DU_STEP_MAG = 0.10
HIDDEN_POLE = 10.0
WEIGHTS_DIR = EXPERIMENTS_ROOT / "results" / "rl_numpy" / "per_plant"
ROBUSTNESS_TESTS_DIR = EXPERIMENTS_ROOT / "results" / "metrics" / "robustness_tests"
MEASUREMENT_NOISE_STRESS_CFG = first_order_measurement_noise_stress_test()
HIDDEN_POLE_STRESS_CFG = first_order_hidden_pole_stress_test()
ROBUSTNESS_MPC_HORIZONS = (20, 35, 50)

TAIL_FRACTION = 0.25
GROWTH_FRACTION = 0.10
FAIL_ERROR_ABS = max(10.0, 5.0 * abs(R_STEP))
FAIL_U_ABS = 50.0
FAIL_TAIL_RMS_FLOOR = 0.20
FAIL_TAIL_GROWTH_RATIO = 1.20
BENCHMARK_WARMUP_REPEATS = 3
BENCHMARK_MEASURE_REPEATS = 12
MPC_SWEEP_WARMUP_REPEATS = 2
MPC_SWEEP_MEASURE_REPEATS = 12
MPC_SWEEP_REPLAY_LOOPS = 8
TRADEOFF_RUNTIME_WARMUP_REPEATS = 2
TRADEOFF_RUNTIME_MEASURE_REPEATS = 12
TRADEOFF_RUNTIME_TARGET_STEPS = 100_000

RANK_METRICS = (
    "mean_cost",
    "peak_error",
    "tail_rms_error",
    "tail_growth_ratio",
    "max_abs_u",
    "failure",
)

METRIC_EXPLANATIONS = (
    ("rank", "average within-plant/scenario rank across cost, peak, tail RMS, tail growth, max |u|, and failure flag; lower is better"),
    ("cost", "mean rollout stage cost using the same qy/ru/qu objective as training and MPC"),
    ("peak", "maximum absolute tracking error over the rollout"),
    ("tail_rms", "RMS tracking error over the last 25% of the rollout; catches bad end behavior"),
    ("tail_g", "RMS(last 10%) / RMS(previous 10%); above 1 means the error is growing again near the end"),
    ("fails", "number of runs flagged as failed by the threshold rule"),
    ("p95_ms", "95th percentile controller step time in milliseconds, measured from controller.step(...) only"),
    ("util_%", "runtime utilization = 100 * p95_step_time / dt"),
)


def step_reference(level: float = 1.0):
    def _r(t: float) -> np.ndarray:
        return np.array([level], dtype=float)

    return _r


def require_file(path: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            f"Run: python {PROJECT_ROOT / 'experiments' / 'first_order' / 'train_first_order_manifest_controllers.py'}"
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


def build_controllers(plant_id: str, *, nominal, use_bounds: bool = False) -> list:
    u_bounds = nominal.u_bounds if use_bounds else None
    u_min = family.u_min if use_bounds else None
    u_max = family.u_max if use_bounds else None

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
        name="MPC",
    )

    rl_simple_path = str(WEIGHTS_DIR / f"rl_pidfeat__{plant_id}.npz")
    require_file(rl_simple_path)

    rl_simple = BackpropRLController.load_npz(
        rl_simple_path,
        kind="pidfeat",
        dt=family.dt,
        u_bounds=u_bounds,
        u_min=u_min,
        u_max=u_max,
        name="RL (PID-features)",
    )

    controllers = [mpc, rl_simple]
    for variant in first_order_mlp_variants():
        weight_path = str(WEIGHTS_DIR / f"{variant['canonical_filename_prefix']}__{plant_id}.npz")
        require_file(weight_path)
        controllers.append(
            BackpropRLController.load_npz(
                weight_path,
                kind="rich",
                dt=family.dt,
                u_bounds=u_bounds,
                u_min=u_min,
                u_max=u_max,
                mlp_hidden=tuple(int(v) for v in variant["hidden_layers"]),
                mlp_activation=str(variant["activation"]),
                rich_feature_set=str(variant["feature_set"]),
                name=str(variant["label"]),
            )
        )
    return controllers


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(values * values)))


def compute_metrics(traj: Trajectory) -> dict:
    e = np.asarray(traj.r[:, 0] - traj.y[:, 0], dtype=float)
    u = np.asarray(traj.u[:, 0], dtype=float)

    du = np.empty_like(u)
    du[0] = u[0]
    du[1:] = u[1:] - u[:-1]

    stage_cost = (
        float(FIRST_ORDER_COST_WEIGHTS.qy) * (e * e)
        + float(FIRST_ORDER_COST_WEIGHTS.ru) * (du * du)
        + float(FIRST_ORDER_COST_WEIGHTS.qu) * (u * u)
    )

    tail_steps = max(1, int(np.ceil(TAIL_FRACTION * e.size)))
    tail_errors = e[-tail_steps:]

    growth_steps = max(1, int(np.ceil(GROWTH_FRACTION * e.size)))
    last_growth = e[-growth_steps:]
    prev_growth = e[-2 * growth_steps : -growth_steps] if e.size >= 2 * growth_steps else e[:-growth_steps]
    prev_growth_rms = _rms(prev_growth) if prev_growth.size > 0 else 0.0
    last_growth_rms = _rms(last_growth)
    if prev_growth_rms <= 1e-12:
        tail_growth_ratio = float(np.inf) if last_growth_rms > 1e-12 else 1.0
    else:
        tail_growth_ratio = float(last_growth_rms / prev_growth_rms)

    is_finite = bool(np.all(np.isfinite(traj.y)) and np.all(np.isfinite(traj.u)))
    peak_error = float(np.max(np.abs(e))) if e.size else 0.0
    tail_rms_error = _rms(tail_errors)
    max_abs_u = float(np.max(np.abs(u))) if u.size else 0.0
    late_escalation = bool(tail_rms_error > FAIL_TAIL_RMS_FLOOR and tail_growth_ratio > FAIL_TAIL_GROWTH_RATIO)
    failure = bool((not is_finite) or (peak_error > FAIL_ERROR_ABS) or (max_abs_u > FAIL_U_ABS) or late_escalation)

    failure_reasons: List[str] = []
    failure_details: List[str] = []
    if not is_finite:
        failure_reasons.append("non_finite")
        failure_details.append("non_finite values detected in rollout")
    if peak_error > FAIL_ERROR_ABS:
        failure_reasons.append("peak_error")
        failure_details.append(f"peak_error={peak_error:.4g}>{FAIL_ERROR_ABS:g}")
    if max_abs_u > FAIL_U_ABS:
        failure_reasons.append("max_abs_u")
        failure_details.append(f"max_abs_u={max_abs_u:.4g}>{FAIL_U_ABS:g}")
    if late_escalation:
        failure_reasons.append("late_escalation")
        failure_details.append(
            f"tail_rms={tail_rms_error:.4g}>{FAIL_TAIL_RMS_FLOOR:g} and tail_g={tail_growth_ratio:.4g}>{FAIL_TAIL_GROWTH_RATIO:g}"
        )

    return {
        "mean_cost": float(np.mean(stage_cost)),
        "peak_error": peak_error,
        "tail_rms_error": tail_rms_error,
        "tail_growth_ratio": tail_growth_ratio,
        "max_abs_u": max_abs_u,
        "late_escalation": late_escalation,
        "failure": failure,
        "failure_reason": ",".join(failure_reasons),
        "failure_detail": "; ".join(failure_details),
    }


def benchmark_controller_replay(
    *,
    controller,
    traj: Trajectory,
    warmup_repeats: int,
    measure_repeats: int,
    replay_loops_per_repeat: int = 1,
) -> dict:
    t = np.asarray(traj.t, dtype=float)
    r = np.asarray(traj.r, dtype=float)
    obs = np.asarray(traj.obs, dtype=float)
    dt = float(traj.info.get("dt", 0.0))

    if int(replay_loops_per_repeat) <= 0:
        raise ValueError("replay_loops_per_repeat must be > 0")

    elapsed_ns: List[int] = []
    steps_per_repeat = int(t.size) * int(replay_loops_per_repeat)

    total_repeats = int(warmup_repeats) + int(measure_repeats)
    for repeat_idx in range(total_repeats):
        start_ns = time.perf_counter_ns()
        for _ in range(int(replay_loops_per_repeat)):
            controller.reset()
            for k in range(t.size):
                controller.step(r=r[k], obs=obs[k], t=float(t[k]))
        elapsed = time.perf_counter_ns() - start_ns
        if repeat_idx >= warmup_repeats:
            elapsed_ns.append(int(elapsed))

    samples_ns = np.asarray(elapsed_ns, dtype=np.float64)
    samples_ms = (samples_ns / float(steps_per_repeat)) / 1e6
    p95_ms = float(np.percentile(samples_ms, 95))
    utilization_pct = float(100.0 * p95_ms / (dt * 1000.0)) if dt > 0 else float("inf")

    return {
        "mean_step_ms": float(np.mean(samples_ms)),
        "median_step_ms": float(np.median(samples_ms)),
        "p95_step_ms": p95_ms,
        "max_step_ms": float(np.max(samples_ms)),
        "utilization_pct": utilization_pct,
        "benchmark_samples": int(samples_ms.size),
    }


def replay_loops_for_target_steps(
    *,
    traj: Trajectory,
    measure_repeats: int,
    target_steps: int,
) -> int:
    steps_per_rollout = int(np.asarray(traj.t, dtype=float).size)
    denom = max(1, steps_per_rollout * int(measure_repeats))
    return max(1, int(np.ceil(float(target_steps) / float(denom))))


def _assign_group_ranks(rows: list[dict]) -> None:
    grouped: Dict[Tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["scenario"], row["plant_id"]), []).append(row)

    for grouped_rows in grouped.values():
        for metric in RANK_METRICS:
            values = np.array(
                [
                    float(row[metric]) if metric != "failure" else float(bool(row[metric]))
                    for row in grouped_rows
                ],
                dtype=float,
            )
            order = np.argsort(values, kind="stable")
            sorted_values = values[order]
            ranks = np.empty_like(values)

            pos = 0
            while pos < sorted_values.size:
                end = pos + 1
                while end < sorted_values.size and np.isclose(sorted_values[end], sorted_values[pos], atol=1e-12, rtol=1e-12):
                    end += 1
                avg_rank = 0.5 * ((pos + 1) + end)
                ranks[order[pos:end]] = avg_rank
                pos = end

            for idx, row in enumerate(grouped_rows):
                row[f"rank_{metric}"] = float(ranks[idx])

        row_rank_keys = [f"rank_{metric}" for metric in RANK_METRICS]
        for row in grouped_rows:
            row["composite_rank"] = float(np.mean([row[key] for key in row_rank_keys]))


def _aggregate_rows(
    rows: Sequence[dict],
    *,
    group_keys: Sequence[str],
    aggregate_metrics: Optional[Sequence[str]] = None,
) -> list[dict]:
    grouped: Dict[Tuple[object, ...], list[dict]] = {}
    for row in rows:
        key = tuple(row[group_key] for group_key in group_keys)
        grouped.setdefault(key, []).append(row)

    metrics_to_aggregate = tuple(
        aggregate_metrics
        if aggregate_metrics is not None
        else (
            "mean_cost",
            "peak_error",
            "tail_rms_error",
            "tail_growth_ratio",
            "max_abs_u",
            "composite_rank",
            "mean_step_ms",
            "median_step_ms",
            "p95_step_ms",
            "max_step_ms",
            "utilization_pct",
        )
    )

    summary_rows = []
    for key in sorted(grouped):
        group_rows = grouped[key]
        summary = {group_key: key[idx] for idx, group_key in enumerate(group_keys)}
        for metric in metrics_to_aggregate:
            values = np.array([float(row[metric]) for row in group_rows], dtype=float)
            summary[f"mean_{metric}"] = float(np.mean(values))
            summary[f"median_{metric}"] = float(np.median(values))
        failure_rows = [row for row in group_rows if bool(row["failure"])]
        late_rows = [row for row in group_rows if bool(row["late_escalation"])]
        summary["failure_count"] = int(len(failure_rows))
        summary["late_escalation_count"] = int(len(late_rows))
        summary["failed_plants"] = ",".join(sorted({str(row["plant_id"]) for row in failure_rows}))
        summary["failure_reasons"] = ",".join(
            sorted(
                {
                    reason
                    for row in failure_rows
                    for reason in str(row.get("failure_reason", "")).split(",")
                    if reason
                }
            )
        )
        failure_labels = []
        for row in failure_rows:
            if "scenario" in row:
                failure_labels.append(f"{row['scenario']}:{row['plant_id']}")
            elif "N" in row:
                failure_labels.append(f"N={row['N']}:{row['plant_id']}")
            else:
                failure_labels.append(str(row.get("plant_id", "unknown")))
        summary["failure_cases"] = "; ".join(sorted(set(failure_labels)))
        detail_labels = []
        for row in failure_rows:
            detail = str(row.get("failure_detail", "")).strip()
            if detail:
                label = row.get("plant_id", row.get("N", "unknown"))
                detail_labels.append(f"{label} ({detail})")
        summary["failure_details"] = "; ".join(sorted(set(detail_labels)))
        summary["late_escalation_plants"] = ",".join(sorted({str(row["plant_id"]) for row in late_rows}))
        summary["num_runs"] = int(len(group_rows))
        summary_rows.append(summary)
    return summary_rows


def _failure_breakdown_rows(rows: Sequence[dict]) -> list[dict]:
    grouped: Dict[Tuple[str, str], list[dict]] = {}
    for row in rows:
        if not bool(row["failure"]):
            continue
        grouped.setdefault((str(row["scenario"]), str(row["controller"])), []).append(row)

    breakdown = []
    for scenario, controller in sorted(grouped):
        group_rows = grouped[(scenario, controller)]
        failure_rows = [row for row in group_rows if bool(row["failure"])]
        late_rows = [row for row in group_rows if bool(row["late_escalation"])]
        breakdown.append(
            {
                "scenario": scenario,
                "controller": controller,
                "failure_count": int(len(failure_rows)),
                "failed_plants": ",".join(sorted({str(row["plant_id"]) for row in failure_rows})),
                "failure_reasons": ",".join(
                    sorted(
                        {
                            reason
                            for row in failure_rows
                            for reason in str(row.get("failure_reason", "")).split(",")
                            if reason
                        }
                    )
                ),
                "failure_details": "; ".join(
                    sorted(
                        {
                            f"{row['plant_id']} ({row['failure_detail']})"
                            for row in failure_rows
                            if str(row.get("failure_detail", "")).strip()
                        }
                    )
                ),
                "late_escalation_plants": ",".join(sorted({str(row["plant_id"]) for row in late_rows})),
            }
        )
    return breakdown


def _format_float(value: float) -> str:
    value = float(value)
    if not np.isfinite(value):
        return "inf"
    if value == 0.0:
        return "0"
    abs_value = abs(value)
    if abs_value >= 1e4 or abs_value < 1e-3:
        return f"{value:.3e}"
    return f"{value:.4f}"


def _write_csv(path: Path, rows: Sequence[dict], *, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _markdown_table(rows: Sequence[dict], *, columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, bool):
                cells.append("yes" if value else "no")
            elif isinstance(value, (int, np.integer)):
                cells.append(str(int(value)))
            elif isinstance(value, (float, np.floating)):
                cells.append(_format_float(float(value)))
            else:
                cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, rule, *body])


def _text_table(rows: Sequence[dict], *, columns: Sequence[Tuple[str, str]]) -> str:
    rendered_rows: List[List[str]] = []
    headers = [header for header, _ in columns]

    for row in rows:
        rendered = []
        for _, key in columns:
            value = row[key]
            if isinstance(value, bool):
                rendered.append("yes" if value else "no")
            elif isinstance(value, (int, np.integer)):
                rendered.append(str(int(value)))
            elif isinstance(value, (float, np.floating)):
                rendered.append(_format_float(float(value)))
            else:
                rendered.append(str(value))
        rendered_rows.append(rendered)

    widths = []
    for col_idx, header in enumerate(headers):
        width = len(header)
        for row in rendered_rows:
            width = max(width, len(row[col_idx]))
        widths.append(width)

    def _fmt_line(values: Sequence[str]) -> str:
        return "  ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    header_line = _fmt_line(headers)
    rule_line = "  ".join("-" * width for width in widths)
    body_lines = [_fmt_line(row) for row in rendered_rows]
    return "\n".join([header_line, rule_line, *body_lines])


def _write_summary_markdown(
    path: Path,
    *,
    overall_rows: Sequence[dict],
    scenario_rows: Sequence[dict],
    failure_rows: Sequence[dict],
) -> None:
    content = [
        "# First-order robustness metrics",
        "",
        "Scenarios:",
        "- `nominal`: nominal plant, no disturbance",
        f"- `disturbed`: measurement noise sigma={SIGMA_Y:g}, input disturbance sigma={SIGMA_DU:g}, step {DU_STEP_MAG:g} at {DU_STEP_TIME:g}s",
        f"- `hidden_pole`: actual plant has extra hidden pole `-( {HIDDEN_POLE:g} )`, controllers remain nominal",
        "",
        "Metrics (lower is better):",
        "- `mean_cost`: rollout mean of the same quadratic stage cost used in the controller objective",
        "- `peak_error`: max absolute tracking error",
        "- `tail_rms_error`: RMS tracking error over the last 25% of the rollout",
        "- `tail_growth_ratio`: RMS(last 10%) / RMS(previous 10%)",
        "- `max_abs_u`: max absolute control action",
        "- `composite_rank`: average rank across the five metrics plus failure flag within each plant/scenario",
        "",
        "Compact table columns:",
        *(f"- `{name}`: {description}" for name, description in METRIC_EXPLANATIONS),
        "",
        "Runtime measurement notes:",
        f"- controller timing uses replayed `controller.step(...)` calls only",
        f"- warmup repeats = {BENCHMARK_WARMUP_REPEATS}, measured repeats = {BENCHMARK_MEASURE_REPEATS}",
        "",
        "Failure rule:",
        f"- failure if non-finite, `peak_error > {FAIL_ERROR_ABS:g}`, `max_abs_u > {FAIL_U_ABS:g}`, or late escalation (`tail_rms_error > {FAIL_TAIL_RMS_FLOOR:g}` and `tail_growth_ratio > {FAIL_TAIL_GROWTH_RATIO:g}`)",
        "",
        "## Overall average across all plants and scenarios",
        _markdown_table(
            overall_rows,
            columns=(
                "controller",
                "mean_composite_rank",
                "mean_mean_cost",
                "mean_peak_error",
                "mean_tail_rms_error",
                "mean_tail_growth_ratio",
                "mean_p95_step_ms",
                "mean_utilization_pct",
                "mean_max_abs_u",
                "failure_count",
                "late_escalation_count",
                "num_runs",
            ),
        ),
        "",
        "Compact text view:",
        "```text",
        _text_table(
            overall_rows,
            columns=(
                ("controller", "controller"),
                ("rank", "mean_composite_rank"),
                ("cost", "mean_mean_cost"),
                ("peak", "mean_peak_error"),
                ("tail_rms", "mean_tail_rms_error"),
                ("tail_g", "mean_tail_growth_ratio"),
                ("p95_ms", "mean_p95_step_ms"),
                ("util_%", "mean_utilization_pct"),
                ("fails", "failure_count"),
            ),
        ),
        "```",
        "",
        "## Failure breakdown",
        (
            "_No failures detected._"
            if not failure_rows
            else _markdown_table(
                failure_rows,
                columns=(
                    "scenario",
                    "controller",
                    "failure_count",
                    "failed_plants",
                    "failure_reasons",
                    "failure_details",
                    "late_escalation_plants",
                ),
            )
        ),
        "",
        "## Scenario averages across plants",
        _markdown_table(
            scenario_rows,
            columns=(
                "scenario",
                "controller",
                "mean_composite_rank",
                "mean_mean_cost",
                "mean_peak_error",
                "mean_tail_rms_error",
                "mean_tail_growth_ratio",
                "mean_p95_step_ms",
                "mean_utilization_pct",
                "mean_max_abs_u",
                "failure_count",
                "late_escalation_count",
                "num_runs",
            ),
        ),
        "",
        "Compact text view:",
        "```text",
        _text_table(
            scenario_rows,
            columns=(
                ("scenario", "scenario"),
                ("controller", "controller"),
                ("rank", "mean_composite_rank"),
                ("cost", "mean_mean_cost"),
                ("peak", "mean_peak_error"),
                ("tail_rms", "mean_tail_rms_error"),
                ("tail_g", "mean_tail_growth_ratio"),
                ("p95_ms", "mean_p95_step_ms"),
                ("util_%", "mean_utilization_pct"),
                ("fails", "failure_count"),
            ),
        ),
        "```",
        "",
    ]
    path.write_text("\n".join(content))


def _load_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            "Run the relevant robustness-test scripts first so the survival summaries exist."
        )
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _stress_scale_values(schedule: Sequence[dict]) -> list[float]:
    values: list[float] = []
    seen: set[float] = set()
    for entry in schedule:
        start = float(entry["start"])
        stop = float(entry["stop"])
        step = float(entry["step"])
        current = start
        while current <= stop + 1e-12:
            rounded = round(float(current), 10)
            if rounded not in seen:
                values.append(rounded)
                seen.add(rounded)
            current += step
    values.sort()
    return values


def _build_mpc_controller_for_horizon(*, nominal, horizon_n: int, use_bounds: bool = False) -> LinearMPCController:
    u_bounds = nominal.u_bounds if use_bounds else None
    return LinearMPCController(
        MPCConfig(
            N=int(horizon_n),
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
        name=f"MPC (N={int(horizon_n)})",
    )


def _measurement_noise_survival_for_mpc_horizons(
    *,
    horizons: Sequence[int],
    use_bounds: bool = USE_BOUNDS,
) -> dict[int, float]:
    schedule = _stress_scale_values(MEASUREMENT_NOISE_STRESS_CFG["stress_scale_schedule"])
    base_sigma = float(MEASUREMENT_NOISE_STRESS_CFG["base_measurement_noise_std"])
    t_final = float(MEASUREMENT_NOISE_STRESS_CFG["t_final_seconds"])
    survived_until: dict[int, float] = {int(n): 0.0 for n in horizons}

    active_horizons = {int(n) for n in horizons}
    for stress_scale in schedule:
        if not active_horizons:
            break
        all_pass_at_scale: dict[int, bool] = {int(n): True for n in active_horizons}
        for plant_id in FIRST_ORDER_PLANT_IDS:
            nominal = build_first_order_grid(family)[plant_id]
            u_bounds = nominal.u_bounds if use_bounds else None
            nominal.u_bounds = u_bounds
            dt = float(nominal.dt)
            n_steps = n_steps_for(t_final=t_final, dt=dt)
            seed_pid = (int(SEED) + int(zlib.crc32(plant_id.encode("utf-8")))) % (2**32)
            rng = np.random.default_rng(seed_pid)
            noise_y = rng.normal(loc=0.0, scale=float(base_sigma) * float(stress_scale), size=(n_steps,))
            scenario = BasicScenario(
                io=nominal.io,
                reference_fn=step_reference(R_STEP),
                measurement_fn=make_measurement_fn_from_trace(noise_y=noise_y, dt=dt),
            )
            cfg = SimConfig(t_final=t_final, seed=SEED)
            for horizon_n in tuple(sorted(active_horizons)):
                controller = _build_mpc_controller_for_horizon(
                    nominal=nominal,
                    horizon_n=horizon_n,
                    use_bounds=use_bounds,
                )
                plant = build_first_order_grid(family)[plant_id]
                plant.u_bounds = u_bounds
                result = simulate_closed_loop(plant=plant, controller=controller, scenario=scenario, cfg=cfg)
                if bool(compute_metrics(result.traj)["failure"]):
                    all_pass_at_scale[horizon_n] = False

        for horizon_n in tuple(sorted(active_horizons)):
            if all_pass_at_scale[horizon_n]:
                survived_until[horizon_n] = float(stress_scale)
            else:
                active_horizons.remove(horizon_n)

    return survived_until


def _hidden_pole_survival_for_mpc_horizons(
    *,
    horizons: Sequence[int],
    use_bounds: bool = USE_BOUNDS,
) -> dict[int, int]:
    max_hidden_order = int(HIDDEN_POLE_STRESS_CFG["max_hidden_order"])
    hidden_pole = float(HIDDEN_POLE_STRESS_CFG["hidden_pole"])
    t_final = float(HIDDEN_POLE_STRESS_CFG["t_final_seconds"])
    survived_until: dict[int, int] = {int(n): 0 for n in horizons}

    active_horizons = {int(n) for n in horizons}
    for hidden_order in range(1, max_hidden_order + 1):
        if not active_horizons:
            break
        all_pass_at_order: dict[int, bool] = {int(n): True for n in active_horizons}
        for plant_id in FIRST_ORDER_PLANT_IDS:
            nominal = build_first_order_grid(family)[plant_id]
            u_bounds = nominal.u_bounds if use_bounds else None
            nominal.u_bounds = u_bounds
            scenario = BasicScenario(
                io=nominal.io,
                reference_fn=step_reference(R_STEP),
            )
            cfg = SimConfig(t_final=t_final, seed=SEED)
            mismatch_plant = build_secret_pole_mismatch_plant(
                plant_id,
                hidden_pole=hidden_pole,
                use_bounds=use_bounds,
            ) if hidden_order == 1 else None
            for horizon_n in tuple(sorted(active_horizons)):
                controller = _build_mpc_controller_for_horizon(
                    nominal=nominal,
                    horizon_n=horizon_n,
                    use_bounds=use_bounds,
                )
                if hidden_order == 1:
                    plant = mismatch_plant
                else:
                    plant = build_hidden_pole_chain_mismatch_plant_for_benchmark(
                        plant_id=plant_id,
                        hidden_pole=hidden_pole,
                        hidden_order=hidden_order,
                        use_bounds=use_bounds,
                    )
                result = simulate_closed_loop(plant=plant, controller=controller, scenario=scenario, cfg=cfg)
                if bool(compute_metrics(result.traj)["failure"]):
                    all_pass_at_order[horizon_n] = False

        for horizon_n in tuple(sorted(active_horizons)):
            if all_pass_at_order[horizon_n]:
                survived_until[horizon_n] = int(hidden_order)
            else:
                active_horizons.remove(horizon_n)

    return survived_until


def build_hidden_pole_chain_mismatch_plant_for_benchmark(
    *,
    plant_id: str,
    hidden_pole: float,
    hidden_order: int,
    use_bounds: bool = False,
) -> DiscreteLTIPlant:
    if hidden_pole <= 0:
        raise ValueError("hidden_pole must be > 0")
    if hidden_order < 1:
        raise ValueError("hidden_order must be >= 1")

    nominal = build_first_order_grid(family)[plant_id]
    p, z = plant_pz_from_id(plant_id)

    if z is None:
        c1 = float(family.k)
        d1 = 0.0
    else:
        c1 = float(family.k * (p - z))
        d1 = float(family.k)

    a = float(hidden_pole)
    n_states = int(hidden_order) + 1
    A_c = np.zeros((n_states, n_states), dtype=float)
    B_c = np.zeros((n_states, 1), dtype=float)

    A_c[0, 0] = float(p)
    B_c[0, 0] = 1.0
    A_c[1, 0] = a * c1
    A_c[1, 1] = -a
    B_c[1, 0] = a * d1
    for idx in range(2, n_states):
        A_c[idx, idx - 1] = a
        A_c[idx, idx] = -a

    A_d, B_d = _discretize_exact(A_c, B_c, dt=family.dt)
    C = np.zeros((1, n_states), dtype=float)
    C[0, -1] = 1.0
    D = np.zeros((1, 1), dtype=float)

    u_bounds = nominal.u_bounds if use_bounds else None
    return DiscreteLTIPlant(
        dt=family.dt,
        A=A_d,
        B=B_d,
        C=C,
        D=D,
        u_bounds=u_bounds,
    )


def _runtime_point_rows(
    *,
    mpc_sweep_rows: Sequence[dict],
    nominal_controller_rows: Sequence[dict],
) -> dict[str, dict]:
    runtime_rows: dict[str, dict] = {
        str(row["controller"]): dict(row)
        for row in nominal_controller_rows
    }

    mpc_n50_row = next((dict(row) for row in mpc_sweep_rows if int(row["N"]) == 50), None)
    if mpc_n50_row is None:
        stable_rows = [dict(row) for row in mpc_sweep_rows if int(row.get("failure_count", 0)) == 0]
        if not stable_rows:
            raise RuntimeError("No nominal-pass MPC horizon rows available for runtime-vs-robustness plot.")
        mpc_n50_row = max(stable_rows, key=lambda row: int(row["N"]))
    runtime_rows["MPC"] = {
        "controller": "MPC",
        "mean_step_ms": float(mpc_n50_row["mean_step_ms"]),
        "median_step_ms": float(mpc_n50_row["median_step_ms"]),
        "p95_step_ms": float(mpc_n50_row["p95_step_ms"]),
        "selected_horizon_n": int(mpc_n50_row["N"]),
    }
    return runtime_rows


def _plot_runtime_vs_robustness(
    *,
    mpc_sweep_rows: Sequence[dict],
    nominal_controller_rows: Sequence[dict],
    figures_dir: Path,
) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    runtime_rows = _runtime_point_rows(
        mpc_sweep_rows=mpc_sweep_rows,
        nominal_controller_rows=nominal_controller_rows,
    )
    mpc_measurement_survival = _measurement_noise_survival_for_mpc_horizons(horizons=ROBUSTNESS_MPC_HORIZONS)
    mpc_hidden_pole_survival = _hidden_pole_survival_for_mpc_horizons(horizons=ROBUSTNESS_MPC_HORIZONS)
    mlp_variant_by_label = {str(variant["label"]): dict(variant) for variant in first_order_mlp_variants()}

    pid_color = "#d97706"
    mpc_color = "#991b1b"
    feature_set_colors = {
        "rich11": "#2563eb",
        "compact6": "#0f766e",
    }

    def _controller_point_style(controller_name: str) -> tuple[str, str, str]:
        if controller_name.startswith("MPC"):
            return mpc_color, "D", "MPC horizons"
        if controller_name == "RL (PID-features)":
            return pid_color, "s", "RL (PID-features)"
        variant = mlp_variant_by_label.get(controller_name)
        if variant is None:
            return "#111827", "o", "Other learned controller"
        feature_set = str(variant["feature_set"])
        color = feature_set_colors.get(feature_set, "#111827")
        return color, "o", f"MLP ({feature_set})"

    def _short_controller_label(controller_name: str) -> str:
        if controller_name.startswith("MPC"):
            return controller_name.replace("MPC (", "MPC ").replace(")", "")
        if controller_name == "RL (PID-features)":
            return "PID-feat"
        variant = mlp_variant_by_label.get(controller_name)
        if variant is None:
            return controller_name
        arch = "_".join(str(v) for v in variant["hidden_layers"])
        feature_tag = "r11" if str(variant["feature_set"]) == "rich11" else "c6"
        return f"{arch} {feature_tag}"

    panel_specs = (
        {
            "title": "Measurement noise",
            "path": ROBUSTNESS_TESTS_DIR / "measurement_noise_stress" / "controller_survival_summary.csv",
            "y_key": "max_pass_all_scale",
            "y_label": "Max survived scale",
            "mpc_survival": mpc_measurement_survival,
        },
        {
            "title": "Hidden pole",
            "path": ROBUSTNESS_TESTS_DIR / "hidden_pole_stress" / "controller_survival_summary.csv",
            "y_key": "max_pass_all_order",
            "y_label": "Max survived order",
            "mpc_survival": mpc_hidden_pole_survival,
        },
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9), sharex=True)
    axes_flat = list(np.ravel(axes))
    saved_paths: list[Path] = []
    seen_labels: set[str] = set()
    learned_label_offsets = {
        0: {
            "RL (PID-features)": (8, 8),
            "16_16 (rich11)": (-56, -18),
            "32_32 (rich11)": (-26, -12),
            "32_32_32 (rich11)": (-12, 10),
        },
        1: {
            "RL (PID-features)": (8, -16),
            "16_16 (rich11)": (-56, 6),
            "32_32 (rich11)": (-44, -18),
            "32_32_32 (rich11)": (10, 10),
        },
    }

    for panel_idx, (ax, spec) in enumerate(zip(axes_flat, panel_specs)):
        survival_rows = _load_csv_rows(spec["path"])
        plotted_rows = []
        for row in survival_rows:
            controller_name = str(row["controller"])
            if controller_name == "MPC":
                continue
            if str(row.get("eligible_after_nominal", "")).lower() != "true":
                continue
            if controller_name not in runtime_rows:
                continue
            plotted_rows.append(
                {
                    "controller": controller_name,
                    "runtime_ms": float(runtime_rows[controller_name]["mean_step_ms"]),
                    "robustness_y": float(row[spec["y_key"]]),
                }
            )

        mpc_curve_rows = []
        for horizon_n in ROBUSTNESS_MPC_HORIZONS:
            focal_runtime_row = next((row for row in mpc_sweep_rows if int(row["N"]) == int(horizon_n)), None)
            if focal_runtime_row is None:
                continue
            mpc_curve_rows.append(
                {
                    "controller": f"MPC (N={int(horizon_n)})",
                    "runtime_ms": float(focal_runtime_row["mean_step_ms"]),
                    "robustness_y": float(spec["mpc_survival"][int(horizon_n)]),
                    "horizon_n": int(horizon_n),
                }
            )

        plotted_rows.sort(key=lambda row: row["runtime_ms"])
        mpc_curve_rows.sort(key=lambda row: row["runtime_ms"])

        if mpc_curve_rows:
            ax.plot(
                [row["runtime_ms"] for row in mpc_curve_rows],
                [row["robustness_y"] for row in mpc_curve_rows],
                color=mpc_color,
                linewidth=1.2,
                alpha=0.8,
                zorder=3,
            )

        mpc_label_done = False
        measurement_offsets = {
            20: (-36, 10),
            35: (-10, 10),
            50: (-6, 10),
        }
        hidden_offsets = {
            20: (-30, 10),
            35: (-10, 10),
            50: (-2, 10),
        }
        for row_idx, row in enumerate(mpc_curve_rows):
            controller_name = str(row["controller"])
            color, marker, legend_label = _controller_point_style(controller_name)
            scatter_label = legend_label if not mpc_label_done else "_nolegend_"
            mpc_label_done = True
            ax.scatter(
                [row["runtime_ms"]],
                [row["robustness_y"]],
                s=92,
                color=color,
                marker=marker,
                edgecolors="white",
                linewidths=0.9,
                label=scatter_label,
                zorder=5,
            )
            ax.annotate(
                f"N={int(row['horizon_n'])}",
                (row["runtime_ms"], row["robustness_y"]),
                textcoords="offset points",
                xytext=(
                    measurement_offsets.get(int(row["horizon_n"]), (-24, 10))
                    if panel_idx == 0
                    else hidden_offsets.get(int(row["horizon_n"]), (-24, 10))
                ),
                fontsize=8.0,
                color=mpc_color,
                zorder=6,
            )

        for row_idx, row in enumerate(plotted_rows):
            controller_name = str(row["controller"])
            color, marker, legend_label = _controller_point_style(controller_name)
            scatter_label = legend_label if legend_label not in seen_labels else "_nolegend_"
            seen_labels.add(legend_label)
            ax.scatter(
                [row["runtime_ms"]],
                [row["robustness_y"]],
                s=96,
                color=color,
                marker=marker,
                edgecolors="white",
                linewidths=0.9,
                label=scatter_label,
                zorder=5,
            )
            dx, dy = learned_label_offsets.get(panel_idx, {}).get(controller_name, (8, 8))
            ax.annotate(
                _short_controller_label(controller_name),
                (row["runtime_ms"], row["robustness_y"]),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=8.5,
                color="#111827",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "fc": "white",
                    "ec": color,
                    "alpha": 0.92,
                },
                zorder=6,
            )

        ax.set_title(spec["title"])
        ax.set_xscale("log")
        ax.set_ylabel(spec["y_label"])
        ax.grid(True, which="both", alpha=0.3)
        ax.margins(x=0.12, y=0.18)

    for ax in axes_flat:
        ax.set_xlabel("Mean controller step time [ms]")

    handles, labels = [], []
    for ax in axes_flat:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.93),
        frameon=True,
    )
    fig.subplots_adjust(top=0.80, bottom=0.16, left=0.08, right=0.98, wspace=0.22)

    save_path_png = figures_dir / "runtime_vs_robustness_tradeoffs.png"
    save_path_pdf = figures_dir / "runtime_vs_robustness_tradeoffs.pdf"
    fig.savefig(save_path_png, dpi=160)
    fig.savefig(save_path_pdf)
    saved_paths.extend([save_path_png, save_path_pdf])

    backend = plt.get_backend().lower()
    if "agg" not in backend:
        plt.show()
    else:
        plt.close(fig)

    return saved_paths


def _plot_nominal_tradeoffs(
    *,
    mpc_sweep_rows: Sequence[dict],
    nominal_controller_rows: Sequence[dict],
    figures_dir: Path,
    plant_id: str,
) -> list[Path]:
    mlp_variant_by_label = {str(variant["label"]): dict(variant) for variant in first_order_mlp_variants()}
    figures_dir.mkdir(parents=True, exist_ok=True)
    mpc_sweep_color = "#f59e9e"
    mpc_line_color = "#991b1b"
    pid_color = "#d97706"
    feature_set_colors = {
        "rich11": "#2563eb",
        "compact6": "#0f766e",
    }
    horizon_n_labels = {14, 15, 20, 30, 40, 50}

    def _pick(row: dict, preferred: str, fallback: str) -> float:
        if preferred in row:
            return float(row[preferred])
        return float(row[fallback])

    x_vals = np.array([_pick(row, "mean_mean_step_ms", "mean_step_ms") for row in mpc_sweep_rows], dtype=float)
    y_tail = np.array([_pick(row, "mean_tail_rms_error", "tail_rms_error") for row in mpc_sweep_rows], dtype=float)
    ns = np.array([row["N"] for row in mpc_sweep_rows], dtype=int)
    mpc_fail_mask = np.array(
        [int(row["failure_count"] if "failure_count" in row else row["failure"]) > 0 for row in mpc_sweep_rows],
        dtype=bool,
    )

    def _controller_point_style(controller_name: str) -> tuple[str, str, str]:
        if controller_name == "RL (PID-features)":
            return pid_color, "s", "RL (PID-features)"
        variant = mlp_variant_by_label.get(controller_name)
        if variant is None:
            return "#111827", "o", "Other learned controller"
        feature_set = str(variant["feature_set"])
        color = feature_set_colors.get(feature_set, "#111827")
        legend_label = f"MLP ({feature_set})"
        return color, "o", legend_label

    def _short_controller_label(controller_name: str) -> str:
        if controller_name == "RL (PID-features)":
            return "PID-feat"
        variant = mlp_variant_by_label.get(controller_name)
        if variant is None:
            return controller_name
        arch = "_".join(str(v) for v in variant["hidden_layers"])
        feature_tag = "r11" if str(variant["feature_set"]) == "rich11" else "c6"
        return f"{arch} {feature_tag}"

    def _plot_tradeoff_panel(
        ax: plt.Axes,
        y_vals: np.ndarray,
        *,
        label_ns: set[int],
    ) -> None:
        y_safe = np.maximum(y_vals, np.finfo(float).tiny)
        stable_mask = ~mpc_fail_mask

        order_by_n = np.argsort(ns)
        stable_order = order_by_n[stable_mask[order_by_n]]

        ax.plot(
            x_vals[stable_order],
            y_safe[stable_order],
            color=mpc_sweep_color,
            linewidth=1.0,
            marker="o",
            markersize=4.2,
            alpha=0.75,
            label="MPC horizons (nominal-pass N)",
            zorder=2,
        )

        horizon_label_offsets = {
            14: (-34, 16),
            15: (8, 8),
            5: (8, 16),
            10: (8, -16),
            20: (-34, -18),
            30: (8, -18),
            40: (8, -18),
            50: (8, 8),
        }
        for row_idx in stable_order:
            n = int(ns[row_idx])
            if n in label_ns:
                dx, dy = horizon_label_offsets.get(n, (6, 6))
                ax.annotate(
                    f"N={n}",
                    (x_vals[row_idx], y_safe[row_idx]),
                    textcoords="offset points",
                    xytext=(dx, dy),
                    fontsize=7.5,
                    color=mpc_line_color,
                )

        annotation_offsets = [(8, 8), (8, -14), (-52, 8), (-52, -14), (8, 20), (-52, 20)]
        seen_labels: set[str] = set()
        learned_rows = sorted(
            nominal_controller_rows,
            key=lambda row: (
                float(row["mean_p95_step_ms"] if "mean_p95_step_ms" in row else row["p95_step_ms"]),
                float(row["mean_tail_rms_error"] if "mean_tail_rms_error" in row else row["tail_rms_error"]),
            ),
        )
        for idx, row in enumerate(learned_rows):
            controller_name = str(row["controller"])
            color, marker, legend_label = _controller_point_style(controller_name)
            y_point = float(row["mean_tail_rms_error"] if "mean_tail_rms_error" in row else row["tail_rms_error"])
            y_point = max(y_point, np.finfo(float).tiny)
            x_point = float(row["mean_mean_step_ms"] if "mean_mean_step_ms" in row else row["mean_step_ms"])
            scatter_label = legend_label if legend_label not in seen_labels else "_nolegend_"
            seen_labels.add(legend_label)
            ax.scatter(
                [x_point],
                [y_point],
                s=88,
                color=color,
                marker=marker,
                edgecolors="white",
                linewidths=0.9,
                label=scatter_label,
                zorder=6,
            )
            dx, dy = annotation_offsets[idx % len(annotation_offsets)]
            ax.annotate(
                _short_controller_label(controller_name),
                (x_point, y_point),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=8.5,
                color="#111827",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "fc": "white",
                    "ec": color,
                    "alpha": 0.92,
                },
                zorder=7,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Mean controller step time [ms]")
        ax.set_ylabel("Mean nominal tail RMS error")
        ax.set_title("Runtime vs nominal RMS error")
        ax.grid(True, which="both", alpha=0.3)
        ax.margins(x=0.08, y=0.20)

    def _make_figure(*, subtitle: str) -> tuple[plt.Figure, Path]:
        fig, ax = plt.subplots(1, 1, figsize=(8.8, 6.0))
        _plot_tradeoff_panel(
            ax,
            y_tail,
            label_ns=horizon_n_labels,
        )
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper center",
            ncol=4,
            bbox_to_anchor=(0.5, 0.90),
            frameon=True,
        )
        fig.subplots_adjust(top=0.80, bottom=0.14, left=0.12, right=0.98)
        fig.suptitle(
            f"First-order nominal tradeoffs on {plant_id}\n"
            f"x-axis is mean controller step time; {subtitle}",
            y=0.965,
        )
        save_path = figures_dir / "runtime_vs_nominal_tradeoffs.png"
        fig.savefig(save_path, dpi=160)
        return fig, save_path

    stale_paths = [
        figures_dir / "runtime_vs_nominal_tradeoffs_full.png",
        figures_dir / "runtime_vs_nominal_tradeoffs_n15plus.png",
        figures_dir / "runtime_vs_nominal_cost.png",
        figures_dir / "runtime_vs_nominal_tail_rms.png",
    ]
    for stale_path in stale_paths:
        if stale_path.exists():
            stale_path.unlink()

    saved_paths: list[Path] = []
    fig, output_path = _make_figure(
        subtitle="Only nominal-pass MPC horizons and nominal-pass learned controllers are shown",
    )
    saved_paths.append(output_path)

    backend = plt.get_backend().lower()
    if "agg" not in backend:
        plt.show()
    else:
        plt.close(fig)

    return saved_paths


def run_all(*, use_bounds: bool = USE_BOUNDS) -> None:
    results_dir = EXPERIMENTS_ROOT / "results" / "metrics" / "first_order_robustness"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = EXPERIMENTS_ROOT / "results" / "figures" / "first_order_robustness"
    figures_dir.mkdir(parents=True, exist_ok=True)

    detailed_rows: List[dict] = []

    for plant_id in FIRST_ORDER_PLANT_IDS:
        nominal = build_first_order_grid(family)[plant_id]
        u_bounds = nominal.u_bounds if use_bounds else None
        nominal.u_bounds = u_bounds
        controllers = build_controllers(plant_id, nominal=nominal, use_bounds=use_bounds)

        nominal_scenario = BasicScenario(
            io=nominal.io,
            reference_fn=step_reference(R_STEP),
        )
        nominal_cfg = SimConfig(t_final=T_FINAL, seed=SEED)
        for controller in controllers:
            plant = build_first_order_grid(family)[plant_id]
            plant.u_bounds = u_bounds
            result = simulate_closed_loop(plant=plant, controller=controller, scenario=nominal_scenario, cfg=nominal_cfg)
            metrics = compute_metrics(result.traj)
            runtime = benchmark_controller_replay(
                controller=controller,
                traj=result.traj,
                warmup_repeats=BENCHMARK_WARMUP_REPEATS,
                measure_repeats=BENCHMARK_MEASURE_REPEATS,
            )
            detailed_rows.append(
                {
                    "scenario": "nominal",
                    "plant_id": plant_id,
                    "controller": controller.name,
                    **metrics,
                    **runtime,
                }
            )

        dt = float(nominal.dt)
        n_steps = n_steps_for(t_final=T_FINAL, dt=dt)
        seed_pid = (int(SEED) + int(zlib.crc32(plant_id.encode("utf-8")))) % (2**32)
        rng = np.random.default_rng(seed_pid)
        noise_y = rng.normal(loc=0.0, scale=float(SIGMA_Y), size=(n_steps,))
        noise_du = rng.normal(loc=0.0, scale=float(SIGMA_DU), size=(n_steps,))
        step_on = (np.arange(n_steps, dtype=float) * dt) >= float(DU_STEP_TIME)
        du = noise_du + (float(DU_STEP_MAG) * step_on.astype(float))

        disturbed_scenario = BasicScenario(
            io=nominal.io,
            reference_fn=step_reference(R_STEP),
            disturbance_u_fn=make_disturbance_u_fn_from_trace(du=du, dt=dt),
            measurement_fn=make_measurement_fn_from_trace(noise_y=noise_y, dt=dt),
        )
        disturbed_cfg = SimConfig(t_final=T_FINAL, seed=SEED)
        controllers = build_controllers(plant_id, nominal=nominal, use_bounds=use_bounds)
        for controller in controllers:
            plant = build_first_order_grid(family)[plant_id]
            plant.u_bounds = u_bounds
            result = simulate_closed_loop(plant=plant, controller=controller, scenario=disturbed_scenario, cfg=disturbed_cfg)
            metrics = compute_metrics(result.traj)
            runtime = benchmark_controller_replay(
                controller=controller,
                traj=result.traj,
                warmup_repeats=BENCHMARK_WARMUP_REPEATS,
                measure_repeats=BENCHMARK_MEASURE_REPEATS,
            )
            detailed_rows.append(
                {
                    "scenario": "disturbed",
                    "plant_id": plant_id,
                    "controller": controller.name,
                    **metrics,
                    **runtime,
                }
            )

        hidden_pole_scenario = BasicScenario(
            io=nominal.io,
            reference_fn=step_reference(R_STEP),
        )
        hidden_pole_cfg = SimConfig(t_final=T_FINAL, seed=SEED)
        controllers = build_controllers(plant_id, nominal=nominal, use_bounds=use_bounds)
        for controller in controllers:
            plant = build_secret_pole_mismatch_plant(
                plant_id,
                hidden_pole=HIDDEN_POLE,
                use_bounds=use_bounds,
            )
            result = simulate_closed_loop(
                plant=plant,
                controller=controller,
                scenario=hidden_pole_scenario,
                cfg=hidden_pole_cfg,
            )
            metrics = compute_metrics(result.traj)
            runtime = benchmark_controller_replay(
                controller=controller,
                traj=result.traj,
                warmup_repeats=BENCHMARK_WARMUP_REPEATS,
                measure_repeats=BENCHMARK_MEASURE_REPEATS,
            )
            detailed_rows.append(
                {
                    "scenario": "hidden_pole",
                    "plant_id": plant_id,
                    "controller": controller.name,
                    **metrics,
                    **runtime,
                }
            )

    _assign_group_ranks(detailed_rows)

    detailed_fieldnames = (
        "scenario",
        "plant_id",
        "controller",
        "mean_cost",
        "peak_error",
        "tail_rms_error",
        "tail_growth_ratio",
        "max_abs_u",
        "mean_step_ms",
        "median_step_ms",
        "p95_step_ms",
        "max_step_ms",
        "utilization_pct",
        "benchmark_samples",
        "late_escalation",
        "failure",
        "failure_reason",
        "failure_detail",
        "rank_mean_cost",
        "rank_peak_error",
        "rank_tail_rms_error",
        "rank_tail_growth_ratio",
        "rank_max_abs_u",
        "rank_failure",
        "composite_rank",
    )
    detailed_path = results_dir / "detailed_metrics.csv"
    _write_csv(detailed_path, detailed_rows, fieldnames=detailed_fieldnames)

    scenario_summary = _aggregate_rows(detailed_rows, group_keys=("scenario", "controller"))
    scenario_summary_path = results_dir / "scenario_summary.csv"
    _write_csv(scenario_summary_path, scenario_summary, fieldnames=scenario_summary[0].keys())

    overall_summary = _aggregate_rows(detailed_rows, group_keys=("controller",))
    overall_summary_path = results_dir / "overall_summary.csv"
    _write_csv(overall_summary_path, overall_summary, fieldnames=overall_summary[0].keys())

    failure_breakdown = _failure_breakdown_rows(detailed_rows)
    failure_breakdown_path = results_dir / "failure_breakdown.csv"
    if failure_breakdown:
        _write_csv(failure_breakdown_path, failure_breakdown, fieldnames=failure_breakdown[0].keys())
    else:
        failure_breakdown_path.write_text(
            "scenario,controller,failure_count,failed_plants,failure_reasons,failure_details,late_escalation_plants\n"
        )

    nominal_rows = [row for row in detailed_rows if row["scenario"] == "nominal"]
    nominal_pass_controllers = {
        str(controller)
        for controller in sorted({str(row["controller"]) for row in nominal_rows})
        if all(
            not bool(row["failure"])
            for row in nominal_rows
            if str(row["controller"]) == str(controller)
        )
    }
    focal_nominal = build_first_order_grid(family)[TRADEOFF_PLANT_ID]
    focal_u_bounds = focal_nominal.u_bounds if use_bounds else None
    focal_nominal.u_bounds = focal_u_bounds
    focal_nominal_scenario = BasicScenario(
        io=focal_nominal.io,
        reference_fn=step_reference(R_STEP),
    )
    focal_nominal_cfg = SimConfig(t_final=T_FINAL, seed=SEED)
    nominal_tradeoff_controller_rows: List[dict] = []
    for controller in build_controllers(TRADEOFF_PLANT_ID, nominal=focal_nominal, use_bounds=use_bounds):
        if str(controller.name) == "MPC" or str(controller.name) not in nominal_pass_controllers:
            continue
        plant = build_first_order_grid(family)[TRADEOFF_PLANT_ID]
        plant.u_bounds = focal_u_bounds
        result = simulate_closed_loop(
            plant=plant,
            controller=controller,
            scenario=focal_nominal_scenario,
            cfg=focal_nominal_cfg,
        )
        tradeoff_loops = replay_loops_for_target_steps(
            traj=result.traj,
            measure_repeats=TRADEOFF_RUNTIME_MEASURE_REPEATS,
            target_steps=TRADEOFF_RUNTIME_TARGET_STEPS,
        )
        runtime = benchmark_controller_replay(
            controller=controller,
            traj=result.traj,
            warmup_repeats=TRADEOFF_RUNTIME_WARMUP_REPEATS,
            measure_repeats=TRADEOFF_RUNTIME_MEASURE_REPEATS,
            replay_loops_per_repeat=tradeoff_loops,
        )
        nominal_tradeoff_controller_rows.append(
            {
                "scenario": "nominal",
                "plant_id": TRADEOFF_PLANT_ID,
                "controller": controller.name,
                **compute_metrics(result.traj),
                **runtime,
            }
        )

    mpc_sweep_rows: List[dict] = []
    focal_mpc_sweep_rows: List[dict] = []

    for horizon_n in range(1, 51):
        horizon_rows: List[dict] = []
        for plant_id in FIRST_ORDER_PLANT_IDS:
            nominal = build_first_order_grid(family)[plant_id]
            u_bounds = nominal.u_bounds if use_bounds else None
            scenario = BasicScenario(
                io=nominal.io,
                reference_fn=step_reference(R_STEP),
            )
            cfg = SimConfig(t_final=T_FINAL, seed=SEED)
            controller = LinearMPCController(
                MPCConfig(
                    N=horizon_n,
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
            plant = build_first_order_grid(family)[plant_id]
            plant.u_bounds = u_bounds
            result = simulate_closed_loop(plant=plant, controller=controller, scenario=scenario, cfg=cfg)
            metrics = compute_metrics(result.traj)
            replay_loops = MPC_SWEEP_REPLAY_LOOPS
            warmup_repeats = MPC_SWEEP_WARMUP_REPEATS
            measure_repeats = MPC_SWEEP_MEASURE_REPEATS
            if plant_id == TRADEOFF_PLANT_ID:
                replay_loops = replay_loops_for_target_steps(
                    traj=result.traj,
                    measure_repeats=TRADEOFF_RUNTIME_MEASURE_REPEATS,
                    target_steps=TRADEOFF_RUNTIME_TARGET_STEPS,
                )
                warmup_repeats = TRADEOFF_RUNTIME_WARMUP_REPEATS
                measure_repeats = TRADEOFF_RUNTIME_MEASURE_REPEATS
            runtime = benchmark_controller_replay(
                controller=controller,
                traj=result.traj,
                warmup_repeats=warmup_repeats,
                measure_repeats=measure_repeats,
                replay_loops_per_repeat=replay_loops,
            )
            horizon_rows.append(
                {
                    "N": int(horizon_n),
                    "plant_id": plant_id,
                    **metrics,
                    **runtime,
                }
            )
            if plant_id == TRADEOFF_PLANT_ID:
                focal_mpc_sweep_rows.append(
                    {
                        "N": int(horizon_n),
                        "plant_id": plant_id,
                        **metrics,
                        **runtime,
                        "failure_count": int(metrics["failure"]),
                    }
                )

        agg = _aggregate_rows(
            horizon_rows,
            group_keys=("N",),
            aggregate_metrics=(
                "mean_cost",
                "peak_error",
                "tail_rms_error",
                "tail_growth_ratio",
                "max_abs_u",
                "mean_step_ms",
                "median_step_ms",
                "p95_step_ms",
                "max_step_ms",
                "utilization_pct",
            ),
        )[0]
        mpc_sweep_rows.append(agg)

    mpc_sweep_path = results_dir / "mpc_horizon_sweep_nominal.csv"
    _write_csv(mpc_sweep_path, mpc_sweep_rows, fieldnames=mpc_sweep_rows[0].keys())
    focal_mpc_sweep_path = results_dir / f"mpc_horizon_sweep_nominal__{TRADEOFF_PLANT_ID}.csv"
    _write_csv(focal_mpc_sweep_path, focal_mpc_sweep_rows, fieldnames=focal_mpc_sweep_rows[0].keys())

    tradeoff_paths = _plot_nominal_tradeoffs(
        mpc_sweep_rows=focal_mpc_sweep_rows,
        nominal_controller_rows=nominal_tradeoff_controller_rows,
        figures_dir=figures_dir,
        plant_id=TRADEOFF_PLANT_ID,
    )
    robustness_tradeoff_paths = _plot_runtime_vs_robustness(
        mpc_sweep_rows=focal_mpc_sweep_rows,
        nominal_controller_rows=nominal_tradeoff_controller_rows,
        figures_dir=figures_dir,
    )

    summary_md_path = results_dir / "summary.md"
    _write_summary_markdown(
        summary_md_path,
        overall_rows=overall_summary,
        scenario_rows=scenario_summary,
        failure_rows=failure_breakdown,
    )

    print(f"Saved detailed metrics: {detailed_path}")
    print(f"Saved scenario summary: {scenario_summary_path}")
    print(f"Saved overall summary: {overall_summary_path}")
    print(f"Saved failure breakdown: {failure_breakdown_path}")
    print(f"Saved MPC horizon sweep: {mpc_sweep_path}")
    print(f"Saved focal MPC horizon sweep: {focal_mpc_sweep_path}")
    for tradeoff_path in tradeoff_paths:
        print(f"Saved tradeoff plot: {tradeoff_path}")
    for tradeoff_path in robustness_tradeoff_paths:
        print(f"Saved robustness-efficiency plot: {tradeoff_path}")
    print(f"Saved markdown summary: {summary_md_path}")
    print()
    print("Metric guide:")
    for name, description in METRIC_EXPLANATIONS:
        print(f"  {name:<8} {description}")
    print()
    print("Overall summary:")
    print(
        _text_table(
            overall_summary,
            columns=(
                ("controller", "controller"),
                ("rank", "mean_composite_rank"),
                ("cost", "mean_mean_cost"),
                ("peak", "mean_peak_error"),
                ("tail_rms", "mean_tail_rms_error"),
                ("tail_g", "mean_tail_growth_ratio"),
                ("p95_ms", "mean_p95_step_ms"),
                ("util_%", "mean_utilization_pct"),
                ("fails", "failure_count"),
            ),
        )
    )
    print()
    print("Scenario summary:")
    print(
        _text_table(
            scenario_summary,
            columns=(
                ("scenario", "scenario"),
                ("controller", "controller"),
                ("rank", "mean_composite_rank"),
                ("cost", "mean_mean_cost"),
                ("peak", "mean_peak_error"),
                ("tail_rms", "mean_tail_rms_error"),
                ("tail_g", "mean_tail_growth_ratio"),
                ("p95_ms", "mean_p95_step_ms"),
                ("util_%", "mean_utilization_pct"),
                ("fails", "failure_count"),
            ),
        )
    )
    print()
    print("Failure breakdown:")
    if failure_breakdown:
        print(
            _text_table(
                failure_breakdown,
                columns=(
                    ("scenario", "scenario"),
                    ("controller", "controller"),
                    ("fails", "failure_count"),
                    ("failed_plants", "failed_plants"),
                    ("why", "failure_reasons"),
                    ("details", "failure_details"),
                    ("late_escalation", "late_escalation_plants"),
                ),
            )
        )
    else:
        print("none")
    print()
    print("MPC horizon sweep (nominal plants, averaged):")
    print(
        _text_table(
            mpc_sweep_rows,
            columns=(
                ("N", "N"),
                ("cost", "mean_mean_cost"),
                ("peak", "mean_peak_error"),
                ("tail_rms", "mean_tail_rms_error"),
                ("tail_g", "mean_tail_growth_ratio"),
                ("p95_ms", "mean_p95_step_ms"),
                ("util_%", "mean_utilization_pct"),
                ("fails", "failure_count"),
            ),
        )
    )
    print()
    print(f"MPC horizon sweep ({TRADEOFF_PLANT_ID}, focal plant):")
    print(
        _text_table(
            focal_mpc_sweep_rows,
            columns=(
                ("N", "N"),
                ("cost", "mean_cost"),
                ("peak", "peak_error"),
                ("tail_rms", "tail_rms_error"),
                ("tail_g", "tail_growth_ratio"),
                ("p95_ms", "p95_step_ms"),
                ("util_%", "utilization_pct"),
                ("fails", "failure_count"),
            ),
        )
    )


if __name__ == "__main__":
    run_all(use_bounds=False)
