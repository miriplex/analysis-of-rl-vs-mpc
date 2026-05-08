from __future__ import annotations

import csv
from pathlib import Path
import sys
from typing import Callable, Dict, List, Sequence, Tuple
import zlib

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
SRC_ROOT = PROJECT_ROOT / "src"
FIRST_ORDER_ROOT = EXPERIMENTS_ROOT / "first_order"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(FIRST_ORDER_ROOT) not in sys.path:
    sys.path.insert(0, str(FIRST_ORDER_ROOT))

from benchmark_first_order_robustness_metrics import (
    SEED,
    USE_BOUNDS,
    _failure_breakdown_rows,
    _markdown_table,
    _text_table,
    compute_metrics,
    make_disturbance_u_fn_from_trace,
    make_measurement_fn_from_trace,
    n_steps_for,
    require_file,
    step_reference,
)
from control_bench.config import FIRST_ORDER_COST_WEIGHTS, FIRST_ORDER_FAMILY as family, FIRST_ORDER_PLANT_IDS
from control_bench.controllers.mpc import LinearMPCController, MPCConfig
from control_bench.controllers.rl_numpy.controller import BackpropRLController
from control_bench.core.scenario_impl import BasicScenario
from control_bench.core.sim import simulate_closed_loop
from control_bench.core.types import SimConfig
from control_bench.experiment_manifest import first_order_disturbance_stress_test, first_order_mlp_variants
from control_bench.plants.first_order_family import build_first_order_grid


STRESS_CFG = first_order_disturbance_stress_test()
T_FINAL = float(STRESS_CFG["t_final_seconds"])
R_STEP = float(STRESS_CFG["reference_step"])
BASE_SIGMA_Y = float(STRESS_CFG["base_measurement_noise_std"])
BASE_SIGMA_DU = float(STRESS_CFG["base_disturbance_noise_std"])
DU_STEP_TIME = float(STRESS_CFG["disturbance_step_time_seconds"])
BASE_DU_STEP_MAG = float(STRESS_CFG["base_disturbance_step_magnitude"])
DISTURBANCE_SCALE_SCHEDULE = tuple(STRESS_CFG["disturbance_scale_schedule"])

WEIGHTS_DIR = EXPERIMENTS_ROOT / "results" / "rl_numpy" / "per_plant"
FIGURES_DIR = (
    EXPERIMENTS_ROOT / "results" / "figures" / "robustness_tests" / str(STRESS_CFG["output_figures_namespace"])
)
METRICS_DIR = (
    EXPERIMENTS_ROOT / "results" / "metrics" / "robustness_tests" / str(STRESS_CFG["output_metrics_namespace"])
)


def disturbance_scale_values() -> List[float]:
    values: List[float] = []
    seen: set[float] = set()
    for entry in DISTURBANCE_SCALE_SCHEDULE:
        start = float(entry["start"])
        stop = float(entry["stop"])
        step = float(entry["step"])
        if step <= 0:
            raise ValueError("disturbance scale schedule step must be > 0")
        current = start
        while current <= stop + 1e-12:
            rounded = round(float(current), 10)
            if rounded not in seen:
                values.append(rounded)
                seen.add(rounded)
            current += step
    values.sort()
    return values


def build_controller_factories(plant_id: str, *, nominal, use_bounds: bool = False) -> Dict[str, Callable[[], object]]:
    u_bounds = nominal.u_bounds if use_bounds else None
    u_min = family.u_min if use_bounds else None
    u_max = family.u_max if use_bounds else None

    factories: Dict[str, Callable[[], object]] = {}
    factories["MPC"] = lambda nominal=nominal, u_bounds=u_bounds: LinearMPCController(
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

    rl_pid_path = str(WEIGHTS_DIR / f"rl_pidfeat__{plant_id}.npz")
    require_file(rl_pid_path)
    factories["RL (PID-features)"] = (
        lambda rl_pid_path=rl_pid_path, u_bounds=u_bounds, u_min=u_min, u_max=u_max: BackpropRLController.load_npz(
            rl_pid_path,
            kind="pidfeat",
            dt=family.dt,
            u_bounds=u_bounds,
            u_min=u_min,
            u_max=u_max,
            name="RL (PID-features)",
        )
    )

    for variant in first_order_mlp_variants():
        label = str(variant["label"])
        weight_path = str(WEIGHTS_DIR / f"{variant['canonical_filename_prefix']}__{plant_id}.npz")
        require_file(weight_path)
        factories[label] = (
            lambda *,
            weight_path=weight_path,
            variant=variant,
            label=label,
            u_bounds=u_bounds,
            u_min=u_min,
            u_max=u_max: BackpropRLController.load_npz(
                weight_path,
                kind="rich",
                dt=family.dt,
                u_bounds=u_bounds,
                u_min=u_min,
                u_max=u_max,
                mlp_hidden=tuple(int(v) for v in variant["hidden_layers"]),
                mlp_activation=str(variant["activation"]),
                rich_feature_set=str(variant["feature_set"]),
                name=label,
            )
        )

    return factories


def _summarize_stage_rows(rows: Sequence[dict], *, group_keys: Sequence[str]) -> List[dict]:
    grouped: Dict[Tuple[object, ...], List[dict]] = {}
    for row in rows:
        grouped.setdefault(tuple(row[key] for key in group_keys), []).append(row)

    summary_rows: List[dict] = []
    for key in sorted(grouped):
        group = grouped[key]
        summary = {group_key: key[idx] for idx, group_key in enumerate(group_keys)}
        summary["num_plants"] = int(len(group))
        summary["failure_count"] = int(sum(bool(row["failure"]) for row in group))
        summary["pass_count"] = int(sum(not bool(row["failure"]) for row in group))
        summary["pass_fraction"] = float(summary["pass_count"] / summary["num_plants"]) if summary["num_plants"] else 0.0
        summary["pass_all_plants"] = bool(summary["failure_count"] == 0)
        summary["failed_plants"] = ",".join(sorted(str(row["plant_id"]) for row in group if bool(row["failure"])))
        summary["late_escalation_plants"] = ",".join(
            sorted(str(row["plant_id"]) for row in group if bool(row["late_escalation"]))
        )
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
        summary["failure_details"] = "; ".join(
            sorted(
                {
                    f"{row['plant_id']} ({row['failure_detail']})"
                    for row in group
                    if bool(row["failure"]) and str(row.get("failure_detail", "")).strip()
                }
            )
        )
        summary["mean_cost"] = float(np.mean([float(row["mean_cost"]) for row in group]))
        summary["worst_cost"] = float(np.max([float(row["mean_cost"]) for row in group]))
        summary["mean_peak_error"] = float(np.mean([float(row["peak_error"]) for row in group]))
        summary["worst_peak_error"] = float(np.max([float(row["peak_error"]) for row in group]))
        summary["mean_tail_rms_error"] = float(np.mean([float(row["tail_rms_error"]) for row in group]))
        summary["worst_tail_rms_error"] = float(np.max([float(row["tail_rms_error"]) for row in group]))
        summary["mean_tail_growth_ratio"] = float(np.mean([float(row["tail_growth_ratio"]) for row in group]))
        summary["worst_tail_growth_ratio"] = float(np.max([float(row["tail_growth_ratio"]) for row in group]))
        summary_rows.append(summary)
    return summary_rows


def _controller_survival_summary(nominal_rows: Sequence[dict], stress_rows: Sequence[dict]) -> List[dict]:
    nominal_by_controller = {str(row["controller"]): row for row in nominal_rows}
    grouped: Dict[str, List[dict]] = {}
    for row in stress_rows:
        grouped.setdefault(str(row["controller"]), []).append(row)

    summary_rows: List[dict] = []
    for controller in sorted(nominal_by_controller):
        nominal = nominal_by_controller[controller]
        controller_rows = sorted(grouped.get(controller, []), key=lambda row: float(row["disturbance_scale"]))
        pass_scales = [float(row["disturbance_scale"]) for row in controller_rows if bool(row["pass_all_plants"])]
        fail_rows = [row for row in controller_rows if not bool(row["pass_all_plants"])]
        first_fail_row = fail_rows[0] if fail_rows else None
        max_pass_all_scale = max(pass_scales) if pass_scales else 0.0
        summary_rows.append(
            {
                "controller": controller,
                "eligible_after_nominal": bool(nominal["pass_all_plants"]),
                "nominal_failure_count": int(nominal["failure_count"]),
                "nominal_failed_plants": str(nominal["failed_plants"]),
                "max_pass_all_scale": float(max_pass_all_scale),
                "first_failure_scale": float(first_fail_row["disturbance_scale"]) if first_fail_row is not None else "",
                "failed_plants_at_first_failure": str(first_fail_row["failed_plants"]) if first_fail_row is not None else "",
                "failure_reasons_at_first_failure": str(first_fail_row["failure_reasons"]) if first_fail_row is not None else "",
                "worst_tail_rms_at_first_failure": (
                    float(first_fail_row["worst_tail_rms_error"]) if first_fail_row is not None else 0.0
                ),
                "last_tested_scale": float(controller_rows[-1]["disturbance_scale"]) if controller_rows else 0.0,
                "num_scales_tested": int(len(controller_rows)),
            }
        )

    summary_rows.sort(
        key=lambda row: (
            int(not bool(row["eligible_after_nominal"])),
            -float(row["max_pass_all_scale"]),
            float(row["first_failure_scale"]) if str(row["first_failure_scale"]).strip() else 10**9,
            float(row["worst_tail_rms_at_first_failure"]),
            str(row["controller"]),
        )
    )
    return summary_rows


def _plot_pass_fraction_heatmap(
    *,
    nominal_rows: Sequence[dict],
    stress_rows: Sequence[dict],
    tested_scales: Sequence[float],
    save_path: Path,
) -> None:
    controller_order = [str(row["controller"]) for row in nominal_rows]
    values = np.zeros((len(controller_order), len(tested_scales) + 1), dtype=float)
    annotations = [["" for _ in range(len(tested_scales) + 1)] for _ in controller_order]

    nominal_by_controller = {str(row["controller"]): row for row in nominal_rows}
    stress_lookup = {(str(row["controller"]), float(row["disturbance_scale"])): row for row in stress_rows}

    for row_idx, controller in enumerate(controller_order):
        nominal = nominal_by_controller[controller]
        values[row_idx, 0] = float(nominal["pass_fraction"])
        annotations[row_idx][0] = f"{int(nominal['pass_count'])}/{int(nominal['num_plants'])}"
        eliminated = False
        for col_idx, disturbance_scale in enumerate(tested_scales, start=1):
            row = stress_lookup.get((controller, float(disturbance_scale)))
            if row is None:
                if eliminated:
                    values[row_idx, col_idx] = 0.0
                    annotations[row_idx][col_idx] = "elim"
                else:
                    values[row_idx, col_idx] = np.nan
                    annotations[row_idx][col_idx] = "-"
            else:
                values[row_idx, col_idx] = float(row["pass_fraction"])
                annotations[row_idx][col_idx] = f"{int(row['pass_count'])}/{int(row['num_plants'])}"
                if not bool(row["pass_all_plants"]):
                    eliminated = True

    fig_h = max(4.8, 0.6 * len(controller_order) + 2.0)
    fig_w = max(8.0, 1.0 * (len(tested_scales) + 1) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    masked = np.ma.masked_invalid(values)
    im = ax.imshow(masked, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fraction of plants passed")

    ax.set_xticks(np.arange(len(tested_scales) + 1))
    ax.set_xticklabels(["nominal"] + [f"x{scale:g}" for scale in tested_scales], rotation=0)
    ax.set_yticks(np.arange(len(controller_order)))
    ax.set_yticklabels(controller_order)
    ax.set_xlabel("Disturbance severity multiplier on the disturbed benchmark traces")
    ax.set_title("Disturbance stress test | pass fraction across first-order plants")

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


def _write_summary_markdown(
    path: Path,
    *,
    nominal_rows: Sequence[dict],
    progression_rows: Sequence[dict],
    survival_rows: Sequence[dict],
    failure_rows: Sequence[dict],
    tested_scales: Sequence[float],
) -> None:
    content = [
        "# Disturbance stress test",
        "",
        "Method:",
        "- Screen all controllers on the nominal first-order plant set.",
        "- Only controllers with zero nominal failures are eligible for the disturbance stress sweep.",
        "- Reuse the standard disturbed benchmark traces and scale all disturbance terms by a severity multiplier `x`.",
        "- Increase `x = start, start + step, ...` until no eligible controller passes all plants.",
        "",
        f"Rollout horizon: `{T_FINAL:g} s`",
        f"Reference step: `{R_STEP:g}`",
        f"Base measurement noise std: `{BASE_SIGMA_Y:g}`",
        f"Base disturbance noise std: `{BASE_SIGMA_DU:g}`",
        f"Base disturbance step magnitude: `{BASE_DU_STEP_MAG:g}` at `t = {DU_STEP_TIME:g} s`",
        f"Tested disturbance scales: `{list(tested_scales)}`",
        "",
        "A disturbance scale `x` applies to:",
        f"- measurement noise: `x * {BASE_SIGMA_Y:g}`",
        f"- disturbance noise: `x * {BASE_SIGMA_DU:g}`",
        f"- disturbance step amplitude: `x * {BASE_DU_STEP_MAG:g}`",
        "",
        "## Nominal screening",
        _markdown_table(
            nominal_rows,
            columns=(
                "controller",
                "pass_all_plants",
                "pass_count",
                "num_plants",
                "failure_count",
                "mean_cost",
                "mean_tail_rms_error",
                "failed_plants",
            ),
        ),
        "",
        "## Survival summary",
        _markdown_table(
            survival_rows,
            columns=(
                "controller",
                "eligible_after_nominal",
                "max_pass_all_scale",
                "first_failure_scale",
                "failed_plants_at_first_failure",
                "failure_reasons_at_first_failure",
                "worst_tail_rms_at_first_failure",
            ),
        ),
        "",
        "## Scale-by-scale progression",
        _markdown_table(
            progression_rows,
            columns=(
                "disturbance_scale",
                "controller",
                "pass_all_plants",
                "pass_count",
                "num_plants",
                "failure_count",
                "mean_cost",
                "worst_tail_rms_error",
                "worst_peak_error",
                "failed_plants",
            ),
        ),
        "",
        "## Failure breakdown",
        (
            _markdown_table(
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
            if failure_rows
            else "_No failures._"
        ),
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def _write_csv_selected(path: Path, rows: Sequence[dict], *, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _scaled_disturbance_traces(*, plant_id: str, n_steps: int, dt: float, disturbance_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    seed_pid = (int(SEED) + int(zlib.crc32(plant_id.encode("utf-8")))) % (2**32)
    rng = np.random.default_rng(seed_pid)
    base_eps_y = rng.normal(loc=0.0, scale=1.0, size=(n_steps,))
    base_eps_du = rng.normal(loc=0.0, scale=1.0, size=(n_steps,))
    noise_y = float(disturbance_scale) * float(BASE_SIGMA_Y) * base_eps_y
    noise_du = float(disturbance_scale) * float(BASE_SIGMA_DU) * base_eps_du
    step_on = (np.arange(n_steps, dtype=float) * float(dt)) >= float(DU_STEP_TIME)
    du = noise_du + (float(disturbance_scale) * float(BASE_DU_STEP_MAG) * step_on.astype(float))
    return noise_y, du


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    nominal_grid = build_first_order_grid(family)
    sim_cfg = SimConfig(t_final=T_FINAL, seed=SEED)

    nominal_detailed_rows: List[dict] = []
    for plant_id in FIRST_ORDER_PLANT_IDS:
        nominal = nominal_grid[plant_id]
        scenario = BasicScenario(io=nominal.io, reference_fn=step_reference(R_STEP))
        factories = build_controller_factories(plant_id, nominal=nominal, use_bounds=USE_BOUNDS)
        for controller_name, factory in factories.items():
            controller = factory()
            result = simulate_closed_loop(plant=nominal, controller=controller, scenario=scenario, cfg=sim_cfg)
            row = {
                "scenario": "nominal_screen",
                "controller": controller_name,
                "plant_id": plant_id,
                "disturbance_scale": 0.0,
                **compute_metrics(result.traj),
            }
            nominal_detailed_rows.append(row)

    nominal_summary_rows = _summarize_stage_rows(nominal_detailed_rows, group_keys=("controller",))
    nominal_summary_rows.sort(
        key=lambda row: (
            int(not bool(row["pass_all_plants"])),
            int(row["failure_count"]),
            float(row["mean_tail_rms_error"]),
            float(row["mean_cost"]),
            str(row["controller"]),
        )
    )

    active_controllers = {
        str(row["controller"])
        for row in nominal_summary_rows
        if bool(row["pass_all_plants"])
    }
    if not active_controllers:
        raise RuntimeError("No controllers passed nominal screening; disturbance stress test cannot proceed.")

    tested_scales = disturbance_scale_values()
    stress_detailed_rows: List[dict] = []

    for disturbance_scale in tested_scales:
        scale_rows: List[dict] = []
        for plant_id in FIRST_ORDER_PLANT_IDS:
            nominal = nominal_grid[plant_id]
            dt = float(nominal.dt)
            n_steps = n_steps_for(t_final=T_FINAL, dt=dt)
            noise_y, du = _scaled_disturbance_traces(
                plant_id=plant_id,
                n_steps=n_steps,
                dt=dt,
                disturbance_scale=float(disturbance_scale),
            )
            scenario = BasicScenario(
                io=nominal.io,
                reference_fn=step_reference(R_STEP),
                disturbance_u_fn=make_disturbance_u_fn_from_trace(du=du, dt=dt),
                measurement_fn=make_measurement_fn_from_trace(noise_y=noise_y, dt=dt),
            )
            factories = build_controller_factories(plant_id, nominal=nominal, use_bounds=USE_BOUNDS)
            for controller_name in sorted(active_controllers):
                controller = factories[controller_name]()
                result = simulate_closed_loop(plant=nominal, controller=controller, scenario=scenario, cfg=sim_cfg)
                row = {
                    "scenario": f"disturbance_scale_{disturbance_scale:g}x",
                    "controller": controller_name,
                    "plant_id": plant_id,
                    "disturbance_scale": float(disturbance_scale),
                    **compute_metrics(result.traj),
                }
                scale_rows.append(row)
        stress_detailed_rows.extend(scale_rows)
        scale_summary = _summarize_stage_rows(scale_rows, group_keys=("disturbance_scale", "controller"))
        failed_this_scale = {
            str(row["controller"])
            for row in scale_summary
            if not bool(row["pass_all_plants"])
        }
        active_controllers -= failed_this_scale
        num_passing = len(active_controllers)
        print(
            f"Completed disturbance stress x{disturbance_scale:g} | controllers passing all plants: {num_passing}"
        )
        if num_passing == 0:
            tested_scales = [value for value in tested_scales if value <= disturbance_scale]
            break

    progression_rows = _summarize_stage_rows(
        stress_detailed_rows,
        group_keys=("disturbance_scale", "controller"),
    )
    progression_rows.sort(
        key=lambda row: (
            float(row["disturbance_scale"]),
            int(not bool(row["pass_all_plants"])),
            int(row["failure_count"]),
            float(row["worst_tail_rms_error"]),
            str(row["controller"]),
        )
    )

    survival_rows = _controller_survival_summary(nominal_summary_rows, progression_rows)
    failure_rows = _failure_breakdown_rows([*nominal_detailed_rows, *stress_detailed_rows])

    detailed_rows = [*nominal_detailed_rows, *stress_detailed_rows]
    _write_csv_selected(
        METRICS_DIR / "detailed_rows.csv",
        detailed_rows,
        fieldnames=(
            "scenario",
            "controller",
            "plant_id",
            "disturbance_scale",
            "mean_cost",
            "peak_error",
            "tail_rms_error",
            "tail_growth_ratio",
            "max_abs_u",
            "late_escalation",
            "failure",
            "failure_reason",
            "failure_detail",
        ),
    )
    _write_csv_selected(
        METRICS_DIR / "nominal_screening.csv",
        nominal_summary_rows,
        fieldnames=(
            "controller",
            "num_plants",
            "pass_count",
            "pass_fraction",
            "pass_all_plants",
            "failure_count",
            "failed_plants",
            "mean_cost",
            "worst_cost",
            "mean_tail_rms_error",
            "worst_tail_rms_error",
            "failure_reasons",
            "failure_details",
        ),
    )
    _write_csv_selected(
        METRICS_DIR / "disturbance_progression.csv",
        progression_rows,
        fieldnames=(
            "disturbance_scale",
            "controller",
            "num_plants",
            "pass_count",
            "pass_fraction",
            "pass_all_plants",
            "failure_count",
            "failed_plants",
            "mean_cost",
            "worst_cost",
            "mean_peak_error",
            "worst_peak_error",
            "mean_tail_rms_error",
            "worst_tail_rms_error",
            "mean_tail_growth_ratio",
            "worst_tail_growth_ratio",
            "failure_reasons",
            "failure_details",
            "late_escalation_plants",
        ),
    )
    _write_csv_selected(
        METRICS_DIR / "controller_survival_summary.csv",
        survival_rows,
        fieldnames=(
            "controller",
            "eligible_after_nominal",
            "nominal_failure_count",
            "nominal_failed_plants",
            "max_pass_all_scale",
            "first_failure_scale",
            "failed_plants_at_first_failure",
            "failure_reasons_at_first_failure",
            "worst_tail_rms_at_first_failure",
            "last_tested_scale",
            "num_scales_tested",
        ),
    )
    _write_csv_selected(
        METRICS_DIR / "failure_breakdown.csv",
        failure_rows,
        fieldnames=(
            "scenario",
            "controller",
            "failure_count",
            "failed_plants",
            "failure_reasons",
            "failure_details",
            "late_escalation_plants",
        ),
    )

    _write_summary_markdown(
        METRICS_DIR / "summary.md",
        nominal_rows=nominal_summary_rows,
        progression_rows=progression_rows,
        survival_rows=survival_rows,
        failure_rows=failure_rows,
        tested_scales=tested_scales,
    )

    _plot_pass_fraction_heatmap(
        nominal_rows=nominal_summary_rows,
        stress_rows=progression_rows,
        tested_scales=tested_scales,
        save_path=FIGURES_DIR / "disturbance_pass_fraction_heatmap.png",
    )

    print("\nNominal screening:")
    print(
        _text_table(
            nominal_summary_rows,
            columns=(
                ("controller", "controller"),
                ("pass_all", "pass_all_plants"),
                ("passes", "pass_count"),
                ("plants", "num_plants"),
                ("fails", "failure_count"),
                ("cost", "mean_cost"),
                ("tail_rms", "mean_tail_rms_error"),
                ("failed_plants", "failed_plants"),
            ),
        )
    )

    print("\nStress survival summary:")
    print(
        _text_table(
            survival_rows,
            columns=(
                ("controller", "controller"),
                ("eligible", "eligible_after_nominal"),
                ("max_x", "max_pass_all_scale"),
                ("first_fail_x", "first_failure_scale"),
                ("failed_plants", "failed_plants_at_first_failure"),
                ("why", "failure_reasons_at_first_failure"),
                ("tail@fail", "worst_tail_rms_at_first_failure"),
            ),
        )
    )

    if failure_rows:
        print("\nFailure breakdown:")
        print(
            _text_table(
                failure_rows,
                columns=(
                    ("scenario", "scenario"),
                    ("controller", "controller"),
                    ("fails", "failure_count"),
                    ("failed_plants", "failed_plants"),
                    ("why", "failure_reasons"),
                    ("details", "failure_details"),
                ),
            )
        )

    print(f"\nAll figures saved to: {FIGURES_DIR}")
    print(f"All metrics saved to: {METRICS_DIR}")


if __name__ == "__main__":
    main()
