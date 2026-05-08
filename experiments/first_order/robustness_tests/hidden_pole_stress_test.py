from __future__ import annotations

import csv
import os
from pathlib import Path
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
    R_STEP,
    SEED,
    T_FINAL,
    USE_BOUNDS,
    _discretize_exact,
    _failure_breakdown_rows,
    _markdown_table,
    _text_table,
    compute_metrics,
    plant_pz_from_id,
    require_file,
    step_reference,
)
from control_bench.config import FIRST_ORDER_COST_WEIGHTS, FIRST_ORDER_FAMILY as family, FIRST_ORDER_PLANT_IDS
from control_bench.controllers.mpc import LinearMPCController, MPCConfig
from control_bench.controllers.rl_numpy.controller import BackpropRLController
from control_bench.core.scenario_impl import BasicScenario
from control_bench.core.sim import simulate_closed_loop
from control_bench.core.types import SimConfig
from control_bench.experiment_manifest import first_order_hidden_pole_stress_test, first_order_mlp_variants
from control_bench.plants.first_order_family import build_first_order_grid
from control_bench.plants.lti_state_space import DiscreteLTIPlant


STRESS_CFG = first_order_hidden_pole_stress_test()
MAX_HIDDEN_ORDER = int(STRESS_CFG["max_hidden_order"])
HIDDEN_POLE = float(STRESS_CFG["hidden_pole"])
WEIGHTS_DIR = EXPERIMENTS_ROOT / "results" / "rl_numpy" / "per_plant"
FIGURES_DIR = (
    EXPERIMENTS_ROOT / "results" / "figures" / "robustness_tests" / str(STRESS_CFG["output_figures_namespace"])
)
METRICS_DIR = (
    EXPERIMENTS_ROOT / "results" / "metrics" / "robustness_tests" / str(STRESS_CFG["output_metrics_namespace"])
)


def build_hidden_pole_chain_mismatch_plant(
    plant_id: str,
    *,
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

    # Cascade n identical hidden poles after the nominal first-order plant while preserving DC gain.
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
    factories["RL (PID-features)"] = lambda rl_pid_path=rl_pid_path, u_bounds=u_bounds, u_min=u_min, u_max=u_max: BackpropRLController.load_npz(
        rl_pid_path,
        kind="pidfeat",
        dt=family.dt,
        u_bounds=u_bounds,
        u_min=u_min,
        u_max=u_max,
        name="RL (PID-features)",
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
        controller_rows = sorted(grouped.get(controller, []), key=lambda row: int(row["hidden_order"]))
        pass_orders = [int(row["hidden_order"]) for row in controller_rows if bool(row["pass_all_plants"])]
        fail_rows = [row for row in controller_rows if not bool(row["pass_all_plants"])]
        first_fail_row = fail_rows[0] if fail_rows else None
        max_pass_all_order = max(pass_orders) if pass_orders else 0
        summary_rows.append(
            {
                "controller": controller,
                "eligible_after_nominal": bool(nominal["pass_all_plants"]),
                "nominal_failure_count": int(nominal["failure_count"]),
                "nominal_failed_plants": str(nominal["failed_plants"]),
                "max_pass_all_order": int(max_pass_all_order),
                "first_failure_order": int(first_fail_row["hidden_order"]) if first_fail_row is not None else "",
                "failed_plants_at_first_failure": str(first_fail_row["failed_plants"]) if first_fail_row is not None else "",
                "failure_reasons_at_first_failure": str(first_fail_row["failure_reasons"]) if first_fail_row is not None else "",
                "worst_tail_rms_at_first_failure": (
                    float(first_fail_row["worst_tail_rms_error"]) if first_fail_row is not None else 0.0
                ),
                "last_tested_order": int(controller_rows[-1]["hidden_order"]) if controller_rows else 0,
                "num_orders_tested": int(len(controller_rows)),
            }
        )

    summary_rows.sort(
        key=lambda row: (
            int(not bool(row["eligible_after_nominal"])),
            -int(row["max_pass_all_order"]),
            int(row["first_failure_order"]) if str(row["first_failure_order"]).strip() else 10**9,
            float(row["worst_tail_rms_at_first_failure"]),
            str(row["controller"]),
        )
    )
    return summary_rows


def _plot_pass_fraction_heatmap(
    *,
    nominal_rows: Sequence[dict],
    stress_rows: Sequence[dict],
    save_path: Path,
) -> None:
    controller_order = [str(row["controller"]) for row in nominal_rows]
    max_hidden_order = max([0] + [int(row["hidden_order"]) for row in stress_rows])
    values = np.zeros((len(controller_order), max_hidden_order + 1), dtype=float)
    annotations = [["" for _ in range(max_hidden_order + 1)] for _ in controller_order]

    nominal_by_controller = {str(row["controller"]): row for row in nominal_rows}
    stress_lookup = {(str(row["controller"]), int(row["hidden_order"])): row for row in stress_rows}

    for row_idx, controller in enumerate(controller_order):
        nominal = nominal_by_controller[controller]
        values[row_idx, 0] = float(nominal["pass_fraction"])
        annotations[row_idx][0] = f"{int(nominal['pass_count'])}/{int(nominal['num_plants'])}"
        eliminated = False
        for hidden_order in range(1, max_hidden_order + 1):
            row = stress_lookup.get((controller, hidden_order))
            if row is None:
                if eliminated:
                    values[row_idx, hidden_order] = 0.0
                    annotations[row_idx][hidden_order] = "elim"
                else:
                    values[row_idx, hidden_order] = np.nan
                    annotations[row_idx][hidden_order] = "-"
            else:
                values[row_idx, hidden_order] = float(row["pass_fraction"])
                annotations[row_idx][hidden_order] = f"{int(row['pass_count'])}/{int(row['num_plants'])}"
                if not bool(row["pass_all_plants"]):
                    eliminated = True

    fig_h = max(4.8, 0.6 * len(controller_order) + 2.0)
    fig_w = max(8.0, 1.0 * (max_hidden_order + 1) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    masked = np.ma.masked_invalid(values)
    im = ax.imshow(masked, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fraction of plants passed")

    ax.set_xticks(np.arange(max_hidden_order + 1))
    ax.set_xticklabels(["nominal"] + [f"n={idx}" for idx in range(1, max_hidden_order + 1)], rotation=0)
    ax.set_yticks(np.arange(len(controller_order)))
    ax.set_yticklabels(controller_order)
    ax.set_xlabel("Hidden-pole mismatch order in (10 / (s + 10))^n")
    ax.set_title("Hidden-pole stress test | pass fraction across first-order plants")

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
    tested_hidden_orders: Sequence[int],
) -> None:
    content = [
        "# Hidden-pole stress test",
        "",
        "The stress plant is defined as:",
        "",
        "`G_real(s) = G_nominal(s) * (10 / (s + 10))^n`",
        "",
        "Method:",
        "- Screen all controllers on the nominal first-order plant set.",
        "- Only controllers with zero nominal failures are eligible for the hidden-pole stress sweep.",
        "- Increase hidden-pole order `n = 1, 2, ...` until no eligible controller passes all plants.",
        "",
        f"Rollout horizon: `{T_FINAL:g} s`",
        f"Reference step: `{R_STEP:g}`",
        f"Hidden pole location: `{HIDDEN_POLE:g}`",
        f"Tested hidden orders: `{list(tested_hidden_orders)}`",
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
                "max_pass_all_order",
                "first_failure_order",
                "failed_plants_at_first_failure",
                "failure_reasons_at_first_failure",
                "worst_tail_rms_at_first_failure",
            ),
        ),
        "",
        "## Order-by-order progression",
        _markdown_table(
            progression_rows,
            columns=(
                "hidden_order",
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


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    nominal_grid = build_first_order_grid(family)
    sim_cfg = SimConfig(t_final=T_FINAL, seed=SEED)

    nominal_detailed_rows: List[dict] = []
    all_controller_labels: set[str] = set()

    for plant_id in FIRST_ORDER_PLANT_IDS:
        nominal = nominal_grid[plant_id]
        scenario = BasicScenario(io=nominal.io, reference_fn=step_reference(R_STEP))
        factories = build_controller_factories(plant_id, nominal=nominal, use_bounds=USE_BOUNDS)
        all_controller_labels.update(factories.keys())
        for controller_name, factory in factories.items():
            controller = factory()
            result = simulate_closed_loop(plant=nominal, controller=controller, scenario=scenario, cfg=sim_cfg)
            row = {
                "scenario": "nominal_screen",
                "controller": controller_name,
                "plant_id": plant_id,
                "hidden_order": 0,
                **compute_metrics(result.traj),
            }
            nominal_detailed_rows.append(row)

    nominal_summary_rows = _summarize_stage_rows(
        nominal_detailed_rows,
        group_keys=("controller",),
    )
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
        raise RuntimeError("No controllers passed nominal screening; hidden-pole stress test cannot proceed.")

    stress_detailed_rows: List[dict] = []
    tested_hidden_orders: List[int] = []
    for hidden_order in range(1, MAX_HIDDEN_ORDER + 1):
        tested_hidden_orders.append(hidden_order)
        order_rows: List[dict] = []
        for plant_id in FIRST_ORDER_PLANT_IDS:
            nominal = nominal_grid[plant_id]
            hidden_plant = build_hidden_pole_chain_mismatch_plant(
                plant_id,
                hidden_pole=HIDDEN_POLE,
                hidden_order=hidden_order,
                use_bounds=USE_BOUNDS,
            )
            scenario = BasicScenario(io=hidden_plant.io, reference_fn=step_reference(R_STEP))
            factories = build_controller_factories(plant_id, nominal=nominal, use_bounds=USE_BOUNDS)
            for controller_name in sorted(active_controllers):
                controller = factories[controller_name]()
                result = simulate_closed_loop(plant=hidden_plant, controller=controller, scenario=scenario, cfg=sim_cfg)
                row = {
                    "scenario": f"hidden_order_{hidden_order}",
                    "controller": controller_name,
                    "plant_id": plant_id,
                    "hidden_order": hidden_order,
                    **compute_metrics(result.traj),
                }
                order_rows.append(row)
        stress_detailed_rows.extend(order_rows)
        order_summary = _summarize_stage_rows(order_rows, group_keys=("hidden_order", "controller"))
        failed_this_order = {
            str(row["controller"])
            for row in order_summary
            if not bool(row["pass_all_plants"])
        }
        active_controllers -= failed_this_order
        any_controller_still_passes = bool(active_controllers)
        print(f"Completed hidden-order stress n={hidden_order} | controllers passing all plants: {len(active_controllers)}")
        if not any_controller_still_passes:
            break

    progression_rows = _summarize_stage_rows(
        stress_detailed_rows,
        group_keys=("hidden_order", "controller"),
    )
    progression_rows.sort(
        key=lambda row: (
            int(row["hidden_order"]),
            int(not bool(row["pass_all_plants"])),
            int(row["failure_count"]),
            float(row["worst_tail_rms_error"]),
            str(row["controller"]),
        )
    )

    survival_rows = _controller_survival_summary(nominal_summary_rows, progression_rows)
    failure_rows = _failure_breakdown_rows(
        [*nominal_detailed_rows, *stress_detailed_rows]
    )

    detailed_rows = [*nominal_detailed_rows, *stress_detailed_rows]
    _write_csv_selected(
        METRICS_DIR / "detailed_rows.csv",
        detailed_rows,
        fieldnames=(
            "scenario",
            "controller",
            "plant_id",
            "hidden_order",
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
        METRICS_DIR / "hidden_pole_progression.csv",
        progression_rows,
        fieldnames=(
            "hidden_order",
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
            "max_pass_all_order",
            "first_failure_order",
            "failed_plants_at_first_failure",
            "failure_reasons_at_first_failure",
            "worst_tail_rms_at_first_failure",
            "last_tested_order",
            "num_orders_tested",
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
        tested_hidden_orders=tested_hidden_orders,
    )

    _plot_pass_fraction_heatmap(
        nominal_rows=nominal_summary_rows,
        stress_rows=progression_rows,
        save_path=FIGURES_DIR / "hidden_pole_pass_fraction_heatmap.png",
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
                ("max_n", "max_pass_all_order"),
                ("first_fail_n", "first_failure_order"),
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
