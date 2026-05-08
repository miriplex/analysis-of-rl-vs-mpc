from __future__ import annotations

import zlib
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark_first_order_robustness_metrics import (
    _aggregate_rows,
    _assign_group_ranks,
    _failure_breakdown_rows,
    _markdown_table,
    _text_table,
    _write_csv,
    build_secret_pole_mismatch_plant,
    compute_metrics,
    make_disturbance_u_fn_from_trace,
    make_measurement_fn_from_trace,
    n_steps_for,
)
from control_bench.config import FIRST_ORDER_FAMILY as family, FIRST_ORDER_PLANT_IDS
from control_bench.core.scenario_impl import BasicScenario
from control_bench.core.sim import simulate_closed_loop
from control_bench.core.types import SimConfig
from control_bench.controllers.rl_numpy.controller import BackpropRLController
from control_bench.experiment_manifest import first_order_mlp_variant_study, first_order_mlp_variants
from control_bench.plants.first_order_family import build_first_order_grid


METRIC_COLUMNS = (
    "mean_cost",
    "peak_error",
    "tail_rms_error",
    "tail_growth_ratio",
    "max_abs_u",
    "composite_rank",
)

METRIC_EXPLANATIONS = (
    ("rank", "average within-plant/scenario rank across cost, peak, tail RMS, tail growth, max |u|, and failure flag; lower is better"),
    ("cost", "mean rollout stage cost under the same qy/ru/qu objective"),
    ("peak", "maximum absolute tracking error"),
    ("tail_rms", "RMS tracking error over the last 25 percent of the rollout"),
    ("tail_g", "RMS(last 10 percent) / RMS(previous 10 percent)"),
    ("max_u", "maximum absolute control magnitude"),
    ("fails", "number of runs flagged as failed by the threshold rule"),
)


def step_reference(level: float = 1.0):
    def _r(t: float) -> np.ndarray:
        return np.array([level], dtype=float)

    return _r


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\nRun: python /Users/leonidsokolov/PycharmProjects/3rd_year_project_sem2/control_bench/experiments/first_order/train_first_order_manifest_controllers.py"
        )


def _disturbed_traces(*, plant_id: str, t_final: float, sigma_y: float, sigma_du: float, step_time: float, step_mag: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    dt = float(family.dt)
    n_steps = n_steps_for(t_final=t_final, dt=dt)
    seed_pid = (int(seed) + int(zlib.crc32(plant_id.encode("utf-8")))) % (2**32)
    rng = np.random.default_rng(seed_pid)

    noise_y = rng.normal(loc=0.0, scale=float(sigma_y), size=(n_steps,))
    noise_du = rng.normal(loc=0.0, scale=float(sigma_du), size=(n_steps,))
    step_on = (np.arange(n_steps, dtype=float) * dt) >= float(step_time)
    du = noise_du + (float(step_mag) * step_on.astype(float))
    return noise_y, du


def _build_mlp_controller(*, variant: dict, plant_id: str, use_bounds: bool = False):
    nominal = build_first_order_grid(family)[plant_id]
    u_bounds = nominal.u_bounds if use_bounds else None
    u_min = family.u_min if use_bounds else None
    u_max = family.u_max if use_bounds else None
    weights_dir = EXPERIMENTS_ROOT / "results" / "rl_numpy" / "per_plant"
    weight_path = weights_dir / f"{variant['canonical_filename_prefix']}__{plant_id}.npz"
    require_file(weight_path)
    return BackpropRLController.load_npz(
        str(weight_path),
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


def _write_summary_markdown(
    path: Path,
    *,
    study_cfg: dict,
    overall_rows: Sequence[dict],
    scenario_rows: Sequence[dict],
    failure_rows: Sequence[dict],
) -> None:
    scenarios = dict(study_cfg["scenarios"])
    disturbed = dict(scenarios["disturbed"])
    hidden_pole = dict(scenarios["hidden_pole_mismatch"])
    content = [
        "# First-order MLP variant robustness metrics",
        "",
        f"Study script: `{study_cfg['script']}`",
        f"Nominal comparison horizon: `{float(study_cfg['t_final_seconds']):g} s`",
        f"Reference step: `{float(study_cfg['reference_step']):g}`",
        "",
        "Scenarios:",
        "- `nominal`: nominal plant, no disturbance",
        (
            f"- `disturbed`: measurement noise sigma={float(disturbed['sigma_y']):g}, "
            f"input disturbance sigma={float(disturbed['sigma_du']):g}, "
            f"step {float(disturbed['disturbance_step_magnitude']):g} at {float(disturbed['disturbance_step_time_seconds']):g}s"
        ),
        (
            f"- `hidden_pole`: actual plant has extra hidden pole `-( {float(hidden_pole['hidden_pole']):g} )`, "
            "controllers remain nominal"
        ),
        "",
        "Metrics (lower is better):",
        *(f"- `{name}`: {description}" for name, description in METRIC_EXPLANATIONS),
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
                ("max_u", "mean_max_abs_u"),
                ("fails", "failure_count"),
            ),
        ),
        "```",
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
        "",
        "## Scenario summary",
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
                "mean_max_abs_u",
                "failure_count",
                "late_escalation_count",
                "num_runs",
            ),
        ),
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    variants = first_order_mlp_variants()
    study_cfg = first_order_mlp_variant_study()
    scenarios_cfg = dict(study_cfg["scenarios"])
    figures_dir = EXPERIMENTS_ROOT / "results" / "figures" / "first_order_mlp_variants"
    metrics_dir = EXPERIMENTS_ROOT / "results" / "metrics" / "first_order_mlp_variants"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    t_final = float(study_cfg["t_final_seconds"])
    seed = 0
    r_step = float(study_cfg["reference_step"])

    detailed_rows: List[dict] = []

    for plant_id in FIRST_ORDER_PLANT_IDS:
        dummy = build_first_order_grid(family)[plant_id]
        nominal_scenario = BasicScenario(io=dummy.io, reference_fn=step_reference(r_step))
        sim_cfg = SimConfig(t_final=t_final, seed=seed)

        disturbed_cfg = dict(scenarios_cfg["disturbed"])
        noise_y, du = _disturbed_traces(
            plant_id=plant_id,
            t_final=t_final,
            sigma_y=float(disturbed_cfg["sigma_y"]),
            sigma_du=float(disturbed_cfg["sigma_du"]),
            step_time=float(disturbed_cfg["disturbance_step_time_seconds"]),
            step_mag=float(disturbed_cfg["disturbance_step_magnitude"]),
            seed=seed,
        )
        disturbed_scenario = BasicScenario(
            io=dummy.io,
            reference_fn=step_reference(r_step),
            disturbance_u_fn=make_disturbance_u_fn_from_trace(du=du, dt=float(dummy.dt)),
            measurement_fn=make_measurement_fn_from_trace(noise_y=noise_y, dt=float(dummy.dt)),
        )

        hidden_cfg = dict(scenarios_cfg["hidden_pole_mismatch"])
        nominal_runs: Dict[str, object] = {}

        for variant in variants:
            label = str(variant["label"])
            variant_id = str(variant["id"])

            controller = _build_mlp_controller(variant=variant, plant_id=plant_id, use_bounds=False)
            nominal_plant = build_first_order_grid(family)[plant_id]
            nominal_res = simulate_closed_loop(plant=nominal_plant, controller=controller, scenario=nominal_scenario, cfg=sim_cfg)
            nominal_runs[label] = nominal_res.traj
            detailed_rows.append(
                {
                    "scenario": "nominal",
                    "plant_id": plant_id,
                    "controller": label,
                    "variant_id": variant_id,
                    **compute_metrics(nominal_res.traj),
                }
            )

            controller = _build_mlp_controller(variant=variant, plant_id=plant_id, use_bounds=False)
            disturbed_plant = build_first_order_grid(family)[plant_id]
            disturbed_res = simulate_closed_loop(plant=disturbed_plant, controller=controller, scenario=disturbed_scenario, cfg=sim_cfg)
            detailed_rows.append(
                {
                    "scenario": "disturbed",
                    "plant_id": plant_id,
                    "controller": label,
                    "variant_id": variant_id,
                    **compute_metrics(disturbed_res.traj),
                }
            )

            controller = _build_mlp_controller(variant=variant, plant_id=plant_id, use_bounds=False)
            hidden_plant = build_secret_pole_mismatch_plant(
                plant_id,
                hidden_pole=float(hidden_cfg["hidden_pole"]),
                use_bounds=False,
            )
            hidden_res = simulate_closed_loop(plant=hidden_plant, controller=controller, scenario=nominal_scenario, cfg=sim_cfg)
            detailed_rows.append(
                {
                    "scenario": "hidden_pole",
                    "plant_id": plant_id,
                    "controller": label,
                    "variant_id": variant_id,
                    **compute_metrics(hidden_res.traj),
                }
            )

        t = next(iter(nominal_runs.values())).t
        r_trace = np.array([nominal_scenario.reference(float(ti))[0] for ti in t], dtype=float)

        plt.figure(figsize=(9, 5.2))
        plt.plot(t, r_trace, "k--", linewidth=2.0, label="reference")
        for label, traj in nominal_runs.items():
            plt.plot(traj.t, traj.y[:, 0], linewidth=2.0, label=label)
        plt.title(f"{plant_id} | MLP variant comparison | output | nominal | {t_final:g}s")
        plt.xlabel("Time [s]")
        plt.ylabel("Output y")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / f"{plant_id}__output.png", dpi=160)
        plt.close()

        plt.figure(figsize=(9, 5.2))
        for label, traj in nominal_runs.items():
            plt.plot(traj.t, traj.u[:, 0], linewidth=2.0, label=label)
        plt.title(f"{plant_id} | MLP variant comparison | control | nominal | {t_final:g}s")
        plt.xlabel("Time [s]")
        plt.ylabel("Control u")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / f"{plant_id}__control.png", dpi=160)
        plt.close()

        print(f"Saved nominal MLP-variant plots and metrics for {plant_id}")

    _assign_group_ranks(detailed_rows)

    overall_rows = _aggregate_rows(detailed_rows, group_keys=("controller",), aggregate_metrics=METRIC_COLUMNS)
    overall_rows = sorted(overall_rows, key=lambda row: (float(row["mean_composite_rank"]), str(row["controller"])))

    scenario_rows = _aggregate_rows(detailed_rows, group_keys=("scenario", "controller"), aggregate_metrics=METRIC_COLUMNS)
    scenario_rows = sorted(
        scenario_rows,
        key=lambda row: (str(row["scenario"]), float(row["mean_composite_rank"]), str(row["controller"])),
    )
    failure_rows = _failure_breakdown_rows(detailed_rows)

    _write_csv(
        metrics_dir / "detailed_metrics.csv",
        detailed_rows,
        fieldnames=(
            "scenario",
            "plant_id",
            "controller",
            "variant_id",
            "mean_cost",
            "peak_error",
            "tail_rms_error",
            "tail_growth_ratio",
            "max_abs_u",
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
        ),
    )
    _write_csv(metrics_dir / "overall_summary.csv", overall_rows, fieldnames=tuple(overall_rows[0].keys()) if overall_rows else ("controller",))
    _write_csv(metrics_dir / "scenario_summary.csv", scenario_rows, fieldnames=tuple(scenario_rows[0].keys()) if scenario_rows else ("scenario", "controller"))
    _write_csv(
        metrics_dir / "failure_breakdown.csv",
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
        metrics_dir / "summary.md",
        study_cfg=study_cfg,
        overall_rows=overall_rows,
        scenario_rows=scenario_rows,
        failure_rows=failure_rows,
    )

    print("\nMetric guide:")
    for short_name, description in METRIC_EXPLANATIONS:
        print(f"- {short_name}: {description}")

    print("\nOverall summary:")
    print(
        _text_table(
            overall_rows,
            columns=(
                ("controller", "controller"),
                ("rank", "mean_composite_rank"),
                ("cost", "mean_mean_cost"),
                ("peak", "mean_peak_error"),
                ("tail_rms", "mean_tail_rms_error"),
                ("tail_g", "mean_tail_growth_ratio"),
                ("max_u", "mean_max_abs_u"),
                ("fails", "failure_count"),
            ),
        )
    )

    print("\nScenario summary:")
    print(
        _text_table(
            scenario_rows,
            columns=(
                ("scenario", "scenario"),
                ("controller", "controller"),
                ("rank", "mean_composite_rank"),
                ("cost", "mean_mean_cost"),
                ("tail_rms", "mean_tail_rms_error"),
                ("fails", "failure_count"),
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
                    ("late_escalation", "late_escalation_plants"),
                ),
            )
        )
    else:
        print("\nFailure breakdown:\nNo failures.")

    print(f"\nAll figures saved to: {figures_dir}")
    print(f"All metrics saved to: {metrics_dir}")


if __name__ == "__main__":
    main()
