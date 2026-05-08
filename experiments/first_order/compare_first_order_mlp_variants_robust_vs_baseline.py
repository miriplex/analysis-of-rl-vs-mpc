from __future__ import annotations

import os
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
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

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
from control_bench.controllers.rl_numpy.controller import BackpropRLController
from control_bench.core.scenario_impl import BasicScenario
from control_bench.core.sim import simulate_closed_loop
from control_bench.core.types import SimConfig
from control_bench.experiment_manifest import (
    first_order_mlp_variants,
    first_order_robust_vs_baseline_study,
)
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


def _parse_env_list(name: str, *, allowed: Sequence[str]) -> list[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return list(allowed)
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    allowed_set = set(allowed)
    invalid = [item for item in requested if item not in allowed_set]
    if invalid:
        raise ValueError(f"{name} contains unknown entries: {invalid}")
    return requested


def step_reference(level: float = 1.0):
    def _r(t: float) -> np.ndarray:
        return np.array([level], dtype=float)

    return _r


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


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


def _build_mlp_controller(*, variant: dict, plant_id: str, weights_dir: Path, name_suffix: str) -> BackpropRLController:
    nominal = build_first_order_grid(family)[plant_id]
    weight_path = weights_dir / f"{variant['canonical_filename_prefix']}__{plant_id}.npz"
    require_file(weight_path)
    return BackpropRLController.load_npz(
        str(weight_path),
        kind="rich",
        dt=family.dt,
        u_bounds=None,
        u_min=family.u_min,
        u_max=family.u_max,
        mlp_hidden=tuple(int(v) for v in variant["hidden_layers"]),
        mlp_activation=str(variant["activation"]),
        rich_feature_set=str(variant["feature_set"]),
        name=f"{variant['label']} | {name_suffix}",
    )


def _paired_summary_rows(summary_rows: Sequence[dict], *, group_keys: Sequence[str]) -> list[dict]:
    grouped: Dict[Tuple[object, ...], Dict[str, dict]] = {}
    for row in summary_rows:
        key = tuple(row[group_key] for group_key in group_keys)
        grouped.setdefault(key, {})[str(row["training_track"])] = row

    paired_rows: List[dict] = []
    for key in sorted(grouped):
        by_track = grouped[key]
        baseline = by_track.get("baseline")
        robust = by_track.get("robust")
        if baseline is None or robust is None:
            continue

        row = {group_key: key[idx] for idx, group_key in enumerate(group_keys)}
        row["baseline_mean_cost"] = float(baseline["mean_mean_cost"])
        row["robust_mean_cost"] = float(robust["mean_mean_cost"])
        row["delta_mean_cost"] = float(robust["mean_mean_cost"] - baseline["mean_mean_cost"])
        row["baseline_tail_rms"] = float(baseline["mean_tail_rms_error"])
        row["robust_tail_rms"] = float(robust["mean_tail_rms_error"])
        row["delta_tail_rms"] = float(robust["mean_tail_rms_error"] - baseline["mean_tail_rms_error"])
        row["baseline_failures"] = int(baseline["failure_count"])
        row["robust_failures"] = int(robust["failure_count"])
        row["delta_failures"] = int(robust["failure_count"] - baseline["failure_count"])
        paired_rows.append(row)

    return paired_rows


def _write_summary_markdown(
    path: Path,
    *,
    study_cfg: dict,
    overall_rows: Sequence[dict],
    scenario_rows: Sequence[dict],
    failure_rows: Sequence[dict],
    paired_overall_rows: Sequence[dict],
    paired_scenario_rows: Sequence[dict],
) -> None:
    scenarios = dict(study_cfg["scenarios"])
    disturbed = dict(scenarios["disturbed"])
    hidden_pole = dict(scenarios["hidden_pole_mismatch"])
    content = [
        "# First-order MLP robust-vs-baseline comparison",
        "",
        f"Study script: `{study_cfg['script']}`",
        f"Nominal comparison horizon: `{float(study_cfg['t_final_seconds']):g} s`",
        f"Reference step: `{float(study_cfg['reference_step']):g}`",
        f"Robust results namespace: `{study_cfg['robust_results_namespace']}`",
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
        "## Overall summary",
        _markdown_table(
            overall_rows,
            columns=(
                "controller",
                "variant_id",
                "training_track",
                "mean_composite_rank",
                "mean_mean_cost",
                "mean_tail_rms_error",
                "failure_count",
            ),
        ),
        "",
        "## Paired overall deltas (robust - baseline)",
        _markdown_table(
            paired_overall_rows,
            columns=(
                "variant_id",
                "baseline_mean_cost",
                "robust_mean_cost",
                "delta_mean_cost",
                "baseline_tail_rms",
                "robust_tail_rms",
                "delta_tail_rms",
                "baseline_failures",
                "robust_failures",
                "delta_failures",
            ),
        ),
        "",
        "## Paired scenario deltas (robust - baseline)",
        _markdown_table(
            paired_scenario_rows,
            columns=(
                "scenario",
                "variant_id",
                "baseline_mean_cost",
                "robust_mean_cost",
                "delta_mean_cost",
                "baseline_tail_rms",
                "robust_tail_rms",
                "delta_tail_rms",
                "baseline_failures",
                "robust_failures",
                "delta_failures",
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
        "",
        "## Scenario summary",
        _markdown_table(
            scenario_rows,
            columns=(
                "scenario",
                "controller",
                "variant_id",
                "training_track",
                "mean_composite_rank",
                "mean_mean_cost",
                "mean_tail_rms_error",
                "failure_count",
            ),
        ),
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    study_cfg = first_order_robust_vs_baseline_study()
    all_variants = first_order_mlp_variants()

    plant_ids = _parse_env_list("FIRST_ORDER_ROBUST_COMPARE_PLANT_IDS", allowed=list(FIRST_ORDER_PLANT_IDS))
    variant_ids = _parse_env_list(
        "FIRST_ORDER_ROBUST_COMPARE_VARIANT_IDS",
        allowed=[str(variant["id"]) for variant in all_variants],
    )
    variants = [variant for variant in all_variants if str(variant["id"]) in set(variant_ids)]

    figures_dir = EXPERIMENTS_ROOT / "results" / "figures" / str(study_cfg["output_figures_namespace"])
    metrics_dir = EXPERIMENTS_ROOT / "results" / "metrics" / str(study_cfg["output_metrics_namespace"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    baseline_weights_dir = EXPERIMENTS_ROOT / "results" / "rl_numpy" / "per_plant"
    robust_weights_dir = EXPERIMENTS_ROOT / "results" / "rl_numpy" / str(study_cfg["robust_results_namespace"]) / "per_plant"

    t_final = float(study_cfg["t_final_seconds"])
    seed = 0
    r_step = float(study_cfg["reference_step"])
    scenarios_cfg = dict(study_cfg["scenarios"])

    detailed_rows: List[dict] = []

    for plant_id in plant_ids:
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

        for variant in variants:
            base_runs: Dict[str, object] = {}
            for training_track, weights_dir, name_suffix in (
                ("baseline", baseline_weights_dir, "baseline"),
                ("robust", robust_weights_dir, "robust"),
            ):
                controller_label = f"{variant['label']} | {name_suffix}"

                controller = _build_mlp_controller(
                    variant=variant,
                    plant_id=plant_id,
                    weights_dir=weights_dir,
                    name_suffix=name_suffix,
                )
                nominal_plant = build_first_order_grid(family)[plant_id]
                nominal_res = simulate_closed_loop(plant=nominal_plant, controller=controller, scenario=nominal_scenario, cfg=sim_cfg)
                detailed_rows.append(
                    {
                        "scenario": "nominal",
                        "plant_id": plant_id,
                        "controller": controller_label,
                        "variant_id": str(variant["id"]),
                        "variant_label": str(variant["label"]),
                        "training_track": training_track,
                        **compute_metrics(nominal_res.traj),
                    }
                )
                base_runs[controller_label] = nominal_res.traj

                controller = _build_mlp_controller(
                    variant=variant,
                    plant_id=plant_id,
                    weights_dir=weights_dir,
                    name_suffix=name_suffix,
                )
                disturbed_plant = build_first_order_grid(family)[plant_id]
                disturbed_res = simulate_closed_loop(plant=disturbed_plant, controller=controller, scenario=disturbed_scenario, cfg=sim_cfg)
                detailed_rows.append(
                    {
                        "scenario": "disturbed",
                        "plant_id": plant_id,
                        "controller": controller_label,
                        "variant_id": str(variant["id"]),
                        "variant_label": str(variant["label"]),
                        "training_track": training_track,
                        **compute_metrics(disturbed_res.traj),
                    }
                )

                controller = _build_mlp_controller(
                    variant=variant,
                    plant_id=plant_id,
                    weights_dir=weights_dir,
                    name_suffix=name_suffix,
                )
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
                        "controller": controller_label,
                        "variant_id": str(variant["id"]),
                        "variant_label": str(variant["label"]),
                        "training_track": training_track,
                        **compute_metrics(hidden_res.traj),
                    }
                )

            t = next(iter(base_runs.values())).t
            r_trace = np.array([nominal_scenario.reference(float(ti))[0] for ti in t], dtype=float)

            plt.figure(figsize=(9, 5.2))
            plt.plot(t, r_trace, "k--", linewidth=2.0, label="reference")
            for label, traj in base_runs.items():
                plt.plot(traj.t, traj.y[:, 0], linewidth=2.0, label=label)
            plt.title(f"{plant_id} | {variant['label']} | baseline vs robust | output | nominal | {t_final:g}s")
            plt.xlabel("Time [s]")
            plt.ylabel("Output y")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(figures_dir / f"{plant_id}__{variant['id']}__output.png", dpi=160)
            plt.close()

            plt.figure(figsize=(9, 5.2))
            for label, traj in base_runs.items():
                plt.plot(traj.t, traj.u[:, 0], linewidth=2.0, label=label)
            plt.title(f"{plant_id} | {variant['label']} | baseline vs robust | control | nominal | {t_final:g}s")
            plt.xlabel("Time [s]")
            plt.ylabel("Control u")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(figures_dir / f"{plant_id}__{variant['id']}__control.png", dpi=160)
            plt.close()

            print(f"Saved baseline-vs-robust nominal plots and metrics for {plant_id} | {variant['label']}")

    _assign_group_ranks(detailed_rows)

    overall_rows = _aggregate_rows(detailed_rows, group_keys=("variant_id", "variant_label", "training_track", "controller"), aggregate_metrics=METRIC_COLUMNS)
    overall_rows = sorted(
        overall_rows,
        key=lambda row: (str(row["variant_id"]), 0 if str(row["training_track"]) == "baseline" else 1, float(row["mean_composite_rank"])),
    )
    scenario_rows = _aggregate_rows(
        detailed_rows,
        group_keys=("scenario", "variant_id", "variant_label", "training_track", "controller"),
        aggregate_metrics=METRIC_COLUMNS,
    )
    scenario_rows = sorted(
        scenario_rows,
        key=lambda row: (
            str(row["scenario"]),
            str(row["variant_id"]),
            0 if str(row["training_track"]) == "baseline" else 1,
            float(row["mean_composite_rank"]),
        ),
    )
    failure_rows = _failure_breakdown_rows(detailed_rows)
    paired_overall_rows = _paired_summary_rows(overall_rows, group_keys=("variant_id", "variant_label"))
    paired_scenario_rows = _paired_summary_rows(scenario_rows, group_keys=("scenario", "variant_id", "variant_label"))

    _write_csv(
        metrics_dir / "detailed_metrics.csv",
        detailed_rows,
        fieldnames=(
            "scenario",
            "plant_id",
            "controller",
            "variant_id",
            "variant_label",
            "training_track",
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
    _write_csv(metrics_dir / "overall_summary.csv", overall_rows, fieldnames=tuple(overall_rows[0].keys()) if overall_rows else ("variant_id",))
    _write_csv(metrics_dir / "scenario_summary.csv", scenario_rows, fieldnames=tuple(scenario_rows[0].keys()) if scenario_rows else ("scenario", "variant_id"))
    _write_csv(metrics_dir / "paired_overall_summary.csv", paired_overall_rows, fieldnames=tuple(paired_overall_rows[0].keys()) if paired_overall_rows else ("variant_id",))
    _write_csv(metrics_dir / "paired_scenario_summary.csv", paired_scenario_rows, fieldnames=tuple(paired_scenario_rows[0].keys()) if paired_scenario_rows else ("scenario", "variant_id"))
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
        paired_overall_rows=paired_overall_rows,
        paired_scenario_rows=paired_scenario_rows,
    )

    print("\nMetric guide:")
    for short_name, description in METRIC_EXPLANATIONS:
        print(f"- {short_name}: {description}")

    print("\nOverall summary:")
    print(
        _text_table(
            overall_rows,
            columns=(
                ("variant", "variant_id"),
                ("track", "training_track"),
                ("rank", "mean_composite_rank"),
                ("cost", "mean_mean_cost"),
                ("tail_rms", "mean_tail_rms_error"),
                ("fails", "failure_count"),
            ),
        )
    )

    print("\nPaired overall deltas (robust - baseline):")
    print(
        _text_table(
            paired_overall_rows,
            columns=(
                ("variant", "variant_id"),
                ("d_cost", "delta_mean_cost"),
                ("d_tail", "delta_tail_rms"),
                ("d_fail", "delta_failures"),
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
    else:
        print("\nFailure breakdown:\nNo failures.")

    print(f"\nAll figures saved to: {figures_dir}")
    print(f"All metrics saved to: {metrics_dir}")


if __name__ == "__main__":
    main()
