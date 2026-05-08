from __future__ import annotations

from pathlib import Path
import sys
from typing import Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
SRC_ROOT = PROJECT_ROOT / "src"
FIRST_ORDER_ROOT = EXPERIMENTS_ROOT / "first_order"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(FIRST_ORDER_ROOT) not in sys.path:
    sys.path.insert(0, str(FIRST_ORDER_ROOT))

from benchmark_first_order_robustness_metrics import make_disturbance_u_fn_from_trace, make_measurement_fn_from_trace
from control_bench.experiment_manifest import first_order_step_disturbance_stress_test
from disturbance_stress_common import run_stress_test, stress_scale_values


CFG = first_order_step_disturbance_stress_test()
T_FINAL = float(CFG["t_final_seconds"])
R_STEP = float(CFG["reference_step"])
STEP_TIME_SECONDS = float(CFG["disturbance_step_time_seconds"])
BASE_STEP_MAGNITUDE = float(CFG["base_disturbance_step_magnitude"])
SCALE_SCHEDULE = tuple(CFG["stress_scale_schedule"])
FIGURES_DIR = EXPERIMENTS_ROOT / "results" / "figures" / "robustness_tests" / str(CFG["output_figures_namespace"])
METRICS_DIR = EXPERIMENTS_ROOT / "results" / "metrics" / "robustness_tests" / str(CFG["output_metrics_namespace"])


def _trace_builder(plant_id: str, n_steps: int, dt: float, stress_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    del plant_id
    noise_y = np.zeros((n_steps,), dtype=float)
    step_on = (np.arange(n_steps, dtype=float) * float(dt)) >= float(STEP_TIME_SECONDS)
    du = float(stress_scale) * float(BASE_STEP_MAGNITUDE) * step_on.astype(float)
    return noise_y, du


def main() -> None:
    tested_scales = stress_scale_values(SCALE_SCHEDULE)
    run_stress_test(
        figures_dir=FIGURES_DIR,
        metrics_dir=METRICS_DIR,
        title="Step-disturbance stress test",
        summary_lines=(
            "Method:",
            "- Screen all controllers on the nominal first-order plant set.",
            "- Only controllers with zero nominal failures are eligible for the step-disturbance stress sweep.",
            "- Increase the disturbance step amplitude by a multiplier `x` while keeping measurement noise and input disturbance noise at zero.",
            f"- Base disturbance step magnitude: `{BASE_STEP_MAGNITUDE:g}` at `t = {STEP_TIME_SECONDS:g} s`",
        ),
        heatmap_xlabel="Step-disturbance multiplier on the baseline disturbance step amplitude",
        heatmap_title="Step-disturbance stress test | pass fraction across first-order plants",
        progress_label="step-disturbance stress",
        scenario_prefix="step_disturbance_scale",
        tested_scales=tested_scales,
        t_final=T_FINAL,
        r_step=R_STEP,
        trace_builder=_trace_builder,
        disturbance_builder=make_disturbance_u_fn_from_trace,
        measurement_builder=make_measurement_fn_from_trace,
    )


if __name__ == "__main__":
    main()
