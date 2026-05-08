from __future__ import annotations

from pathlib import Path
import sys
from typing import Tuple
import zlib

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
SRC_ROOT = PROJECT_ROOT / "src"
FIRST_ORDER_ROOT = EXPERIMENTS_ROOT / "first_order"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(FIRST_ORDER_ROOT) not in sys.path:
    sys.path.insert(0, str(FIRST_ORDER_ROOT))

from benchmark_first_order_robustness_metrics import SEED, make_disturbance_u_fn_from_trace, make_measurement_fn_from_trace
from control_bench.experiment_manifest import first_order_measurement_noise_stress_test
from disturbance_stress_common import run_stress_test, stress_scale_values


CFG = first_order_measurement_noise_stress_test()
T_FINAL = float(CFG["t_final_seconds"])
R_STEP = float(CFG["reference_step"])
BASE_SIGMA_Y = float(CFG["base_measurement_noise_std"])
SCALE_SCHEDULE = tuple(CFG["stress_scale_schedule"])
FIGURES_DIR = EXPERIMENTS_ROOT / "results" / "figures" / "robustness_tests" / str(CFG["output_figures_namespace"])
METRICS_DIR = EXPERIMENTS_ROOT / "results" / "metrics" / "robustness_tests" / str(CFG["output_metrics_namespace"])


def _trace_builder(plant_id: str, n_steps: int, dt: float, stress_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    del dt
    seed_pid = (int(SEED) + int(zlib.crc32(f"meas::{plant_id}".encode("utf-8")))) % (2**32)
    rng = np.random.default_rng(seed_pid)
    base_eps_y = rng.normal(loc=0.0, scale=1.0, size=(n_steps,))
    noise_y = float(stress_scale) * float(BASE_SIGMA_Y) * base_eps_y
    du = np.zeros((n_steps,), dtype=float)
    return noise_y, du


def main() -> None:
    tested_scales = stress_scale_values(SCALE_SCHEDULE)
    run_stress_test(
        figures_dir=FIGURES_DIR,
        metrics_dir=METRICS_DIR,
        title="Measurement-noise stress test",
        summary_lines=(
            "Method:",
            "- Screen all controllers on the nominal first-order plant set.",
            "- Only controllers with zero nominal failures are eligible for the measurement-noise stress sweep.",
            "- Increase the measurement-noise standard deviation by a multiplier `x` while keeping input disturbance noise and step disturbance at zero.",
            f"- Base measurement noise standard deviation: `{BASE_SIGMA_Y:g}`",
        ),
        heatmap_xlabel="Measurement-noise multiplier on the baseline sensor-noise standard deviation",
        heatmap_title="Measurement-noise stress test | pass fraction across first-order plants",
        progress_label="measurement-noise stress",
        scenario_prefix="measurement_noise_scale",
        tested_scales=tested_scales,
        t_final=T_FINAL,
        r_step=R_STEP,
        trace_builder=_trace_builder,
        disturbance_builder=make_disturbance_u_fn_from_trace,
        measurement_builder=make_measurement_fn_from_trace,
    )


if __name__ == "__main__":
    main()
