from __future__ import annotations

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

from pendulum_stress_common import (
    measurement_noise_trace,
    run_stress_test,
    simulate_case,
)


BASE_MEASUREMENT_STD = np.array(
    [
        0.002,  # x [m]
        0.02,  # x_dot [m/s]
        np.deg2rad(0.25),  # theta [rad]
        np.deg2rad(1.0),  # theta_dot [rad/s]
    ],
    dtype=float,
)
TESTED_SCALES = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0)
FIGURES_DIR = EXPERIMENTS_ROOT / "results" / "figures" / "inverted_pendulum_robustness" / "measurement_noise_stress"
METRICS_DIR = EXPERIMENTS_ROOT / "results" / "metrics" / "inverted_pendulum_robustness" / "measurement_noise_stress"


def _simulate_level(controller, case_label: str, x0: np.ndarray, level: float | int) -> dict:
    level = float(level)
    dt = 0.02
    n_steps = int(np.floor(8.0 / dt)) + 1
    noise = None if level <= 0.0 else measurement_noise_trace(
        case_label=case_label,
        n_steps=n_steps,
        stress_scale=level,
        base_std=BASE_MEASUREMENT_STD,
    )
    return simulate_case(
        controller=controller,
        x0=x0,
        measurement_noise=noise,
        hidden_chain=None,
    )


def main() -> None:
    run_stress_test(
        title="Inverted pendulum measurement-noise stress test",
        figures_dir=FIGURES_DIR,
        metrics_dir=METRICS_DIR,
        summary_lines=(
            "Method:",
            "- Use the nominally passing pendulum controllers from the nonlinear nominal screen.",
            "- Re-screen them on the 18-case local nonlinear stabilization grid.",
            "- Apply additive measurement noise to the observed state only.",
            "- Use absorbing elimination: once a controller fails a stress level, it is not tested at higher levels.",
            f"- Base measurement-noise standard deviation: `{BASE_MEASUREMENT_STD.tolist()}` in `[x, x_dot, theta, theta_dot]` units.",
        ),
        progression_key="stress_scale",
        tested_levels=TESTED_SCALES,
        nominal_label="nominal",
        level_label_fn=lambda level: f"x{float(level):g}",
        xlabel="Measurement-noise multiplier on the baseline pendulum state-estimation noise",
        heatmap_title="Pendulum measurement-noise stress | pass fraction across local nonlinear cases",
        simulate_level=_simulate_level,
    )


if __name__ == "__main__":
    main()
