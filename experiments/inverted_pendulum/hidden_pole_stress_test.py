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
    build_hidden_pole_chain,
    run_stress_test,
    simulate_case,
)


HIDDEN_POLE = 100.0
MAX_HIDDEN_ORDER = 5
TESTED_ORDERS = tuple(range(1, MAX_HIDDEN_ORDER + 1))
FIGURES_DIR = EXPERIMENTS_ROOT / "results" / "figures" / "inverted_pendulum_robustness" / "hidden_pole_stress"
METRICS_DIR = EXPERIMENTS_ROOT / "results" / "metrics" / "inverted_pendulum_robustness" / "hidden_pole_stress"


def _simulate_level(controller, case_label: str, x0: np.ndarray, level: float | int) -> dict:
    level = int(level)
    hidden_chain = None
    if level > 0:
        hidden_chain = build_hidden_pole_chain(hidden_pole=HIDDEN_POLE, hidden_order=level, dt=0.02)
    return simulate_case(
        controller=controller,
        x0=x0,
        measurement_noise=None,
        hidden_chain=hidden_chain,
    )


def main() -> None:
    run_stress_test(
        title="Inverted pendulum hidden-pole stress test",
        figures_dir=FIGURES_DIR,
        metrics_dir=METRICS_DIR,
        summary_lines=(
            "Method:",
            "- Use the nominally passing pendulum controllers from the nonlinear nominal screen.",
            "- Re-screen them on the 18-case local nonlinear stabilization grid.",
            f"- Add an unmodelled actuator lag chain `H(s) = (a / (s + a))^n` with `a = {HIDDEN_POLE:g} rad/s` between controller output and plant input.",
            "- Increase the hidden-pole order `n` while keeping the controller model unchanged.",
            "- Use absorbing elimination: once a controller fails an order, it is not tested at higher orders.",
        ),
        progression_key="hidden_order",
        tested_levels=TESTED_ORDERS,
        nominal_label="nominal",
        level_label_fn=lambda level: f"n={int(level)}",
        xlabel="Hidden actuator-lag mismatch order in (10 / (s + 10))^n",
        heatmap_title="Pendulum hidden-pole stress | pass fraction across local nonlinear cases",
        simulate_level=_simulate_level,
    )


if __name__ == "__main__":
    main()
