from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from control_bench.config import SECOND_ORDER_FAMILY, SECOND_ORDER_PLANT_IDS
from control_bench.plants.second_order_family import build_second_order_suite, tf_coeffs_for_plant_id


def _fmt_complex(z: complex) -> str:
    zc = complex(z)
    if abs(zc.imag) < 1e-10:
        return f"{zc.real:.3g}"
    sign = "+" if zc.imag >= 0 else "-"
    return f"{zc.real:.3g}{sign}{abs(zc.imag):.3g}j"


def _root_zero(b1: float, b0: float) -> Optional[complex]:
    if abs(float(b1)) < 1e-12:
        return None
    return complex(-float(b0) / float(b1))


def run() -> None:
    plants = build_second_order_suite(SECOND_ORDER_FAMILY)

    # We record y_k *before* applying the input at step k:
    #   y_k = plant.output()  (uses current x_k and last applied input u_{k-1})
    #   then apply u_k via plant.step(u_k)
    # This matches the repo’s closed-loop simulator convention.
    t_final = 10.0
    u_step = np.array([1.0], dtype=float)

    fig, axes = plt.subplots(5, 2, figsize=(12, 14), sharex=True)
    axes = axes.reshape(-1)

    for i, plant_id in enumerate(SECOND_ORDER_PLANT_IDS):
        plant = plants[plant_id]
        dt = float(plant.dt)
        n_steps = int(np.floor(t_final / dt)) + 1

        t = np.arange(n_steps, dtype=float) * dt
        y = np.zeros((n_steps,), dtype=float)

        plant.reset(x0=np.zeros((2,), dtype=float))

        for k in range(n_steps):
            y[k] = float(plant.output()[0])
            plant.step(u_step)

        b1, b0, a1, a0 = tf_coeffs_for_plant_id(plant_id, SECOND_ORDER_FAMILY)
        poles = np.roots([1.0, float(a1), float(a0)])
        zero = _root_zero(b1, b0)

        title = f"{plant_id}\n"
        title += f"poles: {_fmt_complex(poles[0])}, {_fmt_complex(poles[1])}"
        if zero is not None:
            title += f"\nzero: {_fmt_complex(zero)}"

        ax = axes[i]
        ax.plot(t, y, linewidth=2.0)
        ax.set_title(title)
        ax.grid(True)
        ax.set_ylabel("y")
        if i >= 8:
            ax.set_xlabel("Time [s]")

    fig.suptitle("Open-loop step responses (u = 1) | 2nd-order suite", y=0.995)
    plt.tight_layout()

    out_path = EXPERIMENTS_ROOT / "results" / "figures" / "second_order" / "open_loop_second_order.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    print(f"Saved: {out_path}")

    plt.show()


if __name__ == "__main__":
    run()

