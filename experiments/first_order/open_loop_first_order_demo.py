from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from control_bench.config import FIRST_ORDER_FAMILY, FIRST_ORDER_PLANT_IDS
from control_bench.plants.first_order_family import build_first_order_grid


POLE_ORDER = ("stable", "integrator", "unstable")
ZERO_ORDER = ("no_zero", "lhp_zero", "rhp_zero")


def _humanize_pole(pole_tag: str) -> str:
    if pole_tag == "stable":
        return "Stable"
    if pole_tag == "integrator":
        return "Integrator"
    if pole_tag == "unstable":
        return "Unstable"
    raise ValueError(f"Unknown pole tag: {pole_tag}")


def _humanize_zero(zero_tag: str) -> str:
    if zero_tag == "no_zero":
        return "No zero"
    if zero_tag == "lhp_zero":
        return "LHP zero"
    if zero_tag == "rhp_zero":
        return "RHP zero"
    raise ValueError(f"Unknown zero tag: {zero_tag}")


def _plant_pz_from_id(plant_id: str) -> tuple[float, Optional[float]]:
    pole_tag, zero_tag = plant_id.split("__", 1)

    if pole_tag == "stable":
        p = float(FIRST_ORDER_FAMILY.stable_p)
    elif pole_tag == "integrator":
        p = 0.0
    elif pole_tag == "unstable":
        p = float(FIRST_ORDER_FAMILY.unstable_p)
    else:
        raise ValueError(f"Unknown pole tag: {pole_tag}")

    if zero_tag == "no_zero":
        z = None
    elif zero_tag == "lhp_zero":
        z = float(FIRST_ORDER_FAMILY.lhp_z)
    elif zero_tag == "rhp_zero":
        z = float(FIRST_ORDER_FAMILY.rhp_z)
    else:
        raise ValueError(f"Unknown zero tag: {zero_tag}")

    return p, z


def _continuous_step_response(
    *, p: float, z: Optional[float], k: float, t_final: float, u_step: float, n_points: int = 1200
) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, float(t_final), int(n_points), dtype=float)
    step = float(u_step)

    if z is None:
        if abs(p) < 1e-12:
            y = k * step * t
        else:
            y = (k * step / p) * (np.exp(p * t) - 1.0)
        return t, y

    if abs(p) < 1e-12:
        y_plus = k * step - k * z * step * t[1:]
    else:
        y_plus = k * step + (k * (p - z) * step / p) * (np.exp(p * t[1:]) - 1.0)

    t_plot = np.concatenate(([0.0, 0.0], t[1:]))
    y_plot = np.concatenate(([0.0, k * step], y_plus))
    return t_plot, y_plot


def run() -> None:
    plants = build_first_order_grid(FIRST_ORDER_FAMILY)
    if set(plants.keys()) != set(FIRST_ORDER_PLANT_IDS):
        raise RuntimeError("Configured first-order plant ids do not match the plant family.")

    t_final = 4.0
    u_step = 1.0

    fig, axes = plt.subplots(3, 3, figsize=(15.0, 9.6))

    for row_idx, pole_tag in enumerate(POLE_ORDER):
        for col_idx, zero_tag in enumerate(ZERO_ORDER):
            plant_id = f"{pole_tag}__{zero_tag}"
            p, z = _plant_pz_from_id(plant_id)
            t, y = _continuous_step_response(
                p=p,
                z=z,
                k=FIRST_ORDER_FAMILY.k,
                t_final=t_final,
                u_step=u_step,
            )

            ax = axes[row_idx, col_idx]
            ax.plot(t, y, color="#202020", linewidth=2.2)
            ax.axhline(0.0, color="#bdbdbd", linewidth=1.0, linestyle="--")
            ax.grid(True, alpha=0.28)
            ax.set_xlim(float(t[0]), float(t[-1]))

            if row_idx == 0:
                zero_label = _humanize_zero(zero_tag)
                zero_value = "none" if z is None else f"{z:g}"
                ax.set_title(f"{zero_label}\n$z={zero_value}$", fontsize=14, pad=12)
            if col_idx == 0:
                ax.set_ylabel(f"{_humanize_pole(pole_tag)}\n$p={p:g}$\nOutput y", fontsize=12)
            if row_idx == len(POLE_ORDER) - 1:
                ax.set_xlabel("Time [s]", fontsize=12)

    fig.suptitle("Open-loop unit-step responses of the 9 first-order benchmark plants", fontsize=18, y=0.995)
    fig.tight_layout(rect=[0.03, 0.03, 1.0, 0.965], h_pad=2.2)

    out_dir = EXPERIMENTS_ROOT / "results" / "figures" / "first_order"
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "open_loop_first_order_suite.png"
    pdf_path = out_dir / "open_loop_first_order_suite.pdf"
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    run()
