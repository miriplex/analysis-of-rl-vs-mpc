from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle, Rectangle
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from control_bench.config import INVERTED_PENDULUM_NONLINEAR, INVERTED_PENDULUM_RENDER
from control_bench.plants.inverted_pendulum_nonlinear import build_nonlinear_inverted_pendulum


T_FINAL = 10.0
U_OPEN_LOOP = 0.0
X0 = np.array([0.0, 0.0, np.deg2rad(5.0), 0.0], dtype=float)
VISUAL_POLE_SCALE = INVERTED_PENDULUM_RENDER.visual_pole_scale
VISUAL_BOB_RADIUS = INVERTED_PENDULUM_RENDER.visual_bob_radius
OUT_DIR = EXPERIMENTS_ROOT / "results" / "figures" / "inverted_pendulum"
OUT_PATH = OUT_DIR / "open_loop_nonlinear_inverted_pendulum_10s_presentation.mp4"


def _simulate() -> tuple[np.ndarray, np.ndarray]:
    plant = build_nonlinear_inverted_pendulum(INVERTED_PENDULUM_NONLINEAR)
    plant.reset(x0=X0)

    dt = float(plant.dt)
    n_steps = int(np.floor(T_FINAL / dt)) + 1
    t = np.arange(n_steps, dtype=float) * dt
    x_hist = np.zeros((n_steps, 4), dtype=float)

    for k in range(n_steps):
        x_hist[k] = plant.state
        if k < n_steps - 1:
            plant.step(np.array([U_OPEN_LOOP], dtype=float))

    return t, x_hist


def run() -> None:
    t, x_hist = _simulate()
    params = INVERTED_PENDULUM_NONLINEAR

    visual_pole_length = float(params.pole_length * VISUAL_POLE_SCALE)
    track_half_width = 1.8
    cart_width = 0.30
    cart_height = 0.17
    max_cart_travel = 1.5

    fig, ax = plt.subplots(figsize=(12.8, 7.2), facecolor="white")
    fig.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.08)

    ax.set_xlim(-track_half_width, track_half_width)
    ax.set_ylim(-visual_pole_length - 0.22, visual_pole_length + 0.22)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.axhline(0.0, color="#3f3f46", linewidth=3.0, zorder=1)
    ax.plot([0.0, 0.0], [0.0, visual_pole_length], "--", color="#d4d4d8", linewidth=2.0, zorder=1)

    cart_patch = Rectangle(
        (-cart_width / 2.0, -cart_height / 2.0),
        cart_width,
        cart_height,
        facecolor="#3b82f6",
        edgecolor="#111827",
        linewidth=1.6,
        zorder=4,
    )
    ax.add_patch(cart_patch)
    pole_line, = ax.plot([], [], color="#0f172a", linewidth=4.0, zorder=5)
    bob_patch = Circle(
        (0.0, visual_pole_length),
        radius=VISUAL_BOB_RADIUS * 1.1,
        facecolor="#ef4444",
        edgecolor="#111827",
        linewidth=1.2,
        zorder=6,
    )
    ax.add_patch(bob_patch)

    title_text = ax.text(
        0.5,
        1.04,
        "Inverted Pendulum Open Loop",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=24,
        weight="bold",
        color="#111827",
    )
    subtitle_text = ax.text(
        0.5,
        1.00,
        "Nonlinear model, u(t) = 0 N, initial tilt = 5 deg",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=14,
        color="#4b5563",
    )
    time_text = ax.text(
        0.03,
        0.95,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        weight="bold",
        color="#111827",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#d1d5db", alpha=0.9),
    )

    fps = int(round(1.0 / float(params.dt)))
    metadata = {"title": "Open-loop nonlinear inverted pendulum presentation", "artist": "Codex"}
    writer = FFMpegWriter(fps=fps, metadata=metadata, bitrate=2200, codec="libx264")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(OUT_PATH), dpi=150):
        for k, tk in enumerate(t):
            cart_x_phys = float(x_hist[k, 0])
            theta = float(x_hist[k, 2])
            cart_x = float(np.clip(cart_x_phys, -max_cart_travel, max_cart_travel))
            pivot_x = cart_x
            pivot_y = 0.0
            bob_x = pivot_x - visual_pole_length * np.sin(theta)
            bob_y = pivot_y + visual_pole_length * np.cos(theta)

            cart_patch.set_xy((cart_x - cart_width / 2.0, -cart_height / 2.0))
            pole_line.set_data([pivot_x, bob_x], [pivot_y, bob_y])
            bob_patch.center = (bob_x, bob_y)
            time_text.set_text(f"t = {tk:4.2f} s")

            writer.grab_frame()

    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    run()
