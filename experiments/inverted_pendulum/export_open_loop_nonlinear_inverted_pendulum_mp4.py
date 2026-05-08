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
OUT_PATH = OUT_DIR / "open_loop_nonlinear_inverted_pendulum_10s.mp4"


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
    cart_width = 0.28
    cart_height = 0.16
    max_cart_travel = 1.5

    fig = plt.figure(figsize=(12.8, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.0, 1.0])
    ax_anim = fig.add_subplot(grid[:, 0])
    ax_theta = fig.add_subplot(grid[0, 1])
    ax_cart = fig.add_subplot(grid[1, 1], sharex=ax_theta)

    ax_anim.set_xlim(-track_half_width, track_half_width)
    ax_anim.set_ylim(-visual_pole_length - 0.18, visual_pole_length + 0.18)
    ax_anim.set_aspect("equal")
    ax_anim.set_xticks([])
    ax_anim.set_yticks([])
    ax_anim.set_title("Open-loop nonlinear cart-pole", fontsize=18, pad=12)
    ax_anim.axhline(0.0, color="0.25", linewidth=2.2)
    ax_anim.plot([0.0, 0.0], [0.0, visual_pole_length], "--", color="0.80", linewidth=1.5)

    cart_patch = Rectangle(
        (-cart_width / 2.0, -cart_height / 2.0),
        cart_width,
        cart_height,
        facecolor="#4c6ef5",
        edgecolor="black",
        linewidth=1.2,
    )
    ax_anim.add_patch(cart_patch)
    pole_line, = ax_anim.plot([], [], color="#111827", linewidth=3.0)
    bob_patch = Circle(
        (0.0, visual_pole_length),
        radius=VISUAL_BOB_RADIUS,
        facecolor="#dc2626",
        edgecolor="black",
        linewidth=1.0,
    )
    ax_anim.add_patch(bob_patch)

    time_text = ax_anim.text(
        -track_half_width + 0.08,
        visual_pole_length + 0.08,
        "",
        fontsize=13,
        weight="bold",
    )
    state_text = ax_anim.text(
        -track_half_width + 0.08,
        visual_pole_length - 0.22,
        "",
        fontsize=11,
        family="monospace",
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#d1d5db", alpha=0.92),
    )

    theta_deg = np.rad2deg(x_hist[:, 2])
    line_theta, = ax_theta.plot([], [], color="#16a34a", linewidth=2.4)
    dot_theta, = ax_theta.plot([], [], "o", color="#16a34a", markersize=6)
    line_cart, = ax_cart.plot([], [], color="#ea580c", linewidth=2.4)
    dot_cart, = ax_cart.plot([], [], "o", color="#ea580c", markersize=6)

    for ax in (ax_theta, ax_cart):
        ax.grid(True, alpha=0.25)
        ax.axhline(0.0, color="0.75", linestyle="--", linewidth=1.0)
        ax.set_xlim(float(t[0]), float(t[-1]))

    ax_theta.set_title("Pole angle", fontsize=15, pad=8)
    ax_theta.set_ylabel("theta [deg]")
    ax_theta.set_ylim(float(theta_deg.min()) - 10.0, float(theta_deg.max()) + 10.0)
    ax_cart.set_title("Cart position", fontsize=15, pad=8)
    ax_cart.set_ylabel("x [m]")
    ax_cart.set_xlabel("Time [s]")
    ax_cart.set_ylim(float(x_hist[:, 0].min()) - 0.15, float(x_hist[:, 0].max()) + 0.15)

    info_lines = [
        "Nonlinear open-loop case",
        f"u(t) = {U_OPEN_LOOP:.1f} N",
        f"x0 = [0, 0, {np.rad2deg(X0[2]):.1f} deg, 0]",
        f"dt = {params.dt:.2f} s",
    ]
    fig.text(
        0.70,
        0.93,
        "\n".join(info_lines),
        ha="left",
        va="top",
        fontsize=10,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8fafc", edgecolor="#cbd5e1"),
    )
    fig.suptitle("Inverted pendulum open-loop response", fontsize=22, y=0.99)

    fps = int(round(1.0 / float(params.dt)))
    metadata = {"title": "Open-loop nonlinear inverted pendulum", "artist": "Codex"}
    writer = FFMpegWriter(
        fps=fps,
        metadata=metadata,
        bitrate=2400,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(OUT_PATH), dpi=160):
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
            state_text.set_text(
                f"x     = {x_hist[k, 0]: .3f} m\n"
                f"x_dot = {x_hist[k, 1]: .3f} m/s\n"
                f"theta = {np.rad2deg(x_hist[k, 2]): .1f} deg\n"
                f"thdot = {x_hist[k, 3]: .3f} rad/s"
            )

            line_theta.set_data(t[: k + 1], theta_deg[: k + 1])
            dot_theta.set_data([t[k]], [theta_deg[k]])
            line_cart.set_data(t[: k + 1], x_hist[: k + 1, 0])
            dot_cart.set_data([t[k]], [x_hist[k, 0]])

            writer.grab_frame()

    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    run()
