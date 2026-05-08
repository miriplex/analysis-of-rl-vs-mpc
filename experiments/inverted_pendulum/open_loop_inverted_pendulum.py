from __future__ import annotations

import time
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from control_bench.config import (
    INVERTED_PENDULUM_LINEARIZED,
    INVERTED_PENDULUM_OPEN_LOOP,
    INVERTED_PENDULUM_RENDER,
)
from control_bench.plants.inverted_pendulum import (
    build_linearized_inverted_pendulum,
    continuous_linearized_matrices,
)


T_FINAL = INVERTED_PENDULUM_OPEN_LOOP.t_final
U_OPEN_LOOP = INVERTED_PENDULUM_OPEN_LOOP.u_open_loop
X0 = np.array(INVERTED_PENDULUM_OPEN_LOOP.x0, dtype=float)
REALTIME = INVERTED_PENDULUM_RENDER.realtime
HOLD_FINAL_FRAME = INVERTED_PENDULUM_RENDER.hold_final_frame
VISUAL_POLE_SCALE = INVERTED_PENDULUM_RENDER.visual_pole_scale
VISUAL_BOB_RADIUS = INVERTED_PENDULUM_RENDER.visual_bob_radius
ANGLE_VALIDITY_LIMIT_RAD = INVERTED_PENDULUM_OPEN_LOOP.angle_limit_rad


def _state_labels() -> tuple[str, str, str, str]:
    return ("x [m]", "x_dot [m/s]", "theta [rad]", "theta_dot [rad/s]")


def _format_matrix(A: np.ndarray) -> str:
    return np.array2string(A, precision=4, suppress_small=True)


def run() -> None:
    params = INVERTED_PENDULUM_LINEARIZED
    plant = build_linearized_inverted_pendulum(params)
    A_c, B_c, _, _ = continuous_linearized_matrices(params)
    eigvals_c = np.linalg.eigvals(A_c)
    unstable_rates = [float(val.real) for val in eigvals_c if float(val.real) > 1e-9]
    unstable_rate = max(unstable_rates) if unstable_rates else 0.0

    dt = float(plant.dt)
    n_steps = int(np.floor(T_FINAL / dt)) + 1
    t = np.arange(n_steps, dtype=float) * dt
    x_hist = np.zeros((n_steps, 4), dtype=float)

    plant.reset(x0=X0)

    fig = plt.figure(figsize=(13.5, 6.5))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.6], height_ratios=[1.0, 1.0])
    ax_anim = fig.add_subplot(grid[:, 0])
    ax_state_top = fig.add_subplot(grid[0, 1])
    ax_state_bottom = fig.add_subplot(grid[1, 1], sharex=ax_state_top)

    cart_width = 0.28
    cart_height = 0.16
    track_half_width = 1.4
    pole_length = float(params.pole_length)
    visual_pole_length = float(pole_length * VISUAL_POLE_SCALE)
    max_cart_travel = 1.2

    ax_anim.set_xlim(-track_half_width, track_half_width)
    ax_anim.set_ylim(-0.25, visual_pole_length + 0.15)
    ax_anim.set_aspect("equal")
    ax_anim.grid(True, alpha=0.25)
    ax_anim.set_title("Open-loop linearized inverted pendulum")
    ax_anim.axhline(0.0, color="0.35", linewidth=2.0)
    ax_anim.plot([0.0, 0.0], [0.0, visual_pole_length], "--", color="0.75", linewidth=1.5, label="upright")

    cart_patch = Rectangle(
        (-cart_width / 2.0, -cart_height / 2.0),
        cart_width,
        cart_height,
        facecolor="#5c7cfa",
        edgecolor="black",
        linewidth=1.2,
    )
    ax_anim.add_patch(cart_patch)
    pole_line, = ax_anim.plot([], [], color="#111827", linewidth=3.0)
    bob_patch = Circle(
        (0.0, visual_pole_length),
        radius=VISUAL_BOB_RADIUS,
        facecolor="#e03131",
        edgecolor="black",
        linewidth=1.0,
    )
    ax_anim.add_patch(bob_patch)
    text_x = -track_half_width + 0.06
    time_text = ax_anim.text(text_x, visual_pole_length + 0.07, "", fontsize=11)
    state_text = ax_anim.text(text_x, visual_pole_length - 0.12, "", fontsize=10, family="monospace")
    stop_text = ax_anim.text(text_x, -0.20, "", fontsize=10, color="#b91c1c")

    labels = _state_labels()
    state_lines_top = [
        ax_state_top.plot([], [], linewidth=2.0, label=labels[0])[0],
        ax_state_top.plot([], [], linewidth=2.0, label=labels[2])[0],
    ]
    state_lines_bottom = [
        ax_state_bottom.plot([], [], linewidth=2.0, label=labels[1])[0],
        ax_state_bottom.plot([], [], linewidth=2.0, label=labels[3])[0],
    ]

    ax_state_top.set_ylabel("Position / angle")
    ax_state_bottom.set_ylabel("Velocity")
    ax_state_bottom.set_xlabel("Time [s]")
    ax_state_top.grid(True, alpha=0.3)
    ax_state_bottom.grid(True, alpha=0.3)
    ax_state_top.legend(loc="upper right")
    ax_state_bottom.legend(loc="upper right")
    ax_state_top.set_title("State trajectories")

    info_lines = [
        "Linearized around upright origin",
        f"u(t) = {U_OPEN_LOOP:g} N",
        f"x0 = [{X0[0]:.3f}, {X0[1]:.3f}, {X0[2]:.3f}, {X0[3]:.3f}]",
        f"unstable pole ~= +{unstable_rate:.3f} 1/s",
        f"visual pole scale = {VISUAL_POLE_SCALE:.1f}x",
        f"stop when |theta| > {np.rad2deg(ANGLE_VALIDITY_LIMIT_RAD):.1f} deg",
        "",
        "A_c =",
        _format_matrix(A_c),
        "",
        "B_c =",
        _format_matrix(B_c),
    ]
    fig.text(0.015, 0.98, "\n".join(info_lines), va="top", ha="left", family="monospace", fontsize=9)

    plt.tight_layout(rect=[0.16, 0.0, 1.0, 1.0])
    plt.ion()
    plt.show(block=False)

    stop_reason = ""
    steps_completed = 0

    for k, tk in enumerate(t):
        x_hist[k] = plant.state
        steps_completed = k + 1

        cart_x = float(np.clip(x_hist[k, 0], -max_cart_travel, max_cart_travel))
        theta = float(x_hist[k, 2])
        pivot_x = cart_x
        pivot_y = 0.0
        bob_x = pivot_x - visual_pole_length * np.sin(theta)
        bob_y = pivot_y + visual_pole_length * np.cos(theta)

        cart_patch.set_xy((cart_x - cart_width / 2.0, -cart_height / 2.0))
        pole_line.set_data([pivot_x, bob_x], [pivot_y, bob_y])
        bob_patch.center = (bob_x, bob_y)

        time_text.set_text(f"t = {tk:5.2f} s")
        state_text.set_text(
            f"x     = {x_hist[k, 0]: .3f} m\n"
            f"x_dot = {x_hist[k, 1]: .3f} m/s\n"
            f"theta = {x_hist[k, 2]: .3f} rad\n"
            f"thdot = {x_hist[k, 3]: .3f} rad/s"
        )
        if stop_reason:
            stop_text.set_text(stop_reason)

        for line, idx in zip(state_lines_top, (0, 2)):
            line.set_data(t[: k + 1], x_hist[: k + 1, idx])
        for line, idx in zip(state_lines_bottom, (1, 3)):
            line.set_data(t[: k + 1], x_hist[: k + 1, idx])

        ax_state_top.relim()
        ax_state_top.autoscale_view()
        ax_state_bottom.relim()
        ax_state_bottom.autoscale_view()

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)

        if not np.all(np.isfinite(x_hist[k])):
            stop_reason = "Stopped: state became non-finite."
            break
        if abs(float(x_hist[k, 2])) > ANGLE_VALIDITY_LIMIT_RAD:
            stop_reason = (
                "Stopped: linearization no longer trustworthy "
                f"(|theta| > {np.rad2deg(ANGLE_VALIDITY_LIMIT_RAD):.1f} deg)."
            )
            break
        if k == n_steps - 1:
            continue

        plant.step(np.array([U_OPEN_LOOP], dtype=float))
        if REALTIME:
            time.sleep(dt)

    if not stop_reason:
        stop_reason = f"Completed full horizon: {steps_completed * dt:.2f} s"
    stop_text.set_text(stop_reason)

    out_dir = EXPERIMENTS_ROOT / "results" / "figures" / "inverted_pendulum"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "open_loop_linearized_inverted_pendulum.png"
    fig.savefig(out_path, dpi=160)
    print(f"Saved: {out_path}")
    print(stop_reason)

    if HOLD_FINAL_FRAME:
        plt.ioff()
        plt.show()
    else:
        plt.pause(0.5)
        plt.close(fig)


if __name__ == "__main__":
    run()
