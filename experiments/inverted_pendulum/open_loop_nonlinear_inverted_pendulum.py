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
    INVERTED_PENDULUM_NONLINEAR,
    INVERTED_PENDULUM_NONLINEAR_OPEN_LOOP,
    INVERTED_PENDULUM_RENDER,
)
from control_bench.plants.inverted_pendulum_nonlinear import build_nonlinear_inverted_pendulum


T_FINAL = INVERTED_PENDULUM_NONLINEAR_OPEN_LOOP.t_final
U_OPEN_LOOP = INVERTED_PENDULUM_NONLINEAR_OPEN_LOOP.u_open_loop
X0 = np.array(INVERTED_PENDULUM_NONLINEAR_OPEN_LOOP.x0, dtype=float)
REALTIME = INVERTED_PENDULUM_RENDER.realtime
HOLD_FINAL_FRAME = INVERTED_PENDULUM_RENDER.hold_final_frame
VISUAL_POLE_SCALE = INVERTED_PENDULUM_RENDER.visual_pole_scale
VISUAL_BOB_RADIUS = INVERTED_PENDULUM_RENDER.visual_bob_radius
ANGLE_LIMIT_RAD = INVERTED_PENDULUM_NONLINEAR_OPEN_LOOP.angle_limit_rad


def run() -> None:
    params = INVERTED_PENDULUM_NONLINEAR
    plant = build_nonlinear_inverted_pendulum(params)

    dt = float(plant.dt)
    n_steps = int(np.floor(T_FINAL / dt)) + 1
    t = np.arange(n_steps, dtype=float) * dt
    x_hist = np.zeros((n_steps, 4), dtype=float)

    plant.reset(x0=X0)

    fig = plt.figure(figsize=(15.5, 8.5), constrained_layout=True)
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.05, 1.55],
        height_ratios=[1.0, 1.0, 0.65],
    )
    ax_anim = fig.add_subplot(grid[0:2, 0])
    ax_state = fig.add_subplot(grid[0, 1])
    ax_rate = fig.add_subplot(grid[1, 1], sharex=ax_state)
    ax_info = fig.add_subplot(grid[2, 0])
    ax_theta = fig.add_subplot(grid[2, 1], sharex=ax_state)

    cart_width = 0.28
    cart_height = 0.16
    track_half_width = 2.0
    visual_pole_length = float(params.pole_length * VISUAL_POLE_SCALE)
    max_cart_travel = 1.8

    ax_anim.set_xlim(-track_half_width, track_half_width)
    ax_anim.set_ylim(-0.45, visual_pole_length + 0.18)
    ax_anim.set_aspect("equal")
    ax_anim.grid(True, alpha=0.25)
    ax_anim.set_title("Animation", fontsize=15, pad=8)
    ax_anim.axhline(0.0, color="0.35", linewidth=2.0)
    ax_anim.plot([0.0, 0.0], [0.0, visual_pole_length], "--", color="0.75", linewidth=1.5)

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
    text_x = -track_half_width + 0.08
    time_text = ax_anim.text(text_x, visual_pole_length + 0.08, "", fontsize=11)
    state_text = ax_anim.text(text_x, visual_pole_length - 0.18, "", fontsize=10, family="monospace")
    stop_text = ax_anim.text(text_x, -0.35, "", fontsize=10, color="#b91c1c")

    line_x, = ax_state.plot([], [], linewidth=2.4, color="#1d4ed8", label="x [m]")
    line_theta, = ax_state.plot([], [], linewidth=2.4, color="#16a34a", label="theta [rad]")
    line_xdot, = ax_rate.plot([], [], linewidth=2.4, color="#ea580c", label="x_dot [m/s]")
    line_thetadot, = ax_rate.plot([], [], linewidth=2.4, color="#dc2626", label="theta_dot [rad/s]")
    line_theta_deg, = ax_theta.plot([], [], linewidth=2.4, color="#7c3aed", label="theta [deg]")

    for ax in (ax_state, ax_rate, ax_theta):
        ax.grid(True, alpha=0.3)
        ax.axhline(0.0, color="0.7", linewidth=1.2, linestyle="--")

    ax_state.set_ylabel("Position / angle")
    ax_rate.set_ylabel("Velocity / rate")
    ax_theta.set_ylabel("Angle [deg]")
    ax_theta.set_xlabel("Time [s]")
    ax_state.legend(loc="upper right", ncol=2)
    ax_rate.legend(loc="upper right", ncol=2)
    ax_theta.legend(loc="upper right")
    ax_state.set_title("Position and angle", fontsize=15, pad=8)
    ax_rate.set_title("Velocity and angular rate", fontsize=15, pad=8)
    ax_theta.set_title("Angle in degrees", fontsize=15, pad=8)

    ax_info.axis("off")
    limit_text = "no angle stop" if not np.isfinite(ANGLE_LIMIT_RAD) else f"angle stop = {np.rad2deg(ANGLE_LIMIT_RAD):.1f} deg"
    info_lines = [
        "Setup",
        f"dt = {dt:.3f} s, t_final = {T_FINAL:.1f} s",
        f"u_open_loop = {U_OPEN_LOOP:.2f} N",
        f"x0 = [{X0[0]:.3f}, {X0[1]:.3f}, {X0[2]:.3f}, {X0[3]:.3f}]",
        limit_text,
        "",
        "Physical params",
        f"M = {params.cart_mass:.3f} kg, m = {params.pole_mass:.3f} kg",
        f"l = {params.pole_length:.3f} m, I = {params.pole_inertia:.4f}",
        f"b = {params.cart_damping:.3f}, g = {params.gravity:.2f}",
        "",
        "Model",
        "Nonlinear cart-pole with the upright angle convention",
    ]
    ax_info.text(
        0.02,
        0.98,
        "\n".join(info_lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor="#cbd5e1"),
        transform=ax_info.transAxes,
    )
    fig.suptitle("Nonlinear Inverted Pendulum Open Loop", fontsize=18, y=0.985)

    plt.ion()
    plt.show(block=False)

    stop_reason = ""
    steps_completed = 0

    for k, tk in enumerate(t):
        x_hist[k] = plant.state
        steps_completed = k + 1

        pivot_x, pivot_y, bob_x_phys, bob_y_phys = plant.cart_pole_points()
        bob_x = pivot_x + VISUAL_POLE_SCALE * (bob_x_phys - pivot_x)
        bob_y = pivot_y + VISUAL_POLE_SCALE * (bob_y_phys - pivot_y)
        cart_x = float(np.clip(pivot_x, -max_cart_travel, max_cart_travel))

        cart_patch.set_xy((cart_x - cart_width / 2.0, -cart_height / 2.0))
        pole_line.set_data([cart_x, bob_x], [pivot_y, bob_y])
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

        line_x.set_data(t[: k + 1], x_hist[: k + 1, 0])
        line_theta.set_data(t[: k + 1], x_hist[: k + 1, 2])
        line_xdot.set_data(t[: k + 1], x_hist[: k + 1, 1])
        line_thetadot.set_data(t[: k + 1], x_hist[: k + 1, 3])
        line_theta_deg.set_data(t[: k + 1], np.rad2deg(x_hist[: k + 1, 2]))

        ax_state.relim()
        ax_state.autoscale_view()
        ax_rate.relim()
        ax_rate.autoscale_view()
        ax_theta.relim()
        ax_theta.autoscale_view()

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)

        if not np.all(np.isfinite(x_hist[k])):
            stop_reason = "Stopped: state became non-finite."
            break
        if np.isfinite(ANGLE_LIMIT_RAD) and abs(float(x_hist[k, 2])) > ANGLE_LIMIT_RAD:
            stop_reason = f"Stopped: |theta| > {np.rad2deg(ANGLE_LIMIT_RAD):.1f} deg."
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
    out_path = out_dir / "open_loop_nonlinear_inverted_pendulum.png"
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
