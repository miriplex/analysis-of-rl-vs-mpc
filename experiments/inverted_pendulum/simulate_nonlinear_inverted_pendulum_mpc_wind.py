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
    INVERTED_PENDULUM_NONLINEAR_MPC_EXPERIMENT,
    INVERTED_PENDULUM_NONLINEAR_WIND_DISTURBANCE,
    INVERTED_PENDULUM_RENDER,
)
from control_bench.controllers.inverted_pendulum_mpc import (
    InvertedPendulumMPCConfig,
    InvertedPendulumMPCController,
)
from control_bench.plants.inverted_pendulum import build_linearized_inverted_pendulum
from control_bench.plants.inverted_pendulum_nonlinear import build_nonlinear_inverted_pendulum


T_FINAL = INVERTED_PENDULUM_NONLINEAR_WIND_DISTURBANCE.t_final
X0 = np.array(INVERTED_PENDULUM_NONLINEAR_WIND_DISTURBANCE.x0, dtype=float)
REFERENCE = np.array(INVERTED_PENDULUM_NONLINEAR_WIND_DISTURBANCE.reference, dtype=float)
REALTIME = INVERTED_PENDULUM_RENDER.realtime
HOLD_FINAL_FRAME = INVERTED_PENDULUM_RENDER.hold_final_frame
VISUAL_POLE_SCALE = INVERTED_PENDULUM_RENDER.visual_pole_scale
VISUAL_BOB_RADIUS = INVERTED_PENDULUM_RENDER.visual_bob_radius
ANGLE_FAILURE_LIMIT_RAD = INVERTED_PENDULUM_NONLINEAR_WIND_DISTURBANCE.angle_limit_rad
CART_FAILURE_LIMIT_M = INVERTED_PENDULUM_NONLINEAR_WIND_DISTURBANCE.cart_limit_m
CART_STOP_TEXT = "none" if not np.isfinite(CART_FAILURE_LIMIT_M) else f"{CART_FAILURE_LIMIT_M:.2f} m"
WIND_START_S = INVERTED_PENDULUM_NONLINEAR_WIND_DISTURBANCE.wind_start_s
WIND_DURATION_S = INVERTED_PENDULUM_NONLINEAR_WIND_DISTURBANCE.wind_duration_s
WIND_PEAK_FORCE = INVERTED_PENDULUM_NONLINEAR_WIND_DISTURBANCE.wind_peak_force
MPC_CFG = InvertedPendulumMPCConfig(
    N=20,
    q_state_diag=INVERTED_PENDULUM_NONLINEAR_MPC_EXPERIMENT.q_state_diag,
    ru=INVERTED_PENDULUM_NONLINEAR_MPC_EXPERIMENT.ru,
    qu=INVERTED_PENDULUM_NONLINEAR_MPC_EXPERIMENT.qu,
    use_terminal_cost=True,
    pgd_iters=INVERTED_PENDULUM_NONLINEAR_MPC_EXPERIMENT.pgd_iters,
    pgd_step_size=INVERTED_PENDULUM_NONLINEAR_MPC_EXPERIMENT.pgd_step_size,
)


def _wind_force(t: float) -> float:
    if t < WIND_START_S or t > WIND_START_S + WIND_DURATION_S:
        return 0.0
    phase = (t - WIND_START_S) / WIND_DURATION_S
    return float(WIND_PEAK_FORCE * np.sin(np.pi * phase))


def run() -> None:
    params = INVERTED_PENDULUM_NONLINEAR
    plant = build_nonlinear_inverted_pendulum(params)
    nominal = build_linearized_inverted_pendulum(params)
    controller = InvertedPendulumMPCController(
        MPC_CFG,
        dt=nominal.dt,
        A=nominal.A,
        B=nominal.B,
        u_bounds=plant.u_bounds,
        name="MPC",
    )

    dt = float(plant.dt)
    n_steps = int(np.floor(T_FINAL / dt)) + 1
    t = np.arange(n_steps, dtype=float) * dt
    x_hist = np.zeros((n_steps, 4), dtype=float)
    u_cmd_hist = np.zeros((n_steps,), dtype=float)
    d_hist = np.zeros((n_steps,), dtype=float)
    u_total_hist = np.zeros((n_steps,), dtype=float)

    plant.reset(x0=X0)
    controller.reset()

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
    ax_force = fig.add_subplot(grid[2, 1], sharex=ax_state)
    ax_info = fig.add_subplot(grid[2, 0])

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
        facecolor="#2b8a3e",
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
    gust_arrow = ax_anim.annotate(
        "",
        xy=(0.55, visual_pole_length * 0.78),
        xytext=(-0.10, visual_pole_length * 0.78),
        arrowprops=dict(arrowstyle="->", linewidth=2.5, color="#0f766e"),
    )
    gust_label = ax_anim.text(0.02, visual_pole_length * 0.83, "", fontsize=10, color="#0f766e")
    text_x = -track_half_width + 0.08
    time_text = ax_anim.text(text_x, visual_pole_length + 0.08, "", fontsize=11)
    state_text = ax_anim.text(text_x, visual_pole_length - 0.18, "", fontsize=10, family="monospace")
    stop_text = ax_anim.text(text_x, -0.35, "", fontsize=10, color="#b91c1c")

    state_line_x, = ax_state.plot([], [], linewidth=2.4, color="#1d4ed8", label="x [m]")
    state_line_theta, = ax_state.plot([], [], linewidth=2.4, color="#16a34a", label="theta [rad]")
    rate_line_xdot, = ax_rate.plot([], [], linewidth=2.4, color="#ea580c", label="x_dot [m/s]")
    rate_line_thetadot, = ax_rate.plot([], [], linewidth=2.4, color="#dc2626", label="theta_dot [rad/s]")
    cmd_line, = ax_force.plot([], [], linewidth=2.4, color="#991b1b", label="u_mpc [N]")
    dist_line, = ax_force.plot([], [], linewidth=2.4, color="#0f766e", linestyle="--", label="d_wind [N]")
    total_line, = ax_force.plot([], [], linewidth=2.4, color="#7c3aed", label="u_total [N]")

    for ax in (ax_state, ax_rate, ax_force):
        ax.grid(True, alpha=0.3)
        ax.axhline(0.0, color="0.7", linewidth=1.2, linestyle="--")

    ax_state.set_ylabel("Position / angle")
    ax_rate.set_ylabel("Velocity / rate")
    ax_force.set_ylabel("Force [N]")
    ax_force.set_xlabel("Time [s]")
    ax_state.legend(loc="upper right", ncol=2)
    ax_rate.legend(loc="upper right", ncol=2)
    ax_force.legend(loc="upper right", ncol=3)
    ax_state.set_title("Position and angle", fontsize=15, pad=8)
    ax_rate.set_title("Velocity and angular rate", fontsize=15, pad=8)
    ax_force.set_title("MPC force, wind, and applied total", fontsize=15, pad=8)

    ax_info.axis("off")
    info_lines = [
        "Setup",
        f"dt = {dt:.3f} s, horizon N = {MPC_CFG.N}, t_final = {T_FINAL:.1f} s",
        f"x0 = [{X0[0]:.3f}, {X0[1]:.3f}, {X0[2]:.3f}, {X0[3]:.3f}]",
        f"reference = [{REFERENCE[0]:.1f}, {REFERENCE[1]:.1f}, {REFERENCE[2]:.1f}, {REFERENCE[3]:.1f}]",
        "",
        "Real plant",
        "Nonlinear cart-pole",
        "",
        "Wind gust",
        f"start = {WIND_START_S:.2f} s, duration = {WIND_DURATION_S:.2f} s",
        f"peak force = {WIND_PEAK_FORCE:.2f} N",
        "",
        "Controller model",
        "Upright linearization inside MPC",
        f"Qx diag = {MPC_CFG.q_state_diag}",
        f"ru = {MPC_CFG.ru:g}, qu = {MPC_CFG.qu:g}",
        f"angle stop = {np.rad2deg(ANGLE_FAILURE_LIMIT_RAD):.1f} deg, cart stop = {CART_STOP_TEXT}",
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
    fig.suptitle("Nonlinear Inverted Pendulum with Linear MPC Under Wind Disturbance", fontsize=18, y=0.985)

    plt.ion()
    plt.show(block=False)

    stop_reason = ""
    steps_completed = 0

    for k, tk in enumerate(t):
        obs = plant.observe()
        u_cmd = controller.step(r=REFERENCE, obs=obs, t=tk)
        d_wind = _wind_force(tk)
        u_total = float(u_cmd[0]) + d_wind

        x_hist[k] = plant.state
        u_cmd_hist[k] = float(u_cmd[0])
        d_hist[k] = d_wind
        u_total_hist[k] = u_total
        steps_completed = k + 1

        pivot_x, pivot_y, bob_x_phys, bob_y_phys = plant.cart_pole_points()
        bob_x = pivot_x + VISUAL_POLE_SCALE * (bob_x_phys - pivot_x)
        bob_y = pivot_y + VISUAL_POLE_SCALE * (bob_y_phys - pivot_y)
        cart_x = float(np.clip(pivot_x, -max_cart_travel, max_cart_travel))

        cart_patch.set_xy((cart_x - cart_width / 2.0, -cart_height / 2.0))
        pole_line.set_data([cart_x, bob_x], [pivot_y, bob_y])
        bob_patch.center = (bob_x, bob_y)

        gust_mag = 0.16 + 0.34 * (abs(d_wind) / max(WIND_PEAK_FORCE, 1e-9))
        gust_arrow.set_visible(abs(d_wind) > 1e-6)
        if d_wind >= 0.0:
            gust_arrow.set_position((-gust_mag, visual_pole_length * 0.78))
            gust_arrow.xy = (gust_mag, visual_pole_length * 0.78)
        else:
            gust_arrow.set_position((gust_mag, visual_pole_length * 0.78))
            gust_arrow.xy = (-gust_mag, visual_pole_length * 0.78)
        gust_label.set_text("" if abs(d_wind) <= 1e-6 else f"wind = {d_wind:+.2f} N")

        time_text.set_text(f"t = {tk:5.2f} s")
        state_text.set_text(
            f"x      = {x_hist[k, 0]: .3f} m\n"
            f"x_dot  = {x_hist[k, 1]: .3f} m/s\n"
            f"theta  = {x_hist[k, 2]: .3f} rad\n"
            f"thdot  = {x_hist[k, 3]: .3f} rad/s\n"
            f"u_mpc  = {u_cmd_hist[k]: .3f} N\n"
            f"d_wind = {d_hist[k]: .3f} N\n"
            f"u_tot  = {u_total_hist[k]: .3f} N"
        )
        if stop_reason:
            stop_text.set_text(stop_reason)

        state_line_x.set_data(t[: k + 1], x_hist[: k + 1, 0])
        state_line_theta.set_data(t[: k + 1], x_hist[: k + 1, 2])
        rate_line_xdot.set_data(t[: k + 1], x_hist[: k + 1, 1])
        rate_line_thetadot.set_data(t[: k + 1], x_hist[: k + 1, 3])
        cmd_line.set_data(t[: k + 1], u_cmd_hist[: k + 1])
        dist_line.set_data(t[: k + 1], d_hist[: k + 1])
        total_line.set_data(t[: k + 1], u_total_hist[: k + 1])

        ax_state.relim()
        ax_state.autoscale_view()
        ax_rate.relim()
        ax_rate.autoscale_view()
        ax_force.relim()
        ax_force.autoscale_view()

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.001)

        if not np.all(np.isfinite(x_hist[k])) or not np.isfinite(u_total_hist[k]):
            stop_reason = "Stopped: state or force became non-finite."
            break
        if abs(float(x_hist[k, 2])) > ANGLE_FAILURE_LIMIT_RAD:
            stop_reason = (
                "Stopped: pendulum fell outside the local stabilization region "
                f"(|theta| > {np.rad2deg(ANGLE_FAILURE_LIMIT_RAD):.1f} deg)."
            )
            break
        if np.isfinite(CART_FAILURE_LIMIT_M) and abs(float(x_hist[k, 0])) > CART_FAILURE_LIMIT_M:
            stop_reason = (
                "Stopped: cart left the operating region "
                f"(|x| > {CART_FAILURE_LIMIT_M:.2f} m)."
            )
            break
        if k == n_steps - 1:
            continue

        plant.step(np.array([u_total], dtype=float))
        if REALTIME:
            time.sleep(dt)

    if not stop_reason:
        stop_reason = f"Completed full horizon: {steps_completed * dt:.2f} s"
    stop_text.set_text(stop_reason)

    out_dir = EXPERIMENTS_ROOT / "results" / "figures" / "inverted_pendulum"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "closed_loop_nonlinear_inverted_pendulum_mpc_wind.png"
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
