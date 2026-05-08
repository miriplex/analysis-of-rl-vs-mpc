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
THIS_DIR = Path(__file__).resolve().parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from benchmark_inverted_pendulum_rl_variants_nominal import (
    _build_eval_plant,
    _experiment_cfg,
    _load_rl_controller,
)
from control_bench.config import INVERTED_PENDULUM_RENDER
from inverted_pendulum_rl_variants import variant_by_id


PLANT_KIND = "nonlinear"
VARIANT_ID = "pidfeat"
FPS = 30
REPLAY_T_FINAL = 5.0
OUT_DIR = EXPERIMENTS_ROOT / "results" / "figures" / "inverted_pendulum"
OUT_PATH = OUT_DIR / "pidfeat_nominal_failure_5s.mp4"
PREVIEW_PATH = OUT_DIR / "pidfeat_nominal_failure_5s_preview.png"


def _simulate() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    plant = _build_eval_plant(PLANT_KIND)
    _, exp_cfg = _experiment_cfg(PLANT_KIND)
    controller = _load_rl_controller(variant_by_id(VARIANT_ID))

    dt = float(plant.dt)
    n_steps = int(np.floor(REPLAY_T_FINAL / dt)) + 1
    t = np.arange(n_steps, dtype=float) * dt
    x_hist = np.zeros((n_steps, 4), dtype=float)
    u_hist = np.zeros((n_steps,), dtype=float)

    x0 = np.asarray(exp_cfg.x0, dtype=float)
    r = np.asarray(exp_cfg.reference, dtype=float).reshape(4)
    angle_limit = float(exp_cfg.angle_limit_rad)

    plant.reset(x0=x0)
    controller.reset()

    failure_time = float("nan")
    for k, tk in enumerate(t):
        obs = plant.observe()
        u_cmd = controller.step(r=r, obs=obs, t=float(tk))
        x_hist[k] = plant.state
        u_hist[k] = float(u_cmd[0])
        if np.isnan(failure_time) and abs(float(x_hist[k, 2])) > angle_limit:
            failure_time = float(tk)
        if k < n_steps - 1:
            plant.step(u_cmd)

    return t, x_hist, u_hist, failure_time


def _frame_indices(t: np.ndarray) -> np.ndarray:
    n_frames = int(round(REPLAY_T_FINAL * FPS)) + 1
    frame_times = np.linspace(0.0, REPLAY_T_FINAL, n_frames, dtype=float)
    return np.searchsorted(t, frame_times, side="left").clip(max=len(t) - 1)


def _set_frame(
    *,
    frame_idx: int,
    t: np.ndarray,
    x_hist: np.ndarray,
    u_hist: np.ndarray,
    failure_time: float,
    cart_patch,
    pole_line,
    bob_patch,
    time_text,
    state_text,
    failure_text,
    line_theta,
    dot_theta,
    line_cart,
    dot_cart,
    line_u,
    dot_u,
    visual_pole_length: float,
    max_cart_travel: float,
) -> None:
    tk = float(t[frame_idx])
    x = float(x_hist[frame_idx, 0])
    theta = float(x_hist[frame_idx, 2])
    theta_deg = float(np.rad2deg(theta))
    pivot_x = float(np.clip(x, -max_cart_travel, max_cart_travel))
    pivot_y = 0.0
    bob_x = pivot_x - visual_pole_length * np.sin(theta)
    bob_y = pivot_y + visual_pole_length * np.cos(theta)

    cart_patch.set_xy((pivot_x - 0.14, -0.08))
    pole_line.set_data([pivot_x, bob_x], [pivot_y, bob_y])
    bob_patch.center = (bob_x, bob_y)

    time_text.set_text(f"t = {tk:4.2f} s")
    state_text.set_text(
        f"x     = {x:+.2f} m\n"
        f"x_dot = {x_hist[frame_idx, 1]:+.2f} m/s\n"
        f"theta = {theta_deg:+.1f} deg\n"
        f"thdot = {x_hist[frame_idx, 3]:+.2f} rad/s\n"
        f"u     = {u_hist[frame_idx]:+.2f} N"
    )
    if np.isfinite(failure_time):
        if tk >= failure_time:
            failure_text.set_text(f"Nominal benchmark fails here: |theta| > 90° at t = {failure_time:.2f} s")
            failure_text.set_color("#b91c1c")
        else:
            failure_text.set_text(f"Nominal failure threshold: |theta| > 90° (crossed at {failure_time:.2f} s)")
            failure_text.set_color("#92400e")

    theta_deg_hist = np.rad2deg(x_hist[: frame_idx + 1, 2])
    line_theta.set_data(t[: frame_idx + 1], theta_deg_hist)
    dot_theta.set_data([tk], [theta_deg])
    line_cart.set_data(t[: frame_idx + 1], x_hist[: frame_idx + 1, 0])
    dot_cart.set_data([tk], [x])
    line_u.set_data(t[: frame_idx + 1], u_hist[: frame_idx + 1])
    dot_u.set_data([tk], [u_hist[frame_idx]])


def run() -> None:
    t, x_hist, u_hist, failure_time = _simulate()
    frame_indices = _frame_indices(t)

    visual_pole_length = 0.3 * float(INVERTED_PENDULUM_RENDER.visual_pole_scale)
    bob_radius = float(INVERTED_PENDULUM_RENDER.visual_bob_radius)
    track_half_width = 1.8
    max_cart_travel = track_half_width - 0.12

    fig = plt.figure(figsize=(12.8, 7.2))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 1.0])
    ax_anim = fig.add_subplot(grid[:, 0])
    ax_theta = fig.add_subplot(grid[0, 1])
    ax_cart = fig.add_subplot(grid[1, 1], sharex=ax_theta)
    ax_u = ax_cart.twinx()
    fig.subplots_adjust(left=0.055, right=0.97, top=0.86, bottom=0.1, wspace=0.24, hspace=0.32)

    ax_anim.set_xlim(-track_half_width, track_half_width)
    ax_anim.set_ylim(-visual_pole_length - 0.14, visual_pole_length + 0.14)
    ax_anim.set_aspect("equal")
    ax_anim.set_xticks([])
    ax_anim.set_yticks([])
    ax_anim.set_title("PID-features controller", fontsize=16, pad=10)
    ax_anim.axhline(0.0, color="0.25", linewidth=2.2)
    ax_anim.plot([0.0, 0.0], [0.0, visual_pole_length], "--", color="#d1d5db", linewidth=1.5)

    cart_patch = Rectangle(
        (-0.14, -0.08),
        0.28,
        0.16,
        facecolor="#4c6ef5",
        edgecolor="black",
        linewidth=1.2,
    )
    ax_anim.add_patch(cart_patch)
    pole_line, = ax_anim.plot([], [], color="#111827", linewidth=3.0)
    bob_patch = Circle(
        (0.0, visual_pole_length),
        radius=bob_radius,
        facecolor="#dc2626",
        edgecolor="black",
        linewidth=1.0,
    )
    ax_anim.add_patch(bob_patch)

    time_text = ax_anim.text(
        0.02,
        0.97,
        "",
        fontsize=14,
        weight="bold",
        va="top",
        transform=ax_anim.transAxes,
    )
    state_text = ax_anim.text(
        0.02,
        0.80,
        "",
        fontsize=10.5,
        family="monospace",
        va="top",
        transform=ax_anim.transAxes,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#d1d5db", alpha=0.92),
    )
    failure_text = ax_anim.text(
        0.02,
        0.02,
        "",
        fontsize=11,
        weight="bold",
        transform=ax_anim.transAxes,
    )

    theta_deg = np.rad2deg(x_hist[:, 2])
    line_theta, = ax_theta.plot([], [], color="#16a34a", linewidth=2.4, label="theta [deg]")
    dot_theta, = ax_theta.plot([], [], "o", color="#16a34a", markersize=6)
    line_cart, = ax_cart.plot([], [], color="#ea580c", linewidth=2.4, label="x [m]")
    dot_cart, = ax_cart.plot([], [], "o", color="#ea580c", markersize=6)
    line_u, = ax_u.plot([], [], color="#2563eb", linewidth=2.0, label="u [N]")
    dot_u, = ax_u.plot([], [], "o", color="#2563eb", markersize=5)

    for ax in (ax_theta, ax_cart):
        ax.grid(True, alpha=0.25)
        ax.axhline(0.0, color="0.75", linestyle="--", linewidth=1.0)
        ax.set_xlim(float(t[0]), float(t[-1]))
    ax_u.axhline(0.0, color="0.85", linestyle=":", linewidth=1.0)

    if np.isfinite(failure_time):
        for ax in (ax_theta, ax_cart, ax_u):
            ax.axvline(failure_time, color="#b91c1c", linestyle="--", linewidth=1.2, alpha=0.9)
        ax_theta.axhline(90.0, color="#b91c1c", linestyle=":", linewidth=1.0, alpha=0.8)
        ax_theta.axhline(-90.0, color="#b91c1c", linestyle=":", linewidth=1.0, alpha=0.8)

    ax_theta.set_title("Pole angle", fontsize=15, pad=8)
    ax_theta.set_ylabel("theta [deg]")
    ax_theta.set_ylim(float(theta_deg.min()) - 30.0, float(theta_deg.max()) + 30.0)
    ax_cart.set_title("Cart position and control", fontsize=15, pad=8)
    ax_cart.set_ylabel("x [m]")
    ax_cart.set_xlabel("Time [s]")
    ax_cart.set_ylim(float(x_hist[:, 0].min()) - 5.0, float(x_hist[:, 0].max()) + 5.0)
    ax_u.set_ylabel("u [N]", color="#2563eb")
    ax_u.tick_params(axis="y", colors="#2563eb")
    ax_u.set_ylim(float(u_hist.min()) - 5.0, float(u_hist.max()) + 5.0)

    fig.suptitle("PID-features failure on the nonlinear nominal pendulum", fontsize=19, y=0.965)
    fig.text(
        0.055,
        0.915,
        "5° initial lean, zero reference, nominal failure when |theta| > 90°",
        ha="left",
        va="top",
        fontsize=11,
        color="0.35",
    )

    writer = FFMpegWriter(
        fps=FPS,
        codec="libx264",
        bitrate=2600,
        metadata={"title": "PID-features nominal failure", "artist": "Codex"},
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-profile:v", "high", "-level:v", "4.0"],
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(OUT_PATH), dpi=150):
        for frame_idx in frame_indices:
            _set_frame(
                frame_idx=frame_idx,
                t=t,
                x_hist=x_hist,
                u_hist=u_hist,
                failure_time=failure_time,
                cart_patch=cart_patch,
                pole_line=pole_line,
                bob_patch=bob_patch,
                time_text=time_text,
                state_text=state_text,
                failure_text=failure_text,
                line_theta=line_theta,
                dot_theta=dot_theta,
                line_cart=line_cart,
                dot_cart=dot_cart,
                line_u=line_u,
                dot_u=dot_u,
                visual_pole_length=visual_pole_length,
                max_cart_travel=max_cart_travel,
            )
            writer.grab_frame()

    preview_frame = min(len(frame_indices) - 1, int(round(2.5 * FPS)))
    preview_idx = int(frame_indices[preview_frame])
    _set_frame(
        frame_idx=preview_idx,
        t=t,
        x_hist=x_hist,
        u_hist=u_hist,
        failure_time=failure_time,
        cart_patch=cart_patch,
        pole_line=pole_line,
        bob_patch=bob_patch,
        time_text=time_text,
        state_text=state_text,
        failure_text=failure_text,
        line_theta=line_theta,
        dot_theta=dot_theta,
        line_cart=line_cart,
        dot_cart=dot_cart,
        line_u=line_u,
        dot_u=dot_u,
        visual_pole_length=visual_pole_length,
        max_cart_travel=max_cart_travel,
    )
    fig.savefig(PREVIEW_PATH, dpi=120)
    plt.close(fig)

    print(f"Saved: {OUT_PATH}")
    print(f"Saved: {PREVIEW_PATH}")
    print(f"failure_time={failure_time:.3f}s")


if __name__ == "__main__":
    run()
