from __future__ import annotations

import csv
from dataclasses import dataclass
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
    _load_mpc_controller,
    _load_rl_controller,
)
from control_bench.config import INVERTED_PENDULUM_RENDER
from inverted_pendulum_rl_variants import VARIANT_METRICS_DIR, resolve_variants


PLANT_KIND = "nonlinear"
FPS = 30
REPLAY_T_FINAL = 10.0
OUT_DIR = EXPERIMENTS_ROOT / "results" / "figures" / "inverted_pendulum"
OUT_PATH = OUT_DIR / "nominal_nonlinear_controller_grid.mp4"
PREVIEW_PATH = OUT_DIR / "nominal_nonlinear_controller_grid_preview.png"


@dataclass(frozen=True)
class ControllerRun:
    variant_id: str
    display_name: str
    x_hist: np.ndarray
    x_interp: np.ndarray
    theta_interp: np.ndarray


def _summary_csv_path() -> Path:
    return VARIANT_METRICS_DIR / "nominal_screen" / "nominal_summary__inverted_pendulum_linearized__nonlinear.csv"


def _short_name(name: str) -> str:
    return name.replace("history8", "hist8")


def _load_passing_rows() -> list[dict]:
    csv_path = _summary_csv_path()
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [row for row in rows if row["pass_nominal"] == "True"]


def _simulate_controller(*, variant_id: str, display_name: str) -> np.ndarray:
    plant = _build_eval_plant(PLANT_KIND)
    _, exp_cfg = _experiment_cfg(PLANT_KIND)
    dt = float(plant.dt)
    t_final = float(REPLAY_T_FINAL)
    n_steps = int(np.floor(t_final / dt)) + 1
    x0 = np.asarray(exp_cfg.x0, dtype=float)
    r = np.asarray(exp_cfg.reference, dtype=float).reshape(4)

    variants = {variant.variant_id: variant for variant in resolve_variants()}
    if variant_id == "mpc":
        controller = _load_mpc_controller(PLANT_KIND)
    else:
        controller = _load_rl_controller(variants[variant_id])

    x_hist = np.zeros((n_steps, 4), dtype=float)
    plant.reset(x0=x0)
    controller.reset()

    for k in range(n_steps):
        obs = plant.observe()
        u_cmd = controller.step(r=r, obs=obs, t=k * dt)
        x_hist[k] = plant.state
        if k < n_steps - 1:
            plant.step(u_cmd)
    return x_hist


def _interp_series(x_hist: np.ndarray, *, dt: float, t_frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t_sim = np.arange(x_hist.shape[0], dtype=float) * dt
    x_interp = np.interp(t_frames, t_sim, x_hist[:, 0])
    theta_interp = np.interp(t_frames, t_sim, x_hist[:, 2])
    return x_interp, theta_interp


def _load_runs() -> tuple[list[ControllerRun], np.ndarray]:
    rows = _load_passing_rows()
    dt = float(_build_eval_plant(PLANT_KIND).dt)
    t_frames = np.arange(0.0, float(REPLAY_T_FINAL) + 1e-12, 1.0 / FPS, dtype=float)

    runs: list[ControllerRun] = []
    for row in rows:
        x_hist = _simulate_controller(variant_id=row["variant_id"], display_name=row["display_name"])
        x_interp, theta_interp = _interp_series(x_hist, dt=dt, t_frames=t_frames)
        runs.append(
            ControllerRun(
                variant_id=str(row["variant_id"]),
                display_name=_short_name(str(row["display_name"])),
                x_hist=x_hist,
                x_interp=x_interp,
                theta_interp=theta_interp,
            )
        )
    return runs, t_frames


def _set_frame(
    *,
    frame_idx: int,
    frame_time: float,
    runs: list[ControllerRun],
    panel_artists: list[dict],
    time_text,
    visual_pole_length: float,
) -> None:
    time_text.set_text(f"t = {frame_time:4.2f} s")
    for run, artists in zip(runs, panel_artists):
        x = float(run.x_interp[frame_idx])
        theta = float(run.theta_interp[frame_idx])
        pivot_x = x
        pivot_y = 0.0
        bob_x = pivot_x - visual_pole_length * np.sin(theta)
        bob_y = pivot_y + visual_pole_length * np.cos(theta)

        artists["cart"].set_xy((x - artists["cart_width"] / 2.0, -artists["cart_height"] / 2.0))
        artists["pole"].set_data([pivot_x, bob_x], [pivot_y, bob_y])
        artists["bob"].center = (bob_x, bob_y)
        artists["state"].set_text(f"x={x:+.2f} m\nθ={np.rad2deg(theta):+.1f}°")


def run() -> None:
    runs, t_frames = _load_runs()
    if len(runs) != 10:
        raise RuntimeError(f"Expected 10 passing controllers for the nonlinear nominal screen, got {len(runs)}.")

    all_x = np.concatenate([run.x_hist[:, 0] for run in runs])
    track_half_width = max(0.45, float(np.max(np.abs(all_x))) + 0.18)
    visual_pole_length = 0.3 * float(INVERTED_PENDULUM_RENDER.visual_pole_scale)
    bob_radius = float(INVERTED_PENDULUM_RENDER.visual_bob_radius)

    fig, axes = plt.subplots(2, 5, figsize=(19.2, 10.8), dpi=100)
    axes = axes.reshape(-1)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Nonlinear nominal inverted pendulum benchmark: controllers that pass the nominal screen",
        fontsize=24,
        y=0.985,
    )
    subtitle = fig.text(
        0.5,
        0.953,
        "5° initial lean, zero reference, nonlinear plant, 10 s replay; controllers selected from the nominal screen",
        ha="center",
        va="top",
        fontsize=13,
        color="#4b5563",
    )
    time_text = fig.text(
        0.5,
        0.922,
        "",
        ha="center",
        va="top",
        fontsize=18,
        weight="bold",
    )

    panel_artists: list[dict] = []
    for ax, run in zip(axes, runs):
        ax.set_xlim(-track_half_width, track_half_width)
        ax.set_ylim(-0.18, visual_pole_length + 0.14)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(run.display_name, fontsize=12, pad=8)
        ax.axhline(0.0, color="0.28", linewidth=1.8)
        ax.plot([0.0, 0.0], [0.0, visual_pole_length], "--", color="#d1d5db", linewidth=1.2)

        cart_width = 0.18
        cart_height = 0.10
        cart_patch = Rectangle(
            (-cart_width / 2.0, -cart_height / 2.0),
            cart_width,
            cart_height,
            facecolor="#4c6ef5",
            edgecolor="black",
            linewidth=1.0,
        )
        ax.add_patch(cart_patch)
        pole_line, = ax.plot([], [], color="#111827", linewidth=2.4)
        bob_patch = Circle(
            (0.0, visual_pole_length),
            radius=bob_radius,
            facecolor="#dc2626",
            edgecolor="black",
            linewidth=0.9,
        )
        ax.add_patch(bob_patch)
        state_text = ax.text(
            0.03,
            0.05,
            "",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.8,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#d1d5db", alpha=0.92),
        )
        panel_artists.append(
            {
                "cart": cart_patch,
                "pole": pole_line,
                "bob": bob_patch,
                "state": state_text,
                "cart_width": cart_width,
                "cart_height": cart_height,
            }
        )

    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.89], w_pad=1.6, h_pad=1.8)

    writer = FFMpegWriter(
        fps=FPS,
        codec="libx264",
        bitrate=2800,
        metadata={"title": "Nominal nonlinear inverted pendulum controller grid", "artist": "Codex"},
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-profile:v", "high", "-level:v", "4.0"],
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(OUT_PATH), dpi=100):
        for frame_idx, frame_time in enumerate(t_frames):
            _set_frame(
                frame_idx=frame_idx,
                frame_time=float(frame_time),
                runs=runs,
                panel_artists=panel_artists,
                time_text=time_text,
                visual_pole_length=visual_pole_length,
            )
            writer.grab_frame()

    preview_idx = len(t_frames) // 2
    _set_frame(
        frame_idx=preview_idx,
        frame_time=float(t_frames[preview_idx]),
        runs=runs,
        panel_artists=panel_artists,
        time_text=time_text,
        visual_pole_length=visual_pole_length,
    )
    fig.savefig(PREVIEW_PATH, dpi=120)
    plt.close(fig)

    print(f"Saved: {OUT_PATH}")
    print(f"Saved: {PREVIEW_PATH}")


if __name__ == "__main__":
    run()
