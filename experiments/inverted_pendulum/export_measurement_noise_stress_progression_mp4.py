from __future__ import annotations

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

from control_bench.config import INVERTED_PENDULUM_NONLINEAR, INVERTED_PENDULUM_RENDER
from control_bench.plants.inverted_pendulum_nonlinear import build_nonlinear_inverted_pendulum
from measurement_noise_stress_test import BASE_MEASUREMENT_STD, TESTED_SCALES
from pendulum_stress_common import (
    ANGLE_LIMIT_DEG,
    LOCAL_SETTLE_RATE_NORM,
    LOCAL_SETTLE_THETA_DEG,
    LOCAL_SETTLE_X_M,
    REFERENCE,
    T_FINAL,
    load_controller,
    local_case_grid,
    measurement_noise_trace,
    passing_controller_specs,
)


FPS = 30
STAGE_DURATION_S = 10.0
BENCHMARK_T_FINAL = float(T_FINAL)
TRACK_HALF_WIDTH = 0.75
OUT_DIR = EXPERIMENTS_ROOT / "results" / "figures" / "inverted_pendulum"
OUT_PATH = OUT_DIR / "measurement_noise_stress_progression_grid.mp4"
PREVIEW_PATH = OUT_DIR / "measurement_noise_stress_progression_grid_preview.png"
DISPLAYED_SCALES = (0.5, 2.0, 4.0, 5.0, 6.0)


@dataclass(frozen=True)
class CaseRun:
    case_label: str
    stress_scale: float
    pass_case: bool
    failure_reason: str
    failure_time_s: float
    final_state_norm: float
    max_abs_theta_deg: float
    max_abs_x_m: float
    t_used: np.ndarray
    x_used: np.ndarray


@dataclass(frozen=True)
class StagePanel:
    variant_id: str
    display_name: str
    case_label: str
    pass_stage: bool
    fail_trigger_s: float
    failure_reason: str
    freeze_frame_idx: int
    x_frames: np.ndarray
    theta_frames: np.ndarray


@dataclass(frozen=True)
class StageData:
    stress_scale: float
    panels_by_variant: dict[str, StagePanel]


def _short_name(name: str) -> str:
    return name.replace("history8", "hist8")


def _pretty_case_label(case_label: str) -> str:
    x_part, theta_part = case_label.split("__")
    x0 = x_part.replace("x", "")
    theta0 = theta_part.replace("theta", "")
    return f"x0={x0} m, θ0={theta0}°"


def _pretty_failure_reason(reason: str) -> str:
    if not reason or reason == "--":
        return ""
    labels = {
        "angle_limit": "angle limit",
        "nonfinite_observation": "non-finite observation",
        "nonfinite_noisy_observation": "non-finite noisy observation",
        "nonfinite_control": "non-finite control",
        "nonfinite_next_state": "non-finite next state",
        "final_theta_not_settled": "final angle not settled",
        "final_x_not_settled": "final position not settled",
        "final_rate_not_settled": "final rate not settled",
    }
    parts = [labels.get(token.strip(), token.strip().replace("_", " ")) for token in reason.split(",") if token.strip()]
    return "\n".join(parts)


def _simulate_case_full(*, controller, x0: np.ndarray, case_label: str, stress_scale: float) -> CaseRun:
    plant = build_nonlinear_inverted_pendulum(INVERTED_PENDULUM_NONLINEAR)
    dt = float(plant.dt)
    n_steps = int(np.floor(BENCHMARK_T_FINAL / dt)) + 1
    t = np.arange(n_steps, dtype=float) * dt
    angle_limit_rad = np.deg2rad(ANGLE_LIMIT_DEG)
    noise = measurement_noise_trace(
        case_label=case_label,
        n_steps=n_steps,
        stress_scale=float(stress_scale),
        base_std=BASE_MEASUREMENT_STD,
    )

    plant.reset(x0=x0)
    controller.reset()

    x_hist = np.zeros((n_steps, 4), dtype=float)
    failure_reason = "--"
    failure_time_s = float("nan")
    steps_completed = 0
    hard_stop = False

    for k, tk in enumerate(t):
        obs = plant.observe()
        if not np.all(np.isfinite(obs)):
            failure_reason = "nonfinite_observation"
            failure_time_s = float(tk)
            hard_stop = True
            break

        obs_ctrl = obs + noise[k]
        if not np.all(np.isfinite(obs_ctrl)):
            failure_reason = "nonfinite_noisy_observation"
            failure_time_s = float(tk)
            hard_stop = True
            break

        u_cmd = controller.step(r=REFERENCE, obs=obs_ctrl, t=float(tk))
        if not np.all(np.isfinite(u_cmd)):
            failure_reason = "nonfinite_control"
            failure_time_s = float(tk)
            hard_stop = True
            break

        x_hist[k] = plant.state
        steps_completed = k + 1

        theta = float(x_hist[k, 2])
        if np.isnan(failure_time_s) and abs(theta) > angle_limit_rad:
            failure_reason = "angle_limit"
            failure_time_s = float(tk)

        if k < n_steps - 1:
            plant.step(u_cmd)
            if not np.all(np.isfinite(plant.state)):
                if np.isnan(failure_time_s):
                    failure_reason = "nonfinite_next_state"
                    failure_time_s = float(tk + dt)
                hard_stop = True
                break

    used_len = max(steps_completed, 1)
    x_used = x_hist[:used_len]
    t_used = t[:used_len]
    if hard_stop and used_len == 1 and steps_completed == 0:
        x_used[0] = np.asarray(x0, dtype=float)

    final_state = x_used[-1]
    survive_horizon = bool(np.isnan(failure_time_s) and steps_completed == n_steps)
    settle_failures: list[str] = []
    if abs(float(final_state[2])) > np.deg2rad(LOCAL_SETTLE_THETA_DEG):
        settle_failures.append("final_theta_not_settled")
    if abs(float(final_state[0])) > LOCAL_SETTLE_X_M:
        settle_failures.append("final_x_not_settled")
    if float(np.linalg.norm(final_state[[1, 3]])) > LOCAL_SETTLE_RATE_NORM:
        settle_failures.append("final_rate_not_settled")

    pass_case = bool(survive_horizon and not settle_failures)
    if survive_horizon and settle_failures:
        failure_reason = ",".join(settle_failures)

    return CaseRun(
        case_label=case_label,
        stress_scale=float(stress_scale),
        pass_case=pass_case,
        failure_reason=failure_reason,
        failure_time_s=float(failure_time_s),
        final_state_norm=float(np.linalg.norm(final_state)),
        max_abs_theta_deg=float(np.rad2deg(np.max(np.abs(x_used[:, 2])))),
        max_abs_x_m=float(np.max(np.abs(x_used[:, 0]))),
        t_used=t_used,
        x_used=x_used,
    )


def _choose_representative(case_runs: list[CaseRun]) -> CaseRun:
    failed = [run for run in case_runs if not run.pass_case]
    if failed:
        catastrophic = [
            run
            for run in failed
            if np.isfinite(run.failure_time_s)
            and any(token in run.failure_reason for token in ("angle_limit", "nonfinite"))
        ]
        if catastrophic:
            catastrophic.sort(key=lambda run: (run.failure_time_s, -run.max_abs_theta_deg, -run.final_state_norm))
            return catastrophic[0]
        return max(
            failed,
            key=lambda run: (run.final_state_norm, run.max_abs_theta_deg, run.max_abs_x_m),
        )
    return max(
        case_runs,
        key=lambda run: (run.final_state_norm, run.max_abs_theta_deg, run.max_abs_x_m),
    )


def _interp_display(run: CaseRun, stage_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clamped_t = np.minimum(stage_times, BENCHMARK_T_FINAL)
    x_frames = np.interp(clamped_t, run.t_used, run.x_used[:, 0])
    theta_frames = np.interp(clamped_t, run.t_used, run.x_used[:, 2])
    return x_frames, theta_frames


def _build_stage_data() -> tuple[list[StageData], list, list[str]]:
    stage_times = np.arange(int(round(STAGE_DURATION_S * FPS)), dtype=float) / FPS
    specs = list(passing_controller_specs())
    controllers = {spec.variant_id: load_controller(spec) for spec in specs}
    survivors = specs[:]
    cases = local_case_grid()
    stages: list[StageData] = []
    logs: list[str] = []

    for stress_scale in DISPLAYED_SCALES:
        if not survivors:
            break
        stage_panels: dict[str, StagePanel] = {}
        next_survivors: list = []
        logs.append(f"x={float(stress_scale):g}: {len(survivors)} entering")

        for spec in survivors:
            controller = controllers[spec.variant_id]
            case_runs = [
                _simulate_case_full(
                    controller=controller,
                    x0=x0,
                    case_label=case_label,
                    stress_scale=float(stress_scale),
                )
                for case_label, x0 in cases
            ]
            pass_stage = all(run.pass_case for run in case_runs)
            rep = _choose_representative(case_runs)
            x_frames, theta_frames = _interp_display(rep, stage_times)

            if pass_stage:
                fail_trigger_s = float("inf")
                freeze_frame_idx = len(stage_times) - 1
                next_survivors.append(spec)
            elif np.isfinite(rep.failure_time_s) and any(
                token in rep.failure_reason for token in ("angle_limit", "nonfinite")
            ):
                fail_trigger_s = float(rep.failure_time_s)
                freeze_frame_idx = min(len(stage_times) - 1, int(np.searchsorted(stage_times, fail_trigger_s, side="left")))
            else:
                fail_trigger_s = BENCHMARK_T_FINAL
                freeze_frame_idx = min(
                    len(stage_times) - 1,
                    int(np.searchsorted(stage_times, BENCHMARK_T_FINAL, side="left")),
                )

            stage_panels[spec.variant_id] = StagePanel(
                variant_id=spec.variant_id,
                display_name=_short_name(spec.display_name),
                case_label=_pretty_case_label(rep.case_label),
                pass_stage=pass_stage,
                fail_trigger_s=fail_trigger_s,
                failure_reason=_pretty_failure_reason(rep.failure_reason),
                freeze_frame_idx=freeze_frame_idx,
                x_frames=x_frames,
                theta_frames=theta_frames,
            )
            logs.append(
                f"  {spec.display_name}: {'pass' if pass_stage else 'fail'} "
                f"(rep {rep.case_label}; reason={rep.failure_reason or '--'})"
            )

        stages.append(StageData(stress_scale=float(stress_scale), panels_by_variant=stage_panels))
        survivors = next_survivors

    return stages, specs, logs


def _configure_stage(*, stage: StageData, slot_specs: list, axes: np.ndarray, panel_artists: list[dict]) -> None:
    active_variant_ids = set(stage.panels_by_variant.keys())

    for ax, artists, spec in zip(axes, panel_artists, slot_specs):
        ax.set_visible(True)
        ax.set_title(_short_name(spec.display_name), fontsize=12, pad=8)
        ax.set_xlim(-TRACK_HALF_WIDTH, TRACK_HALF_WIDTH)
        ax.set_ylim(-0.18, artists["visual_pole_length"] + 0.14)
        ax.set_xticks([])
        ax.set_yticks([])
        panel = stage.panels_by_variant.get(spec.variant_id)

        artists["case"].set_visible(False)
        artists["fail"].set_visible(False)
        artists["eliminated"].set_visible(False)

        if panel is None:
            for spine in ax.spines.values():
                spine.set_linewidth(1.3)
                spine.set_color("#e5e7eb")
            artists["cart"].set_visible(False)
            artists["pole"].set_visible(False)
            artists["bob"].set_visible(False)
            artists["eliminated"].set_text("eliminated")
            artists["eliminated"].set_visible(True)
            continue

        for spine in ax.spines.values():
            spine.set_linewidth(1.6)
            spine.set_color("#cbd5e1")
        artists["case"].set_text(panel.case_label)
        artists["case"].set_visible(True)
        artists["cart"].set_visible(True)
        artists["pole"].set_visible(True)
        artists["bob"].set_visible(True)


def _set_stage_frame(
    *,
    stage: StageData,
    slot_specs: list,
    local_frame_idx: int,
    local_time_s: float,
    axes: np.ndarray,
    panel_artists: list[dict],
    time_text,
) -> None:
    time_text.set_text(f"noise scale x = {stage.stress_scale:g} | stage t = {local_time_s:4.2f} s")

    for ax, artists, spec in zip(axes, panel_artists, slot_specs):
        panel = stage.panels_by_variant.get(spec.variant_id)
        if panel is None:
            continue

        frame_idx = local_frame_idx
        failed_now = (not panel.pass_stage) and (local_time_s >= panel.fail_trigger_s)
        if failed_now:
            frame_idx = panel.freeze_frame_idx

        x = float(panel.x_frames[frame_idx])
        theta = float(panel.theta_frames[frame_idx])
        pivot_x = float(np.clip(x, -TRACK_HALF_WIDTH + 0.08, TRACK_HALF_WIDTH - 0.08))
        pivot_y = 0.0
        bob_x = pivot_x - artists["visual_pole_length"] * np.sin(theta)
        bob_y = pivot_y + artists["visual_pole_length"] * np.cos(theta)

        artists["cart"].set_xy((pivot_x - artists["cart_width"] / 2.0, -artists["cart_height"] / 2.0))
        artists["pole"].set_data([pivot_x, bob_x], [pivot_y, bob_y])
        artists["bob"].center = (bob_x, bob_y)

        outline = "#b91c1c" if failed_now else "#cbd5e1"
        width = 3.0 if failed_now else 1.6
        for spine in ax.spines.values():
            spine.set_color(outline)
            spine.set_linewidth(width)
        artists["fail"].set_visible(failed_now)
        if failed_now:
            artists["fail"].set_text(f"FAIL\n{panel.failure_reason}")


def run() -> None:
    stages, slot_specs, logs = _build_stage_data()
    if not stages:
        raise RuntimeError("No measurement-noise stages were generated.")

    visual_pole_length = 0.3 * float(INVERTED_PENDULUM_RENDER.visual_pole_scale)
    bob_radius = float(INVERTED_PENDULUM_RENDER.visual_bob_radius)
    cart_width = 0.18
    cart_height = 0.10
    frames_per_stage = int(round(STAGE_DURATION_S * FPS))
    total_frames = frames_per_stage * len(stages)

    fig, axes = plt.subplots(2, 5, figsize=(19.2, 10.8), dpi=100)
    axes = axes.reshape(-1)
    fig.patch.set_facecolor("white")
    fig.suptitle("Measurement-noise stress progression on the nonlinear inverted pendulum", fontsize=24, y=0.985)
    fig.text(
        0.5,
        0.953,
        "10 s per noise stage; red frame = benchmark fail at the current noise scale; only survivors continue",
        ha="center",
        va="top",
        fontsize=13,
        color="#4b5563",
    )
    time_text = fig.text(0.5, 0.922, "", ha="center", va="top", fontsize=18, weight="bold")

    panel_artists: list[dict] = []
    for ax, spec in zip(axes, slot_specs):
        ax.axhline(0.0, color="0.28", linewidth=1.8)
        ax.plot([0.0, 0.0], [0.0, visual_pole_length], "--", color="#d1d5db", linewidth=1.2)
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
        case_text = ax.text(
            0.03,
            0.04,
            "",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.5,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#e5e7eb", alpha=0.92),
        )
        fail_text = ax.text(
            0.50,
            0.55,
            "",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
            weight="bold",
            color="#b91c1c",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#fca5a5", alpha=0.96),
        )
        eliminated_text = ax.text(
            0.50,
            0.55,
            "eliminated",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
            color="#9ca3af",
        )
        panel_artists.append(
            {
                "cart": cart_patch,
                "pole": pole_line,
                "bob": bob_patch,
                "case": case_text,
                "fail": fail_text,
                "eliminated": eliminated_text,
                "cart_width": cart_width,
                "cart_height": cart_height,
                "visual_pole_length": visual_pole_length,
            }
        )

    fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.89], w_pad=1.4, h_pad=1.8)

    writer = FFMpegWriter(
        fps=FPS,
        codec="libx264",
        bitrate=3200,
        metadata={"title": "Measurement-noise stress progression", "artist": "Codex"},
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-profile:v", "high", "-level:v", "4.0"],
    )

    current_stage_idx = -1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(OUT_PATH), dpi=100):
        for global_frame_idx in range(total_frames):
            stage_idx = global_frame_idx // frames_per_stage
            local_frame_idx = global_frame_idx % frames_per_stage
            local_time_s = local_frame_idx / FPS
            stage = stages[stage_idx]

            if stage_idx != current_stage_idx:
                _configure_stage(stage=stage, slot_specs=slot_specs, axes=axes, panel_artists=panel_artists)
                current_stage_idx = stage_idx

            _set_stage_frame(
                stage=stage,
                slot_specs=slot_specs,
                local_frame_idx=local_frame_idx,
                local_time_s=local_time_s,
                axes=axes,
                panel_artists=panel_artists,
                time_text=time_text,
            )
            writer.grab_frame()

    preview_stage_idx = min(len(stages) - 1, 2)
    preview_local_frame_idx = min(frames_per_stage - 1, int(round(8.6 * FPS)))
    _configure_stage(stage=stages[preview_stage_idx], slot_specs=slot_specs, axes=axes, panel_artists=panel_artists)
    _set_stage_frame(
        stage=stages[preview_stage_idx],
        slot_specs=slot_specs,
        local_frame_idx=preview_local_frame_idx,
        local_time_s=preview_local_frame_idx / FPS,
        axes=axes,
        panel_artists=panel_artists,
        time_text=time_text,
    )
    fig.savefig(PREVIEW_PATH, dpi=120)
    plt.close(fig)

    print(f"Saved: {OUT_PATH}")
    print(f"Saved: {PREVIEW_PATH}")
    print("Stage log:")
    for line in logs:
        print(line)


if __name__ == "__main__":
    run()
