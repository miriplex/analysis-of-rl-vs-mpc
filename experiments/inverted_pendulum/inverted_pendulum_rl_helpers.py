from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from control_bench.config import INVERTED_PENDULUM_LINEARIZED
from control_bench.controllers.inverted_pendulum_rl import (
    InvertedPendulumRLBPTTConfig,
    InvertedPendulumRolloutSpec,
)
from control_bench.plants.inverted_pendulum import build_linearized_inverted_pendulum


PENDULUM_RL_PLANT_ID = "inverted_pendulum_linearized"


def build_inverted_pendulum_training_plant():
    return build_linearized_inverted_pendulum(INVERTED_PENDULUM_LINEARIZED)


def _sample_state(
    rng: np.random.Generator,
    *,
    x_range: tuple[float, float],
    xdot_range: tuple[float, float],
    theta_range_rad: tuple[float, float],
    thetadot_range: tuple[float, float],
    min_norm: float,
) -> np.ndarray:
    while True:
        x0 = np.array(
            [
                rng.uniform(*x_range),
                rng.uniform(*xdot_range),
                rng.uniform(*theta_range_rad),
                rng.uniform(*thetadot_range),
            ],
            dtype=np.float64,
        )
        if float(np.linalg.norm(x0)) > min_norm:
            return x0


def sample_inverted_pendulum_training_rollout(
    rng: np.random.Generator,
    cfg: InvertedPendulumRLBPTTConfig,
) -> InvertedPendulumRolloutSpec:
    T = int(cfg.horizon_steps)
    zeros_ref = np.zeros((T, 4), dtype=np.float64)
    dice = float(rng.random())

    if dice < 0.50:
        x0 = _sample_state(
            rng,
            x_range=(-0.15, 0.15),
            xdot_range=(-0.35, 0.35),
            theta_range_rad=(-np.deg2rad(8.0), np.deg2rad(8.0)),
            thetadot_range=(-0.8, 0.8),
            min_norm=0.03,
        )
        return InvertedPendulumRolloutSpec(r=zeros_ref, du=np.zeros(T, dtype=np.float64), x0=x0, mode="regulation")

    if dice < 0.75:
        du = np.zeros((T,), dtype=np.float64)
        start_idx = int(rng.integers(max(1, int(0.20 * T)), max(2, int(0.70 * T))))
        amp = float(rng.uniform(-0.8, 0.8))
        if abs(amp) < 0.15:
            amp = 0.15 if amp >= 0.0 else -0.15
        du[start_idx:] = amp
        return InvertedPendulumRolloutSpec(
            r=zeros_ref,
            du=du,
            x0=np.zeros((4,), dtype=np.float64),
            mode="disturbance_rejection",
        )

    x0 = _sample_state(
        rng,
        x_range=(-0.25, 0.25),
        xdot_range=(-0.40, 0.40),
        theta_range_rad=(-np.deg2rad(3.0), np.deg2rad(3.0)),
        thetadot_range=(-0.4, 0.4),
        min_norm=0.02,
    )
    return InvertedPendulumRolloutSpec(r=zeros_ref, du=np.zeros(T, dtype=np.float64), x0=x0, mode="regulation")


def build_inverted_pendulum_validation_rollouts(
    cfg: InvertedPendulumRLBPTTConfig,
) -> tuple[InvertedPendulumRolloutSpec, ...]:
    T = int(cfg.horizon_steps)
    zeros_ref = np.zeros((T, 4), dtype=np.float64)

    def regulation(x0: np.ndarray) -> InvertedPendulumRolloutSpec:
        return InvertedPendulumRolloutSpec(
            r=zeros_ref,
            du=np.zeros((T,), dtype=np.float64),
            x0=np.asarray(x0, dtype=np.float64),
            mode="regulation",
        )

    def disturbance(start_frac: float, amp: float) -> InvertedPendulumRolloutSpec:
        du = np.zeros((T,), dtype=np.float64)
        start_idx = min(T - 1, max(1, int(round(start_frac * T))))
        du[start_idx:] = float(amp)
        return InvertedPendulumRolloutSpec(
            r=zeros_ref,
            du=du,
            x0=np.zeros((4,), dtype=np.float64),
            mode="disturbance_rejection",
        )

    return (
        regulation(np.array([0.0, 0.0, np.deg2rad(5.0), 0.0], dtype=np.float64)),
        regulation(np.array([0.0, 0.0, -np.deg2rad(5.0), 0.0], dtype=np.float64)),
        regulation(np.array([0.12, 0.0, 0.0, 0.0], dtype=np.float64)),
        regulation(np.array([-0.12, 0.0, 0.0, 0.0], dtype=np.float64)),
        regulation(np.array([0.10, 0.0, np.deg2rad(4.0), 0.0], dtype=np.float64)),
        disturbance(0.30, 0.5),
        disturbance(0.60, -0.5),
    )
