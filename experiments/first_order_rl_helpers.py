from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Optional

import numpy as np

SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from control_bench.config import FIRST_ORDER_FAMILY as family
from control_bench.controllers.rl_numpy import PlantParams, RLBPTTConfig
from control_bench.controllers.rl_numpy.types import RolloutSpec
from control_bench.plants.first_order_family import build_first_order_grid
from control_bench.plants.lti_siso import DiscreteLTISISOPlant
from control_bench.plants.second_order_family import expm_small


def extract_first_order_plant_params() -> Dict[str, PlantParams]:
    plants = build_first_order_grid(family)
    out: Dict[str, PlantParams] = {}
    for plant_id, plant in plants.items():
        out[plant_id] = PlantParams(
            A=float(plant.A),
            B=float(plant.B),
            C=float(plant.C),
            D=float(plant.D),
            dt=float(plant.dt),
        )
    return out


def _plant_pz_from_id(plant_id: str) -> tuple[float, Optional[float]]:
    pole_tag, zero_tag = plant_id.split("__", 1)

    if pole_tag == "stable":
        p = float(family.stable_p)
    elif pole_tag == "integrator":
        p = 0.0
    elif pole_tag == "unstable":
        p = float(family.unstable_p)
    else:
        raise ValueError(f"Unknown pole tag: {pole_tag}")

    if zero_tag == "no_zero":
        z = None
    elif zero_tag == "lhp_zero":
        z = float(family.lhp_z)
    elif zero_tag == "rhp_zero":
        z = float(family.rhp_z)
    else:
        raise ValueError(f"Unknown zero tag: {zero_tag}")

    return p, z


def _plant_params_from_continuous(*, p: float, z: Optional[float], k: float) -> PlantParams:
    plant = DiscreteLTISISOPlant.from_continuous_first_order(
        dt=family.dt,
        p=p,
        k=k,
        z=z,
        u_bounds=None,
    )
    return PlantParams(A=float(plant.A), B=float(plant.B), C=float(plant.C), D=float(plant.D), dt=float(plant.dt))


def _sample_randomized_first_order_params(
    plant_id: str,
    *,
    rng: np.random.Generator,
    pole_rel_range: float = 0.05,
    zero_rel_range: float = 0.05,
    gain_rel_range: float = 0.05,
) -> tuple[float, Optional[float], float]:
    p_nom, z_nom = _plant_pz_from_id(plant_id)
    p = p_nom if p_nom == 0.0 else float(p_nom * (1.0 + rng.uniform(-pole_rel_range, pole_rel_range)))
    if z_nom is None or z_nom == 0.0:
        z = z_nom
    else:
        z = float(z_nom * (1.0 + rng.uniform(-zero_rel_range, zero_rel_range)))
    k = float(family.k * (1.0 + rng.uniform(-gain_rel_range, gain_rel_range)))
    return p, z, k


def _discretize_exact(A_c: np.ndarray, B_c: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    A_c = np.asarray(A_c, dtype=float)
    B_c = np.asarray(B_c, dtype=float)
    n = int(A_c.shape[0])

    M = np.zeros((n + 1, n + 1), dtype=float)
    M[:n, :n] = A_c
    M[:n, n:] = B_c

    Md = np.asarray(expm_small(M * float(dt)), dtype=float)
    return Md[:n, :n], Md[:n, n:]


def build_hidden_pole_augmented_first_order_plant(
    *,
    p: float,
    z: Optional[float],
    k: float,
    hidden_pole: float,
) -> PlantParams:
    if hidden_pole <= 0:
        raise ValueError("hidden_pole must be > 0")

    if z is None:
        c1 = float(k)
        d1 = 0.0
    else:
        c1 = float(k * (p - z))
        d1 = float(k)

    a = float(hidden_pole)
    A_c = np.array(
        [
            [float(p), 0.0],
            [a * c1, -a],
        ],
        dtype=float,
    )
    B_c = np.array(
        [
            [1.0],
            [a * d1],
        ],
        dtype=float,
    )
    A_d, B_d = _discretize_exact(A_c, B_c, dt=family.dt)
    return PlantParams(
        A=A_d,
        B=B_d,
        C=np.array([[0.0, 1.0]], dtype=np.float64),
        D=np.array([[0.0]], dtype=np.float64),
        dt=float(family.dt),
    )


def build_randomized_hidden_pole_first_order_plant(
    plant_id: str,
    *,
    rng: np.random.Generator,
    hidden_pole_low: float,
    hidden_pole_high: float,
    pole_rel_range: float = 0.05,
    zero_rel_range: float = 0.05,
    gain_rel_range: float = 0.05,
) -> tuple[PlantParams, float, float, Optional[float], float]:
    p, z, k = _sample_randomized_first_order_params(
        plant_id,
        rng=rng,
        pole_rel_range=pole_rel_range,
        zero_rel_range=zero_rel_range,
        gain_rel_range=gain_rel_range,
    )
    a = float(rng.uniform(hidden_pole_low, hidden_pole_high))
    return build_hidden_pole_augmented_first_order_plant(p=p, z=z, k=k, hidden_pole=a), a, p, z, k


def hidden_pole_state_from_output(
    *,
    y0: float,
    p: float,
    z: Optional[float],
    k: float,
) -> np.ndarray:
    if z is None:
        c1 = float(k)
    else:
        c1 = float(k * (p - z))
    if abs(c1) < 1e-12:
        x_nom = 0.0
    else:
        x_nom = float(y0) / c1
    return np.array([x_nom, float(y0)], dtype=np.float64)


def build_randomized_first_order_plant(
    plant_id: str,
    *,
    rng: np.random.Generator,
    pole_rel_range: float = 0.05,
    zero_rel_range: float = 0.05,
    gain_rel_range: float = 0.05,
) -> PlantParams:
    p, z, k = _sample_randomized_first_order_params(
        plant_id,
        rng=rng,
        pole_rel_range=pole_rel_range,
        zero_rel_range=zero_rel_range,
        gain_rel_range=gain_rel_range,
    )
    return _plant_params_from_continuous(p=p, z=z, k=k)


def _sample_nonzero_uniform(
    rng: np.random.Generator,
    low: float,
    high: float,
    *,
    exclude_center: Optional[float] = None,
    exclude_radius: float = 0.0,
) -> float:
    while True:
        value = float(rng.uniform(low, high))
        if exclude_center is None:
            if abs(value) > exclude_radius:
                return value
        elif abs(value - exclude_center) > exclude_radius:
            return value


def sample_first_order_training_rollout(
    rng: np.random.Generator,
    cfg: RLBPTTConfig,
) -> RolloutSpec:
    """
    Episode mix:
    - 50% tracking
    - 25% regulation
    - 25% disturbance rejection
    """
    horizon_steps = int(cfg.horizon_steps)
    dice = float(rng.random())

    if dice < 0.50:
        r0 = _sample_nonzero_uniform(
            rng,
            -1.8,
            1.8,
            exclude_center=-1.0,
            exclude_radius=0.05,
        )
        x0 = _sample_nonzero_uniform(rng, -0.5, 0.5, exclude_radius=0.02)
        return RolloutSpec(
            r=np.full(horizon_steps, r0, dtype=np.float64),
            du=np.zeros(horizon_steps, dtype=np.float64),
            x0=x0,
            mode="tracking",
        )

    if dice < 0.75:
        if float(rng.random()) < 0.35:
            x0 = 0.0
        else:
            x0 = _sample_nonzero_uniform(rng, -0.5, 0.5, exclude_radius=0.02)
        return RolloutSpec(
            r=np.zeros(horizon_steps, dtype=np.float64),
            du=np.zeros(horizon_steps, dtype=np.float64),
            x0=x0,
            mode="regulation",
        )

    du = np.zeros(horizon_steps, dtype=np.float64)
    step_start = int(
        rng.integers(
            max(1, int(0.2 * horizon_steps)),
            max(2, int(0.7 * horizon_steps)),
        )
    )
    amp = _sample_nonzero_uniform(rng, -0.25, 0.25, exclude_radius=0.05)
    du[step_start:] = amp
    x0 = 0.0 if float(rng.random()) < 0.8 else float(rng.uniform(-0.1, 0.1))
    return RolloutSpec(
        r=np.zeros(horizon_steps, dtype=np.float64),
        du=du,
        x0=x0,
        mode="disturbance_rejection",
    )


def sample_first_order_robust_training_rollout(
    rng: np.random.Generator,
    cfg: RLBPTTConfig,
    *,
    plant_id: str,
    tracking_probability: float = 0.45,
    regulation_probability: float = 0.20,
    measurement_noise_std: float = 0.01,
    disturbance_noise_std: float = 0.01,
    pole_rel_range: float = 0.05,
    zero_rel_range: float = 0.05,
    gain_rel_range: float = 0.05,
    hidden_pole_low: Optional[float] = None,
    hidden_pole_high: Optional[float] = None,
) -> RolloutSpec:
    """
    Robustified rollout distribution for rich MLP training.

    Changes versus the nominal sampler:
    - 45% tracking, 20% regulation, 35% disturbance rejection
    - measurement noise on every rollout
    - mild plant randomization around the nominal pole/zero/gain
    - disturbance episodes include both step disturbance and small Gaussian input noise
    """
    horizon_steps = int(cfg.horizon_steps)
    dice = float(rng.random())
    noise_y = rng.normal(loc=0.0, scale=float(measurement_noise_std), size=(horizon_steps,))
    use_hidden_pole = (
        hidden_pole_low is not None
        and hidden_pole_high is not None
        and float(hidden_pole_low) > 0.0
        and float(hidden_pole_high) >= float(hidden_pole_low)
    )
    if use_hidden_pole:
        plant_override, _, p, z, k = build_randomized_hidden_pole_first_order_plant(
            plant_id,
            rng=rng,
            hidden_pole_low=float(hidden_pole_low),
            hidden_pole_high=float(hidden_pole_high),
            pole_rel_range=pole_rel_range,
            zero_rel_range=zero_rel_range,
            gain_rel_range=gain_rel_range,
        )
    else:
        plant_override = build_randomized_first_order_plant(
            plant_id,
            rng=rng,
            pole_rel_range=pole_rel_range,
            zero_rel_range=zero_rel_range,
            gain_rel_range=gain_rel_range,
        )
        p = z = k = None

    tracking_threshold = float(tracking_probability)
    regulation_threshold = tracking_threshold + float(regulation_probability)

    if dice < tracking_threshold:
        r0 = _sample_nonzero_uniform(
            rng,
            -1.8,
            1.8,
            exclude_center=-1.0,
            exclude_radius=0.05,
        )
        x0_scalar = _sample_nonzero_uniform(rng, -0.5, 0.5, exclude_radius=0.02)
        x0 = (
            hidden_pole_state_from_output(y0=x0_scalar, p=float(p), z=z, k=float(k))
            if use_hidden_pole
            else x0_scalar
        )
        return RolloutSpec(
            r=np.full(horizon_steps, r0, dtype=np.float64),
            du=np.zeros(horizon_steps, dtype=np.float64),
            noise_y=noise_y,
            x0=x0,
            plant_override=plant_override,
            mode="tracking",
        )

    if dice < regulation_threshold:
        if float(rng.random()) < 0.35:
            x0_scalar = 0.0
        else:
            x0_scalar = _sample_nonzero_uniform(rng, -0.5, 0.5, exclude_radius=0.02)
        x0 = (
            hidden_pole_state_from_output(y0=x0_scalar, p=float(p), z=z, k=float(k))
            if use_hidden_pole
            else x0_scalar
        )
        return RolloutSpec(
            r=np.zeros(horizon_steps, dtype=np.float64),
            du=np.zeros(horizon_steps, dtype=np.float64),
            noise_y=noise_y,
            x0=x0,
            plant_override=plant_override,
            mode="regulation",
        )

    du = rng.normal(loc=0.0, scale=float(disturbance_noise_std), size=(horizon_steps,))
    step_start = int(
        rng.integers(
            max(1, int(0.2 * horizon_steps)),
            max(2, int(0.7 * horizon_steps)),
        )
    )
    amp = _sample_nonzero_uniform(rng, -0.25, 0.25, exclude_radius=0.05)
    du[step_start:] += amp
    x0_scalar = 0.0 if float(rng.random()) < 0.8 else float(rng.uniform(-0.1, 0.1))
    x0 = (
        hidden_pole_state_from_output(y0=x0_scalar, p=float(p), z=z, k=float(k))
        if use_hidden_pole
        else x0_scalar
    )
    return RolloutSpec(
        r=np.zeros(horizon_steps, dtype=np.float64),
        du=du,
        noise_y=noise_y,
        x0=x0,
        plant_override=plant_override,
        mode="disturbance_rejection",
    )


def sample_first_order_universal_robust_training_rollout(
    rng: np.random.Generator,
    cfg: RLBPTTConfig,
    *,
    plant_id: str,
    tracking_probability: float = 0.35,
    regulation_probability: float = 0.20,
    measurement_noise_std_low: float = 0.01,
    measurement_noise_std_high: float = 0.02,
    disturbance_noise_std_low: float = 0.01,
    disturbance_noise_std_high: float = 0.02,
    disturbance_step_time_seconds_low: float = 2.0,
    disturbance_step_time_seconds_high: float = 8.0,
    disturbance_step_magnitude_abs_low: float = 0.03,
    disturbance_step_magnitude_abs_high: float = 0.20,
    disturbance_both_signs: bool = True,
    pole_rel_range: float = 0.02,
    zero_rel_range: float = 0.0,
    gain_rel_range: float = 0.02,
) -> RolloutSpec:
    """
    Broader robust-training rollout distribution meant to improve universal robustness
    without fitting one exact benchmark disturbance case.
    """
    horizon_steps = int(cfg.horizon_steps)
    dice = float(rng.random())

    meas_sigma = float(rng.uniform(measurement_noise_std_low, measurement_noise_std_high))
    noise_y = rng.normal(loc=0.0, scale=meas_sigma, size=(horizon_steps,))
    plant_override = build_randomized_first_order_plant(
        plant_id,
        rng=rng,
        pole_rel_range=pole_rel_range,
        zero_rel_range=zero_rel_range,
        gain_rel_range=gain_rel_range,
    )

    tracking_threshold = float(tracking_probability)
    regulation_threshold = tracking_threshold + float(regulation_probability)

    if dice < tracking_threshold:
        r0 = _sample_nonzero_uniform(
            rng,
            -1.8,
            1.8,
            exclude_center=-1.0,
            exclude_radius=0.05,
        )
        x0 = _sample_nonzero_uniform(rng, -0.5, 0.5, exclude_radius=0.02)
        return RolloutSpec(
            r=np.full(horizon_steps, r0, dtype=np.float64),
            du=np.zeros(horizon_steps, dtype=np.float64),
            noise_y=noise_y,
            x0=x0,
            plant_override=plant_override,
            mode="tracking",
        )

    if dice < regulation_threshold:
        if float(rng.random()) < 0.35:
            x0 = 0.0
        else:
            x0 = _sample_nonzero_uniform(rng, -0.5, 0.5, exclude_radius=0.02)
        return RolloutSpec(
            r=np.zeros(horizon_steps, dtype=np.float64),
            du=np.zeros(horizon_steps, dtype=np.float64),
            noise_y=noise_y,
            x0=x0,
            plant_override=plant_override,
            mode="regulation",
        )

    du_sigma = float(rng.uniform(disturbance_noise_std_low, disturbance_noise_std_high))
    du = rng.normal(loc=0.0, scale=du_sigma, size=(horizon_steps,))
    step_time_seconds = float(rng.uniform(disturbance_step_time_seconds_low, disturbance_step_time_seconds_high))
    step_start = int(np.clip(np.round(step_time_seconds / float(cfg.dt if hasattr(cfg, "dt") else family.dt)), 1, horizon_steps - 1))
    amp_abs = float(rng.uniform(disturbance_step_magnitude_abs_low, disturbance_step_magnitude_abs_high))
    amp_sign = float(rng.choice(np.array([-1.0, 1.0], dtype=float))) if disturbance_both_signs else 1.0
    du[step_start:] += amp_sign * amp_abs
    x0 = 0.0 if float(rng.random()) < 0.8 else float(rng.uniform(-0.1, 0.1))
    return RolloutSpec(
        r=np.zeros(horizon_steps, dtype=np.float64),
        du=du,
        noise_y=noise_y,
        x0=x0,
        plant_override=plant_override,
        mode="disturbance_rejection",
    )


def build_first_order_validation_rollouts(cfg: RLBPTTConfig) -> tuple[RolloutSpec, ...]:
    horizon_steps = int(cfg.horizon_steps)

    def const_rollout(r0: float, x0: float, mode: str) -> RolloutSpec:
        return RolloutSpec(
            r=np.full(horizon_steps, r0, dtype=np.float64),
            du=np.zeros(horizon_steps, dtype=np.float64),
            x0=x0,
            mode=mode,
        )

    def disturbance_rollout(start_frac: float, amp: float) -> RolloutSpec:
        du = np.zeros(horizon_steps, dtype=np.float64)
        start_idx = min(horizon_steps - 1, max(1, int(round(start_frac * horizon_steps))))
        du[start_idx:] = amp
        return RolloutSpec(
            r=np.zeros(horizon_steps, dtype=np.float64),
            du=du,
            x0=0.0,
            mode="disturbance_rejection",
        )

    return (
        const_rollout(-1.5, 0.25, "tracking"),
        const_rollout(-0.5, -0.15, "tracking"),
        const_rollout(0.5, 0.15, "tracking"),
        const_rollout(1.5, -0.25, "tracking"),
        const_rollout(0.0, 0.0, "regulation"),
        const_rollout(0.0, 0.40, "regulation"),
        disturbance_rollout(0.30, 0.15),
        disturbance_rollout(0.60, -0.15),
    )


def build_first_order_robust_validation_rollouts(
    cfg: RLBPTTConfig,
    *,
    plant_id: Optional[str] = None,
    measurement_noise_std: float = 0.01,
    disturbance_noise_std: float = 0.02,
    hidden_pole_cases: tuple[float, ...] = (),
    seed: int = 12345,
) -> tuple[RolloutSpec, ...]:
    horizon_steps = int(cfg.horizon_steps)
    rng = np.random.default_rng(seed)

    def const_rollout(r0: float, x0: float, mode: str) -> RolloutSpec:
        noise_y = rng.normal(loc=0.0, scale=float(measurement_noise_std), size=(horizon_steps,))
        return RolloutSpec(
            r=np.full(horizon_steps, r0, dtype=np.float64),
            du=np.zeros(horizon_steps, dtype=np.float64),
            noise_y=noise_y,
            x0=x0,
            mode=mode,
        )

    def disturbance_rollout(start_frac: float, amp: float) -> RolloutSpec:
        du = rng.normal(loc=0.0, scale=float(disturbance_noise_std), size=(horizon_steps,))
        noise_y = rng.normal(loc=0.0, scale=float(measurement_noise_std), size=(horizon_steps,))
        start_idx = min(horizon_steps - 1, max(1, int(round(start_frac * horizon_steps))))
        du[start_idx:] += amp
        return RolloutSpec(
            r=np.zeros(horizon_steps, dtype=np.float64),
            du=du,
            noise_y=noise_y,
            x0=0.0,
            mode="disturbance_rejection",
        )
    rollouts = [
        const_rollout(-1.5, 0.25, "tracking"),
        const_rollout(-0.5, -0.15, "tracking"),
        const_rollout(0.5, 0.15, "tracking"),
        const_rollout(1.5, -0.25, "tracking"),
        const_rollout(0.0, 0.0, "regulation"),
        const_rollout(0.0, 0.40, "regulation"),
        disturbance_rollout(0.30, 0.15),
        disturbance_rollout(0.60, -0.15),
    ]

    if hidden_pole_cases:
        if plant_id is None:
            raise ValueError("plant_id must be provided when hidden_pole_cases are used in validation")
        p_nom, z_nom = _plant_pz_from_id(plant_id)
        k_nom = float(family.k)
        for idx, hidden_pole in enumerate(hidden_pole_cases):
            lag_plant = build_hidden_pole_augmented_first_order_plant(
                p=p_nom,
                z=z_nom,
                k=k_nom,
                hidden_pole=float(hidden_pole),
            )
            lag_noise_y = rng.normal(loc=0.0, scale=float(measurement_noise_std), size=(horizon_steps,))
            lag_x0_output = 0.40 if idx % 2 == 0 else -0.25
            rollouts.append(
                RolloutSpec(
                    r=np.zeros(horizon_steps, dtype=np.float64),
                    du=np.zeros(horizon_steps, dtype=np.float64),
                    noise_y=lag_noise_y,
                    x0=hidden_pole_state_from_output(y0=lag_x0_output, p=p_nom, z=z_nom, k=k_nom),
                    plant_override=lag_plant,
                    mode="lag_mismatch",
                )
            )
            lag_du = rng.normal(loc=0.0, scale=float(disturbance_noise_std), size=(horizon_steps,))
            start_frac = 0.30 if idx % 2 == 0 else 0.60
            amp = 0.10 if idx % 2 == 0 else -0.10
            start_idx = min(horizon_steps - 1, max(1, int(round(start_frac * horizon_steps))))
            lag_du[start_idx:] += amp
            rollouts.append(
                RolloutSpec(
                    r=np.zeros(horizon_steps, dtype=np.float64),
                    du=lag_du,
                    noise_y=lag_noise_y.copy(),
                    x0=hidden_pole_state_from_output(y0=0.0, p=p_nom, z=z_nom, k=k_nom),
                    plant_override=lag_plant,
                    mode="lag_mismatch",
                )
            )

    return tuple(rollouts)


def build_first_order_universal_robust_validation_rollouts(
    cfg: RLBPTTConfig,
    *,
    measurement_noise_std: float = 0.015,
    disturbance_noise_std: float = 0.015,
    disturbance_cases: tuple[tuple[float, float], ...] = ((2.5, 0.08), (5.0, -0.12), (7.5, 0.15)),
    seed: int = 54321,
) -> tuple[RolloutSpec, ...]:
    horizon_steps = int(cfg.horizon_steps)
    rng = np.random.default_rng(seed)

    def const_rollout(r0: float, x0: float, mode: str) -> RolloutSpec:
        noise_y = rng.normal(loc=0.0, scale=float(measurement_noise_std), size=(horizon_steps,))
        return RolloutSpec(
            r=np.full(horizon_steps, r0, dtype=np.float64),
            du=np.zeros(horizon_steps, dtype=np.float64),
            noise_y=noise_y,
            x0=x0,
            mode=mode,
        )

    def disturbance_rollout(step_time_seconds: float, amp: float) -> RolloutSpec:
        du = rng.normal(loc=0.0, scale=float(disturbance_noise_std), size=(horizon_steps,))
        noise_y = rng.normal(loc=0.0, scale=float(measurement_noise_std), size=(horizon_steps,))
        start_idx = int(np.clip(np.round(float(step_time_seconds) / float(family.dt)), 1, horizon_steps - 1))
        du[start_idx:] += float(amp)
        return RolloutSpec(
            r=np.zeros(horizon_steps, dtype=np.float64),
            du=du,
            noise_y=noise_y,
            x0=0.0,
            mode="disturbance_rejection",
        )

    return (
        const_rollout(-1.5, 0.25, "tracking"),
        const_rollout(-0.5, -0.15, "tracking"),
        const_rollout(0.5, 0.15, "tracking"),
        const_rollout(1.5, -0.25, "tracking"),
        const_rollout(0.0, 0.0, "regulation"),
        const_rollout(0.0, 0.40, "regulation"),
        *(disturbance_rollout(step_time_seconds, amp) for step_time_seconds, amp in disturbance_cases),
    )
