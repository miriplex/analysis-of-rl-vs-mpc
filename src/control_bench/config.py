from __future__ import annotations

from dataclasses import dataclass
from math import radians
from typing import Optional

from .plants.first_order_family import FirstOrderFamilyParams
from .plants.inverted_pendulum import InvertedPendulumParams
from .plants.second_order_family import SecondOrderFamilyParams, SecondOrderSinglePlantParams


@dataclass(frozen=True)
class QuadraticCostWeights:
    qy: float
    ru: float
    qu: float


@dataclass(frozen=True)
class InvertedPendulumRenderConfig:
    realtime: bool
    hold_final_frame: bool
    visual_pole_scale: float
    visual_bob_radius: float


@dataclass(frozen=True)
class InvertedPendulumOpenLoopConfig:
    t_final: float
    u_open_loop: float
    x0: tuple[float, float, float, float]
    angle_limit_rad: float


@dataclass(frozen=True)
class InvertedPendulumMPCExperimentConfig:
    t_final: float
    x0: tuple[float, float, float, float]
    reference: tuple[float, float, float, float]
    angle_limit_rad: float
    cart_limit_m: float
    N: int
    q_state_diag: tuple[float, float, float, float]
    ru: float
    qu: float
    pgd_iters: int
    pgd_step_size: Optional[float] = None


@dataclass(frozen=True)
class InvertedPendulumWindDisturbanceConfig:
    t_final: float
    x0: tuple[float, float, float, float]
    reference: tuple[float, float, float, float]
    angle_limit_rad: float
    cart_limit_m: float
    wind_start_s: float
    wind_duration_s: float
    wind_peak_force: float


@dataclass(frozen=True)
class InvertedPendulumRLTrainingConfig:
    plant_id: str
    horizon_seconds: float
    q_state_diag: tuple[float, float, float, float]
    ru: float
    qu: float
    use_terminal_cost: bool
    pid_steps: int
    pid_lr: float
    pid_seeds: int
    rich_steps: int
    rich_lr: float
    rich_seeds: int
    rich_hidden_layers: tuple[int, ...]
    rich_activation: str
    rich_grad_clip_norm: Optional[float]
    eval_every: int
    early_stop_patience_evals: int
    early_stop_rel_improve: float
    early_stop_best_val_threshold: float

# ---------------------------------------------------------------------
# UNIVERSAL PARAMETERS FOR FIRST-ORDER EXPERIMENTS
# ---------------------------------------------------------------------
# Change values here ONCE, and both training + evaluation will match.
# If you change these, you MUST retrain RL (weights are plant-specific).
# ---------------------------------------------------------------------

FIRST_ORDER_FAMILY = FirstOrderFamilyParams(
    dt=0.1,
    k=1.0,

    # Poles
    stable_p=-1.0,
    unstable_p=+1.0,

    # Zeros (IMPORTANT: avoid pole-zero cancellation p == z)
    lhp_z=-2.0,
    rhp_z=+2.0,

    # Actuator bounds (recommended for fair comparison)
    u_min=None,
    u_max=None,
)

# Shared 9-plant ordering (use everywhere so plots/tables align)
FIRST_ORDER_PLANT_IDS = [
    "stable__no_zero", "stable__lhp_zero", "stable__rhp_zero",
    "integrator__no_zero", "integrator__lhp_zero", "integrator__rhp_zero",
    "unstable__no_zero", "unstable__lhp_zero", "unstable__rhp_zero",
]

FIRST_ORDER_COST_WEIGHTS = QuadraticCostWeights(
    qy=1.0,
    ru=2.0,
    qu=0,
)

# ---------------------------------------------------------------------
# UNIVERSAL PARAMETERS FOR SECOND-ORDER (2-STATE) EXPERIMENTS
# ---------------------------------------------------------------------

SECOND_ORDER_FAMILY = SecondOrderFamilyParams(
    dt=0.02,
    k=1.0,
    wn=1.0,
    zeta_ud=0.2,
    zeta_cd=1.0,
    zeta_od=2.0,
    sigma=0.3,
    a=0.5,
    b=1.0,
    z_lhp=2.0,
    z_rhp=2.0,
    # bounds OFF for the open-loop demo
    u_min=None,
    u_max=None,
)

SECOND_ORDER_PLANT_IDS = [
    "stable_ud",
    "stable_cd",
    "stable_od",
    "unstable_saddle",
    "unstable_osc",
    "double_integrator",
    "int_plus_pole",
    "stable_ud_lhp_zero",
    "stable_ud_rhp_zero",
    "boss_unstable_osc_rhp_zero",
]

SECOND_ORDER_SINGLE_LIGHTLY_DAMPED = SecondOrderSinglePlantParams(
    plant_id="lightly_damped_custom",
    dt=0.05,
    b1=0.0,
    b0=1.0,
    a1=0.5,
    a0=1.0,
    u_min=None,
    u_max=None,
)

INVERTED_PENDULUM_LINEARIZED = InvertedPendulumParams(
    dt=0.02,
    cart_mass=0.5,
    pole_mass=4,
    cart_damping=0.1,
    pole_length=0.3,
    pole_inertia=0.006,
    gravity=9.8,
    u_min=None,
    u_max=None,
)

INVERTED_PENDULUM_NONLINEAR = InvertedPendulumParams(
    dt=INVERTED_PENDULUM_LINEARIZED.dt,
    cart_mass=INVERTED_PENDULUM_LINEARIZED.cart_mass,
    pole_mass=INVERTED_PENDULUM_LINEARIZED.pole_mass,
    cart_damping=INVERTED_PENDULUM_LINEARIZED.cart_damping,
    pole_length=INVERTED_PENDULUM_LINEARIZED.pole_length,
    pole_inertia=INVERTED_PENDULUM_LINEARIZED.pole_inertia,
    gravity=INVERTED_PENDULUM_LINEARIZED.gravity,
    u_min=INVERTED_PENDULUM_LINEARIZED.u_min,
    u_max=INVERTED_PENDULUM_LINEARIZED.u_max,
)

INVERTED_PENDULUM_RENDER = InvertedPendulumRenderConfig(
    realtime=True,
    hold_final_frame=False,
    visual_pole_scale=2.4,
    visual_bob_radius=0.055,
)

INVERTED_PENDULUM_OPEN_LOOP = InvertedPendulumOpenLoopConfig(
    t_final=8.0,
    u_open_loop=0.0,
    x0=(0.0, 0.0, radians(5.0), 0.0),
    angle_limit_rad=radians(20.0),
)

INVERTED_PENDULUM_MPC_EXPERIMENT = InvertedPendulumMPCExperimentConfig(
    t_final=8.0,
    x0=(0.0, 0.0, radians(5.0), 0.0),
    reference=(0.0, 0.0, 0.0, 0.0),
    angle_limit_rad=radians(20.0),
    cart_limit_m=float("inf"),
    N=60,
    q_state_diag=(1.0, 0.1, 40.0, 4.0),
    ru=2.0,
    qu=0.0,
    pgd_iters=80,
    pgd_step_size=None,
)

INVERTED_PENDULUM_WIND_DISTURBANCE = InvertedPendulumWindDisturbanceConfig(
    t_final=8.0,
    x0=(0.0, 0.0, 0.0, 0.0),
    reference=(0.0, 0.0, 0.0, 0.0),
    angle_limit_rad=radians(20.0),
    cart_limit_m=float("inf"),
    wind_start_s=1.0,
    wind_duration_s=1.0,
    wind_peak_force=0.7,
)

INVERTED_PENDULUM_NONLINEAR_OPEN_LOOP = InvertedPendulumOpenLoopConfig(
    t_final=8.0,
    u_open_loop=0.0,
    x0=(0.0, 0.0, radians(5.0), 0.0),
    angle_limit_rad=float("inf"),
)

INVERTED_PENDULUM_NONLINEAR_MPC_EXPERIMENT = InvertedPendulumMPCExperimentConfig(
    t_final=8.0,
    x0=(0.0, 0.0, radians(5.0), 0.0),
    reference=(0.0, 0.0, 0.0, 0.0),
    angle_limit_rad=radians(90.0),
    cart_limit_m=float("inf"),
    N=60,
    q_state_diag=(1.0, 0.1, 40.0, 4.0),
    ru=0.1,
    qu=0.01,
    pgd_iters=80,
    pgd_step_size=None,
)

INVERTED_PENDULUM_NONLINEAR_WIND_DISTURBANCE = InvertedPendulumWindDisturbanceConfig(
    t_final=8.0,
    x0=(0.0, 0.0, 0.0, 0.0),
    reference=(0.0, 0.0, 0.0, 0.0),
    angle_limit_rad=radians(90.0),
    cart_limit_m=float("inf"),
    wind_start_s=1.0,
    wind_duration_s=2.0,
    wind_peak_force=2,
)

INVERTED_PENDULUM_RL_TRAINING = InvertedPendulumRLTrainingConfig(
    plant_id="inverted_pendulum_linearized",
    horizon_seconds=INVERTED_PENDULUM_MPC_EXPERIMENT.t_final,
    q_state_diag=INVERTED_PENDULUM_NONLINEAR_MPC_EXPERIMENT.q_state_diag,
    ru=INVERTED_PENDULUM_NONLINEAR_MPC_EXPERIMENT.ru,
    qu=INVERTED_PENDULUM_NONLINEAR_MPC_EXPERIMENT.qu,
    use_terminal_cost=True,
    pid_steps=100000,
    pid_lr=3e-3,
    pid_seeds=1,
    rich_steps=100000,
    rich_lr=5e-4,
    rich_seeds=5,
    rich_hidden_layers=(16, 16),
    rich_activation="tanh",
    rich_grad_clip_norm=1.0,
    eval_every=100,
    early_stop_patience_evals=10,
    early_stop_rel_improve=0.01,
    early_stop_best_val_threshold=1.0,
)
