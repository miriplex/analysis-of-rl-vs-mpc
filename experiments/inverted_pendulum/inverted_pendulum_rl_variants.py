from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from control_bench.config import INVERTED_PENDULUM_RL_TRAINING
from control_bench.controllers.inverted_pendulum_rl import MLPPolicySpec, pendulum_feature_dim


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
PLANT_ID = INVERTED_PENDULUM_RL_TRAINING.plant_id

VARIANT_RESULTS_ROOT = EXPERIMENTS_ROOT / "results" / "rl_numpy" / "inverted_pendulum_linearized_variants"
VARIANT_WEIGHTS_DIR = VARIANT_RESULTS_ROOT / "per_variant"
VARIANT_FIGURES_DIR = EXPERIMENTS_ROOT / "results" / "figures" / "rl_numpy" / "inverted_pendulum_linearized_variants"
VARIANT_METRICS_DIR = EXPERIMENTS_ROOT / "results" / "metrics" / "inverted_pendulum_linearized_variants"


@dataclass(frozen=True)
class PendulumRLVariant:
    variant_id: str
    display_name: str
    kind: str
    hidden_layers: tuple[int, ...] = ()
    activation: str = "tanh"
    steps: int = 100000
    lr: float = 5e-4
    seeds: int = 1
    grad_clip_norm: Optional[float] = None

    @property
    def is_pid(self) -> bool:
        return self.kind == "pidfeat"

    @property
    def canonical_weight_path(self) -> Path:
        return VARIANT_WEIGHTS_DIR / f"rl_{self.variant_id}__{PLANT_ID}.npz"

    def seeded_weight_path(self, seed_index: int) -> Path:
        return VARIANT_WEIGHTS_DIR / f"rl_{self.variant_id}__{PLANT_ID}__seed{seed_index}.npz"

    def mlp_spec(self) -> Optional[MLPPolicySpec]:
        if self.is_pid:
            return None
        return MLPPolicySpec(
            input_dim=pendulum_feature_dim(self.kind),
            hidden_layers=self.hidden_layers,
            activation=self.activation,
        )


PENDULUM_RL_VARIANTS: tuple[PendulumRLVariant, ...] = (
    PendulumRLVariant(
        variant_id="pidfeat",
        display_name="PID-features",
        kind="pidfeat",
        steps=INVERTED_PENDULUM_RL_TRAINING.pid_steps,
        lr=INVERTED_PENDULUM_RL_TRAINING.pid_lr,
        seeds=INVERTED_PENDULUM_RL_TRAINING.pid_seeds,
        grad_clip_norm=None,
    ),
    PendulumRLVariant(
        variant_id="state6_16_16",
        display_name="state6 16_16",
        kind="state6",
        hidden_layers=(16, 16),
        activation=INVERTED_PENDULUM_RL_TRAINING.rich_activation,
        steps=INVERTED_PENDULUM_RL_TRAINING.rich_steps,
        lr=INVERTED_PENDULUM_RL_TRAINING.rich_lr,
        seeds=1,
        grad_clip_norm=INVERTED_PENDULUM_RL_TRAINING.rich_grad_clip_norm,
    ),
    PendulumRLVariant(
        variant_id="state6_32_32",
        display_name="state6 32_32",
        kind="state6",
        hidden_layers=(32, 32),
        activation=INVERTED_PENDULUM_RL_TRAINING.rich_activation,
        steps=INVERTED_PENDULUM_RL_TRAINING.rich_steps,
        lr=INVERTED_PENDULUM_RL_TRAINING.rich_lr,
        seeds=1,
        grad_clip_norm=INVERTED_PENDULUM_RL_TRAINING.rich_grad_clip_norm,
    ),
    PendulumRLVariant(
        variant_id="state6_32_32_32",
        display_name="state6 32_32_32",
        kind="state6",
        hidden_layers=(32, 32, 32),
        activation=INVERTED_PENDULUM_RL_TRAINING.rich_activation,
        steps=INVERTED_PENDULUM_RL_TRAINING.rich_steps,
        lr=INVERTED_PENDULUM_RL_TRAINING.rich_lr,
        seeds=1,
        grad_clip_norm=INVERTED_PENDULUM_RL_TRAINING.rich_grad_clip_norm,
    ),
    PendulumRLVariant(
        variant_id="history8_16_16",
        display_name="history8 16_16",
        kind="history8",
        hidden_layers=(16, 16),
        activation=INVERTED_PENDULUM_RL_TRAINING.rich_activation,
        steps=INVERTED_PENDULUM_RL_TRAINING.rich_steps,
        lr=INVERTED_PENDULUM_RL_TRAINING.rich_lr,
        seeds=1,
        grad_clip_norm=INVERTED_PENDULUM_RL_TRAINING.rich_grad_clip_norm,
    ),
    PendulumRLVariant(
        variant_id="history8_32_32",
        display_name="history8 32_32",
        kind="history8",
        hidden_layers=(32, 32),
        activation=INVERTED_PENDULUM_RL_TRAINING.rich_activation,
        steps=INVERTED_PENDULUM_RL_TRAINING.rich_steps,
        lr=INVERTED_PENDULUM_RL_TRAINING.rich_lr,
        seeds=1,
        grad_clip_norm=INVERTED_PENDULUM_RL_TRAINING.rich_grad_clip_norm,
    ),
    PendulumRLVariant(
        variant_id="history8_32_32_32",
        display_name="history8 32_32_32",
        kind="history8",
        hidden_layers=(32, 32, 32),
        activation=INVERTED_PENDULUM_RL_TRAINING.rich_activation,
        steps=INVERTED_PENDULUM_RL_TRAINING.rich_steps,
        lr=INVERTED_PENDULUM_RL_TRAINING.rich_lr,
        seeds=1,
        grad_clip_norm=INVERTED_PENDULUM_RL_TRAINING.rich_grad_clip_norm,
    ),
    PendulumRLVariant(
        variant_id="rich11_16_16",
        display_name="rich11 16_16",
        kind="rich11_pendulum",
        hidden_layers=(16, 16),
        activation=INVERTED_PENDULUM_RL_TRAINING.rich_activation,
        steps=INVERTED_PENDULUM_RL_TRAINING.rich_steps,
        lr=INVERTED_PENDULUM_RL_TRAINING.rich_lr,
        seeds=1,
        grad_clip_norm=INVERTED_PENDULUM_RL_TRAINING.rich_grad_clip_norm,
    ),
    PendulumRLVariant(
        variant_id="rich11_32_32",
        display_name="rich11 32_32",
        kind="rich11_pendulum",
        hidden_layers=(32, 32),
        activation=INVERTED_PENDULUM_RL_TRAINING.rich_activation,
        steps=INVERTED_PENDULUM_RL_TRAINING.rich_steps,
        lr=INVERTED_PENDULUM_RL_TRAINING.rich_lr,
        seeds=1,
        grad_clip_norm=INVERTED_PENDULUM_RL_TRAINING.rich_grad_clip_norm,
    ),
    PendulumRLVariant(
        variant_id="rich11_32_32_32",
        display_name="rich11 32_32_32",
        kind="rich11_pendulum",
        hidden_layers=(32, 32, 32),
        activation=INVERTED_PENDULUM_RL_TRAINING.rich_activation,
        steps=INVERTED_PENDULUM_RL_TRAINING.rich_steps,
        lr=INVERTED_PENDULUM_RL_TRAINING.rich_lr,
        seeds=1,
        grad_clip_norm=INVERTED_PENDULUM_RL_TRAINING.rich_grad_clip_norm,
    ),
)


def variant_by_id(variant_id: str) -> PendulumRLVariant:
    for variant in PENDULUM_RL_VARIANTS:
        if variant.variant_id == variant_id:
            return variant
    raise KeyError(f"Unknown pendulum RL variant: {variant_id}")


def resolve_variants(selected: Optional[Iterable[str]] = None) -> tuple[PendulumRLVariant, ...]:
    if selected is None:
        return PENDULUM_RL_VARIANTS
    selected_ids = tuple(selected)
    return tuple(variant_by_id(variant_id) for variant_id in selected_ids)
