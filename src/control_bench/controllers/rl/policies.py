from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn


def squash_to_bounds(u_raw: torch.Tensor, u_min: Optional[float], u_max: Optional[float]) -> torch.Tensor:
    """
    Smoothly map (-inf, inf) -> [u_min, u_max] using tanh.
    If bounds are None, returns u_raw unchanged.
    """
    if (u_min is None) or (u_max is None):
        return u_raw
    mid = 0.5 * (u_max + u_min)
    half = 0.5 * (u_max - u_min)
    return mid + half * torch.tanh(u_raw)


class LinearPIDFeaturePolicy(nn.Module):
    """
    Simple learned policy:
        u = W*[e, ie, de] + b
    where ie is integral of error and de is derivative of error.

    This is effectively a learned PID-like controller.
    """
    input_dim: int = 3

    def __init__(self, *, use_bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(self.input_dim, 1, bias=use_bias)

    def forward(self, feat: torch.Tensor, u_min: Optional[float], u_max: Optional[float]) -> torch.Tensor:
        # feat shape: (..., 3)
        u_raw = self.linear(feat).squeeze(-1)
        return squash_to_bounds(u_raw, u_min, u_max)


@dataclass(frozen=True)
class MLPPolicySpec:
    """
    Defines an MLP policy architecture for readability and reproducibility.
    """
    input_dim: int
    hidden_layers: Sequence[int] = (16, 16)
    activation: str = "tanh"  # "tanh" or "relu"
    use_bias: bool = True


class MLPPolicy(nn.Module):
    """
    More expressive policy:
        u = MLP(feat)

    Recommended feature vector for first-order LTI:
        feat = [r, y, e, ie, de, u_prev]  (dim=6)
    """
    def __init__(self, spec: MLPPolicySpec) -> None:
        super().__init__()
        self.spec = spec

        if spec.activation.lower() == "tanh":
            act = nn.Tanh
        elif spec.activation.lower() == "relu":
            act = nn.ReLU
        else:
            raise ValueError(f"Unknown activation: {spec.activation}")

        layers = []
        in_dim = spec.input_dim
        for h in spec.hidden_layers:
            layers.append(nn.Linear(in_dim, int(h), bias=spec.use_bias))
            layers.append(act())
            in_dim = int(h)

        layers.append(nn.Linear(in_dim, 1, bias=spec.use_bias))
        self.net = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor, u_min: Optional[float], u_max: Optional[float]) -> torch.Tensor:
        # feat shape: (..., input_dim)
        u_raw = self.net(feat).squeeze(-1)
        return squash_to_bounds(u_raw, u_min, u_max)