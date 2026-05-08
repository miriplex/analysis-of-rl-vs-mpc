from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PlantParams:
    """Discrete SISO plant parameters (matches your LTI plant convention)."""
    A: np.ndarray | float
    B: np.ndarray | float
    C: np.ndarray | float
    D: np.ndarray | float
    dt: float

    def __post_init__(self) -> None:
        A = np.asarray(self.A, dtype=np.float64)
        if A.ndim == 0:
            A = A.reshape(1, 1)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"PlantParams.A must be square, got shape {A.shape}")
        n = int(A.shape[0])

        B = np.asarray(self.B, dtype=np.float64)
        if B.ndim == 0:
            B = B.reshape(1, 1)
        elif B.ndim == 1:
            if B.size != n:
                raise ValueError(f"PlantParams.B must have length {n}, got {B.size}")
            B = B.reshape(n, 1)
        elif B.shape != (n, 1):
            raise ValueError(f"PlantParams.B must have shape ({n}, 1), got {B.shape}")

        C = np.asarray(self.C, dtype=np.float64)
        if C.ndim == 0:
            C = C.reshape(1, 1)
        elif C.ndim == 1:
            if C.size != n:
                raise ValueError(f"PlantParams.C must have length {n}, got {C.size}")
            C = C.reshape(1, n)
        elif C.shape != (1, n):
            raise ValueError(f"PlantParams.C must have shape (1, {n}), got {C.shape}")

        D = np.asarray(self.D, dtype=np.float64)
        if D.ndim == 0:
            D = D.reshape(1, 1)
        elif D.shape != (1, 1):
            raise ValueError(f"PlantParams.D must be scalar or shape (1, 1), got {D.shape}")

        dt = float(self.dt)
        if dt <= 0.0:
            raise ValueError(f"PlantParams.dt must be > 0, got {dt}")

        object.__setattr__(self, "A", A)
        object.__setattr__(self, "B", B)
        object.__setattr__(self, "C", C)
        object.__setattr__(self, "D", D)
        object.__setattr__(self, "dt", dt)

    @property
    def state_dim(self) -> int:
        return int(self.A.shape[0])


@dataclass(frozen=True)
class RLBPTTConfig:
    """
    RL training config for BPTT rollout.

    Loss (mean over horizon):
      qy*(y-r)^2 + ru*(u-u_prev)^2 + qu*u^2

    u_min/u_max are *soft* action bounds (tanh squash inside the policy).
    """
    horizon_steps: int = 250
    qy: float = 1.0
    ru: float = 1.0
    qu: float = 0.01
    r0: float = 1.0
    u_min: Optional[float] = None
    u_max: Optional[float] = None


@dataclass(frozen=True)
class RolloutSpec:
    """
    One training/validation rollout specification.

    - `r`: reference sequence over the horizon
    - `du`: additive input disturbance sequence applied to the plant input
    - `noise_y`: additive measurement noise sequence seen by the controller
    - `x0`: initial plant state vector
    - `plant_override`: optional rollout-specific plant for domain randomization
    - `mode`: human-readable label for logging/metadata
    """

    r: np.ndarray
    du: np.ndarray
    noise_y: Optional[np.ndarray] = None
    x0: np.ndarray | float = 0.0
    plant_override: Optional[PlantParams] = None
    mode: str = "tracking"

    def __post_init__(self) -> None:
        r = np.asarray(self.r, dtype=np.float64).reshape(-1)
        du = np.asarray(self.du, dtype=np.float64).reshape(-1)
        if r.size == 0:
            raise ValueError("RolloutSpec.r must be non-empty")
        if du.shape != r.shape:
            raise ValueError(f"RolloutSpec.du must match r shape, got {du.shape} vs {r.shape}")
        if self.noise_y is None:
            noise_y = np.zeros_like(r)
        else:
            noise_y = np.asarray(self.noise_y, dtype=np.float64).reshape(-1)
            if noise_y.shape != r.shape:
                raise ValueError(f"RolloutSpec.noise_y must match r shape, got {noise_y.shape} vs {r.shape}")
        x0 = np.asarray(self.x0, dtype=np.float64).reshape(-1)
        if x0.size == 0:
            raise ValueError("RolloutSpec.x0 must be non-empty")
        object.__setattr__(self, "r", r)
        object.__setattr__(self, "du", du)
        object.__setattr__(self, "noise_y", noise_y)
        object.__setattr__(self, "x0", x0)


Params = Dict[str, np.ndarray]
Grads = Dict[str, np.ndarray]
