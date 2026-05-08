from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class IOSpec:
    """
    Declares the I/O dimensions so we can validate controller/plant compatibility.
    - ref_dim: dimension of reference signal r(t)
    - obs_dim: dimension of observation given to controller
    - act_dim: dimension of control action u(t)
    - out_dim: dimension of plant output y(t) used for evaluation/metrics
    """
    ref_dim: int
    obs_dim: int
    act_dim: int
    out_dim: int


@dataclass(frozen=True)
class Bounds:
    """Simple box constraints (per-channel)."""
    low: Array
    high: Array

    def __post_init__(self) -> None:
        low = np.asarray(self.low, dtype=float).reshape(-1)
        high = np.asarray(self.high, dtype=float).reshape(-1)
        if low.shape != high.shape:
            raise ValueError(f"Bounds shape mismatch: low{low.shape} vs high{high.shape}")
        if np.any(low > high):
            raise ValueError("Bounds invalid: some low > high")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    def clip(self, x: Array) -> Array:
        x = np.asarray(x, dtype=float).reshape(-1)
        return np.minimum(np.maximum(x, self.low), self.high)


def as_vector(x: Any, dim: int, name: str) -> Array:
    """
    Convert scalars/lists/arrays into a 1D numpy vector of expected length.
    Raises a clean error if dimensions don't match.
    """
    v = np.asarray(x, dtype=float).reshape(-1)
    if v.size != dim:
        raise ValueError(f"{name} dimension mismatch: expected {dim}, got {v.size} (value={x})")
    return v


@dataclass
class Trajectory:
    """
    Time-series log of a single simulation run.

    Arrays are shaped:
      t:        (N,)
      r, obs:   (N, ref_dim), (N, obs_dim)
      y:        (N, out_dim)
      u, u_raw: (N, act_dim)  (u_raw is controller output before saturation)
    """
    t: Array
    r: Array
    obs: Array
    y: Array
    u: Array
    u_raw: Array
    info: Dict[str, Any]

    @staticmethod
    def empty(n_steps: int, io: IOSpec) -> "Trajectory":
        t = np.zeros((n_steps,), dtype=float)
        r = np.zeros((n_steps, io.ref_dim), dtype=float)
        obs = np.zeros((n_steps, io.obs_dim), dtype=float)
        y = np.zeros((n_steps, io.out_dim), dtype=float)
        u = np.zeros((n_steps, io.act_dim), dtype=float)
        u_raw = np.zeros((n_steps, io.act_dim), dtype=float)
        return Trajectory(t=t, r=r, obs=obs, y=y, u=u, u_raw=u_raw, info={})


@dataclass(frozen=True)
class SimConfig:
    """Configuration for a simulation rollout."""
    t_final: float
    max_steps: Optional[int] = None  # if None, derived from t_final/dt
    seed: Optional[int] = None