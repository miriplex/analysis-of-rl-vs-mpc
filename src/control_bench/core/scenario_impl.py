from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .types import Array, Bounds, IOSpec, as_vector


@dataclass
class BasicScenario:
    """
    A simple scenario:
    - reference: callable r(t)
    - disturbance_u: callable du(t), optional
    - measurement_noise: callable that maps y -> y_noisy, optional
    - saturation: clip using plant bounds (or none if bounds is None)
    """
    io: IOSpec
    reference_fn: Callable[[float], Array]
    disturbance_u_fn: Optional[Callable[[float], Array]] = None
    measurement_fn: Optional[Callable[[Array, float], Array]] = None

    def reset(self, *, seed: Optional[int] = None) -> None:
        if seed is not None:
            np.random.seed(seed)

    def reference(self, t: float) -> Array:
        return as_vector(self.reference_fn(t), self.io.ref_dim, "reference")

    def disturbance_u(self, t: float) -> Array:
        if self.disturbance_u_fn is None:
            return np.zeros((self.io.act_dim,), dtype=float)
        return as_vector(self.disturbance_u_fn(t), self.io.act_dim, "disturbance_u")

    def measurement(self, y: Array, t: float) -> Array:
        if self.measurement_fn is None:
            # default: controller sees y directly (only valid when obs_dim == out_dim)
            return as_vector(y, self.io.obs_dim, "measurement")
        return as_vector(self.measurement_fn(y, t), self.io.obs_dim, "measurement")

    def saturate(self, u_raw: Array, bounds: Optional[Bounds]) -> Array:
        u_raw = as_vector(u_raw, self.io.act_dim, "u_raw")
        if bounds is None:
            return u_raw
        return bounds.clip(u_raw)