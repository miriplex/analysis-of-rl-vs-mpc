from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np

from .types import Array, Bounds, IOSpec


@runtime_checkable
class Plant(Protocol):
    """
    Minimal plant interface.

    - observe(): what the controller sees (can be output y or a richer state-like vector)
    - output():  what you evaluate against the reference (tracking metrics)
    """
    dt: float
    io: IOSpec
    u_bounds: Optional[Bounds]

    def reset(self, *, seed: Optional[int] = None, x0: Optional[Array] = None) -> None:
        ...

    def step(self, u: Array) -> None:
        ...

    def observe(self) -> Array:
        ...

    def output(self) -> Array:
        ...


@runtime_checkable
class Controller(Protocol):
    """
    Minimal controller interface.

    Controllers can maintain internal state (integrators, filters, MPC warm-start, RNN hidden state, etc).
    """
    io: IOSpec  # expected dims: ref_dim/obs_dim/act_dim; out_dim is irrelevant for controllers

    def reset(self) -> None:
        ...

    def step(self, r: Array, obs: Array, t: float) -> Array:
        ...


class Scenario(Protocol):
    """
    Scenario defines reference signals + optional noise/disturbances/constraints.

    Keep this deliberately lightweight: it should not "own" the plant, only modulate signals.
    """

    def reset(self, *, seed: Optional[int] = None) -> None:
        ...

    def reference(self, t: float) -> Array:
        ...

    def disturbance_u(self, t: float) -> Array:
        """Additive disturbance on control input (default: zero)."""
        ...

    def measurement(self, y: Array, t: float) -> Array:
        """Return the observation after measurement noise/distortion (default: y)."""
        ...

    def saturate(self, u_raw: Array, bounds: Optional[Bounds]) -> Array:
        """Apply actuator constraints (default: clip to plant bounds if provided)."""
        ...