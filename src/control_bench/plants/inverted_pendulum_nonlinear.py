from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from ..core.protocols import Plant
from ..core.types import Array, Bounds, IOSpec, as_vector
from .inverted_pendulum import InvertedPendulumParams


def _maybe_make_bounds(params: InvertedPendulumParams) -> Optional[Bounds]:
    if params.u_min is None and params.u_max is None:
        return None
    if params.u_min is None or params.u_max is None:
        raise ValueError("If using pendulum bounds, provide both u_min and u_max.")
    return Bounds(low=[params.u_min], high=[params.u_max])


def continuous_nonlinear_dynamics(
    params: InvertedPendulumParams,
    x: Array,
    u: float,
) -> np.ndarray:
    """
    Nonlinear cart-pole dynamics around the upright angle convention.

    State:
        x = [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

    Angle convention:
        pole_angle = 0 means upright.
        Positive angle rotates the pendulum to the cart's left.

    Dynamics:
        (M + m) x_ddot + b x_dot - m l cos(theta) theta_ddot + m l sin(theta) theta_dot^2 = u
        (I + m l^2) theta_ddot - m l cos(theta) x_ddot - m g l sin(theta) = 0

    These equations are solved exactly as a 2x2 system for x_ddot and theta_ddot.
    Their small-angle linearization matches the linearized pendulum model used elsewhere.
    """

    xv = np.asarray(x, dtype=float).reshape(-1)
    if xv.size != 4:
        raise ValueError(f"x must have length 4, got {xv.size}")

    _, x_dot, theta, theta_dot = xv
    M = float(params.cart_mass)
    m = float(params.pole_mass)
    b = float(params.cart_damping)
    l = float(params.pole_length)
    I = float(params.pole_inertia)
    g = float(params.gravity)

    sin_theta = float(np.sin(theta))
    cos_theta = float(np.cos(theta))

    rhs_1 = float(u) - b * float(x_dot) - m * l * sin_theta * float(theta_dot) ** 2
    rhs_2 = m * g * l * sin_theta

    denom = (M + m) * (I + m * l * l) - (m * l * cos_theta) ** 2
    if abs(denom) < 1e-12:
        raise ValueError("Nonlinear inverted pendulum dynamics became singular.")

    x_ddot = ((I + m * l * l) * rhs_1 + (m * l * cos_theta) * rhs_2) / denom
    theta_ddot = ((m * l * cos_theta) * rhs_1 + (M + m) * rhs_2) / denom

    return np.array([x_dot, x_ddot, theta_dot, theta_ddot], dtype=float)


@dataclass
class NonlinearInvertedPendulumPlant(Plant):
    dt: float
    params: InvertedPendulumParams
    u_bounds: Optional[Bounds] = None
    io: IOSpec = IOSpec(ref_dim=4, obs_dim=4, act_dim=1, out_dim=4)

    _x: Array = field(default_factory=lambda: np.zeros((4,), dtype=float))

    def __post_init__(self) -> None:
        self.dt = float(self.dt)
        self.params = InvertedPendulumParams(**self.params.__dict__)
        if self._x.shape != (4,):
            self._x = np.zeros((4,), dtype=float)

    def reset(self, *, seed: Optional[int] = None, x0: Optional[Array] = None) -> None:
        if seed is not None:
            np.random.seed(seed)

        if x0 is None:
            self._x = np.zeros((4,), dtype=float)
            return

        x0v = np.asarray(x0, dtype=float).reshape(-1)
        if x0v.size != 4:
            raise ValueError(f"x0 must have length 4, got {x0v.size}")
        self._x = x0v.astype(float, copy=True)

    def _rhs(self, x: np.ndarray, u: float) -> np.ndarray:
        return continuous_nonlinear_dynamics(self.params, x, u)

    def step(self, u: Array) -> None:
        uv = float(as_vector(u, 1, "u")[0])
        if self.u_bounds is not None:
            uv = float(self.u_bounds.clip(np.array([uv], dtype=float))[0])

        dt = self.dt
        x = self._x
        k1 = self._rhs(x, uv)
        k2 = self._rhs(x + 0.5 * dt * k1, uv)
        k3 = self._rhs(x + 0.5 * dt * k2, uv)
        k4 = self._rhs(x + dt * k3, uv)
        self._x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def observe(self) -> Array:
        return self._x.astype(float, copy=True)

    def output(self) -> Array:
        return self._x.astype(float, copy=True)

    @property
    def state(self) -> np.ndarray:
        return self._x.astype(float, copy=True)

    def cart_pole_points(self) -> Tuple[float, float, float, float]:
        cart_x = float(self._x[0])
        theta = float(self._x[2])
        pivot_x = cart_x
        pivot_y = 0.0
        bob_x = pivot_x - float(self.params.pole_length) * np.sin(theta)
        bob_y = pivot_y + float(self.params.pole_length) * np.cos(theta)
        return pivot_x, pivot_y, bob_x, bob_y


def build_nonlinear_inverted_pendulum(
    params: InvertedPendulumParams,
) -> NonlinearInvertedPendulumPlant:
    return NonlinearInvertedPendulumPlant(
        dt=float(params.dt),
        params=params,
        u_bounds=_maybe_make_bounds(params),
    )
