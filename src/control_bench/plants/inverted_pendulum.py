from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

from ..core.protocols import Plant
from ..core.types import Array, Bounds, IOSpec, as_vector
from .second_order_family import expm_small


@dataclass(frozen=True)
class InvertedPendulumParams:
    """
    Linearized cart-pole around the upright equilibrium.

    State:
        x = [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

    Angle convention:
        pole_angle = 0 means upright.
        Positive angle rotates the pendulum to the cart's left.

    Continuous-time linearized model:
        x_dot = A_c x + B_c u

    The standard small-angle linearization around the origin uses:
        p = I (M + m) + M m l^2

        A_c = [[0, 1, 0, 0],
               [0, -(I + m l^2) b / p,  m^2 g l^2 / p, 0],
               [0, 0, 0, 1],
               [0, -m l b / p, m g l (M + m) / p, 0]]

        B_c = [[0],
               [(I + m l^2) / p],
               [0],
               [m l / p]]

    This is the usual unstable upright linearization for a force-actuated cart.
    """

    dt: float = 0.02
    cart_mass: float = 0.5
    pole_mass: float = 0.2
    cart_damping: float = 0.1
    pole_length: float = 0.3
    pole_inertia: float = 0.006
    gravity: float = 9.8
    u_min: Optional[float] = None
    u_max: Optional[float] = None


def _maybe_make_bounds(params: InvertedPendulumParams) -> Optional[Bounds]:
    if params.u_min is None and params.u_max is None:
        return None
    if params.u_min is None or params.u_max is None:
        raise ValueError("If using pendulum bounds, provide both u_min and u_max.")
    return Bounds(low=[params.u_min], high=[params.u_max])


def continuous_linearized_matrices(
    params: InvertedPendulumParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    M = float(params.cart_mass)
    m = float(params.pole_mass)
    b = float(params.cart_damping)
    l = float(params.pole_length)
    I = float(params.pole_inertia)
    g = float(params.gravity)

    p = I * (M + m) + M * m * l * l
    if abs(p) < 1e-12:
        raise ValueError("Inverted pendulum linearization is singular for the provided parameters.")

    A_c = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -((I + m * l * l) * b) / p, (m * m * g * l * l) / p, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, -(m * l * b) / p, (m * g * l * (M + m)) / p, 0.0],
        ],
        dtype=float,
    )
    B_c = np.array(
        [
            [0.0],
            [(I + m * l * l) / p],
            [0.0],
            [(m * l) / p],
        ],
        dtype=float,
    )
    C_c = np.eye(4, dtype=float)
    D_c = np.zeros((4, 1), dtype=float)
    return A_c, B_c, C_c, D_c


def _discretize_exact(A_c: np.ndarray, B_c: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    dt = float(dt)
    if dt <= 0.0:
        raise ValueError("dt must be > 0")

    A_c = np.asarray(A_c, dtype=float)
    B_c = np.asarray(B_c, dtype=float)
    n = int(A_c.shape[0])

    aug = np.zeros((n + 1, n + 1), dtype=float)
    aug[:n, :n] = A_c
    aug[:n, n:] = B_c

    exp_aug = np.asarray(expm_small(aug * dt), dtype=float)
    A_d = exp_aug[:n, :n]
    B_d = exp_aug[:n, n:]
    return A_d, B_d


@dataclass
class LinearizedInvertedPendulumPlant(Plant):
    dt: float
    A: Array
    B: Array
    C: Array
    D: Array
    pole_length: float
    u_bounds: Optional[Bounds] = None
    io: IOSpec = IOSpec(ref_dim=4, obs_dim=4, act_dim=1, out_dim=4)

    _x: Array = field(default_factory=lambda: np.zeros((4,), dtype=float))

    def __post_init__(self) -> None:
        self.dt = float(self.dt)
        self.A = np.asarray(self.A, dtype=float)
        self.B = np.asarray(self.B, dtype=float).reshape(4, 1)
        self.C = np.asarray(self.C, dtype=float).reshape(4, 4)
        self.D = np.asarray(self.D, dtype=float).reshape(4, 1)
        self.pole_length = float(self.pole_length)
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

    def step(self, u: Array) -> None:
        uv = float(as_vector(u, 1, "u")[0])
        if self.u_bounds is not None:
            uv = float(self.u_bounds.clip(np.array([uv], dtype=float))[0])
        self._x = (self.A @ self._x) + (self.B[:, 0] * uv)

    def observe(self) -> Array:
        return self._x.astype(float, copy=True)

    def output(self) -> Array:
        return (self.C @ self._x.reshape(-1, 1)).reshape(-1)

    @property
    def state(self) -> np.ndarray:
        return self._x.astype(float, copy=True)

    def cart_pole_points(self) -> Tuple[float, float, float, float]:
        cart_x = float(self._x[0])
        theta = float(self._x[2])
        pivot_x = cart_x
        pivot_y = 0.0
        bob_x = pivot_x - self.pole_length * np.sin(theta)
        bob_y = pivot_y + self.pole_length * np.cos(theta)
        return pivot_x, pivot_y, bob_x, bob_y


def build_linearized_inverted_pendulum(
    params: InvertedPendulumParams,
) -> LinearizedInvertedPendulumPlant:
    A_c, B_c, C_c, D_c = continuous_linearized_matrices(params)
    A_d, B_d = _discretize_exact(A_c, B_c, float(params.dt))
    return LinearizedInvertedPendulumPlant(
        dt=float(params.dt),
        A=A_d,
        B=B_d,
        C=C_c,
        D=D_c,
        pole_length=float(params.pole_length),
        u_bounds=_maybe_make_bounds(params),
    )
