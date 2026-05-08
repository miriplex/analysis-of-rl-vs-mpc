from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..core.protocols import Plant
from ..core.types import Array, Bounds, IOSpec, as_vector


@dataclass
class DiscreteLTISISOPlant(Plant):
    """
    Discrete-time SISO LTI plant in state-space form:

        x_{k+1} = A x_k + B u_k
        y_k     = C x_k + D u_k

    Notes
    - This class is deliberately "universal" for any 1-state SISO LTI system.
    - Later, you can generalize to N-state by making A,B,C matrices.
    """
    dt: float
    A: float
    B: float
    C: float
    D: float
    u_bounds: Optional[Bounds] = None
    io: IOSpec = IOSpec(ref_dim=1, obs_dim=1, act_dim=1, out_dim=1)

    # internal state
    _x: float = 0.0
    _last_u: float = 0.0

    def reset(self, *, seed: Optional[int] = None, x0: Optional[Array] = None) -> None:
        if seed is not None:
            np.random.seed(seed)

        if x0 is None:
            self._x = 0.0
        else:
            x0v = np.asarray(x0, dtype=float).reshape(-1)
            if x0v.size != 1:
                raise ValueError(f"x0 must be scalar/length-1 for this plant, got {x0v.size}")
            self._x = float(x0v[0])

        self._last_u = 0.0

    def step(self, u: Array) -> None:
        uv = as_vector(u, 1, "u")[0]
        self._last_u = float(uv)
        self._x = float(self.A * self._x + self.B * uv)

    def output(self) -> Array:
        y = self.C * self._x + self.D * self._last_u
        return np.array([y], dtype=float)

    def observe(self) -> Array:
        # For SISO baseline experiments, controller observes y directly.
        # Later (e.g. pendulum), observe() can return a richer vector.
        return self.output()

    @staticmethod
    def from_continuous_first_order(
        *,
        dt: float,
        p: float,
        k: float,
        z: Optional[float],
        u_bounds: Optional[Bounds] = None,
    ) -> "DiscreteLTISISOPlant":
        """
        Build a discrete-time 1-state plant corresponding to these continuous-time transfer functions:

        If z is None (no zero):
            G(s) = k / (s - p)

        If z is provided:
            G(s) = k (s - z) / (s - p)   (proper but not strictly proper -> includes D term)

        We realize:
            A_c = p, B_c = 1

        For no-zero:
            C = k, D = 0

        For with-zero:
            G(s)=k(1 + (p - z)/(s - p))  =>  D = k, C = k(p - z)
        """
        if dt <= 0:
            raise ValueError("dt must be > 0")

        # exact discretization for scalar A_c = p, B_c = 1
        A_c = float(p)
        if abs(A_c) < 1e-12:
            A_d = 1.0
            B_d = dt
        else:
            A_d = float(np.exp(A_c * dt))
            B_d = float((A_d - 1.0) / A_c)

        if z is None:
            C = float(k)
            D = 0.0
        else:
            C = float(k * (p - z))
            D = float(k)

        return DiscreteLTISISOPlant(
            dt=dt,
            A=A_d,
            B=B_d,
            C=C,
            D=D,
            u_bounds=u_bounds,
        )