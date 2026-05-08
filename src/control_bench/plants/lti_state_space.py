from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.protocols import Plant
from ..core.types import Array, Bounds, IOSpec, as_vector


@dataclass
class DiscreteLTIPlant(Plant):
    """
    Generic discrete-time, SISO, n-state LTI plant:

        x_{k+1} = A x_k + B u_k
        y_k     = C x_k + D u_{k-1}

    Notes
    - The output uses the **last applied input** (u_{k-1}) to avoid an algebraic loop.
      This matches the simulator convention used throughout this repo:
        - the simulator reads y_k via plant.output()
        - then computes u_k
        - then advances the plant with plant.step(u_k)
    - observe() returns output() for now (baseline SISO setting).
    """

    dt: float
    A: Array  # (n, n)
    B: Array  # (n, 1)
    C: Array  # (1, n)
    D: Array  # (1, 1)
    u_bounds: Optional[Bounds] = None
    io: IOSpec = IOSpec(ref_dim=1, obs_dim=1, act_dim=1, out_dim=1)

    # internal state
    _x: Array = field(default_factory=lambda: np.zeros((0,), dtype=float))
    _last_u: float = 0.0

    def __post_init__(self) -> None:
        self.dt = float(self.dt)
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")

        A = np.asarray(self.A, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"A must be square (n,n), got shape {A.shape}")
        n = int(A.shape[0])

        B = np.asarray(self.B, dtype=float)
        if B.ndim == 1:
            if B.size != n:
                raise ValueError(f"B must have length n={n}, got {B.size}")
            B = B.reshape(n, 1)
        elif B.shape != (n, 1):
            raise ValueError(f"B must have shape (n,1) with n={n}, got {B.shape}")

        C = np.asarray(self.C, dtype=float)
        if C.ndim == 1:
            if C.size != n:
                raise ValueError(f"C must have length n={n}, got {C.size}")
            C = C.reshape(1, n)
        elif C.shape != (1, n):
            raise ValueError(f"C must have shape (1,n) with n={n}, got {C.shape}")

        D = np.asarray(self.D, dtype=float)
        if D.ndim == 0:
            D = D.reshape(1, 1)
        elif D.shape != (1, 1):
            raise ValueError(f"D must be scalar or shape (1,1), got {D.shape}")

        self.A = A
        self.B = B
        self.C = C
        self.D = D

        # initialize state to zero of correct dimension
        if self._x.size != n:
            self._x = np.zeros((n,), dtype=float)

    def reset(self, *, seed: Optional[int] = None, x0: Optional[Array] = None) -> None:
        if seed is not None:
            np.random.seed(seed)

        n = int(self.A.shape[0])
        if x0 is None:
            self._x = np.zeros((n,), dtype=float)
        else:
            x0v = np.asarray(x0, dtype=float).reshape(-1)
            if x0v.size != n:
                raise ValueError(f"x0 must have length n={n}, got {x0v.size}")
            self._x = x0v.astype(float, copy=True)

        self._last_u = 0.0

    def step(self, u: Array) -> None:
        uv = float(as_vector(u, 1, "u")[0])
        self._last_u = uv
        self._x = (self.A @ self._x) + (self.B.reshape(-1) * uv)

    def output(self) -> Array:
        y_state = float((self.C @ self._x.reshape(-1, 1)).item())
        y_input = float(self.D.reshape(1, 1).item()) * float(self._last_u)
        return np.array([y_state + y_input], dtype=float)

    def observe(self) -> Array:
        # Baseline: controller observes y directly.
        return self.output()

