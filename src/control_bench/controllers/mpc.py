from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from control_bench.core.types import Array, Bounds, IOSpec, as_vector


@dataclass(frozen=True)
class MPCConfig:
    """
    Linear SISO MPC with cost:

        J = sum_{i=1..N} qy * (y_{k+i|k} - r)^2
          + sum_{i=0..N-1} ru * (Δu_{k+i})^2
          + sum_{i=0..N-1} qu * (u_{k+i})^2

    where:
        Δu_k = u_k - u_{k-1}
        Δu_{k+i} = u_{k+i} - u_{k+i-1}

    Notes:
    - reference assumed constant over the horizon (good for steps).
    - if u_bounds are provided, we solve using projected gradient descent (PGD).
      Otherwise we solve the unconstrained linear system exactly.
    """
    N: int = 10
    qy: float = 1.0
    ru: float = 1.0
    qu: float = 0.01

    pgd_iters: int = 60
    pgd_step_size: Optional[float] = None


def _normalize_state_space(
    *,
    A: np.ndarray | float,
    B: np.ndarray | float,
    C: np.ndarray | float,
    D: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    A_arr = np.asarray(A, dtype=float)
    if A_arr.ndim == 0:
        A_arr = A_arr.reshape(1, 1)
    if A_arr.ndim != 2 or A_arr.shape[0] != A_arr.shape[1]:
        raise ValueError(f"A must be square, got shape {A_arr.shape}")
    n = int(A_arr.shape[0])

    B_arr = np.asarray(B, dtype=float)
    if B_arr.ndim == 0:
        B_arr = B_arr.reshape(1, 1)
    elif B_arr.ndim == 1:
        if B_arr.size != n:
            raise ValueError(f"B must have length {n}, got {B_arr.size}")
        B_arr = B_arr.reshape(n, 1)
    elif B_arr.shape != (n, 1):
        raise ValueError(f"B must have shape ({n}, 1), got {B_arr.shape}")

    C_arr = np.asarray(C, dtype=float)
    if C_arr.ndim == 0:
        C_arr = C_arr.reshape(1, 1)
    elif C_arr.ndim == 1:
        if C_arr.size != n:
            raise ValueError(f"C must have length {n}, got {C_arr.size}")
        C_arr = C_arr.reshape(1, n)
    elif C_arr.shape != (1, n):
        raise ValueError(f"C must have shape (1, {n}), got {C_arr.shape}")

    D_arr = np.asarray(D, dtype=float)
    if D_arr.ndim == 0:
        D_arr = D_arr.reshape(1, 1)
    elif D_arr.shape != (1, 1):
        raise ValueError(f"D must be scalar or shape (1, 1), got {D_arr.shape}")

    return A_arr, B_arr.reshape(-1), C_arr.reshape(-1), float(D_arr.item())


class LinearMPCController:
    """
    Generic SISO MPC for discrete state-space models:

        x_{k+1} = A x_k + B u_k
        y_{k+1} = C x_{k+1} + D u_k

    State estimate handling:
    - 1-state path: preserve the existing algebraic reconstruction
      x_k = (y_k - D*u_{k-1}) / C
    - multi-state path: maintain an internal nominal x_hat updated by the model
      after each chosen control move. This is intentionally observer-free.
    """

    def __init__(
        self,
        cfg: MPCConfig,
        *,
        dt: float,
        A: np.ndarray | float,
        B: np.ndarray | float,
        C: np.ndarray | float,
        D: np.ndarray | float,
        u_bounds: Optional[Bounds] = None,
        name: str = "MPC",
    ) -> None:
        self.cfg = cfg
        self.dt = float(dt)
        self.u_bounds = u_bounds
        self.name = name

        if self.dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if cfg.N <= 0:
            raise ValueError("MPC horizon N must be > 0")
        if cfg.qy < 0 or cfg.ru < 0 or cfg.qu < 0:
            raise ValueError("MPC weights qy, ru, qu must be >= 0")

        self.A, self.B, self.C, self.D = _normalize_state_space(A=A, B=B, C=C, D=D)
        self.n = int(self.A.shape[0])
        self._scalar_state_mode = (self.n == 1)
        if self._scalar_state_mode and abs(float(self.C[0])) < 1e-12:
            raise ValueError("C is too close to zero; cannot estimate x from y.")

        self.io = IOSpec(ref_dim=1, obs_dim=1, act_dim=1, out_dim=1)
        self.N = int(cfg.N)

        self.F = np.zeros((self.N, self.n), dtype=float)
        self.G = np.zeros((self.N, self.N), dtype=float)

        A_power = np.eye(self.n, dtype=float)
        markov = np.zeros((self.N,), dtype=float)
        for lag in range(self.N):
            markov[lag] = float(self.C @ (A_power @ self.B))
            A_power = A_power @ self.A

        A_power = self.A.copy()
        for i in range(self.N):
            self.F[i, :] = self.C @ A_power
            for j in range(i + 1):
                self.G[i, j] = markov[i - j]
            self.G[i, i] += self.D
            A_power = A_power @ self.A

        self.Q = self.cfg.qy * np.eye(self.N)
        self.S = self.cfg.ru * np.eye(self.N)
        self.R = self.cfg.qu * np.eye(self.N)

        self.E = np.zeros((self.N, self.N))
        for i in range(self.N):
            if i == 0:
                self.E[0, 0] = 1.0
            else:
                self.E[i, i] = 1.0
                self.E[i, i - 1] = -1.0

        self.e_prev = np.zeros((self.N,))
        self.e_prev[0] = 1.0

        self.H = self.G.T @ (self.Q @ self.G) + self.E.T @ (self.S @ self.E) + self.R
        self.Bvec = self.E.T @ (self.S @ self.e_prev)

        self._U_prev = np.zeros((self.N,), dtype=float)
        self.u_prev = 0.0
        self._x_hat = np.zeros((self.n,), dtype=float)

        if self.u_bounds is not None:
            if self.cfg.pgd_iters <= 0:
                raise ValueError("pgd_iters must be > 0 when using bounded MPC")

            if self.cfg.pgd_step_size is None:
                eigs = np.linalg.eigvalsh(self.H)
                L = float(np.max(eigs)) if eigs.size else 1.0
                self._alpha = 1.0 / max(L, 1e-9)
            else:
                self._alpha = float(self.cfg.pgd_step_size)
        else:
            self._alpha = None

    def reset(self) -> None:
        self.u_prev = 0.0
        self._U_prev[:] = 0.0
        self._x_hat[:] = 0.0

    def step(self, r: Array, obs: Array, t: float) -> Array:
        r0 = float(as_vector(r, 1, "r")[0])
        yk = float(as_vector(obs, 1, "obs")[0])

        if self._scalar_state_mode:
            xk = np.array([(yk - self.D * self.u_prev) / self.C[0]], dtype=float)
        else:
            xk = self._x_hat.copy()

        R = np.full((self.N,), r0, dtype=float)
        c = self.F @ xk - R

        g_y = self.G.T @ (self.Q @ c)
        q = g_y - self.Bvec * self.u_prev

        if self.u_bounds is None:
            U = np.linalg.solve(self.H, -q)
        else:
            U = np.empty_like(self._U_prev)
            U[:-1] = self._U_prev[1:]
            U[-1] = self._U_prev[-1]

            umin = float(self.u_bounds.low[0])
            umax = float(self.u_bounds.high[0])
            alpha = float(self._alpha)
            for _ in range(self.cfg.pgd_iters):
                grad = self.H @ U + q
                U = U - alpha * grad
                U = np.clip(U, umin, umax)

        u = float(U[0])

        if not self._scalar_state_mode:
            self._x_hat = (self.A @ xk) + (self.B * u)

        self.u_prev = u
        self._U_prev[:] = U

        return np.array([u], dtype=float)
