from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from control_bench.core.types import Array, Bounds, IOSpec, as_vector
from control_bench.plants.inverted_pendulum import InvertedPendulumParams
from control_bench.plants.inverted_pendulum_nonlinear import continuous_nonlinear_dynamics
from control_bench.plants.second_order_family import expm_small


@dataclass(frozen=True)
class InvertedPendulumMPCConfig:
    """
    Full-state regulation MPC for the linearized inverted pendulum.

    Cost:
        J = sum_{i=1..N} (x_{k+i|k} - r)^T Q (x_{k+i|k} - r)
          + sum_{i=0..N-1} ru * (Δu_{k+i})^2
          + sum_{i=0..N-1} qu * (u_{k+i})^2

    This matches the first-order structure, except the tracking term uses the
    4-state pendulum state vector instead of a scalar output.
    """

    N: int = 60
    q_state_diag: tuple[float, float, float, float] = (1.0, 0.1, 40.0, 4.0)
    ru: float = 2.0
    qu: float = 0.01
    use_terminal_cost: bool = False
    terminal_riccati_iters: int = 500
    terminal_riccati_tol: float = 1e-9
    pgd_iters: int = 80
    pgd_step_size: Optional[float] = None


def _solve_discrete_riccati_iterative(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    *,
    max_iters: int,
    tol: float,
) -> np.ndarray:
    if max_iters <= 0:
        raise ValueError("terminal_riccati_iters must be > 0")
    if tol <= 0.0:
        raise ValueError("terminal_riccati_tol must be > 0")

    P = np.array(Q, dtype=float, copy=True)
    for _ in range(max_iters):
        BtPB = B.T @ P @ B
        gain_denom = R + BtPB
        inv_term = np.linalg.inv(gain_denom)
        P_next = A.T @ P @ A - A.T @ P @ B @ inv_term @ B.T @ P @ A + Q
        if float(np.max(np.abs(P_next - P))) <= tol:
            P = P_next
            break
        P = P_next
    return 0.5 * (P + P.T)


def _solve_augmented_delta_u_terminal_matrix(
    A: np.ndarray,
    B: np.ndarray,
    Qx: np.ndarray,
    *,
    ru: float,
    qu: float,
    max_iters: int,
    tol: float,
) -> np.ndarray:
    n = int(A.shape[0])
    A_aug = np.block([[A, B], [np.zeros((1, n)), np.ones((1, 1))]])
    B_aug = np.vstack([B, np.ones((1, 1))])

    Q_aug = np.zeros((n + 1, n + 1), dtype=float)
    Q_aug[:n, :n] = Qx
    Q_aug[n, n] = float(qu)

    R_aug = np.array([[max(float(ru + qu), 1e-9)]], dtype=float)
    N_aug = np.zeros((n + 1, 1), dtype=float)
    N_aug[n, 0] = float(qu)

    R_inv = np.linalg.inv(R_aug)
    A_bar = A_aug - B_aug @ (R_inv @ N_aug.T)
    Q_bar = Q_aug - N_aug @ (R_inv @ N_aug.T)
    P = _solve_discrete_riccati_iterative(
        A_bar,
        B_aug,
        Q_bar,
        R_aug,
        max_iters=max_iters,
        tol=tol,
    )
    return 0.5 * (P + P.T)


def _build_prediction_matrices(
    A: np.ndarray,
    B: np.ndarray,
    *,
    N: int,
    c: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(A.shape[0])
    F = np.zeros((N * n, n), dtype=float)
    G = np.zeros((N * n, N), dtype=float)
    Cbar = np.zeros((N * n,), dtype=float)

    A_power = A.copy()
    c_step = np.zeros((n,), dtype=float) if c is None else np.asarray(c, dtype=float).reshape(n)
    c_accum = c_step.copy()
    for i in range(N):
        row = slice(i * n, (i + 1) * n)
        F[row, :] = A_power
        Cbar[row] = c_accum
        for j in range(i + 1):
            lag_power = np.linalg.matrix_power(A, i - j)
            G[row, j] = (lag_power @ B).reshape(-1)
        A_power = A_power @ A
        c_accum = A @ c_accum + c_step
    return F, G, Cbar


def _continuous_jacobians_numerical(
    params: InvertedPendulumParams,
    x_bar: np.ndarray,
    u_bar: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_bar = np.asarray(x_bar, dtype=float).reshape(4)
    u_bar = float(u_bar)
    f_bar = continuous_nonlinear_dynamics(params, x_bar, u_bar)

    n = x_bar.size
    A_c = np.zeros((n, n), dtype=float)
    for i in range(n):
        step = 1e-6 * max(1.0, abs(float(x_bar[i])))
        xp = x_bar.copy()
        xm = x_bar.copy()
        xp[i] += step
        xm[i] -= step
        fp = continuous_nonlinear_dynamics(params, xp, u_bar)
        fm = continuous_nonlinear_dynamics(params, xm, u_bar)
        A_c[:, i] = (fp - fm) / (2.0 * step)

    step_u = 1e-6 * max(1.0, abs(u_bar))
    fp = continuous_nonlinear_dynamics(params, x_bar, u_bar + step_u)
    fm = continuous_nonlinear_dynamics(params, x_bar, u_bar - step_u)
    B_c = ((fp - fm) / (2.0 * step_u)).reshape(n, 1)
    c_c = f_bar - A_c @ x_bar - B_c[:, 0] * u_bar
    return A_c, B_c, c_c


def _discretize_affine_exact(
    A_c: np.ndarray,
    B_c: np.ndarray,
    c_c: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(A_c.shape[0])
    aug = np.zeros((n + 2, n + 2), dtype=float)
    aug[:n, :n] = A_c
    aug[:n, n : n + 1] = B_c.reshape(n, 1)
    aug[:n, n + 1] = np.asarray(c_c, dtype=float).reshape(n)
    exp_aug = np.asarray(expm_small(aug * float(dt)), dtype=float)
    A_d = exp_aug[:n, :n]
    B_d = exp_aug[:n, n : n + 1]
    c_d = exp_aug[:n, n + 1]
    return A_d, B_d, c_d


class InvertedPendulumMPCController:
    def __init__(
        self,
        cfg: InvertedPendulumMPCConfig,
        *,
        dt: float,
        A: np.ndarray,
        B: np.ndarray,
        u_bounds: Optional[Bounds] = None,
        name: str = "MPC",
    ) -> None:
        self.cfg = cfg
        self.dt = float(dt)
        self.u_bounds = u_bounds
        self.name = name

        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        if self.A.shape != (4, 4):
            raise ValueError(f"A must have shape (4,4), got {self.A.shape}")
        if self.B.ndim == 1:
            if self.B.size != 4:
                raise ValueError(f"B must have length 4, got {self.B.size}")
            self.B = self.B.reshape(4, 1)
        elif self.B.shape != (4, 1):
            raise ValueError(f"B must have shape (4,1), got {self.B.shape}")

        if self.dt <= 0.0:
            raise ValueError("dt must be > 0")
        if self.cfg.N <= 0:
            raise ValueError("MPC horizon must be > 0")
        if self.cfg.ru < 0.0 or self.cfg.qu < 0.0:
            raise ValueError("ru and qu must be >= 0")

        q_diag = np.asarray(self.cfg.q_state_diag, dtype=float).reshape(-1)
        if q_diag.shape != (4,):
            raise ValueError("q_state_diag must have length 4")
        if np.any(q_diag < 0.0):
            raise ValueError("q_state_diag entries must be >= 0")

        self.io = IOSpec(ref_dim=4, obs_dim=4, act_dim=1, out_dim=4)
        self.n = 4
        self.N = int(self.cfg.N)

        self.F, self.G, self.Cbar = _build_prediction_matrices(self.A, self.B, N=self.N)

        q_block = np.diag(q_diag)
        self.Qbar = np.kron(np.eye(self.N), q_block)
        self.terminal_P_aug = None
        if self.cfg.use_terminal_cost:
            self.terminal_P_aug = _solve_augmented_delta_u_terminal_matrix(
                self.A,
                self.B,
                q_block,
                ru=self.cfg.ru,
                qu=self.cfg.qu,
                max_iters=int(self.cfg.terminal_riccati_iters),
                tol=float(self.cfg.terminal_riccati_tol),
            )
        self.S = self.cfg.ru * np.eye(self.N)
        self.R = self.cfg.qu * np.eye(self.N)

        self.E = np.zeros((self.N, self.N), dtype=float)
        for i in range(self.N):
            self.E[i, i] = 1.0
            if i > 0:
                self.E[i, i - 1] = -1.0

        self.e_prev = np.zeros((self.N,), dtype=float)
        self.e_prev[0] = 1.0
        self.e_last = np.zeros((self.N,), dtype=float)
        self.e_last[-1] = 1.0

        self.H = self.G.T @ (self.Qbar @ self.G) + self.E.T @ (self.S @ self.E) + self.R
        self.Bvec = self.E.T @ (self.S @ self.e_prev)

        self._U_prev = np.zeros((self.N,), dtype=float)
        self.u_prev = 0.0

        if self.u_bounds is not None:
            if self.cfg.pgd_iters <= 0:
                raise ValueError("pgd_iters must be > 0 when bounds are enabled")
            if self.cfg.pgd_step_size is None:
                eigs = np.linalg.eigvalsh(self.H)
                lipschitz = float(np.max(eigs)) if eigs.size else 1.0
                self._alpha = 1.0 / max(lipschitz, 1e-9)
            else:
                self._alpha = float(self.cfg.pgd_step_size)
        else:
            self._alpha = None

    def reset(self) -> None:
        self._U_prev[:] = 0.0
        self.u_prev = 0.0

    def step(self, r: Array, obs: Array, t: float) -> Array:
        r_vec = as_vector(r, 4, "r")
        xk = as_vector(obs, 4, "obs")

        R_stack = np.tile(r_vec, self.N)
        c = self.F @ xk + self.Cbar - R_stack
        g_x = self.G.T @ (self.Qbar @ c)
        q = g_x - self.Bvec * self.u_prev

        if self.terminal_P_aug is not None:
            F_last = self.F[-self.n :, :]
            G_last = self.G[-self.n :, :]
            C_last = self.Cbar[-self.n :]
            Gz = np.vstack([G_last, self.e_last.reshape(1, -1)])
            fz = np.concatenate([F_last @ xk + C_last, np.array([0.0])])
            q = q + Gz.T @ (self.terminal_P_aug @ fz)
            H = self.H + Gz.T @ (self.terminal_P_aug @ Gz)
        else:
            H = self.H

        if self.u_bounds is None:
            U = np.linalg.solve(H, -q)
        else:
            U = np.empty_like(self._U_prev)
            U[:-1] = self._U_prev[1:]
            U[-1] = self._U_prev[-1]

            umin = float(self.u_bounds.low[0])
            umax = float(self.u_bounds.high[0])
            if self.cfg.pgd_step_size is None:
                eigs = np.linalg.eigvalsh(H)
                lipschitz = float(np.max(eigs)) if eigs.size else 1.0
                alpha = 1.0 / max(lipschitz, 1e-9)
            else:
                alpha = float(self._alpha)
            for _ in range(self.cfg.pgd_iters):
                grad = H @ U + q
                U = np.clip(U - alpha * grad, umin, umax)

        u = float(U[0])
        self.u_prev = u
        self._U_prev[:] = U
        return np.array([u], dtype=float)


class SuccessiveLinearizedInvertedPendulumMPCController:
    def __init__(
        self,
        cfg: InvertedPendulumMPCConfig,
        *,
        dt: float,
        params: InvertedPendulumParams,
        u_bounds: Optional[Bounds] = None,
        name: str = "MPC (successive linearization)",
    ) -> None:
        self.cfg = cfg
        self.dt = float(dt)
        self.params = params
        self.u_bounds = u_bounds
        self.name = name

        if self.dt <= 0.0:
            raise ValueError("dt must be > 0")
        if self.cfg.N <= 0:
            raise ValueError("MPC horizon must be > 0")
        if self.cfg.ru < 0.0 or self.cfg.qu < 0.0:
            raise ValueError("ru and qu must be >= 0")

        q_diag = np.asarray(self.cfg.q_state_diag, dtype=float).reshape(-1)
        if q_diag.shape != (4,):
            raise ValueError("q_state_diag must have length 4")
        if np.any(q_diag < 0.0):
            raise ValueError("q_state_diag entries must be >= 0")

        self.io = IOSpec(ref_dim=4, obs_dim=4, act_dim=1, out_dim=4)
        self.n = 4
        self.N = int(self.cfg.N)
        self.q_block = np.diag(q_diag)
        self.S = self.cfg.ru * np.eye(self.N)
        self.R = self.cfg.qu * np.eye(self.N)

        self.E = np.zeros((self.N, self.N), dtype=float)
        for i in range(self.N):
            self.E[i, i] = 1.0
            if i > 0:
                self.E[i, i - 1] = -1.0

        self.e_prev = np.zeros((self.N,), dtype=float)
        self.e_prev[0] = 1.0
        self.Bvec = self.E.T @ (self.S @ self.e_prev)
        self._U_prev = np.zeros((self.N,), dtype=float)
        self.u_prev = 0.0

    def reset(self) -> None:
        self._U_prev[:] = 0.0
        self.u_prev = 0.0

    def _qbar(self, A_d: np.ndarray, B_d: np.ndarray) -> np.ndarray:
        Qbar = np.kron(np.eye(self.N), self.q_block)
        if self.cfg.use_terminal_cost:
            r_eff = max(float(self.cfg.ru + self.cfg.qu), 1e-9)
            P = _solve_discrete_riccati_iterative(
                A_d,
                B_d,
                self.q_block,
                np.array([[r_eff]], dtype=float),
                max_iters=int(self.cfg.terminal_riccati_iters),
                tol=float(self.cfg.terminal_riccati_tol),
            )
            Qbar[-self.n :, -self.n :] += P
        return Qbar

    def step(self, r: Array, obs: Array, t: float) -> Array:
        r_vec = as_vector(r, 4, "r")
        xk = as_vector(obs, 4, "obs")

        A_c, B_c, c_c = _continuous_jacobians_numerical(self.params, xk, self.u_prev)
        A_d, B_d, c_d = _discretize_affine_exact(A_c, B_c, c_c, self.dt)
        F, G, Cbar = _build_prediction_matrices(A_d, B_d, N=self.N, c=c_d)
        Qbar = self._qbar(A_d, B_d)

        R_stack = np.tile(r_vec, self.N)
        c = F @ xk + Cbar - R_stack
        H = G.T @ (Qbar @ G) + self.E.T @ (self.S @ self.E) + self.R
        q = G.T @ (Qbar @ c) - self.Bvec * self.u_prev

        if self.u_bounds is None:
            U = np.linalg.solve(H, -q)
        else:
            U = np.empty_like(self._U_prev)
            U[:-1] = self._U_prev[1:]
            U[-1] = self._U_prev[-1]

            if self.cfg.pgd_step_size is None:
                eigs = np.linalg.eigvalsh(H)
                lipschitz = float(np.max(eigs)) if eigs.size else 1.0
                alpha = 1.0 / max(lipschitz, 1e-9)
            else:
                alpha = float(self.cfg.pgd_step_size)

            umin = float(self.u_bounds.low[0])
            umax = float(self.u_bounds.high[0])
            for _ in range(self.cfg.pgd_iters):
                grad = H @ U + q
                U = np.clip(U - alpha * grad, umin, umax)

        u = float(U[0])
        self.u_prev = u
        self._U_prev[:] = U
        return np.array([u], dtype=float)
