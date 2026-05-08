from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from control_bench.core.types import Array, Bounds, IOSpec, as_vector
from .inverted_pendulum_mpc import _solve_augmented_delta_u_terminal_matrix
from .rl_numpy.optim import Adam
from .rl_numpy.policies import LinearFeaturePolicy, MLPPolicy, MLPPolicySpec
from .rl_numpy.training import load_policy_npz


PENDULUM_PID_INPUT_DIM = 5
PENDULUM_STATE6_INPUT_DIM = 6
PENDULUM_RICH_INPUT_DIM = PENDULUM_STATE6_INPUT_DIM
PENDULUM_HISTORY8_INPUT_DIM = 8
PENDULUM_RICH11_INPUT_DIM = 11

Params = Dict[str, np.ndarray]
Grads = Dict[str, np.ndarray]


@dataclass(frozen=True)
class InvertedPendulumRLBPTTConfig:
    horizon_steps: int = 400
    q_state_diag: tuple[float, float, float, float] = (1.0, 0.1, 40.0, 4.0)
    ru: float = 2.0
    qu: float = 0.01
    u_min: Optional[float] = None
    u_max: Optional[float] = None
    use_terminal_cost: bool = False
    terminal_riccati_iters: int = 500
    terminal_riccati_tol: float = 1e-9
    terminal_P_aug: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass(frozen=True)
class InvertedPendulumRolloutSpec:
    r: np.ndarray
    du: np.ndarray
    x0: np.ndarray
    mode: str = "regulation"

    def __post_init__(self) -> None:
        r = np.asarray(self.r, dtype=np.float64)
        du = np.asarray(self.du, dtype=np.float64).reshape(-1)
        x0 = np.asarray(self.x0, dtype=np.float64).reshape(-1)

        if r.ndim != 2 or r.shape[1] != 4:
            raise ValueError(f"RolloutSpec.r must have shape (T, 4), got {r.shape}")
        if du.ndim != 1 or du.shape[0] != r.shape[0]:
            raise ValueError(f"RolloutSpec.du must have shape ({r.shape[0]},), got {du.shape}")
        if x0.shape != (4,):
            raise ValueError(f"RolloutSpec.x0 must have shape (4,), got {x0.shape}")

        object.__setattr__(self, "r", r)
        object.__setattr__(self, "du", du)
        object.__setattr__(self, "x0", x0)


@dataclass(frozen=True)
class InvertedPendulumRLRuntimeConfig:
    dt: float
    kind: str  # "pidfeat", "state6"/"rich", "history8", or "rich11_pendulum"
    u_min: Optional[float] = None
    u_max: Optional[float] = None


def _normalize_kind(kind: str) -> str:
    kind = str(kind)
    if kind == "rich":
        return "state6"
    if kind == "rich11":
        return "rich11_pendulum"
    return kind


def pendulum_feature_dim(kind: str) -> int:
    kind = _normalize_kind(kind)
    if kind == "pidfeat":
        return PENDULUM_PID_INPUT_DIM
    if kind == "state6":
        return PENDULUM_STATE6_INPUT_DIM
    if kind == "history8":
        return PENDULUM_HISTORY8_INPUT_DIM
    if kind == "rich11_pendulum":
        return PENDULUM_RICH11_INPUT_DIM
    raise ValueError(f"Unknown pendulum RL kind: {kind}")


def _clip_gradients(grads: Grads, max_norm: Optional[float]) -> Grads:
    if max_norm is None:
        return grads
    if max_norm <= 0:
        raise ValueError("max_norm must be > 0 when gradient clipping is enabled")

    total_sq_norm = 0.0
    for grad in grads.values():
        total_sq_norm += float(np.sum(np.square(grad)))

    total_norm = float(np.sqrt(total_sq_norm))
    if total_norm <= max_norm:
        return grads

    scale = float(max_norm / (total_norm + 1e-12))
    return {key: grad * scale for key, grad in grads.items()}


def _prepare_rollout(
    *,
    cfg: InvertedPendulumRLBPTTConfig,
    r_seq: Optional[np.ndarray],
    du_seq: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    T = int(cfg.horizon_steps)

    if r_seq is None:
        r = np.zeros((T, 4), dtype=np.float64)
    else:
        r = np.asarray(r_seq, dtype=np.float64)
        if r.shape != (T, 4):
            raise ValueError(f"r_seq must have shape ({T}, 4), got {r.shape}")

    if du_seq is None:
        du = np.zeros((T,), dtype=np.float64)
    else:
        du = np.asarray(du_seq, dtype=np.float64).reshape(-1)
        if du.shape != (T,):
            raise ValueError(f"du_seq must have shape ({T},), got {du.shape}")

    return r, du


def _state_vector(x0: np.ndarray | float) -> np.ndarray:
    x0_vec = np.asarray(x0, dtype=np.float64).reshape(-1)
    if x0_vec.shape != (4,):
        raise ValueError(f"x0 must have shape (4,), got {x0_vec.shape}")
    return x0_vec.astype(np.float64, copy=True)


def build_terminal_cost_matrix(
    *,
    A: np.ndarray,
    B: np.ndarray,
    cfg: InvertedPendulumRLBPTTConfig,
) -> Optional[np.ndarray]:
    if not bool(cfg.use_terminal_cost):
        return None
    q_diag = np.asarray(cfg.q_state_diag, dtype=np.float64).reshape(4)
    q_block = np.diag(q_diag)
    A = np.asarray(A, dtype=np.float64).reshape(4, 4)
    B = np.asarray(B, dtype=np.float64).reshape(4, 1)
    return _solve_augmented_delta_u_terminal_matrix(
        A,
        B,
        q_block,
        ru=float(cfg.ru),
        qu=float(cfg.qu),
        max_iters=int(cfg.terminal_riccati_iters),
        tol=float(cfg.terminal_riccati_tol),
    )


def _terminal_cost_and_gradients(
    *,
    terminal_P_aug: Optional[np.ndarray],
    x_terminal: np.ndarray,
    u_terminal_prev: float,
    scale: float,
) -> tuple[float, np.ndarray, float]:
    if terminal_P_aug is None:
        return 0.0, np.zeros((4,), dtype=np.float64), 0.0
    z = np.concatenate([np.asarray(x_terminal, dtype=np.float64).reshape(4), np.array([float(u_terminal_prev)])])
    Pz = terminal_P_aug @ z
    terminal_cost = float(scale * (z @ Pz))
    grad = 2.0 * scale * Pz
    return terminal_cost, grad[:4].copy(), float(grad[4])


def _resolved_terminal_cost_matrix(
    *,
    A: np.ndarray,
    B: np.ndarray,
    cfg: InvertedPendulumRLBPTTConfig,
) -> Optional[np.ndarray]:
    if not bool(cfg.use_terminal_cost):
        return None
    if cfg.terminal_P_aug is not None:
        return np.asarray(cfg.terminal_P_aug, dtype=np.float64)
    return build_terminal_cost_matrix(A=A, B=B, cfg=cfg)


def _pid_features(error_state: np.ndarray, int_theta_error: float) -> np.ndarray:
    return np.array(
        [
            error_state[2],  # theta error
            int_theta_error,
            error_state[3],  # theta_dot error
            error_state[0],  # x error
            error_state[1],  # x_dot error
        ],
        dtype=np.float64,
    )


def _state6_features(error_state: np.ndarray, int_theta_error: float, u_prev: float) -> np.ndarray:
    return np.array(
        [
            error_state[0],
            error_state[1],
            error_state[2],
            error_state[3],
            int_theta_error,
            u_prev,
        ],
        dtype=np.float64,
    )


def _rich_features(error_state: np.ndarray, int_theta_error: float, u_prev: float) -> np.ndarray:
    return _state6_features(error_state, int_theta_error, u_prev)


def _history8_features(
    error_state: np.ndarray,
    int_theta_error: float,
    u_prev: float,
    prev_error_state: np.ndarray,
) -> np.ndarray:
    return np.array(
        [
            error_state[0],
            error_state[1],
            error_state[2],
            error_state[3],
            int_theta_error,
            u_prev,
            prev_error_state[0],
            prev_error_state[2],
        ],
        dtype=np.float64,
    )


def _rich11_pendulum_features(
    error_state: np.ndarray,
    int_theta_error: float,
    u_prev: float,
    prev_error_state: np.ndarray,
    du_prev: float,
) -> np.ndarray:
    return np.array(
        [
            error_state[0],
            error_state[1],
            error_state[2],
            error_state[3],
            int_theta_error,
            u_prev,
            prev_error_state[0],
            prev_error_state[1],
            prev_error_state[2],
            prev_error_state[3],
            du_prev,
        ],
        dtype=np.float64,
    )


def bptt_rollout_and_grads_pidfeat(
    *,
    policy,
    A: np.ndarray,
    B: np.ndarray,
    cfg: InvertedPendulumRLBPTTConfig,
    x0: np.ndarray | float,
    r_seq: Optional[np.ndarray] = None,
    du_seq: Optional[np.ndarray] = None,
    dt: float,
) -> tuple[float, Grads]:
    T = int(cfg.horizon_steps)
    r_arr, d_arr = _prepare_rollout(cfg=cfg, r_seq=r_seq, du_seq=du_seq)

    A = np.asarray(A, dtype=np.float64).reshape(4, 4)
    B = np.asarray(B, dtype=np.float64).reshape(4)
    q_diag = np.asarray(cfg.q_state_diag, dtype=np.float64).reshape(4)
    terminal_P_aug = _resolved_terminal_cost_matrix(A=A, B=B, cfg=cfg)
    dt = float(dt)

    x = np.zeros((T + 1, 4), dtype=np.float64)
    err = np.zeros((T, 4), dtype=np.float64)
    ie_theta = np.zeros((T,), dtype=np.float64)
    u_cmd = np.zeros((T,), dtype=np.float64)
    du_cmd = np.zeros((T,), dtype=np.float64)
    caches = [None] * T

    x[0] = _state_vector(x0)

    u_prev = 0.0
    ie_prev = 0.0
    for k in range(T):
        err[k] = r_arr[k] - x[k]
        ie_theta[k] = ie_prev + err[k, 2] * dt
        feat = _pid_features(err[k], ie_theta[k])
        u_cmd[k], caches[k] = policy.forward(feat, u_min=cfg.u_min, u_max=cfg.u_max)
        du_cmd[k] = u_cmd[k] - u_prev
        x[k + 1] = (A @ x[k]) + (B * (u_cmd[k] + d_arr[k]))
        u_prev = u_cmd[k]
        ie_prev = ie_theta[k]

    state_cost = np.sum((err * err) * q_diag.reshape(1, 4), axis=1)
    scale = 1.0 / T
    terminal_cost, g_x_terminal, g_u_terminal_prev = _terminal_cost_and_gradients(
        terminal_P_aug=terminal_P_aug,
        x_terminal=x[T],
        u_terminal_prev=u_cmd[T - 1] if T > 0 else 0.0,
        scale=scale,
    )
    loss = float(np.mean(state_cost + cfg.ru * (du_cmd * du_cmd) + cfg.qu * (u_cmd * u_cmd)) + terminal_cost)

    grads_total: Grads = {key: np.zeros_like(value) for key, value in policy.params.items()}
    gx = np.zeros((T + 1, 4), dtype=np.float64)
    gu = np.zeros((T,), dtype=np.float64)
    gerr = np.zeros((T, 4), dtype=np.float64)
    gie = np.zeros((T,), dtype=np.float64)
    gx[T] += g_x_terminal
    if T > 0:
        gu[T - 1] += g_u_terminal_prev

    for k in range(T - 1, -1, -1):
        g_x = gx[k].copy()
        g_u = float(gu[k])
        g_err = gerr[k].copy()
        g_ie = float(gie[k])

        g_x += A.T @ gx[k + 1]
        g_u += float(B @ gx[k + 1])

        g_err += scale * (2.0 * q_diag * err[k])
        g_du = scale * (2.0 * cfg.ru * du_cmd[k])
        g_u += scale * (2.0 * cfg.qu * u_cmd[k]) + g_du
        g_u_prev = -g_du

        pg, dfeat = policy.backward(g_u, caches[k])
        for key in grads_total:
            grads_total[key] += pg[key]

        g_err[2] += float(dfeat[0])
        g_ie += float(dfeat[1])
        g_err[3] += float(dfeat[2])
        g_err[0] += float(dfeat[3])
        g_err[1] += float(dfeat[4])

        g_ie_prev = g_ie
        g_err[2] += g_ie * dt

        g_x += -g_err
        gx[k] = g_x

        if k > 0:
            gu[k - 1] += g_u_prev
            gie[k - 1] += g_ie_prev

    return loss, grads_total


def bptt_rollout_and_grads_rich(
    *,
    policy,
    A: np.ndarray,
    B: np.ndarray,
    cfg: InvertedPendulumRLBPTTConfig,
    x0: np.ndarray | float,
    r_seq: Optional[np.ndarray] = None,
    du_seq: Optional[np.ndarray] = None,
    dt: float,
) -> tuple[float, Grads]:
    return bptt_rollout_and_grads_state6(
        policy=policy,
        A=A,
        B=B,
        cfg=cfg,
        x0=x0,
        r_seq=r_seq,
        du_seq=du_seq,
        dt=dt,
    )


def bptt_rollout_and_grads_state6(
    *,
    policy,
    A: np.ndarray,
    B: np.ndarray,
    cfg: InvertedPendulumRLBPTTConfig,
    x0: np.ndarray | float,
    r_seq: Optional[np.ndarray] = None,
    du_seq: Optional[np.ndarray] = None,
    dt: float,
) -> tuple[float, Grads]:
    T = int(cfg.horizon_steps)
    r_arr, d_arr = _prepare_rollout(cfg=cfg, r_seq=r_seq, du_seq=du_seq)

    A = np.asarray(A, dtype=np.float64).reshape(4, 4)
    B = np.asarray(B, dtype=np.float64).reshape(4)
    q_diag = np.asarray(cfg.q_state_diag, dtype=np.float64).reshape(4)
    terminal_P_aug = _resolved_terminal_cost_matrix(A=A, B=B, cfg=cfg)
    dt = float(dt)

    x = np.zeros((T + 1, 4), dtype=np.float64)
    err = np.zeros((T, 4), dtype=np.float64)
    ie_theta = np.zeros((T,), dtype=np.float64)
    u_cmd = np.zeros((T,), dtype=np.float64)
    du_cmd = np.zeros((T,), dtype=np.float64)
    caches = [None] * T

    x[0] = _state_vector(x0)

    u_prev = 0.0
    ie_prev = 0.0
    for k in range(T):
        err[k] = r_arr[k] - x[k]
        ie_theta[k] = ie_prev + err[k, 2] * dt
        feat = _state6_features(err[k], ie_theta[k], u_prev)
        u_cmd[k], caches[k] = policy.forward(feat, u_min=cfg.u_min, u_max=cfg.u_max)
        du_cmd[k] = u_cmd[k] - u_prev
        x[k + 1] = (A @ x[k]) + (B * (u_cmd[k] + d_arr[k]))
        u_prev = u_cmd[k]
        ie_prev = ie_theta[k]

    state_cost = np.sum((err * err) * q_diag.reshape(1, 4), axis=1)
    scale = 1.0 / T
    terminal_cost, g_x_terminal, g_u_terminal_prev = _terminal_cost_and_gradients(
        terminal_P_aug=terminal_P_aug,
        x_terminal=x[T],
        u_terminal_prev=u_cmd[T - 1] if T > 0 else 0.0,
        scale=scale,
    )
    loss = float(np.mean(state_cost + cfg.ru * (du_cmd * du_cmd) + cfg.qu * (u_cmd * u_cmd)) + terminal_cost)

    grads_total: Grads = {key: np.zeros_like(value) for key, value in policy.params.items()}
    gx = np.zeros((T + 1, 4), dtype=np.float64)
    gu = np.zeros((T,), dtype=np.float64)
    gerr = np.zeros((T, 4), dtype=np.float64)
    gie = np.zeros((T,), dtype=np.float64)
    gx[T] += g_x_terminal
    if T > 0:
        gu[T - 1] += g_u_terminal_prev

    for k in range(T - 1, -1, -1):
        g_x = gx[k].copy()
        g_u = float(gu[k])
        g_err = gerr[k].copy()
        g_ie = float(gie[k])

        g_x += A.T @ gx[k + 1]
        g_u += float(B @ gx[k + 1])

        g_err += scale * (2.0 * q_diag * err[k])
        g_du = scale * (2.0 * cfg.ru * du_cmd[k])
        g_u += scale * (2.0 * cfg.qu * u_cmd[k]) + g_du
        g_u_prev = -g_du

        pg, dfeat = policy.backward(g_u, caches[k])
        for key in grads_total:
            grads_total[key] += pg[key]

        g_err[0] += float(dfeat[0])
        g_err[1] += float(dfeat[1])
        g_err[2] += float(dfeat[2])
        g_err[3] += float(dfeat[3])
        g_ie += float(dfeat[4])
        g_u_prev += float(dfeat[5])

        g_ie_prev = g_ie
        g_err[2] += g_ie * dt

        g_x += -g_err
        gx[k] = g_x

        if k > 0:
            gu[k - 1] += g_u_prev
            gie[k - 1] += g_ie_prev

    return loss, grads_total


def bptt_rollout_and_grads_history8(
    *,
    policy,
    A: np.ndarray,
    B: np.ndarray,
    cfg: InvertedPendulumRLBPTTConfig,
    x0: np.ndarray | float,
    r_seq: Optional[np.ndarray] = None,
    du_seq: Optional[np.ndarray] = None,
    dt: float,
) -> tuple[float, Grads]:
    T = int(cfg.horizon_steps)
    r_arr, d_arr = _prepare_rollout(cfg=cfg, r_seq=r_seq, du_seq=du_seq)

    A = np.asarray(A, dtype=np.float64).reshape(4, 4)
    B = np.asarray(B, dtype=np.float64).reshape(4)
    q_diag = np.asarray(cfg.q_state_diag, dtype=np.float64).reshape(4)
    terminal_P_aug = _resolved_terminal_cost_matrix(A=A, B=B, cfg=cfg)
    dt = float(dt)

    x = np.zeros((T + 1, 4), dtype=np.float64)
    err = np.zeros((T, 4), dtype=np.float64)
    ie_theta = np.zeros((T,), dtype=np.float64)
    u_cmd = np.zeros((T,), dtype=np.float64)
    du_cmd = np.zeros((T,), dtype=np.float64)
    caches = [None] * T

    x[0] = _state_vector(x0)

    u_prev = 0.0
    ie_prev = 0.0
    prev_err = np.zeros((4,), dtype=np.float64)
    for k in range(T):
        err[k] = r_arr[k] - x[k]
        ie_theta[k] = ie_prev + err[k, 2] * dt
        feat = _history8_features(err[k], ie_theta[k], u_prev, prev_err)
        u_cmd[k], caches[k] = policy.forward(feat, u_min=cfg.u_min, u_max=cfg.u_max)
        du_cmd[k] = u_cmd[k] - u_prev
        x[k + 1] = (A @ x[k]) + (B * (u_cmd[k] + d_arr[k]))
        u_prev = u_cmd[k]
        ie_prev = ie_theta[k]
        prev_err = err[k].copy()

    state_cost = np.sum((err * err) * q_diag.reshape(1, 4), axis=1)
    scale = 1.0 / T
    terminal_cost, g_x_terminal, g_u_terminal_prev = _terminal_cost_and_gradients(
        terminal_P_aug=terminal_P_aug,
        x_terminal=x[T],
        u_terminal_prev=u_cmd[T - 1] if T > 0 else 0.0,
        scale=scale,
    )
    loss = float(np.mean(state_cost + cfg.ru * (du_cmd * du_cmd) + cfg.qu * (u_cmd * u_cmd)) + terminal_cost)

    grads_total: Grads = {key: np.zeros_like(value) for key, value in policy.params.items()}
    gx = np.zeros((T + 1, 4), dtype=np.float64)
    gu = np.zeros((T,), dtype=np.float64)
    gerr = np.zeros((T, 4), dtype=np.float64)
    gie = np.zeros((T,), dtype=np.float64)
    gx[T] += g_x_terminal
    if T > 0:
        gu[T - 1] += g_u_terminal_prev

    for k in range(T - 1, -1, -1):
        g_x = gx[k].copy()
        g_u = float(gu[k])
        g_err = gerr[k].copy()
        g_ie = float(gie[k])

        g_x += A.T @ gx[k + 1]
        g_u += float(B @ gx[k + 1])

        g_err += scale * (2.0 * q_diag * err[k])
        g_du = scale * (2.0 * cfg.ru * du_cmd[k])
        g_u += scale * (2.0 * cfg.qu * u_cmd[k]) + g_du
        g_u_prev = -g_du

        pg, dfeat = policy.backward(g_u, caches[k])
        for key in grads_total:
            grads_total[key] += pg[key]

        g_err[0] += float(dfeat[0])
        g_err[1] += float(dfeat[1])
        g_err[2] += float(dfeat[2])
        g_err[3] += float(dfeat[3])
        g_ie += float(dfeat[4])
        g_u_prev += float(dfeat[5])
        if k > 0:
            gerr[k - 1, 0] += float(dfeat[6])
            gerr[k - 1, 2] += float(dfeat[7])

        g_ie_prev = g_ie
        g_err[2] += g_ie * dt

        g_x += -g_err
        gx[k] = g_x

        if k > 0:
            gu[k - 1] += g_u_prev
            gie[k - 1] += g_ie_prev

    return loss, grads_total


def bptt_rollout_and_grads_rich11_pendulum(
    *,
    policy,
    A: np.ndarray,
    B: np.ndarray,
    cfg: InvertedPendulumRLBPTTConfig,
    x0: np.ndarray | float,
    r_seq: Optional[np.ndarray] = None,
    du_seq: Optional[np.ndarray] = None,
    dt: float,
) -> tuple[float, Grads]:
    T = int(cfg.horizon_steps)
    r_arr, d_arr = _prepare_rollout(cfg=cfg, r_seq=r_seq, du_seq=du_seq)

    A = np.asarray(A, dtype=np.float64).reshape(4, 4)
    B = np.asarray(B, dtype=np.float64).reshape(4)
    q_diag = np.asarray(cfg.q_state_diag, dtype=np.float64).reshape(4)
    terminal_P_aug = _resolved_terminal_cost_matrix(A=A, B=B, cfg=cfg)
    dt = float(dt)

    x = np.zeros((T + 1, 4), dtype=np.float64)
    err = np.zeros((T, 4), dtype=np.float64)
    ie_theta = np.zeros((T,), dtype=np.float64)
    u_cmd = np.zeros((T,), dtype=np.float64)
    du_cmd = np.zeros((T,), dtype=np.float64)
    caches = [None] * T

    x[0] = _state_vector(x0)

    u_prev = 0.0
    u_prev_prev = 0.0
    ie_prev = 0.0
    prev_err = np.zeros((4,), dtype=np.float64)
    for k in range(T):
        err[k] = r_arr[k] - x[k]
        ie_theta[k] = ie_prev + err[k, 2] * dt
        du_prev = u_prev - u_prev_prev
        feat = _rich11_pendulum_features(err[k], ie_theta[k], u_prev, prev_err, du_prev)
        u_cmd[k], caches[k] = policy.forward(feat, u_min=cfg.u_min, u_max=cfg.u_max)
        du_cmd[k] = u_cmd[k] - u_prev
        x[k + 1] = (A @ x[k]) + (B * (u_cmd[k] + d_arr[k]))
        u_prev_prev = u_prev
        u_prev = u_cmd[k]
        ie_prev = ie_theta[k]
        prev_err = err[k].copy()

    state_cost = np.sum((err * err) * q_diag.reshape(1, 4), axis=1)
    scale = 1.0 / T
    terminal_cost, g_x_terminal, g_u_terminal_prev = _terminal_cost_and_gradients(
        terminal_P_aug=terminal_P_aug,
        x_terminal=x[T],
        u_terminal_prev=u_cmd[T - 1] if T > 0 else 0.0,
        scale=scale,
    )
    loss = float(np.mean(state_cost + cfg.ru * (du_cmd * du_cmd) + cfg.qu * (u_cmd * u_cmd)) + terminal_cost)

    grads_total: Grads = {key: np.zeros_like(value) for key, value in policy.params.items()}
    gx = np.zeros((T + 1, 4), dtype=np.float64)
    gu = np.zeros((T,), dtype=np.float64)
    gerr = np.zeros((T, 4), dtype=np.float64)
    gie = np.zeros((T,), dtype=np.float64)
    gx[T] += g_x_terminal
    if T > 0:
        gu[T - 1] += g_u_terminal_prev

    for k in range(T - 1, -1, -1):
        g_x = gx[k].copy()
        g_u = float(gu[k])
        g_err = gerr[k].copy()
        g_ie = float(gie[k])

        g_x += A.T @ gx[k + 1]
        g_u += float(B @ gx[k + 1])

        g_err += scale * (2.0 * q_diag * err[k])
        g_du = scale * (2.0 * cfg.ru * du_cmd[k])
        g_u += scale * (2.0 * cfg.qu * u_cmd[k]) + g_du
        g_u_prev = -g_du

        pg, dfeat = policy.backward(g_u, caches[k])
        for key in grads_total:
            grads_total[key] += pg[key]

        g_err[0] += float(dfeat[0])
        g_err[1] += float(dfeat[1])
        g_err[2] += float(dfeat[2])
        g_err[3] += float(dfeat[3])
        g_ie += float(dfeat[4])
        g_u_prev += float(dfeat[5]) + float(dfeat[10])
        if k > 0:
            gerr[k - 1, 0] += float(dfeat[6])
            gerr[k - 1, 1] += float(dfeat[7])
            gerr[k - 1, 2] += float(dfeat[8])
            gerr[k - 1, 3] += float(dfeat[9])
        if k > 1:
            gu[k - 2] -= float(dfeat[10])

        g_ie_prev = g_ie
        g_err[2] += g_ie * dt

        g_x += -g_err
        gx[k] = g_x

        if k > 0:
            gu[k - 1] += g_u_prev
            gie[k - 1] += g_ie_prev

    return loss, grads_total


def rollout_metrics(
    *,
    policy,
    A: np.ndarray,
    B: np.ndarray,
    cfg: InvertedPendulumRLBPTTConfig,
    kind: str,
    rollout: InvertedPendulumRolloutSpec,
    dt: float,
    tail_fraction: float = 0.25,
) -> dict:
    T = int(cfg.horizon_steps)
    if rollout.r.shape != (T, 4):
        raise ValueError(f"rollout.r must have shape ({T}, 4), got {rollout.r.shape}")

    A = np.asarray(A, dtype=np.float64).reshape(4, 4)
    B = np.asarray(B, dtype=np.float64).reshape(4)
    q_diag = np.asarray(cfg.q_state_diag, dtype=np.float64).reshape(4)
    terminal_P_aug = _resolved_terminal_cost_matrix(A=A, B=B, cfg=cfg)
    x = np.zeros((T + 1, 4), dtype=np.float64)
    err = np.zeros((T, 4), dtype=np.float64)
    u_cmd = np.zeros((T,), dtype=np.float64)
    ie_theta = 0.0
    u_prev = 0.0

    x[0] = _state_vector(rollout.x0)
    for k in range(T):
        err[k] = rollout.r[k] - x[k]
        ie_theta += err[k, 2] * dt
        normalized_kind = _normalize_kind(kind)
        if normalized_kind == "pidfeat":
            feat = _pid_features(err[k], ie_theta)
        elif normalized_kind == "state6":
            feat = _state6_features(err[k], ie_theta, u_prev)
        elif normalized_kind == "history8":
            prev_err = err[k - 1] if k > 0 else np.zeros((4,), dtype=np.float64)
            feat = _history8_features(err[k], ie_theta, u_prev, prev_err)
        elif normalized_kind == "rich11_pendulum":
            prev_err = err[k - 1] if k > 0 else np.zeros((4,), dtype=np.float64)
            prev_u_prev = u_cmd[k - 2] if k > 1 else 0.0
            feat = _rich11_pendulum_features(err[k], ie_theta, u_prev, prev_err, u_prev - prev_u_prev)
        else:
            raise ValueError(f"Unknown kind: {kind}")
        u_cmd[k], _ = policy.forward(feat, u_min=cfg.u_min, u_max=cfg.u_max)
        x[k + 1] = (A @ x[k]) + (B * (u_cmd[k] + rollout.du[k]))
        u_prev = u_cmd[k]

    du_cmd = np.empty_like(u_cmd)
    du_cmd[0] = u_cmd[0]
    du_cmd[1:] = u_cmd[1:] - u_cmd[:-1]
    stage_cost = np.sum((err * err) * q_diag.reshape(1, 4), axis=1) + cfg.ru * (du_cmd * du_cmd) + cfg.qu * (
        u_cmd * u_cmd
    )
    terminal_cost, _, _ = _terminal_cost_and_gradients(
        terminal_P_aug=terminal_P_aug,
        x_terminal=x[T],
        u_terminal_prev=u_cmd[T - 1] if T > 0 else 0.0,
        scale=1.0 / T,
    )

    tail_len = max(1, int(np.ceil(tail_fraction * T)))
    tail_states = x[T - tail_len + 1 : T + 1]
    return {
        "mean_loss": float(np.mean(stage_cost) + terminal_cost),
        "tail_state_cost": float(np.mean(np.sum((tail_states * tail_states) * q_diag.reshape(1, 4), axis=1))),
        "final_state_norm": float(np.linalg.norm(x[-1])),
    }


def summarize_validation_metrics(
    *,
    policy,
    A: np.ndarray,
    B: np.ndarray,
    cfg: InvertedPendulumRLBPTTConfig,
    kind: str,
    validation_rollouts: Sequence[InvertedPendulumRolloutSpec],
    dt: float,
    tail_fraction: float = 0.25,
) -> dict:
    grouped_losses = {
        "regulation": [],
        "disturbance_rejection": [],
    }
    tail_costs = []
    rollout_summaries = []

    for rollout in validation_rollouts:
        metrics = rollout_metrics(
            policy=policy,
            A=A,
            B=B,
            cfg=cfg,
            kind=kind,
            rollout=rollout,
            dt=dt,
            tail_fraction=tail_fraction,
        )
        grouped_losses.setdefault(rollout.mode, []).append(float(metrics["mean_loss"]))
        tail_costs.append(float(metrics["tail_state_cost"]))
        rollout_summaries.append(
            {
                "mode": rollout.mode,
                "mean_loss": float(metrics["mean_loss"]),
                "tail_state_cost": float(metrics["tail_state_cost"]),
                "final_state_norm": float(metrics["final_state_norm"]),
            }
        )

    def _mean_or_inf(values: Sequence[float]) -> float:
        return float(np.mean(values)) if values else float(np.inf)

    all_losses = [entry["mean_loss"] for entry in rollout_summaries]
    return {
        "overall_mean": _mean_or_inf(all_losses),
        "reg_mean": _mean_or_inf(grouped_losses.get("regulation", [])),
        "dist_mean": _mean_or_inf(grouped_losses.get("disturbance_rejection", [])),
        "tail_state_cost_mean": _mean_or_inf(tail_costs),
        "rollout_metrics": rollout_summaries,
        "tail_fraction": float(tail_fraction),
    }


def train_one_policy_with_validation(
    *,
    policy,
    A: np.ndarray,
    B: np.ndarray,
    dt: float,
    cfg: InvertedPendulumRLBPTTConfig,
    steps: int,
    lr: float,
    kind: str,
    episode_sampler: Callable[[np.random.Generator, InvertedPendulumRLBPTTConfig], InvertedPendulumRolloutSpec],
    validation_rollouts: Sequence[InvertedPendulumRolloutSpec],
    eval_every: int = 100,
    early_stop_patience_evals: int = 10,
    early_stop_rel_improve: float = 0.01,
    early_stop_best_val_threshold: float = np.inf,
    rng: Optional[np.random.Generator] = None,
    grad_clip_norm: Optional[float] = None,
) -> dict:
    if eval_every <= 0:
        raise ValueError("eval_every must be > 0")
    if early_stop_patience_evals <= 0:
        raise ValueError("early_stop_patience_evals must be > 0")
    if not validation_rollouts:
        raise ValueError("validation_rollouts must be non-empty")

    opt = Adam(lr=lr)
    rng = np.random.default_rng(0) if rng is None else rng

    A = np.asarray(A, dtype=np.float64).reshape(4, 4)
    B = np.asarray(B, dtype=np.float64).reshape(4)
    dt = float(dt)

    best_val_loss = np.inf
    best_params = {k: v.copy() for k, v in policy.params.items()}
    no_improve_evals = 0
    steps_completed = 0
    updates = []
    train_hist = []
    val_hist = []
    train_window = []

    for it in range(steps):
        rollout = episode_sampler(rng, cfg)
        normalized_kind = _normalize_kind(kind)
        if normalized_kind == "pidfeat":
            loss, grads = bptt_rollout_and_grads_pidfeat(
                policy=policy,
                A=A,
                B=B,
                cfg=cfg,
                x0=rollout.x0,
                r_seq=rollout.r,
                du_seq=rollout.du,
                dt=dt,
            )
        elif normalized_kind == "state6":
            loss, grads = bptt_rollout_and_grads_state6(
                policy=policy,
                A=A,
                B=B,
                cfg=cfg,
                x0=rollout.x0,
                r_seq=rollout.r,
                du_seq=rollout.du,
                dt=dt,
            )
        elif normalized_kind == "history8":
            loss, grads = bptt_rollout_and_grads_history8(
                policy=policy,
                A=A,
                B=B,
                cfg=cfg,
                x0=rollout.x0,
                r_seq=rollout.r,
                du_seq=rollout.du,
                dt=dt,
            )
        elif normalized_kind == "rich11_pendulum":
            loss, grads = bptt_rollout_and_grads_rich11_pendulum(
                policy=policy,
                A=A,
                B=B,
                cfg=cfg,
                x0=rollout.x0,
                r_seq=rollout.r,
                du_seq=rollout.du,
                dt=dt,
            )
        else:
            raise ValueError(f"Unknown kind: {kind}")

        grads = _clip_gradients(grads, grad_clip_norm)
        opt.step(policy.params, grads)
        train_window.append(float(loss))
        steps_completed = it + 1

        if (it + 1) % eval_every != 0:
            continue

        train_mean = float(np.mean(train_window)) if train_window else float(loss)
        train_window = []
        validation_metrics = summarize_validation_metrics(
            policy=policy,
            A=A,
            B=B,
            cfg=cfg,
            kind=kind,
            validation_rollouts=validation_rollouts,
            dt=dt,
        )
        val_loss = float(validation_metrics["overall_mean"])

        updates.append(it + 1)
        train_hist.append(train_mean)
        val_hist.append(val_loss)

        rel_drop = (best_val_loss - val_loss) / max(abs(best_val_loss), 1e-12) if np.isfinite(best_val_loss) else np.inf
        improved = val_loss < best_val_loss and (not np.isfinite(best_val_loss) or rel_drop >= early_stop_rel_improve)
        if improved:
            best_val_loss = val_loss
            best_params = {k: v.copy() for k, v in policy.params.items()}
            no_improve_evals = 0
        else:
            no_improve_evals += 1

        if np.isfinite(best_val_loss) and best_val_loss <= early_stop_best_val_threshold and no_improve_evals >= early_stop_patience_evals:
            break

    if not updates or best_val_loss == np.inf:
        validation_metrics = summarize_validation_metrics(
            policy=policy,
            A=A,
            B=B,
            cfg=cfg,
            kind=kind,
            validation_rollouts=validation_rollouts,
            dt=dt,
        )
        best_val_loss = float(validation_metrics["overall_mean"])
        best_params = {k: v.copy() for k, v in policy.params.items()}
        if not updates:
            updates.append(steps_completed)
            train_hist.append(float(np.mean(train_window)) if train_window else float("nan"))
            val_hist.append(best_val_loss)

    for key, value in best_params.items():
        policy.params[key] = value.copy()

    final_validation_metrics = summarize_validation_metrics(
        policy=policy,
        A=A,
        B=B,
        cfg=cfg,
        kind=kind,
        validation_rollouts=validation_rollouts,
        dt=dt,
    )
    final_val_loss = float(final_validation_metrics["overall_mean"])

    return {
        "final_loss": float(train_hist[-1] if train_hist else 0.0),
        "best_val_loss": float(best_val_loss),
        "val_loss": np.asarray(val_hist, dtype=np.float64),
        "train_loss": np.asarray(train_hist, dtype=np.float64),
        "updates": np.asarray(updates, dtype=np.int64),
        "validation_metrics": final_validation_metrics,
        "steps_completed": int(steps_completed),
        "stopped_early": bool(steps_completed < steps),
        "final_val_loss": float(final_val_loss),
    }


class InvertedPendulumRLController:
    def __init__(
        self,
        *,
        policy,
        cfg: InvertedPendulumRLRuntimeConfig,
        u_bounds: Optional[Bounds] = None,
        name: str = "RL",
    ) -> None:
        self.policy = policy
        self.cfg = cfg
        self.u_bounds = u_bounds
        self.name = name
        self.io = IOSpec(ref_dim=4, obs_dim=4, act_dim=1, out_dim=1)
        self._int_theta_error = 0.0
        self._u_prev = 0.0
        self._u_prev_prev = 0.0
        self._prev_error_state = np.zeros((4,), dtype=np.float64)

    def reset(self) -> None:
        self._int_theta_error = 0.0
        self._u_prev = 0.0
        self._u_prev_prev = 0.0
        self._prev_error_state = np.zeros((4,), dtype=np.float64)

    def step(self, r: Array, obs: Array, t: float) -> Array:
        r_vec = as_vector(r, 4, "r")
        obs_vec = as_vector(obs, 4, "obs")
        error_state = r_vec - obs_vec

        dt = float(self.cfg.dt)
        self._int_theta_error += error_state[2] * dt

        normalized_kind = _normalize_kind(self.cfg.kind)
        if normalized_kind == "pidfeat":
            feat = _pid_features(error_state, self._int_theta_error)
        elif normalized_kind == "state6":
            feat = _state6_features(error_state, self._int_theta_error, self._u_prev)
        elif normalized_kind == "history8":
            feat = _history8_features(
                error_state,
                self._int_theta_error,
                self._u_prev,
                self._prev_error_state,
            )
        elif normalized_kind == "rich11_pendulum":
            feat = _rich11_pendulum_features(
                error_state,
                self._int_theta_error,
                self._u_prev,
                self._prev_error_state,
                self._u_prev - getattr(self, "_u_prev_prev", 0.0),
            )
        else:
            raise ValueError(f"Unknown RL kind: {self.cfg.kind}")

        u, _ = self.policy.forward(feat, u_min=self.cfg.u_min, u_max=self.cfg.u_max)
        if self.u_bounds is not None:
            u = float(self.u_bounds.clip(np.array([u], dtype=float))[0])
        self._u_prev_prev = float(self._u_prev)
        self._u_prev = float(u)
        self._prev_error_state = error_state.astype(np.float64, copy=True)
        return np.array([u], dtype=float)

    @staticmethod
    def load_npz(
        path: str,
        *,
        kind: str,
        dt: float,
        u_bounds: Optional[Bounds] = None,
        u_min: Optional[float] = None,
        u_max: Optional[float] = None,
        mlp_hidden: tuple[int, ...] = (16, 16),
        mlp_activation: str = "tanh",
        name: str = "RL",
    ) -> "InvertedPendulumRLController":
        params, _meta = load_policy_npz(path)
        normalized_kind = _normalize_kind(kind)
        if normalized_kind == "pidfeat":
            policy = LinearFeaturePolicy(PENDULUM_PID_INPUT_DIM, seed=0)
        elif normalized_kind in {"state6", "history8", "rich11_pendulum"}:
            spec = MLPPolicySpec(
                input_dim=pendulum_feature_dim(normalized_kind),
                hidden_layers=tuple(mlp_hidden),
                activation=mlp_activation,
            )
            policy = MLPPolicy(spec, seed=0)
        else:
            raise ValueError(f"Unknown kind: {kind}")

        for key, value in params.items():
            policy.params[key] = value.astype(np.float64)

        cfg = InvertedPendulumRLRuntimeConfig(dt=dt, kind=normalized_kind, u_min=u_min, u_max=u_max)
        return InvertedPendulumRLController(policy=policy, cfg=cfg, u_bounds=u_bounds, name=name)
