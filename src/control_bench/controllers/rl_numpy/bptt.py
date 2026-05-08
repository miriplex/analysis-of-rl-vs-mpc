from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .feature_sets import build_first_order_mlp_features, first_order_mlp_feature_names
from .types import Grads, PlantParams, RLBPTTConfig


def _prepare_rollout_sequences(
    *,
    cfg: RLBPTTConfig,
    r_seq: Optional[np.ndarray],
    du_seq: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    T = int(cfg.horizon_steps)

    if r_seq is None:
        r = np.full(T, float(cfg.r0), dtype=np.float64)
    else:
        r = np.asarray(r_seq, dtype=np.float64).reshape(-1)
        if r.size != T:
            raise ValueError(f"r_seq length must match horizon_steps={T}, got {r.size}")

    if du_seq is None:
        du = np.zeros(T, dtype=np.float64)
    else:
        du = np.asarray(du_seq, dtype=np.float64).reshape(-1)
        if du.size != T:
            raise ValueError(f"du_seq length must match horizon_steps={T}, got {du.size}")

    return r, du


def _prepare_measurement_noise(
    *,
    cfg: RLBPTTConfig,
    noise_y_seq: Optional[np.ndarray],
) -> np.ndarray:
    T = int(cfg.horizon_steps)
    if noise_y_seq is None:
        return np.zeros(T, dtype=np.float64)
    noise_y = np.asarray(noise_y_seq, dtype=np.float64).reshape(-1)
    if noise_y.size != T:
        raise ValueError(f"noise_y_seq length must match horizon_steps={T}, got {noise_y.size}")
    return noise_y


def _plant_arrays(plant: PlantParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    A = np.asarray(plant.A, dtype=np.float64)
    B = np.asarray(plant.B, dtype=np.float64).reshape(-1)
    C = np.asarray(plant.C, dtype=np.float64).reshape(-1)
    D = float(np.asarray(plant.D, dtype=np.float64).reshape(1, 1).item())
    return A, B, C, D


def _state_vector(x0: np.ndarray | float, plant: PlantParams) -> np.ndarray:
    x0_vec = np.asarray(x0, dtype=np.float64).reshape(-1)
    if x0_vec.size != plant.state_dim:
        raise ValueError(
            f"x0 must have length {plant.state_dim} for this plant, got {x0_vec.size}"
        )
    return x0_vec.astype(np.float64, copy=True)


def _plant_output(C: np.ndarray, x: np.ndarray, D: float, u_prev_actual: float) -> float:
    return float(C @ x) + float(D) * float(u_prev_actual)


def bptt_rollout_and_grads_pidfeat(
    *,
    policy,
    plant: PlantParams,
    cfg: RLBPTTConfig,
    x0: np.ndarray | float = 0.0,
    r_seq: Optional[np.ndarray] = None,
    du_seq: Optional[np.ndarray] = None,
    noise_y_seq: Optional[np.ndarray] = None,
) -> Tuple[float, Grads]:
    """
    BPTT for the simple policy with features [e, sum_e, delta_e].

    Plant convention:
      y_k     = C*x_k + D*u_actual_{k-1}
      x_{k+1} = A*x_k + B*(u_k + d_k)

    PID-like feature convention:
      ie_k = ie_{k-1} + e_k * dt
      de_k = (e_k - e_{k-1}) / dt

    Stage cost (mean over horizon):
      qy*(y-r)^2 + ru*(u-u_prev)^2 + qu*u^2
    """
    T = int(cfg.horizon_steps)
    r_arr, d_arr = _prepare_rollout_sequences(cfg=cfg, r_seq=r_seq, du_seq=du_seq)
    noise_arr = _prepare_measurement_noise(cfg=cfg, noise_y_seq=noise_y_seq)

    A, B, C, D = _plant_arrays(plant)
    n = int(A.shape[0])

    qy = float(cfg.qy)
    ru = float(cfg.ru)
    qu = float(cfg.qu)
    dt = float(plant.dt)

    x = np.zeros((T + 1, n), dtype=np.float64)
    y = np.zeros(T, dtype=np.float64)
    e = np.zeros(T, dtype=np.float64)
    ie = np.zeros(T, dtype=np.float64)
    de = np.zeros(T, dtype=np.float64)
    u_cmd = np.zeros(T, dtype=np.float64)
    du_cmd = np.zeros(T, dtype=np.float64)

    caches = [None] * T

    x[0] = _state_vector(x0, plant)

    u_prev_cmd = 0.0
    u_prev_actual = 0.0
    e_prev = 0.0
    ie_prev = 0.0

    for k in range(T):
        y_true = _plant_output(C, x[k], D, u_prev_actual)
        y[k] = y_true + noise_arr[k]
        e[k] = r_arr[k] - y[k]

        ie[k] = ie_prev + e[k] * dt
        de[k] = (e[k] - e_prev) / dt

        feat = np.array([e[k], ie[k], de[k]], dtype=np.float64)
        u_cmd[k], caches[k] = policy.forward(feat, u_min=cfg.u_min, u_max=cfg.u_max)

        du_cmd[k] = u_cmd[k] - u_prev_cmd
        u_actual = u_cmd[k] + d_arr[k]
        x[k + 1] = (A @ x[k]) + (B * u_actual)

        u_prev_cmd = u_cmd[k]
        u_prev_actual = u_actual
        e_prev = e[k]
        ie_prev = ie[k]

    loss = float(np.mean(qy * (e * e) + ru * (du_cmd * du_cmd) + qu * (u_cmd * u_cmd)))

    grads_total: Grads = {key: np.zeros_like(value) for key, value in policy.params.items()}

    gx = np.zeros((T + 1, n), dtype=np.float64)
    gu = np.zeros(T, dtype=np.float64)
    ge = np.zeros(T, dtype=np.float64)
    gie = np.zeros(T, dtype=np.float64)

    scale = 1.0 / T

    for k in range(T - 1, -1, -1):
        g_x = gx[k].copy()
        g_u = float(gu[k])
        g_e = float(ge[k])
        g_ie = float(gie[k])

        g_x += A.T @ gx[k + 1]
        g_u += float(B @ gx[k + 1])

        g_e += scale * (2.0 * qy * e[k])
        g_du = scale * (2.0 * ru * du_cmd[k])
        g_u += scale * (2.0 * qu * u_cmd[k])

        g_u += g_du
        g_u_prev = -g_du

        pg, dfeat = policy.backward(g_u, caches[k])
        for key in grads_total:
            grads_total[key] += pg[key]

        g_e += float(dfeat[0])
        g_ie += float(dfeat[1])
        g_de = float(dfeat[2])

        g_e += g_de / dt
        g_e_prev = -g_de / dt

        g_ie_prev = g_ie
        g_e += g_ie * dt

        g_y = -g_e

        g_x += C * g_y
        g_u_prev += g_y * D

        gx[k] = g_x

        if k > 0:
            gu[k - 1] += g_u_prev
            ge[k - 1] += g_e_prev
            gie[k - 1] += g_ie_prev

    return loss, grads_total


def bptt_rollout_and_grads_rich(
    *,
    policy,
    plant: PlantParams,
    cfg: RLBPTTConfig,
    x0: np.ndarray | float = 0.0,
    r_seq: Optional[np.ndarray] = None,
    du_seq: Optional[np.ndarray] = None,
    noise_y_seq: Optional[np.ndarray] = None,
    feature_set: str = "rich11",
) -> Tuple[float, Grads]:
    """
    BPTT for the rich MLP policy with features:
      [r_k, y_k, y_{k-1}, y_{k-2}, e_k, e_{k-1}, e_{k-2}, ie_k, de_k, u_{k-1}, u_{k-2}]

    Same plant convention and cost as pidfeat.
    """
    T = int(cfg.horizon_steps)
    r_arr, d_arr = _prepare_rollout_sequences(cfg=cfg, r_seq=r_seq, du_seq=du_seq)
    noise_arr = _prepare_measurement_noise(cfg=cfg, noise_y_seq=noise_y_seq)

    A, B, C, D = _plant_arrays(plant)
    n = int(A.shape[0])

    qy = float(cfg.qy)
    ru = float(cfg.ru)
    qu = float(cfg.qu)
    dt = float(plant.dt)

    x = np.zeros((T + 1, n), dtype=np.float64)
    y = np.zeros(T, dtype=np.float64)
    e = np.zeros(T, dtype=np.float64)
    ie = np.zeros(T, dtype=np.float64)
    de = np.zeros(T, dtype=np.float64)
    u_cmd = np.zeros(T, dtype=np.float64)
    du_cmd = np.zeros(T, dtype=np.float64)

    caches = [None] * T
    feature_index = {name: idx for idx, name in enumerate(first_order_mlp_feature_names(feature_set))}

    x[0] = _state_vector(x0, plant)

    u_prev_cmd = 0.0
    u_prev2_cmd = 0.0
    u_prev_actual = 0.0
    e_prev = 0.0
    e_prev2 = 0.0
    ie_prev = 0.0
    y_prev = 0.0
    y_prev2 = 0.0

    for k in range(T):
        y_true = _plant_output(C, x[k], D, u_prev_actual)
        y[k] = y_true + noise_arr[k]
        e[k] = r_arr[k] - y[k]

        ie[k] = ie_prev + e[k] * dt
        de[k] = (e[k] - e_prev) / dt

        feat = build_first_order_mlp_features(
            feature_set=feature_set,
            r_k=r_arr[k],
            y_k=y[k],
            y_prev1=y_prev,
            y_prev2=y_prev2,
            e_k=e[k],
            e_prev1=e_prev,
            e_prev2=e_prev2,
            ie_k=ie[k],
            de_k=de[k],
            u_prev1=u_prev_cmd,
            u_prev2=u_prev2_cmd,
        )
        u_cmd[k], caches[k] = policy.forward(feat, u_min=cfg.u_min, u_max=cfg.u_max)

        du_cmd[k] = u_cmd[k] - u_prev_cmd
        u_actual = u_cmd[k] + d_arr[k]
        x[k + 1] = (A @ x[k]) + (B * u_actual)

        y_prev2 = y_prev
        y_prev = y[k]
        e_prev2 = e_prev
        e_prev = e[k]
        u_prev2_cmd = u_prev_cmd
        u_prev_cmd = u_cmd[k]
        u_prev_actual = u_actual
        ie_prev = ie[k]

    loss = float(np.mean(qy * (e * e) + ru * (du_cmd * du_cmd) + qu * (u_cmd * u_cmd)))

    grads_total: Grads = {key: np.zeros_like(value) for key, value in policy.params.items()}

    gx = np.zeros((T + 1, n), dtype=np.float64)
    gu = np.zeros(T, dtype=np.float64)
    ge = np.zeros(T, dtype=np.float64)
    gie = np.zeros(T, dtype=np.float64)
    gy = np.zeros(T, dtype=np.float64)

    scale = 1.0 / T

    for k in range(T - 1, -1, -1):
        g_x = gx[k].copy()
        g_u = float(gu[k])
        g_e = float(ge[k])
        g_ie = float(gie[k])
        g_y = float(gy[k])

        g_x += A.T @ gx[k + 1]
        g_u += float(B @ gx[k + 1])

        g_e += scale * (2.0 * qy * e[k])
        g_du = scale * (2.0 * ru * du_cmd[k])
        g_u += scale * (2.0 * qu * u_cmd[k])

        g_u += g_du
        g_u_prev = -g_du

        pg, dfeat = policy.backward(g_u, caches[k])
        for key in grads_total:
            grads_total[key] += pg[key]

        def d(name: str) -> float:
            idx = feature_index.get(name)
            return float(dfeat[idx]) if idx is not None else 0.0

        g_y += d("y_k")
        g_e += d("e_k")
        g_ie += d("ie_k")
        g_de = d("de_k")
        g_u_prev += d("u_prev1")

        if k > 0:
            gy[k - 1] += d("y_prev1")
            ge[k - 1] += d("e_prev1")
        if k > 1:
            gy[k - 2] += d("y_prev2")
            ge[k - 2] += d("e_prev2")
            gu[k - 2] += d("u_prev2")

        g_e += g_de / dt
        g_e_prev = -g_de / dt

        g_ie_prev = g_ie
        g_e += g_ie * dt

        g_y += -g_e

        g_x += C * g_y
        g_u_prev += g_y * D

        gx[k] = g_x

        if k > 0:
            gu[k - 1] += g_u_prev
            ge[k - 1] += g_e_prev
            gie[k - 1] += g_ie_prev

    return loss, grads_total


def rollout_loss_pidfeat(
    *,
    policy,
    plant: PlantParams,
    cfg: RLBPTTConfig,
    x0: np.ndarray | float = 0.0,
    r_seq: Optional[np.ndarray] = None,
    du_seq: Optional[np.ndarray] = None,
    noise_y_seq: Optional[np.ndarray] = None,
) -> float:
    metrics = rollout_metrics_pidfeat(
        policy=policy,
        plant=plant,
        cfg=cfg,
        x0=x0,
        r_seq=r_seq,
        du_seq=du_seq,
        noise_y_seq=noise_y_seq,
    )
    return float(metrics["mean_loss"])


def rollout_metrics_pidfeat(
    *,
    policy,
    plant: PlantParams,
    cfg: RLBPTTConfig,
    x0: np.ndarray | float = 0.0,
    r_seq: Optional[np.ndarray] = None,
    du_seq: Optional[np.ndarray] = None,
    noise_y_seq: Optional[np.ndarray] = None,
    tail_fraction: float = 0.25,
) -> dict:
    T = int(cfg.horizon_steps)
    if not (0.0 < float(tail_fraction) <= 1.0):
        raise ValueError("tail_fraction must be in (0, 1]")

    r_arr, d_arr = _prepare_rollout_sequences(cfg=cfg, r_seq=r_seq, du_seq=du_seq)
    noise_arr = _prepare_measurement_noise(cfg=cfg, noise_y_seq=noise_y_seq)

    A, B, C, D = _plant_arrays(plant)

    x = _state_vector(x0, plant)
    u_prev_cmd = 0.0
    u_prev_actual = 0.0
    e_prev = 0.0
    ie_prev = 0.0
    dt = float(plant.dt)

    stage_losses = np.zeros(T, dtype=np.float64)
    e_hist = np.zeros(T, dtype=np.float64)
    for k in range(T):
        y = _plant_output(C, x, D, u_prev_actual) + noise_arr[k]
        e = r_arr[k] - y
        e_hist[k] = e
        ie = ie_prev + e * dt
        de = (e - e_prev) / dt
        feat = np.array([e, ie, de], dtype=np.float64)
        u_cmd, _ = policy.forward(feat, u_min=cfg.u_min, u_max=cfg.u_max)
        du_cmd = u_cmd - u_prev_cmd
        stage_losses[k] = cfg.qy * (e * e) + cfg.ru * (du_cmd * du_cmd) + cfg.qu * (u_cmd * u_cmd)

        u_actual = u_cmd + d_arr[k]
        x = (A @ x) + (B * u_actual)
        u_prev_cmd = u_cmd
        u_prev_actual = u_actual
        e_prev = e
        ie_prev = ie

    tail_steps = max(1, int(np.ceil(float(tail_fraction) * T)))
    tail_slice = slice(T - tail_steps, T)
    return {
        "mean_loss": float(np.mean(stage_losses)),
        "tail_mse": float(np.mean(e_hist[tail_slice] * e_hist[tail_slice])),
        "final_error": float(e_hist[-1]),
    }


def rollout_loss_rich(
    *,
    policy,
    plant: PlantParams,
    cfg: RLBPTTConfig,
    x0: np.ndarray | float = 0.0,
    r_seq: Optional[np.ndarray] = None,
    du_seq: Optional[np.ndarray] = None,
    noise_y_seq: Optional[np.ndarray] = None,
    feature_set: str = "rich11",
) -> float:
    metrics = rollout_metrics_rich(
        policy=policy,
        plant=plant,
        cfg=cfg,
        x0=x0,
        r_seq=r_seq,
        du_seq=du_seq,
        noise_y_seq=noise_y_seq,
        feature_set=feature_set,
    )
    return float(metrics["mean_loss"])


def rollout_metrics_rich(
    *,
    policy,
    plant: PlantParams,
    cfg: RLBPTTConfig,
    x0: np.ndarray | float = 0.0,
    r_seq: Optional[np.ndarray] = None,
    du_seq: Optional[np.ndarray] = None,
    noise_y_seq: Optional[np.ndarray] = None,
    tail_fraction: float = 0.25,
    feature_set: str = "rich11",
) -> dict:
    T = int(cfg.horizon_steps)
    if not (0.0 < float(tail_fraction) <= 1.0):
        raise ValueError("tail_fraction must be in (0, 1]")

    r_arr, d_arr = _prepare_rollout_sequences(cfg=cfg, r_seq=r_seq, du_seq=du_seq)
    noise_arr = _prepare_measurement_noise(cfg=cfg, noise_y_seq=noise_y_seq)

    A, B, C, D = _plant_arrays(plant)

    x = _state_vector(x0, plant)
    u_prev_cmd = 0.0
    u_prev2_cmd = 0.0
    u_prev_actual = 0.0
    e_prev = 0.0
    e_prev2 = 0.0
    ie_prev = 0.0
    y_prev = 0.0
    y_prev2 = 0.0
    dt = float(plant.dt)

    stage_losses = np.zeros(T, dtype=np.float64)
    e_hist = np.zeros(T, dtype=np.float64)
    for k in range(T):
        y = _plant_output(C, x, D, u_prev_actual) + noise_arr[k]
        e = r_arr[k] - y
        e_hist[k] = e
        ie = ie_prev + e * dt
        de = (e - e_prev) / dt
        feat = build_first_order_mlp_features(
            feature_set=feature_set,
            r_k=r_arr[k],
            y_k=y,
            y_prev1=y_prev,
            y_prev2=y_prev2,
            e_k=e,
            e_prev1=e_prev,
            e_prev2=e_prev2,
            ie_k=ie,
            de_k=de,
            u_prev1=u_prev_cmd,
            u_prev2=u_prev2_cmd,
        )
        u_cmd, _ = policy.forward(feat, u_min=cfg.u_min, u_max=cfg.u_max)
        du_cmd = u_cmd - u_prev_cmd
        stage_losses[k] = cfg.qy * (e * e) + cfg.ru * (du_cmd * du_cmd) + cfg.qu * (u_cmd * u_cmd)

        u_actual = u_cmd + d_arr[k]
        x = (A @ x) + (B * u_actual)
        y_prev2 = y_prev
        y_prev = y
        e_prev2 = e_prev
        e_prev = e
        u_prev2_cmd = u_prev_cmd
        u_prev_cmd = u_cmd
        u_prev_actual = u_actual
        ie_prev = ie

    tail_steps = max(1, int(np.ceil(float(tail_fraction) * T)))
    tail_slice = slice(T - tail_steps, T)
    return {
        "mean_loss": float(np.mean(stage_losses)),
        "tail_mse": float(np.mean(e_hist[tail_slice] * e_hist[tail_slice])),
        "final_error": float(e_hist[-1]),
    }
