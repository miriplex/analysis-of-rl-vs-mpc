from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .types import Params, Grads


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _tanh_grad(tanh_x: np.ndarray) -> np.ndarray:
    # derivative of tanh wrt input when you already have tanh(x)
    return 1.0 - tanh_x * tanh_x


def squash_to_bounds(u_raw: float, u_min: Optional[float], u_max: Optional[float]) -> Tuple[float, float]:
    """
    Smooth squash using tanh:
      u = mid + half * tanh(u_raw)

    Returns (u, du/du_raw).
    If bounds are None, returns (u_raw, 1.0).
    """
    if (u_min is None) or (u_max is None):
        return float(u_raw), 1.0

    mid = 0.5 * (u_max + u_min)
    half = 0.5 * (u_max - u_min)
    t = np.tanh(u_raw)
    u = mid + half * t
    du_du_raw = half * (1.0 - t * t)
    return float(u), float(du_du_raw)


class LinearPIDFeaturePolicy:
    """
    Simple policy:
      u_raw = W @ feat
      u = squash(u_raw)

    feat = [e, ie, de] (shape (3,))
    """

    def __init__(self, *, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.params: Params = {
            "W": rng.normal(scale=0.1, size=(1, 3)).astype(np.float64),
        }

    def forward(
        self,
        feat: np.ndarray,
        *,
        u_min: Optional[float],
        u_max: Optional[float],
    ) -> Tuple[float, Dict]:
        feat = np.asarray(feat, dtype=np.float64).reshape(3)

        W = self.params["W"]

        u_raw = float((W @ feat.reshape(3, 1)).item())
        u, du_du_raw = squash_to_bounds(u_raw, u_min, u_max)

        cache = {
            "feat": feat,
            "u_raw": u_raw,
            "du_du_raw": du_du_raw,
        }
        return u, cache

    def backward(self, dL_du: float, cache: Dict) -> Tuple[Grads, np.ndarray]:
        """
        Returns:
          grads: dict matching self.params
          dL_dfeat: shape (3,)
        """
        feat = cache["feat"]
        du_du_raw = float(cache["du_du_raw"])

        dL_du_raw = float(dL_du) * du_du_raw  # chain through squash

        # u_raw = W feat
        dW = dL_du_raw * feat.reshape(1, 3)

        # dL/dfeat = W^T * dL/du_raw
        W = self.params["W"]
        dfeat = (W.reshape(3) * dL_du_raw).astype(np.float64)

        grads: Grads = {"W": dW}
        return grads, dfeat


class LinearFeaturePolicy:
    """
    Generic linear-feature policy:
      u_raw = W @ feat
      u = squash(u_raw)

    Useful when the feature vector is still hand-designed, but no longer the
    fixed first-order PID triplet.
    """

    def __init__(self, input_dim: int, *, seed: int = 0) -> None:
        if int(input_dim) <= 0:
            raise ValueError("input_dim must be > 0")
        self.input_dim = int(input_dim)
        rng = np.random.default_rng(seed)
        self.params: Params = {
            "W": rng.normal(scale=0.1, size=(1, self.input_dim)).astype(np.float64),
        }

    def forward(
        self,
        feat: np.ndarray,
        *,
        u_min: Optional[float],
        u_max: Optional[float],
    ) -> Tuple[float, Dict]:
        feat = np.asarray(feat, dtype=np.float64).reshape(self.input_dim)
        W = self.params["W"]

        u_raw = float((W @ feat.reshape(self.input_dim, 1)).item())
        u, du_du_raw = squash_to_bounds(u_raw, u_min, u_max)
        cache = {
            "feat": feat,
            "u_raw": u_raw,
            "du_du_raw": du_du_raw,
            "input_dim": self.input_dim,
        }
        return u, cache

    def backward(self, dL_du: float, cache: Dict) -> Tuple[Grads, np.ndarray]:
        feat = cache["feat"]
        du_du_raw = float(cache["du_du_raw"])
        dL_du_raw = float(dL_du) * du_du_raw

        dW = dL_du_raw * feat.reshape(1, self.input_dim)
        W = self.params["W"]
        dfeat = (W.reshape(self.input_dim) * dL_du_raw).astype(np.float64)

        grads: Grads = {"W": dW}
        return grads, dfeat


@dataclass(frozen=True)
class MLPPolicySpec:
    input_dim: int
    hidden_layers: Tuple[int, ...] = (16, 16)
    activation: str = "tanh"  # "tanh" or "relu"


class MLPPolicy:
    """
    MLP policy:
      z1 = W1 x + b1; a1 = act(z1)
      z2 = W2 a1 + b2; a2 = act(z2)
      ...
      zL = WL a_{L-1} + bL; u_raw = zL
      u = squash(u_raw)

    Designed for rich features, e.g. input_dim=11:
      [r_k, y_k, y_{k-1}, y_{k-2}, e_k, e_{k-1}, e_{k-2}, ie_k, de_k, u_{k-1}, u_{k-2}]
    """

    def __init__(self, spec: MLPPolicySpec, *, seed: int = 0) -> None:
        self.spec = spec
        rng = np.random.default_rng(seed)

        dims = [spec.input_dim, *spec.hidden_layers, 1]
        self.params: Params = {}

        for i in range(len(dims) - 1):
            fan_in = dims[i]
            fan_out = dims[i + 1]
            # Xavier-ish init for tanh; fine for relu too at this scale
            scale = np.sqrt(1.0 / max(1, fan_in))
            self.params[f"W{i+1}"] = rng.normal(scale=scale, size=(fan_out, fan_in)).astype(np.float64)
            self.params[f"b{i+1}"] = np.zeros((fan_out,), dtype=np.float64)

    def _act(self, z: np.ndarray) -> np.ndarray:
        if self.spec.activation == "tanh":
            return np.tanh(z)
        if self.spec.activation == "relu":
            return np.maximum(0.0, z)
        raise ValueError(f"Unknown activation: {self.spec.activation}")

    def _act_grad_from_a(self, a: np.ndarray, z: np.ndarray) -> np.ndarray:
        if self.spec.activation == "tanh":
            return 1.0 - a * a
        if self.spec.activation == "relu":
            return (z > 0.0).astype(np.float64)
        raise ValueError(f"Unknown activation: {self.spec.activation}")

    def forward(
        self,
        feat: np.ndarray,
        *,
        u_min: Optional[float],
        u_max: Optional[float],
    ) -> Tuple[float, Dict]:
        x = np.asarray(feat, dtype=np.float64).reshape(self.spec.input_dim)

        caches: List[Dict] = []
        a = x

        n_layers = 1 + len(self.spec.hidden_layers)  # last layer outputs scalar
        for layer in range(1, n_layers + 1):
            W = self.params[f"W{layer}"]
            b = self.params[f"b{layer}"]
            z = W @ a + b

            if layer < n_layers:
                a_next = self._act(z)
            else:
                a_next = z  # linear output

            caches.append({"a_in": a, "z": z, "a_out": a_next, "layer": layer})
            a = a_next

        u_raw = float(a.reshape(-1)[0])
        u, du_du_raw = squash_to_bounds(u_raw, u_min, u_max)

        cache = {
            "caches": caches,
            "u_raw": u_raw,
            "du_du_raw": du_du_raw,
            "input_dim": self.spec.input_dim,
            "n_layers": n_layers,
        }
        return u, cache

    def backward(self, dL_du: float, cache: Dict) -> Tuple[Grads, np.ndarray]:
        """
        Backprop through MLP + squash.

        Returns:
          grads: dict matching self.params
          dL_dfeat: shape (input_dim,)
        """
        du_du_raw = float(cache["du_du_raw"])
        dL_du_raw = float(dL_du) * du_du_raw

        caches: List[Dict] = cache["caches"]
        n_layers = int(cache["n_layers"])

        grads: Grads = {k: np.zeros_like(v) for k, v in self.params.items()}

        # output layer: u_raw = z_L (scalar)
        delta = np.array([dL_du_raw], dtype=np.float64)  # shape (1,)

        for idx in range(n_layers - 1, -1, -1):
            c = caches[idx]
            layer = c["layer"]
            a_in = c["a_in"]      # shape (fan_in,)
            z = c["z"]            # shape (fan_out,)
            a_out = c["a_out"]    # shape (fan_out,)

            # For hidden layers: delta <- delta * act'(z)
            if layer < n_layers:
                act_grad = self._act_grad_from_a(a_out, z)
                delta = delta * act_grad

            # Gradients for affine: z = W a_in + b
            grads[f"W{layer}"] += delta.reshape(-1, 1) @ a_in.reshape(1, -1)
            grads[f"b{layer}"] += delta.reshape(-1)

            # Propagate to previous activation
            W = self.params[f"W{layer}"]
            delta = (W.T @ delta.reshape(-1)).reshape(-1)

        dfeat = delta.reshape(cache["input_dim"]).astype(np.float64)
        return grads, dfeat
