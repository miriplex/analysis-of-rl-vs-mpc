from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .types import Params, Grads


@dataclass
class Adam:
    lr: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def __post_init__(self) -> None:
        self.t = 0
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}

    def step(self, params: Params, grads: Grads) -> None:
        self.t += 1
        for k, p in params.items():
            g = grads[k]
            if k not in self.m:
                self.m[k] = np.zeros_like(p)
                self.v[k] = np.zeros_like(p)

            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * (g * g)

            m_hat = self.m[k] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1.0 - self.beta2 ** self.t)

            params[k] = p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)