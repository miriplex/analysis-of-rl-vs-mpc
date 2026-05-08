from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from control_bench.core.types import Array, Bounds, IOSpec, as_vector
from .feature_sets import build_first_order_mlp_features, first_order_mlp_feature_input_dim
from .policies import LinearPIDFeaturePolicy, MLPPolicy, MLPPolicySpec
from .training import load_policy_npz


@dataclass(frozen=True)
class RLRuntimeConfig:
    dt: float
    kind: str  # "pidfeat" or "rich"
    u_min: Optional[float] = None
    u_max: Optional[float] = None
    rich_feature_set: str = "rich11"


class BackpropRLController:
    """
    Controller wrapper compatible with your core simulator.

    Internal PID-like features use time-consistent discrete approximations:
      sum_e  <- sum_e + e * dt
      delta_e = (e - e_prev) / dt
    """

    def __init__(
        self,
        *,
        policy,
        cfg: RLRuntimeConfig,
        u_bounds: Optional[Bounds] = None,
        name: str = "RL",
    ) -> None:
        self.policy = policy
        self.cfg = cfg
        self.u_bounds = u_bounds
        self.name = name

        self.io = IOSpec(ref_dim=1, obs_dim=1, act_dim=1, out_dim=1)

        self._sum_e = 0.0
        self._e_prev1 = 0.0
        self._e_prev2 = 0.0
        self._y_prev1 = 0.0
        self._y_prev2 = 0.0
        self._u_prev1 = 0.0
        self._u_prev2 = 0.0

    def reset(self) -> None:
        self._sum_e = 0.0
        self._e_prev1 = 0.0
        self._e_prev2 = 0.0
        self._y_prev1 = 0.0
        self._y_prev2 = 0.0
        self._u_prev1 = 0.0
        self._u_prev2 = 0.0

    def step(self, r: Array, obs: Array, t: float) -> Array:
        r0 = float(as_vector(r, 1, "r")[0])
        y0 = float(as_vector(obs, 1, "obs")[0])

        e = r0 - y0

        dt = float(self.cfg.dt)
        self._sum_e += e * dt
        delta_e = (e - self._e_prev1) / dt

        if self.cfg.kind == "pidfeat":
            feat = np.array([e, self._sum_e, delta_e], dtype=np.float64)
        elif self.cfg.kind == "rich":
            feat = build_first_order_mlp_features(
                feature_set=self.cfg.rich_feature_set,
                r_k=r0,
                y_k=y0,
                y_prev1=self._y_prev1,
                y_prev2=self._y_prev2,
                e_k=e,
                e_prev1=self._e_prev1,
                e_prev2=self._e_prev2,
                ie_k=self._sum_e,
                de_k=delta_e,
                u_prev1=self._u_prev1,
                u_prev2=self._u_prev2,
            )
        else:
            raise ValueError(f"Unknown RL kind: {self.cfg.kind}")

        u, _cache = self.policy.forward(feat, u_min=self.cfg.u_min, u_max=self.cfg.u_max)

        # optional hard clip to match plant constraints exactly
        if self.u_bounds is not None:
            u = float(self.u_bounds.clip(np.array([u], dtype=float))[0])

        self._y_prev2 = self._y_prev1
        self._y_prev1 = y0
        self._e_prev2 = self._e_prev1
        self._e_prev1 = e
        self._u_prev2 = self._u_prev1
        self._u_prev1 = u
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
        mlp_hidden=(16, 16),
        mlp_activation="tanh",
        rich_feature_set: str = "rich11",
        name: str = "RL",
    ) -> "BackpropRLController":
        params, _meta = load_policy_npz(path)

        if kind == "pidfeat":
            policy = LinearPIDFeaturePolicy(seed=0)
        elif kind == "rich":
            spec = MLPPolicySpec(
                input_dim=first_order_mlp_feature_input_dim(rich_feature_set),
                hidden_layers=tuple(mlp_hidden),
                activation=mlp_activation,
            )
            policy = MLPPolicy(spec, seed=0)
        else:
            raise ValueError(f"Unknown kind: {kind}")

        # overwrite initial params with loaded params
        for k, v in params.items():
            policy.params[k] = v.astype(np.float64)

        cfg = RLRuntimeConfig(
            dt=dt,
            kind=kind,
            u_min=u_min,
            u_max=u_max,
            rich_feature_set=rich_feature_set,
        )
        return BackpropRLController(policy=policy, cfg=cfg, u_bounds=u_bounds, name=name)
