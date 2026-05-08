from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from control_bench.core.types import Array, Bounds, IOSpec, as_vector


@dataclass(frozen=True)
class PIDConfig:
    """
    Standard discrete PID:

        u = Kp*e + Ki*∫e dt + Kd*de/dt

    Options:
    - derivative_filter_tau: first-order low-pass filter time-constant for derivative term
    - anti_windup: "conditional" (default) or "backcalc"
    - kaw: anti-windup back-calculation gain (used only if anti_windup="backcalc")
    """
    kp: float
    ki: float
    kd: float
    derivative_filter_tau: float = 0.0  # 0 = no filtering
    anti_windup: str = "conditional"    # "conditional" or "backcalc"
    kaw: float = 0.0                    # only for backcalc


class PIDController:
    """
    SISO PID controller implementing the core Controller protocol:
      - reset()
      - step(r, obs, t) -> u

    Assumes obs = y (scalar measurement). If you later want state feedback,
    wrap/transform observations before feeding PID.
    """

    def __init__(
        self,
        cfg: PIDConfig,
        *,
        dt: float,
        u_bounds: Optional[Bounds] = None,
        name: str = "PID",
    ) -> None:
        self.cfg = cfg
        self.dt = float(dt)
        if self.dt <= 0:
            raise ValueError("dt must be > 0")

        self.u_bounds = u_bounds
        self.name = name

        # Controller I/O spec (SISO)
        self.io = IOSpec(ref_dim=1, obs_dim=1, act_dim=1, out_dim=1)

        # internal state
        self._i = 0.0
        self._e_prev = 0.0
        self._d_filt = 0.0  # filtered derivative state (in "error rate" units)

    def reset(self) -> None:
        self._i = 0.0
        self._e_prev = 0.0
        self._d_filt = 0.0

    def step(self, r: Array, obs: Array, t: float) -> Array:
        r0 = float(as_vector(r, 1, "r")[0])
        y0 = float(as_vector(obs, 1, "obs")[0])

        e = r0 - y0

        # --- derivative term (optionally filtered) ---
        de = (e - self._e_prev) / self.dt

        if self.cfg.derivative_filter_tau > 0:
            # First-order low-pass on derivative:
            # d_filt <- alpha*d_filt + (1-alpha)*de
            tau = float(self.cfg.derivative_filter_tau)
            alpha = tau / (tau + self.dt)
            self._d_filt = alpha * self._d_filt + (1.0 - alpha) * de
            d_term = self.cfg.kd * self._d_filt
        else:
            d_term = self.cfg.kd * de

        # --- integral term update (anti-windup handled below) ---
        i_candidate = self._i + self.cfg.ki * e * self.dt

        # raw (unclipped) control
        u_unsat = self.cfg.kp * e + i_candidate + d_term

        # saturation (if bounds provided)
        if self.u_bounds is None:
            u = u_unsat
            self._i = i_candidate
        else:
            u = float(self.u_bounds.clip(np.array([u_unsat]))[0])

            if self.cfg.anti_windup == "backcalc":
                # Back-calculation: push integral toward the saturated output
                # i <- i_candidate + kaw*(u - u_unsat)
                self._i = i_candidate + self.cfg.kaw * (u - u_unsat)
            else:
                # Conditional integration: only accept integral update if not saturating
                # OR if integration would reduce saturation.
                sat = (u != u_unsat)
                if (not sat) or (sat and np.sign(e) != np.sign(u_unsat - u)):
                    self._i = i_candidate
                # else: keep previous integral (windup prevention)

        self._e_prev = e
        return np.array([u], dtype=float)