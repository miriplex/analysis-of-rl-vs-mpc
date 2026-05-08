from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from ..core.types import Bounds
from .lti_siso import DiscreteLTISISOPlant


class PoleType(str, Enum):
    STABLE = "stable"         # p < 0
    INTEGRATOR = "integrator" # p = 0  (k/s)
    UNSTABLE = "unstable"     # p > 0


class ZeroType(str, Enum):
    NO_ZERO = "no_zero"   # numerator constant (strictly proper)
    LHP_ZERO = "lhp_zero" # z < 0  (minimum phase)
    RHP_ZERO = "rhp_zero" # z > 0  (non-minimum phase)


@dataclass(frozen=True)
class FirstOrderFamilyParams:
    """
    Parameters defining *representative* plants.

    You can later sweep magnitudes (e.g. stable poles -0.2, -1, -5) without changing architecture.
    """
    dt: float = 0.02
    k: float = 1.0

    stable_p: float = -1.0
    unstable_p: float = +1.0

    lhp_z: float = -2.0
    rhp_z: float = +2.0

    # Optional actuator bounds (SISO)
    u_min: Optional[float] = None
    u_max: Optional[float] = None


def build_first_order_grid(params: FirstOrderFamilyParams) -> Dict[str, DiscreteLTISISOPlant]:
    """
    Returns a dict mapping a human-readable plant_id -> plant instance.

    The 9 plants are:
      PoleType ∈ {stable, integrator, unstable}
      ZeroType ∈ {no_zero, lhp_zero, rhp_zero}

    We intentionally define "neutral zero" as "no_zero" (rather than z=0),
    because z=0 can create pole-zero cancellation when p=0 (integrator),
    collapsing the plant into a pure gain, which breaks the intended taxonomy.
    """
    if params.u_min is not None or params.u_max is not None:
        if params.u_min is None or params.u_max is None:
            raise ValueError("If using bounds, provide both u_min and u_max.")
        bounds = Bounds(low=[params.u_min], high=[params.u_max])
    else:
        bounds = None

    def pole_value(pt: PoleType) -> float:
        if pt == PoleType.STABLE:
            return params.stable_p
        if pt == PoleType.INTEGRATOR:
            return 0.0
        if pt == PoleType.UNSTABLE:
            return params.unstable_p
        raise ValueError(f"Unknown PoleType: {pt}")

    def zero_value(zt: ZeroType) -> Optional[float]:
        if zt == ZeroType.NO_ZERO:
            return None
        if zt == ZeroType.LHP_ZERO:
            return params.lhp_z
        if zt == ZeroType.RHP_ZERO:
            return params.rhp_z
        raise ValueError(f"Unknown ZeroType: {zt}")

    plants: Dict[str, DiscreteLTISISOPlant] = {}

    for pt in PoleType:
        for zt in ZeroType:
            p = pole_value(pt)
            z = zero_value(zt)

            plant = DiscreteLTISISOPlant.from_continuous_first_order(
                dt=params.dt,
                p=p,
                k=params.k,
                z=z,
                u_bounds=bounds,
            )

            plant_id = f"{pt.value}__{zt.value}"
            plants[plant_id] = plant

    return plants