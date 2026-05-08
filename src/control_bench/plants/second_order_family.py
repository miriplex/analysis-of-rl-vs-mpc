from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ..core.types import Bounds
from .lti_state_space import DiscreteLTIPlant


@dataclass(frozen=True)
class SecondOrderFamilyParams:
    """
    Parameterization for a small suite of representative 2nd-order SISO plants.

    All plants are built as continuous-time transfer functions, realized in a
    2-state controllable canonical form, then discretized exactly.
    """

    dt: float
    k: float
    wn: float

    # Damping ratios for stable 2nd order: under / critically / over damped
    zeta_ud: float
    zeta_cd: float
    zeta_od: float

    # Unstable complex pair (s^2 - 2*sigma*s + wn^2)
    sigma: float

    # Unstable saddle: (s-a)(s+b)
    a: float
    b: float

    # Zeros
    z_lhp: float
    z_rhp: float

    # Optional actuator bounds (SISO)
    u_min: Optional[float] = None
    u_max: Optional[float] = None


@dataclass(frozen=True)
class SecondOrderSinglePlantParams:
    """
    Explicit single 2nd-order SISO transfer function:

        G(s) = (b1*s + b0) / (s^2 + a1*s + a0)
    """

    plant_id: str
    dt: float
    b1: float
    b0: float
    a1: float
    a0: float
    u_min: Optional[float] = None
    u_max: Optional[float] = None


def tf_coeffs_for_plant_id(
    plant_id: str, params: SecondOrderFamilyParams
) -> Tuple[float, float, float, float]:
    """
    Return (b1, b0, a1, a0) for:

        G(s) = (b1*s + b0) / (s^2 + a1*s + a0)

    All plants in this suite are strictly proper (no s^2 term in the numerator),
    so the canonical realization below uses D_c = 0 for all cases.
    """
    k = float(params.k)
    wn = float(params.wn)

    if plant_id in ("stable_ud", "stable_cd", "stable_od", "stable_ud_lhp_zero", "stable_ud_rhp_zero"):
        if plant_id == "stable_ud":
            zeta = float(params.zeta_ud)
        elif plant_id == "stable_cd":
            zeta = float(params.zeta_cd)
        elif plant_id == "stable_od":
            zeta = float(params.zeta_od)
        else:
            zeta = float(params.zeta_ud)

        # Denominator: s^2 + 2*zeta*wn*s + wn^2
        a1 = 2.0 * zeta * wn
        a0 = wn * wn

        # Numerator:
        # - no-zero: k*wn^2
        # - with zero: k*wn^2*(s ± z)
        if plant_id == "stable_ud_lhp_zero":
            b1 = k * wn * wn
            b0 = k * wn * wn * float(params.z_lhp)  # (s + z_lhp)
        elif plant_id == "stable_ud_rhp_zero":
            b1 = k * wn * wn
            b0 = -k * wn * wn * float(params.z_rhp)  # (s - z_rhp)
        else:
            b1 = 0.0
            b0 = k * wn * wn

        return float(b1), float(b0), float(a1), float(a0)

    if plant_id == "unstable_saddle":
        # G(s) = k / ((s-a)(s+b)) = k / (s^2 + (b-a)s - ab)
        a = float(params.a)
        b = float(params.b)
        a1 = (b - a)
        a0 = -(a * b)
        b1 = 0.0
        b0 = k
        return float(b1), float(b0), float(a1), float(a0)

    if plant_id in ("unstable_osc", "boss_unstable_osc_rhp_zero"):
        # Denominator: s^2 - 2*sigma*s + wn^2
        sigma = float(params.sigma)
        a1 = -2.0 * sigma
        a0 = wn * wn

        if plant_id == "boss_unstable_osc_rhp_zero":
            b1 = k * wn * wn
            b0 = -k * wn * wn * float(params.z_rhp)  # (s - z_rhp)
        else:
            b1 = 0.0
            b0 = k * wn * wn

        return float(b1), float(b0), float(a1), float(a0)

    if plant_id == "double_integrator":
        # G(s) = k / s^2
        a1 = 0.0
        a0 = 0.0
        b1 = 0.0
        b0 = k
        return float(b1), float(b0), float(a1), float(a0)

    if plant_id == "int_plus_pole":
        # G(s) = k / (s(s+wn)) = k / (s^2 + wn*s)
        a1 = wn
        a0 = 0.0
        b1 = 0.0
        b0 = k
        return float(b1), float(b0), float(a1), float(a0)

    raise ValueError(f"Unknown second-order plant_id: {plant_id}")


def _canonical_continuous_ss(
    *,
    a1: float,
    a0: float,
    b1: float,
    b0: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Build a continuous-time controllable canonical realization for:

        G(s) = (b1*s + b0) / (s^2 + a1*s + a0)

    Choose:
        A_c = [[0,  1],
               [-a0, -a1]]
        B_c = [[0],
               [1]]
        C_c = [b0, b1]
        D_c = 0

    Derivation (sketch):
      For this canonical form,
        (sI - A_c)^{-1} B_c = [1, s]^T / (s^2 + a1*s + a0)
      Therefore,
        G(s) = C_c (sI - A_c)^{-1} B_c
             = (b0*1 + b1*s) / (s^2 + a1*s + a0)
             = (b1*s + b0) / (s^2 + a1*s + a0)
      which matches the desired numerator/denominator with D_c = 0.
    """
    A_c = np.array([[0.0, 1.0], [-float(a0), -float(a1)]], dtype=float)
    B_c = np.array([[0.0], [1.0]], dtype=float)
    C_c = np.array([[float(b0), float(b1)]], dtype=float)
    D_c = np.array([[0.0]], dtype=float)
    return A_c, B_c, C_c, D_c


def _expm_series_scaling(A: np.ndarray, *, terms: int = 30) -> np.ndarray:
    """
    Simple scaling-and-squaring + truncated Taylor series.

    This is a robust fallback for small matrices (we only ever use 3x3 here).
    """
    A = np.asarray(A, dtype=float)
    n = int(A.shape[0])
    I = np.eye(n, dtype=float)

    norm_1 = float(np.linalg.norm(A, ord=1))
    if norm_1 == 0.0:
        return I

    s = int(max(0, np.ceil(np.log2(norm_1))))
    A_scaled = A / (2.0**s)

    E = I.copy()
    term = I.copy()
    for k in range(1, int(terms) + 1):
        term = (term @ A_scaled) / float(k)
        E = E + term

    # square back up
    for _ in range(s):
        E = E @ E

    return E


def expm_small(A: np.ndarray) -> np.ndarray:
    """
    Matrix exponential for small matrices (<= 3x3 in this project).

    Primary implementation:
      expm(A) = V diag(exp(lam)) V^{-1}  (eigen-decomposition)

    Fallback:
      scaling-and-squaring Taylor series (robust for defective/ill-conditioned cases).
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"expm_small expects a square matrix, got shape {A.shape}")

    try:
        lam, V = np.linalg.eig(A)
        cond = float(np.linalg.cond(V))
        if not np.isfinite(cond) or cond > 1e12:
            raise np.linalg.LinAlgError(f"Ill-conditioned eigenvectors (cond={cond:.2e})")

        Vinv = np.linalg.inv(V)
        E = V @ np.diag(np.exp(lam)) @ Vinv
        return np.real_if_close(E, tol=1000)
    except np.linalg.LinAlgError:
        return _expm_series_scaling(A, terms=30)


def _discretize_exact(A_c: np.ndarray, B_c: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact discretization for (A_c, B_c) using an augmented matrix exponential:

        M = [[A_c, B_c],
             [ 0 ,  0 ]]

        expm(M*dt) = [[A_d, B_d],
                      [ 0 ,  1 ]]
    """
    dt = float(dt)
    if dt <= 0:
        raise ValueError("dt must be > 0")

    A_c = np.asarray(A_c, dtype=float)
    B_c = np.asarray(B_c, dtype=float)
    n = int(A_c.shape[0])
    if A_c.shape != (n, n):
        raise ValueError(f"A_c must be square, got {A_c.shape}")
    if B_c.shape != (n, 1):
        raise ValueError(f"B_c must have shape (n,1), got {B_c.shape}")

    M = np.zeros((n + 1, n + 1), dtype=float)
    M[:n, :n] = A_c
    M[:n, n:] = B_c

    Md = expm_small(M * dt)
    if np.iscomplexobj(Md):
        imag_max = float(np.max(np.abs(np.imag(Md))))
        if imag_max <= 1e-12:
            Md = np.real(Md)
    A_d = np.asarray(Md[:n, :n], dtype=float)
    B_d = np.asarray(Md[:n, n:], dtype=float)
    return A_d, B_d


def _maybe_make_bounds(u_min: Optional[float], u_max: Optional[float]) -> Optional[Bounds]:
    if u_min is not None or u_max is not None:
        if u_min is None or u_max is None:
            raise ValueError("If using bounds, provide both u_min and u_max.")
        return Bounds(low=[float(u_min)], high=[float(u_max)])
    return None


def build_second_order_single_plant(params: SecondOrderSinglePlantParams) -> DiscreteLTIPlant:
    dt = float(params.dt)
    if dt <= 0:
        raise ValueError("params.dt must be > 0")

    bounds = _maybe_make_bounds(params.u_min, params.u_max)
    A_c, B_c, C_c, D_c = _canonical_continuous_ss(
        a1=float(params.a1),
        a0=float(params.a0),
        b1=float(params.b1),
        b0=float(params.b0),
    )
    A_d, B_d = _discretize_exact(A_c, B_c, dt)
    return DiscreteLTIPlant(dt=dt, A=A_d, B=B_d, C=C_c, D=D_c, u_bounds=bounds)


def build_second_order_suite_plant(
    plant_id: str,
    params: SecondOrderFamilyParams,
) -> DiscreteLTIPlant:
    dt = float(params.dt)
    if dt <= 0:
        raise ValueError("params.dt must be > 0")

    bounds = _maybe_make_bounds(params.u_min, params.u_max)
    b1, b0, a1, a0 = tf_coeffs_for_plant_id(plant_id, params)
    A_c, B_c, C_c, D_c = _canonical_continuous_ss(a1=a1, a0=a0, b1=b1, b0=b0)
    A_d, B_d = _discretize_exact(A_c, B_c, dt)
    return DiscreteLTIPlant(dt=dt, A=A_d, B=B_d, C=C_c, D=D_c, u_bounds=bounds)


def build_second_order_suite(params: SecondOrderFamilyParams) -> Dict[str, DiscreteLTIPlant]:
    """
    Factory that returns a dict mapping plant_id -> DiscreteLTIPlant.

    Plant IDs included (exact):
      1) stable_ud
      2) stable_cd
      3) stable_od
      4) unstable_saddle
      5) unstable_osc
      6) double_integrator
      7) int_plus_pole
      8) stable_ud_lhp_zero
      9) stable_ud_rhp_zero
      10) boss_unstable_osc_rhp_zero
    """
    dt = float(params.dt)
    if dt <= 0:
        raise ValueError("params.dt must be > 0")

    bounds = _maybe_make_bounds(params.u_min, params.u_max)

    plant_ids = [
        "stable_ud",
        "stable_cd",
        "stable_od",
        "unstable_saddle",
        "unstable_osc",
        "double_integrator",
        "int_plus_pole",
        "stable_ud_lhp_zero",
        "stable_ud_rhp_zero",
        "boss_unstable_osc_rhp_zero",
    ]

    plants: Dict[str, DiscreteLTIPlant] = {}

    for plant_id in plant_ids:
        plants[plant_id] = build_second_order_suite_plant(plant_id, params)

    return plants
