from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np


FIRST_ORDER_MLP_FEATURE_SET_ORDER = {
    "rich11": [
        "r_k",
        "y_k",
        "y_prev1",
        "y_prev2",
        "e_k",
        "e_prev1",
        "e_prev2",
        "ie_k",
        "de_k",
        "u_prev1",
        "u_prev2",
    ],
    "compact6": [
        "r_k",
        "y_k",
        "e_k",
        "ie_k",
        "de_k",
        "u_prev1",
    ],
}


def first_order_mlp_feature_input_dim(feature_set: str) -> int:
    try:
        return int(len(FIRST_ORDER_MLP_FEATURE_SET_ORDER[feature_set]))
    except KeyError as exc:
        raise ValueError(f"Unknown first-order MLP feature set: {feature_set}") from exc


def first_order_mlp_feature_names(feature_set: str) -> List[str]:
    try:
        return list(FIRST_ORDER_MLP_FEATURE_SET_ORDER[feature_set])
    except KeyError as exc:
        raise ValueError(f"Unknown first-order MLP feature set: {feature_set}") from exc


def build_first_order_mlp_features(
    *,
    feature_set: str,
    r_k: float,
    y_k: float,
    y_prev1: float,
    y_prev2: float,
    e_k: float,
    e_prev1: float,
    e_prev2: float,
    ie_k: float,
    de_k: float,
    u_prev1: float,
    u_prev2: float,
) -> np.ndarray:
    values: Dict[str, float] = {
        "r_k": float(r_k),
        "y_k": float(y_k),
        "y_prev1": float(y_prev1),
        "y_prev2": float(y_prev2),
        "e_k": float(e_k),
        "e_prev1": float(e_prev1),
        "e_prev2": float(e_prev2),
        "ie_k": float(ie_k),
        "de_k": float(de_k),
        "u_prev1": float(u_prev1),
        "u_prev2": float(u_prev2),
    }
    try:
        keys: Iterable[str] = FIRST_ORDER_MLP_FEATURE_SET_ORDER[feature_set]
    except KeyError as exc:
        raise ValueError(f"Unknown first-order MLP feature set: {feature_set}") from exc
    return np.array([values[key] for key in keys], dtype=np.float64)
