from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any, Dict, List


_MANIFEST_CACHE: Dict[str, Any] | None = None


def manifest_path() -> Path:
    return Path(__file__).resolve().parents[2] / "EXPERIMENT_MANIFEST.yaml"


def _load_with_python_yaml(path: Path) -> Dict[str, Any] | None:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Experiment manifest must decode to a mapping")
    return data


def _load_with_ruby_yaml(path: Path) -> Dict[str, Any]:
    cmd = [
        "ruby",
        "-e",
        (
            "require 'yaml'; "
            "require 'json'; "
            "data = YAML.load_file(ARGV[0]); "
            "puts JSON.generate(data)"
        ),
        str(path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    if not isinstance(data, dict):
        raise ValueError("Experiment manifest must decode to a mapping")
    return data


def load_experiment_manifest() -> Dict[str, Any]:
    global _MANIFEST_CACHE
    if _MANIFEST_CACHE is not None:
        return _MANIFEST_CACHE

    path = manifest_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing experiment manifest: {path}")

    data = _load_with_python_yaml(path)
    if data is None:
        data = _load_with_ruby_yaml(path)

    _MANIFEST_CACHE = data
    return data


def first_order_manifest() -> Dict[str, Any]:
    root = load_experiment_manifest()
    return dict(root["first_order"])


def first_order_controller_registry() -> Dict[str, Any]:
    return dict(first_order_manifest()["rl_training"]["controller_registry"])


def _deep_merge_manifest(defaults: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(defaults)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_manifest(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def first_order_mlp_variant_study() -> Dict[str, Any]:
    return dict(first_order_manifest()["evaluation"]["mlp_variant_study"])


def first_order_robust_vs_baseline_study() -> Dict[str, Any]:
    return dict(first_order_manifest()["evaluation"]["robust_vs_baseline"])


def first_order_hidden_pole_stress_test() -> Dict[str, Any]:
    return dict(first_order_manifest()["evaluation"]["hidden_pole_stress_test"])


def first_order_disturbance_stress_test() -> Dict[str, Any]:
    return dict(first_order_manifest()["evaluation"]["disturbance_stress_test"])


def first_order_measurement_noise_stress_test() -> Dict[str, Any]:
    return dict(first_order_manifest()["evaluation"]["measurement_noise_stress_test"])


def first_order_input_noise_stress_test() -> Dict[str, Any]:
    return dict(first_order_manifest()["evaluation"]["input_noise_stress_test"])


def first_order_step_disturbance_stress_test() -> Dict[str, Any]:
    return dict(first_order_manifest()["evaluation"]["step_disturbance_stress_test"])


def first_order_mlp_feature_sets() -> Dict[str, Dict[str, Any]]:
    return dict(first_order_controller_registry()["mlp_feature_sets"])


def first_order_mlp_variants() -> List[Dict[str, Any]]:
    registry = first_order_controller_registry()
    defaults = dict(registry.get("mlp_training_defaults", {}))
    return [
        _deep_merge_manifest(defaults, dict(variant))
        for variant in registry["mlp_variants"]
    ]


def first_order_default_mlp_variant() -> Dict[str, Any]:
    registry = first_order_controller_registry()
    default_id = str(registry["default_mlp_variant_id"])
    for variant in first_order_mlp_variants():
        if str(variant["id"]) == default_id:
            return dict(variant)
    raise KeyError(f"Default first-order MLP variant '{default_id}' not found in manifest")


def first_order_mlp_variant_by_id(variant_id: str) -> Dict[str, Any]:
    for variant in first_order_mlp_variants():
        if str(variant["id"]) == str(variant_id):
            return dict(variant)
    raise KeyError(f"Unknown first-order MLP variant: {variant_id}")


def first_order_universal_robust_track() -> Dict[str, Any]:
    return dict(first_order_manifest()["rl_training"]["universal_robust_track"])
