"""Shared gate calibration for checkpoint serving and RQ1 evaluation."""
from __future__ import annotations
import json
import math
from pathlib import Path
from collections.abc import Mapping


def load_gate_calibration(
    path: Path, *, required: bool = False, probability_source: str = "generated",
    allow_source_mismatch: bool = False,
) -> dict[str, float] | None:
    if path.is_dir():
        path = path / 'gate_calibration.json'
    if not path.is_file():
        if required:
            raise FileNotFoundError(f'gate calibration is missing: {path}')
        return None
    raw = json.loads(path.read_text())
    if not isinstance(raw, Mapping):
        raise TypeError('gate calibration must be a JSON object')
    if raw.get("probability_source", "generated") != probability_source:
        if allow_source_mismatch:
            return None
        raise ValueError(f"gate calibration probability source differs from {probability_source}: {path}")
    result = {'logit_bias': raw.get('logit_bias'), 'temperature': raw.get('temperature', 1.0)}
    for name, value in result.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'invalid gate calibration {name}: {path}')
    if result['temperature'] <= 0:
        raise ValueError('gate calibration temperature must be positive')
    return {k: float(v) for k, v in result.items()}


def calibrate_gate(probability: float, calibration: Mapping[str, float] | None) -> float:
    if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
        raise ValueError('gate probability must be a finite number between zero and one')
    if calibration is None:
        return float(probability)
    p = min(max(float(probability), 1e-5), 1 - 1e-5)
    z = (math.log(p / (1-p)) + calibration['logit_bias']) / calibration['temperature']
    return 1 / (1 + math.exp(-max(min(z, 30), -30)))
