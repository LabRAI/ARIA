"""Shared job configuration contracts and strict value validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from common.model_api import ApiConfig


class AriaSettings(Protocol):
    enabled: bool
    mode: str


@dataclass(frozen=True)
class OutputConfig:
    existing_output_policy: str = "error"


@dataclass(frozen=True)
class ExperimentConfig:
    data_dir: Path
    external_dir: Path
    output_dir: Path
    workspace_dir: Path
    log_level: str
    api: ApiConfig | None
    aria: AriaSettings
    experiments: list[dict[str, Any]]
    prepare: dict[str, Any]
    output: OutputConfig = field(default_factory=OutputConfig)
    repair_datasets: list[dict[str, Any]] = field(default_factory=list)
    baselines: list[dict[str, Any]] = field(default_factory=list)
    baseline_workers: int = 1
    config_path: Path | None = None
    job_type: str = "inference"
    runtime_kind: str = "endpoint"
    generalization: dict[str, Any] = field(default_factory=dict)


def as_bool(value: Any, default: bool, *, name: str = "boolean value") -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise TypeError(f"{name} must be a YAML boolean")


def normalize_existing_output_policy(value: Any) -> str:
    if value is None:
        return "error"
    policy = str(value)
    if policy not in {"error", "resume", "timestamp", "overwrite"}:
        raise ValueError(
            "unsupported output.existing_output_policy: "
            f"{value!r} (expected error, resume, timestamp, or overwrite)"
        )
    return policy


def required_keys(raw: dict[str, Any], required: set[str], name: str) -> None:
    missing = sorted(required - set(raw))
    if missing:
        raise KeyError(f"missing required keys in {name}: {', '.join(missing)}")
