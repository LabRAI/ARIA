"""Shared, method-neutral baseline contracts."""

from baselines.core.datasets import load_baseline_samples
from baselines.core.evaluation import aggregate_results
from baselines.core.models import ModelAdapter
from baselines.core.types import (
    BaselineEvent,
    BaselineSample,
    ExecutionRecord,
    MethodResult,
)

__all__ = [
    "BaselineEvent",
    "BaselineSample",
    "ExecutionRecord",
    "MethodResult",
    "ModelAdapter",
    "aggregate_results",
    "load_baseline_samples",
]
