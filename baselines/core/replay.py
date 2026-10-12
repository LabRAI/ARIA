"""Explicit host-owned replay boundary for executable baseline evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from baselines.core.types import BaselineSample, ExecutionRecord


class ReplayBackend(Protocol):
    """A host adapter that restores state, executes, and invokes the evaluator."""

    def execute(
        self,
        sample: BaselineSample,
        intervention: Mapping[str, Any] | None,
        *,
        label: str,
    ) -> ExecutionRecord: ...


class EvidenceBackend(Protocol):
    """Optional ErrorProbe tool-grounding surface."""

    def verify_evidence(
        self, sample: BaselineSample, probe: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


def execute(
    backend: ReplayBackend | None,
    sample: BaselineSample,
    intervention: Mapping[str, Any] | None,
    *,
    label: str,
) -> ExecutionRecord | None:
    if backend is None:
        return None
    record = backend.execute(sample, intervention, label=label)
    if not isinstance(record, ExecutionRecord):
        raise TypeError("replay backend must return ExecutionRecord")
    return record
