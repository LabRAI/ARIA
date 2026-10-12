"""Canonical inputs and outputs shared by every baseline method."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class BaselineEvent:
    event_id: str
    role: str
    kind: str
    content: str = ""
    tool_calls: tuple[dict[str, Any], ...] = ()
    tool_results: tuple[dict[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "role": self.role,
            "kind": self.kind,
            "content": self.content,
            "tool_calls": [dict(item) for item in self.tool_calls],
            "tool_results": [dict(item) for item in self.tool_results],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BaselineSample:
    sample_id: str
    task_id: str
    task: str
    events: tuple[BaselineEvent, ...]
    constraints: Mapping[str, Any] = field(default_factory=dict)
    feedback: str | None = None
    benchmark: str | None = None
    mas: str | None = None
    split: str | None = None
    source_evaluation: Mapping[str, Any] = field(default_factory=dict)
    control_evaluation: Mapping[str, Any] = field(default_factory=dict)
    hidden_reference: Mapping[str, Any] = field(default_factory=dict, repr=False)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def model_view(self, *, include_feedback: bool) -> dict[str, Any]:
        """Return the only sample representation permitted in model prompts."""

        return {
            "sample_id": self.sample_id,
            "task": self.task,
            "events": [event.to_dict() for event in self.events],
            "constraints": dict(self.constraints),
            "feedback": self.feedback if include_feedback else None,
            "benchmark": self.benchmark,
            "mas": self.mas,
        }

    @property
    def source_success(self) -> bool | None:
        value = self.source_evaluation.get("success")
        if not isinstance(value, bool):
            value = self.source_evaluation.get("correct")
        return value if isinstance(value, bool) else None

    @property
    def control_success(self) -> bool | None:
        value = self.control_evaluation.get("success")
        if not isinstance(value, bool):
            value = self.control_evaluation.get("correct")
        return value if isinstance(value, bool) else self.source_success

    @property
    def control_score(self) -> float | None:
        value = self.control_evaluation.get("score")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        value = self.source_evaluation.get("score")
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


@dataclass(frozen=True)
class ModelCallRecord:
    stage: str
    messages: tuple[dict[str, Any], ...]
    raw_response: str
    parsed_response: Any
    validation_error: str | None
    response_metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "messages": [dict(message) for message in self.messages],
            "raw_response": self.raw_response,
            "parsed_response": self.parsed_response,
            "validation": {
                "status": "valid" if self.validation_error is None else "invalid",
                "error": self.validation_error,
            },
            "response_metadata": dict(self.response_metadata),
        }


@dataclass(frozen=True)
class ExecutionRecord:
    label: str
    status: str
    evaluation: Mapping[str, Any] = field(default_factory=dict)
    execution_mode: str = "unknown"
    execution_count: float = 1.0
    artifact_refs: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def success(self) -> bool | None:
        value = self.evaluation.get("success")
        if not isinstance(value, bool):
            value = self.evaluation.get("correct")
        return value if isinstance(value, bool) else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "status": self.status,
            "evaluation": dict(self.evaluation),
            "execution_mode": self.execution_mode,
            "execution_count": self.execution_count,
            "artifact_refs": dict(self.artifact_refs),
            "error": self.error,
        }


@dataclass
class MethodResult:
    sample_id: str
    method: str
    setting: str
    status: str
    intervention: dict[str, Any] | None = None
    attribution: dict[str, Any] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    model_calls: list[ModelCallRecord] = field(default_factory=list)
    executions: list[ExecutionRecord] = field(default_factory=list)
    source_execution_count: float = 1.0
    selected_execution_label: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def execution_budget(self) -> float:
        executed = sum(execution.execution_count for execution in self.executions)
        if self.setting == "active":
            # The source row is the matched uninterrupted control. A native
            # active execution replaces that control; it is not charged on top
            # of it. Multiple full retries still increase EB.
            return max(self.source_execution_count, executed)
        return self.source_execution_count + executed

    @property
    def selected_execution(self) -> ExecutionRecord | None:
        if self.selected_execution_label is not None:
            for execution in self.executions:
                if execution.label == self.selected_execution_label:
                    return execution
        return self.executions[-1] if self.executions else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "aria-baseline-result-v1",
            "sample_id": self.sample_id,
            "method": self.method,
            "setting": self.setting,
            "status": self.status,
            "intervention": self.intervention,
            "attribution": self.attribution,
            "diagnostics": self.diagnostics,
            "model_calls": [call.to_dict() for call in self.model_calls],
            "executions": [execution.to_dict() for execution in self.executions],
            "source_execution_count": self.source_execution_count,
            "execution_budget": self.execution_budget,
            "selected_execution_label": self.selected_execution_label,
            "provenance": self.provenance,
        }


def dataclass_dict(value: Any) -> dict[str, Any]:
    """Small public helper for method-specific dataclasses."""

    return asdict(value)
