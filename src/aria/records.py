"""Schema-checked attribution, incident, and repair records.

Models may identify a causal failure and propose replacement events.  The host
still owns replay, tool execution, artifact materialization, and verification,
so model output can never assert that an action already happened.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

ANSWER_LABEL_FIELDS = frozenset(
    {
        "reference_answer",
        "reference_solution",
        "reference_patch",
        "gold_answer",
        "gold_patch",
        "canonical_solution",
        "expected_answer",
        "test",
        "test_patch",
        "fail_to_pass",
        "pass_to_pass",
    }
)

ATTRIBUTION_VERSION = "ARIA-RCA-v1"
ATTRIBUTION_CODES = frozenset(
    {
        "RA-1.1",
        "RA-1.2",
        "RA-1.3",
        "RA-1.4",
        "RA-2.1",
        "RA-2.2",
        "RA-2.3",
        "RA-2.4",
        "RA-3.1",
        "RA-3.2",
        "RA-3.3",
        "RA-3.4",
        "RA-4.1",
        "RA-4.2",
        "RA-4.3",
        "RA-4.4",
        "RA-5.1",
        "RA-5.2",
        "RA-5.3",
        "RA-5.4",
        "RA-5.5",
        "RA-6.1",
        "RA-6.2",
        "RA-6.3",
        "RA-6.4",
        "RA-6.5",
    }
)
PATCH_OPERATORS = frozenset(
    {
        "replace_task_interpretation",
        "replace_plan",
        "replace_decomposition",
        "replace_role_plan",
        "replace_retrieval",
        "reground_evidence",
        "replace_evidence_extraction",
        "remove_unsupported_claim",
        "replace_inference",
        "replace_computation",
        "replace_empirical_claim_with_proof",
        "replace_decision",
        "restore_shared_state",
        "replace_handoff",
        "preserve_agent_evidence",
        "break_nonprogress_cycle",
        "execute_tool_call",
        "replace_tool_call",
        "recover_tool_failure",
        "materialize_artifact",
        "replace_implementation",
        "add_verification",
        "strengthen_verification",
        "reject_failed_candidate",
        "resume_to_completion",
        "normalize_output",
    }
)

_USE_INCIDENT_FEEDBACK = object()
MAX_REPAIR_REPLACEMENT_EVENTS = 8
MAX_REPAIR_JSON_CHARS = 12000


@dataclass(frozen=True)
class TrajectoryEvent:
    """One observable host event with a stable identifier."""

    event_id: str
    role: str
    kind: str
    content: str = ""
    tool_calls: tuple[dict[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls, raw: Mapping[str, Any], *, name: str = "event"
    ) -> TrajectoryEvent:
        event_id = _required_text(raw, "event_id", name)
        role = _required_text(raw, "role", name)
        kind = _required_text(raw, "kind", name)
        tool_calls_raw = raw.get("tool_calls", [])
        if not isinstance(tool_calls_raw, list) or not all(
            isinstance(item, Mapping) for item in tool_calls_raw
        ):
            raise TypeError(f"{name}.tool_calls must be a list of mappings")
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise TypeError(f"{name}.metadata must be a mapping")
        return cls(
            event_id=event_id,
            role=role,
            kind=kind,
            content=str(raw.get("content", "")),
            tool_calls=tuple(dict(item) for item in tool_calls_raw),
            metadata=dict(metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "role": self.role,
            "kind": self.kind,
            "content": self.content,
            "tool_calls": [dict(item) for item in self.tool_calls],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FailureLocalization:
    """The first observable causal error and the last retained trajectory frontier."""

    first_bad_event_ids: tuple[str, ...]
    retained_frontier_ids: tuple[str, ...]
    evidence: tuple[dict[str, str], ...]
    confidence: float

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any],
        *,
        events: Sequence[TrajectoryEvent],
        name: str = "localization",
    ) -> FailureLocalization:
        unexpected = set(raw) - {
            "first_bad_event_ids",
            "retained_frontier_ids",
            "evidence",
            "confidence",
        }
        if unexpected:
            raise ValueError(f"{name} has unsupported fields: {sorted(unexpected)}")
        confidence = raw.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            raise TypeError(f"{name}.confidence must be numeric")
        confidence_value = float(confidence)
        if not 0.0 <= confidence_value <= 1.0:
            raise ValueError(f"{name}.confidence must be between 0 and 1")
        first_bad_event_ids = _text_tuple(raw, "first_bad_event_ids", name=name)
        retained_frontier_ids = _text_tuple(
            raw, "retained_frontier_ids", name=name
        )
        evidence = _attribution_evidence(raw, name=name)
        if not first_bad_event_ids:
            raise ValueError(f"{name}.first_bad_event_ids must not be empty")
        events_by_id = {event.event_id: event for event in events}
        event_ids = set(events_by_id)
        unknown = set((*first_bad_event_ids, *retained_frontier_ids)) - event_ids
        if unknown:
            raise ValueError(
                f"{name} references unknown trajectory events: {sorted(unknown)}"
            )
        event_index = {event.event_id: index for index, event in enumerate(events)}
        earliest_bad_index = min(
            event_index[event_id] for event_id in first_bad_event_ids
        )
        if any(
            event_index[event_id] >= earliest_bad_index
            for event_id in retained_frontier_ids
        ):
            raise ValueError(
                f"{name}.retained_frontier_ids must precede the first bad event"
            )
        if not evidence:
            raise ValueError(f"{name}.evidence must not be empty")
        for index, item in enumerate(evidence):
            if item["event_id"] not in event_ids:
                raise ValueError(
                    f"{name}.evidence[{index}] references an unknown trajectory event"
                )
            if item["quote"] not in events_by_id[item["event_id"]].content:
                raise ValueError(
                    f"{name}.evidence[{index}].quote is absent from its trajectory event"
                )
        if not set(first_bad_event_ids) & {
            item["event_id"] for item in evidence
        }:
            raise ValueError(
                f"{name}.evidence must support at least one first bad event"
            )
        return cls(
            first_bad_event_ids=first_bad_event_ids,
            retained_frontier_ids=retained_frontier_ids,
            evidence=evidence,
            confidence=confidence_value,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_bad_event_ids": list(self.first_bad_event_ids),
            "retained_frontier_ids": list(self.retained_frontier_ids),
            "evidence": [dict(item) for item in self.evidence],
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class RepairAttribution:
    """One RA causal diagnosis grounded in a separately generated localization."""

    diagnosis: str
    primary_code: str
    secondary_codes: tuple[str, ...] = ()
    first_bad_event_ids: tuple[str, ...] = ()
    retained_frontier_ids: tuple[str, ...] = ()
    responsible_agents: tuple[dict[str, str], ...] = ()
    evidence: tuple[dict[str, str], ...] = ()
    localization_confidence: float = 0.0
    counterfactual: str = ""
    patch_operator: str = ""
    confidence: float = 0.0

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any],
        *,
        events: Sequence[TrajectoryEvent],
        localization: FailureLocalization | None = None,
        name: str = "attribution",
    ) -> RepairAttribution:
        unexpected = set(raw) - {
            "diagnosis",
            "primary_code",
            "secondary_codes",
            "first_bad_event_ids",
            "retained_frontier_ids",
            "responsible_agents",
            "responsible_agent_ids",
            "evidence",
            "localization_confidence",
            "counterfactual",
            "patch_operator",
            "confidence",
        }
        if unexpected:
            raise ValueError(f"{name} has unsupported fields: {sorted(unexpected)}")
        confidence = raw.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            raise TypeError(f"{name}.confidence must be numeric")
        confidence_value = float(confidence)
        if not 0.0 <= confidence_value <= 1.0:
            raise ValueError(f"{name}.confidence must be between 0 and 1")
        diagnosis = _required_text(raw, "diagnosis", name)
        primary_code = _required_text(raw, "primary_code", name)
        secondary_codes = _text_tuple(raw, "secondary_codes", name=name)
        responsible_agents = _responsible_agents(raw, name=name)
        counterfactual = _required_text(raw, "counterfactual", name)
        patch_operator = _required_text(raw, "patch_operator", name)
        if primary_code not in ATTRIBUTION_CODES:
            raise ValueError(f"{name}.primary_code must be an ARIA RA-* leaf")
        if any(code not in ATTRIBUTION_CODES for code in secondary_codes):
            raise ValueError(f"{name}.secondary_codes must contain RA-* leaves")
        if primary_code in secondary_codes:
            raise ValueError(f"{name}.secondary_codes repeats the primary code")
        if patch_operator not in PATCH_OPERATORS:
            raise ValueError(f"{name}.patch_operator is unsupported")
        selected_localization = localization or FailureLocalization.from_dict(
            {
                "first_bad_event_ids": raw.get("first_bad_event_ids"),
                "retained_frontier_ids": raw.get("retained_frontier_ids", []),
                "evidence": raw.get("evidence", []),
                "confidence": raw.get("localization_confidence", confidence_value),
            },
            events=events,
            name=f"{name}.localization",
        )

        return cls(
            diagnosis=diagnosis,
            primary_code=primary_code,
            secondary_codes=secondary_codes,
            first_bad_event_ids=selected_localization.first_bad_event_ids,
            retained_frontier_ids=selected_localization.retained_frontier_ids,
            responsible_agents=responsible_agents,
            evidence=selected_localization.evidence,
            localization_confidence=selected_localization.confidence,
            counterfactual=counterfactual,
            patch_operator=patch_operator,
            confidence=confidence_value,
        )

    @property
    def localization(self) -> FailureLocalization:
        return FailureLocalization(
            first_bad_event_ids=self.first_bad_event_ids,
            retained_frontier_ids=self.retained_frontier_ids,
            evidence=self.evidence,
            confidence=self.localization_confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnosis": self.diagnosis,
            "primary_code": self.primary_code,
            "secondary_codes": list(self.secondary_codes),
            "first_bad_event_ids": list(self.first_bad_event_ids),
            "retained_frontier_ids": list(self.retained_frontier_ids),
            "responsible_agents": [dict(item) for item in self.responsible_agents],
            "evidence": [dict(item) for item in self.evidence],
            "localization_confidence": self.localization_confidence,
            "counterfactual": self.counterfactual,
            "patch_operator": self.patch_operator,
            "confidence": self.confidence,
        }

    def model_output(self) -> dict[str, Any]:
        """Return the exact repair-attribution JSON emitted by ARIA models."""
        return {
            "diagnosis": self.diagnosis,
            "primary_code": self.primary_code,
            "secondary_codes": list(self.secondary_codes),
            "responsible_agents": [dict(item) for item in self.responsible_agents],
            "counterfactual": self.counterfactual,
            "patch_operator": self.patch_operator,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class RepairRecord:
    """The constrained ARIA output: diagnosis, retained cut, replacement events."""

    diagnosis: str
    cut_event_id: str | None
    replacement_events: tuple[TrajectoryEvent, ...]

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any],
        *,
        events: Sequence[TrajectoryEvent],
        name: str = "repair",
    ) -> RepairRecord:
        unexpected = set(raw) - {
            "diagnosis",
            "cut_event_id",
            "replacement_events",
        }
        if unexpected:
            raise ValueError(f"{name} has unsupported fields: {sorted(unexpected)}")
        diagnosis = _required_text(raw, "diagnosis", name)
        cut_event_id_raw = raw.get("cut_event_id")
        if cut_event_id_raw is not None and not isinstance(cut_event_id_raw, str):
            raise TypeError(f"{name}.cut_event_id must be a string or null")
        event_ids = {event.event_id for event in events}
        if cut_event_id_raw is not None and cut_event_id_raw not in event_ids:
            raise ValueError(f"{name}.cut_event_id does not identify an incident event")
        replacement_raw = raw.get("replacement_events")
        if not isinstance(replacement_raw, list) or not replacement_raw:
            raise ValueError(
                f"{name}.replacement_events must contain at least one event"
            )
        if len(replacement_raw) > MAX_REPAIR_REPLACEMENT_EVENTS:
            raise ValueError(
                f"{name}.replacement_events exceeds {MAX_REPAIR_REPLACEMENT_EVENTS} events"
            )
        if len(json.dumps(raw, ensure_ascii=True, sort_keys=True)) > MAX_REPAIR_JSON_CHARS:
            raise ValueError(
                f"{name} exceeds the {MAX_REPAIR_JSON_CHARS}-character training limit"
            )
        replacement_events = tuple(
            TrajectoryEvent.from_dict(item, name=f"{name}.replacement_events[{index}]")
            for index, item in enumerate(replacement_raw)
            if isinstance(item, Mapping)
        )
        if len(replacement_events) != len(replacement_raw):
            raise TypeError(f"{name}.replacement_events must be a list of mappings")
        replacement_ids = [event.event_id for event in replacement_events]
        if len(set(replacement_ids)) != len(replacement_ids):
            raise ValueError(
                f"{name}.replacement_events contains duplicate event_id values"
            )
        retained_ids = {
            event.event_id
            for event in events[: cls(diagnosis, cut_event_id_raw, ()).cut_index(events) + 1]
        }
        if retained_ids & set(replacement_ids):
            raise ValueError(f"{name}.replacement_events collide with retained events")
        if any(event.kind == "tool_result" for event in replacement_events):
            raise ValueError(f"{name} may not assert unexecuted tool results")
        return cls(diagnosis, cut_event_id_raw, replacement_events)

    def cut_index(self, events: Sequence[TrajectoryEvent]) -> int:
        if self.cut_event_id is None:
            return -1
        for index, event in enumerate(events):
            if event.event_id == self.cut_event_id:
                return index
        raise ValueError("repair cut_event_id is absent from the incident")

    def to_dict(self) -> dict[str, Any]:
        return {
            "diagnosis": self.diagnosis,
            "cut_event_id": self.cut_event_id,
            "replacement_events": [
                event.to_dict() for event in self.replacement_events
            ],
        }


@dataclass(frozen=True)
class IncidentRecord:
    """A repair incident suitable for answer-blind ARIA training or inference."""

    incident_id: str
    events: tuple[TrajectoryEvent, ...]
    feedback: str | None
    constraints: Mapping[str, Any]
    attribution: RepairAttribution | None = None
    repair: RepairRecord | None = None
    family_id: str | None = None
    verified_successful: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any],
        *,
        require_repair: bool = False,
        require_verified_success: bool = False,
    ) -> IncidentRecord:
        incident_id = _required_text(raw, "incident_id", "incident")
        events_raw = raw.get("events")
        if not isinstance(events_raw, list) or not events_raw:
            raise ValueError("incident.events must contain at least one event")
        if not all(isinstance(item, Mapping) for item in events_raw):
            raise TypeError("incident.events must be a list of mappings")
        events = tuple(
            TrajectoryEvent.from_dict(item, name=f"incident.events[{index}]")
            for index, item in enumerate(events_raw)
        )
        event_ids = [event.event_id for event in events]
        if len(set(event_ids)) != len(event_ids):
            raise ValueError("incident.events contains duplicate event_id values")
        feedback_raw = raw.get("feedback")
        if feedback_raw is not None and not isinstance(feedback_raw, str):
            raise TypeError("incident.feedback must be a string or null")
        constraints = raw.get("constraints", {})
        if not isinstance(constraints, Mapping):
            raise TypeError("incident.constraints must be a mapping")
        attribution_raw = raw.get("attribution")
        if attribution_raw is not None and not isinstance(attribution_raw, Mapping):
            raise TypeError("incident.attribution must be a mapping")
        attribution = (
            RepairAttribution.from_dict(
                attribution_raw, events=events, name="incident.attribution"
            )
            if isinstance(attribution_raw, Mapping)
            else None
        )
        repair_raw = raw.get("repair")
        if repair_raw is not None and not isinstance(repair_raw, Mapping):
            raise TypeError("incident.repair must be a mapping")
        repair = (
            RepairRecord.from_dict(repair_raw, events=events)
            if isinstance(repair_raw, Mapping)
            else None
        )
        if require_repair and repair is None:
            raise ValueError("a verified training incident requires incident.repair")
        family_raw = raw.get("family_id")
        if family_raw is not None and not isinstance(family_raw, str):
            raise TypeError("incident.family_id must be a string or null")
        metadata = raw.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise TypeError("incident.metadata must be a mapping")
        verified_successful = raw.get("verified_successful", False)
        if not isinstance(verified_successful, bool):
            raise TypeError("incident.verified_successful must be a boolean")
        if require_verified_success and not verified_successful:
            raise ValueError(
                "post-training incidents must set verified_successful: true"
            )
        return cls(
            incident_id=incident_id,
            events=events,
            feedback=feedback_raw,
            constraints=dict(constraints),
            attribution=attribution,
            repair=repair,
            family_id=family_raw,
            verified_successful=verified_successful,
            metadata=dict(metadata),
        )

    def model_input(
        self,
        *,
        feedback: str | None | object = _USE_INCIDENT_FEEDBACK,
    ) -> dict[str, Any]:
        """Return only fields permitted in an ARIA model prompt."""

        visible_feedback = (
            self.feedback if feedback is _USE_INCIDENT_FEEDBACK else feedback
        )
        return {
            "incident_id": self.incident_id,
            "events": [event.to_dict() for event in self.events],
            "feedback": visible_feedback,
            "constraints": _without_answer_labels(self.constraints),
        }

    def patch_model_input(
        self,
        attribution: RepairAttribution | None = None,
        *,
        feedback: str | None | object = _USE_INCIDENT_FEEDBACK,
    ) -> dict[str, Any]:
        selected = attribution or self.attribution
        if selected is None:
            raise ValueError("structured patch input requires an RA attribution")
        return {
            **self.model_input(feedback=feedback),
            "localization": selected.localization.to_dict(),
            "attribution": selected.model_output(),
        }

    def attribution_model_input(
        self,
        localization: FailureLocalization,
        *,
        feedback: str | None | object = _USE_INCIDENT_FEEDBACK,
    ) -> dict[str, Any]:
        return {
            **self.model_input(feedback=feedback),
            "localization": localization.to_dict(),
        }

    def assemble(self, repair: RepairRecord) -> tuple[TrajectoryEvent, ...]:
        cut_index = repair.cut_index(self.events)
        return (*self.events[: cut_index + 1], *repair.replacement_events)


def _required_text(raw: Mapping[str, Any], key: str, name: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}.{key} must be a non-empty string")
    return value


def _optional_text(raw: Mapping[str, Any], key: str, name: str) -> str | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}.{key} must be a non-empty string or null")
    return value


def _required_enum(
    raw: Mapping[str, Any], key: str, allowed: frozenset[str], *, name: str
) -> str:
    value = _required_text(raw, key, name)
    if value not in allowed:
        raise ValueError(f"{name}.{key} must be one of {sorted(allowed)}")
    return value


def _text_tuple(raw: Mapping[str, Any], key: str, *, name: str) -> tuple[str, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise TypeError(f"{name}.{key} must be a list of non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{name}.{key} must not contain duplicates")
    return tuple(value)


def _responsible_agents(
    raw: Mapping[str, Any], *, name: str
) -> tuple[dict[str, str], ...]:
    value = raw.get("responsible_agents")
    if value is None and "responsible_agent_ids" in raw:
        ids = _text_tuple(raw, "responsible_agent_ids", name=name)
        return tuple({"agent_id": item, "role": item} for item in ids)
    if value is None:
        return ()
    if not isinstance(value, list):
        raise TypeError(f"{name}.responsible_agents must be a list")
    agents: list[dict[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TypeError(f"{name}.responsible_agents[{index}] must be an object")
        agent_id = _required_text(item, "agent_id", f"{name}.responsible_agents[{index}]")
        role = _required_text(item, "role", f"{name}.responsible_agents[{index}]")
        agents.append({"agent_id": agent_id, "role": role})
    return tuple(agents)


def _attribution_evidence(
    raw: Mapping[str, Any], *, name: str
) -> tuple[dict[str, str], ...]:
    value = raw.get("evidence", [])
    if not isinstance(value, list):
        raise TypeError(f"{name}.evidence must be a list")
    evidence: list[dict[str, str]] = []
    for index, item in enumerate(value):
        item_name = f"{name}.evidence[{index}]"
        if not isinstance(item, Mapping):
            raise TypeError(f"{item_name} must be an object")
        source = _required_text(item, "source", item_name)
        if source not in {"trajectory", "artifact", "workspace"}:
            raise ValueError(f"{item_name}.source is unsupported")
        evidence.append(
            {
                "event_id": _required_text(item, "event_id", item_name),
                "source": source,
                "quote": _required_text(item, "quote", item_name),
                "reason": _required_text(item, "reason", item_name),
            }
        )
    return tuple(evidence)


def _without_answer_labels(raw: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _without_answer_labels_value(value)
        for key, value in raw.items()
        if str(key) not in ANSWER_LABEL_FIELDS
    }


def _without_answer_labels_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _without_answer_labels(value)
    if isinstance(value, list):
        return [_without_answer_labels_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_without_answer_labels_value(item) for item in value)
    return value
