"""Answer-blind model generation for ARIA repair attribution and patches."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aria.prompts import (
    ATTRIBUTION_INSTRUCTION,
    LOCALIZATION_INSTRUCTION,
    STRUCTURED_REPAIR_INSTRUCTION,
    canonical_model_json,
)
from aria.records import (
    ANSWER_LABEL_FIELDS,
    FailureLocalization,
    IncidentRecord,
    RepairAttribution,
    RepairRecord,
    TrajectoryEvent,
)
from common.model_api import ModelApiError
from repair.operator import build_original_problem, build_repair_prompt, diagnose_failure

TRAJECTORY_REPLAY_OPERATOR_ID = "model/aria-repair-v4-localize-attribute-patch-answer-blind"
LEGACY_HEURISTIC_OPERATOR_ID = "heuristic/aria-repair-v2-answer-blind"
MAX_EVENT_CONTENT_CHARS = 12000


@dataclass(frozen=True)
class ProposalCandidate:
    candidate_id: str
    candidate_index: int
    output: FailureLocalization | RepairAttribution | RepairRecord | None
    raw_output: Mapping[str, Any] | None
    validation_status: str
    validation_error: str | None
    response_metadata: Mapping[str, Any]
    generation_attempts: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        output = self.output
        serialized_output = (
            output.model_output()
            if isinstance(output, RepairAttribution)
            else output.to_dict()
            if output is not None
            else None
        )
        return {
            "candidate_id": self.candidate_id,
            "candidate_index": self.candidate_index,
            "output": serialized_output,
            "raw_output": dict(self.raw_output) if self.raw_output is not None else None,
            "validation": {
                "status": self.validation_status,
                "error": self.validation_error,
            },
            "response_metadata": dict(self.response_metadata),
            "generation_attempts": [dict(item) for item in self.generation_attempts],
        }


@dataclass(frozen=True)
class GeneratedRepairTrial:
    trial_id: str
    trial_index: int
    localization_candidate: ProposalCandidate
    attribution_candidate: ProposalCandidate
    patch_candidate: ProposalCandidate

    @property
    def localization(self) -> FailureLocalization | None:
        output = self.localization_candidate.output
        return output if isinstance(output, FailureLocalization) else None

    @property
    def attribution(self) -> RepairAttribution | None:
        output = self.attribution_candidate.output
        return output if isinstance(output, RepairAttribution) else None

    @property
    def repair(self) -> RepairRecord | None:
        output = self.patch_candidate.output
        return output if isinstance(output, RepairRecord) else None

    @property
    def complete(self) -> bool:
        return (
            self.localization is not None
            and self.attribution is not None
            and self.repair is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "trial_index": self.trial_index,
            "localization": self.localization_candidate.to_dict(),
            "attribution": self.attribution_candidate.to_dict(),
            "patch": self.patch_candidate.to_dict(),
            "complete": self.complete,
        }


@dataclass(frozen=True)
class GeneratedRepairEpisode:
    incident: IncidentRecord
    trials: tuple[GeneratedRepairTrial, ...]
    generation_source: str

    @property
    def complete_trials(self) -> tuple[GeneratedRepairTrial, ...]:
        return tuple(trial for trial in self.trials if trial.complete)

    @property
    def attribution(self) -> RepairAttribution:
        trial = next(iter(self.complete_trials), None)
        if trial is None or trial.attribution is None:
            raise ValueError("repair episode has no complete attribution")
        return trial.attribution

    @property
    def adjudication(self) -> Mapping[str, Any]:
        return {
            "method": "independent_repair_trials",
            "status": "generated",
            "trial_count": len(self.trials),
            "complete_trial_count": len(self.complete_trials),
        }

    @property
    def repairs(self) -> tuple[RepairRecord, ...]:
        return tuple(
            trial.repair for trial in self.complete_trials if trial.repair is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident": self.incident.model_input(),
            "repair_trials": [trial.to_dict() for trial in self.trials],
            "generation_source": self.generation_source,
            "adjudication": dict(self.adjudication),
        }


class StructuredRepairGenerationError(RuntimeError):
    """No schema-valid attribution or patch candidate could be generated."""


def generate_repair_episode(
    *,
    source: Mapping[str, Any],
    client: Any | None,
    repair_trials: int,
    structured_output_retries: int,
    max_events: int = 64,
) -> GeneratedRepairEpisode:
    """Generate independent localization -> RA -> patch repair trials."""

    if repair_trials < 1:
        raise ValueError("repair_trials must be positive")
    if structured_output_retries < 0:
        raise ValueError("structured_output_retries must be non-negative")
    incident = build_incident(source, max_events=max_events)
    if client is None:
        return _heuristic_episode(source, incident)

    trials = tuple(
        _generate_repair_trial(
            client,
            incident,
            trial_index=index,
            retries=structured_output_retries,
        )
        for index in range(repair_trials)
    )
    if not any(trial.complete for trial in trials):
        errors = []
        for trial in trials:
            for candidate in (
                trial.localization_candidate,
                trial.attribution_candidate,
                trial.patch_candidate,
            ):
                if candidate.validation_error:
                    errors.append(candidate.validation_error)
        raise StructuredRepairGenerationError(
            "no complete schema-valid repair trial: " + "; ".join(errors[:3])
        )
    return GeneratedRepairEpisode(
        incident=incident,
        trials=trials,
        generation_source="model",
    )


def build_incident(
    source: Mapping[str, Any],
    *,
    max_events: int = 64,
    failed: bool = True,
) -> IncidentRecord:
    result = _mapping(source.get("result"))
    trace_events = source.get("trace_events")
    if not isinstance(trace_events, list):
        trace_events = []
    normalized = [_normalize_event(item, index) for index, item in enumerate(trace_events)]
    normalized = [item for item in normalized if item is not None]
    if not any(_event_is_observable(item) for item in normalized):
        messages = source.get("messages")
        if not isinstance(messages, list):
            messages = []
        normalized = [_normalize_event(item, index) for index, item in enumerate(messages)]
        normalized = [item for item in normalized if item is not None]
    if not normalized:
        final_output = str(result.get("final_output") or result.get("final_answer") or "")
        normalized = [
            TrajectoryEvent(
                event_id="source:event:0001",
                role="assistant",
                kind="message",
                content=final_output or "Completed without an observable final event.",
            )
        ]
    artifact_events = source.get("artifact_events")
    if isinstance(artifact_events, list):
        seen_event_ids = {event.event_id for event in normalized}
        for index, raw in enumerate(artifact_events, start=len(normalized)):
            event = _normalize_event(raw, index)
            if event is not None and event.event_id not in seen_event_ids:
                normalized.append(event)
                seen_event_ids.add(event.event_id)
    events = _bounded_events(normalized, max_events=max_events)
    metadata = _mapping(result.get("sample_metadata"))
    task_text = build_original_problem(dict(result), dict(metadata))
    constraints = {
        "task_text": task_text,
        "benchmark": result.get("benchmark") or metadata.get("benchmark"),
        "mas": result.get("mas"),
        "answer_blind": True,
    }
    return IncidentRecord(
        incident_id=str(
            result.get("run_id")
            or result.get("sample_id")
            or metadata.get("sample_id")
            or "repair-incident"
        ),
        events=tuple(events),
        feedback=(
            "The public evaluator resolved this trajectory as incorrect; no reference answer is disclosed."
            if failed
            else None
        ),
        constraints=_without_answer_labels(constraints),
    )


def materialize_continuation_task(
    incident: IncidentRecord,
    attribution: RepairAttribution,
    repair: RepairRecord,
) -> dict[str, Any]:
    """Convert a runtime patch proposal into the current native rerun input.

    Native MAS runners do not expose uniform checkpoint restoration yet, so this
    materialization is explicitly a non-exact continuation. The structured patch
    remains the model target and the host records the limitation.
    """

    task_text = str(incident.constraints.get("task_text") or "")
    cut_index = repair.cut_index(incident.events)
    retained = [event.to_dict() for event in incident.events[: cut_index + 1]]
    payload = {
        "localization": attribution.localization.to_dict(),
        "attribution": attribution.model_output(),
        "retained_prefix": retained,
        "replacement_events": [event.to_dict() for event in repair.replacement_events],
    }
    prompt_text = (
        f"{task_text}\n\n"
        "[ARIA RUNTIME PATCH]\n"
        "The host validated the following answer-blind causal patch. Continue the task "
        "from the retained state and enact the replacement events. Do not repeat the "
        "discarded failed suffix and do not claim tool results before executing them.\n"
        f"{json.dumps(payload, ensure_ascii=True, sort_keys=True)}"
    )
    return {
        "initial_problem": task_text,
        "patched_task": prompt_text,
        "prompt_text": prompt_text,
        "retained_prefix": retained,
        "repair": repair.to_dict(),
        "localization": attribution.localization.to_dict(),
        "attribution": attribution.model_output(),
        "execution_mode": "non_exact_continuation",
        "answer_blind": True,
    }


def _generate_repair_trial(
    client: Any,
    incident: IncidentRecord,
    *,
    trial_index: int,
    retries: int,
) -> GeneratedRepairTrial:
    trial_id = f"repair-trial:{trial_index:02d}"
    localization_candidate = _generate_localization_candidate(
        client,
        incident,
        trial_index=trial_index,
        retries=retries,
    )
    localization = localization_candidate.output
    if not isinstance(localization, FailureLocalization):
        return GeneratedRepairTrial(
            trial_id=trial_id,
            trial_index=trial_index,
            localization_candidate=localization_candidate,
            attribution_candidate=_skipped_candidate(
                f"{trial_id}:attribution", trial_index, "localization_invalid"
            ),
            patch_candidate=_skipped_candidate(
                f"{trial_id}:patch", trial_index, "localization_invalid"
            ),
        )
    attribution_candidate = _generate_attribution_candidate(
        client,
        incident,
        localization,
        trial_index=trial_index,
        retries=retries,
    )
    attribution = attribution_candidate.output
    if not isinstance(attribution, RepairAttribution):
        return GeneratedRepairTrial(
            trial_id=trial_id,
            trial_index=trial_index,
            localization_candidate=localization_candidate,
            attribution_candidate=attribution_candidate,
            patch_candidate=_skipped_candidate(
                f"{trial_id}:patch", trial_index, "attribution_invalid"
            ),
        )
    patch_candidate = _generate_patch_candidate(
        client,
        incident,
        attribution,
        trial_index=trial_index,
        retries=retries,
    )
    return GeneratedRepairTrial(
        trial_id=trial_id,
        trial_index=trial_index,
        localization_candidate=localization_candidate,
        attribution_candidate=attribution_candidate,
        patch_candidate=patch_candidate,
    )


def _generate_localization_candidate(
    client: Any,
    incident: IncidentRecord,
    *,
    trial_index: int,
    retries: int,
) -> ProposalCandidate:
    messages = _messages(
        LOCALIZATION_INSTRUCTION,
        incident.model_input(),
    )

    def validate(raw: Mapping[str, Any]) -> FailureLocalization:
        return FailureLocalization.from_dict(
            raw,
            events=incident.events,
            name=f"repair trial {trial_index} localization",
        )

    return _generate_candidate(
        client,
        messages,
        validate,
        candidate_id=f"repair-trial:{trial_index:02d}:localization",
        candidate_index=trial_index,
        retries=retries,
    )


def _generate_attribution_candidate(
    client: Any,
    incident: IncidentRecord,
    localization: FailureLocalization,
    *,
    trial_index: int,
    retries: int,
) -> ProposalCandidate:
    messages = _messages(
        ATTRIBUTION_INSTRUCTION,
        incident.attribution_model_input(localization),
    )

    def validate(raw: Mapping[str, Any]) -> RepairAttribution:
        return RepairAttribution.from_dict(
            raw,
            events=incident.events,
            localization=localization,
            name=f"repair trial {trial_index} attribution",
        )

    return _generate_candidate(
        client,
        messages,
        validate,
        candidate_id=f"repair-trial:{trial_index:02d}:attribution",
        candidate_index=trial_index,
        retries=retries,
    )


def _generate_patch_candidate(
    client: Any,
    incident: IncidentRecord,
    attribution: RepairAttribution,
    *,
    trial_index: int,
    retries: int,
) -> ProposalCandidate:
    canonical_diagnosis = json.dumps(attribution.diagnosis, ensure_ascii=True)
    messages = _messages(
        (
            f"{STRUCTURED_REPAIR_INSTRUCTION} Derive the patch from the accepted attribution. "
            "The diagnosis field is immutable: copy the following exact JSON string value "
            "byte-for-byte as the diagnosis value, without paraphrasing or normalizing it: "
            f"{canonical_diagnosis}."
        ),
        incident.patch_model_input(attribution),
    )

    def validate(raw: Mapping[str, Any]) -> RepairRecord:
        repair = RepairRecord.from_dict(
            raw,
            events=incident.events,
            name=f"repair trial {trial_index} patch",
        )
        if repair.diagnosis != attribution.diagnosis:
            raise ValueError("patch diagnosis must equal the accepted attribution diagnosis")
        first_bad_index = min(
            index
            for index, event in enumerate(incident.events)
            if event.event_id in attribution.first_bad_event_ids
        )
        if repair.cut_index(incident.events) >= first_bad_index:
            raise ValueError("patch cut must precede the first bad event")
        discarded = incident.events[repair.cut_index(incident.events) + 1 :]
        if _event_signature(repair.replacement_events) == _event_signature(discarded):
            raise ValueError("replacement_events reproduce the failed suffix unchanged")
        return repair

    return _generate_candidate(
        client,
        messages,
        validate,
        candidate_id=f"repair-trial:{trial_index:02d}:patch",
        candidate_index=trial_index,
        retries=retries,
    )


def _skipped_candidate(
    candidate_id: str,
    candidate_index: int,
    reason: str,
) -> ProposalCandidate:
    return ProposalCandidate(
        candidate_id=candidate_id,
        candidate_index=candidate_index,
        output=None,
        raw_output=None,
        validation_status="skipped",
        validation_error=reason,
        response_metadata={},
        generation_attempts=(),
    )


def _generate_candidate(
    client: Any,
    messages: list[dict[str, Any]],
    validator: Any,
    *,
    candidate_id: str,
    candidate_index: int,
    retries: int,
) -> ProposalCandidate:
    logger = logging.getLogger("masfm.repair_dataset")
    current_messages = list(messages)
    last_raw: Mapping[str, Any] | None = None
    last_error: Exception | None = None
    metadata: dict[str, Any] = {}
    generation_attempts: list[dict[str, Any]] = []
    for attempt in range(retries + 1):
        text: str | None = None
        parsed_raw: Mapping[str, Any] | None = None
        try:
            text, metadata = _invoke(client, current_messages)
            raw = _decode_json_object(text)
            parsed_raw = raw
            last_raw = raw
            output = validator(raw)
            generation_attempts.append(
                {
                    "attempt": attempt + 1,
                    "messages": [dict(message) for message in current_messages],
                    "raw_response": text,
                    "parsed_response": dict(raw),
                    "validation": {"status": "valid", "error": None},
                    "response_metadata": dict(metadata),
                }
            )
            return ProposalCandidate(
                candidate_id=candidate_id,
                candidate_index=candidate_index,
                output=output,
                raw_output=raw,
                validation_status="valid",
                validation_error=None,
                response_metadata={**metadata, "structured_attempt": attempt + 1},
                generation_attempts=tuple(generation_attempts),
            )
        except ModelApiError:
            raise
        except Exception as exc:  # noqa: BLE001 - invalid proposals are data, then retried.
            last_error = exc
            generation_attempts.append(
                {
                    "attempt": attempt + 1,
                    "messages": [dict(message) for message in current_messages],
                    "raw_response": text,
                    "parsed_response": dict(parsed_raw) if parsed_raw is not None else None,
                    "validation": {
                        "status": "invalid",
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                    "response_metadata": dict(metadata),
                }
            )
            logger.warning(
                "Invalid structured repair output candidate=%s attempt=%s/%s error=%s",
                candidate_id,
                attempt + 1,
                retries + 1,
                exc,
            )
            current_messages = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        "The previous response failed host validation: "
                        f"{type(exc).__name__}: {exc}. Return a corrected JSON object only."
                    ),
                },
            ]
    return ProposalCandidate(
        candidate_id=candidate_id,
        candidate_index=candidate_index,
        output=None,
        raw_output=last_raw,
        validation_status="invalid",
        validation_error=(
            f"{type(last_error).__name__}: {last_error}" if last_error else "unknown"
        ),
        response_metadata=metadata,
        generation_attempts=tuple(generation_attempts),
    )


def _invoke(client: Any, messages: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    invoke = getattr(client, "invoke", None)
    if callable(invoke):
        response = invoke(messages)
        text = getattr(response, "text", None)
        if not isinstance(text, str):
            raise TypeError("model invoke() response must expose string text")
        return text, {
            "response_id": getattr(response, "response_id", None),
            "protocol": getattr(response, "protocol", None),
            "usage": dict(getattr(response, "usage", {}) or {}),
            "latency_seconds": getattr(response, "latency_seconds", None),
            "attempt_count": getattr(response, "attempt_count", None),
        }
    complete = getattr(client, "complete", None)
    if not callable(complete):
        raise TypeError("repair model client must implement invoke() or complete()")
    text = complete(messages)
    if not isinstance(text, str):
        raise TypeError("model complete() must return a string")
    return text, {"protocol": "injected-client"}


def _heuristic_episode(
    source: Mapping[str, Any], incident: IncidentRecord
) -> GeneratedRepairEpisode:
    result = dict(_mapping(source.get("result")))
    messages = source.get("messages") if isinstance(source.get("messages"), list) else []
    traces = source.get("trace_events") if isinstance(source.get("trace_events"), list) else []
    diagnosis = diagnose_failure(result, messages, traces)
    repair_prompt = build_repair_prompt(result, diagnosis)
    benchmark = str(result.get("benchmark") or "")
    primary_code, operator = {
        "gaia": ("RA-2.4", "remove_unsupported_claim"),
        "humaneval": ("RA-5.5", "replace_implementation"),
        "swe-bench-lite": ("RA-5.5", "replace_implementation"),
        "olympiadbench": ("RA-3.2", "replace_computation"),
        "gsm-plus": ("RA-3.2", "replace_computation"),
        "mmlu": ("RA-3.4", "replace_decision"),
    }.get(benchmark, ("RA-6.5", "normalize_output"))
    first_bad = incident.events[-1]
    cut = incident.events[-2].event_id if len(incident.events) > 1 else None
    role = first_bad.role
    attribution = RepairAttribution.from_dict(
        {
            "diagnosis": str(diagnosis.get("issue") or "The failed event requires replacement."),
            "primary_code": primary_code,
            "secondary_codes": [],
            "first_bad_event_ids": [first_bad.event_id],
            "retained_frontier_ids": [cut] if cut else [],
            "responsible_agents": [{"agent_id": role, "role": role}],
            "evidence": [
                {
                    "event_id": first_bad.event_id,
                    "source": "trajectory",
                    "quote": first_bad.content[:MAX_EVENT_CONTENT_CHARS] or "empty event",
                    "reason": str(diagnosis.get("issue") or "Evaluator-resolved failure."),
                }
            ],
            "counterfactual": "Replacing this event with the bounded repair instruction permits a new continuation.",
            "patch_operator": operator,
            "confidence": 0.25,
        },
        events=incident.events,
        name="heuristic fallback attribution",
    )
    repair = RepairRecord.from_dict(
        {
            "diagnosis": attribution.diagnosis,
            "cut_event_id": cut,
            "replacement_events": [
                {
                    "event_id": "patch:event:0001",
                    "role": role,
                    "kind": "message",
                    "content": str(repair_prompt.get("patch_instruction") or diagnosis.get("attention_instruction")),
                    "tool_calls": [],
                    "metadata": {
                        "replaces": [first_bad.event_id],
                        "fallback": True,
                    },
                }
            ],
        },
        events=incident.events,
        name="heuristic fallback patch",
    )
    localization_candidate = ProposalCandidate(
        candidate_id="repair-trial:00:localization",
        candidate_index=0,
        output=attribution.localization,
        raw_output=attribution.localization.to_dict(),
        validation_status="valid",
        validation_error=None,
        response_metadata={"source": "heuristic_fallback"},
    )
    attribution_candidate = ProposalCandidate(
        candidate_id="repair-trial:00:attribution",
        candidate_index=0,
        output=attribution,
        raw_output=attribution.model_output(),
        validation_status="valid",
        validation_error=None,
        response_metadata={"source": "heuristic_fallback"},
    )
    patch_candidate = ProposalCandidate(
        candidate_id="repair-trial:00:patch",
        candidate_index=0,
        output=repair,
        raw_output=repair.to_dict(),
        validation_status="valid",
        validation_error=None,
        response_metadata={"source": "heuristic_fallback"},
    )
    return GeneratedRepairEpisode(
        incident=incident,
        trials=(
            GeneratedRepairTrial(
                trial_id="repair-trial:00",
                trial_index=0,
                localization_candidate=localization_candidate,
                attribution_candidate=attribution_candidate,
                patch_candidate=patch_candidate,
            ),
        ),
        generation_source="heuristic_fallback",
    )


def _normalize_event(raw: Any, index: int) -> TrajectoryEvent | None:
    if not isinstance(raw, Mapping):
        return None
    record = _mapping(raw.get("record"))
    role = str(
        raw.get("mas_agent_role")
        or raw.get("mas_agent_id")
        or raw.get("role")
        or raw.get("source")
        or record.get("source")
        or "agent"
    )
    content = _normalize_observable_text(
        str(
            raw.get("content")
            or raw.get("text")
            or raw.get("message")
            or raw.get("result")
            or record.get("content")
            or ""
        )
    )[:MAX_EVENT_CONTENT_CHARS]
    tool_calls = raw.get("tool_calls") or record.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        tool_calls = []
    event_id = str(raw.get("event_id") or f"source:event:{index + 1:04d}")
    if not event_id.startswith("source:"):
        event_id = f"source:{event_id}"
    return TrajectoryEvent(
        event_id=event_id,
        role=role,
        kind=str(raw.get("kind") or record.get("type") or "message"),
        content=content,
        tool_calls=tuple(
            dict(_without_answer_labels(item))
            for item in tool_calls
            if isinstance(item, Mapping)
        ),
        metadata=_without_answer_labels(
            {
                "turn": raw.get("turn") or record.get("turn") or index + 1,
                "hook": raw.get("hook"),
                "tool_results": raw.get("tool_results") or [],
                "artifact": raw.get("artifact"),
            }
        ),
    )


def _event_is_observable(event: TrajectoryEvent) -> bool:
    return bool(
        event.content.strip()
        or event.tool_calls
        or event.metadata.get("tool_results")
        or event.metadata.get("artifact")
    )


def _normalize_observable_text(value: str) -> str:
    return "".join(" " if unicodedata.category(char) == "Zs" else char for char in value)


def _bounded_events(
    events: Sequence[TrajectoryEvent], *, max_events: int
) -> list[TrajectoryEvent]:
    if max_events < 1:
        raise ValueError("attribution_max_events must be positive")
    if len(events) <= max_events:
        return list(events)
    prefix_count = min(4, max_events // 4)
    return [*events[:prefix_count], *events[-(max_events - prefix_count) :]]


def _messages(instruction: str, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": canonical_model_json(payload)},
    ]


def _decode_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*\n(?P<payload>.*)\n\s*```",
        candidate,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced is not None:
        candidate = fenced.group("payload").strip()
    raw = json.loads(candidate)
    if not isinstance(raw, dict):
        raise TypeError("model completion JSON must be an object")
    return raw


def _without_answer_labels(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_answer_labels(item)
            for key, item in value.items()
            if str(key) not in ANSWER_LABEL_FIELDS
        }
    if isinstance(value, list):
        return [_without_answer_labels(item) for item in value]
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _event_signature(events: Sequence[TrajectoryEvent]) -> tuple[tuple[Any, ...], ...]:
    return tuple(
        (
            event.role,
            event.kind,
            event.content.strip(),
            json.dumps(list(event.tool_calls), sort_keys=True),
        )
        for event in events
    )
