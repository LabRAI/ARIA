"""The shared repair prompt used after attribution-only baselines."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from baselines.core.method import validate_intervention
from baselines.core.models import ModelAdapter
from baselines.core.types import BaselineSample, ModelCallRecord

REPAIR_SYSTEM = """You repair a failed multi-agent execution.
Return JSON only with diagnosis, cut_event_id, and replacement_events.
Retain the valid prefix and replace the smallest sufficient suffix while obeying task constraints.
cut_event_id is the last event retained in the repaired trajectory. If the first bad event is E_j,
cut_event_id must be E_{j-1}; never use the event being replaced as the cut. Use null only when
nothing can be retained. Replacement event IDs must be new patch IDs, not source event IDs, and
must be unique within replacement_events. Replacement events need role, kind, content, tool_calls,
and metadata. They may propose tool calls but must never claim an unexecuted tool result. Hidden
reference answers are unavailable. Use at most four concise replacement events and never copy the
full source trajectory into the response. Keep the complete JSON response under 6000 characters;
each replacement content should be at most 1200 characters."""

REPAIR_SYSTEM += """
Compacted or omitted source content is unavailable evidence, not executable code. Never copy
compaction markers or incomplete tool arguments into a repair. A repair may instruct the host
to inspect files or use tools; only the host continuation can produce their results and the final
answer. Do not fabricate host observations, successful command output, or workspace artifacts.
"""

MINIMAL_REPAIR_RETRY = """Your compact retry was also invalid or truncated. Return exactly one replacement_events item
as a message with content under 400 characters, tool_calls [], and metadata {}. Do not emit tool calls,
tool arguments, code blocks, test suites, or copied trajectory text. Keep diagnosis under 20 words."""


def repair_from_context(
    model: ModelAdapter,
    sample: BaselineSample,
    *,
    stage: str,
    attribution: Mapping[str, Any] | None = None,
    reflection: str | None = None,
) -> tuple[dict[str, Any], ModelCallRecord]:
    payload = sample.model_view(include_feedback=True)
    if attribution is not None:
        payload["attribution"] = dict(attribution)
        hint = _attribution_cut_hint(sample, attribution)
        if hint is not None:
            attributed_event_id, cut_hint_event_id, step_index = hint
            payload["repair_cut_hint"] = {
                "attributed_failure_event_id": attributed_event_id,
                "cut_hint_event_id": cut_hint_event_id,
                "attributed_failure_step_index": step_index,
                "rule": "cut_event_id is the last retained event immediately before the attributed failure event",
            }
    if reflection is not None:
        payload["reflection"] = reflection

    def validator(value: Any) -> dict[str, Any]:
        normalized = validate_intervention(value, sample)
        if attribution is not None:
            hint = _attribution_cut_hint(sample, attribution)
            if hint is not None:
                _, cut_hint_event_id, step_index = hint
                positions = {
                    event.event_id: index for index, event in enumerate(sample.events)
                }
                cut = normalized.get("cut_event_id")
                cut_index = -1 if cut is None else positions[cut]
                if cut_index >= step_index:
                    normalized["cut_event_id"] = cut_hint_event_id
                    _append_normalization(
                        normalized,
                        "cut_normalized_from_attribution",
                    )
        return normalized

    return model.json_call(
        stage=stage,
        system=REPAIR_SYSTEM,
        payload=payload,
        validator=validator,
        allow_truncated_containers=True,
        final_retry_instruction=MINIMAL_REPAIR_RETRY,
    )


def _attribution_cut_hint(
    sample: BaselineSample, attribution: Mapping[str, Any]
) -> tuple[str, str | None, int] | None:
    """Resolve a method-generated attribution to the immediately prior cut."""

    positions = {event.event_id: index for index, event in enumerate(sample.events)}
    raw_index = attribution.get("step_index")
    index: int | None = None
    if isinstance(raw_index, int) and not isinstance(raw_index, bool):
        if 0 <= raw_index < len(sample.events):
            index = raw_index
    if index is None:
        event_id = attribution.get("event_id")
        if isinstance(event_id, str):
            index = positions.get(event_id)
    if index is None:
        return None
    return (
        sample.events[index].event_id,
        sample.events[index - 1].event_id if index > 0 else None,
        index,
    )


def _append_normalization(value: dict[str, Any], marker: str) -> None:
    existing = value.get("_schema_normalizations")
    if not isinstance(existing, list):
        existing = []
        value["_schema_normalizations"] = existing
    if marker not in existing:
        existing.append(marker)
