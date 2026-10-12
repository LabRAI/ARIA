"""Method protocol and shared validation helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from baselines.core.models import ModelAdapter, require_keys
from baselines.core.replay import ReplayBackend
from baselines.core.types import BaselineSample, MethodResult


class BaselineMethod(ABC):
    name: str
    setting: str
    requires_model: bool = True
    provenance: Mapping[str, Any] = {}

    def __init__(self, **options: Any) -> None:
        self.options = dict(options)

    @abstractmethod
    def run(
        self,
        sample: BaselineSample,
        model: ModelAdapter | None,
        replay: ReplayBackend | None = None,
    ) -> MethodResult: ...

    def result(self, sample: BaselineSample, *, status: str) -> MethodResult:
        return MethodResult(
            sample_id=sample.sample_id,
            method=self.name,
            setting=self.setting,
            status=status,
            provenance=dict(self.provenance),
        )

    def require_model(self, model: ModelAdapter | None) -> ModelAdapter:
        if model is None:
            raise ValueError(f"baseline method {self.name} requires an api model")
        return model


def validate_attribution(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError("attribution must be an object")
    value = dict(raw)
    aliases = {
        "responsible_agent": ("agent", "role", "responsible_role"),
        "step_index": (
            "failure_step_index",
            "zero_based_step_index",
            "step",
            "index",
        ),
        "reason": ("diagnosis", "rationale", "explanation"),
    }
    normalizations: list[str] = []
    for target, candidates in aliases.items():
        if target in value:
            continue
        for candidate in candidates:
            if candidate in value:
                value[target] = value[candidate]
                normalizations.append(f"{candidate}->{target}")
                break
    value = require_keys(
        value, {"responsible_agent", "step_index", "reason"}, name="attribution"
    )
    if not isinstance(value["step_index"], int) or isinstance(
        value["step_index"], bool
    ):
        raise TypeError("attribution.step_index must be an integer")
    if normalizations:
        value["_schema_normalizations"] = normalizations
    return value


def validate_intervention(raw: Any, sample: BaselineSample) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError("intervention must be an object")
    value = dict(raw)
    normalizations: list[str] = []
    aliases = {
        "diagnosis": ("reason", "rationale", "hypothesis"),
        "cut_event_id": ("cut_event", "cut_id"),
        "replacement_events": ("patch_events", "replacement_event", "replacements"),
    }
    for target, candidates in aliases.items():
        if target in value:
            continue
        for candidate in candidates:
            if candidate in value:
                value[target] = value[candidate]
                normalizations.append(f"{candidate}->{target}")
                break
    value = require_keys(
        value, {"diagnosis", "cut_event_id", "replacement_events"}, name="intervention"
    )
    cut = value["cut_event_id"]
    event_ids = {event.event_id for event in sample.events}
    if cut not in event_ids and str(cut or "").strip().lower() in {
        "source:event:0000",
        "event:0000",
        "before_start",
        "before-start",
    }:
        cut = None
        value["cut_event_id"] = None
        normalizations.append("before_first_event_sentinel->null_cut")
    if isinstance(cut, str) and cut not in event_ids:
        prefix, separator, suffix = cut.rpartition(":")
        if separator and suffix.isdigit():
            requested = int(suffix)
            candidates: list[tuple[int, str]] = []
            for event_id in event_ids:
                event_prefix, event_separator, event_suffix = event_id.rpartition(":")
                if (
                    event_separator
                    and event_prefix == prefix
                    and event_suffix.isdigit()
                    and int(event_suffix) < requested
                ):
                    candidates.append((int(event_suffix), event_id))
            if candidates:
                cut = max(candidates)[1]
                value["cut_event_id"] = cut
                normalizations.append("cut_event_id->nearest_prior_source_event")
    if cut is not None and cut not in event_ids:
        raise ValueError("intervention.cut_event_id is not in the trajectory")
    replacements = value["replacement_events"]
    if isinstance(replacements, dict):
        replacements = [replacements]
        normalizations.append("replacement_event_object->list")
    if not isinstance(replacements, list) or not replacements:
        raise ValueError("intervention.replacement_events must be non-empty")
    fallback_role = next(
        (
            event.role
            for event in reversed(sample.events)
            if cut is None or event.event_id == cut
        ),
        sample.events[-1].role,
    )
    normalized_replacements: list[dict[str, Any]] = []
    source_event_ids = {event.event_id for event in sample.events}
    used_replacement_ids: set[str] = set()
    next_patch_index = 0
    for index, raw_event in enumerate(replacements):
        event = dict(raw_event) if isinstance(raw_event, Mapping) else None
        if event is None:
            normalizations.append(f"replacement[{index}]:drop_non_object")
            continue
        if "content" not in event:
            for alias in ("text", "message", "instruction", "modified_content"):
                if alias in event:
                    event["content"] = event[alias]
                    normalizations.append(f"replacement[{index}]:{alias}->content")
                    break
        tool_calls = event.get("tool_calls")
        if (
            (not isinstance(event.get("content"), str) or not event["content"].strip())
            and isinstance(tool_calls, list)
            and any(isinstance(call, Mapping) and call.get("name") for call in tool_calls)
        ):
            event["content"] = "Execute the proposed tool call with the supplied arguments."
            normalizations.append(f"replacement[{index}]:tool_call->content")
        if not isinstance(event.get("content"), str) or not event["content"].strip():
            normalizations.append(f"replacement[{index}]:drop_empty")
            continue
        if not isinstance(event.get("role"), str) or not event["role"].strip():
            event["role"] = fallback_role
            normalizations.append(f"replacement[{index}]:default_role")
        if not isinstance(event.get("kind"), str) or not event["kind"].strip():
            event["kind"] = "message"
            normalizations.append(f"replacement[{index}]:default_kind")
        if not isinstance(event, dict):
            raise TypeError(f"replacement event {index} must be an object")
        require_keys(
            event, {"role", "kind", "content"}, name=f"replacement event {index}"
        )
        if event["kind"] == "tool_result":
            raise ValueError("an intervention cannot assert an unexecuted tool result")
        event_id = event.get("event_id")
        if not isinstance(event_id, str) or not event_id.strip():
            event_id = None
        else:
            event_id = event_id.strip()
        if (
            event_id is None
            or event_id in source_event_ids
            or event_id in used_replacement_ids
        ):
            original = event_id or "<missing>"
            while True:
                candidate = f"baseline:patch:{next_patch_index:04d}"
                next_patch_index += 1
                if (
                    candidate not in source_event_ids
                    and candidate not in used_replacement_ids
                ):
                    break
            event_id = candidate
            normalizations.append(
                f"replacement[{index}]:event_id:{original}->{event_id}"
            )
        event["event_id"] = event_id
        used_replacement_ids.add(event_id)
        event.setdefault("tool_calls", [])
        event.setdefault("metadata", {})
        normalized_replacements.append(event)
    if not normalized_replacements:
        raise ValueError("intervention.replacement_events has no executable event")
    value["replacement_events"] = normalized_replacements
    # Apply the host's real schema inside structured generation, so the model
    # can retry instead of discovering an incompatible record only at replay.
    from aria.records import RepairRecord, TrajectoryEvent

    RepairRecord.from_dict(
        {key: value[key] for key in ("diagnosis", "cut_event_id", "replacement_events")},
        events=tuple(TrajectoryEvent.from_dict(event.to_dict()) for event in sample.events),
    )
    if normalizations:
        value["_schema_normalizations"] = normalizations
    return value
