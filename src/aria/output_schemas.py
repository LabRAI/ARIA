"""Model-output schemas; semantic grounding remains a host validation step."""

from __future__ import annotations

from typing import Any, Sequence

from aria.records import ATTRIBUTION_CODES, PATCH_OPERATORS


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def output_schema(task: str, event_ids: Sequence[str]) -> dict[str, Any]:
    text = {"type": "string", "minLength": 1}
    confidence = {"type": "number", "minimum": 0, "maximum": 1}
    event_id = {"type": "string", "enum": list(event_ids)}
    ids = {"type": "array", "items": event_id}
    if task == "deployment_gate":
        return _object({"gate_probability": confidence})
    if not event_ids:
        raise ValueError("structured ARIA decoding requires visible event IDs")
    if task == "failure_localization":
        return _object({
            "first_bad_event_ids": {**ids, "minItems": 1},
            "retained_frontier_ids": ids,
            "evidence": {"type": "array", "minItems": 1, "maxItems": 1, "items": _object({
                "event_id": event_id, "source": {"enum": ["trajectory", "artifact", "workspace"]},
                "quote": {**text, "maxLength": 160}, "reason": text,
            })},
            "confidence": confidence,
        })
    if task == "repair_attribution":
        code = {"type": "string", "enum": sorted(ATTRIBUTION_CODES)}
        return _object({
            "diagnosis": text, "primary_code": code, "secondary_codes": {"type": "array", "items": code},
            "responsible_agents": {"type": "array", "minItems": 1,
                                   "items": _object({"agent_id": text, "role": text})},
            "counterfactual": text, "patch_operator": {"enum": sorted(PATCH_OPERATORS)},
            "confidence": confidence,
        })
    if task == "structured_repair":
        return _object({
            "diagnosis": text, "cut_event_id": {"enum": [None, *event_ids]},
            "replacement_events": {"type": "array", "minItems": 1, "maxItems": 8, "items": _object({
                "event_id": text, "role": text, "kind": {"type": "string", "not": {"const": "tool_result"}},
                "content": {"type": "string"}, "tool_calls": {"type": "array", "items": {"type": "object"}},
                "metadata": {"type": "object", "properties": {"replaces": ids}, "additionalProperties": False},
            })},
        })
    raise ValueError(f"unsupported ARIA output schema: {task}")
