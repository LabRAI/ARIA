"""Token-budgeted ARIA messages without slicing instructions or JSON syntax."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from aria.prompts import canonical_model_json

CONTENT_CAPS = (4000, 2000, 1000, 400, 160, 0)
PROMPT_BUDGET_CONTRACT = "structured-json-v3-event-aliases"


def bounded_text(value: str, cap: int) -> str:
    if cap <= 0:
        return ""
    if len(value) <= cap:
        return value
    marker = "...[elided]..."
    if cap <= len(marker) + 2:
        return value[:cap]
    remaining = cap - len(marker)
    head = remaining // 2
    return value[:head] + marker + value[-(remaining - head):]


def reduced_model_input(value: Mapping[str, Any], cap: int) -> dict[str, Any]:
    """Retain identities and decision fields, shorten only evidence payloads."""
    result = dict(value)
    schema = value.get("event_schema")
    content_key = schema.get("content", "content") if isinstance(schema, Mapping) else "content"
    if content_key not in {"content", "text", "body"}:
        raise ValueError("event_schema declares an unsupported content field")
    if isinstance(value.get("events"), list):
        events = []
        for source in value["events"]:
            event = dict(source)
            if isinstance(event.get(content_key), str):
                event[content_key] = bounded_text(event[content_key], cap)
            event["tool_calls"] = [
                _bounded_tool_call(call, cap) for call in event.get("tool_calls", [])
            ]
            if isinstance(event.get("metadata"), Mapping):
                event["metadata"] = _bounded_metadata(event["metadata"], cap)
            events.append(event)
        result["events"] = events
    if isinstance(value.get("constraints"), Mapping):
        constraints = dict(value["constraints"])
        if isinstance(constraints.get("task_text"), str):
            constraints["task_text"] = bounded_text(constraints["task_text"], max(cap, 512))
        result["constraints"] = constraints
    return result


def _bounded_metadata(value: Any, cap: int) -> Any:
    if isinstance(value, Mapping):
        return {
            key: item if key in {"event_id", "role", "kind", "name", "path", "uri"}
            else _bounded_metadata(item, cap)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_bounded_metadata(item, cap) for item in value]
    if isinstance(value, str):
        return bounded_text(value, cap)
    return value


def _bounded_tool_call(call: Mapping[str, Any], cap: int) -> dict[str, Any]:
    result = dict(call)
    for key in ("arguments", "input", "content"):
        value = result.get(key)
        if isinstance(value, str):
            result[key] = bounded_text(value, cap)
        elif isinstance(value, (Mapping, list)):
            encoded = json.dumps(value, ensure_ascii=True, sort_keys=True)
            if len(encoded) > cap:
                result[key] = bounded_text(encoded, cap)
    if isinstance(result.get("function"), Mapping):
        result["function"] = _bounded_tool_call(result["function"], cap)
    return result


def fit_messages(
    messages: list[dict[str, Any]],
    *,
    render: Callable[[list[dict[str, Any]]], str],
    tokenizer: Any,
    max_tokens: int,
    response_prefix: str = "",
    reduce_payload: Callable[[Mapping[str, Any], int], dict[str, Any]] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Fit model-visible JSON; fail explicitly when identities alone exceed budget."""
    def measure(items: list[dict[str, Any]]) -> tuple[str, int]:
        prompt = render(items) + response_prefix
        return prompt, len(tokenizer.encode(prompt, add_special_tokens=False))

    prompt, original_tokens = measure(messages)
    report = {"original_prompt_tokens": original_tokens, "prompt_reduced": False,
              "prompt_content_cap": None, "truncated_prompt": False}
    if original_tokens <= max_tokens:
        return prompt, report
    if len(messages) != 2 or messages[1].get("role") != "user":
        raise ValueError("over-budget prompt requires a system message and one ARIA JSON input")
    try:
        payload = json.loads(messages[1]["content"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("cannot compact over-budget non-JSON input without losing instructions") from exc
    if not isinstance(payload, dict):
        raise ValueError("ARIA prompt input must be a JSON object")
    for cap in CONTENT_CAPS:
        reduced = [dict(message) for message in messages]
        reduced[1]["content"] = canonical_model_json((reduce_payload or reduced_model_input)(payload, cap))
        prompt, tokens = measure(reduced)
        if tokens <= max_tokens:
            return prompt, {**report, "prompt_reduced": True, "prompt_content_cap": cap}
    raise ValueError(
        f"ARIA prompt identities and instructions require {tokens} tokens, exceeding "
        f"max_input_tokens={max_tokens}; increase the input budget"
    )
