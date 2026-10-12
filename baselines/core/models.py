"""Model-neutral structured generation with auditable call records."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from baselines.core.types import ModelCallRecord


class CompletionClient(Protocol):
    def complete(self, messages: list[dict[str, Any]]) -> str: ...


Validator = Callable[[Any], Any]


@dataclass
class ModelAdapter:
    client: CompletionClient
    prompt_char_budget: int | None = None

    def json_call(
        self,
        *,
        stage: str,
        system: str,
        payload: Mapping[str, Any],
        validator: Validator | None = None,
        allow_truncated_containers: bool = False,
        final_retry_instruction: str | None = None,
        canonical_json: bool = False,
        request_options: Mapping[str, Any] | None = None,
    ) -> tuple[Any, ModelCallRecord]:
        model_payload, payload_meta = _prepare_payload(
            payload, max_chars=self.prompt_char_budget
        )
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": json.dumps(
                    model_payload, ensure_ascii=canonical_json, sort_keys=True,
                    **({"separators": (",", ":")} if canonical_json else {}),
                ),
            },
        ]
        raw = ""
        metadata: dict[str, Any] = {}
        parsed: Any = None
        error: str | None = None
        attempts = 1
        partial_json_recovered = False
        generation_attempts: list[dict[str, Any]] = []
        can_retry = callable(getattr(self.client, "invoke_json", None))
        max_attempts = 3 if can_retry and final_retry_instruction else 2 if can_retry else 1
        for attempt in range(max_attempts):
            attempts = attempt + 1
            raw, metadata = (self._invoke(messages, request_options=request_options)
                             if request_options else self._invoke(messages))
            parsed = None
            error = None
            try:
                parsed = parse_json_object(raw)
                if validator is not None:
                    parsed = validator(parsed)
            except (TypeError, ValueError, KeyError) as exc:
                error = str(exc)
            stopped_for_length = str(metadata.get("stop_reason") or "").lower() in {
                "length",
                "max_tokens",
                "max_output_tokens",
            }
            if (
                error is not None
                and allow_truncated_containers
                and (attempt + 1 == max_attempts or not can_retry)
            ):
                try:
                    parsed = parse_truncated_container_object(raw)
                    if validator is not None:
                        parsed = validator(parsed)
                    if isinstance(parsed, dict):
                        normalizations = parsed.setdefault(
                            "_schema_normalizations", []
                        )
                        if "truncated_json_container_recovery" not in normalizations:
                            normalizations.append("truncated_json_container_recovery")
                    partial_json_recovered = True
                    error = None
                except (TypeError, ValueError, KeyError):
                    pass
            generation_attempts.append({
                "messages": [dict(message) for message in messages],
                "raw_response": raw,
                "parsed_response": parsed,
                "validation_error": error,
                "response_metadata": dict(metadata),
            })
            if error is None:
                break
            if attempt + 1 < max_attempts:
                retry_instruction = (
                    f"Your previous response failed the strict schema validator: "
                    f"{error}. Return a corrected JSON object only. Do not explain the correction."
                )
                if str(metadata.get("stop_reason") or "").lower() in {
                    "length",
                    "max_tokens",
                    "max_output_tokens",
                }:
                    retry_instruction = (
                        "Your previous JSON response hit the output-token limit and was truncated. "
                        "Return the smallest valid JSON object for the requested schema. Use one "
                        "concise item when the schema permits a list, omit all narrative detail, "
                        "and do not copy any input text."
                    )
                if attempt >= 1 and final_retry_instruction:
                    retry_instruction = final_retry_instruction
                messages = [
                    {
                        "role": "system",
                        "content": f"{system}\n{retry_instruction}",
                    },
                    messages[1],
                ]
                continue
            break
        metadata = {
            **metadata,
            "validation_attempts": attempts,
            "partial_json_recovered": partial_json_recovered,
            "generation_attempts": generation_attempts,
            **payload_meta,
        }
        record = ModelCallRecord(
            stage=stage,
            messages=tuple(messages),
            raw_response=raw,
            parsed_response=parsed,
            validation_error=error,
            response_metadata=metadata,
        )
        if error is not None:
            raise StructuredGenerationError(error, record)
        return parsed, record

    def _invoke(self, messages: list[dict[str, Any]], *, request_options: Mapping[str, Any] | None = None) -> tuple[str, dict[str, Any]]:
        invoke_json = getattr(self.client, "invoke_json", None)
        if callable(invoke_json):
            response = (invoke_json(messages, request_options=dict(request_options))
                        if request_options else invoke_json(messages))
            text = getattr(response, "text", None)
            if not isinstance(text, str):
                raise TypeError("model invoke_json() response must expose string .text")
            to_record = getattr(response, "to_record", None)
            metadata = to_record() if callable(to_record) else {}
            return text, metadata
        invoke = getattr(self.client, "invoke", None)
        if callable(invoke):
            response = invoke(messages)
            text = getattr(response, "text", None)
            if not isinstance(text, str):
                raise TypeError("model invoke() response must expose string .text")
            to_record = getattr(response, "to_record", None)
            metadata = to_record() if callable(to_record) else {}
            return text, metadata
        return self.client.complete(messages), {}


class StructuredGenerationError(ValueError):
    def __init__(self, message: str, record: ModelCallRecord) -> None:
        super().__init__(message)
        self.record = record


def _prepare_payload(
    payload: Mapping[str, Any], *, max_chars: int | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep model requests below the endpoint context window.

    The loaded :class:`BaselineSample` remains complete.  Only the serialized
    model view is compacted when a long native trajectory would exceed the
    serving budget.  Event IDs, roles, kinds, and head/tail content are kept so
    attribution and repair can still ground their response in the trajectory.
    """

    value = dict(payload)
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if max_chars is None or max_chars < 1 or len(encoded) <= max_chars:
        return value, {
            "prompt_compacted": False,
            "prompt_original_chars": len(encoded),
            "prompt_chars": len(encoded),
        }

    # The problem itself is not expendable trajectory context. Reserve room for
    # it before compacting events; otherwise long traces erase the actual task.
    task = value.get("task")
    constraints = value.get("constraints")
    nested_task = constraints.get("task_text") if isinstance(constraints, Mapping) else None
    if not isinstance(task, str) and isinstance(nested_task, str):
        # ARIA uses constraints.task_text rather than a top-level task. Apply
        # the same protection, then restore its canonical input layout.
        lifted = {**value, "constraints": {k: v for k, v in constraints.items() if k != "task_text"},
                  "task": nested_task}
        compacted, meta = _prepare_payload(lifted, max_chars=max_chars)
        compacted.setdefault("constraints", {})["task_text"] = compacted.pop("task")
        meta["prompt_original_chars"] = len(encoded)
        meta["prompt_chars"] = len(json.dumps(compacted, ensure_ascii=False, sort_keys=True))
        return compacted, meta
    protected = {"task": task} if isinstance(task, str) else {}
    remaining = {key: item for key, item in value.items() if key not in protected}
    reserve = len(json.dumps(protected, ensure_ascii=False, sort_keys=True))
    if protected and reserve + 256 < max_chars:
        compacted, _ = _prepare_payload(remaining, max_chars=max_chars - reserve - 4)
        compacted = {**compacted, **protected}
        encoded = json.dumps(compacted, ensure_ascii=False, sort_keys=True)
        return compacted, {
            "prompt_compacted": True,
            "prompt_original_chars": len(json.dumps(value, ensure_ascii=False, sort_keys=True)),
            "prompt_chars": len(encoded),
            "task_preserved": True,
        }

    compacted = _compact_value(value, max_chars=max_chars)
    encoded = json.dumps(compacted, ensure_ascii=False, sort_keys=True)
    for factor in (0.75, 0.5, 0.25):
        if len(encoded) <= max_chars:
            break
        compacted = _compact_value(
            value, max_chars=max(256, int(max_chars * factor)), aggressive=True
        )
        encoded = json.dumps(compacted, ensure_ascii=False, sort_keys=True)
    if len(encoded) > max_chars:
        # Keep the schema fields but bound every remaining string. This is a
        # last-resort guard for unusually large task text or feedback fields.
        compacted = _compact_value(
            compacted, max_chars=max(256, max_chars // 2), aggressive=True
        )
        encoded = json.dumps(compacted, ensure_ascii=False, sort_keys=True)
    if len(encoded) > max_chars:
        compacted = _fit_payload_to_char_budget(compacted, max_chars=max_chars)
        encoded = json.dumps(compacted, ensure_ascii=False, sort_keys=True)
    return compacted, {
        "prompt_compacted": True,
        "prompt_original_chars": len(
            json.dumps(value, ensure_ascii=False, sort_keys=True)
        ),
        "prompt_chars": len(encoded),
    }


def _compact_value(
    value: Any, *, max_chars: int, aggressive: bool = False, key: str = ""
) -> Any:
    if isinstance(value, Mapping):
        result = {
            str(name): _compact_value(
                item, max_chars=max_chars, aggressive=aggressive, key=str(name)
            )
            for name, item in value.items()
        }
        return result
    if isinstance(value, list):
        if key in {
            "events",
            "prefix",
            "trial_events",
            "visible_prefix",
            "current_event",
        } and value and all(isinstance(item, Mapping) for item in value):
            return _compact_event_list(value, max_chars=max_chars, aggressive=aggressive)
        return [
            _compact_value(
                item, max_chars=max_chars, aggressive=aggressive, key=key
            )
            for item in value
        ]
    if isinstance(value, str):
        limit = 160 if aggressive else 480
        if len(value) <= limit:
            return value
        head = max(1, limit // 2)
        tail = max(1, limit - head)
        return value[:head] + "\n...[prompt content compacted]...\n" + value[-tail:]
    return value


def _compact_event_list(
    values: list[Mapping[str, Any]], *, max_chars: int, aggressive: bool
) -> list[dict[str, Any]]:
    # Keep the prompt bound hard. The previous implementation shortened event
    # text but left nested metadata/tool payloads large enough to exceed vLLM's
    # context window.
    limits = (96, 48, 24, 0) if aggressive else (2048, 1024, 512, 256, 192, 96, 48, 24, 0)
    for content_limit in limits:
        compacted = [
            _compact_event(item, content_limit=content_limit) for item in values
        ]
        if len(json.dumps(compacted, ensure_ascii=False, sort_keys=True)) <= max_chars:
            return compacted

    # Preserve stable boundary IDs when even the full event list cannot fit.
    # The complete events remain in BaselineSample and are used for replay; only
    # this model-facing view is lossy.
    minimal = [_compact_event(item, content_limit=0) for item in values]
    if len(minimal) > 2:
        for count in range(min(len(minimal), 32), 1, -1):
            head = max(1, count // 2)
            selected = minimal[:head] + minimal[-(count - head) :]
            if len(json.dumps(selected, ensure_ascii=False, sort_keys=True)) <= max_chars:
                return selected
    return minimal[:2]


def _fit_payload_to_char_budget(value: Any, *, max_chars: int) -> Any:
    """Bound arbitrary nested payloads, including non-event lists."""

    for list_limit in (4, 2, 1, 0):
        for string_limit in (96, 48, 24, 8, 0):
            candidate = _budget_value(
                value,
                list_limit=list_limit,
                string_limit=string_limit,
            )
            if len(json.dumps(candidate, ensure_ascii=False, sort_keys=True)) <= max_chars:
                return candidate
    # Prompt keys are useful for the model, but oversized values are not. The
    # final fallback preserves the top-level object shape with empty values.
    if isinstance(value, Mapping):
        return {str(key): None for key in value}
    return None


def _budget_value(
    value: Any, *, list_limit: int, string_limit: int, key: str = ""
) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _budget_value(
                item,
                list_limit=list_limit,
                string_limit=string_limit,
                key=str(key),
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        if list_limit <= 0:
            return []
        selected = value[:list_limit]
        return [
            _budget_value(
                item,
                list_limit=list_limit,
                string_limit=string_limit,
                key="",
            )
            for item in selected
        ]
    if isinstance(value, str):
        if key == "id" or key.endswith("_id"):
            return value
        if len(value) <= string_limit:
            return value
        if string_limit <= 0:
            return ""
        return value[:string_limit]
    return value


def _compact_event(item: Mapping[str, Any], *, content_limit: int) -> dict[str, Any]:
    content = str(item.get("content") or item.get("text") or item.get("message") or "")
    if content_limit == 0:
        content = ""
    elif len(content) > content_limit:
        head = max(1, content_limit // 2)
        tail = max(1, content_limit - head)
        content = content[:head] + " ...[compacted]... " + content[-tail:]

    compact_calls: list[dict[str, Any]] = []
    tool_calls = item.get("tool_calls") or []
    if isinstance(tool_calls, list):
        for call in tool_calls[:4]:
            if not isinstance(call, Mapping):
                continue
            function = call.get("function")
            function = function if isinstance(function, Mapping) else {}
            arguments = call.get("arguments") or function.get("arguments") or ""
            encoded = arguments if isinstance(arguments, str) else json.dumps(arguments)
            compact_call = {"name": call.get("name") or function.get("name")}
            if len(encoded) <= max(160, content_limit):
                compact_call["arguments"] = arguments
            else:
                compact_call["arguments_omitted"] = True
            compact_calls.append(compact_call)
    return {
        "event_id": item.get("event_id") or item.get("id"),
        "role": item.get("role") or item.get("source") or item.get("name"),
        "kind": item.get("kind") or item.get("type") or "message",
        "content": content,
        "tool_calls": compact_calls,
    }


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    candidates = [stripped]
    if "```" in stripped:
        blocks = stripped.split("```")
        candidates.extend(block.removeprefix("json").strip() for block in blocks[1::2])
    left = stripped.find("{")
    right = stripped.rfind("}")
    if left >= 0 and right > left:
        candidates.append(stripped[left : right + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("model response is not a JSON object")


def parse_truncated_container_object(text: str) -> dict[str, Any]:
    """Close only incomplete object/array containers, never partial atoms."""

    from partial_json_parser import Allow, loads

    parsed = loads(text, Allow.OBJ | Allow.ARR)
    if not isinstance(parsed, dict):
        raise ValueError("truncated model response is not a JSON object")
    return parsed


def require_keys(raw: Any, keys: set[str], *, name: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError(f"{name} must be an object")
    missing = sorted(keys - set(raw))
    if missing:
        raise KeyError(f"{name} missing keys: {', '.join(missing)}")
    return raw
