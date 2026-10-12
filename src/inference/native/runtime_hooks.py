"""Small message-boundary hooks for native runtimes.

The native Magentic-One and AG2 integrations execute in child processes.  A
parent-owned :class:`aria.runtime.AriaRuntime` therefore cannot intercept a
child model call directly.  This module serializes the runtime bootstrap state
and installs a deliberately small pre/post wrapper around the native model
client.  It supports prompt injection and audit logging at the pending message
boundary; it does not claim arbitrary host rollback.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Mapping, Sequence
from functools import wraps
from typing import Any, Callable

from aria.runtime import AriaConfig, AriaRuntime, RuntimeMemory
from aria.model import AriaModelConfig
from inference.native.model_hooks import intercept_native_response

__all__ = [
    "export_runtime_bootstrap",
    "install_async_client_hook",
    "install_sync_client_hook",
    "restore_runtime",
    "runtime_from_bootstrap",
    "runtime_profile_events",
]


def export_runtime_bootstrap(aria: AriaRuntime | None) -> dict[str, Any] | None:
    """Serialize the parent runtime's config and bootstrap memory."""

    if aria is None:
        return None
    return {
        "config": dataclasses.asdict(aria.config),
        "state": {
            "memory": aria.memory.snapshot(),
            "failure_reports": list(aria.failure_reports),
            "patch_history": list(aria.patch_history),
            "model_interventions": list(aria.model_interventions),
        },
    }


def runtime_from_bootstrap(value: Mapping[str, Any] | None) -> AriaRuntime | None:
    """Recreate a child runtime from :func:`export_runtime_bootstrap`."""

    if not isinstance(value, Mapping):
        return None
    raw_config = value.get("config")
    if not isinstance(raw_config, Mapping):
        return None
    config_values = dict(raw_config)
    if isinstance(config_values.get("model"), Mapping):
        config_values["model"] = AriaModelConfig(**config_values["model"])
    for key in ("verification_keywords", "finalization_keywords"):
        if key in config_values:
            config_values[key] = tuple(config_values[key] or ())
    aria = AriaRuntime(AriaConfig(**config_values))
    state = value.get("state")
    if not isinstance(state, Mapping):
        return aria
    memory = state.get("memory")
    if isinstance(memory, Mapping):
        for field in dataclasses.fields(RuntimeMemory):
            stored = memory.get(field.name)
            if isinstance(stored, Mapping):
                setattr(aria.memory, field.name, copy.deepcopy(dict(stored)))
    for name in ("failure_reports", "patch_history", "model_interventions"):
        stored = state.get(name)
        if isinstance(stored, list):
            setattr(aria, name, copy.deepcopy(stored))
    return aria


def restore_runtime(target: AriaRuntime, source: AriaRuntime | None) -> None:
    """Copy child state into a parent runtime when both live in one process."""

    if source is None:
        return
    target.memory = source.memory
    target.events = source.events
    target.failure_reports = source.failure_reports
    target.patch_history = source.patch_history
    target.pending_patch = source.pending_patch
    target.model_interventions = source.model_interventions


def runtime_profile_events(profile: Mapping[str, Any] | None) -> tuple[str, str, str]:
    """Return pre/post/finalize event names from a serialized profile."""

    value = profile if isinstance(profile, Mapping) else {}
    return (
        str(value.get("pre_event") or "before_native_model_call"),
        str(value.get("post_event") or "after_native_model_call"),
        str(value.get("finalize_event") or "before_native_finalize"),
    )


def install_async_client_hook(
    client: Any,
    aria: AriaRuntime,
    profile: Mapping[str, Any] | None,
    trace: list[dict[str, Any]],
    *,
    role: str,
) -> Any:
    """Wrap an async ``client.create`` call at a native message boundary."""

    original = client.create
    pre_event, post_event, _ = runtime_profile_events(profile)
    counter = {"turn": 0}

    @wraps(original)
    async def create_with_runtime(messages: Any, *args: Any, **kwargs: Any) -> Any:
        counter["turn"] += 1
        turn = counter["turn"]
        prompt = message_text(messages)
        patched_prompt, patch_record = aria.before_action(
            pre_event, turn, role, prompt
        )
        prepared = replace_last_message(messages, patched_prompt) if patched_prompt != prompt else messages
        trace.append(
            {
                "kind": "aria_before_model_call",
                "turn": turn,
                "role": role,
                "hook": pre_event,
                "prompt": patched_prompt,
                "patch": patch_record,
            }
        )
        try:
            response = await original(prepared, *args, **kwargs)
        except Exception as exc:
            trace.append(
                {
                    "kind": "aria_model_call_error",
                    "turn": turn,
                    "role": role,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            raise
        extra = kwargs.get("extra_create_args") or {}
        response = intercept_native_response(
            aria, response, prepared, role=role,
            protected=bool(kwargs.get("json_output") or extra.get("response_format")),
        )
        content = response_text(response)
        detection = aria.after_action(post_event, turn, role, content)
        trace.append(
            {
                "kind": "aria_after_model_call",
                "turn": turn,
                "role": role,
                "hook": post_event,
                "content": content,
                "detection": detection,
            }
        )
        return response

    client.create = create_with_runtime
    return original


def install_sync_client_hook(
    client_class: Any,
    aria: AriaRuntime,
    profile: Mapping[str, Any] | None,
    trace: list[dict[str, Any]],
    *,
    role: str,
) -> Callable[[], None]:
    """Wrap a sync OpenAI-compatible ``create`` class method.

    The returned callback restores the original method.  Native AG2 calls are
    sequential, but the turn counter remains local to the wrapper so the
    audit stream is deterministic.
    """

    original = client_class.create
    pre_event, post_event, _ = runtime_profile_events(profile)
    counter = {"turn": 0}

    @wraps(original)
    def create_with_runtime(client: Any, params: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        counter["turn"] += 1
        turn = counter["turn"]
        prompt = message_text(params.get("messages"))
        patched_prompt, patch_record = aria.before_action(
            pre_event, turn, role, prompt
        )
        prepared = dict(params)
        if patched_prompt != prompt:
            prepared["messages"] = replace_last_message(
                params.get("messages") or [], patched_prompt
            )
        trace.append(
            {
                "kind": "aria_before_model_call",
                "turn": turn,
                "role": role,
                "hook": pre_event,
                "prompt": patched_prompt,
                "patch": patch_record,
            }
        )
        try:
            response = original(client, prepared, *args, **kwargs)
        except Exception as exc:
            trace.append(
                {
                    "kind": "aria_model_call_error",
                    "turn": turn,
                    "role": role,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            raise
        response = intercept_native_response(
            aria, response, prepared.get("messages", []), role=role,
            protected=bool(prepared.get("response_format")),
        )
        content = response_text(response)
        detection = aria.after_action(post_event, turn, role, content)
        trace.append(
            {
                "kind": "aria_after_model_call",
                "turn": turn,
                "role": role,
                "hook": post_event,
                "content": content,
                "detection": detection,
            }
        )
        return response

    client_class.create = create_with_runtime

    def restore() -> None:
        client_class.create = original

    return restore


def message_text(messages: Any) -> str:
    """Extract a compact textual prompt from dict or AutoGen messages."""

    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return str(messages or "")
    for item in reversed(messages):
        content = getattr(item, "content", None)
        if content is None and isinstance(item, Mapping):
            content = item.get("content")
        text = _content_text(content)
        if text:
            return text
    return ""


def response_text(response: Any) -> str:
    """Extract response text without making framework response assumptions."""

    choices = getattr(response, "choices", None)
    if isinstance(choices, Sequence) and choices:
        message = getattr(choices[0], "message", None)
        if message is None and isinstance(choices[0], Mapping):
            message = choices[0].get("message")
        content = getattr(message, "content", None)
        if content is None and isinstance(message, Mapping):
            content = message.get("content")
        text = _content_text(content)
        if text:
            return text
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls is None and isinstance(message, Mapping):
            tool_calls = message.get("tool_calls")
        return f"tool_calls={len(tool_calls or [])}"
    return str(getattr(response, "content", "") or "")


def replace_last_message(messages: Any, content: str) -> Any:
    """Replace the last textual message while preserving its container type."""

    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        return messages
    values = list(messages)
    for index in range(len(values) - 1, -1, -1):
        item = values[index]
        current = getattr(item, "content", None)
        if current is None and isinstance(item, Mapping):
            current = item.get("content")
        if not _content_text(current):
            continue
        values[index] = _copy_with_content(item, content)
        return tuple(values) if isinstance(messages, tuple) else values
    return messages


def _copy_with_content(item: Any, content: str) -> Any:
    if isinstance(item, Mapping):
        value = dict(item)
        value["content"] = content
        return value
    model_copy = getattr(item, "model_copy", None)
    if callable(model_copy):
        try:
            return model_copy(update={"content": content})
        except Exception:  # noqa: BLE001 - try the framework-neutral fallbacks
            pass
    if dataclasses.is_dataclass(item):
        try:
            return dataclasses.replace(item, content=content)
        except (TypeError, ValueError):
            pass
    copied = copy.copy(item)
    try:
        setattr(copied, "content", content)
        return copied
    except Exception:  # noqa: BLE001 - immutable framework messages
        return item


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return str(value.get("text") or value.get("content") or "")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        parts = [_content_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    return str(value or "")
