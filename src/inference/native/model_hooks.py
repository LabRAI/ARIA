"""Post-trained ARIA at a blocking, pre-delivery native response boundary.

Only one plain-text pending message can be substituted. Tools, structured
protocol responses, multi-event patches and historical rollback remain shadow
proposals until a host implements those operations explicitly.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from collections.abc import Mapping, Sequence
from typing import Any
from pathlib import Path

from aria.deployment import load_active_repair_controller
from aria.hook import ActiveAction
from aria.hook.controller import validate_pending_substitution
from aria.hook.profiles import SUBSTITUTE_ONLY_CAPABILITIES
from aria.records import TrajectoryEvent

def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _jsonable(dump())
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    return str(value)


def _event(message: Any, index: int, role: str) -> TrajectoryEvent:
    content = _get(message, "content", "")
    calls = _jsonable(_get(message, "tool_calls", []) or [])
    return TrajectoryEvent(
        event_id=f"source:event:{index:04d}",
        role=str(_get(message, "source") or _get(message, "role") or role),
        kind="tool_call" if calls else "tool_result" if _get(message, "role") == "tool" else "message",
        content=content if isinstance(content, str) else json.dumps(_jsonable(content), ensure_ascii=True),
        tool_calls=tuple(calls),
        metadata={},
    )


def intercept_native_text(aria: Any, content: str, messages: Any, *, role: str,
                          protected: bool = False, pending_tool_calls: Any = None) -> str:
    """Inspect actual host context plus an undelivered text response."""

    if aria is None or not aria.config.enabled or aria.config.model is None or aria.config.mode not in {"active", "shadow"}:
        return content
    if any(item.get("enacted") for item in aria.model_interventions):
        return content
    if aria.config.runtime_policy != "aria":
        from inference.native.baseline_hooks import intercept_baseline_text

        return intercept_baseline_text(aria, content, messages, role=role, protected=protected,
                                       pending_tool_calls=pending_tool_calls)
    contract = aria.memory.contract_memory
    index = len(aria.model_interventions) + 1
    incident_id = str(contract.get("incident_id") or "native-live")
    audit: dict[str, Any] = {
        "incident_id": incident_id, "boundary": index, "role": role,
        "execution_mode": "live_pre_delivery_text_substitution",
        "context_scope": "actual_native_model_request",
        "original_content": content, "delivered_content": content, "enacted": False,
    }
    controller = None
    call_start = 0
    try:
        controller = getattr(aria, "_model_controller", None)
        if controller is None:
            controller = load_active_repair_controller(
                aria.config.model, gate_threshold=aria.config.gate_threshold,
                shadow=aria.config.mode == "shadow", fail_open=aria.config.fail_open,
            )
            aria._model_controller = controller
        controller.begin(incident_id, constraints={
            "task_text": str(contract.get("task") or ""),
            "benchmark": str(contract.get("benchmark") or ""),
            "mas": str(contract.get("mas") or ""), "answer_blind": True,
        }, feedback=None)
        context = messages if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)) else []
        for number, message in enumerate(context, start=1):
            controller.observe(_event(message, number, role))
        pending = _event({"content": content, "role": role,
                          "tool_calls": pending_tool_calls or []}, len(context) + 1, role)
        audit["observable_events"] = [event.to_dict() for event in (*controller.events, pending)]
        calls = getattr(controller.model, "calls", [])
        call_start = len(calls)
        decision = controller.intercept(pending, capabilities=SUBSTITUTE_ONLY_CAPABILITIES)
        audit["model_calls"] = copy.deepcopy(calls[call_start:])
        decision = validate_pending_substitution(decision, content, role=role, protected=protected)
        if decision.action == ActiveAction.SUBSTITUTE:
            replacement = decision.patch_events[0]
            content = replacement.content
            audit["enacted"] = True
            controller.record_enacted(decision)
        audit["intervention"] = decision.to_dict()
        if decision.error is not None:
            audit["error"] = decision.error
        audit["delivered_content"] = content
    except Exception as exc:  # noqa: BLE001 - explicit host fail-open policy
        audit["error"] = f"{type(exc).__name__}: {exc}"
        if not aria.config.fail_open:
            raise
    finally:
        if controller is not None:
            audit["model_calls"] = copy.deepcopy(getattr(controller.model, "calls", [])[call_start:])
        aria.model_interventions.append(audit)
        _persist_audit(aria)
    return content


def _persist_audit(aria: Any) -> None:
    path = aria.memory.contract_memory.get("model_audit_path")
    if path:
        from common.artifacts import atomic_write_jsonl

        atomic_write_jsonl(Path(path), aria.model_interventions)


def intercept_native_response(aria: Any, response: Any, messages: Any, *, role: str,
                              protected: bool = False) -> Any:
    """Preserve native response types and all protocol metadata on substitution."""

    if aria is None or aria.config.model is None:
        return response
    choices = _get(response, "choices")
    message = _get(choices[0], "message") if choices else response
    content = _get(message, "content")
    if not isinstance(content, str):
        calls = _jsonable(_get(message, "tool_calls") or [])
        legacy = _jsonable(_get(message, "function_call"))
        if calls or legacy:
            # A tool-only response is still a real pre-delivery boundary. Observe
            # its exact payload, but never substitute text for executable calls.
            observation = json.dumps({"tool_calls": calls, **({"function_call": legacy} if legacy else {})},
                                     ensure_ascii=True, sort_keys=True)
            before = len(aria.model_interventions)
            intercept_native_text(aria, observation, messages, role=role, protected=True,
                                  pending_tool_calls=calls)
            if len(aria.model_interventions) > before:
                aria.model_interventions[-1].update(
                    response_boundary="protected_tool_only", host_original_content=_jsonable(content),
                    host_tool_calls=calls, host_function_call=legacy, host_response_unchanged=True)
                _persist_audit(aria)
        return response
    delivered = intercept_native_text(
        aria, content, messages, role=role,
        protected=bool(protected or _get(message, "tool_calls") or _get(message, "function_call")),
    )
    if delivered == content:
        return response
    try:
        copied = copy.deepcopy(response)
        target = _get(_get(copied, "choices")[0], "message") if choices else copied
        if isinstance(target, dict):
            target["content"] = delivered
        else:
            setattr(target, "content", delivered)
    except Exception as exc:  # noqa: BLE001 - unsupported immutable host response
        aria.model_interventions[-1].update(
            enacted=False, delivered_content=content, error=f"response substitution failed: {exc}",
        )
        _persist_audit(aria)
        if not aria.config.fail_open:
            raise
        return response
    return copied
