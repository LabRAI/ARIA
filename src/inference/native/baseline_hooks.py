"""Existing runtime baselines at the same native text boundary as ARIA."""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from baselines.core.models import ModelAdapter, StructuredGenerationError
from baselines.core.types import BaselineEvent, BaselineSample, ModelCallRecord
from baselines.registry import build_method
from common.model_api import ApiConfig, ModelApiClient


def _reduce_runtime_payload(value, cap):
    from aria.prompt_budget import bounded_text

    def compact(item, key=""):
        if isinstance(item, dict):
            return {name: compact(part, name) for name, part in item.items()}
        if isinstance(item, list):
            return [compact(part, key) for part in item]
        if isinstance(item, str) and not (key.endswith("_id") or key in {"id", "role", "kind", "action", "status", "label"}):
            return bounded_text(item, max(cap, 512) if key in {"task", "task_text"} else cap)
        return item

    return compact(value)


class AuditedRuntimeModel(ModelAdapter):
    """Preserve completed calls even when a later stage of the policy fails."""

    def __init__(self, client: Any, *, tokenizer=None, max_input_tokens=None) -> None:
        super().__init__(client)
        self.records: list[ModelCallRecord] = []
        self.inflight_messages: list[dict[str, Any]] = []
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens

    def _invoke(self, messages, **kwargs):
        budget = None
        if self.tokenizer is not None:
            from aria.prompt_budget import fit_messages

            serialized, budget = fit_messages(messages,
                render=lambda value: json.dumps(value, ensure_ascii=True, separators=(",", ":")),
                tokenizer=self.tokenizer, max_tokens=self.max_input_tokens, reduce_payload=_reduce_runtime_payload)
            # ModelAdapter records this same list, so the audit describes the actual request.
            messages[:] = json.loads(serialized)
        self.inflight_messages = copy.deepcopy(messages)
        text, metadata = super()._invoke(messages, **kwargs)
        return text, {**metadata, **({"context_budget": budget} if budget is not None else {})}

    def json_call(self, **kwargs):
        try:
            value, record = super().json_call(**kwargs)
        except StructuredGenerationError as exc:
            self.records.append(exc.record)
            raise
        except Exception as exc:
            self.records.append(ModelCallRecord(
                stage=kwargs["stage"], messages=tuple(self.inflight_messages),
                raw_response="", parsed_response=None, validation_error=str(exc),
                response_metadata={"exception_type": type(exc).__name__},
            ))
            raise
        self.records.append(record)
        return value, record


def _load_policy(aria: Any) -> tuple[Any, AuditedRuntimeModel, dict[str, Any]]:
    cached = getattr(aria, "_native_baseline_policy", None)
    if cached is not None:
        return cached
    config = aria.config.model
    if config.provider != "openai-chat-completions":
        raise ValueError("native runtime baselines require an OpenAI-compatible endpoint")
    client = ModelApiClient(ApiConfig(
        provider="openai-compatible", protocol="openai-chat-completions",
        base_url=config.base_url, model=config.model, api_key_env=config.api_key_env,
        api_key_required=config.api_key_required, timeout_seconds=config.timeout_seconds,
        max_tokens=config.max_tokens, temperature=config.temperature, seed=config.seed,
        retries=0, retry_seconds=0.0,
    ))
    tokenizer = None
    if getattr(config, "max_input_tokens", None) is not None:
        from aria.model import OpenAIChatAriaModel, _EndpointChatTokenizer

        tokenizer = _EndpointChatTokenizer(OpenAIChatAriaModel(config))
    cached = (build_method(aria.config.runtime_policy, **(aria.config.runtime_policy_options or {})),
              AuditedRuntimeModel(client, tokenizer=tokenizer, max_input_tokens=getattr(config, "max_input_tokens", None)), {})
    aria._native_baseline_policy = cached
    return cached


def intercept_baseline_text(aria: Any, content: str, messages: Any, *, role: str,
                            protected: bool, pending_tool_calls: Any = None) -> str:
    from inference.native.model_hooks import _event, _persist_audit

    contract = aria.memory.contract_memory
    boundary = len(aria.model_interventions) + 1
    incident_id = str(contract.get("incident_id") or "native-live")
    context = messages if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)) else []
    events = []
    for index, value in enumerate([*context, {"role": role, "content": content,
                                            "tool_calls": pending_tool_calls or []}], start=1):
        event = replace(_event(value, index, role), event_id=f"live:boundary:{boundary}:event:{index}")
        events.append(BaselineEvent(event_id=event.event_id, role=event.role, kind=event.kind,
                                    content=event.content, tool_calls=event.tool_calls, metadata=event.metadata))
    sample = BaselineSample(
        sample_id=incident_id, task_id=incident_id, task=str(contract.get("task") or ""),
        events=tuple(events), benchmark=contract.get("benchmark"), mas=contract.get("mas"),
        constraints={"answer_blind": True, "task_text": str(contract.get("task") or ""),
                     "benchmark": str(contract.get("benchmark") or ""), "mas": str(contract.get("mas") or "")},
        feedback=None,
    )
    audit: dict[str, Any] = {
        "incident_id": incident_id, "boundary": boundary, "role": role,
        "runtime_policy": aria.config.runtime_policy,
        "execution_mode": "live_pre_delivery_text_substitution",
        "context_scope": "actual_native_model_request",
        "observable_events": [event.to_dict() for event in events],
        "original_content": content, "delivered_content": content, "enacted": False,
        "intervention": {"action": "deliver", "reason": "policy_allow"},
    }
    model = None
    first_call = 0
    try:
        method, model, state = _load_policy(aria)
        first_call = len(model.records)
        result = method.run(sample, model, start_index=len(events) - 1, runtime_state=state)
        proposal = result.intervention
        audit["policy_diagnostics"] = result.diagnostics
        audit["baseline_proposal"] = proposal
        if proposal is not None:
            candidate = proposal.get("sanitized_content") if aria.config.runtime_policy == "spark_to_fire" else proposal.get("modified_content")
            try:
                structured = isinstance(json.loads(content), (dict, list))
            except (TypeError, ValueError):
                structured = False
            reason = None
            if aria.config.mode == "shadow":
                reason = "shadow_policy_triggered"
            elif protected or structured:
                reason = "degraded:protected_native_response"
            elif proposal.get("target_event_id") != events[-1].event_id:
                reason = "degraded:no_rollback"
            elif proposal.get("action") not in {"rewrite", "block_and_rollback"}:
                reason = f"degraded:unsupported_native_action:{proposal.get('action')}"
            elif not isinstance(candidate, str) or not candidate.strip():
                reason = "degraded:no_text_replacement"
            elif candidate == content:
                reason = "degraded:no_content_change"
            if reason:
                audit["intervention"] = {"action": "shadow", "reason": reason}
            else:
                audit["intervention"] = {"action": "substitute", "reason": "policy_text_substitution"}
                audit["delivered_content"] = candidate
                audit["enacted"] = True
                content = candidate
            if aria.config.runtime_policy == "safeagent" and state.get("prior_interventions"):
                state["prior_interventions"][-1].update(
                    host_enacted=audit["enacted"], host_action=audit["intervention"]["action"],
                )
    except Exception as exc:
        audit["error"] = f"{type(exc).__name__}: {exc}"
        audit["intervention"] = {"action": "model_error", "reason": type(exc).__name__}
        if not aria.config.fail_open:
            raise
    finally:
        audit["model_calls"] = [call.to_dict() for call in model.records[first_call:]] if model is not None else []
        aria.model_interventions.append(audit)
        _persist_audit(aria)
    return content
