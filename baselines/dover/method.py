"""DoVer trial segmentation, hypothesis generation, intervention, and replay."""

from __future__ import annotations

from typing import Any, ClassVar

from baselines.core.method import BaselineMethod
from baselines.core.models import ModelAdapter, require_keys
from baselines.core.replay import ReplayBackend, execute
from baselines.core.types import BaselineSample

SEGMENT_SYSTEM = """Implement DoVer trial segmentation. A trial starts with initial planning or a
major re-plan and includes its execution steps. Small execution-detail changes are not re-plans.
Return JSON only with initial_planning_step (integer) and plan_update_steps (list of integers)."""
HYPOTHESIS_SYSTEM = """Implement DoVer failure-attribution hypothesis generation for one trial.
Propose the step or agent whose targeted change may alter the outcome. Return JSON only with
responsible_agent, failure_step_index, hypothesis, and intervention_scope (orchestrator_plan or
subagent_capability). Return only these four fields, keep the hypothesis under 40 words, and never
copy task text or trial_events into the response."""
INTERVENTION_SYSTEM = """Implement DoVer intervention generation. Convert the hypothesis into one
testable edit to a plan, instruction, message, or step. Return JSON only with type, target_event_id,
content, reason, and role. The edit will be replayed from a real checkpoint; do not claim success."""


class DoVer(BaselineMethod):
    name = "dover"
    setting = "passive"
    provenance: ClassVar[dict[str, Any]] = {
        "paper": "DoVer: Intervention-Driven Auto Debugging for LLM Multi-Agent Systems (ICLR 2026)",
        "official_repository": "https://github.com/microsoft/ACV/tree/main/misc/DoVer",
        "official_commit": "994ac15db6fff8a5131bbf5a26e84e352e676796",
        "adaptation": "host replay backend supplies framework checkpoint/replay hooks",
    }

    def run(
        self,
        sample: BaselineSample,
        model: ModelAdapter | None,
        replay: ReplayBackend | None = None,
    ):
        selected = self.require_model(model)
        result = self.result(sample, status="proposed")
        segmentation, segment_call = selected.json_call(
            stage="dover.trial_segmentation",
            system=SEGMENT_SYSTEM,
            payload=sample.model_view(include_feedback=True),
            validator=lambda value: _validate_segmentation(value, sample),
        )
        result.model_calls.append(segment_call)
        trials = _trials(segmentation, len(sample.events))
        attempts: list[dict[str, Any]] = []
        ranked: list[tuple[float, bool, dict[str, Any], str | None]] = []
        for trial_index, (start, stop) in enumerate(trials):
            hypothesis, hypothesis_call = selected.json_call(
                stage=f"dover.hypothesis.{trial_index:02d}",
                system=HYPOTHESIS_SYSTEM,
                payload={
                    "task": sample.task,
                    "feedback": sample.feedback,
                    "trial_index": trial_index,
                    "trial_events": [
                        event.to_dict() for event in sample.events[start:stop]
                    ],
                },
                validator=lambda value, lo=start, hi=stop: _validate_hypothesis(
                    value, sample, lo, hi
                ),
            )
            result.model_calls.append(hypothesis_call)
            edit, intervention_call = selected.json_call(
                stage=f"dover.intervention.{trial_index:02d}",
                system=INTERVENTION_SYSTEM,
                payload={
                    "sample": sample.model_view(include_feedback=True),
                    "hypothesis": hypothesis,
                },
                validator=lambda value, h=hypothesis: _validate_edit(value, sample, h),
            )
            result.model_calls.append(intervention_call)
            intervention = _make_intervention(sample, hypothesis, edit)
            label = f"dover-trial-{trial_index:02d}"
            execution = execute(replay, sample, intervention, label=label)
            if execution is not None:
                result.executions.append(execution)
                score = _score(execution.evaluation)
                succeeded = execution.success is True
                ranked.append((score, succeeded, intervention, label))
            else:
                ranked.append((0.0, False, intervention, None))
            attempts.append(
                {
                    "trial_index": trial_index,
                    "range": [start, stop],
                    "hypothesis": hypothesis,
                    "intervention": intervention,
                    "execution": execution.to_dict() if execution is not None else None,
                }
            )
        result.diagnostics["segmentation"] = segmentation
        result.diagnostics["intervention_attempts"] = attempts
        if ranked:
            score, succeeded, intervention, label = max(
                ranked, key=lambda value: (value[1], value[0])
            )
            result.intervention = intervention
            target = intervention["target_event_id"]
            target_index = next(
                index
                for index, event in enumerate(sample.events)
                if event.event_id == target
            )
            result.attribution = {
                "event_id": target,
                "step_index": target_index,
                "validated": succeeded if replay is not None else None,
                "progress_score": score if replay is not None else None,
            }
            result.selected_execution_label = label
            if replay is not None:
                result.status = "executed"
        return result


def _validate_segmentation(raw: Any, sample: BaselineSample) -> dict[str, Any]:
    value = require_keys(
        raw, {"initial_planning_step", "plan_update_steps"}, name="segmentation"
    )
    indices = [value["initial_planning_step"], *value["plan_update_steps"]]
    if not isinstance(value["initial_planning_step"], int) or not isinstance(
        value["plan_update_steps"], list
    ):
        raise TypeError("DoVer segmentation uses integer planning indices")
    if any(not isinstance(index, int) or isinstance(index, bool) for index in indices):
        raise TypeError("DoVer planning indices must be integers")
    normalizations: list[str] = []
    if indices and all(index >= 1 for index in indices) and any(
        index == len(sample.events) for index in indices
    ):
        value["initial_planning_step"] -= 1
        value["plan_update_steps"] = [
            index - 1 for index in value["plan_update_steps"]
        ]
        indices = [value["initial_planning_step"], *value["plan_update_steps"]]
        normalizations.append("one_based_planning_indices->zero_based")
    filtered_updates = [
        index for index in value["plan_update_steps"] if 0 <= index < len(sample.events)
    ]
    if filtered_updates != value["plan_update_steps"]:
        value["plan_update_steps"] = filtered_updates
        normalizations.append("drop_out_of_range_plan_updates")
    if not 0 <= value["initial_planning_step"] < len(sample.events):
        raise ValueError("DoVer planning index is outside the trajectory")
    value["plan_update_steps"] = sorted(set(value["plan_update_steps"]))
    if normalizations:
        value["_schema_normalizations"] = normalizations
    return value


def _trials(segmentation: dict[str, Any], event_count: int) -> list[tuple[int, int]]:
    starts = sorted(
        {segmentation["initial_planning_step"], *segmentation["plan_update_steps"]}
    )
    return [
        (start, starts[index + 1] if index + 1 < len(starts) else event_count)
        for index, start in enumerate(starts)
    ]


def _validate_hypothesis(
    raw: Any, sample: BaselineSample, start: int, stop: int
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError("DoVer hypothesis must be an object")
    raw = dict(raw)
    normalizations: list[str] = []
    if not any(
        key in raw
        for key in (
            "failure_step_index",
            "step_index",
            "step",
            "index",
            "responsible_agent",
            "agent",
            "role",
            "hypothesis",
            "diagnosis",
            "reason",
            "intervention_scope",
        )
    ) and isinstance(raw.get("trial_events"), list):
        trial_events = [
            item for item in raw["trial_events"] if isinstance(item, dict)
        ]
        if trial_events:
            terminal = trial_events[-1]
            terminal_id = terminal.get("event_id")
            positions = {
                event.event_id: index for index, event in enumerate(sample.events)
            }
            terminal_index = positions.get(str(terminal_id))
            if terminal_index is None:
                terminal_index = stop - 1
            terminal_index = min(max(terminal_index, start), stop - 1)
            raw = {
                "responsible_agent": terminal.get("role") or "unknown",
                "failure_step_index": terminal_index,
                "hypothesis": "Test the terminal step because the trial has not produced a correct outcome.",
                "intervention_scope": "subagent_capability",
            }
            normalizations.append("echoed_trial_payload->terminal_trial_hypothesis")
    if "failure_step_index" not in raw:
        for alias in ("step_index", "step", "index"):
            if alias in raw:
                raw["failure_step_index"] = raw[alias]
                normalizations.append(f"{alias}->failure_step_index")
                break
    if "responsible_agent" not in raw:
        raw["responsible_agent"] = raw.get("agent") or raw.get("role") or "unknown"
        normalizations.append("default_responsible_agent")
    if "hypothesis" not in raw:
        raw["hypothesis"] = raw.get("reason") or raw.get("diagnosis") or "targeted intervention hypothesis"
        normalizations.append("reason->hypothesis")
    scope = str(raw.get("intervention_scope") or "").lower().replace("-", "_")
    scope_aliases = {
        "orchestrator": "orchestrator_plan",
        "plan": "orchestrator_plan",
        "orchestrator_instruction": "orchestrator_plan",
        "orchestrator_plan_edit": "orchestrator_plan",
        "subagent": "subagent_capability",
        "agent": "subagent_capability",
        "tool": "subagent_capability",
        "capability": "subagent_capability",
        "agent_capability": "subagent_capability",
        "executor_plan": "subagent_capability",
        "executor": "subagent_capability",
    }
    if scope in scope_aliases:
        raw["intervention_scope"] = scope_aliases[scope]
        normalizations.append(f"intervention_scope:{scope}->{raw['intervention_scope']}")
    value = require_keys(
        raw,
        {"responsible_agent", "failure_step_index", "hypothesis", "intervention_scope"},
        name="DoVer hypothesis",
    )
    index = value["failure_step_index"]
    if isinstance(index, int) and not isinstance(index, bool) and not start <= index < stop:
        if 0 <= index < stop - start:
            index += start
            value["failure_step_index"] = index
            normalizations.append("local_failure_step_index->global")
    if not isinstance(index, int) or isinstance(index, bool):
        raise ValueError("DoVer failure_step_index must lie within its trial")
    if not start <= index < stop:
        bounded = min(max(index, start), stop - 1)
        value["failure_step_index"] = bounded
        normalizations.append("failure_step_index->nearest_trial_boundary")
        index = bounded
    if value["intervention_scope"] not in {"orchestrator_plan", "subagent_capability"}:
        raise ValueError("DoVer intervention_scope is unsupported")
    value["event_id"] = sample.events[index].event_id
    if normalizations:
        value["_schema_normalizations"] = normalizations
    return value


def _validate_edit(
    raw: Any, sample: BaselineSample, hypothesis: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError("DoVer edit must be an object")
    raw = dict(raw)
    normalizations: list[str] = []
    nested_key = next(
        (
            key
            for key in ("edit", "intervention")
            if isinstance(raw.get(key), dict)
        ),
        None,
    )
    nested = raw.get(nested_key) if nested_key is not None else None
    if isinstance(nested, dict) and not any(
        key in raw for key in ("type", "target_event_id", "content", "reason", "role")
    ):
        raw = dict(nested)
        normalizations.append(f"nested_{nested_key}->edit")
    aliases = {
        "target_event_id": ("event_id", "target", "step_event_id"),
        "content": ("message", "instruction", "text", "modified_content"),
        "reason": ("diagnosis", "rationale", "hypothesis"),
        "role": ("agent", "responsible_agent"),
    }
    for target, candidates in aliases.items():
        if target in raw:
            continue
        for alias in candidates:
            if alias in raw:
                raw[target] = raw[alias]
                normalizations.append(f"{alias}->{target}")
                break
    if "target_event_id" not in raw and hypothesis.get("event_id") in {
        event.event_id for event in sample.events
    }:
        raw["target_event_id"] = hypothesis["event_id"]
        normalizations.append("hypothesis.event_id->target_event_id")
    kind = str(raw.get("type") or "message_edit").lower().replace("-", "_")
    kind_aliases = {
        "edit": "message_edit",
        "testable_edit": "message_edit",
        "edit_plan": "plan_edit",
        "plan": "plan_edit",
        "replan": "plan_edit",
        "instruction": "instruction_edit",
        "message": "message_edit",
        "rewrite": "message_edit",
        "step": "step_edit",
        "action": "step_edit",
    }
    if kind in kind_aliases:
        raw["type"] = kind_aliases[kind]
        normalizations.append(f"type:{kind}->{raw['type']}")
    if "role" not in raw and raw.get("target_event_id"):
        raw["role"] = next(
            (event.role for event in sample.events if event.event_id == raw["target_event_id"]),
            "assistant",
        )
        normalizations.append("target_event.role->role")
    value = require_keys(
        raw, {"type", "target_event_id", "content", "reason", "role"}, name="DoVer edit"
    )
    if value["target_event_id"] not in {event.event_id for event in sample.events}:
        raise ValueError("DoVer edit target is absent from the trajectory")
    if value["type"] not in {
        "plan_edit",
        "instruction_edit",
        "message_edit",
        "step_edit",
    }:
        raise ValueError("DoVer edit type is unsupported")
    if normalizations:
        value["_schema_normalizations"] = normalizations
    return value


def _make_intervention(
    sample: BaselineSample, hypothesis: dict[str, Any], edit: dict[str, Any]
) -> dict[str, Any]:
    index = next(
        index
        for index, event in enumerate(sample.events)
        if event.event_id == edit["target_event_id"]
    )
    return {
        "type": edit["type"],
        "target_event_id": edit["target_event_id"],
        "cut_event_id": sample.events[index - 1].event_id if index > 0 else None,
        "replacement_events": [
            {
                "event_id": f"dover:patch:{index:04d}",
                "role": edit["role"],
                "kind": "message",
                "content": edit["content"],
                "tool_calls": [],
                "metadata": {
                    "hypothesis": hypothesis["hypothesis"],
                    "reason": edit["reason"],
                },
            }
        ],
        "checkpoint_required": True,
        "replay": "preserve_prefix_and_execute_from_intervention",
    }


def _score(evaluation: Any) -> float:
    value = evaluation.get("score") if isinstance(evaluation, dict) else None
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else 0.0
    )
