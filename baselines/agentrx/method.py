"""AgentRx guarded-constraint diagnosis followed by shared repair."""

from __future__ import annotations

from typing import Any, ClassVar

from baselines.core.method import BaselineMethod, validate_attribution
from baselines.core.models import ModelAdapter, require_keys
from baselines.core.prompting import repair_from_context
from baselines.core.replay import ReplayBackend, execute
from baselines.core.types import BaselineSample

GLOBAL_SYSTEM = """Implement AgentRx global constraint synthesis. From the task constraints,
tool schemas, roles, and policy, produce executable guarded invariants. Return JSON only as
{"constraints": [{"constraint_id": string, "guard": string, "assertion": string,
"check_type": "programmatic"|"semantic"}]}. Return at most eight concise constraints. Do not use a
hidden reference answer or copy any input fields into the response."""

DYNAMIC_SYSTEM = """Implement AgentRx dynamic constraint synthesis for the current trajectory
prefix. Produce context-dependent guarded invariants implied by earlier observations and plans.
Return the same {"constraints": [...]} schema as global synthesis with at most eight concise
constraints. Avoid restating irrelevant rules and never copy the prefix or global constraints."""

CHECK_SYSTEM = """Implement the AgentRx guarded checker for one trajectory step. Evaluate each
applicable constraint against the prefix and current step. Return JSON only as
{"evaluations": [{"constraint_id": string, "verdict": "SAT"|"VIOL"|"SKIP",
"evidence": string}]}. Keep evidence concise and precisely identify visible trajectory content;
never copy the prefix, current event, or constraints into the response."""

JUDGE_SYSTEM = """Implement the AgentRx root-cause judge. Given the task, trajectory, taxonomy
checklist, and step-indexed validation log, select the first violation sufficient to explain the final
failure. Violations are evidence and may be overridden by trajectory context. Return JSON only with
responsible_agent, zero-based step_index, reason, and failure_category."""
CONSTRAINT_RETRY = """Return only {"constraints": [...]} with at most four unique concise
constraints. Do not repeat a constraint or copy task, prefix, events, roles, or input fields."""
CHECK_RETRY = """Return only {"evaluations": [...]} with at most one concise evaluation per
supplied constraint_id. Do not repeat IDs and do not copy prefix, events, or constraints."""

TAXONOMY = [
    "Instruction/Plan Adherence Failure",
    "Invention of New Information",
    "Invalid Invocation",
    "Misinterpretation of Tool Output / Handoff Failure",
    "Intent-Plan Misalignment",
    "Under-specified User Intent",
    "Intent Not Supported",
    "Guardrails Triggered",
    "System Failure",
    "Inconclusive",
]


class AgentRxRepair(BaselineMethod):
    name = "agentrx"
    setting = "passive"
    provenance: ClassVar[dict[str, Any]] = {
        "paper": "AgentRx: Diagnosing AI Agent Failures from Execution Trajectories (2026)",
        "official_repository": "https://github.com/microsoft/AgentRx",
        "official_commit": "f228165bfec60a801fd5fedd9d8ffe0f9de0c69d",
        "license": "MIT",
        "adaptation": "ARIA events are already the normalized IR; hidden ground truth is disabled",
    }

    def run(
        self,
        sample: BaselineSample,
        model: ModelAdapter | None,
        replay: ReplayBackend | None = None,
    ):
        selected = self.require_model(model)
        result = self.result(sample, status="proposed")
        global_constraints, call = selected.json_call(
            stage="agentrx.global_constraints",
            system=GLOBAL_SYSTEM,
            payload={
                "task": sample.task,
                "constraints": sample.constraints,
                "roles": sorted({event.role for event in sample.events}),
                "tool_schemas": sample.metadata.get("tool_schemas", []),
                "domain_policy": sample.metadata.get("domain_policy"),
            },
            validator=_validate_constraints,
            allow_truncated_containers=True,
            final_retry_instruction=CONSTRAINT_RETRY,
        )
        result.model_calls.append(call)
        validation_log: list[dict[str, Any]] = []
        skipped_empty_events = 0
        max_events = int(self.options.get("max_events", len(sample.events)))
        for index, event in enumerate(sample.events[:max_events]):
            if not event.content.strip() and not event.tool_calls and not event.tool_results:
                skipped_empty_events += 1
                continue
            prefix = [item.to_dict() for item in sample.events[: index + 1]]
            dynamic, dynamic_call = selected.json_call(
                stage=f"agentrx.dynamic_constraints.{index:04d}",
                system=DYNAMIC_SYSTEM,
                payload={
                    "task": sample.task,
                    "prefix": prefix,
                    "global_constraints": global_constraints["constraints"],
                },
                validator=_validate_constraints,
                allow_truncated_containers=True,
                final_retry_instruction=CONSTRAINT_RETRY,
            )
            result.model_calls.append(dynamic_call)
            constraints = _deduplicate_constraints(
                [*global_constraints["constraints"], *dynamic["constraints"]]
            )
            checked, check_call = selected.json_call(
                stage=f"agentrx.check.{index:04d}",
                system=CHECK_SYSTEM,
                payload={
                    "prefix": prefix,
                    "current_event": event.to_dict(),
                    "constraints": constraints,
                },
                validator=lambda value, known=constraints: _validate_checks(
                    value, known
                ),
                allow_truncated_containers=True,
                final_retry_instruction=CHECK_RETRY,
            )
            result.model_calls.append(check_call)
            for evaluation in checked["evaluations"]:
                if evaluation["verdict"] == "VIOL":
                    validation_log.append(
                        {"step_index": index, "event_id": event.event_id, **evaluation}
                    )
        attribution, judge_call = selected.json_call(
            stage="agentrx.root_cause_judge",
            system=JUDGE_SYSTEM,
            payload={
                **sample.model_view(include_feedback=True),
                "taxonomy": TAXONOMY,
                "validation_log": validation_log,
            },
            validator=lambda value: _validate_judgment(value, sample),
        )
        result.model_calls.append(judge_call)
        result.attribution = attribution
        result.diagnostics["validation_log"] = validation_log
        result.diagnostics["evaluated_event_count"] = min(
            max_events, len(sample.events)
        ) - skipped_empty_events
        result.diagnostics["skipped_empty_event_count"] = skipped_empty_events
        intervention, repair_call = repair_from_context(
            selected,
            sample,
            stage="agentrx.shared_repair",
            attribution=attribution,
        )
        result.model_calls.append(repair_call)
        result.intervention = intervention
        execution = execute(replay, sample, intervention, label="agentrx-repair")
        if execution is not None:
            result.executions.append(execution)
            result.selected_execution_label = execution.label
            result.status = "executed"
        return result


def _validate_constraints(raw: Any) -> dict[str, Any]:
    value = require_keys(raw, {"constraints"}, name="constraints")
    if not isinstance(value["constraints"], list):
        raise TypeError("constraints.constraints must be a list")
    normalizations: list[str] = []
    normalized: list[dict[str, Any]] = []
    for index, raw_constraint in enumerate(value["constraints"]):
        if not isinstance(raw_constraint, dict):
            normalizations.append(f"constraint[{index}]:drop_non_object")
            continue
        constraint = dict(raw_constraint)
        required_constraint_keys = {
            "constraint_id",
            "guard",
            "assertion",
            "check_type",
        }
        if not required_constraint_keys <= set(constraint):
            normalizations.append(f"constraint[{index}]:drop_incomplete")
            continue
        if "guard" not in constraint:
            constraint["guard"] = "always"
            normalizations.append(f"constraint[{index}]:default_guard")
        if str(constraint.get("check_type") or "").lower() in {
            "logical",
            "llm",
            "natural_language",
        }:
            constraint["check_type"] = "semantic"
            normalizations.append(f"constraint[{index}]:check_type->semantic")
        require_keys(
            constraint,
            {"constraint_id", "guard", "assertion", "check_type"},
            name=f"constraint {index}",
        )
        if constraint["check_type"] not in {"programmatic", "semantic"}:
            raise ValueError("constraint.check_type must be programmatic or semantic")
        normalized.append(constraint)
        if len(normalized) == 8:
            if index + 1 < len(value["constraints"]):
                normalizations.append("constraints:truncate_to_8")
            break
    if value["constraints"] and not normalized:
        # Constraint synthesis is advisory. A model may correctly conclude
        # that no executable invariant applies; represent that as a no-op so
        # the checker and root-cause judge can still run.
        normalizations.append("constraints:all_invalid->empty_noop")
    value["constraints"] = normalized
    if normalizations:
        value["_schema_normalizations"] = normalizations
    return value


def _deduplicate_constraints(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for value in values:
        by_id[str(value["constraint_id"])] = value
    return list(by_id.values())


def _validate_checks(raw: Any, constraints: list[dict[str, Any]]) -> dict[str, Any]:
    value = require_keys(raw, {"evaluations"}, name="checks")
    if not isinstance(value["evaluations"], list):
        raise TypeError("checks.evaluations must be a list")
    known = {str(item["constraint_id"]) for item in constraints}
    normalizations: list[str] = []
    known_order = [str(item["constraint_id"]) for item in constraints]
    normalized: list[dict[str, Any]] = []
    seen_constraint_ids: set[str] = set()
    for index, raw_item in enumerate(value["evaluations"]):
        if not isinstance(raw_item, dict):
            normalizations.append(f"check[{index}]:drop_non_object")
            continue
        item = dict(raw_item)
        required = {"constraint_id", "verdict", "evidence"}
        if not required <= set(item):
            normalizations.append(f"check[{index}]:drop_incomplete")
            continue
        if "constraint_id" not in item:
            item["constraint_id"] = known_order[min(index, len(known_order) - 1)]
            normalizations.append(f"check[{index}]:default_constraint_id")
        original_constraint_id = str(item["constraint_id"])
        item["constraint_id"] = _resolve_constraint_id(
            item["constraint_id"], known_order
        )
        if item["constraint_id"] != original_constraint_id:
            normalizations.append(
                f"check[{index}].constraint_id->{item['constraint_id']}"
            )
        item["constraint_id"] = _resolve_constraint_id(
            item["constraint_id"], known_order
        )
        if str(item.get("verdict") or "").upper() in {"SKIPPED", "PASS", "PASSED"}:
            item["verdict"] = "SKIP" if str(item.get("verdict")).upper() == "SKIPPED" else "SAT"
            normalizations.append(f"check[{index}].verdict->{item['verdict']}")
        if str(item.get("verdict") or "").upper() in {"VIOLATED", "VIOLATION", "FAIL", "FAILED"}:
            item["verdict"] = "VIOL"
            normalizations.append(f"check[{index}].verdict->VIOL")
        if "evidence" not in item:
            item["evidence"] = str(item.get("reason") or item.get("explanation") or "")
            normalizations.append(f"check[{index}]:default_evidence")
        require_keys(
            item, {"constraint_id", "verdict", "evidence"}, name=f"check {index}"
        )
        if str(item["constraint_id"]) not in known:
            normalizations.append(f"check[{index}]:drop_unknown_constraint")
            continue
        constraint_id = str(item["constraint_id"])
        if constraint_id in seen_constraint_ids:
            normalizations.append(f"check[{index}]:drop_duplicate_constraint")
            continue
        seen_constraint_ids.add(constraint_id)
        if item["verdict"] not in {"SAT", "VIOL", "SKIP"}:
            raise ValueError("check.verdict must be SAT, VIOL, or SKIP")
        normalized.append(item)
    if value["evaluations"] and not normalized:
        raise ValueError("checks contain no known constraints")
    value["evaluations"] = normalized
    if normalizations:
        value["_schema_normalizations"] = normalizations
    return value


def _resolve_constraint_id(value: Any, known: list[str]) -> str:
    candidate = str(value)
    if candidate in known:
        return candidate
    folded = candidate.casefold()
    for item in known:
        if item.casefold() == folded:
            return item
    for suffix in (
        "_assertion",
        "_guard",
        "_programmatic",
        "_semantic",
        "_check",
    ):
        if folded.endswith(suffix):
            base = folded[: -len(suffix)]
            for item in known:
                if item.casefold() == base:
                    return item
    try:
        index = int(candidate)
    except (TypeError, ValueError):
        return candidate
    if 1 <= index <= len(known):
        return known[index - 1]
    if 0 <= index < len(known):
        return known[index]
    return candidate


def _validate_judgment(raw: Any, sample: BaselineSample) -> dict[str, Any]:
    if isinstance(raw, dict) and "step_index" not in raw:
        event_id = raw.get("event_id")
        positions = {event.event_id: index for index, event in enumerate(sample.events)}
        if event_id in positions:
            raw = {**raw, "step_index": positions[event_id]}
    value = validate_attribution(raw)
    require_keys(value, {"failure_category"}, name="AgentRx judgment")
    normalizations = value.setdefault("_schema_normalizations", [])
    if not 0 <= value["step_index"] < len(sample.events):
        requested = value["step_index"]
        event_ids = {event.event_id: index for index, event in enumerate(sample.events)}
        exact_id = f"source:event:{requested:04d}"
        if exact_id in event_ids:
            value["step_index"] = event_ids[exact_id]
            normalizations.append("step_index->source_event_suffix_index")
        else:
            candidates = []
            for event_id, index in event_ids.items():
                suffix = event_id.rsplit(":", 1)[-1]
                if suffix.isdigit() and int(suffix) <= requested:
                    candidates.append((int(suffix), index))
            if candidates:
                value["step_index"] = max(candidates)[1]
                normalizations.append("step_index->nearest_source_event_suffix")
    if not 0 <= value["step_index"] < len(sample.events):
        raise ValueError("AgentRx step_index is outside the trajectory")
    value["event_id"] = sample.events[value["step_index"]].event_id
    return value
