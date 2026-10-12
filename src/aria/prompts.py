"""Stable model instructions and JSON ordering shared by training and serving."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from aria.records import PATCH_OPERATORS

PATCH_OPERATOR_OPTIONS = ", ".join(sorted(PATCH_OPERATORS))
ATTRIBUTION_TAXONOMY = (
    "RA-1.1 task semantics or constraint misunderstanding -> replace_task_interpretation; "
    "RA-1.2 output or termination contract misunderstanding -> replace_plan; "
    "RA-1.3 faulty decomposition or subgoal -> replace_decomposition; "
    "RA-1.4 wrong capability or role plan -> replace_role_plan; "
    "RA-2.1 retrieval omitted or insufficient -> replace_retrieval; "
    "RA-2.2 wrong source, entity, version, time, or scope -> reground_evidence; "
    "RA-2.3 evidence extraction or aggregation error -> replace_evidence_extraction; "
    "RA-2.4 unsupported claim or hallucination -> remove_unsupported_claim; "
    "RA-3.1 invalid inference or proof gap -> replace_inference; "
    "RA-3.2 arithmetic, symbolic, or substitution error -> replace_computation; "
    "RA-3.3 finite evidence treated as proof -> replace_empirical_claim_with_proof; "
    "RA-3.4 wrong decision from correct intermediate work -> replace_decision; "
    "RA-4.1 context, artifact, or tool state lost at handoff -> restore_shared_state; "
    "RA-4.2 correct evidence ignored or overridden -> preserve_agent_evidence; "
    "RA-4.3 role or task handoff failure -> replace_handoff; "
    "RA-4.4 reset, repetition, or non-progress -> break_nonprogress_cycle; "
    "RA-5.1 intended or pseudo tool call not executed -> execute_tool_call; "
    "RA-5.2 wrong tool, query, arguments, or workspace -> replace_tool_call; "
    "RA-5.3 recoverable tool failure not recovered -> recover_tool_failure; "
    "RA-5.4 artifact not materialized or applied -> materialize_artifact; "
    "RA-5.5 implementation or edit incomplete or incorrect -> replace_implementation; "
    "RA-6.1 no verification -> add_verification; "
    "RA-6.2 weak or incomplete proxy verification -> strengthen_verification; "
    "RA-6.3 contradiction or failed check accepted -> reject_failed_candidate; "
    "RA-6.4 premature termination or missing result -> resume_to_completion; "
    "RA-6.5 output, extraction, or serialization error -> normalize_output"
)

GATE_INSTRUCTION = (
    "You are the ARIA intervention gate. Given only the observable trajectory "
    "prefix, estimate whether the pending prefix requires intervention. Return "
    "one JSON object only with gate_probability, a number from 0 to 1."
)

_LOCALIZATION_OUTPUT_INSTRUCTION = (
    "Identify the "
    "earliest observable event or concurrent events that introduced the error. Return "
    "one JSON object only with first_bad_event_ids, retained_frontier_ids, evidence, "
    "and confidence. first_bad_event_ids must be non-empty. retained_frontier_ids must "
    "precede the earliest bad event. Return exactly one evidence item for an event also "
    "listed in first_bad_event_ids. The evidence item must contain event_id, source, "
    "quote, and reason. source must be trajectory, artifact, or workspace. quote must "
    "be a consecutive verbatim substring of that event's content, at most 160 "
    "characters, with no ellipsis or paraphrase. confidence must be from 0 to 1. Use "
    "only observable input; never use a hidden answer. Return JSON only, with exactly "
    "this shape: "
    '{"first_bad_event_ids":["source:event:..."],"retained_frontier_ids":'
    '["source:event:..."],"evidence":[{"event_id":"source:event:...",'
    '"source":"trajectory","quote":"exact bounded quote","reason":"..."}],'
    '"confidence":0.0}.'
)

LOCALIZATION_INSTRUCTION = (
    "You are the ARIA causal-error localization model. The supplied trajectory is a "
    "completed, evaluator-confirmed failure with valid infrastructure. "
    + _LOCALIZATION_OUTPUT_INSTRUCTION
)

RUNTIME_LOCALIZATION_INSTRUCTION = (
    "You are the ARIA causal-error localization model. The deployment gate requested "
    "review of an unfinished, observable runtime prefix. No benchmark verdict is "
    "available; the gate can be wrong. Ground the candidate error in the supplied "
    "events and reflect uncertainty in confidence. "
    + _LOCALIZATION_OUTPUT_INSTRUCTION
)

_ATTRIBUTION_OUTPUT_INSTRUCTION = (
    "Use only the observable task, trajectory, localization, tool results, and "
    "artifacts. Assign exactly one RA-1.1 through RA-6.5 primary_code and return "
    "diagnosis, secondary_codes, responsible_agents, counterfactual, patch_operator, "
    "and confidence. The diagnosis must explain the localized error mechanism, and the "
    "counterfactual must explain why replacing it permits a correct continuation. "
    "Return exactly this JSON shape: "
    '{"diagnosis":"...","primary_code":"RA-x.y","secondary_codes":[],'
    '"responsible_agents":[{"agent_id":"...","role":"..."}],'
    '"counterfactual":"...","patch_operator":"...","confidence":0.0}. '
    f"Use this exact primary-code and operator mapping: {ATTRIBUTION_TAXONOMY}. "
    f"patch_operator must be exactly one of: {PATCH_OPERATOR_OPTIONS}. "
    "Never use a hidden answer or claim that an unexecuted action "
    "happened. Return the JSON object only, with no code fence or prose."
)

ATTRIBUTION_INSTRUCTION = (
    "You are the ARIA repair-attribution model. The input contains an "
    "evaluator-confirmed failed trajectory and a host-validated causal localization. "
    + _ATTRIBUTION_OUTPUT_INSTRUCTION
)

RUNTIME_ATTRIBUTION_INSTRUCTION = (
    "You are the ARIA repair-attribution model. The input contains an unfinished "
    "runtime prefix and a candidate localization whose structure and quoted evidence "
    "were host-validated. No evaluator has confirmed a failure or the diagnosis. "
    + _ATTRIBUTION_OUTPUT_INSTRUCTION
)

STRUCTURED_REPAIR_INSTRUCTION = (
    "You are the ARIA structured trajectory-patch model. Given the observable task "
    "constraints and trajectory, infer the failure mechanism and return one JSON "
    "object only with diagnosis, cut_event_id and replacement_events, in that order. "
    "If the input supplies a diagnosis, use it; otherwise infer it from the incident. "
    "The cut is the last retained event before the first causal bad event. Each "
    "replacement event requires event_id, role, kind, content, tool_calls, and "
    "metadata. The response must have exactly this shape: "
    '{"diagnosis":"failure mechanism","cut_event_id":'
    '"source:event:...","replacement_events":[{"event_id":"patch:event:0001",'
    '"role":"...","kind":"message","content":"executable replacement",'
    '"tool_calls":[],"metadata":{"replaces":["source:event:..."]}}]}. '
    "Use null for cut_event_id only when nothing can be retained. Produce executable "
    "replacement content, not a placeholder. You may propose tool calls or workspace "
    "edits but never emit tool_result or assert an unexecuted result. Return the JSON "
    "object only, with no code fence or prose."
)

REPAIR_INSTRUCTION = (
    "You are the ARIA repair decision model. Given an evaluator-confirmed failed "
    "trajectory, localize the earliest causal error, assign one RA-1.1 through RA-6.5 "
    "attribution, and propose an executable patch containing diagnosis, cut_event_id, "
    "and replacement_events. Never use hidden answers or assert unexecuted tool results."
)

TRANSFER_DIAGNOSIS_INSTRUCTION = (
    "You are the ARIA transfer diagnosis model. Given an incident and, when present, "
    "verified repair context, return one JSON object only with diagnosis. State the "
    "shared failure mechanism, not incident-specific content."
)


def instruction_for_task(task: str) -> str:
    if task == "deployment_gate":
        return GATE_INSTRUCTION
    if task == "failure_localization":
        return LOCALIZATION_INSTRUCTION
    if task == "structured_repair":
        return STRUCTURED_REPAIR_INSTRUCTION
    if task == "repair_attribution":
        return ATTRIBUTION_INSTRUCTION
    if task == "repair_preference":
        return STRUCTURED_REPAIR_INSTRUCTION
    if task == "transfer_diagnosis_preference":
        return TRANSFER_DIAGNOSIS_INSTRUCTION
    if task in {
        "transfer_diagnosis_internalization",
        "training_ready_repair",
    }:
        return REPAIR_INSTRUCTION
    raise ValueError(f"unsupported ARIA training task: {task}")


def canonical_model_json(value: Mapping[str, Any]) -> str:
    """Serialize model inputs identically regardless of host mapping order."""

    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def ordered_training_output(
    task: str,
    output: Mapping[str, Any],
) -> dict[str, Any]:
    """Preserve the paper's diagnosis-first autoregressive repair schema."""

    if task == "deployment_gate":
        fields = ("gate_probability",)
    elif task == "failure_localization":
        fields = (
            "first_bad_event_ids",
            "retained_frontier_ids",
            "evidence",
            "confidence",
        )
    elif task == "repair_attribution":
        fields = (
            "diagnosis",
            "primary_code",
            "secondary_codes",
            "responsible_agents",
            "counterfactual",
            "patch_operator",
            "confidence",
        )
    elif task == "structured_repair":
        fields = ("diagnosis", "cut_event_id", "replacement_events")
    elif task == "transfer_diagnosis_preference":
        fields = ("diagnosis",)
    else:
        fields = ("diagnosis", "cut_event_id", "replacement_events")
    missing = [field for field in fields if field not in output]
    if missing:
        raise ValueError(f"ARIA {task} output is missing fields: {missing}")
    return {field: output[field] for field in fields}
