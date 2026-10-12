"""Read-only native delivery/backend evidence, not a new scorer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def native_delivery_evidence(row: Mapping[str, Any]) -> dict[str, Any] | None:
    if (row.get("benchmark") not in {"swe-bench-lite", "gaia"} or row.get("mas") != "magentic-one"
            or row.get("status") == "invalid"):
        return None
    executions = row.get("executions") or []
    label = row.get("selected_execution_label")
    execution = next((item for item in executions if item.get("label") == label),
                     executions[-1] if executions else None)
    artifacts: dict[str, str] = {}

    def read_json(path: Path) -> Any:
        raw = path.read_bytes()
        artifacts[str(path.resolve())] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    if execution is None:
        if not (row.get("method") == "no_intervention"
                or (row.get("setting") == "active" and row.get("intervention") is None)):
            return None
        control_path = row.get("paired_control_path")
        if not control_path:
            return None
        control = read_json(Path(control_path))
        execution = control.get("execution") or {}
        if execution.get("evaluation") != row.get("control_evaluation"):
            raise ValueError("delivery audit control differs from the recorded outcome")
    evaluation = execution.get("evaluation") or {}
    if execution.get("status") != "completed" or not isinstance(evaluation.get("correct"), bool):
        return None
    result_path = (execution.get("artifact_refs") or {}).get("result_path")
    if not result_path:
        return None
    result = read_json(Path(result_path))
    if result.get("status") != "completed" or not result.get("native_messages_path"):
        return None
    messages = read_json(Path(result["native_messages_path"]))
    if not isinstance(messages, list) or not all(isinstance(item, dict) for item in messages):
        raise ValueError("delivery audit requires the native message list")
    terminal_responses = sum(item.get("source") == "ComputerTerminal" for item in messages)
    terminal_selections = sum(
        item.get("type") == "SelectSpeakerEvent" and isinstance(item.get("content"), list)
        and "ComputerTerminal" in item["content"] for item in messages
    )
    coder_code_messages = sum(
        item.get("source") == "Coder" and isinstance(item.get("content"), str)
        and "```" in item["content"] for item in messages
    )
    endpoint_errors = sum(
        item.get("source") == "WebSurfer"
        and isinstance(item.get("content", item.get("text")), str)
        and "/tokenize" in item.get("content", item.get("text"))
        and "HTTPStatusError" in item.get("content", item.get("text"))
        for item in messages
    )
    if endpoint_errors:
        category = "websurfer_model_endpoint_error"
    elif evaluation.get("harness_status") != "empty_patch" or evaluation["correct"] is not False:
        return None
    elif terminal_responses:
        category = "terminal_responded_without_patch"
    elif terminal_selections:
        category = "terminal_selected_without_recorded_response"
    elif coder_code_messages:
        category = "coder_code_not_dispatched_to_terminal"
    else:
        category = "no_recorded_terminal_activity"
    return {
        "category": category,
        "terminal_responses": terminal_responses,
        "terminal_selections": terminal_selections,
        "coder_code_fenced_messages": coder_code_messages,
        "websurfer_endpoint_errors": endpoint_errors,
        "artifacts": artifacts,
        "scope": "completed Magentic-One GAIA/SWE backend faults or SWE empty patches; code fences do not establish executable or correct code",
    }


def native_trial_backend_faults(row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Audit recorded DoVer executions, including trials not selected for delivery."""
    if row.get("method") != "dover" or row.get("status") == "invalid":
        return {}
    # Final publication verifies this supersession against both source datasets.
    superseded = set((row.get("capability_correction") or {}).get("superseded_execution_labels", []))
    faults = {}
    for execution in row.get("executions") or []:
        label = execution["label"]
        if label in superseded:
            continue
        evidence = native_delivery_evidence({**row, "executions": [execution], "selected_execution_label": label})
        if evidence and evidence.get("websurfer_endpoint_errors"):
            faults[label] = evidence
    return faults
