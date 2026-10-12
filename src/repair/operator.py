"""Answer-blind failure localization and bounded repair prompts.

The repair operator is intentionally isolated from benchmark references.  It
may inspect the original task, the failed output, and the normalized native
trajectory, but it must not read evaluator reference fields or hidden tests.
"""

from __future__ import annotations

import json
from typing import Any

ANSWER_BLIND_OPERATOR_ID = "heuristic/aria-repair-v2-answer-blind"
ARTIFACT_BENCHMARKS = {"humaneval", "swe-bench-lite"}
MAX_LOCALIZATION_CONTENT_CHARS = 4000
HIDDEN_OPERATOR_FIELDS = {
    "answer",
    "answer_index",
    "canonical_solution",
    "fail_to_pass",
    "gold",
    "gold_answer",
    "gold_patch",
    "pass_to_pass",
    "patch",
    "reference",
    "reference_answer",
    "reference_answers",
    "reference_solution",
    "seed_answer",
    "seed_solution",
    "test",
    "test_patch",
    "tests",
}


def diagnose_failure(
    result: dict[str, Any],
    messages: list[dict[str, Any]],
    trace_events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Localize a failed run without consulting its benchmark reference."""

    result = answer_blind_result(result)
    benchmark = str(result.get("benchmark") or "")
    evaluation = result.get("evaluation") if isinstance(result.get("evaluation"), dict) else {}
    metadata = result.get("sample_metadata") if isinstance(result.get("sample_metadata"), dict) else {}
    predicted = str(evaluation.get("predicted_answer") or "").strip()
    if not predicted:
        predicted = _last_final_answer(messages, trace_events)

    message_location = locate_answer_message(messages, predicted)
    trace_location = locate_answer_trace_event(trace_events, predicted)
    artifact_location = None
    if benchmark in ARTIFACT_BENCHMARKS and not (message_location or trace_location):
        artifact_location = locate_artifact_output(result)
    issue = _diagnose_issue(benchmark, predicted)
    return {
        "benchmark": benchmark,
        "predicted_answer": predicted or None,
        "evaluation_method": evaluation.get("method"),
        "trajectory_stage": (
            "artifact_generation" if artifact_location is not None else "assistant_final_answer"
        ),
        "message_location": message_location,
        "trace_location": trace_location,
        "artifact_location": artifact_location,
        "failure_type": issue["failure_type"],
        "issue": issue["issue"],
        "attention_instruction": issue["attention_instruction"],
        "repair_strategy": issue["repair_strategy"],
        "evidence": build_diagnosis_evidence(
            predicted,
            message_location,
            trace_location,
            artifact_location,
        ),
        "operator_id": ANSWER_BLIND_OPERATOR_ID,
        "answer_blind": True,
        "task_fingerprint_source": "model_visible_task",
        "benchmark_context": _public_benchmark_context(benchmark, metadata),
    }


def _diagnose_issue(benchmark: str, predicted: str) -> dict[str, str]:
    if benchmark == "mmlu":
        if predicted.upper() not in {"A", "B", "C", "D"}:
            return _issue(
                "invalid_choice_format",
                "The failed trajectory did not produce one extractable MMLU option letter.",
                "Re-evaluate every listed option against the question and return exactly one letter "
                "from A, B, C, or D in the final line.",
                "recompare_options_and_normalize",
            )
        return _issue(
            "wrong_choice_answer",
            "The failed trajectory selected an option that did not pass evaluation.",
            "Re-evaluate all listed options from the question itself, explicitly check the decisive "
            "condition, and return exactly one letter from A, B, C, or D. Do not reuse the previous "
            "choice without deriving it again.",
            "recompare_options",
        )
    if benchmark == "gsm-plus":
        return _issue(
            "wrong_arithmetic_or_reasoning",
            "The failed trajectory produced a GSM-Plus answer that did not pass evaluation.",
            "Re-read the current problem without relying on a seed problem or previous answer. "
            "List the required quantities and operations, rebuild the equation, check units and "
            "arithmetic, and then return the requested final value.",
            "recompute_from_current_problem",
        )
    if benchmark == "gaia":
        return _issue(
            "grounding_or_calculation_error",
            "The failed trajectory produced a GAIA answer that did not pass evaluation.",
            "Re-solve the question from its stated sources. Verify every retrieved fact, attachment, "
            "calculation, requested unit, and exact output format before finalizing.",
            "reground_recompute_and_verify",
        )
    if benchmark == "humaneval":
        return _issue(
            "implementation_or_test_failure",
            "The failed HumanEval implementation did not pass its evaluator.",
            "Re-read the function contract, inspect edge cases and return behavior, replace the failed "
            "implementation step, and run a syntax check plus task-derived tests before finalizing.",
            "reimplement_and_test",
        )
    if benchmark == "olympiadbench":
        return _issue(
            "mathematical_reasoning_error",
            "The failed OlympiadBench solution did not pass automatic scoring.",
            "Derive the solution again from the problem statement, verify each transformation and "
            "boundary condition, and simplify the final numeric or symbolic answer.",
            "rederive_and_verify",
        )
    if benchmark == "swe-bench-lite":
        return _issue(
            "patch_or_verification_failure",
            "The failed SWE-Bench-Lite run did not resolve the repository issue.",
            "Reproduce the reported behavior in the checked-out repository, identify the first faulty "
            "assumption or code path, apply the smallest source change, and run focused regression tests.",
            "reproduce_patch_and_test",
        )
    return _issue(
        "wrong_final_answer" if predicted else "missing_final_answer",
        "The failed trajectory did not produce an evaluator-passing result.",
        "Re-solve the task from the original prompt, verify each required condition, and produce the "
        "requested final artifact or answer without copying the previous result.",
        "recompute_and_verify",
    )


def _issue(
    failure_type: str,
    issue: str,
    attention_instruction: str,
    repair_strategy: str,
) -> dict[str, str]:
    return {
        "failure_type": failure_type,
        "issue": issue,
        "attention_instruction": attention_instruction,
        "repair_strategy": repair_strategy,
    }


def _public_benchmark_context(benchmark: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Retain only task-visible context; hidden labels/tests never enter diagnosis."""

    allowed = {
        "mmlu": ("subject",),
        "gaia": ("level", "file_name"),
        "swe-bench-lite": ("repo", "instance_id"),
    }.get(benchmark, ())
    return {key: metadata[key] for key in allowed if metadata.get(key) is not None}


def _last_final_answer(
    messages: list[dict[str, Any]],
    trace_events: list[dict[str, Any]],
) -> str:
    for item in reversed([*messages, *trace_events]):
        text = message_text(item) if isinstance(item, dict) else ""
        marker = "FINAL_ANSWER:"
        if marker in text:
            return text.rsplit(marker, 1)[-1].strip().splitlines()[0]
    return ""


def locate_answer_message(
    messages: list[dict[str, Any]],
    predicted: str,
) -> dict[str, Any] | None:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not isinstance(message, dict):
            continue
        content = message_text(message)
        if is_answer_message(message, content, predicted):
            return {
                "message_index": index,
                "role": message.get("role") or message.get("source"),
                "content": content,
            }
    return None


def locate_answer_trace_event(
    trace_events: list[dict[str, Any]],
    predicted: str,
) -> dict[str, Any] | None:
    for index in range(len(trace_events) - 1, -1, -1):
        event = trace_events[index]
        if not isinstance(event, dict):
            continue
        content = trace_event_text(event)
        record = event.get("record") if isinstance(event.get("record"), dict) else event
        if content and (
            "FINAL_ANSWER" in content
            or (predicted and predicted in content and is_answer_message(record, content, predicted))
        ):
            return {
                "event_index": index,
                "event_id": event.get("event_id"),
                "turn": event.get("turn") or record.get("turn"),
                "role": event.get("role") or event.get("source") or record.get("source"),
                "kind": event.get("kind") or record.get("type"),
                "content": content,
            }
    return None


def is_answer_message(message: dict[str, Any], content: str, predicted: str) -> bool:
    role = str(message.get("role") or "")
    source = str(message.get("source") or "")
    message_type = str(message.get("type") or "")
    if "FINAL_ANSWER" not in content and not (predicted and predicted in content):
        return False
    if role == "assistant":
        return True
    if source in {"MagenticOneOrchestrator", "Coder", "QA Engineer", "Verifier"}:
        return message_type != "SelectSpeakerEvent"
    return bool(source)


def message_text(message: dict[str, Any]) -> str:
    for key in ("content", "text", "result"):
        value = message.get(key)
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)
    return ""


def trace_event_text(event: dict[str, Any]) -> str:
    text = message_text(event)
    if text:
        return text
    record = event.get("record")
    return message_text(record) if isinstance(record, dict) else ""


def locate_artifact_output(result: dict[str, Any]) -> dict[str, Any] | None:
    """Locate a model-visible terminal artifact without reading evaluator secrets."""

    content = str(result.get("final_output") or "").strip()
    if not content:
        return None
    benchmark = str(result.get("benchmark") or "")
    return {
        "event_id": "failed:artifact:0001",
        "source": "result.final_output",
        "kind": "solution" if benchmark == "humaneval" else "workspace_patch",
        "role": "assistant",
        "content": bounded_localization_content(content),
    }


def bounded_localization_content(content: str) -> str:
    """Keep the most recent artifact-producing context within the prompt budget."""

    if len(content) <= MAX_LOCALIZATION_CONTENT_CHARS:
        return content
    return "[earlier artifact output truncated]\n" + content[-MAX_LOCALIZATION_CONTENT_CHARS:]


def build_diagnosis_evidence(
    predicted: str,
    message_location: dict[str, Any] | None,
    trace_location: dict[str, Any] | None,
    artifact_location: dict[str, Any] | None = None,
) -> list[str]:
    evidence: list[str] = []
    if predicted:
        evidence.append(f"predicted_answer={predicted}")
    if message_location is not None:
        evidence.append(f"assistant_message_index={message_location.get('message_index')}")
    if trace_location is not None:
        evidence.append(f"trace_event_index={trace_location.get('event_index')}")
    if artifact_location is not None:
        evidence.append(f"artifact_source={artifact_location.get('source')}")
    return evidence


def build_repair_prompt(
    result: dict[str, Any],
    diagnosis: dict[str, Any],
) -> dict[str, Any]:
    """Build a model-visible patch from answer-blind diagnosis fields only."""

    if diagnosis.get("answer_blind") is not True:
        raise ValueError("repair prompt requires an answer-blind diagnosis")
    result = answer_blind_result(result)
    metadata = result.get("sample_metadata") if isinstance(result.get("sample_metadata"), dict) else {}
    original_problem = build_original_problem(result, metadata)
    instruction = str(diagnosis["attention_instruction"])
    failure_location = build_failure_location(diagnosis)
    failed_step = build_failed_step(diagnosis)
    step_modification = build_step_modification(diagnosis)
    prompt_text = (
        "Repair the localized failed step in the previous trajectory.\n"
        f"Failed step location: {format_failure_location(failure_location)}\n"
        f"Failed step content: {failed_step.get('content') or '<missing>'}\n"
        f"Failure type: {diagnosis['failure_type']}\n"
        f"What went wrong: {diagnosis['issue']}\n"
        f"Modify this step by: {step_modification['instruction']}\n"
        "Re-solve the original problem from the current statement and produce a corrected result."
    )
    previous = diagnosis.get("predicted_answer")
    previous_instruction = (
        f"Do not repeat the previous result `{previous}` unless the new derivation proves it."
        if previous
        else "Do not repeat the previous artifact without re-deriving and validating it."
    )
    patched_task = (
        f"{original_problem}\n\n"
        "Repair the localized failed step from the previous trajectory:\n"
        f"- Failed step location: {format_failure_location(failure_location)}.\n"
        f"- Failed step content: {failed_step.get('content') or '<missing>'}\n"
        f"- What went wrong: {diagnosis['issue']}\n"
        f"- Modify this step by: {instruction}\n\n"
        "Solve the original problem again from the current statement only. Before finalizing, "
        "verify that the result follows from the current task. "
        f"{previous_instruction}"
    )
    return {
        "operator_id": ANSWER_BLIND_OPERATOR_ID,
        "answer_blind": True,
        "initial_problem": original_problem,
        "prompt_text": prompt_text,
        "failure_location": failure_location,
        "failed_step": failed_step,
        "step_modification": step_modification,
        "attention_issue": diagnosis["issue"],
        "attention_instruction": instruction,
        "patched_task": patched_task,
    }


def answer_blind_result(result: dict[str, Any]) -> dict[str, Any]:
    """Expose only fields the failed MAS itself could observe."""

    evaluation = result.get("evaluation") if isinstance(result.get("evaluation"), dict) else {}
    metadata = result.get("sample_metadata") if isinstance(result.get("sample_metadata"), dict) else {}
    return {
        "benchmark": result.get("benchmark"),
        "final_output": result.get("final_output"),
        "final_answer": result.get("final_answer"),
        "transcript": _strip_hidden_value(result.get("transcript") or []),
        "sample_metadata": _strip_hidden_value(metadata),
        "evaluation": {
            key: evaluation.get(key)
            for key in ("method", "predicted_answer", "answer_extracted")
            if evaluation.get(key) is not None
        },
    }


def _strip_hidden_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _strip_hidden_value(item)
            for key, item in value.items()
            if str(key).lower() not in HIDDEN_OPERATOR_FIELDS
        }
    if isinstance(value, list):
        return [_strip_hidden_value(item) for item in value]
    return value


def build_original_problem(result: dict[str, Any], metadata: dict[str, Any]) -> str:
    benchmark = str(result.get("benchmark") or metadata.get("benchmark") or "").lower()
    if benchmark == "mmlu":
        problem = format_mmlu_problem(metadata)
        if problem:
            return problem
    problem = str(
        metadata.get("task")
        or metadata.get("question")
        or metadata.get("problem_statement")
        or metadata.get("prompt")
        or ""
    )
    return problem or first_user_problem(result)


def format_mmlu_problem(metadata: dict[str, Any]) -> str:
    question = str(metadata.get("question") or metadata.get("task") or "").strip()
    choices = metadata.get("choices") if isinstance(metadata.get("choices"), list) else []
    parts: list[str] = []
    if metadata.get("subject"):
        parts.append(f"Subject: {metadata['subject']}")
    if question:
        parts.append(f"Question: {question}")
    if choices:
        parts.append(
            "Options:\n"
            + "\n".join(
                f"({chr(ord('A') + index)}) {choice}"
                for index, choice in enumerate(choices[:4])
            )
        )
    return "\n".join(parts)


def build_failure_location(diagnosis: dict[str, Any]) -> dict[str, Any]:
    return {
        "trajectory_stage": diagnosis.get("trajectory_stage"),
        "message_location": diagnosis.get("message_location"),
        "trace_location": diagnosis.get("trace_location"),
        "artifact_location": diagnosis.get("artifact_location"),
    }


def build_failed_step(diagnosis: dict[str, Any]) -> dict[str, Any]:
    message_location = diagnosis.get("message_location") or {}
    trace_location = diagnosis.get("trace_location") or {}
    artifact_location = diagnosis.get("artifact_location") or {}
    return {
        "stage": diagnosis.get("trajectory_stage"),
        "content": (
            message_location.get("content")
            or trace_location.get("content")
            or artifact_location.get("content")
        ),
        "predicted_answer": diagnosis.get("predicted_answer"),
        "failure_type": diagnosis.get("failure_type"),
        "issue": diagnosis.get("issue"),
    }


def build_step_modification(diagnosis: dict[str, Any]) -> dict[str, Any]:
    target = (
        "artifact-producing" if diagnosis.get("artifact_location") else "answer-producing"
    )
    return {
        "operation": "replace_failed_step",
        "target_stage": diagnosis.get("trajectory_stage"),
        "strategy": diagnosis.get("repair_strategy"),
        "instruction": diagnosis.get("attention_instruction"),
        "expected_change": (
            f"Replace the failed {target} step with a newly derived step that follows the "
            "current problem statement and the localized repair instruction."
        ),
    }


def format_failure_location(location: dict[str, Any]) -> str:
    parts = [str(location.get("trajectory_stage") or "unknown_stage")]
    message_location = location.get("message_location") or {}
    trace_location = location.get("trace_location") or {}
    artifact_location = location.get("artifact_location") or {}
    if message_location.get("message_index") is not None:
        parts.append(f"message_index={message_location['message_index']}")
    if trace_location.get("event_index") is not None:
        parts.append(f"trace_event_index={trace_location['event_index']}")
    if artifact_location.get("source"):
        parts.append(f"artifact_source={artifact_location['source']}")
    return ", ".join(parts)


def first_user_problem(result: dict[str, Any]) -> str:
    for item in result.get("transcript", []) or []:
        if not isinstance(item, dict) or not item.get("content"):
            continue
        if item.get("role") == "user" or item.get("source") == "user":
            return str(item["content"])
    return ""
