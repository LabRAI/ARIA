"""ARIA repair dataset record construction and per-sample serialization."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from common.artifacts import atomic_write_jsonl, utc_now
from common.runtime_support import runtime_state_from_result, safe_name
from repair.operator import ANSWER_BLIND_OPERATOR_ID
from repair.schema import REPAIR_SCHEMA_VERSION
from repair.utils import sanitize_result


def build_success_noop_record(
    *,
    name: str,
    spec: dict[str, Any],
    config: Any,
    source_batch_dir: Path,
    source: dict[str, Any],
) -> dict[str, Any]:
    """Build a minimal v4 ``success_noop`` record from a correctly-answered run."""
    result = source["result"]
    metadata = result.get("sample_metadata") if isinstance(result.get("sample_metadata"), dict) else {}
    evaluation = result.get("evaluation") if isinstance(result.get("evaluation"), dict) else {}
    runtime = build_runtime_context(result, config)
    runtime_id = str(runtime["runtime_id"])
    model_id = str(runtime["model_id"])
    mas_id = str(result.get("mas") or "unknown-mas")
    benchmark_id = str(result.get("benchmark") or metadata.get("benchmark") or "")
    sample_id = str(result.get("sample_id") or metadata.get("sample_id") or f"sample-{time.time_ns()}")
    config_id = str(spec.get("config_id") or f"config/{safe_name(name)}")
    split = str(spec.get("split") or spec.get("dataset_split") or "train")
    sample_dir = f"samples/{safe_name(sample_id)}"
    success_eval = normalize_v4_evaluation(evaluation, benchmark_id)
    return {
        "schema_version": REPAIR_SCHEMA_VERSION,
        "record_id": f"aria-{safe_name(benchmark_id)}-{safe_name(mas_id)}-{safe_name(sample_id)}-noop",
        "dataset_id": name,
        "sample_id": sample_id,
        "task_id": str(metadata.get("task_id") or sample_id),
        "trajectory_id": str(result.get("run_id") or f"{mas_id}-{benchmark_id}-{sample_id}"),
        "split": split,
        "record_type": "success_noop",
        "source_refs": {
            "mas_id": mas_id,
            "benchmark_id": benchmark_id,
            "runtime_id": runtime_id,
            "model_id": model_id,
            "teacher_model_id": None,
            "repair_operator_id": "noop",
            "config_id": config_id,
            "host_adapter_id": host_adapter_id(result, mas_id),
            "sample_dir": sample_dir,
            "failed_dir": None,
            "repaired_dir": None,
            "source_batch_dir": str(source_batch_dir),
        },
        "sample": {
            "benchmark": benchmark_id,
            "benchmark_split": metadata.get("split") or spec.get("sample_split"),
            "input_format": "text",
            "task_text": metadata.get("question") or metadata.get("task") or "",
            "reference": {
                "answer": evaluation.get("reference_answer", metadata.get("reference_answer")),
                "answer_format": answer_format_for_benchmark(benchmark_id),
                "evaluation_method": evaluation.get("method"),
                "hidden_from_model": True,
            },
            "metadata": strip_large_sample_metadata(metadata),
        },
        "failed_run": None,
        "control_run": None,
        "localization": None,
        "patch": None,
        "repaired_run": None,
        "outcome": {
            "source_success": True,
            "repair_attempted": False,
            "repair_success": False,
            "repaired_success": False,
            "attempt_count": 0,
            "delta_score": None,
            "new_failure_labels": [],
            "outcome_notes": "success_noop: no intervention needed",
        },
        "quality": {
            "natural_or_synthetic": str(spec.get("natural_or_synthetic", "natural")),
            "dedup_group": sample_id,
            "annotation_status": str(spec.get("annotation_status", "auto_labeled")),
            "audit_status": str(spec.get("audit_status", "not_audited")),
            "created_at": utc_now(),
        },
        "success_run": {
            "status": result.get("status"),
            "trajectory_ref": f"{sample_dir}/failed/trajectory.jsonl",
            "messages_ref": f"{sample_dir}/failed/messages.json",
            "result_ref": f"{sample_dir}/failed/result.json",
            "final_output": result.get("final_output") or result.get("final_answer"),
            "evaluation": success_eval,
        },
    }

def build_v4_dataset_record(
    *,
    name: str,
    spec: dict[str, Any],
    config: Any,
    source_batch_dir: Path,
    source: dict[str, Any],
    diagnosis: dict[str, Any],
    repair: dict[str, Any],
    failed_snapshot: dict[str, Any],
) -> dict[str, Any]:
    result = source["result"]
    metadata = result.get("sample_metadata") if isinstance(result.get("sample_metadata"), dict) else {}
    evaluation = result.get("evaluation") if isinstance(result.get("evaluation"), dict) else {}
    patch_payload = repair["patch"]
    mas_id = str(result.get("mas") or metadata.get("mas") or "unknown-mas")
    benchmark_id = str(result.get("benchmark") or metadata.get("benchmark") or diagnosis.get("benchmark") or "")
    sample_id = str(result.get("sample_id") or metadata.get("sample_id") or f"sample-{time.time_ns()}")
    runtime = build_runtime_context(result, config)
    runtime_id = str(runtime["runtime_id"])
    model_id = str(runtime["model_id"])
    config_id = str(spec.get("config_id") or f"config/{safe_name(name)}")
    trajectory_id = str(result.get("run_id") or f"{mas_id}-{benchmark_id}-{sample_id}")
    split = str(spec.get("split") or spec.get("dataset_split") or "train")
    sample_dir = f"samples/{safe_name(sample_id)}"
    failed_eval = normalize_v4_evaluation(evaluation, benchmark_id)
    repaired_eval = (
        normalize_v4_evaluation(repair.get("primary_evaluation") or {}, benchmark_id)
        if repair.get("enabled")
        else None
    )
    control_payload = repair.get("control") if isinstance(repair.get("control"), dict) else {}
    control_eval = (
        normalize_v4_evaluation(control_payload.get("evaluation") or {}, benchmark_id)
        if repair.get("enabled")
        else None
    )
    localization = build_v4_localization(diagnosis)
    patch = build_v4_patch(sample_id, diagnosis, patch_payload, localization, sample_dir)
    repaired_run = build_v4_repaired_run(repair, sample_dir, repaired_eval)
    control_run = build_v4_control_run(repair, sample_dir, control_eval)
    source_success = failed_eval.get("success") is True
    repair_attempted = repair.get("enabled") is True
    repaired_success = repaired_eval.get("success") is True if isinstance(repaired_eval, dict) else False
    control_success = control_eval.get("success") is True if isinstance(control_eval, dict) else False
    record_type = v4_record_type(source_success, repair_attempted, repaired_success)
    return {
        "schema_version": REPAIR_SCHEMA_VERSION,
        "record_id": f"aria-{safe_name(benchmark_id)}-{safe_name(mas_id)}-{safe_name(sample_id)}-r001",
        "dataset_id": name,
        "sample_id": sample_id,
        "task_id": str(metadata.get("task_id") or sample_id),
        "trajectory_id": trajectory_id,
        "split": split,
        "record_type": record_type,
        "source_refs": {
            "mas_id": mas_id,
            "benchmark_id": benchmark_id,
            "runtime_id": runtime_id,
            "model_id": model_id,
            "teacher_model_id": spec.get("teacher_model_id"),
            "repair_operator_id": str(
                spec.get("repair_operator_id") or ANSWER_BLIND_OPERATOR_ID
            ),
            "config_id": config_id,
            "host_adapter_id": host_adapter_id(result, mas_id),
            "sample_dir": sample_dir,
            "failed_dir": f"{sample_dir}/failed",
            "control_dir": f"{sample_dir}/control" if repair_attempted else None,
            "repaired_dir": f"{sample_dir}/repaired" if repair_attempted else None,
            "source_batch_dir": str(source_batch_dir),
        },
        "sample": {
            "benchmark": benchmark_id,
            "benchmark_split": metadata.get("split") or spec.get("sample_split"),
            "input_format": "text",
            "task_text": metadata.get("question") or metadata.get("task") or patch_payload.get("initial_problem") or "",
            "reference": {
                "answer": evaluation.get("reference_answer", metadata.get("reference_answer")),
                "answer_format": answer_format_for_benchmark(benchmark_id),
                "evaluation_method": evaluation.get("method"),
                "hidden_from_model": True,
            },
            "metadata": strip_large_sample_metadata(metadata),
        },
        "failed_run": {
            "status": result.get("status"),
            "trajectory_ref": f"{sample_dir}/failed/trajectory.jsonl",
            "messages_ref": f"{sample_dir}/failed/messages.json",
            "result_ref": f"{sample_dir}/failed/result.json",
            "log_ref": f"{sample_dir}/failed/run.log",
            "event_count": len(source.get("trace_events") or []),
            "final_output": result.get("final_output") or result.get("final_answer"),
            "evaluation": failed_eval,
            "failure_window_ref": f"{sample_dir}/repair/failure_window.jsonl",
            "failure_window_event_count": len(extract_failure_window(source.get("trace_events") or [], diagnosis)),
        },
        "localization": localization,
        "patch": patch,
        "control_run": control_run,
        "repaired_run": repaired_run,
        "outcome": {
            "source_success": source_success,
            "repair_attempted": repair_attempted,
            "repair_success": repaired_success,
            "repaired_success": repaired_success,
            "control_success": control_success,
            "attributable_repair_success": repaired_success and not control_success,
            "paired_outcome": paired_outcome(control_success, repaired_success),
            "attempt_count": len(repair.get("attempts") or []),
            "delta_score": score_delta(failed_eval, repaired_eval),
            "new_failure_labels": [],
            "outcome_notes": None,
        },
        "quality": {
            "natural_or_synthetic": str(spec.get("natural_or_synthetic", "natural")),
            "dedup_group": sample_id,
            "annotation_status": str(spec.get("annotation_status", "auto_labeled")),
            "audit_status": str(spec.get("audit_status", "not_audited")),
            "created_at": utc_now(),
        },
    }


def v4_record_type(source_success: bool, repair_attempted: bool, repaired_success: bool) -> str:
    if source_success and not repair_attempted:
        return "success_noop"
    if repair_attempted and repaired_success:
        return "failed_successful_repair"
    return "failed_failed_repair"


def paired_outcome(control_success: bool, repaired_success: bool) -> str:
    if control_success and repaired_success:
        return "both_pass"
    if control_success:
        return "control_only"
    if repaired_success:
        return "patched_only"
    return "both_fail"


def host_adapter_id(result: dict[str, Any], mas_id: str) -> str:
    bridge = result.get("bridge") if isinstance(result.get("bridge"), dict) else {}
    adapter_slug = bridge.get("adapter_slug")
    native_head = result.get("native_repo_head")
    if adapter_slug and native_head:
        return f"{adapter_slug}@{native_head}"
    if native_head:
        return f"{mas_id}@{native_head}"
    return str(adapter_slug or mas_id)


def normalize_v4_evaluation(evaluation: dict[str, Any], benchmark_id: str) -> dict[str, Any]:
    correct = evaluation.get("correct")
    score = evaluation.get("score")
    return {
        "success": correct if isinstance(correct, bool) else None,
        "score": score,
        "predicted_answer": evaluation.get("predicted_answer"),
        "reference_answer": evaluation.get("reference_answer"),
        "answer_extracted": evaluation.get("answer_extracted"),
        "evaluator_id": evaluator_id(benchmark_id, evaluation.get("method")),
        "method": evaluation.get("method"),
        "reason": evaluation.get("reason"),
    }


def evaluator_id(benchmark_id: str, method: Any) -> str:
    if method:
        return f"{safe_name(benchmark_id)}-{safe_name(str(method))}-v1"
    return f"{safe_name(benchmark_id)}-unknown-evaluator-v1"


def build_v4_localization(diagnosis: dict[str, Any]) -> dict[str, Any]:
    first_bad_event_id = diagnosis_event_id(diagnosis)
    role = diagnosis_role(diagnosis)
    evidence_quote = failed_step_quote(diagnosis)
    evidence_source = "artifact" if diagnosis.get("artifact_location") else "trajectory"
    failure_type = str(diagnosis.get("failure_type") or "wrong_final_answer")
    return {
        "mast_labels": mast_labels_for_failure(failure_type),
        "failure_stage": "before_finalize",
        "first_bad_event_id": first_bad_event_id,
        "suspect_mas_agent_id": safe_name(role or "assistant"),
        "suspect_mas_agent_role": role or "assistant",
        "failure_type": failure_type,
        "failure_evidence": [
            {
                "event_id": first_bad_event_id,
                "source": evidence_source,
                "quote": evidence_quote,
                "reason": diagnosis.get("issue"),
            }
        ],
        "label_source": "heuristic",
        "label_confidence": 0.65,
    }


def diagnosis_event_id(diagnosis: dict[str, Any]) -> str:
    trace_location = diagnosis.get("trace_location") if isinstance(diagnosis.get("trace_location"), dict) else {}
    event_id = trace_location.get("event_id")
    if event_id:
        text = str(event_id)
        return text if text.startswith("failed:") else text.replace("source:", "failed:", 1)
    index = trace_location.get("event_index")
    if isinstance(index, int):
        return f"failed:event:{index + 1:04d}"
    artifact_location = (
        diagnosis.get("artifact_location")
        if isinstance(diagnosis.get("artifact_location"), dict)
        else {}
    )
    if artifact_location.get("event_id"):
        return str(artifact_location["event_id"])
    return "failed:event:0001"


def diagnosis_role(diagnosis: dict[str, Any]) -> str:
    trace_location = diagnosis.get("trace_location") if isinstance(diagnosis.get("trace_location"), dict) else {}
    message_location = diagnosis.get("message_location") if isinstance(diagnosis.get("message_location"), dict) else {}
    artifact_location = diagnosis.get("artifact_location") if isinstance(diagnosis.get("artifact_location"), dict) else {}
    return str(
        trace_location.get("mas_agent_role")
        or trace_location.get("role")
        or message_location.get("role")
        or artifact_location.get("role")
        or "assistant"
    )


def failed_step_quote(diagnosis: dict[str, Any]) -> str:
    message_location = diagnosis.get("message_location") if isinstance(diagnosis.get("message_location"), dict) else {}
    trace_location = diagnosis.get("trace_location") if isinstance(diagnosis.get("trace_location"), dict) else {}
    artifact_location = diagnosis.get("artifact_location") if isinstance(diagnosis.get("artifact_location"), dict) else {}
    return str(
        message_location.get("content")
        or trace_location.get("content")
        or artifact_location.get("content")
        or diagnosis.get("issue")
        or ""
    )


def mast_labels_for_failure(failure_type: str) -> list[str]:
    """Map a fine-grained failure_type to one or more MAST-14 labels.

    Mapping covers 14 failure modes (FM-1.1 through FM-4.2).  Each bucket is
    designed to be mutually exclusive at its primary label; a secondary label is
    added only when strong co-occurrence is expected.
    """
    # FM-1.x: Task / goal understanding errors
    if failure_type in {"problem_understanding_error"}:
        return ["FM-1.1"]  # Task Misunderstanding
    if failure_type in {"seed_answer_copying"}:
        return ["FM-1.2"]  # Goal Drift (reusing prior answer instead of current goal)

    # FM-2.x: Grounding / world-knowledge / tool errors
    if failure_type in {"unsupported_assumption"}:
        return ["FM-2.1"]  # Capability Overestimation (assumes missing info exists)
    if failure_type in {"external_fact_or_calculation_error", "unit_scale_error"}:
        return ["FM-2.2"]  # Grounding Error
    if failure_type in {"distractor_used"}:
        return ["FM-2.3"]  # Context Confusion

    # FM-3.x: Reasoning / planning errors
    if failure_type in {"wrong_operation_direction", "reversing_operation_error"}:
        return ["FM-3.1"]  # Planning Error (sets up wrong algebraic structure)
    if failure_type in {
        "wrong_arithmetic_or_reasoning",
        "missed_added_operation",
        "numerical_substitution_error",
        "number_format_conversion_error",
        "digit_expansion_error",
        "wrong_choice_answer",
        "wrong_final_answer",
    }:
        return ["FM-3.2"]  # Reasoning / Arithmetic Error
    if failure_type in {
        "invalid_choice_format",
        "missing_choice_answer",
        "answer_format_alias_error",
    }:
        return ["FM-3.3"]  # Output Format Error

    # FM-4.x: Tool / execution errors
    if "tool" in failure_type or "execute" in failure_type:
        return ["FM-4.1"]  # Tool Execution Error
    if "ground" in failure_type:
        return ["FM-4.2"]  # Grounding / retrieval failure

    # Fallback: generic reasoning error
    return ["FM-3.2"]


def build_v4_patch(
    sample_id: str,
    diagnosis: dict[str, Any],
    patch_payload: dict[str, Any],
    localization: dict[str, Any],
    sample_dir: str,
) -> dict[str, Any]:
    step_modification = patch_payload.get("step_modification") if isinstance(patch_payload.get("step_modification"), dict) else {}
    strategy = str(step_modification.get("strategy") or diagnosis.get("repair_strategy") or "recompute_and_verify")
    instruction = str(step_modification.get("instruction") or patch_payload.get("attention_instruction") or "")
    return {
        "patch_id": f"patch:{sample_id}:001",
        "operator_id": patch_payload.get("operator_id") or ANSWER_BLIND_OPERATOR_ID,
        "answer_blind": patch_payload.get("answer_blind") is True,
        "patch_family": patch_family_for_failure(str(diagnosis.get("failure_type") or "")),
        "patch_strategy": strategy,
        "patch_target": {
            "mas_agent_id": localization.get("suspect_mas_agent_id"),
            "mas_agent_role": localization.get("suspect_mas_agent_role"),
            "scope": "failed_step",
            "target_event_id": localization.get("first_bad_event_id"),
        },
        "patch_timing": "before_finalize",
        "intervention_level": "repair",
        "operation": str(step_modification.get("operation") or "replace_failed_step"),
        "patch_instruction": instruction,
        "expected_change": str(
            step_modification.get("expected_change")
            or "Replace the failed step with a recomputed answer-producing step."
        ),
        "prompt_ref": f"{sample_dir}/repair/prompt.txt",
        "patch_ref": f"{sample_dir}/repair/patch.json",
        "candidate_index": 0,
    }


def extract_failure_window(
    trace_events: list[dict[str, Any]],
    diagnosis: dict[str, Any],
    window_size: int = 2,
) -> list[dict[str, Any]]:
    """Return a slice of trace events centred on the first bad event.

    The window is [first_bad_index - window_size, first_bad_index + window_size],
    clamped to the available range.  Falls back to the last ``window_size * 2``
    events when no event index is available.
    """
    if not trace_events:
        return []
    trace_location = diagnosis.get("trace_location") if isinstance(diagnosis.get("trace_location"), dict) else {}
    event_index = trace_location.get("event_index")
    if not isinstance(event_index, int):
        start = max(0, len(trace_events) - window_size * 2)
        return trace_events[start:]
    start = max(0, event_index - window_size)
    end = min(len(trace_events), event_index + window_size + 1)
    return trace_events[start:end]


def patch_family_for_failure(failure_type: str) -> str:
    if failure_type in {"unsupported_assumption"}:
        return "contract_patch"
    if failure_type in {"external_fact_or_calculation_error", "answer_format_alias_error"}:
        return "grounding_patch"
    if failure_type in {"invalid_choice_format", "missing_choice_answer"}:
        return "guard_patch"
    return "verification_patch"


def build_v4_repaired_run(
    repair: dict[str, Any],
    sample_dir: str,
    repaired_eval: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if repair.get("enabled") is not True:
        return None
    trajectory = repair.get("trajectory") if isinstance(repair.get("trajectory"), dict) else {}
    return {
        "enabled": True,
        "status": repair.get("status"),
        "trajectory_ref": f"{sample_dir}/repaired/trajectory.jsonl",
        "messages_ref": f"{sample_dir}/repaired/messages.json",
        "result_ref": f"{sample_dir}/repaired/result.json",
        "log_ref": f"{sample_dir}/repaired/run.log",
        "final_output": trajectory.get("final_output"),
        "evaluation": repaired_eval,
    }


def build_v4_control_run(
    repair: dict[str, Any],
    sample_dir: str,
    control_eval: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if repair.get("enabled") is not True:
        return None
    control = repair.get("control") if isinstance(repair.get("control"), dict) else {}
    trajectory = control.get("trajectory") if isinstance(control.get("trajectory"), dict) else {}
    return {
        "enabled": True,
        "status": control.get("status"),
        "trajectory_ref": f"{sample_dir}/control/trajectory.jsonl",
        "messages_ref": f"{sample_dir}/control/messages.json",
        "result_ref": f"{sample_dir}/control/result.json",
        "log_ref": f"{sample_dir}/control/run.log",
        "final_output": trajectory.get("final_output"),
        "evaluation": control_eval,
    }


def answer_format_for_benchmark(benchmark_id: str) -> str:
    if benchmark_id == "gsm-plus":
        return "number"
    if benchmark_id == "mmlu":
        return "choice_letter"
    if benchmark_id == "olympiadbench":
        return "number_or_expression"
    if benchmark_id in {"humaneval", "programdev", "programdev-v2"}:
        return "artifact"
    if benchmark_id in {"gaia", "swe-bench-lite", "appworld-test-c"}:
        return "text"
    return "text"


def score_delta(failed_eval: dict[str, Any], repaired_eval: dict[str, Any] | None) -> float | None:
    if repaired_eval is None:
        return None
    failed_score = failed_eval.get("score")
    repaired_score = repaired_eval.get("score")
    if isinstance(failed_score, (int, float)) and isinstance(repaired_score, (int, float)):
        return float(repaired_score) - float(failed_score)
    return None


def write_v4_sample_files(
    record_root: Path,
    source: dict[str, Any],
    diagnosis: dict[str, Any],
    repair: dict[str, Any],
    record: dict[str, Any],
) -> None:
    repair_root = record_root / "repair"
    repair_root.mkdir(parents=True, exist_ok=True)
    (record_root / "sample.json").write_text(json.dumps(record["sample"], indent=2, ensure_ascii=False) + "\n")
    (record_root / "labels.json").write_text(
        json.dumps(
            {
                "localization": record["localization"],
                "diagnosis": diagnosis,
                "quality": record["quality"],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    (repair_root / "patch.json").write_text(json.dumps(record["patch"], indent=2, ensure_ascii=False) + "\n")
    prompt_text = str((repair.get("patch") or {}).get("prompt_text") or (repair.get("patch") or {}).get("patched_task") or "")
    (repair_root / "prompt.txt").write_text(prompt_text + ("\n" if prompt_text and not prompt_text.endswith("\n") else ""))
    failure_window = extract_failure_window(source.get("trace_events") or [], diagnosis)
    atomic_write_jsonl(repair_root / "failure_window.jsonl", failure_window)
    atomic_write_jsonl(repair_root / "attempts.jsonl", repair.get("attempts") or [])
    atomic_write_jsonl(
        repair_root / "candidate_patches.jsonl",
        [
            {
                "candidate_index": 0,
                "patch": record["patch"],
                "legacy_patch": repair.get("patch"),
            }
        ],
    )
    (repair_root / "policy_decision.json").write_text(
        json.dumps(
            {
                "intervention_level": record["patch"]["intervention_level"],
                "repair_attempted": record["outcome"]["repair_attempted"],
                "reason": "failure row selected from source batch error trajectories",
                "source_result_path": str(source["result_path"]),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


def strip_large_sample_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key not in {"question", "task", "reference_solution", "seed_solution"}
    }


def build_runtime_context(result: dict[str, Any], config: Any | None) -> dict[str, Any]:
    api = getattr(config, "api", None) if config is not None else None
    mas_id = str(result.get("mas") or "unknown-mas")
    benchmark_id = str(result.get("benchmark") or "unknown-benchmark")
    runtime = runtime_state_from_result(result, api)
    return {
        "runtime_id": runtime["runtime_id"],
        "mas_id": mas_id,
        "benchmark_id": benchmark_id,
        "host_adapter": host_adapter_id(result, mas_id),
        "native_repo_path": runtime["native_repo_path"],
        "native_repo_head": runtime["native_repo_head"],
        "provider": runtime["provider"],
        "protocol": runtime["protocol"],
        "base_url": runtime["base_url"],
        "endpoint_url": runtime["endpoint_url"],
        "api_key_env": runtime["api_key_env"],
        "api_key_required": runtime["api_key_required"],
        "model_id": runtime["model_id"],
        "provider_model_name": runtime["model"],
        "generation": runtime["generation"],
    }


def compact_result_for_dataset(result: dict[str, Any]) -> dict[str, Any]:
    result = sanitize_result(result)
    keep = {
        "run_id",
        "sample_id",
        "sample_index",
        "sample_trace_id",
        "status",
        "error",
        "evaluation",
        "final_output",
        "final_answer",
        "model",
        "provider",
        "runtime_id",
        "model_id",
    }
    return {key: result.get(key) for key in sorted(keep) if key in result}
