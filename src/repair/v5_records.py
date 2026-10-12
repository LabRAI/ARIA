"""ARIA repair-dataset-v6 records and host-derived verification fields."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from aria.records import ATTRIBUTION_VERSION, RepairRecord, TrajectoryEvent
from common.artifacts import atomic_write_json, atomic_write_jsonl, utc_now
from common.runtime_support import safe_name
from repair.generator import (
    TRAJECTORY_REPLAY_OPERATOR_ID,
    GeneratedRepairEpisode,
    build_incident,
    materialize_continuation_task,
)
from repair.records import (
    answer_format_for_benchmark,
    build_runtime_context,
    host_adapter_id,
    normalize_v4_evaluation,
    paired_outcome,
    score_delta,
    strip_large_sample_metadata,
)
from repair.schema import REPAIR_SCHEMA_VERSION


def build_v5_dataset_record(
    *,
    name: str,
    spec: dict[str, Any],
    config: Any,
    source_batch_dir: Path,
    source: dict[str, Any],
    episode: GeneratedRepairEpisode,
    execution: dict[str, Any],
) -> dict[str, Any]:
    result = _mapping(source.get("result"))
    metadata = _mapping(result.get("sample_metadata"))
    evaluation = _mapping(result.get("evaluation"))
    mas_id = str(result.get("mas") or metadata.get("mas") or "unknown-mas")
    benchmark_id = str(
        result.get("benchmark") or metadata.get("benchmark") or "unknown-benchmark"
    )
    sample_id = str(
        result.get("sample_id")
        or metadata.get("sample_id")
        or f"sample-{time.time_ns()}"
    )
    sample_dir = f"samples/{safe_name(sample_id)}"
    runtime = build_runtime_context(dict(result), config)
    repair_attempted = execution.get("enabled") is True
    request_stages = []
    if episode.generation_source == "model":
        for stage, candidate_getter in (
            ("localization", lambda trial: trial.localization_candidate),
            ("attribution", lambda trial: trial.attribution_candidate),
            ("patch_generation", lambda trial: trial.patch_candidate),
        ):
            if any(candidate_getter(trial).generation_attempts for trial in episode.trials):
                request_stages.append(stage)
    if repair_attempted:
        request_stages.extend(("control", "patched_trials"))
    repair_endpoint = _repair_endpoint(config, request_stages=request_stages)
    control = _mapping(execution.get("control"))
    control_eval = (
        normalize_v4_evaluation(_mapping(control.get("evaluation")), benchmark_id)
        if repair_attempted
        else None
    )
    control_success = (
        control_eval.get("success") is True if isinstance(control_eval, dict) else None
    )
    trial_records = _trial_records(
        episode,
        execution,
        control_success=control_success,
        benchmark_id=benchmark_id,
        sample_dir=sample_dir,
    )
    selected = _select_trial(trial_records, execution)
    paired = selected.get("paired_outcome") if selected else None
    repaired_eval = selected.get("evaluation") if selected else None
    verified_successful = bool(selected and selected.get("verified_successful") is True)
    localization_payload = dict(_mapping(selected.get("localization"))) if selected else None
    attribution_payload = {
        "attribution_version": ATTRIBUTION_VERSION,
        **dict(_mapping(selected.get("attribution"))),
        "label_source": _label_source(
            episode,
            verified_successful,
            execution_mode=execution.get("execution_mode"),
        ),
        "trial_id": selected.get("trial_id") if selected else None,
    }
    source_eval = normalize_v4_evaluation(dict(evaluation), benchmark_id)
    teacher_model_id = (
        spec.get("teacher_model_id")
        or repair_endpoint.get("model_id")
        or runtime.get("model_id")
    )
    record_type = _record_type(repair_attempted, verified_successful)
    return {
        "schema_version": REPAIR_SCHEMA_VERSION,
        "record_id": f"aria-{safe_name(benchmark_id)}-{safe_name(mas_id)}-{safe_name(sample_id)}-v3",
        "dataset_id": name,
        "sample_id": sample_id,
        "task_id": str(metadata.get("task_id") or sample_id),
        "trajectory_id": str(
            result.get("run_id") or f"{mas_id}-{benchmark_id}-{sample_id}"
        ),
        "split": str(spec.get("split") or spec.get("dataset_split") or "train"),
        "record_type": record_type,
        "source_refs": {
            "mas_id": mas_id,
            "benchmark_id": benchmark_id,
            "runtime_id": runtime["runtime_id"],
            "model_id": runtime["model_id"],
            "teacher_model_id": teacher_model_id,
            "repair_operator_id": spec.get("repair_operator_id")
            or TRAJECTORY_REPLAY_OPERATOR_ID,
            "config_id": str(spec.get("config_id") or f"config/{safe_name(name)}"),
            "host_adapter_id": host_adapter_id(dict(result), mas_id),
            "source_batch_dir": str(source_batch_dir),
            "sample_dir": sample_dir,
            "failed_dir": f"{sample_dir}/failed",
            "control_dir": f"{sample_dir}/control" if repair_attempted else None,
            "repaired_dir": f"{sample_dir}/repaired" if repair_attempted else None,
        },
        "repair_endpoint": repair_endpoint,
        "sample": {
            "benchmark": benchmark_id,
            "benchmark_split": metadata.get("split") or spec.get("sample_split"),
            "input_format": "text",
            "task_text": episode.incident.constraints.get("task_text"),
            "reference": {
                "answer": evaluation.get(
                    "reference_answer", metadata.get("reference_answer")
                ),
                "answer_format": answer_format_for_benchmark(benchmark_id),
                "evaluation_method": evaluation.get("method"),
                "hidden_from_model": True,
            },
            "metadata": strip_large_sample_metadata(dict(metadata)),
        },
        "incident": episode.incident.model_input(),
        "failed_run": {
            "status": result.get("status"),
            "trajectory_ref": f"{sample_dir}/failed/trajectory.jsonl",
            "messages_ref": f"{sample_dir}/failed/messages.json",
            "result_ref": f"{sample_dir}/failed/result.json",
            "log_ref": f"{sample_dir}/failed/run.log",
            "event_count": len(episode.incident.events),
            "final_output": result.get("final_output") or result.get("final_answer"),
            "evaluation": source_eval,
        },
        "localization": localization_payload,
        "attribution": attribution_payload,
        "repair_trials": trial_records,
        "selected_repair_trial_id": selected.get("trial_id") if selected else None,
        "patch": selected.get("repair") if selected else None,
        "control_run": _control_run(execution, sample_dir, control_eval),
        "repaired_run": _repaired_run(
            execution, selected, sample_dir, repaired_eval
        ),
        "outcome": {
            "source_success": source_eval.get("success") is True,
            "repair_attempted": repair_attempted,
            "repair_success": (
                repaired_eval.get("success") is True
                if isinstance(repaired_eval, dict)
                else False
            ),
            "control_success": control_success,
            "paired_outcome": paired,
            "verified_successful": verified_successful,
            "attempt_count": len(trial_records),
            "delta_score": score_delta(source_eval, repaired_eval),
            "execution_mode": execution.get("execution_mode"),
        },
        "training": _training_disposition(
            episode,
            selected=selected,
            trials=trial_records,
            source_eval=source_eval,
            repair_attempted=repair_attempted,
        ),
        "quality": {
            "natural_or_synthetic": str(
                spec.get("natural_or_synthetic", "natural")
            ),
            "dedup_group": sample_id,
            "annotation_status": str(
                spec.get("annotation_status", "teacher_labeled")
            ),
            "audit_status": str(spec.get("audit_status", "not_audited")),
            "created_at": utc_now(),
        },
    }


def build_v5_gate_success_record(
    *,
    name: str,
    spec: dict[str, Any],
    config: Any,
    source_batch_dir: Path,
    source: dict[str, Any],
) -> dict[str, Any]:
    incident = build_incident(
        source,
        max_events=int(spec.get("attribution_max_events", 64)),
        failed=False,
    )
    result = _mapping(source.get("result"))
    metadata = _mapping(result.get("sample_metadata"))
    evaluation = _mapping(result.get("evaluation"))
    mas_id = str(result.get("mas") or "unknown-mas")
    benchmark_id = str(result.get("benchmark") or metadata.get("benchmark") or "")
    sample_id = str(result.get("sample_id") or metadata.get("sample_id") or "sample")
    runtime = build_runtime_context(dict(result), config)
    sample_dir = f"samples/{safe_name(sample_id)}"
    return {
        "schema_version": REPAIR_SCHEMA_VERSION,
        "record_id": f"aria-{safe_name(benchmark_id)}-{safe_name(mas_id)}-{safe_name(sample_id)}-gate-success-v3",
        "dataset_id": name,
        "sample_id": sample_id,
        "task_id": str(metadata.get("task_id") or sample_id),
        "trajectory_id": str(result.get("run_id") or sample_id),
        "split": str(spec.get("split") or "train"),
        "record_type": "gate_success_sample",
        "source_refs": {
            "mas_id": mas_id,
            "benchmark_id": benchmark_id,
            "runtime_id": runtime["runtime_id"],
            "model_id": runtime["model_id"],
            "teacher_model_id": None,
            "repair_operator_id": "noop",
            "config_id": str(spec.get("config_id") or f"config/{safe_name(name)}"),
            "host_adapter_id": host_adapter_id(dict(result), mas_id),
            "source_batch_dir": str(source_batch_dir),
            "sample_dir": sample_dir,
            "failed_dir": None,
            "control_dir": None,
            "repaired_dir": None,
        },
        "repair_endpoint": _repair_endpoint(config, request_stages=[]),
        "sample": {
            "benchmark": benchmark_id,
            "benchmark_split": metadata.get("split") or spec.get("sample_split"),
            "input_format": "text",
            "task_text": incident.constraints.get("task_text"),
            "reference": {
                "answer": evaluation.get(
                    "reference_answer", metadata.get("reference_answer")
                ),
                "answer_format": answer_format_for_benchmark(benchmark_id),
                "evaluation_method": evaluation.get("method"),
                "hidden_from_model": True,
            },
            "metadata": strip_large_sample_metadata(dict(metadata)),
        },
        "incident": incident.model_input(),
        "failed_run": None,
        "localization": None,
        "attribution": None,
        "repair_trials": [],
        "selected_repair_trial_id": None,
        "patch": None,
        "control_run": None,
        "repaired_run": None,
        "success_run": {
            "status": result.get("status"),
            "trajectory_ref": f"{sample_dir}/success/trajectory.jsonl",
            "messages_ref": f"{sample_dir}/success/messages.json",
            "result_ref": f"{sample_dir}/success/result.json",
            "final_output": result.get("final_output") or result.get("final_answer"),
            "evaluation": normalize_v4_evaluation(dict(evaluation), benchmark_id),
        },
        "outcome": {
            "source_success": True,
            "repair_attempted": False,
            "repair_success": False,
            "control_success": None,
            "paired_outcome": None,
            "verified_successful": False,
            "attempt_count": 0,
            "delta_score": None,
            "execution_mode": None,
        },
        "training": {
            "gate_sft_eligible": True,
            "localization_sft_eligible": False,
            "attribution_sft_eligible": False,
            "structured_repair_sft_eligible": False,
            "repair_dpo_eligible": False,
            "reason": "gate_success_sample_negative",
        },
        "quality": {
            "natural_or_synthetic": str(
                spec.get("natural_or_synthetic", "natural")
            ),
            "dedup_group": sample_id,
            "annotation_status": "host_verified",
            "audit_status": str(spec.get("audit_status", "not_audited")),
            "created_at": utc_now(),
        },
    }


def write_v5_sample_files(
    record_root: Path,
    source: dict[str, Any],
    episode: GeneratedRepairEpisode,
    execution: dict[str, Any],
    record: dict[str, Any],
) -> None:
    repair_root = record_root / "repair"
    repair_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(record_root / "sample.json", record["sample"])
    atomic_write_json(repair_root / "incident.json", record["incident"])
    atomic_write_json(
        record_root / "labels.json",
        {
            "localization": record["localization"],
            "attribution": record["attribution"],
            "quality": record["quality"],
        },
    )
    atomic_write_jsonl(
        repair_root / "repair_trials.jsonl",
        record["repair_trials"],
    )
    atomic_write_json(
        repair_root / "patch.json",
        record["patch"],
    )
    atomic_write_json(
        repair_root / "policy_decision.json",
        {
            "localization": record["localization"],
            "attribution": record["attribution"],
            "selected_repair_trial_id": record["selected_repair_trial_id"],
            "execution_mode": record["outcome"]["execution_mode"],
            "training": record["training"],
            "source_result_path": str(source.get("result_path") or ""),
        },
    )
    atomic_write_jsonl(
        repair_root / "attempts.jsonl", execution.get("attempts") or []
    )


def _trial_records(
    episode: GeneratedRepairEpisode,
    execution: Mapping[str, Any],
    *,
    control_success: bool | None,
    benchmark_id: str,
    sample_dir: str,
) -> list[dict[str, Any]]:
    executed = {
        str(item.get("trial_id")): item
        for item in execution.get("trials", [])
        if isinstance(item, Mapping) and item.get("trial_id")
    }
    records: list[dict[str, Any]] = []
    for trial in episode.trials:
        repair = trial.repair
        attribution = trial.attribution
        localization = trial.localization
        run = _mapping(executed.get(trial.trial_id))
        evaluation = (
            normalize_v4_evaluation(_mapping(run.get("evaluation")), benchmark_id)
            if run
            else None
        )
        patched_success = (
            evaluation.get("success") is True if isinstance(evaluation, dict) else None
        )
        paired = (
            paired_outcome(bool(control_success), bool(patched_success))
            if isinstance(control_success, bool) and isinstance(patched_success, bool)
            else None
        )
        materialization = _mapping(run.get("materialization"))
        expected_materialization = (
            materialize_continuation_task(
                episode.incident,
                attribution,
                repair,
            )
            if repair is not None and attribution is not None
            else None
        )
        materialization_verified, evidence = _verify_materialization(
            repair,
            run,
            expected_materialization=expected_materialization,
        )
        records.append(
            {
                "trial_id": trial.trial_id,
                "trial_index": trial.trial_index,
                "generation": trial.to_dict(),
                "localization": localization.to_dict()
                if localization is not None
                else None,
                "attribution": attribution.model_output()
                if attribution is not None
                else None,
                "repair": repair.to_dict() if repair is not None else None,
                "execution": {
                    "attempted": bool(run),
                    "run_id": run.get("run_id"),
                    "status": run.get("status"),
                    "execution_mode": materialization.get("execution_mode"),
                    "materialization_verified": materialization_verified,
                    "materialization_evidence": evidence,
                    "trajectory_ref": (
                        f"{sample_dir}/trials/trial-{trial.trial_index + 1:02d}/trajectory.jsonl"
                        if run
                        else None
                    ),
                    "final_output": _mapping(
                        _mapping(run.get("trajectory")).get("trajectory")
                    ).get("final_output"),
                },
                "evaluation": evaluation,
                "paired_outcome": paired,
                "verified_successful": bool(
                    paired == "patched_only" and materialization_verified
                ),
            }
        )
    return records


def _verify_materialization(
    repair: RepairRecord | None,
    run: Mapping[str, Any],
    *,
    expected_materialization: Mapping[str, Any] | None,
) -> tuple[bool, list[dict[str, Any]]]:
    if repair is None or not run:
        return False, []
    materialization = _mapping(run.get("materialization"))
    if materialization.get("repair") != repair.to_dict():
        return False, [{"kind": "repair_materialization_mismatch"}]
    if (
        expected_materialization is None
        or dict(materialization) != dict(expected_materialization)
    ):
        return False, [{"kind": "materialization_payload_mismatch"}]
    patched_task = materialization.get("patched_task")
    if not isinstance(patched_task, str) or not patched_task.strip():
        return False, [{"kind": "missing_patched_task"}]
    evidence: list[dict[str, Any]] = [{"kind": "materialization_payload_matched"}]
    nested_payload = _mapping(run.get("trajectory"))
    result = _mapping(nested_payload.get("result"))
    trajectory = _mapping(nested_payload.get("trajectory"))
    trace_events = trajectory.get("trace_events")
    if not isinstance(trace_events, list):
        trace_events = []
    used_tool_call_indexes: set[int] = set()
    for event in repair.replacement_events:
        for tool_call in event.tool_calls:
            observed = _find_tool_execution(
                trace_events,
                tool_call,
                used_call_indexes=used_tool_call_indexes,
            )
            if observed is None:
                name, _ = _tool_call_signature(tool_call)
                return False, [
                    {
                        "kind": "missing_tool_execution",
                        "tool": name or None,
                        "replacement_event_id": event.event_id,
                    }
                ]
            call_index, result_index = observed
            used_tool_call_indexes.add(call_index)
            evidence.append(
                {
                    "kind": "tool_execution_observed",
                    "tool": _tool_call_signature(tool_call)[0],
                    "replacement_event_id": event.event_id,
                    "tool_call_event_id": _mapping(trace_events[call_index]).get(
                        "event_id"
                    ),
                    "tool_result_event_id": _mapping(trace_events[result_index]).get(
                        "event_id"
                    ),
                }
            )
        if event.kind == "workspace_edit":
            workspace_evidence = _workspace_materialization_evidence(
                nested_payload,
                result,
            )
            if workspace_evidence is None:
                return False, [
                    {
                        "kind": "missing_workspace_diff",
                        "replacement_event_id": event.event_id,
                    }
                ]
            evidence.append(
                {
                    "kind": "workspace_diff_observed",
                    "replacement_event_id": event.event_id,
                    **workspace_evidence,
                }
            )
    if len(evidence) == 1:
        evidence.append({"kind": "message_continuation_completed"})
    return True, evidence


def _find_tool_execution(
    trace_events: Sequence[Any],
    proposed_call: Mapping[str, Any],
    *,
    used_call_indexes: set[int],
) -> tuple[int, int] | None:
    proposed_name, proposed_arguments = _tool_call_signature(proposed_call)
    if not proposed_name:
        return None
    for call_index, raw_event in enumerate(trace_events):
        if call_index in used_call_indexes or not isinstance(raw_event, Mapping):
            continue
        observed_calls = raw_event.get("tool_calls")
        if not isinstance(observed_calls, list):
            continue
        matched = any(
            isinstance(observed_call, Mapping)
            and _tool_call_signature(observed_call)
            == (proposed_name, proposed_arguments)
            for observed_call in observed_calls
        )
        if not matched:
            continue
        result_index = _find_tool_result_event(
            trace_events,
            call_index=call_index,
            tool_name=proposed_name,
        )
        if result_index is not None:
            return call_index, result_index
    return None


def _find_tool_result_event(
    trace_events: Sequence[Any],
    *,
    call_index: int,
    tool_name: str,
) -> int | None:
    for result_index in range(call_index, len(trace_events)):
        raw_event = trace_events[result_index]
        if not isinstance(raw_event, Mapping):
            continue
        tool_results = raw_event.get("tool_results")
        if isinstance(tool_results, list) and tool_results:
            return result_index
        if result_index == call_index:
            continue
        observed_calls = raw_event.get("tool_calls")
        if isinstance(observed_calls, list) and observed_calls:
            return None
        kind = str(raw_event.get("kind") or "").strip().lower()
        content = str(raw_event.get("content") or "").strip()
        lowered = content.lower()
        if content and (
            kind in {"tool_result", "tool_response", "after_act", "observation"}
            or (
                tool_name.lower() in lowered
                and any(
                    marker in lowered
                    for marker in ("observed output", "executed", "tool result", "result of")
                )
            )
        ):
            return result_index
    return None


def _tool_call_signature(tool_call: Mapping[str, Any]) -> tuple[str, str]:
    function = _mapping(tool_call.get("function"))
    name = str(tool_call.get("name") or function.get("name") or "").strip()
    arguments = tool_call.get("arguments", function.get("arguments"))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = arguments.strip()
    return name, json.dumps(
        arguments,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _workspace_materialization_evidence(
    nested_payload: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any] | None:
    evaluation = _mapping(nested_payload.get("evaluation"))
    if not evaluation:
        evaluation = _mapping(result.get("evaluation"))
    patch_bytes = evaluation.get("model_patch_bytes")
    changed_files = evaluation.get("changed_files")
    snapshot = _mapping(nested_payload.get("snapshot"))
    files = snapshot.get("files")
    model_patch = None
    if isinstance(files, list):
        model_patch = next(
            (
                item
                for item in files
                if isinstance(item, Mapping) and item.get("name") == "model.patch"
            ),
            None,
        )
    if (
        not isinstance(patch_bytes, int)
        or isinstance(patch_bytes, bool)
        or patch_bytes <= 0
        or not isinstance(changed_files, list)
        or not changed_files
        or not isinstance(model_patch, Mapping)
        or model_patch.get("bytes") != patch_bytes
    ):
        return None
    return {
        "patch_bytes": patch_bytes,
        "changed_files": [str(path) for path in changed_files],
    }


def _select_trial(
    trials: Sequence[dict[str, Any]], execution: Mapping[str, Any]
) -> dict[str, Any] | None:
    for trial in trials:
        if trial.get("verified_successful") is True:
            return trial
    selected_id = execution.get("selected_trial_id")
    for trial in trials:
        if trial.get("trial_id") == selected_id:
            return trial
    return next((trial for trial in trials if trial.get("repair") is not None), None)


def _training_disposition(
    episode: GeneratedRepairEpisode,
    *,
    selected: Mapping[str, Any] | None,
    trials: Sequence[Mapping[str, Any]],
    source_eval: Mapping[str, Any],
    repair_attempted: bool,
) -> dict[str, Any]:
    source_resolved_failure = source_eval.get("success") is False
    model_labeled = episode.generation_source == "model"
    verified = bool(selected and selected.get("verified_successful") is True)
    has_rejected = any(
        trial.get("paired_outcome") in {"both_fail", "control_only"}
        for trial in trials
        if selected is None
        or trial.get("trial_id") != selected.get("trial_id")
    )
    # Observed failed suffix is always available as a schema-matched fallback
    # negative, so verified patches remain DPO-eligible without a second failed
    # model candidate.
    return {
        "gate_sft_eligible": source_resolved_failure and model_labeled,
        "localization_sft_eligible": source_resolved_failure and model_labeled,
        "attribution_sft_eligible": source_resolved_failure and model_labeled,
        "structured_repair_sft_eligible": verified and model_labeled,
        "repair_dpo_eligible": verified and model_labeled,
        "verified_repair_trials": sum(
            trial.get("verified_successful") is True for trial in trials
        ),
        "has_executed_rejected_candidate": has_rejected,
        "reason": (
            "heuristic_fallback_excluded"
            if not model_labeled
            else "verified_patched_only"
            if verified
            else "repair_not_replay_verified"
            if repair_attempted
            else "repair_not_executed"
        ),
    }


def _control_run(
    execution: Mapping[str, Any],
    sample_dir: str,
    evaluation: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if execution.get("enabled") is not True:
        return None
    control = _mapping(execution.get("control"))
    trajectory = _mapping(control.get("trajectory"))
    return {
        "status": control.get("status"),
        "trajectory_ref": f"{sample_dir}/control/trajectory.jsonl",
        "messages_ref": f"{sample_dir}/control/messages.json",
        "result_ref": f"{sample_dir}/control/result.json",
        "final_output": trajectory.get("final_output"),
        "evaluation": dict(evaluation) if evaluation is not None else None,
    }


def _repaired_run(
    execution: Mapping[str, Any],
    selected: Mapping[str, Any] | None,
    sample_dir: str,
    evaluation: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if execution.get("enabled") is not True:
        return None
    selected_execution = _mapping(selected.get("execution")) if selected else {}
    trajectory = _mapping(execution.get("trajectory"))
    return {
        "status": execution.get("status"),
        "trajectory_ref": selected_execution.get("trajectory_ref")
        or f"{sample_dir}/repaired/trajectory.jsonl",
        "messages_ref": (
            str(selected_execution.get("trajectory_ref")).replace(
                "trajectory.jsonl", "messages.json"
            )
            if selected_execution.get("trajectory_ref")
            else f"{sample_dir}/repaired/messages.json"
        ),
        "result_ref": (
            str(selected_execution.get("trajectory_ref")).replace(
                "trajectory.jsonl", "result.json"
            )
            if selected_execution.get("trajectory_ref")
            else f"{sample_dir}/repaired/result.json"
        ),
        "final_output": selected_execution.get("final_output")
        or trajectory.get("final_output"),
        "evaluation": dict(evaluation) if evaluation is not None else None,
    }


def _label_source(
    episode: GeneratedRepairEpisode,
    verified: bool,
    *,
    execution_mode: Any,
) -> str:
    if episode.generation_source != "model":
        return episode.generation_source
    if not verified:
        return "teacher_plus_host"
    if execution_mode == "non_exact_continuation":
        return "teacher_plus_host_plus_non_exact_continuation"
    return "teacher_plus_host_plus_exact_replay"


def _record_type(repair_attempted: bool, verified_successful: bool) -> str:
    if not repair_attempted:
        return "repair_proposed"
    return (
        "failed_successful_repair"
        if verified_successful
        else "failed_failed_repair"
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _repair_endpoint(
    config: Any,
    *,
    request_stages: Sequence[str],
) -> dict[str, Any]:
    """Return the configured repair endpoint on the individual sample record."""

    api = getattr(config, "api", None)
    return {
        "runtime_id": getattr(api, "runtime_id", None),
        "model_id": getattr(api, "model_id", None),
        "provider": getattr(api, "provider", None),
        "protocol": getattr(api, "protocol", None),
        "base_url": getattr(api, "base_url", None),
        "endpoint_url": getattr(api, "endpoint_url", None),
        "model": getattr(api, "model", None),
        "reasoning_effort": getattr(api, "reasoning_effort", None),
        "temperature": getattr(api, "temperature", None),
        "max_tokens": getattr(api, "max_tokens", None),
        "request_stages": list(request_stages),
        "used": bool(request_stages),
    }
