"""Paired unpatched/control and patched repair execution."""

from __future__ import annotations

import logging
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from common.artifacts import atomic_write_json, sample_metadata
from common.catalog import find_benchmark, find_mas
from common.evaluation import (
    evaluate_sample as evaluate_benchmark_sample,
)
from common.evaluation import (
    validate_benchmark_evaluator,
)
from common.failure_classification import (
    classify_infrastructure_failure,
    stable_infrastructure_reason,
)
from common.host_adapter import get_host_adapter
from common.preparation import git_head
from common.runtime_support import safe_name
from inference.native import NativeRunContext, get_native_mas_backend
from repair.artifacts import (
    native_messages_path,
    native_trace_path,
    snapshot_run_artifacts,
    write_standard_repair_run_artifacts,
)
from repair.operator import build_original_problem
from repair.records import compact_result_for_dataset
from repair.utils import load_json, load_jsonl


class RepairSampleInvalid(RuntimeError):
    """Expected sample-level infrastructure failure that is safe to continue past."""

    def __init__(
        self,
        *,
        sample_id: str,
        run_id: str,
        variant: str,
        stage: str,
        category: str,
        reason: str,
        output_dir: Path,
    ) -> None:
        self.sample_id = sample_id
        self.run_id = run_id
        self.variant = variant
        self.stage = stage
        self.category = category
        self.reason = reason
        self.output_dir = output_dir
        failure = "evaluator is unresolved" if stage == "evaluation" else "native rerun failed"
        super().__init__(
            f"repair sample invalid: sample={sample_id} variant={variant} "
            f"stage={stage} category={category}; {failure}: {reason}"
        )


def run_repaired_sample(
    *,
    config: Any,
    spec: dict[str, Any],
    source: dict[str, Any],
    repair_prompt: dict[str, Any] | None,
    repair_output_root: Path,
    dataset_sample_root: Path,
    repair_trials: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run one control and one independently materialized run per repair trial."""

    source_result = source["result"]
    mas_slug = str(source_result.get("mas") or "")
    benchmark_slug = str(source_result.get("benchmark") or "")
    ensure_supported_repair_rerun(mas_slug, benchmark_slug)
    control_sample = build_control_sample(source_result)
    repo_root = Path(__file__).resolve().parents[2]
    clone_path = resolve_repo_path(config, "external_dir") / "mas" / find_mas(mas_slug).clone_dir
    logger = logging.getLogger("masfm.repair_dataset")
    sample_root = repair_output_root / "samples" / safe_name(
        str(source_result["sample_id"])
    )
    if sample_root.exists():
        logger.info(
            "Resetting unfinished repair reruns for sample id=%s",
            source_result["sample_id"],
        )
        shutil.rmtree(sample_root)

    control_root = sample_root / "control" / "attempt-01"
    control_result = _execute_rerun_variant(
        config=config,
        spec=spec,
        source_result=source_result,
        sample=control_sample,
        mas_slug=mas_slug,
        benchmark_slug=benchmark_slug,
        repo_root=repo_root,
        clone_path=clone_path,
        run_root=control_root,
        run_id=f"{spec.get('name', 'repair')}-{control_sample['sample_id']}-control-01",
        variant="control",
    )
    control_payload = _rerun_payload(
        control_result,
        control_root,
        mas_slug,
        snapshot_run_artifacts(control_root, dataset_sample_root / "control"),
    )

    trials = list(repair_trials or [])
    if not trials:
        if not isinstance(repair_prompt, dict):
            raise ValueError("repair rerun requires repair_prompt or repair_trials")
        trials = [
            {
                "trial_id": "legacy:00",
                "trial_index": 0,
                "materialization": repair_prompt,
                "repair": None,
            }
        ]
    if not trials:
        raise ValueError("repair rerun has no executable repair trial")
    attempt_results: list[dict[str, Any]] = []
    trial_payloads: list[dict[str, Any]] = []
    selected_result: dict[str, Any] | None = None
    selected_root: Path | None = None
    selected_trial: dict[str, Any] | None = None
    first_result: tuple[dict[str, Any], Path, dict[str, Any]] | None = None
    for attempt, trial in enumerate(trials, start=1):
        materialization = trial.get("materialization")
        if not isinstance(materialization, dict):
            raise TypeError(f"repair trial {attempt} materialization must be a mapping")
        patched_task = materialization.get("patched_task")
        if not isinstance(patched_task, str) or not patched_task.strip():
            raise ValueError(f"repair trial {attempt} has no patched_task")
        patched_sample = build_repair_sample(source_result, patched_task)
        trial_id = str(
            trial.get("trial_id") or f"repair-trial:{attempt - 1:02d}"
        )
        trial_index = int(trial.get("trial_index", attempt - 1))
        run_root = sample_root / "patched" / f"trial-{attempt:02d}"
        result = _execute_rerun_variant(
            config=config,
            spec=spec,
            source_result=source_result,
            sample=patched_sample,
            mas_slug=mas_slug,
            benchmark_slug=benchmark_slug,
            repo_root=repo_root,
            clone_path=clone_path,
            run_root=run_root,
            run_id=(
                f"{spec.get('name', 'repair')}-{patched_sample['sample_id']}"
                f"-patched-{attempt:02d}"
            ),
            variant="patched",
        )
        trial_snapshot = snapshot_run_artifacts(
            run_root,
            dataset_sample_root / "trials" / f"trial-{attempt:02d}",
        )
        trial_result = {
            "attempt": attempt,
            "trial_id": trial_id,
            "trial_index": trial_index,
            "run_id": result.get("run_id"),
            "output_dir": str(run_root),
            "status": result.get("status"),
            "evaluation": result.get("evaluation"),
            "localization": trial.get("localization"),
            "attribution": trial.get("attribution"),
            "repair": trial.get("repair"),
            "materialization": materialization,
            "snapshot": trial_snapshot,
            "trajectory": _rerun_payload(
                result,
                run_root,
                mas_slug,
                trial_snapshot,
            ),
        }
        attempt_results.append(
            {
                key: trial_result[key]
                for key in (
                    "attempt",
                    "trial_id",
                    "trial_index",
                    "run_id",
                    "output_dir",
                    "status",
                    "evaluation",
                )
            }
        )
        trial_payloads.append(trial_result)
        if first_result is None:
            first_result = (result, run_root, trial)
        if (
            selected_result is None
            and (result.get("evaluation") or {}).get("correct") is True
        ):
            selected_result = result
            selected_root = run_root
            selected_trial = trial_result

    if selected_result is None and first_result is not None:
        selected_result, selected_root, _ = first_result
        selected_trial = trial_payloads[0]
    if selected_result is None or selected_root is None or selected_trial is None:
        raise RuntimeError("patched rerun did not produce a result")
    patched_payload = _rerun_payload(
        selected_result,
        selected_root,
        mas_slug,
        snapshot_run_artifacts(selected_root, dataset_sample_root / "repaired"),
    )
    patched_payload.update(
        {
            "control": control_payload,
            "attempts": attempt_results,
            "trials": trial_payloads,
            "selected_trial_id": selected_trial["trial_id"],
            "selected_trial_index": selected_trial["trial_index"],
            "selected_localization": selected_trial.get("localization"),
            "selected_attribution": selected_trial.get("attribution"),
            "selected_repair": selected_trial.get("repair"),
            "selected_materialization": selected_trial.get("materialization"),
            "primary_evaluation": selected_result.get("evaluation"),
            "primary_success": (selected_result.get("evaluation") or {}).get("correct") is True,
            "execution_mode": str(
                (selected_trial.get("materialization") or {}).get(
                    "execution_mode", "prompt_conditioned_rerun"
                )
            ),
        }
    )
    return patched_payload


def _execute_rerun_variant(
    *,
    config: Any,
    spec: dict[str, Any],
    source_result: dict[str, Any],
    sample: dict[str, Any],
    mas_slug: str,
    benchmark_slug: str,
    repo_root: Path,
    clone_path: Path,
    run_root: Path,
    run_id: str,
    variant: str,
) -> dict[str, Any]:
    result_path = run_root / "result.json"
    run_root.mkdir(parents=True, exist_ok=True)
    try:
        native = run_native_repair_attempt(
            config=config,
            spec=spec,
            repo_root=repo_root,
            clone_path=clone_path,
            run_root=run_root,
            sample=sample,
            mas_slug=mas_slug,
            benchmark_slug=benchmark_slug,
        )
    except Exception as exc:
        category = classify_infrastructure_failure(exc)
        if category and category != "native_invalid":
            raise _sample_invalid(
                sample=sample,
                run_id=run_id,
                variant=variant,
                stage="native",
                error=exc,
                run_root=run_root,
                category=category,
            ) from exc
        raise
    if native.status != "completed":
        raise _sample_invalid(
            sample=sample,
            run_id=run_id,
            variant=variant,
            stage="native",
            error=native.error or native.status,
            run_root=run_root,
        )
    result = build_repair_run_result(
        config=config,
        source_result=source_result,
        sample=sample,
        run_id=run_id,
        run_root=run_root,
        clone_path=clone_path,
        native=native,
        mas_slug=mas_slug,
        benchmark_slug=benchmark_slug,
    )
    try:
        result["evaluation"] = evaluate_benchmark_sample(
            benchmark_slug=benchmark_slug,
            sample=sample,
            result=result,
            run_root=run_root,
            evaluation_config=spec.get("evaluation") or {},
        )
    except Exception as exc:
        category = classify_infrastructure_failure(exc)
        if category and category != "native_invalid":
            reason = stable_infrastructure_reason(exc, category)
            result["evaluation"] = {
                "benchmark": benchmark_slug,
                "score": None,
                "correct": None,
                "method": "evaluation_error",
                "reason": reason,
            }
            write_standard_repair_run_artifacts(run_root, result, mas_slug)
            atomic_write_json(result_path, result)
            raise _sample_invalid(
                sample=sample,
                run_id=run_id,
                variant=variant,
                stage="evaluation",
                error=reason,
                run_root=run_root,
                category=category,
            ) from exc
        raise
    write_standard_repair_run_artifacts(run_root, result, mas_slug)
    atomic_write_json(result_path, result)
    if result.get("status") != "completed":
        raise _sample_invalid(
            sample=sample,
            run_id=run_id,
            variant=variant,
            stage="native",
            error=result.get("error") or result.get("status"),
            run_root=run_root,
        )
    require_resolved_evaluation(
        result,
        run_id,
        sample_id=str(sample["sample_id"]),
        variant=variant,
        run_root=run_root,
    )
    return result


def require_resolved_evaluation(
    result: dict[str, Any],
    run_id: str,
    *,
    sample_id: str | None = None,
    variant: str = "unknown",
    run_root: Path | None = None,
) -> None:
    evaluation = result.get("evaluation") if isinstance(result.get("evaluation"), dict) else {}
    if not isinstance(evaluation.get("correct"), bool):
        error = evaluation.get("reason") or evaluation.get("method") or "missing evaluation"
        category = classify_infrastructure_failure(error)
        raise RepairSampleInvalid(
            sample_id=str(sample_id or result.get("sample_id") or "unknown"),
            run_id=run_id,
            variant=variant,
            stage="evaluation",
            category=str(category or "evaluator_unresolved"),
            reason=stable_infrastructure_reason(error, category),
            output_dir=run_root or Path(str(result.get("output_dir") or ".")),
        )


def _sample_invalid(
    *,
    sample: dict[str, Any],
    run_id: str,
    variant: str,
    stage: str,
    error: Any,
    run_root: Path,
    category: str | None = None,
) -> RepairSampleInvalid:
    resolved_category = category or classify_infrastructure_failure(error) or "native_invalid"
    return RepairSampleInvalid(
        sample_id=str(sample.get("sample_id") or "unknown"),
        run_id=run_id,
        variant=variant,
        stage=stage,
        category=resolved_category,
        reason=stable_infrastructure_reason(error, resolved_category),
        output_dir=run_root,
    )


def _rerun_payload(
    result: dict[str, Any],
    run_root: Path,
    mas_slug: str,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    messages_path = native_messages_path(result, run_root, mas_slug)
    trace_path = native_trace_path(result, run_root, mas_slug)
    messages = load_json(messages_path) if messages_path.exists() else []
    trace = load_jsonl(trace_path) if trace_path.exists() else []
    return {
        "status": result.get("status"),
        "success": (result.get("evaluation") or {}).get("correct") is True,
        "run_id": result.get("run_id"),
        "output_dir": str(run_root),
        "api_key_env": result.get("api_key_env"),
        "snapshot": snapshot,
        "evaluation": result.get("evaluation"),
        "result": compact_result_for_dataset(result),
        "trajectory": {
            "messages": messages,
            "trace_events": trace,
            "transcript": result.get("transcript", []),
            "final_output": result.get("final_output") or result.get("final_answer"),
        },
    }


def ensure_supported_repair_rerun(mas_slug: str, benchmark_slug: str) -> None:
    get_native_mas_backend(mas_slug)
    validate_benchmark_evaluator(benchmark_slug)


def run_native_repair_attempt(
    *,
    config: Any,
    spec: dict[str, Any],
    repo_root: Path,
    clone_path: Path,
    run_root: Path,
    sample: dict[str, Any],
    mas_slug: str,
    benchmark_slug: str,
) -> Any:
    mas = find_mas(mas_slug)
    benchmark = find_benchmark(benchmark_slug)
    return get_native_mas_backend(mas_slug).run(
        NativeRunContext(
            repo_root=repo_root,
            clone_path=clone_path,
            run_root=run_root,
            workspace_path=run_root / "workspace",
            sample=sample,
            mas=mas,
            benchmark=benchmark,
            host_adapter=get_host_adapter(mas_slug),
            aria=None,
            options=spec,
            api_settings=api_settings_from_config(config),
            default_timeout_seconds=int(getattr(config.api, "timeout_seconds", 240)),
        )
    )


def build_repair_run_result(
    *,
    config: Any,
    source_result: dict[str, Any],
    sample: dict[str, Any],
    run_id: str,
    run_root: Path,
    clone_path: Path,
    native: Any,
    mas_slug: str,
    benchmark_slug: str,
) -> dict[str, Any]:
    api = getattr(config, "api", None)
    native_backend = get_native_mas_backend(mas_slug)
    return {
        "run_id": run_id,
        "mas": mas_slug,
        "benchmark": benchmark_slug,
        "sample_id": sample.get("sample_id"),
        "sample_index": sample.get("sample_index"),
        "sample_trace_id": source_result.get("sample_trace_id"),
        "sample_metadata": sample_metadata(sample),
        "model": getattr(api, "model", source_result.get("model")),
        "base_url": getattr(api, "base_url", source_result.get("base_url")),
        "provider": getattr(api, "provider", source_result.get("provider", "openai-compatible")),
        "protocol": getattr(api, "protocol", source_result.get("protocol")),
        "endpoint_url": getattr(api, "endpoint_url", source_result.get("endpoint_url")),
        "reasoning_effort": getattr(api, "reasoning_effort", source_result.get("reasoning_effort")),
        "upstream_provider": getattr(
            api,
            "upstream_provider",
            source_result.get("upstream_provider"),
        ),
        "allow_fallbacks": getattr(
            api,
            "allow_fallbacks",
            source_result.get("allow_fallbacks"),
        ),
        "runtime_id": getattr(api, "runtime_id", source_result.get("runtime_id")),
        "model_id": getattr(api, "model_id", source_result.get("model_id")),
        "api_key_env": getattr(api, "api_key_env", source_result.get("api_key_env")),
        "api_key_required": getattr(api, "api_key_required", source_result.get("api_key_required")),
        "native_repo_path": str(clone_path),
        "native_repo_head": git_head(clone_path),
        "status": native.status,
        "output_dir": str(run_root),
        "workspace_path": str(run_root / "workspace"),
        "transcript": native.transcript,
        "final_output": native.final_output,
        "native_messages_path": str(run_root / native_backend.messages_filename),
        "native_trace_path": str(run_root / native_backend.trace_filename),
        "error": native.error,
    }


def build_repair_sample(source_result: dict[str, Any], patched_task: str) -> dict[str, Any]:
    return _base_rerun_sample(source_result, patched_task)


def build_control_sample(source_result: dict[str, Any]) -> dict[str, Any]:
    metadata = source_result.get("sample_metadata") if isinstance(source_result.get("sample_metadata"), dict) else {}
    original_task = build_original_problem(source_result, metadata)
    if not original_task:
        raise ValueError("cannot reconstruct model-visible source task for control rerun")
    return _base_rerun_sample(source_result, original_task)


def _base_rerun_sample(source_result: dict[str, Any], task: str) -> dict[str, Any]:
    metadata = source_result.get("sample_metadata") if isinstance(source_result.get("sample_metadata"), dict) else {}
    evaluation = source_result.get("evaluation") if isinstance(source_result.get("evaluation"), dict) else {}
    sample = dict(metadata)
    sample["task"] = task
    sample["question"] = metadata.get("question") or task
    sample["sample_id"] = source_result.get("sample_id") or metadata.get("sample_id")
    sample["sample_index"] = source_result.get("sample_index", metadata.get("sample_index"))
    sample["reference_answer"] = str(
        evaluation.get("reference_answer", metadata.get("reference_answer", ""))
    )
    return sample


def api_settings_from_config(config: Any) -> dict[str, Any]:
    api = getattr(config, "api", None)
    if api is None:
        raise RuntimeError("repair rerun requires config.api")
    return {
        "base_url": api.base_url,
        "endpoint_url": api.endpoint_url,
        "model": api.model,
        "api_key_env": getattr(api, "api_key_env", "OPENAI_API_KEY"),
        "api_key_required": getattr(api, "api_key_required", True),
        "provider": getattr(api, "provider", "openai-compatible"),
        "protocol": api.protocol,
        "runtime_id": getattr(api, "runtime_id", None),
        "model_id": getattr(api, "model_id", None),
        "max_tokens": getattr(api, "max_tokens", 2048),
        "context_window_tokens": getattr(api, "context_window_tokens", None),
        "temperature": getattr(api, "temperature", 0.0),
        "reasoning_effort": getattr(api, "reasoning_effort", None),
        "timeout_seconds": getattr(api, "timeout_seconds", 240),
        "retries": getattr(api, "retries", 4),
        "retry_seconds": getattr(api, "retry_seconds", 5.0),
        "seed": getattr(api, "seed", None),
        "upstream_provider": getattr(api, "upstream_provider", None),
        "allow_fallbacks": getattr(api, "allow_fallbacks", None),
        "anthropic_version": getattr(api, "anthropic_version", "2023-06-01"),
    }


def resolve_repo_path(config: Any, attr: str) -> Path:
    value = getattr(config, attr)
    path = value if isinstance(value, Path) else Path(str(value))
    if path.is_absolute():
        return path
    return (Path(__file__).resolve().parents[2] / path).resolve()
