"""Adapters from ARIA repair datasets and portable JSONL into baseline samples."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from baselines.core.aria_records import adapt_aria_record, is_aria_record
from baselines.core.types import BaselineEvent, BaselineSample

#: ``aria-v6`` is retained as the historical spelling of ``aria-record``.
INPUT_FORMATS = frozenset(
    {
        "auto",
        "aria-record",
        "aria-v6",
        "aria-incident",
        "aria-full",
        "stage3-environment",
        "inference-results",
        "generic-jsonl",
    }
)

ANSWER_FIELDS = frozenset(
    {
        "reference_answer",
        "reference_solution",
        "reference_patch",
        "gold_answer",
        "gold_patch",
        "canonical_solution",
        "expected_answer",
        "ground_truth",
        "answer",
        "test",
        "test_patch",
        "fail_to_pass",
        "pass_to_pass",
    }
)


def load_baseline_samples(
    path: Path,
    *,
    input_format: str = "auto",
    sample_limit: int | None = None,
    dataset_root: Path | None = None,
    max_events: int = 64,
) -> list[BaselineSample]:
    """Load evaluation samples from one JSONL file.

    ``dataset_root`` resolves the artifact references inside ARIA repair-dataset
    records and defaults to the directory holding ``path``, which is where an
    ARIA dataset keeps its ``samples/`` tree.
    """

    if input_format not in INPUT_FORMATS:
        supported = ", ".join(sorted(INPUT_FORMATS))
        raise ValueError(
            f"unsupported baseline input format {input_format!r}; expected one of: {supported}"
        )
    path = resolve_baseline_input_path(path, input_format=input_format)
    root = (dataset_root or path.parent).resolve()
    samples: list[BaselineSample] = []
    for index, row in enumerate(_read_rows(path)):
        detected = detect_format(row) if input_format == "auto" else input_format
        if detected in {"aria-record", "aria-v6"}:
            sample = adapt_aria_record(
                row, index=index, dataset_root=root, max_events=max_events
            )
        elif detected == "aria-incident":
            sample = adapt_aria_incident(row, index=index)
        elif detected in {"aria-full", "stage3-environment"}:
            sample = adapt_stage3_environment(row, index=index)
        elif detected == "inference-results":
            sample = adapt_inference_result(
                row,
                index=index,
                results_root=path.parent.resolve(),
                max_events=max_events,
            )
        elif detected == "generic-jsonl":
            sample = adapt_generic(row, index=index)
        else:
            raise ValueError(f"unsupported baseline input format: {detected}")
        samples.append(sample)
        if sample_limit is not None and len(samples) >= sample_limit:
            break
    return samples


def resolve_baseline_input_path(path: Path, *, input_format: str = "auto") -> Path:
    """Resolve a JSONL file or a downloaded ARIA-Full dataset directory."""

    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.is_dir():
        raise FileNotFoundError(f"baseline input does not exist: {candidate}")
    preferred = (
        ("results.jsonl",)
        if input_format == "inference-results"
        else ("dataset.jsonl", "eval_pool.jsonl", "environment.jsonl", "env_all.jsonl")
    )
    for name in preferred:
        found = candidate / name
        if found.is_file():
            return found.resolve()
    names = {"dataset.jsonl", "eval_pool.jsonl", "environment.jsonl", "env_all.jsonl"}
    matches = sorted(item for item in candidate.rglob("*.jsonl") if item.name in names)
    if len(matches) == 1:
        return matches[0].resolve()
    detail = ", ".join(str(item) for item in matches[:5]) or "no JSONL source found"
    raise FileNotFoundError(
        f"baseline input directory {candidate} must contain one of "
        f"dataset.jsonl/eval_pool.jsonl/environment.jsonl/env_all.jsonl; found {detail}"
    )


def detect_format(row: Mapping[str, Any]) -> str:
    if isinstance(row.get("incident"), Mapping):
        return "aria-incident"
    if str(row.get("schema_version") or "") == "aria-stage3-environment-v1":
        return "stage3-environment"
    # ARIA-Full exports may omit the schema version while retaining the
    # answer-blind environment fields. Keep this detection structural so a
    # generic JSONL row is never accidentally treated as a scored incident.
    if all(key in row for key in ("cell", "events", "source", "control")):
        return "aria-full"
    if is_aria_record(row):
        return "aria-record"
    if row.get("output_dir") and isinstance(row.get("evaluation"), Mapping):
        return "inference-results"
    return "generic-jsonl"


def adapt_stage3_environment(
    row: Mapping[str, Any], *, index: int
) -> BaselineSample:
    """Adapt one ARIA-Full/Stage-3 environment incident.

    The aggregate ARIA-Full release stores answer-blind trajectories inline and
    keeps evaluator-only answers under ``evaluator_sample``. This adapter makes
    that representation usable by the baseline runner without exposing the
    answer or feedback policy to model prompts.
    """

    # ``build_eval_pools.py`` wraps an environment row as an inline ARIA
    # incident. Accept that derived form under the same explicit ``aria-full``
    # input format so callers can point at either the aggregate environment or
    # a materialized test-pool file.
    if isinstance(row.get("incident"), Mapping):
        return adapt_aria_incident(row, index=index)

    cell = row.get("cell") if isinstance(row.get("cell"), Mapping) else {}
    source = row.get("source") if isinstance(row.get("source"), Mapping) else {}
    constraints = (
        row.get("constraints") if isinstance(row.get("constraints"), Mapping) else {}
    )
    evaluator_sample = (
        row.get("evaluator_sample")
        if isinstance(row.get("evaluator_sample"), Mapping)
        else {}
    )
    sample_id = str(row.get("incident_id") or row.get("sample_id") or index)
    task_id = str(row.get("task_id") or sample_id)
    task = str(constraints.get("task_text") or evaluator_sample.get("task") or "")
    events = _events(row.get("events"), prefix="source")
    hidden_reference = {
        str(key): value
        for key, value in evaluator_sample.items()
        if str(key).lower() in ANSWER_FIELDS
    }
    evaluation = dict(source)
    return BaselineSample(
        sample_id=sample_id,
        task_id=task_id,
        task=task,
        events=events,
        constraints=_without_answers(constraints),
        feedback=(str(row["feedback"]) if row.get("feedback") is not None else None),
        benchmark=str(cell.get("benchmark_id") or constraints.get("benchmark") or "")
        or None,
        mas=str(cell.get("mas_id") or constraints.get("mas") or "") or None,
        split=str(row.get("split") or "") or None,
        source_evaluation=evaluation,
        control_evaluation=(
            dict(row.get("control") or {})
            if isinstance(row.get("control"), Mapping)
            else dict(evaluation)
        ),
        hidden_reference=hidden_reference,
        metadata={
            "schema_version": row.get("schema_version"),
            "record_id": row.get("record_id"),
            "dataset_id": row.get("source_dataset_id") or "ARIA-Full",
            "record_type": row.get("record_type"),
            "host_model": cell.get("host_model_id"),
            "source_result": dict(row.get("source_result") or {})
            if isinstance(row.get("source_result"), Mapping)
            else {},
            "control_evaluation": dict(row.get("control") or {})
            if isinstance(row.get("control"), Mapping)
            else {},
        },
    )


def adapt_inference_result(
    row: Mapping[str, Any],
    *,
    index: int,
    results_root: Path,
    max_events: int = 64,
) -> BaselineSample:
    """Adapt one compact inference ``results.jsonl`` row.

    Active baselines require the complete task pool, including runs that the
    uninterrupted host solved.  The inference batch is that pool; repair
    datasets are intentionally unsuitable because admission keeps failures.
    """

    from repair.artifacts import sanitize_trace_events
    from repair.generator import build_incident
    from repair.repair_dataset import observable_artifact_events
    from repair.utils import load_json, load_jsonl

    run_root = _artifact_path(results_root, row.get("output_dir"))
    if run_root is None or not run_root.is_dir():
        raise FileNotFoundError(
            f"inference result {row.get('sample_id') or index} has no local output_dir"
        )
    result_path = run_root / "result.json"
    loaded = load_json(result_path) if result_path.is_file() else None
    result = dict(loaded) if isinstance(loaded, Mapping) else dict(row)
    result.update({key: value for key, value in row.items() if value is not None})

    trajectory_path = run_root / "trajectory.jsonl"
    if not trajectory_path.is_file():
        trajectory_path = _artifact_path(run_root, result.get("native_trace_path"))
    trace_events = (
        sanitize_trace_events(load_jsonl(trajectory_path))
        if trajectory_path is not None and trajectory_path.is_file()
        else []
    )
    messages_path = run_root / "messages.json"
    messages = load_json(messages_path) if messages_path.is_file() else []
    if not isinstance(messages, list):
        messages = []
    source = {
        "run_dir": run_root,
        "result": result,
        "messages": messages,
        "trace_events": trace_events,
        "artifact_events": observable_artifact_events(run_root, result),
    }
    incident = build_incident(source, max_events=max_events)
    metadata = (
        dict(result["sample_metadata"])
        if isinstance(result.get("sample_metadata"), Mapping)
        else {}
    )
    evaluation = (
        dict(result["evaluation"])
        if isinstance(result.get("evaluation"), Mapping)
        else {}
    )
    hidden_reference = {
        str(key): value
        for key, value in {**metadata, **evaluation}.items()
        if str(key).lower() in ANSWER_FIELDS
    }
    sample_id = str(result.get("sample_id") or row.get("sample_id") or index)
    return BaselineSample(
        sample_id=sample_id,
        task_id=str(metadata.get("task_id") or sample_id),
        task=str(incident.constraints.get("task_text") or ""),
        events=tuple(
            BaselineEvent(
                event_id=event.event_id,
                role=event.role,
                kind=event.kind,
                content=event.content,
                tool_calls=tuple(dict(item) for item in event.tool_calls),
                tool_results=tuple(
                    dict(item)
                    for item in (event.metadata.get("tool_results") or [])
                    if isinstance(item, Mapping)
                ),
                metadata=dict(event.metadata),
            )
            for event in incident.events
        ),
        constraints=dict(incident.constraints),
        feedback=incident.feedback,
        benchmark=str(result.get("benchmark") or "") or None,
        mas=str(result.get("mas") or "") or None,
        split=str(metadata.get("split") or metadata.get("sample_split") or "") or None,
        source_evaluation=evaluation,
        control_evaluation=(
            dict(row.get("control_run", {}).get("evaluation") or {})
            if isinstance(row.get("control_run"), Mapping)
            else dict(evaluation)
        ),
        hidden_reference=hidden_reference,
        metadata={
            "input_kind": "inference-results",
            "host_model": result.get("model"),
            "source_result": result,
            "artifact_paths": {
                "run_dir": str(run_root),
                "result_path": str(result_path),
                "trajectory_path": str(trajectory_path) if trajectory_path else None,
                "messages_path": str(messages_path),
            },
        },
    )


def adapt_aria_incident(row: Mapping[str, Any], *, index: int) -> BaselineSample:
    """Adapt a record that inlines its incident instead of referencing it."""

    incident = _mapping(row.get("incident"), "incident")
    sample = _mapping(row.get("sample"), "sample")
    source_refs = _mapping(row.get("source_refs") or {}, "source_refs")
    # A sampled successful run stores an explicit null for the failed-run block
    # rather than omitting the key, so the default has to cover null too.
    failed_run = _mapping(row.get("failed_run") or {}, "failed_run")
    events = _events(incident.get("events"), prefix="source")
    hidden_reference = _mapping(sample.get("reference") or {}, "sample.reference")
    task = str(sample.get("task_text") or incident.get("task") or "")
    evaluation = _mapping(failed_run.get("evaluation") or {}, "failed_run.evaluation")
    if "success" not in evaluation and "correct" not in evaluation:
        # A sampled successful run has no failed_run block; its outcome lives in
        # the record's outcome summary, and dropping it would remove every
        # passing task from an active evaluation pool.
        outcome = _mapping(row.get("outcome") or {}, "outcome")
        if isinstance(outcome.get("source_success"), bool):
            evaluation = {
                **evaluation,
                "success": outcome["source_success"],
                "score": 1.0 if outcome["source_success"] else 0.0,
                "method": "recorded-source-execution",
            }
    control_run = _mapping(row.get("control_run") or {}, "control_run")
    control_evaluation = _mapping(
        control_run.get("evaluation") or row.get("control_evaluation") or {},
        "control_run.evaluation",
    )
    if not control_evaluation:
        control_evaluation = dict(evaluation)
    return BaselineSample(
        sample_id=str(row.get("sample_id") or incident.get("incident_id") or index),
        task_id=str(row.get("task_id") or row.get("sample_id") or index),
        task=task,
        events=events,
        constraints=_without_answers(incident.get("constraints") or {}),
        feedback=(
            str(incident["feedback"]) if incident.get("feedback") is not None else None
        ),
        benchmark=str(source_refs.get("benchmark_id") or sample.get("benchmark") or "")
        or None,
        mas=str(source_refs.get("mas_id") or "") or None,
        split=str(row.get("split") or "") or None,
        source_evaluation=evaluation,
        control_evaluation=control_evaluation,
        hidden_reference=hidden_reference,
        metadata={
            "schema_version": row.get("schema_version"),
            "record_id": row.get("record_id"),
            "evaluator_sample": dict(row.get("evaluator_sample") or {})
            if isinstance(row.get("evaluator_sample"), Mapping)
            else {},
            "source_result": dict(row.get("source_result") or {})
            if isinstance(row.get("source_result"), Mapping) else {},
            "host_model": source_refs.get("model_id"),
        },
    )


def adapt_generic(row: Mapping[str, Any], *, index: int) -> BaselineSample:
    raw_events = (
        row.get("events")
        or row.get("trajectory")
        or row.get("history")
        or row.get("conversation")
        or row.get("messages")
    )
    events = _events(raw_events, prefix="generic")
    task = (
        row.get("task")
        or row.get("question")
        or row.get("query")
        or row.get("prompt")
        or ""
    )
    hidden = {key: row[key] for key in ANSWER_FIELDS if key in row}
    evaluation = (
        row.get("evaluation") if isinstance(row.get("evaluation"), Mapping) else {}
    )
    constraints = (
        row.get("constraints") if isinstance(row.get("constraints"), Mapping) else {}
    )
    return BaselineSample(
        sample_id=str(row.get("sample_id") or row.get("id") or index),
        task_id=str(
            row.get("task_id") or row.get("sample_id") or row.get("id") or index
        ),
        task=str(task),
        events=events,
        constraints=_without_answers(constraints),
        feedback=str(row["feedback"]) if row.get("feedback") is not None else None,
        benchmark=str(row.get("benchmark") or "") or None,
        mas=str(row.get("mas") or row.get("system") or "") or None,
        split=str(row.get("split") or "") or None,
        source_evaluation=dict(evaluation),
        hidden_reference=hidden,
        metadata={
            key: value
            for key, value in row.items()
            if key not in ANSWER_FIELDS
            and key
            not in {
                "events",
                "trajectory",
                "history",
                "conversation",
                "messages",
                "constraints",
                "evaluation",
            }
        },
    )


def _events(raw: Any, *, prefix: str) -> tuple[BaselineEvent, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("baseline sample must contain a non-empty event list")
    normalized: list[BaselineEvent] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise TypeError(f"event {index} must be an object")
        event_id = str(
            item.get("event_id") or item.get("id") or f"{prefix}:event:{index:04d}"
        )
        role = str(
            item.get("role")
            or item.get("mas_agent_role")
            or item.get("mas_agent_id")
            or item.get("name")
            or item.get("source")
            or "assistant"
        )
        content = item.get("content", item.get("message", item.get("text", "")))
        tool_calls = item.get("tool_calls", [])
        tool_results = item.get("tool_results", [])
        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]
        if not isinstance(tool_results, list):
            tool_results = [tool_results]
        normalized.append(
            BaselineEvent(
                event_id=event_id,
                role=role,
                kind=str(item.get("kind") or item.get("type") or "message"),
                content=str(content),
                tool_calls=tuple(
                    dict(value) for value in tool_calls if isinstance(value, Mapping)
                ),
                tool_results=tuple(
                    dict(value) for value in tool_results if isinstance(value, Mapping)
                ),
                metadata={
                    key: value
                    for key, value in item.items()
                    if key
                    not in {
                        "event_id",
                        "id",
                        "role",
                        "mas_agent_role",
                        "mas_agent_id",
                        "name",
                        "source",
                        "content",
                        "message",
                        "text",
                        "kind",
                        "type",
                        "tool_calls",
                        "tool_results",
                    }
                },
            )
        )
    ids = [event.event_id for event in normalized]
    if len(ids) != len(set(ids)):
        raise ValueError("baseline event IDs must be unique")
    return tuple(normalized)


def _without_answers(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _without_answers(item)
            for key, item in value.items()
            if str(key).lower() not in ANSWER_FIELDS
        }
    if isinstance(value, list):
        return [_without_answers(item) for item in value]
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return dict(value)


def _artifact_path(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _read_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} must contain a JSON object")
            yield row
