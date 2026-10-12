"""Adapter from on-disk ARIA repair-dataset records into baseline samples.

An ARIA repair dataset is a ``dataset.jsonl`` of records whose failed run is
stored beside it under ``samples/<sample_id>/failed/``.  The record itself never
inlines the trajectory; it references it through ``failed_run.trajectory_ref``.

Baselines must observe exactly what ARIA observes, so this module does not
re-normalize trajectories on its own.  It reconstructs the same ``source``
mapping that :func:`repair.repair_dataset.load_source_artifacts` produces and
hands it to :func:`repair.generator.build_incident`, which is the single place
where ARIA decides which events are model-visible and answer-blind.  A
difference between an ARIA row and a baseline row is therefore a difference of
method, never of input.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from baselines.core.types import BaselineEvent, BaselineSample

ARIA_RECORD_SCHEMA_VERSIONS = frozenset(
    {
        "aria-repair-dataset-v4",
        "aria-repair-dataset-v5",
        "aria-repair-dataset-v6",
    }
)


def is_aria_record(row: Mapping[str, Any]) -> bool:
    """True for a repair-dataset record whose failed run lives on disk."""

    if str(row.get("schema_version") or "") in ARIA_RECORD_SCHEMA_VERSIONS:
        return True
    return isinstance(row.get("failed_run"), Mapping) and isinstance(
        row.get("sample"), Mapping
    )


def adapt_aria_record(
    row: Mapping[str, Any],
    *,
    index: int,
    dataset_root: Path,
    max_events: int = 64,
) -> BaselineSample:
    """Build one baseline sample from an ARIA repair-dataset record."""

    from repair.generator import build_incident

    sample_section = _mapping(row.get("sample", {}), "sample")
    failed_run = _mapping(row.get("failed_run", {}), "failed_run")
    source_refs = _mapping(row.get("source_refs", {}), "source_refs")
    source, artifact_paths = load_record_source(
        row,
        dataset_root=dataset_root,
        sample_section=sample_section,
        failed_run=failed_run,
        source_refs=source_refs,
    )
    incident = build_incident(source, max_events=max_events)
    sample_id = str(
        row.get("sample_id") or row.get("record_id") or incident.incident_id or index
    )
    evaluation = failed_run.get("evaluation")
    if not isinstance(evaluation, Mapping):
        evaluation = _mapping(source["result"].get("evaluation", {}), "evaluation")
    if not isinstance(evaluation.get("success"), bool) and not isinstance(
        evaluation.get("correct"), bool
    ):
        # A sampled successful run carries no failed-run block: its outcome is
        # in the record's outcome summary. Without this an active pool would
        # lose every passing task and could not be scored at all.
        outcome = _mapping(row.get("outcome"), "outcome")
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
    benchmark = (
        str(source_refs.get("benchmark_id") or sample_section.get("benchmark") or "")
        or None
    )
    mas = str(source_refs.get("mas_id") or source["result"].get("mas") or "") or None
    return BaselineSample(
        sample_id=sample_id,
        task_id=str(row.get("task_id") or sample_id),
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
        benchmark=benchmark,
        mas=mas,
        split=str(row.get("split") or "") or None,
        source_evaluation=dict(evaluation),
        control_evaluation=control_evaluation,
        hidden_reference=_mapping(
            sample_section.get("reference", {}), "sample.reference"
        ),
        metadata={
            "schema_version": row.get("schema_version"),
            "record_id": row.get("record_id"),
            "dataset_id": row.get("dataset_id"),
            "record_type": row.get("record_type"),
            "source_refs": dict(source_refs),
            "host_model": source["result"].get("model") or source_refs.get("model_id"),
            "artifact_paths": artifact_paths,
            "source_result": source["result"],
            "localization": row.get("localization"),
        },
    )


def load_record_source(
    row: Mapping[str, Any],
    *,
    dataset_root: Path,
    sample_section: Mapping[str, Any],
    failed_run: Mapping[str, Any],
    source_refs: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str | None]]:
    """Rebuild the ``source`` mapping that ARIA's incident builder consumes.

    Referenced artifacts are read when the dataset ships them.  A dataset
    exported as records alone (a Hugging Face upload of ``dataset.jsonl``, for
    instance) still yields a usable sample, reconstructed from the fields the
    record itself carries.
    """

    from repair.artifacts import sanitize_trace_events
    from repair.utils import load_json, load_jsonl

    trajectory_path = _reference(dataset_root, failed_run.get("trajectory_ref"))
    messages_path = _reference(dataset_root, failed_run.get("messages_ref"))
    result_path = _reference(dataset_root, failed_run.get("result_ref"))

    result: dict[str, Any] = {}
    if result_path is not None and result_path.is_file():
        loaded = load_json(result_path)
        if isinstance(loaded, Mapping):
            result = dict(loaded)
    result = _complete_result(result, row, sample_section, failed_run, source_refs)

    trace_events: list[dict[str, Any]] = []
    if trajectory_path is not None and trajectory_path.is_file():
        trace_events = sanitize_trace_events(load_jsonl(trajectory_path))
    messages: list[dict[str, Any]] = []
    if messages_path is not None and messages_path.is_file():
        loaded_messages = load_json(messages_path)
        if isinstance(loaded_messages, list):
            messages = loaded_messages

    artifact_events: list[dict[str, Any]] = []
    run_dir = trajectory_path.parent if trajectory_path is not None else None
    if run_dir is not None and run_dir.is_dir():
        from repair.repair_dataset import observable_artifact_events

        artifact_events = observable_artifact_events(run_dir, result)

    source = {
        "run_dir": run_dir,
        "result": result,
        "messages": messages,
        "trace_events": trace_events,
        "artifact_events": artifact_events,
    }
    artifact_paths = {
        "run_dir": str(run_dir) if run_dir is not None else None,
        "result_path": str(result_path) if result_path is not None else None,
        "trajectory_path": str(trajectory_path)
        if trajectory_path is not None
        else None,
        "messages_path": str(messages_path) if messages_path is not None else None,
        "materialized": str(bool(trace_events or messages)),
    }
    return source, artifact_paths


def _complete_result(
    result: dict[str, Any],
    row: Mapping[str, Any],
    sample_section: Mapping[str, Any],
    failed_run: Mapping[str, Any],
    source_refs: Mapping[str, Any],
) -> dict[str, Any]:
    """Fill the fields the incident builder needs from the record itself."""

    metadata = result.get("sample_metadata")
    if not isinstance(metadata, Mapping):
        metadata = sample_section.get("metadata")
    result["sample_metadata"] = dict(metadata) if isinstance(metadata, Mapping) else {}
    result.setdefault("sample_id", row.get("sample_id"))
    result.setdefault("run_id", row.get("record_id") or row.get("trajectory_id"))
    if not result.get("benchmark"):
        result["benchmark"] = source_refs.get("benchmark_id") or sample_section.get(
            "benchmark"
        )
    if not result.get("mas"):
        result["mas"] = source_refs.get("mas_id")
    if not result.get("final_output"):
        result["final_output"] = failed_run.get("final_output") or ""
    if not isinstance(result.get("evaluation"), Mapping) and isinstance(
        failed_run.get("evaluation"), Mapping
    ):
        result["evaluation"] = dict(failed_run["evaluation"])
    task_text = sample_section.get("task_text")
    if isinstance(task_text, str) and task_text.strip():
        result.setdefault("task", task_text)
        result["sample_metadata"].setdefault("question", task_text)
    return result


def _reference(dataset_root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (dataset_root / path)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return dict(value)
