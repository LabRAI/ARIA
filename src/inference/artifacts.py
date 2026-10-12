"""Inference workspace, checkpoint, and standard artifact helpers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from common.artifacts import atomic_write_json, atomic_write_jsonl, read_jsonl_lines, utc_now
from common.model_api import ApiConfig
from common.runtime_support import runtime_state_from_result, safe_run_id


def write_sample_run_state(
    run_root: Path,
    *,
    run_id: str,
    status: str,
    mas_slug: str,
    benchmark_slug: str,
    sample_id: str | None,
    error: str | None = None,
) -> None:
    atomic_write_json(
        run_root / "run_state.json",
        {
            "schema_version": "aria-run-state-v1",
            "run_id": run_id,
            "status": status,
            "mas": mas_slug,
            "benchmark": benchmark_slug,
            "sample_id": sample_id,
            "updated_at": utc_now(),
            "error": error,
        },
    )


def resolve_workspace_path(
    config: Any,
    mas_slug: str,
    benchmark_slug: str,
    run_id: str,
    sample_id: str | None = None,
) -> Path:
    pair_dir = f"{safe_run_id(mas_slug)}-{safe_run_id(benchmark_slug)}"
    leaf = safe_run_id(sample_id or run_id)
    return (config.workspace_dir / pair_dir / leaf).resolve()


def prepare_sample_workspace(workspace_path: Path, *, reset: bool) -> None:
    """Give each batch attempt a clean workspace while retaining a stable path."""

    if reset and workspace_path.exists():
        if workspace_path.is_dir() and not workspace_path.is_symlink():
            shutil.rmtree(workspace_path)
        else:
            workspace_path.unlink()
    workspace_path.mkdir(parents=True, exist_ok=True)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def clear_stale_aria_artifacts(run_root: Path) -> None:
    for filename in ("aria_runtime.json", "aria_patch_history.json", "aria_event_log.jsonl", "aria_model_interventions.jsonl"):
        (run_root / filename).unlink(missing_ok=True)


def compact_batch_result(result: dict[str, Any]) -> dict[str, Any]:
    keep_keys = {
        "run_id",
        "batch_run_id",
        "mas",
        "benchmark",
        "sample_id",
        "sample_index",
        "sample_trace_id",
        "sample_metadata",
        "model",
        "base_url",
        "provider",
        "protocol",
        "endpoint_url",
        "reasoning_effort",
        "upstream_provider",
        "allow_fallbacks",
        "runtime_id",
        "model_id",
        "api_key_env",
        "api_key_required",
        "status",
        "error",
        "failure_category",
        "output_dir",
        "workspace_path",
        "native_messages_path",
        "native_trace_path",
        "evaluation",
    }
    return {key: result.get(key) for key in keep_keys if key in result}


def load_latest_batch_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    results: dict[str, dict[str, Any]] = {}
    lines = read_jsonl_lines(path)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            is_partial_tail = index == len(lines) - 1 and not line.endswith("\n")
            if is_partial_tail:
                break
            raise ValueError(f"invalid JSONL record in {path} at line {index + 1}") from exc
        if not isinstance(payload, dict):
            raise TypeError(f"JSONL record in {path} at line {index + 1} must be an object")
        sample_id = payload.get("sample_id")
        if sample_id:
            results[str(sample_id)] = payload
    return results


def load_existing_batch_results(
    path: Path,
    *,
    require_resolved_evaluation: bool = False,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for sample_id, payload in load_latest_batch_results(path).items():
        evaluation = payload.get("evaluation")
        resolved = isinstance(evaluation, dict) and isinstance(
            evaluation.get("correct"), bool
        )
        if payload.get("status") == "completed" and (
            resolved or not require_resolved_evaluation
        ):
            results[sample_id] = payload
    return results


def refresh_standard_run_artifacts(run_root: Path, result: dict[str, Any]) -> None:
    runtime = runtime_state_from_result(result)
    messages_path = (
        Path(str(result["native_messages_path"]))
        if result.get("native_messages_path")
        else run_root / "messages.json"
    )
    trace_path = (
        Path(str(result["native_trace_path"]))
        if result.get("native_trace_path")
        else run_root / "trajectory.jsonl"
    )
    messages = load_json_if_exists(
        messages_path,
        messages_from_transcript(result.get("transcript", [])),
    )
    trace_events = load_jsonl_if_exists(trace_path)
    if not trace_events:
        trace_events = trajectory_from_transcript(result.get("transcript", []), phase="source")
    write_standard_run_artifacts(
        run_root=run_root,
        result=result,
        messages=messages,
        trace_events=trace_events,
        evaluation=result.get("evaluation"),
        runtime_state=runtime,
    )


def write_standard_run_artifacts(
    *,
    run_root: Path,
    result: dict[str, Any],
    messages: Any,
    trace_events: list[dict[str, Any]],
    evaluation: dict[str, Any] | None,
    runtime_state: dict[str, Any],
) -> None:
    if not isinstance(messages, list):
        messages = []
    normalized_trace = normalize_trajectory_events(trace_events, result=result)
    atomic_write_json(run_root / "messages.json", messages)
    atomic_write_jsonl(run_root / "trajectory.jsonl", normalized_trace)
    atomic_write_json(run_root / "runtime_state.json", runtime_state)
    if evaluation is not None:
        atomic_write_json(run_root / "evaluation.json", evaluation)


def messages_from_transcript(transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    messages = []
    for item in transcript:
        if not isinstance(item, dict):
            continue
        messages.append(
            {
                "role": "assistant",
                "source": item.get("role") or item.get("source"),
                "content": item.get("content") or item.get("text") or item.get("result") or "",
                "turn": item.get("turn"),
            }
        )
    return messages


def trajectory_from_transcript(
    transcript: list[dict[str, Any]],
    *,
    phase: str,
) -> list[dict[str, Any]]:
    events = []
    for index, item in enumerate(transcript, start=1):
        if not isinstance(item, dict):
            continue
        turn = int(item.get("turn") or index)
        role = str(item.get("role") or item.get("source") or "assistant")
        events.append(
            {
                "event_id": f"{phase}:event:{index:04d}",
                "turn": turn,
                "mas_agent_id": safe_run_id(role) or "assistant",
                "mas_agent_role": role,
                "kind": "assistant_message",
                "hook": "after_agent_step",
                "content": str(item.get("content") or item.get("text") or item.get("result") or ""),
                "tool_calls": item.get("tool_calls") or [],
                "tool_results": item.get("tool_results") or [],
                "native_event": {"source": "transcript", "native_index": index - 1},
                "detector": item.get("detector"),
            }
        )
    return events


def normalize_trajectory_events(
    events: list[dict[str, Any]],
    *,
    result: dict[str, Any],
    phase: str = "source",
) -> list[dict[str, Any]]:
    normalized = []
    for index, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            continue
        record = event.get("record") if isinstance(event.get("record"), dict) else {}
        role = str(
            event.get("mas_agent_id")
            or event.get("role")
            or event.get("source")
            or record.get("source")
            or "agent"
        )
        normalized.append(
            {
                "event_id": str(event.get("event_id") or f"{phase}:event:{index:04d}"),
                "turn": event.get("turn") or record.get("turn") or index,
                "mas_agent_id": safe_run_id(role) or "agent",
                "mas_agent_role": str(event.get("mas_agent_role") or role),
                "kind": str(event.get("kind") or record.get("type") or "runtime_event"),
                "hook": event.get("hook"),
                "content": event_text(event),
                "tool_calls": event.get("tool_calls") or record.get("tool_calls") or [],
                "tool_results": event.get("tool_results") or [],
                "native_event": {
                    "source": result.get("mas") or "unknown",
                    "native_index": index - 1,
                    "native_kind": event.get("kind"),
                    "record": record or None,
                },
                "detector": event.get("detector"),
            }
        )
    if not normalized and result.get("transcript"):
        return trajectory_from_transcript(result.get("transcript", []), phase=phase)
    return normalized


def event_text(event: dict[str, Any]) -> str:
    for key in ("content", "text", "result"):
        value = event.get(key)
        if value is not None:
            return json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
    record = event.get("record")
    if isinstance(record, dict):
        for key in ("content", "text", "result"):
            value = record.get(key)
            if value is not None:
                return (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else str(value)
                )
    return ""


def build_runtime_state(result: dict[str, Any], api: ApiConfig) -> dict[str, Any]:
    return runtime_state_from_result(result, api)


def load_json_if_exists(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def load_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    lines = read_jsonl_lines(path)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            is_partial_tail = index == len(lines) - 1 and not line.endswith("\n")
            if is_partial_tail:
                break
            raise ValueError(f"invalid JSONL record in {path} at line {index + 1}") from exc
        if not isinstance(payload, dict):
            raise TypeError(f"JSONL record in {path} at line {index + 1} must be an object")
        rows.append(payload)
    return rows
