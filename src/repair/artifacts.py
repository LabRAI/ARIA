"""Canonical artifact snapshots for failed, control, and patched runs."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from common.artifacts import atomic_write_json, atomic_write_jsonl
from common.runtime_support import runtime_state_from_result, safe_name
from inference.native import NATIVE_MAS_BACKENDS, get_native_mas_backend
from repair.operator import trace_event_text
from repair.utils import load_json, load_jsonl, sanitize_result

STANDARD_SNAPSHOT_FILENAMES = (
    "result.json",
    "messages.json",
    "trajectory.jsonl",
    "runtime_state.json",
    "evaluation.json",
    "model.patch",
    "run.log",
    "stdout.log",
    "stderr.log",
)
SNAPSHOT_FILENAMES = STANDARD_SNAPSHOT_FILENAMES + tuple(
    filename
    for backend in NATIVE_MAS_BACKENDS.values()
    for filename in backend.artifact_filenames
)
NATIVE_MESSAGE_FILENAMES = tuple(
    backend.messages_filename for backend in NATIVE_MAS_BACKENDS.values()
)
NATIVE_TRACE_FILENAMES = tuple(
    backend.trace_filename for backend in NATIVE_MAS_BACKENDS.values()
)
NATIVE_STDOUT_FILENAMES = tuple(
    filename
    for backend in NATIVE_MAS_BACKENDS.values()
    for filename in backend.artifact_filenames
    if filename.endswith("_stdout.log")
)
NATIVE_STDERR_FILENAMES = tuple(
    filename
    for backend in NATIVE_MAS_BACKENDS.values()
    for filename in backend.artifact_filenames
    if filename.endswith("_stderr.log")
)


def snapshot_run_artifacts(source_dir: Path, destination_dir: Path) -> dict[str, Any]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, Any]] = []
    for filename in SNAPSHOT_FILENAMES:
        source = source_dir / filename
        if not source.exists():
            continue
        destination = destination_dir / filename
        if filename == "result.json":
            write_sanitized_result_artifact(source, destination)
        elif filename.endswith("_native_trace.jsonl"):
            write_sanitized_trace_artifact(source, destination)
        else:
            shutil.copy2(source, destination)
        files.append(_file_record(filename, destination))
    ensure_standard_snapshot_files(source_dir, destination_dir, files)
    return {"dir": str(destination_dir), "files": files}


def ensure_standard_snapshot_files(
    source_dir: Path,
    destination_dir: Path,
    files: list[dict[str, Any]],
) -> None:
    existing = {str(item.get("name")) for item in files if isinstance(item, dict)}
    _copy_first_alias(
        destination_dir,
        files,
        existing,
        "messages.json",
        NATIVE_MESSAGE_FILENAMES,
    )
    if "trajectory.jsonl" not in existing:
        for candidate in NATIVE_TRACE_FILENAMES:
            source = destination_dir / candidate
            if source.exists():
                destination = destination_dir / "trajectory.jsonl"
                write_sanitized_trace_artifact(source, destination)
                files.append(_file_record("trajectory.jsonl", destination))
                existing.add("trajectory.jsonl")
                break

    result_path = source_dir / "result.json"
    result = load_json(result_path) if result_path.exists() else None
    if isinstance(result, dict) and "evaluation.json" not in existing and isinstance(result.get("evaluation"), dict):
        destination = destination_dir / "evaluation.json"
        atomic_write_json(destination, result["evaluation"])
        files.append(_file_record("evaluation.json", destination))
        existing.add("evaluation.json")
    if isinstance(result, dict) and "runtime_state.json" not in existing:
        destination = destination_dir / "runtime_state.json"
        atomic_write_json(destination, runtime_state_from_result(result))
        files.append(_file_record("runtime_state.json", destination))
        existing.add("runtime_state.json")

    _copy_first_alias(
        destination_dir,
        files,
        existing,
        "stdout.log",
        NATIVE_STDOUT_FILENAMES,
    )
    _copy_first_alias(
        destination_dir,
        files,
        existing,
        "stderr.log",
        NATIVE_STDERR_FILENAMES,
    )


def _copy_first_alias(
    destination_dir: Path,
    files: list[dict[str, Any]],
    existing: set[str],
    canonical: str,
    candidates: tuple[str, ...],
) -> None:
    if canonical in existing:
        return
    for candidate in candidates:
        source = destination_dir / candidate
        if not source.exists():
            continue
        destination = destination_dir / canonical
        shutil.copy2(source, destination)
        files.append(_file_record(canonical, destination))
        existing.add(canonical)
        return


def _file_record(name: str, path: Path) -> dict[str, Any]:
    return {"name": name, "path": str(path), "bytes": path.stat().st_size}


def native_messages_filename(mas_slug: str) -> str:
    return get_native_mas_backend(mas_slug).messages_filename


def native_trace_filename(mas_slug: str) -> str:
    return get_native_mas_backend(mas_slug).trace_filename


def native_messages_path(result: dict[str, Any], run_root: Path, mas_slug: str) -> Path:
    if result.get("native_messages_path"):
        return Path(str(result["native_messages_path"])).resolve()
    return (run_root / native_messages_filename(mas_slug)).resolve()


def native_trace_path(result: dict[str, Any], run_root: Path, mas_slug: str) -> Path:
    if result.get("native_trace_path"):
        return Path(str(result["native_trace_path"])).resolve()
    return (run_root / native_trace_filename(mas_slug)).resolve()


def write_standard_repair_run_artifacts(
    run_root: Path,
    result: dict[str, Any],
    mas_slug: str,
) -> None:
    messages_path = native_messages_path(result, run_root, mas_slug)
    trace_path = native_trace_path(result, run_root, mas_slug)
    messages = load_json(messages_path) if messages_path.exists() else []
    trace_events = load_jsonl(trace_path) if trace_path.exists() else []
    atomic_write_json(run_root / "messages.json", messages if isinstance(messages, list) else [])
    atomic_write_jsonl(
        run_root / "trajectory.jsonl",
        normalize_repair_trace_events(trace_events, result),
    )
    if isinstance(result.get("evaluation"), dict):
        atomic_write_json(run_root / "evaluation.json", result["evaluation"])
    atomic_write_json(run_root / "runtime_state.json", runtime_state_from_result(result))


def normalize_repair_trace_events(
    events: list[dict[str, Any]],
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            continue
        record = event.get("record") if isinstance(event.get("record"), dict) else {}
        role = str(
            event.get("mas_agent_role")
            or event.get("role")
            or event.get("source")
            or record.get("source")
            or "agent"
        )
        normalized.append(
            {
                "event_id": str(event.get("event_id") or f"repaired:event:{index:04d}"),
                "turn": event.get("turn") or record.get("turn") or index,
                "mas_agent_id": str(event.get("mas_agent_id") or safe_name(role) or "agent"),
                "mas_agent_role": role,
                "kind": str(event.get("kind") or record.get("type") or "runtime_event"),
                "hook": event.get("hook"),
                "content": trace_event_text(event),
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
    if normalized:
        return normalized
    for index, item in enumerate(result.get("transcript") or [], start=1):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or item.get("source") or "assistant")
        normalized.append(
            {
                "event_id": f"repaired:event:{index:04d}",
                "turn": item.get("turn") or index,
                "mas_agent_id": safe_name(role),
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
    return normalized


def write_sanitized_result_artifact(source: Path, destination: Path) -> None:
    payload = load_json(source)
    atomic_write_json(destination, sanitize_result(payload) if isinstance(payload, dict) else payload)


def write_sanitized_trace_artifact(source: Path, destination: Path) -> None:
    atomic_write_jsonl(destination, sanitize_trace_events(load_jsonl(source)))


def sanitize_trace_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [clean for event in events if (clean := sanitize_trace_event(event)) is not None]


def sanitize_trace_event(event: dict[str, Any]) -> dict[str, Any] | None:
    if str(event.get("kind") or "").startswith("finalize_patch"):
        return None
    blocked = {"patch", "detection", "aria", "policy"}
    return {
        key: value
        for key, value in event.items()
        if key not in blocked and not str(key).startswith("aria_")
    }
