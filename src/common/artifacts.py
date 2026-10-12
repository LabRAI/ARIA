"""Atomic artifact writes and resumable batch state."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RUN_STATE_SCHEMA = "aria-run-state-v1"


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sample_metadata(sample: dict[str, Any]) -> dict[str, Any]:
    """Retain evaluator metadata while excluding model-visible task bodies."""

    skipped_keys = {"task", "reference_solution", "seed_solution"}
    return {key: value for key, value in sample.items() if key not in skipped_keys}


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    atomic_write_text(path, text)


def append_jsonl_fsync(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl_lines(path: Path) -> list[str]:
    """Read physical JSONL records without treating Unicode separators as newlines."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        return handle.readlines()


@dataclass
class BatchCheckpoint:
    path: Path
    run_id: str
    total_samples: int
    completed_sample_ids: set[str] = field(default_factory=set)
    failed_sample_ids: set[str] = field(default_factory=set)
    invalid_sample_ids: set[str] = field(default_factory=set)
    started_at: str = field(default_factory=utc_now)
    status: str = "pending"
    active_sample_id: str | None = None
    last_error: str | None = None

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        run_id: str,
        total_samples: int,
        completed_sample_ids: set[str],
    ) -> BatchCheckpoint:
        if path.exists():
            payload = json.loads(path.read_text())
            if payload.get("schema_version") != RUN_STATE_SCHEMA:
                raise ValueError(f"unsupported run state schema in {path}")
            if payload.get("run_id") != run_id:
                raise ValueError(f"run state id mismatch in {path}")
            previous_total = int(payload.get("total_samples", -1))
            if previous_total > total_samples:
                raise ValueError(
                    f"run state sample count cannot shrink for {run_id}: "
                    f"{previous_total} > {total_samples}"
                )
            checkpoint = cls(
                path=path,
                run_id=run_id,
                total_samples=total_samples,
                completed_sample_ids=set(payload.get("completed_sample_ids", [])),
                failed_sample_ids=set(payload.get("failed_sample_ids", [])),
                invalid_sample_ids=set(payload.get("invalid_sample_ids", [])),
                started_at=str(payload.get("started_at") or utc_now()),
                status=str(payload.get("status") or "pending"),
                active_sample_id=payload.get("active_sample_id"),
                last_error=payload.get("last_error"),
            )
        else:
            checkpoint = cls(path=path, run_id=run_id, total_samples=total_samples)
        checkpoint.completed_sample_ids.update(completed_sample_ids)
        checkpoint.failed_sample_ids.difference_update(checkpoint.completed_sample_ids)
        checkpoint.invalid_sample_ids.difference_update(checkpoint.completed_sample_ids)
        checkpoint.failed_sample_ids.difference_update(checkpoint.invalid_sample_ids)
        checkpoint.active_sample_id = None
        checkpoint.last_error = None
        checkpoint.status = "running"
        checkpoint.save()
        return checkpoint

    def start_sample(self, sample_id: str) -> None:
        self.status = "running"
        self.active_sample_id = sample_id
        self.last_error = None
        self.save()

    def finish_sample(self, sample_id: str, status: str, error: str | None = None) -> None:
        self.active_sample_id = None
        if status == "completed":
            self.completed_sample_ids.add(sample_id)
            self.failed_sample_ids.discard(sample_id)
            self.invalid_sample_ids.discard(sample_id)
        elif status == "invalid":
            self.completed_sample_ids.discard(sample_id)
            self.failed_sample_ids.discard(sample_id)
            self.invalid_sample_ids.add(sample_id)
        else:
            self.completed_sample_ids.discard(sample_id)
            self.invalid_sample_ids.discard(sample_id)
            self.failed_sample_ids.add(sample_id)
        self.last_error = error
        self.save()

    def finish(self, status: str) -> None:
        if status not in {"completed", "partial", "failed", "interrupted"}:
            raise ValueError(f"unsupported terminal run status: {status}")
        self.status = status
        self.active_sample_id = None
        self.save()

    def save(self) -> None:
        atomic_write_json(
            self.path,
            {
                "schema_version": RUN_STATE_SCHEMA,
                "run_id": self.run_id,
                "status": self.status,
                "started_at": self.started_at,
                "updated_at": utc_now(),
                "total_samples": self.total_samples,
                "completed_samples": len(self.completed_sample_ids),
                "failed_samples": len(self.failed_sample_ids),
                "invalid_samples": len(self.invalid_sample_ids),
                "completed_sample_ids": sorted(self.completed_sample_ids),
                "failed_sample_ids": sorted(self.failed_sample_ids),
                "invalid_sample_ids": sorted(self.invalid_sample_ids),
                "active_sample_id": self.active_sample_id,
                "last_error": self.last_error,
            },
        )
