"""Small JSON, path, and record utilities shared by repair modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def resolve_path(value: str | Path) -> Path:
    path = value if isinstance(value, Path) else Path(str(value))
    return path.expanduser().resolve()


def resolve_optional_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return resolve_path(value)


def first_existing_path(*paths: Path | None) -> Path | None:
    for path in paths:
        if path is not None and path.exists():
            return path
    return None


def source_batch_context(source_batch_dir: Path) -> tuple[str, str]:
    benchmark_slug = source_batch_dir.parent.name
    mas_slug = source_batch_dir.parent.parent.name
    if not mas_slug or not benchmark_slug:
        raise ValueError(f"cannot infer mas/benchmark from source batch dir: {source_batch_dir}")
    return mas_slug, benchmark_slug


def sanitize_result(result: dict[str, Any]) -> dict[str, Any]:
    blocked = {"aria", "aria_finalize_record", "bridge"}
    return {
        key: value
        for key, value in result.items()
        if key not in blocked and not key.startswith("aria_")
    }
