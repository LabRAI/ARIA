"""Config-driven passive ARIA repair over completed incident JSONL records."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from aria.deployment import load_passive_repair_service
from aria.model import AriaModelConfig
from aria.records import IncidentRecord


def run_passive_repairs_from_spec(
    model_config: AriaModelConfig,
    spec: Mapping[str, Any],
    *,
    config_path: Path | None,
) -> dict[str, Any]:
    """Repair each submitted completed incident and mark outputs as proposed.

    This config-only route intentionally has no replay environment.  Users who
    need executed correctness use ``PassiveRepairService.repair(..., replay_environment=...)``
    with their actual MAS workspace/replayer.
    """

    root = config_path.parent.resolve() if config_path is not None else Path.cwd()
    input_path = _resolve(root, str(spec["input_path"]))
    output_path = _resolve(root, str(spec["output_path"]))
    service = load_passive_repair_service(model_config)
    rows: list[dict[str, Any]] = []
    with input_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise TypeError(f"passive input line {line_number} must be an object")
            incident = IncidentRecord.from_dict(raw)
            rows.append(
                {
                    "incident_id": incident.incident_id,
                    "trajectory_event_count": len(incident.events),
                    "replay_required_for_success": True,
                    **service.repair(incident).to_dict(),
                }
            )
    _atomic_write_jsonl(output_path, rows)
    return {
        "type": "aria_passive_repair",
        "status": "completed",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "sample_count": len(rows),
        "execution_status": "proposed",
        "replay_required_for_success": True,
    }


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
