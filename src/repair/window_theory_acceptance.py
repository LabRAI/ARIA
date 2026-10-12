"""Operational Window-Theory trace-acceptance audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from common.window_theory_selection import load_latest_results
from repair.operator import build_repair_prompt, diagnose_failure
from repair.repair_dataset import load_source_artifacts


def evaluate_trace_acceptance(batch_dir: Path) -> dict[str, Any]:
    """Apply the operational trace-acceptance contract to cognitive failures."""

    rows = load_latest_results(batch_dir / "results.jsonl")
    eligible: list[str] = []
    rejected: list[dict[str, str]] = []
    cognitive_failures = 0
    for sample_id, row in rows.items():
        evaluation = row.get("evaluation") if isinstance(row.get("evaluation"), dict) else {}
        if row.get("status") != "completed" or evaluation.get("correct") is not False:
            continue
        cognitive_failures += 1
        try:
            source = load_source_artifacts(row)
            require_trace_artifacts(source)
            diagnosis = diagnose_failure(
                source["result"],
                source["messages"],
                source["trace_events"],
            )
            if not diagnosis.get("evidence"):
                raise ValueError("diagnosis has no localization evidence")
            if not (
                diagnosis.get("message_location")
                or diagnosis.get("trace_location")
                or diagnosis.get("artifact_location")
            ):
                raise ValueError("diagnosis has no message, trace, or artifact location")
            patch = build_repair_prompt(source["result"], diagnosis)
            if patch.get("answer_blind") is not True:
                raise ValueError("repair patch is not answer-blind")
            if not str(patch.get("patched_task") or "").strip():
                raise ValueError("repair patch has no patched task")
            eligible.append(sample_id)
        except Exception as exc:  # noqa: BLE001 - persist an auditable rejection reason.
            rejected.append({"sample_id": sample_id, "reason": f"{type(exc).__name__}: {exc}"})
    return {
        "schema_version": "window-theory-trace-acceptance-v1",
        "batch_dir": str(batch_dir),
        "cognitive_failure_count": cognitive_failures,
        "eligible_failure_count": len(eligible),
        "acceptance_rate": len(eligible) / cognitive_failures if cognitive_failures else None,
        "eligible_failure_ids": eligible,
        "rejected_failures": rejected,
    }


def require_trace_artifacts(source: dict[str, Any]) -> None:
    result_path = Path(str(source.get("result_path") or ""))
    if not result_path.exists():
        raise FileNotFoundError("missing source result.json")
    result = source.get("result") if isinstance(source.get("result"), dict) else {}
    if result.get("status") != "completed":
        raise ValueError("source result is not terminal completed")
    if not source.get("messages"):
        raise ValueError("normalized messages are empty")
    if not source.get("trace_events"):
        raise ValueError("normalized trajectory is empty")
    run_dir = Path(str(source.get("run_dir") or result_path.parent))
    required = ("messages.json", "trajectory.jsonl", "runtime_state.json", "evaluation.json")
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise FileNotFoundError("missing standard artifacts: " + ", ".join(missing))
