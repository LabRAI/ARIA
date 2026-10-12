"""Artifact-derived Window-Theory metrics and per-surface model selection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from common.artifacts import read_jsonl_lines
from common.window_theory import one_sided_exact_mcnemar, wilson_interval


@dataclass(frozen=True)
class SourceMetrics:
    total: int
    infrastructure_invalid: int
    valid: int
    cognitive_failures: int
    correct: int
    infrastructure_rate: float | None
    cognitive_failure_rate: float | None
    cognitive_failure_ci: tuple[float, float]
    triage: str
    completed_sample_ids: tuple[str, ...]
    invalid_sample_ids: tuple[str, ...]
    failure_sample_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RepairMetrics:
    cognitive_failures: int
    eligible_failures: int
    attempted_repairs: int
    control_passes: int
    patched_passes: int
    control_only: int
    patched_only: int
    audited: int
    audit_passes: int
    acceptance_rate: float | None
    control_rate: float | None
    repair_rate: float | None
    delta_repair: float | None
    quality_rate: float | None
    mcnemar_pvalue: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_latest_results(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line_number, line in enumerate(read_jsonl_lines(path), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL in {path} at line {line_number}") from exc
        if not isinstance(row, dict):
            raise TypeError(f"JSONL row in {path} at line {line_number} must be an object")
        sample_id = row.get("sample_id")
        if sample_id is not None:
            rows[str(sample_id)] = row
    return rows


def source_metrics(batch_dir: Path) -> SourceMetrics:
    rows = load_latest_results(batch_dir / "results.jsonl")
    valid_rows: list[dict[str, Any]] = []
    invalid_ids: list[str] = []
    failure_ids: list[str] = []
    completed_ids: list[str] = []
    correct = 0
    for sample_id, row in rows.items():
        evaluation = row.get("evaluation") if isinstance(row.get("evaluation"), dict) else {}
        resolved = isinstance(evaluation.get("correct"), bool)
        if row.get("status") != "completed" or not resolved:
            invalid_ids.append(sample_id)
            continue
        valid_rows.append(row)
        completed_ids.append(sample_id)
        if evaluation["correct"] is True:
            correct += 1
        else:
            failure_ids.append(sample_id)

    total = len(rows)
    invalid = len(invalid_ids)
    valid = len(valid_rows)
    failures = len(failure_ids)
    failure_rate = failures / valid if valid else None
    ci = wilson_interval(failures, valid)
    if valid == 0:
        triage = "insufficient"
    elif ci[1] < 0.15:
        triage = "ceiling"
    elif ci[0] > 0.70:
        triage = "floor"
    else:
        triage = "candidate"
    return SourceMetrics(
        total=total,
        infrastructure_invalid=invalid,
        valid=valid,
        cognitive_failures=failures,
        correct=correct,
        infrastructure_rate=invalid / total if total else None,
        cognitive_failure_rate=failure_rate,
        cognitive_failure_ci=ci,
        triage=triage,
        completed_sample_ids=tuple(completed_ids),
        invalid_sample_ids=tuple(invalid_ids),
        failure_sample_ids=tuple(failure_ids),
    )


def repair_metrics(
    *,
    cognitive_failures: int,
    eligible_failures: int,
    records: list[dict[str, Any]],
    audit_pass_statuses: set[str] | None = None,
) -> RepairMetrics:
    pass_statuses = audit_pass_statuses or {"passed", "approved", "accepted"}
    attempted = [
        row
        for row in records
        if isinstance(row.get("outcome"), dict)
        and row["outcome"].get("repair_attempted") is True
    ]
    control_passes = sum(row["outcome"].get("control_success") is True for row in attempted)
    patched_passes = sum(row["outcome"].get("repair_success") is True for row in attempted)
    paired = [str(row["outcome"].get("paired_outcome") or "") for row in attempted]
    control_only = paired.count("control_only")
    patched_only = paired.count("patched_only")
    audit_statuses = [
        str((row.get("quality") or {}).get("audit_status") or "not_audited").lower()
        for row in attempted
    ]
    audited_statuses = [
        status
        for status in audit_statuses
        if status not in {"", "not_audited", "pending", "auto_labeled"}
    ]
    audit_passes = sum(status in pass_statuses for status in audited_statuses)
    control_rate = control_passes / len(attempted) if attempted else None
    patched_rate = patched_passes / len(attempted) if attempted else None
    return RepairMetrics(
        cognitive_failures=cognitive_failures,
        eligible_failures=eligible_failures,
        attempted_repairs=len(attempted),
        control_passes=control_passes,
        patched_passes=patched_passes,
        control_only=control_only,
        patched_only=patched_only,
        audited=len(audited_statuses),
        audit_passes=audit_passes,
        acceptance_rate=(eligible_failures / cognitive_failures if cognitive_failures else None),
        control_rate=control_rate,
        repair_rate=patched_rate,
        delta_repair=(
            patched_rate - control_rate
            if patched_rate is not None and control_rate is not None
            else None
        ),
        quality_rate=(audit_passes / len(audited_statuses) if audited_statuses else None),
        mcnemar_pvalue=(
            one_sided_exact_mcnemar(control_only, patched_only) if attempted else None
        ),
    )


def passes_window_gates(source: SourceMetrics, repair: RepairMetrics) -> tuple[bool, list[str]]:
    failures: list[str] = []
    checks = (
        (source.cognitive_failure_rate is not None and 0.20 <= source.cognitive_failure_rate <= 0.60, "F_cog"),
        (source.infrastructure_rate is not None and source.infrastructure_rate <= 0.05, "I"),
        (repair.acceptance_rate is not None and repair.acceptance_rate >= 0.70, "A"),
        (repair.repair_rate is not None and repair.repair_rate >= 0.30, "R"),
        (repair.delta_repair is not None and repair.delta_repair > 0.0, "DeltaR"),
        (repair.quality_rate is not None and repair.quality_rate >= 0.80, "Q"),
        (repair.mcnemar_pvalue is not None and repair.mcnemar_pvalue < 0.05, "McNemar"),
    )
    for passed, name in checks:
        if not passed:
            failures.append(name)
    return (not failures, failures)


def attributable_density(
    source: SourceMetrics,
    repair: RepairMetrics,
    gamma: float | None,
) -> float | None:
    values = (
        source.cognitive_failure_rate,
        repair.acceptance_rate,
        repair.delta_repair,
        repair.quality_rate,
        gamma,
    )
    if any(value is None for value in values) or gamma is None or gamma <= 0:
        return None
    return (
        float(source.cognitive_failure_rate)
        * float(repair.acceptance_rate)
        * float(repair.delta_repair)
        * float(repair.quality_rate)
        / gamma
    )
