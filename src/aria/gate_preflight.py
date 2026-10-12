"""Static preflight checks for ARIA deployment-gate corpora.

The gate is a binary prefix classifier.  A corpus that contains almost no
positive prefixes can produce a perfectly finite SFT loss while learning the
degenerate policy "never intervene".  These checks run before any optimizer
stage and make that failure mode explicit in the training manifest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from aria.corpus import load_rows, summarize_gate_rows


def gate_preflight(
    train_path: Path,
    validation_path: Path | None = None,
    *,
    min_positive_rate: float = 0.20,
    max_positive_rate: float = 0.80,
    min_rows: int = 30,
    require_both_labels_per_cell: bool = True,
) -> dict[str, Any]:
    """Validate train/validation gate label support and return an audit report."""

    if not 0.0 < min_positive_rate <= 0.5:
        raise ValueError("min_positive_rate must lie in (0, 0.5]")
    if not 0.5 <= max_positive_rate < 1.0:
        raise ValueError("max_positive_rate must lie in [0.5, 1)")
    if min_positive_rate > max_positive_rate:
        raise ValueError("min_positive_rate must not exceed max_positive_rate")
    train_rows = load_rows(Path(train_path))
    validation_rows = (
        load_rows(Path(validation_path)) if validation_path is not None else []
    )
    report: dict[str, Any] = {
        "schema_version": "aria-gate-preflight-v1",
        "train_path": str(Path(train_path).resolve()),
        "validation_path": (
            str(Path(validation_path).resolve()) if validation_path is not None else None
        ),
        "thresholds": {
            "min_positive_rate": min_positive_rate,
            "max_positive_rate": max_positive_rate,
            "min_rows": min_rows,
            "require_both_labels_per_cell": require_both_labels_per_cell,
        },
        "train": summarize_gate_rows(train_rows),
        "validation": (
            summarize_gate_rows(validation_rows) if validation_path is not None else None
        ),
        "violations": [],
        "status": "passed",
    }
    violations: list[str] = []
    for name, summary, required in (
        ("train", report["train"], True),
        ("validation", report["validation"], validation_path is not None),
    ):
        if not required:
            continue
        assert isinstance(summary, Mapping)
        rows = int(summary.get("rows") or 0)
        if rows < min_rows:
            violations.append(f"{name} gate rows {rows} < required minimum {min_rows}")
        rate = summary.get("positive_rate")
        if not isinstance(rate, (int, float)) or not (
            min_positive_rate <= float(rate) <= max_positive_rate
        ):
            violations.append(
                f"{name} gate positive rate {rate!r} is outside "
                f"[{min_positive_rate}, {max_positive_rate}]"
            )
        positions = summary.get("positions") or {}
        missing_positions = [
            position for position in ("early", "middle", "late") if not positions.get(position)
        ]
        if missing_positions:
            violations.append(f"{name} gate is missing prefix positions {missing_positions}")
        if require_both_labels_per_cell:
            for cell, labels in (summary.get("label_by_source_cell") or {}).items():
                if not all(str(label) in labels for label in (0, 1)):
                    violations.append(
                        f"{name} source cell {cell!r} does not contain both gate labels"
                    )
    report["violations"] = violations
    if violations:
        report["status"] = "failed"
        raise ValueError(
            "ARIA gate preflight failed:\n- " + "\n- ".join(violations)
        )
    return report


__all__ = ["gate_preflight"]
