"""Compare training and held-out tasks across all ARIA record layouts."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable


def task_keys(row: Mapping[str, Any]) -> set[str]:
    """Use task IDs and normalized task text; incident IDs differ across hosts."""
    metadata = row.get('metadata') or {}
    incident = row.get('incident') or row.get('input') or row
    constraints = incident.get('constraints') or {}
    cell = row.get('cell') or {}
    refs = row.get('source_refs') or {}
    benchmark = constraints.get('benchmark') or metadata.get('benchmark_id') or cell.get('benchmark_id') or refs.get('benchmark_id')
    keys = set()
    task_id = metadata.get('task_id') or metadata.get('source_task_id') or row.get('task_id')
    if benchmark and task_id:
        keys.add(f'id:{benchmark}:{task_id}')
    text = constraints.get('task_text') or (row.get('sample') or {}).get('task_text') or (row.get('evaluator_sample') or {}).get('task')
    if benchmark and isinstance(text, str) and text.strip():
        digest = hashlib.sha256(' '.join(text.split()).encode()).hexdigest()
        keys.add(f'text:{benchmark}:{digest}')
    if not keys:
        raise ValueError('split audit requires benchmark and task ID or task text')
    return keys


def rows_from_path(path: Path) -> Iterable[dict[str, Any]]:
    paths = [path / 'sft.jsonl', path / 'dpo.jsonl'] if path.is_dir() else [path]
    for source in paths:
        with source.open(encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)


def audit_disjoint(training_paths: list[Path], holdout_paths: list[Path]) -> dict[str, Any]:
    heldout = set()
    for path in holdout_paths:
        for row in rows_from_path(path):
            heldout.update(task_keys(row))
    results = []
    for path in training_paths:
        count = 0
        overlap_rows = 0
        overlaps = set()
        for row in rows_from_path(path):
            count += 1
            matched = task_keys(row) & heldout
            if matched:
                overlap_rows += 1
                overlaps.update(matched)
        results.append({'path': str(path), 'rows': count, 'overlap_rows': overlap_rows,
                        'overlap_keys': len(overlaps), 'examples': sorted(overlaps)[:10]})
    return {'schema_version': 'aria-split-audit-v1',
            'status': 'failed' if any(r['overlap_rows'] for r in results) else 'passed',
            'holdout_paths': [str(p) for p in holdout_paths], 'training': results}
