"""Deterministic, bounded training balance without changing outcome labels."""

from __future__ import annotations

import math
import hashlib
import json
import random
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from typing import Any


def cell_key(row: Mapping[str, Any]) -> tuple[str, str]:
    metadata = row.get("metadata") or {}
    return (str(metadata.get("mas_id") or "unknown"),
            str(metadata.get("benchmark_id") or "unknown"))


def balanced_take(rows: Sequence[dict[str, Any]], limit: int, *, seed: int) -> list[dict[str, Any]]:
    """Rotate cells/labels and incident lineages before taking another prefix."""
    if limit < 1:
        raise ValueError("training row limit must be positive")
    buckets = defaultdict(lambda: defaultdict(list))
    for row in rows:
        label = (row.get("target") or {}).get("gate_probability", "non-gate")
        buckets[(*cell_key(row), str(label))][row["incident_id"]].append(row)
    rng = random.Random(seed)
    cells = list(sorted(buckets))
    rng.shuffle(cells)
    queues = {}
    for cell in cells:
        incidents = list(sorted(buckets[cell]))
        rng.shuffle(incidents)
        lineage = []
        for incident in incidents:
            items = sorted(buckets[cell][incident], key=lambda r: r["record_id"])
            rng.shuffle(items)
            lineage.append(deque(items))
        queues[cell] = deque(lineage)
    selected = []
    while cells and len(selected) < limit:
        pending = []
        for cell in cells:
            incident = queues[cell].popleft()
            selected.append(incident.popleft())
            if incident:
                queues[cell].append(incident)
            if queues[cell]:
                pending.append(cell)
            if len(selected) == limit:
                break
        cells = pending
    return selected


def loss_weights(rows: Sequence[Mapping[str, Any]], *, mode: str, max_ratio: float = 4.0) -> dict[str, float]:
    """Equalize tasks and temper cell imbalance with a capped square-root weight."""
    if mode not in {"none", "balanced_cell_task"}:
        raise ValueError("unknown training row weighting")
    if not math.isfinite(max_ratio) or max_ratio < 1:
        raise ValueError("max training weight ratio must be finite and at least one")
    keys = []
    for row in rows:
        key = row.get("record_id") or row.get("incident_id") or (row.get("input") or {}).get("incident_id")
        keys.append(str(key) if key else hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
    if len(set(keys)) != len(keys):
        raise ValueError("training weights require unique record IDs")
    if not rows or mode == "none":
        return dict.fromkeys(keys, 1.0)
    tasks = Counter(str(row["task"]) for row in rows)
    cells = Counter((str(row["task"]), cell_key(row)) for row in rows)
    cell_counts = Counter(task for task, _ in cells)
    weights = []
    for row in rows:
        task = str(row["task"])
        task_weight = len(rows) / (len(tasks) * tasks[task])
        cell_weight = math.sqrt(tasks[task] / (cell_counts[task] * cells[task, cell_key(row)]))
        weights.append(min(max_ratio, max(1.0 / max_ratio, task_weight * cell_weight)))
    mean = sum(weights) / len(weights)
    return {key: weight / mean for key, weight in zip(keys, weights)}
