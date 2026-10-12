"""Corpus partitioning and held-out cell views for the ARIA experiment grid.

Section~\\ref{subsec:exp-setup} treats the corpus as one dataset partitioned
once, at the level of the task identifier, into 60/20/20 train/validation/test
stratified over source cells, and defines two generalization axes that withhold
a *property* of the source cell rather than a set of tasks.  Both operations
live here so that a partition is produced once and every downstream consumer -
post-training, ablation, and evaluation - reads the same assignment.

A source cell is the triple ``(mas_id, benchmark_id, host_model_id)`` of
Section~\\ref{subsec:dataset-creation}.  ``host_model_id`` is not stored as its
own field by the corpus builder, so it is recovered from the source dataset
identifier, which encodes the selected host model of the cell.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SPLIT_NAMES = ("train", "validation", "test")
SPLIT_DIRECTORY_NAMES = {"train": "train", "validation": "val", "test": "test"}
DEFAULT_SPLIT_PROPORTIONS = (0.60, 0.20, 0.20)
CORPUS_SPLIT_SCHEMA_VERSION = "aria-corpus-split-v2"
CORPUS_VIEW_SCHEMA_VERSION = "aria-corpus-view-v1"

# ``repair-v4-selected-<mas>-<benchmark>-<runtime>-<host model>`` is the
# identifier the dataset builder writes; the host model is the suffix after the
# runtime token.  A dataset id that does not match leaves the host model
# unknown, which is reported rather than guessed.
_SOURCE_DATASET_PATTERN = re.compile(
    r"^repair-v\d+-(?:selected|manual)-(?P<rest>.+)$"
)
_RUNTIME_TOKENS = ("callapi", "vllm", "endpoint", "api")
_HOST_TRANSPORT_PREFIXES = (
    "openrouter-siliconflow-",
    "openrouter-",
    "siliconflow-",
)

__all__ = [
    "CORPUS_SPLIT_SCHEMA_VERSION",
    "CORPUS_VIEW_SCHEMA_VERSION",
    "DEFAULT_SPLIT_PROPORTIONS",
    "format_split_proportions",
    "SPLIT_DIRECTORY_NAMES",
    "SPLIT_NAMES",
    "HeldOutAxis",
    "SourceCell",
    "assign_splits",
    "build_corpus_view",
    "cell_of_row",
    "host_model_of_row",
    "load_rows",
    "row_task_id",
    "summarize_gate_rows",
    "split_aria_full",
    "write_jsonl",
]


@dataclass(frozen=True)
class SourceCell:
    """The ``(A, B, M)`` triple that a corpus row was produced by."""

    mas_id: str
    benchmark_id: str
    host_model_id: str

    def key(self) -> str:
        return f"{self.mas_id}/{self.benchmark_id}/{self.host_model_id}"


@dataclass(frozen=True)
class HeldOutAxis:
    """One RQ2 generalization axis, withholding a source-cell property."""

    name: str
    field: str
    values: tuple[str, ...]

    def excludes(self, cell: SourceCell) -> bool:
        return getattr(cell, self.field) in self.values

    @classmethod
    def parse(cls, name: str, values: Sequence[str]) -> HeldOutAxis:
        field = {
            "unseen_system": "mas_id",
            "unseen_host_model": "host_model_id",
            "unseen_benchmark": "benchmark_id",
        }.get(name)
        if field is None:
            raise ValueError(
                "held-out axis must be unseen_system, unseen_host_model, or "
                f"unseen_benchmark; got {name!r}"
            )
        cleaned = tuple(str(value) for value in values if str(value).strip())
        if not cleaned:
            raise ValueError(f"held-out axis {name} requires at least one value")
        return cls(name=name, field=field, values=cleaned)


def host_model_of_row(row: Mapping[str, Any]) -> str:
    """Recover the host model of a row's source cell from its dataset id."""

    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        return "unknown"
    explicit = metadata.get("host_model_id")
    if isinstance(explicit, str) and explicit.strip():
        return explicit
    dataset_ids = metadata.get("source_dataset_ids")
    if not isinstance(dataset_ids, list) or not dataset_ids:
        return "unknown"
    mas_id = str(metadata.get("mas_id") or "")
    benchmark_id = str(metadata.get("benchmark_id") or "")
    resolved = {
        _host_model_of_dataset_id(
            str(value), mas_id=mas_id, benchmark_id=benchmark_id
        )
        for value in dataset_ids
    }
    resolved.discard("unknown")
    if len(resolved) != 1:
        # A row assembled from several cells has no single host model; treating
        # it as one would silently leak a withheld model into a held-out run.
        return "unknown"
    return resolved.pop()


def _host_model_of_dataset_id(
    dataset_id: str, *, mas_id: str = "", benchmark_id: str = ""
) -> str:
    dataset_id = re.sub(r"endpoint-([abc])(?=-)", r"endpoint\1", dataset_id)
    match = _SOURCE_DATASET_PATTERN.match(dataset_id)
    if match is None:
        return "unknown"
    rest = match.group("rest")

    # V6 selected and manual dataset identities both carry the source cell,
    # but their transport tokens occur on opposite sides of it:
    #
    #   selected-<mas>-<benchmark>-endpoint-b-<host>
    #   selected-<mas>-<benchmark>-openrouter-siliconflow-<host>
    #   manual-vllm-<mas>-<benchmark>-<host>-full
    #
    # MAS and benchmark names contain hyphens, so splitting the identifier and
    # looking for the first runtime-looking token cannot recover the boundary.
    # The row already stores the canonical MAS and benchmark; use those exact
    # values to remove the cell prefix before interpreting the transport.
    if mas_id and benchmark_id:
        manual_prefix = f"vllm-{mas_id}-{benchmark_id}-"
        selected_prefix = f"{mas_id}-{benchmark_id}-"
        if rest.startswith(manual_prefix):
            host = rest[len(manual_prefix) :]
            return host.removesuffix("-full") or "unknown"
        if rest.startswith(selected_prefix):
            host = rest[len(selected_prefix) :]
            tokens = host.split("-")
            if tokens and any(
                tokens[0].startswith(prefix) for prefix in _RUNTIME_TOKENS
            ):
                host = "-".join(tokens[1:])
            else:
                for prefix in _HOST_TRANSPORT_PREFIXES:
                    if host.startswith(prefix):
                        host = host[len(prefix) :]
                        break
            return host or "unknown"

    # Backward-compatible fallback for older selected identities whose rows do
    # not carry explicit cell metadata.
    tokens = rest.split("-")
    for index, token in enumerate(tokens):
        if any(token.startswith(prefix) for prefix in _RUNTIME_TOKENS):
            suffix = tokens[index + 1 :]
            return "-".join(suffix) if suffix else "unknown"
    return "unknown"


def cell_of_row(row: Mapping[str, Any]) -> SourceCell:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    return SourceCell(
        mas_id=str(metadata.get("mas_id") or "unknown"),
        benchmark_id=str(metadata.get("benchmark_id") or "unknown"),
        host_model_id=host_model_of_row(row),
    )


def row_task_id(row: Mapping[str, Any]) -> str:
    """Return the task identifier a row belongs to.

    Partitioning at the task identifier is what keeps a repaired trajectory and
    its own failed continuation from landing on opposite sides of a split.  The
    training-ready row carries the incident identifier, whose trailing UUID is
    the source task; the dataset prefix is stripped so the same benchmark task
    observed under two cells is deduplicated to one identifier.
    """

    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("task_id", "source_task_id"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value
    incident_id = row.get("incident_id")
    if isinstance(incident_id, str) and incident_id.strip():
        return _task_id_from_incident(incident_id)
    record_id = row.get("record_id")
    if isinstance(record_id, str) and record_id.strip():
        return record_id.split("::", 1)[0]
    raise ValueError("ARIA corpus row carries neither incident_id nor record_id")


_UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)


def _task_id_from_incident(incident_id: str) -> str:
    match = _UUID_PATTERN.search(incident_id)
    return match.group(0) if match else incident_id


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{number} must be a JSON object")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    temporary.replace(path)
    return count


def assign_splits(
    rows: Sequence[Mapping[str, Any]],
    *,
    proportions: Sequence[float] = DEFAULT_SPLIT_PROPORTIONS,
    seed: int = 0,
) -> dict[str, str]:
    """Partition task identifiers 60/20/20, stratified over source cells.

    Assignment is deterministic in ``seed`` and in the task identifier, so a
    re-run of the builder reproduces the partition without storing per-row
    state, and a task identifier that appears under two cells receives one
    assignment rather than two.  Stratification is applied per cell, with each
    cell's tasks ordered by a seeded hash and cut at the requested proportions,
    which keeps every cell's corpus proportion in each partition even when the
    cell is small.
    """

    if len(proportions) != 3:
        raise ValueError("split proportions must give train, validation, and test")
    if abs(sum(proportions) - 1.0) > 1e-9 or any(value < 0 for value in proportions):
        raise ValueError("split proportions must be non-negative and sum to one")

    tasks_by_cell: dict[str, set[str]] = defaultdict(set)
    cell_of_task: dict[str, str] = {}
    for row in rows:
        task_id = row_task_id(row)
        cell = cell_of_row(row).key()
        # A task seen under several cells is stratified under the first cell in
        # sorted order, so the assignment does not depend on row ordering.
        previous = cell_of_task.get(task_id)
        if previous is None or cell < previous:
            cell_of_task[task_id] = cell
    for task_id, cell in cell_of_task.items():
        tasks_by_cell[cell].add(task_id)

    assignment: dict[str, str] = {}
    for cell, task_ids in sorted(tasks_by_cell.items()):
        ordered = sorted(task_ids, key=lambda value: _split_rank(seed, cell, value))
        total = len(ordered)
        train_end = round(total * proportions[0])
        validation_end = train_end + round(total * proportions[1])
        # A cell too small to fill three partitions keeps every task in train
        # rather than emitting a one-row test cell that no metric can read.
        if total < 3:
            train_end, validation_end = total, total
        else:
            train_end = min(max(train_end, 1), total - 2)
            validation_end = min(max(validation_end, train_end + 1), total - 1)
        for index, task_id in enumerate(ordered):
            if index < train_end:
                assignment[task_id] = "train"
            elif index < validation_end:
                assignment[task_id] = "validation"
            else:
                assignment[task_id] = "test"
    return assignment


def _split_rank(seed: int, cell: str, task_id: str) -> str:
    digest = hashlib.sha256(f"{seed}:{cell}:{task_id}".encode()).hexdigest()
    return digest


def build_corpus_view(
    source_dir: Path,
    output_dir: Path,
    *,
    split: str | None = None,
    splits: Sequence[str] | None = None,
    held_out: HeldOutAxis | None = None,
    seed: int = 0,
    proportions: Sequence[float] = DEFAULT_SPLIT_PROPORTIONS,
    max_rows_per_task: Mapping[str, int] | None = None,
    task_filter: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Materialize one filtered corpus directory plus its provenance manifest.

    ``held_out`` removes every row whose source cell carries the withheld
    property, which is how RQ2 trains a checkpoint from which those cells are
    absent.  ``max_rows_per_task`` caps a task family - in practice the
    ``deployment_gate`` family, which outnumbers every other family by two
    orders of magnitude - by a deterministic per-incident stride rather than a
    prefix cut, so the cap preserves the prefix-position distribution the gate
    is trained on.
    """

    requested_splits = _requested_splits(split, splits)
    sft_rows = load_rows(source_dir / "sft.jsonl")
    dpo_rows = load_rows(source_dir / "dpo.jsonl")
    assignment = assign_splits(
        [*sft_rows, *dpo_rows], proportions=proportions, seed=seed
    )

    dropped_held_out = 0
    dropped_split = 0
    dropped_task = 0
    kept: dict[str, list[dict[str, Any]]] = {"sft": [], "dpo": []}
    for name, rows in (("sft", sft_rows), ("dpo", dpo_rows)):
        for row in rows:
            task_id = row_task_id(row)
            assigned = assignment[task_id]
            if requested_splits is not None and assigned not in requested_splits:
                dropped_split += 1
                continue
            if held_out is not None and held_out.excludes(cell_of_row(row)):
                dropped_held_out += 1
                continue
            if task_filter is not None and str(row.get("task")) not in task_filter:
                dropped_task += 1
                continue
            annotated = dict(row)
            metadata = dict(annotated.get("metadata") or {})
            cell = cell_of_row(row)
            metadata.update(
                {
                    "split": assigned,
                    "corpus_split": assigned,
                    "task_id": task_id,
                    "host_model_id": cell.host_model_id,
                    "source_cell": cell.key(),
                }
            )
            annotated["metadata"] = metadata
            kept[name].append(annotated)

    capped_counts: dict[str, int] = {}
    sampling_audits: dict[str, Any] = {}
    if max_rows_per_task:
        kept["sft"], capped_counts, sampling_audits = _cap_task_families(
            kept["sft"], max_rows_per_task, seed=seed
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        "sft": write_jsonl(output_dir / "sft.jsonl", kept["sft"]),
        "dpo": write_jsonl(output_dir / "dpo.jsonl", kept["dpo"]),
    }
    manifest = {
        "schema_version": CORPUS_VIEW_SCHEMA_VERSION,
        "dataset_schema_version": "aria-training-ready-v3",
        "source_dir": str(source_dir),
        "splits": list(requested_splits) if requested_splits else list(SPLIT_NAMES),
        "split_seed": seed,
        "split_proportions": list(proportions),
        "held_out_axis": (
            {"name": held_out.name, "field": held_out.field, "values": list(held_out.values)}
            if held_out is not None
            else None
        ),
        "counts": counts,
        "dropped": {
            "other_split": dropped_split,
            "held_out_cell": dropped_held_out,
            "other_task_family": dropped_task,
        },
        "task_family_caps": dict(max_rows_per_task or {}),
        "task_family_counts": _task_counts(kept["sft"]),
        "capped_task_families": capped_counts,
        "task_family_sampling": sampling_audits,
        "split_assignment_counts": _assignment_counts(assignment),
        "cells": sorted({cell_of_row(row).key() for row in kept["sft"]}),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def split_aria_full(
    source_dir: Path,
    output_dir: Path,
    *,
    seed: int = 0,
    proportions: Sequence[float] = DEFAULT_SPLIT_PROPORTIONS,
) -> dict[str, Any]:
    """Materialize ARIA-Full as separate train/val/test directories.

    The aggregate release stores all rows in two JSONL files. This helper
    applies :func:`assign_splits` once over both files, then streams rows into
    three views. Assignment is at task level, so every prefix and preference
    pair for a task stays in one split.
    """

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    source_paths = {name: source_dir / f"{name}.jsonl" for name in ("sft", "dpo")}
    missing = [str(path) for path in source_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "ARIA-Full source is missing required JSONL file(s): "
            + ", ".join(missing)
        )

    # Keep only the metadata needed to compute the deterministic assignment;
    # the full payload is parsed again while writing each view.
    assignment_rows: list[dict[str, Any]] = []
    for path in source_paths.values():
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(f"{path}:{line_number} must be a JSON object")
                metadata = row.get("metadata")
                metadata = metadata if isinstance(metadata, Mapping) else {}
                assignment_rows.append(
                    {
                        "incident_id": row.get("incident_id"),
                        "record_id": row.get("record_id"),
                        "metadata": {
                            key: metadata.get(key)
                            for key in (
                                "task_id",
                                "source_task_id",
                                "mas_id",
                                "benchmark_id",
                                "source_dataset_ids",
                            )
                            if key in metadata
                        },
                    }
                )
    assignment = assign_splits(
        assignment_rows, proportions=proportions, seed=seed
    )

    split_counts: dict[str, dict[str, int]] = {
        split: {"sft": 0, "dpo": 0, "tasks": 0} for split in SPLIT_NAMES
    }
    tasks_by_split: dict[str, set[str]] = {split: set() for split in SPLIT_NAMES}
    statistics = _SplitStatistics()
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_NAMES:
        split_dir = output_dir / SPLIT_DIRECTORY_NAMES[split]
        split_dir.mkdir(parents=True, exist_ok=True)
        for name, path in source_paths.items():
            _stream_split_jsonl(
                path,
                split_dir / f"{name}.jsonl",
                split=split,
                assignment=assignment,
                tasks=tasks_by_split[split],
                count=split_counts[split],
                kind=name,
                statistics=statistics,
            )
        split_counts[split]["tasks"] = len(tasks_by_split[split])
        split_manifest = {
            "schema_version": CORPUS_VIEW_SCHEMA_VERSION,
            "dataset_schema_version": "aria-training-ready-v3",
            "dataset_id": "ARIA-Full",
            "source_dir": str(source_dir.resolve()),
            "split": split,
            "directory": SPLIT_DIRECTORY_NAMES[split],
            "split_seed": seed,
            "split_proportions": list(proportions),
            "counts": dict(split_counts[split]),
            "repair_samples": statistics.sample_count(split),
        }
        (split_dir / "manifest.json").write_text(
            json.dumps(split_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    stats_markdown = statistics.render_markdown(
        source_dir=source_dir,
        seed=seed,
        proportions=proportions,
    )
    _write_text_atomic(output_dir / "STATS.md", stats_markdown)

    manifest = {
        "schema_version": CORPUS_SPLIT_SCHEMA_VERSION,
        "dataset_schema_version": "aria-training-ready-v3",
        "dataset_id": "ARIA-Full",
        "source_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "split_seed": seed,
        "split_proportions": list(proportions),
        "split_directories": {
            split: SPLIT_DIRECTORY_NAMES[split] for split in SPLIT_NAMES
        },
        "counts": split_counts,
        "repair_samples": {
            split: statistics.sample_count(split) for split in SPLIT_NAMES
        },
        "task_counts": {
            split: len(tasks) for split, tasks in tasks_by_split.items()
        },
        "assignment_counts": _assignment_counts(assignment),
        "splits": list(SPLIT_NAMES),
        "stats_path": str((output_dir / "STATS.md").resolve()),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


class _SplitStatistics:
    """Collect split-aware sample and training-row statistics while streaming."""

    def __init__(self) -> None:
        self.training_rows: dict[tuple[str, str, str], int] = defaultdict(int)
        self.tasks_by_split: dict[str, set[str]] = {
            split: set() for split in SPLIT_NAMES
        }
        self.samples: dict[tuple[str, str, str], dict[str, Any]] = {}

    def observe(
        self, row: Mapping[str, Any], *, kind: str, split: str
    ) -> None:
        cell = _mas_benchmark_key(row)
        self.training_rows[(kind, split, cell)] += 1
        self.tasks_by_split[split].add(row_task_id(row))
        if kind != "sft":
            return

        metadata = row.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        sample_id = _sample_id(row)
        key = (split, cell, sample_id)
        if key in self.samples:
            return

        record_type = str(metadata.get("record_type") or "unknown")
        paired_outcome = str(metadata.get("paired_outcome") or "")
        if not paired_outcome:
            paired_outcome = (
                "not_applicable"
                if record_type == "gate_success_sample"
                else "unresolved"
            )
        repair_success = metadata.get("repair_success")
        if not isinstance(repair_success, bool):
            repair_success = paired_outcome in {"patched_only", "both_pass"}
        control_success = metadata.get("control_success")
        if not isinstance(control_success, bool):
            control_success = paired_outcome in {"control_only", "both_pass"}
        repair_attempted = metadata.get("repair_attempted")
        if not isinstance(repair_attempted, bool):
            repair_attempted = record_type != "gate_success_sample"
        self.samples[key] = {
            "split": split,
            "cell": cell,
            "sample_id": sample_id,
            "record_type": record_type,
            "repair_attempted": repair_attempted,
            "repair_success": repair_success,
            "verified_successful": metadata.get("verified_successful") is True,
            "control_success": control_success,
            "paired_outcome": paired_outcome,
        }

    def sample_count(self, split: str | None) -> int:
        return sum(
            split is None or item["split"] == split
            for item in self.samples.values()
        )

    def metric_count(self, metric: str, split: str | None = None) -> int:
        return sum(
            bool(item.get(metric))
            for item in self.samples.values()
            if split is None or item["split"] == split
        )

    def outcome_count(self, outcome: str, split: str | None = None) -> int:
        return sum(
            item["paired_outcome"] == outcome
            for item in self.samples.values()
            if split is None or item["split"] == split
        )

    def matrix(
        self, metric: str, *, kind: str | None = None
    ) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {split: 0 for split in SPLIT_NAMES}
        )
        if metric == "samples":
            for item in self.samples.values():
                counts[item["cell"]][item["split"]] += 1
            return dict(counts)

        for (row_kind, split, cell), count in self.training_rows.items():
            if kind is not None and row_kind != kind:
                continue
            counts[cell][split] += count
        return dict(counts)

    def training_row_count(self, split: str | None, *, kind: str | None = None) -> int:
        return sum(
            count
            for (row_kind, row_split, _), count in self.training_rows.items()
            if (kind is None or row_kind == kind)
            and (split is None or row_split == split)
        )

    def gate_success_count(self, split: str | None = None) -> int:
        return sum(
            item["record_type"] == "gate_success_sample"
            for item in self.samples.values()
            if split is None or item["split"] == split
        )

    def cell_repair_metrics(self) -> dict[str, dict[str, int]]:
        metrics: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "records": 0,
                "attempted_repairs": 0,
                "repair_successes": 0,
                "verified_successes": 0,
                "control_successes": 0,
                "gate_success_samples": 0,
            }
        )
        for item in self.samples.values():
            cell_metrics = metrics[item["cell"]]
            cell_metrics["records"] += 1
            cell_metrics["attempted_repairs"] += int(item["repair_attempted"])
            cell_metrics["repair_successes"] += int(item["repair_success"])
            cell_metrics["verified_successes"] += int(item["verified_successful"])
            cell_metrics["control_successes"] += int(item["control_success"])
            cell_metrics["gate_success_samples"] += int(
                item["record_type"] == "gate_success_sample"
            )
        for cell, cell_metrics in metrics.items():
            cell_metrics["patched_only"] = self._cell_outcome_count(
                cell, "patched_only"
            )
        return dict(metrics)

    def _cell_outcome_count(self, cell: str, outcome: str) -> int:
        return sum(
            item["cell"] == cell and item["paired_outcome"] == outcome
            for item in self.samples.values()
        )

    def render_markdown(
        self,
        *,
        source_dir: Path,
        seed: int,
        proportions: Sequence[float],
    ) -> str:
        cells = set(self.matrix("samples")) | set(self.matrix("training_rows"))
        mas_order, benchmark_order = _matrix_axes(cells)
        sample_counts = self.matrix("samples")
        training_counts = self.matrix("training_rows")
        sft_counts = self.matrix("training_rows", kind="sft")
        dpo_counts = self.matrix("training_rows", kind="dpo")
        split_ratio = format_split_proportions(proportions)
        lines = [
            "# ARIA-Full-Processed Statistics",
            "",
            f"Generated at: `{_utc_now()}`",
            "",
            "## Scope",
            "",
            f"Source dataset: `{source_dir.resolve()}`.",
            "",
            "The split is assigned deterministically at task/incident level and "
            f"stratified by source cell with seed `{seed}` and proportions "
            f"`{split_ratio}` (train/val/test). All rows for one task remain in "
            "the same split.",
            "",
            "Counts in the first matrix are unique repair samples. Counts in the "
            "training-row matrices count projected rows, including deployment "
            "gate prefixes.",
            "",
            "## Repair Samples",
            "",
            "Unique source repair samples by MAS and benchmark (`train/val/test`):",
            "",
        ]
        lines.extend(_matrix_markdown(sample_counts, mas_order, benchmark_order))
        lines.extend(
            [
                "",
                "## Training Rows",
                "",
                "Projected `SFT + DPO` rows by MAS and benchmark (`train/val/test`):",
                "",
            ]
        )
        lines.extend(_matrix_markdown(training_counts, mas_order, benchmark_order))
        lines.extend(
            [
                "",
                "### SFT Rows",
                "",
            ]
        )
        lines.extend(_matrix_markdown(sft_counts, mas_order, benchmark_order))
        lines.extend(
            [
                "",
                "### DPO Rows",
                "",
            ]
        )
        lines.extend(_matrix_markdown(dpo_counts, mas_order, benchmark_order))

        lines.extend(
            [
                "",
                "## Overall Counts",
                "",
                "| Metric | Total | Train | Val | Test |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        overall_rows = [
            ("Unique repair samples", self._sample_counts()),
            ("Repair attempts", self._metric_counts("repair_attempted")),
            (
                "Repair successes (`repair_success=true`)",
                self._metric_counts("repair_success"),
            ),
            (
                "Host-verified successful repairs",
                self._metric_counts("verified_successful"),
            ),
            ("Control successes", self._metric_counts("control_success")),
            ("Gate-success samples", self._gate_success_counts()),
            ("SFT rows", self._training_row_counts(kind="sft")),
            ("DPO rows", self._training_row_counts(kind="dpo")),
            ("Total training rows", self._training_row_counts()),
            ("Tasks", self._task_counts()),
        ]
        for label, counts in overall_rows:
            lines.append(
                f"| {label} | {counts['total']} | {counts['train']} | "
                f"{counts['validation']} | {counts['test']} |"
            )

        outcomes = {
            outcome: self.outcome_count(outcome)
            for outcome in (
                "patched_only",
                "both_pass",
                "control_only",
                "both_fail",
                "unresolved",
            )
        }
        lines.extend(
            [
                "",
                "## Repair Outcomes",
                "",
                "Repair success is inferred from `paired_outcome` when the "
                "projected row does not carry an explicit boolean. `Verified` "
                "requires host verification.",
                "",
                "| Outcome | Samples |",
                "| --- | ---: |",
            ]
        )
        lines.extend(
            f"| `{outcome}` | {count} |" for outcome, count in outcomes.items() if count
        )

        lines.extend(
            [
                "",
                "## Repair Outcomes by Cell",
                "",
                "| MAS | Benchmark | Samples | Attempts | Repair success | "
                "Verified | Control success | Patched only | Gate-success samples |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        cell_metrics = self.cell_repair_metrics()
        for cell in sorted(cell_metrics):
            mas, benchmark = cell.split("/", 1)
            metrics = cell_metrics[cell]
            lines.append(
                f"| `{mas}` | `{benchmark}` | {metrics['records']} | "
                f"{metrics['attempted_repairs']} | {metrics['repair_successes']} | "
                f"{metrics['verified_successes']} | {metrics['control_successes']} | "
                f"{metrics['patched_only']} | {metrics['gate_success_samples']} |"
            )
        lines.append("")
        return "\n".join(lines)

    def _sample_counts(self) -> dict[str, int]:
        return self._split_totals(self.sample_count)

    def _metric_counts(self, metric: str) -> dict[str, int]:
        return self._split_totals(lambda split: self.metric_count(metric, split))

    def _gate_success_counts(self) -> dict[str, int]:
        return self._split_totals(self.gate_success_count)

    def _training_row_counts(self, *, kind: str | None = None) -> dict[str, int]:
        return self._split_totals(
            lambda split: self.training_row_count(split, kind=kind)
        )

    def _task_counts(self) -> dict[str, int]:
        counts = {split: len(self.tasks_by_split[split]) for split in SPLIT_NAMES}
        counts["total"] = sum(counts.values())
        return counts

    @staticmethod
    def _split_totals(counter: Callable[[str | None], int]) -> dict[str, int]:
        counts = {split: int(counter(split)) for split in SPLIT_NAMES}
        counts["total"] = int(counter(None))
        return counts


def _mas_benchmark_key(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    mas = metadata.get("mas_id") or "unknown"
    benchmark = metadata.get("benchmark_id") or "unknown"
    return f"{mas}/{benchmark}"


def _sample_id(row: Mapping[str, Any]) -> str:
    return row_task_id(row)


def _matrix_axes(cells: Iterable[str]) -> tuple[list[str], list[str]]:
    preferred_mas = ["ag2", "magentic-one", "metagpt", "openmanus"]
    preferred_benchmarks = [
        "gsm-plus",
        "mmlu",
        "humaneval",
        "olympiadbench",
        "gaia",
        "swe-bench-lite",
    ]
    parsed = [cell.split("/", 1) for cell in cells if "/" in cell]
    mas = {item[0] for item in parsed}
    benchmarks = {item[1] for item in parsed}
    mas_order = [value for value in preferred_mas if value in mas]
    mas_order.extend(sorted(mas - set(mas_order)))
    benchmark_order = [
        value for value in preferred_benchmarks if value in benchmarks
    ]
    benchmark_order.extend(sorted(benchmarks - set(benchmark_order)))
    return mas_order, benchmark_order


def _matrix_markdown(
    counts: Mapping[str, Mapping[str, int]],
    mas_order: Sequence[str],
    benchmark_order: Sequence[str],
) -> list[str]:
    def value(mas: str, benchmark: str) -> str:
        cell = counts.get(f"{mas}/{benchmark}", {})
        return "/".join(str(cell.get(split, 0)) for split in SPLIT_NAMES)

    def total_for_mas(mas: str) -> str:
        return "/".join(
            str(
                sum(
                    counts.get(f"{mas}/{benchmark}", {}).get(split, 0)
                    for benchmark in benchmark_order
                )
            )
            for split in SPLIT_NAMES
        )

    def total_for_benchmark(benchmark: str) -> str:
        return "/".join(
            str(
                sum(
                    counts.get(f"{mas}/{benchmark}", {}).get(split, 0)
                    for mas in mas_order
                )
            )
            for split in SPLIT_NAMES
        )

    lines = [
        "| MAS \\ Benchmark | "
        + " | ".join(benchmark_order)
        + " | Total |",
        "| --- | " + " | ".join("---:" for _ in benchmark_order) + " | ---: |",
    ]
    for mas in mas_order:
        values = [value(mas, benchmark) for benchmark in benchmark_order]
        values.append(total_for_mas(mas))
        lines.append(f"| **{mas}** | " + " | ".join(values) + " |")
    total_values = [total_for_benchmark(benchmark) for benchmark in benchmark_order]
    total_values.append(
        "/".join(
            str(
                sum(
                    counts.get(f"{mas}/{benchmark}", {}).get(split, 0)
                    for mas in mas_order
                    for benchmark in benchmark_order
                )
            )
            for split in SPLIT_NAMES
        )
    )
    lines.append("| **Total** | " + " | ".join(total_values) + " |")
    return lines


def format_split_proportions(proportions: Sequence[float]) -> str:
    formatted = []
    for value in proportions:
        percent = value * 100
        rounded = round(percent)
        formatted.append(
            str(int(rounded)) if abs(percent - rounded) < 1e-9 else f"{percent:g}"
        )
    return "/".join(formatted)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _stream_split_jsonl(
    source_path: Path,
    output_path: Path,
    *,
    split: str,
    assignment: Mapping[str, str],
    tasks: set[str],
    count: dict[str, int],
    kind: str,
    statistics: _SplitStatistics,
) -> None:
    """Write one split file without retaining source payloads in memory."""

    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with source_path.open(encoding="utf-8") as source, temporary.open(
        "w", encoding="utf-8"
    ) as target:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{source_path}:{line_number} must be a JSON object")
            task_id = row_task_id(row)
            if assignment.get(task_id) != split:
                continue
            annotated = dict(row)
            metadata = dict(annotated.get("metadata") or {})
            metadata.update({"split": split, "corpus_split": split, "task_id": task_id})
            annotated["metadata"] = metadata
            target.write(
                json.dumps(annotated, ensure_ascii=False, sort_keys=True) + "\n"
            )
            count[kind] += 1
            tasks.add(task_id)
            statistics.observe(annotated, kind=kind, split=split)
    temporary.replace(output_path)


def _requested_splits(
    split: str | None, splits: Sequence[str] | None
) -> tuple[str, ...] | None:
    names: list[str] = []
    if split is not None:
        names.append(split)
    if splits is not None:
        names.extend(splits)
    if not names:
        return None
    unknown = [name for name in names if name not in SPLIT_NAMES]
    if unknown:
        raise ValueError(f"unknown corpus split(s): {sorted(set(unknown))}")
    return tuple(dict.fromkeys(names))


def _cap_task_families(
    rows: list[dict[str, Any]],
    caps: Mapping[str, int],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("task"))].append(row)
    kept: list[dict[str, Any]] = []
    capped: dict[str, int] = {}
    audits: dict[str, Any] = {}
    for task, task_rows in grouped.items():
        cap = int(caps.get(task, 0))
        if cap <= 0 or len(task_rows) <= cap:
            kept.extend(task_rows)
            continue
        if task == "deployment_gate":
            selected, audit = _sample_gate_rows(task_rows, cap=cap, seed=seed)
            capped[task] = len(task_rows) - len(selected)
            audits[task] = audit
            kept.extend(selected)
            continue
        # Stride within each incident so a capped gate family keeps early,
        # middle, and late prefixes of every incident instead of all prefixes
        # of the first few incidents.
        by_incident: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in task_rows:
            by_incident[str(row.get("incident_id"))].append(row)
        incident_ids = sorted(by_incident)
        per_incident = max(1, cap // max(1, len(incident_ids)))
        selected: list[dict[str, Any]] = []
        for incident_id in incident_ids:
            incident_rows = by_incident[incident_id]
            stride = max(1, len(incident_rows) // per_incident)
            selected.extend(incident_rows[::stride][:per_incident])
        if len(selected) > cap:
            stride = max(1, len(selected) // cap)
            selected = selected[::stride][:cap]
        capped[task] = len(task_rows) - len(selected)
        kept.extend(selected)
    return kept, capped, audits


def _sample_gate_rows(
    rows: Sequence[dict[str, Any]], *, cap: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a label-, cell-, and prefix-position-balanced gate corpus.

    A global per-incident stride is unsafe when the number of incidents exceeds
    the cap: it assigns one row to each incident and therefore selects almost
    exclusively the first, pre-failure prefix.  Gate sampling instead reserves
    half the budget for each label and round-robins over
    ``source_cell x early/middle/late`` strata.  Within a stratum, rows from
    distinct incidents are consumed before a second row from the same incident.
    """

    if cap < 2:
        raise ValueError("deployment_gate cap must be at least 2")
    positions = _gate_positions(rows)
    by_label: dict[int, list[dict[str, Any]]] = {0: [], 1: []}
    for row in rows:
        by_label[_gate_label(row)].append(row)
    quotas = _binary_gate_quotas(by_label, cap)
    selected: list[dict[str, Any]] = []
    for label in (0, 1):
        strata: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in by_label[label]:
            strata[(cell_of_row(row).key(), positions[id(row)])].append(row)
        selected.extend(
            _round_robin_gate_strata(
                strata,
                quota=quotas[label],
                seed=seed,
                label=label,
            )
        )
    # Preserve source order for stable staging and readable diffs. Trainer-side
    # epoch shuffling remains responsible for optimization order.
    source_order = {id(row): index for index, row in enumerate(rows)}
    selected.sort(key=lambda row: source_order[id(row)])
    annotated: list[dict[str, Any]] = []
    for row in selected:
        copied = dict(row)
        metadata = dict(copied.get("metadata") or {})
        metadata.update(
            {
                "gate_label": _gate_label(row),
                "gate_position": positions[id(row)],
                "gate_sampling": "balanced-label-cell-position-v1",
            }
        )
        copied["metadata"] = metadata
        annotated.append(copied)
    audit = {
        "strategy": "balanced-label-cell-position-v1",
        "requested_cap": cap,
        "label_quotas": {str(key): value for key, value in quotas.items()},
        "before": summarize_gate_rows(rows, positions=positions),
        "after": summarize_gate_rows(annotated),
    }
    return annotated, audit


def _binary_gate_quotas(
    by_label: Mapping[int, Sequence[Mapping[str, Any]]], cap: int
) -> dict[int, int]:
    target = {0: cap // 2, 1: cap - cap // 2}
    quota = {
        label: min(target[label], len(by_label.get(label, ()))) for label in (0, 1)
    }
    remaining = cap - sum(quota.values())
    while remaining > 0:
        progressed = False
        for label in (0, 1):
            available = len(by_label.get(label, ())) - quota[label]
            if available <= 0:
                continue
            quota[label] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            break
    return quota


def _round_robin_gate_strata(
    strata: Mapping[tuple[str, str], Sequence[dict[str, Any]]],
    *,
    quota: int,
    seed: int,
    label: int,
) -> list[dict[str, Any]]:
    ordered: dict[tuple[str, str], list[dict[str, Any]]] = {
        key: _incident_diverse_gate_order(
            value, seed=seed, salt=f"label={label}|cell={key[0]}|position={key[1]}"
        )
        for key, value in strata.items()
        if value
    }
    keys = sorted(
        ordered,
        key=lambda key: _stable_rank(
            seed, f"label={label}|cell={key[0]}|position={key[1]}"
        ),
    )
    offsets = dict.fromkeys(keys, 0)
    selected: list[dict[str, Any]] = []
    while len(selected) < quota:
        progressed = False
        for key in keys:
            index = offsets[key]
            values = ordered[key]
            if index >= len(values):
                continue
            selected.append(values[index])
            offsets[key] = index + 1
            progressed = True
            if len(selected) >= quota:
                break
        if not progressed:
            break
    return selected


def _incident_diverse_gate_order(
    rows: Sequence[dict[str, Any]], *, seed: int, salt: str
) -> list[dict[str, Any]]:
    by_incident: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_incident[str(row.get("incident_id") or row.get("record_id"))].append(row)
    incident_ids = sorted(
        by_incident,
        key=lambda incident_id: _stable_rank(seed, f"{salt}|incident={incident_id}"),
    )
    for incident_id in incident_ids:
        by_incident[incident_id].sort(
            key=lambda row: _stable_rank(
                seed, f"{salt}|row={row.get('record_id') or row.get('incident_id')}"
            )
        )
    ordered: list[dict[str, Any]] = []
    depth = 0
    while True:
        added = False
        for incident_id in incident_ids:
            values = by_incident[incident_id]
            if depth < len(values):
                ordered.append(values[depth])
                added = True
        if not added:
            return ordered
        depth += 1


def _gate_positions(rows: Sequence[Mapping[str, Any]]) -> dict[int, str]:
    maximum_by_incident: dict[str, int] = defaultdict(int)
    for row in rows:
        incident_id = str(row.get("incident_id") or row.get("record_id"))
        maximum_by_incident[incident_id] = max(
            maximum_by_incident[incident_id], _gate_prefix_length(row)
        )
    result: dict[int, str] = {}
    for row in rows:
        incident_id = str(row.get("incident_id") or row.get("record_id"))
        maximum = maximum_by_incident[incident_id]
        fraction = _gate_prefix_length(row) / maximum if maximum else 0.0
        result[id(row)] = (
            "early" if fraction <= 1 / 3 else "middle" if fraction <= 2 / 3 else "late"
        )
    return result


def _gate_prefix_length(row: Mapping[str, Any]) -> int:
    model_input = row.get("input")
    events = model_input.get("events") if isinstance(model_input, Mapping) else []
    return len(events) if isinstance(events, list) else 0


def _gate_label(row: Mapping[str, Any]) -> int:
    target = row.get("target")
    probability = target.get("gate_probability") if isinstance(target, Mapping) else 0
    return 1 if isinstance(probability, (int, float)) and float(probability) >= 0.5 else 0


def _stable_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def summarize_gate_rows(
    rows: Sequence[Mapping[str, Any]], *, positions: Mapping[int, str] | None = None
) -> dict[str, Any]:
    """Return an auditable distribution report for deployment-gate rows."""

    gate_rows = [row for row in rows if str(row.get("task")) == "deployment_gate"]
    resolved_positions = positions or _gate_positions(gate_rows)
    labels: dict[str, int] = defaultdict(int)
    position_counts: dict[str, int] = defaultdict(int)
    by_mas: dict[str, int] = defaultdict(int)
    by_benchmark: dict[str, int] = defaultdict(int)
    by_source_cell: dict[str, int] = defaultdict(int)
    label_by_source_cell: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    position_by_source_cell: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    incidents: set[str] = set()
    for row in gate_rows:
        label = str(_gate_label(row))
        metadata = row.get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        cell = cell_of_row(row)
        cell_key = cell.key()
        position = str(
            metadata.get("gate_position") or resolved_positions.get(id(row), "unknown")
        )
        labels[label] += 1
        position_counts[position] += 1
        by_mas[cell.mas_id] += 1
        by_benchmark[cell.benchmark_id] += 1
        by_source_cell[cell_key] += 1
        label_by_source_cell[cell_key][label] += 1
        position_by_source_cell[cell_key][position] += 1
        incidents.add(str(row.get("incident_id") or row.get("record_id")))
    count = len(gate_rows)
    positives = labels.get("1", 0)
    return {
        "rows": count,
        "incidents": len(incidents),
        "labels": dict(sorted(labels.items())),
        "positive_rate": positives / count if count else None,
        "positions": dict(sorted(position_counts.items())),
        "by_mas": dict(sorted(by_mas.items())),
        "by_benchmark": dict(sorted(by_benchmark.items())),
        "by_source_cell": dict(sorted(by_source_cell.items())),
        "label_by_source_cell": {
            key: dict(sorted(value.items()))
            for key, value in sorted(label_by_source_cell.items())
        },
        "position_by_source_cell": {
            key: dict(sorted(value.items()))
            for key, value in sorted(position_by_source_cell.items())
        },
    }


def _task_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row.get("task"))] += 1
    return dict(sorted(counts.items()))


def _assignment_counts(assignment: Mapping[str, str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in assignment.values():
        counts[value] += 1
    return dict(sorted(counts.items()))
