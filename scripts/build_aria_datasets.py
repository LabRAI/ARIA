#!/usr/bin/env python3
"""Project V6 repair records into the training-ready ARIA-Full corpus.

By default this reads the 24 maintained repair configs under
``configs/manual_api_repair`` and ``configs/manual_vllm_repair``, resolves each
declared ``dataset_dir``, and projects the detailed ``dataset.jsonl`` audit
records with ``repair.training_projection`` -- the same projector the repair
runtime uses.  Reading the records instead of each experiment's materialized
``training_ready/`` directory keeps the aggregate corpus current with the
projection contract without re-running any repair experiment.

Usage:
    uv run python scripts/build_aria_datasets.py
    uv run python scripts/build_aria_datasets.py --allow-missing
    uv run python scripts/build_aria_datasets.py --dataset-dir <v6-dataset-dir>
    uv run python scripts/build_aria_datasets.py \
        --source-root outputs/repair \
        --source-root outputs/repair-opensource \
        --source-root outputs/repair-mini
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from aria.corpus import (
    DEFAULT_SPLIT_PROPORTIONS,
    SPLIT_NAMES,
    assign_splits,
    cell_of_row,
    format_split_proportions,
    row_task_id,
)
from aria.training import validate_training_ready_task_row
from common.artifacts import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_text,
    utc_now,
)
from repair.schema import REPAIR_SCHEMA_VERSION, TRAINING_READY_SCHEMA_VERSION
from repair.training_projection import project_record

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "ARIA-Full"
DEFAULT_CONFIG_ROOTS = (
    PROJECT_ROOT / "configs" / "manual_api_repair",
    PROJECT_ROOT / "configs" / "manual_vllm_repair",
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / DATASET_ID / "training_ready"
TRAINING_MANIFEST_SCHEMA_VERSION = "aria-training-ready-manifest-v3"
COLLECTION_MANIFEST_SCHEMA_VERSION = "aria-training-ready-collection-v4"
CORPUS_REPORT_SCHEMA_VERSION = "aria-training-ready-report-v1"
CONFIDENCE_BANDS = ((0.8, "high"), (0.5, "medium"), (0.0, "low"))
HEURISTIC_LABEL_SOURCES = frozenset({"heuristic", "heuristic_fallback"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-root",
        action="append",
        type=Path,
        help="repair config file or directory; repeatable",
    )
    parser.add_argument(
        "--dataset-dir",
        action="append",
        type=Path,
        default=[],
        help="explicit V6 repair dataset directory; repeatable",
    )
    parser.add_argument(
        "--source-root",
        action="append",
        type=Path,
        default=[],
        help="repair output root to scan for consumable V6 datasets; repeatable",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="aggregate training_ready output directory",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="skip configured repair datasets that have not produced artifacts yet",
    )
    parser.add_argument(
        "--allow-invalid-records",
        action="store_true",
        help="report unprojectable repair records instead of failing the build",
    )
    return parser.parse_args()


def dataset_dirs_from_configs(config_roots: Sequence[Path]) -> list[Path]:
    dataset_dirs: list[Path] = []
    for config_path in _config_paths(config_roots):
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise TypeError(f"config root must be a mapping: {config_path}")
        repair_specs = raw.get("repair_datasets", [])
        if not isinstance(repair_specs, list):
            raise TypeError(f"repair_datasets must be a list: {config_path}")
        for index, spec in enumerate(repair_specs):
            if not isinstance(spec, Mapping):
                raise TypeError(
                    f"repair_datasets[{index}] must be a mapping: {config_path}"
                )
            if spec.get("schema_version") != REPAIR_SCHEMA_VERSION:
                continue
            value = spec.get("dataset_dir")
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"repair_datasets[{index}].dataset_dir is missing: {config_path}"
                )
            dataset_dirs.append(_project_path(Path(value)))
    return _unique_paths(dataset_dirs)


def build_aria_corpus(
    dataset_dirs: Sequence[Path],
    output_dir: Path,
    *,
    allow_missing: bool = False,
    allow_invalid_records: bool = False,
    excluded_sources: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Project every configured V6 repair dataset into one post-training corpus."""

    resolved_output = _project_path(output_dir)
    resolved_sources = _unique_paths(_project_path(path) for path in dataset_dirs)
    if not resolved_sources:
        raise ValueError("no V6 repair dataset directories were selected")

    rows_by_kind: dict[str, dict[str, dict[str, Any]]] = {"sft": {}, "dpo": {}}
    fingerprints: dict[str, dict[str, str]] = {"sft": {}, "dpo": {}}
    source_records: list[dict[str, Any]] = []
    missing: list[str] = []
    statistics = _CorpusStatistics()
    for dataset_root in resolved_sources:
        if dataset_root == resolved_output or resolved_output in dataset_root.parents:
            raise ValueError("aggregate output_dir may not contain a source dataset")
        try:
            descriptor = _describe_source_dataset(dataset_root)
        except FileNotFoundError:
            if not allow_missing:
                raise
            missing.append(_display_path(dataset_root))
            continue
        projected = Counter()
        records_read = 0
        for line_number, record in _iter_records(dataset_root / "dataset.jsonl"):
            records_read += 1
            origin = f"{_display_path(dataset_root)}/dataset.jsonl:{line_number}"
            statistics.observe_record(record)
            try:
                projected_sft, projected_dpo = project_record(record)
            except (KeyError, TypeError, ValueError) as exc:
                statistics.observe_unprojectable(record, origin=origin, error=exc)
                if allow_invalid_records:
                    continue
                raise ValueError(
                    f"repair record could not be projected: {origin}: {exc}"
                ) from exc
            for kind, rows in (("sft", projected_sft), ("dpo", projected_dpo)):
                preference = kind == "dpo"
                for row in rows:
                    validated = dict(
                        validate_training_ready_task_row(
                            row,
                            name=f"{origin}:{row.get('record_id')}",
                            preference=preference,
                        )
                    )
                    record_id = str(validated["record_id"])
                    fingerprint = _semantic_fingerprint(
                        validated, preference=preference
                    )
                    existing = fingerprints[kind].get(record_id)
                    if existing is not None:
                        if existing != fingerprint:
                            raise ValueError(
                                f"conflicting {kind} record_id across repair "
                                f"datasets: {record_id}"
                            )
                        _merge_source_metadata(
                            rows_by_kind[kind][record_id], descriptor
                        )
                        continue
                    _merge_source_metadata(validated, descriptor)
                    rows_by_kind[kind][record_id] = validated
                    fingerprints[kind][record_id] = fingerprint
                    projected[str(validated["task"])] += 1
                    projected[kind] += 1
                    statistics.observe_row(validated, preference=preference)
        statistics.observe_invalid_samples(dataset_root / "invalid_samples.jsonl")
        descriptor["records_read"] = records_read
        descriptor["projected_counts"] = dict(sorted(projected.items()))
        descriptor["training_ready_drift"] = _training_ready_drift(
            descriptor.get("stored_training_counts"), projected
        )
        source_records.append(descriptor)

    if not source_records:
        raise ValueError("no completed or partial V6 repair datasets were available")
    sft_rows = sorted(rows_by_kind["sft"].values(), key=lambda row: row["record_id"])
    dpo_rows = sorted(rows_by_kind["dpo"].values(), key=lambda row: row["record_id"])
    if not sft_rows:
        raise ValueError("aggregate ARIA corpus contains no SFT rows")
    _validate_dpo_alignment(sft_rows, dpo_rows)

    resolved_output.mkdir(parents=True, exist_ok=True)
    sft_path = resolved_output / "sft.jsonl"
    dpo_path = resolved_output / "dpo.jsonl"
    atomic_write_jsonl(sft_path, sft_rows)
    atomic_write_jsonl(dpo_path, dpo_rows)
    task_counts: dict[str, int] = {}
    for row in [*sft_rows, *dpo_rows]:
        task = str(row["task"])
        task_counts[task] = task_counts.get(task, 0) + 1
    selected_dataset_dirs = {
        str(source.get("dataset_dir")) for source in source_records
    }
    effective_excluded_sources = [
        dict(source)
        for source in excluded_sources
        if str(source.get("dataset_dir")) not in selected_dataset_dirs
    ]
    manifest = {
        "schema_version": COLLECTION_MANIFEST_SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "dataset_schema_version": TRAINING_READY_SCHEMA_VERSION,
        "source_dataset_schema_version": REPAIR_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "output_dir": _display_path(resolved_output),
        "sources": source_records,
        "excluded_sources": effective_excluded_sources,
        "missing_sources": missing,
        "counts": {
            **task_counts,
            "sft": len(sft_rows),
            "dpo": len(dpo_rows),
            "source_datasets": len(source_records),
            "missing_source_datasets": len(missing),
        },
        "datasets": {
            "sft": _file_manifest(sft_path, len(sft_rows)),
            "dpo": _file_manifest(dpo_path, len(dpo_rows)),
        },
        "selection_policy": {
            "source": "V6 dataset.jsonl records projected by repair.training_projection",
            "validation": "aria.training.validate_training_ready_task_row per row",
            "deduplication": "record_id plus model-visible semantic fingerprint",
            "dpo_alignment": "input/chosen must match structured_repair SFT",
        },
    }
    atomic_write_json(resolved_output / "manifest.json", manifest)
    report = statistics.build_report(
        counts=manifest["counts"],
        sources=source_records,
        missing=missing,
    )
    atomic_write_json(resolved_output / "report.json", report)
    atomic_write_text(
        resolved_output / "STATS.md",
        _build_stats_markdown(
            sft_rows=sft_rows,
            dpo_rows=dpo_rows,
            manifest=manifest,
            report=report,
            statistics=statistics,
            split_seed=0,
        ),
    )
    manifest["report"] = report
    return manifest


class _CorpusStatistics:
    """Accumulate the acceptance figures required before ARIA post-training."""

    def __init__(self) -> None:
        self.record_types: Counter[str] = Counter()
        self.splits: Counter[str] = Counter()
        self.mas: Counter[str] = Counter()
        self.benchmarks: Counter[str] = Counter()
        self.cells: Counter[str] = Counter()
        self.paired_outcomes: Counter[str] = Counter()
        self.paired_outcomes_by_cell: dict[str, Counter[str]] = defaultdict(Counter)
        self.execution_modes: Counter[str] = Counter()
        self.primary_codes: Counter[str] = Counter()
        self.confidence_bands: Counter[str] = Counter()
        self.label_sources: Counter[str] = Counter()
        self.training_eligibility: Counter[str] = Counter()
        self.invalid_categories: Counter[str] = Counter()
        self.invalid_stages: Counter[str] = Counter()
        self.invalid_attempts = 0
        self.records = 0
        self.rows = 0
        self.attempted_repairs = 0
        self.repair_successes = 0
        self.verified_successes = 0
        self.control_successes = 0
        self.gate_success_samples = 0
        self.attempted_repairs_by_cell: Counter[str] = Counter()
        self.repair_successes_by_cell: Counter[str] = Counter()
        self.verified_successes_by_cell: Counter[str] = Counter()
        self.control_successes_by_cell: Counter[str] = Counter()
        self.gate_success_samples_by_cell: Counter[str] = Counter()
        self.row_tasks: Counter[str] = Counter()
        self.row_codes: Counter[str] = Counter()
        self.row_cells: Counter[str] = Counter()
        self.heuristic_positive_rows = 0
        self.unprojectable: list[dict[str, Any]] = []

    def observe_record(self, record: Mapping[str, Any]) -> None:
        self.records += 1
        refs = _mapping(record.get("source_refs"))
        outcome = _mapping(record.get("outcome"))
        attribution = _mapping(record.get("attribution"))
        training = _mapping(record.get("training"))
        mas = str(refs.get("mas_id") or "unknown")
        benchmark = str(refs.get("benchmark_id") or "unknown")
        cell = f"{mas}/{benchmark}"
        self.record_types[str(record.get("record_type") or "unknown")] += 1
        self.splits[str(record.get("split") or "unknown")] += 1
        self.mas[mas] += 1
        self.benchmarks[benchmark] += 1
        self.cells[cell] += 1
        paired = outcome.get("paired_outcome")
        if isinstance(paired, str) and paired:
            self.paired_outcomes[paired] += 1
            self.paired_outcomes_by_cell[cell][paired] += 1
        elif outcome.get("repair_attempted") is True:
            self.paired_outcomes["unresolved"] += 1
            self.paired_outcomes_by_cell[cell]["unresolved"] += 1
        self.execution_modes[
            str(outcome.get("execution_mode") or "not_executed")
        ] += 1
        if record.get("record_type") == "gate_success_sample":
            self.gate_success_samples += 1
            self.gate_success_samples_by_cell[cell] += 1
        if outcome.get("repair_attempted") is True:
            self.attempted_repairs += 1
            self.attempted_repairs_by_cell[cell] += 1
        if outcome.get("repair_success") is True:
            self.repair_successes += 1
            self.repair_successes_by_cell[cell] += 1
        if outcome.get("verified_successful") is True:
            self.verified_successes += 1
            self.verified_successes_by_cell[cell] += 1
        if outcome.get("control_success") is True:
            self.control_successes += 1
            self.control_successes_by_cell[cell] += 1
        code = attribution.get("primary_code")
        if isinstance(code, str) and code:
            self.primary_codes[code] += 1
            self.confidence_bands[
                _confidence_band(attribution.get("confidence"))
            ] += 1
        label_source = attribution.get("label_source")
        if isinstance(label_source, str) and label_source:
            self.label_sources[label_source] += 1
        for key, value in training.items():
            if key.endswith("_eligible") and value is True:
                self.training_eligibility[str(key)] += 1

    def observe_row(self, row: Mapping[str, Any], *, preference: bool) -> None:
        metadata = _mapping(row.get("metadata"))
        task = str(row.get("task") or "unknown")
        self.rows += 1
        self.row_tasks[task] += 1
        code = metadata.get("primary_code")
        if isinstance(code, str) and code:
            self.row_codes[code] += 1
        mas = str(metadata.get("mas_id") or "unknown")
        benchmark = str(metadata.get("benchmark_id") or "unknown")
        self.row_cells[f"{mas}/{benchmark}"] += 1
        positive = preference or task == "structured_repair"
        operator = metadata.get("repair_operator_id")
        if positive and isinstance(operator, str) and operator.startswith("heuristic"):
            self.heuristic_positive_rows += 1

    def observe_invalid_samples(self, path: Path) -> None:
        if not path.is_file():
            return
        for _, event in _iter_records(path):
            self.invalid_attempts += 1
            self.invalid_categories[str(event.get("category") or "unknown")] += 1
            self.invalid_stages[str(event.get("stage") or "unknown")] += 1

    def observe_unprojectable(
        self, record: Mapping[str, Any], *, origin: str, error: BaseException
    ) -> None:
        self.unprojectable.append(
            {
                "origin": origin,
                "record_id": record.get("record_id"),
                "sample_id": record.get("sample_id"),
                "error": f"{type(error).__name__}: {error}",
            }
        )

    def build_report(
        self,
        *,
        counts: Mapping[str, Any],
        sources: Sequence[Mapping[str, Any]],
        missing: Sequence[str],
    ) -> dict[str, Any]:
        return {
            "schema_version": CORPUS_REPORT_SCHEMA_VERSION,
            "dataset_id": DATASET_ID,
            "generated_at": utc_now(),
            "corpus": {
                "source_datasets": len(sources),
                "missing_source_datasets": len(missing),
                "repair_records": self.records,
                "training_rows": self.rows,
                "attempted_repairs": self.attempted_repairs,
                "repair_successes": self.repair_successes,
                "verified_successes": self.verified_successes,
                "control_successes": self.control_successes,
                "gate_success_samples": self.gate_success_samples,
                **{
                    key: value
                    for key, value in counts.items()
                    if key not in {"source_datasets", "missing_source_datasets"}
                },
            },
            "tasks": dict(self.row_tasks.most_common()),
            "records": {
                "by_record_type": dict(self.record_types.most_common()),
                "by_split": dict(self.splits.most_common()),
                "by_mas": dict(self.mas.most_common()),
                "by_benchmark": dict(self.benchmarks.most_common()),
                "by_mas_benchmark": dict(self.cells.most_common()),
                "repair_metrics_by_mas_benchmark": {
                    cell: {
                        "records": self.cells[cell],
                        "attempted_repairs": self.attempted_repairs_by_cell[cell],
                        "repair_successes": self.repair_successes_by_cell[cell],
                        "verified_successes": self.verified_successes_by_cell[cell],
                        "control_successes": self.control_successes_by_cell[cell],
                        "gate_success_samples": self.gate_success_samples_by_cell[
                            cell
                        ],
                        "patched_only": self.paired_outcomes_by_cell[cell][
                            "patched_only"
                        ],
                    }
                    for cell in sorted(self.cells)
                },
                "training_eligibility": dict(self.training_eligibility.most_common()),
            },
            "paired_outcomes": {
                "total": dict(self.paired_outcomes.most_common()),
                "by_mas_benchmark": {
                    cell: dict(outcomes.most_common())
                    for cell, outcomes in sorted(self.paired_outcomes_by_cell.items())
                },
            },
            "execution_modes": dict(self.execution_modes.most_common()),
            "attribution": {
                "record_primary_codes": dict(sorted(self.primary_codes.items())),
                "row_primary_codes": dict(sorted(self.row_codes.items())),
                "confidence_bands": dict(sorted(self.confidence_bands.items())),
                "label_sources": dict(self.label_sources.most_common()),
            },
            "rows_by_mas_benchmark": dict(self.row_cells.most_common()),
            "infrastructure_invalid": {
                "attempts": self.invalid_attempts,
                "by_category": dict(self.invalid_categories.most_common()),
                "by_stage": dict(self.invalid_stages.most_common()),
            },
            "validation": {
                "policy": "a leaked answer label or invalid row aborts the build",
                "schema_validated_rows": self.rows,
                "answer_leakage_checked_rows": self.rows,
                "answer_leakage_violations": 0,
                "heuristic_fallback_positive_rows": self.heuristic_positive_rows,
                "unprojectable_records": len(self.unprojectable),
                "unprojectable": self.unprojectable[:50],
            },
        }


def _describe_source_dataset(dataset_root: Path) -> dict[str, Any]:
    manifest_path = dataset_root / "manifest.json"
    dataset_path = dataset_root / "dataset.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing V6 repair manifest: {manifest_path}")
    if not dataset_path.is_file():
        raise FileNotFoundError(f"missing V6 repair records: {dataset_path}")
    manifest = _read_mapping(manifest_path)
    if manifest.get("schema_version") != REPAIR_SCHEMA_VERSION:
        raise ValueError(f"source is not a V6 repair dataset: {manifest_path}")
    if manifest.get("status") not in {"completed", "partial"}:
        raise ValueError(
            f"repair dataset status is not consumable: {manifest.get('status')!r}"
        )
    stored_counts: dict[str, Any] | None = None
    training_manifest_path = dataset_root / "training_ready" / "manifest.json"
    if training_manifest_path.is_file():
        training_manifest = _read_mapping(training_manifest_path)
        if training_manifest.get("schema_version") != TRAINING_MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported training-ready manifest: {training_manifest_path}"
            )
        if (
            training_manifest.get("dataset_schema_version")
            != TRAINING_READY_SCHEMA_VERSION
        ):
            raise ValueError(
                f"training-ready row schema mismatch: {training_manifest_path}"
            )
        stored_counts = dict(_mapping(training_manifest.get("counts")))
    source_dataset_dir = manifest.get("source_dataset_dir")
    canonical_dataset_dir = (
        str(source_dataset_dir)
        if isinstance(source_dataset_dir, str) and source_dataset_dir.strip()
        else _display_path(dataset_root)
    )
    descriptor = {
        "dataset_id": manifest.get("dataset_id") or manifest.get("name"),
        "dataset_dir": canonical_dataset_dir,
        "status": manifest.get("status"),
        "records": _mapping(manifest.get("counts")).get("records"),
        "stored_training_counts": stored_counts,
        "dataset_sha256": _mapping(manifest.get("checksums")).get("dataset.jsonl"),
    }
    if canonical_dataset_dir != _display_path(dataset_root):
        descriptor["read_dataset_dir"] = _display_path(dataset_root)
    if manifest.get("staging") is True:
        descriptor["staged"] = True
        if isinstance(manifest.get("staging_reason"), str):
            descriptor["staging_reason"] = manifest["staging_reason"]
        if isinstance(manifest.get("staging_state"), Mapping):
            descriptor["staging_state"] = dict(manifest["staging_state"])
    return descriptor


def _build_stats_markdown(
    *,
    sft_rows: Sequence[Mapping[str, Any]],
    dpo_rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    statistics: _CorpusStatistics,
    split_seed: int,
) -> str:
    """Render the human-readable corpus summary from the fresh projection."""

    split_counts = _split_cell_counts(sft_rows, dpo_rows, seed=split_seed)
    sample_counts = split_counts["samples"]
    training_counts = split_counts["training_rows"]
    cells = set(sample_counts) | set(training_counts) | {
        _mas_benchmark_key_from_cell(cell) for cell in statistics.cells
    }
    mas_order, benchmark_order = _matrix_axes(cells)

    counts = manifest["counts"]
    corpus = report["corpus"]
    outcomes = report["paired_outcomes"]["total"]
    invalid = report["infrastructure_invalid"]
    validation = report["validation"]
    lines = [
        "# ARIA-Full Statistics",
        "",
        f"Generated at: `{manifest['generated_at']}`",
        "",
        "## Scope",
        "",
        "This release is built from the selected non-empty V6 repair "
        "`dataset.jsonl` sources listed in `manifest.json`. Legacy V4 sources "
        "and other non-consumable sources are not projected.",
        "",
        "Split counts use the repository's deterministic task-level partition "
        f"with seed `{split_seed}` and proportions "
        f"`{format_split_proportions(DEFAULT_SPLIT_PROPORTIONS)}`. Each cell is "
        "shown as `train/val/test`; all rows belonging to one task stay in the "
        "same split.",
        "",
        "## Repair Samples",
        "",
        "Unique task/sample IDs represented by projected SFT rows, grouped by "
        "MAS and benchmark:",
        "",
    ]
    lines.extend(_matrix_markdown(sample_counts, mas_order, benchmark_order))
    staged_sources = [
        source for source in manifest["sources"] if source.get("staged") is True
    ]
    if staged_sources:
        lines.extend(
            [
                "",
                f"This release includes {len(staged_sources)} explicitly staged "
                "partial V6 source(s). Staged records are included for dataset "
                "construction only; they do not claim full repair-run completion.",
            ]
        )
    lines.extend(
        [
            "",
            "## Training Rows",
            "",
            "Projected training rows (`SFT + DPO`) by MAS and benchmark:",
            "",
        ]
    )
    lines.extend(_matrix_markdown(training_counts, mas_order, benchmark_order))
    lines.extend(
        [
            "",
            "## Overall Counts",
            "",
            "| Metric | Count |",
            "| --- | ---: |",
            f"| Source datasets | {counts['source_datasets']} |",
            f"| MAS x Benchmark cells with data | {len(statistics.cells)} |",
            f"| Repair records/samples | {corpus['repair_records']} |",
            f"| Repair attempts | {statistics.attempted_repairs} |",
            f"| Repair successes (`repair_success=true`) | {statistics.repair_successes} |",
            f"| Host-verified successful repairs | {statistics.verified_successes} |",
            f"| Control successes | {statistics.control_successes} |",
            f"| Gate-success samples | {statistics.gate_success_samples} |",
            f"| Paired outcome: `patched_only` | {outcomes.get('patched_only', 0)} |",
            f"| Paired outcome: `both_pass` | {outcomes.get('both_pass', 0)} |",
            f"| Paired outcome: `control_only` | {outcomes.get('control_only', 0)} |",
            f"| Paired outcome: `both_fail` | {outcomes.get('both_fail', 0)} |",
            f"| SFT rows | {counts['sft']} |",
            f"| DPO rows | {counts['dpo']} |",
            f"| Total training rows | {corpus['training_rows']} |",
            f"| Infrastructure-invalid attempts | {invalid['attempts']} |",
            f"| Unprojectable records | {validation['unprojectable_records']} |",
            f"| Answer-leakage violations | {validation['answer_leakage_violations']} |",
            "",
            "## Training Task Counts",
            "",
            "| Task | Rows |",
            "| --- | ---: |",
        ]
    )
    for task, count in report["tasks"].items():
        lines.append(f"| `{task}` | {count} |")
    lines.extend(
        [
            "",
            "## Repair Outcomes by Cell",
            "",
            "Counts are audit-record counts for each source cell; `verified` "
            "means host-verified successful repair and `patched_only` is the "
            "paired causal outcome.",
            "",
            "| MAS | Benchmark | Records | Attempts | Repair success | Verified | Control success | Patched only | Gate-success samples |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    metrics_by_cell = report["records"]["repair_metrics_by_mas_benchmark"]
    for cell in sorted(metrics_by_cell):
        mas, benchmark = cell.split("/", 1)
        metrics = metrics_by_cell[cell]
        lines.append(
            f"| `{mas}` | `{benchmark}` | {metrics['records']} | "
            f"{metrics['attempted_repairs']} | {metrics['repair_successes']} | "
            f"{metrics['verified_successes']} | {metrics['control_successes']} | "
            f"{metrics['patched_only']} | {metrics['gate_success_samples']} |"
        )
    lines.extend(
        [
            "",
            "## Paired Outcomes",
            "",
            "| Outcome | Samples |",
            "| --- | ---: |",
        ]
    )
    for outcome, count in outcomes.items():
        lines.append(f"| `{outcome}` | {count} |")
    excluded = manifest.get("excluded_sources") or []
    if excluded:
        lines.extend(
            [
                "",
                "## Excluded Sources",
                "",
                "These directories were discovered under the requested repair "
                "roots but were not projected:",
                "",
                "| Dataset directory | Reason |",
                "| --- | --- |",
            ]
        )
        for source in excluded:
            lines.append(
                f"| `{source.get('dataset_dir', 'unknown')}` | "
                f"`{source.get('reason', 'unknown')}` |"
            )
    lines.extend(
        [
            "",
            "## Source Datasets",
            "",
            "| Dataset directory | Status | Records read | Projected SFT | Projected DPO |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for source in manifest["sources"]:
        projected = source.get("projected_counts") or {}
        status = str(source.get("status", "unknown"))
        if source.get("staged") is True:
            status += ", staged"
        lines.append(
            "| `{dataset_dir}` | `{status}` | {records} | {sft} | {dpo} |".format(
                dataset_dir=source["dataset_dir"],
                status=status,
                records=source.get("records_read", source.get("records", 0)),
                sft=projected.get("sft", 0),
                dpo=projected.get("dpo", 0),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _split_cell_counts(
    sft_rows: Sequence[Mapping[str, Any]],
    dpo_rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, dict[str, dict[str, int]]]:
    assignment = assign_splits([*sft_rows, *dpo_rows], seed=seed)
    samples: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {split: set() for split in SPLIT_NAMES}
    )
    training_rows: dict[str, dict[str, int]] = defaultdict(
        lambda: {split: 0 for split in SPLIT_NAMES}
    )
    for kind, rows in (("sft", sft_rows), ("dpo", dpo_rows)):
        for row in rows:
            task_id = row_task_id(row)
            split = assignment[task_id]
            cell = _mas_benchmark_key(row)
            training_rows[cell][split] += 1
            if kind == "sft":
                samples[cell][split].add(task_id)
    return {
        "samples": {
            cell: {split: len(values[split]) for split in SPLIT_NAMES}
            for cell, values in samples.items()
        },
        "training_rows": {
            cell: dict(values) for cell, values in training_rows.items()
        },
    }


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


def _mas_benchmark_key(row: Mapping[str, Any]) -> str:
    metadata = _mapping(row.get("metadata"))
    return (
        f"{metadata.get('mas_id') or 'unknown'}"
        f"/{metadata.get('benchmark_id') or 'unknown'}"
    )


def _mas_benchmark_key_from_cell(cell: str) -> str:
    return "/".join(cell.split("/")[:2])


def _matrix_markdown(
    counts: Mapping[str, Mapping[str, int]],
    mas_order: Sequence[str],
    benchmark_order: Sequence[str],
) -> list[str]:
    def value(mas: str, benchmark: str) -> str:
        cell = counts.get(f"{mas}/{benchmark}", {})
        return "/".join(str(cell.get(split, 0)) for split in SPLIT_NAMES)

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
        row_values = [value(mas, benchmark) for benchmark in benchmark_order]
        row_values.append(
            "/".join(
                str(
                    sum(
                        counts.get(f"{mas}/{benchmark}", {}).get(split, 0)
                        for benchmark in benchmark_order
                    )
                )
                for split in SPLIT_NAMES
            )
        )
        lines.append(f"| **{mas}** | " + " | ".join(row_values) + " |")
    total_values = [total_for_benchmark(benchmark) for benchmark in benchmark_order]
    all_total = "/".join(
        str(
            sum(
                counts.get(f"{mas}/{benchmark}", {}).get(split, 0)
                for mas in mas_order
                for benchmark in benchmark_order
            )
        )
        for split in SPLIT_NAMES
    )
    lines.append("| **Total** | " + " | ".join([*total_values, all_total]) + " |")
    return lines


def _training_ready_drift(
    stored: Mapping[str, Any] | None, projected: Mapping[str, int]
) -> dict[str, Any] | None:
    """Report where a stored ``training_ready/`` differs from a fresh projection."""

    if stored is None:
        return None
    differences = {
        key: {"stored": stored.get(key), "projected": projected.get(key, 0)}
        for key in sorted(set(stored) | set(projected))
        if stored.get(key) != projected.get(key, 0)
    }
    return {"in_sync": not differences, "differences": differences}


def _validate_dpo_alignment(
    sft_rows: Sequence[Mapping[str, Any]],
    dpo_rows: Sequence[Mapping[str, Any]],
) -> None:
    repairs: dict[str, str] = {}
    for row in sft_rows:
        if row.get("task") != "structured_repair":
            continue
        source_record_id = _required_source_record_id(row)
        if source_record_id in repairs:
            raise ValueError(
                "multiple structured_repair SFT rows for source_record_id: "
                f"{source_record_id}"
            )
        repairs[source_record_id] = _fingerprint(
            {"input": row.get("input"), "target": row.get("target")}
        )
    for row in dpo_rows:
        source_record_id = _required_source_record_id(row)
        expected = repairs.get(source_record_id)
        if expected is None:
            raise ValueError(
                "DPO row has no structured_repair SFT source: "
                f"{source_record_id}"
            )
        actual = _fingerprint(
            {"input": row.get("input"), "target": row.get("chosen")}
        )
        if actual != expected:
            raise ValueError(
                "DPO input/chosen conflicts with structured_repair SFT: "
                f"{source_record_id}"
            )


def _merge_source_metadata(row: dict[str, Any], source: Mapping[str, Any]) -> None:
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    merged = dict(metadata)
    dataset_id = str(source.get("dataset_id") or source.get("dataset_dir"))
    dataset_dir = str(source.get("dataset_dir"))
    ids = {str(value) for value in merged.get("source_dataset_ids", [])}
    dirs = {str(value) for value in merged.get("source_dataset_dirs", [])}
    ids.add(dataset_id)
    dirs.add(dataset_dir)
    merged["source_dataset_ids"] = sorted(ids)
    merged["source_dataset_dirs"] = sorted(dirs)
    row["metadata"] = merged


def _required_source_record_id(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    value = metadata.get("source_record_id") if isinstance(metadata, Mapping) else None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{row.get('record_id')}.metadata.source_record_id must be non-empty"
        )
    return value


def _confidence_band(value: Any) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "unscored"
    for threshold, band in CONFIDENCE_BANDS:
        if float(value) >= threshold:
            return band
    return "unscored"


def _semantic_fingerprint(row: Mapping[str, Any], *, preference: bool) -> str:
    fields = (
        ("task", "input", "chosen", "rejected", "rejection_reason")
        if preference
        else ("task", "input", "target")
    )
    return _fingerprint({field: row.get(field) for field in fields})


def _fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _file_manifest(path: Path, records: int) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": path.name,
        "records": records,
        "bytes": len(data),
        "sha256": f"sha256:{hashlib.sha256(data).hexdigest()}",
    }


def _iter_records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Stream one JSONL audit file; V6 record files are far too large to slurp."""

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number} must contain a JSON object")
            yield line_number, value


def _read_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _config_paths(roots: Sequence[Path]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        resolved = _project_path(root)
        if resolved.is_file():
            paths.append(resolved)
        elif resolved.is_dir():
            paths.extend(sorted(resolved.rglob("*.yml")))
        else:
            raise FileNotFoundError(f"repair config root does not exist: {resolved}")
    return sorted(set(paths))


def discover_source_datasets(
    source_roots: Sequence[Path],
) -> tuple[list[Path], list[dict[str, str]]]:
    """Discover consumable V6 datasets under repair output roots.

    Discovery is intentionally conservative: a source must have a V6 manifest,
    a non-empty ``dataset.jsonl``, and a terminal consumable status.  The
    excluded directories are returned for provenance instead of being silently
    ignored in the aggregate manifest.
    """

    selected: list[Path] = []
    excluded: list[dict[str, str]] = []
    for root in _unique_paths(_project_path(path) for path in source_roots):
        if not root.is_dir():
            raise FileNotFoundError(f"source root does not exist: {root}")
        candidates = sorted(root.glob("*/*/*/dataset"))
        for dataset_root in candidates:
            manifest_path = dataset_root / "manifest.json"
            dataset_path = dataset_root / "dataset.jsonl"
            display = _display_path(dataset_root)
            if not manifest_path.is_file():
                excluded.append(
                    {"dataset_dir": display, "reason": "missing_manifest"}
                )
                continue
            try:
                manifest = _read_mapping(manifest_path)
            except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                excluded.append(
                    {
                        "dataset_dir": display,
                        "reason": f"invalid_manifest:{type(exc).__name__}",
                    }
                )
                continue
            if manifest.get("schema_version") != REPAIR_SCHEMA_VERSION:
                excluded.append(
                    {
                        "dataset_dir": display,
                        "reason": f"unsupported_schema:{manifest.get('schema_version')}",
                    }
                )
                continue
            if not dataset_path.is_file():
                excluded.append(
                    {"dataset_dir": display, "reason": "missing_dataset_jsonl"}
                )
                continue
            with dataset_path.open(encoding="utf-8") as handle:
                has_records = any(line.strip() for line in handle)
            if not has_records:
                excluded.append({"dataset_dir": display, "reason": "empty_dataset"})
                continue
            if manifest.get("status") not in {"completed", "partial"}:
                excluded.append(
                    {
                        "dataset_dir": display,
                        "reason": f"non_consumable_status:{manifest.get('status')}",
                    }
                )
                continue
            run_state_path = dataset_root / "run_state.json"
            if run_state_path.is_file():
                try:
                    run_state = _read_mapping(run_state_path)
                except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    excluded.append(
                        {
                            "dataset_dir": display,
                            "reason": f"invalid_run_state:{type(exc).__name__}",
                        }
                    )
                    continue
                if run_state.get("status") == "running" or run_state.get(
                    "active_sample_id"
                ):
                    excluded.append(
                        {"dataset_dir": display, "reason": "run_state_running"}
                    )
                    continue
            selected.append(dataset_root)
    return _unique_paths(selected), excluded


def _project_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    return list(dict.fromkeys(Path(path).resolve() for path in paths))


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    config_roots = tuple(
        args.config_root
        if args.config_root
        else (() if args.source_root else DEFAULT_CONFIG_ROOTS)
    )
    configured = dataset_dirs_from_configs(config_roots)
    discovered, excluded = discover_source_datasets(args.source_root)
    explicit = [_project_path(path) for path in args.dataset_dir]
    sources = _unique_paths([*configured, *discovered, *explicit])
    manifest = build_aria_corpus(
        sources,
        args.output_dir,
        allow_missing=args.allow_missing,
        allow_invalid_records=args.allow_invalid_records,
        excluded_sources=excluded,
    )
    counts = manifest["counts"]
    report = manifest["report"]
    print(
        f"Built {DATASET_ID} training_ready: "
        f"sources={counts['source_datasets']} "
        f"records={report['corpus']['repair_records']} "
        f"sft={counts['sft']} dpo={counts['dpo']} "
        f"output={manifest['output_dir']}"
    )
    for task, count in report["tasks"].items():
        print(f"  {task}: {count}")
    drifted = [
        source["dataset_dir"]
        for source in manifest["sources"]
        if (source.get("training_ready_drift") or {}).get("in_sync") is False
    ]
    if drifted:
        print(
            "  stale per-experiment training_ready (aggregate re-projected): "
            + ", ".join(drifted)
        )
    unprojectable = report["validation"]["unprojectable_records"]
    if unprojectable:
        print(f"  unprojectable repair records: {unprojectable}")


if __name__ == "__main__":
    main()
