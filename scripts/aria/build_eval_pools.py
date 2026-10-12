"""Materialize the evaluation pools of Section~\\ref{subsec:exp-setup}.

Three pools are produced from the same partition the training corpora use, so a
task is never both trained on and evaluated:

``in_domain``
    The test partition of every source cell.  RQ1 reports on this pool: the
    passive setting restricts it to failed trajectories, and the active setting
    keeps every task, since a runtime system does not know in advance which run
    will fail.

``unseen_system`` and ``unseen_host_model``
    The withheld cells of RQ2, evaluated with the checkpoint trained without
    them.  These withhold a property of the source cell rather than a set of
    tasks, so the pool is defined by the cell rather than by the partition; the
    test partition of those cells is used by default so that pool sizes stay
    comparable with the in-domain pool.

The output is a V6 record file with the split rewritten to the shared
assignment, which the baseline dataset loader already reads.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from aria.corpus import (
    DEFAULT_SPLIT_PROPORTIONS,
    HeldOutAxis,
    assign_splits,
    host_model_of_row,
    load_rows,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=REPO_ROOT / "outputs/repair")
    parser.add_argument(
        "--environment",
        type=Path,
        default=None,
        help="Build an evaluation pool from env_all.jsonl. This is the runnable "
        "fallback for a prompt-stripped training_ready-only corpus.",
    )
    parser.add_argument(
        "--processed-split",
        type=Path,
        default=None,
        help="ARIA-Full-Processed split directory or sft.jsonl whose task IDs "
        "define the evaluation split. When supplied with --environment, the "
        "environment rows are filtered to this task-level split.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--splits", default="test")
    parser.add_argument(
        "--axis",
        default="in_domain",
        choices=("in_domain", "unseen_system", "unseen_host_model", "unseen_benchmark"),
    )
    parser.add_argument(
        "--values",
        default="",
        help="Cell property values that define a held-out pool, comma separated.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_split_name = "test"
    if args.processed_split is not None:
        requested = {value.strip() for value in args.splits.split(",") if value.strip()}
        if len(requested) != 1 or not requested <= {"train", "validation", "test"}:
            raise ValueError("a processed split requires exactly one train, validation, or test label")
        processed_split_name = next(iter(requested))
    if args.environment is not None:
        if not args.environment.is_file():
            raise SystemExit(f"environment corpus does not exist: {args.environment}")
        environment_rows = load_rows(args.environment)
        if args.processed_split is not None:
            records = _records_from_processed_split(
                args.processed_split, environment_rows=environment_rows, split=processed_split_name
            )
            _write_pool(
                args,
                records,
                source_paths=[args.environment, args.processed_split],
                preassigned=True,
                source_kind=f"aria-full-processed-{processed_split_name}+stage3-environment",
            )
            return
        _write_pool(
            args,
            [_eval_record_from_environment(row) for row in environment_rows],
            source_paths=[
                args.environment,
                *([args.processed_split] if args.processed_split is not None else []),
            ],
            preassigned=True,
        )
        return
    if args.processed_split is not None:
        records = _records_from_processed_split(args.processed_split, split=processed_split_name)
        _write_pool(
            args,
            records,
            source_paths=[args.processed_split],
            preassigned=True,
            source_kind=f"aria-full-processed-{processed_split_name}",
        )
        return
    dataset_paths = sorted(args.dataset_root.rglob("dataset/dataset.jsonl"))
    if not dataset_paths:
        raise SystemExit(f"no V6 dataset.jsonl beneath {args.dataset_root}")

    records: list[dict[str, Any]] = []
    for path in dataset_paths:
        dataset_id = path.parent.parent.name
        for row in load_rows(path):
            row["__dataset_id"] = dataset_id
            records.append(row)

    assignment = assign_splits(
        [
            {
                "incident_id": record["incident"]["incident_id"],
                "metadata": {
                    "task_id": record.get("task_id"),
                    "mas_id": (record.get("source_refs") or {}).get("mas_id"),
                    "benchmark_id": (record.get("source_refs") or {}).get("benchmark_id"),
                    "source_dataset_ids": [record["__dataset_id"]],
                },
            }
            for record in records
        ],
        proportions=DEFAULT_SPLIT_PROPORTIONS,
        seed=args.split_seed,
    )

    for record in records:
        task_id = str(record.get("task_id") or record["incident"]["incident_id"])
        record["split"] = assignment.get(task_id, "train")
    _write_pool(args, records, source_paths=dataset_paths, preassigned=True)


def _write_pool(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    *,
    source_paths: list[Path],
    preassigned: bool,
    source_kind: str | None = None,
) -> None:
    held_out = None
    if args.axis != "in_domain":
        values = [value for value in args.values.split(",") if value.strip()]
        held_out = HeldOutAxis.parse(args.axis, values)
    keep_splits = (
        None
        if args.splits.strip().lower() == "all"
        else {value for value in args.splits.split(",") if value.strip()}
    )

    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    seen_sample_ids: dict[str, dict[str, Any]] = {}
    duplicate_sample_ids: Counter[str] = Counter()
    for record in records:
        task_id = str(record.get("task_id") or record["incident"]["incident_id"])
        split = str(record.get("split") or "train") if preassigned else "train"
        cell = _cell(record)
        in_held_out = held_out is not None and _matches(cell, held_out)
        if held_out is not None and not in_held_out:
            stats["skipped_other_cell"] += 1
            continue
        if keep_splits is not None and split not in keep_splits:
            stats["skipped_split"] += 1
            continue
        row = {key: value for key, value in record.items() if key != "__dataset_id"}
        row["split"] = split
        row.setdefault("source_refs", {})
        row["source_refs"] = {
            **row["source_refs"],
            "host_model_id": cell["host_model_id"],
        }
        sample_id = str(row.get("sample_id") or row.get("incident_id") or "")
        if sample_id in seen_sample_ids:
            previous = seen_sample_ids[sample_id]
            if json.dumps(previous, ensure_ascii=True, sort_keys=True) != json.dumps(
                row, ensure_ascii=True, sort_keys=True
            ):
                raise ValueError(
                    "evaluation pool contains conflicting records for sample_id "
                    f"{sample_id}"
                )
            duplicate_sample_ids[sample_id] += 1
            continue
        seen_sample_ids[sample_id] = row
        rows.append(row)
        stats[f"cell:{cell['mas_id']}/{cell['benchmark_id']}"] += 1
        stats[f"record_type:{record.get('record_type')}"] += 1
        source_success = _source_success(record)
        stats[f"source_success:{source_success}"] += 1

    count = write_jsonl(args.output, rows)
    manifest = {
        "schema_version": "aria-eval-pool-v1",
        "axis": args.axis,
        "held_out_values": list(held_out.values) if held_out else [],
        "splits": sorted(keep_splits) if keep_splits else ["train", "validation", "test"],
        "split_proportions": list(DEFAULT_SPLIT_PROPORTIONS),
        "split_assignment": "task-level, stratified by source cell",
        "split_seed": args.split_seed,
        "records": count,
        "duplicate_sample_ids": dict(sorted(duplicate_sample_ids.items())),
        "duplicate_record_count": int(sum(duplicate_sample_ids.values())),
        "tasks": len({row.get("task_id") for row in rows}),
        "counts": dict(sorted(stats.items())),
        "cells": sorted({f"{_cell(r)['mas_id']}/{_cell(r)['benchmark_id']}/{_cell(r)['host_model_id']}" for r in rows}),
        "source_datasets": [str(path) for path in source_paths],
        "source_kind": (
            source_kind
            or ("stage3-environment" if args.environment is not None else "v6-repair-datasets")
        ),
        "output": str(args.output),
    }
    args.output.with_name(args.output.stem + "_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _eval_record_from_environment(row: dict[str, Any]) -> dict[str, Any]:
    """Adapt one answer-blind environment incident to the baseline V6 reader."""

    incident_id = str(row.get("incident_id") or row.get("task_id"))
    task_id = str(row.get("task_id") or incident_id)
    cell = row.get("cell") or {}
    source = dict(row.get("source") or {})
    control = dict(row.get("control") or {})
    certified = row.get("certified") or {}
    constraints = dict(row.get("constraints") or {})
    task_text = str(
        constraints.get("task_text")
        or (row.get("evaluator_sample") or {}).get("task")
        or ""
    )
    return {
        "schema_version": "aria-eval-pool-from-environment-v1",
        "record_id": f"eval::{incident_id}",
        "sample_id": incident_id,
        "task_id": task_id,
        "split": str(row.get("split") or "train"),
        "record_type": str(row.get("record_type") or "unknown"),
        "source_refs": {
            "mas_id": str(cell.get("mas_id") or "unknown"),
            "benchmark_id": str(cell.get("benchmark_id") or "unknown"),
            "model_id": f"model/{cell.get('host_model_id') or 'unknown'}",
            "host_model_id": str(cell.get("host_model_id") or "unknown"),
        },
        "incident": {
            "incident_id": incident_id,
            "events": row.get("events") or [],
            "feedback": row.get("feedback"),
            "constraints": constraints,
        },
        "sample": {
            "task_text": task_text,
            "metadata": {"sample_id": task_id},
            # training-ready fallback is intentionally answer-blind.
            "reference": {},
        },
        # Evaluator-only fields are retained outside the model-visible sample
        # view so local official evaluators can score new baseline repairs.
        "evaluator_sample": dict(row.get("evaluator_sample") or {}),
        "ledger": list(row.get("ledger") or []),
        "family_id": row.get("family_id"),
        "localization": row.get("localization"),
        "attribution": row.get("attribution"),
        "patch": certified.get("repair"),
        "failed_run": {"evaluation": source},
        "control_run": {"evaluation": control},
        "outcome": {
            "source_success": source.get("success"),
            "verified_successful": bool(row.get("verified_successful", False)),
        },
        "reward_capabilities": list(row.get("reward_capabilities") or []),
    }


def _cell(record: dict[str, Any]) -> dict[str, str]:
    refs = record.get("source_refs") or {}
    model = str(refs.get("model_id") or "unknown")
    return {
        "mas_id": str(refs.get("mas_id") or "unknown"),
        "benchmark_id": str(refs.get("benchmark_id") or "unknown"),
        "host_model_id": model.split("/", 1)[1] if model.startswith("model/") else model,
    }


def _matches(cell: dict[str, str], held_out: HeldOutAxis) -> bool:
    return cell[held_out.field] in held_out.values


def _source_success(record: dict[str, Any]) -> Any:
    failed = (record.get("failed_run") or {}).get("evaluation") or {}
    if isinstance(failed.get("success"), bool):
        return failed["success"]
    return (record.get("outcome") or {}).get("source_success")


def _load_processed_task_ids(path: Path) -> set[str]:
    """Read task-level identities from an ARIA-Full-Processed split.

    The processed release is a training projection and therefore is not itself
    executable.  Its task IDs are nevertheless the authoritative split
    assignment.  The runnable trajectory/evaluator payload comes from the
    Stage-3 environment passed via ``--environment``.
    """

    candidate = path
    if candidate.is_dir():
        candidate = candidate / "sft.jsonl"
    if not candidate.is_file():
        raise SystemExit(f"processed split does not exist: {candidate}")
    task_ids: set[str] = set()
    with candidate.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{candidate}:{line_number} must be a JSON object")
            metadata = row.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            task_id = (
                metadata.get("task_id")
                or row.get("task_id")
                or row.get("incident_id")
            )
            if task_id:
                task_ids.add(str(task_id))
    if not task_ids:
        raise SystemExit(f"processed split has no task IDs: {candidate}")
    return task_ids


def _records_from_processed_split(
    path: Path, *, environment_rows: list[dict[str, Any]] | None = None, split: str = "test",
) -> list[dict[str, Any]]:
    """Build one answer-blind eval record per incident from a processed split.

    ``ARIA-Full-Processed`` contains projected training rows rather than native
    source results.  Failed incidents retain their complete trajectory in the
    localization/attribution inputs; successful gate samples retain a prefix,
    so the longest recorded prefix is used as the runtime-visible trajectory.
    The public evaluator fields are intentionally not copied into the record.
    """

    if split not in {"train", "validation", "test"}:
        raise ValueError("processed pool split must be train, validation, or test")
    candidate = path / "sft.jsonl" if path.is_dir() else path
    if not candidate.is_file():
        raise SystemExit(f"processed split does not exist: {candidate}")
    grouped: dict[str, list[dict[str, Any]]] = {}
    with candidate.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{candidate}:{line_number} must be a JSON object")
            incident_id = str(row.get("incident_id") or row.get("sample_id") or "")
            if incident_id:
                grouped.setdefault(incident_id, []).append(row)

    by_incident: dict[str, dict[str, Any]] = {}
    for row in environment_rows or []:
        incident_id = str(row.get("incident_id") or "")
        if incident_id:
            by_incident[incident_id] = row

    records: list[dict[str, Any]] = []
    for incident_id, rows in sorted(grouped.items()):
        metadata = dict(rows[0].get("metadata") or {})
        record_type = str(metadata.get("record_type") or "unknown")
        task_id = str(metadata.get("task_id") or incident_id)
        source_success = record_type == "gate_success_sample"
        control_success = source_success
        environment_row = by_incident.get(incident_id)
        # The same benchmark task may have several source runs. Only exact
        # incident identity authorizes borrowing a trajectory or its ledger.
        if environment_row is not None:
            # Prefer the full answer-blind environment row when available. It
            # carries the original complete trajectory and any recorded ledger
            # entries, while the processed split still defines membership.
            record = _eval_record_from_environment(environment_row)
            record["split"] = split
            record["processed_split"] = split
            # The processed split owns the evaluation-pool membership and
            # source/control labels. The older environment is only a source of
            # longer trajectories and recorded ledger entries.
            record["failed_run"] = {
                "evaluation": {
                    "success": source_success,
                    "score": float(source_success),
                }
            }
            record["control_run"] = {
                "evaluation": {
                    "success": control_success,
                    "score": float(control_success),
                }
            }
            record["outcome"] = {"source_success": source_success}
            records.append(record)
            continue
        if record_type == "gate_success_sample":
            candidates = [row for row in rows if row.get("task") == "deployment_gate"]
            selected = max(
                candidates or rows,
                key=lambda row: len((row.get("input") or {}).get("events") or []),
            )
            source_success = True
            control_success = True
        else:
            selected = next(
                (row for row in rows if row.get("task") == "failure_localization"),
                next(
                    (row for row in rows if row.get("task") == "repair_attribution"),
                    rows[0],
                ),
            )
            source_success = False
            control_success = False
        model_input = dict(selected.get("input") or {})
        constraints = dict(model_input.get("constraints") or {})
        events = list(model_input.get("events") or [])
        if not events:
            raise ValueError(f"processed incident has no trajectory events: {incident_id}")
        mas = str(metadata.get("mas_id") or constraints.get("mas") or "unknown")
        benchmark = str(
            metadata.get("benchmark_id") or constraints.get("benchmark") or "unknown"
        )
        host_model = host_model_of_row(
            {
                "metadata": {
                    "mas_id": mas,
                    "benchmark_id": benchmark,
                    "source_dataset_ids": metadata.get("source_dataset_ids") or [],
                }
            }
        )
        records.append(
            {
                "schema_version": "aria-eval-pool-from-processed-v1",
                "record_id": f"processed::{incident_id}",
                "sample_id": incident_id,
                "task_id": task_id,
                "split": split,
                "record_type": record_type,
                "source_refs": {
                    "mas_id": mas,
                    "benchmark_id": benchmark,
                    "host_model_id": str(
                        metadata.get("host_model_id") or host_model or "unknown"
                    ),
                    "model_id": f"model/{metadata.get('host_model_id') or host_model or 'unknown'}",
                },
                "incident": {
                    "incident_id": incident_id,
                    "events": events,
                    "feedback": model_input.get("feedback"),
                    "constraints": constraints,
                },
                "sample": {
                    "task_text": str(constraints.get("task_text") or ""),
                    "metadata": {"sample_id": str(metadata.get("task_id") or incident_id)},
                    "reference": {},
                },
                "evaluator_sample": {},
                "failed_run": {
                    "evaluation": {"success": source_success, "score": float(source_success)}
                },
                "control_run": {
                    "evaluation": {"success": control_success, "score": float(control_success)}
                },
                "outcome": {"source_success": source_success},
            }
        )
    if not records:
        raise SystemExit(f"processed split has no incidents: {candidate}")
    return records


if __name__ == "__main__":
    main()
