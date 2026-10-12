"""Build the Stage 3 environment corpus from the V6 repair datasets.

Section~\\ref{subsubsec:stage3} reuses the verification apparatus assembled for
corpus construction as the environment in which the final training stage obtains
its rewards.  This script materializes that apparatus as one JSONL file: per
incident, the answer-blind model view, the certified cut, the evaluator-only
sample carrying the hidden reference answer, the matched control outcome, and
every splice the corpus actually executed together with its recorded verdict.

The split assignment is produced by :mod:`aria.corpus` from the *same* seed and
proportions as the training corpus view, so an incident is never a Stage 3
training incident in one artifact and a test incident in another.
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
    cell_of_row,
    load_rows,
    row_task_id,
    write_jsonl,
)
from aria.post_training.environment import ENVIRONMENT_SCHEMA_VERSION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="A V6 dataset.jsonl path; repeat for several source cells.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "repair",
        help="Discover every dataset.jsonl beneath this root when no --dataset is given.",
    )
    parser.add_argument(
        "--training-ready",
        type=Path,
        default=None,
        help="Build the recorded-execution environment from an ARIA training_ready "
        "directory when the source V6 dataset.jsonl files are unavailable.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--splits",
        default="",
        help="Comma-separated splits to keep; empty keeps all three.",
    )
    parser.add_argument("--held-out-axis", default="")
    parser.add_argument("--held-out-values", default="")
    parser.add_argument(
        "--require-verified",
        action="store_true",
        help="Keep only incidents whose repair was execution-verified.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Truncate to the last N events; 0 keeps the whole trajectory, which is\n"
        "the default because the trainer fits its own prompt budget and a\n"
        "truncated log would strand the certified cut outside the record.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.training_ready is not None:
        if args.dataset:
            raise SystemExit("--training-ready and --dataset are mutually exclusive")
        records = _training_ready_environment_rows(
            args.training_ready,
            split_seed=args.split_seed,
        )
        _write_selected_rows(args, records, source_paths=[args.training_ready])
        return
    dataset_paths = _dataset_paths(args)
    if not dataset_paths:
        raise SystemExit("no V6 dataset.jsonl was found")

    records: list[dict[str, Any]] = []
    for path in dataset_paths:
        for row in load_rows(path):
            row["__source_dataset_id"] = _dataset_id(path)
            records.append(row)
    if not records:
        raise SystemExit("the V6 datasets contain no records")

    split_rows = [
        {
            "incident_id": record["incident"]["incident_id"],
            "metadata": {
                "task_id": record.get("task_id"),
                "mas_id": (record.get("source_refs") or {}).get("mas_id"),
                "benchmark_id": (record.get("source_refs") or {}).get("benchmark_id"),
                "source_dataset_ids": [record["__source_dataset_id"]],
            },
        }
        for record in records
    ]
    assignment = assign_splits(
        split_rows, proportions=DEFAULT_SPLIT_PROPORTIONS, seed=args.split_seed
    )

    _write_selected_rows(args, [
        incident
        for record in records
        if (incident := _environment_row(record, assignment, max_events=args.max_events))
        is not None
    ], source_paths=dataset_paths)


def _write_selected_rows(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
    *,
    source_paths: list[Path],
) -> None:
    held_out = None
    if args.held_out_axis:
        held_out = HeldOutAxis.parse(
            args.held_out_axis,
            [value for value in args.held_out_values.split(",") if value.strip()],
        )
    keep_splits = {value for value in args.splits.split(",") if value.strip()}

    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for incident in records:
        if keep_splits and incident["split"] not in keep_splits:
            stats["skipped_split"] += 1
            continue
        if held_out is not None and _excluded(incident, held_out):
            stats["skipped_held_out"] += 1
            continue
        if args.require_verified and not incident["verified_successful"]:
            stats["skipped_unverified"] += 1
            continue
        rows.append(incident)
        stats[f"split:{incident['split']}"] += 1
        stats[f"cell:{incident['cell']['mas_id']}/{incident['cell']['benchmark_id']}"] += 1
        stats[f"ledger_entries:{min(len(incident['ledger']), 4)}"] += 1

    count = write_jsonl(args.output, rows)
    manifest = {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "output": str(args.output),
        "incidents": count,
        "source_datasets": [str(path) for path in source_paths],
        "source_kind": (
            "training-ready-recorded-outcomes"
            if args.training_ready is not None
            else "v6-repair-datasets"
        ),
        "split_seed": args.split_seed,
        "split_proportions": list(DEFAULT_SPLIT_PROPORTIONS),
        "splits_kept": sorted(keep_splits) or ["train", "validation", "test"],
        "held_out_axis": (
            {"name": held_out.name, "values": list(held_out.values)}
            if held_out is not None
            else None
        ),
        "require_verified": bool(args.require_verified),
        "counts": dict(sorted(stats.items())),
        "families": len({row["family_id"] for row in rows}),
        "cells": sorted({_cell_key(row) for row in rows}),
        "executed_ledger_entries": sum(len(row["ledger"]) for row in rows),
    }
    manifest_path = args.output.with_name(args.output.stem + "_manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _training_ready_environment_rows(
    source_dir: Path,
    *,
    split_seed: int,
) -> list[dict[str, Any]]:
    """Recover the replayable subset intentionally retained by training-ready V3.

    V3 is prompt-stripped and therefore cannot drive a fresh official evaluator
    or native host rerun.  It does retain the answer-blind incident, the verified
    chosen splice, the observed failed continuation, and paired-outcome labels.
    Those are sufficient for the explicitly labelled ``ledger`` backend and make
    the partial public corpus runnable without pretending it contains hidden
    evaluator references.
    """

    sft_path = source_dir / "sft.jsonl"
    dpo_path = source_dir / "dpo.jsonl"
    if not sft_path.is_file() or not dpo_path.is_file():
        raise SystemExit(
            f"training-ready source requires sft.jsonl and dpo.jsonl: {source_dir}"
        )
    sft_rows = load_rows(sft_path)
    dpo_rows = load_rows(dpo_path)
    assignment = assign_splits(
        [*sft_rows, *dpo_rows],
        proportions=DEFAULT_SPLIT_PROPORTIONS,
        seed=split_seed,
    )
    by_incident: dict[str, list[dict[str, Any]]] = {}
    dpo_by_incident: dict[str, list[dict[str, Any]]] = {}
    for row in sft_rows:
        by_incident.setdefault(str(row.get("incident_id")), []).append(row)
    for row in dpo_rows:
        dpo_by_incident.setdefault(str(row.get("incident_id")), []).append(row)

    result: list[dict[str, Any]] = []
    for incident_id, rows in sorted(by_incident.items()):
        base = _most_informative_training_row(rows)
        model_input = base.get("input")
        if not isinstance(model_input, dict):
            continue
        events = model_input.get("events")
        if not isinstance(events, list) or not events:
            continue
        structured = next(
            (row for row in rows if row.get("task") == "structured_repair"),
            None,
        )
        repair = (
            dict(structured["target"])
            if structured is not None and isinstance(structured.get("target"), dict)
            else None
        )
        localization = _teacher_object(
            structured,
            "localization",
            rows,
            fallback_task="failure_localization",
        )
        attribution = _teacher_object(
            structured,
            "attribution",
            rows,
            fallback_task="repair_attribution",
        )
        metadata = dict(base.get("metadata") or {})
        cell = cell_of_row(base)
        task_id = row_task_id(base)
        source_success = str(metadata.get("record_type")) == "gate_success_sample"
        control_success = _paired_control_success(
            metadata.get("paired_outcome"),
            default=source_success,
        )
        event_ids = {str(event.get("event_id")) for event in events if isinstance(event, dict)}
        first_bad = list((localization or {}).get("first_bad_event_ids") or [])
        ledger: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for preference in dpo_by_incident.get(incident_id, []):
            for label, field, success in (
                ("verified_patch", "chosen", True),
                ("observed_failed_continuation", "rejected", False),
            ):
                candidate = preference.get(field)
                if not isinstance(candidate, dict):
                    continue
                cut = candidate.get("cut_event_id")
                replacement = candidate.get("replacement_events")
                if cut is not None and str(cut) not in event_ids:
                    continue
                if not isinstance(replacement, list) or not replacement:
                    continue
                signature = (str(cut), json.dumps(replacement, sort_keys=True))
                if signature in seen:
                    continue
                seen.add(signature)
                ledger.append(
                    {
                        "label": label,
                        "cut_event_id": cut,
                        "replacement_events": replacement,
                        "success": success,
                        "score": 1.0 if success else 0.0,
                        "execution_mode": str(
                            (preference.get("metadata") or {}).get("execution_mode")
                            or "recorded_training_ready_outcome"
                        ),
                    }
                )
        result.append(
            {
                "schema_version": ENVIRONMENT_SCHEMA_VERSION,
                "incident_id": incident_id,
                "task_id": task_id,
                "split": assignment[task_id],
                "cell": {
                    "mas_id": cell.mas_id,
                    "benchmark_id": cell.benchmark_id,
                    "host_model_id": cell.host_model_id,
                },
                "family_id": str(
                    metadata.get("family_id")
                    or metadata.get("primary_code")
                    or "unknown"
                ),
                "events": events,
                "feedback": model_input.get("feedback"),
                "constraints": model_input.get("constraints") or {},
                "localization": localization,
                "attribution": attribution,
                "certified": {
                    "cut_event_id": repair.get("cut_event_id") if repair else None,
                    "first_bad_event_ids": first_bad,
                    "repair": repair,
                },
                # Hidden references are deliberately absent from training-ready V3.
                "evaluator_sample": {
                    "task": (model_input.get("constraints") or {}).get("task_text", ""),
                    "sample_id": task_id,
                },
                "control": {
                    "success": control_success,
                    "score": 1.0 if control_success else 0.0,
                },
                "source": {
                    "success": source_success,
                    "score": 1.0 if source_success else 0.0,
                },
                "ledger": ledger,
                "verified_successful": bool(metadata.get("verified_successful", False)),
                "record_type": str(metadata.get("record_type") or "unknown"),
                "source_dataset_id": _training_source_dataset_id(metadata),
                "reward_capabilities": ["ledger"],
            }
        )
    return result


def _most_informative_training_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    priority = {
        "structured_repair": 4,
        "repair_attribution": 3,
        "failure_localization": 2,
        "deployment_gate": 1,
    }
    return max(
        rows,
        key=lambda row: (
            priority.get(str(row.get("task")), 0),
            len(((row.get("input") or {}).get("events") or [])),
        ),
    )


def _teacher_object(
    structured: dict[str, Any] | None,
    key: str,
    rows: list[dict[str, Any]],
    *,
    fallback_task: str,
) -> dict[str, Any] | None:
    if structured is not None:
        value = (structured.get("input") or {}).get(key)
        if isinstance(value, dict):
            return dict(value)
    fallback = next((row for row in rows if row.get("task") == fallback_task), None)
    value = fallback.get("target") if fallback is not None else None
    return dict(value) if isinstance(value, dict) else None


def _paired_control_success(value: Any, *, default: bool) -> bool:
    if value in {"both_pass", "control_only"}:
        return True
    if value in {"both_fail", "patched_only"}:
        return False
    return default


def _training_source_dataset_id(metadata: dict[str, Any]) -> str:
    values = metadata.get("source_dataset_ids")
    if isinstance(values, list) and values:
        return str(values[0])
    return "training-ready-v3"


def _dataset_paths(args: argparse.Namespace) -> list[Path]:
    if args.dataset:
        return [Path(value).resolve() for value in args.dataset]
    return sorted(args.dataset_root.rglob("dataset/dataset.jsonl"))


def _dataset_id(path: Path) -> str:
    # .../outputs/repair/<mas>/<benchmark>/<dataset id>/dataset/dataset.jsonl
    return path.parent.parent.name


def _cell_key(row: dict[str, Any]) -> str:
    cell = row["cell"]
    return f"{cell['mas_id']}/{cell['benchmark_id']}/{cell['host_model_id']}"


def _excluded(row: dict[str, Any], held_out: HeldOutAxis) -> bool:
    field = {
        "mas_id": "mas_id",
        "benchmark_id": "benchmark_id",
        "host_model_id": "host_model_id",
    }[held_out.field]
    return str(row["cell"][field]) in held_out.values


def _environment_row(
    record: dict[str, Any],
    assignment: dict[str, str],
    *,
    max_events: int,
) -> dict[str, Any] | None:
    incident = record.get("incident")
    if not isinstance(incident, dict):
        return None
    events = incident.get("events")
    if not isinstance(events, list) or not events:
        return None
    # The prompt budget bounds how many events a model can observe; the tail is
    # what carries the failure, so an over-long trajectory keeps its tail.
    if max_events > 0 and len(events) > max_events:
        events = events[-max_events:]
    event_ids = {str(event.get("event_id")) for event in events}

    source_refs = record.get("source_refs") or {}
    sample = record.get("sample") or {}
    sample_metadata = dict(sample.get("metadata") or {})
    reference = sample.get("reference") or {}
    evaluator_sample = dict(sample_metadata)
    evaluator_sample.setdefault("reference_answer", reference.get("answer"))
    evaluator_sample["task"] = sample.get("task_text") or ""
    evaluator_sample["question"] = sample_metadata.get("question") or sample.get(
        "task_text"
    )
    evaluator_sample["sample_id"] = record.get("sample_id")

    patch = record.get("patch") if isinstance(record.get("patch"), dict) else None
    localization = (
        record.get("localization")
        if isinstance(record.get("localization"), dict)
        else None
    )
    first_bad = [
        value
        for value in ((localization or {}).get("first_bad_event_ids") or [])
        if str(value) in event_ids
    ]
    cut_event_id = (patch or {}).get("cut_event_id")
    if cut_event_id is not None and str(cut_event_id) not in event_ids:
        cut_event_id = None

    outcome = record.get("outcome") or {}
    control_eval = (record.get("control_run") or {}).get("evaluation") or {}
    failed_eval = (record.get("failed_run") or {}).get("evaluation") or {}

    ledger = _ledger(record, patch, event_ids)
    attribution = (
        record.get("attribution")
        if isinstance(record.get("attribution"), dict)
        else None
    )
    task_id = str(record.get("task_id") or record["incident"]["incident_id"])
    return {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "incident_id": str(incident["incident_id"]),
        "task_id": task_id,
        "split": assignment.get(task_id, "train"),
        "cell": {
            "mas_id": str(source_refs.get("mas_id") or "unknown"),
            "benchmark_id": str(source_refs.get("benchmark_id") or "unknown"),
            "host_model_id": _host_model_id(source_refs),
        },
        "family_id": str((attribution or {}).get("primary_code") or "unknown"),
        "events": events,
        "feedback": incident.get("feedback"),
        "constraints": incident.get("constraints") or {},
        "localization": localization,
        "attribution": _model_visible_attribution(attribution),
        "certified": {
            "cut_event_id": cut_event_id,
            "first_bad_event_ids": first_bad,
            "repair": patch,
        },
        "evaluator_sample": evaluator_sample,
        "control": {
            "success": control_eval.get("success"),
            "score": control_eval.get("score"),
        },
        "source": {
            "success": failed_eval.get("success", outcome.get("source_success")),
            "score": failed_eval.get("score"),
        },
        "ledger": ledger,
        "verified_successful": bool(outcome.get("verified_successful", False)),
        "record_type": str(record.get("record_type") or "unknown"),
        "source_dataset_id": record["__source_dataset_id"],
    }


def _host_model_id(source_refs: dict[str, Any]) -> str:
    """Normalize ``model/<id>`` to the bare host-model identifier.

    The held-out axes of RQ2 are named by the host model, so the identifier
    stored here has to match the one :mod:`aria.corpus` recovers from the source
    dataset id, or a withheld model would only be withheld from one artifact.
    """

    value = str(source_refs.get("model_id") or "unknown")
    return value.split("/", 1)[1] if value.startswith("model/") else value


def _model_visible_attribution(attribution: dict[str, Any] | None) -> dict[str, Any] | None:
    if attribution is None:
        return None
    return {
        key: attribution[key]
        for key in (
            "diagnosis",
            "primary_code",
            "secondary_codes",
            "responsible_agents",
            "counterfactual",
            "patch_operator",
            "confidence",
        )
        if key in attribution
    }


def _ledger(
    record: dict[str, Any],
    patch: dict[str, Any] | None,
    event_ids: set[str],
) -> list[dict[str, Any]]:
    """Collect every splice this incident actually executed and had scored."""

    entries: list[dict[str, Any]] = []
    seen: set[tuple[Any, str]] = set()

    def add(
        label: str,
        cut_event_id: Any,
        replacement_events: Any,
        evaluation: Any,
        execution_mode: str,
    ) -> None:
        if not isinstance(evaluation, dict):
            return
        success = evaluation.get("success")
        if success is None:
            success = evaluation.get("correct")
        if not isinstance(success, bool):
            return
        if not isinstance(replacement_events, list) or not replacement_events:
            return
        if cut_event_id is not None and str(cut_event_id) not in event_ids:
            return
        signature = (
            cut_event_id,
            json.dumps(replacement_events, sort_keys=True)[:512],
        )
        if signature in seen:
            return
        seen.add(signature)
        entries.append(
            {
                "label": label,
                "cut_event_id": cut_event_id,
                "replacement_events": replacement_events,
                "success": success,
                "score": evaluation.get("score"),
                "execution_mode": execution_mode,
            }
        )

    for trial in record.get("repair_trials") or []:
        if not isinstance(trial, dict):
            continue
        repair = trial.get("repair")
        if not isinstance(repair, dict):
            continue
        add(
            str(trial.get("trial_id") or "trial"),
            repair.get("cut_event_id"),
            repair.get("replacement_events"),
            trial.get("evaluation"),
            str((trial.get("execution") or {}).get("execution_mode") or "unknown"),
        )

    if patch is not None:
        add(
            "verified_patch",
            patch.get("cut_event_id"),
            patch.get("replacement_events"),
            (record.get("repaired_run") or {}).get("evaluation"),
            str((record.get("outcome") or {}).get("execution_mode") or "unknown"),
        )

    # The observed failed continuation is an executed splice too: it is the
    # trajectory the harness actually ran, and the evaluator rejected it.
    incident_events = record.get("incident", {}).get("events") or []
    if patch is not None and patch.get("cut_event_id") is not None:
        cut = str(patch["cut_event_id"])
        index = next(
            (
                position
                for position, event in enumerate(incident_events)
                if str(event.get("event_id")) == cut
            ),
            None,
        )
        if index is not None and index + 1 < len(incident_events):
            add(
                "observed_failed_continuation",
                cut,
                incident_events[index + 1 :],
                (record.get("failed_run") or {}).get("evaluation"),
                "source_execution",
            )
    return entries


if __name__ == "__main__":
    main()
