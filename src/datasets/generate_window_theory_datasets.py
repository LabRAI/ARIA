"""Build Window-Theory evidence and trainer-ready ARIA datasets.

This generator is intentionally scoped to the fixed Window-Theory repair layout::

    outputs/window_theory_repair/repair_datasets/
      <mas>/<benchmark>/<cell>/dataset.jsonl

It runs in two layers.  ``evidence`` resolves every canonical artifact referenced
by a v4 repair row and writes one self-contained JSONL file per benchmark.
``training-ready`` projects verified evidence into compact ``prompt``/``chosen``
and ``prompt``/``chosen``/``rejected`` corpora for SFT and DPO.

No inference result is changed.  Re-running the generator atomically replaces
only ``outputs/window_theory_datasets`` snapshots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO, TextIO

from common.window_theory import BENCHMARK_SLUGS, MAS_SLUGS, RUNTIMES
from repair.schema import LEGACY_REPAIR_SCHEMA_VERSION as REPAIR_SCHEMA_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPAIR_ROOT = (
    PROJECT_ROOT / "outputs" / "window_theory_repair" / "repair_datasets"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "window_theory_datasets"

EVIDENCE_SCHEMA_VERSION = "aria-window-theory-evidence-v1"
MANIFEST_SCHEMA_VERSION = "aria-window-theory-evidence-manifest-v1"
TRAINING_READY_SCHEMA_VERSION = "aria-training-ready-v1"
TRAINING_MANIFEST_SCHEMA_VERSION = "aria-training-ready-manifest-v1"
GLOBAL_DATASET_ID = "window-theory-repair"

RUN_PHASES = ("failed", "control", "repaired")
REPAIR_JSON_FILES = (
    "patch.json",
    "policy_decision.json",
)
REPAIR_JSONL_FILES = (
    "failure_window.jsonl",
    "attempts.jsonl",
    "candidate_patches.jsonl",
)


@dataclass(frozen=True)
class SourceRecord:
    """One latest-per-sample record from a Window-Theory repair cell."""

    dataset_path: Path
    dataset_root: Path
    line_number: int
    raw_line_count: int
    record: dict[str, Any]


@dataclass
class JsonlWriter:
    """Streaming JSONL writer that also computes the final byte checksum."""

    handle: TextIO
    digest: Any
    count: int = 0
    byte_count: int = 0

    def write(self, payload: Mapping[str, Any]) -> None:
        encoded = (
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        self.handle.write(encoded.decode("utf-8"))
        self.digest.update(encoded)
        self.count += 1
        self.byte_count += len(encoded)

    @property
    def checksum(self) -> str:
        return f"sha256:{self.digest.hexdigest()}"


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate self-contained ARIA training data from Window-Theory "
            "repair records."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("all", "evidence", "training-ready"),
        default="all",
        help=(
            "materialize evidence, project existing evidence into SFT/DPO data, "
            "or run both layers (default: all)"
        ),
    )
    parser.add_argument(
        "--repair-root",
        type=Path,
        default=DEFAULT_REPAIR_ROOT,
        help="Window-Theory repair_datasets root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="destination root (default: outputs/window_theory_datasets)",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def load_jsonl(path: Path) -> list[Any]:
    values: list[Any] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


@contextmanager
def atomic_jsonl_writer(path: Path) -> Iterator[JsonlWriter]:
    """Yield a streaming writer and atomically install its output on success."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            writer = JsonlWriter(handle=handle, digest=hashlib.sha256())
            yield writer
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def latest_source_records(dataset_path: Path) -> tuple[list[SourceRecord], int]:
    """Return the last valid row for each sample ID in one resumable cell."""

    latest: dict[str, tuple[int, dict[str, Any]]] = {}
    raw_line_count = 0
    with dataset_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw_line_count += 1
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {dataset_path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(value, dict):
                raise TypeError(f"{dataset_path}:{line_number} must be a JSON object")
            if value.get("schema_version") != REPAIR_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported repair schema at {dataset_path}:{line_number}: "
                    f"{value.get('schema_version')!r}"
                )
            sample_id = value.get("sample_id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(
                    f"{dataset_path}:{line_number} has no non-empty sample_id"
                )
            latest[sample_id] = (line_number, value)

    dataset_root = dataset_path.parent.resolve()
    records = [
        SourceRecord(
            dataset_path=dataset_path.resolve(),
            dataset_root=dataset_root,
            line_number=line_number,
            raw_line_count=raw_line_count,
            record=record,
        )
        for _, (line_number, record) in sorted(latest.items())
    ]
    return records, raw_line_count


def discover_benchmark_sources(
    repair_root: Path,
    benchmark: str,
    *,
    mas_slugs: Iterable[str] = MAS_SLUGS,
) -> list[SourceRecord]:
    """Discover all available MAS/runtime cells for one benchmark."""

    if benchmark not in BENCHMARK_SLUGS:
        raise ValueError(f"unsupported Window-Theory benchmark: {benchmark}")
    root = repair_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"repair dataset root does not exist: {root}")

    records: list[SourceRecord] = []
    for mas in mas_slugs:
        benchmark_root = root / mas / benchmark
        if not benchmark_root.exists():
            continue
        for dataset_path in sorted(benchmark_root.glob("*/dataset.jsonl")):
            cell_records, _ = latest_source_records(dataset_path)
            records.extend(cell_records)
    return records


def benchmark_cell_status(
    repair_root: Path,
    benchmark: str,
    *,
    mas_slugs: Iterable[str] = MAS_SLUGS,
) -> list[dict[str, Any]]:
    """Inspect checked, no-op, partial, and complete cells independently."""

    root = repair_root.resolve()
    statuses: list[dict[str, Any]] = []
    for mas in mas_slugs:
        benchmark_root = root / mas / benchmark
        if not benchmark_root.exists():
            continue
        for cell_root in sorted(
            path for path in benchmark_root.iterdir() if path.is_dir()
        ):
            acceptance_path = cell_root / "trace_acceptance.json"
            dataset_path = cell_root / "dataset.jsonl"
            run_state_path = cell_root / "run_state.json"
            acceptance = (
                load_json(acceptance_path) if acceptance_path.is_file() else None
            )
            eligible_ids = (
                acceptance.get("eligible_failure_ids", [])
                if isinstance(acceptance, dict)
                else []
            )
            if not isinstance(eligible_ids, list):
                raise TypeError(
                    f"trace acceptance eligible_failure_ids must be a list: "
                    f"{acceptance_path}"
                )
            latest_records = 0
            if dataset_path.is_file():
                records, _ = latest_source_records(dataset_path)
                latest_records = len(records)
            run_state = load_json(run_state_path) if run_state_path.is_file() else None
            run_status = (
                run_state.get("status") if isinstance(run_state, dict) else None
            )
            checked = acceptance is not None
            eligible = len(eligible_ids)
            complete = bool(
                checked
                and (
                    eligible == 0
                    or (run_status == "completed" and latest_records == eligible)
                )
            )
            statuses.append(
                {
                    "path": cell_root.relative_to(root).as_posix(),
                    "mas": mas,
                    "runtime": cell_root.name.removeprefix(
                        f"wt30-repair-{mas}-{benchmark}-"
                    ),
                    "checked": checked,
                    "source_failures": (
                        acceptance.get("cognitive_failure_count")
                        if isinstance(acceptance, dict)
                        else None
                    ),
                    "eligible": eligible if checked else None,
                    "records": latest_records,
                    "run_status": run_status,
                    "complete": complete,
                }
            )
    return statuses


def resolve_artifact(dataset_root: Path, reference: str, *, name: str) -> Path:
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"{name} must be a non-empty relative artifact reference")
    relative = Path(reference)
    if relative.is_absolute():
        raise ValueError(f"{name} must be relative, got: {reference}")
    root = dataset_root.resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{name} escapes its repair dataset: {reference}")
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {name}: {resolved}")
    return resolved


def load_phase(
    source: SourceRecord,
    phase: str,
) -> dict[str, Any]:
    summary = source.record.get(f"{phase}_run")
    if not isinstance(summary, dict):
        raise TypeError(
            f"{source.dataset_path}:{source.line_number} has no {phase}_run mapping"
        )

    prefix = f"{source.dataset_path}:{source.line_number} {phase}_run"
    trajectory_path = resolve_artifact(
        source.dataset_root,
        summary.get("trajectory_ref"),
        name=f"{prefix}.trajectory_ref",
    )
    messages_path = resolve_artifact(
        source.dataset_root,
        summary.get("messages_ref"),
        name=f"{prefix}.messages_ref",
    )
    result_path = resolve_artifact(
        source.dataset_root,
        summary.get("result_ref"),
        name=f"{prefix}.result_ref",
    )
    phase_root = trajectory_path.parent
    runtime_path = phase_root / "runtime_state.json"
    evaluation_path = phase_root / "evaluation.json"
    if not runtime_path.is_file():
        raise FileNotFoundError(f"missing canonical runtime state: {runtime_path}")
    if not evaluation_path.is_file():
        raise FileNotFoundError(f"missing canonical evaluation: {evaluation_path}")

    messages = json.loads(messages_path.read_text(encoding="utf-8"))
    if not isinstance(messages, list):
        raise TypeError(f"canonical messages must be a JSON list: {messages_path}")
    return {
        "summary": summary,
        "trajectory": load_jsonl(trajectory_path),
        "messages": messages,
        "result": load_json(result_path),
        "runtime_state": load_json(runtime_path),
        "evaluation": load_json(evaluation_path),
    }


def load_repair_artifacts(source: SourceRecord) -> dict[str, Any]:
    refs = source.record.get("source_refs")
    if not isinstance(refs, dict):
        raise TypeError(
            f"{source.dataset_path}:{source.line_number} has no source_refs mapping"
        )
    sample_root = resolve_directory_reference(
        source.dataset_root,
        refs.get("sample_dir"),
        name=(f"{source.dataset_path}:{source.line_number} source_refs.sample_dir"),
    )
    repair_root = sample_root / "repair"
    if not repair_root.is_dir():
        raise FileNotFoundError(f"missing repair artifact directory: {repair_root}")

    payload: dict[str, Any] = {
        "sample": load_json(sample_root / "sample.json"),
        "labels": load_json(sample_root / "labels.json"),
        "prompt": (repair_root / "prompt.txt").read_text(encoding="utf-8"),
    }
    for filename in REPAIR_JSON_FILES:
        payload[filename.removesuffix(".json")] = load_json(repair_root / filename)
    for filename in REPAIR_JSONL_FILES:
        payload[filename.removesuffix(".jsonl")] = load_jsonl(repair_root / filename)
    return payload


def resolve_directory_reference(
    dataset_root: Path, reference: Any, *, name: str
) -> Path:
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"{name} must be a non-empty relative directory reference")
    relative = Path(reference)
    if relative.is_absolute():
        raise ValueError(f"{name} must be relative, got: {reference}")
    root = dataset_root.resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"{name} escapes its repair dataset: {reference}")
    if not resolved.is_dir():
        raise FileNotFoundError(f"missing {name}: {resolved}")
    return resolved


def portable_value(value: Any) -> Any:
    """Replace this checkout's absolute root while preserving source content."""

    marker = "$PROJECT_ROOT"
    project_root = str(PROJECT_ROOT.resolve())
    if isinstance(value, str):
        return value.replace(project_root, marker)
    if isinstance(value, list):
        return [portable_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): portable_value(item) for key, item in value.items()}
    return value


def runtime_slug(record: Mapping[str, Any]) -> str:
    refs = record.get("source_refs")
    if not isinstance(refs, Mapping):
        raise TypeError("repair record source_refs must be a mapping")
    runtime_id = refs.get("runtime_id")
    if not isinstance(runtime_id, str) or not runtime_id:
        raise ValueError("repair record source_refs.runtime_id must be non-empty")
    return runtime_id.removeprefix("runtime/")


def evidence_selection(
    record: Mapping[str, Any], runs: Mapping[str, Any]
) -> dict[str, Any]:
    """Classify evidence without embedding trainer prompts or answer labels."""

    outcome = record.get("outcome")
    if not isinstance(outcome, Mapping):
        raise TypeError("repair record outcome must be a mapping")

    paired_outcome = outcome.get("paired_outcome")
    repaired = runs.get("repaired")
    repaired_events = (
        repaired.get("trajectory") if isinstance(repaired, Mapping) else None
    )
    repaired_success = outcome.get("repair_success") is True
    trajectory_sft_eligible = bool(repaired_success and repaired_events)
    causal_sft_eligible = trajectory_sft_eligible and paired_outcome == "patched_only"
    if trajectory_sft_eligible:
        eligibility_reason = (
            "verified_patched_only"
            if causal_sft_eligible
            else "verified_repaired_but_non_attributable"
        )
    else:
        eligibility_reason = "repaired_evaluation_not_successful"

    return {
        "trajectory_sft_eligible": trajectory_sft_eligible,
        "causal_repair_sft_eligible": causal_sft_eligible,
        "eligibility_reason": eligibility_reason,
        "paired_outcome": paired_outcome,
        "control_success": outcome.get("control_success") is True,
        "repaired_success": repaired_success,
    }


def standalone_record(source: SourceRecord, repair_root: Path) -> dict[str, Any]:
    record = source.record
    refs = record.get("source_refs")
    if not isinstance(refs, dict):
        raise TypeError("repair record source_refs must be a mapping")
    mas = refs.get("mas_id")
    benchmark = refs.get("benchmark_id")
    sample_id = record.get("sample_id")
    if not all(
        isinstance(value, str) and value for value in (mas, benchmark, sample_id)
    ):
        raise ValueError(
            "repair record MAS, benchmark, and sample IDs must be non-empty"
        )

    expected = source.dataset_path.relative_to(repair_root.resolve()).parts
    if len(expected) < 4 or expected[0] != mas or expected[1] != benchmark:
        raise ValueError(
            f"record identity does not match source path: {source.dataset_path}"
        )
    runtime = runtime_slug(record)
    if runtime not in RUNTIMES:
        raise ValueError(f"unsupported Window-Theory runtime: {runtime}")

    runs = {phase: load_phase(source, phase) for phase in RUN_PHASES}
    repair_artifacts = load_repair_artifacts(source)
    identifier = f"wt-repair::{benchmark}::{mas}::{runtime}::{sample_id}"
    dataset_relative_path = source.dataset_path.relative_to(repair_root.resolve())
    payload = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "record_id": identifier,
        "dataset_id": GLOBAL_DATASET_ID,
        "split": record.get("split") or "train",
        "provenance": {
            "window_theory": True,
            "mas": mas,
            "benchmark": benchmark,
            "runtime": runtime,
            "runtime_id": refs.get("runtime_id"),
            "model_id": refs.get("model_id"),
            "host_adapter_id": refs.get("host_adapter_id"),
            "repair_operator_id": refs.get("repair_operator_id"),
            "source_dataset_id": record.get("dataset_id"),
            "source_record_id": record.get("record_id"),
            "source_dataset_path": dataset_relative_path.as_posix(),
            "source_line_number": source.line_number,
        },
        "repair_record": record,
        "runs": runs,
        "repair_artifacts": repair_artifacts,
    }
    payload["selection"] = evidence_selection(record, runs)
    return portable_value(payload)


def collect_benchmark(
    repair_root: Path,
    output_root: Path,
    benchmark: str,
    *,
    mas_slugs: Iterable[str] = MAS_SLUGS,
) -> dict[str, Any]:
    """Stage 1: materialize all MAS repair rows for one benchmark."""

    repair_root = repair_root.resolve()
    output_root = output_root.resolve()
    mas_order = tuple(mas_slugs)
    cell_statuses = benchmark_cell_status(
        repair_root,
        benchmark,
        mas_slugs=mas_order,
    )
    sources = discover_benchmark_sources(
        repair_root,
        benchmark,
        mas_slugs=mas_order,
    )
    if not sources:
        raise ValueError(f"no Window-Theory repair records found for {benchmark}")

    sources.sort(
        key=lambda source: (
            str(source.record["source_refs"]["mas_id"]),
            runtime_slug(source.record),
            str(source.record["sample_id"]),
        )
    )
    dataset_path = output_root / benchmark / "dataset.jsonl"
    record_ids: set[str] = set()
    mas_counts: Counter[str] = Counter()
    runtime_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    training_counts: Counter[str] = Counter()
    cell_stats: dict[str, dict[str, Any]] = {}

    with atomic_jsonl_writer(dataset_path) as writer:
        for source in sources:
            payload = standalone_record(source, repair_root)
            record_id = str(payload["record_id"])
            if record_id in record_ids:
                raise ValueError(f"duplicate standalone repair record_id: {record_id}")
            record_ids.add(record_id)
            writer.write(payload)

            provenance = payload["provenance"]
            selection = payload["selection"]
            mas_counts[str(provenance["mas"])] += 1
            runtime_counts[str(provenance["runtime"])] += 1
            outcome_counts[str(selection["paired_outcome"])] += 1
            if selection["trajectory_sft_eligible"]:
                training_counts["trajectory_sft_eligible"] += 1
            if selection["causal_repair_sft_eligible"]:
                training_counts["causal_repair_sft_eligible"] += 1

            cell_key = str(provenance["source_dataset_path"])
            if cell_key not in cell_stats:
                cell_stats[cell_key] = {
                    "path": cell_key,
                    "mas": provenance["mas"],
                    "runtime": provenance["runtime"],
                    "raw_records": source.raw_line_count,
                    "latest_records": 0,
                    "source_sha256": sha256_file(source.dataset_path),
                }
            cell_stats[cell_key]["latest_records"] += 1

        checksum = writer.checksum
        byte_count = writer.byte_count
        record_count = writer.count

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_schema_version": EVIDENCE_SCHEMA_VERSION,
        "dataset_id": f"{GLOBAL_DATASET_ID}/{benchmark}",
        "scope": "benchmark",
        "benchmark": benchmark,
        "generated_at": utc_now(),
        "source_repair_root": portable_value(str(repair_root)),
        "dataset_path": f"{benchmark}/dataset.jsonl",
        "records": record_count,
        "bytes": byte_count,
        "sha256": checksum,
        "record_cells": len(cell_stats),
        "checked_cells": sum(1 for cell in cell_statuses if cell["checked"]),
        "complete_cells": sum(1 for cell in cell_statuses if cell["complete"]),
        "expected_cells": len(mas_order) * len(RUNTIMES),
        "matrix_complete": sum(1 for cell in cell_statuses if cell["complete"])
        == len(mas_order) * len(RUNTIMES),
        "counts": {
            "mas": dict(sorted(mas_counts.items())),
            "runtime": dict(sorted(runtime_counts.items())),
            "paired_outcome": dict(sorted(outcome_counts.items())),
            "training": dict(sorted(training_counts.items())),
        },
        "source_cells": [cell_stats[key] for key in sorted(cell_stats)],
        "matrix_cells": cell_statuses,
    }
    atomic_write_json(output_root / benchmark / "manifest.json", manifest)
    return manifest


def copy_validated_jsonl(
    source_path: Path,
    destination: BinaryIO,
    digest: Any,
    *,
    benchmark: str,
    seen_record_ids: set[str],
) -> tuple[int, int, Counter[str], Counter[str]]:
    count = 0
    byte_count = 0
    outcome_counts: Counter[str] = Counter()
    training_counts: Counter[str] = Counter()
    with source_path.open("rb") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON at {source_path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise TypeError(f"{source_path}:{line_number} must be a JSON object")
            if payload.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported dataset row at {source_path}:{line_number}"
                )
            provenance = payload.get("provenance")
            if (
                not isinstance(provenance, dict)
                or provenance.get("benchmark") != benchmark
            ):
                raise ValueError(
                    f"benchmark mismatch at {source_path}:{line_number}; expected {benchmark}"
                )
            record_id = payload.get("record_id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError(f"missing record_id at {source_path}:{line_number}")
            if record_id in seen_record_ids:
                raise ValueError(f"duplicate global repair record_id: {record_id}")
            seen_record_ids.add(record_id)

            if not line.endswith(b"\n"):
                line += b"\n"
            destination.write(line)
            digest.update(line)
            count += 1
            byte_count += len(line)

            selection = payload.get("selection")
            if isinstance(selection, dict):
                outcome_counts[str(selection.get("paired_outcome"))] += 1
                if selection.get("trajectory_sft_eligible") is True:
                    training_counts["trajectory_sft_eligible"] += 1
                if selection.get("causal_repair_sft_eligible") is True:
                    training_counts["causal_repair_sft_eligible"] += 1
    return count, byte_count, outcome_counts, training_counts


def combine_benchmarks(
    output_root: Path,
    *,
    benchmarks: Iterable[str] = BENCHMARK_SLUGS,
) -> dict[str, Any]:
    """Stage 2: validate and combine benchmark snapshots into one JSONL."""

    output_root = output_root.resolve()
    benchmark_order = tuple(benchmarks)
    dataset_path = output_root / "dataset.jsonl"
    temporary = dataset_path.with_name(f".{dataset_path.name}.{os.getpid()}.tmp")
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    total_records = 0
    total_bytes = 0
    seen_record_ids: set[str] = set()
    source_manifests: list[dict[str, Any]] = []
    outcome_counts: Counter[str] = Counter()
    training_counts: Counter[str] = Counter()
    record_cells = 0
    checked_cells = 0
    complete_cells = 0
    expected_cells = 0
    matrix_complete = True

    try:
        with temporary.open("wb") as destination:
            for benchmark in benchmark_order:
                benchmark_root = output_root / benchmark
                source_path = benchmark_root / "dataset.jsonl"
                manifest_path = benchmark_root / "manifest.json"
                if not source_path.is_file() or not manifest_path.is_file():
                    raise FileNotFoundError(
                        f"missing collected benchmark dataset or manifest: {benchmark_root}"
                    )
                manifest = load_json(manifest_path)
                expected_checksum = manifest.get("sha256")
                actual_checksum = sha256_file(source_path)
                if expected_checksum != actual_checksum:
                    raise ValueError(
                        f"benchmark dataset checksum mismatch for {benchmark}: "
                        f"{expected_checksum!r} != {actual_checksum!r}"
                    )
                count, byte_count, outcomes, training = copy_validated_jsonl(
                    source_path,
                    destination,
                    digest,
                    benchmark=benchmark,
                    seen_record_ids=seen_record_ids,
                )
                if count != manifest.get("records"):
                    raise ValueError(
                        f"benchmark record count mismatch for {benchmark}: "
                        f"{count} != {manifest.get('records')!r}"
                    )
                total_records += count
                total_bytes += byte_count
                outcome_counts.update(outcomes)
                training_counts.update(training)
                record_cells += int(manifest.get("record_cells") or 0)
                checked_cells += int(manifest.get("checked_cells") or 0)
                complete_cells += int(manifest.get("complete_cells") or 0)
                expected_cells += int(manifest.get("expected_cells") or 0)
                matrix_complete = (
                    matrix_complete and manifest.get("matrix_complete") is True
                )
                source_manifests.append(
                    {
                        "benchmark": benchmark,
                        "path": f"{benchmark}/dataset.jsonl",
                        "records": count,
                        "bytes": byte_count,
                        "sha256": actual_checksum,
                        "record_cells": manifest.get("record_cells"),
                        "checked_cells": manifest.get("checked_cells"),
                        "complete_cells": manifest.get("complete_cells"),
                        "expected_cells": manifest.get("expected_cells"),
                        "matrix_complete": manifest.get("matrix_complete"),
                    }
                )
            destination.flush()
            os.fsync(destination.fileno())
        temporary.replace(dataset_path)
    finally:
        temporary.unlink(missing_ok=True)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_schema_version": EVIDENCE_SCHEMA_VERSION,
        "dataset_id": GLOBAL_DATASET_ID,
        "scope": "all-benchmarks",
        "generated_at": utc_now(),
        "dataset_path": "dataset.jsonl",
        "benchmark_order": list(benchmark_order),
        "records": total_records,
        "bytes": total_bytes,
        "sha256": f"sha256:{digest.hexdigest()}",
        "record_cells": record_cells,
        "checked_cells": checked_cells,
        "complete_cells": complete_cells,
        "expected_cells": expected_cells,
        "matrix_complete": matrix_complete,
        "counts": {
            "paired_outcome": dict(sorted(outcome_counts.items())),
            "training": dict(sorted(training_counts.items())),
        },
        "benchmark_datasets": source_manifests,
    }
    atomic_write_json(output_root / "manifest.json", manifest)
    return manifest


def _required_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _required_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _normalized_event(raw: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    event_id = _required_text(raw.get("event_id"), name=f"{name}.event_id")
    role = next(
        (
            value
            for value in (
                raw.get("role"),
                raw.get("mas_agent_role"),
                raw.get("mas_agent_id"),
            )
            if isinstance(value, str) and value.strip()
        ),
        "unknown",
    )
    kind = raw.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        kind = "message"
    tool_calls = raw.get("tool_calls", [])
    if not isinstance(tool_calls, list) or not all(
        isinstance(item, Mapping) for item in tool_calls
    ):
        raise TypeError(f"{name}.tool_calls must be a list of mappings")
    metadata = {
        key: portable_value(raw[key])
        for key in (
            "turn",
            "mas_agent_id",
            "mas_agent_role",
            "hook",
            "tool_results",
            "detector",
        )
        if raw.get(key) is not None
    }
    return {
        "event_id": event_id,
        "role": role,
        "kind": kind,
        "content": str(raw.get("content", "")),
        "tool_calls": [dict(item) for item in tool_calls],
        "metadata": metadata,
    }


def _failure_evidence_events(
    localization: Mapping[str, Any],
    *,
    existing_ids: set[str],
) -> list[dict[str, Any]]:
    raw_items = localization.get("failure_evidence", [])
    if not isinstance(raw_items, list):
        raise TypeError("repair_record.localization.failure_evidence must be a list")
    role = next(
        (
            value
            for value in (
                localization.get("suspect_mas_agent_role"),
                localization.get("suspect_mas_agent_id"),
            )
            if isinstance(value, str) and value.strip()
        ),
        "repair_target",
    )
    events: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, Mapping):
            raise TypeError(f"failure_evidence[{index}] must be a mapping")
        event_id = raw.get("event_id")
        if not isinstance(event_id, str) or not event_id or event_id in existing_ids:
            continue
        source = raw.get("source")
        events.append(
            {
                "event_id": event_id,
                "role": role,
                "kind": "artifact" if source == "artifact" else "failure_evidence",
                "content": str(raw.get("quote", "")),
                "tool_calls": [],
                "metadata": {
                    key: portable_value(raw[key])
                    for key in ("source", "reason")
                    if raw.get(key) is not None
                },
            }
        )
        existing_ids.add(event_id)
    return events


def _localized_target_event_id(
    record: Mapping[str, Any],
    events: list[dict[str, Any]],
) -> str:
    localization = _required_mapping(
        record.get("localization"), name="repair_record.localization"
    )
    patch = _required_mapping(record.get("patch"), name="repair_record.patch")
    patch_target = patch.get("patch_target", {})
    if not isinstance(patch_target, Mapping):
        raise TypeError("repair_record.patch.patch_target must be a mapping")
    requested = patch_target.get("target_event_id") or localization.get(
        "first_bad_event_id"
    )
    requested = _required_text(requested, name="localized target event ID")
    event_ids = {str(event["event_id"]) for event in events}
    if requested in event_ids:
        return requested
    source_alias = requested.replace("failed:event:", "source:event:", 1)
    if source_alias in event_ids:
        return source_alias

    evidence_events = _failure_evidence_events(localization, existing_ids=event_ids)
    events.extend(evidence_events)
    if requested in event_ids:
        return requested

    role = next(
        (
            value
            for value in (
                localization.get("suspect_mas_agent_role"),
                localization.get("suspect_mas_agent_id"),
            )
            if isinstance(value, str) and value.strip()
        ),
        "repair_target",
    )
    events.append(
        {
            "event_id": requested,
            "role": role,
            "kind": "failure_marker",
            "content": "",
            "tool_calls": [],
            "metadata": {"synthetic": True, "reason": "localized target marker"},
        }
    )
    return requested


def _diagnosis(localization: Mapping[str, Any]) -> str:
    failure_type = localization.get("failure_type")
    raw_evidence = localization.get("failure_evidence", [])
    reason = None
    if isinstance(raw_evidence, list):
        for item in raw_evidence:
            if isinstance(item, Mapping) and isinstance(item.get("reason"), str):
                reason = item["reason"].strip()
                if reason:
                    break
    parts = [
        value.strip()
        for value in (failure_type, reason)
        if isinstance(value, str) and value.strip()
    ]
    return ": ".join(parts) or "The observed continuation failed evaluation."


def _validate_repair_payload(
    repair: Mapping[str, Any],
    *,
    event_ids: set[str],
    name: str,
) -> None:
    _required_text(repair.get("diagnosis"), name=f"{name}.diagnosis")
    cut_event_id = repair.get("cut_event_id")
    if cut_event_id is not None and cut_event_id not in event_ids:
        raise ValueError(f"{name}.cut_event_id does not identify a prompt event")
    replacements = repair.get("replacement_events")
    if not isinstance(replacements, list) or not replacements:
        raise ValueError(f"{name}.replacement_events must be a non-empty list")
    replacement_ids: list[str] = []
    for index, replacement in enumerate(replacements):
        if not isinstance(replacement, Mapping):
            raise TypeError(f"{name}.replacement_events[{index}] must be a mapping")
        normalized = _normalized_event(
            replacement,
            name=f"{name}.replacement_events[{index}]",
        )
        replacement_ids.append(str(normalized["event_id"]))
    if len(set(replacement_ids)) != len(replacement_ids):
        raise ValueError(f"{name}.replacement_events contains duplicate event IDs")


def _training_ready_row(evidence: Mapping[str, Any]) -> dict[str, Any]:
    record_id = _required_text(evidence.get("record_id"), name="evidence.record_id")
    record = _required_mapping(
        evidence.get("repair_record"), name="evidence.repair_record"
    )
    runs = _required_mapping(evidence.get("runs"), name="evidence.runs")
    failed = _required_mapping(runs.get("failed"), name="evidence.runs.failed")
    trajectory = failed.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError(f"{record_id} has no failed trajectory for training")
    if not all(isinstance(item, Mapping) for item in trajectory):
        raise TypeError(f"{record_id} failed trajectory must contain mappings")
    events = [
        _normalized_event(item, name=f"{record_id}.failed_trajectory[{index}]")
        for index, item in enumerate(trajectory)
    ]
    event_ids = [str(event["event_id"]) for event in events]
    if len(set(event_ids)) != len(event_ids):
        raise ValueError(f"{record_id} failed trajectory has duplicate event IDs")

    sample = _required_mapping(record.get("sample"), name="repair_record.sample")
    localization = _required_mapping(
        record.get("localization"), name="repair_record.localization"
    )
    patch = _required_mapping(record.get("patch"), name="repair_record.patch")
    if patch.get("answer_blind") is not True:
        raise ValueError(f"{record_id} repair is not marked answer_blind")
    target_event_id = _localized_target_event_id(record, events)
    target_index = next(
        index
        for index, event in enumerate(events)
        if event["event_id"] == target_event_id
    )
    cut_event_id = events[target_index - 1]["event_id"] if target_index else None

    raw_evidence = localization.get("failure_evidence", [])
    feedback = None
    if isinstance(raw_evidence, list):
        for item in raw_evidence:
            if isinstance(item, Mapping) and isinstance(item.get("reason"), str):
                feedback = item["reason"].strip() or None
                if feedback:
                    break
    prompt = {
        "incident_id": record_id,
        "events": events,
        "feedback": feedback,
        "constraints": {
            "task_text": sample.get("task_text"),
            "localization": {
                key: portable_value(localization[key])
                for key in (
                    "mast_labels",
                    "failure_stage",
                    "failure_type",
                    "first_bad_event_id",
                    "suspect_mas_agent_id",
                    "suspect_mas_agent_role",
                    "label_confidence",
                )
                if localization.get(key) is not None
            },
            "answer_blind": True,
        },
    }

    patch_instruction = _required_text(
        patch.get("patch_instruction"), name=f"{record_id} patch_instruction"
    )
    patch_target = patch.get("patch_target", {})
    replacement_role = "repair_target"
    if isinstance(patch_target, Mapping):
        for key in ("mas_agent_role", "mas_agent_id"):
            value = patch_target.get(key)
            if isinstance(value, str) and value.strip():
                replacement_role = value
                break
    patch_id = patch.get("patch_id")
    replacement_event = {
        "event_id": f"repair:{patch_id or record_id}:instruction",
        "role": replacement_role,
        "kind": "message",
        "content": patch_instruction,
        "tool_calls": [],
        "metadata": {
            key: portable_value(patch[key])
            for key in (
                "patch_id",
                "operation",
                "patch_family",
                "patch_strategy",
                "patch_timing",
                "intervention_level",
                "patch_target",
                "expected_change",
            )
            if patch.get(key) is not None
        },
    }
    chosen = {
        "diagnosis": _diagnosis(localization),
        "cut_event_id": cut_event_id,
        "replacement_events": [replacement_event],
    }
    rejected = {
        "diagnosis": "Keep the observed failed continuation without the verified repair.",
        "cut_event_id": cut_event_id,
        "replacement_events": events[target_index:],
    }

    prompt_event_ids = {str(event["event_id"]) for event in events}
    _validate_repair_payload(
        chosen,
        event_ids=prompt_event_ids,
        name=f"{record_id}.chosen",
    )
    _validate_repair_payload(
        rejected,
        event_ids=prompt_event_ids,
        name=f"{record_id}.rejected",
    )
    selection = _required_mapping(evidence.get("selection"), name="evidence.selection")
    return {
        "schema_version": TRAINING_READY_SCHEMA_VERSION,
        "record_id": record_id,
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
        "metadata": {
            "evidence_record_id": record_id,
            "split": evidence.get("split"),
            "provenance": portable_value(evidence.get("provenance", {})),
            "paired_outcome": selection.get("paired_outcome"),
            "causal_repair": selection.get("causal_repair_sft_eligible") is True,
        },
    }


def materialize_training_ready(
    evidence_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Project evidence into compact SFT and causal DPO corpora."""

    evidence_path = evidence_path.resolve()
    output_root = output_root.resolve()
    if not evidence_path.is_file():
        raise FileNotFoundError(f"evidence dataset does not exist: {evidence_path}")
    evidence_manifest_path = evidence_path.parent / "manifest.json"
    evidence_manifest = (
        load_json(evidence_manifest_path) if evidence_manifest_path.is_file() else None
    )
    if evidence_manifest is not None:
        if evidence_manifest.get("dataset_schema_version") != EVIDENCE_SCHEMA_VERSION:
            raise ValueError(
                f"evidence manifest has unsupported dataset schema: {evidence_manifest_path}"
            )
        actual_evidence_sha256 = sha256_file(evidence_path)
        if evidence_manifest.get("sha256") != actual_evidence_sha256:
            raise ValueError(
                f"evidence checksum mismatch: {evidence_manifest.get('sha256')!r} "
                f"!= {actual_evidence_sha256!r}"
            )
    sft_path = output_root / "sft.jsonl"
    dpo_path = output_root / "dpo.jsonl"
    seen: set[str] = set()
    evidence_records = 0
    outcome_counts: Counter[str] = Counter()

    with (
        atomic_jsonl_writer(sft_path) as sft_writer,
        atomic_jsonl_writer(dpo_path) as dpo_writer,
        evidence_path.open(encoding="utf-8") as source,
    ):
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            evidence_records += 1
            try:
                evidence = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid evidence JSON at {evidence_path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(evidence, Mapping):
                raise TypeError(
                    f"evidence line {line_number} must contain a JSON object"
                )
            if evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
                raise ValueError(
                    f"unsupported evidence schema at {evidence_path}:{line_number}"
                )
            record_id = _required_text(
                evidence.get("record_id"), name=f"evidence line {line_number} record_id"
            )
            if record_id in seen:
                raise ValueError(f"duplicate evidence record_id: {record_id}")
            seen.add(record_id)
            selection = _required_mapping(
                evidence.get("selection"), name=f"{record_id}.selection"
            )
            outcome_counts[str(selection.get("paired_outcome"))] += 1
            if selection.get("trajectory_sft_eligible") is not True:
                continue
            row = _training_ready_row(evidence)
            sft_writer.write(
                {key: value for key, value in row.items() if key != "rejected"}
            )
            if selection.get("causal_repair_sft_eligible") is True:
                dpo_writer.write(row)

        if (
            evidence_manifest is not None
            and evidence_manifest.get("records") != evidence_records
        ):
            raise ValueError(
                "evidence record count does not match its collection manifest"
            )
        datasets = {
            "sft": {
                "path": "sft.jsonl",
                "records": sft_writer.count,
                "bytes": sft_writer.byte_count,
                "sha256": sft_writer.checksum,
            },
            "dpo": {
                "path": "dpo.jsonl",
                "records": dpo_writer.count,
                "bytes": dpo_writer.byte_count,
                "sha256": dpo_writer.checksum,
            },
        }

    manifest = {
        "schema_version": TRAINING_MANIFEST_SCHEMA_VERSION,
        "dataset_schema_version": TRAINING_READY_SCHEMA_VERSION,
        "dataset_id": f"{GLOBAL_DATASET_ID}/training-ready",
        "generated_at": utc_now(),
        "source_evidence_path": portable_value(str(evidence_path)),
        "source_evidence_sha256": sha256_file(evidence_path),
        "selection_policy": {
            "sft": "verified repaired trajectory",
            "dpo": "patched_only causal repair",
        },
        "evidence_records": evidence_records,
        "counts": {
            "paired_outcome": dict(sorted(outcome_counts.items())),
            "sft": datasets["sft"]["records"],
            "dpo": datasets["dpo"]["records"],
        },
        "datasets": datasets,
    }
    atomic_write_json(output_root / "manifest.json", manifest)
    return manifest


def generate(
    repair_root: Path = DEFAULT_REPAIR_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    stage: str = "all",
) -> dict[str, Any]:
    if stage not in {"all", "evidence", "training-ready"}:
        raise ValueError(f"unsupported generation stage: {stage}")
    output_root = output_root.resolve()
    evidence_root = output_root / "evidence"
    training_root = output_root / "training_ready"
    benchmark_manifests: list[dict[str, Any]] = []
    if stage in {"all", "evidence"}:
        for benchmark in BENCHMARK_SLUGS:
            benchmark_manifests.append(
                collect_benchmark(repair_root, evidence_root, benchmark)
            )
    evidence_manifest = (
        combine_benchmarks(evidence_root)
        if stage in {"all", "evidence"}
        else load_json(evidence_root / "manifest.json")
    )
    training_manifest = (
        materialize_training_ready(evidence_root / "dataset.jsonl", training_root)
        if stage in {"all", "training-ready"}
        else None
    )
    return {
        "stage": stage,
        "output_root": str(output_root),
        "benchmarks": benchmark_manifests,
        "evidence": evidence_manifest,
        "training_ready": training_manifest,
    }


def main() -> None:
    args = parse_args()
    result = generate(
        repair_root=args.repair_root,
        output_root=args.output_root,
        stage=args.stage,
    )
    evidence = result["evidence"]
    training_ready = result["training_ready"]
    summary = {
        "stage": result["stage"],
        "output_root": result["output_root"],
        "benchmarks": [
            {
                "benchmark": manifest["benchmark"],
                "records": manifest["records"],
                "complete_cells": manifest["complete_cells"],
                "expected_cells": manifest["expected_cells"],
            }
            for manifest in result["benchmarks"]
        ],
        "evidence": (
            {
                "dataset_path": "evidence/dataset.jsonl",
                "records": evidence["records"],
                "bytes": evidence["bytes"],
                "sha256": evidence["sha256"],
                "complete_cells": evidence["complete_cells"],
                "expected_cells": evidence["expected_cells"],
                "matrix_complete": evidence["matrix_complete"],
                "counts": evidence["counts"],
            }
            if isinstance(evidence, dict)
            else None
        ),
        "training_ready": (
            {
                "sft": training_ready["datasets"]["sft"],
                "dpo": training_ready["datasets"]["dpo"],
            }
            if isinstance(training_ready, dict)
            else None
        ),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
