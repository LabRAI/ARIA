"""Dataset-level metadata, manifest, and README generation."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from common.artifacts import atomic_write_text, utc_now
from common.preparation import git_head
from repair.schema import REPAIR_SCHEMA_VERSION


def build_metadata(
    *,
    name: str,
    spec: dict[str, Any],
    source_batch_dir: Path,
    dataset_root: Path,
    config: Any,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = getattr(config, "config_path", None)
    config_path_value = str(config_path) if config_path is not None else None
    return {
        "dataset_id": name,
        "schema_version": REPAIR_SCHEMA_VERSION,
        "created_at": utc_now(),
        "builder": {
            "command": f"uv run python main.py -c {config_path_value}" if config_path_value else None,
            "config_path": config_path_value,
            "config_sha256": (
                f"sha256:{file_sha256(Path(config_path_value))}"
                if config_path_value and Path(config_path_value).exists()
                else None
            ),
            "code_git_head": git_head(repo_root),
        },
        "source_batches": [
            {
                "source_batch_id": str(spec.get("source_batch_id") or source_batch_dir.name),
                "path": str(source_batch_dir),
                "result_file": "results.jsonl",
            }
        ],
        "evaluators": unique_evaluators(records),
        "label_taxonomy": {
            "primary": "ARIA-RCA-v1",
            "cognitive_families": [
                "RA-1",
                "RA-2",
                "RA-3",
                "RA-4",
                "RA-5",
                "RA-6",
            ],
            "mast_compatibility": "mast-14",
        },
        "paths": {
            "dataset_dir": str(dataset_root),
            "dataset_jsonl": "dataset.jsonl",
            "invalid_samples_jsonl": "invalid_samples.jsonl",
            "samples_dir": "samples",
        },
    }


def unique_evaluators(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        benchmark = (record.get("sample") or {}).get("benchmark")
        for run_key in ("failed_run", "control_run", "repaired_run"):
            run = record.get(run_key)
            if not isinstance(run, dict):
                continue
            evaluation = run.get("evaluation") if isinstance(run.get("evaluation"), dict) else {}
            evaluator = evaluation.get("evaluator_id")
            if evaluator and evaluator not in by_id:
                by_id[str(evaluator)] = {
                    "evaluator_id": evaluator,
                    "benchmark": benchmark,
                    "method": evaluation.get("method"),
                }
    return list(by_id.values())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    *,
    name: str,
    spec: dict[str, Any],
    source_batch_dir: Path,
    dataset_root: Path,
    repair_output_root: Path,
    records: list[dict[str, Any]],
    schema_version: str = REPAIR_SCHEMA_VERSION,
    status: str = "completed",
    invalid_sample_ids: set[str] | None = None,
    invalid_attempts: int = 0,
) -> dict[str, Any]:
    if schema_version != REPAIR_SCHEMA_VERSION:
        raise ValueError(f"unsupported repair schema: {schema_version}")
    successful = sum(
        1
        for item in records
        if item["outcome"].get("verified_successful") is True
    )
    attempted = sum(1 for item in records if item["outcome"].get("repair_attempted") is True)
    failed = sum(
        1
        for item in records
        if item["outcome"].get("repair_attempted") is True
        and item["outcome"].get("verified_successful") is not True
    )
    splits: dict[str, int] = {}
    for item in records:
        split = str(item.get("split") or "train")
        splits[split] = splits.get(split, 0) + 1
    dataset_path = dataset_root / "dataset.jsonl"
    return {
        "dataset_id": name,
        "name": name,
        "schema_version": schema_version,
        "created_at": utc_now(),
        "status": status,
        "paths": {
            "dataset_jsonl": "dataset.jsonl",
            "invalid_samples_jsonl": "invalid_samples.jsonl",
            "metadata_json": "metadata.json",
            "samples_dir": "samples",
        },
        "counts": {
            "records": len(records),
            "samples": len({item.get("sample_id") for item in records}),
            "attempted_repairs": attempted,
            "successful_repairs": successful,
            "failed_repairs": failed,
            "paired_outcomes": {
                outcome: sum(
                    1
                    for item in records
                    if item.get("outcome", {}).get("paired_outcome") == outcome
                )
                for outcome in ("patched_only", "both_pass", "control_only", "both_fail")
            },
            "training_ready": {
                key: sum(
                    1
                    for item in records
                    if item.get("training", {}).get(key) is True
                )
                for key in (
                    "gate_sft_eligible",
                    "localization_sft_eligible",
                    "attribution_sft_eligible",
                    "structured_repair_sft_eligible",
                    "repair_dpo_eligible",
                )
            },
            "gate_success_samples": sum(
                1
                for item in records
                if item.get("record_type")
                in {"gate_success_sample", "success_noop"}
            ),
            "infrastructure_invalid": len(invalid_sample_ids or set()),
            "invalid_attempts": invalid_attempts,
        },
        "splits": splits,
        "checksums": {
            "dataset.jsonl": f"sha256:{file_sha256(dataset_path)}" if dataset_path.exists() else None,
        },
        "source_batch_dir": str(source_batch_dir),
        "dataset_dir": str(dataset_root),
        "repair_output_root": str(repair_output_root),
        "sample_ids": [item["sample_id"] for item in records],
        "invalid_sample_ids": sorted(invalid_sample_ids or set()),
        "spec": spec,
    }


def write_readme(dataset_root: Path, manifest: dict[str, Any]) -> None:
    counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}
    text = (
        f"# {manifest['name']}\n\n"
        f"Schema: `{REPAIR_SCHEMA_VERSION}`\n\n"
        "This folder contains canonical ARIA repair-supervision records generated from failed MAS runs.\n\n"
        "- `dataset.jsonl`: V6 localization, RA attribution, repair trials, and paired outcomes.\n"
        "- `invalid_samples.jsonl`: non-training audit events for infrastructure-invalid samples.\n"
        "- `metadata.json`: dataset builder, source-batch, evaluator, and config provenance.\n"
        "- Each `dataset.jsonl` sample stores its own `repair_endpoint`; API key values are never stored.\n"
        "- `manifest.json`: inventory, counts, split statistics and checksums.\n"
        "- `samples/<sample_id>/failed/`: source run artifacts.\n"
        "- `samples/<sample_id>/control/`: unpatched paired-rerun artifacts.\n"
        "- `samples/<sample_id>/repair/`: model candidates, host validation, policy, and attempts.\n"
        "- `samples/<sample_id>/repaired/`: primary patched-rerun artifacts.\n\n"
        "- `training_ready/`: gate, attribution, patch SFT, and repair DPO JSONL.\n\n"
        f"Records: {counts.get('records', 0)}. "
        f"Attempted repairs: {counts.get('attempted_repairs', 0)}. "
        f"Successful repairs: {counts.get('successful_repairs', 0)}.\n"
        f"Infrastructure-invalid samples: {counts.get('infrastructure_invalid', 0)}.\n"
    )
    atomic_write_text(dataset_root / "README.md", text)
