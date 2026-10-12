"""External MAS checkout and benchmark cache preparation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from common.catalog import (
    BenchmarkSpec,
    selected_benchmarks,
    selected_mas,
)
from common.mast_data import extract_task_text, load_gaia_items, load_mad_records

GAIA_REPO_ID = "gaia-benchmark/GAIA"


def huggingface_token() -> str | None:
    """Return the configured Hugging Face token without persisting it."""

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import HfFolder
    except ImportError:
        return None
    return HfFolder.get_token()


def clone_mas_specs(
    external_dir: Path,
    slugs: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    mas_root = external_dir / "mas"
    mas_root.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in selected_mas(slugs):
        destination = mas_root / spec.clone_dir
        if destination.joinpath(".git").exists():
            status = "exists"
        else:
            command = ["git", "clone", "--depth", "1"]
            if spec.branch:
                command.extend(["--branch", spec.branch])
            command.extend([spec.repo_url, str(destination)])
            subprocess.run(command, check=True)
            status = "cloned"
        results.append(
            {
                "slug": spec.slug,
                "name": spec.paper_name,
                "status": status,
                "path": str(destination),
                "head": git_head(destination),
                "repo_url": spec.repo_url,
            }
        )
    return results


def download_benchmark_specs(
    data_dir: Path,
    slugs: Iterable[str] | None = None,
    *,
    force: bool = False,
) -> list[dict[str, Any]]:
    benchmark_root = data_dir / "benchmarks"
    benchmark_root.mkdir(parents=True, exist_ok=True)
    results = []
    specs = tuple(selected_benchmarks(slugs))
    mad_records = load_mad_records(data_dir) if any(s.mad_name for s in specs) else []
    for spec in specs:
        destination = benchmark_root / spec.slug
        destination.mkdir(parents=True, exist_ok=True)
        downloaded_files = []
        skipped_files = []
        for remote_file in spec.files:
            output_path = destination / remote_file.path
            if output_path.exists() and not force:
                skipped_files.append(str(output_path))
            else:
                download_file(remote_file.url, output_path)
                downloaded_files.append(str(output_path))

        sample_path = write_benchmark_sample(destination, spec, mad_records)
        manifest = {
            "slug": spec.slug,
            "paper_name": spec.paper_name,
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "partial" if spec.skip_reason else "downloaded",
            "skip_reason": spec.skip_reason,
            "source_files": [asdict(item) for item in spec.files],
            "downloaded_files": downloaded_files,
            "already_present_files": skipped_files,
            "sample_file": str(sample_path) if sample_path else None,
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        )
        results.append(manifest)
    return results


def cache_gaia_attachments(
    data_dir: Path,
    *,
    split: str = "validation",
    offset: int = 0,
    limit: int | None = None,
    sample_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Cache attachments referenced by the selected gated GAIA samples."""

    metadata_path = data_dir / "benchmarks" / "gaia" / "2023" / split / "metadata.parquet"
    token = huggingface_token()
    if not metadata_path.exists():
        if not token:
            raise RuntimeError(
                "GAIA metadata is missing and prepare.gaia_attachments requires "
                "HF_TOKEN or HUGGINGFACE_HUB_TOKEN"
            )
        from huggingface_hub import hf_hub_download

        cached = Path(
            hf_hub_download(
                repo_id=GAIA_REPO_ID,
                repo_type="dataset",
                filename=f"2023/{split}/metadata.parquet",
                token=token,
            )
        )
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, metadata_path)

    items = load_gaia_items(data_dir, split=split)
    if sample_ids is not None:
        by_id = {str(item.get("sample_id")): item for item in items}
        missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
        if missing:
            raise ValueError(
                "GAIA metadata is missing requested sample IDs: "
                + ", ".join(missing[:10])
            )
        selected = [by_id[sample_id] for sample_id in sample_ids]
    else:
        selected = items[offset:] if limit is None else items[offset : offset + limit]
    results: list[dict[str, Any]] = []
    for item in selected:
        file_name = str(item.get("file_name") or "").strip()
        if not file_name:
            continue
        destination = metadata_path.parent / file_name
        if destination.exists():
            results.append(
                {
                    "sample_id": item.get("sample_id"),
                    "file_name": file_name,
                    "path": str(destination),
                    "status": "exists",
                }
            )
            continue
        if not token:
            raise RuntimeError(
                f"GAIA attachment {file_name} is missing; export HF_TOKEN or "
                "HUGGINGFACE_HUB_TOKEN after accepting dataset access"
            )
        from huggingface_hub import hf_hub_download

        cached = Path(
            hf_hub_download(
                repo_id=GAIA_REPO_ID,
                repo_type="dataset",
                filename=f"2023/{split}/{file_name}",
                token=token,
            )
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cached, destination)
        results.append(
            {
                "sample_id": item.get("sample_id"),
                "file_name": file_name,
                "path": str(destination),
                "status": "downloaded",
            }
        )
    return results


def write_benchmark_sample(
    destination: Path,
    spec: BenchmarkSpec,
    mad_records: list[dict[str, Any]],
) -> Path | None:
    if spec.mad_name is None:
        return None
    for record in mad_records:
        if record.get("benchmark_name") == spec.mad_name:
            sample = {
                "source": "mcemri/MAD",
                "benchmark": spec.paper_name,
                "mad_benchmark_name": spec.mad_name,
                "mas_name": record.get("mas_name"),
                "llm_name": record.get("llm_name"),
                "trace_id": record.get("trace_id"),
                "task": extract_task_text(record),
                "record": record,
            }
            output_path = destination / "sample.json"
            output_path.write_text(
                json.dumps(sample, ensure_ascii=False, indent=2) + "\n"
            )
            return output_path
    return None


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    command = [
        "curl",
        "-L",
        "--fail",
        "--show-error",
        "--progress-bar",
        "--connect-timeout",
        "30",
        "--max-time",
        "600",
        "-o",
        str(tmp_path),
        url,
    ]
    subprocess.run(command, check=True)
    shutil.move(tmp_path, destination)


def git_head(path: Path) -> str | None:
    if not path.joinpath(".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()
