"""Freeze the paper's Qwen scaling and stage-ablation matrix from YAML."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

from aria.corpus import SourceCell, cell_of_row
from aria.split_audit import rows_from_path, task_keys
from common.artifacts import atomic_write_json, atomic_write_jsonl, atomic_write_text
from masfm.config import load_config
from scripts.aria.build_stage3_environment import _training_ready_environment_rows

AXES = ("in_domain", "unseen_system", "unseen_host_model")
STAGES = ("sft", "dpo", "transfer_diagnosis", "grpo_repair", "grpo_gate")
GPU_UUID = "0"


def cell(row):
    raw = row.get("cell") or row.get("source_refs")
    if raw:
        return SourceCell(str(raw.get("mas_id", "unknown")),
                          str(raw.get("benchmark_id", "unknown")),
                          str(raw.get("host_model_id", "unknown")))
    return cell_of_row(row)


def withheld(row, axis, spec):
    source = cell(row)
    if "unknown" in (source.mas_id, source.benchmark_id, source.host_model_id):
        raise ValueError("unresolved source-cell provenance")
    if axis == "unseen_system":
        return source.mas_id == spec["unseen_system"]
    if axis == "unseen_host_model":
        return source.host_model_id in spec["unseen_host_models"]
    return False


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_yaml(path, value):
    atomic_write_text(path, yaml.safe_dump(value, sort_keys=False))


def prepare_views(spec, root):
    source = Path(spec["source"])
    validation = Path(spec["validation"])
    test = Path(spec["test"])
    heldout_keys = set()
    for path in (validation, test):
        for row in rows_from_path(path):
            heldout_keys.update(task_keys(row))
    raw = {name: list(rows_from_path(source / f"{name}.jsonl")) for name in ("sft", "dpo")}
    val_rows = [r for r in rows_from_path(validation / "sft.jsonl") if r["task"] == "deployment_gate"]
    pool = list(rows_from_path(Path(spec["test_pool"])))
    if any(r.get("split") != "test" for r in pool):
        raise ValueError("formal evaluation requires the frozen test split")
    views = {}
    for axis in AXES:
        output = root / "corpora" / axis
        kept = {}
        for name, rows in raw.items():
            kept[name] = [r for r in rows if not withheld(r, axis, spec)
                          and not task_keys(r) & heldout_keys]
            if not kept[name]:
                raise ValueError(f"empty {axis}/{name}")
            atomic_write_jsonl(output / "train" / f"{name}.jsonl", kept[name])
        patch_keys = {(r["incident_id"], json.dumps(r["input"], sort_keys=True),
                       json.dumps(r["target"], sort_keys=True))
                      for r in kept["sft"] if r["task"] == "structured_repair"}
        if any((r["incident_id"], json.dumps(r["input"], sort_keys=True),
                json.dumps(r["chosen"], sort_keys=True)) not in patch_keys for r in kept["dpo"]):
            raise ValueError("preference/SFT lineage mismatch")
        gates = [r for r in val_rows if not withheld(r, axis, spec)]
        atomic_write_jsonl(output / "validation_gate.jsonl", gates)
        # Reconstruct all source cells; the older env_all omitted several GPT/Magentic cells.
        environment = _training_ready_environment_rows(output / "train", split_seed=spec["data_seed"])
        for row in environment:
            row["split"] = "train"
            if withheld(row, axis, spec) or task_keys(row) & heldout_keys:
                raise ValueError("environment leaks a held-out source cell or task")
        atomic_write_jsonl(output / "env_train.jsonl", environment)
        evaluation = [r for r in pool if axis == "in_domain" or withheld(r, axis, spec)]
        if not evaluation:
            raise ValueError(f"empty held-out evaluation: {axis}")
        atomic_write_jsonl(output / "test_pool.jsonl", evaluation)
        manifest = {
            "schema_version": "aria-rq23-view-v1", "dataset_schema_version": "aria-training-ready-v3",
            "axis": axis, "unseen_system": spec["unseen_system"],
            "unseen_host_models": spec["unseen_host_models"],
            "source": str(source), "source_sha256": {n: sha(source / f"{n}.jsonl") for n in raw},
            "rows": {n: len(v) for n, v in kept.items()},
            "excluded_rows": {n: len(raw[n]) - len(v) for n, v in kept.items()},
            "tasks": dict(Counter(r["task"] for r in kept["sft"])),
            "cells": dict(Counter(cell(r).key() for r in kept["sft"])),
            "validation_gate_rows": len(gates), "environment_incidents": len(environment),
            "test_incidents": len(evaluation), "split_audit": "passed",
            "environment_provenance": "Reconstructed observable incidents; native evaluator resolves original tasks. Ledger labels are not online rewards.",
            "eligibility": "Inherited processed V3 export; raw V6 execution evidence is not reverified.",
            "sha256": {str(p.relative_to(output)): sha(p) for p in output.rglob("*.jsonl")},
        }
        atomic_write_json(output / "manifest.json", manifest)
        atomic_write_json(output / "train" / "manifest.json", manifest)
        views[axis] = {"root": str(output), **manifest}
    return views


def training_jobs(spec, root, views):
    jobs, checkpoints = [], {}
    repo = Path(spec["repo_root"])
    base = yaml.safe_load(Path(spec["training_template"]).read_text())
    for size in spec["sizes"]:
        for axis in AXES:
            for seed in spec["seeds"]:
                key = f"qwen25-{size}/{axis}/seed-{seed}"
                previous, dependency = f"Qwen/Qwen2.5-{size.upper()}-Instruct", None
                for stage in STAGES:
                    name = f"{key}/{stage}"
                    config, checkpoint = training_config(base, spec, root, views[axis], name, stage,
                                                          previous, seed, size)
                    path = root / "configs" / f"{name}.yml"
                    write_yaml(path, config)
                    load_config(path)
                    jobs.append({"id": name, "kind": "training", "size": size, "axis": axis,
                                 "seed": seed, "stage": stage, "config": str(path),
                                 "depends_on": [dependency] if dependency else [],
                                 "memory_gib": spec["memory_gib"][size], "checkpoint": checkpoint,
                                 "estimated_hours": spec["estimated_stage_hours"][size][stage],
                                 "environment": str(repo / ".venv-aria-training")})
                    checkpoints[name] = checkpoint
                    previous, dependency = checkpoint, name
                if size == "7b" and axis != "unseen_host_model":
                    previous, dependency = checkpoints[f"{key}/sft"], f"{key}/sft"
                    for stage in ("grpo_repair", "grpo_gate"):
                        name = f"{key}/stage1_3/{stage}"
                        config, checkpoint = training_config(base, spec, root, views[axis], name,
                                                              stage, previous, seed, size)
                        path = root / "configs" / f"{name}.yml"
                        write_yaml(path, config)
                        load_config(path)
                        jobs.append({"id": name, "kind": "training", "size": size, "axis": axis,
                                     "seed": seed, "stage": stage, "config": str(path),
                                     "depends_on": [dependency], "memory_gib": spec["memory_gib"][size],
                                     "checkpoint": checkpoint,
                                     "estimated_hours": spec["estimated_stage_hours"][size][stage],
                                     "environment": str(repo / ".venv-aria-training")})
                        checkpoints[name] = checkpoint
                        previous, dependency = checkpoint, name
    return jobs, checkpoints


def training_config(base, spec, root, view, name, stage, previous, seed, size):
    config = copy.deepcopy(base)
    post = config["aria"]["post_training"]
    output = root / "training" / name
    post.update(base_model=previous, source_path=str(Path(view["root"]) / "train"),
                output_dir=str(output), stages=[stage], random_seed=seed,
                gate_validation_path=str(Path(view["root"]) / "validation_gate.jsonl"),
                environment_path=str(Path(view["root"]) / "env_train.jsonl"),
                environment={"kind": "native_by_cell", "config_manifest": str(root / "host_manifest.json"), "seed": seed},
                evaluation_holdout_paths=[spec["validation"], spec["test"]], resume=True)
    options = {**spec["trainer"], **spec["stage_options"][stage], "seed": seed,
               "cuda-memory-fraction": spec["memory_gib"][size] / spec["gpu_capacity_gib"]}
    chain = name.split("/stage1_3/")[0] if "/stage1_3/" in name else name.rsplit("/", 1)[0]
    if stage == "grpo_gate":
        options["grpo-gate-anchor-path"] = str(Path(view["root"]) / "train" / "sft.jsonl")
    command = ["env", f"CUDA_VISIBLE_DEVICES={GPU_UUID}",
               f"UV_PROJECT_ENVIRONMENT={Path(spec['repo_root']) / '.venv-aria-training'}",
               "UV_NO_SYNC=1", "bash", "scripts/aria/launch_transformers.sh"]
    for argument in ("stage", "base-model", "datasets-json", "output-dir", "reference-model", "environment-spec"):
        placeholder = {"datasets-json": "datasets"}.get(argument, argument.replace("-", "_"))
        value = "{" + placeholder + "}"
        if argument == "reference-model" and stage != "sft":
            anchor = "sft" if stage in {"dpo", "transfer_diagnosis"} or "/stage1_3/" in name else "transfer_diagnosis"
            value = str(root / "training" / chain / anchor / "checkpoints" / anchor / "checkpoint")
        command.extend([f"--{argument}", value])
    for option, value in options.items():
        command.extend([f"--{option}", str(value).lower() if isinstance(value, bool) else str(value)])
    post["backend"] = {"type": "command", "command": command}
    return config, str(output / "checkpoints" / stage / "checkpoint")


def build(spec):
    root = Path(spec["output"])
    if root.exists():
        raise FileExistsError(f"refusing to overwrite frozen study: {root}")
    if spec["gpu_uuid"] != GPU_UUID or spec["gpu_index"] != 1:
        raise ValueError("this study is restricted to physical GPU 1")
    if spec["seeds"] != [42, 43, 44] or spec["sizes"] != ["1.5b", "3b", "7b", "14b"]:
        raise ValueError("formal matrix requires the four Qwen sizes and three registered seeds")
    if spec["inspection_seconds"] < 1800:
        raise ValueError("progress inspection interval must be at least 30 minutes")
    views = prepare_views(spec, root)
    jobs, checkpoints = training_jobs(spec, root, views)
    manifest = {"schema_version": "aria-rq23-study-v1", "spec": spec, "views": views,
                "jobs": jobs, "checkpoints": checkpoints,
                "training_jobs": len(jobs), "unique_full_chains": 36,
                "estimated_serial_training_hours": sum(j["estimated_hours"] for j in jobs),
                "selection_rule": "Pre-registered terminal checkpoint at fixed supervised update budgets; no test-based selection.",
                "compute_qualification": "Stage 3 accepted update counts must be audited across scales; flat/unresolved groups cannot be counted as learning.",
                "status": "prepared_not_evaluated"}
    atomic_write_json(root / "manifest.json", manifest)
    write_yaml(root / "study.yml", spec)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(yaml.safe_load(args.config.read_text()))
    print(json.dumps({k: manifest[k] for k in ("training_jobs", "estimated_serial_training_hours", "status")}, indent=2))


if __name__ == "__main__":
    main()
