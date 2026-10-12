"""Frozen, answer-blind RQ2 deployment pools and matched native executions."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import yaml

from aria.corpus import cell_of_row, row_task_id
from common.artifacts import atomic_write_json
from common.catalog import find_benchmark
from common.mast_data import load_benchmark_items
from common.preparation import download_benchmark_specs
from common.runtime_support import safe_run_id

SEEN_MAS = frozenset({"openmanus", "metagpt", "magentic-one", "ag2"})
SEEN_BENCHMARKS = frozenset({"gsm-plus", "mmlu", "humaneval", "olympiadbench", "gaia", "swe-bench-lite"})
AXES = {"reference", "unseen_benchmark", "unseen_system", "unseen_host_model", "joint"}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def model_identity(value: str) -> str:
    return value.lower().removeprefix("model/").replace("_", "-").split("/")[-1]


def validate_spec(spec: dict) -> None:
    allowed = {"fitting_paths", "seen_host_models", "prepare_only", "seed", "project_environments"}
    if not isinstance(spec, dict) or set(spec) - allowed:
        raise ValueError("invalid generalization settings or unknown keys")
    for key in ("fitting_paths", "seen_host_models"):
        if not isinstance(spec.get(key), list) or not spec[key] or not all(isinstance(v, str) and v for v in spec[key]):
            raise ValueError(f"generalization.{key} requires a non-empty string list")
    if not isinstance(spec.get("prepare_only", False), bool):
        raise ValueError("generalization.prepare_only must be a boolean")
    if type(spec.get("seed", 42)) is not int:
        raise ValueError("generalization.seed must be an integer")
    environments = spec.get("project_environments", {})
    if not isinstance(environments, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in environments.items()):
        raise ValueError("generalization.project_environments must map MAS names to paths")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def task_texts(value: Any) -> set[str]:
    texts = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"task", "task_text", "question"} and isinstance(item, str):
                texts.add(" ".join(item.split()))
            else:
                texts.update(task_texts(item))
    elif isinstance(value, list):
        for item in value:
            texts.update(task_texts(item))
    return texts


def prepare_pools(config) -> dict:
    spec = config.generalization
    validate_spec(spec)
    rows, fitting = [], []
    for raw_path in spec["fitting_paths"]:
        path = Path(raw_path).resolve()
        data = read_jsonl(path)
        if not data:
            raise ValueError(f"empty fitting corpus: {path}")
        for row in data:
            cell = cell_of_row(row)
            if not cell.mas_id or not cell.benchmark_id or cell.host_model_id == "unknown":
                raise ValueError("fitting_paths must contain canonical aria-full rows with complete source-cell provenance")
        rows.extend(data)
        fitting.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(data)})
    seen_hosts = {model_identity(v) for v in spec["seen_host_models"]}
    for row in rows:
        cell = cell_of_row(row)
        if cell.mas_id not in SEEN_MAS or cell.benchmark_id not in SEEN_BENCHMARKS or model_identity(cell.host_model_id) not in seen_hosts:
            raise ValueError(f"fitting corpus includes an undeclared source cell: {cell}")
    fit_ids = {row_task_id(row) for row in rows}
    fit_text = set().union(*(task_texts(row) for row in rows))
    if not fit_text:
        raise ValueError("fitting corpus has no task text for overlap audit")
    if config.prepare.get("download_benchmarks"):
        download_benchmark_specs(config.data_dir, sorted({e["benchmark"] for e in config.experiments}))
    from inference.runner import effective_api_config, validate_expanded_experiments

    validate_expanded_experiments(config, config.experiments)

    pools, names = [], set()
    for experiment in config.experiments:
        name = experiment.get("name", "")
        if not name or name in names or safe_run_id(name) != name:
            raise ValueError("generalization requires unique plain experiment names")
        names.add(name)
        if "matrix" in experiment:
            raise ValueError("generalization uses explicit cells, not matrix expansion")
        axis = experiment.get("generalization_axis")
        if axis not in AXES:
            raise ValueError(f"unknown generalization axis: {axis}")
        api = effective_api_config(config.api, experiment)
        unseen = (experiment["benchmark"] not in SEEN_BENCHMARKS,
                  experiment["mas"] not in SEEN_MAS,
                  model_identity(api.model) not in seen_hosts)
        expected = {"reference": (False, False, False), "unseen_benchmark": (True, False, False),
                    "unseen_system": (False, True, False), "unseen_host_model": (False, False, True)}
        if (axis == "joint" and sum(unseen) < 2) or (axis != "joint" and unseen != expected[axis]):
            raise ValueError(f"cell {name} is confounded for axis {axis}: unseen benchmark/system/host={unseen}")
        if api.model == config.aria.model.model:
            raise ValueError("host model and ARIA checkpoint must be distinct")
        if experiment.get("sample_source") != "full" or experiment.get("sample_split") != "test":
            raise ValueError("generalization requires full test datasets")
        limit = experiment.get("sample_limit")
        if type(limit) is not int or limit < 1:
            raise ValueError("generalization requires a positive sample_limit")
        items = load_benchmark_items(config.data_dir, experiment["benchmark"], source="full", split="test",
                                     sample_ids=experiment.get("sample_ids"))
        # Exclude shared task IDs/text even across host/system axes. BBH loader
        # interleaves task families, so prefix sampling retains task diversity.
        eligible = [item for item in items if item["sample_id"] not in fit_ids
                    and not (task_texts(item) & fit_text)]
        offset = int(experiment.get("sample_offset", 0))
        selected = eligible[offset:offset + limit]
        if len({item["sample_id"] for item in selected}) != len(selected):
            raise ValueError(f"{name}: duplicate sample IDs are not a valid evaluation pool")
        if len(selected) != limit:
            raise ValueError(f"{name}: only {len(selected)} eligible tasks for requested {limit}")
        files = find_benchmark(experiment["benchmark"]).files
        cache_root = config.data_dir / "benchmarks" / experiment["benchmark"]
        pools.append({"name": name, "axis": axis, "mas": experiment["mas"],
                      "benchmark": experiment["benchmark"], "host_model": api.model,
                      "excluded_fitting_tasks": len(items) - len(eligible),
                      "sample_ids": [item["sample_id"] for item in selected],
                      "samples_sha256": digest(selected),
                      "source_files": [{"url": file.url, "sha256": hashlib.sha256((cache_root / file.path).read_bytes()).hexdigest()}
                                       for file in files]})
    return {"schema_version": "aria-generalization-v1", "fitting_corpora": fitting, "pools": pools}


def summarize_pairs(control: dict[str, dict], active: dict[str, dict], sample_ids: list[str], *, seed: int = 42) -> dict:
    from aria.runtime_metrics import inspection_failure

    pairs, invalid = [], []
    for sample_id in sample_ids:
        left, right = control.get(sample_id, {}), active.get(sample_id, {})
        evaluations = [row.get("evaluation") or {} for row in (left, right)]
        if any(row.get("status") != "completed" for row in (left, right)) or any(type(e.get("correct")) is not bool for e in evaluations):
            invalid.append({"sample_id": sample_id, "reason": "unresolved native/evaluator outcome"})
            continue
        audits = (right.get("aria") or {}).get("model_interventions", [])
        if not audits:
            invalid.append({"sample_id": sample_id, "reason": "missing live ARIA inspection evidence"})
            continue

        # Transport/model-loader failures remain invalid under fail-open;
        # malformed structured predictions are counted as harmful interventions.
        failures = [inspection_failure(a) for a in audits]
        if "infrastructure" in failures:
            invalid.append({"sample_id": sample_id, "reason": "ARIA endpoint/runtime failure"})
            continue
        structural = "malformed" in failures
        enacted = any(a.get("enacted") for a in audits)
        score_left, score_right = (float(e.get("score") if e.get("score") is not None else e["correct"]) for e in evaluations)
        pairs.append({"sample_id": sample_id, "control": int(evaluations[0]["correct"]),
                      "active": int(evaluations[1]["correct"]),
                      "harmful": int(structural or (enacted and score_right < score_left)),
                      "intervened": int(enacted),
                      "unintervened_discordance": int(not enacted and evaluations[0]["correct"] != evaluations[1]["correct"])})
    n = len(pairs)
    mean = lambda key: sum(p[key] for p in pairs) / n if n else None
    rng = random.Random(seed)
    deltas = sorted(sum((p["active"] - p["control"]) for p in rng.choices(pairs, k=n)) / n
                    for _ in range(2000)) if n else []
    return {"pool_size": len(sample_ids), "resolved_pairs": n, "invalid_count": len(invalid),
            "control_sr": mean("control"), "aria_sr": mean("active"), "hir": mean("harmful"),
            "intervention_rate": mean("intervened"),
            "unintervened_discordance": sum(p["unintervened_discordance"] for p in pairs),
            "sr_delta_ci95": [deltas[50], deltas[1949]] if deltas else None,
            "control_eb": sum(row.get("execution_attempts", 1) for key, row in control.items() if key in sample_ids) / len(sample_ids) if sample_ids else None,
            "aria_eb": sum(row.get("execution_attempts", 1) for key, row in active.items() if key in sample_ids) / len(sample_ids) if sample_ids else None,
            "planned_executions_per_task": 2,
            "pairs": pairs, "invalid_samples": invalid}


def run_generalization(config) -> list[dict]:
    from masfm.config import StrictSafeLoader, load_config

    root = (config.output_dir / "generalization").resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".writer.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = prepare_pools(config)
        raw = yaml.load(config.config_path.read_text(), Loader=StrictSafeLoader)
        frozen = json.loads(json.dumps(raw))
        frozen["generalization"].pop("prepare_only", None)
        manifest["configuration"] = frozen
        calibration = config.aria.model.gate_calibration_path
        if calibration:
            manifest["calibration_sha256"] = hashlib.sha256(Path(calibration).read_bytes()).hexdigest()
        path = root / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("frozen generalization manifest changed; use a new output_dir")
        atomic_write_json(path, manifest)
        reports = []
        for experiment, pool in zip(config.experiments, manifest["pools"]):
            arms = {}
            for arm in ("control", "aria"):
                child = {k: v for k, v in raw.items() if k != "generalization"}
                child["job"] = "inference" if arm == "control" else "aria"
                child["aria"] = {"enabled": False, "mode": "off"} if arm == "control" else raw["aria"]
                child["output_dir"] = str(root / pool["name"] / arm)
                child["output"] = {"existing_output_policy": "resume"}
                child["prepare"] = {"clone_mas": False, "download_benchmarks": False}
                entry = {k: v for k, v in experiment.items() if k != "generalization_axis"}
                entry.update(sample_ids=pool["sample_ids"], sample_offset=0, sample_limit=len(pool["sample_ids"]),
                             batch=True, native=True, resume=True, resume_require_resolved_evaluation=True)
                child["experiments"] = [entry]
                child_path = root / "configs" / f"{pool['name']}-{arm}.yml"
                child_path.parent.mkdir(parents=True, exist_ok=True)
                child_path.write_text(yaml.safe_dump(child, sort_keys=False))
                load_config(child_path)  # Validate before any endpoint is invoked.
                if not config.generalization.get("prepare_only", False):
                    env = os.environ.copy()
                    project_env = config.generalization.get("project_environments", {}).get(pool["mas"], f".venv-{pool['mas']}")
                    env["UV_PROJECT_ENVIRONMENT"] = str(Path(project_env).resolve())
                    if not (Path(project_env) / "bin" / "python").exists():
                        raise FileNotFoundError(f"missing MAS project environment: {project_env}; install its uv extra first")
                    subprocess.run(["uv", "run", "--no-sync", "python", "main.py", "-c", str(child_path)], env=env, check=True)
                batch = Path(child["output_dir"]) / child["job"] / pool["mas"] / pool["benchmark"] / pool["name"]
                records = {}
                results = batch / "results.jsonl"
                if results.exists():
                    for row in read_jsonl(results):
                        full_path = Path(row.get("output_dir") or batch / "samples" / row["sample_id"]) / "result.json"
                        full = json.loads(full_path.read_text()) if full_path.exists() else row
                        attempts = records.get(row["sample_id"], {}).get("execution_attempts", 0) + 1
                        records[row["sample_id"]] = {**full, "evaluation": row.get("evaluation"), "execution_attempts": attempts}
                arms[arm] = records
            report = {**pool, **summarize_pairs(arms["control"], arms["aria"], pool["sample_ids"],
                                               seed=config.generalization.get("seed", 42))}
            reports.append(report)
            atomic_write_json(root / "report.json", {"schema_version": "aria-generalization-report-v1",
                                                     "phase": "prepared" if config.generalization.get("prepare_only", False) else "evaluated",
                                                     "cells": reports})
        return reports
