"""Recover explicit No-intervention rows from already observed native controls."""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

from baselines.reexecute import execute_saved_row
from baselines.registry import build_method
from baselines.core.datasets import load_baseline_samples
from baselines.rescore import read_rows, recover_reference_control
from common.runtime_support import safe_name
from masfm.config import parse_api_config


def read_observed_control(sample, source_spec, control_path):
    """Validate an existing control without launching or rewriting anything."""
    options = source_spec["replay_backend"]["options"]
    expected_path = Path(options["control_output_root"]).resolve() / safe_name(sample.sample_id) / "control_execution.json"
    path = Path(control_path).resolve()
    if path != expected_path:
        raise ValueError("reference control path differs from the declared host")
    control_bytes = path.read_bytes()
    payload = json.loads(control_bytes)
    expected_identity = {"host_api": asdict(parse_api_config({"api": options["host_api"]}, required=True)),
        "native_options": options["native_options"], "benchmark": sample.benchmark, "mas": sample.mas,
        "task_sha256": hashlib.sha256(sample.task.encode()).hexdigest()}
    if payload["identity"] != expected_identity:
        raise ValueError("reference control has a different task or host identity")
    execution = payload["execution"]
    evaluation = execution["evaluation"]
    if execution["status"] != "completed" or not isinstance(evaluation.get("correct"), bool):
        raise ValueError("reference control has no completed benchmark verdict")
    if sample.benchmark == "swe-bench-lite":
        if evaluation.get("method") != "swebench_official_docker" or evaluation.get("harness_status") not in {"completed", "empty_patch"}:
            raise ValueError("reference control has no SWE evaluator outcome")
    elif sample.benchmark != "gaia" or evaluation.get("method") != "gaia_official_quasi_exact_match":
        raise ValueError("reference control has no GAIA evaluator outcome")
    result_path = Path(execution["artifact_refs"]["result_path"])
    evaluation_path = result_path.with_name("evaluation.json")
    native_bytes, evaluation_bytes = result_path.read_bytes(), evaluation_path.read_bytes()
    native = json.loads(native_bytes)
    if native["status"] != "completed" or json.loads(evaluation_bytes) != evaluation:
        raise ValueError("reference control differs from its native artifacts")
    return payload, {str(item): hashlib.sha256(raw).hexdigest() for item, raw in
                     ((path, control_bytes), (result_path, native_bytes), (evaluation_path, evaluation_bytes))}


def project_reference_row(sample, original, source_spec, control_path):
    if (source_spec.get("method") != "no_intervention" or source_spec.get("replay_backend", {}).get("type") != "native"
            or original.get("method") != "no_intervention" or original.get("setting") != "active"
            or original.get("intervention") is not None or original.get("model_calls")):
        raise ValueError("reference projection cannot replace a gate or repair decision")
    for key in ("sample_id", "task_id", "mas", "benchmark"):
        if original.get(key) != getattr(sample, key):
            raise ValueError("reference source differs from the evaluation sample")
    payload, artifacts = read_observed_control(sample, source_spec, control_path)
    path = Path(control_path).resolve()
    execution = payload["execution"]
    evaluation = execution["evaluation"]

    class ExistingControl:
        def prepare_sample(self, item):
            return replace(item, control_evaluation=evaluation, metadata={**item.metadata,
                "paired_control_path": str(path), "control_execution_count": execution["execution_count"]})

        def execute(self, *args, **kwargs):
            raise AssertionError("reference projection must not execute a native task")

    _, _, row = execute_saved_row(sample, copy.deepcopy(original), build_method("no_intervention"), None, ExistingControl())
    row["reexecution"]["source_results"] = source_spec["reexecute"]["source_results"]
    return row, artifacts


def reference_recovery_overlay(source_path, sample_id, input_path):
    source_path, input_path = Path(source_path).resolve(), Path(input_path).resolve()
    state = json.loads(source_path.with_name("run_state.json").read_text())
    spec = state["identity"]["spec"]
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != state["identity"]["input_sha256"]:
        raise ValueError("reference recovery input pool changed")
    source = next(row for row in read_rows(source_path) if row["sample_id"] == sample_id)
    if (spec.get("method") != "no_intervention" or source.get("method") != "no_intervention" or source.get("status") != "invalid"
            or source.get("model_calls") or source.get("intervention") is not None
            or source.get("diagnostics", {}).get("failure_category") != "infrastructure"):
        raise ValueError("standalone reference recovery requires an explicit failed reference")
    sample = next(item for item in load_baseline_samples(input_path, input_format=spec["input_format"],
        max_events=spec.get("max_events", 100000)) if item.sample_id == sample_id)
    if any(source.get(key) != getattr(sample, key) for key in ("task_id", "mas", "benchmark")):
        raise ValueError("reference source belongs to another evaluation sample")
    path = Path(spec["replay_backend"]["options"]["control_output_root"]) / safe_name(sample_id) / "control_execution.json"
    _, artifacts = read_observed_control(sample, spec, path)
    recovered = recover_reference_control(source, sample, spec, path)
    if recovered is source:
        raise ValueError("reference still has no recoverable control")
    recovered["reference_control_recovery"].update(standalone=True, source_results=str(source_path),
        previous_row_sha256=hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        input_path=str(input_path), control_artifacts=artifacts, new_model_calls=0, new_native_executions=0)
    return recovered


def verify_reference_recovery_overlay(row):
    metadata = row["reference_control_recovery"]
    expected = reference_recovery_overlay(metadata["source_results"], row["sample_id"], metadata["input_path"])
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"reference recovery differs from its source evidence: {key}")
