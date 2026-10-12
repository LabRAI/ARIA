"""Verified recovery of saved decisions whose shared-cache preflight never ran."""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess

from baselines.capability_correction import _control, _digest, _read, _verdict
from baselines.core.datasets import load_baseline_samples
from baselines.rescore import read_rows
from baselines.runner import _result_from_row
from common.evaluation import load_swe_bench_report
from masfm.config import parse_api_config


def validate_preflight_source(previous, decision, failed_control):
    error = previous.get("diagnostics", {}).get("error", "")
    if (previous.get("status") != "invalid" or previous.get("executions")
            or previous.get("diagnostics", {}).get("failure_category") != "infrastructure"
            or "workspace preflight failed" not in error or "Repository not found" not in error
            or previous.get("benchmark") != "swe-bench-lite" or previous.get("method") == "dover"):
        raise ValueError("recovery requires an unexecuted cache-preflight failure")
    if (failed_control.get("status") != "invalid" or failed_control.get("execution_count") != 0
            or failed_control.get("evaluation", {}).get("correct") is not None
            or not isinstance(failed_control.get("error"), str) or not failed_control["error"]
            or failed_control["error"] not in error):
        raise ValueError("failed control does not prove a zero-execution preflight")
    if (decision.get("status") == "invalid"
            or (decision.get("intervention") is None and decision.get("method") != "no_intervention")):
        raise ValueError("original method has no complete saved decision")
    for key in ("sample_id", "task_id", "method", "setting", "mas", "benchmark", "model_calls"):
        if previous.get(key) != decision.get(key):
            raise ValueError("original decision or model-call provenance differs")
    reenactment = previous.get("reexecution", {})
    if (reenactment.get("generation_reused") is not False
            or reenactment.get("previous_status") != decision["status"]
            or reenactment.get("previous_executions") != decision.get("executions", [])):
        raise ValueError("failed row does not preserve its original source execution")


def _artifacts(row, execution):
    _verdict(row, execution)
    if execution.execution_count != 1:
        raise ValueError("preflight recovery must execute each saved decision once")
    root = Path(execution.artifact_refs["run_root"])
    paths = [Path(execution.artifact_refs["result_path"]), root / "evaluation.json", root / "swebench_workspace_setup.json"]
    native, evaluation, setup = [json.loads(path.read_text()) for path in paths]
    workspace = Path(native["workspace_path"]).resolve()
    git_dir = workspace / ".git"
    request_path = root / "native_replay_request.json"
    requested_mode = setup["workspace_mode"]
    if request_path.exists():
        requested_mode = json.loads(request_path.read_text())["spec"].get("swebench_workspace_mode")
        paths.append(request_path)
    if (not git_dir.is_dir() or git_dir.is_symlink() or (git_dir / "commondir").exists()
            or (git_dir / "objects/info/alternates").exists() or requested_mode != "isolated-clone"):
        raise ValueError("private checkout is not independently configured")
    common = subprocess.run(["git", "-C", str(workspace), "rev-parse", "--git-common-dir"],
                            check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    if (workspace / common).resolve() != git_dir.resolve():
        raise ValueError("private checkout has an external Git common directory")
    paths.append(git_dir / "config")
    if (native["status"] != "completed" or evaluation != execution.evaluation
            or setup["workspace_mode"] not in {"isolated-clone", "shared-worktree"}
            or setup["base_commit"] != evaluation["base_commit"]
            or Path(setup["workspace_path"]).resolve() != Path(native["workspace_path"]).resolve()):
        raise ValueError("native result, private checkout, or evaluation differs")
    paths.extend(Path(evaluation[key]) for key in ("model_patch_path", "predictions_path", "dataset_path"))
    prediction = json.loads(Path(evaluation["predictions_path"]).read_text())
    dataset = json.loads(Path(evaluation["dataset_path"]).read_text())
    if (len(prediction) != 1 or len(dataset) != 1
            or prediction[0]["instance_id"] != evaluation["instance_id"]
            or dataset[0]["instance_id"] != evaluation["instance_id"]
            or dataset[0]["base_commit"] != setup["base_commit"]
            or prediction[0]["model_patch"] != Path(evaluation["model_patch_path"]).read_text()):
        raise ValueError("SWE submission does not match the recovered native output")
    if evaluation["harness_status"] == "completed":
        report_path, report = load_swe_bench_report(root, root / "swebench_harness", evaluation["harness_run_id"], evaluation["instance_id"])
        if (report is None or str(report_path) != evaluation["harness_report_path"]
                or report != evaluation["harness_report"] or evaluation["harness_returncode"] != 0
                or report["resolved"] is not execution.success):
            raise ValueError("recovered evaluation differs from its official report")
        paths.append(report_path)
    elif prediction[0]["model_patch"].strip() or execution.success is not False:
        raise ValueError("invalid empty-patch outcome")
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def preflight_recovery_row(previous_path, corrected_path, sample_id, *, admission_path):
    previous_path, corrected_path, admission_path = map(lambda path: Path(path).resolve(), (previous_path, corrected_path, admission_path))
    admission = json.loads(admission_path.read_text())
    target = next(item for item in admission["targets"] if item["sample_id"] == sample_id and Path(item["source_results"]).resolve() == previous_path)
    previous, before_state = _read(previous_path, sample_id)
    decision_path = Path(target["decision_source"])
    decision = next(item for item in read_rows(decision_path) if item["sample_id"] == sample_id)
    old_control = json.loads(Path(target["failed_control_snapshot"]).read_text())
    if (_digest(previous) != target["source_row_sha256"] or _digest(decision) != target["decision_sha256"]
            or _digest(old_control) != target["failed_control_sha256"]):
        raise ValueError("preflight admission sources changed")
    validate_preflight_source(previous, decision, old_control["execution"])
    if (Path(previous["reexecution"]["source_results"]).resolve() != decision_path.resolve()
            or hashlib.sha256(decision_path.read_bytes()).hexdigest() != before_state["identity"]["source_sha256"]):
        raise ValueError("original decision file differs from failed experiment identity")
    corrected, after_state = _read(corrected_path, sample_id)
    manifest = json.loads(corrected_path.with_name("manifest.json").read_text())
    if after_state["status"] != "completed" or manifest["identity"] != after_state["identity"]:
        raise ValueError("preflight recovery is not a complete verified experiment")
    for name, expected in manifest["artifacts"].items():
        if hashlib.sha256((corrected_path.parent / name).read_bytes()).hexdigest() != expected:
            raise ValueError("recovered experiment artifacts changed")
    old_spec, new_spec = before_state["identity"]["spec"], after_state["identity"]["spec"]
    if before_state["identity"]["input_sha256"] != after_state["identity"]["input_sha256"]:
        raise ValueError("preflight recovery changed evaluation pool")
    for key in ("method", "model_identity", "input_format", "split", "max_events", "prompt_char_budget", "method_options"):
        if old_spec.get(key) != new_spec.get(key):
            raise ValueError("preflight recovery changed method settings")
    expected = copy.deepcopy(old_spec["replay_backend"])
    expected["options"]["native_options"]["swebench_workspace_mode"] = "isolated-clone"
    expected["options"]["control_output_root"] = new_spec["replay_backend"]["options"]["control_output_root"]
    if expected != new_spec["replay_backend"]:
        raise ValueError("preflight recovery changed host settings beyond Git isolation")
    if after_state["identity"].get("python_prefix") != new_spec["reexecute"].get("project_environment"):
        raise ValueError("recovery did not run in its declared evaluator environment")
    frozen = Path(new_spec["reexecute"]["source_results"])
    if (hashlib.sha256(frozen.read_bytes()).hexdigest() != after_state["identity"]["source_sha256"]
            or next(item for item in read_rows(frozen) if item["sample_id"] == sample_id) != decision):
        raise ValueError("recovery changed the frozen method decision")
    for key in ("sample_id", "task_id", "method", "setting", "mas", "benchmark", "intervention", "attribution", "model_calls", "source_execution_count"):
        if corrected.get(key) != decision.get(key):
            raise ValueError("recovery changed the saved decision or model calls")
    if corrected["status"] == "invalid" or not corrected["reexecution"]["generation_reused"]:
        raise ValueError("preflight recovery did not execute the saved decision")
    input_path = Path(new_spec["input_path"])
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != after_state["identity"]["input_sha256"]:
        raise ValueError("recovery pool changed")
    sample = next(item for item in load_baseline_samples(input_path, input_format=new_spec["input_format"], max_events=new_spec.get("max_events", 100000)) if item.sample_id == sample_id)
    options = old_spec["replay_backend"]["options"]
    identity = {"host_api": asdict(parse_api_config({"api": options["host_api"]}, required=True)),
        "native_options": options["native_options"], "benchmark": sample.benchmark, "mas": sample.mas,
        "task_sha256": hashlib.sha256(sample.task.encode()).hexdigest()}
    if old_control["identity"] != identity:
        raise ValueError("failed control belongs to a different host or task")
    new_control, artifacts = _control(corrected, new_spec)
    expected_identity = copy.deepcopy(identity)
    expected_identity["native_options"]["swebench_workspace_mode"] = "isolated-clone"
    if new_control["identity"] != expected_identity:
        raise ValueError("corrected control changed task identity")
    result = _result_from_row(corrected)
    control_result = _result_from_row({**corrected, "executions": [new_control["execution"]], "selected_execution_label": new_control["execution"]["label"]})
    artifacts.update(_artifacts(corrected, control_result.selected_execution))
    if not result.executions and result.method == "no_intervention":
        result.executions = control_result.executions
        result.selected_execution_label = control_result.selected_execution_label
    if len(result.executions) != 1:
        raise ValueError("preflight recovery must contain exactly one corrected execution")
    artifacts.update(_artifacts(corrected, result.selected_execution))
    failed = _result_from_row({**corrected, "executions": [old_control["execution"]]}).executions[0]
    result.executions.insert(0, replace(failed, label="before-preflight-recovery:" + failed.label))
    return {**copy.deepcopy(corrected), **result.to_dict(), "model_calls": copy.deepcopy(corrected["model_calls"]), "preflight_recovery": {
        "previous_results": str(previous_path), "corrected_results": str(corrected_path), "admission": str(admission_path),
        "previous_row_sha256": _digest(previous), "corrected_row_sha256": _digest(corrected),
        "admission_sha256": hashlib.sha256(admission_path.read_bytes()).hexdigest(),
        "decision_source": str(decision_path), "decision_sha256": _digest(decision),
        "previous_diagnostics": copy.deepcopy(previous["diagnostics"]), "artifacts": artifacts,
        "extra_native_executions": result.selected_execution.execution_count,
        "policy": "restore the original saved decision after zero-execution cache preflight failure; one private-workspace execution, not best-of-N"}}


def verify_preflight_recovery(row):
    info = row["preflight_recovery"]
    expected = preflight_recovery_row(info["previous_results"], info["corrected_results"], row["sample_id"], admission_path=info["admission"])
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"preflight recovery differs from verified artifacts: {key}")
