"""Reconstruct an uncommitted single native execution only on an exact request match."""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import re

from baselines.core.types import ExecutionRecord
from baselines.reference_projection import read_observed_control
from baselines.reexecute import execute_saved_row
from baselines.registry import build_method
from baselines.replay.native import materialize_in_workspace
from masfm.config import parse_api_config
from common.runtime_support import safe_name


def verify_failed_execution_projection(row, state, source_path, input_path):
    """Recheck an exit-120 reconstruction before admitting a new native run."""
    from baselines.core.datasets import load_baseline_samples
    from baselines.rescore import read_rows

    audit_path = Path(source_path).with_name("audit.json")
    audit = json.loads(audit_path.read_text())
    identity = state["identity"]
    projection = identity.get("persistence_projection", {})
    if (projection.get("kind") != "execution"
            or hashlib.sha256(audit_path.read_bytes()).hexdigest() != projection.get("audit_sha256")):
        raise ValueError("failed invocation has no verified persistence projection")
    spec = identity["spec"]
    original_path = Path(spec["reexecute"]["source_results"])
    if hashlib.sha256(original_path.read_bytes()).hexdigest() != identity["source_sha256"]:
        raise ValueError("failed-invocation decision source changed")
    input_path = Path(input_path)
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != identity["input_sha256"]:
        raise ValueError("failed-invocation input pool changed")
    sid = row["sample_id"]
    original = next(item for item in read_rows(original_path) if item["sample_id"] == sid)
    sample = next(item for item in load_baseline_samples(input_path, input_format=spec["input_format"],
        max_events=spec.get("max_events", 100000)) if item.sample_id == sid)
    label = original.get("selected_execution_label") or "saved-repair"
    root = Path(spec["output_dir"]) / "replay" / safe_name(sid) / safe_name(label)
    witness = audit.get("failure_witnesses", {}).get(sid)
    if not witness:
        raise ValueError("failed invocation has no parent-log witness")
    expected, artifacts = project_execution_row(sample, original, spec, root,
        method_api_model=None, failure_witness=witness)
    if expected != row or artifacts != audit["artifacts"][sid]:
        raise ValueError("failed invocation differs from its saved execution evidence")
    return {"category": "native_exit_120", "failure_witness": witness, "artifacts": artifacts}


def project_execution_row(sample, original, source_spec, run_root, *, method_api_model, failure_witness=None):
    if (original.get("method") in {"no_intervention", "dover"} or original.get("status") == "invalid"
            or original.get("intervention") is None or source_spec.get("method") != original.get("method")
            or source_spec.get("replay_backend", {}).get("type") != "native"):
        raise ValueError("projection requires one complete saved intervention, not a regenerated or multi-trial decision")
    for key in ("sample_id", "task_id", "mas", "benchmark"):
        if original.get(key) != getattr(sample, key):
            raise ValueError("saved decision belongs to another sample")
    root = Path(run_root).resolve()
    options = source_spec["replay_backend"]["options"]
    label = original.get("selected_execution_label") or "saved-repair"
    expected_root = Path(options.get("output_root") or Path(source_spec["output_dir"]) / "replay") / safe_name(sample.sample_id) / safe_name(label)
    if root != expected_root.resolve() or options.get("native_options", {}).get("delivery_retries", 0):
        raise ValueError("native run path or retry protocol differs")
    paths = [root / "native_replay_request.json"]
    if failure_witness is None:
        paths.extend((root / "result.json", root / "evaluation.json"))
    raw = [path.read_bytes() for path in paths]
    request = json.loads(raw[0])
    native = json.loads(raw[1]) if failure_witness is None else None
    evaluation = json.loads(raw[2]) if failure_witness is None else None
    digests = {str(path): hashlib.sha256(value).hexdigest() for path, value in zip(paths, raw)}
    api = parse_api_config({"api": options["host_api"]}, required=True)
    expected_spec = {**source_spec, **options.get("native_options", {}), "evaluation": options.get("evaluation", {}),
        "max_tokens": int(api.max_tokens), "context_window_tokens": int(options.get("context_window_tokens") or api.context_window_tokens or 32768)}
    if request["api_config"] != asdict(api) or request["spec"] != expected_spec:
        raise ValueError("native request changed host, budgets, or method settings")
    if (request["mas_slug"] != sample.mas or request["benchmark_slug"] != sample.benchmark
            or request["sample"]["sample_id"] != sample.sample_id or Path(request["run_root"]).resolve() != root):
        raise ValueError("native request identity differs")
    workspace = root / "workspace"
    if native is not None:
        if (native["status"] != "completed" or native["sample_id"] != sample.sample_id
                or native["mas"] != sample.mas or native["benchmark"] != sample.benchmark
                or any(native.get(key) != getattr(api, key) for key in ("model", "provider", "protocol", "base_url"))):
            raise ValueError("native result identity or completion differs")
        expected_evaluator = "swebench_official_docker" if sample.benchmark == "swe-bench-lite" else "gaia_official_quasi_exact_match"
        if evaluation.get("benchmark") != sample.benchmark or (
            isinstance(evaluation.get("correct"), bool) and evaluation.get("method") != expected_evaluator
        ):
            raise ValueError("native evaluation belongs to another benchmark or evaluator")
        if Path(native["workspace_path"]).resolve() != workspace:
            raise ValueError("native result points at another workspace")
    task_id = request["sample"].get("instance_id")
    if not task_id:
        match = re.search(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$", sample.sample_id)
        task_id = match.group() if match else None
    attachment = (request["sample"].get("file_path") or request["sample"].get("attachment_path")) if sample.benchmark == "gaia" else None
    known_files = {}
    if attachment:
        path = root / "gaia_attachment.json"
        raw = path.read_bytes()
        relocation = json.loads(raw)
        destination = Path(relocation["workspace_path"])
        if (Path(relocation["source_path"]).resolve() != Path(attachment).resolve()
                or destination.resolve() != workspace / "attachments" / Path(attachment).name):
            raise ValueError("attachment relocation differs from the executed workspace")
        known_files[Path(attachment).name] = destination
        digests[str(path)] = hashlib.sha256(raw).hexdigest()
    continuation = materialize_in_workspace(sample, original["intervention"], workspace,
        max_chars=options.get("continuation_char_budget"), task_id=task_id,
        source_workspace=(sample.metadata.get("source_result") or {}).get("workspace_path"),
        repo_name=str(request["sample"].get("repo") or "").rsplit("/", 1)[-1], known_files=known_files)
    if continuation["task"] != request["sample"]["task"]:
        raise ValueError("saved intervention does not reproduce the exact executed continuation")
    control_path = Path(options["control_output_root"]) / safe_name(sample.sample_id) / "control_execution.json"
    control, control_digests = read_observed_control(sample, source_spec, control_path)
    digests.update(control_digests)
    if failure_witness is None:
        execution = ExecutionRecord(label, "completed", evaluation=evaluation, execution_mode=continuation["execution_mode"],
            artifact_refs={"run_root": str(root), "result_path": str(root / "result.json"),
                "cut_event_id": continuation.get("cut_event_id"), "retained_event_count": len(continuation.get("retained_prefix", [])),
                "source_host_model": sample.metadata.get("host_model"), "replay_host_model": api.model,
                "baseline_method_model": method_api_model, "host_identity_mode": "explicit_configured_host"})
    else:
        witness_path = Path(failure_witness).resolve()
        witness_raw = witness_path.read_bytes()
        witness = json.loads(witness_raw)
        pattern = (r"^\d{4}-\d{2}-\d{2} .* - masfm\.baselines\.replay - WARNING - baseline replay failed sample="
            + re.escape(sample.sample_id) + r" label=" + re.escape(label) + r": native child exited 120:\s*$")
        line = witness.get("line", "")
        if (not re.fullmatch(pattern, line) or line not in Path(witness["source_log"]).read_text().splitlines()
                or any((root / name).exists() for name in ("result.json", "evaluation.json", "native_replay_result.json"))):
            raise ValueError("failed invocation lacks its exact parent-log witness or has a retained outcome")
        stderr_path = root / "native_replay_stderr.log"
        stderr = stderr_path.read_bytes()
        if stderr:
            raise ValueError("exit-120 reconstruction requires the observed empty stderr")
        digests[str(witness_path)] = hashlib.sha256(witness_raw).hexdigest()
        digests[str(stderr_path)] = hashlib.sha256(stderr).hexdigest()
        reason = "RuntimeError: native child exited 120: "
        execution = ExecutionRecord(label, "invalid", execution_mode="native_failed", execution_count=1.0,
            evaluation={"score": None, "correct": None, "method": "unresolved", "reason": reason, "failure_category": "infrastructure"},
            artifact_refs={"run_root": str(root)}, error=reason)

    class ObservedExecution:
        def prepare_sample(self, item):
            return replace(item, control_evaluation=control["execution"]["evaluation"], metadata={**item.metadata,
                "paired_control_path": str(control_path.resolve()), "control_execution_count": control["execution"]["execution_count"]})

        def execute(self, item, intervention, *, label):
            if intervention != original["intervention"] or label != execution.label:
                raise ValueError("reconstruction attempted a different intervention")
            return execution

    _, _, row = execute_saved_row(sample, copy.deepcopy(original), build_method(original["method"]), None, ObservedExecution())
    if row["status"] == "invalid" or not row["reexecution"]["generation_reused"]:
        raise ValueError("reconstruction did not preserve the saved decision")
    row["reexecution"]["source_results"] = source_spec["reexecute"]["source_results"]
    if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected for path, expected in digests.items()):
        raise ValueError("native artifacts changed during reconstruction")
    return row, digests
