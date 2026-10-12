"""Exact-set reconciliation of persistence recovery and separately executed missing rows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from baselines.capability_correction import verify_invocation_correction
from baselines.rescore import read_rows
from baselines.runner import _result_from_row
from common.runtime_support import safe_name


def row_hash(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def merge_recovered_rows(base, additions, replacements, expected_ids):
    expected = list(expected_ids)
    if len(set(expected)) != len(expected):
        raise ValueError("reconciliation pool contains duplicate IDs")
    combined = {}
    for rows in [base, *additions]:
        for row in rows:
            sid = row["sample_id"]
            if sid in combined:
                raise ValueError("reconciliation sources overlap without a declared correction")
            if sid not in expected:
                raise ValueError("reconciliation includes a sample outside the fixed pool")
            combined[sid] = row
    for row in replacements:
        sid = row["sample_id"]
        metadata = row.get("invocation_correction", {})
        if sid not in combined or metadata.get("previous_row_sha256") != row_hash(combined[sid]):
            raise ValueError("replacement is not bound to the exact earlier execution")
        combined[sid] = row
    if set(combined) != set(expected):
        raise ValueError("reconciliation does not cover the complete evaluation pool")
    return [combined[sid] for sid in expected]


def read_verified_component(path, *, allow_projection=False):
    path = Path(path)
    state = json.loads(path.with_name("run_state.json").read_text())
    manifest = json.loads(path.with_name("manifest.json").read_text())
    projection = state.get("identity", {}).get("persistence_projection")
    if state["status"] != "completed" and not (allow_projection and projection and state["status"] == "recovering"):
        raise ValueError("reconciliation component is not complete")
    if "identity" in manifest and manifest["identity"] != state.get("identity"):
        raise ValueError("reconciliation component identity differs from its manifest")
    for name, digest in manifest["artifacts"].items():
        if hashlib.sha256((path.parent / name).read_bytes()).hexdigest() != digest:
            raise ValueError("reconciliation component artifact changed")
    if projection:
        audit = path.with_name("audit.json")
        if hashlib.sha256(audit.read_bytes()).hexdigest() != projection["audit_sha256"]:
            raise ValueError("persistence projection audit changed")
        for evidence in json.loads(audit.read_text()).get("artifacts", {}).values():
            for name, digest in evidence.items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                    raise ValueError("persistence projection evidence changed")
    rows = read_rows(path)
    recorded_count = state.get("completed_sample_count", state.get("sample_count"))
    if recorded_count != len(rows):
        raise ValueError("reconciliation component row count differs from its state")
    return rows, state


def verify_missing_execution_rows(rows, state, admission, base_identity):
    identity = state["identity"]
    spec = identity["spec"]
    original_spec = base_identity["spec"]
    if identity["input_sha256"] != base_identity["input_sha256"]:
        raise ValueError("missing-execution recovery changed the evaluation pool")
    for key in ("method", "model_identity", "input_format", "split", "max_events", "prompt_char_budget", "method_options"):
        if spec.get(key) != original_spec.get(key):
            raise ValueError("missing-execution recovery changed method identity")
    import copy
    expected = copy.deepcopy(original_spec["replay_backend"])
    expected["options"]["native_options"].update(admission["native_options_delta"])
    expected["options"]["control_output_root"] = spec["replay_backend"]["options"]["control_output_root"]
    if expected != spec["replay_backend"]:
        raise ValueError("missing-execution recovery changed unapproved host options")
    if not spec["reexecute"].get("project_environment") or identity.get("python_prefix") != spec["reexecute"]["project_environment"]:
        raise ValueError("missing-execution recovery changed evaluator environment")
    frozen = Path(spec["reexecute"]["source_results"])
    if hashlib.sha256(frozen.read_bytes()).hexdigest() != identity["source_sha256"]:
        raise ValueError("missing-execution decisions changed")
    decisions = {row["sample_id"]: row for row in read_rows(frozen)}
    targets = {item["sample_id"]: item for item in admission["targets"]}
    if set(decisions) != set(targets) or {row["sample_id"] for row in rows} != set(targets):
        raise ValueError("missing-execution recovery lost admitted samples")
    for row in rows:
        original = decisions[row["sample_id"]]
        if row_hash(original) != targets[row["sample_id"]]["decision_sha256"]:
            raise ValueError("missing-execution admission changed")
        for key in ("method", "setting", "mas", "benchmark", "task_id"):
            if row.get(key) != original.get(key):
                raise ValueError("missing-execution row identity changed")
        calls = row.get("model_calls", [])
        prior_calls = original.get("model_calls", [])
        policy = admission.get("decision_policy", "reuse-complete")
        if policy not in {"reuse-complete", "retry-invalid"}:
            raise ValueError("unknown missing-execution recovery policy")
        if row["reexecution"]["source_results"] != str(frozen):
            raise ValueError("recovery row names a different decision source")
        receipt_path = Path(spec["output_dir"]) / "completed_samples" / safe_name(row["sample_id"]) / "result.json"
        receipt = json.loads(receipt_path.read_text())
        if (receipt["row"] != row or receipt["row_sha256"] != row_hash(row)
                or receipt["identity_sha256"] != hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()):
            raise ValueError("recovery row differs from its completed-worker receipt")
        if policy == "reuse-complete":
            if original["status"] == "invalid" or original.get("intervention") is None:
                raise ValueError("saved-decision recovery has no complete source decision")
            if calls != prior_calls:
                raise ValueError("saved-decision recovery regenerated method calls")
            if row["status"] != "invalid" and any(row.get(key) != original.get(key) for key in ("intervention", "attribution")):
                raise ValueError("saved-decision recovery changed intervention")
        else:
            if original["status"] != "invalid" or calls[:len(prior_calls)] != prior_calls or row["reexecution"]["generation_reused"]:
                raise ValueError("generation recovery lost original failed-call history")
        result = _result_from_row(row)
        if row["execution_budget"] != result.execution_budget:
            raise ValueError("recovered execution budget differs from recorded attempts")


def reconcile_panel(base_path, additions, replacement_path, expected_ids):
    base, state = read_verified_component(base_path, allow_projection=True)
    groups = []
    for path, admission_path in additions:
        rows, component_state = read_verified_component(path)
        admission = json.loads(Path(admission_path).read_text())
        verify_missing_execution_rows(rows, component_state, admission, state["identity"])
        groups.append(rows)
    replacements, _ = read_verified_component(replacement_path)
    for row in replacements:
        verify_invocation_correction(row)
    return merge_recovered_rows(base, groups, replacements, expected_ids)


def verify_reconciled_source(path):
    path = Path(path)
    state = json.loads(path.with_name("run_state.json").read_text())
    manifest = json.loads(path.with_name("manifest.json").read_text())
    identity = state["identity"]
    if state["status"] != "completed" or manifest["identity"] != identity:
        raise ValueError("reconciled source is not complete")
    for source, digest in identity["source_artifacts"].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != digest:
            raise ValueError("reconciliation input artifact changed")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["artifacts"]["results.jsonl"]:
        raise ValueError("reconciled result changed")
    expected = reconcile_panel(identity["base_projection"], identity["additions"], identity["replacement"], identity["expected_ids"])
    if read_rows(path) != expected:
        raise ValueError("reconciled rows differ from verified source components")
