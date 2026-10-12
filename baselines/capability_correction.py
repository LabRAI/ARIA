"""Evidence-bound overlays for the text-only native capability correction."""

from __future__ import annotations

import copy
from dataclasses import asdict, replace
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from baselines.core.datasets import load_baseline_samples
from baselines.delivery_audit import native_delivery_evidence, native_trial_backend_faults
from baselines.rescore import read_rows
from baselines.runner import _result_from_row
from common.runtime_support import safe_name
from masfm.config import parse_api_config


def _digest(row: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def _read(path: Path, sample_id: str):
    rows = read_rows(path)
    row = next((row for row in rows if row["sample_id"] == sample_id), None)
    if row is None:
        raise ValueError(f"correction sample is absent: {path}")
    state = json.loads(path.with_name("run_state.json").read_text())
    return row, state


def _control(row, spec):
    options = spec["replay_backend"]["options"]
    path = Path(row["paired_control_path"]).resolve()
    expected_path = Path(options["control_output_root"]).resolve() / safe_name(row["sample_id"]) / "control_execution.json"
    if path != expected_path:
        raise ValueError("control path does not belong to the declared sample and host")
    payload = json.loads(path.read_text())
    expected = {
        "host_api": asdict(parse_api_config({"api": options["host_api"]}, required=True)),
        "native_options": options.get("native_options"), "benchmark": row["benchmark"], "mas": row["mas"],
        "task_sha256": payload["identity"]["task_sha256"],
    }
    if payload["identity"] != expected or payload["execution"]["evaluation"] != row["control_evaluation"]:
        raise ValueError("control identity or verdict differs from its declared row")
    return payload, {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}


def _verdict(row, execution):
    if execution is None or execution.status != "completed" or execution.success is None:
        raise ValueError("capability correction requires a completed benchmark verdict")
    evaluation = execution.evaluation
    if evaluation.get("method") == "controller-abstention":
        if ((row.get("intervention") or {}).get("action") not in {"terminate", "escalate"}
                or execution.success is not False or execution.execution_count != 0):
            raise ValueError("invalid controller stop outcome")
        return
    if row["benchmark"] == "swe-bench-lite":
        if (evaluation.get("method") != "swebench_official_docker"
                or evaluation.get("harness_status") not in {"completed", "empty_patch"}):
            raise ValueError("capability correction requires the SWE evaluator verdict")
    elif evaluation.get("method") != "gaia_official_quasi_exact_match":
        raise ValueError("capability correction requires the GAIA evaluator verdict")


def _dover_trials(row):
    attempts = row.get("diagnostics", {}).get("intervention_attempts") or []
    executions = row.get("executions") or []
    if not attempts or len(attempts) != len(executions):
        raise ValueError("DoVer correction must retain every recorded trial")
    labels = [execution["label"] for execution in executions]
    if len(set(labels)) != len(labels) or [attempt["execution"]["label"] for attempt in attempts] != labels:
        raise ValueError("DoVer trial labels or order differ from executions")
    if any(attempt["execution"] != execution for attempt, execution in zip(attempts, executions)):
        raise ValueError("DoVer trial execution metadata differs")
    return [{key: value for key, value in attempt.items() if key != "execution"} for attempt in attempts], labels


def capability_correction_row(previous_path: Path, corrected_path: Path, sample_id: str,
                              *, witness_path: Path, _kind: str = "capability") -> dict[str, Any]:
    """Keep all charged attempts; accept either corrected verdict, never the higher one."""
    if _kind not in {"capability", "context_budget", "invocation"}:
        raise ValueError("unknown host-correction kind")
    budget_correction = _kind == "context_budget"
    invocation_correction = _kind == "invocation"
    metadata_key = "invocation_correction" if invocation_correction else "context_budget_correction" if budget_correction else "capability_correction"
    delta = ({"context_budget_mode": "server-tokenizer", "swebench_workspace_mode": "isolated-clone"}
             if budget_correction else {"model_vision": False, "code_execution_environment": "per-sample"})
    if invocation_correction:
        delta = {"code_execution_environment": "per-sample", "swebench_workspace_mode": "isolated-clone", "command_rejection_policy": "feedback"}
    previous_path, corrected_path, witness_path = (path.resolve() for path in (previous_path, corrected_path, witness_path))
    previous, before_state = _read(previous_path, sample_id)
    if previous.get("reference_control_recovery", {}).get("standalone"):
        from baselines.reference_projection import verify_reference_recovery_overlay
        verify_reference_recovery_overlay(previous)
    corrected, after_state = _read(corrected_path, sample_id)
    witness, witness_state = _read(witness_path, sample_id)
    if after_state["status"] != "completed":
        raise ValueError("correction source is not terminal")
    manifest = json.loads(corrected_path.with_name("manifest.json").read_text())
    if manifest["identity"] != after_state["identity"]:
        raise ValueError("correction manifest identity differs")
    for name, expected in manifest["artifacts"].items():
        if hashlib.sha256((corrected_path.parent / name).read_bytes()).hexdigest() != expected:
            raise ValueError("correction artifact hash mismatch")
    identities = [state["identity"] for state in (before_state, after_state, witness_state)]
    specs = [identity["spec"] for identity in identities]
    before_spec, after_spec, witness_spec = specs
    if any(spec.get("replay_backend", {}).get("type") != "native" for spec in specs):
        raise ValueError("capability correction requires native source experiments")
    for row, spec in zip((previous, corrected, witness), specs):
        if row["method"] != spec["method"]:
            raise ValueError("correction row method differs from its source specification")
    if (len({identity["input_sha256"] for identity in identities}) != 1
            or len({spec["model_identity"] for spec in specs}) != 1):
        raise ValueError("correction pool or backbone differs")
    input_path = Path(after_spec["input_path"])
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != after_state["identity"]["input_sha256"]:
        raise ValueError("correction input pool changed")
    for key in ("sample_id", "task_id", "mas", "benchmark"):
        if previous[key] != corrected[key] or previous[key] != witness[key]:
            raise ValueError("correction or witness sample differs")
    is_dover = previous["method"] == "dover"
    stable = ("method", "setting", "model_calls", "source_execution_count")
    if not is_dover:
        stable += ("intervention", "attribution")
    for key in stable:
        if previous.get(key) != corrected.get(key):
            raise ValueError("correction changed the method decision or model calls")
    if (previous.get(metadata_key)
            or previous["status"] == "invalid" or corrected["status"] == "invalid"
            or not corrected.get("reexecution", {}).get("generation_reused")):
        raise ValueError("correction requires an unrepeated, complete saved decision")
    if is_dover:
        if (_dover_trials(previous) != _dover_trials(corrected)
                or previous["diagnostics"].get("segmentation") != corrected["diagnostics"].get("segmentation")):
            raise ValueError("DoVer correction changed trial decisions, count, or order")
    for key in ("method", "model_identity", "input_format", "split", "max_events", "prompt_char_budget", "method_options"):
        if before_spec.get(key) != after_spec.get(key):
            raise ValueError(f"correction changed method input settings: {key}")
    old_options, new_options, witness_options = [spec["replay_backend"]["options"] for spec in specs]
    if budget_correction and (previous["mas"] != "metagpt" or is_dover
            or old_options["native_options"].get("context_budget_mode") == "server-tokenizer"):
        raise ValueError("context recovery requires an uncorrected single MetaGPT execution")
    if invocation_correction and (previous["mas"] != "ag2" or is_dover):
        raise ValueError("invocation recovery requires the recorded single AG2 failure")
    if _kind == "capability" and old_options["native_options"].get("model_vision") is False:
        raise ValueError("source already uses the corrected text-only capability declaration")
    if old_options["host_api"] != new_options["host_api"] or old_options["host_api"] != witness_options["host_api"]:
        raise ValueError("correction changed host endpoint or decoding settings")
    expected_options = copy.deepcopy(old_options)
    expected_options["native_options"].update(delta)
    expected_options["control_output_root"] = new_options["control_output_root"]
    if expected_options != new_options or old_options["native_options"] != witness_options["native_options"]:
        raise ValueError("correction changed unapproved host options")
    frozen_path = Path(after_spec["reexecute"]["source_results"])
    if (hashlib.sha256(frozen_path.read_bytes()).hexdigest() != after_state["identity"]["source_sha256"]
            or next((row for row in read_rows(frozen_path) if row["sample_id"] == sample_id), None) != previous):
        raise ValueError("correction does not reuse the exact frozen source row")

    old_control, old_refs = _control(previous, before_spec)
    new_control, new_refs = _control(corrected, after_spec)
    witness_control, witness_refs = _control(witness, witness_spec)
    expected_control = copy.deepcopy(old_control["identity"])
    expected_control["native_options"].update(delta)
    if new_control["identity"] != expected_control or witness_control["identity"] != old_control["identity"]:
        raise ValueError("correction changed the matched control task or host")
    if invocation_correction:
        from baselines.execution_projection import verify_failed_execution_projection
        evidence = verify_failed_execution_projection(previous, before_state, previous_path, input_path)
    elif budget_correction:
        failed = _result_from_row(previous).selected_execution
        error = failed.error or "" if failed else ""
        match = re.search(r"maximum context length is\s+(\d+)\s+tokens.*?request has\s+(\d+)\s+input tokens", error, re.I | re.S)
        if (failed is None or failed.status not in {"invalid", "failed"} or failed.success is not None
                or failed.evaluation.get("failure_category") != "infrastructure" or match is None):
            raise ValueError("context recovery has no recorded infrastructure overflow")
        evidence = {"category": "context_overflow", "error": error,
            "context_window_tokens": int(match.group(1)), "prompt_tokens": int(match.group(2))}
    else:
        evidence = native_delivery_evidence(witness)
        if not evidence or not evidence.get("websurfer_endpoint_errors"):
            evidence = next(iter(native_trial_backend_faults(witness).values()), None)
        if not evidence or not evidence.get("websurfer_endpoint_errors"):
            evidence = native_delivery_evidence({**witness, "method": "no_intervention", "setting": "active",
                "status": "no_intervention", "intervention": None, "executions": [], "selected_execution_label": None})
        if not evidence or not evidence.get("websurfer_endpoint_errors"):
            raise ValueError("capability correction has no observed endpoint-fault witness")
    corrected_control_row = {**corrected, "method": "no_intervention", "setting": "active",
        "status": "no_intervention", "intervention": None, "executions": [], "selected_execution_label": None}
    for candidate in (corrected, corrected_control_row):
        audit = native_delivery_evidence(candidate)
        if audit and audit.get("websurfer_endpoint_errors"):
            raise ValueError("corrected execution or shared control still has endpoint faults")
    if native_trial_backend_faults(corrected):
        raise ValueError("corrected DoVer trials still have endpoint faults")

    old, new = _result_from_row(previous), _result_from_row(corrected)
    for result, row, control in ((old, previous, old_control), (new, corrected, new_control)):
        if not result.executions and result.setting == "active" and result.intervention is None:
            result.executions = _result_from_row({**row, "executions": [control["execution"]],
                "selected_execution_label": control["execution"]["label"]}).executions
            result.selected_execution_label = control["execution"]["label"]
        if not result.executions or (not is_dover and len(result.executions) != 1):
            raise ValueError("capability correction requires one original and one corrected execution")
        if any(not math.isfinite(value) or value < 0 for value in
               [result.source_execution_count, *(execution.execution_count for execution in result.executions)]):
            raise ValueError("invalid execution accounting")
    selected = new.selected_execution
    _verdict(corrected, selected)
    if is_dover:
        for execution in new.executions:
            _verdict(corrected, execution)
        winner = max(new.executions, key=lambda execution: (execution.success is True, float(execution.evaluation.get("score") or 0.0)))
        attempts = corrected["diagnostics"]["intervention_attempts"]
        proposal = next(attempt["intervention"] for attempt in attempts if attempt["execution"]["label"] == winner.label)
        target = proposal["target_event_id"]
        samples = load_baseline_samples(input_path, input_format=after_spec["input_format"],
                                       max_events=after_spec.get("max_events", 100000))
        sample = next(sample for sample in samples if sample.sample_id == sample_id)
        index = next((i for i, event in enumerate(sample.events) if event.event_id == target), None)
        expected_attribution = {"event_id": target, "step_index": index, "validated": winner.success is True,
                                "progress_score": winner.evaluation.get("score")}
        if (index is None or new.selected_execution_label != winner.label or new.intervention != proposal
                or new.attribution != expected_attribution):
            raise ValueError("DoVer selection differs from the complete corrected trial set")
    corrected_control_result = _result_from_row({**corrected_control_row, "executions": [new_control["execution"]],
        "selected_execution_label": new_control["execution"]["label"]})
    _verdict(corrected_control_row, corrected_control_result.selected_execution)
    budget_artifacts = {}
    if budget_correction or invocation_correction:
        from baselines.preflight_recovery import _artifacts
        environment = after_spec["reexecute"].get("project_environment")
        if not environment or after_state["identity"].get("python_prefix") != environment:
            raise ValueError("context recovery lacks its declared evaluator environment")
        for execution in (selected, corrected_control_result.selected_execution):
            budget_artifacts.update(_artifacts(corrected, execution))
            if invocation_correction:
                continue
            native_path = Path(execution.artifact_refs["run_root"]) / "metagpt_output.json"
            native = json.loads(native_path.read_text())
            if len(native["model_calls"]) != 4:
                raise ValueError("context recovery did not complete all native roles")
            for call in native["model_calls"]:
                audit = call.get("input_budget")
                limit = new_options["host_api"]["context_window_tokens"] - new_options["host_api"]["max_tokens"] - 256
                if (not audit or audit["prompt_tokens"] > limit
                        or call["metadata"].get("context_budget") is not None
                        or call["metadata"]["usage"]["prompt_tokens"] > limit):
                    raise ValueError("corrected native request did not preserve its declared token budget")
            budget_artifacts[str(native_path)] = hashlib.sha256(native_path.read_bytes()).hexdigest()
    prefix = "before-invocation-correction" if invocation_correction else "before-context-correction" if budget_correction else "before-capability-correction"
    prior = [replace(execution, label=f"{prefix}:{execution.label}") for execution in old.executions]
    extra = sum(execution.execution_count for execution in new.executions)
    new.executions = [*prior, *new.executions]
    new.selected_execution_label = selected.label
    trial_metadata = {"replayed_trial_count": len(prior), "superseded_execution_labels": [execution.label for execution in prior]} if is_dover else {}
    return {**copy.deepcopy(corrected), **new.to_dict(), "model_calls": copy.deepcopy(corrected["model_calls"]),
        metadata_key: {
            "previous_results": str(previous_path), "corrected_results": str(corrected_path),
            "previous_row_sha256": _digest(previous), "corrected_row_sha256": _digest(corrected),
            "witness_results": str(witness_path), "witness_row_sha256": _digest(witness), "witness": evidence,
            "control_artifacts": {**old_refs, **new_refs, **witness_refs},
            "previous_control_evaluation": copy.deepcopy(previous["control_evaluation"]),
            "extra_native_executions": extra,
            **trial_metadata,
            "native_options_delta": delta,
            **({"budget_artifacts": budget_artifacts} if budget_correction else {}),
            **({"execution_artifacts": budget_artifacts} if invocation_correction else {}),
            "policy": "one fault-admitted paired-cohort correction, not best-of-N; retain both executions and the corrected verdict",
        }}


def context_budget_correction_row(previous_path, corrected_path, sample_id):
    return capability_correction_row(Path(previous_path), Path(corrected_path), sample_id,
        witness_path=Path(previous_path), _kind="context_budget")


def invocation_correction_row(previous_path, corrected_path, sample_id):
    return capability_correction_row(Path(previous_path), Path(corrected_path), sample_id,
        witness_path=Path(previous_path), _kind="invocation")


def verify_invocation_correction(row):
    metadata = row["invocation_correction"]
    expected = invocation_correction_row(metadata["previous_results"], metadata["corrected_results"], row["sample_id"])
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"invocation correction differs from its verified sources: {key}")


def verify_context_budget_correction(row):
    metadata = row["context_budget_correction"]
    expected = context_budget_correction_row(metadata["previous_results"], metadata["corrected_results"], row["sample_id"])
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"context recovery differs from its verified sources: {key}")


def verify_capability_correction(row: dict[str, Any]) -> None:
    metadata = row["capability_correction"]
    expected = capability_correction_row(Path(metadata["previous_results"]), Path(metadata["corrected_results"]),
        row["sample_id"], witness_path=Path(metadata["witness_results"]))
    for key in ("sample_id", "task_id", "method", "setting", "status", "intervention", "attribution",
                "model_calls", "diagnostics", "executions", "selected_execution_label", "source_execution_count",
                "execution_budget", "control_evaluation", "control_success", "capability_correction"):
        if row.get(key) != expected.get(key):
            raise ValueError(f"capability correction differs from its verified sources: {key}")
