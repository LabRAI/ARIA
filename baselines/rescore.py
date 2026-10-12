"""Model-free, YAML-dispatched corrections of a fixed baseline result pool."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Mapping

from baselines.core.datasets import load_baseline_samples
from baselines.core.evaluation import aggregate_grouped_results, aggregate_results, _outcome
from baselines.delivery_audit import native_delivery_evidence, native_trial_backend_faults
from baselines.registry import build_method
from baselines.replay.ledger import LedgerReplayBackend
from baselines.runner import (
    _atomic_write_jsonl, _resolve, _result_from_row, _result_row,
    _select_evaluation_pool, _sha256,
)
from common.artifacts import atomic_write_json, read_jsonl_lines
from common.runtime_support import safe_name


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in read_jsonl_lines(path) if line.strip()]
    ids = [row["sample_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate sample IDs in {path}")
    return rows


def sample_with_control(sample: Any, row: Mapping[str, Any]) -> Any:
    control = row.get("control_evaluation")
    if not isinstance(control, Mapping) or not control:
        value = row.get("control_success")
        control = {"correct": value} if isinstance(value, bool) else sample.control_evaluation
    return replace(sample, control_evaluation=dict(control))


def controller_adapter_rejections(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Identify recorded control actions rejected by the old text-patch adapter."""
    actions, benchmarks = Counter(), Counter()
    for row in rows:
        if row.get("setting") != "active":
            continue
        result = _result_from_row(row)
        execution = result.selected_execution
        action = (result.intervention or {}).get("action")
        if execution is None or action not in {"block", "replan", "rollback", "block_and_rollback"}:
            continue
        evaluation = execution.evaluation
        if (evaluation.get("method") == "static-validator"
                and "replacement_events" in str(evaluation.get("reason", ""))):
            actions[action] += 1
            benchmarks[str(row.get("benchmark"))] += 1
    return {"count": sum(actions.values()), "by_action": dict(sorted(actions.items())),
            "by_benchmark": dict(sorted(benchmarks.items()))}


def recovery_row(previous: Mapping[str, Any], recovered: Mapping[str, Any], *,
                 previous_path: Path, recovered_path: Path) -> dict[str, Any]:
    """Replace an infrastructure-invalid execution with one verified re-execution."""
    fields = ("sample_id", "task_id", "method", "setting", "mas", "benchmark", "intervention",
              "model_calls", "control_evaluation", "source_execution_count")
    if any(previous.get(key) != recovered.get(key) for key in fields):
        raise ValueError("execution recovery changed the sample, decision, model calls, or control")
    old = _result_from_row(previous)
    new = _result_from_row(recovered)
    before, after = old.selected_execution, new.selected_execution
    if (len(old.executions) != 1 or before is None or before.status not in {"invalid", "failed"}
            or before.success is not None or before.evaluation.get("failure_category") != "infrastructure"):
        raise ValueError("execution recovery requires one infrastructure-invalid execution")
    if (len(new.executions) != 1 or new.status == "invalid" or after is None
            or after.status != "completed" or after.success is None
            or after.evaluation.get("method") == "static-validator"):
        raise ValueError("execution recovery requires a completed benchmark verdict")
    prior_label = f"before-recovery:{before.label}"
    if prior_label == after.label:
        raise ValueError("execution recovery labels collide")
    new.executions = [replace(before, label=prior_label), after]
    new.selected_execution_label = after.label
    return {
        **copy.deepcopy(dict(recovered)), **new.to_dict(), "model_calls": copy.deepcopy(recovered.get("model_calls", [])),
        "execution_recovery": {
            "previous_results": str(previous_path), "recovered_results": str(recovered_path),
            "previous_row_sha256": hashlib.sha256(json.dumps(dict(previous), sort_keys=True).encode()).hexdigest(),
            "recovered_row_sha256": hashlib.sha256(json.dumps(dict(recovered), sort_keys=True).encode()).hexdigest(),
            "extra_native_executions": after.execution_count,
            "policy": "one same-decision re-execution after an infrastructure-invalid outcome; not best-of-N",
        },
    }


def merge_sources(paths: list[Path], samples: Mapping[str, Any], method: Any) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for index, path in enumerate(paths):
        state_path = path.parent / "run_state.json"
        if state_path.is_file() and json.loads(state_path.read_text()).get("status") != "completed":
            raise ValueError(f"correction source is not terminal: {path}")
        if state_path.is_file() and json.loads(state_path.read_text()).get("identity", {}).get("kind") == "persistence-reconciliation":
            from baselines.reconciliation import verify_reconciled_source
            verify_reconciled_source(path)
        rows = read_rows(path)
        ids = {row["sample_id"] for row in rows}
        if (index == 0 and ids != set(samples)) or not ids <= set(samples):
            raise ValueError(f"source does not match the fixed evaluation pool: {path}")
        for row in rows:
            sample = samples[row["sample_id"]]
            if row["method"] != method.name or row["setting"] != method.setting:
                raise ValueError(f"source method/setting mismatch: {path}")
            if any(row.get(key) != getattr(sample, key) for key in ("benchmark", "mas", "task_id")):
                raise ValueError(f"source sample identity mismatch: {row['sample_id']}")
            recovery = row.get("execution_recovery") or row.get("capability_correction") or row.get("evaluation_recovery") or row.get("preflight_recovery")
            recovery = recovery or row.get("context_budget_correction")
            recovery = recovery or row.get("invocation_correction")
            if not recovery and row.get("reference_control_recovery", {}).get("standalone"):
                recovery = row["reference_control_recovery"]
            if recovery and index:
                prior = {key: value for key, value in merged.get(sample.sample_id, {}).items()
                         if key != "correction_source"}
                digest = hashlib.sha256(json.dumps(prior, sort_keys=True).encode()).hexdigest()
                if digest != recovery.get("previous_row_sha256"):
                    raise ValueError("execution recovery does not match the preceding result row")
            merged[sample.sample_id] = {**row, "correction_source": str(path)}
    return [merged[key] for key in samples]


def rescore_row(row: dict[str, Any], sample: Any, replay: Any) -> dict[str, Any]:
    result = _result_from_row(row)
    if not result.executions or result.status == "invalid":
        return row
    attempts = result.diagnostics.get("intervention_attempts", [])
    interventions = {
        attempt["execution"]["label"]: attempt["intervention"]
        for attempt in attempts if attempt.get("execution") and attempt.get("intervention")
    }
    previous = []
    for index, execution in enumerate(result.executions):
        intervention = interventions.get(execution.label, result.intervention)
        if intervention is None or execution.evaluation.get("method") == "static-validator":
            continue
        if execution.execution_mode == "not_executed":
            continue
        corrected = replay.execute(sample, intervention, label=execution.label)
        # Rescoring an existing candidate does not consume a new MAS execution.
        result.executions[index] = replace(
            corrected, execution_count=execution.execution_count,
            artifact_refs={**dict(corrected.artifact_refs), "previous_execution": execution.to_dict()},
        )
        previous.append(execution.label)
    if not previous:
        return row
    if result.method == "dover":
        selected = max(result.executions, key=lambda item: (
            item.success is True, float(item.evaluation.get("score") or 0.0),
        ))
        result.selected_execution_label = selected.label
        result.intervention = interventions[selected.label]
        if result.attribution is not None:
            target = result.intervention["target_event_id"]
            result.attribution.update(
                event_id=target, step_index=next(i for i, e in enumerate(sample.events) if e.event_id == target),
                validated=selected.success is True, progress_score=selected.evaluation.get("score"),
            )
        for attempt in attempts:
            old = attempt.get("execution")
            if old:
                attempt["execution"] = next(e.to_dict() for e in result.executions if e.label == old["label"])
    return {
        **row, **_result_row(result.to_dict(), sample),
        "model_calls": row.get("model_calls", []),
        "correction": {"reason": "humaneval candidate materialization", "rescored_labels": previous},
    }


def rescore_native_gaia_row(row: dict[str, Any], sample: Any) -> dict[str, Any]:
    from common.evaluation import evaluate_sample

    result = _result_from_row(row)
    changed = []

    def score(evaluation, path):
        if not path or not evaluation.get("reference_answer"):
            return None
        path = Path(path)
        if not path.is_file():
            return None
        native = json.loads(path.read_text())
        return evaluate_sample(
            benchmark_slug="gaia", sample={"reference_answer": evaluation["reference_answer"]},
            result=native, run_root=path.parent,
        )

    for index, execution in enumerate(result.executions):
        path = execution.artifact_refs.get("result_path")
        evaluation = score(execution.evaluation, path)
        if evaluation is not None:
            result.executions[index] = replace(execution, evaluation=evaluation, artifact_refs={
                **dict(execution.artifact_refs), "previous_evaluation": dict(execution.evaluation),
                "scored_result_sha256": _sha256(Path(path)),
            })
            changed.append(execution.label)
    control = row.get("control_evaluation") or {}
    messages = control.get("messages_path")
    control_path = Path(messages).parent / "result.json" if messages else None
    corrected_control = score(control, control_path)
    if corrected_control is not None:
        row["control_evaluation"] = corrected_control
        row["control_success"] = corrected_control.get("correct")
        changed.append("unpatched-control")
    if not changed:
        return row
    if result.method == "dover" and result.executions:
        attempts = result.diagnostics.get("intervention_attempts", [])
        selected = max(result.executions, key=lambda item: (
            item.success is True, float(item.evaluation.get("score") or 0.0),
        ))
        for attempt in attempts:
            old = attempt.get("execution")
            if old:
                attempt["execution"] = next(e.to_dict() for e in result.executions if e.label == old["label"])
                if old["label"] == selected.label:
                    result.intervention = attempt["intervention"]
        result.selected_execution_label = selected.label
        if result.attribution is not None:
            target = result.intervention["target_event_id"]
            result.attribution.update(event_id=target,
                                      step_index=next(i for i, e in enumerate(sample.events) if e.event_id == target),
                                      validated=selected.success is True, progress_score=selected.evaluation.get("score"))
    return {**row, **result.to_dict(), "model_calls": row.get("model_calls", []),
            "correction": {"reason": "GAIA delivered answer precedence", "rescored_labels": changed,
                           "previous_control_evaluation": control}}


def recover_reference_control(row, sample, source_spec, control_path):
    """Recover only an explicit reference row from its resolved same-host control."""
    if (row.get("method") != "no_intervention" or row.get("status") != "invalid"
            or row.get("intervention") is not None
            or row.get("diagnostics", {}).get("failure_category") != "infrastructure"):
        return row
    if not control_path.is_file():
        return row
    payload = json.loads(control_path.read_text())
    execution = payload.get("execution") or {}
    evaluation = execution.get("evaluation") or {}
    if execution.get("status") != "completed" or not isinstance(evaluation.get("correct"), bool):
        return row
    from masfm.config import parse_api_config

    options = source_spec["replay_backend"]["options"]
    expected = {
        "host_api": asdict(parse_api_config({"api": options["host_api"]}, required=True)),
        "native_options": options.get("native_options"),
        "task_sha256": hashlib.sha256(sample.task.encode()).hexdigest(),
        "benchmark": sample.benchmark, "mas": sample.mas,
    }
    if payload.get("identity") != expected:
        raise ValueError("recovered reference control identity differs from its source experiment")
    if sample.benchmark == "swe-bench-lite" and evaluation.get("harness_status") not in {"completed", "empty_patch"}:
        return row
    if sample.benchmark == "gaia" and evaluation.get("method") != "gaia_official_quasi_exact_match":
        return row
    recovered = copy.deepcopy(row)
    previous = recovered["diagnostics"]
    recovered.update(
        status="no_intervention", executions=[], selected_execution_label=None,
        source_execution_count=float(execution.get("execution_count", 1.0)),
        control_evaluation=evaluation, control_success=evaluation["correct"],
        paired_control_path=str(control_path), diagnostics={"recovered_reference_control": True},
        reference_control_recovery={"control_path": str(control_path), "control_sha256": _sha256(control_path),
                                    "previous_diagnostics": previous},
    )
    return recovered


def run_rescore_spec(config: Any, spec: Mapping[str, Any]) -> dict[str, Any]:
    root = config.config_path.parent.resolve()
    options = spec["rescore"]
    paths = [_resolve(root, path) for path in options["sources"]]
    output = _resolve(root, spec["output_dir"])
    if any(output == path.parent for path in paths):
        raise ValueError("corrections must not overwrite source results")
    input_path = _resolve(root, spec["input_path"])
    method = build_method(spec["method"])
    code_root = Path(__file__).resolve().parents[1]
    reference_controls = {}
    control_sources = {}
    if method.name == "no_intervention":
        for path in paths:
            source_state_path = path.parent / "run_state.json"
            if not source_state_path.is_file():
                continue
            source_state = json.loads(source_state_path.read_text())
            source_spec = source_state.get("identity", {}).get("spec", {})
            backend = source_spec.get("replay_backend") or {}
            options_spec = backend.get("options") or {}
            if backend.get("type") == "native" and options_spec.get("paired_control"):
                control_root = _resolve(code_root, options_spec["control_output_root"])
                reference_controls[str(path)] = (source_spec, control_root)
                control_sources.update({str(p): _sha256(p) for p in control_root.glob("*/control_execution.json")})
    identity = {
        "spec": dict(spec), "input_sha256": _sha256(input_path),
        "sources": {str(path): _sha256(path) for path in paths},
        "control_sources": control_sources,
        "code": {path: _sha256(code_root / path) for path in (
            "baselines/rescore.py", "baselines/core/evaluation.py",
            "baselines/delivery_audit.py",
            "baselines/capability_correction.py",
            "baselines/preflight_recovery.py",
            "baselines/reevaluate.py",
            "baselines/replay/ledger.py",
            "src/aria/post_training/environment.py", "src/common/evaluation.py",
        )},
    }
    state_path = output / "run_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("identity") != identity:
            raise ValueError("correction identity changed; use a new output directory")
        if state.get("status") == "completed":
            manifest = json.loads((output / "manifest.json").read_text())
            for name, digest in manifest["artifacts"].items():
                if _sha256(output / name) != digest:
                    raise ValueError(f"correction artifact changed: {name}")
            return json.loads((output / "summary.json").read_text())
    loaded = load_baseline_samples(
        input_path, input_format=spec.get("input_format", "auto"),
        max_events=spec.get("max_events", 100000),
    )
    samples, excluded = _select_evaluation_pool(loaded, setting=method.setting, split=spec.get("split"))
    samples_by_id = {sample.sample_id: sample for sample in samples}
    if len(samples_by_id) != len(samples):
        raise ValueError("duplicate samples in evaluation pool")
    rows = merge_sources(paths, samples_by_id, method)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(state_path, {"status": "running", "identity": identity})
    replay = LedgerReplayBackend(
        environment={"kind": "evaluator", "data_dir": str(config.data_dir.resolve())},
        run_root=output / "evaluator",
    ) if "humaneval" in options.get("benchmarks", []) else None
    corrected_rows, pairs = [], []
    changes: Counter[str] = Counter()
    outcome_audit = defaultdict(Counter)
    for row in rows:
        if row.get("execution_recovery"):
            changes["recovered_executions"] += 1
            changes["recovery_extra_native_executions"] += row["execution_recovery"]["extra_native_executions"]
        if row.get("preflight_recovery"):
            changes["preflight_recoveries"] += 1
            changes["preflight_native_executions"] += row["preflight_recovery"]["extra_native_executions"]
        if row.get("capability_correction"):
            changes["capability_corrections"] += 1
            changes["capability_extra_native_executions"] += row["capability_correction"]["extra_native_executions"]
        if row.get("context_budget_correction"):
            changes["context_budget_corrections"] += 1
            changes["context_extra_native_executions"] += row["context_budget_correction"]["extra_native_executions"]
        if row.get("invocation_correction"):
            changes["invocation_corrections"] += 1
            changes["invocation_extra_native_executions"] += row["invocation_correction"]["extra_native_executions"]
        if row.get("evaluation_recovery"):
            changes["reevaluated_rows"] += 1
            changes["recovered_evaluations"] += len(row["evaluation_recovery"]["rescored_labels"])
        sample = sample_with_control(samples_by_id[row["sample_id"]], row)
        before = _outcome(sample, _result_from_row(row))[0]
        reference = reference_controls.get(row.get("correction_source"))
        if reference is not None:
            source_spec, control_root = reference
            recovered = recover_reference_control(
                row, sample, source_spec, control_root / safe_name(sample.sample_id) / "control_execution.json",
            )
            if recovered is not row:
                row = recovered
                sample = sample_with_control(samples_by_id[row["sample_id"]], row)
                changes["recovered_controls"] += 1
        rescored = False
        if sample.benchmark in options.get("benchmarks", []):
            candidate = copy.deepcopy(row)
            previous = candidate.pop("correction", None)
            row = (rescore_native_gaia_row(candidate, sample) if sample.benchmark == "gaia"
                   else rescore_row(candidate, sample, replay))
            rescored = "correction" in row
            if previous is not None:
                if rescored:
                    row.setdefault("correction_history", []).append(previous)
                else:
                    row["correction"] = previous
            sample = sample_with_control(samples_by_id[row["sample_id"]], row)
        result = _result_from_row(row)
        after, outcome = _outcome(sample, result)
        delivery = native_delivery_evidence(row)
        if delivery is not None:
            row["native_delivery_audit"] = delivery
        trial_faults = native_trial_backend_faults(row)
        if result.method == "dover":
            row["native_trial_backend_faults"] = trial_faults
        evaluation = result.selected_execution.evaluation if result.selected_execution else sample.control_evaluation
        category = "passed" if after is True else "benchmark_incorrect"
        if after is None:
            category = "method_invalid" if outcome == "method_invalid" else "evaluator_unresolved"
        elif after is False:
            if evaluation.get("method") == "controller-abstention":
                category = "controller_stop"
            elif evaluation.get("harness_status") == "empty_patch":
                category = delivery["category"] if delivery else "no_repository_patch"
            elif evaluation.get("harness_status") == "completed" and sample.benchmark == "swe-bench-lite":
                category = "official_tests_failed"
            elif sample.benchmark == "gaia":
                category = "no_final_answer" if evaluation.get("answer_extracted") is False else "gaia_answer_mismatch"
        if delivery and delivery.get("websurfer_endpoint_errors"):
            category = "passed_with_websurfer_endpoint_error" if after is True else "websurfer_model_endpoint_error"
        if trial_faults:
            category = "passed_with_native_trial_endpoint_error" if after is True else "native_trial_endpoint_error"
        outcome_audit[sample.benchmark][category] += 1
        if rescored:
            changes["rescored_rows"] += 1
            changes["new_passes"] += after is True and before is not True
            changes["lost_passes"] += before is True and after is not True
        row["execution_budget"] = result.execution_budget
        corrected_rows.append(row)
        pairs.append((sample, result))
    summary = aggregate_results(pairs)
    axis = "benchmark" if method.setting == "passive" else "mas"
    summary.update(
        method=method.name, setting=method.setting, model_identity=spec.get("model_identity"),
        group_axis=axis, groups=aggregate_grouped_results(pairs, axis),
        groups_by_benchmark=aggregate_grouped_results(pairs, "benchmark"),
        groups_by_mas=aggregate_grouped_results(pairs, "mas"),
        input_path=str(input_path), results_path=str(output / "results.jsonl"),
        input_sample_count=len(loaded), excluded_sample_count=excluded,
        outcome_audit_by_benchmark={key: dict(value) for key, value in outcome_audit.items()},
        controller_adapter_rejections=controller_adapter_rejections(corrected_rows),
        correction={"sources": identity["sources"], **dict(changes), "new_model_calls": 0,
                    "benchmarks": options.get("benchmarks", [])},
    )
    _atomic_write_jsonl(output / "results.jsonl", corrected_rows)
    atomic_write_json(output / "summary.json", summary)
    atomic_write_json(output / "manifest.json", {
        "schema_version": "aria-baseline-correction-v1", "identity": identity,
        "artifacts": {name: _sha256(output / name) for name in ("results.jsonl", "summary.json")},
    })
    atomic_write_json(state_path, {"status": "completed", "identity": identity, "sample_count": len(rows)})
    print(f"corrected {method.name} {spec.get('model_identity')}: {dict(changes)} "
          f"SR={summary['success_rate']:.4f} HIR={summary['harmful_intervention_rate']:.4f}", flush=True)
    return summary
