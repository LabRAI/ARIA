"""Re-enact saved baseline decisions on a corrected native host, from YAML."""

from __future__ import annotations

import copy
import fcntl
import json
import hashlib
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from baselines.core.datasets import load_baseline_samples
from baselines.core.evaluation import aggregate_results, aggregate_grouped_results
from baselines.core.models import ModelAdapter, StructuredGenerationError
from baselines.registry import build_method
from baselines.rescore import read_rows
from baselines.runner import (
    _atomic_write_jsonl, _resolve, _result_from_row, _result_row, _sha256,
    _select_evaluation_pool, _load_replay_backend,
    _run_bounded_samples,
)
from common.artifacts import atomic_write_json
from common.model_api import ModelApiClient, ModelApiError
from common.runtime_support import safe_name

LOGGER = logging.getLogger("masfm.baselines.reexecute")


def execute_saved_row(sample, source, method, model, replay):
    result = _result_from_row(source)
    original_calls = copy.deepcopy(source.get("model_calls", []))
    preparing = True
    try:
        sample = replay.prepare_sample(sample)
        preparing = False
        if result.status == "invalid":
            # No complete method decision was produced. Retry its ordinary
            # protocol once, not best-of-N and not until a task passes.
            result = method.run(sample, model, replay)
            calls = original_calls + [call.to_dict() for call in result.model_calls]
            reused = False
        else:
            calls = original_calls
            reused = True
            if result.method == "no_intervention" or (result.setting == "active" and result.intervention is None):
                result.executions = []
                result.selected_execution_label = None
            elif result.method == "dover":
                attempts = copy.deepcopy(result.diagnostics.get("intervention_attempts", []))
                if not attempts:
                    raise ValueError("saved DoVer result is missing its attempted interventions")
                result.executions = []
                for index, attempt in enumerate(attempts):
                    label = (attempt.get("execution") or {}).get("label", f"dover-trial-{index:02d}")
                    execution = replay.execute(sample, attempt["intervention"], label=label)
                    result.executions.append(execution)
                    attempt["execution"] = execution.to_dict()
                winner = max(range(len(attempts)), key=lambda i: (
                    result.executions[i].success is True,
                    float(result.executions[i].evaluation.get("score") or 0.0),
                ))
                selected = result.executions[winner]
                result.selected_execution_label = selected.label
                result.intervention = attempts[winner]["intervention"]
                target = result.intervention["target_event_id"]
                result.attribution = {
                    "event_id": target,
                    "step_index": next(i for i, event in enumerate(sample.events) if event.event_id == target),
                    "validated": selected.success is True,
                    "progress_score": selected.evaluation.get("score"),
                }
                result.diagnostics["intervention_attempts"] = attempts
            elif result.intervention is not None:
                label = result.selected_execution_label or "saved-repair"
                result.executions = [replay.execute(sample, result.intervention, label=label)]
                result.selected_execution_label = label
            else:
                raise ValueError("passive source has no executable decision")
    except Exception as exc:
        result = method.result(sample, status="invalid")
        result.diagnostics["error"] = f"{type(exc).__name__}: {exc}"
        result.diagnostics["failure_category"] = (
            "infrastructure" if preparing or isinstance(exc, (ModelApiError, TimeoutError, ConnectionError)) else "method"
        )
        calls = original_calls
        if isinstance(exc, StructuredGenerationError):
            calls.append(exc.record.to_dict())
        reused = False
    if result.setting == "active" and result.intervention is None:
        result.source_execution_count = float(sample.metadata.get("control_execution_count", 1.0))
    row = _result_row(result.to_dict(), sample)
    row["model_calls"] = calls
    row["reexecution"] = {
        "generation_reused": reused,
        "previous_executions": source.get("executions", []),
        "previous_status": source.get("status"),
        "source_control_evaluation": source.get("control_evaluation", {}),
    }
    return sample, result, row


def run_reexecution_spec(config: Any, spec: dict[str, Any]) -> dict[str, Any]:
    root = config.config_path.parent.resolve()
    output = _resolve(root, spec["output_dir"])
    source_path = _resolve(root, spec["reexecute"]["source_results"])
    if output == source_path.parent:
        raise ValueError("native re-execution must not overwrite its source")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".writer.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_locked(config, spec, root, output, source_path)


def _run_locked(config, spec, root, output, source_path):
    environment = spec["reexecute"].get("project_environment")
    if environment and Path(sys.prefix).resolve() != _resolve(root, environment):
        raise ValueError("re-execution is not running in its declared evaluator environment")
    method = build_method(spec["method"], **dict(spec.get("method_options", {})))
    input_path = _resolve(root, spec["input_path"])
    source = {row["sample_id"]: row for row in read_rows(source_path)}
    loaded = load_baseline_samples(input_path, input_format=spec.get("input_format", "auto"),
                                   max_events=spec.get("max_events", 100000))
    samples, _ = _select_evaluation_pool(loaded, setting=method.setting, split=spec.get("split"))
    if spec.get("benchmark_filter"):
        samples = [sample for sample in samples if sample.benchmark in spec["benchmark_filter"]]
    selected_ids = spec["reexecute"].get("sample_ids")
    if selected_ids:
        samples = [sample for sample in samples if sample.sample_id in selected_ids]
        if {sample.sample_id for sample in samples} != set(selected_ids):
            raise ValueError("diagnostic sample IDs are not in the declared evaluation pool")
    elif set(source) != {sample.sample_id for sample in samples}:
        raise ValueError("source rows do not cover the complete declared evaluation pool")
    if not samples:
        raise ValueError("empty native re-execution pool")
    for sample in samples:
        row = source[sample.sample_id]
        if row["method"] != method.name or row["setting"] != method.setting:
            raise ValueError("source method identity differs from re-execution method")
        if any(row[key] != getattr(sample, key) for key in ("mas", "benchmark", "task_id")):
            raise ValueError("source sample identity differs from the evaluation pool")
    code_root = Path(__file__).resolve().parents[1]
    code_paths = ["baselines/reexecute.py", "baselines/replay/native.py", "baselines/replay/materialize.py",
                  "src/common/evaluation.py", "src/inference/native/shared.py",
                  "src/inference/native/code_environment.py", "src/inference/native/token_budget.py"]
    code_paths += [str(path.relative_to(code_root)) for path in (code_root / "src/inference/native").glob("*/*.py")]
    identity = {"spec": dict(spec), "source_sha256": _sha256(source_path), "input_sha256": _sha256(input_path),
                "code": {path: _sha256(code_root / path) for path in sorted(set(code_paths))}}
    if environment:
        from importlib.metadata import version
        identity.update(python_prefix=str(Path(sys.prefix).resolve()),
                        evaluator_packages={name: version(name) for name in ("httpx", "swebench")})
    state_path = output / "run_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state and state.get("identity") != identity:
        raise ValueError("native re-execution identity changed; use a new output directory")
    results_path = output / "results.jsonl"
    rows = {row["sample_id"]: row for row in read_rows(results_path)} if results_path.exists() else {}
    if not set(rows) <= {sample.sample_id for sample in samples}:
        raise ValueError("resumed results include samples outside the declared pool")
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    receipts = output / "completed_samples"
    names = [safe_name(sample.sample_id) for sample in samples]
    if len(names) != len(set(names)):
        raise ValueError("sample IDs collide in completion-receipt paths")
    recovered = False
    for sample in samples:
        path = receipts / safe_name(sample.sample_id) / "result.json"
        if sample.sample_id in rows or not path.exists():
            continue
        receipt = json.loads(path.read_text())
        saved = receipt["row"]
        if receipt.get("schema_version") != "baseline-sample-completion-v1" or receipt.get("identity_sha256") != identity_hash:
            raise ValueError("completion receipt belongs to a different experiment")
        if hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest() != receipt.get("row_sha256"):
            raise ValueError("completion receipt row checksum differs")
        if (saved.get("sample_id") != sample.sample_id or saved.get("method") != method.name
                or saved.get("setting") != method.setting
                or saved.get("reexecution", {}).get("source_results") != str(source_path)
                or any(saved.get(key) != getattr(sample, key) for key in ("task_id", "benchmark", "mas"))):
            raise ValueError("completion receipt has a different sample or method identity")
        rows[sample.sample_id] = saved
        recovered = True
    if recovered:
        _atomic_write_jsonl(results_path, [rows[s.sample_id] for s in samples if s.sample_id in rows])
    pairs = {}
    for sample in samples:
        if sample.sample_id in rows:
            saved = rows[sample.sample_id]
            pairs[sample.sample_id] = (replace(sample, control_evaluation=saved["control_evaluation"]), _result_from_row(saved))
    pending = [sample for sample in samples if sample.sample_id not in rows]
    client = ModelApiClient(config.api) if config.api is not None else None
    model = ModelAdapter(client, prompt_char_budget=spec.get("prompt_char_budget", 48000)) if client else None
    atomic_write_json(state_path, {"status": "running", "identity": identity,
                                   "completed_sample_count": len(rows), "sample_count": len(samples)})

    def run_one(sample):
        replay = _load_replay_backend(spec["replay_backend"], spec={**spec, "output_dir": str(output)}, config=config)
        instance = build_method(spec["method"], **dict(spec.get("method_options", {})))
        completed = execute_saved_row(sample, copy.deepcopy(source[sample.sample_id]), instance, model, replay)
        completed[2]["reexecution"]["source_results"] = str(source_path)
        atomic_write_json(receipts / safe_name(sample.sample_id) / "result.json", {
            "schema_version": "baseline-sample-completion-v1", "identity_sha256": identity_hash,
            "row_sha256": hashlib.sha256(json.dumps(completed[2], sort_keys=True).encode()).hexdigest(), "row": completed[2],
        })
        return completed

    committed_count = len(rows)

    def commit(sample, result, row):
        nonlocal committed_count
        rows[sample.sample_id] = row
        pairs[sample.sample_id] = (sample, result)
        _atomic_write_jsonl(results_path, [rows[s.sample_id] for s in samples if s.sample_id in rows])
        committed_count = len(rows)
        atomic_write_json(state_path, {"status": "running", "identity": identity,
                                       "completed_sample_count": len(rows), "sample_count": len(samples)})
        selected = result.selected_execution
        success = None if result.status == "invalid" else (selected.success if selected else sample.control_success)
        print(f"native {spec.get('model_identity')} {method.name} {len(rows)}/{len(samples)} "
              f"{sample.mas}/{sample.benchmark} success={success} "
              f"status={result.status}", flush=True)

    def failed(exc):
        LOGGER.error("native re-execution stopped after a worker or persistence failure: %s", spec["name"],
                     exc_info=(type(exc), exc, exc.__traceback__))
        atomic_write_json(state_path, {"status": "failed", "identity": identity,
            "completed_sample_count": committed_count, "sample_count": len(samples),
            "error": f"{type(exc).__name__}: {exc}", "stage": "sample_execution_or_commit",
            "recovery": "verified completion receipts may be folded into results.jsonl without re-execution"})

    _run_bounded_samples(pending, run_one, commit, spec.get("sample_workers", 2), on_error=failed)
    all_pairs = [pairs[sample.sample_id] for sample in samples]
    summary = aggregate_results(all_pairs)
    axis = "benchmark" if method.setting == "passive" else "mas"
    summary.update(method=method.name, setting=method.setting, model_identity=spec.get("model_identity"),
                   group_axis=axis, groups=aggregate_grouped_results(all_pairs, axis),
                   groups_by_benchmark=aggregate_grouped_results(all_pairs, "benchmark"),
                   groups_by_mas=aggregate_grouped_results(all_pairs, "mas"),
                   input_path=str(input_path), results_path=str(results_path),
                   reexecution={"source_results": str(source_path), "stored_decisions": sum(r["reexecution"]["generation_reused"] for r in rows.values())})
    atomic_write_json(output / "summary.json", summary)
    atomic_write_json(output / "manifest.json", {"identity": identity, "artifacts": {
        name: _sha256(output / name) for name in ("results.jsonl", "summary.json")}})
    atomic_write_json(state_path, {"status": "completed", "identity": identity, "sample_count": len(rows)})
    return summary
