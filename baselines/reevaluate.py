"""YAML-only, model-free recovery of unresolved official SWE evaluations."""

from __future__ import annotations

import copy
from dataclasses import replace
import fcntl
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

from baselines.core.datasets import load_baseline_samples
from baselines.core.evaluation import aggregate_results
from baselines.registry import build_method
from baselines.rescore import read_rows, sample_with_control
from baselines.runner import _atomic_write_jsonl, _resolve, _result_from_row, _select_evaluation_pool
from common.artifacts import atomic_write_json
from common.evaluation import evaluate_saved_swe_bench_submission, load_swe_bench_report


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reevaluate_row(row, sample, output, evaluation_options):
    if row["benchmark"] != "swe-bench-lite" or row["status"] == "invalid":
        raise ValueError("reevaluation requires a saved native SWE outcome")
    result = _result_from_row(copy.deepcopy(row))
    changed, artifacts = [], {}
    for index, execution in enumerate(result.executions):
        old = execution.evaluation
        if old.get("method") != "swebench_official_docker" or old.get("correct") is not None:
            continue
        paths = {key: Path(old[key]) for key in ("predictions_path", "dataset_path", "model_patch_path")}
        digests = {str(path): _sha(path) for path in paths.values()}
        predictions = json.loads(paths["predictions_path"].read_text())
        dataset = json.loads(paths["dataset_path"].read_text())
        if len(predictions) != 1 or len(dataset) != 1:
            raise ValueError("reevaluation requires one frozen prediction and dataset record")
        prediction, record = predictions[0], dataset[0]
        if (prediction["instance_id"] != old["instance_id"] or record["instance_id"] != old["instance_id"]
                or record["base_commit"] != old["base_commit"]
                or (old.get("model_name_or_path") is not None and prediction["model_name_or_path"] != old["model_name_or_path"])
                or prediction["model_patch"] != paths["model_patch_path"].read_bytes().decode("utf-8")):
            raise ValueError("frozen SWE submission identity or patch changed")
        evaluation = evaluate_saved_swe_bench_submission(record, prediction, output / execution.label,
                                                       evaluation_config=evaluation_options)
        if any(_sha(Path(path)) != digest for path, digest in digests.items()):
            raise ValueError("source submission changed during reevaluation")
        result.executions[index] = replace(execution, evaluation={**old, **evaluation}, artifact_refs={
            **dict(execution.artifact_refs), "previous_evaluation": copy.deepcopy(dict(old)),
            "reevaluation_artifacts": digests,
        })
        changed.append(execution.label)
        artifacts.update(digests)
    if not changed:
        raise ValueError("no unresolved official SWE evaluation is eligible")
    if result.method == "dover":
        by_label = {execution.label: execution for execution in result.executions}
        attempts = result.diagnostics["intervention_attempts"]
        for attempt in attempts:
            attempt["execution"] = by_label[attempt["execution"]["label"]].to_dict()
        selected = max(result.executions, key=lambda execution: (execution.success is True, float(execution.evaluation.get("score") or 0.0)))
        result.selected_execution_label = selected.label
        result.intervention = next(attempt["intervention"] for attempt in attempts if attempt["execution"]["label"] == selected.label)
        target = result.intervention["target_event_id"]
        result.attribution = {"event_id": target, "step_index": next(i for i, event in enumerate(sample.events) if event.event_id == target),
                              "validated": selected.success is True, "progress_score": selected.evaluation.get("score")}
    updated = {**copy.deepcopy(row), **result.to_dict(), "model_calls": copy.deepcopy(row["model_calls"])}
    updated["evaluation_recovery"] = {
        "previous_row_sha256": hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(),
        "rescored_labels": changed, "source_artifacts": artifacts, "extra_mas_executions": 0,
        "policy": "reevaluate exact frozen submissions after unresolved harness outcomes; no model calls or workspace edits",
    }
    return updated


def run_reevaluation_spec(config, spec):
    root = config.config_path.parent.resolve()
    options = spec["reevaluate"]
    environment = _resolve(root, options["project_environment"])
    if Path(sys.prefix).resolve() != environment:
        raise ValueError("reevaluation is not running in its declared project environment")
    output, source = _resolve(root, spec["output_dir"]), _resolve(root, options["source_results"])
    if output == source.parent:
        raise ValueError("reevaluation cannot overwrite its source")
    method = build_method(spec["method"])
    input_path = _resolve(root, spec["input_path"])
    samples, _ = _select_evaluation_pool(load_baseline_samples(input_path, input_format=spec["input_format"],
        max_events=spec.get("max_events", 100000)), setting=method.setting, split=spec.get("split"))
    samples = {sample.sample_id: sample for sample in samples}
    rows = {row["sample_id"]: row for row in read_rows(source)}
    ids = options["sample_ids"]
    if len(ids) != len(set(ids)) or not set(ids) <= rows.keys() or not set(ids) <= samples.keys():
        raise ValueError("reevaluation IDs do not match the fixed source pool")
    identity = {"spec": dict(spec), "input_sha256": _sha(input_path), "source_sha256": _sha(source),
                "python_prefix": str(environment), "packages": {name: version(name) for name in ("swebench", "httpx", "ghapi", "fastspec")},
                "code": {name: _sha(Path(__file__).resolve().parents[1] / name) for name in ("baselines/reevaluate.py", "src/common/evaluation.py")}}
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".writer.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = output / "run_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state["identity"] != identity:
                raise ValueError("reevaluation identity changed; use a new output directory")
            if state["status"] == "completed":
                manifest = json.loads((output / "manifest.json").read_text())
                if manifest["identity"] != identity or any(_sha(output / name) != digest for name, digest in manifest["artifacts"].items()):
                    raise ValueError("reevaluation artifact hash mismatch")
                return json.loads((output / "summary.json").read_text())
        saved_path = output / "results.jsonl"
        saved = {row["sample_id"]: row for row in read_rows(saved_path)} if saved_path.exists() else {}
        if not saved.keys() <= set(ids):
            raise ValueError("reevaluation output contains unexpected sample IDs")
        atomic_write_json(state_path, {"status": "running", "identity": identity, "sample_count": len(ids)})
        for sid in ids:
            row = rows[sid]
            sample = sample_with_control(samples[sid], row)
            if row["method"] != method.name or row["setting"] != method.setting or any(row[key] != getattr(sample, key) for key in ("mas", "benchmark", "task_id")):
                raise ValueError("reevaluation source identity differs from the pool")
            if sid not in saved:
                saved[sid] = reevaluate_row(row, sample, output / "evaluations" / sid, options.get("evaluation", {}))
                saved[sid]["evaluation_recovery"]["source_results"] = str(source)
                _atomic_write_jsonl(saved_path, [saved[key] for key in ids if key in saved])
                atomic_write_json(state_path, {"status": "running", "identity": identity, "sample_count": len(ids), "completed_sample_count": len(saved)})
        summary = aggregate_results((sample_with_control(samples[sid], saved[sid]), _result_from_row(saved[sid])) for sid in ids)
        summary.update(method=method.name, model_identity=spec.get("model_identity"), model_calls=0, extra_mas_executions=0)
        atomic_write_json(output / "summary.json", summary)
        atomic_write_json(output / "manifest.json", {"identity": identity, "artifacts": {
            name: _sha(output / name) for name in ("results.jsonl", "summary.json")}})
        atomic_write_json(state_path, {"status": "completed", "identity": identity, "sample_count": len(ids)})
        print(f"reevaluated {method.name}: {len(ids)} frozen submissions; no new MAS executions", flush=True)
        return summary


def verify_evaluation_recovery(row, sample=None):
    metadata = row["evaluation_recovery"]
    before = next((item for item in read_rows(Path(metadata["source_results"])) if item["sample_id"] == row["sample_id"]), None)
    if before is None or hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest() != metadata["previous_row_sha256"]:
        raise ValueError("reevaluation source row changed")
    for key in ("sample_id", "task_id", "method", "setting", "status", "mas", "benchmark", "model_calls",
                "source_execution_count", "control_evaluation", "control_success"):
        if row.get(key) != before.get(key):
            raise ValueError(f"reevaluation changed a non-evaluator field: {key}")
    if metadata.get("extra_mas_executions") != 0:
        raise ValueError("reevaluation cannot charge a new MAS execution")
    for path, digest in metadata["source_artifacts"].items():
        if _sha(Path(path)) != digest:
            raise ValueError("frozen reevaluation artifact changed")
    if len(row["executions"]) != len(before["executions"]):
        raise ValueError("reevaluation changed the execution count")
    changed = set(metadata["rescored_labels"])
    seen = set()
    for old, new in zip(before["executions"], row["executions"]):
        for key in ("label", "status", "execution_count", "execution_mode", "error"):
            if old.get(key) != new.get(key):
                raise ValueError("reevaluation changed an execution identity or cost")
        if new["label"] not in changed:
            if new != old:
                raise ValueError("reevaluation changed an unselected evaluator result")
            continue
        seen.add(new["label"])
        if old["evaluation"].get("correct") is not None or new["artifact_refs"].get("previous_evaluation") != old["evaluation"]:
            raise ValueError("reevaluation replaced a resolved result or lost its prior verdict")
        old_eval, evaluation = old["evaluation"], new["evaluation"]
        for key in ("predictions_path", "dataset_path"):
            if json.loads(Path(old_eval[key]).read_text()) != json.loads(Path(evaluation[key]).read_text()):
                raise ValueError("reevaluation changed the frozen submission")
        if Path(old_eval["model_patch_path"]).read_bytes() != Path(evaluation["model_patch_path"]).read_bytes():
            raise ValueError("reevaluation changed patch bytes")
        if evaluation["harness_status"] == "completed":
            root = Path(evaluation["model_patch_path"]).parent
            path, report = load_swe_bench_report(root, root / "swebench_harness", evaluation["harness_run_id"], evaluation["instance_id"])
            if (report is None or str(path) != evaluation.get("harness_report_path") or report != evaluation.get("harness_report")
                    or evaluation.get("harness_returncode") != 0 or evaluation["correct"] is not report["resolved"]
                    or evaluation.get("score") != float(report["resolved"])):
                raise ValueError("reevaluation disagrees with the official report")
        elif evaluation["harness_status"] == "empty_patch":
            if Path(evaluation["model_patch_path"]).read_text().strip() or evaluation["correct"] is not False:
                raise ValueError("invalid empty-patch reevaluation")
        elif evaluation.get("correct") is not None or evaluation.get("score") is not None:
            raise ValueError("unresolved reevaluation was converted to a benchmark verdict")
    if seen != changed or not changed:
        raise ValueError("reevaluation labels do not match recorded executions")
    old_result, new_result = _result_from_row(before), _result_from_row(row)
    if row["execution_budget"] != old_result.execution_budget or new_result.execution_budget != old_result.execution_budget:
        raise ValueError("reevaluation changed EB")
    if row["method"] != "dover":
        for key in ("intervention", "attribution", "selected_execution_label", "diagnostics"):
            if row.get(key) != before.get(key):
                raise ValueError("reevaluation changed the method decision")
    else:
        from baselines.capability_correction import _dover_trials

        if _dover_trials(before) != _dover_trials(row) or sample is None:
            raise ValueError("reevaluation changed DoVer trial decisions")
        winner = max(new_result.executions, key=lambda item: (item.success is True, float(item.evaluation.get("score") or 0.0)))
        proposal = next(item["intervention"] for item in row["diagnostics"]["intervention_attempts"] if item["execution"]["label"] == winner.label)
        target = proposal["target_event_id"]
        attribution = {"event_id": target, "step_index": next(i for i, event in enumerate(sample.events) if event.event_id == target),
                       "validated": winner.success is True, "progress_score": winner.evaluation.get("score")}
        if row["selected_execution_label"] != winner.label or row["intervention"] != proposal or row["attribution"] != attribution:
            raise ValueError("reevaluation selected an invalid DoVer winner")
