"""YAML-dispatched baseline dataset runner with resumable JSONL artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from contextlib import suppress
from pathlib import Path
from typing import Any

from baselines.core.datasets import load_baseline_samples, resolve_baseline_input_path
from baselines.core.evaluation import aggregate_grouped_results, aggregate_results
from baselines.core.models import ModelAdapter, StructuredGenerationError
from baselines.core.types import MethodResult
from baselines.registry import build_method
from common.artifacts import atomic_write_json
from common.config import ExperimentConfig
from common.model_api import ModelApiError


def _run_bounded_samples(samples, run_one, commit, workers, on_error=None):
    """Do not queue the rest of a corpus past a failed persistence boundary."""
    remaining = iter(samples)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()

        def submit(count):
            for _ in range(count):
                try:
                    sample = next(remaining)
                except StopIteration:
                    break
                pending.add(pool.submit(run_one, sample))

        submit(workers)
        try:
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    commit(*future.result())
                submit(len(done))
        except BaseException as exc:
            for future in pending:
                future.cancel()
            if on_error is not None:
                with suppress(Exception):
                    on_error(exc)
            raise


def run_baselines(config: ExperimentConfig, client: Any | None) -> list[dict[str, Any]]:
    workers = int(getattr(config, "baseline_workers", 1))
    if workers < 1:
        raise ValueError("baseline_workers must be positive")
    if workers == 1 or len(config.baselines) <= 1:
        return [run_baseline_spec(config, spec, client) for spec in config.baselines]

    # Each method owns its output directory and resume ledger, so method-level
    # parallelism is safe.  The per-sample worker pool remains inside
    # run_baseline_spec and is controlled by that method's YAML entry.
    summaries: list[dict[str, Any] | None] = [None] * len(config.baselines)
    with ThreadPoolExecutor(max_workers=min(workers, len(config.baselines))) as pool:
        futures = {
            pool.submit(run_baseline_spec, config, spec, client): index
            for index, spec in enumerate(config.baselines)
        }
        for future in as_completed(futures):
            summaries[futures[future]] = future.result()
    return [summary for summary in summaries if summary is not None]


def run_baseline_spec(
    config: ExperimentConfig, spec: Mapping[str, Any], client: Any | None
) -> dict[str, Any]:
    if "reevaluate" in spec:
        from baselines.reevaluate import run_reevaluation_spec

        return run_reevaluation_spec(config, spec)
    if "reexecute" in spec:
        from baselines.reexecute import run_reexecution_spec

        return run_reexecution_spec(config, spec)
    if "rescore" in spec:
        from baselines.rescore import run_rescore_spec

        return run_rescore_spec(config, spec)
    root = (
        config.config_path.parent.resolve()
        if config.config_path is not None
        else Path.cwd()
    )
    input_path = resolve_baseline_input_path(
        _resolve(root, str(spec["input_path"])),
        input_format=str(spec.get("input_format", "auto")),
    )
    output_dir = _resolve(root, str(spec["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    resume = bool(spec.get("resume", True))
    method = build_method(str(spec["method"]), **dict(spec.get("method_options", {})))
    local_identity = None
    if spec.get("local_model") is not None:
        if client is not None:
            raise ValueError("baseline local_model cannot be combined with an endpoint client")
        from baselines.core.local_model import LocalTransformersClient

        local = dict(spec["local_model"])
        model_path = str(local.pop("model_path"))
        if model_path.startswith((".", "/")):
            model_path = str(_resolve(root, model_path))
        client = LocalTransformersClient(model_path, **local)
        local_identity = client.identity()
    sample_workers = int(spec.get("sample_workers", 1))
    if sample_workers < 1:
        raise ValueError("baseline sample_workers must be positive")
    method_api = (
        _api_identity(getattr(config, "api", None)) if method.requires_model else None
    )
    if local_identity is not None:
        method_api = local_identity
    run_identity = _run_identity(
        method=method.name,
        setting=method.setting,
        method_options=dict(spec.get("method_options", {})),
        method_api=method_api,
        replay_backend=spec.get("replay_backend") or {"type": "none"},
        model_identity=spec.get("model_identity"),
        input_path=input_path,
        input_sha256=_sha256(input_path),
        split=str(spec["split"]) if spec.get("split") is not None else None,
        sample_limit=spec.get("sample_limit"),
        max_events=spec.get("max_events", 64),
        prompt_char_budget=spec.get("prompt_char_budget", 100_000),
        benchmark_filter=spec.get("benchmark_filter"),
    )
    _validate_resume_identity(
        output_dir,
        results_path,
        run_identity,
        resume=resume,
        allow_legacy_resume=bool(spec.get("allow_legacy_resume", False)),
    )
    existing = _load_results(results_path) if resume else {}
    loaded_samples = load_baseline_samples(
        input_path,
        input_format=str(spec.get("input_format", "auto")),
        sample_limit=None,
        dataset_root=(
            _resolve(root, str(spec["dataset_root"]))
            if spec.get("dataset_root")
            else None
        ),
        max_events=int(spec.get("max_events", 64)),
    )
    samples, excluded = _select_evaluation_pool(
        loaded_samples,
        setting=method.setting,
        split=str(spec["split"]) if spec.get("split") is not None else None,
    )
    benchmark_filter = spec.get("benchmark_filter")
    if benchmark_filter is not None:
        if not isinstance(benchmark_filter, list) or not benchmark_filter or not all(
            isinstance(value, str) and value for value in benchmark_filter
        ):
            raise TypeError("baseline benchmark_filter must be a non-empty list of slugs")
        before = len(samples)
        samples = [sample for sample in samples if sample.benchmark in benchmark_filter]
        excluded += before - len(samples)
    # Active ARIA measurements can be restricted to the native frameworks with
    # a tested pre/post hook.  Keep this as a runner-level pool operation so the
    # denominator, grouped counts, and matched reference all describe the same
    # two-MAS deployment surface rather than silently scoring unsupported hosts.
    mas_filter = (spec.get("method_options") or {}).get("mas_filter")
    if mas_filter is not None:
        if not isinstance(mas_filter, (list, tuple, set)) or not all(
            isinstance(value, str) and value.strip() for value in mas_filter
        ):
            raise TypeError("baseline method_options.mas_filter must be a non-empty list of MAS slugs")
        allowed = {value.strip().lower() for value in mas_filter}
        if not allowed:
            raise ValueError("baseline method_options.mas_filter must not be empty")
        before = len(samples)
        samples = [sample for sample in samples if str(sample.mas or "").lower() in allowed]
        excluded += before - len(samples)
    if spec.get("sample_limit") is not None:
        samples = samples[: int(spec["sample_limit"])]
    if not samples:
        raise ValueError(
            f"baseline method {method.name} has an empty {method.setting} evaluation pool"
        )
    if method.requires_model and client is None:
        raise ValueError(
            f"baseline method {method.name} requires top-level api configuration"
        )
    model = (
        ModelAdapter(
            client,
            prompt_char_budget=int(spec.get("prompt_char_budget", 100_000)),
        )
        if client is not None
        else None
    )
    replay = _load_replay_backend(
        spec.get("replay_backend"),
        spec={**spec, "output_dir": str(output_dir)},
        config=config,
    )
    # Keep every previously persisted row in the live ledger.  A resumed run
    # may replace only one sample at a time; writing just the rows that happen
    # to be resumable would silently drop invalid/unresolved rows if the
    # process is interrupted before those samples are retried.
    rows_by_id: dict[str, dict[str, Any]] = {
        sample.sample_id: dict(existing[sample.sample_id])
        for sample in samples
        if sample.sample_id in existing
    }

    def ordered_rows() -> list[dict[str, Any]]:
        return [
            rows_by_id[sample.sample_id]
            for sample in samples
            if sample.sample_id in rows_by_id
        ]

    pairs = []
    run_state = {
        "schema_version": "aria-baseline-run-state-v1",
        "status": "running",
        "method": method.name,
        "input_path": str(input_path),
        "input_sample_count": len(loaded_samples),
        "pool_sample_count": len(samples),
        "excluded_sample_count": excluded,
        "run_identity": run_identity,
        "active_sample_id": None,
        "completed_sample_ids": [],
        "failed_sample_ids": [],
    }
    atomic_write_json(output_dir / "run_state.json", run_state)
    pending_samples = []
    for sample in samples:
        existing_row = existing.get(sample.sample_id)
        row_resume_check = getattr(replay, "resume_row_is_valid", None)
        if (
            existing_row is not None
            and _row_is_resumable(existing_row)
            and (not callable(row_resume_check) or row_resume_check(existing_row))
        ):
            if isinstance(existing_row.get("control_evaluation"), Mapping):
                from dataclasses import replace
                sample = replace(sample, control_evaluation=existing_row["control_evaluation"])
            result = _result_from_row(existing_row)
            resumed_row = _result_row(existing_row, sample)
            rows_by_id[sample.sample_id] = resumed_row
            pairs.append((sample, result))
            run_state["completed_sample_ids"].append(sample.sample_id)
            continue
        pending_samples.append(sample)

    def run_one(sample: Any) -> tuple[Any, MethodResult, Exception | None]:
        selected_method = (
            method
            if sample_workers == 1
            else build_method(str(spec["method"]), **dict(spec.get("method_options", {})))
        )
        preparing = True
        try:
            prepare_sample = getattr(replay, "prepare_sample", None)
            if callable(prepare_sample):
                sample = prepare_sample(sample)
            preparing = False
            return sample, selected_method.run(sample, model, replay), None
        except Exception as exc:  # noqa: BLE001 - handled by the runner policy
            if spec.get("on_sample_error", "stop") != "continue":
                raise
            result = selected_method.result(sample, status="invalid")
            result.diagnostics["error"] = f"{type(exc).__name__}: {exc}"
            if preparing or isinstance(exc, (ModelApiError, TimeoutError, ConnectionError)):
                result.diagnostics["failure_category"] = "infrastructure"
            if isinstance(exc, StructuredGenerationError):
                result.model_calls.append(exc.record)
                candidate = _intervention_candidate(exc.record)
                if candidate is not None:
                    result.intervention = candidate
                    result.diagnostics["structurally_valid"] = False
            return sample, result, exc

    def commit(sample: Any, result: MethodResult, error: Exception | None) -> None:
        if error is not None:
            run_state["failed_sample_ids"].append(sample.sample_id)
        if result.intervention is not None:
            result.diagnostics.setdefault("structurally_valid", True)
        row = _result_row(result.to_dict(), sample)
        rows_by_id[sample.sample_id] = row
        pairs.append((sample, result))
        run_state["completed_sample_ids"].append(sample.sample_id)
        run_state["active_sample_id"] = None
        atomic_write_json(output_dir / "run_state.json", run_state)
        _atomic_write_jsonl(results_path, ordered_rows())

    def record_failure(exc: BaseException) -> None:
        run_state["status"] = "failed"
        run_state["error"] = f"{type(exc).__name__}: {exc}"
        # ENOSPC may also prevent recording the failure; preserve the original
        # exception and let resume use the last durable result ledger.
        with suppress(OSError):
            atomic_write_json(output_dir / "run_state.json", run_state)

    if sample_workers == 1:
        for sample in pending_samples:
            run_state["active_sample_id"] = sample.sample_id
            atomic_write_json(output_dir / "run_state.json", run_state)
            try:
                selected_sample, result, error = run_one(sample)
                commit(selected_sample, result, error)
            except Exception as exc:
                record_failure(exc)
                raise
    elif pending_samples:
        run_state["active_sample_id"] = pending_samples[0].sample_id
        atomic_write_json(output_dir / "run_state.json", run_state)
        try:
            _run_bounded_samples(pending_samples, run_one, commit, sample_workers)
        except BaseException as exc:
            record_failure(exc)
            raise

    # Resume loads results by sample ID, so this also canonicalizes legacy or
    # interrupted JSONL files that contain duplicate rows for one sample.
    rows = ordered_rows()
    _atomic_write_jsonl(results_path, rows)
    summary = aggregate_results(pairs)
    group_axis = "benchmark" if method.setting == "passive" else "mas"
    summary.update(
        {
            "method": method.name,
            "setting": method.setting,
            "pool_contract": (
                "failed source trajectories"
                if method.setting == "passive"
                else "all source test tasks"
            ),
            "group_axis": group_axis,
            "groups": aggregate_grouped_results(pairs, group_axis),
            "groups_by_benchmark": aggregate_grouped_results(pairs, "benchmark"),
            "groups_by_mas": aggregate_grouped_results(pairs, "mas"),
            "task_count": len({sample.task_id for sample, _ in pairs}),
            "input_sample_count": len(loaded_samples),
            "excluded_sample_count": excluded,
            "input_path": str(input_path),
            "results_path": str(results_path),
        }
    )
    atomic_write_json(output_dir / "summary.json", summary)
    manifest = {
        "schema_version": "aria-baseline-manifest-v1",
        "method": method.name,
        "provenance": dict(method.provenance),
        "method_options": dict(spec.get("method_options", {})),
        "method_api": method_api,
        "run_identity": run_identity,
        "replay_backend": str((spec.get("replay_backend") or {}).get("type", "none")),
        "input": {"path": str(input_path), "sha256": _sha256(input_path)},
        "artifacts": {
            "results.jsonl": _sha256(results_path),
            "summary.json": _sha256(output_dir / "summary.json"),
        },
        "sample_count": len(rows),
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    run_state.update({"status": "completed", "active_sample_id": None})
    atomic_write_json(output_dir / "run_state.json", run_state)
    return summary


def _result_row(row: Mapping[str, Any], sample: Any) -> dict[str, Any]:
    """Persist the fixed evaluation identity and uninterrupted control outcome."""

    return {
        **dict(row),
        "task_id": sample.task_id,
        "benchmark": sample.benchmark,
        "mas": sample.mas,
        "split": sample.split,
        "source_success": sample.source_success,
        "control_success": sample.control_success,
        "control_evaluation": dict(sample.control_evaluation),
        "paired_control_path": sample.metadata.get("paired_control_path"),
    }


def _load_replay_backend(
    value: Any, *, spec: Mapping[str, Any], config: ExperimentConfig
) -> Any | None:
    """Resolve native execution, recorded evaluator outcomes, or proposal-only mode."""

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("baseline replay_backend must be a mapping")
    backend_type = value.get("type", "none")
    options = dict(value.get("options", {}))
    if backend_type == "none":
        return None
    if backend_type == "native":
        from baselines.replay.native import build_native_replay_backend

        return build_native_replay_backend(options, dict(spec), config)
    if backend_type == "ledger":
        from baselines.replay.ledger import build_ledger_replay_backend

        return build_ledger_replay_backend(options, dict(spec), config)
    raise ValueError(
        "baseline replay_backend.type must be none, ledger, or native"
    )


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _load_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["sample_id"])] = row
    return rows


def _row_is_resumable(row: Mapping[str, Any]) -> bool:
    """Return whether an existing result is strong enough to skip on resume.

    Sample IDs alone are not completion markers: a prior endpoint outage,
    malformed model response, or unresolved evaluator decision is persisted as
    an auditable row and must be retried after the dependency recovers.
    """

    if str(row.get("status") or "") == "invalid":
        return False
    method = str(row.get("method") or "")
    setting = str(row.get("setting") or "")
    if method == "no_intervention":
        return True
    if setting == "active" and row.get("intervention") is None:
        return True
    # Proposal-only jobs intentionally have no replay backend. They are a
    # completed schema-inspection artifact and should remain resumable.
    if str(row.get("status") or "") == "proposed" and not row.get("executions"):
        return True
    executions = row.get("executions")
    if not isinstance(executions, list) or not executions:
        return False
    selected_label = row.get("selected_execution_label")
    selected = next(
        (
            execution
            for execution in executions
            if isinstance(execution, Mapping)
            and (
                selected_label is None
                or execution.get("label") == selected_label
            )
        ),
        None,
    )
    if not isinstance(selected, Mapping):
        return False
    if str(selected.get("status") or "") != "completed":
        return False
    evaluation = selected.get("evaluation")
    if not isinstance(evaluation, Mapping):
        return False
    success = evaluation.get("success")
    if not isinstance(success, bool):
        success = evaluation.get("correct")
    return isinstance(success, bool)


def _result_from_row(row: Mapping[str, Any]) -> MethodResult:
    from baselines.core.types import ExecutionRecord

    result = MethodResult(
        sample_id=str(row["sample_id"]),
        method=str(row["method"]),
        setting=str(row["setting"]),
        status=str(row["status"]),
        intervention=row.get("intervention"),
        attribution=row.get("attribution"),
        diagnostics=dict(row.get("diagnostics", {})),
        source_execution_count=float(row.get("source_execution_count", 1.0)),
        selected_execution_label=row.get("selected_execution_label"),
        provenance=dict(row.get("provenance", {})),
    )
    result.executions = [
        ExecutionRecord(
            label=str(item["label"]),
            status=str(item["status"]),
            evaluation=dict(item.get("evaluation", {})),
            execution_mode=str(item.get("execution_mode", "unknown")),
            execution_count=float(item.get("execution_count", 1.0)),
            artifact_refs=dict(item.get("artifact_refs", {})),
            error=item.get("error"),
        )
        for item in row.get("executions", [])
    ]
    return result


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_evaluation_pool(
    samples: list[Any], *, setting: str, split: str | None
) -> tuple[list[Any], int]:
    selected = [sample for sample in samples if split is None or sample.split == split]
    if setting == "passive":
        unknown = [
            sample.sample_id for sample in selected if sample.source_success is None
        ]
        if unknown:
            raise ValueError(
                "passive baseline pool requires a resolved source evaluation; missing for: "
                + ", ".join(unknown[:10])
            )
        selected = [sample for sample in selected if sample.source_success is False]
    elif setting == "active":
        # A runtime system does not know in advance which run will fail, so the
        # active pool is every task rather than the failed ones.  What the pool
        # must satisfy is that property directly: both outcomes present, and no
        # task left unresolved.
        outcomes = {sample.source_success for sample in selected}
        unresolved = [
            sample.sample_id for sample in selected if sample.source_success is None
        ]
        if unresolved:
            raise ValueError(
                "active baseline pool requires a resolved uninterrupted outcome "
                "for every task; missing for: " + ", ".join(unresolved[:10])
            )
        if not {True, False} <= outcomes:
            raise ValueError(
                "active baseline pool must contain both successful and failed "
                "uninterrupted host runs; reporting SR only on runs already "
                "known to have failed would presuppose the gate's decision"
            )
    else:
        raise ValueError(f"unsupported baseline setting: {setting}")
    return selected, len(samples) - len(selected)


def _intervention_candidate(record: Any) -> dict[str, Any] | None:
    stage = str(getattr(record, "stage", ""))
    if not any(
        token in stage for token in ("repair", "intervention", "core", "screen")
    ):
        return None
    parsed = getattr(record, "parsed_response", None)
    if not isinstance(parsed, Mapping):
        return None
    nested = parsed.get("intervention")
    return dict(nested) if isinstance(nested, Mapping) else dict(parsed)


def _api_identity(api: Any | None) -> dict[str, Any] | None:
    if api is None:
        return None
    return {
        "provider": api.provider,
        "protocol": api.protocol,
        "base_url": api.base_url,
        "model": api.model,
        "model_id": api.model_id,
        "runtime_id": api.runtime_id,
        "reasoning_effort": api.reasoning_effort,
        "temperature": api.temperature,
        "seed": api.seed,
        "max_tokens": api.max_tokens,
    }


def _run_identity(**payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, default=str
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_resume_identity(
    output_dir: Path,
    results_path: Path,
    expected: str,
    *,
    resume: bool,
    allow_legacy_resume: bool = False,
) -> None:
    if not resume or not results_path.exists():
        return
    manifest_path = output_dir / "manifest.json"
    state_path = output_dir / "run_state.json"
    identity_path = (
        manifest_path
        if manifest_path.is_file()
        else state_path
        if state_path.is_file()
        else None
    )
    if identity_path is None:
        if allow_legacy_resume:
            return
        raise ValueError(
            f"cannot resume baseline results without an identity artifact in {output_dir}"
        )
    identity_record = json.loads(identity_path.read_text(encoding="utf-8"))
    actual = (
        identity_record.get("run_identity")
        if isinstance(identity_record, Mapping)
        else None
    )
    if actual != expected:
        if allow_legacy_resume:
            return
        raise ValueError(
            "baseline resume identity changed; use a distinct output_dir for the "
            "input, method options, backbone, split, or replay mode"
        )
