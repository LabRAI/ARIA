"""Build repair-supervision datasets from completed cognitive failures."""

from __future__ import annotations

import logging
import random
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from common.artifacts import (
    BatchCheckpoint,
    append_jsonl_fsync,
    atomic_write_json,
    atomic_write_jsonl,
    utc_now,
)
from common.model_api import ModelApiError, is_capacity_exhausted_error
from common.runtime_support import safe_name
from inference.native import get_native_mas_backend
from repair.artifacts import sanitize_trace_events, snapshot_run_artifacts
from repair.generator import (
    TRAJECTORY_REPLAY_OPERATOR_ID,
    StructuredRepairGenerationError,
    generate_repair_episode,
    materialize_continuation_task,
)
from repair.metadata import build_manifest, build_metadata, write_readme
from repair.rerun import (
    RepairSampleInvalid,
    run_repaired_sample,
)
from repair.schema import REPAIR_SCHEMA_VERSION
from repair.training_projection import write_training_ready
from repair.utils import (
    first_existing_path,
    load_json,
    load_jsonl,
    resolve_optional_path,
    resolve_path,
    source_batch_context,
)
from repair.v5_records import (
    build_v5_dataset_record,
    build_v5_gate_success_record,
    write_v5_sample_files,
)


def call_with_capacity_wait(
    call: Callable[[], Any],
    *,
    wait_seconds: float,
    max_waits: int,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[float, float], float] = random.uniform,
    on_wait: Callable[[int, float, BaseException], None] | None = None,
) -> Any:
    """Keep a transient provider-capacity outage out of the invalid ledger."""

    waits = 0
    while True:
        try:
            return call()
        except (ModelApiError, RepairSampleInvalid) as exc:
            reason = exc.reason if isinstance(exc, RepairSampleInvalid) else str(exc)
            if wait_seconds <= 0 or not is_capacity_exhausted_error(reason):
                raise
            if max_waits > 0 and waits >= max_waits:
                raise
            waits += 1
            delay = jitter(wait_seconds * 0.8, wait_seconds * 1.2)
            logging.getLogger("masfm.repair_dataset").warning(
                "Upstream model account capacity unavailable; keeping sample active "
                "and waiting %.1fs before capacity retry %s%s",
                delay,
                waits,
                f"/{max_waits}" if max_waits > 0 else " (unbounded)",
            )
            if on_wait is not None:
                on_wait(waits, delay, exc)
            sleep(delay)


def _record_capacity_wait(
    checkpoint: BatchCheckpoint,
    *,
    wait_number: int,
    delay: float,
    error: BaseException,
) -> None:
    checkpoint.last_error = (
        "upstream model account capacity unavailable; "
        f"capacity_wait={wait_number} next_retry_seconds={delay:.1f}: {error}"
    )
    checkpoint.save()


def run_repair_dataset(
    config: Any,
    spec: dict[str, Any],
    index: int = 0,
    *,
    client: Any | None = None,
) -> dict[str, Any]:
    """Generate an attribution-driven V6 repair dataset from inference failures."""

    logger = logging.getLogger("masfm.repair_dataset")
    name = str(spec.get("name") or f"repair-dataset-{index:02d}")
    operator_id = str(
        spec.get("repair_operator_id") or TRAJECTORY_REPLAY_OPERATOR_ID
    )
    if operator_id != TRAJECTORY_REPLAY_OPERATOR_ID:
        raise ValueError(
            f"unsupported repair operator: {operator_id}; "
            f"expected {TRAJECTORY_REPLAY_OPERATOR_ID}"
        )
    schema_version = str(spec.get("schema_version") or REPAIR_SCHEMA_VERSION)
    if schema_version != REPAIR_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported repair dataset schema: {schema_version}; "
            f"expected {REPAIR_SCHEMA_VERSION}"
        )

    source_batch_dir = resolve_path(spec["source_batch_dir"])
    if bool(spec.get("require_inference_completed", False)):
        require_completed_source_batch(source_batch_dir)
    source_mas_slug, source_benchmark_slug = source_batch_context(source_batch_dir)
    dataset_root = resolve_path(
        spec.get("dataset_dir")
        or Path(config.data_dir) / "repair_datasets" / safe_name(name)
    )
    repair_output_root = resolve_path(
        spec.get("repair_output_dir")
        or Path(config.output_dir).resolve()
        / "repair"
        / safe_name(source_mas_slug)
        / safe_name(source_benchmark_slug)
        / safe_name(name)
    )
    dataset_root.mkdir(parents=True, exist_ok=True)
    repair_output_root.mkdir(parents=True, exist_ok=True)

    dataset_path = dataset_root / "dataset.jsonl"
    invalid_path = dataset_root / "invalid_samples.jsonl"
    resume_enabled = bool(spec.get("resume", True))
    skip_sample_ids = {str(item) for item in spec.get("skip_sample_ids", [])}
    on_sample_error = str(spec.get("on_sample_error") or "fail")
    retry_invalid = bool(spec.get("retry_invalid", False))
    max_consecutive_endpoint_invalid = max(
        0, int(spec.get("max_consecutive_endpoint_invalid", 0))
    )
    capacity_wait_seconds = float(spec.get("capacity_wait_seconds", 0))
    capacity_wait_max_attempts = int(spec.get("capacity_wait_max_attempts", 0))
    if dataset_path.exists() and not resume_enabled:
        raise FileExistsError(
            f"repair dataset already exists: {dataset_path}; "
            "enable resume or use a new dataset_dir"
        )
    records = load_jsonl(dataset_path) if dataset_path.exists() else []
    incompatible = {
        str(record.get("schema_version"))
        for record in records
        if isinstance(record, dict)
        and record.get("schema_version") != schema_version
    }
    if incompatible:
        raise ValueError(
            "existing repair dataset uses incompatible schema(s) "
            f"{sorted(incompatible)}; the active repair contract requires a new dataset_dir"
        )
    invalid_events = load_jsonl(invalid_path) if invalid_path.exists() else []
    existing_sample_ids = {
        str(record["sample_id"])
        for record in records
        if isinstance(record, dict) and record.get("sample_id")
    }

    failure_rows = select_failure_rows(source_batch_dir, spec)
    if bool(spec.get("require_trace_acceptance", False)):
        from repair.window_theory_acceptance import evaluate_trace_acceptance

        acceptance = evaluate_trace_acceptance(source_batch_dir)
        atomic_write_json(dataset_root / "trace_acceptance.json", acceptance)
        eligible = set(acceptance["eligible_failure_ids"])
        failure_rows = [
            row for row in failure_rows if str(row.get("sample_id")) in eligible
        ]
    sample_limit = spec.get("sample_limit")
    if sample_limit is not None:
        failure_rows = failure_rows[: int(sample_limit)]
    rerun_enabled = bool(spec.get("rerun", True))
    gate_success_rate = float(spec.get("gate_success_sample_rate", 0.25))
    default_gate_success_limit = (
        max(1, int(len(failure_rows) * gate_success_rate))
        if failure_rows and gate_success_rate > 0
        else 0
    )
    gate_success_limit = int(
        spec.get("gate_success_sample_limit", default_gate_success_limit)
    )
    gate_success_rows = select_gate_success_rows(
        source_batch_dir, gate_success_limit
    )
    planned_sample_ids = {
        str(row["sample_id"])
        for row in [*failure_rows, *gate_success_rows]
    }
    unexpected_existing = existing_sample_ids - planned_sample_ids
    if unexpected_existing:
        raise ValueError(
            "existing repair dataset contains samples outside the current selection: "
            + ", ".join(sorted(unexpected_existing)[:10])
        )

    checkpoint = BatchCheckpoint.open(
        dataset_root / "run_state.json",
        run_id=name,
        total_samples=len(planned_sample_ids),
        completed_sample_ids=existing_sample_ids & planned_sample_ids,
    )
    if retry_invalid:
        checkpoint.invalid_sample_ids.difference_update(planned_sample_ids)
        checkpoint.save()
    logger.info(
        "Building repair dataset name=%s schema=%s failures=%s source=%s output=%s rerun=%s",
        name,
        schema_version,
        len(failure_rows),
        source_batch_dir,
        dataset_root,
        rerun_enabled,
    )
    consecutive_endpoint_invalid = 0
    endpoint_circuit_open = False

    for position, row in enumerate(failure_rows, start=1):
        sample_id = str(row["sample_id"])
        if sample_id in existing_sample_ids:
            logger.info(
                "Skipping completed repair sample %s/%s id=%s",
                position,
                len(failure_rows),
                sample_id,
            )
            continue
        if sample_id in checkpoint.invalid_sample_ids:
            logger.info(
                "Skipping terminal invalid repair sample %s/%s id=%s",
                position,
                len(failure_rows),
                sample_id,
            )
            continue
        if sample_id in skip_sample_ids:
            reason = "skipped by repair config"
            checkpoint.finish_sample(sample_id, "failed", reason)
            logger.warning("Skipping configured repair sample id=%s", sample_id)
            continue
        checkpoint.start_sample(sample_id)
        logger.info(
            "Processing repair sample %s/%s id=%s",
            position,
            len(failure_rows),
            sample_id,
        )
        try:
            record_root = dataset_root / "samples" / safe_name(sample_id)
            rerun_sample_root = repair_output_root / "samples" / safe_name(sample_id)
            if record_root.exists():
                logger.info("Resetting unfinished repair sample artifacts id=%s", sample_id)
                shutil.rmtree(record_root)
            if rerun_sample_root.exists():
                shutil.rmtree(rerun_sample_root)
            source = load_source_artifacts(row)
            snapshot_run_artifacts(
                source["run_dir"],
                record_root / "failed",
            )
            episode = call_with_capacity_wait(
                lambda: generate_repair_episode(
                    source=source,
                    client=client,
                    repair_trials=int(spec.get("repair_trials", 1)),
                    structured_output_retries=int(
                        spec.get("structured_output_retries", 2)
                    ),
                    max_events=int(spec.get("attribution_max_events", 64)),
                ),
                wait_seconds=capacity_wait_seconds,
                max_waits=capacity_wait_max_attempts,
                on_wait=lambda wait_number, delay, error: _record_capacity_wait(
                    checkpoint,
                    wait_number=wait_number,
                    delay=delay,
                    error=error,
                ),
            )
            executable_trials = []
            for trial in episode.complete_trials:
                if trial.repair is None or trial.attribution is None:
                    continue
                materialization = materialize_continuation_task(
                    episode.incident,
                    trial.attribution,
                    trial.repair,
                )
                executable_trials.append(
                    {
                        "trial_id": trial.trial_id,
                        "trial_index": trial.trial_index,
                        "localization": trial.localization.to_dict()
                        if trial.localization is not None
                        else None,
                        "attribution": trial.attribution.model_output(),
                        "repair": trial.repair.to_dict(),
                        "materialization": materialization,
                    }
                )
            execution_enabled = bool(rerun_enabled and executable_trials)
            repair_payload: dict[str, Any] = {
                "status": "not_run",
                "enabled": execution_enabled,
                "execution_mode": (
                    "non_exact_continuation" if execution_enabled else None
                ),
                "attempts": [],
                "trials": [],
                "api_key_env": getattr(getattr(config, "api", None), "api_key_env", None),
            }
            if execution_enabled:
                repair_payload.update(
                    call_with_capacity_wait(
                        lambda: run_repaired_sample(
                            config=config,
                            spec=spec,
                            source=source,
                            repair_prompt=None,
                            repair_output_root=repair_output_root,
                            dataset_sample_root=record_root,
                            repair_trials=executable_trials,
                        ),
                        wait_seconds=capacity_wait_seconds,
                        max_waits=capacity_wait_max_attempts,
                        on_wait=lambda wait_number, delay, error: _record_capacity_wait(
                            checkpoint,
                            wait_number=wait_number,
                            delay=delay,
                            error=error,
                        ),
                    )
                )
            record = build_v5_dataset_record(
                name=name,
                spec=spec,
                config=config,
                source_batch_dir=source_batch_dir,
                source=source,
                episode=episode,
                execution=repair_payload,
            )
            write_v5_sample_files(
                record_root, source, episode, repair_payload, record
            )
            records.append(record)
            append_jsonl_fsync(dataset_path, record)
            checkpoint.finish_sample(sample_id, "completed")
            consecutive_endpoint_invalid = 0
        except ModelApiError as exc:
            invalid = RepairSampleInvalid(
                sample_id=sample_id,
                run_id=f"{name}-{sample_id}-proposal",
                variant="proposal",
                stage="repair_generation",
                category="endpoint_invalid",
                reason=str(exc),
                output_dir=(
                    dataset_root / "samples" / safe_name(sample_id) / "repair"
                ),
            )
            if on_sample_error != "continue":
                checkpoint.finish_sample(sample_id, "failed", str(invalid))
                checkpoint.finish("failed")
                raise invalid from exc
            invalid_event = build_invalid_sample_event(name=name, error=invalid)
            invalid_events.append(invalid_event)
            append_jsonl_fsync(invalid_path, invalid_event)
            checkpoint.finish_sample(sample_id, "invalid", invalid.reason)
            consecutive_endpoint_invalid += 1
            logger.warning(
                "Continuing after repair-generation endpoint invalid id=%s error=%s",
                sample_id,
                exc,
            )
            if (
                max_consecutive_endpoint_invalid
                and consecutive_endpoint_invalid >= max_consecutive_endpoint_invalid
            ):
                checkpoint.last_error = (
                    "endpoint circuit opened after "
                    f"{consecutive_endpoint_invalid} consecutive invalid samples"
                )
                logger.error(
                    "Stopping repair batch after %s consecutive endpoint-invalid samples",
                    consecutive_endpoint_invalid,
                )
                endpoint_circuit_open = True
                break
            continue
        except StructuredRepairGenerationError as exc:
            invalid = RepairSampleInvalid(
                sample_id=sample_id,
                run_id=f"{name}-{sample_id}-proposal",
                variant="proposal",
                stage="repair_generation",
                category="structured_output_invalid",
                reason=str(exc),
                output_dir=(
                    dataset_root / "samples" / safe_name(sample_id) / "repair"
                ),
            )
            if on_sample_error != "continue":
                checkpoint.finish_sample(sample_id, "failed", str(invalid))
                checkpoint.finish("failed")
                raise invalid from exc
            invalid_event = build_invalid_sample_event(name=name, error=invalid)
            invalid_events.append(invalid_event)
            append_jsonl_fsync(invalid_path, invalid_event)
            checkpoint.finish_sample(sample_id, "invalid", invalid.reason)
            consecutive_endpoint_invalid = 0
            logger.warning(
                "Continuing after invalid structured repair sample id=%s error=%s",
                sample_id,
                exc,
            )
            continue
        except RepairSampleInvalid as exc:
            if on_sample_error != "continue":
                checkpoint.finish_sample(sample_id, "failed", str(exc))
                checkpoint.finish("failed")
                raise
            invalid_event = build_invalid_sample_event(name=name, error=exc)
            invalid_events.append(invalid_event)
            append_jsonl_fsync(invalid_path, invalid_event)
            checkpoint.finish_sample(sample_id, "invalid", exc.reason)
            consecutive_endpoint_invalid = (
                consecutive_endpoint_invalid + 1
                if exc.category == "endpoint_invalid"
                else 0
            )
            logger.warning(
                "Continuing after invalid repair sample id=%s category=%s stage=%s variant=%s",
                sample_id,
                exc.category,
                exc.stage,
                exc.variant,
            )
            if (
                max_consecutive_endpoint_invalid
                and consecutive_endpoint_invalid >= max_consecutive_endpoint_invalid
            ):
                checkpoint.last_error = (
                    "endpoint circuit opened after "
                    f"{consecutive_endpoint_invalid} consecutive invalid samples"
                )
                logger.error(
                    "Stopping repair batch after %s consecutive endpoint-invalid samples",
                    consecutive_endpoint_invalid,
                )
                endpoint_circuit_open = True
                break
            continue
        except BaseException as exc:
            checkpoint.finish_sample(sample_id, "failed", str(exc))
            checkpoint.finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
            raise

    if endpoint_circuit_open:
        gate_success_rows = []
    if gate_success_rows:
        logger.info(
            "Adding %s gate-success samples to dataset %s",
            len(gate_success_rows),
            name,
        )
    for gate_success_row in gate_success_rows:
        sample_id = str(gate_success_row["sample_id"])
        if sample_id in existing_sample_ids:
            continue
        checkpoint.start_sample(sample_id)
        try:
            gate_success_source = load_source_artifacts(gate_success_row)
            record_root = dataset_root / "samples" / safe_name(sample_id)
            if record_root.exists():
                shutil.rmtree(record_root)
            snapshot_run_artifacts(
                gate_success_source["run_dir"], record_root / "success"
            )
            gate_success_record = build_v5_gate_success_record(
                name=name,
                spec=spec,
                config=config,
                source_batch_dir=source_batch_dir,
                source=gate_success_source,
            )
            records.append(gate_success_record)
            append_jsonl_fsync(dataset_path, gate_success_record)
            checkpoint.finish_sample(sample_id, "completed")
        except BaseException as exc:
            checkpoint.finish_sample(sample_id, "failed", str(exc))
            checkpoint.finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
            raise

    try:
        atomic_write_jsonl(dataset_path, records)
        training_manifest = write_training_ready(dataset_root, records)
        final_status = (
            "partial"
            if checkpoint.failed_sample_ids or checkpoint.invalid_sample_ids
            else "completed"
        )
        metadata = build_metadata(
            name=name,
            spec=spec,
            source_batch_dir=source_batch_dir,
            dataset_root=dataset_root,
            config=config,
            records=records,
        )
        atomic_write_json(dataset_root / "metadata.json", metadata)
        manifest = build_manifest(
            name=name,
            spec=spec,
            source_batch_dir=source_batch_dir,
            dataset_root=dataset_root,
            repair_output_root=repair_output_root,
            records=records,
            schema_version=schema_version,
            status=final_status,
            invalid_sample_ids=checkpoint.invalid_sample_ids,
            invalid_attempts=len(invalid_events),
        )
        atomic_write_json(dataset_root / "manifest.json", manifest)
        write_readme(dataset_root, manifest)
        checkpoint.finish(final_status)
    except BaseException as exc:
        checkpoint.last_error = str(exc)
        checkpoint.finish("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed")
        raise

    logger.info("Finished repair dataset name=%s output=%s", name, dataset_root)
    return {
        "type": "repair_dataset",
        "name": name,
        "status": final_status,
        "dataset_dir": str(dataset_root),
        "dataset_path": str(dataset_path),
        "manifest_path": str(dataset_root / "manifest.json"),
        "metadata_path": str(dataset_root / "metadata.json"),
        "training_ready_dir": str(dataset_root / "training_ready"),
        "training_ready_counts": training_manifest["counts"],
        "schema_version": schema_version,
        "sample_count": len(records),
        "invalid_samples": len(checkpoint.invalid_sample_ids),
        "invalid_attempts": len(invalid_events),
        "invalid_path": str(invalid_path),
        "endpoint_circuit_open": endpoint_circuit_open,
        "successful_repairs": sum(
            1
            for item in records
            if item["outcome"].get("verified_successful") is True
        ),
        "patched_passes": sum(
            1
            for item in records
            if item["outcome"].get("repair_success") is True
        ),
    }


def build_invalid_sample_event(
    *,
    name: str,
    error: RepairSampleInvalid,
) -> dict[str, Any]:
    """Build a non-training audit event for a terminal infrastructure invalid."""

    return {
        "schema_version": "aria-repair-invalid-v1",
        "dataset_id": name,
        "sample_id": error.sample_id,
        "status": "invalid",
        "category": error.category,
        "stage": error.stage,
        "variant": error.variant,
        "run_id": error.run_id,
        "reason": error.reason,
        "output_dir": str(error.output_dir),
        "retryable": True,
        "created_at": utc_now(),
    }


def select_gate_success_rows(
    source_batch_dir: Path, limit: int
) -> list[dict[str, Any]]:
    results_by_sample = load_results_by_sample(source_batch_dir / "results.jsonl")
    rows = [
        item
        for item in results_by_sample.values()
        if (item.get("evaluation") or {}).get("correct") is True
        and item.get("status") == "completed"
        and item.get("output_dir")
    ]
    return rows[:limit]


def select_failure_rows(
    source_batch_dir: Path,
    spec: dict[str, Any],
) -> list[dict[str, Any]]:
    """Select only evaluator-resolved cognitive failures.

    Failed native runs and unresolved official harness results are diagnostics,
    never repair supervision.
    """

    results_by_sample = load_results_by_sample(source_batch_dir / "results.jsonl")
    candidates = (
        load_jsonl(source_batch_dir / "error_trajectories.jsonl")
        if (source_batch_dir / "error_trajectories.jsonl").exists()
        else list(results_by_sample.values())
    )
    merged: list[dict[str, Any]] = []
    for row in candidates:
        sample_id = row.get("sample_id")
        if sample_id is None:
            continue
        full = dict(row)
        full.update(results_by_sample.get(str(sample_id), {}))
        evaluation = full.get("evaluation") if isinstance(full.get("evaluation"), dict) else {}
        if full.get("status") != "completed" or evaluation.get("correct") is not False:
            continue
        merged.append(full)

    sample_ids = spec.get("sample_ids")
    if sample_ids:
        wanted = {str(item) for item in sample_ids}
        merged = [item for item in merged if str(item.get("sample_id")) in wanted]
    return merged


def require_completed_source_batch(source_batch_dir: Path) -> None:
    """Require one resolved latest result for every sample in the source manifest."""

    manifest_path = source_batch_dir / "batch_manifest.json"
    results_path = source_batch_dir / "results.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing source batch manifest: {manifest_path}")
    if not results_path.exists():
        raise FileNotFoundError(f"missing source batch results: {results_path}")

    manifest = load_json(manifest_path)
    sample_ids = manifest.get("sample_ids") if isinstance(manifest, dict) else None
    if not isinstance(sample_ids, list) or not all(
        isinstance(sample_id, str) and sample_id for sample_id in sample_ids
    ):
        raise ValueError(
            f"source batch manifest has invalid sample_ids: {manifest_path}"
        )
    expected = set(sample_ids)
    if len(expected) != len(sample_ids):
        raise ValueError(
            f"source batch manifest has duplicate sample_ids: {manifest_path}"
        )

    results = load_results_by_sample(results_path)
    missing = expected - set(results)
    unexpected = set(results) - expected
    unresolved = {
        sample_id
        for sample_id in expected & set(results)
        if results[sample_id].get("status") != "completed"
        or not isinstance(
            (results[sample_id].get("evaluation") or {}).get("correct"),
            bool,
        )
    }
    if missing or unexpected or unresolved:
        raise RuntimeError(
            f"source inference is not formally complete: {source_batch_dir} "
            f"(resolved={len(expected) - len(missing) - len(unresolved)}/{len(expected)}, "
            f"missing={len(missing)}, unresolved={len(unresolved)}, "
            f"unexpected={len(unexpected)})"
        )


def load_results_by_sample(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for item in load_jsonl(path):
        sample_id = item.get("sample_id")
        if sample_id is not None:
            rows[str(sample_id)] = item
    return rows


def load_source_artifacts(row: dict[str, Any]) -> dict[str, Any]:
    run_dir = resolve_path(row["output_dir"])
    result_path = run_dir / "result.json"
    result = load_json(result_path)
    mas_slug = str(result.get("mas") or row.get("mas") or "")
    try:
        native_backend = get_native_mas_backend(mas_slug)
    except NotImplementedError:
        native_backend = None
    native_messages_path = (
        run_dir / native_backend.messages_filename if native_backend is not None else None
    )
    native_trace_path = (
        run_dir / native_backend.trace_filename if native_backend is not None else None
    )
    evaluation_path = first_existing_path(
        run_dir / "evaluation.json",
        resolve_optional_path((result.get("evaluation") or {}).get("evaluation_path")),
    )
    if (
        evaluation_path is not None
        and not isinstance(result.get("evaluation"), dict)
    ):
        loaded_evaluation = load_json(evaluation_path)
        if isinstance(loaded_evaluation, dict):
            result["evaluation"] = loaded_evaluation

    messages_path = first_existing_path(
        resolve_optional_path((result.get("evaluation") or {}).get("messages_path")),
        resolve_optional_path(result.get("messages_path")),
        resolve_optional_path(row.get("messages_path")),
        run_dir / "messages.json",
        resolve_optional_path(result.get("native_messages_path")),
        native_messages_path,
    ) or (run_dir / "messages.json")
    trace_path = first_existing_path(
        resolve_optional_path((result.get("evaluation") or {}).get("trajectory_path")),
        resolve_optional_path(result.get("trajectory_path")),
        resolve_optional_path(row.get("trajectory_path")),
        run_dir / "trajectory.jsonl",
        resolve_optional_path(result.get("native_trace_path")),
        native_trace_path,
    ) or (run_dir / "trajectory.jsonl")
    messages = load_json(messages_path) if messages_path.exists() else []
    return {
        "run_dir": run_dir,
        "result_path": result_path,
        "evaluation_path": evaluation_path,
        "messages_path": messages_path,
        "trace_path": trace_path,
        "result": result,
        "messages": messages if isinstance(messages, list) else [],
        "trace_events": sanitize_trace_events(
            load_jsonl(trace_path) if trace_path.exists() else []
        ),
        "artifact_events": observable_artifact_events(run_dir, result),
    }


def observable_artifact_events(
    run_dir: Path,
    result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Expose host-observed, answer-blind artifacts to causal attribution."""

    if str(result.get("benchmark") or "") != "swe-bench-lite":
        return []
    patch_path = run_dir / "model.patch"
    if not patch_path.is_file():
        return []
    patch = patch_path.read_text(encoding="utf-8", errors="replace")
    evaluation = (
        result.get("evaluation")
        if isinstance(result.get("evaluation"), dict)
        else {}
    )
    content = patch if patch.strip() else "Host-observed workspace diff is empty."
    return [
        {
            "event_id": "source:artifact:workspace-patch",
            "role": "host",
            "kind": "workspace_artifact",
            "content": content,
            "artifact": {
                "kind": "workspace_patch",
                "path": patch_path.name,
                "bytes": len(patch.encode("utf-8")),
                "changed_files": list(evaluation.get("changed_files") or []),
                "host_observed": True,
            },
        }
    ]
