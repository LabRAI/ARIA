"""Config-driven inference orchestration over native MAS backends."""

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tqdm import tqdm

from aria.runtime import AriaRuntime
from common.artifacts import (
    BatchCheckpoint,
    append_jsonl_fsync,
    atomic_write_json,
    atomic_write_text,
    sample_metadata,
)
from common.bridge import build_bridge_spec
from common.catalog import (
    find_benchmark,
    find_mas,
)
from common.config import (
    ExperimentConfig,
    as_bool,
    normalize_existing_output_policy,
    required_keys,
)
from common.evaluation import (
    evaluate_sample,
    validate_benchmark_evaluator,
    write_batch_summary,
)
from common.failure_classification import classify_infrastructure_failure
from common.host_adapter import get_host_adapter
from common.mast_data import load_benchmark_items, load_benchmark_sample
from common.model_api import ApiConfig, ModelApiClient, ModelResponse
from common.preparation import (
    cache_gaia_attachments,
    clone_mas_specs,
    download_benchmark_specs,
    git_head,
)
from common.runtime_support import configure_root_logging, safe_run_id
from inference.artifacts import (
    build_runtime_state,
    clear_stale_aria_artifacts,
    compact_batch_result,
    format_duration,
    load_existing_batch_results,
    load_json_if_exists,
    load_jsonl_if_exists,
    load_latest_batch_results,
    messages_from_transcript,
    prepare_sample_workspace,
    refresh_standard_run_artifacts,
    resolve_workspace_path,
    trajectory_from_transcript,
    write_sample_run_state,
    write_standard_run_artifacts,
)
from inference.native import (
    NativeRunContext,
    get_native_mas_backend,
    validate_native_mas_protocol,
)


@dataclass(frozen=True)
class RunDestination:
    run_id: str
    run_root: Path


def job_output_root(config: ExperimentConfig, job_type: str | None = None) -> Path:
    resolved_job_type = job_type or str(getattr(config, "job_type", "inference"))
    return (config.output_dir / safe_run_id(resolved_job_type)).resolve()


def resolve_run_destination(
    config: ExperimentConfig,
    mas_slug: str,
    benchmark_slug: str,
    run_id: str,
) -> RunDestination:
    policy = normalize_existing_output_policy(config.output.existing_output_policy)
    base_run_id = str(run_id)
    base_root = (job_output_root(config) / mas_slug / benchmark_slug / safe_run_id(base_run_id)).resolve()
    if not base_root.exists():
        return RunDestination(run_id=base_run_id, run_root=base_root)
    if policy == "resume":
        return RunDestination(run_id=base_run_id, run_root=base_root)
    if policy == "overwrite":
        shutil.rmtree(base_root)
        return RunDestination(run_id=base_run_id, run_root=base_root)
    if policy == "error":
        raise FileExistsError(
            f"run output already exists: {base_root}; set "
            "output.existing_output_policy to resume, overwrite, or timestamp"
        )

    timestamped_run_id = f"{base_run_id}-{timestamp_suffix()}"
    candidate_root = (job_output_root(config) / mas_slug / benchmark_slug / safe_run_id(timestamped_run_id)).resolve()
    while candidate_root.exists():
        timestamped_run_id = f"{base_run_id}-{timestamp_suffix()}"
        candidate_root = (job_output_root(config) / mas_slug / benchmark_slug / safe_run_id(timestamped_run_id)).resolve()
    return RunDestination(run_id=timestamped_run_id, run_root=candidate_root)


def run_output_has_results(run_root: Path) -> bool:
    return any((run_root / marker).exists() for marker in ("result.json", "results.jsonl"))


def timestamp_suffix() -> str:
    return datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")


def run_inference(
    config: ExperimentConfig,
    client: ModelApiClient,
) -> list[dict[str, Any]]:
    configure_root_logging(config.log_level)
    logger = logging.getLogger("masfm")
    experiments = expand_experiment_matrix(config.experiments)
    validate_expanded_experiments(config, experiments)
    logger.info("Starting inference job: count=%s aria=%s", len(experiments), config.aria.enabled)

    if config.prepare.get("clone_mas"):
        slugs = sorted({item["mas"] for item in experiments})
        logger.info("Verifying MAS clones: %s", ", ".join(slugs))
        clone_mas_specs(config.external_dir, slugs)
    if config.prepare.get("download_benchmarks"):
        slugs = sorted({item["benchmark"] for item in experiments})
        logger.info("Verifying benchmark cache: %s", ", ".join(slugs))
        download_benchmark_specs(config.data_dir, slugs)
    if config.prepare.get("gaia_attachments"):
        gaia_experiments = [
            item for item in experiments if str(item.get("benchmark")) == "gaia"
        ]
        for item in gaia_experiments:
            cached = cache_gaia_attachments(
                config.data_dir,
                split=str(item.get("sample_split", "validation")),
                offset=int(item.get("sample_offset", 0)),
                limit=(
                    int(item["sample_limit"])
                    if item.get("sample_limit") is not None
                    else None
                ),
                sample_ids=(
                    [str(value) for value in item["sample_ids"]]
                    if item.get("sample_ids") is not None
                    else None
                ),
            )
            logger.info(
                "Verified GAIA attachments experiment=%s count=%s",
                item.get("name"),
                len(cached),
            )

    results = []
    for index, experiment in enumerate(experiments):
        try:
            if is_batch_experiment(experiment):
                result = run_batch_experiment(config, client, experiment, index)
            else:
                result = run_single_experiment(config, client, experiment, index)
        except Exception as exc:
            logger.exception(
                "Experiment failed mas=%s benchmark=%s",
                experiment.get("mas"),
                experiment.get("benchmark"),
            )
            result = write_failed_experiment(config, experiment, index, exc)
        finally:
            run_logger_name = str(
                experiment.get("name")
                or f"{index:02d}-{experiment.get('mas', 'unknown-mas')}-{experiment.get('benchmark', 'unknown-benchmark')}"
            )
            close_run_logger(logging.getLogger(f"masfm.run.{safe_run_id(run_logger_name)}"))
        results.append(result)
    logger.info("Finished inference job")
    return results


def expand_experiment_matrix(experiments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand YAML-declared MAS/benchmark/model matrices into concrete runs."""

    expanded: list[dict[str, Any]] = []
    for experiment in experiments:
        if "matrix" not in experiment:
            expanded.append(experiment)
            continue

        matrix = experiment.get("matrix") or {}
        mas_values = matrix_values(matrix, "mas", experiment.get("mas"))
        benchmark_values = matrix_values(matrix, "benchmark", experiment.get("benchmark"))
        runtime_values = matrix_runtime_values(matrix, experiment)

        base = {key: value for key, value in experiment.items() if key != "matrix"}
        template = str(base.get("name_template") or base.get("name") or "{mas}-{benchmark}-{model}")
        for mas_slug in mas_values:
            for benchmark_slug in benchmark_values:
                for runtime in runtime_values:
                    item = dict(base)
                    item["mas"] = mas_slug
                    item["benchmark"] = benchmark_slug
                    apply_runtime_overrides(item, runtime)
                    runtime_name = str(
                        item.get("runtime")
                        or item.get("runtime_name")
                        or item.get("runtime_id")
                        or item.get("model")
                        or "default-runtime"
                    )
                    item.setdefault(
                        "name",
                        template.format(
                            mas=safe_run_id(str(mas_slug)),
                            benchmark=safe_run_id(str(benchmark_slug)),
                            model=safe_run_id(str(item.get("model") or "default-model")),
                            provider=safe_run_id(str(item.get("provider") or "default-provider")),
                            runtime=safe_run_id(runtime_name),
                            llm=safe_run_id(runtime_name),
                        ),
                    )
                    item.pop("name_template", None)
                    expanded.append(item)
    return expanded


def matrix_runtime_values(matrix: dict[str, Any], experiment: dict[str, Any]) -> list[dict[str, Any]]:
    del experiment
    values = matrix["runtimes"]
    runtimes: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        runtime = dict(value)
        runtime["runtime"] = str(runtime.pop("name"))
        runtimes.append(runtime)
    return runtimes


def validate_expanded_experiments(
    config: ExperimentConfig,
    experiments: list[dict[str, Any]],
) -> None:
    for index, experiment in enumerate(experiments):
        required_keys(experiment, {"name", "mas", "benchmark", "native"}, f"expanded experiment[{index}]")
        mas = find_mas(str(experiment["mas"]))
        benchmark_slug = str(experiment["benchmark"])
        find_benchmark(benchmark_slug)
        if mas.slug == "langgraph" and benchmark_slug not in mas.paper_benchmarks:
            raise ValueError(f"LangGraph Debate has no execution tools for benchmark {benchmark_slug}")
        evaluation = experiment.get("evaluation") or {}
        harness = evaluation.get("harness")
        expected_harness = {
            "gaia": "gaia-official",
            "swe-bench-lite": "swebench-official",
        }.get(benchmark_slug)
        if harness is not None and harness != expected_harness:
            raise ValueError(
                f"expanded experiment[{index}].evaluation.harness={harness} "
                f"does not match benchmark {benchmark_slug}"
            )
        if is_batch_experiment(experiment):
            validate_benchmark_evaluator(benchmark_slug)
        api = effective_api_config(config.api, experiment)
        if as_bool(experiment["native"], False, name=f"expanded experiment[{index}].native"):
            validate_native_mas_protocol(mas.slug, api.protocol)


def apply_runtime_overrides(item: dict[str, Any], runtime: dict[str, Any]) -> None:
    for key, value in runtime.items():
        normalized_key = "runtime" if key in {"name", "id"} else key
        if value is not None:
            item[normalized_key] = value


def matrix_values(matrix: dict[str, Any], key: str, fallback: Any) -> list[Any]:
    plural = f"{key}s"
    if plural in matrix:
        value = matrix[plural]
    elif key in matrix:
        value = matrix[key]
    else:
        value = fallback
    if value is None:
        return [None]
    if isinstance(value, list):
        return value
    return [value]


def is_batch_experiment(experiment: dict[str, Any]) -> bool:
    return bool(experiment.get("batch")) or str(experiment.get("sample_source", "sample")) == "full"


def effective_api_config(api: ApiConfig, overrides: dict[str, Any]) -> ApiConfig:
    provider = str(overrides.get("provider", api.provider))
    protocol = str(overrides.get("protocol", api.protocol))
    protocol_changed = protocol != api.protocol
    api_key_required = as_bool(
        overrides.get("api_key_required"),
        api.api_key_required,
    )
    runtime_name = overrides.get("runtime") or overrides.get("runtime_name")
    runtime_id = overrides.get("runtime_id", api.runtime_id)
    if runtime_id is None and runtime_name:
        runtime_id = f"runtime/{safe_run_id(str(runtime_name))}"
    temperature_raw = overrides.get(
        "temperature",
        None if protocol_changed else api.temperature,
    )
    reasoning_effort = (
        str(overrides["reasoning_effort"])
        if overrides.get("reasoning_effort") is not None
        else (None if protocol_changed else api.reasoning_effort)
    )
    seed = (
        int(overrides["seed"])
        if overrides.get("seed") is not None
        else (None if protocol_changed else api.seed)
    )
    api_key_env_raw = overrides.get("api_key_env", api.api_key_env)
    return ApiConfig(
        provider=provider,
        protocol=protocol,
        base_url=str(overrides.get("base_url", api.base_url)),
        model=str(overrides.get("model", api.model)),
        api_key_env=str(api_key_env_raw) if api_key_env_raw is not None else None,
        api_key_required=api_key_required,
        timeout_seconds=int(overrides.get("timeout_seconds", api.timeout_seconds)),
        max_tokens=int(overrides.get("max_tokens", api.max_tokens)),
        context_window_tokens=(
            int(overrides["context_window_tokens"])
            if overrides.get("context_window_tokens") is not None
            else api.context_window_tokens
        ),
        temperature=float(temperature_raw) if temperature_raw is not None else None,
        reasoning_effort=reasoning_effort,
        retries=int(overrides.get("retries", api.retries)),
        retry_seconds=float(overrides.get("retry_seconds", api.retry_seconds)),
        runtime_id=runtime_id,
        model_id=overrides.get("model_id", api.model_id),
        seed=seed,
        anthropic_version=str(overrides.get("anthropic_version", api.anthropic_version)),
        upstream_provider=(
            str(overrides["upstream_provider"])
            if overrides.get("upstream_provider") is not None
            else (None if provider != api.provider else api.upstream_provider)
        ),
        allow_fallbacks=(
            as_bool(
                overrides["allow_fallbacks"],
                True,
                name="runtime.allow_fallbacks",
            )
            if overrides.get("allow_fallbacks") is not None
            else (None if provider != api.provider else api.allow_fallbacks)
        ),
    )


def client_for_api(client: Any, api: ApiConfig) -> Any:
    if isinstance(client, ModelApiClient) and client.config != api:
        return ModelApiClient(api)
    return client


def invoke_model(client: Any, messages: list[dict[str, Any]]) -> ModelResponse:
    if isinstance(client, ModelApiClient):
        return client.invoke(messages)
    content = client.complete(messages)
    if not isinstance(content, str):
        raise TypeError("injected model client complete() must return a string")
    return ModelResponse(
        text=content,
        response_id=None,
        usage={},
        stop_reason=None,
        latency_seconds=0.0,
        attempt_count=1,
        protocol="injected-test-client",
        raw={},
    )


def api_settings(api: ApiConfig) -> dict[str, Any]:
    return {
        "provider": api.provider,
        "protocol": api.protocol,
        "base_url": api.base_url,
        "endpoint_url": api.endpoint_url,
        "model": api.model,
        "model_id": api.model_id or f"model/{safe_run_id(api.model)}",
        "runtime_id": api.runtime_id,
        "api_key_env": api.api_key_env,
        "api_key_required": api.api_key_required,
        "max_tokens": api.max_tokens,
        "context_window_tokens": api.context_window_tokens,
        "temperature": api.temperature,
        "timeout_seconds": api.timeout_seconds,
        "retries": api.retries,
        "retry_seconds": api.retry_seconds,
        "reasoning_effort": api.reasoning_effort,
        "anthropic_version": api.anthropic_version,
        "seed": api.seed,
        "upstream_provider": api.upstream_provider,
        "allow_fallbacks": api.allow_fallbacks,
    }


def ensure_native_clone_present(external_dir: Path, mas: Any) -> None:
    """Abort early if a native MAS run is missing its source checkout.

    The MAS repos under external/mas/* are gitignored, so on a fresh machine
    (e.g. a cluster node) they may be absent while prepare.clone_mas is false.
    Without this guard every sample fails identically deep in the runner
    (ModuleNotFoundError), wasting a full batch before the cause is visible.
    """

    clone_path = external_dir / "mas" / mas.clone_dir
    native_backend = get_native_mas_backend(mas.slug)
    marker = native_backend.clone_marker
    if not marker:  # Package-backed native framework.
        return
    present = clone_path.is_dir() and (clone_path / marker).exists()
    if present:
        return
    raise FileNotFoundError(
        f"native {mas.slug} run requires its source checkout at {clone_path}, "
        "but it is missing or incomplete. external/mas/* is gitignored, so clone it "
        f"on this machine first: `git clone {mas.repo_url} {clone_path}` "
        "(or set prepare.clone_mas: true in the run config)."
    )


def load_evaluator_retry_result(
    previous: dict[str, Any] | None,
    sample_root: Path,
    *,
    require_resolved_evaluation: bool,
) -> dict[str, Any] | None:
    """Load a completed trajectory whose evaluator must be retried in place."""

    result_path = sample_root / "result.json"
    if (
        not require_resolved_evaluation
        or not isinstance(previous, dict)
        or previous.get("status") != "completed"
    ):
        return None
    if not result_path.is_file():
        raise FileNotFoundError(
            f"completed sample is missing evaluator-retry artifact: {result_path}"
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if not isinstance(result, dict) or result.get("status") != "completed":
        raise ValueError(f"invalid completed sample artifact for evaluator retry: {result_path}")
    return result


def run_batch_experiment(
    config: ExperimentConfig,
    client: ModelApiClient,
    experiment: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    mas_slug = str(experiment["mas"])
    benchmark_slug = str(experiment["benchmark"])
    run_id = str(experiment.get("name") or f"{index:02d}-{mas_slug}-{benchmark_slug}-batch")
    destination = resolve_run_destination(config, mas_slug, benchmark_slug, run_id)
    run_id = destination.run_id
    batch_root = destination.run_root
    experiment["name"] = run_id
    batch_root.mkdir(parents=True, exist_ok=True)
    logger = build_run_logger(batch_root, config.log_level, run_id)

    api = effective_api_config(config.api, experiment)
    runtime_client = client_for_api(client, api)
    mas = find_mas(mas_slug)
    benchmark = find_benchmark(benchmark_slug)
    native = as_bool(experiment.get("native"), False, name="experiment.native")
    if native:
        validate_native_mas_protocol(mas.slug, api.protocol)
        ensure_native_clone_present(config.external_dir, mas)

    source = str(experiment.get("sample_source", "full"))
    limit_raw = experiment.get("sample_limit")
    sample_limit = int(limit_raw) if limit_raw is not None else None
    offset = int(experiment.get("sample_offset", 0))
    split = str(experiment.get("sample_split", "validation"))
    sample_ids = (
        [str(value) for value in experiment["sample_ids"]]
        if experiment.get("sample_ids") is not None
        else None
    )
    samples = load_benchmark_items(
        config.data_dir,
        benchmark.slug,
        source=source,
        limit=sample_limit,
        offset=offset,
        split=split,
        sample_ids=sample_ids,
    )
    logger.info(
        "Starting batch experiment mas=%s benchmark=%s samples=%s source=%s offset=%s limit=%s",
        mas.slug,
        benchmark.slug,
        len(samples),
        source,
        offset,
        sample_limit,
    )

    manifest = {
        "run_id": run_id,
        "mas": mas.slug,
        "benchmark": benchmark.slug,
        "source": source,
        "split": split,
        "offset": offset,
        "limit": sample_limit,
        "sample_count": len(samples),
        "sample_ids": [str(sample.get("sample_id")) for sample in samples],
        "replicate_id": experiment.get("replicate_id"),
        "host_seed": api.seed,
        "model": api.model,
        "base_url": api.base_url,
        "provider": api.provider,
        "protocol": api.protocol,
        "endpoint_url": api.endpoint_url,
        "reasoning_effort": api.reasoning_effort,
        "upstream_provider": api.upstream_provider,
        "allow_fallbacks": api.allow_fallbacks,
        "evaluation": experiment.get("evaluation") or {},
        "runtime_id": api.runtime_id
        or f"runtime/{safe_run_id(mas.slug)}-{safe_run_id(benchmark.slug)}-{safe_run_id(api.model)}",
        "output_dir": str(batch_root),
    }
    manifest_path = batch_root / "batch_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("replicate_id") != manifest["replicate_id"]:
            raise ValueError("cannot resume a different replicate_id under the same run identity")
    atomic_write_json(manifest_path, manifest)

    results_path = batch_root / "results.jsonl"
    resume = as_bool(experiment.get("resume"), True, name="experiment.resume")
    require_resolved_evaluation = as_bool(
        experiment.get("resume_require_resolved_evaluation"),
        False,
        name="experiment.resume_require_resolved_evaluation",
    )
    if results_path.exists() and not resume:
        raise FileExistsError(
            f"batch results already exist at {results_path}; enable experiment.resume "
            "or choose overwrite/timestamp output policy"
        )
    latest_by_sample = load_latest_batch_results(results_path) if resume else {}
    completed_by_sample = (
        load_existing_batch_results(
            results_path,
            require_resolved_evaluation=require_resolved_evaluation,
        )
        if resume
        else {}
    )
    selected_sample_ids = {
        str(sample.get("sample_id") or f"sample-{position:05d}")
        for position, sample in enumerate(samples)
    }
    unexpected_completed = set(completed_by_sample) - selected_sample_ids
    if unexpected_completed:
        raise ValueError(
            f"resumed batch {run_id} changed its sample selection; completed IDs are "
            "absent from the current YAML: "
            + ", ".join(sorted(unexpected_completed)[:10])
        )
    checkpoint = BatchCheckpoint.open(
        batch_root / "run_state.json",
        run_id=run_id,
        total_samples=len(samples),
        completed_sample_ids=set(completed_by_sample),
    )
    stop_after_failures = int(experiment.get("stop_after_failures", 0) or 0)
    judge_enabled = as_bool(experiment.get("judge"), False, name="experiment.judge")
    consecutive_failures = 0
    results: list[dict[str, Any]] = []
    batch_started = time.monotonic()
    show_progress = as_bool(
        experiment.get("progress"), True, name="experiment.progress"
    ) and sys.stderr.isatty()

    try:
        with tqdm(
            total=len(samples),
            desc=f"{mas.slug}/{benchmark.slug}",
            unit="sample",
            dynamic_ncols=True,
            disable=not show_progress,
        ) as progress:
            for position, sample in enumerate(samples, start=1):
                sample_id = str(sample.get("sample_id") or f"sample-{position - 1:05d}")
                if sample_id in completed_by_sample:
                    existing = completed_by_sample[sample_id]
                    results.append(existing)
                    logger.info(
                        "Skipping completed sample %s/%s id=%s",
                        position,
                        len(samples),
                        sample_id,
                    )
                    progress.update(1)
                    continue

                sample_root = (batch_root / "samples" / safe_run_id(sample_id)).resolve()
                sample_run_id = f"{run_id}-{sample_id}"
                sample_experiment = {
                    **experiment,
                    "name": sample_run_id,
                    "batch_parent": run_id,
                    "sample_id": sample_id,
                    "sample_index": sample.get("sample_index"),
                }
                result = load_evaluator_retry_result(
                    latest_by_sample.get(sample_id),
                    sample_root,
                    require_resolved_evaluation=require_resolved_evaluation,
                )
                if result is not None:
                    checkpoint.start_sample(sample_id)
                    try:
                        result["evaluation"] = evaluate_sample(
                            benchmark_slug=benchmark.slug,
                            sample=sample,
                            result=result,
                            run_root=sample_root,
                            judge_client=runtime_client,
                            judge_enabled=judge_enabled,
                            evaluation_config=experiment.get("evaluation") or {},
                        )
                    except Exception as exc:
                        logger.exception("Evaluator retry failed id=%s", sample_id)
                        result["evaluation"] = {
                            "benchmark": benchmark.slug,
                            "score": None,
                            "correct": None,
                            "method": "evaluation_retry",
                            "reason": str(exc),
                        }
                    refresh_standard_run_artifacts(sample_root, result)
                    atomic_write_json(sample_root / "result.json", result)
                    result["sample_id"] = sample_id
                    result["sample_index"] = sample.get("sample_index")
                    result["batch_run_id"] = run_id
                    compact = compact_batch_result(result)
                    results.append(compact)
                    append_jsonl_fsync(results_path, compact)
                    checkpoint.finish_sample(
                        sample_id,
                        "completed"
                        if isinstance(result["evaluation"].get("correct"), bool)
                        else "evaluation_unresolved",
                        None,
                    )
                    elapsed_seconds = time.monotonic() - batch_started
                    progress.update(1)
                    logger.info(
                        "Retried evaluator %s/%s id=%s correct=%s elapsed=%s output=%s",
                        position,
                        len(samples),
                        sample_id,
                        result["evaluation"].get("correct"),
                        format_duration(elapsed_seconds),
                        sample_root,
                    )
                    continue

                checkpoint.start_sample(sample_id)
                sample_started = time.monotonic()
                try:
                    result = run_single_experiment(
                        config,
                        runtime_client,
                        sample_experiment,
                        index,
                        sample=sample,
                        run_root_override=sample_root,
                    )
                    result["evaluation"] = evaluate_sample(
                        benchmark_slug=benchmark.slug,
                        sample=sample,
                        result=result,
                        run_root=sample_root,
                        judge_client=runtime_client,
                        judge_enabled=judge_enabled,
                        evaluation_config=experiment.get("evaluation") or {},
                    )
                    refresh_standard_run_artifacts(sample_root, result)
                    atomic_write_json(sample_root / "result.json", result)
                    consecutive_failures = (
                        0
                        if result.get("status") == "completed"
                        else consecutive_failures + 1
                    )
                except Exception as exc:
                    logger.exception("Batch sample failed id=%s", sample_id)
                    result = write_failed_experiment(
                        config,
                        sample_experiment,
                        index,
                        exc,
                        run_root_override=sample_root,
                    )
                    result["evaluation"] = evaluate_sample(
                        benchmark_slug=benchmark.slug,
                        sample=sample,
                        result=result,
                        run_root=sample_root,
                        evaluation_config=experiment.get("evaluation") or {},
                    )
                    refresh_standard_run_artifacts(sample_root, result)
                    atomic_write_json(sample_root / "result.json", result)
                    consecutive_failures += 1
                finally:
                    close_run_logger(
                        logging.getLogger(f"masfm.run.{safe_run_id(sample_run_id)}")
                    )

                result["sample_id"] = sample_id
                result["sample_index"] = sample.get("sample_index")
                result["batch_run_id"] = run_id
                execution_timing = {
                    "sample_id": sample_id,
                    "wall_seconds": time.monotonic() - sample_started,
                    "scope": "native execution plus benchmark evaluation; excludes batch queue wait",
                }
                atomic_write_json(sample_root / "execution_timing.json", execution_timing)
                append_jsonl_fsync(batch_root / "execution_timings.jsonl", execution_timing)
                compact = compact_batch_result(result)
                results.append(compact)
                append_jsonl_fsync(results_path, compact)
                checkpoint.finish_sample(
                    sample_id,
                    str(result.get("status")),
                    str(result.get("error")) if result.get("error") else None,
                )
                elapsed_seconds = time.monotonic() - batch_started
                processed = len(results)
                remaining = max(len(samples) - processed, 0)
                avg_seconds = elapsed_seconds / processed if processed else 0.0
                eta_seconds = avg_seconds * remaining
                progress.set_postfix(
                    status=str(result.get("status")),
                    correct=str(result.get("evaluation", {}).get("correct")),
                )
                progress.update(1)
                logger.info(
                    "Finished sample %s/%s id=%s status=%s correct=%s elapsed=%s eta=%s avg_sample_seconds=%.2f output=%s",
                    position,
                    len(samples),
                    sample_id,
                    result.get("status"),
                    result.get("evaluation", {}).get("correct"),
                    format_duration(elapsed_seconds),
                    format_duration(eta_seconds),
                    avg_seconds,
                    sample_root,
                )
                if stop_after_failures and consecutive_failures >= stop_after_failures:
                    logger.error(
                        "Stopping batch after %s consecutive failures",
                        consecutive_failures,
                    )
                    break
    except KeyboardInterrupt:
        checkpoint.finish("interrupted")
        logger.warning("Batch interrupted; resume from %s", batch_root)
        raise
    except BaseException as exc:
        checkpoint.last_error = str(exc)
        checkpoint.finish("failed")
        raise

    summary = write_batch_summary(batch_root, results)
    terminal_samples = checkpoint.completed_sample_ids | checkpoint.failed_sample_ids
    batch_status = "completed" if len(terminal_samples) == len(samples) else "partial"
    checkpoint.finish(batch_status)
    batch_result = {
        **manifest,
        "status": batch_status,
        "completed_samples": len(results),
        "summary": summary,
        "results_path": str(results_path),
        "correct_trajectories_path": str(batch_root / "correct_trajectories.jsonl"),
        "error_trajectories_path": str(batch_root / "error_trajectories.jsonl"),
    }
    atomic_write_json(batch_root / "result.json", batch_result)
    logger.info("Finished batch experiment output=%s summary=%s", batch_root, summary)
    return batch_result


def run_single_experiment(
    config: ExperimentConfig,
    client: ModelApiClient,
    experiment: dict[str, Any],
    index: int,
    *,
    sample: dict[str, Any] | None = None,
    run_root_override: Path | None = None,
) -> dict[str, Any]:
    mas_slug = str(experiment["mas"])
    benchmark_slug = str(experiment["benchmark"])
    run_id = str(experiment.get("name") or f"{index:02d}-{mas_slug}-{benchmark_slug}")
    if run_root_override is None:
        destination = resolve_run_destination(config, mas_slug, benchmark_slug, run_id)
        run_id = destination.run_id
        run_root = destination.run_root
    else:
        run_root = run_root_override.resolve()
    experiment["name"] = run_id

    api = client.config if isinstance(client, ModelApiClient) else effective_api_config(config.api, experiment)
    mas = find_mas(mas_slug)
    benchmark = find_benchmark(benchmark_slug)
    if sample is None:
        sample = load_benchmark_sample(config.data_dir, benchmark.slug)
    clone_path = config.external_dir / "mas" / mas.clone_dir
    adapter = get_host_adapter(mas.slug)
    bridge_spec = build_bridge_spec(config.external_dir, mas)
    run_root.mkdir(parents=True, exist_ok=True)
    write_sample_run_state(
        run_root,
        run_id=run_id,
        status="running",
        mas_slug=mas.slug,
        benchmark_slug=benchmark.slug,
        sample_id=str(sample.get("sample_id")) if sample.get("sample_id") else None,
    )
    workspace_path = resolve_workspace_path(
        config,
        mas.slug,
        benchmark.slug,
        run_id,
        str(sample.get("sample_id")) if sample.get("sample_id") else None,
    )
    prepare_sample_workspace(
        workspace_path,
        reset=experiment.get("batch_parent") is not None,
    )
    aria = AriaRuntime(config.aria) if config.aria.enabled else None
    if aria is not None:
        aria.memory.contract_memory["incident_id"] = run_id
        aria.memory.contract_memory["model_audit_path"] = str(run_root / "aria_model_interventions.jsonl")
    if aria is None:
        clear_stale_aria_artifacts(run_root)

    logger = build_run_logger(run_root, config.log_level, run_id)
    logger.info("Starting experiment mas=%s benchmark=%s", mas.slug, benchmark.slug)
    logger.debug(
        "Sample id=%s index=%s trace_id=%s task=%s",
        sample.get("sample_id"),
        sample.get("sample_index"),
        sample.get("trace_id"),
        sample.get("task"),
    )

    task = str(sample["task"])
    native = as_bool(experiment.get("native"), False, name="experiment.native")
    if native:
        validate_native_mas_protocol(mas.slug, api.protocol)
        ensure_native_clone_present(config.external_dir, mas)
        result = run_native_experiment(
            api=api,
            experiment=experiment,
            mas=mas,
            benchmark=benchmark,
            sample=sample,
            clone_path=clone_path,
            run_root=run_root,
            aria=aria,
            adapter=adapter,
            bridge_spec=bridge_spec,
            logger=logger,
            workspace_path=workspace_path,
        )
        write_sample_run_state(
            run_root,
            run_id=run_id,
            status=str(result.get("status")),
            mas_slug=mas.slug,
            benchmark_slug=benchmark.slug,
            sample_id=str(sample.get("sample_id")) if sample.get("sample_id") else None,
            error=str(result.get("error")) if result.get("error") else None,
        )
        logger.info("Finished experiment mas=%s benchmark=%s output=%s", mas.slug, benchmark.slug, run_root)
        return result

    max_agent_turns = int(experiment.get("max_agent_turns", len(mas.roles)))
    transcript = []
    shared_context = ""
    if aria is not None:
        aria.memory.contract_memory["mode"] = config.aria.mode
        adapter.bootstrap(aria, mas, benchmark, task, list(mas.roles[:max_agent_turns]))
    for turn, role in enumerate(mas.roles[:max_agent_turns], start=1):
        action = adapter.build_action(mas, benchmark, role, task, shared_context, turn)
        patch_record = None
        if aria is not None:
            action, patch_record = adapter.before_action(aria, action)
            if patch_record is not None:
                logger.info("ARIA patch injected role=%s turn=%s type=%s", role, turn, patch_record["patch"]["patch_type"])
        logger.debug("Calling role=%s turn=%s prompt=%s", role, turn, action.prompt)
        request_messages = [
            {"role": "system", "content": action.system_message},
            {"role": "user", "content": action.prompt},
        ]
        model_response = invoke_model(client, request_messages)
        content = model_response.text
        detection_record = None
        if aria is not None:
            detection_record = adapter.after_action(aria, action, content)
            if detection_record is not None:
                logger.info(
                    "ARIA detected failure role=%s turn=%s modes=%s level=%s",
                    role,
                    turn,
                    ",".join(detection_record["report"]["failure_modes"]),
                    detection_record["policy"]["level"],
                )
        transcript.append(
            {
                "turn": turn,
                "role": role,
                "content": content,
                "model_call": model_response.to_record(),
            }
        )
        append_jsonl_fsync(
            run_root / "model_calls.jsonl",
            {
                "turn": turn,
                "role": role,
                "request": request_messages,
                "response": {
                    **model_response.to_record(),
                    "text": content,
                },
            },
        )
        atomic_write_text(run_root / f"{turn:02d}_{safe_run_id(role)}.txt", content + "\n")
        shared_context += f"\n\n[{role}]\n{content}"
        logger.info("Completed role=%s turn=%s chars=%s", role, turn, len(content))

    finalize_patch = None
    if aria is not None and transcript:
        finalize_patch = adapter.before_finalize(aria, mas, benchmark, transcript)

    result = {
        "run_id": run_id,
        "mas": mas.slug,
        "mas_name": mas.paper_name,
        "benchmark": benchmark.slug,
        "benchmark_name": benchmark.paper_name,
        "sample_id": sample.get("sample_id"),
        "sample_index": sample.get("sample_index"),
        "sample_trace_id": sample.get("trace_id"),
        "sample_metadata": sample_metadata(sample),
        "model": api.model,
        "base_url": api.base_url,
        "provider": api.provider,
        "protocol": api.protocol,
        "endpoint_url": api.endpoint_url,
        "reasoning_effort": api.reasoning_effort,
        "upstream_provider": api.upstream_provider,
        "allow_fallbacks": api.allow_fallbacks,
        "runtime_id": api.runtime_id
        or f"runtime/{safe_run_id(mas.slug)}-{safe_run_id(benchmark.slug)}-{safe_run_id(api.model)}",
        "model_id": api.model_id or f"model/{safe_run_id(api.model)}",
        "api_key_env": api.api_key_env,
        "api_key_required": api.api_key_required,
        "native_repo_path": str(clone_path),
        "native_repo_head": git_head(clone_path),
        "bridge": bridge_spec.to_dict(),
        "roles": list(mas.roles[:max_agent_turns]),
        "status": "completed",
        "output_dir": str(run_root),
        "workspace_path": str(workspace_path),
        "final_answer": transcript[-1]["content"] if transcript else "",
        "transcript": transcript,
        "aria": aria.export_state() if aria is not None else {"enabled": False},
        "aria_finalize_record": finalize_patch,
    }
    write_standard_run_artifacts(
        run_root=run_root,
        result=result,
        messages=messages_from_transcript(transcript),
        trace_events=trajectory_from_transcript(transcript, phase="source"),
        evaluation=result.get("evaluation"),
        runtime_state=build_runtime_state(result, api),
    )
    atomic_write_json(run_root / "result.json", result)
    write_sample_run_state(
        run_root,
        run_id=run_id,
        status="completed",
        mas_slug=mas.slug,
        benchmark_slug=benchmark.slug,
        sample_id=str(sample.get("sample_id")) if sample.get("sample_id") else None,
    )
    logger.info("Finished experiment mas=%s benchmark=%s output=%s", mas.slug, benchmark.slug, run_root)
    return result


def run_native_experiment(
    *,
    api: ApiConfig,
    experiment: dict[str, Any],
    mas: Any,
    benchmark: Any,
    sample: dict[str, Any],
    clone_path: Path,
    run_root: Path,
    aria: AriaRuntime | None,
    adapter: Any,
    bridge_spec: Any,
    logger: logging.Logger,
    workspace_path: Path,
) -> dict[str, Any]:
    native_backend = get_native_mas_backend(mas.slug)
    if benchmark.slug == "swe-bench-lite" and experiment.get("swebench_workspace_mode"):
        sample = {**sample, "_swebench_workspace_mode": experiment["swebench_workspace_mode"]}
    logger.info("Launching %s", native_backend.launch_description)
    native = native_backend.run(
        NativeRunContext(
            repo_root=Path(__file__).resolve().parents[2],
            clone_path=clone_path,
            run_root=run_root,
            workspace_path=workspace_path,
            sample=sample,
            mas=mas,
            benchmark=benchmark,
            host_adapter=adapter,
            aria=aria,
            options=experiment,
            api_settings=api_settings(api),
            default_timeout_seconds=api.timeout_seconds,
        )
    )
    result = {
        "run_id": experiment.get("name") or f"{mas.slug}-{benchmark.slug}-native",
        "mas": mas.slug,
        "mas_name": mas.paper_name,
        "benchmark": benchmark.slug,
        "benchmark_name": benchmark.paper_name,
        "sample_id": sample.get("sample_id"),
        "sample_index": sample.get("sample_index"),
        "sample_trace_id": sample.get("trace_id"),
        "sample_metadata": sample_metadata(sample),
        "model": api.model,
        "base_url": api.base_url,
        "provider": api.provider,
        "protocol": api.protocol,
        "endpoint_url": api.endpoint_url,
        "reasoning_effort": api.reasoning_effort,
        "upstream_provider": api.upstream_provider,
        "allow_fallbacks": api.allow_fallbacks,
        "runtime_id": api.runtime_id
        or f"runtime/{safe_run_id(mas.slug)}-{safe_run_id(benchmark.slug)}-{safe_run_id(api.model)}",
        "model_id": api.model_id or f"model/{safe_run_id(api.model)}",
        "api_key_env": api.api_key_env,
        "api_key_required": api.api_key_required,
        "native_repo_path": str(clone_path),
        "native_repo_head": git_head(clone_path),
        "bridge": bridge_spec.to_dict(),
        "roles": list(mas.roles),
        "status": native.status,
        "output_dir": str(run_root),
        "workspace_path": str(workspace_path),
        "transcript": native.transcript,
        "final_output": native.final_output,
        "native_messages_path": str(run_root / native_backend.messages_filename),
        "native_trace_path": str(run_root / native_backend.trace_filename),
        "aria": native.aria_export if native.aria_export is not None else {"enabled": False},
        "error": native.error,
        "failure_category": classify_infrastructure_failure(native.error),
    }
    write_standard_run_artifacts(
        run_root=run_root,
        result=result,
        messages=load_json_if_exists(run_root / native_backend.messages_filename, []),
        trace_events=load_jsonl_if_exists(run_root / native_backend.trace_filename),
        evaluation=result.get("evaluation"),
        runtime_state=build_runtime_state(result, api),
    )
    atomic_write_json(run_root / "result.json", result)
    return result


def write_failed_experiment(
    config: ExperimentConfig,
    experiment: dict[str, Any],
    index: int,
    exc: Exception,
    *,
    run_root_override: Path | None = None,
) -> dict[str, Any]:
    mas_slug = str(experiment.get("mas", "unknown-mas"))
    benchmark_slug = str(experiment.get("benchmark", "unknown-benchmark"))
    api = effective_api_config(config.api, experiment)
    run_id = str(experiment.get("name") or f"{index:02d}-{mas_slug}-{benchmark_slug}")
    run_root = (
        run_root_override
        or (job_output_root(config) / mas_slug / benchmark_slug / safe_run_id(run_id)).resolve()
    )
    run_root.mkdir(parents=True, exist_ok=True)
    workspace_path = resolve_workspace_path(
        config,
        mas_slug,
        benchmark_slug,
        run_id,
        str(experiment.get("sample_id")) if experiment.get("sample_id") else None,
    )
    workspace_path.mkdir(parents=True, exist_ok=True)
    result = {
        "run_id": run_id,
        "mas": mas_slug,
        "benchmark": benchmark_slug,
        "model": api.model,
        "base_url": api.base_url,
        "provider": api.provider,
        "protocol": api.protocol,
        "endpoint_url": api.endpoint_url,
        "reasoning_effort": api.reasoning_effort,
        "runtime_id": api.runtime_id
        or f"runtime/{safe_run_id(mas_slug)}-{safe_run_id(benchmark_slug)}-{safe_run_id(api.model)}",
        "model_id": api.model_id or f"model/{safe_run_id(api.model)}",
        "api_key_env": api.api_key_env,
        "api_key_required": api.api_key_required,
        "status": "failed",
        "error": str(exc),
        "failure_category": classify_infrastructure_failure(exc),
        "output_dir": str(run_root),
        "workspace_path": str(workspace_path),
        "sample_id": experiment.get("sample_id"),
        "sample_index": experiment.get("sample_index"),
    }
    write_standard_run_artifacts(
        run_root=run_root,
        result=result,
        messages=[],
        trace_events=[
            {
                "event_id": "source:event:0001",
                "turn": 1,
                "mas_agent_id": "runtime",
                "mas_agent_role": "runtime",
                "kind": "run_failed",
                "hook": "run_exception",
                "content": str(exc),
                "tool_calls": [],
                "tool_results": [],
                "native_event": {"source": "masfm", "native_index": 0, "error_type": type(exc).__name__},
                "detector": None,
            }
        ],
        evaluation=None,
        runtime_state=build_runtime_state(result, api),
    )
    atomic_write_json(run_root / "result.json", result)
    write_sample_run_state(
        run_root,
        run_id=run_id,
        status="failed",
        mas_slug=mas_slug,
        benchmark_slug=benchmark_slug,
        sample_id=str(experiment.get("sample_id")) if experiment.get("sample_id") else None,
        error=str(exc),
    )
    return result


def build_run_logger(run_root: Path, level: str, name: str) -> logging.Logger:
    logger = logging.getLogger(f"masfm.run.{safe_run_id(name)}")
    close_run_logger(logger)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(run_root / "run.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(file_handler)
    logger.propagate = True
    if level != "DEBUG":
        logger.setLevel(logging.INFO)
    return logger


def close_run_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
