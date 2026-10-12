"""Strict YAML parser for MASFM inference, repair, and ARIA jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from aria.model import AriaModelConfig
from aria.runtime import AriaConfig
from common.config import (
    ExperimentConfig,
    OutputConfig,
    as_bool,
    normalize_existing_output_policy,
    required_keys,
)
from common.model_api import ApiConfig


class StrictSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: StrictSafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    seen: set[Any] = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=False)
        if key in seen:
            raise ValueError(f"duplicate YAML key: {key}")
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)

TOP_LEVEL_KEYS = {
    "job",
    "runtime",
    "data_dir",
    "external_dir",
    "output_dir",
    "workspace_dir",
    "api",
    "vllm",
    "logging",
    "prepare",
    "output",
    "aria",
    "experiments",
    "repair_datasets",
    "baselines",
    "baseline_workers",
}
API_KEYS = {
    "provider",
    "protocol",
    "base_url",
    "model",
    "api_key_env",
    "api_key_required",
    "timeout_seconds",
    "max_tokens",
    "context_window_tokens",
    "temperature",
    "reasoning_effort",
    "retries",
    "retry_seconds",
    "runtime_id",
    "model_id",
    "seed",
    "anthropic_version",
    "upstream_provider",
    "allow_fallbacks",
}
EVALUATION_KEYS = {
    "harness",
    "timeout_seconds",
    "process_timeout_seconds",
    "max_workers",
    "cache_level",
    "clean",
    "namespace",
    "docker_platform",
    "use_host_docker_credentials",
}
EXPERIMENT_KEYS = {
    "command_rejection_policy",
    "context_budget_mode",
    "swebench_workspace_mode",
    "replicate_id",
    "name",
    "name_template",
    "mas",
    "benchmark",
    "native",
    "batch",
    "sample_source",
    "sample_limit",
    "sample_offset",
    "sample_split",
    "sample_ids",
    "resume",
    "resume_require_resolved_evaluation",
    "progress",
    "judge",
    "evaluation",
    "max_steps",
    "max_agent_turns",
    "max_turns",
    "max_stalls",
    "native_timeout_seconds",
    "model_vision",
    "code_execution_environment",
    "stop_after_failures",
    "matrix",
    *API_KEYS,
}
POST_TRAINING_KEYS = {
    "repair_conditioning",
    "evaluation_holdout_paths",
    "base_model",
    "source_path",
    "output_dir",
    "environment_path",
    "environment",
    "feedback_dropout_probability",
    "diagnosis_uplift_threshold",
    "random_seed",
    "resume",
    "stages",
    "ablation_variant",
    "gate_validation_path",
    "gate_min_positive_rate",
    "gate_max_positive_rate",
    "backend",
}
STAGE3_ENVIRONMENT_KINDS = {"evaluator", "ledger", "native", "native_by_cell", "chain"}
POST_TRAINING_BACKEND_KEYS = {"type", "command"}
ARIA_MODEL_KEYS = {
    "system_role_mode",
    "tokenizer_endpoint",
    "runtime_context",
    "structured_output_retries",
    "seed",
    "gate_probability_source",
    "max_input_tokens",
    "gate_calibration_path",
    "provider",
    "model",
    "base_url",
    "api_key_env",
    "api_key_required",
    "timeout_seconds",
    "max_tokens",
    "temperature",
}
ARIA_PASSIVE_KEYS = {"input_path", "output_path"}
REPAIR_DATASET_KEYS = {
    "name",
    "schema_version",
    "source_batch_dir",
    "dataset_dir",
    "repair_output_dir",
    "split",
    "rerun",
    "native",
    "resume",
    "skip_sample_ids",
    "on_sample_error",
    "retry_invalid",
    "max_consecutive_endpoint_invalid",
    "capacity_wait_seconds",
    "capacity_wait_max_attempts",
    "max_steps",
    "max_turns",
    "max_stalls",
    "native_timeout_seconds",
    "repair_trials",
    "structured_output_retries",
    "attribution_max_events",
    "sample_limit",
    "sample_ids",
    "gate_success_sample_rate",
    "gate_success_sample_limit",
    "require_inference_completed",
    "require_trace_acceptance",
    "teacher_model_id",
    "repair_operator_id",
    "config_id",
    "sample_split",
    "natural_or_synthetic",
    "annotation_status",
    "audit_status",
    "evaluation",
}
BASELINE_KEYS = {
    "reevaluate",
    "reexecute",
    "rescore",
    "local_model",
    "benchmark_filter",
    "name",
    "method",
    "input_path",
    "input_format",
    "dataset_root",
    "max_events",
    "output_dir",
    "sample_limit",
    "sample_workers",
    "prompt_char_budget",
    "allow_legacy_resume",
    "split",
    "resume",
    "on_sample_error",
    "method_options",
    "model_identity",
    "replay_backend",
}
BASELINE_INPUT_FORMATS = {
    "auto",
    "aria-record",
    "aria-v6",
    "aria-incident",
    "aria-full",
    "stage3-environment",
    "inference-results",
    "generic-jsonl",
}
BASELINE_REPLAY_TYPES = {"none", "ledger", "native"}


def load_config(path: Path) -> ExperimentConfig:
    raw = yaml.load(path.read_text(), Loader=StrictSafeLoader)
    if not isinstance(raw, dict):
        raise TypeError(f"config root must be a mapping: {path}")
    reject_unknown_keys(raw, TOP_LEVEL_KEYS, "config")
    job_type = resolve_job_type(raw)
    runtime_kind = resolve_runtime_kind(raw)
    api = parse_api_config(
        raw, required=job_type not in {"aria-training", "aria-passive", "baselines"}
    )
    logging_raw = mapping_section(raw, "logging")
    reject_unknown_keys(logging_raw, {"level"}, "logging")
    log_level = str(logging_raw.get("level", "INFO")).upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ValueError(f"unsupported logging.level: {log_level}")

    output_raw = mapping_section(raw, "output")
    reject_unknown_keys(output_raw, {"existing_output_policy"}, "output")
    output = OutputConfig(
        existing_output_policy=normalize_existing_output_policy(
            output_raw.get("existing_output_policy")
        )
    )
    aria = parse_aria_config(mapping_section(raw, "aria"))
    experiments = mapping_list(raw.get("experiments", []), "experiments")
    repair_datasets = mapping_list(raw.get("repair_datasets", []), "repair_datasets")
    baselines = mapping_list(raw.get("baselines", []), "baselines")
    baseline_workers = raw.get("baseline_workers", 1)
    if (
        not isinstance(baseline_workers, int)
        or isinstance(baseline_workers, bool)
        or baseline_workers < 1
    ):
        raise TypeError("baseline_workers must be a positive integer")
    for index, experiment in enumerate(experiments):
        reject_unknown_keys(experiment, EXPERIMENT_KEYS, f"experiments[{index}]")
        validate_experiment(experiment, index)
    for index, repair_spec in enumerate(repair_datasets):
        reject_unknown_keys(
            repair_spec, REPAIR_DATASET_KEYS, f"repair_datasets[{index}]"
        )
        validate_repair_spec(repair_spec, index)
    for index, baseline_spec in enumerate(baselines):
        reject_unknown_keys(baseline_spec, BASELINE_KEYS, f"baselines[{index}]")
        validate_baseline_spec(baseline_spec, index)
        backend = baseline_spec.get("replay_backend")
        if (
            api is None
            and isinstance(backend, dict)
            and backend.get("type") == "native"
        ):
            host_api = (backend.get("options") or {}).get("host_api")
            if not isinstance(host_api, dict):
                raise ValueError(
                    f"baselines[{index}].replay_backend.type: {backend['type']} re-runs the "
                    "host system and requires top-level api configuration or an explicit "
                    "replay_backend.options.host_api"
                )
            required_keys(host_api, {
                "provider", "protocol", "base_url", "model", "api_key_required",
            }, f"baselines[{index}].replay_backend.options.host_api")
            parse_api_config({"api": host_api}, required=True)

    prepare = mapping_section(raw, "prepare")
    reject_unknown_keys(
        prepare,
        {"clone_mas", "download_benchmarks", "gaia_attachments"},
        "prepare",
    )
    for key, value in prepare.items():
        as_bool(value, False, name=f"prepare.{key}")
    if job_type == "aria-training" and runtime_kind != "training":
        raise ValueError("aria-training job requires runtime.kind: training")
    if job_type != "aria-training" and runtime_kind == "training":
        raise ValueError("runtime.kind: training is only valid for job: aria-training")
    validate_job_contents(job_type, experiments, repair_datasets, baselines, aria)
    return ExperimentConfig(
        data_dir=Path(raw.get("data_dir", "data")),
        external_dir=Path(raw.get("external_dir", "external")),
        output_dir=Path(raw.get("output_dir", "outputs")),
        workspace_dir=Path(raw.get("workspace_dir", "workspace")),
        log_level=log_level,
        api=api,
        aria=aria,
        output=output,
        experiments=experiments,
        prepare=prepare,
        repair_datasets=repair_datasets,
        baselines=baselines,
        baseline_workers=baseline_workers,
        config_path=path,
        job_type=job_type,
        runtime_kind=runtime_kind,
    )


def parse_api_config(raw: dict[str, Any], *, required: bool) -> ApiConfig | None:
    api_raw = mapping_section(raw, "api", required=required)
    if not api_raw:
        return None
    reject_unknown_keys(api_raw, API_KEYS, "api")
    required_keys(api_raw, {"provider", "protocol", "base_url", "model"}, "api")
    api_key_required = as_bool(api_raw.get("api_key_required"), True)
    if api_key_required:
        required_keys(api_raw, {"api_key_env"}, "api")
    temperature_raw = api_raw.get("temperature")
    return ApiConfig(
        provider=str(api_raw["provider"]),
        protocol=str(api_raw["protocol"]),
        base_url=str(api_raw["base_url"]),
        model=str(api_raw["model"]),
        api_key_env=(
            str(api_raw["api_key_env"])
            if api_raw.get("api_key_env") is not None
            else None
        ),
        api_key_required=api_key_required,
        timeout_seconds=int(api_raw.get("timeout_seconds", 120)),
        max_tokens=int(api_raw.get("max_tokens", 512)),
        context_window_tokens=(
            int(api_raw["context_window_tokens"])
            if api_raw.get("context_window_tokens") is not None
            else None
        ),
        temperature=float(temperature_raw) if temperature_raw is not None else None,
        reasoning_effort=(
            str(api_raw["reasoning_effort"])
            if api_raw.get("reasoning_effort") is not None
            else None
        ),
        retries=int(api_raw.get("retries", 0)),
        retry_seconds=float(api_raw.get("retry_seconds", 5.0)),
        runtime_id=api_raw.get("runtime_id"),
        model_id=api_raw.get("model_id"),
        seed=int(api_raw["seed"]) if api_raw.get("seed") is not None else None,
        anthropic_version=str(api_raw.get("anthropic_version", "2023-06-01")),
        upstream_provider=(
            str(api_raw["upstream_provider"])
            if api_raw.get("upstream_provider") is not None
            else None
        ),
        allow_fallbacks=(
            as_bool(api_raw["allow_fallbacks"], True, name="api.allow_fallbacks")
            if api_raw.get("allow_fallbacks") is not None
            else None
        ),
    )


def parse_aria_config(raw: dict[str, Any]) -> AriaConfig:
    reject_unknown_keys(
        raw,
        {
            "enabled",
            "mode",
            "max_history_events",
            "max_pending_patches",
            "repeat_threshold",
            "verification_keywords",
            "finalization_keywords",
            "post_training",
            "model",
            "gate_threshold",
            "fail_open",
            "runtime_policy",
            "runtime_policy_options",
            "passive",
        },
        "aria",
    )
    mode = str(raw.get("mode", "shadow"))
    if mode not in {"off", "shadow", "active", "passive"}:
        raise ValueError(f"unsupported aria.mode: {mode}")
    post_training = parse_post_training_config(raw.get("post_training"))
    model = parse_aria_model_config(raw.get("model"))
    runtime_policy = str(raw.get("runtime_policy", "aria"))
    if runtime_policy not in {"aria", "spark_to_fire", "safeagent"}:
        raise ValueError("unsupported aria.runtime_policy")
    policy_options = raw.get("runtime_policy_options")
    if policy_options is not None and not isinstance(policy_options, dict):
        raise TypeError("aria.runtime_policy_options must be a mapping")
    if runtime_policy != "aria" and model is None:
        raise ValueError("a baseline runtime policy requires aria.model endpoint settings")
    if runtime_policy != "aria" and model.provider != "openai-chat-completions":
        raise ValueError("native baseline runtime policies require an endpoint model")
    permitted_policy_options = {"policy", "risk_threshold", "max_retries"} if runtime_policy == "spark_to_fire" else set()
    reject_unknown_keys(policy_options or {}, permitted_policy_options, "aria.runtime_policy_options")
    passive = parse_passive_config(raw.get("passive"))
    gate_threshold = float(raw.get("gate_threshold", 0.5))
    if not 0.0 <= gate_threshold <= 1.0:
        raise ValueError("aria.gate_threshold must be between 0 and 1")
    return AriaConfig(
        enabled=as_bool(raw.get("enabled"), False),
        mode=mode,
        max_history_events=int(raw.get("max_history_events", 24)),
        max_pending_patches=int(raw.get("max_pending_patches", 4)),
        repeat_threshold=int(raw.get("repeat_threshold", 2)),
        verification_keywords=tuple(
            raw.get("verification_keywords", ("verify", "test", "check", "assert"))
        ),
        finalization_keywords=tuple(
            raw.get("finalization_keywords", ("final", "answer", "complete", "done"))
        ),
        post_training=post_training,
        model=model,
        gate_threshold=gate_threshold,
        fail_open=as_bool(raw.get("fail_open"), True, name="aria.fail_open"),
        runtime_policy=runtime_policy,
        runtime_policy_options=dict(policy_options or {}),
        passive=passive,
    )


def parse_aria_model_config(value: Any) -> AriaModelConfig | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("aria.model must be a mapping")
    reject_unknown_keys(value, ARIA_MODEL_KEYS, "aria.model")
    required_keys(value, {"provider", "model"}, "aria.model")
    config = AriaModelConfig(
        tokenizer_endpoint=value.get("tokenizer_endpoint"),
        runtime_context=as_bool(value.get("runtime_context"), False, name="aria.model.runtime_context"),
        structured_output_retries=value.get("structured_output_retries", 0),
        seed=value.get("seed"),
        gate_calibration_path=value.get("gate_calibration_path"),
        gate_probability_source=str(value.get("gate_probability_source", "generated")),
        system_role_mode=str(value.get("system_role_mode", "native")),
        max_input_tokens=int(value["max_input_tokens"]) if value.get("max_input_tokens") is not None else None,
        provider=str(value["provider"]),
        model=str(value["model"]),
        base_url=str(value["base_url"]) if value.get("base_url") is not None else None,
        api_key_env=(
            str(value["api_key_env"]) if value.get("api_key_env") is not None else None
        ),
        api_key_required=as_bool(
            value.get("api_key_required"), True, name="aria.model.api_key_required"
        ),
        timeout_seconds=int(value.get("timeout_seconds", 120)),
        max_tokens=int(value.get("max_tokens", 2048)),
        temperature=(
            float(value.get("temperature", 0.0))
            if value.get("temperature", 0.0) is not None
            else None
        ),
    )
    config.validate()
    return config


def parse_passive_config(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("aria.passive must be a mapping")
    reject_unknown_keys(value, ARIA_PASSIVE_KEYS, "aria.passive")
    required_keys(value, ARIA_PASSIVE_KEYS, "aria.passive")
    for key in ARIA_PASSIVE_KEYS:
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"aria.passive.{key} must be a non-empty string")
    return dict(value)


def parse_post_training_config(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("aria.post_training must be a mapping")
    reject_unknown_keys(value, POST_TRAINING_KEYS, "aria.post_training")
    if value.get("repair_conditioning", "incident") not in {"incident", "cascade"}:
        raise ValueError("aria.post_training.repair_conditioning must be incident or cascade")
    required_keys(
        value,
        {"base_model", "source_path", "output_dir", "backend"},
        "aria.post_training",
    )
    for key in ("base_model", "source_path", "output_dir"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"aria.post_training.{key} must be a non-empty string")
    backend = value["backend"]
    if not isinstance(backend, dict):
        raise TypeError("aria.post_training.backend must be a mapping")
    reject_unknown_keys(
        backend, POST_TRAINING_BACKEND_KEYS, "aria.post_training.backend"
    )
    required_keys(backend, {"type"}, "aria.post_training.backend")
    backend_type = backend["type"]
    if backend_type not in {"artifact-only", "command"}:
        raise ValueError(
            "aria.post_training.backend.type must be artifact-only or command"
        )
    if backend_type == "command":
        command = backend.get("command")
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(item, str) for item in command)
        ):
            raise TypeError(
                "aria.post_training.backend.command must be a non-empty list of strings"
            )
    elif "command" in backend:
        raise ValueError(
            "aria.post_training.backend.command is only valid for type: command"
        )
    for key in (
        "feedback_dropout_probability",
        "diagnosis_uplift_threshold",
        "gate_min_positive_rate",
        "gate_max_positive_rate",
    ):
        if key in value and not isinstance(value[key], (int, float)):
            raise TypeError(f"aria.post_training.{key} must be numeric")
    if (
        "feedback_dropout_probability" in value
        and not 0 <= float(value["feedback_dropout_probability"]) <= 1
    ):
        raise ValueError(
            "aria.post_training.feedback_dropout_probability must be between 0 and 1"
        )
    for key in ("gate_min_positive_rate", "gate_max_positive_rate"):
        if key in value and not 0 < float(value[key]) < 1:
            raise ValueError(f"aria.post_training.{key} must be between 0 and 1")
    if (
        "gate_min_positive_rate" in value
        and "gate_max_positive_rate" in value
        and float(value["gate_min_positive_rate"])
        > float(value["gate_max_positive_rate"])
    ):
        raise ValueError(
            "aria.post_training.gate_min_positive_rate must not exceed "
            "gate_max_positive_rate"
        )
    if "random_seed" in value and (
        not isinstance(value["random_seed"], int)
        or isinstance(value["random_seed"], bool)
    ):
        raise TypeError("aria.post_training.random_seed must be an integer")
    if "stages" in value and "ablation_variant" in value:
        raise ValueError(
            "give aria.post_training either stages or ablation_variant, not both"
        )
    if "ablation_variant" in value:
        variant = value["ablation_variant"]
        expected_variants = {"stage1_only", "stage1_2", "stage1_3", "full"}
        if not isinstance(variant, str) or variant not in expected_variants:
            raise ValueError(
                "aria.post_training.ablation_variant must be one of "
                + ", ".join(sorted(expected_variants))
            )
    if "stages" in value:
        stages = value["stages"]
        if not isinstance(stages, list) or not stages:
            raise TypeError("aria.post_training.stages must be a non-empty list")
        if not all(isinstance(stage, str) for stage in stages):
            raise TypeError("aria.post_training.stages entries must be strings")
        expected = ["sft", "dpo", "transfer_diagnosis", "grpo_repair", "grpo_gate"]
        if len(stages) != len(set(stages)) or any(
            stage not in expected for stage in stages
        ):
            raise ValueError(
                "aria.post_training.stages entries must be unique values drawn "
                "from " + ", ".join(expected)
            )
        if stages != [stage for stage in expected if stage in stages]:
            raise ValueError(
                "aria.post_training.stages must follow the order "
                + ", ".join(expected)
            )
        if "sft" not in stages and not str(value.get("base_model", "")).strip().startswith(
            ("outputs/", "../", "/")
        ):
            raise ValueError(
                "aria.post_training.stages must include sft, or base_model must "
                "point at an ARIA checkpoint that ran it; the record schema and "
                "the gate are introduced there"
            )
        if ("grpo_gate" in stages and "grpo_repair" not in stages
                and not str(value.get("base_model", "")).endswith("/grpo_repair/checkpoint")):
            raise ValueError(
                "aria.post_training.stages: grpo_gate requires grpo_repair; the "
                "gate is optimized after the repair policy"
            )
    stage_names = value.get("stages") or (
        ["sft", "dpo", "transfer_diagnosis", "grpo_repair", "grpo_gate"]
        if value.get("ablation_variant") in {None, "full"}
        else {
            "stage1_only": ["sft"],
            "stage1_2": ["sft", "dpo", "transfer_diagnosis"],
            "stage1_3": ["sft", "grpo_repair", "grpo_gate"],
        }[str(value["ablation_variant"])]
    )
    if any(name.startswith("grpo_") for name in stage_names):
        environment_path = value.get("environment_path")
        if not isinstance(environment_path, str) or not environment_path.strip():
            raise ValueError(
                "aria.post_training.environment_path is required whenever a "
                "grpo stage runs; Stage 3 scores its samples by re-executing "
                "them in the environment retained from corpus construction"
            )
    if "environment" in value:
        environment = value["environment"]
        if not isinstance(environment, dict):
            raise TypeError("aria.post_training.environment must be a mapping")
        kind = environment.get("kind", "evaluator")
        if kind not in STAGE3_ENVIRONMENT_KINDS:
            raise ValueError(
                "aria.post_training.environment.kind must be one of "
                + ", ".join(sorted(STAGE3_ENVIRONMENT_KINDS))
            )
    if "environment_path" in value and (
        not isinstance(value["environment_path"], str)
        or not value["environment_path"].strip()
    ):
        raise ValueError("aria.post_training.environment_path must be a non-empty string")
    if "resume" in value:
        as_bool(value["resume"], True, name="aria.post_training.resume")
    return dict(value)


def resolve_runtime_kind(raw: dict[str, Any]) -> str:
    runtime_raw = mapping_section(raw, "runtime", required=True)
    reject_unknown_keys(runtime_raw, {"kind"}, "runtime")
    required_keys(runtime_raw, {"kind"}, "runtime")
    value = str(runtime_raw["kind"])
    if value not in {"endpoint", "vllm", "training"}:
        raise ValueError(f"unsupported runtime.kind: {value}")
    if value == "vllm":
        vllm_raw = mapping_section(raw, "vllm", required=True)
        reject_unknown_keys(
            vllm_raw,
            {
                "host",
                "port",
                "hf_model_name",
                "served_model_name",
                "startup_timeout_seconds",
                "poll_interval_seconds",
                "tensor_parallel_size",
                "gpu_memory_utilization",
                "cuda_visible_devices",
                "dtype",
                "max_model_len",
                "extra_args",
            },
            "vllm",
        )
    return value


def resolve_job_type(raw: dict[str, Any]) -> str:
    if "job" not in raw:
        raise KeyError("config requires job")
    value = raw["job"]
    if not isinstance(value, str):
        raise TypeError("job must be a string")
    if value not in {
        "inference",
        "repair",
        "aria",
        "aria-training",
        "aria-passive",
        "mixed",
        "baselines",
    }:
        raise ValueError(f"unsupported job type: {value}")
    return value


def mapping_section(
    raw: dict[str, Any],
    key: str,
    *,
    required: bool = False,
) -> dict[str, Any]:
    if key not in raw:
        if required:
            raise KeyError(f"config requires {key}")
        return {}
    value = raw[key]
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a mapping")
    return dict(value)


def mapping_list(value: Any, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise TypeError(f"{name}[{index}] must be a mapping")
        result.append(dict(item))
    return result


def reject_unknown_keys(raw: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"unknown keys in {name}: {', '.join(unknown)}")


def validate_experiment(experiment: dict[str, Any], index: int) -> None:
    name = f"experiments[{index}]"
    if "command_rejection_policy" in experiment:
        if experiment["command_rejection_policy"] not in ("abort", "feedback"):
            raise ValueError(f"{name}.command_rejection_policy must be abort or feedback")
        if experiment.get("mas") != "ag2" or experiment.get("native") is not True:
            raise ValueError(f"{name}.command_rejection_policy requires an explicit native AG2 experiment")
    if "replicate_id" in experiment and (not isinstance(experiment["replicate_id"], str) or not experiment["replicate_id"].strip()):
        raise TypeError(f"{name}.replicate_id must be a non-empty string")
    if "model_vision" in experiment and not isinstance(experiment["model_vision"], bool):
        raise TypeError(f"{name}.model_vision must be a boolean")
    if experiment.get("code_execution_environment", "shared") not in {"shared", "per-sample"}:
        raise ValueError(f"{name}.code_execution_environment must be shared or per-sample")
    if experiment.get("context_budget_mode", "legacy") not in {"legacy", "server-tokenizer"}:
        raise ValueError(f"{name}.context_budget_mode must be legacy or server-tokenizer")
    if experiment.get("swebench_workspace_mode", "shared-worktree") not in {"shared-worktree", "isolated-clone"}:
        raise ValueError(f"{name}.swebench_workspace_mode must be shared-worktree or isolated-clone")
    if "matrix" in experiment:
        matrix = experiment["matrix"]
        if not isinstance(matrix, dict):
            raise TypeError(f"{name}.matrix must be a mapping")
        reject_unknown_keys(matrix, {"mas", "benchmarks", "runtimes"}, f"{name}.matrix")
        required_keys(matrix, {"mas", "benchmarks", "runtimes"}, f"{name}.matrix")
        for key in ("mas", "benchmarks"):
            values = matrix[key]
            if (
                not isinstance(values, list)
                or not values
                or not all(isinstance(value, str) and value for value in values)
            ):
                raise TypeError(
                    f"{name}.matrix.{key} must be a non-empty list of strings"
                )
        runtimes = matrix["runtimes"]
        if not isinstance(runtimes, list) or not runtimes:
            raise ValueError(f"{name}.matrix.runtimes must be a non-empty list")
        for runtime_index, runtime in enumerate(runtimes):
            if not isinstance(runtime, dict):
                raise TypeError(
                    f"{name}.matrix.runtimes[{runtime_index}] must be a mapping"
                )
            reject_unknown_keys(
                runtime,
                API_KEYS | {"name"},
                f"{name}.matrix.runtimes[{runtime_index}]",
            )
            required_keys(
                runtime,
                {"name", "provider", "protocol", "base_url", "model", "api_key_env"},
                f"{name}.matrix.runtimes[{runtime_index}]",
            )
    else:
        required_keys(experiment, {"name", "mas", "benchmark", "native"}, name)
        for key in ("name", "mas", "benchmark"):
            if not isinstance(experiment[key], str) or not experiment[key]:
                raise TypeError(f"{name}.{key} must be a non-empty string")
    for key in (
        "native",
        "batch",
        "resume",
        "resume_require_resolved_evaluation",
        "progress",
        "judge",
    ):
        if key in experiment:
            as_bool(experiment[key], False, name=f"{name}.{key}")
    if "sample_source" in experiment and experiment["sample_source"] not in {
        "sample",
        "full",
    }:
        raise ValueError(f"{name}.sample_source must be sample or full")
    for key in (
        "sample_limit",
        "sample_offset",
        "max_steps",
        "max_agent_turns",
        "max_turns",
        "max_stalls",
        "native_timeout_seconds",
        "stop_after_failures",
    ):
        value = experiment.get(key)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise TypeError(f"{name}.{key} must be null or a non-negative integer")
    if "sample_ids" in experiment:
        sample_ids = experiment["sample_ids"]
        if (
            not isinstance(sample_ids, list)
            or not sample_ids
            or not all(isinstance(value, str) and value for value in sample_ids)
        ):
            raise TypeError(f"{name}.sample_ids must be a non-empty list of strings")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError(f"{name}.sample_ids must not contain duplicates")
    if "evaluation" in experiment:
        validate_evaluation_config(experiment["evaluation"], f"{name}.evaluation")


def validate_evaluation_config(value: Any, name: str) -> None:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a mapping")
    reject_unknown_keys(value, EVALUATION_KEYS, name)
    harness = value.get("harness")
    if harness not in {"gaia-official", "swebench-official"}:
        raise ValueError(f"{name}.harness must be gaia-official or swebench-official")
    for key in ("timeout_seconds", "process_timeout_seconds", "max_workers"):
        item = value.get(key)
        if item is not None and (
            not isinstance(item, int) or isinstance(item, bool) or item <= 0
        ):
            raise TypeError(f"{name}.{key} must be a positive integer")
    if "clean" in value:
        as_bool(value["clean"], False, name=f"{name}.clean")
    if "use_host_docker_credentials" in value:
        as_bool(
            value["use_host_docker_credentials"],
            True,
            name=f"{name}.use_host_docker_credentials",
        )
    cache_level = value.get("cache_level")
    if cache_level is not None and cache_level not in {
        "none",
        "base",
        "env",
        "instance",
    }:
        raise ValueError(f"{name}.cache_level must be none, base, env, or instance")


def validate_job_contents(
    job_type: str,
    experiments: list[dict[str, Any]],
    repair_datasets: list[dict[str, Any]],
    baselines: list[dict[str, Any]],
    aria: AriaConfig,
) -> None:
    if job_type == "baselines":
        if not baselines:
            raise ValueError("baselines job requires at least one baselines entry")
        if experiments or repair_datasets:
            raise ValueError(
                "baselines job must not define experiments or repair_datasets"
            )
    elif baselines:
        raise ValueError("baselines entries are only valid for job: baselines")
    if job_type == "inference" and repair_datasets:
        raise ValueError("inference job must not define repair_datasets")
    if job_type == "repair" and experiments:
        raise ValueError("repair job must not define experiments")
    if job_type == "aria" and not aria.enabled:
        raise ValueError("aria job requires aria.enabled: true")
    if job_type == "aria-training":
        if aria.post_training is None:
            raise ValueError("aria-training job requires aria.post_training")
        if experiments or repair_datasets or baselines:
            raise ValueError(
                "aria-training job must not define experiments or repair_datasets"
            )
    elif aria.post_training is not None:
        raise ValueError("aria.post_training is only valid for job: aria-training")
    if job_type == "aria-passive":
        if aria.model is None or aria.passive is None:
            raise ValueError("aria-passive job requires aria.model and aria.passive")
        if experiments or repair_datasets or baselines:
            raise ValueError(
                "aria-passive job must not define experiments or repair_datasets"
            )
    elif aria.passive is not None:
        raise ValueError("aria.passive is only valid for job: aria-passive")
    if job_type not in {"aria", "aria-training", "aria-passive"} and aria.enabled:
        raise ValueError("aria.enabled is only valid for job: aria")


def validate_baseline_spec(spec: dict[str, Any], index: int) -> None:
    name = f"baselines[{index}]"
    if "reevaluate" in spec:
        evaluation = spec["reevaluate"]
        if not isinstance(evaluation, dict):
            raise TypeError(f"{name}.reevaluate must be a mapping")
        reject_unknown_keys(evaluation, {"source_results", "sample_ids", "evaluation", "project_environment"}, f"{name}.reevaluate")
        for key in ("source_results", "project_environment"):
            if not isinstance(evaluation.get(key), str) or not evaluation[key].strip():
                raise ValueError(f"{name}.reevaluate.{key} is required")
        if (not isinstance(evaluation.get("sample_ids"), list) or not evaluation["sample_ids"]
                or not all(isinstance(value, str) and value for value in evaluation["sample_ids"])):
            raise ValueError(f"{name}.reevaluate.sample_ids must be a non-empty string list")
        if not isinstance(evaluation.get("evaluation", {}), dict):
            raise TypeError(f"{name}.reevaluate.evaluation must be a mapping")
        if any(key in spec for key in ("reexecute", "rescore", "local_model", "sample_limit", "benchmark_filter")):
            raise ValueError(f"{name}.reevaluate cannot change method generation or sampling")
    if "reexecute" in spec:
        replay = spec["reexecute"]
        if not isinstance(replay, dict):
            raise TypeError(f"{name}.reexecute must be a mapping")
        reject_unknown_keys(replay, {"source_results", "sample_ids", "project_environment"}, f"{name}.reexecute")
        if "project_environment" in replay and (
            not isinstance(replay["project_environment"], str) or not replay["project_environment"].strip()
        ):
            raise ValueError(f"{name}.reexecute.project_environment must be an environment path")
        if "sample_ids" in replay and (not isinstance(replay["sample_ids"], list)
                or not replay["sample_ids"] or not all(isinstance(item, str) for item in replay["sample_ids"])):
            raise ValueError(f"{name}.reexecute.sample_ids must be a non-empty string list")
        if not isinstance(replay.get("source_results"), str) or not replay["source_results"].strip():
            raise ValueError(f"{name}.reexecute.source_results must be a result path")
        if any(key in spec for key in ("rescore", "local_model")):
            raise ValueError(f"{name}.reexecute cannot be combined with rescore or local_model")
        if (spec.get("replay_backend") or {}).get("type") != "native":
            raise ValueError(f"{name}.reexecute requires a native replay backend")
    if "rescore" in spec:
        rescore = spec["rescore"]
        if not isinstance(rescore, dict):
            raise TypeError(f"{name}.rescore must be a mapping")
        reject_unknown_keys(rescore, {"sources", "benchmarks"}, f"{name}.rescore")
        sources = rescore.get("sources")
        if not isinstance(sources, list) or not sources or not all(
            isinstance(path, str) and path.strip() for path in sources
        ):
            raise ValueError(f"{name}.rescore.sources must be non-empty result paths")
        benchmarks = rescore.get("benchmarks", [])
        if not isinstance(benchmarks, list) or any(value not in {"humaneval", "gaia"} for value in benchmarks):
            raise ValueError(f"{name}.rescore.benchmarks supports humaneval and saved-native gaia")
        if any(key in spec for key in ("local_model", "sample_limit", "benchmark_filter")):
            raise ValueError(f"{name}.rescore must preserve the full pool and cannot load a model")
    if "local_model" in spec:
        local = spec["local_model"]
        if not isinstance(local, dict):
            raise TypeError(f"{name}.local_model must be a mapping")
        reject_unknown_keys(local, {
            "model_path", "device", "dtype", "max_new_tokens", "gate_max_new_tokens",
            "max_input_tokens", "temperature", "top_p", "seed",
            "gate_probability_source",
        }, f"{name}.local_model")
        if not isinstance(local.get("model_path"), str) or not local["model_path"].strip():
            raise TypeError(f"{name}.local_model.model_path must be a non-empty string")
        for key in ("max_new_tokens", "gate_max_new_tokens", "max_input_tokens"):
            if key in local and (type(local[key]) is not int or local[key] < 1):
                raise ValueError(f"{name}.local_model.{key} must be a positive integer")
        if local.get("gate_probability_source", "generated") not in {"generated", "label_likelihood"}:
            raise ValueError(f"{name}.local_model.gate_probability_source is unsupported")
        if (local.get("gate_probability_source") == "label_likelihood"
                and isinstance(spec.get("method_options"), dict)
                and spec["method_options"].get("gate_calibration_path")):
            raise ValueError("label likelihood cannot reuse a generated-value gate calibration")
    required_keys(spec, {"name", "method", "input_path", "output_dir"}, name)
    for key in ("name", "method", "input_path", "output_dir"):
        if not isinstance(spec[key], str) or not spec[key].strip():
            raise TypeError(f"{name}.{key} must be a non-empty string")
    from baselines.registry import ALIASES, BASELINE_METHODS

    canonical = ALIASES.get(str(spec["method"]).lower(), str(spec["method"]).lower())
    if canonical not in BASELINE_METHODS:
        raise ValueError(f"{name}.method is unsupported: {spec['method']}")
    if spec.get("input_format", "auto") not in BASELINE_INPUT_FORMATS:
        supported = ", ".join(sorted(BASELINE_INPUT_FORMATS))
        raise ValueError(f"{name}.input_format must be one of: {supported}")
    if "dataset_root" in spec and (
        not isinstance(spec["dataset_root"], str) or not spec["dataset_root"].strip()
    ):
        raise TypeError(f"{name}.dataset_root must be a non-empty string")
    if "max_events" in spec and (
        not isinstance(spec["max_events"], int)
        or isinstance(spec["max_events"], bool)
        or spec["max_events"] <= 0
    ):
        raise TypeError(f"{name}.max_events must be a positive integer")
    if "sample_limit" in spec and (
        not isinstance(spec["sample_limit"], int)
        or isinstance(spec["sample_limit"], bool)
        or spec["sample_limit"] <= 0
    ):
        raise TypeError(f"{name}.sample_limit must be a positive integer")
    if "sample_workers" in spec and (
        not isinstance(spec["sample_workers"], int)
        or isinstance(spec["sample_workers"], bool)
        or spec["sample_workers"] <= 0
    ):
        raise TypeError(f"{name}.sample_workers must be a positive integer")
    if "prompt_char_budget" in spec and (
        not isinstance(spec["prompt_char_budget"], int)
        or isinstance(spec["prompt_char_budget"], bool)
        or spec["prompt_char_budget"] < 10_000
    ):
        raise TypeError(f"{name}.prompt_char_budget must be an integer >= 10000")
    if "allow_legacy_resume" in spec:
        as_bool(
            spec["allow_legacy_resume"],
            False,
            name=f"{name}.allow_legacy_resume",
        )
    if "split" in spec and (
        not isinstance(spec["split"], str) or not spec["split"].strip()
    ):
        raise TypeError(f"{name}.split must be a non-empty string")
    if "resume" in spec:
        as_bool(spec["resume"], True, name=f"{name}.resume")
    if spec.get("on_sample_error", "stop") not in {"stop", "continue"}:
        raise ValueError(f"{name}.on_sample_error must be stop or continue")
    if "method_options" in spec and not isinstance(spec["method_options"], dict):
        raise TypeError(f"{name}.method_options must be a mapping")
    if "model_identity" in spec and (
        not isinstance(spec["model_identity"], str) or not spec["model_identity"].strip()
    ):
        raise TypeError(f"{name}.model_identity must be a non-empty string")
    if "replay_backend" in spec:
        backend = spec["replay_backend"]
        if not isinstance(backend, dict):
            raise TypeError(f"{name}.replay_backend must be a mapping")
        reject_unknown_keys(backend, {"type", "options"}, f"{name}.replay_backend")
        backend_type = backend.get("type", "none")
        if backend_type not in BASELINE_REPLAY_TYPES:
            supported = ", ".join(sorted(BASELINE_REPLAY_TYPES))
            raise ValueError(f"{name}.replay_backend.type must be one of: {supported}")
        if "options" in backend and not isinstance(backend["options"], dict):
            raise TypeError(f"{name}.replay_backend.options must be a mapping")


def validate_repair_spec(spec: dict[str, Any], index: int) -> None:
    name = f"repair_datasets[{index}]"
    required_keys(
        spec,
        {
            "name",
            "schema_version",
            "source_batch_dir",
            "dataset_dir",
            "repair_output_dir",
            "split",
            "rerun",
            "resume",
            "native",
            "repair_trials",
        },
        name,
    )
    if spec["schema_version"] != "aria-repair-dataset-v6":
        raise ValueError(f"{name}.schema_version must be aria-repair-dataset-v6")
    operator_id = spec.get(
        "repair_operator_id",
        "model/aria-repair-v4-localize-attribute-patch-answer-blind",
    )
    if operator_id != "model/aria-repair-v4-localize-attribute-patch-answer-blind":
        raise ValueError(
            f"{name}.repair_operator_id must be "
            "model/aria-repair-v4-localize-attribute-patch-answer-blind"
        )
    for key in (
        "rerun",
        "resume",
        "native",
        "require_inference_completed",
        "require_trace_acceptance",
        "retry_invalid",
    ):
        if key in spec:
            as_bool(spec[key], False, name=f"{name}.{key}")
    if "skip_sample_ids" in spec and not isinstance(spec["skip_sample_ids"], list):
        raise TypeError(f"{name}.skip_sample_ids must be a list")
    on_sample_error = spec.get("on_sample_error", "fail")
    if on_sample_error not in {"fail", "continue"}:
        raise ValueError(f"{name}.on_sample_error must be 'fail' or 'continue'")
    for key in (
        "repair_trials",
        "structured_output_retries",
        "attribution_max_events",
        "max_consecutive_endpoint_invalid",
        "capacity_wait_seconds",
        "capacity_wait_max_attempts",
        "max_steps",
        "max_turns",
        "max_stalls",
        "native_timeout_seconds",
        "sample_limit",
        "gate_success_sample_limit",
    ):
        if key in spec and (
            not isinstance(spec[key], int)
            or isinstance(spec[key], bool)
            or spec[key] < 0
        ):
            raise TypeError(f"{name}.{key} must be a non-negative integer")
    for key in ("repair_trials", "attribution_max_events"):
        if key in spec and spec[key] < 1:
            raise ValueError(f"{name}.{key} must be at least one")
    if "gate_success_sample_rate" in spec and (
        not isinstance(spec["gate_success_sample_rate"], (int, float))
        or isinstance(spec["gate_success_sample_rate"], bool)
    ):
        raise TypeError(f"{name}.gate_success_sample_rate must be numeric")
    if "gate_success_sample_rate" in spec and not 0 <= float(
        spec["gate_success_sample_rate"]
    ) <= 1:
        raise ValueError(
            f"{name}.gate_success_sample_rate must be between zero and one"
        )
    if "sample_ids" in spec and (
        not isinstance(spec["sample_ids"], list)
        or not all(isinstance(value, str) for value in spec["sample_ids"])
    ):
        raise TypeError(f"{name}.sample_ids must be a list of strings")
    if "evaluation" in spec:
        value = spec["evaluation"]
        if not isinstance(value, dict):
            raise TypeError(f"{name}.evaluation must be a mapping")
        reject_unknown_keys(value, EVALUATION_KEYS, f"{name}.evaluation")
