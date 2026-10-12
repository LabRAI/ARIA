"""Top-level job dispatcher for inference, repair, and ARIA workflows."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from aria.passive import run_passive_repairs_from_spec
from aria.training import run_post_training_from_spec
from baselines.runner import run_baselines
from common.config import ExperimentConfig
from common.model_api import ModelApiClient
from inference.runner import run_inference
from repair.runner import run_repair


def run_from_config(config: ExperimentConfig) -> list[dict[str, Any]]:
    if config.job_type == "baselines":
        client = ModelApiClient(config.api) if config.api is not None else None
        return run_baselines(config, client)
    if config.job_type == "aria-training":
        if config.aria.post_training is None:
            raise ValueError("aria-training job requires aria.post_training")
        return [
            run_post_training_from_spec(
                config.aria.post_training,
                config_path=config.config_path,
            )
        ]
    if config.job_type == "aria-passive":
        if config.aria.model is None or config.aria.passive is None:
            raise ValueError("aria-passive job requires aria.model and aria.passive")
        return [
            run_passive_repairs_from_spec(
                config.aria.model,
                config.aria.passive,
                config_path=config.config_path,
            )
        ]
    if config.api is None:
        raise ValueError(f"{config.job_type} job requires api configuration")
    return run_from_config_with_client(config, ModelApiClient(config.api))


def run_from_config_with_client(config: ExperimentConfig, client: Any) -> list[dict[str, Any]]:
    job_type = config.job_type
    if job_type == "repair":
        return run_repair(config, client)
    if job_type == "aria":
        if not config.aria.enabled:
            raise ValueError("ARIA job requires aria.enabled: true")
        return run_inference(config, client)
    if job_type == "inference":
        return run_inference(config, client)
    if job_type == "baselines":
        return run_baselines(config, client)
    if job_type == "mixed":
        inference_config = replace(config, job_type="inference")
        repair_config = replace(config, job_type="repair")
        return [*run_inference(inference_config, client), *run_repair(repair_config, client)]
    raise ValueError(f"unsupported job type: {job_type}")
