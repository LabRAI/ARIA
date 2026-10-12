"""Repair-dataset job runner."""

from __future__ import annotations

import logging
from typing import Any

from common.config import ExperimentConfig
from common.runtime_support import configure_root_logging
from repair.repair_dataset import run_repair_dataset


def run_repair(config: ExperimentConfig, client: Any) -> list[dict[str, Any]]:
    configure_root_logging(config.log_level)
    logger = logging.getLogger("masfm.repair")
    logger.info("Starting repair job: repair_datasets=%s", len(config.repair_datasets))
    results = [
        run_repair_dataset(config, repair_spec, repair_index, client=client)
        for repair_index, repair_spec in enumerate(config.repair_datasets)
    ]
    logger.info("Finished repair job")
    return results
