"""Stable method registry used by YAML jobs and programmatic evaluation."""

from __future__ import annotations

from typing import Any

from baselines.agentrx import AgentRxRepair
from baselines.aria import AriaActive, AriaPassive
from baselines.dover import DoVer
from baselines.errorprobe import ErrorProbeRepair
from baselines.no_intervention import NoIntervention
from baselines.safeagent import SafeAgent
from baselines.spark_to_fire import SparkToFireGovernance
from baselines.who_when import WhoWhenRepair
from baselines.zero_shot import ZeroShotRepair

BASELINE_METHODS = {
    "aria_passive": AriaPassive,
    "aria_active": AriaActive,
    "zero_shot": ZeroShotRepair,
    "who_when": WhoWhenRepair,
    "agentrx": AgentRxRepair,
    "errorprobe": ErrorProbeRepair,
    "dover": DoVer,
    "no_intervention": NoIntervention,
    "spark_to_fire": SparkToFireGovernance,
    "safeagent": SafeAgent,
}

ALIASES = {
    "aria": "aria_passive",
    "aria-passive": "aria_passive",
    "aria-active": "aria_active",
    "who&when": "who_when",
    "who-when": "who_when",
    "agent_rx": "agentrx",
    "no-intervention": "no_intervention",
    "zero-shot": "zero_shot",
    "spark-to-fire": "spark_to_fire",
    "safe_agent": "safeagent",
}


def build_method(name: str, **options: Any):
    canonical = ALIASES.get(name.lower(), name.lower())
    try:
        method_type = BASELINE_METHODS[canonical]
    except KeyError as exc:
        supported = ", ".join(sorted(BASELINE_METHODS))
        raise ValueError(
            f"unsupported baseline method {name!r}; expected one of: {supported}"
        ) from exc
    return method_type(**options)
