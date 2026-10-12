"""Passive repair service plus compatibility imports for active deployment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from aria.hook import (
    ActiveAction,
    ActiveIntervention,
    ActiveMasHost,
    ActiveRepairController,
    AriaHook,
    DeploymentMode,
    EventNormalizer,
    MasHookProfile,
    enact_intervention,
    get_mas_hook_profile,
)
from aria.model import AriaModel, AriaModelConfig, load_aria_model
from aria.records import IncidentRecord, RepairAttribution, RepairRecord, TrajectoryEvent


@dataclass(frozen=True)
class ReplayResult:
    succeeded: bool
    events: tuple[TrajectoryEvent, ...]
    evidence: Mapping[str, Any]


class ReplayEnvironment(Protocol):
    def replay(self, events: Sequence[TrajectoryEvent]) -> ReplayResult:
        """Execute the assembled trajectory and return real host evidence."""


@dataclass(frozen=True)
class PassiveRepairResult:
    attribution: RepairAttribution | None
    repair: RepairRecord | None
    repaired_events: tuple[TrajectoryEvent, ...]
    execution_status: str
    replay_evidence: Mapping[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "attribution": (
                self.attribution.to_dict() if self.attribution is not None else None
            ),
            "repair": self.repair.to_dict() if self.repair is not None else None,
            "repaired_events": [event.to_dict() for event in self.repaired_events],
            "execution_status": self.execution_status,
            "replay_evidence": dict(self.replay_evidence)
            if self.replay_evidence
            else None,
        }


class PassiveRepairService:
    """Direct model invocation for a completed, externally flagged incident."""

    def __init__(self, model: AriaModel) -> None:
        self.model = model

    def repair(
        self,
        incident: IncidentRecord,
        *,
        replay_environment: ReplayEnvironment | None = None,
    ) -> PassiveRepairResult:
        attribute = getattr(self.model, "attribute", None)
        localize = getattr(self.model, "localize", None)
        attribution = (
            attribute(incident, localize(incident))
            if callable(attribute) and callable(localize)
            else attribute(incident)
            if callable(attribute)
            else None
        )
        repair = (
            self.model.propose_repair(incident, attribution)
            if attribution is not None
            else self.model.propose_repair(incident)
        )
        repaired_events = incident.assemble(repair)
        if replay_environment is None:
            return PassiveRepairResult(
                attribution=attribution,
                repair=repair,
                repaired_events=repaired_events,
                execution_status="proposed",
                replay_evidence=None,
            )
        replay = replay_environment.replay(repaired_events)
        return PassiveRepairResult(
            attribution=attribution,
            repair=repair,
            repaired_events=replay.events,
            execution_status="verified" if replay.succeeded else "replay_failed",
            replay_evidence=dict(replay.evidence),
        )

    def repair_and_verify(
        self,
        incident: IncidentRecord,
        replay_environment: ReplayEnvironment,
    ) -> PassiveRepairResult:
        """Generate the complete splice and require host/evaluator replay.

        This is the production passive entry point.  The looser ``repair``
        method remains available for proposal-only workflows and for the
        config-only JSONL job, but it is intentionally impossible to report a
        verified result here without an explicit replayer.
        """

        if replay_environment is None:
            raise TypeError("repair_and_verify requires a ReplayEnvironment")
        return self.repair(incident, replay_environment=replay_environment)


def load_passive_repair_service(config: AriaModelConfig) -> PassiveRepairService:
    """Load one configured ARIA model for direct passive repair requests."""

    return PassiveRepairService(load_aria_model(config))


def load_active_repair_controller(
    config: AriaModelConfig,
    *,
    gate_threshold: float,
    shadow: bool = False,
    fail_open: bool = True,
) -> ActiveRepairController:
    """Load the controller embedded in a host-side ARIA hook."""

    return ActiveRepairController(
        load_aria_model(config),
        gate_threshold=gate_threshold,
        mode=DeploymentMode.SHADOW if shadow else DeploymentMode.ACTIVE,
        fail_open=fail_open,
    )


def load_active_hook(
    config: AriaModelConfig,
    host: ActiveMasHost,
    *,
    mas: str | None = None,
    profile: MasHookProfile | None = None,
    gate_threshold: float = 0.5,
    shadow: bool = False,
    fail_open: bool = True,
    normalizer: EventNormalizer | None = None,
) -> AriaHook:
    """Load the public active component around a user's MAS host adapter.

    The returned hook owns ARIA-model calls and policy decisions.  ``host``
    remains responsible for pausing, delivering, restoring state, executing
    tools, and reporting post-hook evidence.  ``mas`` selects one of the
    repository's declared profiles; pass ``profile`` for a custom framework.
    """

    selected_profile = profile or get_mas_hook_profile(str(mas or "future-mas"))
    controller = load_active_repair_controller(
        config,
        gate_threshold=gate_threshold,
        shadow=shadow,
        fail_open=fail_open,
    )
    return AriaHook(
        controller,
        host,
        selected_profile,
        normalizer=normalizer,
    )


__all__ = [
    "ActiveAction",
    "ActiveIntervention",
    "ActiveMasHost",
    "ActiveRepairController",
    "DeploymentMode",
    "PassiveRepairResult",
    "PassiveRepairService",
    "ReplayEnvironment",
    "ReplayResult",
    "enact_intervention",
    "load_active_repair_controller",
    "load_active_hook",
    "load_passive_repair_service",
]
