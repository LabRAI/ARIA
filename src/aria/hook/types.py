"""Shared decisions and capability declarations for the ARIA hook."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from aria.model import GatePrediction
from aria.records import RepairAttribution, RepairRecord, TrajectoryEvent


class DeploymentMode(str, Enum):
    ACTIVE = "active"
    SHADOW = "shadow"


class ActiveAction(str, Enum):
    DELIVER = "deliver"
    SUBSTITUTE = "substitute"
    ROLLBACK = "rollback"
    SHADOW = "shadow"
    MODEL_ERROR = "model_error"


class PauseStrategy(str, Enum):
    """How the host keeps an event undelivered while ARIA decides."""

    BLOCKING_HOOK = "blocking_hook"
    EXPLICIT = "explicit"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class MasHookCapabilities:
    """Operations an attached MAS host has actually implemented."""

    pause_strategy: PauseStrategy
    substitute: bool
    rollback: bool
    resume: bool
    reexecute_tool_calls: bool

    @property
    def supports_pause(self) -> bool:
        return self.pause_strategy != PauseStrategy.UNAVAILABLE

    @property
    def realtime_level(self) -> str:
        if not self.supports_pause:
            return "shadow-only"
        if (
            self.substitute
            and self.rollback
            and self.resume
            and self.reexecute_tool_calls
        ):
            return "full"
        if self.substitute and self.reexecute_tool_calls:
            return "substitute-only"
        if self.substitute:
            return "text-substitute-only"
        return "shadow-only"


FULL_REALTIME_CAPABILITIES = MasHookCapabilities(
    pause_strategy=PauseStrategy.BLOCKING_HOOK,
    substitute=True,
    rollback=True,
    resume=True,
    reexecute_tool_calls=True,
)

SHADOW_ONLY_CAPABILITIES = MasHookCapabilities(
    pause_strategy=PauseStrategy.UNAVAILABLE,
    substitute=False,
    rollback=False,
    resume=False,
    reexecute_tool_calls=False,
)


@dataclass(frozen=True)
class MasHookProfile:
    slug: str
    pre_delivery_hook: str
    post_delivery_hook: str
    capabilities: MasHookCapabilities = FULL_REALTIME_CAPABILITIES

    @property
    def supports_pause(self) -> bool:
        return self.capabilities.supports_pause

    @property
    def supports_rollback(self) -> bool:
        return self.capabilities.rollback

    @property
    def reexecute_tool_calls(self) -> bool:
        return self.capabilities.reexecute_tool_calls


@dataclass(frozen=True)
class ActiveIntervention:
    action: ActiveAction
    event: TrajectoryEvent
    gate: GatePrediction | None
    repair: RepairRecord | None
    patch_events: tuple[TrajectoryEvent, ...]
    reason: str
    reexecute_tool_calls: bool = False
    attribution: RepairAttribution | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "event": self.event.to_dict(),
            "gate": asdict(self.gate) if self.gate is not None else None,
            "attribution": (
                self.attribution.to_dict() if self.attribution is not None else None
            ),
            "repair": self.repair.to_dict() if self.repair is not None else None,
            "patch_events": [event.to_dict() for event in self.patch_events],
            "reason": self.reason,
            "reexecute_tool_calls": self.reexecute_tool_calls,
            "error": self.error,
        }
