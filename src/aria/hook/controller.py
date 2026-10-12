"""Controller policy owned by the deployable ARIA hook component."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import json
from typing import Any

from aria.hook.types import (
    FULL_REALTIME_CAPABILITIES,
    ActiveAction,
    ActiveIntervention,
    DeploymentMode,
    MasHookCapabilities,
)
from aria.model import AriaModel
from aria.records import IncidentRecord, RepairAttribution, RepairRecord, TrajectoryEvent


def validate_pending_substitution(decision: ActiveIntervention, content: str, *, role: str,
                                  protected: bool = False) -> ActiveIntervention:
    """The shared last delivery check for native interception and training rollouts."""
    if decision.action != ActiveAction.SUBSTITUTE:
        return decision
    replacement = decision.patch_events[0]
    try:
        structured = isinstance(json.loads(content), (dict, list))
    except (TypeError, ValueError):
        structured = False
    reason = None
    if protected or structured:
        reason = "degraded:protected_native_response"
    elif replacement.role != role or not replacement.content.strip():
        reason = "degraded:incompatible_pending_message"
    elif replacement.content == content:
        reason = "degraded:no_content_change"
    return replace(decision, action=ActiveAction.SHADOW, reason=reason) if reason else decision


class ActiveRepairController:
    """Compare gates, validate model output, and choose one host action."""

    def __init__(
        self,
        model: AriaModel,
        *,
        gate_threshold: float,
        mode: DeploymentMode = DeploymentMode.ACTIVE,
        fail_open: bool = True,
    ) -> None:
        if not 0.0 <= gate_threshold <= 1.0:
            raise ValueError("ARIA gate_threshold must be between 0 and 1")
        self.model = model
        self.gate_threshold = gate_threshold
        self.mode = mode
        self.fail_open = fail_open
        self._incident_id = "active-incident"
        self._constraints: Mapping[str, Any] = {}
        self._feedback: str | None = None
        self._events: list[TrajectoryEvent] = []

    @property
    def events(self) -> tuple[TrajectoryEvent, ...]:
        return tuple(self._events)

    @property
    def incident_id(self) -> str:
        """Current host task/incident identifier used in audit records."""

        return self._incident_id

    def begin(
        self,
        incident_id: str,
        *,
        constraints: Mapping[str, Any] | None = None,
        feedback: str | None = None,
    ) -> None:
        if not incident_id.strip():
            raise ValueError("active ARIA incident_id must be non-empty")
        self._incident_id = incident_id
        self._constraints = dict(constraints or {})
        self._feedback = feedback
        self._events = []

    def intercept(
        self,
        event: TrajectoryEvent,
        *,
        feedback: str | None = None,
        capabilities: MasHookCapabilities = FULL_REALTIME_CAPABILITIES,
    ) -> ActiveIntervention:
        """Inspect one pending event and return a capability-safe decision."""

        if any(existing.event_id == event.event_id for existing in self._events):
            raise ValueError(
                f"active ARIA received duplicate event_id: {event.event_id}"
            )
        if feedback is not None:
            self._feedback = feedback
        self._events.append(event)
        incident = IncidentRecord(
            incident_id=self._incident_id,
            events=tuple(self._events),
            feedback=self._feedback,
            constraints=self._constraints,
        )
        try:
            gate = self.model.predict_gate(incident)
        except Exception as exc:  # noqa: BLE001 - fail-open is an explicit host policy.
            return self._model_error(event, None, "gate", exc)
        if gate.probability < self.gate_threshold:
            return ActiveIntervention(
                action=ActiveAction.DELIVER,
                event=event,
                gate=gate,
                repair=None,
                patch_events=(),
                reason="gate_below_threshold",
            )
        attribution: RepairAttribution | None = None
        try:
            attribute = getattr(self.model, "attribute", None)
            if callable(attribute):
                localize = getattr(self.model, "localize", None)
                attribution = (
                    attribute(incident, localize(incident))
                    if callable(localize)
                    else attribute(incident)
                )
                repair = self.model.propose_repair(incident, attribution)
            else:
                # Compatibility for a pre-RCA model adapter. Production ARIA
                # checkpoints implement attribute() and use the branch above.
                repair = self.model.propose_repair(incident)
            cut_index = self._validate_repair(incident, repair)
        except Exception as exc:  # noqa: BLE001 - invalid model output is recorded, never enacted.
            return self._model_error(event, gate, "repair", exc)

        patch_has_tool_calls = any(
            patch.kind == "tool_call" or patch.tool_calls
            for patch in repair.replacement_events
        )
        if self.mode == DeploymentMode.SHADOW:
            return self._shadow(
                event, gate, repair, "shadow_gate_triggered", attribution
            )
        if not capabilities.supports_pause:
            return self._shadow(event, gate, repair, "degraded:no_pause", attribution)
        if patch_has_tool_calls and not capabilities.reexecute_tool_calls:
            return self._shadow(
                event, gate, repair, "degraded:no_tool_reexecution", attribution
            )

        pending_index = len(incident.events) - 1
        if cut_index == pending_index - 1:
            if not capabilities.substitute:
                return self._shadow(
                    event, gate, repair, "degraded:no_substitute", attribution
                )
            if len(repair.replacement_events) > 1 and not capabilities.resume:
                return self._shadow(
                    event, gate, repair, "degraded:no_resume", attribution
                )
            return ActiveIntervention(
                action=ActiveAction.SUBSTITUTE,
                event=event,
                gate=gate,
                attribution=attribution,
                repair=repair,
                patch_events=repair.replacement_events,
                reason="gate_triggered:substitute",
                reexecute_tool_calls=capabilities.reexecute_tool_calls,
            )
        if not capabilities.rollback or not capabilities.resume:
            return self._shadow(
                event, gate, repair, "degraded:no_rollback", attribution
            )
        return ActiveIntervention(
            action=ActiveAction.ROLLBACK,
            event=event,
            gate=gate,
            attribution=attribution,
            repair=repair,
            patch_events=repair.replacement_events,
            reason="gate_triggered:rollback",
            reexecute_tool_calls=capabilities.reexecute_tool_calls,
        )

    def record_enacted(self, intervention: ActiveIntervention) -> None:
        """Replace the controller trajectory only after host execution succeeds."""

        if intervention.action not in {ActiveAction.SUBSTITUTE, ActiveAction.ROLLBACK}:
            return
        repair = intervention.repair
        if repair is None:
            raise ValueError("enacted intervention is missing its repair record")
        cut_index = repair.cut_index(self._events)
        self._events = [
            *self._events[: cut_index + 1],
            *intervention.patch_events,
        ]

    def observe(self, event: TrajectoryEvent) -> None:
        """Record an already delivered host event without making a new decision."""

        if any(item.event_id == event.event_id for item in self._events):
            raise ValueError(f"active ARIA received duplicate event_id: {event.event_id}")
        self._events.append(event)

    def record_failed(self, intervention: ActiveIntervention) -> None:
        """Drop an undelivered pending event after a host enactment failure.

        A host exception is infrastructure evidence, not a cognitive repair
        outcome. Removing only the pending event lets the caller retry the same
        event without tripping the duplicate-ID guard while preserving the
        already committed prefix.
        """

        if self._events and self._events[-1].event_id == intervention.event.event_id:
            self._events.pop()

    def _validate_repair(
        self,
        incident: IncidentRecord,
        repair: RepairRecord,
    ) -> int:
        cut_index = repair.cut_index(incident.events)
        pending_index = len(incident.events) - 1
        if cut_index >= pending_index:
            raise ValueError("repair retains the pending event")
        retained_ids = {event.event_id for event in incident.events[: cut_index + 1]}
        patch_ids = {event.event_id for event in repair.replacement_events}
        if retained_ids & patch_ids:
            raise ValueError("replacement event_id collides with retained trajectory")
        if any(event.kind == "tool_result" for event in repair.replacement_events):
            raise ValueError(
                "active model patches may not assert unexecuted tool results"
            )
        return cut_index

    def _shadow(
        self,
        event: TrajectoryEvent,
        gate: Any,
        repair: RepairRecord,
        reason: str,
        attribution: RepairAttribution | None = None,
    ) -> ActiveIntervention:
        return ActiveIntervention(
            action=ActiveAction.SHADOW,
            event=event,
            gate=gate,
            attribution=attribution,
            repair=repair,
            patch_events=repair.replacement_events,
            reason=reason,
            reexecute_tool_calls=False,
        )

    def _model_error(
        self,
        event: TrajectoryEvent,
        gate: Any,
        operation: str,
        error: Exception,
    ) -> ActiveIntervention:
        if not self.fail_open:
            raise error
        return ActiveIntervention(
            action=ActiveAction.MODEL_ERROR,
            event=event,
            gate=gate,
            repair=None,
            patch_events=(),
            reason=f"{operation}_error:{type(error).__name__}",
            error=str(error),
        )
