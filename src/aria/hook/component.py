"""Deployable hook that owns interception, normalization, and host execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from aria.hook.controller import ActiveRepairController
from aria.hook.normalizer import EventNormalizer, MappingEventNormalizer
from aria.hook.types import (
    ActiveAction,
    ActiveIntervention,
    MasHookProfile,
    PauseStrategy,
)
from aria.records import TrajectoryEvent


class ActiveMasHost(Protocol):
    """Host operations invoked by the hook, never by the model."""

    def pause_delivery(self, event: TrajectoryEvent) -> None:
        """Pause an event for profiles using explicit rather than blocking hooks."""

    def deliver(self, event: TrajectoryEvent) -> None:
        """Deliver the pending event unchanged."""

    def substitute_pending(
        self,
        event: TrajectoryEvent,
        replacement: TrajectoryEvent,
        *,
        reexecute_tool_calls: bool,
    ) -> None:
        """Replace the pending event and execute changed tool calls when present."""

    def rollback_to(self, cut_event_id: str | None) -> None:
        """Restore host state through the retained cut point."""

    def resume_patch(
        self,
        events: Sequence[TrajectoryEvent],
        *,
        reexecute_tool_calls: bool,
    ) -> None:
        """Resume from replacement events and execute changed tool calls."""


@dataclass(frozen=True)
class HookRegistration:
    """Names of the callbacks installed in a host runtime.

    The registration object is deliberately framework-neutral.  A host may
    expose a pair of ``register_*`` functions, an event-bus subscription, or a
    decorator API; ARIA only needs the callback handles to be installed and,
    optionally, removed by the host.
    """

    pre_hook: str
    post_hook: str
    installed: bool = True


class AriaHook:
    """Small host-side component delivered alongside a trained checkpoint."""

    def __init__(
        self,
        controller: ActiveRepairController,
        host: ActiveMasHost,
        profile: MasHookProfile,
        *,
        normalizer: EventNormalizer | None = None,
    ) -> None:
        self.controller = controller
        self.host = host
        self.profile = profile
        self.normalizer = normalizer or MappingEventNormalizer()
        self.interventions: list[dict[str, Any]] = []
        self.delivery_events: list[dict[str, Any]] = []
        self._registration: HookRegistration | None = None

    def begin(
        self,
        incident_id: str,
        *,
        constraints: Mapping[str, Any] | None = None,
        feedback: str | None = None,
    ) -> None:
        self.controller.begin(incident_id, constraints=constraints, feedback=feedback)

    def before_delivery(
        self,
        native_event: object,
        *,
        feedback: str | None = None,
    ) -> ActiveIntervention:
        """Hold, normalize, decide, and enact one pending native MAS event."""

        event = self.normalizer.normalize(native_event)
        capabilities = self.profile.capabilities
        if capabilities.pause_strategy == PauseStrategy.EXPLICIT:
            self.host.pause_delivery(event)
        intervention = self.controller.intercept(
            event,
            feedback=feedback,
            capabilities=capabilities,
        )
        try:
            enact_intervention(self.host, intervention)
        except Exception as exc:  # noqa: BLE001 - host errors are audit evidence.
            self.controller.record_failed(intervention)
            self.interventions.append(
                {
                    "incident_id": self.controller.incident_id,
                    "profile": self.profile.slug,
                    "hook": self.profile.pre_delivery_hook,
                    "capability_level": capabilities.realtime_level,
                    "intervention": intervention.to_dict(),
                    "enactment_error": f"{type(exc).__name__}: {exc}",
                }
            )
            self.after_delivery(event, error=f"{type(exc).__name__}: {exc}")
            raise
        self.controller.record_enacted(intervention)
        self.interventions.append(
            {
                "incident_id": self.controller.incident_id,
                "profile": self.profile.slug,
                "hook": self.profile.pre_delivery_hook,
                "capability_level": capabilities.realtime_level,
                "intervention": intervention.to_dict(),
            }
        )
        return intervention

    def after_delivery(
        self,
        native_event: object,
        host_result: object | None = None,
        *,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Record the post-delivery outcome without invoking the model.

        ``before_delivery`` is the only gate/repair decision point.  The
        post-hook exists to capture what the host actually delivered (or why it
        failed), including tool/workspace evidence, so an intervention is never
        mistaken for a successful repair merely because the model returned a
        valid JSON patch.
        """

        event = self.normalizer.normalize(native_event)
        record = {
            "incident_id": self.controller.incident_id,
            "profile": self.profile.slug,
            "hook": self.profile.post_delivery_hook,
            "event": event.to_dict(),
            "host_result": _jsonable_host_result(host_result),
            "error": error,
        }
        self.delivery_events.append(record)
        return record

    def install(
        self,
        *,
        register_pre_delivery_hook: Callable[[Callable[..., ActiveIntervention]], Any],
        register_post_delivery_hook: Callable[[Callable[..., dict[str, Any]]], Any],
    ) -> HookRegistration:
        """Install both callbacks in a MAS event loop.

        The pre callback must run while the event is still undelivered.  The
        post callback may receive an optional host result as its second
        positional argument; ``after_delivery`` accepts and records it.  A
        host should retain the returned registration metadata alongside its
        own unsubscription handles.
        """

        if self._registration is not None:
            raise RuntimeError("ARIA hook is already installed")
        register_pre_delivery_hook(self.before_delivery)
        register_post_delivery_hook(self.after_delivery)
        self._registration = HookRegistration(
            pre_hook=self.profile.pre_delivery_hook,
            post_hook=self.profile.post_delivery_hook,
        )
        return self._registration

    @property
    def registration(self) -> HookRegistration | None:
        """Return callback registration metadata, if :meth:`install` ran."""

        return self._registration

    def export_state(self) -> dict[str, Any]:
        """Return an auditable, JSON-compatible deployment snapshot."""

        return {
            "profile": self.profile.slug,
            "pre_delivery_hook": self.profile.pre_delivery_hook,
            "post_delivery_hook": self.profile.post_delivery_hook,
            "capability_level": self.profile.capabilities.realtime_level,
            "interventions": list(self.interventions),
            "delivery_events": list(self.delivery_events),
            "controller_event_count": len(self.controller.events),
        }


def enact_intervention(host: ActiveMasHost, intervention: ActiveIntervention) -> None:
    """Execute a controller decision using only generic host operations."""

    if intervention.action in {
        ActiveAction.DELIVER,
        ActiveAction.SHADOW,
        ActiveAction.MODEL_ERROR,
    }:
        host.deliver(intervention.event)
        return
    if intervention.repair is None or not intervention.patch_events:
        raise ValueError(
            "active repair intervention requires a repair and patch events"
        )
    if intervention.action == ActiveAction.SUBSTITUTE:
        host.substitute_pending(
            intervention.event,
            intervention.patch_events[0],
            reexecute_tool_calls=intervention.reexecute_tool_calls,
        )
        if len(intervention.patch_events) > 1:
            host.resume_patch(
                intervention.patch_events[1:],
                reexecute_tool_calls=intervention.reexecute_tool_calls,
            )
        return
    if intervention.action == ActiveAction.ROLLBACK:
        host.rollback_to(intervention.repair.cut_event_id)
        host.resume_patch(
            intervention.patch_events,
            reexecute_tool_calls=intervention.reexecute_tool_calls,
        )
        return
    raise AssertionError(f"unhandled active intervention: {intervention.action}")


ActiveHookBridge = AriaHook


def _jsonable_host_result(value: object | None) -> Any:
    """Best-effort serialization of framework-specific post-hook results."""

    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            result = to_dict()
            if isinstance(result, (str, int, float, bool, list, dict)):
                return result
        except Exception:  # noqa: BLE001 - telemetry must never break delivery.
            pass
    return {"type": type(value).__name__, "repr": repr(value)}
