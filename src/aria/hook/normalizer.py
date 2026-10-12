"""Configurable conversion from native MAS events to ARIA trajectories."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from aria.records import TrajectoryEvent


class EventNormalizer(Protocol):
    def normalize(self, event: object) -> TrajectoryEvent:
        """Return one schema-valid event with a stable host identifier."""


@dataclass(frozen=True)
class EventFieldMapping:
    event_id: str = "event_id"
    role: str = "role"
    kind: str = "kind"
    content: str = "content"
    tool_calls: str = "tool_calls"
    metadata: str = "metadata"


@dataclass(frozen=True)
class MappingEventNormalizer:
    """Normalize a mapping using configurable native field names."""

    fields: EventFieldMapping = EventFieldMapping()

    def normalize(self, event: object) -> TrajectoryEvent:
        if isinstance(event, TrajectoryEvent):
            return event
        if not isinstance(event, Mapping):
            raise TypeError(
                "ARIA hook events must be TrajectoryEvent or mapping values"
            )
        raw: dict[str, Any] = {
            "event_id": event.get(self.fields.event_id),
            "role": event.get(self.fields.role),
            "kind": event.get(self.fields.kind),
            "content": event.get(self.fields.content, ""),
            "tool_calls": event.get(self.fields.tool_calls, []),
            "metadata": event.get(self.fields.metadata, {}),
        }
        return TrajectoryEvent.from_dict(raw, name="hook event")
