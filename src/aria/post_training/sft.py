"""Stage 1: structured repair SFT and deployment-gate supervision."""

from __future__ import annotations

from typing import Any

from aria.records import IncidentRecord, RepairRecord


def build_sft_rows(
    incident: IncidentRecord,
    *,
    feedback: str | None,
) -> list[dict[str, Any]]:
    """Build Eq. (1) repair targets plus prefix-level gate labels."""

    repair = _required_repair(incident)
    # Equation (1) predicts diagnosis, cut, and continuation from the observable
    # incident.  The projection also contains teacher localization/attribution
    # rows as auxiliary targets, but putting either gold object in this input
    # would leak the fields the deployed one-shot repair call has to predict.
    model_input = incident.model_input(feedback=feedback)
    rows: list[dict[str, Any]] = []
    if incident.attribution is not None:
        rows.append(
            {
                "incident_id": incident.incident_id,
                "task": "failure_localization",
                "input": incident.model_input(feedback=feedback),
                "target": incident.attribution.localization.to_dict(),
            }
        )
        rows.append(
            {
                "incident_id": incident.incident_id,
                "task": "repair_attribution",
                "input": incident.attribution_model_input(
                    incident.attribution.localization,
                    feedback=feedback,
                ),
                "target": incident.attribution.model_output(),
            }
        )
    rows.append(
        {
            "incident_id": incident.incident_id,
            "task": "structured_repair",
            "input": model_input,
            "target": repair.to_dict(),
        }
    )
    rows.extend(_gate_rows(incident, feedback=feedback))
    return rows


def _gate_rows(
    incident: IncidentRecord,
    *,
    feedback: str | None,
) -> list[dict[str, Any]]:
    repair = _required_repair(incident)
    cut_index = repair.cut_index(incident.events)
    first_bad_index = cut_index + 1
    if incident.attribution is not None and incident.attribution.first_bad_event_ids:
        first_bad = set(incident.attribution.first_bad_event_ids)
        first_bad_index = min(
            index
            for index, event in enumerate(incident.events)
            if event.event_id in first_bad
        )
    rows: list[dict[str, Any]] = []
    for index in range(len(incident.events)):
        rows.append(
            {
                "incident_id": incident.incident_id,
                "task": "deployment_gate",
                "input": {
                    **incident.model_input(feedback=feedback),
                    "events": [
                        event.to_dict() for event in incident.events[: index + 1]
                    ],
                },
                "target": {
                    "gate_probability": 1.0 if index >= first_bad_index else 0.0
                },
            }
        )
    repaired = incident.assemble(repair)
    for index in range(cut_index + 1, len(repaired)):
        rows.append(
            {
                "incident_id": incident.incident_id,
                "task": "deployment_gate",
                "input": {
                    **incident.model_input(feedback=feedback),
                    "events": [event.to_dict() for event in repaired[: index + 1]],
                },
                "target": {"gate_probability": 0.0},
            }
        )
    return rows


def _required_repair(incident: IncidentRecord) -> RepairRecord:
    if incident.repair is None:
        raise ValueError("training incident unexpectedly has no repair")
    return incident.repair
