"""Stage 2: outcome-grounded repair preference learning."""

from __future__ import annotations

from typing import Any

from aria.records import IncidentRecord


def build_repair_preference_row(
    incident: IncidentRecord,
    *,
    feedback: str | None,
) -> dict[str, Any]:
    """Pair a verified repair with its observed failed continuation."""

    repair = incident.repair
    if repair is None:
        raise ValueError("training incident unexpectedly has no repair")
    cut_index = repair.cut_index(incident.events)
    model_diagnosis = incident.metadata.get("model_diagnosis")
    has_model_diagnosis = isinstance(model_diagnosis, str) and bool(
        model_diagnosis.strip()
    )
    if not has_model_diagnosis:
        model_diagnosis = repair.diagnosis
    return {
        "incident_id": incident.incident_id,
        "task": "repair_preference",
        # Chosen and rejected serialize the same diagnosis and cut before their
        # continuations diverge.  Keeping the prompt answer-blind makes the DPO
        # contrast exactly the common-context outcome contrast of Section 3.3.2.
        "input": incident.model_input(feedback=feedback),
        "chosen": {
            **repair.to_dict(),
            "diagnosis": model_diagnosis,
        },
        "rejected": {
            "diagnosis": model_diagnosis,
            "cut_event_id": repair.cut_event_id,
            "replacement_events": [
                event.to_dict() for event in incident.events[cut_index + 1 :]
            ],
        },
        "diagnosis_source": "model" if has_model_diagnosis else "repair_fallback",
        "rejection_reason": "observed_failed_continuation",
    }
