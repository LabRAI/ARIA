"""Stage 3: transfer-calibrated diagnosis and internalization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from aria.records import IncidentRecord


@dataclass(frozen=True)
class DiagnosisChoice:
    chosen: str
    rejected: str
    uplift: float
    heldout_uplift: float


def build_transfer_diagnosis_rows(
    incident: IncidentRecord,
    *,
    feedback: str | None,
    uplift_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Build diagnosis DPO followed by diagnosis-internalization SFT data."""

    repair = incident.repair
    if repair is None:
        raise ValueError("training incident unexpectedly has no repair")
    choice = select_diagnosis_choice(incident, uplift_threshold)
    if choice is None:
        return None
    repair_context = {
        "cut_event_id": repair.cut_event_id,
        "replacement_events": [event.to_dict() for event in repair.replacement_events],
    }
    diagnosis_dpo = {
        "incident_id": incident.incident_id,
        "task": "transfer_diagnosis_preference",
        "input": {
            **incident.model_input(feedback=feedback),
            "repair_context": repair_context,
        },
        "chosen": {"diagnosis": choice.chosen},
        "rejected": {"diagnosis": choice.rejected},
        "uplift": choice.uplift,
        "heldout_uplift": choice.heldout_uplift,
    }
    internalization_sft = {
        "incident_id": incident.incident_id,
        "task": "transfer_diagnosis_internalization",
        "input": incident.model_input(feedback=feedback),
        "target": {
            "diagnosis": choice.chosen,
            **repair_context,
        },
        "uplift": choice.uplift,
        "heldout_uplift": choice.heldout_uplift,
    }
    return diagnosis_dpo, internalization_sft


def select_diagnosis_choice(
    incident: IncidentRecord,
    uplift_threshold: float,
) -> DiagnosisChoice | None:
    """Validate externally scored sibling-transfer candidates and select a pair."""

    candidates_raw = incident.metadata.get("diagnosis_candidates", [])
    family_size = incident.metadata.get("family_size")
    if family_size is not None and (
        not isinstance(family_size, int) or family_size < 2
    ):
        return None
    if not isinstance(candidates_raw, list) or len(candidates_raw) < 2:
        return None
    candidates: list[tuple[str, float, float]] = []
    for index, candidate in enumerate(candidates_raw):
        if not isinstance(candidate, Mapping):
            raise TypeError(f"diagnosis_candidates[{index}] must be a mapping")
        text = candidate.get("diagnosis")
        uplift = candidate.get("uplift")
        heldout = candidate.get("heldout_uplift")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"diagnosis_candidates[{index}].diagnosis must be non-empty"
            )
        if not isinstance(uplift, (int, float)) or not isinstance(
            heldout, (int, float)
        ):
            raise TypeError(
                f"diagnosis_candidates[{index}] requires numeric uplift and heldout_uplift"
            )
        candidates.append((text, float(uplift), float(heldout)))
    chosen = max(candidates, key=lambda item: (item[1], -len(item[0])))
    rejected = min(candidates, key=lambda item: (item[1], len(item[0])))
    if chosen[1] < uplift_threshold or chosen[2] <= 0.0:
        return None
    if chosen[0] == rejected[0]:
        return None
    return DiagnosisChoice(
        chosen=chosen[0],
        rejected=rejected[0],
        uplift=chosen[1],
        heldout_uplift=chosen[2],
    )
