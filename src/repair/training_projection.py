"""Project verified repair-dataset-v6 episodes into ARIA SFT and DPO rows."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from aria.records import (
    FailureLocalization,
    IncidentRecord,
    RepairAttribution,
    RepairRecord,
    TrajectoryEvent,
)
from common.artifacts import atomic_write_json, atomic_write_jsonl, utc_now
from repair.schema import REPAIR_SCHEMA_VERSION, TRAINING_READY_SCHEMA_VERSION

MAX_REJECTED_SUFFIX_EVENTS = 3
MAX_REJECTED_EVENT_CHARS = 2000


def write_training_ready(
    dataset_root: Path,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output_root = dataset_root / "training_ready"
    output_root.mkdir(parents=True, exist_ok=True)
    sft_rows: list[dict[str, Any]] = []
    dpo_rows: list[dict[str, Any]] = []
    counts = {
        "deployment_gate": 0,
        "failure_localization": 0,
        "repair_attribution": 0,
        "structured_repair": 0,
        "repair_preference": 0,
    }
    for record in records:
        projected_sft, projected_dpo = project_record(record)
        sft_rows.extend(projected_sft)
        dpo_rows.extend(projected_dpo)
        for row in projected_sft:
            counts[str(row["task"])] += 1
        counts["repair_preference"] += len(projected_dpo)
    sft_path = output_root / "sft.jsonl"
    dpo_path = output_root / "dpo.jsonl"
    atomic_write_jsonl(sft_path, sft_rows)
    atomic_write_jsonl(dpo_path, dpo_rows)
    manifest = {
        "schema_version": "aria-training-ready-manifest-v3",
        "dataset_schema_version": TRAINING_READY_SCHEMA_VERSION,
        "source_dataset_schema_version": REPAIR_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "source_dataset_dir": str(dataset_root),
        "counts": {**counts, "sft": len(sft_rows), "dpo": len(dpo_rows)},
        "datasets": {
            "sft": _file_manifest(sft_path, len(sft_rows)),
            "dpo": _file_manifest(dpo_path, len(dpo_rows)),
        },
        "selection_policy": {
            "gate": "observable prefixes only; post-hoc evaluator fields removed",
            "localization_sft": "host-grounded LLM error localization per repair trial",
            "attribution_sft": "schema-valid RA attribution per repair trial",
            "structured_repair_sft": "host-verified patched_only trials only",
            "repair_dpo": "same prompt/schema; patched_only chosen",
        },
    }
    atomic_write_json(output_root / "manifest.json", manifest)
    return manifest


def project_record(
    record: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if record.get("record_type") in {"gate_success_sample", "success_noop"}:
        incident = _incident(record)
        return _gate_rows(record, incident, first_bad_index=None, all_negative=True), []
    incident = _incident(record)
    training = _mapping(record.get("training"))
    sft: list[dict[str, Any]] = []
    dpo: list[dict[str, Any]] = []
    trial_rows = _repair_trials(record)
    selected_trial_id = record.get("selected_repair_trial_id")
    selected_trial = next(
        (
            trial
            for trial in trial_rows
            if trial.get("trial_id") == selected_trial_id
        ),
        trial_rows[0] if trial_rows else None,
    )
    selected_attribution = (
        _trial_attribution(selected_trial, incident)
        if isinstance(selected_trial, Mapping)
        else None
    )
    first_bad_index = (
        _first_bad_index(incident, selected_attribution)
        if selected_attribution is not None
        else None
    )
    if training.get("gate_sft_eligible") is True:
        sft.extend(
            _gate_rows(
                record,
                incident,
                first_bad_index=first_bad_index,
                all_negative=False,
            )
        )
    for trial in trial_rows:
        trial_id = str(trial.get("trial_id") or "repair-trial:00")
        localization = _trial_localization(trial, incident)
        attribution = _trial_attribution(trial, incident)
        if localization is None or attribution is None:
            continue
        metadata = _training_metadata(
            record,
            trial=trial,
            trial_id=trial_id,
            attribution=attribution,
        )
        if training.get("localization_sft_eligible") is True:
            sft.append(
                {
                    "schema_version": TRAINING_READY_SCHEMA_VERSION,
                    "record_id": f"{record['record_id']}::{trial_id}::localization",
                    "incident_id": incident.incident_id,
                    "task": "failure_localization",
                    "input": incident.model_input(),
                    "target": localization.to_dict(),
                    "metadata": metadata,
                }
            )
        if training.get("attribution_sft_eligible") is True:
            sft.append(
                {
                    "schema_version": TRAINING_READY_SCHEMA_VERSION,
                    "record_id": f"{record['record_id']}::{trial_id}::attribution",
                    "incident_id": incident.incident_id,
                    "task": "repair_attribution",
                    "input": incident.attribution_model_input(localization),
                    "target": attribution.model_output(),
                    "metadata": metadata,
                }
            )
        repair_raw = trial.get("repair")
        if not isinstance(repair_raw, Mapping):
            continue
        repair = RepairRecord.from_dict(
            repair_raw,
            events=incident.events,
            name=f"{record.get('record_id')}.{trial_id}.patch",
        )
        patch_input = incident.patch_model_input(attribution)
        verified = trial.get("verified_successful") is True
        if verified and training.get("structured_repair_sft_eligible") is True:
            sft.append(
                {
                    "schema_version": TRAINING_READY_SCHEMA_VERSION,
                    "record_id": f"{record['record_id']}::{trial_id}::patch",
                    "incident_id": incident.incident_id,
                    "task": "structured_repair",
                    "input": patch_input,
                    "target": repair.to_dict(),
                    "metadata": metadata,
                }
            )
        if verified and training.get("repair_dpo_eligible") is True:
            rejected, rejection_reason = _rejected_patch(
                trial_rows, incident, repair, selected_trial_id=trial_id
            )
            dpo.append(
                {
                    "schema_version": TRAINING_READY_SCHEMA_VERSION,
                    "record_id": f"{record['record_id']}::{trial_id}::preference",
                    "incident_id": incident.incident_id,
                    "task": "repair_preference",
                    "input": patch_input,
                    "chosen": repair.to_dict(),
                    "rejected": rejected.to_dict(),
                    "rejection_reason": rejection_reason,
                    "metadata": metadata,
                }
            )
    if (
        training.get("gate_sft_eligible") is True
        and selected_attribution is not None
        and selected_trial is not None
        and selected_trial.get("verified_successful") is True
    ):
        selected_repair_raw = selected_trial.get("repair")
        if isinstance(selected_repair_raw, Mapping):
            selected_repair = RepairRecord.from_dict(
                selected_repair_raw,
                events=incident.events,
                name=f"{record.get('record_id')}.selected_patch",
            )
            repaired_events = incident.assemble(selected_repair)
            for offset in range(
                selected_repair.cut_index(incident.events) + 1,
                len(repaired_events),
            ):
                sft.append(
                    _gate_row(
                        record,
                        incident,
                        events=repaired_events[: offset + 1],
                        probability=0.0,
                        suffix=f"repaired-{offset:04d}",
                    )
                )
    return sft, dpo


def _gate_rows(
    record: Mapping[str, Any],
    incident: IncidentRecord,
    *,
    first_bad_index: int | None,
    all_negative: bool,
) -> list[dict[str, Any]]:
    if not incident.events:
        return []
    if first_bad_index is None and not all_negative:
        indexes = [len(incident.events) - 1]
    else:
        indexes = list(range(len(incident.events)))
    rows = []
    for index in indexes:
        probability = (
            0.0
            if all_negative
            else 1.0
            if first_bad_index is None or index >= first_bad_index
            else 0.0
        )
        rows.append(
            _gate_row(
                record,
                incident,
                events=incident.events[: index + 1],
                probability=probability,
                suffix=f"source-{index:04d}",
            )
        )
    return rows


def _gate_row(
    record: Mapping[str, Any],
    incident: IncidentRecord,
    *,
    events: Sequence[TrajectoryEvent],
    probability: float,
    suffix: str,
) -> dict[str, Any]:
    model_input = incident.model_input(feedback=None)
    constraints = dict(_mapping(model_input.get("constraints")))
    constraints.pop("evaluation", None)
    model_input["constraints"] = constraints
    model_input["events"] = [event.to_dict() for event in events]
    return {
        "schema_version": TRAINING_READY_SCHEMA_VERSION,
        "record_id": f"{record['record_id']}::gate::{suffix}",
        "incident_id": incident.incident_id,
        "task": "deployment_gate",
        "input": model_input,
        "target": {"gate_probability": probability},
        "metadata": _training_metadata(record),
    }


def _repair_trials(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = record.get("repair_trials")
    if isinstance(raw, list):
        return [trial for trial in raw if isinstance(trial, Mapping)]
    return []


def _trial_localization(
    trial: Mapping[str, Any], incident: IncidentRecord
) -> FailureLocalization | None:
    raw = trial.get("localization")
    if not isinstance(raw, Mapping):
        return None
    return FailureLocalization.from_dict(
        raw,
        events=incident.events,
        name=f"{trial.get('trial_id')}.localization",
    )


def _trial_attribution(
    trial: Mapping[str, Any], incident: IncidentRecord
) -> RepairAttribution | None:
    localization = _trial_localization(trial, incident)
    raw = trial.get("attribution")
    if localization is None or not isinstance(raw, Mapping):
        return None
    allowed = {
        key: value
        for key, value in raw.items()
        if key
        in {
            "diagnosis",
            "primary_code",
            "secondary_codes",
            "responsible_agents",
            "counterfactual",
            "patch_operator",
            "confidence",
        }
    }
    return RepairAttribution.from_dict(
        allowed,
        events=incident.events,
        localization=localization,
        name=f"{trial.get('trial_id')}.attribution",
    )


def _incident(record: Mapping[str, Any]) -> IncidentRecord:
    raw = record.get("incident")
    if not isinstance(raw, Mapping):
        raise TypeError(f"{record.get('record_id')}.incident must be a mapping")
    return IncidentRecord.from_dict(raw)


def _first_bad_index(
    incident: IncidentRecord, attribution: RepairAttribution
) -> int | None:
    if not attribution.first_bad_event_ids:
        return None
    ids = set(attribution.first_bad_event_ids)
    return min(
        index for index, event in enumerate(incident.events) if event.event_id in ids
    )


def _rejected_patch(
    trials: Sequence[Mapping[str, Any]],
    incident: IncidentRecord,
    selected: RepairRecord,
    *,
    selected_trial_id: str,
) -> tuple[RepairRecord, str]:
    for trial in trials:
        if trial.get("trial_id") == selected_trial_id:
            continue
        if trial.get("paired_outcome") not in {"both_fail", "control_only"}:
            continue
        repair_raw = trial.get("repair")
        if not isinstance(repair_raw, Mapping):
            continue
        try:
            return (
                RepairRecord.from_dict(
                    repair_raw,
                    events=incident.events,
                    name="DPO rejected repair trial",
                ),
                str(trial.get("paired_outcome")),
            )
        except (TypeError, ValueError):
            continue
    return (
        _observed_failed_continuation(incident, selected),
        "observed_failed_continuation",
    )


def _observed_failed_continuation(
    incident: IncidentRecord, selected: RepairRecord
) -> RepairRecord:
    """Convert the observed failed suffix into a schema-matched DPO negative.

    The suffix is bounded before validation, and progressively narrowed when a
    verbose trajectory still exceeds the structured-patch limits, so a verified
    patch never loses its preference pair to an unbounded observed suffix.
    """

    cut_index = selected.cut_index(incident.events)
    suffix = incident.events[
        cut_index + 1 : cut_index + 1 + MAX_REJECTED_SUFFIX_EVENTS
    ]
    candidates: list[list[dict[str, Any]]] = []
    for keep_tool_calls in (True, False):
        for length in range(len(suffix), 0, -1):
            candidates.append(
                _bounded_suffix_events(suffix[:length], keep_tool_calls=keep_tool_calls)
            )
    candidates.append(
        [
            {
                "event_id": "patch:event:0001",
                "role": "assistant",
                "kind": "message",
                "content": "Terminate without applying the required repair.",
                "tool_calls": [],
                "metadata": {},
            }
        ]
    )
    last_error: Exception | None = None
    for replacement_events in candidates:
        try:
            return RepairRecord.from_dict(
                {
                    "diagnosis": selected.diagnosis,
                    "cut_event_id": selected.cut_event_id,
                    "replacement_events": replacement_events,
                },
                events=incident.events,
                name="observed failed continuation",
            )
        except (TypeError, ValueError) as exc:
            last_error = exc
    raise ValueError(
        f"observed failed continuation is not schema-valid: {last_error}"
    )


def _bounded_suffix_events(
    suffix: Sequence[TrajectoryEvent], *, keep_tool_calls: bool
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for index, event in enumerate(suffix, start=1):
        content = event.content[:MAX_REJECTED_EVENT_CHARS].strip()
        tool_calls = (
            [dict(item) for item in event.tool_calls]
            if keep_tool_calls and event.kind != "tool_result"
            else []
        )
        events.append(
            {
                "event_id": f"patch:event:{index:04d}",
                "role": event.role,
                "kind": "message" if event.kind == "tool_result" else event.kind,
                "content": content or "Continue the observed failed suffix.",
                "tool_calls": tool_calls,
                "metadata": {"replaces": [event.event_id]},
            }
        )
    return events


def _training_metadata(
    record: Mapping[str, Any],
    *,
    trial: Mapping[str, Any] | None = None,
    trial_id: str | None = None,
    attribution: RepairAttribution | None = None,
) -> dict[str, Any]:
    """Describe one projected row from its own trial, never from a sibling.

    Rows are per repair trial, so the RA family, paired outcome, and execution
    mode must come from the trial that produced the target.  Only record-scoped
    rows such as gate prefixes fall back to the canonical selected trial.
    """

    refs = _mapping(record.get("source_refs"))
    record_attribution = _mapping(record.get("attribution"))
    outcome = _mapping(record.get("outcome"))
    trial_outcome = _mapping(trial)
    primary_code = (
        attribution.primary_code
        if attribution is not None
        else record_attribution.get("primary_code")
    )
    return {
        "source_record_id": (
            f"{record.get('record_id')}::{trial_id}"
            if trial_id is not None
            else record.get("record_id")
        ),
        "repair_trial_id": trial_id,
        "split": record.get("split"),
        "record_type": record.get("record_type"),
        "family_id": primary_code,
        "primary_code": primary_code,
        "paired_outcome": (
            trial_outcome.get("paired_outcome")
            if trial is not None
            else outcome.get("paired_outcome")
        ),
        "verified_successful": (
            trial_outcome.get("verified_successful") is True
            if trial is not None
            else outcome.get("verified_successful") is True
        ),
        "mas_id": refs.get("mas_id"),
        "benchmark_id": refs.get("benchmark_id"),
        "repair_operator_id": refs.get("repair_operator_id"),
        "execution_mode": (
            _mapping(trial_outcome.get("execution")).get("execution_mode")
            or outcome.get("execution_mode")
            if trial is not None
            else outcome.get("execution_mode")
        ),
    }


def _file_manifest(path: Path, count: int) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": path.name,
        "records": count,
        "bytes": len(data),
        "sha256": f"sha256:{hashlib.sha256(data).hexdigest()}",
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
