"""Genealogy-based message governance from From Spark to Fire."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, ClassVar

from baselines.core.method import BaselineMethod
from baselines.core.models import ModelAdapter, require_keys
from baselines.core.replay import ReplayBackend, execute
from baselines.core.types import BaselineSample

SCREEN_SYSTEM = """Implement the Spark-to-Fire message governance representation and screening
layers. Decompose the current message into minimal atomic claims, compare each with confirmed
lineage, and assign Green (entailed), Red (contradicted), or Yellow (unverified). Return JSON only as
{"atoms": [{"claim_id": string, "text": string, "label": "Green"|"Red"|"Yellow",
"reason": string, "depends_on": [claim_id], "risk": number in [0,1]}]}. Return at most eight
claims found in current_event.content. Do not atomize the task, metadata, IDs, roles, tool arrays,
schemas, or confirmed-lineage records, and do not copy the input payload."""
VERIFY_SYSTEM = """Implement targeted verification for one Yellow atomic claim using the task,
confirmed lineage, and visible evidence. Return JSON only with claim_id, verdict (confirmed,
contradicted, or unresolved), reason, and corrected_text (string or null). Evaluate only the single
supplied claim and copy its claim_id exactly; do not substitute a related lineage claim ID."""
SCREEN_RETRY = """Return only {"atoms": [...]} with at most four concise atoms from
current_event.content. Do not copy the task, event object, lineage, metadata, or tool arrays."""
VERIFY_RETRY = """Return only claim_id, verdict, reason, and corrected_text for the one supplied
claim. Copy claim.claim_id exactly. Do not return claim, lineage, visible_prefix, or task."""


class SparkToFireGovernance(BaselineMethod):
    name = "spark_to_fire"
    setting = "active"
    provenance: ClassVar[dict[str, Any]] = {
        "paper": "From Spark to Fire: Modeling and Mitigating Error Cascades in LLM-Based Multi-Agent Collaboration (2026)",
        "paper_code": "https://anonymous.4open.science/r/From-spark-to-fire-6E0C",
        "adaptation": "message-layer plugin over normalized ARIA events",
    }

    def run(
        self,
        sample: BaselineSample,
        model: ModelAdapter | None,
        replay: ReplayBackend | None = None,
        *,
        start_index: int = 0,
        runtime_state: dict[str, Any] | None = None,
    ):
        if not 0 <= start_index <= len(sample.events):
            raise ValueError("Spark start_index must be inside the observed prefix")
        selected = self.require_model(model)
        policy = str(self.options.get("policy", "balanced"))
        if policy not in {"low", "balanced", "strict"}:
            raise ValueError("spark_to_fire policy must be low, balanced, or strict")
        retries = int(self.options.get("max_retries", 2))
        risk_threshold = float(self.options.get("risk_threshold", 0.5))
        lineage: dict[str, dict[str, Any]] = deepcopy((runtime_state or {}).get("lineage", {}))
        event_reports: list[dict[str, Any]] = []
        result = self.result(sample, status="no_intervention")
        for index, event in enumerate(sample.events):
            if index < start_index:
                continue
            if not event.content.strip() and not event.tool_calls:
                event_reports.append(
                    {
                        "event_id": event.event_id,
                        "atoms": [],
                        "verification": [],
                        "red_count": 0,
                        "skipped": "empty_event",
                    }
                )
                continue
            screened, screen_call = selected.json_call(
                stage=f"spark_to_fire.screen.{index:04d}",
                system=SCREEN_SYSTEM,
                payload={
                    "task": sample.task,
                    "current_event": event.to_dict(),
                    "confirmed_lineage": _recent_lineage(lineage, 16),
                },
                validator=lambda value, eid=event.event_id: _validate_atoms(
                    value, lineage, eid
                ),
                allow_truncated_containers=True,
                final_retry_instruction=SCREEN_RETRY,
            )
            result.model_calls.append(screen_call)
            atoms = screened["atoms"]
            red: list[dict[str, Any]] = [
                atom for atom in atoms if atom["label"] == "Red"
            ]
            released: list[dict[str, Any]] = []
            verification: list[dict[str, Any]] = []
            for atom in atoms:
                if atom["label"] == "Green":
                    lineage[atom["claim_id"]] = {
                        **atom,
                        "source_event_id": event.event_id,
                        "status": "confirmed",
                    }
                    released.append(atom)
                    continue
                if atom["label"] == "Red":
                    continue
                should_verify = policy == "strict" or (
                    policy == "balanced" and float(atom["risk"]) >= risk_threshold
                )
                if not should_verify:
                    released.append({**atom, "status": "released_unconfirmed"})
                    continue
                verdict, verify_call = selected.json_call(
                    stage=f"spark_to_fire.verify.{index:04d}.{atom['claim_id']}",
                    system=VERIFY_SYSTEM,
                    payload={
                        "task": sample.task,
                        "claim": atom,
                        "confirmed_lineage": [
                            {
                                "text": item.get("text"),
                                "status": item.get("status"),
                            }
                            for item in _recent_lineage(lineage, 8)
                        ],
                        "visible_prefix": [
                            {
                                "role": item.role,
                                "kind": item.kind,
                                "content": item.content,
                            }
                            for item in sample.events[max(0, index - 7) : index + 1]
                        ],
                    },
                    validator=lambda value, cid=atom["claim_id"]: _validate_verdict(
                        value, cid
                    ),
                    final_retry_instruction=VERIFY_RETRY,
                )
                result.model_calls.append(verify_call)
                verification.append(verdict)
                if verdict["verdict"] == "confirmed":
                    confirmed = {
                        **atom,
                        "source_event_id": event.event_id,
                        "status": "confirmed",
                    }
                    lineage[atom["claim_id"]] = confirmed
                    released.append(confirmed)
                elif verdict["verdict"] == "contradicted":
                    red.append(
                        {
                            **atom,
                            "reason": verdict["reason"],
                            "corrected_text": verdict["corrected_text"],
                        }
                    )
                else:
                    released.append({**atom, "status": "released_high_risk"})
            event_reports.append(
                {
                    "event_id": event.event_id,
                    "atoms": atoms,
                    "verification": verification,
                    "red_count": len(red),
                }
            )
            if not red:
                continue
            intervention = {
                "type": "message_layer_governance",
                "target_event_id": event.event_id,
                "action": "block_and_rollback",
                "sanitized_content": _sanitized_content(released, red),
                "feedback_package": {
                    "rejected_atoms": red,
                    "rewrite_directive": "remove contradicted claims and resubmit",
                    "max_retries": retries,
                    "circuit_breaker": "exclude persistent Red; tag persistent Yellow high-risk",
                },
                "lineage_graph": list(lineage.values()),
            }
            result.intervention = intervention
            result.status = "proposed"
            result.diagnostics["trigger_event_id"] = event.event_id
            execution = execute(
                replay, sample, intervention, label="spark-to-fire-governance"
            )
            if execution is not None:
                result.executions.append(execution)
                result.selected_execution_label = execution.label
                result.status = "executed"
            break
        result.diagnostics["policy"] = policy
        result.diagnostics["event_reports"] = event_reports
        result.diagnostics["lineage_graph"] = list(lineage.values())
        if runtime_state is not None:
            runtime_state["lineage"] = lineage
        return result


def _sanitized_content(
    released: list[dict[str, Any]], red: list[dict[str, Any]]
) -> str:
    """The message the sender is rolled back to, never an empty one.

    Green and confirmed atoms pass through, and a contradicted atom passes only
    in its corrected form.  When screening rejects every atom there is nothing
    left to release, so the block degenerates into the paper's feedback-and-retry
    path: the sender is handed the rejected claims and asked to resubmit.  An
    empty replacement would instead hand the host nothing to execute.
    """

    kept = [str(atom["text"]) for atom in released]
    kept += [str(atom["corrected_text"]) for atom in red if atom.get("corrected_text")]
    if kept:
        return "\n".join(kept)
    rejected = "; ".join(f"{atom['text']} ({atom['reason']})" for atom in red)
    return (
        "Every claim in the previous message was rejected by message-layer screening "
        f"and none of it entered shared state. Rejected claims: {rejected}. "
        "Redo this step from the confirmed lineage and resubmit without them."
    )


def _recent_lineage(
    lineage: dict[str, dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    """Return recent unique lineage claims without repeating identical text."""

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in reversed(list(lineage.values())):
        text = str(item.get("text") or "")
        if not text or text in seen:
            continue
        seen.add(text)
        selected.append(item)
        if len(selected) == limit:
            break
    return list(reversed(selected))


def _validate_atoms(
    raw: Any, lineage: dict[str, dict[str, Any]], event_id: str
) -> dict[str, Any]:
    value = require_keys(raw, {"atoms"}, name="Spark atoms")
    if not isinstance(value["atoms"], list):
        raise TypeError("Spark atoms must be a list")
    local: set[str] = set()
    claim_aliases: dict[str, str] = {}
    normalizations: list[str] = []
    normalized_atoms: list[dict[str, Any]] = []
    for index, raw_atom in enumerate(value["atoms"]):
        if not isinstance(raw_atom, dict):
            normalizations.append(f"atom[{index}]:drop_non_object")
            continue
        atom = dict(raw_atom)
        if "claim_id" not in atom:
            atom["claim_id"] = f"claim:{event_id}:{index:04d}"
            normalizations.append(f"atom[{index}]:default_claim_id")
        if "text" not in atom:
            atom["text"] = atom.get("claim") or atom.get("content") or ""
            normalizations.append(f"atom[{index}]:claim->text")
        if "reason" not in atom:
            atom["reason"] = atom.get("explanation") or "screened claim"
            normalizations.append(f"atom[{index}]:default_reason")
        if "depends_on" not in atom:
            atom["depends_on"] = []
            normalizations.append(f"atom[{index}]:default_depends_on")
        label = str(atom.get("label") or "Yellow").strip().lower()
        label_aliases = {
            "green": "Green",
            "entailed": "Green",
            "confirmed": "Green",
            "true": "Green",
            "red": "Red",
            "contradicted": "Red",
            "false": "Red",
            "yellow": "Yellow",
            "unverified": "Yellow",
            "uncertain": "Yellow",
            "unknown": "Yellow",
        }
        if label in label_aliases:
            atom["label"] = label_aliases[label]
            if atom["label"] != raw_atom.get("label"):
                normalizations.append(f"atom[{index}].label->{atom['label']}")
        require_keys(
            atom,
            {"claim_id", "text", "label", "reason", "depends_on", "risk"},
            name="atom",
        )
        claim_id = str(atom["claim_id"])
        if len(claim_id) > 128 or re.fullmatch(r"[A-Za-z0-9_.:-]+", claim_id) is None:
            original = claim_id
            claim_id = f"claim:{event_id}:{index:04d}"
            atom["claim_id"] = claim_id
            claim_aliases[original] = claim_id
            normalizations.append(f"atom[{index}]:canonicalize_claim_id")
        if claim_id in lineage or claim_id in local:
            original = claim_id
            claim_id = f"claim:{event_id}:{index:04d}"
            suffix = 0
            while claim_id in lineage or claim_id in local:
                suffix += 1
                claim_id = f"claim:{event_id}:{index:04d}:{suffix}"
            atom["claim_id"] = claim_id
            normalizations.append(f"atom[{index}]:deduplicate_claim_id")
        local.add(claim_id)
        if atom["label"] not in {"Green", "Red", "Yellow"}:
            raise ValueError("Spark atom label is invalid")
        if not isinstance(atom["depends_on"], list):
            raise TypeError("Spark atom depends_on must be a list")
        if not isinstance(atom["risk"], (int, float)) or isinstance(atom["risk"], bool):
            raise TypeError("Spark atom risk must be numeric")
        atom["risk"] = float(atom["risk"])
        if not 0 <= atom["risk"] <= 1:
            raise ValueError("Spark atom risk must be between 0 and 1")
        normalized_atoms.append(atom)
        if len(normalized_atoms) == 8:
            if index + 1 < len(value["atoms"]):
                normalizations.append("atoms:truncate_to_8")
            break
    for index, atom in enumerate(normalized_atoms):
        remapped = [
            claim_aliases.get(str(dependency), str(dependency))
            for dependency in atom["depends_on"]
        ]
        if remapped != atom["depends_on"]:
            atom["depends_on"] = remapped
            normalizations.append(f"atom[{index}]:remap_depends_on")
    value["atoms"] = normalized_atoms
    if normalizations:
        value["_schema_normalizations"] = normalizations
    return value


def _validate_verdict(raw: Any, claim_id: str) -> dict[str, Any]:
    value = dict(
        require_keys(
            raw,
            {"claim_id", "verdict", "reason", "corrected_text"},
            name="verification",
        )
    )
    normalizations: list[str] = []
    if str(value["claim_id"]) != claim_id:
        value["claim_id"] = claim_id
        normalizations.append("single_claim_scope->claim_id")
    if value["verdict"] not in {"confirmed", "contradicted", "unresolved"}:
        raise ValueError("verification verdict is invalid")
    if value["corrected_text"] is not None and not isinstance(
        value["corrected_text"], str
    ):
        raise TypeError("verification corrected_text must be string or null")
    if normalizations:
        value["_schema_normalizations"] = normalizations
    return value
