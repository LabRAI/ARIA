"""Paper-described ErrorProbe with verified, in-process episodic memory."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Any, ClassVar

from baselines.core.method import BaselineMethod, validate_attribution
from baselines.core.models import ModelAdapter, require_keys
from baselines.core.prompting import repair_from_context
from baselines.core.replay import ReplayBackend, execute
from baselines.core.types import BaselineSample

TAG_SYSTEM = """ErrorProbe phase 1: operationalize MAST as local anomaly tags. Parse every event
into agent, role, and action; identify lightweight MAST symptom tags and the terminal visible failure
symptom. Return JSON only with tags (event_id, mast_tag, reason) and symptom_event_ids. Keep each
reason under 12 words and never copy event content or the input trajectory into the response."""
DEPENDENCY_SYSTEM = """ErrorProbe phase 1/2: parse information-flow dependencies in this trace.
Return JSON only as {"edges": [{"source": earlier_event_id, "target": later_event_id,
"reason": string}]}. Include only visible causal or information-use relations."""
STRATEGIST_SYSTEM = """ErrorProbe Strategist: using only the backward-traced condensed context,
MAST priors, and retrieved verified patterns, propose root-cause hypotheses. Return JSON only as
{"hypotheses": [{"hypothesis_id": string, "event_id": string, "responsible_agent": string,
"suspected_mode": string, "claim": string}]}.
"""
INVESTIGATOR_SYSTEM = """ErrorProbe Investigator: turn one hypothesis into testable evidence.
Return JSON only with hypothesis_id and probes, where each probe has type (code_exec or
logic_probe), event_id, specification, and expected_observation. Do not claim a probe was executed."""
ARBITER_SYSTEM = """ErrorProbe Arbiter: adjudicate hypotheses using the supplied host-verified probe
results. Discard empty or inconclusive evidence. Return JSON only with responsible_agent,
zero-based step_index, reason, failure_mode, and confidence in [0,1]."""
TAG_RETRY = """Return only tags and symptom_event_ids. Use at most eight concise tag objects; do
not copy task, events, constraints, feedback, or input fields."""
DEPENDENCY_RETRY = """Return only {"edges": [...]} with at most eight concise visible dependency
edges. Do not copy events or other input fields."""
STRATEGY_RETRY = """Return only {"hypotheses": [...]} with at most two concise hypothesis objects.
Do not copy task, events, tags, memory, or input fields."""
INVESTIGATOR_RETRY = """Return only hypothesis_id and exactly one concise probes item with type,
event_id, specification, and expected_observation. Do not copy sample, hypothesis, or events."""
ARBITER_RETRY = """Return only responsible_agent, zero-based step_index, reason, failure_mode, and
confidence. Do not return hypotheses, evidence, events, or input fields."""


class ErrorProbeRepair(BaselineMethod):
    name = "errorprobe"
    setting = "passive"
    provenance: ClassVar[dict[str, Any]] = {
        "paper": "Towards Self-Improving Error Diagnosis in Multi-Agent Systems (2026)",
        "implementation": "paper description; no official code repository identified",
        "pipeline": "MAST decomposition -> backward trace -> Strategist/Investigator/Arbiter -> verified memory",
    }

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        initial = options.get("initial_memory", [])
        if not isinstance(initial, list):
            raise TypeError("errorprobe initial_memory must be a list")
        self.memory: list[dict[str, Any]] = [
            dict(item) for item in initial if isinstance(item, Mapping)
        ]

    def run(
        self,
        sample: BaselineSample,
        model: ModelAdapter | None,
        replay: ReplayBackend | None = None,
    ):
        selected = self.require_model(model)
        result = self.result(sample, status="proposed")
        tags, tag_call = selected.json_call(
            stage="errorprobe.mast_decomposition",
            system=TAG_SYSTEM,
            payload=sample.model_view(include_feedback=True),
            validator=lambda value: _validate_tags(value, sample),
            allow_truncated_containers=True,
            final_retry_instruction=TAG_RETRY,
        )
        result.model_calls.append(tag_call)
        dependencies, dependency_call = selected.json_call(
            stage="errorprobe.dependency_parsing",
            system=DEPENDENCY_SYSTEM,
            payload={"events": [event.to_dict() for event in sample.events]},
            validator=lambda value: _validate_edges(value, sample),
            allow_truncated_containers=True,
            final_retry_instruction=DEPENDENCY_RETRY,
        )
        result.model_calls.append(dependency_call)
        receptive_ids = _backward_trace(
            tags["symptom_event_ids"], dependencies["edges"]
        )
        condensed_events = [
            event.to_dict()
            for event in sample.events
            if event.event_id in receptive_ids
        ]
        signature = _signature(tags, sample)
        retrieved = _retrieve(
            self.memory, signature, int(self.options.get("memory_k", 5))
        )
        strategy, strategist_call = selected.json_call(
            stage="errorprobe.strategist",
            system=STRATEGIST_SYSTEM,
            payload={
                "task": sample.task,
                "condensed_events": condensed_events,
                "mast_tags": tags["tags"],
                "verified_memory": retrieved,
            },
            validator=lambda value: _validate_hypotheses(value, sample, receptive_ids),
            allow_truncated_containers=True,
            final_retry_instruction=STRATEGY_RETRY,
        )
        result.model_calls.append(strategist_call)
        evidence: list[dict[str, Any]] = []
        for hypothesis in strategy["hypotheses"]:
            investigation, investigator_call = selected.json_call(
                stage=f"errorprobe.investigator.{hypothesis['hypothesis_id']}",
                system=INVESTIGATOR_SYSTEM,
                payload={
                    "sample": sample.model_view(include_feedback=True),
                    "hypothesis": hypothesis,
                },
                validator=lambda value, hid=hypothesis["hypothesis_id"]: (
                    _validate_probes(value, hid)
                ),
                allow_truncated_containers=True,
                final_retry_instruction=INVESTIGATOR_RETRY,
            )
            result.model_calls.append(investigator_call)
            for probe in investigation["probes"]:
                verifier = getattr(replay, "verify_evidence", None)
                if callable(verifier):
                    observed = dict(verifier(sample, probe))
                    verified = observed.get("verified") is True
                else:
                    observed = {"verified": False, "reason": "no host evidence backend"}
                    verified = False
                evidence.append(
                    {
                        "hypothesis_id": hypothesis["hypothesis_id"],
                        "probe": probe,
                        "observed": observed,
                        "verified": verified,
                    }
                )
        attribution, arbiter_call = selected.json_call(
            stage="errorprobe.arbiter",
            system=ARBITER_SYSTEM,
            payload={
                "events": [event.to_dict() for event in sample.events],
                "hypotheses": strategy["hypotheses"],
                "evidence": evidence,
            },
            validator=lambda value: _validate_arbiter(value, sample),
            allow_truncated_containers=True,
            final_retry_instruction=ARBITER_RETRY,
        )
        result.model_calls.append(arbiter_call)
        result.attribution = attribution
        threshold = float(self.options.get("memory_threshold", 0.7))
        committed = attribution["confidence"] >= threshold and any(
            item["verified"] for item in evidence
        )
        if committed:
            self.memory.append(
                {
                    "signature": signature,
                    "diagnosis": attribution,
                    "evidence": evidence,
                    "frequency": 1,
                }
            )
        result.diagnostics.update(
            {
                "mast_tags": tags["tags"],
                "dependency_edges": dependencies["edges"],
                "effective_receptive_field": sorted(receptive_ids),
                "retrieved_memory_count": len(retrieved),
                "verified_evidence": evidence,
                "memory_committed": committed,
                "memory_size": len(self.memory),
            }
        )
        intervention, repair_call = repair_from_context(
            selected, sample, stage="errorprobe.shared_repair", attribution=attribution
        )
        result.model_calls.append(repair_call)
        result.intervention = intervention
        execution = execute(replay, sample, intervention, label="errorprobe-repair")
        if execution is not None:
            result.executions.append(execution)
            result.selected_execution_label = execution.label
            result.status = "executed"
        return result


def _validate_tags(raw: Any, sample: BaselineSample) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError("ErrorProbe tags must be an object")
    value = dict(raw)
    normalizations: list[str] = []
    raw_tags = value.get("tags")
    event_tags = value.get("events")
    if raw_tags is None and isinstance(event_tags, list):
        tagged_events = [
            tag
            for tag in event_tags
            if isinstance(tag, Mapping)
            and "event_id" in tag
            and any(key in tag for key in ("mast_tag", "tag", "label"))
        ]
        if tagged_events:
            raw_tags = tagged_events
            normalizations.append("tagged_events->tags")
    if raw_tags is None and all(key in value for key in ("event_id", "mast_tag")):
        # Copy before assigning ``tags`` below; putting ``value`` itself in
        # the list creates a self-referential dict and breaks JSON persistence.
        raw_tags = [dict(value)]
        normalizations.append("tag_object->tags")
    if raw_tags is None:
        raise KeyError("ErrorProbe tags missing keys: tags")
    if not isinstance(raw_tags, list):
        raw_tags = [raw_tags]
        normalizations.append("tag_object->tags")
    symptoms = value.get("symptom_event_ids")
    if symptoms is None:
        for tag in raw_tags:
            if isinstance(tag, Mapping) and isinstance(tag.get("symptom_event_ids"), list):
                symptoms = [*tag["symptom_event_ids"]]
                normalizations.append("tag.symptom_event_ids->symptom_event_ids")
                break
    if symptoms is None:
        symptoms = []
        normalizations.append("default_symptom_event_ids->last_event")
    value["tags"] = raw_tags
    value["symptom_event_ids"] = symptoms
    value = require_keys(value, {"tags", "symptom_event_ids"}, name="ErrorProbe tags")
    event_ids = {event.event_id for event in sample.events}
    if not isinstance(value["tags"], list) or not isinstance(
        value["symptom_event_ids"], list
    ):
        raise TypeError("ErrorProbe tags and symptom_event_ids must be lists")
    normalized_tags: list[dict[str, Any]] = []
    for index, raw_tag in enumerate(value["tags"]):
        if not isinstance(raw_tag, Mapping):
            raise TypeError("ErrorProbe tag must be an object")
        tag = dict(raw_tag)
        if "event_id" not in tag and isinstance(tag.get("step_index"), int):
            step = tag["step_index"]
            if 0 <= step < len(sample.events):
                tag["event_id"] = sample.events[step].event_id
                normalizations.append(f"tag[{index}].step_index->event_id")
        if "mast_tag" not in tag:
            tag["mast_tag"] = str(tag.get("tag") or tag.get("label") or "anomaly")
            normalizations.append(f"tag[{index}]:default_mast_tag")
        if "reason" not in tag:
            tag["reason"] = str(tag.get("explanation") or tag.get("description") or "visible anomaly")
            normalizations.append(f"tag[{index}]:default_reason")
        tag.pop("symptom_event_ids", None)
        require_keys(tag, {"event_id", "mast_tag", "reason"}, name="ErrorProbe tag")
        if tag["event_id"] not in event_ids:
            raise ValueError("ErrorProbe tag references an unknown event")
        normalized_tags.append(tag)
    value["tags"] = normalized_tags
    value["symptom_event_ids"] = [
        str(item) for item in value["symptom_event_ids"] if str(item) in event_ids
    ]
    if (
        not value["symptom_event_ids"]
    ):
        value["symptom_event_ids"] = [sample.events[-1].event_id]
        normalizations.append("default_symptom_event_ids->last_event")
    if normalizations:
        value["_schema_normalizations"] = normalizations
    return value


def _validate_edges(raw: Any, sample: BaselineSample) -> dict[str, Any]:
    value = require_keys(raw, {"edges"}, name="dependency graph")
    if not isinstance(value["edges"], list):
        raise TypeError("dependency graph edges must be a list")
    positions = {event.event_id: index for index, event in enumerate(sample.events)}
    normalized_edges: list[dict[str, Any]] = []
    normalizations: list[str] = []
    for index, raw_edge in enumerate(value["edges"]):
        if isinstance(raw_edge, Mapping):
            edge = dict(raw_edge)
        else:
            raise TypeError("dependency edge must be an object")
        # Qwen occasionally serializes an edge as
        # {"source:event:0007": "source:event:0008", "reason": ...}.
        # Recover that unambiguous pair without inventing a causal relation.
        if "source" not in edge or "target" not in edge:
            event_keys = [key for key in edge if str(key) in positions]
            if len(event_keys) == 1:
                source = event_keys[0]
                target = edge.pop(source)
                edge["source"] = source
                edge["target"] = target
                normalizations.append(f"edge[{index}]:flattened_event_pair")
        edge["source"] = _canonical_event_id(edge.get("source"), positions)
        edge["target"] = _canonical_event_id(edge.get("target"), positions)
        require_keys(edge, {"source", "target", "reason"}, name="dependency edge")
        if edge["source"] not in positions or edge["target"] not in positions:
            raise ValueError("dependency edge references an unknown event")
        if positions[edge["source"]] >= positions[edge["target"]]:
            edge["source"], edge["target"] = edge["target"], edge["source"]
            normalizations.append("reverse_dependency_edge")
        normalized_edges.append(edge)
    value["edges"] = normalized_edges
    if normalizations:
        value["_schema_normalizations"] = normalizations
    return value


def _canonical_event_id(value: Any, positions: dict[str, int]) -> Any:
    if value in positions:
        return value
    text = str(value or "")
    if text in positions:
        return text
    suffix = text.rsplit(":", 1)[-1]
    if suffix.isdigit():
        number = int(suffix)
        candidates = [
            event_id
            for event_id, position in positions.items()
            if position == number
            or event_id.rsplit(":", 1)[-1].isdigit()
            and int(event_id.rsplit(":", 1)[-1]) == number
        ]
        if len(candidates) == 1:
            return candidates[0]
    return value


def _backward_trace(symptoms: list[str], edges: list[dict[str, Any]]) -> set[str]:
    incoming: dict[str, list[str]] = {}
    for edge in edges:
        incoming.setdefault(str(edge["target"]), []).append(str(edge["source"]))
    seen = set(symptoms)
    queue = deque(symptoms)
    while queue:
        target = queue.popleft()
        for source in incoming.get(target, []):
            if source not in seen:
                seen.add(source)
                queue.append(source)
    return seen


def _signature(tags: dict[str, Any], sample: BaselineSample) -> dict[str, Any]:
    return {
        "mast_tags": sorted({str(tag["mast_tag"]) for tag in tags["tags"]}),
        "roles": sorted({event.role for event in sample.events}),
        "benchmark": sample.benchmark,
        "mas": sample.mas,
    }


def _retrieve(
    memory: list[dict[str, Any]], signature: dict[str, Any], k: int
) -> list[dict[str, Any]]:
    current_tags = set(signature["mast_tags"])
    ranked: list[tuple[float, dict[str, Any]]] = []
    for item in memory:
        prior = (
            item.get("signature") if isinstance(item.get("signature"), Mapping) else {}
        )
        prior_tags = set(prior.get("mast_tags", []))
        if not current_tags or not prior_tags or not current_tags & prior_tags:
            continue
        union = current_tags | prior_tags
        score = len(current_tags & prior_tags) / len(union)
        if score >= 0.75:
            ranked.append((score, item))
    return [
        item for _, item in sorted(ranked, key=lambda value: value[0], reverse=True)[:k]
    ]


def _validate_hypotheses(
    raw: Any, sample: BaselineSample, receptive: set[str]
) -> dict[str, Any]:
    value = require_keys(raw, {"hypotheses"}, name="hypotheses")
    if not isinstance(value["hypotheses"], list) or not value["hypotheses"]:
        raise ValueError("ErrorProbe hypotheses must be a non-empty list")
    positions = {event.event_id: index for index, event in enumerate(sample.events)}
    receptive_positions = [positions[event_id] for event_id in receptive if event_id in positions]
    for hypothesis in value["hypotheses"]:
        if not isinstance(hypothesis, Mapping):
            raise TypeError("ErrorProbe hypothesis must be an object")
        if hypothesis.get("event_id") not in receptive and receptive_positions:
            proposed = positions.get(str(hypothesis.get("event_id")))
            nearest = min(
                receptive_positions,
                key=lambda index: abs(index - proposed),
            ) if proposed is not None else receptive_positions[-1]
            hypothesis = dict(hypothesis)
            hypothesis["event_id"] = sample.events[nearest].event_id
        require_keys(
            hypothesis,
            {
                "hypothesis_id",
                "event_id",
                "responsible_agent",
                "suspected_mode",
                "claim",
            },
            name="hypothesis",
        )
        if hypothesis["event_id"] not in receptive:
            raise ValueError("hypothesis is outside the effective receptive field")
    return value


def _validate_probes(raw: Any, hypothesis_id: str) -> dict[str, Any]:
    value = require_keys(raw, {"hypothesis_id", "probes"}, name="investigation")
    if value["hypothesis_id"] != hypothesis_id or not isinstance(value["probes"], list):
        raise ValueError("investigation hypothesis_id/probes mismatch")
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    normalizations: list[str] = []
    for index, raw_probe in enumerate(value["probes"]):
        if not isinstance(raw_probe, Mapping):
            normalizations.append(f"probe[{index}]:drop_non_object")
            continue
        probe = dict(raw_probe)
        require_keys(
            probe,
            {"type", "event_id", "specification", "expected_observation"},
            name="probe",
        )
        if probe["type"] not in {"code_exec", "logic_probe"}:
            raise ValueError("probe.type must be code_exec or logic_probe")
        signature = tuple(
            str(probe[key])
            for key in ("type", "event_id", "specification", "expected_observation")
        )
        if signature in seen:
            normalizations.append(f"probe[{index}]:drop_duplicate")
            continue
        seen.add(signature)
        normalized.append(probe)
        if len(normalized) == 4:
            if index + 1 < len(value["probes"]):
                normalizations.append("probes:truncate_to_4")
            break
    if not normalized:
        raise ValueError("investigation contains no valid probes")
    value["probes"] = normalized
    if normalizations:
        value["_schema_normalizations"] = normalizations
    return value


def _validate_arbiter(raw: Any, sample: BaselineSample) -> dict[str, Any]:
    normalizations: list[str] = []
    if isinstance(raw, Mapping) and isinstance(raw.get("hypotheses"), list):
        candidates = [
            dict(item)
            for item in raw["hypotheses"]
            if isinstance(item, Mapping)
            and isinstance(item.get("confidence"), (int, float))
            and not isinstance(item.get("confidence"), bool)
        ]
        if candidates:
            best_confidence = max(float(item["confidence"]) for item in candidates)
            best = [
                item
                for item in candidates
                if float(item["confidence"]) == best_confidence
            ]
            if best:
                raw = min(
                    best,
                    key=lambda item: (
                        item.get("step_index")
                        if isinstance(item.get("step_index"), int)
                        and not isinstance(item.get("step_index"), bool)
                        else len(sample.events),
                        str(item.get("responsible_agent") or ""),
                        str(item.get("reason") or ""),
                    ),
                )
                normalizations.append(
                    "hypotheses->highest_confidence_earliest_step"
                    if len(best) > 1
                    else "hypotheses->unique_highest_confidence"
                )
    if isinstance(raw, dict) and "step_index" not in raw:
        event_id = raw.get("event_id")
        positions = {event.event_id: index for index, event in enumerate(sample.events)}
        if event_id in positions:
            raw = {**raw, "step_index": positions[event_id]}
    value = validate_attribution(raw)
    require_keys(value, {"failure_mode", "confidence"}, name="arbiter")
    if not 0 <= value["step_index"] < len(sample.events):
        raise ValueError("arbiter step_index is outside the trajectory")
    if not isinstance(value["confidence"], (int, float)) or isinstance(
        value["confidence"], bool
    ):
        raise TypeError("arbiter confidence must be numeric")
    value["confidence"] = float(value["confidence"])
    if not 0 <= value["confidence"] <= 1:
        raise ValueError("arbiter confidence must be between 0 and 1")
    if normalizations:
        existing = value.setdefault("_schema_normalizations", [])
        existing.extend(normalizations)
    value["event_id"] = sample.events[value["step_index"]].event_id
    return value
