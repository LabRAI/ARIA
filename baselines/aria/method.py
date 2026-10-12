"""ARIA as an evaluated method in the passive and active settings.

Section~\\ref{subsec:rq1-effectiveness} evaluates ARIA against baselines under
one harness, so ARIA is registered as a method here rather than driven by its
own runner.  The two settings differ in exactly the two ways the paper states:
the passive method observes a terminated trajectory together with evaluator
feedback and emits one record, while the active method observes only a prefix,
never observes feedback, and emits a record at the first positive gate decision.

The prompts are the ones the model was trained on, imported from
:mod:`aria.prompts`, so an untuned backbone and a post-trained one are asked the
same question and the difference between their rows is the post-training.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, ClassVar

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from aria.calibration import calibrate_gate, load_gate_calibration
from aria.gate_scoring import chat_label_probability
from aria.output_schemas import output_schema
from aria.prompts import instruction_for_task
from aria.records import (
    FailureLocalization,
    IncidentRecord,
    RepairAttribution,
    RepairRecord,
    TrajectoryEvent,
)
from baselines.core.method import BaselineMethod, validate_intervention
from baselines.core.models import ModelAdapter, StructuredGenerationError
from baselines.core.replay import ReplayBackend, execute
from baselines.core.types import BaselineSample, MethodResult

MAX_REPLACEMENT_EVENTS = 8


class _AriaMethod(BaselineMethod):
    provenance: ClassVar[dict[str, Any]] = {
        "source": "docs/sections/03_methodology.tex",
        "record_schema": "diagnosis, cut_event_id, replacement_events",
    }

    def _schema_options(self, task: str, events: tuple[Any, ...]) -> dict[str, Any] | None:
        if not self.options.get("structured_decoding", False):
            return None
        return {"response_format": {"type": "json_schema", "json_schema": {
            "name": task, "strict": True,
            "schema": output_schema(task, [event.event_id for event in events]),
        }}}

    def _repair(
        self,
        sample: BaselineSample,
        model: ModelAdapter,
        result: MethodResult,
        *,
        events: tuple[Any, ...],
        include_feedback: bool,
        stage: str,
    ) -> dict[str, Any] | None:
        payload = _model_input(sample, events, include_feedback=include_feedback)
        incident = _incident(sample, events, include_feedback=include_feedback)

        # Match the training chain: localization, attribution, structured repair.
        localization: dict[str, Any] | None = None
        attribution: dict[str, Any] | None = None
        try:
            localization_value, localization_call = model.json_call(
                stage=f"{stage}.localization",
                canonical_json=True,
                system=instruction_for_task("failure_localization"),
                payload=payload,
                validator=lambda value: _validate_localization(value, incident),
                request_options=self._schema_options("failure_localization", events),
            )
            result.model_calls.append(localization_call)
            localization = dict(localization_value)

            attribution_value, attribution_call = model.json_call(
                stage=f"{stage}.attribution",
                canonical_json=True,
                system=instruction_for_task("repair_attribution"),
                payload={
                    **payload,
                    "localization": localization,
                },
                validator=lambda value: _validate_attribution(
                    value, incident, localization
                ),
                request_options=self._schema_options("repair_attribution", events),
            )
            result.model_calls.append(attribution_call)
            attribution = dict(attribution_value)

            patch_payload = {
                **payload,
                **({"localization": localization} if localization else {}),
                **({"attribution": attribution} if attribution else {}),
            }
            intervention, call = model.json_call(
                stage=stage,
                canonical_json=True,
                system=instruction_for_task("structured_repair"),
                payload=patch_payload,
                validator=lambda value: _validate(value, sample, events),
                request_options=self._schema_options("structured_repair", events),
            )
        except StructuredGenerationError as exc:
            # A malformed record is a real outcome for this method, not an
            # infrastructure failure: HIR counts a structurally invalid
            # intervention as harmful, so it is recorded rather than retried.
            result.model_calls.append(exc.record)
            result.diagnostics["structurally_valid"] = False
            result.diagnostics["structured_generation_error"] = str(exc)
            parsed = exc.record.parsed_response
            result.intervention = dict(parsed) if isinstance(parsed, dict) else None
            result.status = "invalid"
            return None
        result.model_calls.append(call)
        result.intervention = intervention
        if attribution is not None:
            result.attribution = attribution
        if localization is not None:
            result.diagnostics["localization"] = localization
        result.diagnostics["structurally_valid"] = True
        return intervention


class AriaPassive(_AriaMethod):
    """Repair a terminated failed trajectory with evaluator feedback visible."""

    name = "aria_passive"
    setting = "passive"
    provenance = {
        **_AriaMethod.provenance,
        "serving_protocol": "localization -> attribution -> structured_repair",
        "trajectory_scope": "complete_failed_trajectory",
        "replay_required_for_success": True,
    }

    def run(
        self,
        sample: BaselineSample,
        model: ModelAdapter | None,
        replay: ReplayBackend | None = None,
    ) -> MethodResult:
        selected = self.require_model(model)
        result = self.result(sample, status="proposed")
        intervention = self._repair(
            sample,
            selected,
            result,
            events=sample.events,
            include_feedback=True,
            stage="aria.passive.repair",
        )
        if intervention is None:
            return result
        execution = execute(replay, sample, intervention, label="aria-passive")
        if execution is not None:
            result.executions.append(execution)
            result.selected_execution_label = execution.label
            result.status = "executed"
        return result


class AriaActive(_AriaMethod):
    """Gate a running trajectory and intervene at the first positive decision.

    The gate is evaluated on the prefix after each event, and no evaluator
    feedback is exposed at any point.  Abstention delivers the uninterrupted
    host run, which is the matched reference the active pool is scored against.
    """

    name = "aria_active"
    setting = "active"
    provenance = {
        **_AriaMethod.provenance,
        "serving_protocol": "pre_delivery_gate -> localization -> attribution -> structured_repair",
        "hook": "aria.hook.AriaHook",
        "supported_mas": ["openmanus", "metagpt", "magentic-one", "ag2"],
    }

    def __init__(self, **options: Any) -> None:
        super().__init__(**options)
        self.gate_probability_source = str(options.get("gate_probability_source", "generated"))
        if self.gate_probability_source not in {"generated", "label_likelihood"}:
            raise ValueError("unsupported gate_probability_source")
        calibration_path = options.get("gate_calibration_path")
        self.gate_calibration = (load_gate_calibration(Path(calibration_path), required=True,
                                                     probability_source=self.gate_probability_source)
                                 if calibration_path else None)
        self.gate_threshold = float(options.get("gate_threshold", 0.5))
        # Active deployment evaluates the gate at every pre-delivery event.
        # ``max_probes`` optionally limits gate evaluations and is recorded
        # in the result diagnostics.
        max_probes = options.get("max_probes")
        self.max_probes = int(max_probes) if max_probes is not None else None
        if not 0.0 <= self.gate_threshold <= 1.0:
            raise ValueError("aria_active gate_threshold must be between 0 and 1")
        if self.max_probes is not None and self.max_probes < 1:
            raise ValueError("aria_active max_probes must be positive when supplied")

    def run(
        self,
        sample: BaselineSample,
        model: ModelAdapter | None,
        replay: ReplayBackend | None = None,
    ) -> MethodResult:
        selected = self.require_model(model)
        result = self.result(sample, status="proposed")
        probes = _probe_indices(len(sample.events), self.max_probes)
        decisions: list[dict[str, Any]] = []
        trigger: int | None = None
        for probe in probes:
            prefix = sample.events[:probe]
            payload = _model_input(sample, prefix, include_feedback=False)
            try:
                gate, call = selected.json_call(
                    stage=f"aria.active.gate[{probe}]",
                    canonical_json=True,
                    system=instruction_for_task("deployment_gate"),
                    payload=payload,
                    validator=_validate_gate,
                    request_options=({
                        **(self._schema_options("deployment_gate", prefix) or {}),
                        **({"logprobs": True, "top_logprobs": 20}
                           if self.gate_probability_source == "label_likelihood" else {}),
                    } or None),
                )
                result.model_calls.append(call)
                probability = float(gate["gate_probability"])
                if self.gate_probability_source == "label_likelihood":
                    likelihood = chat_label_probability(call.response_metadata.get("token_logprobs"))
                    if likelihood is None:
                        result.status = "invalid"
                        result.diagnostics.update({
                            "gate_probability_source": self.gate_probability_source,
                            "structured_generation_error": "endpoint omitted the gate-label log probabilities",
                            "gate_decisions": decisions, "intervened": False,
                        })
                        return result
                    probability = likelihood
                if self.gate_calibration is not None:
                    if not call.response_metadata.get("gate_calibration_applied"):
                        probability = calibrate_gate(probability, self.gate_calibration)
            except StructuredGenerationError as exc:
                result.model_calls.append(exc.record)
                decisions.append({"prefix": probe, "gate_probability": None,
                                  "validation_error": str(exc)})
                result.status = "invalid"
                result.diagnostics.update({
                    "gate_decisions": decisions,
                    "gate_threshold": self.gate_threshold,
                    "gate_calibration": self.gate_calibration,
                    "gate_evaluated_after_each_event": self.max_probes is None,
                    "probe_count": len(probes),
                    "intervened": False,
                    "structured_generation_error": str(exc),
                })
                return result
            decisions.append({"prefix": probe, "gate_probability": probability})
            if probability >= self.gate_threshold:
                trigger = probe
                break
        result.diagnostics["gate_decisions"] = decisions
        result.diagnostics["gate_threshold"] = self.gate_threshold
        result.diagnostics["gate_probability_source"] = self.gate_probability_source
        result.diagnostics["gate_calibration"] = self.gate_calibration
        result.diagnostics["probe_count"] = len(probes)
        result.diagnostics["gate_evaluated_after_each_event"] = self.max_probes is None
        if trigger is None:
            # No intervention: the source run is delivered unchanged, so the
            # pool row scores the uninterrupted outcome and charges no budget.
            result.status = "abstained"
            result.diagnostics["intervened"] = False
            return result
        result.diagnostics["intervened"] = True
        result.diagnostics["trigger_prefix"] = trigger
        intervention = self._repair(
            sample,
            selected,
            result,
            events=sample.events[:trigger],
            include_feedback=False,
            stage="aria.active.repair",
        )
        if intervention is None:
            return result
        execution = execute(replay, sample, intervention, label="aria-active")
        if execution is not None:
            result.executions.append(execution)
            result.selected_execution_label = execution.label
            result.status = "executed"
        return result


def _model_input(
    sample: BaselineSample, events: tuple[Any, ...], *, include_feedback: bool
) -> dict[str, Any]:
    """The answer-blind view, in the field layout the model was trained on."""

    return _incident(sample, events, include_feedback=include_feedback).model_input()



def _incident(
    sample: BaselineSample,
    events: tuple[Any, ...],
    *,
    include_feedback: bool,
) -> IncidentRecord:
    """Build the exact record shape consumed by the serving validators.

    Baseline events carry a separate ``tool_results`` convenience field.  ARIA
    keeps those observations in event metadata, which preserves the one
    canonical trajectory representation shared by training, passive serving,
    and the active hook.
    """

    normalized: list[TrajectoryEvent] = []
    for item in events:
        metadata = dict(item.metadata)
        if item.tool_results:
            metadata.setdefault("tool_results", [dict(value) for value in item.tool_results])
        normalized.append(
            TrajectoryEvent(
                event_id=item.event_id,
                role=item.role,
                kind=item.kind,
                content=item.content,
                tool_calls=tuple(dict(value) for value in item.tool_calls),
                metadata=metadata,
            )
        )
    constraints = dict(sample.constraints)
    constraints.setdefault("task_text", sample.task)
    if sample.benchmark:
        constraints.setdefault("benchmark", sample.benchmark)
    if sample.mas:
        constraints.setdefault("mas", sample.mas)
    return IncidentRecord(
        incident_id=sample.sample_id,
        events=tuple(normalized),
        feedback=sample.feedback if include_feedback else None,
        constraints=constraints,
    )


def _validate_localization(value: Any, incident: IncidentRecord) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("failure localization must be an object")
    # Small instruction tuned checkpoints often append an obsolete
    # ``fixed_events``/``fixed_answer`` field after the trained schema. Those
    # claims are answer-bearing and are deliberately ignored at the host
    # boundary; only the four localization fields are executable.
    value = {key: value[key] for key in (
        "first_bad_event_ids", "retained_frontier_ids", "evidence", "confidence"
    ) if key in value}
    value = _normalize_localization_evidence(dict(value), incident)
    return FailureLocalization.from_dict(
        value,
        events=incident.events,
        name="ARIA baseline localization",
    ).to_dict()


def _normalize_localization_evidence(
    value: dict[str, Any], incident: IncidentRecord
) -> dict[str, Any]:
    """Keep evidence grounded when a small model paraphrases its quote.

    The model is still responsible for selecting the event and explaining the
    cause. The host-side adapter only replaces a non-verbatim quote with a
    bounded substring from that same selected event, so the schema remains
    executable without inventing evidence or exposing the reference answer.
    """

    events = {event.event_id: event for event in incident.events}
    evidence = value.get("evidence")
    if not isinstance(evidence, list):
        return value
    normalized: list[dict[str, Any]] = []
    for item in evidence:
        if not isinstance(item, dict):
            normalized.append(item)
            continue
        entry = dict(item)
        event = events.get(str(entry.get("event_id")))
        if event is not None:
            quote = str(entry.get("quote") or "")
            if quote not in event.content:
                replacement = _best_grounded_quote(quote, event.content)
                if replacement:
                    entry["quote"] = replacement
                    entry["_schema_normalized"] = "quote->verbatim_event_substring"
        normalized.append(entry)
    value["evidence"] = normalized
    return value


def _best_grounded_quote(proposed: str, content: str) -> str:
    if not content:
        return ""
    words = [word for word in proposed.split() if len(word) > 2]
    if words:
        best_start = min(
            (
                content.lower().find(word.lower())
                for word in words
                if content.lower().find(word.lower()) >= 0
            ),
            default=-1,
        )
        if best_start >= 0:
            return content[best_start : best_start + 160]
    return content[:160]


def _validate_attribution(
    value: Any,
    incident: IncidentRecord,
    localization: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("repair attribution must be an object")
    if localization is None:
        raise ValueError("repair attribution requires a localization")
    value = {key: value[key] for key in (
        "diagnosis", "primary_code", "secondary_codes", "responsible_agents",
        "counterfactual", "patch_operator", "confidence"
    ) if key in value}
    selected = FailureLocalization.from_dict(
        localization,
        events=incident.events,
        name="ARIA baseline localization",
    )
    return RepairAttribution.from_dict(
        value,
        events=incident.events,
        localization=selected,
        name="ARIA baseline attribution",
    ).model_output()


def _validate(value: Any, sample: BaselineSample, events: tuple[Any, ...]) -> dict[str, Any]:
    """Reject a record the harness could not splice, before it is executed."""

    scoped = BaselineSample(
        sample_id=sample.sample_id,
        task_id=sample.task_id,
        task=sample.task,
        events=tuple(events),
        constraints=sample.constraints,
        feedback=sample.feedback,
        benchmark=sample.benchmark,
        mas=sample.mas,
        split=sample.split,
        source_evaluation=sample.source_evaluation,
        metadata=sample.metadata,
    )
    validated = validate_intervention(value, scoped)
    if len(validated["replacement_events"]) > MAX_REPLACEMENT_EVENTS:
        raise ValueError(
            f"replacement_events exceeds the bound of {MAX_REPLACEMENT_EVENTS} events"
        )
    if not isinstance(validated.get("diagnosis"), str) or not validated["diagnosis"].strip():
        raise ValueError("diagnosis must be a non-empty string")
    return validated


def _validate_gate(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or "gate_probability" not in value:
        raise KeyError("gate response missing gate_probability")
    probability = value["gate_probability"]
    if isinstance(probability, bool) or not isinstance(probability, (int, float)):
        raise TypeError("gate_probability must be a number")
    if not 0.0 <= float(probability) <= 1.0:
        raise ValueError("gate_probability must lie between 0 and 1")
    return {"gate_probability": float(probability)}


def _probe_indices(event_count: int, max_probes: int | None = None) -> list[int]:
    """Prefix lengths at which the gate is asked to decide.

    Maintained measurements and the deployable Hook evaluate after every event.
    A positive ``max_probes`` limits the number of gate evaluations; its use is
    reported in the result diagnostics rather than left implicit.
    """

    if event_count <= 0:
        return []
    if max_probes is None or max_probes <= 0 or event_count <= max_probes:
        return list(range(1, event_count + 1))
    step = event_count / max_probes
    return sorted(
        {max(1, min(event_count, round((index + 1) * step))) for index in range(max_probes)}
    )
