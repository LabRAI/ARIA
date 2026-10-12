"""Who&When all-at-once attribution followed by the shared repair prompt."""

from __future__ import annotations

from typing import Any, ClassVar

from baselines.core.method import BaselineMethod, validate_attribution
from baselines.core.models import ModelAdapter
from baselines.core.prompting import repair_from_context
from baselines.core.replay import ReplayBackend, execute
from baselines.core.types import BaselineSample

ATTRIBUTION_SYSTEM = """Analyze the complete multi-agent failure log in one receptive field.
Identify the single agent directly responsible for the failed solution, the zero-based first
decisive error step, and a concise reason. Return JSON only with responsible_agent,
step_index, and reason. Use evaluator feedback, but do not assume access to a gold answer."""


class WhoWhenRepair(BaselineMethod):
    name = "who_when"
    setting = "passive"
    provenance: ClassVar[dict[str, Any]] = {
        "paper": "Which Agent Causes Task Failures and When? (ICML 2025)",
        "official_repository": "https://github.com/mingyin1/Agents_Failure_Attribution",
        "official_commit": "b2bae5c5b06d681d04ea5e9b63b7a30525c04925",
        "official_method": "all_at_once",
        "adaptation": "gold answer omitted to match the answer-blind experiment contract",
    }

    def run(
        self,
        sample: BaselineSample,
        model: ModelAdapter | None,
        replay: ReplayBackend | None = None,
    ):
        selected = self.require_model(model)
        result = self.result(sample, status="proposed")
        attribution, call = selected.json_call(
            stage="who_when.all_at_once",
            system=ATTRIBUTION_SYSTEM,
            payload=sample.model_view(include_feedback=True),
            validator=lambda value: _validate(value, sample),
        )
        result.attribution = attribution
        result.model_calls.append(call)
        intervention, repair_call = repair_from_context(
            selected,
            sample,
            stage="who_when.shared_repair",
            attribution=attribution,
        )
        result.intervention = intervention
        result.model_calls.append(repair_call)
        execution = execute(replay, sample, intervention, label="who-when-repair")
        if execution is not None:
            result.executions.append(execution)
            result.selected_execution_label = execution.label
            result.status = "executed"
        return result


def _validate(raw: Any, sample: BaselineSample) -> dict[str, Any]:
    value = validate_attribution(raw)
    normalizations = value.setdefault("_schema_normalizations", [])
    if value["step_index"] == len(sample.events):
        value["step_index"] -= 1
        normalizations.append("one_based_terminal_step_index->zero_based")
    elif not 0 <= value["step_index"] < len(sample.events):
        event_ids = {event.event_id: index for index, event in enumerate(sample.events)}
        requested = value["step_index"]
        exact_id = f"source:event:{requested:04d}"
        if exact_id in event_ids:
            value["step_index"] = event_ids[exact_id]
            normalizations.append("step_index->source_event_suffix_index")
        else:
            candidates = []
            for event_id, index in event_ids.items():
                suffix = event_id.rsplit(":", 1)[-1]
                if suffix.isdigit() and int(suffix) <= requested:
                    candidates.append((int(suffix), index))
            if candidates:
                value["step_index"] = max(candidates)[1]
                normalizations.append("step_index->nearest_source_event_suffix")
    if not 0 <= value["step_index"] < len(sample.events):
        raise ValueError("attribution.step_index is outside the trajectory")
    value["event_id"] = sample.events[value["step_index"]].event_id
    return value
