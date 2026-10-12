"""Single-call passive repair used for the API-1/API-2 rows."""

from __future__ import annotations

from typing import Any, ClassVar

from baselines.core.method import BaselineMethod
from baselines.core.models import ModelAdapter
from baselines.core.prompting import repair_from_context
from baselines.core.replay import ReplayBackend, execute
from baselines.core.types import BaselineSample


class ZeroShotRepair(BaselineMethod):
    name = "zero_shot"
    setting = "passive"
    provenance: ClassVar[dict[str, Any]] = {
        "source": "docs/sections/05_experiments.tex",
        "adaptation": "shared answer-blind repair record schema",
    }

    def run(
        self,
        sample: BaselineSample,
        model: ModelAdapter | None,
        replay: ReplayBackend | None = None,
    ):
        selected = self.require_model(model)
        result = self.result(sample, status="proposed")
        intervention, call = repair_from_context(
            selected, sample, stage="zero_shot.repair"
        )
        result.intervention = intervention
        result.model_calls.append(call)
        execution = execute(replay, sample, intervention, label="zero-shot-repair")
        if execution is not None:
            result.executions.append(execution)
            result.selected_execution_label = execution.label
            result.status = "executed"
        return result
