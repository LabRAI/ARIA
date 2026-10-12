"""Uninterrupted host-system reference row."""

from __future__ import annotations

from typing import Any, ClassVar

from baselines.core.method import BaselineMethod
from baselines.core.models import ModelAdapter
from baselines.core.replay import ReplayBackend
from baselines.core.types import BaselineSample


class NoIntervention(BaselineMethod):
    name = "no_intervention"
    setting = "active"
    requires_model = False
    provenance: ClassVar[dict[str, Any]] = {
        "source": "docs/sections/05_experiments.tex",
        "execution_budget": 1.0,
    }

    def run(
        self,
        sample: BaselineSample,
        model: ModelAdapter | None,
        replay: ReplayBackend | None = None,
    ):
        result = self.result(sample, status="no_intervention")
        result.diagnostics["source_evaluation_reused"] = True
        return result
