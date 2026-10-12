"""Run a fired training gate through the production three-stage repair policy."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
import json
from typing import Any

from aria.hook.controller import ActiveRepairController, validate_pending_substitution
from aria.hook.profiles import SUBSTITUTE_ONLY_CAPABILITIES
from aria.model import AriaModelConfig, GatePrediction, OpenAIChatAriaModel
from aria.records import IncidentRecord


class CallableRepairModel(OpenAIChatAriaModel):
    """Use the same validators and stage prompts without a second model server."""

    def __init__(self, generate: Callable[[str, Mapping[str, Any]], str], *, retries: int):
        self.config = AriaModelConfig(provider="transformers", model="training-policy",
                                     runtime_context=True, structured_output_retries=retries)
        self.calls = []
        self.generate = generate

    def predict_gate(self, incident):
        # The stochastic training gate already fired; never resample it here.
        return GatePrediction(1.0)

    def _complete(self, *, system, payload):
        call = {"messages": [{"role": "system", "content": system},
                             {"role": "user", "content": json.dumps(payload, sort_keys=True)}]}
        self.calls.append(call)
        generated = self.generate(system, deepcopy(payload))
        if isinstance(generated, tuple):
            text, fitted = generated
            call["input_before_budget"] = deepcopy(payload)
            call["messages"][1]["content"] = json.dumps(fitted, sort_keys=True)
        else:
            text = generated
        call["raw_response"] = text
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("ARIA model must return schema-constrained JSON only") from exc
        if not isinstance(raw, dict):
            raise TypeError("ARIA model JSON response must be an object")
        call["parsed_response"] = deepcopy(raw)
        return raw


def deployed_repair_attempt(incident: IncidentRecord, generate, *, retries: int = 1):
    """Return a proposal and calls; unsupported delivery is not executed or rewarded."""
    if not incident.events:
        raise ValueError("training repair needs an observable pending event")
    policy = CallableRepairModel(generate, retries=retries)
    controller = ActiveRepairController(policy, gate_threshold=0.5, fail_open=True)
    controller.begin(incident.incident_id, constraints=incident.constraints, feedback=None)
    for event in incident.events[:-1]:
        controller.observe(event)
    pending = incident.events[-1]
    decision = controller.intercept(pending, capabilities=SUBSTITUTE_ONLY_CAPABILITIES)
    decision = validate_pending_substitution(decision, pending.content, role=pending.role,
        protected=bool(pending.tool_calls) or pending.kind != "message" or pending.role in {"system", "user", "tool"})
    return decision, policy.calls


def bernoulli_policy_loss(probability, *, old_probability: float, reference_probability,
                          fired: bool, advantage: float, clip_low: float, clip_high: float,
                          kl_coefficient: float, torch):
    """Clipped gate policy objective and exact binary KL; no evaluator labels in inputs."""
    p = probability.clamp(1e-6, 1 - 1e-6)
    ref = reference_probability.detach().clamp(1e-6, 1 - 1e-6)
    old = min(max(old_probability, 1e-6), 1 - 1e-6)
    selected = p if fired else 1 - p
    ratio = selected / (old if fired else 1 - old)
    clipped = ratio.clamp(1 - clip_low, 1 + clip_high)
    surrogate = torch.minimum(ratio * advantage, clipped * advantage)
    kl = p * (p.log() - ref.log()) + (1 - p) * ((1 - p).log() - (1 - ref).log())
    return -surrogate + kl_coefficient * kl
