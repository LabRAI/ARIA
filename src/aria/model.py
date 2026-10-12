"""Model loading and constrained decoding for an ARIA repair checkpoint.

The service depends on a small protocol instead of a specific foundation
model. Endpoint and local Transformers implementations share the same gate,
failure-localization, RA-attribution, and structured-patch outputs.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from aria.chat_templates import prepare_chat_messages, render_chat_prompt
from aria.calibration import calibrate_gate, load_gate_calibration
from aria.gate_scoring import chat_label_probability, label_probability
from aria.prompt_budget import bounded_text, fit_messages

from aria.prompts import (
    ATTRIBUTION_INSTRUCTION,
    GATE_INSTRUCTION,
    LOCALIZATION_INSTRUCTION,
    RUNTIME_LOCALIZATION_INSTRUCTION,
    RUNTIME_ATTRIBUTION_INSTRUCTION,
    STRUCTURED_REPAIR_INSTRUCTION,
    canonical_model_json,
)
from aria.records import (
    FailureLocalization,
    IncidentRecord,
    RepairAttribution,
    RepairRecord,
)


@dataclass(frozen=True)
class GatePrediction:
    probability: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("ARIA gate probability must be between 0 and 1")


class AriaModel(Protocol):
    """The only model-facing contract required by passive and active serving."""

    def predict_gate(self, incident: IncidentRecord) -> GatePrediction:
        """Predict whether the current prefix would draw failure feedback."""

    def localize(self, incident: IncidentRecord) -> FailureLocalization:
        """Identify the earliest causal error step and retained frontier."""

    def attribute(
        self,
        incident: IncidentRecord,
        localization: FailureLocalization | None = None,
    ) -> RepairAttribution:
        """Assign one RA attribution to a host-validated localization."""

    def propose_repair(
        self,
        incident: IncidentRecord,
        attribution: RepairAttribution | None = None,
    ) -> RepairRecord:
        """Produce a schema-valid diagnosis, cut point, and replacement events."""


@dataclass(frozen=True)
class AriaModelConfig:
    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    api_key_required: bool = True
    timeout_seconds: int = 120
    max_tokens: int = 2048
    temperature: float | None = 0.0
    seed: int | None = None
    structured_output_retries: int = 0
    runtime_context: bool = False
    gate_calibration_path: str | None = None
    gate_probability_source: str = "generated"
    system_role_mode: str = "native"
    max_input_tokens: int | None = None
    tokenizer_endpoint: str | None = None

    def validate(self) -> None:
        if self.system_role_mode not in {"native", "user-prefix"}:
            raise ValueError("ARIA system_role_mode must be native or user-prefix")
        if not isinstance(self.runtime_context, bool):
            raise TypeError("ARIA runtime_context must be a boolean")
        if (not isinstance(self.structured_output_retries, int) or isinstance(self.structured_output_retries, bool)
                or not 0 <= self.structured_output_retries <= 2):
            raise ValueError("ARIA structured_output_retries must be an integer from 0 to 2")
        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
            raise TypeError("ARIA model seed must be an integer or null")
        if self.gate_probability_source not in {"generated", "label_likelihood"}:
            raise ValueError("ARIA gate_probability_source must be generated or label_likelihood")
        if self.max_input_tokens is not None and self.max_input_tokens <= 0:
            raise ValueError("ARIA max_input_tokens must be positive")
        if self.tokenizer_endpoint is not None:
            base, tokenizer = urlsplit(self.base_url or ""), urlsplit(self.tokenizer_endpoint)
            if (self.provider != "openai-chat-completions" or self.max_input_tokens is None
                    or tokenizer.scheme not in {"http", "https"}
                    or (base.scheme, base.netloc) != (tokenizer.scheme, tokenizer.netloc)):
                raise ValueError("endpoint token budgeting requires max_input_tokens and a same-origin tokenizer_endpoint")
        if self.provider == "openai-chat-completions" and self.max_input_tokens is not None and not self.tokenizer_endpoint:
            raise ValueError("endpoint max_input_tokens requires an explicit tokenizer_endpoint")
        if self.provider not in {"openai-chat-completions", "transformers"}:
            raise ValueError(
                "ARIA model provider must be openai-chat-completions or transformers"
            )
        if not self.model.strip():
            raise ValueError("ARIA model must be non-empty")
        if self.provider == "openai-chat-completions" and not self.base_url:
            raise ValueError("openai-chat-completions ARIA model requires base_url")
        if (
            self.api_key_required
            and not self.api_key_env
            and self.provider == "openai-chat-completions"
        ):
            raise ValueError(
                "endpoint ARIA model requires api_key_env when api_key_required is true"
            )
        if self.timeout_seconds <= 0 or self.max_tokens <= 0:
            raise ValueError(
                "ARIA model timeout_seconds and max_tokens must be positive"
            )


def load_aria_model(config: AriaModelConfig) -> AriaModel:
    """Load a configured ARIA checkpoint without selecting a base model family."""

    config.validate()
    if config.provider == "openai-chat-completions":
        return OpenAIChatAriaModel(config)
    if config.provider == "transformers":
        return TransformersAriaModel(config)
    raise AssertionError(f"unhandled ARIA model provider: {config.provider}")


class OpenAIChatAriaModel:
    """Constrained JSON service over any OpenAI Chat Completions endpoint."""

    def __init__(self, config: AriaModelConfig) -> None:
        self.config = config
        self.calls: list[dict[str, Any]] = []
        self._gate_calibration = load_gate_calibration(
            Path(config.gate_calibration_path or config.model),
            required=config.gate_calibration_path is not None,
            probability_source=config.gate_probability_source,
        )
        api_key = (
            os.environ.get(config.api_key_env or "") if config.api_key_env else None
        )
        if not api_key and config.api_key_required:
            raise RuntimeError(f"missing API key env var: {config.api_key_env}")
        self._api_key = api_key

    def predict_gate(self, incident: IncidentRecord) -> GatePrediction:
        raw = self._complete(
            system=GATE_INSTRUCTION,
            payload=incident.model_input(),
        )
        probability = raw.get("gate_probability")
        if not isinstance(probability, (int, float)):
            raise TypeError("ARIA gate response must contain numeric gate_probability")
        return GatePrediction(calibrate_gate(probability, self._gate_calibration))

    def attribute(
        self,
        incident: IncidentRecord,
        localization: FailureLocalization | None = None,
    ) -> RepairAttribution:
        selected_localization = localization or self.localize(incident)
        return self.attribute_localized(incident, selected_localization)

    def localize(self, incident: IncidentRecord) -> FailureLocalization:
        return _validated_stage(
            lambda payload: self._complete(system=RUNTIME_LOCALIZATION_INSTRUCTION if self.config.runtime_context else LOCALIZATION_INSTRUCTION, payload=payload),
            incident.model_input(),
            lambda raw: FailureLocalization.from_dict(raw, events=incident.events, name="ARIA model localization"),
            retries=self.config.structured_output_retries, calls=self.calls,
        )

    def attribute_localized(
        self,
        incident: IncidentRecord,
        localization: FailureLocalization,
    ) -> RepairAttribution:
        return _validated_stage(
            lambda payload: self._complete(system=RUNTIME_ATTRIBUTION_INSTRUCTION if self.config.runtime_context else ATTRIBUTION_INSTRUCTION, payload=payload),
            incident.attribution_model_input(localization),
            lambda raw: _parse_attribution(raw, incident, localization, name="ARIA model attribution"),
            retries=self.config.structured_output_retries, calls=self.calls,
        )

    def propose_repair(
        self,
        incident: IncidentRecord,
        attribution: RepairAttribution | None = None,
    ) -> RepairRecord:
        selected = attribution or incident.attribution
        if selected is None:
            selected = self.attribute(incident)
        return _validated_stage(
            lambda payload: self._complete(system=STRUCTURED_REPAIR_INSTRUCTION, payload=payload),
            incident.patch_model_input(selected),
            lambda raw: RepairRecord.from_dict(raw, events=incident.events, name="ARIA model repair"),
            retries=self.config.structured_output_retries, calls=self.calls,
        )

    def _complete(self, *, system: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        score_gate = system == GATE_INSTRUCTION and self.config.gate_probability_source == "label_likelihood"
        body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": canonical_model_json(payload),
                },
            ],
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.config.temperature is not None:
            body["temperature"] = self.config.temperature
        if self.config.seed is not None:
            body["seed"] = self.config.seed
        if score_gate:
            body.update(logprobs=True, top_logprobs=20, max_tokens=min(32, self.config.max_tokens))
        budget = None
        if self.config.tokenizer_endpoint is not None:
            fitted, budget = fit_messages(
                body["messages"], render=lambda items: canonical_model_json(prepare_chat_messages(items, self.config.system_role_mode)),
                tokenizer=_EndpointChatTokenizer(self), max_tokens=self.config.max_input_tokens,
            )
            body["messages"] = json.loads(fitted)
        else:
            body["messages"] = prepare_chat_messages(body["messages"], self.config.system_role_mode)
        request = urllib.request.Request(
            _chat_endpoint(self.config.base_url or ""),
            data=json.dumps(body).encode("utf-8"),
            headers=self._headers(),
            method="POST",
        )
        call: dict[str, Any] = {
            "messages": deepcopy(body["messages"]), "model": self.config.model,
            "generation_settings": {k: v for k, v in body.items() if k not in {"model", "messages"}},
            "context_budget": budget,
        }
        self.calls.append(call)
        try:
            with urllib.request.urlopen(
                request, timeout=self.config.timeout_seconds
            ) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read(1000).decode("utf-8", errors="replace")
            call["error"] = f"HTTP {exc.code}: {detail}"
            raise RuntimeError(
                f"ARIA model endpoint returned HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            call["error"] = str(exc.reason)
            raise RuntimeError(
                f"ARIA model endpoint is unavailable: {exc.reason}"
            ) from exc
        if not isinstance(decoded, Mapping):
            raise TypeError("ARIA model endpoint returned a non-object response")
        try:
            content = decoded["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                "ARIA model endpoint returned no chat completion content"
            ) from exc
        if not isinstance(content, str):
            raise TypeError("ARIA model completion content must be a string")
        call["raw_response"] = content
        call["response_metadata"] = {key: decoded.get(key) for key in ("id", "model", "usage")}
        if score_gate:
            call["response_metadata"]["gate_logprobs"] = decoded["choices"][0].get("logprobs")
        try:
            result = json.loads(content)
        except json.JSONDecodeError as exc:
            call["error"] = str(exc)
            raise ValueError(
                "ARIA model must return schema-constrained JSON only"
            ) from exc
        if not isinstance(result, dict):
            raise TypeError("ARIA model JSON response must be an object")
        call["parsed_response"] = deepcopy(result)
        if score_gate:
            if set(result) != {"gate_probability"}:
                raise ValueError("gate response must contain only gate_probability")
            calibrate_gate(result["gate_probability"], None)
            probability = chat_label_probability(decoded["choices"][0].get("logprobs"))
            if probability is None:
                raise ValueError("endpoint did not return both gate-label log probabilities")
            result["gate_probability"] = probability
        return result

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers


class _EndpointChatTokenizer:
    """Use the serving model's chat template/tokenizer for budgeting only."""

    def __init__(self, model: OpenAIChatAriaModel) -> None:
        self.model = model

    def encode(self, serialized_messages: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        request = urllib.request.Request(
            self.model.config.tokenizer_endpoint,
            data=json.dumps({"model": self.model.config.model, "messages": json.loads(serialized_messages),
                             "add_generation_prompt": True}).encode(),
            headers=self.model._headers(), method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.model.config.timeout_seconds) as response:
                result = json.loads(response.read().decode())
        except (urllib.error.HTTPError, urllib.error.URLError) as exc:
            raise RuntimeError(f"ARIA endpoint tokenization failed: {exc}") from exc
        tokens = result.get("tokens") if isinstance(result, Mapping) else None
        if not isinstance(tokens, list) or not all(type(token) is int for token in tokens):
            raise RuntimeError("ARIA tokenizer endpoint did not return token IDs")
        if result.get("count") != len(tokens):
            raise RuntimeError("ARIA tokenizer count differs from its token IDs")
        return tokens


class TransformersAriaModel:
    """Lazy local loader for any causal-LM checkpoint supported by Transformers."""

    def __init__(self, config: AriaModelConfig) -> None:
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "transformers ARIA serving requires Transformers and a compatible torch runtime; "
                "install the chosen model stack or use an endpoint provider"
            ) from exc
        self.config = config
        self.calls: list[dict[str, Any]] = []
        self._gate_calibration = load_gate_calibration(
            Path(config.gate_calibration_path or config.model),
            required=config.gate_calibration_path is not None,
            probability_source=config.gate_probability_source,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            fix_mistral_regex=True,
        )
        adapter_config = Path(config.model) / "adapter_config.json"
        if adapter_config.is_file():
            try:
                from peft import (
                    AutoPeftModelForCausalLM,  # type: ignore[import-not-found]
                )
            except ImportError as exc:
                raise RuntimeError(
                    "loading an ARIA adapter checkpoint requires PEFT"
                ) from exc
            self._model = AutoPeftModelForCausalLM.from_pretrained(
                config.model,
                dtype="auto",
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                config.model,
                dtype="auto",
            )
        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        else:
            self._device = torch.device("cpu")
        self._model.to(self._device)
        self._model.eval()
        self._model.config.use_cache = True

    def predict_gate(self, incident: IncidentRecord) -> GatePrediction:
        raw = self._generate_json(
            GATE_INSTRUCTION,
            incident.model_input(),
        )
        probability = raw.get("gate_probability")
        if not isinstance(probability, (int, float)):
            raise TypeError("ARIA gate response must contain numeric gate_probability")
        return GatePrediction(calibrate_gate(probability, self._gate_calibration))

    def attribute(
        self,
        incident: IncidentRecord,
        localization: FailureLocalization | None = None,
    ) -> RepairAttribution:
        selected_localization = localization or self.localize(incident)
        return self.attribute_localized(incident, selected_localization)

    def localize(self, incident: IncidentRecord) -> FailureLocalization:
        return _validated_stage(
            lambda payload: self._generate_json(RUNTIME_LOCALIZATION_INSTRUCTION if self.config.runtime_context else LOCALIZATION_INSTRUCTION, payload),
            incident.model_input(),
            lambda raw: FailureLocalization.from_dict(raw, events=incident.events, name="ARIA model localization"),
            retries=self.config.structured_output_retries, calls=self.calls,
        )

    def attribute_localized(
        self,
        incident: IncidentRecord,
        localization: FailureLocalization,
    ) -> RepairAttribution:
        return _validated_stage(
            lambda payload: self._generate_json(RUNTIME_ATTRIBUTION_INSTRUCTION if self.config.runtime_context else ATTRIBUTION_INSTRUCTION, payload),
            incident.attribution_model_input(localization),
            lambda raw: _parse_attribution(raw, incident, localization, name="ARIA model attribution"),
            retries=self.config.structured_output_retries, calls=self.calls,
        )

    def propose_repair(
        self,
        incident: IncidentRecord,
        attribution: RepairAttribution | None = None,
    ) -> RepairRecord:
        selected = attribution or incident.attribution
        if selected is None:
            selected = self.attribute(incident)
        return _validated_stage(
            lambda payload: self._generate_json(STRUCTURED_REPAIR_INSTRUCTION, payload),
            incident.patch_model_input(selected),
            lambda raw: RepairRecord.from_dict(raw, events=incident.events, name="ARIA model repair"),
            retries=self.config.structured_output_retries, calls=self.calls,
        )

    def _generate_json(
        self, instruction: str, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": instruction},
            {
                "role": "user",
                "content": canonical_model_json(payload),
            },
        ]
        call: dict[str, Any] = {"messages": deepcopy(messages), "model": self.config.model,
                               "generation_settings": {"temperature": self.config.temperature, "seed": self.config.seed}}
        self.calls.append(call)
        score_gate = instruction == GATE_INSTRUCTION and self.config.gate_probability_source == "label_likelihood"
        response_prefix = '{"gate_probability":' if score_gate else ""
        generation_limit = min(32, self.config.max_tokens) if score_gate else self.config.max_tokens
        def render(items: list[dict[str, Any]]) -> str:
            if getattr(self._tokenizer, "chat_template", None):
                return render_chat_prompt(self._tokenizer, items, self.config.system_role_mode)
            return f"{items[0]['content']}\nINPUT:\n{items[1]['content']}\nOUTPUT:\n"
        max_context = int(
            getattr(self._model.config, "max_position_embeddings", 0)
            or getattr(self._tokenizer, "model_max_length", 0)
            or 0
        )
        input_budget = max_context - generation_limit if max_context > 0 else (self.config.max_input_tokens or 8192)
        if self.config.max_input_tokens is not None:
            input_budget = min(input_budget, self.config.max_input_tokens)
        if input_budget <= 0:
            raise ValueError("ARIA max_tokens leaves no model context for the incident input")
        prompt, _ = fit_messages(messages, render=render, tokenizer=self._tokenizer,
                                 max_tokens=input_budget, response_prefix=response_prefix)
        tokens = self._tokenizer(prompt, return_tensors="pt", truncation=False, add_special_tokens=False)
        tokens = {key: value.to(self._device) for key, value in tokens.items()}
        generation_config = deepcopy(self._model.generation_config)
        generation_config.do_sample = self.config.temperature not in {None, 0.0}
        if generation_config.do_sample:
            generation_config.temperature = self.config.temperature
        else:
            generation_config.temperature = None
            generation_config.top_p = None
            generation_config.top_k = None
        generation_options: dict[str, Any] = {
            "max_new_tokens": generation_limit,
            "generation_config": generation_config,
            "return_dict_in_generate": score_gate,
            "output_scores": score_gate,
        }
        if self.config.seed is None:
            generated = self._model.generate(**tokens, **generation_options)
        else:
            import torch

            device = torch.device(self._device)
            devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.random.default_generator.manual_seed(self.config.seed)
                if devices:
                    with torch.cuda.device(devices[0]):
                        torch.cuda.manual_seed(self.config.seed)
                generated = self._model.generate(**tokens, **generation_options)
        probability = None
        if score_gate:
            import torch
            probability = label_probability(generated.scores, self._tokenizer, torch,
                int(generated.sequences[0, tokens["input_ids"].shape[1]]))
            generated = generated.sequences
        text = self._tokenizer.decode(
            generated[0][tokens["input_ids"].shape[1] :], skip_special_tokens=True
        )
        text = text if text.lstrip().startswith("{") else response_prefix + text
        call["raw_response"] = text
        raw = _decode_json_object(text)
        if not isinstance(raw, dict):
            raise TypeError("local ARIA model JSON response must be an object")
        call["parsed_response"] = deepcopy(raw)
        if score_gate:
            if set(raw) != {"gate_probability"}:
                raise ValueError("gate response must contain only gate_probability")
            calibrate_gate(raw["gate_probability"], None)
            if probability is None:
                raise ValueError("model did not produce a scorable gate-label token")
            raw["gate_probability"] = probability
        return raw


def _validated_stage(
    generate: Callable[[Mapping[str, Any]], dict[str, Any]], payload: Mapping[str, Any],
    validate: Callable[[dict[str, Any]], Any], *, retries: int, calls: list[dict[str, Any]],
) -> Any:
    """Retry only host-rejected structure/grounding, never benchmark outcomes."""
    current = deepcopy(dict(payload))
    for attempt in range(retries + 1):
        before = len(calls)
        raw = None
        try:
            raw = generate(current)
            value = validate(raw)
        except (ValueError, TypeError) as exc:
            call = calls[-1] if len(calls) > before else None
            if call is not None:
                call["host_validation"] = {"status": "invalid", "error": str(exc), "attempt": attempt + 1}
            if raw is None and (call is None or "raw_response" not in call):
                raise
            if attempt == retries:
                raise
            prior = call.get("raw_response") if call else None
            current = deepcopy(dict(payload))
            current["constraints"] = {**dict(current.get("constraints") or {}), "structured_output_feedback": {
                "validation_error": str(exc),
                "previous_output": bounded_text(prior or canonical_model_json(raw), 2000),
                "requirement": "Correct the JSON using only the original observable evidence. Do not invent evidence or outcomes.",
            }}
            continue
        if len(calls) > before:
            calls[-1]["host_validation"] = {"status": "valid", "error": None, "attempt": attempt + 1}
        return value
    raise AssertionError("unreachable structured generation loop")


def _parse_attribution(
    raw: Mapping[str, Any],
    incident: IncidentRecord,
    localization: FailureLocalization,
    *,
    name: str,
) -> RepairAttribution:
    return RepairAttribution.from_dict(
        raw,
        events=incident.events,
        localization=localization,
        name=name,
    )


def _head_tail_token_tensors(tokens: Mapping[str, Any], budget: int) -> dict[str, Any]:
    input_ids = tokens.get("input_ids")
    if input_ids is None or input_ids.shape[-1] <= budget:
        return dict(tokens)
    head = max(1, budget // 3)
    tail = budget - head
    indexes = [*range(head)]
    if tail:
        indexes.extend(range(input_ids.shape[-1] - tail, input_ids.shape[-1]))
    compact: dict[str, Any] = {}
    for key, value in tokens.items():
        if getattr(value, "ndim", 0) == 2 and value.shape[-1] == input_ids.shape[-1]:
            compact[key] = value[:, indexes]
        else:
            compact[key] = value
    return compact


def _chat_endpoint(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    suffix = "/chat/completions"
    if normalized.endswith(suffix):
        return normalized
    if normalized.endswith("/v1"):
        return normalized + suffix
    return normalized + "/v1" + suffix


def _decode_json_object(text: str) -> dict[str, Any]:
    """Parse one JSON-only object from a local model completion."""

    try:
        raw = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ValueError(
            "local ARIA model must decode schema-constrained JSON"
        ) from exc
    if not isinstance(raw, dict):
        raise TypeError("local ARIA model JSON response must be an object")
    return raw
