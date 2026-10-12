"""Serve an ARIA checkpoint locally so evaluation needs no inference server.

Every baseline reaches its model through :class:`~baselines.core.models.ModelAdapter`,
which needs only ``complete(messages) -> str``.  A post-trained ARIA checkpoint
is a LoRA adapter or a merged model on disk, so this client loads it with
transformers and generates directly.  Using the same adapter for ARIA and for
the baselines is what keeps a difference between rows attributable to the
method rather than to the serving stack.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aria.prompt_budget import PROMPT_BUDGET_CONTRACT, fit_messages
from aria.gate_scoring import label_probability as _gate_label_probability
from aria.calibration import load_gate_calibration

__all__ = ["LocalResponse", "LocalTransformersClient"]


@dataclass
class LocalResponse:
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return dict(self.metadata)


class LocalTransformersClient:
    """Greedy or sampled decoding from a local checkpoint.

    The model is loaded once per process and guarded by a lock, since the
    baseline runner may evaluate several methods against one client.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 512,
        gate_max_new_tokens: int | None = None,
        gate_probability_source: str = "generated",
        max_input_tokens: int = 3072,
        temperature: float = 0.0,
        top_p: float = 1.0,
        seed: int = 0,
    ) -> None:
        self.model_path = str(model_path)
        if gate_probability_source not in {"generated", "label_likelihood"}:
            raise ValueError("gate_probability_source must be generated or label_likelihood")
        self.gate_probability_source = gate_probability_source
        self.max_new_tokens = int(max_new_tokens)
        self.gate_max_new_tokens = (
            int(gate_max_new_tokens) if gate_max_new_tokens is not None else None
        )
        if self.gate_max_new_tokens is not None and self.gate_max_new_tokens < 1:
            raise ValueError("gate_max_new_tokens must be positive when provided")
        self.max_input_tokens = int(max_input_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.seed = int(seed)
        self._lock = threading.Lock()
        self.gate_calibration = _load_gate_calibration(Path(self.model_path), probability_source=gate_probability_source)

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.device = _resolve_device(torch, device)
        self.dtype = _resolve_dtype(torch, self.device, dtype)
        adapter_config = Path(self.model_path) / "adapter_config.json"
        if adapter_config.is_file():
            from peft import PeftModel

            config = json.loads(adapter_config.read_text(encoding="utf-8"))
            foundation = str(config["base_model_name_or_path"])
            self.foundation_model = foundation
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            base = AutoModelForCausalLM.from_pretrained(
                foundation, dtype=self.dtype, low_cpu_mem_usage=True
            )
            self.model = PeftModel.from_pretrained(base, self.model_path)
            self.checkpoint_kind = "peft-adapter"
        else:
            self.foundation_model = self.model_path
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, dtype=self.dtype, low_cpu_mem_usage=True
            )
            self.checkpoint_kind = "standalone"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.to(self.device)
        self.model.eval()
        self.model.config.use_cache = True
        self.call_count = 0

    def identity(self) -> dict[str, Any]:
        return {
            "provider": "local-transformers",
            "model": self.model_path,
            "foundation_model": self.foundation_model,
            "checkpoint_kind": self.checkpoint_kind,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "gate_max_new_tokens": self.gate_max_new_tokens,
            "gate_probability_source": self.gate_probability_source,
            "gate_calibration": self.gate_calibration,
            "max_input_tokens": self.max_input_tokens,
            "prompt_budget_contract": PROMPT_BUDGET_CONTRACT,
            "seed": self.seed,
        }

    def complete(self, messages: list[dict[str, Any]]) -> str:
        response = self.invoke(messages)
        return response.text

    def invoke_json(self, messages: list[dict[str, Any]]) -> LocalResponse:
        """JSON-capable alias used by ModelAdapter's bounded retry path."""
        return self.invoke(messages)

    def invoke(self, messages: list[dict[str, Any]]) -> LocalResponse:
        torch = self._torch
        response_prefix = _response_prefix(messages)
        prompt, budget_metadata = fit_messages(
            messages, render=self._render, tokenizer=self.tokenizer,
            max_tokens=self.max_input_tokens, response_prefix=response_prefix,
        )
        is_gate = bool(messages) and "ARIA intervention gate" in str(
            messages[0].get("content") or ""
        )
        generation_limit = self.max_new_tokens
        if is_gate and self.gate_max_new_tokens is not None:
            generation_limit = min(generation_limit, self.gate_max_new_tokens)
        encoded = self.tokenizer(
            prompt, return_tensors="pt", truncation=False, add_special_tokens=False,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        started = time.perf_counter()
        with self._lock:
            self.call_count += 1
            torch.manual_seed(self.seed + self.call_count)
            sample = self.temperature > 0.0
            with torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    do_sample=sample,
                    temperature=self.temperature if sample else None,
                    top_p=self.top_p if sample else None,
                    max_new_tokens=generation_limit,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=is_gate,
                    output_scores=is_gate,
                )
        prompt_length = encoded["input_ids"].shape[1]
        gate_metadata = {}
        if is_gate:
            gate_metadata = {
                "gate_label_probability": _gate_label_probability(
                    generated.scores, self.tokenizer, torch,
                    int(generated.sequences[0, prompt_length]),
                ),
                "gate_label_probability_kind": "conditional-first-token-0-vs-1",
            }
            generated = generated.sequences
        decoded = self.tokenizer.decode(
            generated[0][prompt_length:], skip_special_tokens=True
        )
        # Some instruction checkpoints repeat the complete JSON object even
        # when the prompt already contains the trained response prefix. Avoid
        # nesting the prefix into a second object; continuation-only samples
        # still receive the prefix required by the training contract.
        text = decoded if decoded.lstrip().startswith("{") else response_prefix + decoded
        if is_gate:
            gate_metadata["gate_probability_source"] = self.gate_probability_source
            gate_metadata["generated_gate_response"] = text
            if self.gate_probability_source == "label_likelihood":
                text = _label_scored_gate(text, gate_metadata["gate_label_probability"])
                if self.gate_calibration is None:
                    gate_metadata["checkpoint_calibration_disabled"] = (
                        "no checkpoint calibration fitted to label likelihoods"
                    )
        text, calibration_applied = _apply_gate_calibration(
            text,
            messages,
            self.gate_calibration,
        )
        return LocalResponse(
            text=text,
            metadata={
                "model": self.model_path,
                "prompt_tokens": int(prompt_length),
                "completion_tokens": int(generated.shape[1] - prompt_length),
                "generation_max_new_tokens": generation_limit,
                "generation_kind": "gate" if is_gate else "repair",
                "response_prefix": response_prefix,
                **budget_metadata,
                "elapsed_seconds": round(time.perf_counter() - started, 4),
                "gate_calibration_applied": calibration_applied,
                **gate_metadata,
            },
        )

    def _render(self, messages: list[dict[str, Any]]) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        parts = [f"{message['role'].upper()}:\n{message['content']}" for message in messages]
        return "\n\n".join([*parts, "ASSISTANT:\n"])


def _label_scored_gate(text: str, probability: float | None) -> str:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text
    value = payload.get("gate_probability") if isinstance(payload, dict) else None
    # Scoring must not turn a malformed generation into a valid decision.
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        return text
    if probability is None or not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("gate label likelihood is unavailable for this generated response")
    payload["gate_probability"] = probability
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def _resolve_device(torch: Any, requested: str) -> Any:
    if requested == "mps" or (requested == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_dtype(torch: Any, device: Any, requested: str) -> Any:
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def _load_gate_calibration(model_path: Path, *, probability_source: str = "generated") -> dict[str, float] | None:
    return load_gate_calibration(
        model_path / "gate_calibration.json", probability_source=probability_source,
        allow_source_mismatch=True,
    )


def _apply_gate_calibration(
    text: str,
    messages: list[dict[str, Any]],
    calibration: Mapping[str, float] | None,
) -> tuple[str, bool]:
    if calibration is None or not messages:
        return text, False
    system = str(messages[0].get("content") or "")
    if "ARIA intervention gate" not in system:
        return text, False
    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError:
        return text, False
    probability = payload.get("gate_probability") if isinstance(payload, dict) else None
    if not isinstance(probability, (int, float)) or isinstance(probability, bool):
        return text, False
    if not 0 <= probability <= 1:
        return text, False
    bounded = min(max(float(probability), 1e-5), 1.0 - 1e-5)
    logit = math.log(bounded / (1.0 - bounded))
    adjusted = (logit + calibration["logit_bias"]) / calibration["temperature"]
    payload["gate_probability"] = 1.0 / (
        1.0 + math.exp(-max(min(adjusted, 30.0), -30.0))
    )
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":")), True


def _response_prefix(messages: list[dict[str, Any]]) -> str:
    """Force the first structural fields used by the training serializer."""
    system = str(messages[0].get("content") or "") if messages else ""
    if "intervention gate" in system:
        return '{"gate_probability":'
    if "causal-error localization" in system:
        return '{"first_bad_event_ids":'
    if "repair-attribution" in system:
        return '{"diagnosis":"'
    if "structured trajectory-patch" in system or "repair decision model" in system:
        return '{"diagnosis":"'
    return ""
