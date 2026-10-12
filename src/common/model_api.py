"""Strict model API protocols used by inference and repair jobs.

Provider names are metadata. ``protocol`` selects the HTTP path, request body,
authentication headers, and response parser. This keeps a self-hosted vLLM
server independent from the OpenAI and Anthropic wire protocols it exposes.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

OPENAI_RESPONSES = "openai-responses"
OPENAI_CHAT_COMPLETIONS = "openai-chat-completions"
ANTHROPIC_MESSAGES = "anthropic-messages"

SUPPORTED_PROVIDERS = {
    "openai",
    "anthropic",
    "vllm",
    "openai-compatible",
    "openrouter",
    "claude-gateway",
}
SUPPORTED_PROTOCOLS = {
    OPENAI_RESPONSES,
    OPENAI_CHAT_COMPLETIONS,
    ANTHROPIC_MESSAGES,
}
PROVIDER_BASE_URLS = {
    "openrouter": "https://api.example.invalid/api/v1",
}
SUPPORTED_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
MASFM_USER_AGENT = "MASFM/1.0"
MAX_RETRY_DELAY_SECONDS = 120.0


class ModelApiError(RuntimeError):
    """The configured model endpoint failed to return a usable response."""


@dataclass(frozen=True)
class ContextBudget:
    """A server-reported context limit and the safe completion budget it leaves."""

    context_window_tokens: int
    prompt_tokens: int
    requested_completion_tokens: int
    safe_completion_tokens: int
    safety_margin_tokens: int


@dataclass(frozen=True)
class ApiConfig:
    provider: str
    protocol: str
    base_url: str
    model: str
    api_key_env: str | None
    api_key_required: bool
    timeout_seconds: int
    max_tokens: int
    retries: int
    retry_seconds: float
    context_window_tokens: int | None = None
    temperature: float | None = None
    reasoning_effort: str | None = None
    runtime_id: str | None = None
    model_id: str | None = None
    seed: int | None = None
    anthropic_version: str = "2023-06-01"
    upstream_provider: str | None = None
    allow_fallbacks: bool | None = None

    def __post_init__(self) -> None:
        if self.provider not in SUPPORTED_PROVIDERS:
            raise ValueError(f"unsupported api.provider: {self.provider}")
        if self.protocol not in SUPPORTED_PROTOCOLS:
            raise ValueError(f"unsupported api.protocol: {self.protocol}")
        if self.provider == "openai" and self.protocol != OPENAI_RESPONSES:
            raise ValueError("api.provider=openai requires api.protocol=openai-responses")
        if self.provider == "anthropic" and self.protocol != ANTHROPIC_MESSAGES:
            raise ValueError("api.provider=anthropic requires api.protocol=anthropic-messages")
        if self.provider == "vllm" and self.protocol == ANTHROPIC_MESSAGES:
            raise ValueError("api.provider=vllm does not use the Anthropic Messages protocol")
        if self.provider == "openrouter" and self.protocol != OPENAI_CHAT_COMPLETIONS:
            raise ValueError(
                f"api.provider={self.provider} requires api.protocol=openai-chat-completions"
            )
        if self.provider == "claude-gateway" and self.protocol != ANTHROPIC_MESSAGES:
            raise ValueError(
                "api.provider=claude-gateway requires api.protocol=anthropic-messages"
            )
        if "claude" in self.model.lower() and self.protocol != ANTHROPIC_MESSAGES:
            raise ValueError("Claude models require api.protocol=anthropic-messages")
        expected_base_url = PROVIDER_BASE_URLS.get(self.provider)
        if expected_base_url and self.base_url.rstrip("/") != expected_base_url:
            raise ValueError(
                f"api.provider={self.provider} requires api.base_url={expected_base_url}"
            )
        _validate_base_url(self.base_url)
        if not self.model.strip():
            raise ValueError("api.model must be non-empty")
        if self.api_key_env is not None and not self.api_key_env.strip():
            raise ValueError("api.api_key_env must be non-empty when provided")
        if self.api_key_required and self.api_key_env is None:
            raise ValueError("api.api_key_env is required when api.api_key_required is true")
        if self.timeout_seconds <= 0:
            raise ValueError("api.timeout_seconds must be greater than zero")
        if self.max_tokens <= 0:
            raise ValueError("api.max_tokens must be greater than zero")
        if self.context_window_tokens is not None:
            if self.context_window_tokens <= 0:
                raise ValueError("api.context_window_tokens must be greater than zero")
            if self.context_window_tokens < self.max_tokens:
                raise ValueError(
                    "api.context_window_tokens must be at least api.max_tokens"
                )
        if self.retries < 0:
            raise ValueError("api.retries must be zero or greater")
        if self.retry_seconds < 0:
            raise ValueError("api.retry_seconds must be zero or greater")
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("api.temperature must be zero or greater")
        if self.reasoning_effort is not None:
            if self.protocol not in {OPENAI_RESPONSES, OPENAI_CHAT_COMPLETIONS}:
                raise ValueError(
                    "api.reasoning_effort is only valid for openai-responses "
                    "or openai-chat-completions"
                )
            if self.reasoning_effort not in SUPPORTED_REASONING_EFFORTS:
                raise ValueError(
                    "api.reasoning_effort must be one of: none, minimal, low, "
                    "medium, high, xhigh, max"
                )
        if self.seed is not None and self.protocol != OPENAI_CHAT_COMPLETIONS:
            raise ValueError("api.seed is only valid for openai-chat-completions")
        if self.protocol == ANTHROPIC_MESSAGES and not self.anthropic_version.strip():
            raise ValueError("api.anthropic_version must be non-empty")
        if self.upstream_provider is not None:
            if self.provider != "openrouter":
                raise ValueError("api.upstream_provider is only valid for api.provider=openrouter")
            if not self.upstream_provider.strip():
                raise ValueError("api.upstream_provider must be non-empty")
        if self.allow_fallbacks is not None and self.provider != "openrouter":
            raise ValueError("api.allow_fallbacks is only valid for api.provider=openrouter")

    @property
    def endpoint_url(self) -> str:
        return endpoint_url(self.base_url, self.protocol)

    @property
    def api_root_url(self) -> str:
        """Return the OpenAI-compatible API root used by native SDK clients."""

        if self.protocol not in {OPENAI_CHAT_COMPLETIONS, OPENAI_RESPONSES}:
            raise ValueError(f"protocol {self.protocol} has no OpenAI-compatible API root")
        return api_root_url(self.base_url)


@dataclass(frozen=True)
class ModelResponse:
    text: str
    response_id: str | None
    usage: dict[str, Any]
    stop_reason: str | None
    latency_seconds: float
    attempt_count: int
    protocol: str
    raw: dict[str, Any] = field(repr=False)
    context_budget: ContextBudget | None = None

    def to_record(self) -> dict[str, Any]:
        record = {
            "response_id": self.response_id,
            "protocol": self.protocol,
            "usage": self.usage,
            "stop_reason": self.stop_reason,
            "latency_seconds": self.latency_seconds,
            "attempt_count": self.attempt_count,
            "context_budget": (
                {
                    "context_window_tokens": self.context_budget.context_window_tokens,
                    "prompt_tokens": self.context_budget.prompt_tokens,
                    "requested_completion_tokens": self.context_budget.requested_completion_tokens,
                    "safe_completion_tokens": self.context_budget.safe_completion_tokens,
                    "safety_margin_tokens": self.context_budget.safety_margin_tokens,
                }
                if self.context_budget is not None
                else None
            ),
        }
        choices = self.raw.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            if isinstance(choices[0].get("logprobs"), dict):
                record["token_logprobs"] = choices[0]["logprobs"]
        return record


class ModelApiClient:
    """Synchronous client for the three explicitly supported model protocols."""

    def __init__(self, config: ApiConfig) -> None:
        api_key = os.environ.get(config.api_key_env) if config.api_key_env else None
        if not api_key and config.api_key_required:
            raise RuntimeError(f"missing API key env var: {config.api_key_env}")
        self.config = config
        self.api_key = api_key

    def complete(self, messages: list[dict[str, Any]]) -> str:
        """Compatibility surface for evaluators and injected test clients."""

        return self.invoke(messages).text

    def invoke(self, messages: list[dict[str, Any]]) -> ModelResponse:
        _validate_messages(messages)
        payload = self._request_payload(messages)
        headers = self._headers()
        started = time.monotonic()
        raw, attempts, context_budget = self._post_json(payload, headers)
        latency = time.monotonic() - started
        return self._parse_response(
            raw,
            latency_seconds=latency,
            attempt_count=attempts,
            context_budget=context_budget,
        )

    def invoke_json(self, messages: list[dict[str, Any]], *, request_options: dict[str, Any] | None = None) -> ModelResponse:
        """Request an object-shaped completion when the protocol supports it."""

        _validate_messages(messages)
        payload = self._request_payload(
            messages, response_format={"type": "json_object"}
        )
        if request_options:
            if self.config.protocol != OPENAI_CHAT_COMPLETIONS:
                raise ValueError("per-call log probabilities require Chat Completions")
            if set(request_options) - {"logprobs", "top_logprobs", "response_format"}:
                raise ValueError("unsupported JSON request options")
            if "response_format" in request_options and not isinstance(request_options["response_format"], dict):
                raise ValueError("response_format must be an object")
            if "logprobs" in request_options and type(request_options["logprobs"]) is not bool:
                raise ValueError("logprobs must be boolean")
            if "top_logprobs" in request_options and (
                type(request_options["top_logprobs"]) is not int or not 0 <= request_options["top_logprobs"] <= 20
            ):
                raise ValueError("top_logprobs must be an integer from 0 to 20")
            payload.update(request_options)
        headers = self._headers()
        started = time.monotonic()
        raw, attempts, context_budget = self._post_json(payload, headers)
        latency = time.monotonic() - started
        return self._parse_response(
            raw,
            latency_seconds=latency,
            attempt_count=attempts,
            context_budget=context_budget,
        )

    def _request_payload(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        config = self.config
        if config.protocol == OPENAI_RESPONSES:
            payload: dict[str, Any] = {
                "model": config.model,
                "input": messages,
                "max_output_tokens": config.max_tokens,
            }
            if config.reasoning_effort is not None:
                payload["reasoning"] = {"effort": config.reasoning_effort}
            if config.temperature is not None:
                payload["temperature"] = config.temperature
            return payload
        if config.protocol == OPENAI_CHAT_COMPLETIONS:
            payload = {
                "model": config.model,
                "messages": messages,
            }
            if config.reasoning_effort is not None:
                payload["max_completion_tokens"] = config.max_tokens
                payload["reasoning_effort"] = config.reasoning_effort
            else:
                payload["max_tokens"] = config.max_tokens
            if config.temperature is not None and config.reasoning_effort is None:
                payload["temperature"] = config.temperature
            if config.seed is not None:
                payload["seed"] = config.seed
            if response_format is not None:
                payload["response_format"] = response_format
            payload.update(
                openai_compatible_extra_body(
                    config.base_url,
                    config.model,
                    upstream_provider=config.upstream_provider,
                    allow_fallbacks=config.allow_fallbacks,
                )
            )
            return payload
        if config.protocol == ANTHROPIC_MESSAGES:
            system_messages = [
                str(message["content"])
                for message in messages
                if message["role"] in {"system", "developer"}
            ]
            conversation = [
                {"role": message["role"], "content": message["content"]}
                for message in messages
                if message["role"] not in {"system", "developer"}
            ]
            if not conversation:
                raise ValueError("anthropic-messages requires at least one user or assistant message")
            payload = {
                "model": config.model,
                "messages": conversation,
                "max_tokens": config.max_tokens,
            }
            if system_messages:
                payload["system"] = "\n\n".join(system_messages)
            if config.temperature is not None:
                payload["temperature"] = config.temperature
            return payload
        raise AssertionError(f"unhandled protocol: {config.protocol}")

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": MASFM_USER_AGENT,
        }
        if self.config.protocol == ANTHROPIC_MESSAGES:
            headers["anthropic-version"] = self.config.anthropic_version
            if self.api_key is not None:
                if self.config.provider == "claude-gateway":
                    headers["Authorization"] = f"Bearer {self.api_key}"
                else:
                    headers["x-api-key"] = self.api_key
        elif self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _post_json(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> tuple[dict[str, Any], int, ContextBudget | None]:
        data = json.dumps(payload).encode("utf-8")
        normal_attempts = self.config.retries + 1
        normal_attempt = 0
        request_attempt = 0
        context_budget: ContextBudget | None = None
        logger = logging.getLogger("masfm.model_api")
        while normal_attempt < normal_attempts:
            request_attempt += 1
            request = urllib.request.Request(
                self.config.endpoint_url,
                data=data,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.config.timeout_seconds,
                ) as response:
                    body = response.read().decode("utf-8")
                try:
                    decoded = json.loads(body)
                except json.JSONDecodeError as exc:
                    normal_attempt += 1
                    if normal_attempt == normal_attempts:
                        raise ModelApiError("model API returned invalid JSON") from exc
                    retry_delay = retry_delay_seconds(
                        self.config.retry_seconds,
                        normal_attempt,
                    )
                    logger.warning(
                        "model API returned malformed JSON; retrying attempt %s/%s "
                        "after %.1fs",
                        normal_attempt + 1,
                        normal_attempts,
                        retry_delay,
                    )
                    if retry_delay:
                        time.sleep(retry_delay)
                    continue
                if not isinstance(decoded, dict):
                    raise ModelApiError("model API response root must be a JSON object")
                if not _response_contains_output_text(decoded, self.config.protocol):
                    normal_attempt += 1
                    if normal_attempt == normal_attempts:
                        return decoded, request_attempt, context_budget
                    retry_delay = retry_delay_seconds(
                        self.config.retry_seconds,
                        normal_attempt,
                    )
                    logger.warning(
                        "model API returned no output text; retrying attempt %s/%s "
                        "after %.1fs",
                        normal_attempt + 1,
                        normal_attempts,
                        retry_delay,
                    )
                    if retry_delay:
                        time.sleep(retry_delay)
                    continue
                return decoded, request_attempt, context_budget
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                budget = context_budget_from_error(detail, payload)
                if budget is not None and context_budget is None:
                    payload = _with_context_budget(payload, budget.safe_completion_tokens)
                    data = json.dumps(payload).encode("utf-8")
                    context_budget = budget
                    logger.warning(
                        "model API context budget %s/%s; retrying once with max output %s",
                        budget.prompt_tokens,
                        budget.context_window_tokens,
                        budget.safe_completion_tokens,
                    )
                    continue
                normal_attempt += 1
                retryable = is_retryable_http_error(exc.code, detail)
                if not retryable or normal_attempt == normal_attempts:
                    raise ModelApiError(
                        f"model API request failed with HTTP {exc.code}: {detail}"
                    ) from exc
                retry_delay = retry_delay_seconds(
                    self.config.retry_seconds,
                    normal_attempt,
                )
                logger.warning(
                    "model API returned HTTP %s; retrying attempt %s/%s after %.1fs",
                    exc.code,
                    normal_attempt + 1,
                    normal_attempts,
                    retry_delay,
                )
                if retry_delay:
                    time.sleep(retry_delay)
            except (
                urllib.error.URLError,
                TimeoutError,
                OSError,
                http.client.HTTPException,
            ) as exc:
                normal_attempt += 1
                if normal_attempt == normal_attempts:
                    raise ModelApiError(f"model API request failed: {exc}") from exc
                retry_delay = retry_delay_seconds(
                    self.config.retry_seconds,
                    normal_attempt,
                )
                logger.warning(
                    "model API transport error %s; retrying attempt %s/%s after %.1fs",
                    type(exc).__name__,
                    normal_attempt + 1,
                    normal_attempts,
                    retry_delay,
                )
                if retry_delay:
                    time.sleep(retry_delay)
        raise AssertionError("retry loop exited without returning or raising")

    def _parse_response(
        self,
        raw: dict[str, Any],
        *,
        latency_seconds: float,
        attempt_count: int,
        context_budget: ContextBudget | None,
    ) -> ModelResponse:
        protocol = self.config.protocol
        if protocol == OPENAI_RESPONSES:
            text_parts = []
            for item in raw.get("output", []):
                if not isinstance(item, dict) or item.get("type") != "message":
                    continue
                for content in item.get("content", []):
                    if isinstance(content, dict) and content.get("type") == "output_text":
                        text_parts.append(str(content.get("text", "")))
            text = "".join(text_parts)
            stop_reason = str(raw.get("status")) if raw.get("status") is not None else None
        elif protocol == OPENAI_CHAT_COMPLETIONS:
            choices = raw.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise ModelApiError("chat completions response is missing choices[0]")
            message = choices[0].get("message")
            if not isinstance(message, dict):
                raise ModelApiError(
                    "chat completions response is missing choices[0].message"
                )
            content = message.get("content")
            try:
                text = _content_text(content)
            except RuntimeError as exc:
                raise ModelApiError(str(exc)) from exc
            stop_reason = (
                str(choices[0].get("finish_reason"))
                if choices[0].get("finish_reason") is not None
                else None
            )
        elif protocol == ANTHROPIC_MESSAGES:
            content = raw.get("content")
            if not isinstance(content, list):
                raise ModelApiError("Anthropic Messages response is missing content")
            text = "".join(
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
            stop_reason = (
                str(raw.get("stop_reason")) if raw.get("stop_reason") is not None else None
            )
        else:
            raise AssertionError(f"unhandled protocol: {protocol}")
        if not text:
            raise ModelApiError(f"{protocol} response contains no output text")
        usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        return ModelResponse(
            text=text,
            response_id=str(raw["id"]) if raw.get("id") is not None else None,
            usage=dict(usage),
            stop_reason=stop_reason,
            latency_seconds=latency_seconds,
            attempt_count=attempt_count,
            protocol=protocol,
            raw=raw,
            context_budget=context_budget,
        )


def is_retryable_http_error(status_code: int, detail: str) -> bool:
    if status_code in {408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}:
        return True
    return status_code == 403 and is_capacity_exhausted_error(detail)


def is_capacity_exhausted_error(detail: str) -> bool:
    """Return whether a gateway explicitly reports transient upstream unavailability."""

    normalized = detail.lower()
    return any(
        marker in normalized
        for marker in (
            "all available accounts exhausted",
            "no available accounts",
            "upstream request failed, please try again",
            "upstream provider service failure",
        )
    )


def retry_delay_seconds(base_seconds: float, failed_attempts: int) -> float:
    """Return capped exponential backoff with jitter for a transient API failure."""

    if base_seconds <= 0:
        return 0.0
    exponential = min(
        base_seconds * (2 ** max(0, failed_attempts - 1)),
        MAX_RETRY_DELAY_SECONDS,
    )
    return min(exponential * random.uniform(0.8, 1.2), MAX_RETRY_DELAY_SECONDS)


def invoke_anthropic_messages_gateway(
    *,
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    retries: int,
    retry_seconds: float,
    anthropic_version: str = "2023-06-01",
) -> tuple[dict[str, Any], int]:
    """Invoke a Bearer-auth Anthropic gateway without SDK fingerprint headers."""

    data = json.dumps(payload).encode("utf-8")
    attempts = retries + 1
    logger = logging.getLogger("masfm.model_api")
    for attempt in range(attempts):
        delay = 0.0
        request = urllib.request.Request(
            endpoint_url(base_url, ANTHROPIC_MESSAGES),
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "anthropic-version": anthropic_version,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": MASFM_USER_AGENT,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                decoded = json.loads(response.read().decode("utf-8"))
            if not isinstance(decoded, dict):
                raise ModelApiError("Anthropic gateway response root must be a JSON object")
            return decoded, attempt + 1
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if (
                attempt + 1 == attempts
                or not is_retryable_http_error(exc.code, detail)
            ):
                raise ModelApiError(
                    f"model API request failed with HTTP {exc.code}: {detail}"
                ) from exc
            delay = retry_delay_seconds(retry_seconds, attempt + 1)
            logger.warning(
                "Anthropic gateway returned HTTP %s; retrying attempt %s/%s after %.1fs",
                exc.code,
                attempt + 2,
                attempts,
                delay,
            )
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            http.client.HTTPException,
        ) as exc:
            if attempt + 1 == attempts:
                raise ModelApiError(f"model API request failed: {exc}") from exc
            delay = retry_delay_seconds(retry_seconds, attempt + 1)
            logger.warning(
                "Anthropic gateway transport error %s; retrying attempt %s/%s after %.1fs",
                type(exc).__name__,
                attempt + 2,
                attempts,
                delay,
            )
        if delay:
            time.sleep(delay)
    raise AssertionError("Anthropic gateway retry loop exited without returning or raising")


def context_budget_from_error(
    detail: str,
    payload: dict[str, Any],
    *,
    safety_margin_tokens: int = 256,
) -> ContextBudget | None:
    """Parse vLLM's deterministic context-budget HTTP 400 response."""

    match = re.search(
        r"maximum context length is\s+(?P<context>\d+)\s+tokens.*?request has\s+"
        r"(?P<prompt>\d+)\s+input tokens",
        detail,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return None
    requested = _payload_completion_tokens(payload)
    if requested is None:
        return None
    context_window = int(match.group("context"))
    prompt_tokens = int(match.group("prompt"))
    safe_completion = context_window - prompt_tokens - safety_margin_tokens
    if safe_completion <= 0 or safe_completion >= requested:
        return None
    return ContextBudget(
        context_window_tokens=context_window,
        prompt_tokens=prompt_tokens,
        requested_completion_tokens=requested,
        safe_completion_tokens=safe_completion,
        safety_margin_tokens=safety_margin_tokens,
    )


def _payload_completion_tokens(payload: dict[str, Any]) -> int | None:
    for field_name in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
        value = payload.get(field_name)
        if isinstance(value, int) and value > 0:
            return value
    return None


def _with_context_budget(payload: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    updated = dict(payload)
    for field_name in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
        if field_name in updated:
            updated[field_name] = max_tokens
            return updated
    raise ValueError("model API payload has no completion-token field")


def endpoint_url(base_url: str, protocol: str) -> str:
    root = api_root_url(base_url)
    paths = {
        OPENAI_RESPONSES: "responses",
        OPENAI_CHAT_COMPLETIONS: "chat/completions",
        ANTHROPIC_MESSAGES: "messages",
    }
    try:
        suffix = paths[protocol]
    except KeyError as exc:
        raise ValueError(f"unsupported protocol: {protocol}") from exc
    return f"{root}/{suffix}"


def api_root_url(base_url: str) -> str:
    """Normalize ``https://host`` and ``https://host/v1`` to one API root."""

    _validate_base_url(base_url)
    value = base_url.rstrip("/")
    parsed = urllib.parse.urlsplit(value)
    if parsed.path.rstrip("/").endswith("/v1"):
        return value
    return f"{value}/v1"


def openai_compatible_extra_body(
    base_url: str,
    model: str,
    *,
    upstream_provider: str | None = None,
    allow_fallbacks: bool | None = None,
) -> dict[str, Any]:
    """Return provider parameters required for a usable final-answer channel."""

    extra: dict[str, Any] = {}
    normalized_base_url = base_url.lower()
    normalized_model = model.lower()
    is_qwen3 = normalized_model.startswith(("qwen/qwen3", "qwen3-"))
    if "api.example.invalid" in normalized_base_url and is_qwen3:
        extra["reasoning"] = {"enabled": False, "exclude": True}
    hostname = urllib.parse.urlsplit(base_url).hostname
    if hostname in {"127.0.0.1", "localhost", "::1"} and is_qwen3:
        extra["chat_template_kwargs"] = {"enable_thinking": False}
    if "api.example.invalid" in normalized_base_url and (
        upstream_provider is not None or allow_fallbacks is not None
    ):
        route: dict[str, Any] = {}
        if upstream_provider is not None:
            route["order"] = [upstream_provider]
        if allow_fallbacks is not None:
            route["allow_fallbacks"] = allow_fallbacks
        extra["provider"] = route
    return extra


def anthropic_payload_from_openai_chat(
    params: dict[str, Any],
    *,
    default_max_tokens: int,
) -> dict[str, Any]:
    """Translate one OpenAI-style chat request into Anthropic Messages format."""

    system_parts: list[str] = []
    messages: list[dict[str, Any]] = []
    for raw_message in params.get("messages", []):
        if not isinstance(raw_message, dict):
            continue
        role = str(raw_message.get("role") or "")
        content = raw_message.get("content")
        if role in {"system", "developer"}:
            text = _openai_content_text(content)
            if text:
                system_parts.append(text)
            continue
        if role == "tool":
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": str(raw_message.get("tool_call_id") or ""),
                            "content": _openai_content_text(content),
                        }
                    ],
                }
            )
            continue
        if role not in {"user", "assistant"}:
            continue
        blocks = _openai_content_blocks(content)
        if role == "assistant":
            for tool_call in raw_message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if not isinstance(function, dict):
                    continue
                arguments = function.get("arguments")
                try:
                    tool_input = json.loads(arguments) if isinstance(arguments, str) else arguments
                except json.JSONDecodeError:
                    tool_input = {"raw_arguments": str(arguments)}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": str(tool_call.get("id") or ""),
                        "name": str(function.get("name") or ""),
                        "input": tool_input if isinstance(tool_input, dict) else {},
                    }
                )
        messages.append(
            {
                "role": role,
                "content": blocks if blocks else "",
            }
        )

    if not messages:
        raise ValueError("Anthropic bridge requires at least one user or assistant message")
    max_tokens = params.get("max_tokens", params.get("max_completion_tokens"))
    payload: dict[str, Any] = {
        "model": str(params["model"]),
        "messages": messages,
        "max_tokens": int(max_tokens or default_max_tokens),
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)
    if params.get("temperature") is not None:
        payload["temperature"] = float(params["temperature"])
    tools = []
    for tool in params.get("tools") or []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            continue
        function = tool.get("function")
        if not isinstance(function, dict):
            continue
        tools.append(
            {
                "name": str(function.get("name") or ""),
                "description": str(function.get("description") or ""),
                "input_schema": function.get("parameters") or {"type": "object"},
            }
        )
    if tools:
        payload["tools"] = tools
        tool_choice = params.get("tool_choice")
        if tool_choice == "required":
            payload["tool_choice"] = {"type": "any"}
        elif isinstance(tool_choice, dict):
            function = tool_choice.get("function")
            if isinstance(function, dict) and function.get("name"):
                payload["tool_choice"] = {
                    "type": "tool",
                    "name": str(function["name"]),
                }
        elif tool_choice != "none":
            payload["tool_choice"] = {"type": "auto"}
    return payload


def openai_chat_response_from_anthropic(raw: dict[str, Any]) -> dict[str, Any]:
    """Translate an Anthropic message response into an OpenAI ChatCompletion shape."""

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in raw.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text_parts.append(str(block.get("text") or ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": str(block.get("id") or ""),
                    "type": "function",
                    "function": {
                        "name": str(block.get("name") or ""),
                        "arguments": json.dumps(
                            block.get("input") if isinstance(block.get("input"), dict) else {},
                            ensure_ascii=False,
                        ),
                    },
                }
            )
    stop_reason = str(raw.get("stop_reason") or "")
    finish_reason = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }.get(stop_reason, "stop")
    usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts) or None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": str(raw.get("id") or "anthropic-message"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": str(raw.get("model") or ""),
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def _openai_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        )
    return "" if content is None else str(content)


def _openai_content_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if isinstance(content, list):
        return [
            {"type": "text", "text": str(item.get("text") or "")}
            for item in content
            if isinstance(item, dict)
            and item.get("type") in {"text", "output_text"}
            and item.get("text")
        ]
    return []


def _response_contains_output_text(raw: dict[str, Any], protocol: str) -> bool:
    if protocol == OPENAI_RESPONSES:
        return any(
            isinstance(content, dict)
            and content.get("type") == "output_text"
            and bool(content.get("text"))
            for item in raw.get("output", [])
            if isinstance(item, dict) and item.get("type") == "message"
            for content in item.get("content", [])
        )
    if protocol == OPENAI_CHAT_COMPLETIONS:
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return False
        message = choices[0].get("message")
        if not isinstance(message, dict):
            return False
        try:
            return bool(_content_text(message.get("content")))
        except RuntimeError:
            return False
    if protocol == ANTHROPIC_MESSAGES:
        content = raw.get("content")
        return isinstance(content, list) and any(
            isinstance(block, dict) and block.get("type") == "text" and bool(block.get("text"))
            for block in content
        )
    raise AssertionError(f"unhandled protocol: {protocol}")


def _validate_base_url(base_url: str) -> None:
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("api.base_url must be an absolute http(s) URL")
    if parsed.query or parsed.fragment:
        raise ValueError("api.base_url must not contain a query string or fragment")
    terminal = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    if terminal in {"responses", "messages", "completions"}:
        raise ValueError(
            "api.base_url must be an API root, not a full operation endpoint"
        )


def _validate_messages(messages: list[dict[str, Any]]) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    allowed_roles = {"system", "developer", "user", "assistant"}
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(f"messages[{index}] must be an object")
        if set(message) != {"role", "content"}:
            raise ValueError(f"messages[{index}] must contain exactly role and content")
        if message["role"] not in allowed_roles:
            raise ValueError(f"unsupported message role: {message['role']}")
        if not isinstance(message["content"], (str, list)):
            raise TypeError(f"messages[{index}].content must be a string or content block list")


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}
        )
    raise RuntimeError("model response content must be a string or content block list")
