"""Classify infrastructure failures without conflating them with wrong answers."""

from __future__ import annotations

from typing import Any


def classify_infrastructure_failure(error: Any) -> str | None:
    """Return a stable category for an unsuccessful native/evaluator result."""

    if error is None:
        return None
    text = str(error).lower()
    if not text:
        return None
    if "maximum context length" in text or "context window" in text:
        return "context_budget_invalid"
    if "tool protocol" in text or "unstructured tool" in text:
        return "tool_protocol_invalid"
    if any(marker in text for marker in ("swebench", "swe-bench", "docker", "harness")):
        return "harness_invalid"
    if any(
        marker in text
        for marker in (
            "http ",
            "api request",
            "connection",
            "timed out",
            "timeout",
            "rate limit",
            "authentication",
            "no choices",
        )
    ):
        return "endpoint_invalid"
    return "native_invalid"


def stable_infrastructure_reason(error: Any, category: str | None = None) -> str:
    """Return an allowlisted reason without copying raw provider payloads."""

    text = str(error or "").lower()
    if "official swe-bench harness did not produce an instance report" in text:
        return "official SWE-Bench harness did not produce an instance report"
    if "harness" in text and "did not produce an instance report" in text:
        return "official benchmark harness did not produce an instance report"
    if "524" in text or "proxy read timeout" in text:
        return "upstream endpoint returned HTTP 524 proxy read timeout"
    if "rate limit" in text or "too many requests" in text:
        return "provider quota or rate limit"
    if "timed out" in text or "timeout" in text:
        return "request or native timeout"
    if "connection" in text:
        return "provider connection failure"
    if "authentication" in text or "unauthorized" in text:
        return "provider authentication failure"
    if "no choices" in text:
        return "provider returned no choices"
    if "maximum context length" in text or "context window" in text:
        return "model context budget exceeded"
    if "tool protocol" in text or "unstructured tool" in text:
        return "provider tool protocol failure"
    if category == "harness_invalid":
        return "official benchmark harness unresolved"
    if category == "endpoint_invalid":
        return "upstream provider service failure"
    if category == "context_budget_invalid":
        return "model context budget exceeded"
    if category == "tool_protocol_invalid":
        return "provider tool protocol failure"
    return "native runtime failure"
