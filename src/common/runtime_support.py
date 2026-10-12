"""Shared run naming, logging, and runtime-state construction."""

from __future__ import annotations

import logging
import time
from typing import Any


def safe_slug(value: str, *, fallback_prefix: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")
    return safe or f"{fallback_prefix}-{time.time_ns()}"


def safe_run_id(value: str) -> str:
    return safe_slug(value, fallback_prefix="run")


def safe_name(value: str) -> str:
    return safe_slug(value, fallback_prefix="dataset")


def configure_root_logging(level: str) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level, logging.INFO))
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, level, logging.INFO))
    console.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    root.addHandler(console)


def runtime_state_from_result(
    result: dict[str, Any],
    api: Any | None = None,
) -> dict[str, Any]:
    mas = str(result.get("mas") or "mas")
    benchmark = str(result.get("benchmark") or "benchmark")
    model = str(result.get("model") or getattr(api, "model", "unknown-model"))
    return {
        "runtime_id": (
            result.get("runtime_id")
            or getattr(api, "runtime_id", None)
            or f"runtime/{safe_run_id(mas)}-{safe_run_id(benchmark)}-{safe_run_id(model)}"
        ),
        "provider": result.get("provider") or getattr(api, "provider", "openai-compatible"),
        "protocol": result.get("protocol") or getattr(api, "protocol", None),
        "base_url": result.get("base_url") or getattr(api, "base_url", None),
        "endpoint_url": result.get("endpoint_url") or getattr(api, "endpoint_url", None),
        "upstream_provider": result.get("upstream_provider") or getattr(api, "upstream_provider", None),
        "allow_fallbacks": result.get(
            "allow_fallbacks",
            getattr(api, "allow_fallbacks", None),
        ),
        "api_key_env": result.get("api_key_env") or getattr(api, "api_key_env", None),
        "api_key_required": result.get(
            "api_key_required",
            getattr(api, "api_key_required", True),
        ),
        "model": model,
        "model_id": (
            result.get("model_id")
            or getattr(api, "model_id", None)
            or f"model/{safe_run_id(model)}"
        ),
        "generation": {
            "temperature": getattr(api, "temperature", None),
            "max_tokens": getattr(api, "max_tokens", None),
            "context_window_tokens": getattr(api, "context_window_tokens", None),
            "seed": getattr(api, "seed", None),
            "reasoning_effort": (
                getattr(api, "reasoning_effort", None)
                if api is not None
                else result.get("reasoning_effort")
            ),
        },
        "native_repo_path": result.get("native_repo_path"),
        "native_repo_head": result.get("native_repo_head"),
        "workspace_path": result.get("workspace_path"),
        "bridge": result.get("bridge"),
    }
