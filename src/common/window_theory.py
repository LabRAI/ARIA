"""Shared Window-Theory runtime, sampling, and statistical contracts."""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from common.mast_data import load_benchmark_items

MAS_SLUGS = ("openmanus", "metagpt", "magentic-one", "ag2")
BENCHMARK_SLUGS = (
    "gsm-plus",
    "mmlu",
    "humaneval",
    "olympiadbench",
    "gaia",
    "swe-bench-lite",
)


@dataclass(frozen=True)
class RuntimeSpec:
    slug: str
    runtime_id: str
    model_id: str
    provider: str
    protocol: str
    base_url: str
    model: str
    api_key_env: str
    max_tokens: int
    context_length: int
    input_price_per_million: float
    output_price_per_million: float
    temperature: float | None = None
    reasoning_effort: str | None = None
    upstream_provider: str | None = None
    allow_fallbacks: bool | None = None

    def api_overrides(self) -> dict[str, Any]:
        values = asdict(self)
        for key in (
            "slug",
            "context_length",
            "input_price_per_million",
            "output_price_per_million",
        ):
            values.pop(key)
        values["api_key_required"] = True
        return {key: value for key, value in values.items() if value is not None}


RUNTIMES = {
    spec.slug: spec
    for spec in (
        RuntimeSpec(
            slug="openrouter-alibaba-qwen3-8b-nonthinking",
            runtime_id="runtime/openrouter-alibaba-qwen3-8b-nonthinking",
            model_id="model/qwen3-8b-04-28",
            provider="openrouter",
            protocol="openai-chat-completions",
            base_url="https://api.example.invalid/api/v1",
            model="qwen/qwen3-8b",
            api_key_env="OPENAI_API_KEY",
            max_tokens=8192,
            temperature=0.0,
            upstream_provider="Alibaba",
            allow_fallbacks=False,
            context_length=131072,
            input_price_per_million=0.117,
            output_price_per_million=0.455,
        ),
        RuntimeSpec(
            slug="openrouter-alibaba-qwen3-32b-nonthinking",
            runtime_id="runtime/openrouter-alibaba-qwen3-32b-nonthinking",
            model_id="model/qwen3-32b-04-28",
            provider="openrouter",
            protocol="openai-chat-completions",
            base_url="https://api.example.invalid/api/v1",
            model="qwen/qwen3-32b",
            api_key_env="OPENAI_API_KEY",
            max_tokens=8192,
            temperature=0.0,
            upstream_provider="Alibaba",
            allow_fallbacks=False,
            context_length=131072,
            input_price_per_million=0.104,
            output_price_per_million=0.416,
        ),
        RuntimeSpec(
            slug="openrouter-deepinfra-llama-3-1-70b-instruct",
            runtime_id="runtime/openrouter-deepinfra-llama-3-1-70b-instruct",
            model_id="model/llama-3-1-70b-instruct",
            provider="openrouter",
            protocol="openai-chat-completions",
            base_url="https://api.example.invalid/api/v1",
            model="meta-llama/llama-3.1-70b-instruct",
            api_key_env="OPENAI_API_KEY",
            max_tokens=8192,
            temperature=0.0,
            upstream_provider="DeepInfra",
            allow_fallbacks=False,
            context_length=131072,
            input_price_per_million=0.4,
            output_price_per_million=0.4,
        ),
        RuntimeSpec(
            slug="endpoint-a-gpt-5-5-medium",
            runtime_id="runtime/endpoint-a-gpt-5-5-medium",
            model_id="model/gpt-5-5",
            provider="openai-compatible",
            protocol="openai-chat-completions",
            base_url="https://api.example.invalid",
            model="gpt-5.5",
            api_key_env="OPENAI_API_KEY",
            max_tokens=8192,
            reasoning_effort="medium",
            context_length=400000,
            input_price_per_million=5.0,
            output_price_per_million=30.0,
        ),
        RuntimeSpec(
            slug="endpoint-a-gpt-5-5-high",
            runtime_id="runtime/endpoint-a-gpt-5-5-high",
            model_id="model/gpt-5-5",
            provider="openai-compatible",
            protocol="openai-chat-completions",
            base_url="https://api.example.invalid",
            model="gpt-5.5",
            api_key_env="OPENAI_API_KEY",
            max_tokens=8192,
            reasoning_effort="high",
            context_length=400000,
            input_price_per_million=5.0,
            output_price_per_million=30.0,
        ),
        RuntimeSpec(
            slug="endpoint-a-gpt-5-5-xhigh",
            runtime_id="runtime/endpoint-a-gpt-5-5-xhigh",
            model_id="model/gpt-5-5",
            provider="openai-compatible",
            protocol="openai-chat-completions",
            base_url="https://api.example.invalid",
            model="gpt-5.5",
            api_key_env="OPENAI_API_KEY",
            max_tokens=8192,
            reasoning_effort="xhigh",
            context_length=400000,
            input_price_per_million=5.0,
            output_price_per_million=30.0,
        ),
        RuntimeSpec(
            slug="endpoint-b-claude-sonnet-4-6",
            runtime_id="runtime/endpoint-b-claude-sonnet-4-6",
            model_id="model/claude-sonnet-4-6",
            provider="claude-gateway",
            protocol="anthropic-messages",
            base_url="https://api.example.invalid",
            model="claude-sonnet-4-6",
            api_key_env="ANTHROPIC_API_KEY",
            max_tokens=8192,
            temperature=0.0,
            context_length=1000000,
            input_price_per_million=3.0,
            output_price_per_million=15.0,
        ),
        RuntimeSpec(
            slug="endpoint-b-claude-opus-4-8",
            runtime_id="runtime/endpoint-b-claude-opus-4-8",
            model_id="model/claude-opus-4-8",
            provider="claude-gateway",
            protocol="anthropic-messages",
            base_url="https://api.example.invalid",
            model="claude-opus-4-8",
            api_key_env="ANTHROPIC_API_KEY",
            max_tokens=8192,
            temperature=0.0,
            context_length=1000000,
            input_price_per_million=5.0,
            output_price_per_million=25.0,
        ),
    )
}

def expected_runtime_slugs(mas: str, benchmark: str) -> tuple[str, ...]:
    """Return all eight runtimes required for every MAS/benchmark surface."""

    if mas not in MAS_SLUGS:
        raise ValueError(f"unsupported MAS: {mas}")
    if benchmark not in BENCHMARK_SLUGS:
        raise ValueError(f"unsupported benchmark: {benchmark}")
    return tuple(RUNTIMES)

UNIQUE_TARGETS = {
    "gsm-plus": 500,
    "mmlu": 500,
    "humaneval": 164,
    "olympiadbench": 500,
    "gaia": 165,
    "swe-bench-lite": 300,
}


def benchmark_split(benchmark: str) -> str:
    return "validation" if benchmark == "gaia" else "test"


def benchmark_stratum(benchmark: str, item: dict[str, Any]) -> str:
    key = {
        "gsm-plus": "perturbation_type",
        "mmlu": "subject",
        "olympiadbench": "subfield",
        "gaia": "level",
        "swe-bench-lite": "repo",
    }.get(benchmark)
    return str(item.get(key) or "all") if key else "all"


def eligible_benchmark_items(data_dir: Path, benchmark: str) -> list[dict[str, Any]]:
    items = load_benchmark_items(
        data_dir,
        benchmark,
        source="full",
        split=benchmark_split(benchmark),
    )
    if benchmark == "olympiadbench":
        items = [
            item
            for item in items
            if int(item.get("sample_index", -1)) >= 1215
            and bool(item.get("reference_answers") or item.get("reference_answer"))
        ]
    return items


def stratified_order(
    items: list[dict[str, Any]],
    *,
    stratum: Callable[[dict[str, Any]], str],
    prefix_ids: list[str] | None = None,
) -> list[str]:
    """Return a deterministic round-robin order with a fixed reusable prefix."""

    by_id = {str(item["sample_id"]): item for item in items}
    prefix = list(prefix_ids or [])
    missing = [sample_id for sample_id in prefix if sample_id not in by_id]
    if missing:
        raise ValueError("sample prefix IDs are missing: " + ", ".join(missing[:10]))
    if len(set(prefix)) != len(prefix):
        raise ValueError("sample prefix IDs must be unique")

    grouped: dict[str, deque[str]] = defaultdict(deque)
    prefix_set = set(prefix)
    for item in items:
        sample_id = str(item["sample_id"])
        if sample_id not in prefix_set:
            grouped[stratum(item)].append(sample_id)

    order = list(prefix)
    strata = sorted(grouped)
    while strata:
        next_strata: list[str] = []
        for name in strata:
            values = grouped[name]
            if values:
                order.append(values.popleft())
            if values:
                next_strata.append(name)
        strata = next_strata
    return order


def build_sample_plan(data_dir: Path) -> dict[str, list[str]]:
    """Build the capped sample plan used only by Window-Theory screening."""

    plan: dict[str, list[str]] = {}
    for benchmark in BENCHMARK_SLUGS:
        items = eligible_benchmark_items(data_dir, benchmark)
        order = stratified_order(
            items,
            stratum=lambda item, benchmark=benchmark: benchmark_stratum(benchmark, item),
        )
        target = min(UNIQUE_TARGETS[benchmark], len(order))
        plan[benchmark] = order[:target]
    return plan


def build_full_benchmark_plan(data_dir: Path) -> dict[str, list[str]]:
    """Return every evaluator-eligible sample ID for each benchmark."""

    plan: dict[str, list[str]] = {}
    for benchmark in BENCHMARK_SLUGS:
        items = eligible_benchmark_items(data_dir, benchmark)
        plan[benchmark] = stratified_order(
            items,
            stratum=lambda item, benchmark=benchmark: benchmark_stratum(
                benchmark, item
            ),
        )
    return plan


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 1.0)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return (max(0.0, center - radius), min(1.0, center + radius))


def one_sided_exact_mcnemar(control_only: int, patched_only: int) -> float:
    discordant = control_only + patched_only
    if discordant == 0:
        return 1.0
    return sum(
        math.comb(discordant, value) * (0.5**discordant)
        for value in range(patched_only, discordant + 1)
    )
