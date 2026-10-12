"""Benchmark capability registry shared by loading, evaluation, and repair."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkRegistration:
    slug: str
    kind: str
    loader_name: str
    evaluator_name: str
    implemented: bool = True


BENCHMARK_REGISTRY = {
    "gsm-plus": BenchmarkRegistration("gsm-plus", "text", "gsm_plus", "gsm_plus"),
    "mmlu": BenchmarkRegistration("mmlu", "text", "mmlu", "mmlu"),
    "humaneval": BenchmarkRegistration("humaneval", "text", "humaneval", "humaneval"),
    "olympiadbench": BenchmarkRegistration(
        "olympiadbench", "text", "olympiadbench", "olympiadbench"
    ),
    "gaia": BenchmarkRegistration("gaia", "agent", "gaia", "gaia"),
    "swe-bench-lite": BenchmarkRegistration(
        "swe-bench-lite", "agent", "swe_bench_lite", "swe_bench_lite"
    ),
}


def get_benchmark_registration(slug: str) -> BenchmarkRegistration:
    try:
        registration = BENCHMARK_REGISTRY[slug]
    except KeyError as exc:
        raise NotImplementedError(f"benchmark adapter is not registered: {slug}") from exc
    if not registration.implemented:
        raise NotImplementedError(f"benchmark adapter is not implemented: {slug}")
    return registration
