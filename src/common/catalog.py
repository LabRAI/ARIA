"""Static MAS and benchmark catalog definitions."""

from __future__ import annotations

import urllib.parse
from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class RemoteFile:
    url: str
    path: str


@dataclass(frozen=True)
class MasSpec:
    slug: str
    paper_name: str
    repo_url: str
    clone_dir: str
    branch: str | None
    launch_entrypoint: str
    paper_benchmarks: tuple[str, ...]
    roles: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkSpec:
    slug: str
    paper_name: str
    mad_name: str | None
    files: tuple[RemoteFile, ...]
    skip_reason: str | None = None


def hf_dataset_url(repo_id: str, path: str) -> str:
    quoted_repo = urllib.parse.quote(repo_id, safe="/")
    quoted_path = urllib.parse.quote(path, safe="/")
    return f"https://huggingface.co/datasets/{quoted_repo}/resolve/main/{quoted_path}?download=true"


MAS_SPECS: tuple[MasSpec, ...] = (
    MasSpec(
        slug="metagpt",
        paper_name="MetaGPT",
        repo_url="https://github.com/FoundationAgents/MetaGPT.git",
        clone_dir="metagpt",
        branch=None,
        launch_entrypoint="python main.py",
        paper_benchmarks=(
            "gsm-plus",
            "mmlu",
            "humaneval",
            "olympiadbench",
            "gaia",
            "swe-bench-lite",
        ),
        roles=("Product Manager", "Architect", "Engineer", "QA Engineer"),
    ),
    MasSpec(
        slug="ag2",
        paper_name="AG2 (MathChat)",
        repo_url="https://github.com/ag2ai/ag2.git",
        clone_dir="ag2",
        branch=None,
        launch_entrypoint="python examples/mathchat/run.py",
        paper_benchmarks=(
            "gsm-plus",
            "mmlu",
            "humaneval",
            "olympiadbench",
            "gaia",
            "swe-bench-lite",
        ),
        roles=("Student", "Python Assistant", "Verifier"),
    ),
    MasSpec(
        slug="magentic-one",
        paper_name="Magentic-One",
        repo_url="https://github.com/microsoft/autogen.git",
        clone_dir="magentic-one",
        branch=None,
        launch_entrypoint="python -m autogen_agentchat.teams.magentic_one",
        paper_benchmarks=(
            "gsm-plus",
            "mmlu",
            "humaneval",
            "olympiadbench",
            "gaia",
            "swe-bench-lite",
        ),
        roles=("Orchestrator", "Web Surfer", "Coder", "File Surfer"),
    ),
    MasSpec(
        slug="openmanus",
        paper_name="OpenManus",
        repo_url="https://github.com/FoundationAgents/OpenManus.git",
        clone_dir="openmanus",
        branch=None,
        launch_entrypoint="python main.py",
        paper_benchmarks=(
            "gsm-plus",
            "mmlu",
            "humaneval",
            "olympiadbench",
            "gaia",
            "swe-bench-lite",
        ),
        roles=("Planner", "Executor", "Verifier"),
    ),
)


BENCHMARK_SPECS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        slug="swe-bench-lite",
        paper_name="SWE-Bench Lite",
        mad_name="SWE-Bench-Lite",
        files=(
            RemoteFile(
                url=hf_dataset_url("SWE-bench/SWE-bench_Lite", "data/test-00000-of-00001.parquet"),
                path="data/test-00000-of-00001.parquet",
            ),
        ),
    ),
    BenchmarkSpec(
        slug="gsm-plus",
        paper_name="GSM-Plus",
        mad_name="GSM",
        files=(
            RemoteFile(
                url=hf_dataset_url("qintongli/GSM-Plus", "data/testmini-00000-of-00001.jsonl"),
                path="data/testmini-00000-of-00001.jsonl",
            ),
            RemoteFile(
                url=hf_dataset_url("qintongli/GSM-Plus", "data/test-00000-of-00001.jsonl"),
                path="data/test-00000-of-00001.jsonl",
            ),
        ),
    ),
    BenchmarkSpec(
        slug="olympiadbench",
        paper_name="OlympiadBench",
        mad_name="Olympiad",
        files=(
            RemoteFile(
                url=hf_dataset_url("lmms-lab/OlympiadBench", "data/test_en-00000-of-00001.parquet"),
                path="data/test_en-00000-of-00001.parquet",
            ),
        ),
    ),
    BenchmarkSpec(
        slug="mmlu",
        paper_name="MMLU",
        mad_name="MMLU",
        files=(
            RemoteFile(
                url=hf_dataset_url("cais/mmlu", "all/test-00000-of-00001.parquet"),
                path="all/test-00000-of-00001.parquet",
            ),
            RemoteFile(
                url=hf_dataset_url("cais/mmlu", "all/validation-00000-of-00001.parquet"),
                path="all/validation-00000-of-00001.parquet",
            ),
        ),
    ),
    BenchmarkSpec(
        slug="gaia",
        paper_name="GAIA",
        mad_name="GAIA",
        files=(),
        skip_reason=(
            "GAIA is auto-gated on Hugging Face; use the MAD-derived sample unless "
            "credentials are available."
        ),
    ),
    BenchmarkSpec(
        slug="humaneval",
        paper_name="HumanEval",
        mad_name=None,
        files=(
            RemoteFile(
                url=hf_dataset_url(
                    "openai/openai_humaneval", "openai_humaneval/test-00000-of-00001.parquet"
                ),
                path="openai_humaneval/test-00000-of-00001.parquet",
            ),
        ),
    ),
)


def list_catalog() -> dict[str, Any]:
    return {
        "mas": [asdict(spec) for spec in MAS_SPECS],
        "benchmarks": [asdict(spec) for spec in BENCHMARK_SPECS],
    }


def find_mas(slug: str) -> MasSpec:
    for spec in MAS_SPECS:
        if spec.slug == slug:
            return spec
    raise KeyError(f"unknown MAS slug: {slug}")


def find_benchmark(slug: str) -> BenchmarkSpec:
    for spec in BENCHMARK_SPECS:
        if spec.slug == slug:
            return spec
    raise KeyError(f"unknown benchmark slug: {slug}")


def selected_mas(slugs: Iterable[str] | None) -> list[MasSpec]:
    return list(MAS_SPECS) if not slugs else [find_mas(slug) for slug in slugs]


def selected_benchmarks(slugs: Iterable[str] | None) -> list[BenchmarkSpec]:
    return list(BENCHMARK_SPECS) if not slugs else [find_benchmark(slug) for slug in slugs]
