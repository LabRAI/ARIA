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
        slug="langgraph", paper_name="LangGraph Debate",
        repo_url="https://github.com/langchain-ai/langgraph.git",
        clone_dir="langgraph", branch=None,
        launch_entrypoint="python -m inference.native.langgraph.launcher",
        paper_benchmarks=("gsm-plus", "mmlu", "olympiadbench", "bbh", "arc-challenge"),
        roles=("Solver", "Critic", "Synthesizer"),
    ),
    MasSpec(
        slug="chatdev",
        paper_name="ChatDev",
        repo_url="https://github.com/OpenBMB/ChatDev.git",
        clone_dir="chatdev",
        branch="chatdev1.0",
        launch_entrypoint="python run.py",
        paper_benchmarks=("programdev",),
        roles=("CEO", "CTO", "Programmer", "Code Reviewer", "Software Test Engineer"),
    ),
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
        slug="hyperagent",
        paper_name="HyperAgent",
        repo_url="https://github.com/FSoft-AI4Code/HyperAgent.git",
        clone_dir="hyperagent",
        branch=None,
        launch_entrypoint="python main.py",
        paper_benchmarks=("swe-bench-lite",),
        roles=("Planner", "Navigator", "Editor", "Executor"),
    ),
    MasSpec(
        slug="appworld",
        paper_name="AppWorld",
        repo_url="https://github.com/StonyBrookNLP/appworld.git",
        clone_dir="appworld",
        branch=None,
        launch_entrypoint="python experiments/run.py",
        paper_benchmarks=("appworld-test-c",),
        roles=("Supervisor", "Service Agent", "Verifier"),
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


from common.ood_benchmarks import ARC_REVISION, BBH_REVISION, BBH_TASKS

BENCHMARK_SPECS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        slug="bbh", paper_name="BIG-Bench Hard", mad_name=None,
        files=tuple(RemoteFile(
            url=f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/{BBH_REVISION}/bbh/{task}.json",
            path=f"{task}.json",
        ) for task in BBH_TASKS),
    ),
    BenchmarkSpec(
        slug="arc-challenge", paper_name="ARC-Challenge", mad_name=None,
        files=(RemoteFile(
            url=f"https://huggingface.co/datasets/allenai/ai2_arc/resolve/{ARC_REVISION}/ARC-Challenge/test-00000-of-00001.parquet",
            path="test.parquet",
        ),),
    ),
    BenchmarkSpec(
        slug="programdev",
        paper_name="ProgramDev",
        mad_name="ProgramDev",
        files=(
            RemoteFile(
                url=(
                    "https://raw.githubusercontent.com/"
                    "multi-agent-systems-failure-taxonomy/MAST/main/"
                    "traces/programdev/programdev_dataset.json"
                ),
                path="programdev_dataset.json",
            ),
        ),
    ),
    BenchmarkSpec(
        slug="programdev-v2",
        paper_name="ProgramDev-v2",
        mad_name=None,
        files=(),
        skip_reason=(
            "The paper describes ProgramDev-v2, but the official MAST repository "
            "does not currently expose a separate easy-download ProgramDev-v2 file."
        ),
    ),
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
        slug="appworld-test-c",
        paper_name="AppWorld Test-C",
        mad_name="Test-C",
        files=(
            RemoteFile(
                url=hf_dataset_url("LukaszTP/AppWorld-Tasks", "samples.json"),
                path="samples.json",
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
