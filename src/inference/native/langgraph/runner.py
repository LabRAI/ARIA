"""LangGraph package backend, isolated from the four original MAS stacks."""
from __future__ import annotations

import json
import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.artifacts import atomic_write_json, atomic_write_jsonl, atomic_write_text
from common.model_api import ANTHROPIC_MESSAGES, OPENAI_CHAT_COMPLETIONS
from inference.native.base import NativeMasBackend, NativeRunContext
from inference.native.runtime_hooks import export_runtime_bootstrap
from inference.native.shared import safe_api_settings, timeout_text, write_aria_artifacts

SUPPORTED_BENCHMARKS = frozenset({"gsm-plus", "mmlu", "olympiadbench", "bbh", "arc-challenge"})


@dataclass(frozen=True)
class LangGraphResult:
    transcript: list[dict[str, Any]]
    final_output: str
    aria_export: dict[str, Any] | None
    status: str
    error: str | None


def run(context: NativeRunContext) -> LangGraphResult:
    if context.benchmark.slug not in SUPPORTED_BENCHMARKS:
        raise ValueError("LangGraph Debate supports text reasoning only; unsupported benchmark: " + context.benchmark.slug)
    root = context.run_root
    root.mkdir(parents=True, exist_ok=True)
    if context.aria is not None:
        context.host_adapter.bootstrap(context.aria, context.mas, context.benchmark,
                                       context.prompt, list(context.mas.roles))
    request = {"task": context.prompt, "api": safe_api_settings(context.api_settings),
               "rounds": int(context.options.get("max_turns", 2)),
               "aria": export_runtime_bootstrap(context.aria)}
    request_path = root / "langgraph_request.json"
    atomic_write_json(request_path, request)
    # Honor the caller's explicit uv environment; otherwise use the dedicated stack.
    interpreter = context.repo_root / ".venv-langgraph" / "bin" / "python"
    if importlib.util.find_spec("langgraph") is not None:
        interpreter = Path(sys.executable)
    if not interpreter.exists():
        raise FileNotFoundError("Install LangGraph: UV_PROJECT_ENVIRONMENT=.venv-langgraph uv sync --extra langgraph")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(context.repo_root / "src"), env.get("PYTHONPATH", "")])
    output_path = root / "langgraph_output.json"
    output_path.unlink(missing_ok=True)
    try:
        process = subprocess.run(
            [str(interpreter), "-m", "inference.native.langgraph.launcher", str(request_path.resolve())],
            cwd=context.repo_root, env=env, capture_output=True, text=True,
            timeout=int(context.options.get("native_timeout_seconds", context.default_timeout_seconds * 8)),
        )
        stdout, stderr = process.stdout, process.stderr
        output = json.loads(output_path.read_text()) if output_path.exists() else {}
        if process.returncode != 0 or output.get("status") != "completed":
            output.update(status="failed", error=output.get("error") or stderr[-4000:] or "missing native output")
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = timeout_text(exc.stdout), timeout_text(exc.stderr)
        output = {"status": "failed", "error": "native LangGraph timeout"}
    atomic_write_text(root / "langgraph_stdout.log", stdout)
    atomic_write_text(root / "langgraph_stderr.log", stderr)
    atomic_write_json(root / "langgraph_messages.json", output.get("messages", []))
    atomic_write_jsonl(root / "langgraph_native_trace.jsonl", output.get("trace", []))
    aria_export = output.get("aria_export")
    if aria_export is not None:
        write_aria_artifacts(root, aria_export)
    return LangGraphResult(output.get("transcript", []), output.get("final_output", ""),
                           aria_export, output["status"], output.get("error"))


BACKEND = NativeMasBackend(
    slug="langgraph", supported_protocols=(OPENAI_CHAT_COMPLETIONS, ANTHROPIC_MESSAGES),
    clone_marker="", artifact_prefix="langgraph", messages_filename="langgraph_messages.json",
    trace_filename="langgraph_native_trace.jsonl",
    artifact_filenames=("langgraph_request.json", "langgraph_output.json", "langgraph_messages.json",
                        "langgraph_native_trace.jsonl", "langgraph_stdout.log", "langgraph_stderr.log"),
    launch_description="native LangGraph solver/critic loop and final synthesis",
    notes="Version-pinned package in .venv-langgraph; no clone required. Text-only, pending-message substitution.",
    run_native=run,
)
