"""MetaGPT native runner and registration.

The ignored MetaGPT checkout remains unmodified.  Each sample is executed in a
fresh child process using the active ``uv --extra metagpt`` environment, while
the parent prepares benchmark workspaces and converts MetaGPT messages into the
standard MASFM trajectory artifacts.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aria.runtime import AriaRuntime
from common.artifacts import atomic_write_json, atomic_write_jsonl
from common.catalog import BenchmarkSpec, MasSpec
from common.host_adapter import HostAdapter
from common.model_api import ANTHROPIC_MESSAGES, OPENAI_CHAT_COMPLETIONS
from inference.native.base import NativeMasBackend, NativeRunContext
from inference.native.shared import (
    hide_benchmark_labels,
    prepare_gaia_attachment,
    prepare_swe_bench_workspace,
    safe_api_settings,
    timeout_text,
    write_aria_artifacts,
)


@dataclass(frozen=True)
class MetaGPTNativeResult:
    transcript: list[dict[str, Any]]
    final_output: str
    aria_export: dict[str, Any] | None
    native_messages: list[dict[str, Any]]
    status: str
    error: str | None = None


def run_metagpt_native(
    *,
    repo_root: Path,
    clone_path: Path,
    run_root: Path,
    prompt: str,
    mas: MasSpec,
    benchmark: BenchmarkSpec,
    adapter: HostAdapter,
    aria: AriaRuntime | None,
    workspace_path: Path,
    sample: dict[str, Any] | None = None,
    max_rounds: int = 4,
    timeout_seconds: int = 2400,
    api_settings: dict[str, Any] | None = None,
    repository_context_mode: str = "legacy",
    context_budget_mode: str = "legacy",
) -> MetaGPTNativeResult:
    """Execute one benchmark sample through a native MetaGPT Team."""

    if not (clone_path / "metagpt").is_dir():
        raise FileNotFoundError(
            f"MetaGPT checkout not found at {clone_path} (missing 'metagpt/' package)"
        )
    settings = dict(api_settings or {})
    if settings.get("protocol") not in {OPENAI_CHAT_COMPLETIONS, ANTHROPIC_MESSAGES}:
        raise ValueError(
            "native MetaGPT requires protocol openai-chat-completions or anthropic-messages"
        )
    if not str(settings.get("model") or "").strip():
        raise ValueError("native MetaGPT requires a non-empty model")
    if max_rounds < 4:
        raise ValueError("native MetaGPT requires at least four rounds for its four roles")
    if repository_context_mode not in {"legacy", "symbol-aware"}:
        raise ValueError("repository_context_mode must be legacy or symbol-aware")
    if context_budget_mode not in {"legacy", "server-tokenizer"}:
        raise ValueError("context_budget_mode must be legacy or server-tokenizer")

    run_root.mkdir(parents=True, exist_ok=True)
    workspace_path.mkdir(parents=True, exist_ok=True)
    sample = dict(sample or {})
    if benchmark.slug == "swe-bench-lite":
        prepare_swe_bench_workspace(workspace_path, sample, run_root)
    elif benchmark.slug == "gaia":
        sample = prepare_gaia_attachment(workspace_path, sample, run_root)

    if aria is not None:
        adapter.bootstrap(aria, mas, benchmark, prompt, list(mas.roles))

    request = {
        "clone_path": str(clone_path),
        "run_root": str(run_root),
        "workspace_path": str(workspace_path),
        "benchmark": benchmark.slug,
        "prompt": prompt,
        "sample": hide_benchmark_labels(sample),
        "api_settings": safe_api_settings(settings),
        "max_rounds": max_rounds,
    }
    if repository_context_mode != "legacy":
        request["repository_context_mode"] = repository_context_mode
    if context_budget_mode != "legacy":
        request["context_budget_mode"] = context_budget_mode
    from inference.native.runtime_hooks import export_runtime_bootstrap

    request["aria_bootstrap"] = export_runtime_bootstrap(aria)
    request_path = run_root / "metagpt_request.json"
    atomic_write_json(request_path, request)
    command = [
        sys.executable,
        "-m",
        "inference.native.metagpt.launcher",
        str(request_path),
    ]
    environment = os.environ.copy()
    python_path = [str(repo_root / "src"), str(clone_path)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    runtime_root = run_root / "_metagpt_runtime"
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    _write_bootstrap_config(runtime_root, settings)
    (runtime_root / "home").mkdir(parents=True, exist_ok=True)
    environment["METAGPT_PROJECT_ROOT"] = str(runtime_root)
    environment["HOME"] = str(runtime_root / "home")

    trace: list[dict[str, Any]] = [
        {
            "kind": "launch",
            "command": command,
            "workspace_path": str(workspace_path),
            "max_rounds": max_rounds,
            "model": settings["model"],
            "reasoning_effort": settings.get("reasoning_effort"),
        }
    ]
    process_started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        returncode: int | None = completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = timeout_text(exc.stdout)
        stderr = timeout_text(exc.stderr)
        returncode = None
        trace.append(
            {
                "kind": "process_timeout",
                "timeout_seconds": timeout_seconds,
                "latency_seconds": time.monotonic() - process_started,
                "stdout_tail": stdout[-4000:],
                "stderr_tail": stderr[-4000:],
            }
        )
    (run_root / "metagpt_stdout.log").write_text(stdout, encoding="utf-8")
    (run_root / "metagpt_stderr.log").write_text(stderr, encoding="utf-8")
    if returncode is not None:
        trace.append(
            {
                "kind": "process_completed",
                "returncode": returncode,
                "latency_seconds": time.monotonic() - process_started,
                "stdout_tail": stdout[-4000:],
                "stderr_tail": stderr[-4000:],
            }
        )

    output_path = run_root / "metagpt_output.json"
    if output_path.exists():
        try:
            output = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            output = {"status": "failed", "error": f"invalid launcher output JSON: {exc}"}
    elif returncode is None:
        output = {
            "status": "failed",
            "error": f"native MetaGPT exceeded {timeout_seconds} seconds",
        }
    else:
        output = {
            "status": "failed",
            "error": f"launcher did not write {output_path}",
        }

    messages = list(output.get("messages", [])) if isinstance(output.get("messages"), list) else []
    transcript = list(output.get("transcript", [])) if isinstance(output.get("transcript"), list) else []
    trace.extend(list(output.get("trace", [])) if isinstance(output.get("trace"), list) else [])
    final_output = str(output.get("final_output") or "")
    status = "completed" if returncode == 0 and output.get("status") == "completed" else "failed"
    error = None if status == "completed" else str(output.get("error") or stderr[-4000:])

    atomic_write_json(run_root / "metagpt_messages.json", messages)
    aria_export = None
    if aria is not None and aria.config.model is not None:
        aria_export = output.get("aria_export") or aria.export_state()
        write_aria_artifacts(run_root, aria_export)
    elif aria is not None:
        profile = adapter.runtime_profile(mas, benchmark)
        for index, item in enumerate(transcript, start=1):
            content = str(item.get("content") or "")
            role = str(item.get("role") or item.get("source") or "MetaGPT")
            aria.after_action(profile.post_event, index, role, content)
        _, finalize_patch = aria.before_finalize(
            len(transcript) + 1,
            "QA Engineer",
            f"before_metagpt_delivery: finalize native MetaGPT run for {benchmark.paper_name}",
        )
        if finalize_patch is not None:
            trace.append({"kind": "finalize_patch", "record": finalize_patch})
        aria_export = aria.export_state()
        write_aria_artifacts(run_root, aria_export)

    atomic_write_jsonl(run_root / "metagpt_native_trace.jsonl", trace)
    return MetaGPTNativeResult(
        transcript=transcript,
        final_output=final_output,
        aria_export=aria_export,
        native_messages=messages,
        status=status,
        error=error,
    )


def _write_bootstrap_config(runtime_root: Path, settings: dict[str, Any]) -> None:
    """Write a secret-free config used only while MetaGPT imports config2."""

    config_dir = runtime_root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    base_url = str(settings["base_url"])
    model = str(settings["model"])
    (config_dir / "config2.yaml").write_text(
        "llm:\n"
        "  api_type: openai\n"
        "  api_key: REDACTED\n"
        f"  base_url: {base_url}\n"
        f"  model: {model}\n"
        "  stream: false\n",
        encoding="utf-8",
    )


def _run(context: NativeRunContext) -> MetaGPTNativeResult:
    return run_metagpt_native(
        repo_root=context.repo_root,
        clone_path=context.clone_path,
        run_root=context.run_root,
        prompt=context.prompt,
        sample=context.sample,
        mas=context.mas,
        benchmark=context.benchmark,
        adapter=context.host_adapter,
        aria=context.aria,
        workspace_path=context.workspace_path,
        max_rounds=int(
            context.options.get(
                "max_turns",
                context.options.get("max_agent_turns", 4),
            )
        ),
        timeout_seconds=int(
            context.options.get(
                "native_timeout_seconds",
                context.default_timeout_seconds * 8,
            )
        ),
        api_settings=context.api_settings,
        repository_context_mode=str(context.options.get("repository_context_mode", "legacy")),
        context_budget_mode=str(context.options.get("context_budget_mode", "legacy")),
    )


BACKEND = NativeMasBackend(
    slug="metagpt",
    supported_protocols=(OPENAI_CHAT_COMPLETIONS, ANTHROPIC_MESSAGES),
    clone_marker="metagpt",
    artifact_prefix="metagpt",
    messages_filename="metagpt_messages.json",
    trace_filename="metagpt_native_trace.jsonl",
    artifact_filenames=(
        "metagpt_messages.json",
        "metagpt_native_trace.jsonl",
        "metagpt_request.json",
        "metagpt_output.json",
        "metagpt_stdout.log",
        "metagpt_stderr.log",
    ),
    launch_description="native MetaGPT Team/Role/Action pipeline",
    notes="Native MetaGPT Team/Role/Action pipeline with a GPT-5.5-compatible model bridge.",
    run_native=_run,
)
