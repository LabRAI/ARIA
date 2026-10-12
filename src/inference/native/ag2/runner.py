"""AG2 native runner and registration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aria.runtime import AriaRuntime
from common.artifacts import atomic_write_json, atomic_write_jsonl, atomic_write_text
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
from inference.native.runtime_hooks import export_runtime_bootstrap


@dataclass(frozen=True)
class AG2NativeResult:
    transcript: list[dict[str, Any]]
    final_output: str
    aria_export: dict[str, Any] | None
    native_messages: list[dict[str, Any]]
    status: str
    error: str | None = None


def run_ag2_native(
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
    max_turns: int = 4,
    timeout_seconds: int = 2400,
    api_settings: dict[str, Any] | None = None,
    command_rejection_policy: str = "abort",
    code_execution_environment: str = "shared",
    code_block_contract: str = "legacy",
    code_block_limit: int | None = None,
) -> AG2NativeResult:
    """Execute one benchmark sample with native AG2 agents."""

    if command_rejection_policy not in {"abort", "feedback"}:
        raise ValueError("command_rejection_policy must be abort or feedback")
    if code_execution_environment not in {"shared", "per-sample"}:
        raise ValueError("code_execution_environment must be shared or per-sample")
    if code_block_contract not in {"legacy", "native"}:
        raise ValueError("code_block_contract must be legacy or native")
    if code_block_limit is not None and (not isinstance(code_block_limit, int) or isinstance(code_block_limit, bool) or code_block_limit < 1):
        raise ValueError("code_block_limit must be a positive integer")
    if not (clone_path / "autogen").is_dir():
        raise FileNotFoundError(
            f"AG2 checkout not found at {clone_path} (missing 'autogen/' package)"
        )
    settings = dict(api_settings or {})
    if settings.get("protocol") not in {OPENAI_CHAT_COMPLETIONS, ANTHROPIC_MESSAGES}:
        raise ValueError(
            "native AG2 requires protocol openai-chat-completions or anthropic-messages"
        )
    if not str(settings.get("model") or "").strip():
        raise ValueError("native AG2 requires a non-empty model")
    if max_turns < 1:
        raise ValueError("native AG2 max_turns must be at least one")

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
        "max_turns": max_turns,
        "aria_bootstrap": export_runtime_bootstrap(aria),
        "aria_profile": (
            adapter.runtime_profile(mas, benchmark).to_dict() if aria is not None else None
        ),
    }
    request_path = run_root / "ag2_request.json"
    if command_rejection_policy != "abort":
        request["command_rejection_policy"] = command_rejection_policy
    if code_execution_environment != "shared":
        request["code_execution_environment"] = code_execution_environment
    if code_block_contract != "legacy":
        request["code_block_contract"] = code_block_contract
    if code_block_limit is not None:
        request["code_block_limit"] = code_block_limit
    atomic_write_json(request_path, request)
    command = [sys.executable, "-m", "inference.native.ag2.launcher", str(request_path)]
    environment = os.environ.copy()
    python_path = [str(repo_root / "src"), str(clone_path)]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    environment["AUTOGEN_USE_DOCKER"] = "0"

    trace: list[dict[str, Any]] = [
        {
            "kind": "launch",
            "command": command,
            "workspace_path": str(workspace_path),
            "max_turns": max_turns,
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
            encoding="utf-8",
            errors="backslashreplace",
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
    atomic_write_text(run_root / "ag2_stdout.log", stdout)
    atomic_write_text(run_root / "ag2_stderr.log", stderr)
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

    output_path = run_root / "ag2_output.json"
    if output_path.exists():
        try:
            output = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            output = {"status": "failed", "error": f"invalid launcher output JSON: {exc}"}
    elif returncode is None:
        output = {"status": "failed", "error": f"native AG2 exceeded {timeout_seconds} seconds"}
    else:
        output = {"status": "failed", "error": f"launcher did not write {output_path}"}

    messages = list(output.get("messages", [])) if isinstance(output.get("messages"), list) else []
    transcript = list(output.get("transcript", [])) if isinstance(output.get("transcript"), list) else []
    trace.extend(list(output.get("trace", [])) if isinstance(output.get("trace"), list) else [])
    final_output = str(output.get("final_output") or "")
    status = "completed" if returncode == 0 and output.get("status") == "completed" else "failed"
    error = None if status == "completed" else str(output.get("error") or stderr[-4000:])

    atomic_write_json(run_root / "ag2_messages.json", messages)
    aria_export = None
    child_aria_export = output.get("aria_export")
    if isinstance(child_aria_export, dict):
        # The native launcher owns the live child-side hook. Avoid replaying the
        # same transcript through the parent runtime a second time.
        aria_export = child_aria_export
        write_aria_artifacts(run_root, aria_export)
    elif aria is not None:
        profile = adapter.runtime_profile(mas, benchmark)
        for index, item in enumerate(transcript, start=1):
            content = str(item.get("content") or "")
            role = str(item.get("role") or item.get("source") or "AG2")
            aria.after_action(profile.post_event, index, role, content)
        _, finalize_patch = aria.before_finalize(
            len(transcript) + 1,
            "Verifier",
            f"before_ag2_delivery: finalize native AG2 run for {benchmark.paper_name}",
        )
        if finalize_patch is not None:
            trace.append({"kind": "finalize_patch", "record": finalize_patch})
        aria_export = aria.export_state()
        write_aria_artifacts(run_root, aria_export)

    atomic_write_jsonl(run_root / "ag2_native_trace.jsonl", trace)
    return AG2NativeResult(
        transcript=transcript,
        final_output=final_output,
        aria_export=aria_export,
        native_messages=messages,
        status=status,
        error=error,
    )


def _run(context: NativeRunContext) -> AG2NativeResult:
    return run_ag2_native(
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
        max_turns=int(
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
        command_rejection_policy=str(context.options.get("command_rejection_policy", "abort")),
        code_execution_environment=str(context.options.get("code_execution_environment", "shared")),
        code_block_contract=str(context.options.get("code_block_contract", "legacy")),
        code_block_limit=context.options.get("code_block_limit"),
    )


BACKEND = NativeMasBackend(
    slug="ag2",
    supported_protocols=(OPENAI_CHAT_COMPLETIONS, ANTHROPIC_MESSAGES),
    clone_marker="autogen",
    artifact_prefix="ag2",
    messages_filename="ag2_messages.json",
    trace_filename="ag2_native_trace.jsonl",
    artifact_filenames=(
        "ag2_messages.json",
        "ag2_native_trace.jsonl",
        "ag2_request.json",
        "ag2_output.json",
        "ag2_stdout.log",
        "ag2_stderr.log",
    ),
    launch_description="native AG2 Student/Python Assistant/Verifier pipeline",
    notes="Native AG2 Student/Python Assistant/Verifier pipeline using the ag2 uv extra.",
    run_native=_run,
)
