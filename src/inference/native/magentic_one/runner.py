"""Magentic-One native runner and registration.

The ignored AutoGen checkout remains unmodified. Each sample runs in a child
process using the active ``uv --extra magentic-one`` environment, while imports
are resolved from the pinned native checkout and outputs are normalized into
the standard MASFM artifact layout.
"""

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
from common.artifacts import atomic_write_json
from common.catalog import BenchmarkSpec, MasSpec
from common.host_adapter import HostAdapter
from common.model_api import OPENAI_CHAT_COMPLETIONS, openai_compatible_extra_body
from inference.native.base import NativeMasBackend, NativeRunContext
from inference.native.shared import (
    gaia_attachment_instruction,
    prepare_gaia_attachment,
    prepare_swe_bench_workspace,
    safe_api_settings,
    timeout_text,
    write_aria_artifacts,
)
from inference.native.runtime_hooks import export_runtime_bootstrap


@dataclass(frozen=True)
class MagenticOneNativeResult:
    transcript: list[dict[str, Any]]
    final_output: str
    aria_export: dict[str, Any] | None
    native_messages: list[dict[str, Any]]
    status: str
    error: str | None = None


def _model_supports_vision(model: str) -> bool:
    normalized = model.lower().replace("_", "-")
    if any(marker in normalized for marker in ("vision", "-vl", "vl-")):
        return True
    return not any(
        marker in normalized
        for marker in ("llama-3.", "llama-3-", "llama3.", "qwen/qwen3", "qwen3-")
    )


def _requires_sequential_tool_calls(model: str) -> bool:
    normalized = model.lower().replace("_", "-")
    return any(
        marker in normalized
        for marker in ("llama-3.", "llama-3-", "llama3.", "llama31")
    )


def run_magentic_one_native(
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
    max_turns: int = 12,
    max_stalls: int = 3,
    timeout_seconds: int = 1200,
    api_settings: dict[str, Any] | None = None,
    code_execution_environment: str = "shared",
    model_vision: bool | None = None,
) -> MagenticOneNativeResult:
    if code_execution_environment not in {"shared", "per-sample"}:
        raise ValueError("code_execution_environment must be shared or per-sample")
    if model_vision is not None and not isinstance(model_vision, bool):
        raise ValueError("model_vision must be a boolean when declared")
    settings = dict(api_settings or {})
    if settings.get("protocol") != OPENAI_CHAT_COMPLETIONS:
        raise ValueError("native Magentic-One requires protocol openai-chat-completions")
    if not str(settings.get("model") or "").strip():
        raise ValueError("native Magentic-One requires a non-empty model")

    source_roots = _magentic_one_source_roots(clone_path)
    run_root.mkdir(parents=True, exist_ok=True)
    workspace_path.mkdir(parents=True, exist_ok=True)
    sample = dict(sample or {})
    if benchmark.slug == "swe-bench-lite":
        prepare_swe_bench_workspace(workspace_path, sample, run_root)
    elif benchmark.slug == "gaia":
        sample = prepare_gaia_attachment(workspace_path, sample, run_root)

    if aria is not None:
        adapter.bootstrap(aria, mas, benchmark, prompt, list(mas.roles))

    payload = {
        "repo_root": str(repo_root),
        "clone_path": str(clone_path),
        "run_root": str(run_root),
        "workspace_path": str(workspace_path),
        "benchmark": benchmark.slug,
        "prompt": _build_magentic_one_prompt(prompt, benchmark.slug, workspace_path, sample),
        "sample": _safe_sample(sample),
        "api_settings": safe_api_settings(settings),
        "max_turns": max_turns,
        "max_stalls": max_stalls,
        "aria_bootstrap": export_runtime_bootstrap(aria),
        "aria_profile": (
            adapter.runtime_profile(mas, benchmark).to_dict() if aria is not None else None
        ),
    }
    request_path = run_root / "magentic_one_request.json"
    if code_execution_environment != "shared":
        payload["code_execution_environment"] = code_execution_environment
    if model_vision is not None:
        payload["model_vision"] = model_vision
    atomic_write_json(request_path, payload)

    env = os.environ.copy()
    env["MASFM_MAGENTIC_ONE_REQUEST"] = str(request_path)
    python_path = [str(repo_root / "src"), *(str(path) for path in source_roots)]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    api_key_env_raw = settings.get("api_key_env")
    api_key_env = str(api_key_env_raw) if api_key_env_raw is not None else None
    if api_key_env and api_key_env in os.environ:
        env["OPENAI_API_KEY"] = os.environ[api_key_env]

    command = [
        sys.executable,
        "-m",
        "inference.native.magentic_one.launcher",
        str(request_path),
    ]
    trace_log: list[dict[str, Any]] = [
        {
            "kind": "launch",
            "command": command,
            "workspace_path": str(workspace_path),
            "max_turns": max_turns,
            "max_stalls": max_stalls,
            "model": settings["model"],
            "reasoning_effort": settings.get("reasoning_effort"),
            "source_roots": [str(path) for path in source_roots],
        }
    ]

    process_started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        returncode: int | None = completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = timeout_text(exc.stdout)
        stderr = timeout_text(exc.stderr)
        returncode = None
        trace_log.append(
            {
                "kind": "process_timeout",
                "timeout_seconds": timeout_seconds,
                "latency_seconds": time.monotonic() - process_started,
                "stdout_tail": stdout[-4000:],
                "stderr_tail": stderr[-4000:],
            }
        )
    (run_root / "magentic_one_stdout.log").write_text(stdout, encoding="utf-8")
    (run_root / "magentic_one_stderr.log").write_text(stderr, encoding="utf-8")
    if returncode is not None:
        trace_log.append(
            {
                "kind": "process_completed",
                "returncode": returncode,
                "latency_seconds": time.monotonic() - process_started,
                "stdout_tail": stdout[-4000:],
                "stderr_tail": stderr[-4000:],
            }
        )

    output_path = run_root / "magentic_one_output.json"
    if output_path.exists():
        try:
            output = json.loads(output_path.read_text())
        except json.JSONDecodeError as exc:
            output = {"status": "failed", "error": f"invalid launcher output JSON: {exc}"}
    else:
        output = {
            "status": "failed",
            "error": (
                f"native Magentic-One timed out after {timeout_seconds} seconds"
                if returncode is None
                else f"launcher did not write {output_path}"
            ),
        }

    native_messages = list(output.get("messages", [])) if isinstance(output.get("messages"), list) else []
    transcript = list(output.get("transcript", [])) if isinstance(output.get("transcript"), list) else []
    final_output = str(output.get("final_output") or "")
    status = "completed" if returncode == 0 and output.get("status") == "completed" else "failed"
    error = None if status == "completed" else str(output.get("error") or stderr[-4000:])

    _write_message_dump(run_root, native_messages)
    trace_log.extend(list(output.get("trace", [])) if isinstance(output.get("trace"), list) else [])

    aria_export = None
    child_aria_export = output.get("aria_export")
    if isinstance(child_aria_export, dict):
        # The native launcher owns the live child-side hook. Do not replay the
        # transcript through the parent runtime and count every event twice.
        aria_export = child_aria_export
        write_aria_artifacts(run_root, aria_export)
    elif aria is not None:
        for item in transcript:
            if not isinstance(item, dict):
                continue
            content = str(item.get("content") or item.get("text") or "")
            role = str(item.get("source") or item.get("role") or "MagenticOne")
            turn = int(item.get("turn") or len(trace_log) + 1)
            aria.after_action(adapter.runtime_profile(mas, benchmark).post_event, turn, role, content)
        _, finalize_patch = aria.before_finalize(
            len(transcript) + 1,
            "Orchestrator",
            f"before_magentic_resolution: finalize native Magentic-One run for {benchmark.paper_name}",
        )
        if finalize_patch is not None:
            trace_log.append({"kind": "finalize_patch", "record": finalize_patch})
        aria_export = aria.export_state()
        write_aria_artifacts(run_root, aria_export)

    _write_native_trace_log(run_root, trace_log)
    return MagenticOneNativeResult(
        transcript=transcript,
        final_output=final_output,
        aria_export=aria_export,
        native_messages=native_messages,
        status=status,
        error=error,
    )


def install_context_budget(client, settings, trace, *, context_factory=None, token_counter=None,
                           completion_factory=None):
    """Bound SDK context without cutting the task, tool arguments, or newest message."""
    from urllib.parse import urlsplit, urlunsplit

    if context_factory is None:
        from autogen_core.model_context import BufferedChatCompletionContext

        context_factory = BufferedChatCompletionContext
    original_create = client.create
    window = int(settings.get("context_window_tokens") or 32768)
    output = int(settings.get("max_tokens") or 4096)

    async def count(messages, tools):
        if token_counter is not None:
            return await token_counter(messages, tools)
        if settings.get("provider") != "vllm":
            return int(client.count_tokens(messages, tools=tools) / 0.8)
        import httpx
        from autogen_ext.models.openai._openai_client import to_oai_type, convert_tools

        url = urlsplit(settings["base_url"])
        endpoint = urlunsplit((url.scheme, url.netloc, url.path.rstrip("/").removesuffix("/v1") + "/tokenize", "", ""))
        payload = {
            "model": settings["model"], "add_generation_prompt": True,
            "messages": [item for message in messages for item in to_oai_type(
                message, model=settings["model"], model_family=client.model_info["family"],
                include_name_in_message=False,
            )],
        }
        if tools:
            payload["tools"] = convert_tools(tools)
        key = os.environ.get(settings.get("api_key_env") or "") or "not-required"
        async with httpx.AsyncClient(timeout=60.0) as http:
            response = await http.post(endpoint, json=payload, headers={"Authorization": f"Bearer {key}"})
            response.raise_for_status()
            return int(response.json()["count"])

    async def invoke(messages, args, kwargs, retry_allowed):
        try:
            return await original_create(messages, *args, **kwargs)
        except Exception as exc:
            schema = kwargs.get("json_output")
            completion = getattr(exc, "completion", None)
            if (type(exc).__name__ != "LengthFinishReasonError" or completion is None
                    or not callable(getattr(schema, "model_validate_json", None))):
                raise
            choices = getattr(completion, "choices", [])
            raw = getattr(choices[0].message, "content", None) if choices else None
            usage = getattr(completion, "usage", None)
            entry = {"kind": "structured_output_length", "raw_response": raw,
                     "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                     "completion_tokens": getattr(usage, "completion_tokens", 0)}
            trace.append(entry)
            parsed = None
            if isinstance(raw, str):
                try:
                    parsed = schema.model_validate_json(raw)
                except (ValueError, TypeError):
                    pass
            if parsed is not None:
                entry["recovery"] = "complete_schema_valid_json"
                content = parsed.model_dump_json()
                if completion_factory is not None:
                    return completion_factory(content, usage)
                from autogen_core.models import CreateResult, RequestUsage

                return CreateResult(content=content, finish_reason="length", cached=False,
                                    usage=RequestUsage(prompt_tokens=entry["prompt_tokens"],
                                                       completion_tokens=entry["completion_tokens"]))
            entry["recovery"] = "unresolved"
            if (retry_allowed and int(settings.get("retries", 0)) > 0 and messages
                    and isinstance(getattr(messages[-1], "content", None), str)
                    and callable(getattr(messages[-1], "model_copy", None))):
                entry["recovery"] = "one_compact_schema_retry"
                last = messages[-1].model_copy(update={"content": messages[-1].content + (
                    "\n\nThe previous structured response exceeded the output limit. "
                    "Return a compact JSON object for the same schema. Keep each reason under "
                    "40 words and the instruction under 80 words. Do not copy the task or history."
                )})
                return await bounded_create([*messages[:-1], last], args, kwargs, False)
            raise

    async def bounded_create(messages, args, kwargs, retry_allowed):
        tools = kwargs.get("tools") or []
        requested = int((kwargs.get("extra_create_args") or {}).get("max_tokens") or output)
        limit = window - requested - 1024
        systems = [message for message in messages if getattr(message, "type", None) == "SystemMessage"]
        history = [message for message in messages if getattr(message, "type", None) != "SystemMessage"]
        before = await count(messages, tools)
        if before <= limit:
            trace.append({"kind": "context_budget", "input_tokens": before,
                          "context_window_tokens": window, "output_token_reserve": requested,
                          "count_source": "vllm_tokenizer" if settings.get("provider") == "vllm" else "estimate"})
            return await invoke(messages, args, kwargs, retry_allowed)
        for size in range(len(history), 0, -1):
            view = await context_factory(buffer_size=size, initial_messages=history).get_messages()
            if not view or view[-1] is not history[-1]:
                continue
            bounded = [*systems, *view]
            tokens = await count(bounded, tools)
            if tokens <= limit:
                trace.append({"kind": "context_compaction", "before_tokens": before,
                              "after_tokens": tokens, "context_window_tokens": window,
                              "output_token_reserve": requested, "retained_messages": len(bounded),
                              "original_messages": len(messages)})
                return await invoke(bounded, args, kwargs, retry_allowed)
        raise ValueError("native request cannot fit its system instructions and last message in the host context")

    async def create(messages, *args, **kwargs):
        return await bounded_create(messages, args, kwargs, True)

    client.create = create


def bound_orchestrator_rounds(orchestrator_type: type) -> None:
    """Bound total rounds and give each newly generated plan a fresh stall window."""
    original = orchestrator_type._orchestrate_step
    if getattr(original, "_aria_total_budget", False):
        return

    async def bounded(self, cancellation_token):
        total = getattr(self, "_aria_total_rounds", 0)
        if self._max_turns is not None and total >= self._max_turns:
            await self._prepare_final_answer("Total turn budget reached, including replanning.", cancellation_token)
            return
        self._aria_total_rounds = total + 1
        await original(self, cancellation_token=cancellation_token)

    bounded._aria_total_budget = True
    orchestrator_type._orchestrate_step = bounded
    original_reenter = getattr(orchestrator_type, "_reenter_outer_loop", None)
    if original_reenter is not None:
        async def reenter(self, cancellation_token):
            previous_stalls = self._n_stalls
            self._n_stalls = 0
            if previous_stalls and callable(getattr(self, "_log_message", None)):
                await self._log_message(f"Host reset consecutive stalls from {previous_stalls} to 0 for the new plan.")
            await original_reenter(self, cancellation_token=cancellation_token)

        orchestrator_type._reenter_outer_loop = reenter


def _build_magentic_one_prompt(
    prompt: str,
    benchmark_slug: str,
    workspace_path: Path,
    sample: dict[str, Any] | None = None,
) -> str:
    sample = sample or {}
    instructions = {
        "gsm-plus": (
            "Solve the math problem carefully. End with exactly one line: "
            "FINAL_ANSWER: <answer>."
        ),
        "mmlu": (
            "Select the single correct option. End with exactly one line containing only "
            "FINAL_ANSWER: <A, B, C, or D>."
        ),
        "humaneval": (
            f"Implement the requested Python function and save the complete runnable function to "
            f"{workspace_path / 'answer.py'}. Use the terminal to syntax-check it. End with exactly "
            "one line: FINAL_ANSWER: saved to answer.py."
        ),
        "olympiadbench": (
            "Solve the problem rigorously and simplify the result. All available problem "
            "information is included in the text; there is no attached image or file to inspect, "
            "and no web lookup is needed. Route the mathematical reasoning directly to the Coder. "
            "End with exactly one line: FINAL_ANSWER: <answer>."
        ),
        "gaia": (
            "Solve this GAIA question using the web, attached file, and tools as needed. End with "
            "exactly one line: FINAL_ANSWER: <answer>."
        ),
        "swe-bench-lite": (
            f"The repository is checked out at {workspace_path}. Inspect it, reproduce the issue "
            "when practical, and implement the fix in that checkout. The Coder does not have a "
            "terminal tool directly: whenever inspection or modification is needed, the Coder MUST "
            "respond with exactly one complete fenced `sh` or `python` code block for "
            "ComputerTerminal to execute next. Never claim that command execution is unavailable. "
            "Use executable commands to modify files, then verify that `git diff --stat` is nonempty "
            "and run focused checks when practical. Do not finish after only describing or printing "
            "a patch. End with exactly one line: FINAL_ANSWER: <brief fix summary>."
        ),
    }
    benchmark_instruction = instructions.get(
        benchmark_slug,
        "Complete the task and end with exactly one line: FINAL_ANSWER: <answer>.",
    )
    attachment = gaia_attachment_instruction(sample) if benchmark_slug == "gaia" else ""
    return f"{benchmark_instruction}\n\n{attachment}Task:\n{prompt}"


def provider_extra_body(api_settings: dict[str, Any]) -> dict[str, Any] | None:
    extra = openai_compatible_extra_body(
        str(api_settings.get("base_url") or ""),
        str(api_settings.get("model") or ""),
        upstream_provider=api_settings.get("upstream_provider"),
        allow_fallbacks=api_settings.get("allow_fallbacks"),
    )
    return extra or None


def provider_supports_structured_output(api_settings: dict[str, Any]) -> bool:
    model = str(api_settings.get("model") or "").lower()
    base_url = str(api_settings.get("base_url") or "").lower()
    if "api.example.invalid" in base_url and model == "qwen/qwen3-8b":
        return True
    # vLLM 0.11 supports response_format=json_schema for local OpenAI-
    # compatible servers.  Magentic-One's progress ledger is a nested Pydantic
    # object; JSON mode alone is too permissive for small local models and can
    # leave the host with an infrastructure-invalid ledger.
    return api_settings.get("provider") == "vllm" and (
        "127.0.0.1" in base_url or "localhost" in base_url
    )


def _safe_sample(sample: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "sample_id",
        "sample_index",
        "instance_id",
        "repo",
        "base_commit",
        "version",
        "file_name",
        "file_path",
        "attachment_path",
        "level",
        "split",
    }
    return {key: value for key, value in sample.items() if key in allowed}


def _magentic_one_source_roots(clone_path: Path) -> list[Path]:
    package_root = clone_path / "python" / "packages"
    roots = [
        package_root / "autogen-core" / "src",
        package_root / "autogen-agentchat" / "src",
        package_root / "autogen-ext" / "src",
    ]
    missing = [str(path) for path in roots if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Magentic-One checkout at {clone_path} is incomplete; missing source roots: {missing}"
        )
    return roots


def _write_message_dump(run_root: Path, messages: list[dict[str, Any]]) -> None:
    (run_root / "magentic_one_messages.json").write_text(json.dumps(messages, indent=2, ensure_ascii=False) + "\n")


def _write_native_trace_log(run_root: Path, trace_log: list[dict[str, Any]]) -> None:
    with (run_root / "magentic_one_native_trace.jsonl").open("w", encoding="utf-8") as handle:
        for item in trace_log:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def is_final_answer_connection_error(error: BaseException | str) -> bool:
    text = str(error).lower()
    return "_prepare_final_answer" in text and any(
        marker in text
        for marker in (
            "apiconnectionerror",
            "connection error",
            "remoteprotocolerror",
            "server disconnected",
        )
    )


def explicit_final_answer_message(messages: list[dict[str, Any]]) -> str:
    for item in reversed(messages):
        text = str(item.get("text") or item.get("content") or "")
        if "final_answer:" in text.lower():
            return text
    return ""


def compact_finalization_prompt(
    task: str,
    messages: list[dict[str, Any]],
    *,
    max_chars: int = 12_000,
) -> str:
    excerpts: list[str] = []
    for item in reversed(messages):
        source = str(item.get("source") or item.get("role") or "agent")
        text = str(item.get("text") or item.get("content") or "").strip()
        if not text or source == "user":
            continue
        excerpts.append(f"{source}:\n{text}")
        if sum(len(excerpt) for excerpt in excerpts) >= max_chars:
            break
    work = "\n\n".join(reversed(excerpts))[-max_chars:]
    return (
        "The native Magentic-One final-answer request failed after the agent rounds "
        "completed. Produce the answer from the compact work record below. Return only "
        "one line in the exact format FINAL_ANSWER: <answer>.\n\n"
        f"Task:\n{task}\n\nAgent work:\n{work}"
    )


def _run(context: NativeRunContext) -> MagenticOneNativeResult:
    return run_magentic_one_native(
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
                context.options.get("magentic_one_max_turns", 12),
            )
        ),
        max_stalls=int(context.options.get("max_stalls", 3)),
        timeout_seconds=int(
            context.options.get(
                "native_timeout_seconds",
                context.default_timeout_seconds * 8,
            )
        ),
        api_settings=context.api_settings,
        code_execution_environment=str(context.options.get("code_execution_environment", "shared")),
        model_vision=context.options.get("model_vision"),
    )


BACKEND = NativeMasBackend(
    slug="magentic-one",
    supported_protocols=(OPENAI_CHAT_COMPLETIONS,),
    clone_marker="python",
    artifact_prefix="magentic_one",
    messages_filename="magentic_one_messages.json",
    trace_filename="magentic_one_native_trace.jsonl",
    artifact_filenames=(
        "magentic_one_messages.json",
        "magentic_one_native_trace.jsonl",
        "magentic_one_request.json",
        "magentic_one_output.json",
        "magentic_one_stdout.log",
        "magentic_one_stderr.log",
        "magentic_one_trace_debug.log",
    ),
    launch_description="native Magentic-One through the active magentic-one uv environment",
    notes="Native AutoGen/Magentic-One adapter using the magentic-one uv extra.",
    run_native=_run,
)
