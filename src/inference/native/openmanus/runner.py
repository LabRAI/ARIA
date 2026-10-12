"""OpenManus native runner and registration.

This module imports the baseline from ``external/mas/openmanus`` without
modifying its source tree. It applies compatibility patches at runtime and
collects a native execution trace that can be consumed by ARIA.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import signal
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aria.runtime import AriaRuntime
from common.catalog import BenchmarkSpec, MasSpec
from common.host_adapter import HostAdapter
from common.model_api import (
    ANTHROPIC_MESSAGES,
    MASFM_USER_AGENT,
    OPENAI_CHAT_COMPLETIONS,
    anthropic_payload_from_openai_chat,
    api_root_url,
    context_budget_from_error,
    invoke_anthropic_messages_gateway,
    is_retryable_http_error,
    openai_chat_response_from_anthropic,
    openai_compatible_extra_body,
    retry_delay_seconds,
)
from inference.native.base import NativeMasBackend, NativeRunContext
from inference.native.openmanus.config import (
    ensure_openmanus_config,
    suppress_openmanus_pydantic_v2_warnings,
)
from inference.native.shared import (
    gaia_attachment_instruction,
    prepare_gaia_attachment,
    prepare_swe_bench_workspace,
    write_aria_artifacts,
)


@dataclass(frozen=True)
class OpenManusNativeResult:
    transcript: list[dict[str, Any]]
    final_output: str
    aria_export: dict[str, Any] | None
    native_messages: list[dict[str, Any]]
    status: str
    error: str | None = None


_OPENMANUS_RUN_CONTEXT: dict[str, Any] = {}


class ToolProtocolError(RuntimeError):
    """The endpoint emitted a tool request as plain content instead of tool_calls."""


def _recover_tool_calls(content: str, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize exact llama3_json requests that a serving parser left in content."""
    import uuid

    declared = {tool["function"]["name"]: tool["function"] for tool in tools if tool.get("type") == "function"}
    decoder = json.JSONDecoder()
    remainder = content.strip()
    if "<tool_call>" in remainder:
        payloads = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", remainder, re.DOTALL)
        if not payloads or len(payloads) != remainder.count("<tool_call>"):
            raise ToolProtocolError("incomplete tool_call envelope")
        remainder = "; ".join(payloads)
    calls = []
    while remainder:
        try:
            value, offset = decoder.raw_decode(remainder)
        except json.JSONDecodeError as exc:
            raise ToolProtocolError("tool JSON is incomplete") from exc
        if not isinstance(value, dict):
            raise ToolProtocolError("tool request must be an object")
        function = value.get("function", value)
        if not isinstance(function, dict) or function.get("name") not in declared:
            raise ToolProtocolError("tool JSON names an undeclared tool")
        name = function["name"]
        arguments = function.get("parameters", function.get("arguments"))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise ToolProtocolError("tool arguments are not JSON") from exc
        if not isinstance(arguments, dict):
            raise ToolProtocolError("tool arguments must be an object")
        required = declared[name].get("parameters", {}).get("required", [])
        if not set(required) <= set(arguments):
            raise ToolProtocolError("tool JSON is missing required arguments")
        calls.append({"id": f"call_{uuid.uuid4().hex}", "type": "function",
                      "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}})
        remainder = remainder[offset:].strip()
        if remainder:
            if not remainder.startswith(";") or not remainder[1:].strip():
                raise ToolProtocolError("unexpected text after tool JSON")
            remainder = remainder[1:].strip()
    if not calls:
        raise ToolProtocolError("empty tool request")
    return calls


def _tool_choice_value(tool_choice: Any) -> str:
    value = getattr(tool_choice, "value", tool_choice)
    return str(value)


def _requires_sequential_tool_calls(model: str) -> bool:
    normalized = model.lower().replace("_", "-")
    return any(
        marker in normalized
        for marker in ("llama-3.", "llama-3-", "llama3.", "llama31")
    )


def _tool_request_fields(
    tools: Any,
    tool_choice: Any,
    *,
    model: str,
) -> dict[str, Any]:
    choice = _tool_choice_value(tool_choice)
    if choice not in {"none", "auto", "required"}:
        raise ValueError(f"Invalid tool_choice: {choice}")

    normalized_tools = list(tools or [])
    if choice == "required" and not normalized_tools:
        raise ValueError("tool_choice=required requires at least one tool")
    if choice == "none" or not normalized_tools:
        return {}

    fields: dict[str, Any] = {
        "tools": normalized_tools,
        "tool_choice": choice,
    }
    if _requires_sequential_tool_calls(model):
        fields["parallel_tool_calls"] = False
    return fields


def _contains_unstructured_tool_request(content: Any) -> bool:
    """Recognize the Llama JSON tool syntax when vLLM failed to parse it."""

    if not isinstance(content, str) or not content.strip().startswith("{"):
        return False
    decoder = json.JSONDecoder()
    remainder = content.lstrip()
    found = 0
    while remainder:
        try:
            value, offset = decoder.raw_decode(remainder)
        except json.JSONDecodeError:
            return False
        if not isinstance(value, dict):
            return False
        function = value.get("function")
        is_request = (
            isinstance(value.get("name"), str)
            and ("parameters" in value or "arguments" in value)
        ) or (
            isinstance(function, dict)
            and isinstance(function.get("name"), str)
            and ("parameters" in function or "arguments" in function)
        )
        if not is_request:
            return False
        found += 1
        remainder = remainder[offset:].lstrip()
        if not remainder:
            return found > 0
        if not remainder.startswith(";"):
            return False
        remainder = remainder[1:].lstrip()
    return False


def _successful_terminate_requested(tool_calls: list[Any]) -> bool:
    for call in tool_calls:
        function = getattr(call, "function", None)
        if str(getattr(function, "name", "")).lower() != "terminate":
            continue
        try:
            arguments = json.loads(getattr(function, "arguments", "") or "{}")
        except json.JSONDecodeError:
            continue
        if isinstance(arguments, dict) and arguments.get("status") == "success":
            return True
    return False


def _tool_result_failed(result: Any) -> bool:
    text = str(result).lower()
    return any(
        marker in text
        for marker in (
            "error:",
            "success': false",
            '"success": false',
            "execution timeout",
            "traceback (most recent call last)",
        )
    )


def run_openmanus_native(
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
    max_steps: int = 20,
    task_kind: str = "",
    api_settings: dict[str, Any] | None = None,
    context_budget_mode: str = "legacy",
) -> OpenManusNativeResult:
    if context_budget_mode not in {"legacy", "server-tokenizer"}:
        raise ValueError("context_budget_mode must be legacy or server-tokenizer")
    settings = dict(api_settings or {})
    if context_budget_mode != "legacy":
        settings["context_budget_mode"] = context_budget_mode
    return asyncio.run(
        _run_openmanus_native_async(
            repo_root=repo_root,
            clone_path=clone_path,
            run_root=run_root,
            prompt=prompt,
            sample=sample or {},
            mas=mas,
            benchmark=benchmark,
            adapter=adapter,
            aria=aria,
            workspace_path=workspace_path,
            max_steps=max_steps,
            task_kind=task_kind,
            api_settings=settings,
        )
    )


async def _run_openmanus_native_async(
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
    sample: dict[str, Any],
    max_steps: int,
    task_kind: str,
    api_settings: dict[str, Any],
) -> OpenManusNativeResult:
    _ensure_openmanus_importable(repo_root, clone_path)
    ensure_openmanus_config(clone_path)
    suppress_openmanus_pydantic_v2_warnings()
    _configure_openmanus_workspace(workspace_path)
    _configure_openmanus_llm(api_settings)
    if task_kind == "swe-bench-lite":
        prepare_swe_bench_workspace(workspace_path, sample, run_root)
    elif task_kind == "gaia":
        sample = prepare_gaia_attachment(workspace_path, sample, run_root)

    from app.agent.manus import Manus  # type: ignore
    from app.schema import Message  # type: ignore

    transcript: list[dict[str, Any]] = []
    trace_log: list[dict[str, Any]] = []
    native_messages: list[dict[str, Any]] = []

    if aria is not None:
        adapter.bootstrap(aria, mas, benchmark, prompt, ["Planner", "Executor", "Verifier"])

    benchmark_prompt = _build_benchmark_prompt(prompt, benchmark.slug or task_kind, workspace_path, sample)
    _set_openmanus_run_context(
        aria=aria,
        adapter=adapter,
        mas=mas,
        benchmark=benchmark,
        transcript=transcript,
        trace_log=trace_log,
        api_settings=api_settings,
    )
    _apply_openmanus_runtime_patches(
    )

    agent = None
    run_started = time.monotonic()
    try:
        agent = await Manus.create()
        agent.max_steps = max_steps
        agent.system_prompt = _build_system_prompt(workspace_path)
        agent.next_step_prompt = _build_next_step_prompt(benchmark.slug or task_kind, workspace_path)
        _configure_agent_tools(
            agent,
            benchmark.slug or task_kind,
            workspace_path=workspace_path,
        )
        final_output = await agent.run(benchmark_prompt)
        native_messages = [msg.to_dict() if isinstance(msg, Message) else msg for msg in agent.memory.messages]
        trace_log.append(
            {"kind": "run_completed", "latency_seconds": time.monotonic() - run_started}
        )
        _write_native_message_dump(run_root, native_messages)
        _write_native_trace_log(run_root, trace_log)
        aria_export = None
        if aria is not None:
            _, finalize_patch = aria.before_finalize(
                len(transcript) + 1,
                "Verifier",
                f"before_openmanus_verification: finalize native OpenManus run for {benchmark.paper_name}",
            )
            if finalize_patch is not None:
                trace_log.append({"kind": "finalize_patch", "record": finalize_patch})
                _write_native_trace_log(run_root, trace_log)
            aria_export = aria.export_state()
            write_aria_artifacts(run_root, aria_export)
        return OpenManusNativeResult(
            transcript=transcript,
            final_output=final_output,
            aria_export=aria_export,
            native_messages=native_messages,
            status="completed",
        )
    except Exception as exc:  # noqa: BLE001 - preserve native trajectory for failed samples.
        native_messages = [
            msg.to_dict() if isinstance(msg, Message) else msg
            for msg in (agent.memory.messages if agent is not None else [])
        ]
        trace_log.append(
            {
                "kind": "run_failed",
                "error": str(exc),
                "error_type": type(exc).__name__,
                "latency_seconds": time.monotonic() - run_started,
            }
        )
        _write_native_message_dump(run_root, native_messages)
        _write_native_trace_log(run_root, trace_log)
        aria_export = aria.export_state() if aria is not None else None
        if aria_export is not None:
            write_aria_artifacts(run_root, aria_export)
        return OpenManusNativeResult(
            transcript=transcript,
            final_output="",
            aria_export=aria_export,
            native_messages=native_messages,
            status="failed",
            error=str(exc),
        )
    finally:
        if agent is not None:
            await agent.cleanup()
        await _close_openmanus_llm_clients()


def _ensure_openmanus_importable(repo_root: Path, clone_path: Path) -> None:
    if not (clone_path / "app").is_dir():
        raise FileNotFoundError(
            f"OpenManus checkout not found at {clone_path} (missing 'app/' package). "
            "external/mas/openmanus is gitignored, so it must be cloned on this machine "
            "before running. Clone it with `git clone "
            "https://github.com/FoundationAgents/OpenManus.git "
            f"{clone_path}` or set prepare.clone_mas: true in the run config."
        )
    for path in (str(clone_path), str(repo_root / "src")):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.environ.setdefault("PYTHONPATH", f"{repo_root / 'src'}:{clone_path}")


def _configure_openmanus_workspace(workspace_path: Path) -> None:
    workspace_path.mkdir(parents=True, exist_ok=True)
    import app.config as openmanus_config  # type: ignore

    openmanus_config.WORKSPACE_ROOT = workspace_path


def _configure_openmanus_llm(api_settings: dict[str, Any]) -> None:
    if not api_settings:
        raise ValueError("native OpenManus requires explicit api settings")
    from app.config import SearchSettings, config  # type: ignore
    from app.llm import LLM  # type: ignore

    if api_settings.get("protocol") not in {
        OPENAI_CHAT_COMPLETIONS,
        ANTHROPIC_MESSAGES,
    }:
        raise ValueError(
            "native OpenManus requires protocol openai-chat-completions or "
            "anthropic-messages"
        )
    base_url = api_root_url(str(api_settings["base_url"]))
    api_key_env_raw = api_settings.get("api_key_env")
    api_key_env = str(api_key_env_raw) if api_key_env_raw is not None else None
    api_key = os.environ.get(api_key_env) if api_key_env else None
    if not api_key and bool(api_settings.get("api_key_required", True)):
        raise RuntimeError(f"missing API key env var: {api_key_env}")
    if not api_key:
        api_key = "not-required"

    for llm_settings in config.llm.values():
        llm_settings.api_type = "openai"
        llm_settings.model = str(api_settings["model"])
        llm_settings.base_url = base_url
        llm_settings.api_key = api_key
        llm_settings.max_tokens = int(api_settings.get("max_tokens", llm_settings.max_tokens))
        if api_settings.get("temperature") is not None:
            llm_settings.temperature = float(api_settings["temperature"])
    config._config.search_config = SearchSettings(
        engine="bing",
        fallback_engines=["DuckDuckGo", "Google", "Baidu"],
        retry_delay=2,
        max_retries=0,
        lang="en",
        country="us",
    )
    LLM._instances.clear()


async def _close_openmanus_llm_clients() -> None:
    from app.llm import LLM  # type: ignore

    clients = [getattr(instance, "client", None) for instance in LLM._instances.values()]
    for client in clients:
        if client is None:
            continue
        close = getattr(client, "close", None) or getattr(client, "aclose", None)
        if close is None:
            continue
        result = close()
        if inspect.isawaitable(result):
            await result
    LLM._instances.clear()


def _build_system_prompt(workspace_path: Path) -> str:
    return (
        "You are OpenManus, an all-capable AI assistant solving benchmark tasks. "
        "Use tools when they help, keep all files inside the assigned workspace, "
        f"and never write outside this directory: {workspace_path}."
    )


def _build_next_step_prompt(task_kind: str, workspace_path: Path) -> str:
    if task_kind in {"gsm-plus", "mmlu", "olympiadbench"}:
        return (
            "Solve the problem directly. Do not call tools. Respond with exactly one final line "
            "formatted as `FINAL_ANSWER: <answer>`."
        )
    if task_kind == "programdev":
        return (
            f"Build the requested program inside {workspace_path}. Create the needed files with "
            "absolute paths, run at least one verification command or Python syntax check, summarize "
            "what was created, and then call the terminate tool with status success."
        )
    if task_kind == "humaneval":
        return (
            f"Complete the Python function in {workspace_path}/solution.py, run the provided check "
            "with Python, fix any failure, and then call the terminate tool with status success."
        )
    if task_kind == "gaia":
        return (
            "Solve the GAIA question using direct URL fetches and web search when needed. "
            "For APOD pages, use the date-coded URLs such as "
            "https://apod.nasa.gov/apod/ap150801.html through ap150807.html. "
            "When the answer is determined, provide exactly one line formatted as "
            "`FINAL_ANSWER: <answer>`, then call the terminate tool with status success."
        )
    if task_kind == "swe-bench-lite":
        return (
            f"Fix the GitHub issue described in the task. All edits must stay inside {workspace_path}. "
            "Use the str_replace_editor to read and modify source files. Run the relevant tests with "
            "Python when the checkout's dependencies are already available. The host environment may "
            "not contain a historical repository's build and test dependencies; if verification is "
            "blocked specifically by missing packages, generated extensions, or incompatible build "
            "tooling, do not build a new environment or install dependencies. Inspect `git diff`, keep "
            "the patch narrowly scoped, and finish—the official SWE-Bench Docker harness will run the "
            "authoritative tests. Call terminate with status success after the patch is ready."
        )
    return "Complete the benchmark task, verify the result, and call terminate with status success."


def _configure_agent_tools(agent: Any, task_kind: str, workspace_path: Path | None = None) -> None:
    from app.schema import ToolChoice  # type: ignore
    from app.tool import ToolCollection  # type: ignore
    from app.tool.str_replace_editor import StrReplaceEditor  # type: ignore
    from app.tool.terminate import Terminate  # type: ignore

    if task_kind in {"gsm-plus", "mmlu", "olympiadbench"}:
        agent.available_tools = ToolCollection()
        agent.tool_choices = ToolChoice.NONE
        agent.special_tool_names = []
    elif task_kind in {"programdev", "humaneval"}:
        if workspace_path is None:
            raise ValueError(f"{task_kind} tools require a workspace path")
        agent.available_tools = ToolCollection(
            StrReplaceEditor(),
            _build_workspace_python_tool(workspace_path),
            Terminate(),
        )
        agent.tool_choices = ToolChoice.AUTO
        agent.special_tool_names = [Terminate().name]
    elif task_kind == "gaia":
        agent.available_tools = ToolCollection(
            _build_fetch_url_tool(),
            _build_yahoo_search_tool(),
            StrReplaceEditor(),
            _build_workspace_python_tool(workspace_path),
            Terminate(),
        )
        agent.tool_choices = ToolChoice.AUTO
        agent.special_tool_names = [Terminate().name]
    elif task_kind == "swe-bench-lite":
        if workspace_path is None:
            raise ValueError("SWE-Bench Lite tools require a workspace path")
        agent.available_tools = ToolCollection(
            _build_workspace_bash_tool(workspace_path),
            StrReplaceEditor(),
            Terminate(),
        )
        agent.tool_choices = ToolChoice.AUTO
        agent.special_tool_names = [Terminate().name]


def _build_workspace_python_tool(workspace_path: Path) -> Any:
    """Run Python without OpenManus's macOS multiprocessing startup penalty.

    The native tool uses ``multiprocessing.Process`` with a five-second timeout.
    macOS starts that child with ``spawn``, which re-imports the full OpenManus
    process and can time out before even trivial code executes.  A direct uv-
    environment subprocess preserves the same Python dependencies while making
    the timeout apply to the submitted code rather than framework startup.
    """
    from app.tool.base import BaseTool  # type: ignore

    class WorkspacePythonExecute(BaseTool):  # type: ignore[misc]
        name: str = "python_execute"
        description: str = (
            "Execute Python code in the assigned benchmark workspace using the current uv "
            "environment. Printed stdout and stderr are returned."
        )
        parameters: dict[str, Any] = {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Use print statements for visible output.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum execution time in seconds (default 60, maximum 300).",
                },
            },
            "required": ["code"],
        }

        async def execute(self, code: str, timeout: int = 60, **_: Any) -> dict[str, Any]:
            limit = max(1, min(int(timeout), 300))
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                code,
                cwd=str(workspace_path),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=limit)
            except TimeoutError:
                process.kill()
                await process.wait()
                return {
                    "observation": f"Execution timeout after {limit} seconds",
                    "success": False,
                }
            output = stdout.decode("utf-8", errors="replace")
            error = stderr.decode("utf-8", errors="replace")
            observation = output
            if error:
                observation = f"{observation}{'' if observation.endswith(chr(10)) or not observation else chr(10)}{error}"
            return {
                "observation": observation[-30_000:],
                "success": process.returncode == 0,
            }

    return WorkspacePythonExecute()


def _build_workspace_bash_tool(workspace_path: Path) -> Any:
    from app.tool.base import BaseTool, CLIResult  # type: ignore

    class WorkspaceBash(BaseTool):  # type: ignore[misc]
        name: str = "bash"
        description: str = (
            "Run shell commands from the checked-out SWE-Bench repository workspace. "
            "Use this to inspect files, run tests, and inspect git diff. Each call is an "
            "independent process, so a timed-out command does not poison later commands."
        )
        parameters: dict[str, Any] = {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to run from the checked-out repository root.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Maximum execution time in seconds (default 120, maximum 600).",
                },
            },
            "required": ["command"],
        }

        async def execute(
            self,
            command: str | None = None,
            restart: bool = False,
            timeout: int = 120,
            **kwargs: Any,
        ) -> Any:
            del restart, kwargs
            if command in {None, ""}:
                return CLIResult(error="no command provided")
            if command == "ctrl+c":
                return CLIResult(system="commands are isolated; there is no persistent process to interrupt")
            limit = max(1, min(int(timeout), 600))
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=str(workspace_path),
                env=os.environ.copy(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=limit)
            except TimeoutError:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
                return CLIResult(error=f"command timed out after {limit} seconds")
            output = stdout.decode("utf-8", errors="replace")
            error = stderr.decode("utf-8", errors="replace")
            if len(output) > 60_000:
                output = f"[earlier stdout truncated]\n{output[-60_000:]}"
            if len(error) > 30_000:
                error = f"[earlier stderr truncated]\n{error[-30_000:]}"
            if process.returncode != 0:
                detail = error or output or f"command exited with status {process.returncode}"
                return CLIResult(error=detail)
            return CLIResult(output=output, error=error or None)

    return WorkspaceBash()


def _build_benchmark_prompt(
    prompt: str,
    task_kind: str,
    workspace_path: Path,
    sample: dict[str, Any] | None = None,
) -> str:
    if task_kind == "gsm-plus":
        return (
            "Benchmark: GSM-Plus\n"
            "Return the final answer in an extractable form.\n\n"
            f"Problem:\n{prompt}\n\n"
            "Required final format: FINAL_ANSWER: <answer>\n"
            "Do not include units unless the answer itself requires them."
        )
    if task_kind == "mmlu":
        return (
            "Benchmark: MMLU\n"
            "Choose exactly one option letter from A, B, C, or D.\n\n"
            f"{prompt}\n\n"
            "Required final format: FINAL_ANSWER: <letter>\n"
            "Do not include explanation or option text in the final answer."
        )
    if task_kind == "olympiadbench":
        return (
            "Benchmark: OlympiadBench\n"
            "Solve the competition mathematics problem. Show your work step by step.\n\n"
            f"Problem:\n{prompt}\n\n"
            "Required final format: FINAL_ANSWER: <numeric or symbolic answer>\n"
            "Simplify the answer as much as possible."
        )
    if task_kind == "programdev":
        return (
            "Benchmark: ProgramDev\n"
            f"Workspace: {workspace_path}\n\n"
            f"Task:\n{prompt}\n\n"
            "Create a runnable implementation in the workspace. Use absolute paths for file tools. "
            "Run a verification or syntax check before termination."
        )
    if task_kind == "humaneval":
        return (
            "Benchmark: HumanEval\n"
            f"Workspace: {workspace_path}\n\n"
            f"{prompt}\n\n"
            "Create / update solution.py in the workspace with a complete Python implementation. "
            "Run the embedded HumanEval check with Python before termination. Keep all files in the workspace."
        )
    if task_kind == "gaia":
        attachment = gaia_attachment_instruction(sample or {})
        return (
            "Benchmark: GAIA\n"
            f"Workspace: {workspace_path}\n\n"
            f"Question:\n{prompt}\n\n"
            f"{attachment}"
            "Use fetch_url first for likely direct source pages. For the first week of August 2015, "
            "APOD URLs follow https://apod.nasa.gov/apod/ap150801.html, ap150802.html, and so on. "
            "Use web search if a direct URL is not enough. "
            "Return the answer in the required final format: FINAL_ANSWER: <answer>."
        )
    if task_kind == "swe-bench-lite":
        repo = (sample or {}).get("repo", "")
        issue_text = prompt
        return (
            "Benchmark: SWE-Bench Lite\n"
            f"Workspace: {workspace_path}\n"
            f"Repository: {repo}\n\n"
            f"Issue:\n{issue_text}\n\n"
            "Reproduce the failing behaviour first, then apply the minimal source-code fix needed "
            "to resolve the issue. Keep all file edits inside the workspace. Run the relevant tests "
            "to confirm the fix. When done, call terminate with status success."
        )
    return prompt


def _build_fetch_url_tool() -> Any:
    from app.tool.base import BaseTool, ToolResult  # type: ignore

    class FetchUrlTool(BaseTool):  # type: ignore[misc]
        name: str = "fetch_url"
        description: str = (
            "Fetch a web page by exact URL and return readable text. Use this when a reliable "
            "source URL is known or can be inferred, such as an official archive page."
        )
        parameters: dict[str, Any] = {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The http or https URL to fetch.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum number of text characters to return. Default is 4000.",
                    "default": 4000,
                },
            },
            "required": ["url"],
        }

        async def execute(self, url: str, max_chars: int = 4000) -> ToolResult:
            if not url.startswith(("https://", "http://")):
                return ToolResult(error="Only http and https URLs are supported.")
            try:
                text = await asyncio.to_thread(_fetch_url_text, url, int(max_chars))
            except Exception as exc:  # noqa: BLE001 - surface fetch failures to the agent.
                return ToolResult(error=f"fetch_url failed for {url}: {exc}")
            return ToolResult(output=text)

    return FetchUrlTool()


def _build_yahoo_search_tool() -> Any:
    from app.tool.base import BaseTool, ToolResult  # type: ignore

    class YahooWebSearchTool(BaseTool):  # type: ignore[misc]
        name: str = "web_search"
        description: str = (
            "Search the public web and return English result titles, direct URLs, and snippets. "
            "Use fetch_url on a result when more source detail is needed."
        )
        parameters: dict[str, Any] = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "num_results": {
                    "type": "integer",
                    "description": "Maximum results to return. Default is 5.",
                    "default": 5,
                },
                "fetch_content": {
                    "type": "boolean",
                    "description": "Accepted for compatibility; use fetch_url for full content.",
                    "default": False,
                },
            },
            "required": ["query"],
        }

        async def execute(
            self,
            query: str,
            num_results: int = 5,
            fetch_content: bool = False,
            **_: Any,
        ) -> ToolResult:
            del fetch_content
            try:
                output = await asyncio.to_thread(
                    _search_yahoo,
                    query,
                    max(1, min(int(num_results), 10)),
                )
            except Exception as exc:  # noqa: BLE001 - surface search failure to the agent.
                return ToolResult(error=f"web_search failed: {exc}")
            return ToolResult(output=output)

    return YahooWebSearchTool()


def _search_yahoo(query: str, num_results: int) -> str:
    url = "https://search.yahoo.com/search?p=" + urllib.parse.quote_plus(query)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "identity",
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        html = response.read(2_000_000).decode("utf-8", errors="replace")

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for result in soup.select("div.algo"):
        anchor = result.select_one("div.compTitle a")
        title_node = result.select_one("h3")
        if anchor is None or title_node is None:
            continue
        target = _decode_yahoo_result_url(str(anchor.get("href") or ""))
        if not target.startswith(("https://", "http://")):
            continue
        title = " ".join(title_node.get_text(" ", strip=True).split())
        description_node = result.select_one("div.compText p")
        description = (
            " ".join(description_node.get_text(" ", strip=True).split())
            if description_node is not None
            else ""
        )
        rows.append((title, target, description))
        if len(rows) >= num_results:
            break

    if not rows:
        raise RuntimeError("Yahoo returned no parseable web results")
    output = [f"Search results for '{query}':"]
    for index, (title, target, description) in enumerate(rows, start=1):
        output.append(f"\n{index}. {title}\n   URL: {target}")
        if description:
            output.append(f"   Description: {description}")
    return "\n".join(output)


def _decode_yahoo_result_url(value: str) -> str:
    match = re.search(r"/RU=([^/]+)/RK=", value)
    return urllib.parse.unquote(match.group(1)) if match else value


def _fetch_url_text(url: str, max_chars: int) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 MASFM/ARIA OpenManus fetch_url"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            body = response.read(1_000_000)
            content_type = response.headers.get("Content-Type", "")
            status = getattr(response, "status", "unknown")
    except urllib.error.HTTPError as exc:
        detail = exc.read(4000).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc

    normalized_content_type = content_type.lower()
    if not any(
        marker in normalized_content_type
        for marker in ("text/", "json", "xml", "javascript", "xhtml")
    ):
        return (
            f"Fetched {url} status={status} content_type={content_type}. "
            "The response is binary; use an HTML/text endpoint or a file-analysis tool instead."
        )

    text = body.decode("utf-8", errors="replace")
    if "html" in content_type.lower() or "<html" in text[:1000].lower():
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(text, "html.parser")
            for element in soup(["script", "style", "nav", "header", "footer"]):
                element.extract()
            text = soup.get_text(" ", strip=True)
        except Exception:
            text = re.sub(r"<[^>]+>", " ", text)
    text = " ".join(text.split())
    if len(text) > max_chars:
        text = text[:max_chars] + "... [truncated]"
    return f"Fetched {url} status={status} content_type={content_type}\n{text}"


def _set_openmanus_run_context(
    *,
    aria: AriaRuntime | None,
    adapter: HostAdapter,
    mas: MasSpec,
    benchmark: BenchmarkSpec,
    transcript: list[dict[str, Any]],
    trace_log: list[dict[str, Any]],
    api_settings: dict[str, Any],
) -> None:
    _OPENMANUS_RUN_CONTEXT.clear()
    _OPENMANUS_RUN_CONTEXT.update(
        {
            "aria": aria,
            "adapter": adapter,
            "mas": mas,
            "benchmark": benchmark,
            "transcript": transcript,
            "trace_log": trace_log,
            "api_settings": dict(api_settings),
            "detected_context_window_tokens": None,
        }
    )


async def _openmanus_chat_completion_with_retry(
    client: Any,
    params: dict[str, Any],
    api_settings: dict[str, Any],
) -> Any:
    """Call the OpenManus SDK client with MASFM's transient endpoint policy."""

    retries = int(api_settings.get("retries", 0))
    retry_seconds = float(api_settings.get("retry_seconds", 0))
    attempts = retries + 1
    request_client = (
        None
        if api_settings.get("protocol") == ANTHROPIC_MESSAGES
        else client.with_options(max_retries=0)
    )
    for attempt in range(attempts):
        try:
            if api_settings.get("protocol") == ANTHROPIC_MESSAGES:
                return await _openmanus_anthropic_chat_completion(
                    params,
                    api_settings,
                )
            assert request_client is not None
            return await request_client.chat.completions.create(**params)
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            retryable = (
                isinstance(status_code, int)
                and is_retryable_http_error(status_code, str(exc))
            ) or type(exc).__name__ in {"APIConnectionError", "APITimeoutError"}
            if not retryable or attempt + 1 == attempts:
                raise
            delay = retry_delay_seconds(retry_seconds, attempt + 1)
            logging.getLogger(__name__).warning(
                "OpenManus model endpoint transient failure status=%s; "
                "retrying attempt %s/%s after %.1fs",
                status_code,
                attempt + 2,
                attempts,
                delay,
            )
            trace_log = _OPENMANUS_RUN_CONTEXT.get("trace_log")
            if isinstance(trace_log, list):
                trace_log.append(
                    {
                        "kind": "endpoint_retry",
                        "status_code": status_code,
                        "attempt": attempt + 1,
                        "next_attempt": attempt + 2,
                        "delay_seconds": delay,
                        "error_type": type(exc).__name__,
                    }
                )
            if delay:
                await asyncio.sleep(delay)
    raise AssertionError("OpenManus endpoint retry loop exited without returning or raising")


async def _repair_unparsed_tool_response(client, params, api_settings, response, trace_log):
    """Retry malformed declared tool output once; never guess argument contents."""
    if not params.get("tools") or not response.choices:
        return response
    message = response.choices[0].message
    content = getattr(message, "content", None) or ""
    if getattr(message, "tool_calls", None) or "<tool_call>" not in content:
        return response
    try:
        _recover_tool_calls(content, params["tools"])
        return response
    except ToolProtocolError as exc:
        trace_log.append({"kind": "tool_protocol_retry", "raw_content": content, "error": str(exc)})
    retry_params = {**params, "messages": [*params["messages"], {
        "role": "user", "content": "Your tool request was not valid JSON and was not executed. "
        "Return a valid call to one declared tool. Escape backslashes in JSON strings correctly."
    }]}
    if api_settings.get("provider") == "vllm":
        retry_params["tool_choice"] = "required"
    retried = await _openmanus_chat_completion_with_retry(client, retry_params, api_settings)
    if not retried.choices:
        raise ToolProtocolError("tool protocol retry returned no choice")
    message = retried.choices[0].message
    if not getattr(message, "tool_calls", None):
        _recover_tool_calls(getattr(message, "content", None) or "", params["tools"])
    return retried


def _compact_openmanus_messages(
    llm: Any,
    messages: list[dict[str, Any]],
    *,
    target_tokens: int,
) -> list[dict[str, Any]]:
    """Bound old native history while preserving the current task and turn."""

    def shorten(value: Any, limit: int) -> Any:
        if isinstance(value, str):
            if len(value) <= limit:
                return value
            head = max(1, limit // 3)
            tail = max(1, limit - head)
            return value[:head] + "\n...[older context compacted]...\n" + value[-tail:]
        if isinstance(value, list):
            return [shorten(item, limit) for item in value]
        if isinstance(value, dict):
            return {key: shorten(item, limit) for key, item in value.items()}
        return value

    def candidate(limit: int, preserve_edges: int) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        last_index = len(messages) - 1
        for index, original in enumerate(messages):
            message = dict(original)
            edge = index < 2 or index >= max(2, last_index - preserve_edges + 1)
            content_limit = max(limit * 2, 256) if edge else limit
            if "content" in message:
                message["content"] = shorten(message["content"], content_limit)
            if not edge and isinstance(message.get("tool_calls"), list):
                calls = []
                for call in message["tool_calls"]:
                    copied = dict(call) if isinstance(call, dict) else call
                    if isinstance(copied, dict) and isinstance(copied.get("function"), dict):
                        function = dict(copied["function"])
                        function["arguments"] = shorten(function.get("arguments", ""), limit)
                        copied["function"] = function
                    calls.append(copied)
                message["tool_calls"] = calls
            result.append(message)
        return result

    for limit in (4096, 2048, 1024, 512, 256, 128, 64):
        for preserve_edges in (8, 4, 2):
            result = candidate(limit, preserve_edges)
            if llm.count_message_tokens(result) <= target_tokens:
                return result
    return candidate(32, 1)


async def _openmanus_anthropic_chat_completion(
    params: dict[str, Any],
    api_settings: dict[str, Any],
) -> Any:
    from openai.types.chat import ChatCompletion  # type: ignore

    api_key_env = api_settings.get("api_key_env")
    api_key = os.environ.get(str(api_key_env)) if api_key_env else None
    if not api_key and bool(api_settings.get("api_key_required", True)):
        raise RuntimeError(f"missing API key env var: {api_key_env}")
    payload = anthropic_payload_from_openai_chat(
        params,
        default_max_tokens=int(api_settings.get("max_tokens", 8192)),
    )
    raw, _ = await asyncio.to_thread(
        invoke_anthropic_messages_gateway,
        base_url=str(api_settings["base_url"]),
        api_key=str(api_key or ""),
        payload=payload,
        timeout_seconds=float(api_settings.get("timeout_seconds", 600)),
        retries=int(api_settings.get("retries", 0)),
        retry_seconds=float(api_settings.get("retry_seconds", 0)),
        anthropic_version=str(api_settings.get("anthropic_version", "2023-06-01")),
    )
    return ChatCompletion.model_validate(openai_chat_response_from_anthropic(raw))


def _apply_openmanus_runtime_patches(
) -> None:
    from app.agent.base import BaseAgent  # type: ignore
    from app.agent.toolcall import ToolCallAgent  # type: ignore
    from app.exceptions import TokenLimitExceeded  # type: ignore
    from app.llm import LLM, MULTIMODAL_MODELS, REASONING_MODELS  # type: ignore
    from app.logger import logger as openmanus_logger  # type: ignore
    from app.schema import AgentState  # type: ignore
    from openai import (  # type: ignore
        APIError,
        AuthenticationError,
        OpenAIError,
        RateLimitError,
    )

    if not getattr(LLM.ask_tool, "_aria_patched", False):
        async def ask_tool_with_usage_guard(
            self,
            messages,
            system_msgs=None,
            timeout: int = 300,
            tools=None,
            tool_choice="auto",
            temperature=None,
            **kwargs,
        ):
            try:
                supports_images = self.model in MULTIMODAL_MODELS
                if system_msgs:
                    system_msgs = self.format_messages(system_msgs, supports_images)
                    messages = system_msgs + self.format_messages(messages, supports_images)
                else:
                    messages = self.format_messages(messages, supports_images)

                if "llama" in self.model.lower():
                    from inference.native.tool_history import sequential_tool_history
                    messages, split_turns = sequential_tool_history(messages)
                    if split_turns:
                        _OPENMANUS_RUN_CONTEXT["trace_log"].append({
                            "kind": "tool_history_serialized", "multi_tool_turns": split_turns,
                            "scope": "single-tool template compatibility; call IDs, arguments and results preserved",
                        })

                request_fields = _tool_request_fields(
                    tools,
                    tool_choice,
                    model=self.model,
                )
                request_tools = request_fields.get("tools", [])

                tools_tokens = 0
                if request_tools:
                    for tool in request_tools:
                        tools_tokens += self.count_tokens(str(tool))
                api_settings = _OPENMANUS_RUN_CONTEXT.get("api_settings", {})
                context_window = (
                    _OPENMANUS_RUN_CONTEXT.get("detected_context_window_tokens")
                    or api_settings.get("context_window_tokens")
                )
                requested_completion_tokens = int(self.max_tokens)
                server_budget = api_settings.get("context_budget_mode") == "server-tokenizer"
                measured = None
                if server_budget:
                    from inference.native.token_budget import count_chat_tokens, fit_chat_history
                    limit = int(context_window or 32768) - requested_completion_tokens - 256
                    messages, measured = await asyncio.to_thread(fit_chat_history, messages,
                        lambda value: count_chat_tokens(api_settings, value, request_tools), limit)
                    _OPENMANUS_RUN_CONTEXT["trace_log"].append({"kind": "server_context_budget",
                        "target_prompt_tokens": limit, "requested_completion_tokens": requested_completion_tokens, **measured})
                elif context_window is not None:
                    target_input_tokens = max(
                        256,
                        int(context_window) - requested_completion_tokens - 128 - tools_tokens,
                    )
                    original_input_tokens = self.count_message_tokens(messages)
                    if original_input_tokens > target_input_tokens:
                        messages = _compact_openmanus_messages(
                            self,
                            messages,
                            target_tokens=target_input_tokens,
                        )
                        compacted_input_tokens = self.count_message_tokens(messages)
                        _OPENMANUS_RUN_CONTEXT["trace_log"].append(
                            {
                                "kind": "context_messages_compacted",
                                "context_window_tokens": int(context_window),
                                "original_prompt_tokens": original_input_tokens,
                                "compacted_prompt_tokens": compacted_input_tokens,
                                "target_prompt_tokens": target_input_tokens,
                                "requested_completion_tokens": requested_completion_tokens,
                            }
                        )
                if measured is not None:
                    input_tokens = measured["prompt_tokens"]
                else:
                    input_tokens = self.count_message_tokens(messages) + tools_tokens
                    input_tokens += tools_tokens

                if not self.check_token_limit(input_tokens):
                    raise TokenLimitExceeded(self.get_limit_error_message(input_tokens))

                if request_tools:
                    for tool in request_tools:
                        if not isinstance(tool, dict) or "type" not in tool:
                            raise ValueError("Each tool must be a dict with 'type' field")

                params = {
                    "model": self.model,
                    "messages": messages,
                    "timeout": timeout,
                    **kwargs,
                    **request_fields,
                }
                params["timeout"] = int(api_settings.get("timeout_seconds", timeout))
                params["extra_headers"] = {"User-Agent": MASFM_USER_AGENT}
                reasoning_effort = api_settings.get("reasoning_effort")
                completion_field = (
                    "max_completion_tokens"
                    if self.model in REASONING_MODELS or reasoning_effort is not None
                    else "max_tokens"
                )
                if context_window is not None:
                    safe_completion_tokens = int(context_window) - input_tokens - 256
                    if 0 < safe_completion_tokens < requested_completion_tokens:
                        requested_completion_tokens = safe_completion_tokens
                        _OPENMANUS_RUN_CONTEXT["trace_log"].append(
                            {
                                "kind": "context_budget_adjusted",
                                "context_window_tokens": int(context_window),
                                "estimated_prompt_tokens": input_tokens,
                                "requested_completion_tokens": int(self.max_tokens),
                                "safe_completion_tokens": requested_completion_tokens,
                                "safety_margin_tokens": 256,
                            }
                        )
                if self.model in REASONING_MODELS or reasoning_effort is not None:
                    params[completion_field] = requested_completion_tokens
                    if reasoning_effort is not None:
                        params["reasoning_effort"] = reasoning_effort
                else:
                    params[completion_field] = requested_completion_tokens
                    params["temperature"] = temperature if temperature is not None else self.temperature
                if api_settings.get("seed") is not None:
                    params["seed"] = int(api_settings["seed"])

                extra_body = openai_compatible_extra_body(
                    str(api_settings.get("base_url") or ""),
                    self.model,
                    upstream_provider=api_settings.get("upstream_provider"),
                    allow_fallbacks=api_settings.get("allow_fallbacks"),
                )
                if extra_body:
                    params["extra_body"] = extra_body

                params["stream"] = False
                started = time.monotonic()
                try:
                    response = await _openmanus_chat_completion_with_retry(
                        self.client,
                        params,
                        api_settings,
                    )
                except OpenAIError as exc:
                    if server_budget:
                        raise
                    budget = context_budget_from_error(
                        str(exc),
                        {completion_field: requested_completion_tokens},
                    )
                    if budget is None:
                        raise
                    _OPENMANUS_RUN_CONTEXT["detected_context_window_tokens"] = (
                        budget.context_window_tokens
                    )
                    params[completion_field] = budget.safe_completion_tokens
                    _OPENMANUS_RUN_CONTEXT["trace_log"].append(
                        {
                            "kind": "context_budget_retry",
                            "context_window_tokens": budget.context_window_tokens,
                            "prompt_tokens": budget.prompt_tokens,
                            "requested_completion_tokens": budget.requested_completion_tokens,
                            "safe_completion_tokens": budget.safe_completion_tokens,
                            "safety_margin_tokens": budget.safety_margin_tokens,
                        }
                    )
                    response = await _openmanus_chat_completion_with_retry(
                        self.client,
                        params,
                        api_settings,
                    )
                if not response.choices or not response.choices[0].message:
                    return None
                response = await _repair_unparsed_tool_response(
                    self.client, params, api_settings, response, _OPENMANUS_RUN_CONTEXT["trace_log"])
                response_message = response.choices[0].message
                if (
                    request_tools
                    and not getattr(response_message, "tool_calls", None)
                    and (_contains_unstructured_tool_request(getattr(response_message, "content", None))
                         or "<tool_call>" in (getattr(response_message, "content", None) or ""))
                ):
                    from openai.types.chat import ChatCompletionMessage

                    raw_content = str(response_message.content)
                    normalized_calls = _recover_tool_calls(raw_content, request_tools)
                    response_message = ChatCompletionMessage.model_validate({
                        **response_message.model_dump(), "content": None, "tool_calls": normalized_calls,
                    })
                    _OPENMANUS_RUN_CONTEXT["trace_log"].append(
                        {
                            "kind": "tool_protocol_recovered",
                            "model": self.model,
                            "tool_choice": request_fields.get("tool_choice", "none"),
                            "content": raw_content,
                            "tool_calls": normalized_calls,
                            "reason": "normalized exact declared tool JSON from response content",
                        }
                    )

                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                self.update_token_count(prompt_tokens, completion_tokens)
                completion_details = getattr(usage, "completion_tokens_details", None)
                _OPENMANUS_RUN_CONTEXT["trace_log"].append(
                    {
                        "kind": "model_call",
                        "model": self.model,
                        "reasoning_effort": reasoning_effort,
                        "seed": api_settings.get("seed"),
                        "tool_count": len(request_tools),
                        "tool_choice": request_fields.get("tool_choice", "none"),
                        "parallel_tool_calls": request_fields.get("parallel_tool_calls"),
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "reasoning_tokens": getattr(
                            completion_details, "reasoning_tokens", None
                        ),
                        "response_id": getattr(response, "id", None),
                        "finish_reason": getattr(response.choices[0], "finish_reason", None),
                        "latency_seconds": time.monotonic() - started,
                    }
                )
                from inference.native.model_hooks import intercept_native_response

                return intercept_native_response(
                    _OPENMANUS_RUN_CONTEXT.get("aria"), response_message, messages,
                    role="Planner",
                )
            except TokenLimitExceeded:
                raise
            except ValueError as exc:
                openmanus_logger.error(f"Validation error in ask_tool: {exc}")
                raise
            except OpenAIError as exc:
                openmanus_logger.error(f"OpenAI API error: {exc}")
                if isinstance(exc, AuthenticationError):
                    openmanus_logger.error("Authentication failed. Check API key.")
                elif isinstance(exc, RateLimitError):
                    openmanus_logger.error("Rate limit exceeded. Consider increasing retry attempts.")
                elif isinstance(exc, APIError):
                    openmanus_logger.error(f"API error: {exc}")
                raise
            except Exception as exc:
                openmanus_logger.error(f"Unexpected error in ask_tool: {exc}")
                raise

        LLM.ask_tool = ask_tool_with_usage_guard  # type: ignore[assignment]
        setattr(LLM.ask_tool, "_aria_patched", True)

    if not getattr(LLM.update_token_count, "_aria_patched", False):
        original_update = LLM.update_token_count

        def update_token_count_safe(self, input_tokens: int | None, completion_tokens: int | None = 0) -> None:
            safe_input = int(input_tokens or 0)
            safe_completion = int(completion_tokens or 0)
            return original_update(self, safe_input, safe_completion)

        LLM.update_token_count = update_token_count_safe  # type: ignore[assignment]
        setattr(LLM.update_token_count, "_aria_patched", True)

    if not getattr(BaseAgent.run, "_aria_patched", False):
        original_run = BaseAgent.run

        async def run_with_trace(self, request: str | None = None):
            result = await original_run(self, request)
            trace_log = _OPENMANUS_RUN_CONTEXT["trace_log"]
            trace_log.append(
                {
                    "kind": "agent_run_completed",
                    "agent": getattr(self, "name", "agent"),
                    "current_step": getattr(self, "current_step", None),
                    "result": result,
                }
            )
            return result

        BaseAgent.run = run_with_trace  # type: ignore[assignment]
        setattr(BaseAgent.run, "_aria_patched", True)

    if not getattr(ToolCallAgent.think, "_aria_patched", False):
        original_think = ToolCallAgent.think

        async def think_with_aria(self):
            aria = _OPENMANUS_RUN_CONTEXT.get("aria")
            adapter = _OPENMANUS_RUN_CONTEXT["adapter"]
            mas = _OPENMANUS_RUN_CONTEXT["mas"]
            benchmark = _OPENMANUS_RUN_CONTEXT["benchmark"]
            transcript = _OPENMANUS_RUN_CONTEXT["transcript"]
            trace_log = _OPENMANUS_RUN_CONTEXT["trace_log"]
            turn = int(getattr(self, "current_step", 0)) + 1
            prompt = getattr(self, "next_step_prompt", "") or ""
            patch_record = None
            if aria is not None:
                prompt, patch_record = aria.before_action(
                    adapter.runtime_profile(mas, benchmark).pre_event,
                    turn,
                    "Planner",
                    prompt,
                )
                self.next_step_prompt = prompt
            trace_log.append(
                {
                    "kind": "before_think",
                    "turn": turn,
                    "role": "Planner",
                    "prompt": prompt,
                    "patch": patch_record,
                }
            )
            had_action = await original_think(self)
            last_message = self.memory.messages[-1] if self.memory.messages else None
            content = ""
            if last_message is not None and getattr(last_message, "content", None):
                content = str(last_message.content)
            if benchmark.slug in {"gsm-plus", "mmlu", "gaia"} and "FINAL_ANSWER" in content:
                self.state = AgentState.FINISHED
            tool_calls = []
            if last_message is not None and getattr(last_message, "tool_calls", None):
                tool_calls = [
                    {
                        "id": call.id,
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    }
                    for call in last_message.tool_calls
                ]
            detection = None
            if aria is not None:
                detection = aria.after_action(
                    adapter.runtime_profile(mas, benchmark).post_event,
                    turn,
                    "Planner",
                    content or f"tool_calls={len(tool_calls)}",
                )
            transcript.append(
                {
                    "turn": turn,
                    "role": "Planner",
                    "content": content,
                    "tool_calls": tool_calls,
                    "had_action": had_action,
                    "detection": detection,
                }
            )
            trace_log.append(
                {
                    "kind": "after_think",
                    "turn": turn,
                    "role": "Planner",
                    "content": content,
                    "tool_calls": tool_calls,
                    "detection": detection,
                }
            )
            return had_action

        ToolCallAgent.think = think_with_aria  # type: ignore[assignment]
        setattr(ToolCallAgent.think, "_aria_patched", True)

    if not getattr(ToolCallAgent.act, "_aria_patched", False):
        original_act = ToolCallAgent.act

        async def act_with_trace(self):
            aria = _OPENMANUS_RUN_CONTEXT.get("aria")
            adapter = _OPENMANUS_RUN_CONTEXT["adapter"]
            mas = _OPENMANUS_RUN_CONTEXT["mas"]
            benchmark = _OPENMANUS_RUN_CONTEXT["benchmark"]
            transcript = _OPENMANUS_RUN_CONTEXT["transcript"]
            trace_log = _OPENMANUS_RUN_CONTEXT["trace_log"]
            turn = int(getattr(self, "current_step", 0))
            tool_calls = list(getattr(self, "tool_calls", []) or [])
            memory_start = len(getattr(self.memory, "messages", []))
            result = await original_act(self)
            tool_results = [
                str(getattr(message, "content", ""))
                for message in getattr(self.memory, "messages", [])[memory_start:]
                if str(getattr(message, "role", "")).lower().endswith("tool")
            ]
            if not tool_results:
                tool_results = [str(result)]
            failed_results = [item for item in tool_results if _tool_result_failed(item)]
            termination_withheld = (
                _successful_terminate_requested(tool_calls) and bool(failed_results)
            )
            if termination_withheld:
                self.state = AgentState.IDLE
                result = (
                    f"{result}\n\nTerminate(success) withheld because a preceding tool "
                    "in this turn failed; repair or verify the failure before terminating."
                )
            transcript.append(
                {
                    "turn": turn,
                    "role": "Executor",
                    "content": result,
                    "tool_results": tool_results,
                }
            )
            trace_log.append(
                {
                    "kind": "after_act",
                    "turn": turn,
                    "role": "Executor",
                    "content": result,
                    "tool_results": tool_results,
                    "termination_withheld": termination_withheld,
                }
            )
            if aria is not None:
                aria.after_action(
                    adapter.runtime_profile(mas, benchmark).post_event,
                    turn,
                    "Executor",
                    result,
                )
            return result

        ToolCallAgent.act = act_with_trace  # type: ignore[assignment]
        setattr(ToolCallAgent.act, "_aria_patched", True)


def _write_native_trace_log(run_root: Path, trace_log: list[dict[str, Any]]) -> None:
    with (run_root / "openmanus_native_trace.jsonl").open("w", encoding="utf-8") as handle:
        for item in trace_log:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def _write_native_message_dump(run_root: Path, messages: list[dict[str, Any]]) -> None:
    (run_root / "openmanus_messages.json").write_text(json.dumps(messages, indent=2, ensure_ascii=False) + "\n")


def _run(context: NativeRunContext) -> OpenManusNativeResult:
    return run_openmanus_native(
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
        max_steps=int(
            context.options.get(
                "max_steps",
                context.options.get("openmanus_max_steps", 20),
            )
        ),
        task_kind=context.benchmark.slug,
        api_settings=context.api_settings,
        context_budget_mode=str(context.options.get("context_budget_mode", "legacy")),
    )


BACKEND = NativeMasBackend(
    slug="openmanus",
    supported_protocols=(OPENAI_CHAT_COMPLETIONS, ANTHROPIC_MESSAGES),
    clone_marker="app",
    artifact_prefix="openmanus",
    messages_filename="openmanus_messages.json",
    trace_filename="openmanus_native_trace.jsonl",
    artifact_filenames=(
        "openmanus_messages.json",
        "openmanus_native_trace.jsonl",
    ),
    launch_description="native OpenManus run through the repo-local ARIA bridge",
    notes="Native OpenManus adapter; the pinned baseline uses OpenAI Chat Completions.",
    run_native=_run,
)
