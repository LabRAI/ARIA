"""Child-process implementation for one native Magentic-One sample."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import traceback
from hashlib import sha256
from pathlib import Path
from typing import Any

from autogen_agentchat.agents import CodeExecutorAgent
from autogen_agentchat.base import TaskResult
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.teams._group_chat._magentic_one._magentic_one_orchestrator import MagenticOneOrchestrator
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_core.models import ModelFamily, UserMessage
from autogen_ext.agents.file_surfer import FileSurfer
from autogen_ext.agents.magentic_one import MagenticOneCoderAgent
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from autogen_ext.code_executors._common import (
    PYTHON_VARIANTS,
    CommandLineCodeResult,
    get_file_name_from_content,
    lang_to_cmd,
    silence_pip,
)
from autogen_ext.code_executors.local import LocalCommandLineCodeExecutor
from autogen_ext.models.openai import OpenAIChatCompletionClient

from inference.native.magentic_one.runner import (
    bound_orchestrator_rounds,
    install_context_budget,
    _model_supports_vision,
    _requires_sequential_tool_calls,
    compact_finalization_prompt,
    explicit_final_answer_message,
    is_final_answer_connection_error,
    provider_extra_body,
    provider_supports_structured_output,
)
from inference.native.runtime_hooks import (
    install_async_client_hook,
    runtime_from_bootstrap,
    runtime_profile_events,
)
from inference.native.code_environment import code_process_environment, prepare_code_environment

LOGGER = logging.getLogger(__name__)


def decode_command_output(value: bytes) -> str:
    """Decode captured command output without failing on legacy bytes."""

    return value.decode("utf-8", errors="replace")


class MagenticOneLocalCommandLineCodeExecutor(LocalCommandLineCodeExecutor):
    """AutoGen 0.7.5 local executor with backend-safe output decoding.

    AutoGen 0.7.5 decodes command stdout and stderr as strict UTF-8. SWE-Bench
    repositories can contain tests or tools that emit legacy-encoded bytes,
    which must not terminate the entire native Magentic-One sample. This keeps
    the pinned executor behavior intact except for tolerant output decoding.
    """

    async def _execute_code_dont_check_setup(
        self,
        code_blocks: list[CodeBlock],
        cancellation_token: CancellationToken,
    ) -> CommandLineCodeResult:
        logs_all = ""
        file_names: list[Path] = []
        exitcode = 0

        for code_block in code_blocks:
            lang, code = code_block.language.lower(), code_block.code
            code = silence_pip(code, lang)
            if lang in PYTHON_VARIANTS:
                lang = "python"

            if lang not in self.SUPPORTED_LANGUAGES:
                exitcode = 1
                logs_all += f"\nunknown language {lang}"
                break

            try:
                filename = get_file_name_from_content(code, self.work_dir)
            except ValueError:
                return CommandLineCodeResult(
                    exit_code=1,
                    output="Filename is not in the workspace",
                    code_file=None,
                )

            if filename is None:
                code_hash = sha256(code.encode()).hexdigest()
                if lang.startswith("python"):
                    extension = "py"
                elif lang in {"pwsh", "powershell", "ps1"}:
                    extension = "ps1"
                else:
                    extension = lang
                filename = f"tmp_code_{code_hash}.{extension}"

            written_file = (self.work_dir / filename).resolve()
            with written_file.open("w", encoding="utf-8") as handle:
                handle.write(code)
            file_names.append(written_file)

            env = os.environ.copy()
            if self._virtual_env_context:
                env = code_process_environment(env, self._virtual_env_context)

            if lang == "python":
                program = (
                    os.path.abspath(self._virtual_env_context.env_exe)
                    if self._virtual_env_context
                    else sys.executable
                )
                extra_args = [str(written_file.absolute())]
            else:
                program = lang_to_cmd(lang)
                if program == "pwsh":
                    extra_args = [
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(written_file.absolute()),
                    ]
                else:
                    extra_args = [str(written_file.absolute())]

            task = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    program,
                    *extra_args,
                    cwd=self.work_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env,
                )
            )
            cancellation_token.link_future(task)

            process = None
            try:
                process = await task
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), self._timeout
                )
                exitcode = process.returncode or 0
            except TimeoutError:
                logs_all += "\nTimeout"
                exitcode = 124
                if process:
                    process.terminate()
                    await process.wait()
                break
            except asyncio.CancelledError:
                logs_all += "\nCancelled"
                exitcode = 125
                if process:
                    process.terminate()
                    await process.wait()
                break

            logs_all += decode_command_output(stderr)
            logs_all += decode_command_output(stdout)
            if exitcode != 0:
                break

        code_file = str(file_names[0]) if file_names else None
        result = CommandLineCodeResult(
            exit_code=exitcode,
            output=logs_all,
            code_file=code_file,
        )

        if self._cleanup_temp_files:
            for file_path in file_names:
                try:
                    file_path.unlink(missing_ok=True)
                except OSError as error:
                    LOGGER.error(
                        "Failed to delete temporary file %s: %s",
                        file_path,
                        error,
                    )

        return result


def main() -> None:
    request_path = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ["MASFM_MAGENTIC_ONE_REQUEST"]).resolve()
    request = json.loads(request_path.read_text())
    output_path = Path(request["run_root"]) / "magentic_one_output.json"
    _configure_trace_log(output_path.parent / "magentic_one_trace_debug.log")
    try:
        payload = asyncio.run(run_native_magentic_one(request))
    except Exception as exc:  # noqa: BLE001 - return a structured failure to the parent runner.
        payload = {
            "status": "failed",
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
            "messages": [],
            "transcript": [],
            "trace": [],
            "final_output": "",
        }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    if payload.get("status") != "completed":
        raise SystemExit(1)


def _configure_trace_log(path: Path) -> None:
    from autogen_agentchat import TRACE_LOGGER_NAME

    logger = logging.getLogger(TRACE_LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)


async def run_native_magentic_one(request: dict[str, Any]) -> dict[str, Any]:
    workspace = Path(request["workspace_path"]).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "downloads").mkdir(parents=True, exist_ok=True)
    (workspace / "web_debug").mkdir(parents=True, exist_ok=True)
    prompt = str(request["prompt"])
    api_settings = dict(request["api_settings"])

    client = build_client(api_settings, model_vision=request.get("model_vision"))
    aria = runtime_from_bootstrap(request.get("aria_bootstrap"))
    profile = request.get("aria_profile")
    messages: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    install_context_budget(client, api_settings, trace)
    if aria is not None:
        install_async_client_hook(
            client,
            aria,
            profile if isinstance(profile, dict) else None,
            trace,
            role="Magentic-One",
        )
    web_surfer: MultimodalWebSurfer | None = None
    code_executor: MagenticOneLocalCommandLineCodeExecutor | None = None
    try:
        file_surfer = FileSurfer("FileSurfer", model_client=client, base_path=str(workspace))
        web_surfer = MultimodalWebSurfer(
            "WebSurfer",
            model_client=client,
            downloads_folder=str(workspace / "downloads"),
            debug_dir=str(workspace / "web_debug"),
            headless=True,
            start_page="https://www.bing.com/",
            to_save_screenshots=False,
        )
        coder = MagenticOneCoderAgent("Coder", model_client=client)
        code_environment = None
        if request.get("code_execution_environment") == "per-sample":
            code_environment = await asyncio.to_thread(prepare_code_environment, Path(request["run_root"]))
            trace.append({"kind": "code_dependency_environment", "mode": "per-sample",
                          "path": str(code_environment.env_dir), "python": str(code_environment.env_exe),
                          "security_sandbox": False})
        code_executor = MagenticOneLocalCommandLineCodeExecutor(
            work_dir=workspace,
            timeout=int(api_settings.get("code_timeout_seconds", 120)),
            virtual_env_context=code_environment,
        )
        terminal = CodeExecutorAgent("ComputerTerminal", code_executor=code_executor)
        bound_orchestrator_rounds(MagenticOneOrchestrator)
        team = MagenticOneGroupChat(
            [file_surfer, web_surfer, coder, terminal],
            model_client=client,
            max_turns=int(request.get("max_turns", 12)),
            max_stalls=int(request.get("max_stalls", 3)),
            final_answer_prompt=(
                "We are working on the following task:\n"
                f"{prompt}\n\n"
                "We have completed the task. Based on the information gathered, provide only "
                "the final answer in this exact format:\nFINAL_ANSWER: <answer>"
            ),
            emit_team_events=True,
        )

        final_result: dict[str, Any] | None = None
        finalization_error: str | None = None
        try:
            async for item in team.run_stream(task=prompt):
                record = serialize_item(item)
                if isinstance(item, TaskResult):
                    final_result = record
                    trace.append({"kind": "task_result", "record": record})
                else:
                    messages.append(record)
                    trace.append({"kind": "message", "record": record})
        except RuntimeError as exc:
            if not messages or not is_final_answer_connection_error(exc):
                raise
            finalization_error = str(exc)
            trace.append(
                {
                    "kind": "native_finalization_error",
                    "error_type": type(exc).__name__,
                    "error": finalization_error,
                }
            )

        completion_mode = "native"
        if finalization_error is None:
            final_output = extract_final_output(messages, final_result)
        else:
            try:
                response = await client.create(
                    [
                        UserMessage(
                            content=compact_finalization_prompt(prompt, messages),
                            source="user",
                        )
                    ],
                        extra_create_args={
                            "max_tokens": min(int(api_settings.get("max_tokens", 1024)), 1024)
                        },
                )
                if not isinstance(response.content, str) or not response.content.strip():
                    raise RuntimeError("compact finalization returned no text")
                final_output = response.content
                completion_mode = "compact_finalization"
                trace.append(
                    {
                        "kind": "compact_finalization",
                        "status": "completed",
                        "final_output": final_output,
                    }
                )
            except Exception as exc:
                final_output = explicit_final_answer_message(messages)
                if not final_output:
                    raise RuntimeError(
                        "compact finalization failed and agent messages contain no explicit final answer"
                    ) from exc
                completion_mode = "explicit_agent_answer_fallback"
                trace.append(
                    {
                        "kind": "compact_finalization",
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "fallback": completion_mode,
                    }
                )
            final_result = {
                "type": "TaskResult",
                "stop_reason": "Recovered from native final-answer connection failure.",
                "message_count": len(messages),
            }
        transcript = [
            {
                "turn": index,
                "source": item.get("source"),
                "role": item.get("source"),
                "type": item.get("type"),
                "content": item.get("text") or item.get("content"),
            }
            for index, item in enumerate(messages, start=1)
            if item.get("text") or item.get("content")
        ]
        aria_export = None
        if aria is not None:
            _, _, finalize_event = runtime_profile_events(
                profile if isinstance(profile, dict) else None
            )
            _, finalize_patch = aria.before_finalize(
                len(transcript) + 1,
                "Orchestrator",
                f"{finalize_event}: Finalize benchmark answer for {request['benchmark']}.",
            )
            if finalize_patch is not None:
                trace.append({"kind": "aria_before_finalize", "record": finalize_patch})
            aria_export = aria.export_state()
        return {
            "status": "completed",
            "model_config": {
                "model": api_settings["model"],
                "reasoning_effort": api_settings.get("reasoning_effort"),
                "seed": api_settings.get("seed"),
                "protocol": api_settings["protocol"],
            },
            "messages": messages,
            "transcript": transcript,
            "trace": trace,
            "task_result": final_result,
            "final_output": final_output,
            "completion_mode": completion_mode,
            "finalization_error": finalization_error,
            "aria_export": aria_export,
        }
    finally:
        if web_surfer is not None:
            await web_surfer.close()
        if code_executor is not None:
            await code_executor.stop()
        await client.close()


def build_client(api_settings: dict[str, Any], *, model_vision: bool | None = None) -> OpenAIChatCompletionClient:
    if model_vision is not None and not isinstance(model_vision, bool):
        raise ValueError("model_vision must be a boolean when declared")
    if api_settings.get("protocol") != "openai-chat-completions":
        raise ValueError("native Magentic-One requires protocol openai-chat-completions")
    base_url = str(api_settings["base_url"]).rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = f"{base_url}/v1"
    api_key_env_raw = api_settings.get("api_key_env")
    api_key_env = str(api_key_env_raw) if api_key_env_raw is not None else None
    api_key = os.environ.get(api_key_env) if api_key_env else None
    if not api_key and bool(api_settings.get("api_key_required", True)):
        raise RuntimeError(f"missing API key env var: {api_key_env}")
    if not api_key:
        api_key = "not-required"
    client_kwargs: dict[str, Any] = {
        "model": str(api_settings["model"]),
        "base_url": base_url,
        "api_key": api_key,
        "max_tokens": int(api_settings.get("max_tokens", 2048)),
        "timeout": float(api_settings.get("timeout_seconds", 240)),
        "max_retries": int(api_settings.get("retries", 4)),
        "model_info": {
            "vision": model_vision if model_vision is not None else _model_supports_vision(str(api_settings["model"])),
            "function_calling": True,
            "json_output": True,
            "structured_output": provider_supports_structured_output(api_settings),
            "family": _model_family(str(api_settings["model"])),
            "multiple_system_messages": True,
        },
        "include_name_in_message": False,
    }
    if _requires_sequential_tool_calls(str(api_settings["model"])):
        client_kwargs["parallel_tool_calls"] = False
    if api_settings.get("temperature") is not None:
        client_kwargs["temperature"] = float(api_settings["temperature"])
    if api_settings.get("reasoning_effort") is not None:
        client_kwargs["reasoning_effort"] = str(api_settings["reasoning_effort"])
    if api_settings.get("seed") is not None:
        client_kwargs["seed"] = int(api_settings["seed"])
    extra_body = provider_extra_body(api_settings)
    if extra_body:
        client_kwargs["extra_body"] = extra_body
    return OpenAIChatCompletionClient(
        **client_kwargs,
    )


def _model_family(model: str) -> str:
    normalized = model.lower()
    if "claude" in normalized and "sonnet" in normalized:
        return ModelFamily.CLAUDE_4_SONNET
    if "gpt-5" in normalized:
        return ModelFamily.GPT_5
    return ModelFamily.UNKNOWN


def serialize_item(item: Any) -> dict[str, Any]:
    if isinstance(item, TaskResult):
        return {
            "type": "TaskResult",
            "stop_reason": item.stop_reason,
            "message_count": len(item.messages),
        }
    if hasattr(item, "model_dump"):
        payload = item.model_dump(mode="json")
        if isinstance(payload, dict):
            payload["text"] = item.to_text() if hasattr(item, "to_text") else payload.get("content")
            return payload
    return {"type": type(item).__name__, "text": str(item)}


def extract_final_output(messages: list[dict[str, Any]], final_result: dict[str, Any] | None) -> str:
    for item in reversed(messages):
        source = str(item.get("source", ""))
        text = str(item.get("text") or item.get("content") or "")
        if source == "MagenticOneOrchestrator" and text:
            return text
    for item in reversed(messages):
        text = str(item.get("text") or item.get("content") or "")
        if text:
            return text
    if final_result is not None:
        return json.dumps(final_result, ensure_ascii=False)
    return ""


if __name__ == "__main__":
    main()
