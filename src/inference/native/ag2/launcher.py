"""Child-process implementation for one native AG2 benchmark sample."""

from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from common.artifacts import atomic_write_json
from common.model_api import (
    ANTHROPIC_MESSAGES,
    MASFM_USER_AGENT,
    anthropic_payload_from_openai_chat,
    invoke_anthropic_messages_gateway,
    openai_chat_response_from_anthropic,
    openai_compatible_extra_body,
)
from inference.native.runtime_hooks import (
    install_sync_client_hook,
    runtime_from_bootstrap,
    runtime_profile_events,
)

REQUEST: dict[str, Any] = {}

NATIVE_CODE_BLOCK_CONTRACT = (
    "Don't use a code block if it is not intended to be executed by the Python Assistant. "
    "Don't include multiple code blocks in one response. Provide one complete executable block, "
    "not an example followed by separate output blocks. Use print for actual output and check "
    "the execution result returned by the Python Assistant before taking the next step. "
)


def build_code_executor(workspace: Path, trace: list[dict[str, Any]], *, timeout: int = 180,
                        rejection_policy: str = "abort", virtual_env_context: Any = None,
                        code_block_limit: int | None = None) -> Any:
    from autogen.coding import LocalCommandLineCodeExecutor
    from autogen.coding.base import CommandLineCodeResult

    if rejection_policy not in {"abort", "feedback"}:
        raise ValueError("command_rejection_policy must be abort or feedback")
    if code_block_limit is not None and (not isinstance(code_block_limit, int) or isinstance(code_block_limit, bool) or code_block_limit < 1):
        raise ValueError("code_block_limit must be a positive integer")

    class SafeLocalCommandLineCodeExecutor(LocalCommandLineCodeExecutor):
        def execute_code_blocks(self, code_blocks: list[Any]) -> Any:
            if code_block_limit is not None and len(code_blocks) > code_block_limit:
                trace.append({"kind": "code_block_limit_rejected", "requested_blocks": len(code_blocks),
                              "code_block_limit": code_block_limit, "executed_blocks": 0})
                return CommandLineCodeResult(exit_code=1, output=(
                    f"Execution request contains {len(code_blocks)} code blocks; the maximum is {code_block_limit}. "
                    "No code block was executed. Submit one complete executable block without copied examples or output blocks."
                ))
            blocks = _protect_plain_comment_code_blocks(code_blocks)
            if rejection_policy == "feedback":
                # Check the complete request before any block can modify the workspace.
                for index, block in enumerate(blocks):
                    try:
                        LocalCommandLineCodeExecutor.sanitize_command(block.language.lower(), block.code)
                    except ValueError as exc:
                        trace.append({"kind": "code_command_rejected", "block_index": index,
                                      "reason": str(exc), "executed_blocks": 0})
                        return CommandLineCodeResult(exit_code=1, output=(
                            f"Native execution policy rejected this request: {exc}\n"
                            "No code block in this request was executed. Revise the request."
                        ))
            return super().execute_code_blocks(blocks)

    return SafeLocalCommandLineCodeExecutor(timeout=timeout, work_dir=workspace,
                                           virtual_env_context=virtual_env_context)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m inference.native.ag2.launcher <request.json>")
    request_path = Path(sys.argv[1]).resolve()
    global REQUEST
    REQUEST = json.loads(request_path.read_text(encoding="utf-8"))
    output_path = Path(REQUEST["run_root"]) / "ag2_output.json"
    try:
        output = _run()
    except Exception as exc:  # noqa: BLE001 - persist native framework failures.
        output = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "messages": [],
            "transcript": [],
            "trace": [{"kind": "launcher_error", "error": f"{type(exc).__name__}: {exc}"}],
            "final_output": "",
        }
    atomic_write_json(output_path, output)
    if output.get("status") != "completed":
        raise SystemExit(1)


def _run() -> dict[str, Any]:
    from autogen import AssistantAgent, LLMConfig, UserProxyAgent, __version__

    settings = dict(REQUEST["api_settings"])
    _install_openrouter_empty_choices_retry(settings)
    _install_anthropic_messages_bridge(settings)
    _install_context_budget_guard(settings)
    aria = runtime_from_bootstrap(REQUEST.get("aria_bootstrap"))
    aria_profile = REQUEST.get("aria_profile")
    trace: list[dict[str, Any]] = []
    if aria is not None:
        from autogen.oai.client import OpenAIClient

        install_sync_client_hook(
            OpenAIClient,
            aria,
            aria_profile if isinstance(aria_profile, dict) else None,
            trace,
            role="AG2",
        )
    api_key_env_raw = settings.get("api_key_env")
    api_key_env = str(api_key_env_raw) if api_key_env_raw is not None else None
    api_key = os.environ.get(api_key_env) if api_key_env else None
    if not api_key and settings.get("api_key_required", True):
        raise RuntimeError(f"missing API key env var: {api_key_env}")

    config_entry: dict[str, Any] = {
        "api_type": "openai",
        "model": settings["model"],
        "api_key": api_key or "not-required",
        "base_url": _api_root(settings),
        "default_headers": {"User-Agent": MASFM_USER_AGENT},
        "max_retries": int(settings.get("retries", 0)),
        "stream": False,
    }
    if settings.get("reasoning_effort") is not None:
        config_entry["max_completion_tokens"] = int(settings["max_tokens"])
        config_entry["reasoning_effort"] = str(settings["reasoning_effort"])
    else:
        config_entry["max_tokens"] = int(settings["max_tokens"])
        if settings.get("temperature") is not None:
            config_entry["temperature"] = float(settings["temperature"])
    if settings.get("seed") is not None:
        config_entry["seed"] = int(settings["seed"])
    extra_body = openai_compatible_extra_body(
        str(settings["base_url"]),
        str(settings["model"]),
        upstream_provider=settings.get("upstream_provider"),
        allow_fallbacks=settings.get("allow_fallbacks"),
    )
    if extra_body:
        config_entry["extra_body"] = extra_body
    llm_config = LLMConfig(
        config_entry,
        timeout=int(settings["timeout_seconds"]),
        cache_seed=None,
    )

    benchmark = str(REQUEST["benchmark"])
    workspace = Path(REQUEST["workspace_path"]).resolve()
    task = _task_prompt(benchmark, str(REQUEST["prompt"]), REQUEST.get("sample") or {}, workspace)
    is_termination = _is_final_answer
    code_environment = None
    if REQUEST.get("code_execution_environment") == "per-sample":
        from inference.native.code_environment import code_process_environment, prepare_code_environment

        code_environment = prepare_code_environment(Path(REQUEST["run_root"]))
        environment = code_process_environment(os.environ, code_environment)
        os.environ.clear()
        os.environ.update(environment)
        trace.append({"kind": "code_dependency_environment", "mode": "per-sample",
                      "path": str(code_environment.env_dir), "security_sandbox": False})
    code_executor = build_code_executor(workspace, trace,
        rejection_policy=REQUEST.get("command_rejection_policy", "abort"), virtual_env_context=code_environment,
        code_block_limit=REQUEST.get("code_block_limit"))
    executor = UserProxyAgent(
        name="Python_Assistant",
        human_input_mode="NEVER",
        is_termination_msg=is_termination,
        max_consecutive_auto_reply=max(int(REQUEST["max_turns"]), 2),
        default_auto_reply=(
            "Continue by checking the latest result. Use another fenced python or shell block when "
            "needed; otherwise finish with FINAL_ANSWER: <answer>."
        ),
        code_execution_config={
            "executor": code_executor,
            "last_n_messages": 1,
        },
        llm_config=False,
    )
    student = AssistantAgent(
        name="Student",
        system_message=_student_system_message(benchmark, workspace, code_block_contract=REQUEST.get("code_block_contract", "legacy")),
        human_input_mode="NEVER",
        is_termination_msg=is_termination,
        llm_config=llm_config,
    )
    verifier = AssistantAgent(
        name="Verifier",
        system_message=_verifier_system_message(benchmark, workspace, code_block_contract=REQUEST.get("code_block_contract", "legacy")),
        human_input_mode="NEVER",
        is_termination_msg=is_termination,
        llm_config=llm_config,
    )

    student_chat = executor.initiate_chat(
        student,
        message=task,
        max_turns=int(REQUEST["max_turns"]),
        summary_method="last_msg",
        silent=True,
    )
    student_messages = _normalize_chat(
        student_chat.chat_history,
        phase="solve",
        initiator="Python Assistant",
        recipient="Student",
    )
    review_prompt = _review_prompt(benchmark, task, student_messages, workspace)
    verifier_chat = executor.initiate_chat(
        verifier,
        message=review_prompt,
        max_turns=max(2, int(REQUEST["max_turns"])),
        summary_method="last_msg",
        silent=True,
    )
    verifier_messages = _normalize_chat(
        verifier_chat.chat_history,
        phase="verify",
        initiator="Python Assistant",
        recipient="Verifier",
    )
    messages = student_messages + verifier_messages
    final_output = (
        _last_final_answer_content(verifier_messages)
        or _last_final_answer_content(student_messages)
        or str(verifier_chat.summary or _last_content(verifier_messages))
    )
    trace.extend(
        [
            {
                "kind": "ag2_runtime",
                "ag2_version": str(__version__),
                "benchmark": benchmark,
                "workspace_path": str(workspace),
                "model": settings["model"],
                "reasoning_effort": settings.get("reasoning_effort"),
                "student_max_turns": int(REQUEST["max_turns"]),
                "verifier_max_turns": max(2, int(REQUEST["max_turns"])),
            },
            {
                "kind": "chat_completed",
                "phase": "solve",
                "message_count": len(student_messages),
                "cost": _json_safe(student_chat.cost),
            },
            {
                "kind": "chat_completed",
                "phase": "verify",
                "message_count": len(verifier_messages),
                "cost": _json_safe(verifier_chat.cost),
            },
        ]
    )
    aria_export = None
    if aria is not None:
        _, _, finalize_event = runtime_profile_events(
            aria_profile if isinstance(aria_profile, dict) else None
        )
        _, finalize_patch = aria.before_finalize(
            len(messages) + 1,
            "Verifier",
            f"{finalize_event}: Finalize benchmark answer for {benchmark}.",
        )
        if finalize_patch is not None:
            trace.append({"kind": "aria_before_finalize", "record": finalize_patch})
        aria_export = aria.export_state()
    return {
        "status": "completed",
        "error": None,
        "messages": messages,
        "transcript": messages,
        "trace": trace,
        "final_output": final_output,
        "aria_export": aria_export,
    }


def _install_openrouter_empty_choices_retry(settings: dict[str, Any]) -> None:
    if settings.get("provider") != "openrouter":
        return

    from autogen.oai.client import OpenAIClient

    original_create = OpenAIClient.create
    if getattr(original_create, "_masfm_retries_empty_choices", False):
        return

    def create_with_retry(client: Any, params: dict[str, Any]) -> Any:
        return _create_with_nonempty_choices(
            lambda: original_create(client, params),
            retries=int(settings.get("retries", 0)),
            retry_seconds=float(settings.get("retry_seconds", 0)),
        )

    create_with_retry._masfm_retries_empty_choices = True  # type: ignore[attr-defined]
    OpenAIClient.create = create_with_retry


def _install_context_budget_guard(settings: dict[str, Any]) -> None:
    """Bound AG2's accumulated chat history before it reaches the host API."""

    if settings.get("protocol") != "openai-chat-completions":
        return
    from autogen.oai.client import OpenAIClient

    original_create = OpenAIClient.create
    if getattr(original_create, "_masfm_context_budget_guard", False):
        return

    context_window = int(settings.get("context_window_tokens") or 32768)
    requested = int(settings.get("max_tokens") or 1024)

    def create_with_context_guard(client: Any, params: dict[str, Any]) -> Any:
        guarded = dict(params)
        messages = guarded.get("messages")
        if isinstance(messages, list):
            target = max(256, context_window - requested - 512)
            guarded["messages"] = _compact_chat_messages(messages, target_tokens=target)
            estimated = _estimate_message_tokens(guarded["messages"])
            field = "max_completion_tokens" if "max_completion_tokens" in guarded else "max_tokens"
            safe = context_window - estimated - 256
            if safe > 0:
                guarded[field] = min(int(guarded.get(field) or requested), safe)
        return original_create(client, guarded)

    create_with_context_guard._masfm_context_budget_guard = True  # type: ignore[attr-defined]
    OpenAIClient.create = create_with_context_guard


def _estimate_message_tokens(messages: list[dict[str, Any]]) -> int:
    # Conservative estimate for Llama/Qwen-compatible tokenizers. The target
    # leaves a margin, so this intentionally overestimates rather than relying
    # on the model-specific tokenizer being importable in every MAS env.
    return max(1, (len(json.dumps(messages, ensure_ascii=False, separators=(",", ":"))) + 2) // 3)


def _compact_chat_messages(messages: list[dict[str, Any]], *, target_tokens: int) -> list[dict[str, Any]]:
    """Keep task framing and recent turns while shrinking old tool output."""

    def shorten(value: Any, limit: int) -> Any:
        if isinstance(value, str):
            if len(value) <= limit:
                return value
            head = max(1, limit // 3)
            return value[:head] + "\n...[older context compacted]...\n" + value[-(limit - head):]
        if isinstance(value, list):
            return [shorten(item, limit) for item in value]
        if isinstance(value, dict):
            return {key: shorten(item, limit) for key, item in value.items()}
        return value

    def candidate(limit: int, edge_count: int) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        last = len(messages) - 1
        for index, original in enumerate(messages):
            message = dict(original)
            edge = index < 2 or index >= max(2, last - edge_count + 1)
            content_limit = max(limit * 2, 512) if edge else limit
            if "content" in message:
                message["content"] = shorten(message["content"], content_limit)
            if isinstance(message.get("tool_calls"), list) and not edge:
                message["tool_calls"] = shorten(message["tool_calls"], limit)
            result.append(message)
        return result

    for limit in (4096, 2048, 1024, 512, 256, 128, 64):
        for edge_count in (8, 4, 2):
            result = candidate(limit, edge_count)
            if _estimate_message_tokens(result) <= target_tokens:
                return result
    return candidate(32, 1)


def _install_anthropic_messages_bridge(settings: dict[str, Any]) -> None:
    if settings.get("protocol") != ANTHROPIC_MESSAGES:
        return

    from autogen.oai.client import OpenAIClient

    original_create = OpenAIClient.create
    if getattr(original_create, "_masfm_anthropic_messages_bridge", False):
        return

    def create_with_anthropic_messages(client: Any, params: dict[str, Any]) -> Any:
        return _create_via_anthropic_messages(settings, params)

    create_with_anthropic_messages._masfm_anthropic_messages_bridge = True  # type: ignore[attr-defined]
    OpenAIClient.create = create_with_anthropic_messages


def _create_via_anthropic_messages(
    settings: dict[str, Any],
    params: dict[str, Any],
) -> Any:
    from openai.types.chat import ChatCompletion

    api_key_env = settings.get("api_key_env")
    api_key = os.environ.get(str(api_key_env)) if api_key_env else None
    if not api_key and settings.get("api_key_required", True):
        raise RuntimeError(f"missing API key env var: {api_key_env}")
    payload = anthropic_payload_from_openai_chat(
        params,
        default_max_tokens=int(settings.get("max_tokens", 8192)),
    )
    raw, _ = invoke_anthropic_messages_gateway(
        base_url=str(settings["base_url"]),
        api_key=str(api_key or ""),
        payload=payload,
        timeout_seconds=float(settings.get("timeout_seconds", 600)),
        retries=int(settings.get("retries", 0)),
        retry_seconds=float(settings.get("retry_seconds", 0)),
        anthropic_version=str(settings.get("anthropic_version", "2023-06-01")),
    )
    return ChatCompletion.model_validate(openai_chat_response_from_anthropic(raw))


def _create_with_nonempty_choices(
    create: Callable[[], Any],
    *,
    retries: int,
    retry_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    attempts = retries + 1
    for attempt in range(attempts):
        response = create()
        if getattr(response, "choices", None):
            return response
        if attempt + 1 < attempts and retry_seconds:
            sleep(retry_seconds)
    raise RuntimeError(
        "OpenAI-compatible API returned no choices "
        f"after {attempts} attempt{'s' if attempts != 1 else ''}"
    )


def _api_root(settings: dict[str, Any]) -> str:
    endpoint = str(settings.get("endpoint_url") or "").rstrip("/")
    suffix = "/chat/completions"
    if endpoint.endswith(suffix):
        return endpoint[: -len(suffix)]
    base = str(settings["base_url"]).rstrip("/")
    return base if base.endswith("/v1") else f"{base}/v1"


def _protect_plain_comment_code_blocks(code_blocks: list[Any]) -> list[Any]:
    """Prevent AG2 from treating an ordinary leading comment as a filename."""

    protected: list[Any] = []
    comment = re.compile(r"^(?:#|//|/\*|<!--)\s*")
    explicit_filename = re.compile(r"^(?:#|//|/\*|<!--)\s*filename\s*:", re.IGNORECASE)
    for block in code_blocks:
        code = str(block.code)
        first_line = code.split("\n", 1)[0].strip()
        if comment.match(first_line) and not explicit_filename.match(first_line):
            code = "\n" + code
        protected.append(type(block)(code=code, language=block.language))
    return protected


def _is_final_answer(message: dict[str, Any]) -> bool:
    content = str(message.get("content") or "")
    has_executable_block = re.search(
        r"```\s*(?:python|py|sh|bash|shell)\b",
        content,
        flags=re.IGNORECASE,
    )
    has_final_answer = re.search(r"FINAL_ANSWER\s*:", content, flags=re.IGNORECASE)
    is_placeholder = re.search(
        r"FINAL_ANSWER\s*:\s*<answer>", content, flags=re.IGNORECASE
    )
    return bool(not has_executable_block and has_final_answer and not is_placeholder)


def _task_prompt(
    benchmark: str,
    prompt: str,
    sample: dict[str, Any],
    workspace: Path,
) -> str:
    context = [f"Benchmark: {benchmark}", f"Workspace: {workspace}", "", prompt]
    file_path = sample.get("file_path") or sample.get("attachment_path")
    if file_path:
        context.extend(["", f"The task attachment is available at: {file_path}"])
    if benchmark == "swe-bench-lite":
        context.extend(
            [
                "",
                (
                    "Modify the checked-out repository in the workspace to solve the issue. "
                    "Inspect files and run focused tests through fenced shell or python blocks. "
                    "Do not only describe a patch."
                ),
            ]
        )
    return "\n".join(context)


def _student_system_message(benchmark: str, workspace: Path, *, code_block_contract: str = "legacy") -> str:
    ending = _answer_contract(benchmark, code_block_contract=code_block_contract)
    code_rules = NATIVE_CODE_BLOCK_CONTRACT if code_block_contract == "native" else ""
    return (
        "You are the Student in an AG2 problem-solving team. Solve the assigned task carefully. "
        "The Python Assistant executes fenced `python`, `sh`, or `bash` blocks in the workspace and returns "
        "their real output. Use it whenever computation, file inspection, editing, or tests are useful. "
        "Never invent command output. Never put FINAL_ANSWER in a message containing an executable code "
        "block; wait for the Python Assistant's real output, then send the final answer separately. "
        f"{code_rules}The workspace is {workspace}. {ending}"
    )


def _verifier_system_message(benchmark: str, workspace: Path, *, code_block_contract: str = "legacy") -> str:
    ending = _answer_contract(benchmark, code_block_contract=code_block_contract)
    code_rules = NATIVE_CODE_BLOCK_CONTRACT if code_block_contract == "native" else ""
    return (
        "You are the Verifier in an AG2 problem-solving team. Independently check the proposed solution and "
        "correct it when needed. You may ask the Python Assistant to execute fenced `python`, `sh`, or `bash` "
        "blocks. For repository tasks, inspect the actual workspace, make any remaining edits, and run focused "
        "tests before delivery. Never put FINAL_ANSWER in a message containing an executable code block; wait "
        f"for the Python Assistant's real output first. {code_rules}The workspace is {workspace}. {ending}"
    )


def _answer_contract(benchmark: str, *, code_block_contract: str = "legacy") -> str:
    if code_block_contract not in {"legacy", "native"}:
        raise ValueError("code_block_contract must be legacy or native")
    if benchmark == "mmlu":
        return "Finish with exactly `FINAL_ANSWER: <A|B|C|D>`."
    if benchmark == "humaneval":
        if code_block_contract == "native":
            return (
                "Use an executable fenced `python` block to write the complete implementation to solution.py. "
                "After the Python Assistant confirms execution, send a separate final message: FINAL_ANSWER: implemented."
            )
        return (
            "Finish with the complete Python implementation in a fenced `python` block, followed by "
            "`FINAL_ANSWER: implemented`."
        )
    if benchmark == "swe-bench-lite":
        return "Do not include a diff in the answer; edit the workspace and finish with `FINAL_ANSWER: patch ready`."
    if benchmark == "olympiadbench":
        return (
            "Solve the problem rigorously, but put only the requested mathematical value or expression after "
            "the marker. Finish with exactly one line `FINAL_ANSWER: \\boxed{<answer>}`."
        )
    return "Finish with exactly one final line in the form `FINAL_ANSWER: <answer>`."


def _review_prompt(
    benchmark: str,
    task: str,
    messages: list[dict[str, Any]],
    workspace: Path,
) -> str:
    history = "\n\n".join(
        f"[{item['source']}]\n{item['content']}" for item in messages
    )
    if len(history) > 48_000:
        history = history[:8_000] + "\n...[middle transcript compacted]...\n" + history[-40_000:]
    history = re.sub(
        r"FINAL_ANSWER\s*:",
        "STUDENT_PROPOSED_ANSWER:",
        history,
        flags=re.IGNORECASE,
    )
    return (
        f"Review and finalize this {benchmark} task.\n\nOriginal task:\n{task}\n\n"
        f"Student/Python Assistant transcript:\n{history}\n\n"
        f"Use the actual workspace at {workspace} when verification is needed."
    )


def _normalize_chat(
    history: list[dict[str, Any]],
    *,
    phase: str,
    initiator: str,
    recipient: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, message in enumerate(history):
        source = initiator if index % 2 == 0 else recipient
        content = message.get("content")
        if isinstance(content, (dict, list)):
            content = json.dumps(content, ensure_ascii=False)
        normalized.append(
            {
                "turn": index + 1,
                "phase": phase,
                "source": source,
                "role": source,
                "content": "" if content is None else str(content),
                "tool_calls": _json_safe(message.get("tool_calls") or []),
            }
        )
    return normalized


def _last_content(messages: list[dict[str, Any]]) -> str:
    return str(messages[-1].get("content") or "") if messages else ""


def _last_final_answer_content(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        content = str(message.get("content") or "")
        if _is_final_answer({"content": content}):
            return content
    return ""


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


if __name__ == "__main__":
    main()
