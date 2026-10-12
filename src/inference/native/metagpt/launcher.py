"""Child-process implementation for one native MetaGPT benchmark sample."""

from __future__ import annotations

import asyncio
import ast
import json
import os
import re
import subprocess
import sys
import traceback
import types
from collections import Counter
from pathlib import Path
from typing import Any

from common.failure_classification import classify_infrastructure_failure
from common.model_api import (
    ANTHROPIC_MESSAGES,
    OPENAI_CHAT_COMPLETIONS,
    ApiConfig,
    ModelApiClient,
)

REQUEST: dict[str, Any] = {}
MODEL_CALLS: list[dict[str, Any]] = []
STAGE_PROMPT_PARTS: dict[str, Any] = {}
WORKSPACE_CONTEXT_PARTS: dict[str, Any] = {}


def main() -> None:
    request_path = Path(sys.argv[1]).resolve()
    global REQUEST
    REQUEST = json.loads(request_path.read_text(encoding="utf-8"))
    output_path = Path(REQUEST["run_root"]) / "metagpt_output.json"
    try:
        payload = asyncio.run(run_native_metagpt())
    except Exception as exc:  # noqa: BLE001 - child must return structured evidence.
        payload = {
            "status": "failed",
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
            "messages": [],
            "transcript": [],
            "trace": [
                *[
                    {
                        "kind": "model_call_failed",
                        "role": item.get("role"),
                        "error": item.get("error"),
                        "failure_category": item.get("failure_category"),
                    }
                    for item in MODEL_CALLS
                    if item.get("error")
                ],
                {"kind": "exception", "traceback": traceback.format_exc()},
            ],
            "model_calls": MODEL_CALLS,
            "final_output": "",
        }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if payload.get("status") != "completed":
        raise SystemExit(1)


async def run_native_metagpt() -> dict[str, Any]:
    clone_path = Path(REQUEST["clone_path"]).resolve()
    if str(clone_path) not in sys.path:
        sys.path.insert(0, str(clone_path))
    _install_minimal_provider_package(clone_path)

    # Imports intentionally live inside the child process. The parent runner and
    # its tests stay independent from MetaGPT's large optional dependency set.
    from metagpt.actions import Action, UserRequirement
    from metagpt.config2 import Config
    from metagpt.configs.llm_config import LLMConfig
    from metagpt.context import Context
    from metagpt.provider.base_llm import BaseLLM
    from metagpt.roles import Role
    from metagpt.team import Team

    api = _api_config(REQUEST["api_settings"])
    api_key = os.environ.get(api.api_key_env) if api.api_key_env else None
    llm_config = LLMConfig(
        api_type="openai",
        api_key=api_key or "not-required",
        base_url=_metagpt_config_base_url(api),
        model=api.model,
        max_token=api.max_tokens,
        temperature=0.0,
        stream=False,
        timeout=api.timeout_seconds,
        calc_usage=False,
    )
    config = Config(llm=llm_config, project_path=str(Path(REQUEST["workspace_path"]).resolve()))
    context = Context(config=config)
    from inference.native.runtime_hooks import runtime_from_bootstrap
    from inference.native.model_hooks import intercept_native_text

    aria = runtime_from_bootstrap(REQUEST.get("aria_bootstrap"))

    class MASFMModelBridge(BaseLLM):
        """MetaGPT BaseLLM backed by MASFM's strict model protocol client."""

        def __init__(self, config: LLMConfig, *, role_label: str):
            self.config = config
            self.model = api.model
            self.pricing_plan = api.model
            self.role_label = role_label
            self.client = ModelApiClient(api)
            self.cost_manager = None

        async def _invoke(self, messages: list[dict[str, Any]]) -> str:
            budget_audit = None
            try:
                if REQUEST.get("context_budget_mode") == "server-tokenizer":
                    from inference.native.token_budget import count_chat_tokens, fit_stage_messages
                    settings = REQUEST["api_settings"]
                    limit = int(settings.get("context_window_tokens") or 32768) - int(settings["max_tokens"]) - 256
                    messages, budget_audit = await asyncio.to_thread(fit_stage_messages, messages, STAGE_PROMPT_PARTS,
                        lambda value: count_chat_tokens(settings, value), limit)
                response = await asyncio.to_thread(self.client.invoke, messages)
            except Exception as exc:
                MODEL_CALLS.append(
                    {
                        "role": self.role_label,
                        "model": api.model,
                        "reasoning_effort": api.reasoning_effort,
                        "messages": messages,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "failure_category": classify_infrastructure_failure(exc),
                    }
                )
                raise
            MODEL_CALLS.append(
                {
                    "role": self.role_label,
                    "model": api.model,
                    "reasoning_effort": api.reasoning_effort,
                    "messages": messages,
                    "response": response.text,
                    "metadata": response.to_record(),
                    **({"input_budget": budget_audit} if budget_audit is not None else {}),
                }
            )
            delivered = intercept_native_text(aria, response.text, messages, role=self.role_label)
            MODEL_CALLS[-1]["delivered_response"] = delivered
            return delivered

        async def acompletion_text(self, messages, stream=False, timeout=None):
            del stream, timeout
            return await self._invoke(messages)

        async def _achat_completion(self, messages, timeout=None):
            del timeout
            return await self._invoke(messages)

        async def acompletion(self, messages, timeout=None):
            del timeout
            return await self._invoke(messages)

        async def _achat_completion_stream(self, messages, timeout=None):
            del timeout
            return await self._invoke(messages)

    def llm_for(role: str) -> MASFMModelBridge:
        return MASFMModelBridge(llm_config, role_label=role)

    class AnalyzeBenchmark(Action):
        async def run(self, history):
            return await self._aask(_stage_prompt("product", history))

    class DesignSolution(Action):
        async def run(self, history):
            return await self._aask(_stage_prompt("architect", history))

    class ImplementSolution(Action):
        async def run(self, history):
            return await self._aask(_stage_prompt("engineer", history))

    class VerifySolution(Action):
        async def run(self, history):
            return await self._aask(_stage_prompt("qa", history))

    class ProductManagerRole(Role):
        name: str = "Alice"
        profile: str = "Product Manager"
        goal: str = "Analyze the benchmark requirement and define an exact success contract"
        constraints: str = "Do not use hidden reference answers or weaken the requested output format"

        def __init__(self):
            super().__init__(context=context, llm=llm_for("Product Manager"))
            self.set_actions([AnalyzeBenchmark])
            self._watch([UserRequirement])

    class ArchitectRole(Role):
        name: str = "Bob"
        profile: str = "Architect"
        goal: str = "Design a concrete, verifiable solution from the requirement analysis"
        constraints: str = "Use available evidence and explicitly address edge cases"

        def __init__(self):
            super().__init__(context=context, llm=llm_for("Architect"))
            self.set_actions([DesignSolution])
            self._watch([AnalyzeBenchmark])

    class EngineerRole(Role):
        name: str = "Alex"
        profile: str = "Engineer"
        goal: str = "Produce the complete benchmark answer or repository patch"
        constraints: str = "Follow the exact benchmark output contract and keep code changes minimal"

        def __init__(self):
            super().__init__(context=context, llm=llm_for("Engineer"))
            self.set_actions([ImplementSolution])
            self._watch([DesignSolution])

    class QAEngineerRole(Role):
        name: str = "Edward"
        profile: str = "QA Engineer"
        goal: str = "Independently verify the proposed solution and emit the final deliverable"
        constraints: str = "Correct mistakes before returning the exact required final format"

        def __init__(self):
            super().__init__(context=context, llm=llm_for("QA Engineer"))
            self.set_actions([VerifySolution])
            self._watch([ImplementSolution])

    team = Team(context=context, use_mgx=False)
    team.hire([ProductManagerRole(), ArchitectRole(), EngineerRole(), QAEngineerRole()])
    team.invest(investment=1000.0)
    memory = await team.run(
        n_round=max(4, int(REQUEST.get("max_rounds", 4))),
        idea=_requirement_prompt(),
        auto_archive=False,
    )
    messages = _serialize_history(memory)
    transcript = [
        {
            "turn": index,
            "role": item.get("role"),
            "source": item.get("sent_from"),
            "cause_by": item.get("cause_by"),
            "content": item.get("content"),
        }
        for index, item in enumerate(messages, start=1)
        if item.get("role") != "user"
    ]
    if len(MODEL_CALLS) != 4 or len(transcript) != 4:
        failed_call = next((item for item in reversed(MODEL_CALLS) if item.get("error")), None)
        cause = (
            f"; root_cause={failed_call['error']}"
            if failed_call is not None
            else ""
        )
        raise RuntimeError(
            f"MetaGPT four-stage pipeline incomplete: model_calls={len(MODEL_CALLS)} "
            f"transcript_messages={len(transcript)}{cause}"
        )
    final_output = str(transcript[-1]["content"])
    artifact_trace = _materialize_benchmark_artifact(final_output, transcript)
    return {
        "status": "completed",
        "messages": messages,
        "transcript": transcript,
        "trace": [
            {
                "kind": "team_completed",
                "roles": ["Product Manager", "Architect", "Engineer", "QA Engineer"],
                "model_calls": len(MODEL_CALLS),
                "model": api.model,
                "reasoning_effort": api.reasoning_effort,
            },
            *artifact_trace,
        ],
        "model_calls": MODEL_CALLS,
        "final_output": final_output,
        "aria_export": aria.export_state() if aria is not None else None,
    }


def _api_config(settings: dict[str, Any]) -> ApiConfig:
    return ApiConfig(
        provider=str(settings["provider"]),
        protocol=str(settings["protocol"]),
        base_url=str(settings["base_url"]),
        model=str(settings["model"]),
        api_key_env=(
            str(settings["api_key_env"])
            if settings.get("api_key_env") is not None
            else None
        ),
        api_key_required=bool(settings.get("api_key_required", True)),
        timeout_seconds=int(settings.get("timeout_seconds", 600)),
        max_tokens=int(settings.get("max_tokens", 8192)),
        context_window_tokens=(
            int(settings["context_window_tokens"])
            if settings.get("context_window_tokens") is not None
            else None
        ),
        temperature=(
            float(settings["temperature"])
            if settings.get("temperature") is not None
            else None
        ),
        reasoning_effort=(
            str(settings["reasoning_effort"])
            if settings.get("reasoning_effort") is not None
            else None
        ),
        retries=int(settings.get("retries", 4)),
        retry_seconds=float(settings.get("retry_seconds", 10)),
        runtime_id=settings.get("runtime_id"),
        model_id=settings.get("model_id"),
        seed=int(settings["seed"]) if settings.get("seed") is not None else None,
    )


def _metagpt_config_base_url(api: ApiConfig) -> str:
    """Return a harmless bootstrap URL for MetaGPT's otherwise unused LLM config."""

    if api.protocol == OPENAI_CHAT_COMPLETIONS:
        return api.api_root_url
    if api.protocol == ANTHROPIC_MESSAGES:
        return api.base_url.rstrip("/")
    raise ValueError(f"unsupported MetaGPT model protocol: {api.protocol}")


def _install_minimal_provider_package(clone_path: Path) -> None:
    """Avoid importing every optional cloud provider from MetaGPT.__init__.

    MetaGPT's provider package eagerly imports Qianfan, DashScope, Gemini,
    Bedrock, and other SDKs even when a run supplies its own BaseLLM.  Native
    benchmark runs only need BaseLLM plus HumanProvider, so expose that exact
    package surface without altering the ignored third-party checkout.
    """

    package_name = "metagpt.provider"
    if package_name in sys.modules:
        return
    if str(clone_path) not in sys.path:
        sys.path.insert(0, str(clone_path))
    package = types.ModuleType(package_name)
    package.__path__ = [str(clone_path / "metagpt" / "provider")]
    package.__package__ = package_name
    sys.modules[package_name] = package
    from metagpt.provider.human_provider import HumanProvider

    package.HumanProvider = HumanProvider


def _requirement_prompt() -> str:
    benchmark = str(REQUEST["benchmark"])
    task = str(REQUEST["prompt"])
    workspace = Path(REQUEST["workspace_path"]).resolve()
    sample = dict(REQUEST.get("sample") or {})
    attachment = sample.get("file_path") or sample.get("attachment_path")
    common = f"Benchmark: {benchmark}\nTask:\n{task}\n"
    if benchmark == "gsm-plus":
        return common + "The final deliverable must end with exactly `FINAL_ANSWER: <answer>`."
    if benchmark == "mmlu":
        return common + "Choose A, B, C, or D. The final deliverable must be `FINAL_ANSWER: <letter>`."
    if benchmark == "humaneval":
        return common + "Return a complete Python implementation in one fenced `python` code block."
    if benchmark == "olympiadbench":
        return common + "Solve rigorously, then end with exactly `FINAL_ANSWER: <answer>`."
    if benchmark == "gaia":
        attachment_note = f"\nAvailable attachment: {attachment}" if attachment else ""
        return common + attachment_note + "\nEnd with exactly `FINAL_ANSWER: <answer>`."
    if benchmark == "swe-bench-lite":
        return (
            common
            + f"Repository workspace: {workspace}\n"
            + "Return only a git-applicable unified diff against the current checkout."
        )
    return common


def _stage_prompt(stage: str, history: list[Any]) -> str:
    benchmark = str(REQUEST["benchmark"])
    history_text = _history_text(history)
    context = _workspace_context() if benchmark == "swe-bench-lite" else ""
    instructions = {
        "product": (
            "As Product Manager, translate the user task into a precise acceptance contract. "
            "Identify the expected answer shape, constraints, ambiguities, and the checks QA must perform. "
            "Do not merely restate the task."
        ),
        "architect": (
            "As Architect, devise the concrete solution strategy. Work through the problem, identify "
            "the relevant evidence or code location, and hand the Engineer an implementable plan."
        ),
        "engineer": _engineer_instruction(benchmark),
        "qa": _qa_instruction(benchmark),
    }
    mandatory = (
        f"{instructions[stage]}\n\n"
        f"Original requirement:\n{_requirement_prompt()}\n\n"
        "Team history:\n"
    )
    prompt = mandatory + history_text + "\n" + context
    if REQUEST.get("context_budget_mode") == "server-tokenizer":
        STAGE_PROMPT_PARTS[prompt] = (mandatory, history_text, WORKSPACE_CONTEXT_PARTS.get(context, [context] if context else []))
    return prompt


def _engineer_instruction(benchmark: str) -> str:
    if benchmark == "swe-bench-lite":
        return (
            "As Engineer, implement the smallest correct fix using the repository context below. "
            "Return only a complete unified diff beginning with `diff --git`; do not return prose."
        )
    if benchmark == "humaneval":
        return (
            "As Engineer, implement the requested function completely. Return exactly one fenced Python "
            "code block containing the full function and any required imports, with no prose after it."
        )
    return (
        "As Engineer, solve the task carefully from the team plan. Include concise reasoning, then end "
        "with exactly one line `FINAL_ANSWER: <answer>` using the benchmark's required answer shape."
    )


def _qa_instruction(benchmark: str) -> str:
    if benchmark == "swe-bench-lite":
        return (
            "As QA Engineer, inspect the Engineer's diff against the issue and repository context. Correct "
            "any defect and return only the final complete git-applicable unified diff beginning with "
            "`diff --git`; do not add review prose or markdown fences."
        )
    if benchmark == "humaneval":
        return (
            "As QA Engineer, test the Engineer's logic mentally against edge cases and correct it if needed. "
            "Return exactly one fenced Python code block with the complete final implementation and no prose."
        )
    if benchmark == "mmlu":
        return (
            "As QA Engineer, independently verify the chosen option. Return exactly one line and nothing "
            "else: `FINAL_ANSWER: <A|B|C|D>`."
        )
    return (
        "As QA Engineer, independently recompute or verify the answer, correct it when needed, and return "
        "exactly one line and nothing else: `FINAL_ANSWER: <answer>`."
    )


def _history_text(history: list[Any]) -> str:
    parts = []
    for item in history:
        role = str(getattr(item, "role", "unknown"))
        sender = str(getattr(item, "sent_from", ""))
        content = str(getattr(item, "content", ""))
        parts.append(f"[{role} from {sender}]\n{content}")
    return "\n\n".join(parts)[-60_000:] or "[no prior messages]"


def _workspace_context() -> str:
    workspace = Path(REQUEST["workspace_path"]).resolve()
    task = str((REQUEST.get("sample") or {}).get("question") or REQUEST["prompt"])
    if not (workspace / ".git").exists():
        return "\nRepository context unavailable.\n"
    mode = REQUEST.get("repository_context_mode", "legacy")
    if mode not in {"legacy", "symbol-aware"}:
        raise ValueError("repository_context_mode must be legacy or symbol-aware")
    selector, extractor = _candidate_files, _relevant_source_context
    if mode == "symbol-aware":
        from inference.native.metagpt.repository_context import candidate_files, relevant_source_context
        selector, extractor = candidate_files, relevant_source_context
    files = selector(workspace, task)
    parts = ["\nRepository context selected from the checked-out base commit:"]
    budget = 60_000 - len(parts[0]) if mode == "symbol-aware" else 60_000
    for index, path in enumerate(files):
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        rel = path.relative_to(workspace)
        heading = f"\n--- {rel} ---\n"
        if mode == "symbol-aware":
            file_budget = min(18_000, max(0, budget // (len(files) - index) - len(heading) - 1))
            if file_budget < 128:
                continue
            chunk = heading + extractor(text, task, max_chars=file_budget)
        else:
            chunk = heading + extractor(text, task)
        if len(chunk) > budget:
            break
        parts.append(chunk)
        budget -= len(chunk) + (1 if mode == "symbol-aware" else 0)
    context = "\n".join(parts)
    if REQUEST.get("context_budget_mode") == "server-tokenizer":
        WORKSPACE_CONTEXT_PARTS[context] = parts
    return context


def _relevant_source_context(text: str, task: str, max_chars: int = 18_000) -> str:
    """Keep task-named definitions even when they occur deep in a large file."""
    if len(text) <= max_chars:
        return text
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text[:max_chars]
    identifiers = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", task))
    nodes = [node for node in ast.walk(tree)
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
             and node.name in identifiers]
    if not nodes:
        return text[:max_chars]
    lines = text.splitlines(keepends=True)
    chunks, remaining = [], max_chars
    covered = []
    for node in sorted(nodes, key=lambda node: (node.end_lineno - node.lineno, node.lineno)):
        start, end = max(0, node.lineno - 4), min(len(lines), node.end_lineno + 3)
        if any(lo <= start and end <= hi for lo, hi in covered):
            continue
        heading = f"\n# Source lines {start + 1}-{end}\n"
        chunk = heading + "".join(lines[start:end])
        if remaining < len(heading) + 128:
            break
        chunks.append(chunk[:remaining])
        remaining -= len(chunks[-1])
        covered.append((start, end))
    return "\n".join(chunks)[:max_chars]


def _candidate_files(workspace: Path, task: str) -> list[Path]:
    direct = re.findall(r"(?<![\w/])([\w./-]+\.(?:py|js|ts|tsx|java|go|rs))", task)
    scores: Counter[str] = Counter()
    for relative in direct:
        candidate = workspace / relative.lstrip("/")
        if candidate.is_file():
            scores[str(candidate.relative_to(workspace))] += 20

    stop = {
        "that", "this", "with", "from", "have", "when", "where", "which", "model",
        "return", "should", "does", "into", "using", "correct", "expected", "error",
        "test", "tests", "python", "class", "function", "value", "issue", "following",
    }
    tokens = [
        token
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", task)
        if token.lower() not in stop
    ]
    for token, _ in Counter(tokens).most_common(14):
        completed = subprocess.run(
            ["git", "-C", str(workspace), "grep", "-l", "-F", token, "--"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        for relative in completed.stdout.splitlines()[:40]:
            if Path(relative).suffix.lower() in {".py", ".js", ".ts", ".tsx", ".java", ".go", ".rs"}:
                scores[relative] += 1
    ranked = sorted(
        scores,
        key=lambda value: (
            -scores[value],
            1 if "test" in value.lower() else 0,
            len(value),
            value,
        ),
    )
    selected = [workspace / relative for relative in ranked[:10]]
    if not selected:
        completed = subprocess.run(
            ["git", "-C", str(workspace), "ls-files", "*.py"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        selected = [workspace / relative for relative in completed.stdout.splitlines()[:6]]
    return selected


def _serialize_history(memory: Any) -> list[dict[str, Any]]:
    if memory is None or not hasattr(memory, "get"):
        return []
    messages = []
    for item in memory.get():
        payload = item.model_dump(mode="json", exclude_none=True)
        if isinstance(payload, dict):
            messages.append(payload)
    return messages


def _materialize_benchmark_artifact(
    final_output: str,
    transcript: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    benchmark = str(REQUEST["benchmark"])
    workspace = Path(REQUEST["workspace_path"]).resolve()
    if benchmark == "humaneval":
        code = _extract_python_code(final_output)
        if not code:
            raise RuntimeError("MetaGPT QA Engineer did not return HumanEval Python code")
        path = workspace / "solution.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code.rstrip() + "\n", encoding="utf-8")
        return [{"kind": "humaneval_solution_written", "path": str(path), "bytes": path.stat().st_size}]
    if benchmark == "swe-bench-lite":
        candidates = [final_output] + [str(item.get("content") or "") for item in reversed(transcript)]
        attempts = []
        extracted_patches = 0
        for content in candidates:
            patch = _extract_diff(content)
            if not patch:
                continue
            extracted_patches += 1
            sections = [patch, *_split_file_diffs(patch)]
            applied_sections = 0
            for section_index, section in enumerate(sections):
                check = subprocess.run(
                    [
                        "git",
                        "-C",
                        str(workspace),
                        "apply",
                        "--check",
                        "--recount",
                        "-",
                    ],
                    input=section,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=120,
                )
                attempt = {
                    "section_index": section_index,
                    "check_returncode": check.returncode,
                    "stderr_tail": check.stderr[-2000:],
                    "patch_bytes": len(section.encode("utf-8")),
                }
                attempts.append(attempt)
                if check.returncode != 0:
                    continue
                apply = subprocess.run(
                    ["git", "-C", str(workspace), "apply", "--recount", "-"],
                    input=section,
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=120,
                )
                attempt["apply_returncode"] = apply.returncode
                attempt["apply_stderr_tail"] = apply.stderr[-2000:]
                if apply.returncode == 0:
                    applied_sections += 1
                    # A successful whole-patch application already contains all
                    # file sections, so no per-file fallback is needed.
                    if section_index == 0:
                        return [{"kind": "swe_bench_patch_applied", "attempts": attempts}]
            if applied_sections:
                return [
                    {
                        "kind": "swe_bench_patch_partially_applied",
                        "applied_file_sections": applied_sections,
                        "attempts": attempts,
                    }
                ]
        return [
            {
                "kind": "swe_bench_patch_rejected",
                "reason": (
                    "no_unified_diff" if extracted_patches == 0 else "no_git_applicable_section"
                ),
                "extracted_patch_candidates": extracted_patches,
                "attempts": attempts,
            }
        ]
    return []


def _extract_python_code(text: str) -> str | None:
    blocks = re.findall(r"```(?:python|py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if blocks:
        return blocks[-1].strip()
    start = text.find("def ")
    return text[start:].strip() if start >= 0 else None


def _extract_diff(text: str) -> str | None:
    blocks = re.findall(r"```(?:diff|patch)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidates = list(reversed(blocks)) + [text]
    for candidate in candidates:
        start = candidate.find("diff --git ")
        if start >= 0:
            return candidate[start:].strip() + "\n"
    return None


def _split_file_diffs(patch: str) -> list[str]:
    return [
        section.strip() + "\n"
        for section in re.split(r"(?=^diff --git )", patch, flags=re.MULTILINE)
        if section.strip().startswith("diff --git ")
    ]


if __name__ == "__main__":
    main()
