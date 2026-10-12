"""Materialize baseline SWE-Bench interventions as repository patches.

The baseline methods emit trajectory events, not native MAS workspace state.
SWE-Bench cannot be scored from that text alone: the intervention must produce
an actual diff in a checkout at the benchmark's exact base commit.  This module
supports the two representations used by the existing baselines:

* a unified diff embedded in a response; or
* executable shell/Python blocks that edit the isolated checkout.

Natural-language-only interventions remain evaluator-unresolved.  They are not
silently converted into incorrect patches.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from common.artifacts import atomic_write_json
from common.evaluation import collect_swe_bench_patch
from inference.native.shared import prepare_swe_bench_workspace

__all__ = [
    "materialize_swe_bench_patch",
    "extract_unified_diffs",
]


_CODE_BLOCK_RE = re.compile(
    r"```(?P<language>[A-Za-z0-9_+.-]*)\s*\n?(?P<code>.*?)```",
    flags=re.IGNORECASE | re.DOTALL,
)
_DIFF_START_RE = re.compile(r"(?m)^diff --git [^\n]+")
_SUPPORTED_SCRIPT_LANGUAGES = {"sh", "bash", "shell", "zsh", "python", "py"}


def materialize_swe_bench_patch(
    *,
    workspace: Path,
    sample: Mapping[str, Any],
    replacement_events: Sequence[Mapping[str, Any]],
    run_root: Path,
    cache_root: Path | None = None,
) -> dict[str, Any]:
    """Prepare a checkout and materialize one baseline intervention.

    The returned mapping is persisted by the caller and intentionally contains
    command tails rather than hidden model state.  ``status`` is one of
    ``patch_applied``, ``script_applied``, or ``invalid``.
    """

    run_root.mkdir(parents=True, exist_ok=True)
    try:
        prepare_swe_bench_workspace(
            workspace,
            dict(sample),
            run_root,
            cache_root=cache_root,
        )
    except Exception as exc:  # noqa: BLE001 - recorded as evaluator evidence
        setup_error = f"{type(exc).__name__}: {exc}"
        payload = {
            "status": "invalid",
            "reason": f"SWE-Bench checkout unavailable: {setup_error}",
            "extracted_diff_count": 0,
            "script_count": 0,
            "commands": [],
        }
        atomic_write_json(run_root / "swebench_patch_materialization.json", payload)
        return payload

    contents = [
        str(event.get("content") or "")
        for event in replacement_events
        if isinstance(event, Mapping) and str(event.get("content") or "").strip()
    ]
    diff_candidates = _unique_preserving_order(
        diff for content in contents for diff in extract_unified_diffs(content)
    )
    attempts: list[dict[str, Any]] = []
    for index, candidate in enumerate(diff_candidates):
        result = _try_apply_patch(workspace, candidate)
        result["candidate_index"] = index
        attempts.append(result)
        if result["applied"]:
            patch = collect_swe_bench_patch(workspace, str(sample.get("base_commit") or ""))
            payload = {
                "status": "patch_applied",
                "source": "unified_diff",
                "patch": patch["patch"],
                "changed_files": patch["changed_files"],
                "commands": [*attempts, *patch["commands"]],
                "extracted_diff_count": len(diff_candidates),
                "script_count": 0,
            }
            atomic_write_json(run_root / "swebench_patch_materialization.json", payload)
            return payload

    script_blocks = [
        (language.lower(), code)
        for content in contents
        for language, code in _script_blocks(content)
    ]
    script_attempts: list[dict[str, Any]] = []
    for index, (language, code) in enumerate(script_blocks):
        script = _rewrite_workspace_paths(code, workspace, sample)
        if language in {"python", "py"}:
            command = [sys.executable, "-"]
        else:
            command = ["bash", "-lc", script]
        try:
            completed = subprocess.run(
                command,
                cwd=workspace,
                input=script if language in {"python", "py"} else None,
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
                env={"PATH": "/usr/local/bin:/usr/bin:/bin", **_safe_environment()},
            )
            attempt = {
                "block_index": index,
                "language": language,
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
            }
        except subprocess.TimeoutExpired as exc:
            attempt = {
                "block_index": index,
                "language": language,
                "returncode": None,
                "stdout_tail": _timeout_text(exc.stdout),
                "stderr_tail": _timeout_text(exc.stderr),
                "timeout_seconds": 300,
            }
        script_attempts.append(attempt)

    patch = collect_swe_bench_patch(workspace, str(sample.get("base_commit") or ""))
    if patch["patch"].strip():
        payload = {
            "status": "script_applied",
            "source": "executable_code_blocks",
            "patch": patch["patch"],
            "changed_files": patch["changed_files"],
            "commands": [*attempts, *script_attempts, *patch["commands"]],
            "extracted_diff_count": len(diff_candidates),
            "script_count": len(script_blocks),
        }
    else:
        payload = {
            "status": "invalid",
            "reason": (
                "intervention did not materialize a non-empty repository patch; "
                "no applicable unified diff or editing script was found"
            ),
            "commands": [*attempts, *script_attempts, *patch["commands"]],
            "extracted_diff_count": len(diff_candidates),
            "script_count": len(script_blocks),
        }
    atomic_write_json(run_root / "swebench_patch_materialization.json", payload)
    return payload


def extract_unified_diffs(text: str) -> list[str]:
    """Extract candidate diffs from fenced or prose model output."""

    candidates: list[str] = []
    for match in _CODE_BLOCK_RE.finditer(text):
        block = match.group("code").strip()
        if _DIFF_START_RE.search(block):
            candidates.append(_clean_diff(block))
    start = _DIFF_START_RE.search(text)
    if start is not None:
        candidates.append(_clean_diff(text[start.start() :]))
    return _unique_preserving_order(item for item in candidates if item.strip())


def _script_blocks(text: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    for match in _CODE_BLOCK_RE.finditer(text):
        language = match.group("language").strip().lower()
        code = match.group("code").strip()
        if language in _SUPPORTED_SCRIPT_LANGUAGES and code:
            blocks.append((language, code))
    return blocks


def _try_apply_patch(workspace: Path, patch: str) -> dict[str, Any]:
    check = subprocess.run(
        ["git", "-C", str(workspace), "apply", "--check", "--recount", "--binary", "-"],
        input=patch,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    result: dict[str, Any] = {
        "operation": "git_apply",
        "check_returncode": check.returncode,
        "check_stderr_tail": check.stderr[-4000:],
        "patch_bytes": len(patch.encode("utf-8")),
        "applied": False,
    }
    if check.returncode != 0:
        return result
    applied = subprocess.run(
        ["git", "-C", str(workspace), "apply", "--recount", "--binary", "-"],
        input=patch,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    result.update(
        {
            "apply_returncode": applied.returncode,
            "apply_stderr_tail": applied.stderr[-4000:],
            "applied": applied.returncode == 0,
        }
    )
    return result


def _clean_diff(value: str) -> str:
    lines = value.replace("\r\n", "\n").splitlines()
    while lines and not lines[0].startswith("diff --git "):
        lines.pop(0)
    while lines and lines[-1].strip() in {"```", "```diff", "```patch"}:
        lines.pop()
    return "\n".join(lines).rstrip() + "\n" if lines else ""


def _rewrite_workspace_paths(
    script: str, workspace: Path, sample: Mapping[str, Any]
) -> str:
    """Map source-run absolute repository paths into the fresh checkout."""

    repo_name = str(sample.get("repo") or "").rsplit("/", 1)[-1].strip()
    if not repo_name:
        return script

    absolute_path = re.compile(r"/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+")

    def replace(match: re.Match[str]) -> str:
        value = match.group(0)
        marker = f"/{repo_name}"
        position = value.find(marker)
        if position < 0:
            return value
        suffix = value[position + len(marker) :]
        return str(workspace) + suffix

    return absolute_path.sub(replace, script)


def _safe_environment() -> dict[str, str]:
    import os

    return {
        key: value
        for key, value in os.environ.items()
        if key in {"HOME", "LANG", "LC_ALL", "PATH", "PYTHONPATH", "PIP_INDEX_URL"}
    }


def _timeout_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")[-4000:]
    return str(value)[-4000:]


def _unique_preserving_order(values: Sequence[str] | Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
