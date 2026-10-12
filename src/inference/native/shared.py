"""Benchmark workspace helpers shared by all native MAS implementations."""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from common.artifacts import atomic_write_json, atomic_write_jsonl


_SWEBENCH_CACHE_LOCK = threading.Lock()


@contextmanager
def _swebench_cache_lock(cache_root: Path):
    """Serialize Git worktree registration across isolated native processes."""
    import fcntl

    cache_root.mkdir(parents=True, exist_ok=True)
    with _SWEBENCH_CACHE_LOCK, (cache_root / ".aria-cache.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)

BENCHMARK_LABEL_FIELDS = frozenset(
    {
        "reference_answer",
        "reference_solution",
        "reference_answers",
        "canonical_solution",
        "answer_index",
        "seed_answer",
        "final_answer",
        "test",
        "test_patch",
        "fail_to_pass",
        "pass_to_pass",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "patch",
        "answer",
        "_swebench_cache_dir",
        "_swebench_workspace_mode",
    }
)


def safe_api_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Remove any inline secret before persisting a launcher request."""

    return {key: value for key, value in settings.items() if key != "api_key"}


def hide_benchmark_labels(sample: dict[str, Any]) -> dict[str, Any]:
    """Remove evaluator-only labels from a model-visible launcher request."""
    def strip(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: strip(item) for key, item in value.items() if key not in BENCHMARK_LABEL_FIELDS}
        if isinstance(value, list):
            return [strip(item) for item in value]
        if isinstance(value, tuple):
            return tuple(strip(item) for item in value)
        return value

    return strip(sample)


def timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def write_aria_artifacts(run_root: Path, payload: dict[str, Any]) -> None:
    atomic_write_json(run_root / "aria_runtime.json", payload)
    atomic_write_json(run_root / "aria_patch_history.json", payload.get("patch_history", []))
    atomic_write_jsonl(run_root / "aria_event_log.jsonl", payload.get("events", []))
    atomic_write_jsonl(run_root / "aria_model_interventions.jsonl", payload.get("model_interventions", []))


def gaia_attachment_instruction(sample: dict[str, Any]) -> str:
    path = sample.get("file_path") or sample.get("attachment_path")
    if not path:
        return ""
    return f"Attached file, if needed: {path}\n\n"


def prepare_gaia_attachment(
    workspace_path: Path,
    sample: dict[str, Any],
    run_root: Path,
) -> dict[str, Any]:
    source_raw = sample.get("file_path") or sample.get("attachment_path")
    if not source_raw:
        return sample
    source = Path(str(source_raw)).resolve()
    if not source.exists():
        raise FileNotFoundError(
            f"GAIA attachment is missing: {source}. Enable prepare.gaia_attachments "
            "and export HF_TOKEN or HUGGINGFACE_HUB_TOKEN."
        )
    attachment_dir = workspace_path / "attachments"
    attachment_dir.mkdir(parents=True, exist_ok=True)
    destination = attachment_dir / source.name
    shutil.copy2(source, destination)
    prepared = dict(sample)
    prepared["file_path"] = str(destination.resolve())
    atomic_write_json(
        run_root / "gaia_attachment.json",
        {
            "source_path": str(source),
            "workspace_path": str(destination.resolve()),
            "bytes": destination.stat().st_size,
        },
    )
    return prepared


def _swe_bench_commit_is_local(bare_repo: Path, base_commit: str) -> bool:
    """Probe local objects without triggering a promisor remote's lazy fetch."""
    # Enumerate stored objects rather than looking up a missing hash: older
    # Git versions can still lazy-fetch after a promisor config override.
    result = subprocess.run(
        ["git", f"--git-dir={bare_repo}", "cat-file", "--batch-all-objects",
         "--batch-check=%(objectname) %(objecttype)"],
        capture_output=True, text=True, check=True, timeout=30,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )
    return f"{base_commit.lower()} commit" in result.stdout.splitlines()


def prepare_swe_bench_workspace(
    workspace_path: Path,
    sample: dict[str, Any],
    run_root: Path,
    *,
    cache_root: Path | None = None,
) -> None:
    repo = str(sample.get("repo") or "").strip()
    base_commit = str(sample.get("base_commit") or "").strip()
    if not repo or not base_commit:
        raise ValueError("SWE-Bench Lite sample requires repo and base_commit")
    mode = sample.get("_swebench_workspace_mode", "shared-worktree")
    if mode not in {"shared-worktree", "isolated-clone"}:
        raise ValueError("SWE workspace mode must be shared-worktree or isolated-clone")
    if mode == "isolated-clone" and (workspace_path / ".git").exists() and not (workspace_path / ".git").is_dir():
        raise ValueError("isolated clone cannot reuse a shared worktree; use a fresh output identity")

    cache_root = cache_root or sample.get("_swebench_cache_dir")
    cache_root = Path(cache_root) if cache_root is not None else workspace_path.parent / ".repo-cache"
    bare_repo = cache_root / f"{repo.replace('/', '__')}.git"
    repo_url = f"https://github.com/{repo}.git"
    command_log: list[dict[str, Any]] = []

    def run(
        command: list[str],
        *,
        timeout: int = 900,
        attempts: int = 6,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        for attempt in range(1, attempts + 1):
            started = time.monotonic()
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                input=input_text,
                timeout=timeout,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            )
            command_log.append(
                {
                    "command": command,
                    "attempt": attempt,
                    "returncode": completed.returncode,
                    "elapsed_seconds": time.monotonic() - started,
                    "stdout_tail": completed.stdout[-4000:],
                    "stderr_tail": completed.stderr[-4000:],
                    **({"stdin": input_text} if input_text is not None else {}),
                }
            )
            atomic_write_json(
                run_root / "swebench_workspace_setup.json",
                {
                    "repo": repo,
                    "repo_url": repo_url,
                    "base_commit": base_commit,
                    "workspace_path": str(workspace_path),
                    "bare_cache_path": str(bare_repo),
                    "workspace_mode": mode,
                    "commands": command_log,
                },
            )
            if completed.returncode == 0:
                return completed
            if attempt == attempts or not is_transient_git_error(completed.stderr):
                break
            time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError(
            f"SWE-Bench workspace setup failed ({completed.returncode}): "
            f"{' '.join(command)}\n{completed.stderr[-2000:]}"
        )

    # Multiple baseline rows may share a repository cache. Keep cache setup and
    # fetch serialized while allowing their isolated worktrees to be prepared
    # by separate workers.
    with _swebench_cache_lock(cache_root):
        if not bare_repo.exists():
            cache_root.mkdir(parents=True, exist_ok=True)
            run(["git", "init", "--bare", str(bare_repo)])
            run(["git", f"--git-dir={bare_repo}", "remote", "add", "origin", repo_url])
            run(["git", f"--git-dir={bare_repo}", "config", "remote.origin.promisor", "true"])
            run(
                [
                    "git",
                    f"--git-dir={bare_repo}",
                    "config",
                    "remote.origin.partialclonefilter",
                    "blob:none",
                ]
            )

        # A missing commit in a promisor repository otherwise makes cat-file
        # perform network I/O under its 30-second local-probe timeout. Fetch
        # explicitly below, with the normal transport timeout and retry log.
        if not _swe_bench_commit_is_local(bare_repo, base_commit):
            run(["git", f"--git-dir={bare_repo}", "fetch", "--depth=1", "origin", base_commit])
        run(["git", f"--git-dir={bare_repo}", "worktree", "prune"])
    if mode == "isolated-clone":
        if not (workspace_path / ".git").exists():
            if workspace_path.exists():
                shutil.rmtree(workspace_path)
            run(["git", "init", str(workspace_path)])
        # Copy only the requested revision through Git transport. A private
        # config, refs, and object store prevent ordinary task Git commands
        # from mutating another task through shared worktree metadata.
        with _swebench_cache_lock(cache_root):
            previous_missing = None
            while True:
                objects = run(["git", f"--git-dir={bare_repo}", "rev-list", "--objects",
                               "--no-object-names", "--max-count=1", "--missing=print", base_commit]).stdout
                missing = sorted(line[1:] for line in objects.splitlines() if line.startswith("?"))
                if not missing:
                    break
                if missing == previous_missing:
                    raise RuntimeError("SWE cache hydration did not recover its missing objects")
                # upload-pack cannot lazily fetch a partial clone's objects.
                # Explicit wants also work on Git versions without --refetch.
                run(["git", f"--git-dir={bare_repo}", "fetch", "--no-filter", "--no-tags",
                     "--no-write-fetch-head", "origin", "--stdin"], input_text="\n".join(missing) + "\n")
                previous_missing = missing
            run(["git", "-C", str(workspace_path), "fetch", "--no-tags", "--depth=1",
                 "--update-shallow", str(bare_repo.resolve()), base_commit])
        run(["git", "-C", str(workspace_path), "config", "--replace-all", "remote.origin.url", repo_url])
        run(["git", "-C", str(workspace_path), "checkout", "--force", "--detach", base_commit])
        run(["git", "-C", str(workspace_path), "clean", "-fdx"])
        actual_head = run(["git", "-C", str(workspace_path), "rev-parse", "HEAD"]).stdout.strip()
        if actual_head != base_commit:
            raise RuntimeError(f"SWE-Bench workspace HEAD mismatch: expected {base_commit}, got {actual_head}")
        return
    if (workspace_path / ".git").exists():
        run(["git", "-C", str(workspace_path), "reset", "--hard", base_commit])
        run(["git", "-C", str(workspace_path), "clean", "-fdx"])
    else:
        if workspace_path.exists():
            shutil.rmtree(workspace_path)
        add_command = [
            "git",
            f"--git-dir={bare_repo}",
            "worktree",
            "add",
            "--force",
            "--detach",
            str(workspace_path),
            base_commit,
        ]
        for attempt in range(1, 4):
            if workspace_path.exists():
                shutil.rmtree(workspace_path)
            try:
                with _swebench_cache_lock(cache_root):
                    run(["git", f"--git-dir={bare_repo}", "worktree", "prune"])
                    run(add_command, attempts=1)
                break
            except RuntimeError:
                if attempt == 3:
                    raise
                with _swebench_cache_lock(cache_root):
                    run(["git", f"--git-dir={bare_repo}", "fetch", "--depth=1", "origin", base_commit])
    actual_head = run(["git", "-C", str(workspace_path), "rev-parse", "HEAD"]).stdout.strip()
    if actual_head != base_commit:
        raise RuntimeError(
            f"SWE-Bench workspace HEAD mismatch: expected {base_commit}, got {actual_head}"
        )


def is_transient_git_error(stderr: str) -> bool:
    text = stderr.lower()
    return any(
        marker in text
        for marker in (
            "connection reset",
            "could not resolve host",
            "early eof",
            "http 429",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
            "http2 framing",
            "operation timed out",
            "remote end hung up unexpectedly",
            "rpc failed",
            "ssl_connect",
            "ssl_error_syscall",
            "temporary failure",
            "tls connection",
            "transferred a partial file",
            "unexpected eof",
        )
    )
