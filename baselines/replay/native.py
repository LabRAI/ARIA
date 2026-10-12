"""Execute a baseline intervention on the real host system and evaluator.

Success Rate and Harmful Intervention Rate are only meaningful when the repaired
trajectory is actually run and scored by the benchmark's official evaluator.
This backend therefore drives the same native multi-agent runner and the same
:func:`common.evaluation.evaluate_sample` entry point that ARIA's own repair
reruns use, so a row for ARIA and a row for a baseline differ by method alone.

Nothing here decides correctness on its own.  The evaluator owns that verdict,
and an evaluator that cannot resolve a run yields ``correct: None``, which the
aggregation counts as unresolved rather than silently as a failure.
"""

from __future__ import annotations

import logging
import os
import subprocess
import json
import signal
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from baselines.core.types import BaselineSample, ExecutionRecord
from baselines.replay.materialize import InterventionNotExecutable, materialize

LOGGER = logging.getLogger("masfm.baselines.replay")
REPO_ROOT = Path(__file__).resolve().parents[2]


class NativeControlUnavailable(RuntimeError):
    """A native control has no authoritative verdict; never a negative reward."""


class NativeReplayBackend:
    """Run one intervention through the host framework and its evaluator."""

    def __init__(
        self,
        *,
        config: Any,
        options: Mapping[str, Any] | None = None,
        spec: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.options = dict(options or {})
        self.spec = dict(spec or {})
        self.output_root = Path(
            self.options.get("output_root")
            or (
                Path(str(self.spec.get("output_dir") or "outputs/baselines")) / "replay"
            )
        )
        self.external_dir = Path(
            self.options.get("external_dir")
            or getattr(config, "external_dir", None)
            or REPO_ROOT / "external"
        )
        if not self.external_dir.is_absolute():
            self.external_dir = (REPO_ROOT / self.external_dir).resolve()
        self.evaluation_options = dict(self.options.get("evaluation") or {})
        self.timeout_seconds = self.options.get("timeout_seconds")
        self._resolver = None

    def prepare_sample(self, sample: BaselineSample) -> BaselineSample:
        """Share one actually executed control for an explicitly changed host."""
        if not self.options.get("paired_control"):
            return sample
        import fcntl
        import hashlib
        from uuid import uuid4
        from common.artifacts import atomic_write_json
        from common.runtime_support import safe_name
        from inference.runner import effective_api_config

        root = Path(self.options["control_output_root"]).resolve() / safe_name(sample.sample_id)
        root.mkdir(parents=True, exist_ok=True)
        host = self._host_config(self._source_result(sample), effective_api_config)
        from dataclasses import asdict
        identity = {"host_api": asdict(host.api), "native_options": self.options.get("native_options"),
                    "task_sha256": hashlib.sha256(sample.task.encode()).hexdigest(),
                    "benchmark": sample.benchmark, "mas": sample.mas}
        path = root / "control_execution.json"
        with (root / ".writer.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            payload = json.loads(path.read_text()) if path.is_file() else {}
            if payload and payload.get("identity") != identity:
                raise ValueError("paired control identity differs; use a new control_output_root")
            execution = payload.get("execution") or {}
            if not isinstance(execution.get("evaluation", {}).get("correct"), bool):
                attempt_root = root / "attempts" / uuid4().hex
                backend = NativeReplayBackend(config=self.config,
                    options={**self.options, "paired_control": False, "output_root": str(attempt_root)}, spec=self.spec)
                control = backend.execute(sample, None, label="unpatched-control")
                execution = control.to_dict()
                attempts = list(payload.get("attempts") or [])
                if not attempts and payload.get("execution"):
                    attempts.append({"execution": payload["execution"], "legacy": True})
                attempts.append({"output_root": str(attempt_root), "execution": execution})
                atomic_write_json(path, {"identity": identity, "execution": execution, "attempts": attempts})
            if not isinstance(execution.get("evaluation", {}).get("correct"), bool):
                detail = execution.get("error") or execution.get("evaluation", {}).get("reason") or "no authoritative verdict"
                raise NativeControlUnavailable(f"paired native control unresolved: {detail}")
        return replace(sample, control_evaluation=execution["evaluation"],
                       metadata={**sample.metadata, "paired_control_path": str(path),
                                 "control_execution_count": execution.get("execution_count", 1.0)})

    def resume_row_is_valid(self, row: Mapping[str, Any]) -> bool:
        """Never resume a boolean result produced by a different native host."""
        if not self.options.get("paired_control"):
            return True
        raw_path = row.get("paired_control_path")
        root = self.options.get("control_output_root")
        # Older native-v6 rows were written before the metadata replacement
        # propagated ``paired_control_path`` into the row.  Their embedded
        # control evaluation and explicit replay-host identity are still
        # sufficient to resume safely; force only genuinely mismatched or
        # unresolved rows through the fresh-control path.
        if not isinstance(raw_path, str) or not isinstance(root, str):
            control = row.get("control_evaluation")
            executions = row.get("executions")
            if not isinstance(control, Mapping) or not isinstance(
                control.get("correct", control.get("success")), bool
            ):
                return False
            if row.get("method") == "no_intervention" or (
                row.get("setting") == "active" and row.get("intervention") is None
            ):
                return True
            if not isinstance(executions, list) or not executions:
                return False
            for execution in executions:
                if not isinstance(execution, Mapping):
                    continue
                refs = execution.get("artifact_refs")
                if isinstance(refs, Mapping) and refs.get("replay_host_model"):
                    return refs.get("replay_host_model") == self._configured_host_model()
            # Active no-intervention rows have a valid embedded control but no
            # execution artifact beyond the control itself.
            return True
        try:
            path = Path(raw_path).resolve()
            expected = Path(root).resolve()
            if expected not in path.parents:
                return False
            payload = json.loads(path.read_text(encoding="utf-8"))
            execution = payload.get("execution") or {}
            return isinstance((execution.get("evaluation") or {}).get("correct"), bool)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return False

    def _configured_host_model(self) -> str | None:
        explicit = self.options.get("host_api")
        return str(explicit.get("model")) if isinstance(explicit, Mapping) else None

    def execute(
        self,
        sample: BaselineSample,
        intervention: Mapping[str, Any] | None,
        *,
        label: str,
    ) -> ExecutionRecord:
        attempts = [self._execute_once(sample, intervention, label=label)]
        retries = int((self.options.get("native_options") or {}).get("delivery_retries", 0))
        if retries < 0 or retries > 1:
            raise ValueError("native delivery_retries must be 0 or 1")
        for index in range(retries):
            if sample.benchmark != "swe-bench-lite" or attempts[-1].evaluation.get("harness_status") != "empty_patch":
                break
            retry_sample = replace(sample, task=sample.task + (
                "\n\n[HOST WORKSPACE CHECK]\n"
                "The preceding native attempt ended without modifying any repository source files. "
                "The task is not delivered. Inspect the checked-out source, then use the terminal or "
                "editor to write the actual fix to disk. A code block or a claim of completion is not "
                "a repository edit. Inspect git diff before finishing. Do not create a demonstration "
                "project or ask a human to apply the change. No benchmark test verdict is supplied."
            ))
            attempts.append(self._execute_once(retry_sample, intervention, label=f"{label}-delivery-{index + 1}"))
        if len(attempts) == 1:
            return attempts[0]
        return replace(attempts[-1], label=label,
                       execution_count=sum(item.execution_count for item in attempts),
                       artifact_refs={**dict(attempts[-1].artifact_refs),
                                      "delivery_attempts": [item.to_dict() for item in attempts]})

    def _execute_once(
        self,
        sample: BaselineSample,
        intervention: Mapping[str, Any] | None,
        *,
        label: str,
    ) -> ExecutionRecord:
        from common.catalog import find_mas
        from common.evaluation import evaluate_sample
        from common.runtime_support import safe_name
        from inference.runner import effective_api_config
        from repair.rerun import (
            build_repair_run_result,
            ensure_supported_repair_rerun,
            run_native_repair_attempt,
        )

        mas_slug = str(sample.mas or "")
        benchmark_slug = str(sample.benchmark or "")
        if intervention and intervention.get("action") in {"terminate", "escalate"}:
            return ExecutionRecord(
                label=label, status="completed", execution_mode="controller_stop",
                execution_count=0.0, evaluation={
                    "correct": False, "success": False, "score": 0.0,
                    "method": "controller-abstention", "reason":
                    "Controller stopped before delivering a task result; no human continuation is configured.",
                },
            )
        if not mas_slug or not benchmark_slug:
            return self._invalid(
                label, "sample does not declare both a mas and a benchmark"
            )
        try:
            ensure_supported_repair_rerun(mas_slug, benchmark_slug)
        except (InterventionNotExecutable, NotImplementedError, ValueError) as exc:
            return self._invalid(label, f"{type(exc).__name__}: {exc}", category="method")

        source_result = self._source_result(sample)
        try:
            host_config = self._host_config(source_result, effective_api_config)
        except (TypeError, ValueError) as exc:
            return self._invalid(label, f"{type(exc).__name__}: {exc}", category="infrastructure")
        try:
            run_sample = self._rerun_sample(sample, source_result, sample.task)
        except (TypeError, ValueError, FileNotFoundError) as exc:
            return self._invalid(label, f"{type(exc).__name__}: {exc}", category="infrastructure")
        run_root = self.output_root / safe_name(sample.sample_id) / safe_name(label)
        run_root.mkdir(parents=True, exist_ok=True)
        workspace = run_root / "workspace"
        known_files = {}
        try:
            if benchmark_slug == "swe-bench-lite":
                from inference.native.shared import prepare_swe_bench_workspace

                prepare_swe_bench_workspace(workspace, run_sample, run_root,
                                           cache_root=self.options.get("swebench_cache_dir"))
            elif benchmark_slug == "gaia":
                from inference.native.shared import prepare_gaia_attachment

                prepared = prepare_gaia_attachment(workspace, run_sample, run_root)
                if prepared.get("file_path"):
                    attachment = Path(prepared["file_path"])
                    known_files[attachment.name] = attachment
        except (OSError, RuntimeError, ValueError) as exc:
            return self._invalid(label, f"workspace preflight failed: {exc}", category="infrastructure")
        task_id = run_sample.get("instance_id")
        if not task_id:
            match = re.search(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$", sample.sample_id)
            task_id = match.group() if match else None
        try:
            continuation = materialize_in_workspace(
                sample, intervention, workspace,
                max_chars=self.options.get("continuation_char_budget"),
                task_id=str(task_id) if task_id else None,
                source_workspace=source_result.get("workspace_path"),
                repo_name=str(run_sample.get("repo") or "").rsplit("/", 1)[-1],
                known_files=known_files,
            )
        except (InterventionNotExecutable, ValueError) as exc:
            return self._invalid(label, f"{type(exc).__name__}: {exc}", category="method")
        run_sample["task"] = continuation["task"]
        run_id = f"baseline-{safe_name(sample.sample_id)}-{safe_name(label)}"
        spec = {**self.spec, **dict(self.options.get("native_options") or {}),
                "evaluation": self.evaluation_options}
        # Native clients must honor the host budget declared in YAML. A hidden
        # 1024-token cap truncates tool arguments, ledgers, and repository diffs.
        spec["max_tokens"] = int(host_config.api.max_tokens)
        spec["context_window_tokens"] = int(
            self.options.get("context_window_tokens") or host_config.api.context_window_tokens or 32768
        )
        try:
            arguments = dict(
                config=host_config,
                spec=spec,
                repo_root=REPO_ROOT,
                clone_path=self.external_dir / "mas" / find_mas(mas_slug).clone_dir,
                run_root=run_root,
                sample=run_sample,
                mas_slug=mas_slug,
                benchmark_slug=benchmark_slug,
            )
            environments = self.options.get("project_environments") or {}
            if mas_slug in environments:
                native = self._isolated_run(arguments, Path(environments[mas_slug]))
            else:
                native = run_native_repair_attempt(**arguments)
        except Exception as exc:  # noqa: BLE001 - host infrastructure boundary
            LOGGER.warning(
                "baseline replay failed sample=%s label=%s: %s",
                sample.sample_id,
                label,
                exc,
            )
            return self._invalid(
                label,
                f"{type(exc).__name__}: {exc}",
                run_root=run_root,
                execution_started=True,
                category="infrastructure",
            )
        if getattr(native, "status", None) != "completed":
            return self._invalid(
                label,
                f"native run status is {getattr(native, 'status', 'unknown')}: "
                f"{getattr(native, 'error', None)}",
                run_root=run_root,
                execution_started=True,
                category="infrastructure",
            )
        result = build_repair_run_result(
            config=host_config,
            source_result=source_result,
            sample=run_sample,
            run_id=run_id,
            run_root=run_root,
            clone_path=self.external_dir / "mas" / find_mas(mas_slug).clone_dir,
            native=native,
            mas_slug=mas_slug,
            benchmark_slug=benchmark_slug,
        )
        from common.artifacts import atomic_write_json

        atomic_write_json(run_root / "result.json", result)
        try:
            evaluation = evaluate_sample(
                benchmark_slug=benchmark_slug,
                sample=run_sample,
                result=result,
                run_root=run_root,
                evaluation_config=self.evaluation_options,
            )
        except Exception as exc:  # noqa: BLE001 - evaluator boundary
            LOGGER.warning(
                "baseline evaluator unresolved sample=%s label=%s: %s",
                sample.sample_id,
                label,
                exc,
            )
            evaluation = {
                "benchmark": benchmark_slug,
                "score": None,
                "correct": None,
                "method": "unresolved",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        atomic_write_json(run_root / "evaluation.json", evaluation)
        return ExecutionRecord(
            label=label,
            status="completed",
            evaluation=dict(evaluation),
            execution_mode=str(continuation["execution_mode"]),
            artifact_refs={
                "run_root": str(run_root),
                "result_path": str(run_root / "result.json"),
                "cut_event_id": continuation.get("cut_event_id"),
                "retained_event_count": len(continuation.get("retained_prefix", [])),
                "source_host_model": sample.metadata.get("host_model"),
                "replay_host_model": host_config.api.model,
                "baseline_method_model": getattr(
                    getattr(self.config, "api", None), "model", None
                ),
                "host_identity_mode": (
                    "explicit_configured_host" if self.options.get("host_api") else "source_host"
                ),
            },
        )

    def _isolated_run(self, arguments: dict[str, Any], environment: Path) -> Any:
        """Keep conflicting MAS stacks, global patches and workspaces in child processes."""
        from dataclasses import asdict
        from types import SimpleNamespace
        from common.artifacts import atomic_write_json
        from inference.native.shared import hide_benchmark_labels

        environment = environment if environment.is_absolute() else REPO_ROOT / environment
        executable = environment / "bin/python"
        if not executable.is_file():
            raise FileNotFoundError(f"missing native MAS environment: {executable}")
        run_root = arguments["run_root"]
        request = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in arguments.items() if key != "config"
        }
        request["sample"] = hide_benchmark_labels(dict(arguments["sample"]))
        request["swebench_cache_dir"] = arguments["sample"].get("_swebench_cache_dir")
        # The child resolves the named environment variable itself, never a persisted key.
        request["api_config"] = asdict(arguments["config"].api)
        request_path = run_root / "native_replay_request.json"
        atomic_write_json(request_path, request)
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), str(REPO_ROOT / "src")])}
        with (run_root / "native_replay_stdout.log").open("w") as stdout, (
            run_root / "native_replay_stderr.log"
        ).open("w") as stderr:
            process = subprocess.Popen(
                [str(executable), "-m", "baselines.replay.worker", str(request_path)],
                cwd=REPO_ROOT, env=env, stdout=stdout, stderr=stderr,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=int(self.timeout_seconds or 3600))
            except BaseException:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                raise
        output = run_root / "native_replay_result.json"
        if returncode != 0 or not output.is_file():
            detail = (run_root / "native_replay_stderr.log").read_text()[-3000:]
            raise RuntimeError(f"native child exited {returncode}: {detail}")
        return SimpleNamespace(**json.loads(output.read_text()))

    def _source_result(self, sample: BaselineSample) -> dict[str, Any]:
        stored = sample.metadata.get("source_result")
        result = dict(stored) if isinstance(stored, Mapping) else {}
        result.setdefault("sample_id", sample.sample_id)
        result.setdefault("mas", sample.mas)
        result.setdefault("benchmark", sample.benchmark)
        return result

    def _host_config(self, source_result: Mapping[str, Any], factory: Any) -> Any:
        """Restore the source host API without borrowing the baseline model API."""

        explicit = self.options.get("host_api")
        if explicit is not None:
            required = {"provider", "protocol", "base_url", "model", "api_key_required"}
            if not isinstance(explicit, Mapping) or not required.issubset(explicit):
                raise ValueError("explicit host_api must declare the complete endpoint identity")
            if getattr(self.config, "api", None) is None:
                from masfm.config import parse_api_config

                host_api = parse_api_config({"api": dict(explicit)}, required=True)
                return replace(self.config, api=host_api)
            return replace(self.config, api=factory(self.config.api, dict(explicit)))
        if getattr(self.config, "api", None) is None:
            raise ValueError("native replay requires explicit host_api when the method has no endpoint api")
        required = ("provider", "protocol", "base_url", "model", "api_key_required")
        missing = [key for key in required if source_result.get(key) is None]
        if missing:
            raise ValueError(
                "source result is missing host api identity fields: "
                + ", ".join(missing)
            )
        expected_model = source_result.get("model")
        recorded_model = source_result.get("model_id")
        overrides = {
            key: source_result[key]
            for key in (
                "provider",
                "protocol",
                "base_url",
                "model",
                "api_key_env",
                "api_key_required",
                "reasoning_effort",
                "runtime_id",
                "model_id",
                "upstream_provider",
                "allow_fallbacks",
            )
            if key in source_result
        }
        if overrides["api_key_required"] and not overrides.get("api_key_env"):
            raise ValueError("source host api requires a missing api_key_env")
        host_api = factory(self.config.api, overrides)
        if host_api.model != expected_model or (
            recorded_model is not None and host_api.model_id != recorded_model
        ):
            raise ValueError("reconstructed host api does not match source identity")
        return replace(self.config, api=host_api)

    def _rerun_sample(
        self,
        sample: BaselineSample,
        source_result: Mapping[str, Any],
        task: str,
    ) -> dict[str, Any]:
        """Build the runner input: patched task for the host, reference for the evaluator."""

        metadata = source_result.get("sample_metadata")
        run_sample: dict[str, Any] = (
            dict(metadata) if isinstance(metadata, Mapping) else {}
        )
        if sample.benchmark in {"gaia", "swe-bench-lite", "gsm-plus", "mmlu", "humaneval", "olympiadbench"}:
            from aria.post_training.environment import BenchmarkSampleResolver, EnvironmentIncident
            from aria.records import TrajectoryEvent

            if self._resolver is None:
                self._resolver = BenchmarkSampleResolver(data_dir=Path(self.config.data_dir))
            incident = EnvironmentIncident(
                incident_id=sample.sample_id, task_id=sample.task_id,
                split=str(sample.split or "test"), mas_id=str(sample.mas),
                benchmark_id=str(sample.benchmark), host_model_id="unknown", family_id="unknown",
                events=tuple(TrajectoryEvent.from_dict(event.to_dict()) for event in sample.events),
                feedback=None, constraints={"task_text": sample.task}, certified_cut_event_id=None,
                certified_cut_index=-1, first_bad_event_ids=(), verified_repair=None,
                attribution=None, localization=None,
            )
            resolved, detail = self._resolver.resolve(incident)
            if resolved is None:
                raise ValueError(f"benchmark preflight failed: {detail}")
            run_sample.update(resolved)
        run_sample["task"] = task
        run_sample["question"] = run_sample.get("question") or sample.task
        run_sample["sample_id"] = sample.sample_id
        run_sample["sample_index"] = source_result.get(
            "sample_index", run_sample.get("sample_index")
        )
        reference = run_sample.get("reference_answer")
        if reference in (None, ""):
            reference = sample.source_evaluation.get("reference_answer")
        if reference in (None, ""):
            reference = sample.hidden_reference.get("answer")
        if reference in (None, ""):
            reference = run_sample.get("reference_answer", "")
        run_sample["reference_answer"] = str(reference)
        if self.options.get("swebench_cache_dir"):
            run_sample["_swebench_cache_dir"] = str(Path(self.options["swebench_cache_dir"]).resolve())
        workspace_mode = (self.options.get("native_options") or {}).get("swebench_workspace_mode")
        if workspace_mode is not None:
            run_sample["_swebench_workspace_mode"] = workspace_mode
        return run_sample

    def _invalid(
        self,
        label: str,
        reason: str,
        *,
        run_root: Path | None = None,
        execution_started: bool = False,
        category: str = "infrastructure",
    ) -> ExecutionRecord:
        return ExecutionRecord(
            label=label,
            status="invalid",
            evaluation={
                "score": None,
                "correct": None,
                "method": "unresolved",
                "reason": reason,
                "failure_category": category,
            },
            execution_mode="native_failed" if execution_started else "not_executed",
            execution_count=1.0 if execution_started else 0.0,
            artifact_refs={"run_root": str(run_root)} if run_root is not None else {},
            error=reason,
        )


def build_native_replay_backend(
    options: Mapping[str, Any],
    spec: Mapping[str, Any],
    config: Any,
) -> NativeReplayBackend:
    """Factory referenced from YAML as ``baselines.replay.native:build_native_replay_backend``."""

    return NativeReplayBackend(config=config, options=options, spec=spec)


_WORKSPACE_PATH_RE = re.compile(
    r"(?:/[A-Za-z0-9_.-]+)*?/workspace(?=/|[\s\"'`\\),\]}:;]|$)"
)


def _rewrite_workspace_paths(value: str, workspace: Path, *, task_id: str | None = None,
                             source_workspace: str | None = None, repo_name: str = "",
                             known_files: Mapping[str, Path] | None = None) -> str:
    """Replace the old workspace root, never the repository-relative filename."""
    patterns = []
    boundary = r"(?=/|[.]?(?:[\s\"'`\\),\]}:;]|$))"
    variants = {task_id, task_id.replace("__", "-"), task_id.replace("_", "-")} if task_id else set()

    def substitute(pattern, replacement):
        nonlocal value
        urls = [(m.start(), m.end()) for m in re.finditer(r"https?://[^\s\"'`]+", value)]
        value = pattern.sub(lambda match: match.group() if any(lo <= match.start() < hi for lo, hi in urls)
                            else replacement(match), value)

    for name, path in (known_files or {}).items():
        pattern = re.compile(r"/(?:[A-Za-z0-9_.-]+/)+" + re.escape(name) + boundary)
        substitute(pattern, lambda _match, path=path: str(path))
    if source_workspace:
        patterns.append(re.compile(re.escape(str(source_workspace).rstrip("/")) + boundary))
    if task_id:
        patterns.append(re.compile(
            _WORKSPACE_PATH_RE.pattern + r"/manual_[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*?/(?:"
            + "|".join(re.escape(item) for item in sorted(variants)) + ")" + boundary
        ))
    for pattern in patterns:
        substitute(pattern, lambda _match: str(workspace))

    legacy = re.compile(_WORKSPACE_PATH_RE.pattern +
        r"/manual_(?:vllm|api)/(?:ag2|openmanus|metagpt|magentic[-_]one)"
        r"/(?:swe-bench-lite|gaia|humaneval|gsm-plus|mmlu|olympiadbench)"
        r"/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?P<tail>(?:/[A-Za-z0-9_.-]+)*)")

    def relocate_legacy(match):
        tail = match.group("tail").lstrip("/")
        punctuation = "." if tail.endswith(".") else ""
        parts = tail.removesuffix(".").split("/") if tail else []
        if parts and parts[0] in variants:
            parts = parts[1:]
        if ".." not in parts:
            for start in range(len(parts)):
                candidate = workspace.joinpath(*parts[start:])
                if candidate.exists() and candidate.resolve().is_relative_to(workspace.resolve()):
                    return str(candidate) + punctuation
        if parts and parts[0] == repo_name and len(parts) > 1:
            if parts[1] == repo_name or not (workspace / repo_name).exists():
                parts = parts[1:]
        return str(workspace.joinpath(*parts)) + punctuation

    substitute(legacy, relocate_legacy)
    substitute(_WORKSPACE_PATH_RE, lambda _match: str(workspace))
    return value


def materialize_in_workspace(sample: BaselineSample, intervention: Mapping[str, Any] | None,
                             workspace: Path, *, max_chars: int | None = None,
                             task_id: str | None = None, source_workspace: str | None = None,
                             repo_name: str = "", known_files: Mapping[str, Path] | None = None) -> dict[str, Any]:
    def rewrite(value):
        if isinstance(value, str):
            return _rewrite_workspace_paths(value, workspace, task_id=task_id, source_workspace=source_workspace,
                                            repo_name=repo_name, known_files=known_files)
        if isinstance(value, Mapping):
            return {key: rewrite(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(rewrite(item) for item in value)
        return value

    rewritten = replace(sample, task=rewrite(sample.task), events=tuple(
        replace(event, content=rewrite(event.content), tool_calls=rewrite(event.tool_calls),
                tool_results=rewrite(event.tool_results), metadata=rewrite(event.metadata))
        for event in sample.events
    ))
    # Path expansion must happen before the budget calculation, not after it.
    return materialize(rewritten, rewrite(intervention), max_chars=max_chars)
