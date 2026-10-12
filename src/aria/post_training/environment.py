"""The Stage 3 execution environment retained from corpus construction.

Section~\\ref{subsubsec:stage3} obtains its rewards by re-executing the policy's
own samples in the environment already assembled to certify a record: an
isolated harness that restores execution from a retained prefix, together with
the benchmark evaluator that decides the outcome.  No model is asked to judge
whether a repair succeeded, so every backend here ends in an official benchmark
evaluator or in a recorded verdict from one.

Three backends implement that contract at different execution costs.

``evaluator``
    Restores the prefix, substitutes the sampled continuation, and scores the
    resulting final state with the benchmark's official evaluator through
    :func:`common.evaluation.evaluate_sample`.  Only the bounded continuation is
    materialized rather than re-run through the host framework, which is the
    ``non_exact_continuation`` execution mode the corpus already records.  A
    continuation whose events do not determine a final state is *unresolved*
    rather than scored.

``ledger``
    Replays the verdicts the corpus recorded for splices that were actually
    executed, keyed by cut point and normalized continuation content.  A sample
    that matches no recorded execution is unresolved.

``native``
    Re-runs the host framework itself and scores the run with the same
    evaluator, which is the configuration used for cluster-scale training.

An unresolved outcome carries no reward and is dropped from its group, in the
same way as a group without reward spread; it is never silently counted as a
failed repair.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from aria.records import IncidentRecord, RepairRecord, TrajectoryEvent

ENVIRONMENT_SCHEMA_VERSION = "aria-stage3-environment-v1"

ACCEPTED = "accepted"
REJECTED = "rejected"
INVALID = "invalid"
UNRESOLVED = "unresolved"

__all__ = [
    "ACCEPTED",
    "ENVIRONMENT_SCHEMA_VERSION",
    "INVALID",
    "REJECTED",
    "UNRESOLVED",
    "EnvironmentIncident",
    "EvaluatorEnvironment",
    "ExecutionOutcome",
    "LedgerEntry",
    "LedgerEnvironment",
    "NativeEnvironment",
    "BenchmarkSampleResolver",
    "RepairEnvironment",
    "build_environment",
    "load_environment_incidents",
]


@dataclass(frozen=True)
class LedgerEntry:
    """One splice that was executed and scored during corpus construction."""

    label: str
    cut_event_id: str | None
    replacement_events: tuple[dict[str, Any], ...]
    success: bool | None
    score: float | None
    execution_mode: str = "unknown"

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> LedgerEntry:
        events = raw.get("replacement_events") or []
        return cls(
            label=str(raw.get("label") or "unlabeled"),
            cut_event_id=raw.get("cut_event_id"),
            replacement_events=tuple(
                dict(item) for item in events if isinstance(item, Mapping)
            ),
            success=raw.get("success") if isinstance(raw.get("success"), bool) else None,
            score=_numeric(raw.get("score")),
            execution_mode=str(raw.get("execution_mode") or "unknown"),
        )


@dataclass(frozen=True)
class EnvironmentIncident:
    """Everything the environment needs to score one incident's samples.

    ``reference`` and ``evaluator_sample`` are evaluator-only: they carry the
    hidden answer label and are never placed in a model prompt.  ``model_input``
    is the answer-blind view, and the two are kept in separate fields so that a
    prompt builder cannot reach the label by accident.
    """

    incident_id: str
    task_id: str
    split: str
    mas_id: str
    benchmark_id: str
    host_model_id: str
    family_id: str
    events: tuple[TrajectoryEvent, ...]
    feedback: str | None
    constraints: Mapping[str, Any]
    certified_cut_event_id: str | None
    certified_cut_index: int
    first_bad_event_ids: tuple[str, ...]
    verified_repair: Mapping[str, Any] | None
    attribution: Mapping[str, Any] | None
    localization: Mapping[str, Any] | None
    evaluator_sample: Mapping[str, Any] = field(repr=False, default_factory=dict)
    control_success: bool | None = None
    control_score: float | None = None
    source_success: bool | None = None
    source_score: float | None = None
    ledger: tuple[LedgerEntry, ...] = ()
    verified_successful: bool = False
    requires_native_control: bool = False

    @property
    def cell_key(self) -> str:
        return f"{self.mas_id}/{self.benchmark_id}/{self.host_model_id}"

    @property
    def matched_control_success(self) -> bool | None:
        """The outcome of this task with no intervention.

        A failed incident has an explicit control re-execution.  A sampled
        successful run has none, because nothing was repaired: its own
        uninterrupted execution *is* the control, so it is used directly.
        """

        if self.requires_native_control:
            return self.control_success
        return self.control_success if self.control_success is not None else self.source_success

    @property
    def matched_control_score(self) -> float | None:
        if self.requires_native_control:
            return self.control_score
        return self.control_score if self.control_score is not None else self.source_score

    def model_input(self, *, feedback: str | None) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "events": [event.to_dict() for event in self.events],
            "feedback": feedback,
            "constraints": dict(self.constraints),
        }

    def prefix_input(self, prefix_length: int, *, feedback: str | None) -> dict[str, Any]:
        """Return the runtime view: only the first ``prefix_length`` events.

        Stage 3a truncates contexts at prefixes drawn from the gate's trigger
        distribution because a runtime intervention observes only
        :math:`\\tau_{\\leq t}`.
        """

        bounded = max(1, min(prefix_length, len(self.events)))
        return {
            "incident_id": self.incident_id,
            "events": [event.to_dict() for event in self.events[:bounded]],
            "feedback": feedback,
            "constraints": dict(self.constraints),
        }

    def as_incident_record(self) -> IncidentRecord:
        return IncidentRecord(
            incident_id=self.incident_id,
            events=self.events,
            feedback=self.feedback,
            constraints=dict(self.constraints),
            family_id=self.family_id,
            verified_successful=self.verified_successful,
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> EnvironmentIncident:
        events = tuple(
            TrajectoryEvent.from_dict(item, name=f"events[{index}]")
            for index, item in enumerate(raw.get("events") or [])
            if isinstance(item, Mapping)
        )
        if not events:
            raise ValueError("environment incident requires at least one event")
        certified = raw.get("certified") or {}
        cut_event_id = certified.get("cut_event_id")
        cut_index = _index_of(events, cut_event_id)
        cell = raw.get("cell") or {}
        control = raw.get("control") or {}
        source = raw.get("source") or {}
        return cls(
            incident_id=str(raw["incident_id"]),
            task_id=str(raw.get("task_id") or raw["incident_id"]),
            split=str(raw.get("split") or "train"),
            mas_id=str(cell.get("mas_id") or "unknown"),
            benchmark_id=str(cell.get("benchmark_id") or "unknown"),
            host_model_id=str(cell.get("host_model_id") or "unknown"),
            family_id=str(raw.get("family_id") or "unknown"),
            events=events,
            feedback=raw.get("feedback"),
            constraints=dict(raw.get("constraints") or {}),
            certified_cut_event_id=cut_event_id,
            certified_cut_index=cut_index,
            first_bad_event_ids=tuple(
                str(value) for value in (certified.get("first_bad_event_ids") or [])
            ),
            verified_repair=certified.get("repair"),
            attribution=raw.get("attribution"),
            localization=raw.get("localization"),
            evaluator_sample=dict(raw.get("evaluator_sample") or {}),
            control_success=_boolean(control.get("success")),
            control_score=_numeric(control.get("score")),
            source_success=_boolean(source.get("success")),
            source_score=_numeric(source.get("score")),
            ledger=tuple(
                LedgerEntry.from_dict(item)
                for item in (raw.get("ledger") or [])
                if isinstance(item, Mapping)
            ),
            verified_successful=bool(raw.get("verified_successful", False)),
            requires_native_control=(raw.get("requires_native_control") is True
                or "fresh_native_execution_required" in (raw.get("reward_capabilities") or [])),
        )


@dataclass(frozen=True)
class ExecutionOutcome:
    """The environment's verdict on one sampled record."""

    status: str
    success: bool | None
    score: float | None
    harmful: bool
    execution_count: float
    grounding: str
    detail: str | None = None

    @property
    def accepted(self) -> bool:
        return self.status == ACCEPTED

    @property
    def resolved(self) -> bool:
        return self.status in {ACCEPTED, REJECTED, INVALID}

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "success": self.success,
            "score": self.score,
            "harmful": self.harmful,
            "execution_count": self.execution_count,
            "grounding": self.grounding,
            "detail": self.detail,
        }


def invalid_outcome(detail: str) -> ExecutionOutcome:
    """A statically screened record is never executed (Section 3.3.3)."""

    return ExecutionOutcome(
        status=INVALID,
        success=False,
        score=None,
        harmful=True,
        execution_count=0.0,
        grounding="static-validator",
        detail=detail,
    )


def unresolved_outcome(detail: str, *, execution_count: float = 0.0) -> ExecutionOutcome:
    return ExecutionOutcome(
        status=UNRESOLVED,
        success=None,
        score=None,
        harmful=False,
        execution_count=execution_count,
        grounding="unresolved",
        detail=detail,
    )


class RepairEnvironment(Protocol):
    """Execute one sampled record and return a benchmark-evaluator verdict."""

    name: str

    def execute(
        self,
        incident: EnvironmentIncident,
        repair: RepairRecord,
    ) -> ExecutionOutcome: ...


# ---------------------------------------------------------------------------
# Backend: official benchmark evaluator on the materialized splice
# ---------------------------------------------------------------------------


class EvaluatorEnvironment:
    """Score the splice's final state with the benchmark's official evaluator.

    The continuation is bounded, so what is executed is the substitution of
    ``e`` into the retained prefix followed by the evaluator's decision on the
    resulting final state.  Tool calls proposed by the continuation are not
    dispatched, so a continuation whose outcome depends on an undispatched tool
    call yields no final state and is reported unresolved rather than scored as
    a failed repair.
    """

    name = "evaluator"

    def __init__(
        self,
        *,
        run_root: Path | None = None,
        data_dir: Path | None = None,
        benchmark_splits: Mapping[str, str] | None = None,
        evaluation_config: Mapping[str, Any] | None = None,
        swebench_cache_dir: Path | None = None,
    ) -> None:
        self._run_root = run_root
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._resolver = (
            BenchmarkSampleResolver(
                data_dir=Path(data_dir), benchmark_splits=benchmark_splits
            )
            if data_dir is not None
            else None
        )
        self._evaluation_config = dict(evaluation_config or {})
        self._swebench_cache_dir = (
            Path(swebench_cache_dir).resolve() if swebench_cache_dir is not None else None
        )

    def _root(self) -> Path:
        if self._run_root is not None:
            self._run_root.mkdir(parents=True, exist_ok=True)
            return self._run_root
        if self._temporary is None:
            self._temporary = tempfile.TemporaryDirectory(prefix="aria-stage3-")
        return Path(self._temporary.name)

    def execute(
        self, incident: EnvironmentIncident, repair: RepairRecord
    ) -> ExecutionOutcome:
        from common.evaluation import evaluate_sample

        sample, resolution = self._evaluator_sample(incident)
        if sample is None:
            return unresolved_outcome(resolution)
        if incident.benchmark_id == "swe-bench-lite":
            return self._execute_swe_bench(incident, repair, sample)
        spliced = _splice(incident, repair)
        final_output = _final_output(spliced)
        if not final_output.strip():
            return unresolved_outcome("continuation determines no final state")
        run_root = self._execution_root(incident, repair)
        workspace = run_root / "workspace"
        if incident.benchmark_id == "humaneval":
            self._write_humaneval_candidate(
                workspace,
                sample,
                repair,
            )
        result = {
            "status": "completed",
            "final_output": final_output,
            "workspace_path": str(workspace),
            "transcript": [
                {"content": event.content, "tool_calls": list(event.tool_calls)}
                for event in spliced
            ],
        }
        try:
            evaluation = evaluate_sample(
                benchmark_slug=incident.benchmark_id,
                sample=sample,
                result=result,
                run_root=run_root,
                evaluation_config=self._evaluation_config,
            )
        except NotImplementedError as exc:
            return unresolved_outcome(f"evaluator unavailable: {exc}")
        except Exception as exc:  # noqa: BLE001 - evaluator boundary
            return unresolved_outcome(f"{type(exc).__name__}: {exc}")
        correct = evaluation.get("correct")
        from common.artifacts import atomic_write_json

        atomic_write_json(run_root / "evaluation.json", evaluation)
        if not isinstance(correct, bool):
            return unresolved_outcome(
                str(evaluation.get("reason") or evaluation.get("method") or "unresolved"),
                execution_count=1.0,
            )
        score = _numeric(evaluation.get("score"))
        return ExecutionOutcome(
            status=ACCEPTED if correct else REJECTED,
            success=correct,
            score=score,
            harmful=_harmful(incident, success=correct, score=score),
            execution_count=1.0,
            grounding="benchmark-evaluator",
            detail=str(evaluation.get("method") or ""),
        )

    @staticmethod
    def _write_humaneval_candidate(
        workspace: Path,
        sample: Mapping[str, Any],
        repair: RepairRecord,
    ) -> None:
        """Materialize a generated code continuation for the unit-test evaluator."""

        workspace.mkdir(parents=True, exist_ok=True)
        entry_point = str(sample.get("entry_point") or "")
        contents = [
            str(event.content or "")
            for event in repair.replacement_events
            if str(event.content or "").strip()
        ]
        candidate = "\n\n".join(contents).strip("\n")
        # Files bypass the evaluator's response-text extractor. Never write
        # Markdown fences into solution.py, even when the function is complete.
        if "```" in candidate:
            from common.evaluation import extract_python_code

            candidate = extract_python_code(candidate) or candidate
        if entry_point and f"def {entry_point}" not in candidate:
            body = candidate
            nonempty = [line for line in body.splitlines() if line.strip()]
            min_indent = min(
                (len(line) - len(line.lstrip()) for line in nonempty),
                default=0,
            )
            if min_indent < 4:
                body = textwrap.indent(body, "    ")
            prompt = str(sample.get("prompt") or "").rstrip()
            candidate = f"{prompt}\n{body}" if prompt else body
        (workspace / "solution.py").write_text(candidate + "\n", encoding="utf-8")

    def score_priority(self, incident: EnvironmentIncident) -> int:
        """Rank incidents that this backend can objectively score.

        Text/code evaluators consume the materialized continuation directly.
        SWE-Bench is deliberately excluded until the continuation has been
        applied to a repository workspace and produces a Git patch.
        """

        sample, _ = self._evaluator_sample(incident)
        return 1 if sample is not None else 100

    def capability(self, incident: EnvironmentIncident) -> str:
        sample, detail = self._evaluator_sample(incident)
        if sample is None:
            return f"evaluator-unavailable:{detail}"
        return f"benchmark-evaluator:{incident.benchmark_id}"

    def _execute_swe_bench(
        self,
        incident: EnvironmentIncident,
        repair: RepairRecord,
        sample: Mapping[str, Any],
    ) -> ExecutionOutcome:
        """Materialize and officially evaluate a repository patch."""

        from baselines.replay.swebench import materialize_swe_bench_patch
        from common.evaluation import evaluate_sample

        run_root = self._execution_root(incident, repair)
        workspace = run_root / "workspace"
        try:
            materialization = materialize_swe_bench_patch(
                workspace=workspace,
                sample=sample,
                replacement_events=[
                    event.to_dict() for event in repair.replacement_events
                ],
                run_root=run_root,
                cache_root=self._swebench_cache_dir,
            )
        except Exception as exc:  # noqa: BLE001 - evaluator boundary
            return unresolved_outcome(
                f"SWE-Bench patch materialization failed: {type(exc).__name__}: {exc}"
            )
        if materialization.get("status") == "invalid":
            return unresolved_outcome(
                str(materialization.get("reason") or "SWE-Bench patch was not materialized")
            )

        result = {
            "status": "completed",
            "workspace_path": str(workspace),
            "final_output": _final_output(_splice(incident, repair)),
            "model": "baseline-patch-replay",
            "reasoning_effort": None,
        }
        try:
            evaluation = evaluate_sample(
                benchmark_slug="swe-bench-lite",
                sample=dict(sample),
                result=result,
                run_root=run_root,
                evaluation_config=self._evaluation_config,
            )
        except Exception as exc:  # noqa: BLE001 - evaluator boundary
            return unresolved_outcome(
                f"SWE-Bench official evaluator failed: {type(exc).__name__}: {exc}",
                execution_count=1.0,
            )
        correct = evaluation.get("correct")
        if not isinstance(correct, bool):
            return unresolved_outcome(
                str(evaluation.get("reason") or "SWE-Bench official evaluator unresolved"),
                execution_count=1.0,
            )
        score = _numeric(evaluation.get("score"))
        return ExecutionOutcome(
            status=ACCEPTED if correct else REJECTED,
            success=correct,
            score=score,
            harmful=_harmful(incident, success=correct, score=score),
            execution_count=1.0,
            grounding="swebench-official",
            detail=str(evaluation.get("reason") or evaluation.get("method") or ""),
        )

    def _evaluator_sample(
        self, incident: EnvironmentIncident
    ) -> tuple[dict[str, Any] | None, str]:
        sample = dict(incident.evaluator_sample)
        if self._resolver is not None:
            resolved, detail = self._resolver.resolve(incident)
            if resolved is not None:
                # The benchmark cache is authoritative for hidden evaluator
                # fields. The environment copy supplies only harmless display
                # metadata such as the model-visible task text.
                sample = {**sample, **resolved}
                return sample, detail
            if not _sample_can_be_evaluated(incident.benchmark_id, sample):
                return None, detail
        if _sample_can_be_evaluated(incident.benchmark_id, sample):
            return sample, "environment-inline"
        return None, "benchmark sample could not be restored from the local cache"

    def _execution_root(
        self, incident: EnvironmentIncident, repair: RepairRecord
    ) -> Path:
        payload = json.dumps(repair.to_dict(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode()).hexdigest()[:16]
        root = self._root() / _safe_path_part(incident.incident_id) / digest
        root.mkdir(parents=True, exist_ok=True)
        return root


class BenchmarkSampleResolver:
    """Restore evaluator-only benchmark fields for a Stage 3 incident.

    ARIA's public training rows are answer blind, so the Stage 3 environment
    names a benchmark task without carrying its gold answer or tests. The
    resolver joins that identity back to the local benchmark cache. No restored
    field is ever returned to a model prompt.
    """

    def __init__(
        self,
        *,
        data_dir: Path,
        benchmark_splits: Mapping[str, str] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.benchmark_splits = {
            "gaia": "validation",
            **dict(benchmark_splits or {}),
        }
        self._items: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._resolved: dict[tuple[str, str, str], tuple[dict[str, Any] | None, str]] = {}

    def resolve(
        self, incident: EnvironmentIncident
    ) -> tuple[dict[str, Any] | None, str]:
        benchmark = incident.benchmark_id
        split = self.benchmark_splits.get(benchmark, incident.split or "validation")
        cache_key = (benchmark, split, incident.task_id)
        if cache_key in self._resolved:
            return self._resolved[cache_key]
        try:
            items = self._load(benchmark, split)
        except (FileNotFoundError, NotImplementedError, TypeError, ValueError) as exc:
            result = (None, f"{type(exc).__name__}: {exc}")
            self._resolved[cache_key] = result
            return result

        identities = {
            str(value)
            for value in (
                incident.task_id,
                incident.incident_id,
                incident.evaluator_sample.get("sample_id"),
            )
            if value not in (None, "")
        }
        normalized_ids = {_normalize_identifier(value) for value in identities}
        exact = [
            item
            for item in items
            if _normalize_identifier(str(item.get("sample_id") or ""))
            in normalized_ids
        ]
        if len(exact) == 1:
            result = (dict(exact[0]), "benchmark-cache-id")
            self._resolved[cache_key] = result
            return result

        suffix = [
            item
            for item in items
            if any(
                _identifier_suffix_match(
                    identity, str(item.get("sample_id") or "")
                )
                for identity in identities
            )
        ]
        if len(suffix) == 1:
            result = (dict(suffix[0]), "benchmark-cache-suffix")
            self._resolved[cache_key] = result
            return result

        task = _normalize_task_text(
            str(incident.evaluator_sample.get("task") or incident.constraints.get("task_text") or "")
        )
        by_task = [
            item
            for item in items
            if task and _normalize_task_text(str(item.get("task") or "")) == task
        ]
        if len(by_task) == 1:
            result = (dict(by_task[0]), "benchmark-cache-task")
            self._resolved[cache_key] = result
            return result

        detail = (
            "benchmark sample identity is ambiguous"
            if exact or suffix or by_task
            else "benchmark sample identity was not found"
        )
        result = (None, detail)
        self._resolved[cache_key] = result
        return result

    def _load(self, benchmark: str, split: str) -> list[dict[str, Any]]:
        key = (benchmark, split)
        if key not in self._items:
            from common.mast_data import load_benchmark_items

            self._items[key] = load_benchmark_items(
                self.data_dir,
                benchmark,
                source="full",
                split=split,
            )
        return self._items[key]


# ---------------------------------------------------------------------------
# Backend: recorded executions from corpus construction
# ---------------------------------------------------------------------------


class LedgerEnvironment:
    """Return the recorded verdict of the executed splice a sample matches.

    Matching requires the same cut point and a normalized-content overlap above
    ``similarity_threshold``; anything else is unresolved.  The verdict is a
    benchmark-evaluator decision on a real execution, so this backend replays
    Stage 3's reward rather than approximating it, at the cost of covering only
    the splices the corpus actually executed.
    """

    name = "ledger"

    def __init__(self, *, similarity_threshold: float = 0.6) -> None:
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("ledger similarity threshold must be in (0, 1]")
        self.similarity_threshold = similarity_threshold

    def execute(
        self, incident: EnvironmentIncident, repair: RepairRecord
    ) -> ExecutionOutcome:
        signature = _content_signature(
            [event.to_dict() for event in repair.replacement_events]
        )
        best: tuple[float, LedgerEntry] | None = None
        for entry in incident.ledger:
            if entry.cut_event_id != repair.cut_event_id:
                continue
            if entry.success is None:
                continue
            similarity = _jaccard(
                signature, _content_signature(list(entry.replacement_events))
            )
            if best is None or similarity > best[0]:
                best = (similarity, entry)
        if best is None or best[0] < self.similarity_threshold:
            return unresolved_outcome(
                "no recorded execution matches this cut and continuation"
            )
        similarity, entry = best
        return ExecutionOutcome(
            status=ACCEPTED if entry.success else REJECTED,
            success=entry.success,
            score=entry.score,
            harmful=_harmful(incident, success=entry.success, score=entry.score),
            execution_count=1.0,
            grounding="recorded-execution",
            detail=f"{entry.label} similarity={similarity:.3f}",
        )

    @staticmethod
    def score_priority(incident: EnvironmentIncident) -> int:
        return 0 if incident.ledger else 100

    @staticmethod
    def capability(incident: EnvironmentIncident) -> str:
        return "recorded-ledger" if incident.ledger else "ledger-unavailable"


# ---------------------------------------------------------------------------
# Backend: host framework re-execution
# ---------------------------------------------------------------------------


class NativeEnvironment:
    """Re-run the host framework on the splice and score it with the evaluator.

    This is the paper configuration.  It needs the framework checkout and the
    host model endpoint of the source cell, so it is selected by configuration
    rather than by default.
    """

    name = "native"

    def __init__(
        self, *, config: Any, spec: Mapping[str, Any] | None = None,
        run_root: Path | None = None,
    ) -> None:
        from baselines.replay.native import NativeReplayBackend

        self.config = config
        self.spec = dict(spec or {})
        options = dict(self.spec)
        if run_root is not None:
            options.setdefault("output_root", str(run_root))
        options.setdefault("paired_control", True)
        if options["paired_control"] is not True:
            raise ValueError("native training requires fresh paired controls")
        if options["paired_control"]:
            root = options.get("output_root")
            if not root:
                raise ValueError("native training requires an isolated output_root")
            options.setdefault("control_output_root", str(Path(root) / "controls"))
        self._backend = NativeReplayBackend(config=config, options=options)

    @staticmethod
    def _sample(incident: EnvironmentIncident) -> Any:
        from baselines.core.types import BaselineEvent, BaselineSample

        return BaselineSample(
            sample_id=incident.incident_id,
            task_id=incident.task_id,
            task=str(incident.constraints.get("task_text") or ""),
            events=tuple(
                BaselineEvent(
                    event_id=event.event_id,
                    role=event.role,
                    kind=event.kind,
                    content=event.content,
                    tool_calls=tuple(dict(call) for call in event.tool_calls),
                    metadata=dict(event.metadata),
                )
                for event in incident.events
            ),
            constraints=dict(incident.constraints),
            feedback=incident.feedback,
            benchmark=incident.benchmark_id,
            mas=incident.mas_id,
            split=incident.split,
            source_evaluation={
                "success": incident.source_success,
                "score": incident.source_score,
            },
            control_evaluation={
                "success": incident.control_success,
                "score": incident.control_score,
            },
            metadata={
                "stage": "aria-stage3",
                "source_result": {"sample_metadata": dict(incident.evaluator_sample)},
            },
        )

    def prepare_incident(self, incident: EnvironmentIncident) -> EnvironmentIncident:
        sample = self._backend.prepare_sample(self._sample(incident))
        return replace(
            incident, control_success=sample.control_success,
            control_score=sample.control_score,
        )

    def execute(
        self, incident: EnvironmentIncident, repair: RepairRecord
    ) -> ExecutionOutcome:
        try:
            incident = self.prepare_incident(incident)
            # Repeated stochastic draws must not share mutable native workspaces.
            record = self._backend.execute(
                self._sample(incident), repair.to_dict(), label=f"stage3-{uuid4().hex}",
            )
        except Exception as exc:  # noqa: BLE001 - native infrastructure boundary
            return unresolved_outcome(f"native execution failed: {type(exc).__name__}: {exc}")
        success = record.success
        score = _numeric(record.evaluation.get("score"))
        if record.evaluation.get("failure_category") == "method":
            return invalid_outcome(record.error or "native host rejected the intervention")
        if record.status != "completed" or success is None:
            return unresolved_outcome(
                record.error or f"native replay status {record.status}",
                execution_count=float(record.execution_count),
            )
        return ExecutionOutcome(
            status=ACCEPTED if success else REJECTED,
            success=success,
            score=score,
            harmful=_harmful(incident, success=success, score=score),
            execution_count=float(record.execution_count),
            grounding="native-rerun",
            detail=str(record.evaluation.get("method") or ""),
        )


class NativeCellEnvironment:
    """Route native rewards to the source cell's declared host, never the repair model."""

    name = "native_by_cell"

    def __init__(self, spec, *, run_root, config_loader=None):
        self.entries = json.loads(Path(spec["config_manifest"]).read_text())["cells"]
        self.backends = {}
        self.run_root = run_root
        self.load_config = config_loader
        self.seed = spec.get("seed")

    def _backend(self, incident):
        key = incident.cell_key
        if key not in self.entries:
            raise ValueError(f"no declared native source host for {key}")
        if key not in self.backends:
            entry = self.entries[key]
            if entry.get("status") != "ready":
                raise RuntimeError(f"native source host not qualified: {key}: {entry.get('status')}")
            if entry["host_model_id"] != incident.host_model_id:
                raise ValueError("native source host identity mismatch")
            if self.load_config is None:
                raise ValueError("native cell routing requires an application-supplied config loader")
            config_path = Path(entry["config"])
            if entry.get("config_sha256"):
                import hashlib
                if hashlib.sha256(config_path.read_bytes()).hexdigest() != entry["config_sha256"]:
                    raise ValueError("qualified native source config changed")
            config = self.load_config(config_path)
            if entry.get("declared_api"):
                from dataclasses import asdict
                if asdict(config.api) != entry["declared_api"]:
                    raise ValueError("native source API differs from its declared identity")
            options = dict(entry["options"])
            explicit = options.get("host_api")
            if isinstance(explicit, Mapping):
                for field in ("model", "protocol", "provider", "base_url", "reasoning_effort"):
                    if explicit.get(field) != getattr(config.api, field):
                        raise ValueError(f"native host override differs from source config: {field}")
            if self.seed is not None and config.api.protocol == "openai-chat-completions":
                config = replace(config, api=replace(config.api, seed=int(self.seed)))
                if isinstance(explicit, Mapping):
                    options["host_api"] = {**explicit, "seed": int(self.seed)}
            self.backends[key] = NativeEnvironment(config=config, spec=options,
                                                   run_root=self.run_root / key)
        return self.backends[key]

    def prepare_incident(self, incident):
        return self._backend(incident).prepare_incident(incident)

    def execute(self, incident, repair):
        return self._backend(incident).execute(incident, repair)


class ChainEnvironment:
    """Try each backend in order and return the first resolved verdict."""

    def __init__(self, backends: Sequence[Any]) -> None:
        if not backends:
            raise ValueError("environment chain requires at least one backend")
        self.backends = tuple(backends)
        self.name = "+".join(str(getattr(item, "name", "backend")) for item in backends)

    def prepare_incident(self, incident: EnvironmentIncident) -> EnvironmentIncident:
        for backend in self.backends:
            prepare = getattr(backend, "prepare_incident", None)
            if callable(prepare):
                incident = prepare(incident)
        return incident

    def execute(
        self, incident: EnvironmentIncident, repair: RepairRecord
    ) -> ExecutionOutcome:
        last = unresolved_outcome("no backend was consulted")
        for backend in self.backends:
            outcome = backend.execute(incident, repair)
            if outcome.resolved:
                return outcome
            last = outcome
        return last

    def score_priority(self, incident: EnvironmentIncident) -> int:
        return min(
            int(getattr(backend, "score_priority", lambda _: 100)(incident))
            for backend in self.backends
        )

    def capability(self, incident: EnvironmentIncident) -> str:
        return "+".join(
            str(getattr(backend, "capability", lambda _: backend.name)(incident))
            for backend in self.backends
        )


def build_environment(
    spec: Mapping[str, Any] | str | None,
    *,
    config: Any | None = None,
    run_root: Path | None = None,
    config_loader: Any | None = None,
) -> Any:
    """Construct the environment named by a config block or a bare kind."""

    if spec is None:
        return EvaluatorEnvironment(run_root=run_root)
    if isinstance(spec, str):
        spec = {"kind": spec}
    kind = str(spec.get("kind") or "evaluator")
    if kind == "native_by_cell":
        return NativeCellEnvironment(spec, run_root=run_root, config_loader=config_loader)
    if kind == "evaluator":
        options = spec.get("options")
        options = dict(options) if isinstance(options, Mapping) else {}
        data_dir = spec.get("data_dir", options.get("data_dir"))
        benchmark_splits = spec.get(
            "benchmark_splits", options.get("benchmark_splits")
        )
        evaluation_config = spec.get(
            "evaluation", options.get("evaluation")
        )
        swebench_cache_dir = spec.get(
            "swebench_cache_dir", options.get("swebench_cache_dir")
        )
        return EvaluatorEnvironment(
            run_root=run_root,
            data_dir=Path(str(data_dir)) if data_dir else None,
            benchmark_splits=(
                dict(benchmark_splits)
                if isinstance(benchmark_splits, Mapping)
                else None
            ),
            evaluation_config=(
                dict(evaluation_config)
                if isinstance(evaluation_config, Mapping)
                else None
            ),
            swebench_cache_dir=(
                Path(str(swebench_cache_dir)) if swebench_cache_dir else None
            ),
        )
    if kind == "ledger":
        return LedgerEnvironment(
            similarity_threshold=float(spec.get("similarity_threshold", 0.6))
        )
    if kind == "native":
        if config is None:
            raise ValueError("the native Stage 3 environment requires an experiment config")
        return NativeEnvironment(config=config, spec=spec.get("options") or {}, run_root=run_root)
    if kind == "chain":
        members = spec.get("backends")
        if not isinstance(members, list) or not members:
            raise ValueError("chain environment requires a non-empty backends list")
        return ChainEnvironment(
            [
                build_environment(member, config=config, run_root=run_root, config_loader=config_loader)
                for member in members
            ]
        )
    raise ValueError(f"unsupported Stage 3 environment kind: {kind}")


def load_environment_incidents(path: Path) -> list[EnvironmentIncident]:
    incidents: list[EnvironmentIncident] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise TypeError(f"{path}:{number} must be a JSON object")
            incidents.append(EnvironmentIncident.from_dict(raw))
    if not incidents:
        raise ValueError(f"Stage 3 environment corpus is empty: {path}")
    return incidents


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _harmful(
    incident: EnvironmentIncident, *, success: bool | None, score: float | None
) -> bool:
    """Apply the harmful-intervention condition of Section 4.1.

    An intervention is harmful when it lowers a graded benchmark score relative
    to no intervention, or when it breaks a run the matched control resolved as
    successful.  Structural invalidity is charged by the static screen instead,
    which is why it does not reappear here.
    """

    control_score = incident.matched_control_score
    if control_score is not None and score is not None and score < control_score:
        return True
    return incident.matched_control_success is True and success is False


def _splice(
    incident: EnvironmentIncident, repair: RepairRecord
) -> tuple[TrajectoryEvent, ...]:
    cut_index = repair.cut_index(incident.events)
    return (*incident.events[: cut_index + 1], *repair.replacement_events)


_FINAL_MARKERS = ("terminate", "final answer", "final_answer", "answer:")


def _final_output(events: Sequence[TrajectoryEvent]) -> str:
    """Return the text the evaluator reads as the run's final state.

    The evaluator extracts an answer from the run's output text, so the final
    state is taken from the trailing content-bearing events of the splice rather
    than from the last event alone, which may be an empty control message.
    """

    parts: list[str] = []
    for event in reversed(events):
        if event.kind == "tool_result":
            continue
        content = (event.content or "").strip()
        if content:
            parts.append(content)
        if len(parts) >= 3:
            break
    return "\n".join(reversed(parts))


def _sample_can_be_evaluated(
    benchmark: str, sample: Mapping[str, Any]
) -> bool:
    if benchmark in {"gsm-plus", "mmlu", "gaia", "olympiadbench"}:
        return bool(str(sample.get("reference_answer") or "").strip())
    if benchmark == "humaneval":
        return all(sample.get(key) not in (None, "") for key in ("test", "entry_point"))
    if benchmark == "swe-bench-lite":
        return all(
            sample.get(key) not in (None, "")
            for key in ("instance_id", "repo", "base_commit")
        )
    return False


def _normalize_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value).strip().lower()).strip("-")


def _identifier_suffix_match(container: str, candidate: str) -> bool:
    left = _normalize_identifier(container)
    right = _normalize_identifier(candidate)
    return bool(right) and (left == right or left.endswith(f"-{right}"))


def _normalize_task_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def _safe_path_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return cleaned[:160] or "incident"


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _content_signature(events: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    parts: list[str] = []
    for event in events:
        content = event.get("content")
        if isinstance(content, str):
            parts.append(content)
        calls = event.get("tool_calls")
        if isinstance(calls, list):
            parts.append(json.dumps(calls, ensure_ascii=True, sort_keys=True))
    return frozenset(_TOKEN_PATTERN.findall(" ".join(parts).lower()))


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _index_of(events: Sequence[TrajectoryEvent], event_id: Any) -> int:
    if not isinstance(event_id, str):
        return -1
    for index, event in enumerate(events):
        if event.event_id == event_id:
            return index
    return -1


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _boolean(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None
