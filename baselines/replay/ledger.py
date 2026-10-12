"""Score an intervention against the executions the corpus already ran.

Success Rate and Harmful Intervention Rate only mean something when the
repaired trajectory is executed and scored by the benchmark's official
evaluator.  :mod:`baselines.replay.native` does that live.  This backend
instead replays the verdicts recorded during corpus construction, using the
Stage 3 environment of :mod:`aria.post_training.environment`, so an evaluation
can be run on a machine without the framework checkouts or the host-model
endpoint at the cost of covering only splices the corpus executed.

Nothing here decides correctness on its own: every verdict it returns was
produced by a benchmark evaluator on a real run.  An intervention that matches
no recorded execution is returned as ``unresolved`` and is counted separately
from a task-incorrect repair, exactly as an unresolved live evaluation would be.
"""

from __future__ import annotations

import copy
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from aria.post_training.environment import (
    ACCEPTED,
    INVALID,
    UNRESOLVED,
    EnvironmentIncident,
    build_environment,
    load_environment_incidents,
)
from aria.post_training.grpo import GrpoConfig, screen_record
from aria.records import RepairRecord, TrajectoryEvent
from baselines.core.types import BaselineSample, ExecutionRecord
from baselines.replay.materialize import replacement_events, resolve_cut_index

__all__ = ["LedgerReplayBackend", "build_ledger_replay_backend"]


class LedgerReplayBackend:
    """Restore the prefix, substitute the record, and read back the verdict."""

    def __init__(
        self,
        *,
        environment_path: Path | None = None,
        environment: Mapping[str, Any] | None = None,
        run_root: Path | None = None,
    ) -> None:
        self.environment_path = Path(environment_path) if environment_path is not None else None
        self.incidents: dict[str, EnvironmentIncident] = {}
        for incident in (load_environment_incidents(self.environment_path) if self.environment_path else []):
            self.incidents[incident.incident_id] = incident
            self.incidents.setdefault(incident.task_id, incident)
        self.environment = build_environment(
            dict(environment or {"kind": "ledger"}), run_root=run_root
        )
        self.config = GrpoConfig()

    def execute(
        self,
        sample: BaselineSample,
        intervention: Mapping[str, Any] | None,
        *,
        label: str,
    ) -> ExecutionRecord:
        incident = self._incident(sample)
        if incident is None:
            return _record(
                label,
                "invalid",
                error=f"no recorded execution environment for sample {sample.sample_id}",
                execution_count=0.0,
            )
        if intervention is None:
            return _record(
                label,
                "completed",
                evaluation={
                    "success": incident.matched_control_success,
                    "correct": incident.matched_control_success,
                    "score": incident.matched_control_score,
                    "method": "recorded-no-intervention-control",
                },
                execution_count=1.0,
                execution_mode="recorded_control",
            )
        try:
            cut_index = resolve_cut_index(sample, intervention)
            normalized_replacements = replacement_events(sample, intervention)
            normalized_intervention = {
                **dict(intervention),
                "diagnosis": str(
                    intervention.get("diagnosis")
                    or intervention.get("reason")
                    or intervention.get("action")
                    or "runtime intervention"
                ),
                "cut_event_id": (
                    sample.events[cut_index].event_id if cut_index >= 0 else None
                ),
                "replacement_events": normalized_replacements,
            }
            record = RepairRecord.from_dict(
                {
                    "diagnosis": normalized_intervention["diagnosis"],
                    "cut_event_id": normalized_intervention.get("cut_event_id"),
                    "replacement_events": normalized_replacements,
                },
                events=incident.events,
            )
        except (TypeError, ValueError) as exc:
            return _record(
                label,
                "completed",
                evaluation={
                    "success": False,
                    "correct": False,
                    "score": 0.0,
                    "method": "static-validator",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "constraint_violation": True,
                },
                execution_count=0.0,
                execution_mode="not_executed",
            )
        # The certified cut is hidden Stage-3 supervision. It must not reject
        # a baseline repair before the ledger/evaluator gets a chance to score it.
        screen = screen_record(
            record,
            incident,
            self.config,
            enforce_certified_cut=False,
        )
        if not screen.valid:
            # A record that fails the static screen is never executed and is
            # charged as a harmful intervention, which is what HIR counts.
            return _record(
                label,
                "completed",
                evaluation={
                    "success": False,
                    "correct": False,
                    "score": 0.0,
                    "method": "static-validator",
                    "reason": screen.reason,
                    "constraint_violation": True,
                },
                execution_count=0.0,
                execution_mode="not_executed",
            )
        outcome = self.environment.execute(incident, record)
        if outcome.status == UNRESOLVED:
            return _record(
                label,
                "invalid",
                error=outcome.detail or "no recorded execution matches this record",
                execution_count=float(outcome.execution_count),
                execution_mode="unresolved",
            )
        return _record(
            label,
            "completed",
            evaluation={
                "success": outcome.success,
                "correct": outcome.success,
                "score": outcome.score,
                "method": outcome.grounding,
                "reason": outcome.detail,
                "constraint_violation": outcome.status == INVALID,
                "accepted": outcome.status == ACCEPTED,
            },
            execution_count=float(outcome.execution_count),
            execution_mode=outcome.grounding,
        )

    def _incident(self, sample: BaselineSample) -> EnvironmentIncident | None:
        for key in (sample.sample_id, sample.task_id):
            incident = self.incidents.get(str(key))
            if incident is not None:
                return incident
        # ARIA-Full-Processed is a prompt projection and may contain test
        # incidents that were not present in the older Stage-3 ledger export.
        # Construct an answer-blind incident from the sample so the chain can
        # fall through to the local official evaluator instead of turning a
        # missing historical ledger row into an infrastructure invalid.
        if not sample.events or not sample.benchmark:
            return None
        source_score = _numeric(sample.source_evaluation.get("score"))
        control_score = sample.control_score
        return EnvironmentIncident(
            incident_id=sample.sample_id,
            task_id=sample.task_id,
            split=str(sample.split or "test"),
            mas_id=str(sample.mas or "unknown"),
            benchmark_id=str(sample.benchmark),
            host_model_id=str(sample.metadata.get("host_model") or "unknown"),
            family_id=str(sample.metadata.get("family_id") or "unknown"),
            events=tuple(
                TrajectoryEvent.from_dict(event.to_dict(), name="baseline event")
                for event in sample.events
            ),
            feedback=sample.feedback,
            constraints=dict(sample.constraints),
            certified_cut_event_id=None,
            certified_cut_index=-1,
            first_bad_event_ids=(),
            verified_repair=None,
            attribution=None,
            localization=None,
            evaluator_sample=(
                dict(sample.metadata.get("evaluator_sample") or {})
                if isinstance(sample.metadata.get("evaluator_sample"), Mapping)
                else {}
            ),
            control_success=sample.control_success,
            control_score=control_score,
            source_success=sample.source_success,
            source_score=source_score,
            ledger=(),
            verified_successful=False,
        )


def _record(
    label: str,
    status: str,
    *,
    evaluation: Mapping[str, Any] | None = None,
    error: str | None = None,
    execution_count: float = 1.0,
    execution_mode: str = "recorded_execution",
) -> ExecutionRecord:
    return ExecutionRecord(
        label=label,
        status=status,
        evaluation=dict(evaluation or {}),
        execution_mode=execution_mode,
        execution_count=execution_count,
        error=error,
    )


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def build_ledger_replay_backend(
    options: Mapping[str, Any],
    spec: Mapping[str, Any],
    config: Any,
) -> LedgerReplayBackend:
    environment_path = options.get("environment_path")
    if not environment_path:
        raise ValueError(
            "replay_backend type ledger requires options.environment_path"
        )
    root = (
        config.config_path.parent.resolve()
        if getattr(config, "config_path", None) is not None
        else Path.cwd()
    )
    resolved = Path(str(environment_path)).expanduser()
    if not resolved.is_absolute():
        resolved = (root / resolved).resolve()
    environment = _resolve_environment_paths(
        dict(options.get("environment") or {"kind": "ledger"}),
        base_dir=root,
    )
    return LedgerReplayBackend(
        environment_path=resolved,
        environment=environment,
        run_root=Path(str(spec.get("output_dir") or "outputs/baselines")) / "replay",
    )


def _resolve_environment_paths(value: Mapping[str, Any], *, base_dir: Path) -> dict[str, Any]:
    """Resolve nested evaluator paths relative to the YAML config directory."""

    result = copy.deepcopy(dict(value))
    for key in ("data_dir", "run_root"):
        raw = result.get(key)
        if isinstance(raw, str) and raw.strip():
            path = Path(raw).expanduser()
            result[key] = str(path if path.is_absolute() else (base_dir / path).resolve())
    raw_cache = result.get("swebench_cache_dir")
    if isinstance(raw_cache, str) and raw_cache.strip():
        path = Path(raw_cache).expanduser()
        result["swebench_cache_dir"] = str(
            path if path.is_absolute() else (base_dir / path).resolve()
        )
    members = result.get("backends")
    if isinstance(members, list):
        result["backends"] = [
            _resolve_environment_paths(item, base_dir=base_dir)
            if isinstance(item, Mapping)
            else item
            for item in members
        ]
    return result
