"""Model-agnostic ARIA post-training data preparation and orchestration.

This module owns the paper's SFT -> DPO -> transfer diagnosis ordering.  The
three objectives live in separate :mod:`aria.post_training` modules and are
independent ablation axes.  This orchestrator intentionally does not choose a
base model or trainer:
those choices are represented by :class:`TrainingBackend`, while ARIA keeps
the repair semantics, answer-blindness, validation, provenance, and resume
contract stable across training stacks.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import subprocess
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from aria.post_training import (
    build_repair_preference_row,
    build_sft_rows,
    build_transfer_diagnosis_rows,
    select_diagnosis_choice,
)
from aria.post_training.transfer_diagnosis import DiagnosisChoice
from aria.gate_preflight import gate_preflight
from aria.prompts import ordered_training_output
from aria.records import (
    ANSWER_LABEL_FIELDS,
    FailureLocalization,
    IncidentRecord,
    RepairAttribution,
    RepairRecord,
)

LEGACY_TRAINING_READY_SCHEMA_VERSION = "aria-training-ready-v1"
TRAINING_READY_SCHEMA_VERSION = "aria-training-ready-v3"
SFT_TASKS = (
    "deployment_gate",
    "failure_localization",
    "repair_attribution",
    "structured_repair",
)
PREFERENCE_TASKS = ("repair_preference",)

__all__ = [
    "ArtifactOnlyTrainingBackend",
    "CommandTrainingBackend",
    "DiagnosisChoice",
    "PostTrainingPlan",
    "PreparedCorpora",
    "TrainingBackend",
    "TrainingRun",
    "TrainingStage",
    "load_verified_incidents",
    "gate_preflight",
    "prepare_corpora",
    "prepare_training_ready_corpora",
    "run_post_training",
    "run_post_training_from_spec",
    "select_diagnosis_choice",
    "validate_training_ready_task_row",
]


class TrainingStage(str, Enum):
    """The five updates of Table 1, in the order the paper applies them."""

    SFT = "sft"
    """Stage 1: structured repair SFT with the prefix-level gate."""

    DPO = "dpo"
    """Stage 2a: the verified repair ranked above its failed continuation."""

    TRANSFER_DIAGNOSIS = "transfer_diagnosis"
    """Stage 2b: frozen-scorer sibling uplift, then internalization."""

    GRPO_REPAIR = "grpo_repair"
    """Stage 3a: transfer-grounded credit assignment on policy samples."""

    GRPO_GATE = "grpo_gate"
    """Stage 3b: the inspection decision under matched-seed outcomes."""


TRAINING_STAGE_ORDER = (
    TrainingStage.SFT,
    TrainingStage.DPO,
    TrainingStage.TRANSFER_DIAGNOSIS,
    TrainingStage.GRPO_REPAIR,
    TrainingStage.GRPO_GATE,
)

STAGE_GROUPS = {
    "stage1": (TrainingStage.SFT,),
    "stage2": (TrainingStage.DPO, TrainingStage.TRANSFER_DIAGNOSIS),
    "stage3": (TrainingStage.GRPO_REPAIR, TrainingStage.GRPO_GATE),
}

# The RQ3 ablation names, so a config can request a variant by the name the
# table uses instead of restating a stage list that has to stay in sync.
ABLATION_VARIANTS = {
    "stage1_only": (TrainingStage.SFT,),
    "stage1_2": (
        TrainingStage.SFT,
        TrainingStage.DPO,
        TrainingStage.TRANSFER_DIAGNOSIS,
    ),
    "stage1_3": (
        TrainingStage.SFT,
        TrainingStage.GRPO_REPAIR,
        TrainingStage.GRPO_GATE,
    ),
    "full": TRAINING_STAGE_ORDER,
}


@dataclass(frozen=True)
class PostTrainingPlan:
    """Inputs and acceptance thresholds independent of a model family."""

    base_model: str
    source_path: Path
    output_dir: Path
    feedback_dropout_probability: float = 0.25
    diagnosis_uplift_threshold: float = 0.0
    random_seed: int = 0
    resume: bool = True
    stages: tuple[TrainingStage, ...] = TRAINING_STAGE_ORDER
    environment_path: Path | None = None
    """The Stage 3 environment corpus; required whenever a GRPO stage runs."""

    environment: Mapping[str, Any] | None = None
    """Which execution backend scores Stage 3 samples (see aria.post_training.environment)."""

    gate_validation_path: Path | None = None
    """Optional held-out gate rows used by the pre-training support check."""

    gate_min_positive_rate: float = 0.20
    gate_max_positive_rate: float = 0.80
    repair_conditioning: str = "incident"

    def _starts_from_stage1(self) -> bool:
        """Whether ``base_model`` is a checkpoint an ARIA Stage 1 produced."""

        candidate = Path(self.base_model)
        if not candidate.is_dir():
            return False
        for report_path in (
            candidate / "training_report.json",
            candidate.parent / "training_report.json",
        ):
            if not report_path.is_file():
                continue
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(report, Mapping) and report.get("stage") in {
                stage.value for stage in TRAINING_STAGE_ORDER
            }:
                return True
        return False

    def validate(self) -> None:
        if self.repair_conditioning not in {"incident", "cascade"}:
            raise ValueError("repair_conditioning must be incident or cascade")
        if not self.base_model.strip():
            raise ValueError("base_model must be non-empty")
        if not self.source_path.exists():
            raise FileNotFoundError(
                f"ARIA training source does not exist: {self.source_path}"
            )
        if self.source_path.is_dir():
            filenames: list[str] = []
            if TrainingStage.SFT in self.stages:
                filenames.append("sft.jsonl")
            if TrainingStage.DPO in self.stages:
                filenames.append("dpo.jsonl")
            if TrainingStage.TRANSFER_DIAGNOSIS in self.stages:
                transfer_paths = (
                    self.source_path / "transfer_diagnosis_sft.jsonl",
                    self.source_path / "transfer_diagnosis_dpo.jsonl",
                )
                if transfer_paths[0].is_file() != transfer_paths[1].is_file():
                    raise FileNotFoundError(
                        "ARIA transfer diagnosis requires both "
                        "transfer_diagnosis_sft.jsonl and "
                        "transfer_diagnosis_dpo.jsonl"
                    )
                if not transfer_paths[0].is_file():
                    # The trainer mines sibling-uplift pairs online from the
                    # current causal repair data when calibrated pairs have not
                    # yet been materialized by a larger data pipeline.
                    filenames.extend(["sft.jsonl", "dpo.jsonl"])
            for filename in filenames:
                path = self.source_path / filename
                if not path.is_file():
                    raise FileNotFoundError(
                        f"ARIA training-ready source is missing {filename}: {path}"
                    )
            _validate_corpus_manifest(self.source_path / "manifest.json")
        if any(stage in self.stages for stage in STAGE_GROUPS["stage3"]):
            if self.environment_path is None:
                raise ValueError(
                    "ARIA Stage 3 requires environment_path: the execution "
                    "environment retained from corpus construction"
                )
            if not self.environment_path.is_file():
                raise FileNotFoundError(
                    f"ARIA Stage 3 environment corpus is missing: {self.environment_path}"
                )
        if not 0.0 <= self.feedback_dropout_probability <= 1.0:
            raise ValueError("feedback_dropout_probability must be between 0 and 1")
        if self.gate_validation_path is not None:
            if not self.gate_validation_path.is_file():
                raise FileNotFoundError(
                    "ARIA gate validation corpus is missing: "
                    f"{self.gate_validation_path}"
                )
            gate_preflight(
                self.source_path / "sft.jsonl",
                self.gate_validation_path,
                min_positive_rate=self.gate_min_positive_rate,
                max_positive_rate=self.gate_max_positive_rate,
            )
        if not self.stages:
            raise ValueError("ARIA post-training must enable at least one stage")
        if len(set(self.stages)) != len(self.stages):
            raise ValueError("ARIA post-training stages must not contain duplicates")
        canonical = tuple(
            stage for stage in TRAINING_STAGE_ORDER if stage in self.stages
        )
        if self.stages != canonical:
            raise ValueError(
                "ARIA post-training stages must follow the order "
                + ", ".join(stage.value for stage in TRAINING_STAGE_ORDER)
            )
        if TrainingStage.SFT not in self.stages and not self._starts_from_stage1():
            # Section 5.3: a variant that omits Stage 1 is not defined, since
            # the output schema and the gate are introduced there.  A run that
            # *starts from* a Stage 1 checkpoint has not omitted it, so the
            # requirement is checked against the base model rather than assumed
            # from the stage list; this is what lets an ablation reuse the
            # identical Stage 1 of another run instead of recomputing it.
            raise ValueError(
                "ARIA post-training must include Stage 1 or start from an ARIA "
                "checkpoint that ran it; the record schema and the gate are "
                "introduced there"
            )
        if (
            TrainingStage.GRPO_GATE in self.stages
            and TrainingStage.GRPO_REPAIR not in self.stages
            and not self._starts_from_verified_repair_policy()
        ):
            # Section 3.3.3: a gate optimized against a weak repair policy
            # converges to never intervening, so 3b follows 3a.
            raise ValueError(
                "ARIA Stage 3b requires Stage 3a; the gate is optimized after "
                "the repair policy, not jointly with it"
            )

    def _starts_from_verified_repair_policy(self) -> bool:
        checkpoint = Path(self.base_model)
        report_path = checkpoint.parent / "training_report.json"
        if not checkpoint.is_dir() or not report_path.is_file():
            return False
        try:
            report = json.loads(report_path.read_text())
        except (OSError, ValueError):
            return False
        return (report.get("materialized") is True
                and report.get("stage") == "grpo_repair"
                and report.get("optimizer_steps", 0) > 0
                and any(checkpoint.glob("*.safetensors")))


@dataclass(frozen=True)
class TrainingRun:
    stage: TrainingStage
    base_model: str
    dataset_paths: tuple[Path, ...]
    output_dir: Path
    reference_model: Path | None
    objectives: tuple[str, ...]
    is_final_stage: bool
    environment_spec: Mapping[str, Any] | None = None
    """Stage 3 only: which execution backend scores the policy's samples."""


class TrainingBackend(Protocol):
    """One explicit weight-update implementation for a chosen model stack."""

    materializes_weights: bool

    def train(self, run: TrainingRun) -> Path:
        """Train one stage and return a materialized checkpoint directory."""


class ArtifactOnlyTrainingBackend:
    """Creates auditable stage artifacts without claiming to update weights.

    It is useful for data QA and CI.  Production configs must use an actual
    backend such as ``CommandTrainingBackend`` or a project-specific adapter.
    """

    materializes_weights = False

    def train(self, run: TrainingRun) -> Path:
        run.output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(
            run.output_dir / "checkpoint.json",
            {
                "materialized": False,
                "reason": "artifact-only backend does not update model weights",
                "stage": run.stage.value,
                "base_model": run.base_model,
                "datasets": [str(path) for path in run.dataset_paths],
                "reference_model": str(run.reference_model)
                if run.reference_model
                else None,
                "objectives": list(run.objectives),
                "is_final_stage": run.is_final_stage,
            },
        )
        return run.output_dir


@dataclass(frozen=True)
class CommandTrainingBackend:
    """Invoke any external trainer without exposing shell interpolation."""

    command: tuple[str, ...]
    materializes_weights = True

    def train(self, run: TrainingRun) -> Path:
        if not self.command:
            raise ValueError("command backend requires a non-empty command")
        run.output_dir.mkdir(parents=True, exist_ok=True)
        values = {
            "stage": run.stage.value,
            "base_model": run.base_model,
            "dataset": str(run.dataset_paths[0]),
            "datasets": json.dumps([str(path) for path in run.dataset_paths]),
            "output_dir": str(run.output_dir),
            "reference_model": str(run.reference_model or ""),
            "objectives": json.dumps(list(run.objectives)),
            "is_final_stage": "true" if run.is_final_stage else "false",
            "environment_spec": json.dumps(
                dict(run.environment_spec) if run.environment_spec else {},
                sort_keys=True,
            ),
        }
        command = [part.format_map(values) for part in self.command]
        log_path = run.output_dir / "trainer.log"
        output_tail: deque[str] = deque(maxlen=200)
        with log_path.open("w", encoding="utf-8") as log_handle:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log_handle.write(line)
                log_handle.flush()
                output_tail.append(line)
            returncode = process.wait()
        _atomic_write_json(
            run.output_dir / "trainer_process.json",
            {
                "command": command,
                "returncode": returncode,
                "log_path": str(log_path),
                "output_tail": "".join(output_tail)[-4000:],
            },
        )
        if returncode != 0:
            raise RuntimeError(
                f"ARIA {run.stage.value} trainer failed with exit code {returncode}"
            )
        checkpoint = run.output_dir / "checkpoint"
        if not checkpoint.is_dir():
            raise FileNotFoundError(
                f"ARIA {run.stage.value} trainer succeeded but did not create {checkpoint}"
            )
        return checkpoint


@dataclass(frozen=True)
class PreparedCorpora:
    sft_path: Path
    repair_dpo_path: Path
    counts: Mapping[str, int]
    transfer_diagnosis_sft_path: Path | None = None
    transfer_diagnosis_dpo_path: Path | None = None
    transfer_diagnosis_mode: str | None = None
    environment_path: Path | None = None


def run_post_training(
    plan: PostTrainingPlan, backend: TrainingBackend
) -> dict[str, Any]:
    """Prepare corpora, run selected stages, and persist a resumable manifest."""

    plan.validate()
    gate_preflight_report = (
        gate_preflight(
            plan.source_path / "sft.jsonl",
            plan.gate_validation_path,
            min_positive_rate=plan.gate_min_positive_rate,
            max_positive_rate=plan.gate_max_positive_rate,
        )
        if plan.gate_validation_path is not None
        else None
    )
    materialized_weights = bool(getattr(backend, "materializes_weights", True))
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = plan.output_dir / "post_training_manifest.json"
    manifest = _load_manifest(manifest_path) if plan.resume else None
    source_sha256 = _source_sha256(plan.source_path)
    backend_identity = _backend_identity(backend)
    backend_fingerprint = _fingerprint(backend_identity)
    if plan.source_path.is_dir():
        corpora = prepare_training_ready_corpora(plan)
    else:
        incidents = load_verified_incidents(plan.source_path)
        corpora = prepare_corpora(plan, incidents)
    corpora_sha256 = _corpora_sha256(corpora, plan.stages)
    if manifest is not None:
        _validate_resume_manifest(
            plan,
            manifest,
            source_sha256=source_sha256,
            corpora_sha256=corpora_sha256,
            backend_fingerprint=backend_fingerprint,
        )

    stage_datasets: dict[TrainingStage, tuple[Path, ...]] = {
        TrainingStage.SFT: (corpora.sft_path,),
        TrainingStage.DPO: (corpora.repair_dpo_path,),
    }
    if plan.environment_path is not None:
        # Stage 3 samples from the policy and scores those samples in the
        # environment, so its dataset is the environment corpus rather than a
        # set of target records.
        stage_datasets[TrainingStage.GRPO_REPAIR] = (plan.environment_path,)
        stage_datasets[TrainingStage.GRPO_GATE] = (plan.environment_path,)
    if (
        corpora.transfer_diagnosis_sft_path is not None
        and corpora.transfer_diagnosis_dpo_path is not None
    ):
        stage_datasets[TrainingStage.TRANSFER_DIAGNOSIS] = (
            corpora.transfer_diagnosis_dpo_path,
            corpora.transfer_diagnosis_sft_path,
        )
    required_counts: dict[TrainingStage, tuple[str, ...]] = {
        TrainingStage.SFT: ("sft",),
        TrainingStage.DPO: ("repair_dpo",),
        TrainingStage.TRANSFER_DIAGNOSIS: (
            "transfer_diagnosis_dpo",
            "transfer_diagnosis_sft",
        ),
        TrainingStage.GRPO_REPAIR: ("environment_incidents",),
        TrainingStage.GRPO_GATE: ("environment_incidents",),
    }
    if corpora.transfer_diagnosis_mode == "online-sibling-uplift":
        required_counts[TrainingStage.TRANSFER_DIAGNOSIS] = ("repair_dpo",)
    stages = [stage for stage in TRAINING_STAGE_ORDER if stage in plan.stages]
    for stage in stages:
        missing = [
            name for name in required_counts[stage] if corpora.counts.get(name, 0) <= 0
        ]
        if stage not in stage_datasets or missing:
            raise ValueError(
                f"ARIA {stage.value} stage requires non-empty corpora: "
                f"{', '.join(missing or required_counts[stage])}"
            )
    checkpoints: dict[str, str] = (
        dict(manifest.get("checkpoints", {})) if manifest else {}
    )
    current_base: str = plan.base_model
    stage1_reference: Path | None = (
        Path(plan.base_model) if TrainingStage.SFT not in stages else None
    )
    stage3_reference: Path | None = None
    for stage in stages:
        existing = (
            Path(checkpoints[stage.value]) if stage.value in checkpoints else None
        )
        if plan.resume and existing is not None and existing.is_dir():
            if stage == TrainingStage.SFT:
                stage1_reference = existing
            if stage in STAGE_GROUPS["stage3"] and stage3_reference is None:
                stage3_reference = Path(current_base)
            current_base = str(existing)
            continue
        if stage == TrainingStage.SFT:
            reference_model = None
        elif stage in STAGE_GROUPS["stage2"]:
            # Both outcome preferences use the frozen Stage 1 policy theta_0.
            reference_model = stage1_reference or Path(current_base)
        else:
            # Stage 3's weak KL anchor is the policy at Stage 3 entry (theta_1
            # for the full pipeline, theta_0 for the Stage-1+3 ablation), not
            # the immediately preceding GRPO sub-stage.
            if stage3_reference is None:
                stage3_reference = Path(current_base)
            reference_model = stage3_reference
        stage_dir = plan.output_dir / "checkpoints" / stage.value
        checkpoint = backend.train(
            TrainingRun(
                stage=stage,
                base_model=current_base,
                dataset_paths=stage_datasets[stage],
                output_dir=stage_dir,
                reference_model=reference_model,
                objectives=_stage_objectives(stage),
                is_final_stage=stage == stages[-1],
                environment_spec=(
                    plan.environment
                    if stage in STAGE_GROUPS["stage3"]
                    else None
                ),
            )
        )
        if not checkpoint.is_dir():
            raise NotADirectoryError(
                f"ARIA trainer returned a non-directory checkpoint: {checkpoint}"
            )
        checkpoints[stage.value] = str(checkpoint.resolve())
        current_base = str(checkpoint)
        if stage == TrainingStage.SFT:
            stage1_reference = checkpoint
        _atomic_write_json(
            manifest_path,
            _manifest_payload(
                plan,
                corpora,
                checkpoints,
                completed_stage=stage.value,
                materialized_weights=materialized_weights,
                backend_identity=backend_identity,
                backend_fingerprint=backend_fingerprint,
                corpora_sha256=corpora_sha256,
                gate_preflight_report=gate_preflight_report,
            ),
        )

    final_manifest = _manifest_payload(
        plan,
        corpora,
        checkpoints,
        completed_stage=stages[-1].value,
        materialized_weights=materialized_weights,
        backend_identity=backend_identity,
        backend_fingerprint=backend_fingerprint,
        corpora_sha256=corpora_sha256,
        gate_preflight_report=gate_preflight_report,
    )
    final_manifest["status"] = "completed"
    _atomic_write_json(manifest_path, final_manifest)
    return final_manifest


def _stage_objectives(stage: TrainingStage) -> tuple[str, ...]:
    if stage == TrainingStage.TRANSFER_DIAGNOSIS:
        return ("dpo", "sft")
    if stage == TrainingStage.GRPO_REPAIR:
        return ("grpo", "transfer_credit")
    if stage == TrainingStage.GRPO_GATE:
        return ("grpo", "matched_seed_gate")
    return (stage.value,)


def _environment_with_config(
    environment: Mapping[str, Any], config_path: Path | None,
) -> dict[str, Any]:
    value = dict(environment)
    if value.get("kind") == "native":
        root = config_path.parent.resolve() if config_path is not None else Path.cwd()
        declared = value.get("experiment_config")
        if declared:
            value["experiment_config"] = str(_resolve_config_path(root, str(declared)))
        elif config_path is not None:
            value["experiment_config"] = str(config_path.resolve())
    if value.get("kind") == "chain":
        value["backends"] = [
            _environment_with_config(member, config_path)
            for member in value.get("backends", [])
        ]
    return value


def run_post_training_from_spec(
    spec: Mapping[str, Any],
    *,
    config_path: Path | None,
) -> dict[str, Any]:
    """Run a YAML-declared plan without binding ARIA to a trainer framework."""

    root = config_path.parent.resolve() if config_path is not None else Path.cwd()
    source_path = _resolve_config_path(root, str(spec["source_path"]))
    output_dir = _resolve_config_path(root, str(spec["output_dir"]))
    base_model = str(spec["base_model"])
    # YAML paths are resolved relative to the config file, just like source and
    # output roots.  This is required for stage1_3 and other ablations that
    # start from a checkpoint produced by an earlier run.
    base_candidate = Path(base_model)
    if not base_candidate.is_absolute() and (
        base_model.startswith(".") or base_candidate.exists()
    ):
        base_model = str(_resolve_config_path(root, base_model))
    plan = PostTrainingPlan(
        base_model=base_model,
        source_path=source_path,
        output_dir=output_dir,
        feedback_dropout_probability=float(
            spec.get("feedback_dropout_probability", 0.25)
        ),
        diagnosis_uplift_threshold=float(spec.get("diagnosis_uplift_threshold", 0.0)),
        random_seed=int(spec.get("random_seed", 0)),
        resume=bool(spec.get("resume", True)),
        stages=_parse_training_stages(
            spec.get("stages"), variant=spec.get("ablation_variant")
        ),
        environment_path=(
            _resolve_config_path(root, str(spec["environment_path"]))
            if spec.get("environment_path")
            else None
        ),
        environment=(
            _environment_with_config(spec["environment"], config_path)
            if isinstance(spec.get("environment"), Mapping) else None
        ),
        gate_validation_path=(
            _resolve_config_path(root, str(spec["gate_validation_path"]))
            if spec.get("gate_validation_path")
            else None
        ),
        gate_min_positive_rate=float(spec.get("gate_min_positive_rate", 0.20)),
        gate_max_positive_rate=float(spec.get("gate_max_positive_rate", 0.80)),
        repair_conditioning=str(spec.get("repair_conditioning", "incident")),
    )
    holdouts = spec.get("evaluation_holdout_paths")
    if holdouts is not None:
        if not isinstance(holdouts, list) or not holdouts or not all(isinstance(p, str) for p in holdouts):
            raise TypeError("evaluation_holdout_paths must be a non-empty list of paths")
        from aria.split_audit import audit_disjoint
        training_paths = [source_path]
        if plan.environment_path is not None:
            training_paths.append(plan.environment_path)
        audit = audit_disjoint(training_paths, [_resolve_config_path(root, p) for p in holdouts])
        output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(output_dir / "split_audit.json", audit)
        if audit["status"] != "passed":
            raise ValueError("ARIA training overlaps evaluation holdouts; see split_audit.json")
    backend_spec = spec["backend"]
    if not isinstance(backend_spec, Mapping):
        raise TypeError("aria.post_training.backend must be a mapping")
    backend_type = str(backend_spec["type"])
    if backend_type == "artifact-only":
        backend: TrainingBackend = ArtifactOnlyTrainingBackend()
    elif backend_type == "command":
        command = backend_spec.get("command")
        if not isinstance(command, list) or not all(
            isinstance(part, str) for part in command
        ):
            raise TypeError(
                "aria.post_training.backend.command must be a list of strings"
            )
        backend = CommandTrainingBackend(tuple(command))
    else:
        raise ValueError(f"unsupported ARIA training backend: {backend_type}")
    manifest = run_post_training(plan, backend)
    return {
        "type": "aria_post_training",
        "status": manifest["status"],
        "output_dir": str(output_dir),
        "manifest_path": str(output_dir / "post_training_manifest.json"),
        "checkpoints": manifest["checkpoints"],
        "materialized_weights": backend_type != "artifact-only",
    }


def _validate_corpus_manifest(path: Path) -> None:
    """Reject a stale aggregate before a trainer is launched against it.

    The corpus manifest is optional so a hand-assembled directory still trains,
    but when it exists it must declare the row schema this module validates.
    """

    if not path.is_file():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise TypeError(f"ARIA corpus manifest must be an object: {path}")
    declared = manifest.get("dataset_schema_version")
    if declared is not None and declared not in {
        TRAINING_READY_SCHEMA_VERSION,
        LEGACY_TRAINING_READY_SCHEMA_VERSION,
    }:
        raise ValueError(
            f"ARIA corpus declares unsupported row schema {declared!r}: {path}"
        )


def _parse_training_stages(
    value: Any, *, variant: Any = None
) -> tuple[TrainingStage, ...]:
    """Resolve a stage list, or an RQ3 ablation variant that names one."""

    if variant is not None:
        if value is not None:
            raise ValueError(
                "give aria.post_training either stages or ablation_variant, not both"
            )
        name = str(variant)
        try:
            return ABLATION_VARIANTS[name]
        except KeyError as exc:
            raise ValueError(
                "aria.post_training.ablation_variant must be one of "
                + ", ".join(sorted(ABLATION_VARIANTS))
            ) from exc
    if value is None:
        return TRAINING_STAGE_ORDER
    if not isinstance(value, list) or not value:
        raise TypeError("aria.post_training.stages must be a non-empty list")
    stages: list[TrainingStage] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, str):
            raise TypeError(f"aria.post_training.stages[{index}] must be a string")
        try:
            stages.append(TrainingStage(raw))
        except ValueError as exc:
            raise ValueError(
                "aria.post_training.stages entries must be one of "
                + ", ".join(stage.value for stage in TRAINING_STAGE_ORDER)
            ) from exc
    return tuple(stages)


def load_verified_incidents(path: Path) -> list[IncidentRecord]:
    incidents: list[IncidentRecord] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise TypeError(
                    f"training source line {line_number} must be a JSON object"
                )
            incident = IncidentRecord.from_dict(
                raw,
                require_repair=True,
                require_verified_success=True,
            )
            if incident.incident_id in seen:
                raise ValueError(
                    f"duplicate training incident_id: {incident.incident_id}"
                )
            seen.add(incident.incident_id)
            incidents.append(incident)
    if not incidents:
        raise ValueError("ARIA post-training source contains no verified incidents")
    return incidents


def prepare_training_ready_corpora(plan: PostTrainingPlan) -> PreparedCorpora:
    """Validate and stage pre-materialized paper-stage corpora."""

    corpus_dir = plan.output_dir / "corpora"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    sft_source = plan.source_path / "sft.jsonl"
    dpo_source = plan.source_path / "dpo.jsonl"
    transfer_sft_source = plan.source_path / "transfer_diagnosis_sft.jsonl"
    transfer_dpo_source = plan.source_path / "transfer_diagnosis_dpo.jsonl"
    sft_path = corpus_dir / "sft.jsonl"
    dpo_path = corpus_dir / "repair_dpo.jsonl"
    transfer_sft_path = corpus_dir / "transfer_diagnosis_sft.jsonl"
    transfer_dpo_path = corpus_dir / "transfer_diagnosis_dpo.jsonl"
    sft_fingerprints: dict[str, str] = {}
    structured_repair_fingerprints: dict[str, str] = {}
    structured_repair_feedback_drop: dict[str, bool] = {}
    sft_task_counts: dict[str, int] = dict.fromkeys(SFT_TASKS, 0)
    drop_feedback: dict[str, bool] = {}
    rng = random.Random(plan.random_seed)

    def transform_sft(
        raw: Mapping[str, Any], line_number: int
    ) -> Sequence[Mapping[str, Any]]:
        if raw.get("schema_version") == TRAINING_READY_SCHEMA_VERSION:
            row = _validate_task_training_row(
                raw,
                name=f"{sft_source}:{line_number}",
                preference=False,
            )
            record_id = str(row["record_id"])
            if record_id in sft_fingerprints:
                raise ValueError(f"duplicate SFT record_id: {record_id}")
            input_payload = row["input"]
            assert isinstance(input_payload, Mapping)
            feedback = input_payload.get("feedback")
            should_drop = bool(
                feedback and rng.random() < plan.feedback_dropout_probability
            )
            transformed = dict(row)
            if should_drop:
                transformed_input = dict(input_payload)
                transformed_input["feedback"] = None
                transformed["input"] = transformed_input
            task = str(transformed["task"])
            if task == "structured_repair":
                transformed["input"] = _observable_incident_input(
                    transformed["input"],
                    name=f"{sft_source}:{line_number}",
                    conditioning=plan.repair_conditioning,
                )
            sft_task_counts[task] = sft_task_counts.get(task, 0) + 1
            fingerprint = _fingerprint(
                {"input": transformed["input"], "target": transformed["target"]}
            )
            sft_fingerprints[record_id] = fingerprint
            drop_feedback[record_id] = should_drop
            if task == "structured_repair":
                source_record_id = _source_record_id(
                    transformed,
                    name=f"{sft_source}:{line_number}",
                )
                if source_record_id in structured_repair_fingerprints:
                    raise ValueError(
                        "duplicate structured-repair SFT source_record_id: "
                        f"{source_record_id}"
                    )
                structured_repair_fingerprints[source_record_id] = fingerprint
                structured_repair_feedback_drop[source_record_id] = should_drop
            return (transformed,)
        record_id, fingerprint = _validate_training_ready_row(
            raw,
            name=f"{sft_source}:{line_number}",
            require_rejected=False,
        )
        if record_id in sft_fingerprints:
            raise ValueError(f"duplicate SFT record_id: {record_id}")
        sft_fingerprints[record_id] = fingerprint
        prompt = raw["prompt"]
        assert isinstance(prompt, Mapping)
        feedback = prompt.get("feedback")
        should_drop = bool(
            feedback and rng.random() < plan.feedback_dropout_probability
        )
        drop_feedback[record_id] = should_drop
        visible_feedback = None if should_drop else feedback
        chosen = raw["chosen"]
        assert isinstance(chosen, Mapping)
        incident = IncidentRecord.from_dict(
            {
                **prompt,
                "feedback": visible_feedback,
                "repair": chosen,
                "verified_successful": True,
            },
            require_repair=True,
            require_verified_success=True,
        )
        return build_sft_rows(incident, feedback=visible_feedback)

    sft_count = _atomic_transform_jsonl(sft_source, sft_path, transform_sft)
    if sft_count <= 0:
        raise ValueError("ARIA training-ready source contains no SFT rows")

    seen_dpo: set[str] = set()

    def transform_dpo(raw: Mapping[str, Any], line_number: int) -> Mapping[str, Any]:
        if raw.get("schema_version") == TRAINING_READY_SCHEMA_VERSION:
            row = dict(
                _validate_task_training_row(
                    raw,
                    name=f"{dpo_source}:{line_number}",
                    preference=True,
                )
            )
            record_id = str(row["record_id"])
            if record_id in seen_dpo:
                raise ValueError(f"duplicate DPO record_id: {record_id}")
            seen_dpo.add(record_id)
            model_input = row["input"]
            assert isinstance(model_input, Mapping)
            row["input"] = _observable_incident_input(
                model_input,
                name=f"{dpo_source}:{line_number}",
                conditioning=plan.repair_conditioning,
            )
            model_input = row["input"]
            source_record_id = _source_record_id(
                row,
                name=f"{dpo_source}:{line_number}",
            )
            should_drop = structured_repair_feedback_drop.get(
                source_record_id, False
            )
            if should_drop and model_input.get("feedback"):
                transformed_input = dict(model_input)
                transformed_input["feedback"] = None
                row["input"] = transformed_input
            expected_fingerprint = structured_repair_fingerprints.get(
                source_record_id
            )
            if expected_fingerprint is None:
                raise ValueError(
                    "DPO row has no matching structured-repair SFT row for "
                    f"source_record_id: {source_record_id}"
                )
            chosen = row["chosen"]
            assert isinstance(chosen, Mapping)
            actual_fingerprint = _fingerprint(
                {"input": row["input"], "target": chosen}
            )
            if actual_fingerprint != expected_fingerprint:
                raise ValueError(
                    "DPO input/chosen does not match structured-repair SFT for "
                    f"source_record_id: {source_record_id}"
                )
            return row
        record_id, fingerprint = _validate_training_ready_row(
            raw,
            name=f"{dpo_source}:{line_number}",
            require_rejected=True,
        )
        if record_id in seen_dpo:
            raise ValueError(f"duplicate DPO record_id: {record_id}")
        seen_dpo.add(record_id)
        if record_id not in sft_fingerprints:
            raise ValueError(f"DPO record_id has no matching SFT row: {record_id}")
        if sft_fingerprints[record_id] != fingerprint:
            raise ValueError(f"SFT/DPO prompt or chosen mismatch for {record_id}")
        row = _training_row_with_feedback(raw, drop=drop_feedback[record_id])
        chosen = dict(row["chosen"])
        rejected = dict(row["rejected"])
        metadata = row.get("metadata", {})
        assert isinstance(metadata, Mapping)
        model_diagnosis = metadata.get("model_diagnosis")
        if not isinstance(model_diagnosis, str) or not model_diagnosis.strip():
            model_diagnosis = chosen["diagnosis"]
            diagnosis_source = "repair_fallback"
        else:
            diagnosis_source = "model"
        chosen["diagnosis"] = model_diagnosis
        rejected["diagnosis"] = model_diagnosis
        row["chosen"] = chosen
        row["rejected"] = rejected
        row["task"] = "repair_preference"
        row["diagnosis_source"] = diagnosis_source
        return row

    dpo_count = _atomic_transform_jsonl(dpo_source, dpo_path, transform_dpo)
    transfer_sft_count = 0
    transfer_dpo_count = 0
    transfer_mode: str | None = None
    if transfer_sft_source.is_file() != transfer_dpo_source.is_file():
        raise FileNotFoundError(
            "ARIA transfer diagnosis requires both transfer_diagnosis_sft.jsonl "
            "and transfer_diagnosis_dpo.jsonl"
        )
    if transfer_sft_source.is_file():
        transfer_mode = "precomputed"
        transfer_sft_count = _atomic_transform_jsonl(
            transfer_sft_source,
            transfer_sft_path,
            lambda raw, line_number: _validate_transfer_training_row(
                raw,
                name=f"{transfer_sft_source}:{line_number}",
                preference=False,
            ),
        )
        transfer_dpo_count = _atomic_transform_jsonl(
            transfer_dpo_source,
            transfer_dpo_path,
            lambda raw, line_number: _validate_transfer_training_row(
                raw,
                name=f"{transfer_dpo_source}:{line_number}",
                preference=True,
            ),
        )
    elif TrainingStage.TRANSFER_DIAGNOSIS in plan.stages:
        if dpo_count <= 0:
            raise ValueError(
                "online transfer diagnosis requires non-empty causal DPO rows"
            )
        transfer_mode = "online-sibling-uplift"
        transfer_sft_path = sft_path
        transfer_dpo_path = dpo_path
    return PreparedCorpora(
        sft_path=sft_path,
        repair_dpo_path=dpo_path,
        transfer_diagnosis_sft_path=(transfer_sft_path if transfer_mode else None),
        transfer_diagnosis_dpo_path=(transfer_dpo_path if transfer_mode else None),
        transfer_diagnosis_mode=transfer_mode,
        counts={
            "sft": sft_count,
            "repair_dpo": dpo_count,
            **sft_task_counts,
            "transfer_diagnosis_sft": transfer_sft_count,
            "transfer_diagnosis_dpo": transfer_dpo_count,
            "environment_incidents": _environment_incident_count(plan),
        },
        environment_path=plan.environment_path,
    )


def _environment_incident_count(plan: PostTrainingPlan) -> int:
    """Count the incidents Stage 3 may sample from, without loading them.

    Stage 3 has no target records to prepare, so the corpus-readiness check it
    submits to is that its environment corpus exists and is non-empty.
    """

    if plan.environment_path is None:
        return 0
    count = 0
    with plan.environment_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _validate_transfer_training_row(
    raw: Mapping[str, Any],
    *,
    name: str,
    preference: bool,
) -> Mapping[str, Any]:
    model_input = raw.get("input")
    if not isinstance(model_input, Mapping):
        raise TypeError(f"{name}.input must be a mapping")
    _assert_answer_blind(model_input, name=f"{name}.input")
    required_outputs = ("chosen", "rejected") if preference else ("target",)
    for field in required_outputs:
        output = raw.get(field)
        if not isinstance(output, Mapping):
            raise TypeError(f"{name}.{field} must be a mapping")
        _assert_answer_blind(output, name=f"{name}.{field}")
    if preference and set(raw["chosen"]) != set(raw["rejected"]):
        raise ValueError(f"{name} chosen/rejected output schemas do not match")
    return raw


def _validate_training_ready_row(
    raw: Mapping[str, Any],
    *,
    name: str,
    require_rejected: bool,
) -> tuple[str, str]:
    if raw.get("schema_version") != LEGACY_TRAINING_READY_SCHEMA_VERSION:
        raise ValueError(f"{name} has unsupported schema_version")
    record_id = raw.get("record_id")
    if not isinstance(record_id, str) or not record_id.strip():
        raise ValueError(f"{name}.record_id must be a non-empty string")
    prompt = raw.get("prompt")
    if not isinstance(prompt, Mapping):
        raise TypeError(f"{name}.prompt must be a mapping")
    unexpected_prompt_keys = set(prompt) - {
        "incident_id",
        "events",
        "feedback",
        "constraints",
    }
    if unexpected_prompt_keys:
        raise ValueError(
            f"{name}.prompt has unsupported fields: {sorted(unexpected_prompt_keys)}"
        )
    _assert_answer_blind(prompt, name=f"{name}.prompt")
    chosen = raw.get("chosen")
    if not isinstance(chosen, Mapping):
        raise TypeError(f"{name}.chosen must be a mapping")
    incident = IncidentRecord.from_dict(
        {**prompt, "repair": chosen, "verified_successful": True},
        require_repair=True,
        require_verified_success=True,
    )
    if incident.incident_id != record_id:
        raise ValueError(f"{name}.prompt.incident_id must equal record_id")
    if require_rejected:
        rejected = raw.get("rejected")
        if not isinstance(rejected, Mapping):
            raise TypeError(f"{name}.rejected must be a mapping")
        RepairRecord.from_dict(
            rejected, events=incident.events, name=f"{name}.rejected"
        )
        if set(chosen) != set(rejected):
            raise ValueError(f"{name} chosen/rejected output schemas do not match")
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError(f"{name}.metadata must be a mapping")
    fingerprint = hashlib.sha256(
        json.dumps(
            {"prompt": prompt, "chosen": chosen},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return record_id, fingerprint


def _validate_task_training_row(
    raw: Mapping[str, Any],
    *,
    name: str,
    preference: bool,
) -> Mapping[str, Any]:
    if raw.get("schema_version") != TRAINING_READY_SCHEMA_VERSION:
        raise ValueError(f"{name} has unsupported schema_version")
    record_id = raw.get("record_id")
    if not isinstance(record_id, str) or not record_id.strip():
        raise ValueError(f"{name}.record_id must be a non-empty string")
    task = raw.get("task")
    expected_tasks = set(PREFERENCE_TASKS) if preference else set(SFT_TASKS)
    if task not in expected_tasks:
        raise ValueError(f"{name}.task must be one of {sorted(expected_tasks)}")
    model_input = raw.get("input")
    if not isinstance(model_input, Mapping):
        raise TypeError(f"{name}.input must be a mapping")
    _assert_answer_blind(model_input, name=f"{name}.input")
    output_fields = ("chosen", "rejected") if preference else ("target",)
    for field in output_fields:
        output = raw.get(field)
        if not isinstance(output, Mapping):
            raise TypeError(f"{name}.{field} must be a mapping")
        _assert_answer_blind(output, name=f"{name}.{field}")
        ordered = ordered_training_output(str(task), output)
        if set(output) != set(ordered):
            raise ValueError(
                f"{name}.{field} has unsupported fields: "
                f"{sorted(set(output) - set(ordered))}"
            )
    incident = IncidentRecord.from_dict(
        {
            "incident_id": model_input.get("incident_id"),
            "events": model_input.get("events"),
            "feedback": model_input.get("feedback"),
            "constraints": model_input.get("constraints", {}),
        }
    )
    if not preference:
        target = raw["target"]
        assert isinstance(target, Mapping)
        if task == "deployment_gate":
            probability = target.get("gate_probability")
            if (
                not isinstance(probability, (int, float))
                or isinstance(probability, bool)
                or not 0.0 <= float(probability) <= 1.0
            ):
                raise ValueError(
                    f"{name}.target.gate_probability must be numeric between 0 and 1"
                )
        elif task == "failure_localization":
            FailureLocalization.from_dict(
                target,
                events=incident.events,
                name=f"{name}.target.localization",
            )
        elif task == "repair_attribution":
            localization_raw = model_input.get("localization")
            if not isinstance(localization_raw, Mapping):
                raise TypeError(f"{name}.input.localization must be an object")
            localization = FailureLocalization.from_dict(
                localization_raw,
                events=incident.events,
                name=f"{name}.input.localization",
            )
            RepairAttribution.from_dict(
                target,
                events=incident.events,
                localization=localization,
                name=f"{name}.target.attribution",
            )
        elif task == "structured_repair":
            RepairRecord.from_dict(
                target,
                events=incident.events,
                name=f"{name}.target",
            )
    if preference:
        chosen = raw["chosen"]
        rejected = raw["rejected"]
        assert isinstance(chosen, Mapping) and isinstance(rejected, Mapping)
        if set(chosen) != set(rejected):
            raise ValueError(f"{name} chosen/rejected output schemas do not match")
        events_raw = model_input.get("events")
        if not isinstance(events_raw, list):
            raise TypeError(f"{name}.input.events must be a list")
        RepairRecord.from_dict(chosen, events=incident.events, name=f"{name}.chosen")
        RepairRecord.from_dict(
            rejected, events=incident.events, name=f"{name}.rejected"
        )
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError(f"{name}.metadata must be a mapping")
    return raw


def _mapping_metadata(raw: Mapping[str, Any]) -> Mapping[str, Any]:
    value = raw.get("metadata", {})
    return value if isinstance(value, Mapping) else {}


def _observable_incident_input(
    value: Any,
    *,
    name: str,
    conditioning: str = "incident",
) -> dict[str, Any]:
    """Keep cascade conditioning for deployment-aligned runs; preserve legacy views."""

    if not isinstance(value, Mapping):
        raise TypeError(f"{name}.input must be a mapping")
    required = ("incident_id", "events", "feedback", "constraints")
    if conditioning == "cascade":
        required += ("localization", "attribution")
    elif conditioning != "incident":
        raise ValueError("repair conditioning must be incident or cascade")
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"{name}.input is missing observable fields: {missing}")
    return {key: value[key] for key in required}


def _source_record_id(raw: Mapping[str, Any], *, name: str) -> str:
    source_record_id = _mapping_metadata(raw).get("source_record_id")
    if not isinstance(source_record_id, str) or not source_record_id.strip():
        raise ValueError(f"{name}.metadata.source_record_id must be non-empty")
    return source_record_id


def validate_training_ready_task_row(
    raw: Mapping[str, Any],
    *,
    name: str,
    preference: bool,
) -> Mapping[str, Any]:
    """Public validator shared by corpus builders and the training loader."""

    return _validate_task_training_row(
        raw,
        name=name,
        preference=preference,
    )


def _assert_answer_blind(value: Any, *, name: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in ANSWER_LABEL_FIELDS:
                raise ValueError(f"{name} contains forbidden answer label field: {key}")
            _assert_answer_blind(item, name=f"{name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_answer_blind(item, name=f"{name}[{index}]")


def _training_row_with_feedback(
    raw: Mapping[str, Any],
    *,
    drop: bool,
) -> dict[str, Any]:
    row = dict(raw)
    prompt = dict(raw["prompt"])
    if drop:
        prompt["feedback"] = None
    row["prompt"] = prompt
    return row


def _atomic_transform_jsonl(
    source: Path,
    destination: Path,
    transform: Callable[
        [Mapping[str, Any], int],
        Mapping[str, Any] | Sequence[Mapping[str, Any]],
    ],
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    count = 0
    try:
        with (
            source.open(encoding="utf-8") as input_handle,
            temporary.open("w", encoding="utf-8") as output_handle,
        ):
            for line_number, line in enumerate(input_handle, start=1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON at {source}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(raw, Mapping):
                    raise TypeError(f"{source}:{line_number} must contain an object")
                transformed = transform(raw, line_number)
                rows = (
                    (transformed,) if isinstance(transformed, Mapping) else transformed
                )
                for row in rows:
                    if not isinstance(row, Mapping):
                        raise TypeError(
                            f"transform for {source}:{line_number} returned a non-object"
                        )
                    output_handle.write(json.dumps(row, sort_keys=True) + "\n")
                    count += 1
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def prepare_corpora(
    plan: PostTrainingPlan, incidents: Sequence[IncidentRecord]
) -> PreparedCorpora:
    """Materialize the three paper-aligned stage corpora."""

    corpus_dir = plan.output_dir / "corpora"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(plan.random_seed)
    sft_rows: list[dict[str, Any]] = []
    transfer_diagnosis_sft_rows: list[dict[str, Any]] = []
    repair_dpo_rows: list[dict[str, Any]] = []
    transfer_diagnosis_dpo_rows: list[dict[str, Any]] = []
    for incident in incidents:
        feedback = _drop_feedback(
            incident.feedback, plan.feedback_dropout_probability, rng
        )
        sft_rows.extend(build_sft_rows(incident, feedback=feedback))
        repair_dpo_rows.append(build_repair_preference_row(incident, feedback=feedback))
        transfer_rows = build_transfer_diagnosis_rows(
            incident,
            feedback=feedback,
            uplift_threshold=plan.diagnosis_uplift_threshold,
        )
        if transfer_rows is not None:
            diagnosis_dpo, diagnosis_sft = transfer_rows
            transfer_diagnosis_dpo_rows.append(diagnosis_dpo)
            transfer_diagnosis_sft_rows.append(diagnosis_sft)
    paths = {
        "sft": corpus_dir / "sft.jsonl",
        "transfer_diagnosis_sft": corpus_dir / "transfer_diagnosis_sft.jsonl",
        "repair_dpo": corpus_dir / "repair_dpo.jsonl",
        "transfer_diagnosis_dpo": corpus_dir / "transfer_diagnosis_dpo.jsonl",
    }
    _atomic_write_jsonl(paths["sft"], sft_rows)
    _atomic_write_jsonl(paths["transfer_diagnosis_sft"], transfer_diagnosis_sft_rows)
    _atomic_write_jsonl(paths["repair_dpo"], repair_dpo_rows)
    _atomic_write_jsonl(paths["transfer_diagnosis_dpo"], transfer_diagnosis_dpo_rows)
    return PreparedCorpora(
        sft_path=paths["sft"],
        transfer_diagnosis_sft_path=paths["transfer_diagnosis_sft"],
        repair_dpo_path=paths["repair_dpo"],
        transfer_diagnosis_dpo_path=paths["transfer_diagnosis_dpo"],
        counts={
            "sft": len(sft_rows),
            "repair_dpo": len(repair_dpo_rows),
            "transfer_diagnosis_sft": len(transfer_diagnosis_sft_rows),
            "transfer_diagnosis_dpo": len(transfer_diagnosis_dpo_rows),
            "environment_incidents": _environment_incident_count(plan),
        },
        environment_path=plan.environment_path,
    )


def _drop_feedback(
    feedback: str | None, probability: float, rng: random.Random
) -> str | None:
    if feedback and rng.random() < probability:
        return None
    return feedback


def _manifest_payload(
    plan: PostTrainingPlan,
    corpora: PreparedCorpora,
    checkpoints: Mapping[str, str],
    *,
    completed_stage: str,
    materialized_weights: bool,
    backend_identity: Mapping[str, Any],
    backend_fingerprint: str,
    corpora_sha256: Mapping[str, str],
    gate_preflight_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    selected_stages = [
        stage.value for stage in TRAINING_STAGE_ORDER if stage in plan.stages
    ]
    pipeline_complete = completed_stage == selected_stages[-1]
    final_checkpoint = (
        checkpoints.get(selected_stages[-1]) if pipeline_complete else None
    )
    checkpoint_kind = _checkpoint_kind(final_checkpoint)
    return {
        "method": "aria-post-training-sft-dpo-transfer-grpo-v4",
        "base_model": plan.base_model,
        "source_path": str(plan.source_path.resolve()),
        "source_kind": (
            "training-ready-directory"
            if plan.source_path.is_dir()
            else "incident-jsonl"
        ),
        "source_sha256": _source_sha256(plan.source_path),
        "feedback_dropout_probability": plan.feedback_dropout_probability,
        "repair_conditioning": plan.repair_conditioning,
        "diagnosis_uplift_threshold": plan.diagnosis_uplift_threshold,
        "random_seed": plan.random_seed,
        "training_backend": dict(backend_identity),
        "backend_fingerprint": backend_fingerprint,
        "corpora_sha256": dict(corpora_sha256),
        "stages": selected_stages,
        "stage_plan": {
            stage.value: {"enabled": stage in plan.stages, "order": index + 1}
            for index, stage in enumerate(TRAINING_STAGE_ORDER)
        },
        "corpora": {
            key: str(path.resolve()) if path is not None else None
            for key, path in {
                "sft": corpora.sft_path,
                "repair_dpo": corpora.repair_dpo_path,
                "transfer_diagnosis_sft": corpora.transfer_diagnosis_sft_path,
                "transfer_diagnosis_dpo": corpora.transfer_diagnosis_dpo_path,
                "environment": plan.environment_path,
            }.items()
        },
        "transfer_diagnosis_mode": corpora.transfer_diagnosis_mode,
        "stage3_environment": dict(plan.environment) if plan.environment else None,
        "gate_preflight": (
            dict(gate_preflight_report) if gate_preflight_report is not None else None
        ),
        "counts": dict(corpora.counts),
        "checkpoints": dict(checkpoints),
        "completed_stage": completed_stage,
        "final_checkpoint": final_checkpoint,
        "materialized_weights": materialized_weights,
        "deliverables": {
            "hook": {
                "python_package": "aria.hook",
                "component": "AriaHook",
            },
            "model_checkpoint": final_checkpoint if materialized_weights else None,
            "model_checkpoint_kind": checkpoint_kind,
            "ready": bool(
                materialized_weights
                and final_checkpoint
                and checkpoint_kind == "standalone"
            ),
        },
        "status": "running",
    }


def _load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"ARIA manifest must be an object: {path}")
    return raw


def _validate_resume_manifest(
    plan: PostTrainingPlan,
    manifest: Mapping[str, Any],
    *,
    source_sha256: str,
    corpora_sha256: Mapping[str, str],
    backend_fingerprint: str,
) -> None:
    expected = {
        "source_sha256": source_sha256,
        "corpora_sha256": dict(corpora_sha256),
        "base_model": plan.base_model,
        "feedback_dropout_probability": plan.feedback_dropout_probability,
        "diagnosis_uplift_threshold": plan.diagnosis_uplift_threshold,
        "random_seed": plan.random_seed,
        "backend_fingerprint": backend_fingerprint,
        "stages": [stage.value for stage in plan.stages],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"cannot resume ARIA post-training after {key} changed: "
                f"{manifest.get(key)!r} != {value!r}"
            )
    if manifest.get("repair_conditioning", "incident") != plan.repair_conditioning:
        raise ValueError("cannot resume ARIA post-training after repair_conditioning changed")


def _backend_identity(backend: TrainingBackend) -> dict[str, Any]:
    if isinstance(backend, CommandTrainingBackend):
        command_files: dict[str, str] = {}
        for part in backend.command:
            path = Path(part)
            if "{" in part or not path.is_file():
                continue
            command_files[part] = _file_sha256(path)
            if path.suffix == ".sh":
                for reference in re.findall(
                    r"[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)+",
                    path.read_text(encoding="utf-8"),
                ):
                    referenced_path = Path(reference)
                    if referenced_path.is_file():
                        command_files[reference] = _file_sha256(referenced_path)
        if "scripts/aria/train_transformers.py" in command_files:
            for relative in (
                "src/aria/prompts.py", "src/aria/prompt_budget.py", "src/aria/records.py",
                "src/aria/training.py", "src/aria/post_training/environment.py",
                "src/aria/post_training/grpo.py",
                "src/aria/gate_scoring.py", "src/aria/calibration.py",
                "src/aria/training_balance.py", "src/aria/model.py", "src/aria/hook/controller.py",
                "src/aria/post_training/deployment_rollout.py",
                "src/common/evaluation.py",
            ):
                command_files[relative] = _file_sha256(Path(relative))
        return {
            "type": "command",
            "command": list(backend.command),
            "command_files": command_files,
        }
    if isinstance(backend, ArtifactOnlyTrainingBackend):
        return {"type": "artifact-only"}
    return {
        "type": f"{type(backend).__module__}.{type(backend).__qualname__}",
        "materializes_weights": bool(getattr(backend, "materializes_weights", True)),
    }


def _fingerprint(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _corpora_sha256(
    corpora: PreparedCorpora,
    stages: Sequence[TrainingStage],
) -> dict[str, str]:
    paths = {
        "sft": (corpora.sft_path if TrainingStage.SFT in stages else None),
        "repair_dpo": (
            corpora.repair_dpo_path if TrainingStage.DPO in stages else None
        ),
        "transfer_diagnosis_sft": (
            corpora.transfer_diagnosis_sft_path
            if TrainingStage.TRANSFER_DIAGNOSIS in stages
            else None
        ),
        "transfer_diagnosis_dpo": (
            corpora.transfer_diagnosis_dpo_path
            if TrainingStage.TRANSFER_DIAGNOSIS in stages
            else None
        ),
        "environment": (
            corpora.environment_path
            if any(stage in stages for stage in STAGE_GROUPS["stage3"])
            else None
        ),
    }
    return {
        name: _file_sha256(path) for name, path in paths.items() if path is not None
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_kind(checkpoint: str | None) -> str | None:
    if checkpoint is None:
        return None
    path = Path(checkpoint)
    if (path / "adapter_config.json").is_file():
        return "peft-adapter"
    has_model_weights = (
        any(path.glob("model*.safetensors")) or (path / "pytorch_model.bin").is_file()
    )
    if (path / "config.json").is_file() and has_model_weights:
        return "standalone"
    if (path / "checkpoint.json").is_file():
        return "artifact-only"
    return "unknown"


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    paths = (
        [path]
        if path.is_file()
        else [
            item
            for item in (
                path / "sft.jsonl",
                path / "dpo.jsonl",
                path / "transfer_diagnosis_sft.jsonl",
                path / "transfer_diagnosis_dpo.jsonl",
            )
            if item.is_file()
        ]
    )
    for item in paths:
        if path.is_dir():
            digest.update(item.relative_to(path).as_posix().encode("utf-8"))
            digest.update(b"\0")
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _resolve_config_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
