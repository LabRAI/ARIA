"""Stage 3: transfer-grounded policy optimization.

This module owns everything about Stage 3 that does not depend on a tensor
library: the static screen that decides whether a sampled record is executed at
all, the execution-grounded reward of Equation~(5), the diagnosis value of
Equation~(6) with its sibling-transfer term, the two-level advantages that give
diagnoses and repairs separate baselines, the group filters that discard a group
carrying no gradient, and the matched-seed gate reward of Equation~(7).

Keeping these here rather than inside the trainer means the reward that Stage 3
optimizes is the same object the evaluation harness reports, and that each piece
is testable without loading a model.
"""

from __future__ import annotations

import json
import math
import random
import re
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from aria.post_training.environment import (
    ACCEPTED,
    INVALID,
    UNRESOLVED,
    EnvironmentIncident,
    ExecutionOutcome,
)
from aria.records import MAX_REPAIR_REPLACEMENT_EVENTS, RepairRecord

STAGE3_SCHEMA_VERSION = "aria-stage3-grpo-v1"

__all__ = [
    "STAGE3_SCHEMA_VERSION",
    "GateRollout",
    "GateRolloutGroup",
    "GrpoConfig",
    "RepairGroup",
    "RepairSample",
    "ScreenResult",
    "build_repair_group",
    "diagnosis_value",
    "gate_reward",
    "gate_rollout_advantages",
    "parse_record",
    "reward",
    "screen_record",
    "standardize",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GrpoConfig:
    """Stage 3 hyper-parameters, named as in Section 3.3.3."""

    # Equation (5): r = 1[accept] - lambda_h * 1[harmful] - lambda_c * b(y)
    harmful_penalty: float = 1.0
    """:math:`\\lambda_h`, the cost of a harmful intervention."""

    cost_penalty: float = 0.05
    """:math:`\\lambda_c`, the price of one full execution of overhead."""

    invalid_reward: float = -1.0
    """The fixed negative reward of a statically screened record."""

    max_replacement_events: int = MAX_REPAIR_REPLACEMENT_EVENTS
    """:math:`L`, the bound on the replacement continuation."""

    # Equation (6): V(d) mixes own-incident and sibling-cell repairs.
    transfer_weight: float = 0.5
    """:math:`\\alpha`, the weight Stage 3 places on cross-cell transfer."""

    diagnosis_group_size: int = 4
    """:math:`m`, diagnoses sampled per incident."""

    repair_group_size: int = 4
    """:math:`n`, repairs sampled per diagnosis."""

    transfer_siblings: int = 1
    """:math:`n_t`, sibling incidents repaired under the same diagnosis."""

    # GRPO objective.
    clip_low: float = 0.2
    clip_high: float = 0.28
    """DAPO's raised upper bound, which keeps sampling entropy from collapsing."""

    kl_coefficient: float = 0.02
    """Weak penalty toward :math:`\\pi_{\\theta_1}`."""

    advantage_epsilon: float = 1e-6
    min_group_std: float = 1e-3
    """Groups without reward spread are discarded (dynamic sampling)."""

    per_cell_standardization: bool = True
    randomize_event_surface: bool = True
    """Domain randomization over the serialized event log."""

    # Stage 3b.
    gate_group_size: int = 4
    gate_threshold: float = 0.5
    gate_cost_penalty: float = 0.05
    """:math:`\\lambda_c` applied to the gate's :math:`\\Delta\\mathrm{EB}`."""

    outer_rounds: int = 1
    """Alternations of repair and gate optimization."""

    def validate(self) -> None:
        if not 0.0 <= self.transfer_weight <= 1.0:
            raise ValueError("transfer weight alpha must be between 0 and 1")
        if self.diagnosis_group_size < 2 or self.repair_group_size < 1:
            raise ValueError("Stage 3a requires m >= 2 diagnoses and n >= 1 repairs")
        if self.gate_group_size < 2:
            raise ValueError("Stage 3b requires at least two gate rollouts per group")
        if self.max_replacement_events < 1:
            raise ValueError("the continuation bound L must be positive")
        if self.clip_high < self.clip_low:
            raise ValueError("the upper clipping bound must not be below the lower one")
        if self.outer_rounds < 1:
            raise ValueError("Stage 3 requires at least one outer round")


# ---------------------------------------------------------------------------
# Static screen
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScreenResult:
    """The outcome of the static validators that gate execution."""

    valid: bool
    reason: str | None
    repair: RepairRecord | None

    @property
    def rejected(self) -> bool:
        return not self.valid


def parse_record(
    text: str, incident: EnvironmentIncident
) -> tuple[RepairRecord | None, str | None]:
    """Parse a sampled record, returning the parse failure instead of raising."""

    payload = _extract_json(text)
    if payload is None:
        return None, "sample is not a single well-formed JSON object"
    if not isinstance(payload, Mapping):
        return None, "sample is not a JSON object"
    fields = {
        key: payload[key]
        for key in ("diagnosis", "cut_event_id", "replacement_events")
        if key in payload
    }
    missing = [
        key
        for key in ("diagnosis", "cut_event_id", "replacement_events")
        if key not in payload
    ]
    if missing:
        return None, f"record is missing fields: {missing}"
    try:
        record = RepairRecord.from_dict(fields, events=incident.events)
    except (TypeError, ValueError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    return record, None


def screen_record(
    record: RepairRecord | None,
    incident: EnvironmentIncident,
    config: GrpoConfig,
    *,
    parse_error: str | None = None,
    max_prefix_length: int | None = None,
    enforce_certified_cut: bool = True,
) -> ScreenResult:
    """Apply the Stage 3 static validators.

    A record must be well formed, must not cut so late that it retains the
    certified fault when ``enforce_certified_cut`` is enabled, must keep its
    continuation within the bound :math:`L`, must replace only events the cut
    discards, and must reference only source events
    present in :math:`\\tau_{\\leq \\hat k}`, since a continuation that cites a
    discarded event asserts context the restored execution will not have.  When
    the sample was drawn at a runtime prefix the cut must additionally satisfy
    :math:`\\hat k \\leq t`.  A record failing any validator is assigned the
    fixed negative reward and never executed.

    Event identifiers are normalized before they are resolved.  The corpus was
    written by a teacher model that occasionally drops a leading zero from an
    index, and a validator strict enough to reject those references would
    penalize the policy for reproducing its own supervision.
    """

    if record is None:
        return ScreenResult(False, parse_error or "record could not be parsed", None)
    cut_index = record.cut_index(incident.events)
    if len(record.replacement_events) > config.max_replacement_events:
        return ScreenResult(
            False,
            f"continuation exceeds the bound of {config.max_replacement_events} events",
            record,
        )
    if enforce_certified_cut:
        if incident.first_bad_event_ids:
            retained = {event.event_id for event in incident.events[: cut_index + 1]}
            if retained & set(incident.first_bad_event_ids):
                return ScreenResult(False, "cut retains the certified fault", record)
        elif incident.certified_cut_index >= 0 and cut_index > incident.certified_cut_index:
            return ScreenResult(False, "cut retains the certified fault", record)
    if max_prefix_length is not None and cut_index + 1 > max_prefix_length:
        return ScreenResult(False, "cut lies beyond the observed runtime prefix", record)

    resolver = {
        _normalize_event_id(event.event_id): index
        for index, event in enumerate(incident.events)
    }
    for event in record.replacement_events:
        replaces = event.metadata.get("replaces")
        if not isinstance(replaces, list):
            continue
        for reference in replaces:
            if not isinstance(reference, str) or reference.startswith("patch:"):
                continue
            position = resolver.get(_normalize_event_id(reference))
            if position is None:
                return ScreenResult(
                    False, f"continuation cites an unknown event {reference}", record
                )
            if position <= cut_index:
                return ScreenResult(
                    False, f"continuation replaces retained event {reference}", record
                )
    for event in record.replacement_events:
        for reference in _cited_event_ids(event.content):
            position = resolver.get(_normalize_event_id(reference))
            if position is not None and position > cut_index:
                return ScreenResult(
                    False,
                    f"continuation depends on discarded event {reference}",
                    record,
                )
    if all(not (event.content or "").strip() for event in record.replacement_events):
        return ScreenResult(False, "continuation has no executable content", record)
    return ScreenResult(True, None, record)


_EVENT_ID_PATTERN = re.compile(r"\b([a-z_]+):event:(\d+)\b")


def _normalize_event_id(value: str) -> str:
    """Zero-pad an event index so ``source:event:006`` resolves like ``0006``."""

    match = _EVENT_ID_PATTERN.fullmatch(value.strip())
    if match is None:
        return value.strip()
    return f"{match.group(1)}:event:{int(match.group(2)):04d}"


def _cited_event_ids(content: str) -> list[str]:
    if not content:
        return []
    return [
        f"{namespace}:event:{index}"
        for namespace, index in _EVENT_ID_PATTERN.findall(content)
    ]


def _extract_json(text: str) -> Any | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```")[1] if "```" in stripped[3:] else stripped[3:]
        if stripped.lstrip().startswith("json"):
            stripped = stripped.lstrip()[4:]
    start = stripped.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(stripped)):
        character = stripped[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(stripped[start : index + 1])
                except json.JSONDecodeError:
                    # Keep scanning: a model can emit a truncated object before
                    # a later complete JSON record in the same completion.
                    break
    # Some instruction-tuned checkpoints prefix a valid object with a short
    # explanation or emit an earlier, truncated object before the final JSON.
    # Try every object boundary with the standard decoder before classifying the
    # sample as malformed; this remains conservative because only a complete
    # JSON object is accepted.
    decoder = json.JSONDecoder()
    for offset, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return value
    return None


# ---------------------------------------------------------------------------
# Equation (5): execution-grounded reward
# ---------------------------------------------------------------------------


def reward(outcome: ExecutionOutcome, config: GrpoConfig) -> float | None:
    """Return :math:`r(y \\mid x)` for one scored sample.

    ``None`` marks an unresolved evaluator or harness outcome, which Section 4.1
    counts separately from a task-incorrect repair and which therefore carries
    no gradient rather than a zero reward.
    """

    if outcome.status == UNRESOLVED:
        return None
    if outcome.status == INVALID:
        return config.invalid_reward
    accepted = 1.0 if outcome.status == ACCEPTED else 0.0
    harmful = config.harmful_penalty if outcome.harmful else 0.0
    overhead = config.cost_penalty * float(outcome.execution_count)
    return accepted - harmful - overhead


# ---------------------------------------------------------------------------
# Stage 3a: groups, diagnosis value, advantages
# ---------------------------------------------------------------------------


@dataclass
class RepairSample:
    """One sampled repair with its screen, execution, and reward."""

    text: str
    record: RepairRecord | None
    screen: ScreenResult
    outcome: ExecutionOutcome
    reward: float | None
    incident_id: str
    is_sibling: bool = False
    is_replay: bool = False
    replay_label: str | None = None
    advantage: float = 0.0
    update_input: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "is_sibling": self.is_sibling,
            "is_replay": self.is_replay,
            "replay_label": self.replay_label,
            "screened": self.screen.valid,
            "screen_reason": self.screen.reason,
            "outcome": self.outcome.to_dict(),
            "reward": self.reward,
            "advantage": self.advantage,
        }


@dataclass
class DiagnosisBranch:
    """One sampled diagnosis together with the repairs it produced."""

    text: str
    """The raw completion, which is what the policy update is scored on."""

    diagnosis: str = ""
    """The parsed diagnosis, which is what conditions the repair samples."""

    own_samples: list[RepairSample] = field(default_factory=list)
    sibling_samples: list[RepairSample] = field(default_factory=list)
    value: float | None = None
    advantage: float = 0.0
    update_instruction: str | None = None
    update_input: Mapping[str, Any] | None = None
    localization_update: tuple[str, Mapping[str, Any], str] | None = None

    def own_rewards(self) -> list[float]:
        return [
            sample.reward for sample in self.own_samples if sample.reward is not None
        ]

    def sibling_rewards(self) -> list[float]:
        return [
            sample.reward
            for sample in self.sibling_samples
            if sample.reward is not None
        ]


@dataclass
class RepairGroup:
    """The Stage 3a group for one incident: m diagnoses, each with n repairs."""

    incident_id: str
    cell_key: str
    prefix_length: int
    branches: list[DiagnosisBranch] = field(default_factory=list)
    dropped_reason: str | None = None
    model_input: Mapping[str, Any] | None = None
    """The context the group was sampled at, replayed when the batch updates."""

    @property
    def has_gradient(self) -> bool:
        return self.dropped_reason is None and any(
            branch.own_samples for branch in self.branches
        )

    def pass_at_k(self) -> dict[str, Any]:
        """pass@1 and pass@k over the group's own-incident repairs.

        Optimization against a binary reward can raise first-sample accuracy
        while narrowing the set of repairs the model can produce at all, so both
        are tracked and reported per round.
        """

        samples = [sample for branch in self.branches for sample in branch.own_samples]
        if not samples:
            return {"pass_at_1": None, "pass_at_k": None, "k": 0}
        first = [branch.own_samples[0] for branch in self.branches if branch.own_samples]
        accepted_first = sum(
            1 for sample in first if sample.outcome.status == ACCEPTED
        )
        accepted_any = any(sample.outcome.status == ACCEPTED for sample in samples)
        return {
            "pass_at_1": accepted_first / len(first) if first else None,
            "pass_at_k": 1.0 if accepted_any else 0.0,
            "k": len(samples),
        }

    def statistics(self) -> dict[str, Any]:
        rewards = [
            sample.reward
            for branch in self.branches
            for sample in branch.own_samples
            if sample.reward is not None
        ]
        return {
            **self.pass_at_k(),
            "incident_id": self.incident_id,
            "cell": self.cell_key,
            "prefix_length": self.prefix_length,
            "diagnoses": len(self.branches),
            "scored_repairs": len(rewards),
            "mean_reward": statistics.fmean(rewards) if rewards else None,
            "accepted": sum(
                1
                for branch in self.branches
                for sample in branch.own_samples
                if sample.outcome.status == ACCEPTED
            ),
            "invalid": sum(
                1
                for branch in self.branches
                for sample in branch.own_samples
                if sample.outcome.status == INVALID
            ),
            "unresolved": sum(
                1
                for branch in self.branches
                for sample in branch.own_samples
                if sample.outcome.status == UNRESOLVED
            ),
            "dropped_reason": self.dropped_reason,
        }


def diagnosis_value(branch: DiagnosisBranch, config: GrpoConfig) -> float | None:
    """Equation (6): the value of a diagnosis is the outcome of its repairs.

    The own-incident term is the mean reward of the repairs the diagnosis
    produced for this incident; the transfer term is the mean reward of the
    repairs it produced for family siblings drawn from *other* source cells, so
    a diagnosis that only fits the incident it was written for is not credited
    for transfer it does not have.  When no sibling was scored, the weight
    collapses onto the own-incident term rather than treating an unscored
    sibling as a zero.
    """

    own = branch.own_rewards()
    siblings = branch.sibling_rewards()
    if not own and not siblings:
        return None
    own_mean = statistics.fmean(own) if own else None
    sibling_mean = statistics.fmean(siblings) if siblings else None
    if sibling_mean is None:
        return own_mean
    if own_mean is None:
        return sibling_mean
    alpha = config.transfer_weight
    return (1.0 - alpha) * own_mean + alpha * sibling_mean


def standardize(
    values: Sequence[float], *, epsilon: float = 1e-6
) -> list[float]:
    """Return group-standardized advantages."""

    if not values:
        return []
    mean = statistics.fmean(values)
    deviation = statistics.pstdev(values) if len(values) > 1 else 0.0
    if deviation <= epsilon:
        return [0.0 for _ in values]
    return [(value - mean) / deviation for value in values]


def build_repair_group(
    incident: EnvironmentIncident,
    branches: Sequence[DiagnosisBranch],
    config: GrpoConfig,
    *,
    prefix_length: int,
    cell_baselines: Mapping[str, tuple[float, float]] | None = None,
) -> RepairGroup:
    """Assemble one group and credit its diagnoses and repairs.

    Diagnoses are credited by :math:`A^d_q = (V(d_q) - \\mu_i)/\\sigma_i` over
    the group's diagnosis values, and repairs by
    :math:`A^r_{q,j} = (r_{q,j} - \\bar r_q)/\\sigma_{i,q}` within their own
    branch, so the two fields do not share one advantage over the whole record.
    A group whose diagnosis values or whose repair rewards carry no spread is
    discarded, in line with the dynamic-sampling filter.
    """

    group = RepairGroup(
        incident_id=incident.incident_id,
        cell_key=incident.cell_key,
        prefix_length=prefix_length,
        branches=list(branches),
    )
    for branch in group.branches:
        branch.value = diagnosis_value(branch, config)

    scored_any = False
    for branch in group.branches:
        rewards = branch.own_rewards()
        if len(rewards) < 2:
            # A single scored repair has no within-branch baseline; the paper
            # standardizes within the branch, so it contributes no repair
            # gradient while still carrying its diagnosis advantage.
            for sample in branch.own_samples:
                sample.advantage = 0.0
            continue
        branch_mean = statistics.fmean(rewards)
        branch_spread = statistics.pstdev(rewards)
        if branch_spread <= config.min_group_std:
            for sample in branch.own_samples:
                sample.advantage = 0.0
            continue
        scored_any = True
        for sample in branch.own_samples:
            sample.advantage = (
                0.0
                if sample.reward is None
                else (sample.reward - branch_mean) / branch_spread
            )

    values = [branch.value for branch in group.branches if branch.value is not None]
    diagnosis_spread = (
        statistics.pstdev(values) if len(values) >= 2 else 0.0
    )
    if diagnosis_spread > config.min_group_std:
        mean = statistics.fmean(values)
        for branch in group.branches:
            branch.advantage = (
                0.0
                if branch.value is None
                else (branch.value - mean) / diagnosis_spread
            )
    else:
        for branch in group.branches:
            branch.advantage = 0.0

    if cell_baselines is not None and config.per_cell_standardization:
        baseline = cell_baselines.get(incident.cell_key)
        if baseline is not None:
            cell_mean, cell_spread = baseline
            if cell_spread > config.min_group_std:
                for branch in group.branches:
                    if branch.value is not None:
                        branch.advantage = (branch.value - cell_mean) / cell_spread

    if not scored_any and diagnosis_spread <= config.min_group_std:
        group.dropped_reason = "neither diagnoses nor repairs carry reward spread"
    return group


def cell_baselines(
    groups: Sequence[RepairGroup], *, epsilon: float = 1e-6
) -> dict[str, tuple[float, float]]:
    """Per-cell moments used to standardize a batch that mixes source cells."""

    by_cell: dict[str, list[float]] = defaultdict(list)
    for group in groups:
        for branch in group.branches:
            if branch.value is not None:
                by_cell[group.cell_key].append(branch.value)
    return {
        cell: (
            statistics.fmean(values),
            statistics.pstdev(values) if len(values) > 1 else 0.0,
        )
        for cell, values in by_cell.items()
        if values
    }


def select_transfer_siblings(
    incident: EnvironmentIncident,
    population: Sequence[EnvironmentIncident],
    config: GrpoConfig,
    *,
    rng: random.Random,
) -> list[EnvironmentIncident]:
    """Draw family siblings from source cells other than the incident's own.

    Equation~(6) requires :math:`c(\\nu_l(i)) \\neq c(i)`, so each sibling
    differs from the incident in its framework, its benchmark, or its host
    model.  When the family has no member outside the incident's cell the
    transfer term is simply unavailable and the value collapses onto the
    own-incident term.
    """

    candidates = [
        other
        for other in population
        if other.family_id == incident.family_id
        and other.incident_id != incident.incident_id
        and other.cell_key != incident.cell_key
    ]
    if not candidates:
        return []
    rng.shuffle(candidates)
    return candidates[: max(0, config.transfer_siblings)]


# ---------------------------------------------------------------------------
# Stage 3b: the inspection decision
# ---------------------------------------------------------------------------


@dataclass
class GateRollout:
    """One complete-task rollout under a sampled gate decision sequence."""

    incident_id: str
    seed: int
    trigger_index: int | None
    """The first event index at which the gate fired, or ``None`` for abstention."""

    decisions: list[tuple[int, float, bool, str]] = field(default_factory=list)
    """``(event index, sampled probability, fired, generated text)`` per prefix."""

    intervened_success: bool | None = None
    control_success: bool | None = None
    execution_delta: float = 0.0
    reward: float | None = None
    advantage: float = 0.0
    detail: str | None = None
    diagnosis: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "seed": self.seed,
            "trigger_index": self.trigger_index,
            "decisions": [
                {"prefix_length": probe, "sampling_probability": probability,
                 "fired": fired, "raw_response": text}
                for probe, probability, fired, text in self.decisions
            ],
            "intervened_success": self.intervened_success,
            "control_success": self.control_success,
            "execution_delta": self.execution_delta,
            "reward": self.reward,
            "advantage": self.advantage,
            "detail": self.detail,
            "diagnosis": self.diagnosis,
        }


@dataclass
class GateRolloutGroup:
    incident_id: str
    rollouts: list[GateRollout] = field(default_factory=list)
    dropped_reason: str | None = None

    @property
    def has_gradient(self) -> bool:
        return self.dropped_reason is None and any(
            abs(rollout.advantage) > 0.0 for rollout in self.rollouts
        )


def gate_reward(
    *,
    intervened_success: bool | None,
    control_success: bool | None,
    execution_delta: float,
    config: GrpoConfig,
) -> float | None:
    """Equation (7): the matched-seed difference in outcome.

    The subtrahend is the same task under the same seed with no intervention,
    which is computed once per task and seed and cached, so a rollout costs one
    execution.  Charging a trigger on a run that would have succeeded is exactly
    the harmful-intervention condition, so no separate penalty term appears.
    """

    if intervened_success is None or control_success is None:
        return None
    difference = float(intervened_success) - float(control_success)
    return difference - config.gate_cost_penalty * float(execution_delta)


def gate_rollout_advantages(
    group: GateRolloutGroup, config: GrpoConfig
) -> GateRolloutGroup:
    """Standardize a gate group, discarding one whose rollouts all return zero."""

    rewards = [
        rollout.reward for rollout in group.rollouts if rollout.reward is not None
    ]
    if len(rewards) < 2:
        group.dropped_reason = "fewer than two resolved rollouts"
        return group
    if all(abs(value) <= config.min_group_std for value in rewards):
        group.dropped_reason = "all rollouts returned zero reward"
        return group
    spread = statistics.pstdev(rewards)
    if spread <= config.min_group_std:
        group.dropped_reason = "rollout rewards carry no spread"
        return group
    mean = statistics.fmean(rewards)
    for rollout in group.rollouts:
        rollout.advantage = (
            0.0 if rollout.reward is None else (rollout.reward - mean) / spread
        )
    return group


def parse_gate_probability(text: str) -> float | None:
    payload = _extract_json(text)
    if not isinstance(payload, Mapping):
        return None
    value = payload.get("gate_probability")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        return None
    return value


# ---------------------------------------------------------------------------
# Domain randomization
# ---------------------------------------------------------------------------

_SURFACE_KEY_ALIASES = (
    {"event_id": "event_id", "role": "role", "kind": "kind", "content": "content"},
    {"event_id": "id", "role": "actor", "kind": "type", "content": "text"},
    {"event_id": "eid", "role": "speaker", "kind": "event_kind", "content": "body"},
)


def randomize_event_surface(
    model_input: Mapping[str, Any], *, rng: random.Random
) -> dict[str, Any]:
    """Randomize the surface form of the serialized event log.

    Each batch mixes source cells whose harnesses serialize events differently,
    so the field names carry a cell-identifying signal the policy should not
    depend on.  Only the surface changes: event identities, ordering, and
    content are preserved, so a target that references ``event_id`` values still
    resolves.
    """

    aliases = rng.choice(_SURFACE_KEY_ALIASES)
    if aliases["event_id"] == "event_id":
        return dict(model_input)
    events = model_input.get("events")
    if not isinstance(events, list):
        return dict(model_input)
    renamed = []
    for event in events:
        if not isinstance(event, Mapping):
            renamed.append(event)
            continue
        item = {
            aliases.get(key, key): value
            for key, value in event.items()
        }
        renamed.append(item)
    return {**model_input, "events": renamed, "event_schema": aliases}


def gate_trigger_prefix(
    incident: EnvironmentIncident, *, rng: random.Random
) -> int:
    """Draw a runtime prefix length from the gate's trigger distribution.

    A runtime intervention observes only :math:`\\tau_{\\leq t}`, and the gate
    fires at or after the certified fault, so the prefix is drawn from the
    events at and beyond the certified cut rather than uniformly over the
    trajectory.
    """

    total = len(incident.events)
    lower = max(1, incident.certified_cut_index + 1)
    if lower >= total:
        return total
    return rng.randint(lower, total)
