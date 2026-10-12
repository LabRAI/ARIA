"""Runtime state and backward-compatible heuristic trace sidecar.

The publishable model-backed active implementation lives in :mod:`aria.hook`.
When a model is configured, native pre-delivery hooks own learned decisions and
this container disables the heuristic nudges. With no model it retains the old
detector-only workflow. This container itself does not perform host rollback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from aria.model import AriaModelConfig


class HookKind(str, Enum):
    PRE = "pre"
    POST = "post"
    TRANSITION = "transition"


class InterventionLevel(str, Enum):
    OBSERVE = "observe"
    NUDGE = "nudge"
    GUARD = "guard"
    REPAIR = "repair"
    ROLLBACK = "rollback"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class AriaConfig:
    enabled: bool = False
    mode: str = "shadow"
    max_history_events: int = 24
    max_pending_patches: int = 4
    repeat_threshold: int = 2
    verification_keywords: tuple[str, ...] = ("verify", "test", "check", "assert")
    finalization_keywords: tuple[str, ...] = ("final", "answer", "complete", "done")
    post_training: dict[str, Any] | None = None
    model: AriaModelConfig | None = None
    gate_threshold: float = 0.5
    fail_open: bool = True
    runtime_policy: str = "aria"
    runtime_policy_options: dict[str, Any] | None = None
    passive: dict[str, Any] | None = None


@dataclass(frozen=True)
class RuntimeEvent:
    kind: HookKind
    name: str
    turn: int
    role: str
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FailureReport:
    failure_modes: tuple[str, ...]
    status: str
    severity: str
    confidence: float
    evidence: tuple[str, ...]
    missing_invariants: tuple[str, ...]
    recommended_patch_families: tuple[str, ...]
    suggested_targets: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RuntimePatch:
    patch_type: str
    target: str
    timing: str
    intent: str
    payload: dict[str, Any]
    reversibility: str
    expected_effect: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyDecision:
    should_intervene: bool
    level: InterventionLevel
    rationale: str
    selected_patch: RuntimePatch | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.selected_patch is not None:
            payload["selected_patch"] = self.selected_patch.to_dict()
        return payload


@dataclass
class RuntimeMemory:
    contract_memory: dict[str, Any] = field(default_factory=dict)
    progress_memory: dict[str, Any] = field(default_factory=lambda: {"completed_steps": [], "open_blockers": []})
    fact_memory: dict[str, Any] = field(default_factory=lambda: {"shared_facts": []})
    grounding_memory: dict[str, Any] = field(default_factory=lambda: {"citations": []})
    verification_memory: dict[str, Any] = field(
        default_factory=lambda: {"required_checks": [], "completed_checks": [], "issues": []}
    )

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


def detect_failure(
    event: RuntimeEvent,
    memory: RuntimeMemory,
    recent_events: list[RuntimeEvent],
    config: AriaConfig,
) -> FailureReport | None:
    content = event.content.lower()
    failure_modes: list[str] = []
    evidence: list[str] = []
    missing_invariants: list[str] = []
    patch_families: list[str] = []

    repeated = _is_repeated_content(event, recent_events, config.repeat_threshold)
    if repeated:
        failure_modes.append("FM-1.3")
        evidence.append("current role output substantially repeats earlier content")
        missing_invariants.append("next_step_differs_from_previous_attempt")
        patch_families.append("progress_patch")

    if event.name == "before_finalize" and not _has_verification_signal(memory, recent_events, config):
        failure_modes.append("FM-3.2")
        evidence.append("finalization attempted without a recorded verification step")
        missing_invariants.append("verification_completed_before_termination")
        patch_families.append("verification_patch")

    if "assumption" in content and "share" not in content:
        failure_modes.append("FM-2.4")
        evidence.append("hidden assumption surfaced without any explicit sharing step")
        missing_invariants.append("shared_precondition_visible_to_executor")
        patch_families.append("communication_patch")

    if not failure_modes:
        return None

    severity = "high" if "FM-3.2" in failure_modes and event.name == "before_finalize" else "moderate"
    status = "at_risk" if severity == "moderate" else "blocking"
    confidence = 0.9 if severity == "high" else 0.72
    return FailureReport(
        failure_modes=tuple(dict.fromkeys(failure_modes)),
        status=status,
        severity=severity,
        confidence=confidence,
        evidence=tuple(evidence),
        missing_invariants=tuple(dict.fromkeys(missing_invariants)),
        recommended_patch_families=tuple(dict.fromkeys(patch_families)),
        suggested_targets=(event.role,),
    )


def generate_patches(report: FailureReport, event: RuntimeEvent, memory: RuntimeMemory) -> list[RuntimePatch]:
    patches: list[RuntimePatch] = []
    for family in report.recommended_patch_families:
        if family == "progress_patch":
            patches.append(
                RuntimePatch(
                    patch_type=family,
                    target=event.role,
                    timing="before_agent_step",
                    intent="force_non_redundant_next_step",
                    payload={
                        "completed_steps": memory.progress_memory.get("completed_steps", []),
                        "instruction": "State one new concrete step that differs from prior attempts.",
                    },
                    reversibility="safe",
                    expected_effect="reduce repeated planning or duplicated output",
                )
            )
        elif family == "verification_patch":
            patches.append(
                RuntimePatch(
                    patch_type=family,
                    target=event.role,
                    timing="before_finalize",
                    intent="require_explicit_verification",
                    payload={
                        "required_checks": memory.verification_memory.get("required_checks", [])
                        or ["verify benchmark-specific success criteria before final answer"],
                    },
                    reversibility="safe",
                    expected_effect="ensure verification occurs before termination",
                )
            )
        elif family == "communication_patch":
            patches.append(
                RuntimePatch(
                    patch_type=family,
                    target=event.role,
                    timing="before_agent_step",
                    intent="broadcast_missing_preconditions",
                    payload={"facts_to_share": memory.fact_memory.get("shared_facts", [])},
                    reversibility="safe",
                    expected_effect="surface hidden assumptions to downstream roles",
                )
            )
    return patches


def decide_policy(report: FailureReport, patches: list[RuntimePatch], config: AriaConfig) -> PolicyDecision:
    if config.mode == "off":
        return PolicyDecision(False, InterventionLevel.OBSERVE, "ARIA disabled by config")
    if not patches:
        return PolicyDecision(False, InterventionLevel.OBSERVE, "detected issue but no applicable patch")
    if config.mode == "shadow":
        return PolicyDecision(False, InterventionLevel.OBSERVE, "shadow mode logs patches without injection", patches[0])
    level = InterventionLevel.GUARD if report.severity == "high" else InterventionLevel.NUDGE
    return PolicyDecision(True, level, "inject minimal structured patch", patches[0])


def apply_patch_to_prompt(prompt: str, decision: PolicyDecision) -> str:
    if not decision.should_intervene or decision.selected_patch is None:
        return prompt
    patch = decision.selected_patch
    patch_block = (
        "\n\n[ARIA PATCH]\n"
        f"type: {patch.patch_type}\n"
        f"intent: {patch.intent}\n"
        f"expected_effect: {patch.expected_effect}\n"
        f"payload: {patch.payload}\n"
    )
    return prompt + patch_block


class AriaRuntime:
    """Hook-driven sidecar runtime attached around host events."""

    def __init__(self, config: AriaConfig) -> None:
        self.config = config
        self.memory = RuntimeMemory()
        self.events: list[RuntimeEvent] = []
        self.failure_reports: list[dict[str, Any]] = []
        self.patch_history: list[dict[str, Any]] = []
        self.pending_patch: RuntimePatch | None = None
        self.model_interventions: list[dict[str, Any]] = []

    def on_transition(self, name: str, turn: int, role: str, metadata: dict[str, Any] | None = None) -> None:
        event = RuntimeEvent(HookKind.TRANSITION, name, turn, role, metadata=metadata or {})
        self._append_event(event)

    def before_action(self, name: str, turn: int, role: str, prompt: str) -> tuple[str, dict[str, Any] | None]:
        event = RuntimeEvent(HookKind.PRE, name, turn, role, content=prompt)
        self._append_event(event)
        if self.config.model is not None:
            return prompt, None
        if self.pending_patch is None:
            return prompt, None
        decision = PolicyDecision(True, InterventionLevel.NUDGE, "apply queued patch", self.pending_patch)
        patched_prompt = apply_patch_to_prompt(prompt, decision)
        patch_record = {"event": event.name, "turn": turn, "role": role, "patch": self.pending_patch.to_dict()}
        self.patch_history.append(patch_record)
        self.pending_patch = None
        return patched_prompt, patch_record

    def after_action(self, name: str, turn: int, role: str, content: str) -> dict[str, Any] | None:
        event = RuntimeEvent(HookKind.POST, name, turn, role, content=content)
        self._append_event(event)
        self._update_memory(event)
        if self.config.model is not None:
            return None
        report = detect_failure(event, self.memory, self.events[-self.config.max_history_events :], self.config)
        if report is None:
            return None
        patches = generate_patches(report, event, self.memory)
        decision = decide_policy(report, patches, self.config)
        if decision.should_intervene and decision.selected_patch is not None:
            self.pending_patch = decision.selected_patch
        report_record = {
            "event": event.name,
            "turn": turn,
            "role": role,
            "report": report.to_dict(),
            "policy": decision.to_dict(),
        }
        self.failure_reports.append(report_record)
        return report_record

    def before_finalize(self, turn: int, role: str, prompt: str) -> tuple[str, dict[str, Any] | None]:
        event = RuntimeEvent(HookKind.PRE, "before_finalize", turn, role, content=prompt)
        if self.config.model is not None:
            self._append_event(event)
            return prompt, None
        report = detect_failure(event, self.memory, self.events[-self.config.max_history_events :], self.config)
        if report is None:
            self._append_event(event)
            return prompt, None
        patches = generate_patches(report, event, self.memory)
        decision = decide_policy(report, patches, self.config)
        record = {"event": "before_finalize", "turn": turn, "role": role, "report": report.to_dict(), "policy": decision.to_dict()}
        self.failure_reports.append(record)
        self._append_event(event)
        if decision.should_intervene:
            return apply_patch_to_prompt(prompt, decision), record
        return prompt, record

    def export_state(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "memory": self.memory.snapshot(),
            "failure_reports": self.failure_reports,
            "patch_history": self.patch_history,
            "event_count": len(self.events),
            "model_interventions": self.model_interventions,
        }

    def _append_event(self, event: RuntimeEvent) -> None:
        self.events.append(event)
        if len(self.events) > self.config.max_history_events * 4:
            self.events = self.events[-self.config.max_history_events * 4 :]

    def _update_memory(self, event: RuntimeEvent) -> None:
        text = event.content.strip()
        if not text:
            return
        self.memory.progress_memory.setdefault("completed_steps", [])
        self.memory.fact_memory.setdefault("shared_facts", [])
        self.memory.verification_memory.setdefault("completed_checks", [])
        if "done" in text.lower() or "completed" in text.lower():
            self.memory.progress_memory["completed_steps"].append({"turn": event.turn, "role": event.role, "text": text})
        if "fact:" in text.lower() or "constraint:" in text.lower():
            self.memory.fact_memory["shared_facts"].append({"turn": event.turn, "role": event.role, "text": text})
        if any(keyword in text.lower() for keyword in self.config.verification_keywords):
            self.memory.verification_memory["completed_checks"].append(
                {"turn": event.turn, "role": event.role, "text": text}
            )


def _has_verification_signal(
    memory: RuntimeMemory, recent_events: list[RuntimeEvent], config: AriaConfig
) -> bool:
    if memory.verification_memory.get("completed_checks"):
        return True
    return any(any(keyword in event.content.lower() for keyword in config.verification_keywords) for event in recent_events)


def _is_repeated_content(event: RuntimeEvent, recent_events: list[RuntimeEvent], threshold: int) -> bool:
    normalized = " ".join(event.content.lower().split())
    if not normalized:
        return False
    matches = 0
    for previous in recent_events[:-1]:
        if previous.role != event.role:
            continue
        if " ".join(previous.content.lower().split()) == normalized:
            matches += 1
        if matches >= threshold:
            return True
    return False
