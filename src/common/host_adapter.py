"""Host adapter layer for attaching ARIA to baseline runtimes.

The adapter contract is intentionally small: baseline-specific logic maps native
execution steps into ARIA hook events without modifying the external baseline
repositories themselves.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol

from common.catalog import BenchmarkSpec, MasSpec


@dataclass(frozen=True)
class HostAction:
    turn: int
    role: str
    prompt: str
    system_message: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class BaselineRuntimeProfile:
    adapter_slug: str
    runtime_family: str
    pre_event: str
    post_event: str
    finalize_event: str
    transition_event: str
    artifact_kind: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AriaRuntimeLike(Protocol):
    """Structural ARIA surface required by host adapters."""

    memory: Any

    def on_transition(
        self,
        hook: str,
        turn: int,
        role: str,
        metadata: dict[str, Any],
    ) -> None: ...

    def before_action(
        self,
        hook: str,
        turn: int,
        role: str,
        prompt: str,
    ) -> tuple[str, dict[str, Any] | None]: ...

    def after_action(
        self,
        hook: str,
        turn: int,
        role: str,
        content: str,
    ) -> dict[str, Any] | None: ...

    def before_finalize(
        self,
        turn: int,
        role: str,
        prompt: str,
    ) -> tuple[str, dict[str, Any] | None]: ...


class HostAdapter(Protocol):
    """Minimal protocol for connecting a host runtime to ARIA."""

    slug: str

    def bootstrap(
        self,
        aria: AriaRuntimeLike,
        mas: MasSpec,
        benchmark: BenchmarkSpec,
        task: str,
        roles: list[str],
    ) -> None: ...

    def runtime_profile(self, mas: MasSpec, benchmark: BenchmarkSpec) -> BaselineRuntimeProfile: ...

    def build_action(
        self,
        mas: MasSpec,
        benchmark: BenchmarkSpec,
        role: str,
        task: str,
        shared_context: str,
        turn: int,
    ) -> HostAction: ...

    def before_action(
        self,
        aria: AriaRuntimeLike,
        action: HostAction,
    ) -> tuple[HostAction, dict[str, Any] | None]: ...

    def after_action(
        self,
        aria: AriaRuntimeLike,
        action: HostAction,
        content: str,
    ) -> dict[str, Any] | None: ...

    def before_finalize(
        self,
        aria: AriaRuntimeLike,
        mas: MasSpec,
        benchmark: BenchmarkSpec,
        transcript: list[dict[str, Any]],
    ) -> dict[str, Any] | None: ...


class GenericPaperRoleAdapter:
    """Default adapter for paper-style role sequencing used in this repo."""

    slug = "generic-paper-role"

    def __init__(self, profile: BaselineRuntimeProfile | None = None) -> None:
        self._profile = profile
        if profile is not None:
            self.slug = profile.adapter_slug

    def runtime_profile(self, mas: MasSpec, benchmark: BenchmarkSpec) -> BaselineRuntimeProfile:
        if self._profile is not None:
            return self._profile
        return BaselineRuntimeProfile(
            adapter_slug=self.slug,
            runtime_family="paper-role-runner",
            pre_event="before_agent_step",
            post_event="after_agent_step",
            finalize_event="before_finalize",
            transition_event="on_start",
            artifact_kind=f"{mas.slug}-role-transcript",
            notes=f"Normalized paper-role adapter for {mas.paper_name} on {benchmark.paper_name}.",
        )

    def bootstrap(
        self,
        aria: AriaRuntimeLike,
        mas: MasSpec,
        benchmark: BenchmarkSpec,
        task: str,
        roles: list[str],
    ) -> None:
        profile = self.runtime_profile(mas, benchmark)
        aria.memory.contract_memory.update(
            {
                "mas": mas.slug,
                "benchmark": benchmark.slug,
                "task": task,
                "roles": roles,
                "adapter": self.slug,
                "runtime_profile": profile.to_dict(),
            }
        )
        aria.memory.verification_memory["required_checks"] = [
            f"verify the {benchmark.paper_name} task before finalization"
        ]
        aria.on_transition(
            profile.transition_event,
            0,
            "system",
            {"mas": mas.slug, "benchmark": benchmark.slug, "runtime_profile": profile.to_dict()},
        )

    def build_action(
        self,
        mas: MasSpec,
        benchmark: BenchmarkSpec,
        role: str,
        task: str,
        shared_context: str,
        turn: int,
    ) -> HostAction:
        prompt = (
            f"MAS: {mas.paper_name}\n"
            f"Benchmark: {benchmark.paper_name}\n"
            f"Task sample:\n{task}\n\n"
            f"Previous team context:\n{shared_context or '[none]'}\n\n"
            f"As {role}, produce your concrete contribution. "
            "If you are a verifier or reviewer, identify whether the current answer is sufficient."
        )
        system_message = (
            f"You are the {role} in {mas.paper_name}. "
            "You must work on the benchmark task concretely and concisely."
        )
        return HostAction(
            turn=turn,
            role=role,
            prompt=prompt,
            system_message=system_message,
            metadata={
                "mas": mas.slug,
                "benchmark": benchmark.slug,
                "runtime_profile": self.runtime_profile(mas, benchmark).to_dict(),
            },
        )

    def before_action(
        self,
        aria: AriaRuntimeLike,
        action: HostAction,
    ) -> tuple[HostAction, dict[str, Any] | None]:
        patched_prompt, patch_record = aria.before_action(
            str(action.metadata["runtime_profile"]["pre_event"]),
            action.turn,
            action.role,
            action.prompt,
        )
        return (
            HostAction(
                turn=action.turn,
                role=action.role,
                prompt=patched_prompt,
                system_message=action.system_message,
                metadata=action.metadata,
            ),
            patch_record,
        )

    def after_action(
        self,
        aria: AriaRuntimeLike,
        action: HostAction,
        content: str,
    ) -> dict[str, Any] | None:
        return aria.after_action(
            str(action.metadata["runtime_profile"]["post_event"]),
            action.turn,
            action.role,
            content,
        )

    def before_finalize(
        self,
        aria: AriaRuntimeLike,
        mas: MasSpec,
        benchmark: BenchmarkSpec,
        transcript: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not transcript:
            return None
        profile = self.runtime_profile(mas, benchmark)
        _, record = aria.before_finalize(
            len(transcript),
            transcript[-1]["role"],
            f"{profile.finalize_event}: Finalize benchmark answer for {benchmark.paper_name}.",
        )
        return record


ADAPTER_REGISTRY: dict[str, HostAdapter] = {
    slug: GenericPaperRoleAdapter(profile)
    for slug, profile in {
        "metagpt": BaselineRuntimeProfile(
            "metagpt-sidecar",
            "spec-to-implementation",
            "before_metagpt_stage",
            "after_metagpt_stage",
            "before_metagpt_delivery",
            "on_metagpt_stage_start",
            "metagpt-stage-log",
            "Maps MetaGPT staged product pipeline to ARIA hooks.",
        ),
        "ag2": BaselineRuntimeProfile(
            "ag2-sidecar",
            "solver-coder-verifier",
            "before_ag2_exchange",
            "after_ag2_exchange",
            "before_ag2_verifier_termination",
            "on_ag2_turn_switch",
            "ag2-conversation-log",
            "Maps AG2 MathChat-style agent exchange to ARIA hooks.",
        ),
        "magentic-one": BaselineRuntimeProfile(
            "magentic-one-sidecar",
            "orchestrated-tool-team",
            "before_magentic_delegate",
            "after_magentic_delegate",
            "before_magentic_resolution",
            "on_magentic_handoff",
            "magentic-orchestration-log",
            "Maps Magentic-One delegation to ARIA hooks.",
        ),
        "openmanus": BaselineRuntimeProfile(
            "openmanus-sidecar",
            "planner-executor-verifier",
            "before_openmanus_step",
            "after_openmanus_step",
            "before_openmanus_verification",
            "on_openmanus_handoff",
            "openmanus-run-log",
            "Maps OpenManus planning/execution handoff to ARIA hooks.",
        ),
    }.items()
}


def get_host_adapter(mas_slug: str) -> HostAdapter:
    try:
        return ADAPTER_REGISTRY[mas_slug]
    except KeyError as exc:
        raise KeyError(f"host adapter is not registered: {mas_slug}") from exc
