"""Conservative capability profiles for maintained and future MAS hosts."""

from __future__ import annotations

from aria.hook.types import (
    SHADOW_ONLY_CAPABILITIES,
    MasHookCapabilities,
    MasHookProfile,
    PauseStrategy,
)

SUBSTITUTE_ONLY_CAPABILITIES = MasHookCapabilities(
    pause_strategy=PauseStrategy.BLOCKING_HOOK,
    substitute=True,
    rollback=False,
    resume=False,
    reexecute_tool_calls=False,
)


NATIVE_MAS_HOOK_PROFILES = {
    "langgraph": MasHookProfile(
        slug="langgraph", pre_delivery_hook="before_graph_node",
        post_delivery_hook="after_graph_node", capabilities=SUBSTITUTE_ONLY_CAPABILITIES,
    ),
    "openmanus": MasHookProfile(
        slug="openmanus",
        pre_delivery_hook="before_agent_step",
        post_delivery_hook="after_agent_step",
        capabilities=SUBSTITUTE_ONLY_CAPABILITIES,
    ),
    "metagpt": MasHookProfile(
        slug="metagpt",
        pre_delivery_hook="before_role_message",
        post_delivery_hook="after_role_message",
        capabilities=SUBSTITUTE_ONLY_CAPABILITIES,
    ),
    "magentic-one": MasHookProfile(
        slug="magentic-one",
        pre_delivery_hook="before_team_message",
        post_delivery_hook="after_team_message",
        capabilities=SUBSTITUTE_ONLY_CAPABILITIES,
    ),
    "ag2": MasHookProfile(
        slug="ag2",
        pre_delivery_hook="before_agent_message",
        post_delivery_hook="after_agent_message",
        capabilities=SUBSTITUTE_ONLY_CAPABILITIES,
    ),
}

# Native response hooks replace one plain-text message. Tool reexecution,
# multi-event resume and historical rollback require an explicit custom host.
ACTIVE_NATIVE_MAS = frozenset(NATIVE_MAS_HOOK_PROFILES)


def supports_realtime_mas(slug: str) -> bool:
    """Whether the repository has a tested active Hook integration profile."""

    return slug.strip().lower() in ACTIVE_NATIVE_MAS


def get_mas_hook_profile(slug: str) -> MasHookProfile:
    """Return a conservative profile; callers may supply proven capabilities."""

    normalized = slug.strip().lower()
    if not normalized:
        raise ValueError("MAS slug must be non-empty")
    return NATIVE_MAS_HOOK_PROFILES.get(
        normalized,
        MasHookProfile(
            slug=normalized,
            pre_delivery_hook="before_event_delivery",
            post_delivery_hook="after_event_delivery",
            capabilities=SHADOW_ONLY_CAPABILITIES,
        ),
    )
