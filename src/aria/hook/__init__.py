"""Public API for the deployable ARIA MAS hook."""

from aria.hook.component import (
    ActiveHookBridge,
    ActiveMasHost,
    AriaHook,
    HookRegistration,
    enact_intervention,
)
from aria.hook.controller import ActiveRepairController
from aria.hook.normalizer import (
    EventFieldMapping,
    EventNormalizer,
    MappingEventNormalizer,
)
from aria.hook.profiles import (
    ACTIVE_NATIVE_MAS,
    NATIVE_MAS_HOOK_PROFILES,
    get_mas_hook_profile,
    supports_realtime_mas,
)
from aria.hook.types import (
    FULL_REALTIME_CAPABILITIES,
    SHADOW_ONLY_CAPABILITIES,
    ActiveAction,
    ActiveIntervention,
    DeploymentMode,
    MasHookCapabilities,
    MasHookProfile,
    PauseStrategy,
)

__all__ = [
    "FULL_REALTIME_CAPABILITIES",
    "ACTIVE_NATIVE_MAS",
    "NATIVE_MAS_HOOK_PROFILES",
    "SHADOW_ONLY_CAPABILITIES",
    "ActiveAction",
    "ActiveHookBridge",
    "ActiveIntervention",
    "ActiveMasHost",
    "ActiveRepairController",
    "AriaHook",
    "HookRegistration",
    "DeploymentMode",
    "EventFieldMapping",
    "EventNormalizer",
    "MappingEventNormalizer",
    "MasHookCapabilities",
    "MasHookProfile",
    "PauseStrategy",
    "enact_intervention",
    "get_mas_hook_profile",
    "supports_realtime_mas",
]
