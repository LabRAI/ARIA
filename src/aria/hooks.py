"""Compatibility imports for the deployable :mod:`aria.hook` package."""

from aria.hook import (
    ACTIVE_NATIVE_MAS,
    NATIVE_MAS_HOOK_PROFILES,
    ActiveHookBridge,
    AriaHook,
    MasHookProfile,
    get_mas_hook_profile,
    supports_realtime_mas,
)

__all__ = [
    "NATIVE_MAS_HOOK_PROFILES",
    "ACTIVE_NATIVE_MAS",
    "ActiveHookBridge",
    "AriaHook",
    "MasHookProfile",
    "get_mas_hook_profile",
    "supports_realtime_mas",
]
