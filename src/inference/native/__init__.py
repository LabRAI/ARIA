"""Public entrypoint for native MAS inference and repair execution."""

from inference.native.base import NativeMasBackend, NativeRunContext
from inference.native.registry import (
    NATIVE_MAS_BACKENDS,
    get_native_mas_backend,
    validate_native_mas_protocol,
)

__all__ = [
    "NATIVE_MAS_BACKENDS",
    "NativeMasBackend",
    "NativeRunContext",
    "get_native_mas_backend",
    "validate_native_mas_protocol",
]
