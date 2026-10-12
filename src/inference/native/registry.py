"""Single dispatch registry for maintained native MAS implementations."""

from inference.native.ag2.runner import BACKEND as AG2_BACKEND
from inference.native.base import NativeMasBackend
from inference.native.magentic_one.runner import BACKEND as MAGENTIC_ONE_BACKEND
from inference.native.metagpt.runner import BACKEND as METAGPT_BACKEND
from inference.native.openmanus.runner import BACKEND as OPENMANUS_BACKEND


NATIVE_MAS_BACKENDS = {
    backend.slug: backend
    for backend in (
        OPENMANUS_BACKEND,
        MAGENTIC_ONE_BACKEND,
        METAGPT_BACKEND,
        AG2_BACKEND,
    )
}


def get_native_mas_backend(slug: str) -> NativeMasBackend:
    try:
        return NATIVE_MAS_BACKENDS[slug]
    except KeyError as exc:
        raise NotImplementedError(f"native MAS adapter is not implemented: {slug}") from exc


def validate_native_mas_protocol(slug: str, protocol: str) -> NativeMasBackend:
    backend = get_native_mas_backend(slug)
    backend.validate_protocol(protocol)
    return backend
