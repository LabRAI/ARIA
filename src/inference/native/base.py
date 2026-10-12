"""Shared contract for native MAS implementations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

NativeRunner = Callable[["NativeRunContext"], Any]


@dataclass(frozen=True)
class NativeRunContext:
    """Inputs shared by source inference and paired repair reruns."""

    repo_root: Path
    clone_path: Path
    run_root: Path
    workspace_path: Path
    sample: dict[str, Any]
    mas: Any
    benchmark: Any
    host_adapter: Any
    aria: Any
    options: Mapping[str, Any]
    api_settings: dict[str, Any]
    default_timeout_seconds: int

    @property
    def prompt(self) -> str:
        return str(self.sample["task"])


@dataclass(frozen=True)
class NativeMasBackend:
    """Registration and entrypoint for one maintained native MAS."""

    slug: str
    supported_protocols: tuple[str, ...]
    clone_marker: str
    artifact_prefix: str
    messages_filename: str
    trace_filename: str
    artifact_filenames: tuple[str, ...]
    launch_description: str
    notes: str
    run_native: NativeRunner

    @property
    def status(self) -> str:
        return "implemented"

    @property
    def implemented(self) -> bool:
        return True

    def validate_protocol(self, protocol: str) -> None:
        if protocol in self.supported_protocols:
            return
        supported = ", ".join(self.supported_protocols) or "none"
        raise ValueError(
            f"native MAS runner {self.slug} does not support protocol {protocol}; "
            f"supported protocols: {supported}"
        )

    def run(self, context: NativeRunContext) -> Any:
        self.validate_protocol(str(context.api_settings.get("protocol") or ""))
        return self.run_native(context)
