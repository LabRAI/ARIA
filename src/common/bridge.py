"""Baseline-native bridge specifications for ARIA sidecar attachment.

These bridge specs describe how a repo-local launcher can attach ARIA to each
external baseline without modifying the external repository source code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from common.catalog import MasSpec
from common.host_adapter import get_host_adapter


@dataclass(frozen=True)
class BridgeSpec:
    mas_slug: str
    adapter_slug: str
    launch_entrypoint: str
    launch_cwd: str
    hook_transport: str
    injected_artifacts: tuple[str, ...]
    required_env: tuple[str, ...]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_bridge_spec(external_dir: Path, mas: MasSpec) -> BridgeSpec:
    adapter = get_host_adapter(mas.slug)
    clone_root = external_dir / "mas" / mas.clone_dir
    return BridgeSpec(
        mas_slug=mas.slug,
        adapter_slug=adapter.slug,
        launch_entrypoint=mas.launch_entrypoint,
        launch_cwd=str(clone_root),
        hook_transport="repo-local sidecar launcher with normalized hook callbacks",
        injected_artifacts=(
            "aria_runtime.json",
            "aria_event_log.jsonl",
            "aria_patch_history.json",
        ),
        required_env=("PYTHONPATH", "OPENAI_API_KEY", "UV_PROJECT_ENVIRONMENT"),
        notes=(
            "This bridge spec is implemented from the repo side only. "
            "It must not require edits inside the external baseline repository. "
            "Launch through uv-managed Python because plain python may fail under the repo's 3.11 requirement."
        ),
    )


def list_bridge_specs(external_dir: Path, specs: tuple[MasSpec, ...]) -> list[BridgeSpec]:
    return [build_bridge_spec(external_dir, spec) for spec in specs]
