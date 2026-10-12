"""Import-time compatibility helpers for the pinned OpenManus checkout."""

from __future__ import annotations

import re
import tomllib
import warnings
from pathlib import Path
from typing import Any

_PLACEHOLDER_CONFIG = (
    # Real LLM settings are injected by the native runner after import.
    '[llm]\n'
    'model = "placeholder"\n'
    'base_url = "http://localhost:8000/v1"\n'
    'api_key = "not-required"\n'
    'api_type = "openai"\n'
    'api_version = ""\n'
    'max_tokens = 2048\n'
    'temperature = 0.0\n'
    '\n'
    '[daytona]\n'
    'daytona_api_key = "disabled"\n'
)


def ensure_openmanus_config(clone_path: Path) -> None:
    """Create the minimal secret-free config required by OpenManus imports."""

    config_path = clone_path / "config" / "config.toml"
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(_PLACEHOLDER_CONFIG, encoding="utf-8")
        return

    raw_config = _read_toml(config_path)
    daytona_config = raw_config.get("daytona")
    if isinstance(daytona_config, dict) and "daytona_api_key" in daytona_config:
        return
    if daytona_config is not None and not isinstance(daytona_config, dict):
        raise RuntimeError(f"invalid OpenManus [daytona] config in {config_path}")

    text = config_path.read_text(encoding="utf-8")
    if daytona_config is None:
        suffix = "" if text.endswith("\n") else "\n"
        config_path.write_text(
            f'{text}{suffix}\n[daytona]\ndaytona_api_key = "disabled"\n',
            encoding="utf-8",
        )
        return

    config_path.write_text(_insert_daytona_api_key(text), encoding="utf-8")


def suppress_openmanus_pydantic_v2_warnings() -> None:
    """Hide a known warning emitted by the pinned OpenManus checkout."""

    warnings.filterwarnings(
        "ignore",
        message=(
            r"Valid config keys have changed in V2:\n"
            r"\* 'underscore_attrs_are_private' has been removed"
        ),
        category=UserWarning,
        module=r"pydantic\._internal\._config",
    )


def _read_toml(config_path: Path) -> dict[str, Any]:
    try:
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise RuntimeError(f"invalid OpenManus config TOML at {config_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(
            f"invalid OpenManus config TOML at {config_path}: root must be a table"
        )
    return data


def _insert_daytona_api_key(text: str) -> str:
    pattern = re.compile(r"(?m)^\s*\[daytona\]\s*(?:#.*)?$")
    match = pattern.search(text)
    if match is None:
        suffix = "" if text.endswith("\n") else "\n"
        return f'{text}{suffix}\n[daytona]\ndaytona_api_key = "disabled"\n'
    insert_at = match.end()
    return f'{text[:insert_at]}\ndaytona_api_key = "disabled"{text[insert_at:]}'
