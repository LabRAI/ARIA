"""Dependency isolation for generated code, not a filesystem security sandbox."""

from __future__ import annotations

import os
from pathlib import Path
import sysconfig
from types import SimpleNamespace
from typing import Mapping
import venv


def prepare_code_environment(run_root: Path, *, shared_site_packages: Path | None = None) -> SimpleNamespace:
    """Keep task installs local while exposing existing host dependencies."""
    directory = run_root.resolve() / "code_environment"
    if directory.is_symlink():
        raise ValueError("task code environment must not be a symbolic link")
    builder = venv.EnvBuilder(with_pip=True, symlinks=os.name != "nt", clear=True)
    builder.create(directory)
    builder.clear = False
    context = builder.ensure_directories(directory)
    host_packages = (shared_site_packages or Path(sysconfig.get_path("purelib"))).resolve()
    if any(char in str(host_packages) for char in "\r\n"):
        raise ValueError("host site-packages path cannot contain newlines")
    scheme = "nt_venv" if os.name == "nt" else "posix_venv"
    packages = Path(sysconfig.get_path("purelib", scheme=scheme, vars={
        "base": str(directory), "platbase": str(directory),
    }))
    (packages / "host_dependencies.pth").write_text(str(host_packages) + "\n", encoding="utf-8")
    return context


def code_process_environment(source: Mapping[str, str], context: SimpleNamespace) -> dict[str, str]:
    env = dict(source)
    for key in ("PYTHONHOME", "PYTHONPATH", "PIP_TARGET", "PIP_PREFIX", "PIP_USER"):
        env.pop(key, None)
    env.update(
        PATH=str(Path(context.bin_path).resolve()) + os.pathsep + env.get("PATH", os.defpath),
        VIRTUAL_ENV=str(Path(context.env_dir).resolve()),
        UV_PROJECT_ENVIRONMENT=str(Path(context.env_dir).resolve()),
        UV_PYTHON=str(Path(context.env_exe).absolute()),
        PYTHONNOUSERSITE="1",
        PIP_CONFIG_FILE=os.devnull,
        PIP_REQUIRE_VIRTUALENV="true",
    )
    return env
