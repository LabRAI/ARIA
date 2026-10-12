"""Internal native replay worker; experiments are still launched through main.py YAML."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from common.artifacts import atomic_write_json
from common.config import ApiConfig
from repair.rerun import run_native_repair_attempt


def main() -> None:
    request = json.loads(Path(sys.argv[1]).read_text())
    config = SimpleNamespace(api=ApiConfig(**request.pop("api_config")))
    request.pop("api_settings", None)
    cache = request.pop("swebench_cache_dir", None)
    if cache:
        request["sample"]["_swebench_cache_dir"] = cache
    workspace_mode = request.get("spec", {}).get("swebench_workspace_mode")
    if workspace_mode is not None:
        request["sample"]["_swebench_workspace_mode"] = workspace_mode
    for key in ("repo_root", "clone_path", "run_root"):
        request[key] = Path(request[key])
    native = run_native_repair_attempt(config=config, **request)
    atomic_write_json(request["run_root"] / "native_replay_result.json", asdict(native))


if __name__ == "__main__":
    main()
