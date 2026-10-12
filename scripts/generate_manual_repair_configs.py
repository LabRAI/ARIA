#!/usr/bin/env python3
"""Generate direct single-cell repair configs from manual inference configs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from repair.generator import TRAJECTORY_REPLAY_OPERATOR_ID
from repair.schema import REPAIR_SCHEMA_VERSION

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FAMILIES = {
    "manual_api": "manual_api_repair",
    "manual_vllm": "manual_vllm_repair",
}
REPAIR_OPTION_KEYS = ("max_steps", "max_turns", "max_stalls", "native_timeout_seconds")
MESSAGES_REPAIR_CELLS = {
    ("ag2", "gaia"): True,
    ("ag2", "humaneval"): False,
    ("ag2", "olympiadbench"): True,
    ("metagpt", "gaia"): True,
    ("metagpt", "swe-bench-lite"): True,
    ("openmanus", "olympiadbench"): True,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate direct repair YAMLs for every manual inference cell."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if generated content differs; do not write files",
    )
    return parser.parse_args()


def load_source(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("job") != "inference":
        raise ValueError(f"expected inference config: {path}")
    experiments = raw.get("experiments")
    if not isinstance(experiments, list) or len(experiments) != 1:
        raise ValueError(f"expected exactly one inference experiment: {path}")
    experiment = experiments[0]
    if not isinstance(experiment, dict):
        raise TypeError(f"experiment must be a mapping: {path}")
    return raw, experiment


def repair_payload(source_path: Path, target_family: str) -> dict[str, Any]:
    raw, experiment = load_source(source_path)
    mas = str(experiment["mas"])
    benchmark = str(experiment["benchmark"])
    source_name = str(experiment["name"])
    repair_name = f"repair-v4-{source_name}"
    output_dir = Path(str(raw.get("output_dir") or "outputs"))
    source_batch_dir = output_dir / "inference" / mas / benchmark / source_name
    repair_root = output_dir / "repair" / mas / benchmark / repair_name
    source_api = raw.get("api")
    if not isinstance(source_api, dict):
        raise TypeError(f"api must be a mapping: {source_path}")
    api = dict(source_api)
    retry_invalid = False
    endpoint_c_retry = MESSAGES_REPAIR_CELLS.get((mas, benchmark))
    if target_family == "manual_api_repair" and endpoint_c_retry is not None:
        model = str(api["model"])
        api.update(
            {
                "runtime_id": f"runtime/endpoint-c-{model}",
                "provider": "claude-gateway",
                "protocol": "anthropic-messages",
                "base_url": "https://api.example.invalid",
                "api_key_env": "ANTHROPIC_API_KEY",
            }
        )
        retry_invalid = endpoint_c_retry
    runtime_slug = str(api.get("runtime_id") or "runtime/unknown").removeprefix(
        "runtime/"
    )

    spec: dict[str, Any] = {
        "name": repair_name,
        "schema_version": REPAIR_SCHEMA_VERSION,
        "source_batch_dir": source_batch_dir.as_posix(),
        "dataset_dir": (repair_root / "dataset").as_posix(),
        "repair_output_dir": (repair_root / "reruns").as_posix(),
        "split": "train",
        "sample_split": experiment.get("sample_split"),
        "rerun": True,
        "native": True,
        "resume": True,
        "on_sample_error": "continue",
        "retry_invalid": retry_invalid,
        "max_consecutive_endpoint_invalid": 3,
        **(
            {
                "capacity_wait_seconds": 300,
                "capacity_wait_max_attempts": 0,
            }
            if target_family == "manual_api_repair" and endpoint_c_retry is not None
            else {}
        ),
        "repair_trials": 1,
        "structured_output_retries": (
            8
            if target_family == "manual_api_repair" and endpoint_c_retry is not None
            else 2
        ),
        "attribution_max_events": 64,
        "gate_success_sample_rate": 0.25,
        "require_inference_completed": False,
        "require_trace_acceptance": True,
        "repair_operator_id": TRAJECTORY_REPLAY_OPERATOR_ID,
        "config_id": f"config/{repair_name}",
    }
    for key in REPAIR_OPTION_KEYS:
        if experiment.get(key) is not None:
            spec[key] = experiment[key]
    if isinstance(experiment.get("evaluation"), dict):
        evaluation = dict(experiment["evaluation"])
        if (
            target_family == "manual_api_repair"
            and endpoint_c_retry is not None
            and benchmark == "swe-bench-lite"
        ):
            evaluation["docker_platform"] = "linux/amd64"
        spec["evaluation"] = evaluation

    return {
        "job": "repair",
        "runtime": dict(raw.get("runtime") or {"kind": "endpoint"}),
        "data_dir": raw.get("data_dir", "data"),
        "external_dir": raw.get("external_dir", "external"),
        "output_dir": output_dir.as_posix(),
        "workspace_dir": (
            Path("workspace") / target_family / mas / benchmark / runtime_slug
        ).as_posix(),
        "output": {"existing_output_policy": "resume"},
        "aria": {"enabled": False, "mode": "off"},
        "api": api,
        "logging": dict(raw.get("logging") or {"level": "INFO"}),
        "prepare": dict(
            raw.get("prepare")
            or {
                "clone_mas": False,
                "download_benchmarks": False,
                "gaia_attachments": False,
            }
        ),
        "experiments": [],
        "repair_datasets": [spec],
    }


def render_payload(payload: dict[str, Any]) -> str:
    return yaml.safe_dump(payload, sort_keys=False, width=100)


def generated_configs(*, check: bool) -> list[Path]:
    mismatches: list[Path] = []
    generated: list[Path] = []
    for source_family, target_family in CONFIG_FAMILIES.items():
        source_root = PROJECT_ROOT / "configs" / source_family
        target_root = PROJECT_ROOT / "configs" / target_family
        for source_path in sorted(source_root.glob("*/*.yml")):
            relative = source_path.relative_to(source_root)
            target_path = target_root / relative
            content = render_payload(repair_payload(source_path, target_family))
            generated.append(target_path)
            if check:
                if (
                    not target_path.exists()
                    or target_path.read_text(encoding="utf-8") != content
                ):
                    mismatches.append(target_path)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(content, encoding="utf-8")
    if mismatches:
        raise RuntimeError(
            "manual repair configs are stale:\n- "
            + "\n- ".join(str(path.relative_to(PROJECT_ROOT)) for path in mismatches)
        )
    return generated


def main() -> None:
    args = parse_args()
    paths = generated_configs(check=args.check)
    action = "validated" if args.check else "generated"
    print(f"{action} {len(paths)} manual repair configs")


if __name__ == "__main__":
    main()
