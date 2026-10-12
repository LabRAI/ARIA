"""Verify a completed LoRA training phase before a later phase may use it."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml

from common.artifacts import atomic_write_json


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify(job: dict) -> dict:
    import torch
    from safetensors import safe_open

    root = Path(job["output_dir"])
    manifest_path = root / "post_training_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "completed" or manifest.get("repair_conditioning") != "cascade":
        raise ValueError("phase must be completed with cascade conditioning")
    if manifest.get("stages") != job["stages"]:
        raise ValueError("phase stages do not match the acceptance contract")
    for name, digest in (manifest.get("corpora_sha256") or {}).items():
        path = (manifest.get("corpora") or {}).get(name)
        if not path or file_sha256(Path(path)) != digest:
            raise ValueError(f"staged training corpus changed: {name}")
    if job.get("freeze_manifest"):
        freeze = json.loads(Path(job["freeze_manifest"]).read_text())
        snapshot = Path(freeze["snapshot"])
        for relative, digest in freeze["source_sha256"].items():
            if file_sha256(snapshot / relative) != digest:
                raise ValueError(f"frozen training source changed: {relative}")
    checked = []
    for stage in job["stages"]:
        checkpoint = root / "checkpoints" / stage / "checkpoint"
        if Path(manifest["checkpoints"][stage]).resolve() != checkpoint.resolve():
            raise ValueError("manifest references a different checkpoint")
        report = json.loads((checkpoint.parent / "training_report.json").read_text())
        if report.get("stage") != stage or report.get("materialized") is not True or report.get("checkpoint_kind") != "peft-adapter":
            raise ValueError("missing materialized LoRA training report")
        losses = [float(value) for values in report["losses"].values() for value in values]
        if not losses or not all(math.isfinite(value) for value in losses):
            raise ValueError("training losses must be nonempty and finite")
        if stage == "sft" and job.get("expected_source_sft_rows") is not None:
            evidence = (report.get("context_budget") or {}).get("sft_evidence") or {}
            if evidence.get("input_rows") != job["expected_source_sft_rows"]:
                raise ValueError("SFT did not use the complete declared source view")
            expected = evidence.get("kept", 0) * report.get("epochs", 0)
            if len(losses) != expected:
                raise ValueError("SFT stopped before completing every accepted training row")
        if stage in {"grpo_repair", "grpo_gate"}:
            grpo = report.get("stage3") or {}
            if grpo.get("environment") not in {"native", "native_by_cell"}:
                raise ValueError("formal GRPO must use a native execution environment")
            if grpo.get("ledger_replay_enabled") is not False or grpo.get("gradient_status") != "updated":
                raise ValueError("GRPO lacks non-ledger optimizer updates")
            if (grpo.get("reward_grounding") or {}).get("native-rerun", 0) <= 0:
                raise ValueError("GRPO has no native-grounded execution outcomes")
            actual = grpo.get("optimizer_steps", 0)
            expected = job.get("expected_grpo_updates", grpo.get("requested_optimizer_steps"))
            if actual <= 0 or (expected and actual != expected):
                raise ValueError("GRPO did not complete its declared optimizer budget")
        adapter = json.loads((checkpoint / "adapter_config.json").read_text())
        if adapter["r"] != job["lora_rank"] or adapter["base_model_name_or_path"] != job["foundation_model"]:
            raise ValueError("adapter backbone or rank differs from the requested model")
        path = checkpoint / "adapter_model.safetensors"
        nonzero_b, tensors = 0, 0
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                value = handle.get_tensor(key)
                if value.numel() == 0 or not bool(torch.isfinite(value).all()):
                    raise ValueError(f"empty or nonfinite LoRA tensor: {key}")
                tensors += 1
                if ".lora_B." in key and bool(value.ne(0).any()):
                    nonzero_b += 1
        if not nonzero_b:
            raise ValueError("all LoRA B matrices are still zero; no materialized update is established")
        digest = file_sha256(path)
        if stage != "sft":
            parent = Path(report.get("base_model", "")) / "adapter_model.safetensors"
            if not parent.is_file() or file_sha256(parent) == digest:
                raise ValueError("later phase has no changed adapter relative to its parent")
        checked.append({"stage": stage, "checkpoint": str(checkpoint), "loss_count": len(losses),
                        "tensor_count": tensors, "nonzero_lora_b": nonzero_b,
                        "adapter_sha256": digest})
    return {"name": job["name"], "status": "materialized_training_verified", "stages": checked,
            "qualification": "weight/training verification only; held-out performance not established"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", type=Path, required=True)
    args = parser.parse_args()
    spec = yaml.safe_load(args.config.read_text())
    report = {"jobs": [verify(job) for job in spec["jobs"]]}
    atomic_write_json(Path(spec["output_path"]), report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
