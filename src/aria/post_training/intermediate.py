"""Publish immutable, explicitly non-final LoRA checkpoints for early evaluation."""

import json
import shutil
from pathlib import Path


def save_intermediate(model, tokenizer, output_dir, *, stage, step, interval, parent, seed):
    if interval <= 0 or step <= 0 or step % interval:
        return None
    destination = Path(output_dir) / "intermediate" / f"step-{step:04d}"
    if destination.is_dir():
        return destination
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    target = model.module if hasattr(model, "module") else model
    target.save_pretrained(temporary, selected_adapters=["default"])
    tokenizer.save_pretrained(temporary)
    (temporary / "intermediate_metadata.json").write_text(json.dumps({
        "stage": stage, "optimizer_steps": step, "parent_checkpoint": str(parent),
        "seed": seed, "status": "intermediate_not_final", "optimizer_state_saved": False,
        "scope": "early evaluation or explicit warm start; not exact optimizer resume",
    }, indent=2) + "\n")
    temporary.rename(destination)
    return destination
