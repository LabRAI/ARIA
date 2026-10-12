"""Reference LoRA trainer for all three ARIA paper stages.

This backend intentionally keeps the optimization loop small and auditable. SFT
uses causal language-model loss. Preference objectives use the standard DPO
loss against frozen per-row reference log probabilities captured at stage start.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict, deque
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from aria.prompt_budget import CONTENT_CAPS as EVENT_CONTENT_CAPS
from aria.prompt_budget import reduced_model_input as _reduced_model_input
from aria.gate_scoring import label_probability
from aria.calibration import load_gate_calibration
from aria.training_balance import loss_weights

from aria.prompts import (
    canonical_model_json,
    instruction_for_task,
    ordered_training_output,
)

# Descending per-event content budgets used to fit an ARIA input to the token
# limit while every event identity stays addressable by the target.
_PROMPT_REDUCTIONS: Counter[Any] = Counter()
_BUDGET_REPORTS: dict[str, Any] = {}
_MIN_EVENT_CONTENT_CHARS = 0
_REQUIRE_VISIBLE_EVIDENCE = False


class PromptBudgetError(ValueError):
    """A complete structured prompt cannot fit the declared token budget."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("sft", "dpo", "transfer_diagnosis", "grpo_repair", "grpo_gate"),
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--datasets-json", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-model", default="")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--grpo-control-attempts", type=int, default=2)
    parser.add_argument("--grpo-checkpoint-interval", type=int, default=0)
    parser.add_argument("--optimizer-steps", type=int, default=0,
                        help="Fixed updates per supervised objective; 0 uses epochs. Rows cycle deterministically.")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--dpo-beta", type=float, default=0.1)
    parser.add_argument("--diagnosis-token-weight", type=float, default=16.0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=0,
        help="Cap the rendered prompt independently of --max-length; 0 lets the "
        "prompt use whatever the target leaves. ARIA targets vary in length by "
        "two orders of magnitude across task families, so a single budget "
        "either truncates a repair record or inflates a gate row to fill it.",
    )
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--row-weighting", choices=("none", "balanced_cell_task"), default="none")
    parser.add_argument("--max-row-weight-ratio", type=float, default=4.0)
    parser.add_argument("--min-event-content-chars", type=int, default=0)
    parser.add_argument("--require-visible-evidence", choices=("true", "false"), default="false")
    parser.add_argument("--dpo-sft-anchor", type=float, default=0.0)
    parser.add_argument("--dpo-label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--skip-oversized-targets",
        choices=("true", "false"),
        default="false",
        help="Drop rows whose target JSON cannot fit the context budget instead "
        "of failing. The count and the affected tasks are reported.",
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--cuda-memory-fraction", type=float, default=1.0)
    parser.add_argument(
        "--gradient-checkpointing",
        choices=("true", "false"),
        default="true",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="auto",
    )
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa", "flex_attention"),
        default="sdpa",
    )
    parser.add_argument("--flex-block-size", type=int, default=0)
    parser.add_argument("--target-only-logits", choices=("true", "false"), default="false")
    parser.add_argument("--transfer-candidates", type=int, default=4)
    parser.add_argument("--transfer-max-siblings", type=int, default=8)
    parser.add_argument("--transfer-max-new-tokens", type=int, default=96)
    parser.add_argument("--transfer-temperature", type=float, default=0.8)
    parser.add_argument("--transfer-uplift-threshold", type=float, default=0.0)
    parser.add_argument(
        "--transfer-family-similarity-threshold",
        type=float,
        default=0.2,
        help="Combined cosine threshold over training-only feedback embeddings and "
        "repair-edit features used to form Stage 2b sibling families.",
    )
    parser.add_argument(
        "--transfer-family-neighbors",
        type=int,
        default=3,
        help="Mutual nearest neighbors retained per incident when fitting Stage 2b families.",
    )
    parser.add_argument(
        "--transfer-min-uplift-std",
        type=float,
        default=0.0,
        help="Skip an incident whose candidate uplifts have a lower standard "
        "deviation than this; such a group yields a preference pair that "
        "carries no information.",
    )
    # Stage 3: transfer-grounded policy optimization.
    parser.add_argument(
        "--environment-spec",
        default="",
        help="JSON naming the Stage 3 execution backend, e.g. {\"kind\":\"ledger\"}.",
    )
    parser.add_argument(
        "--grpo-feedback-dropout",
        type=float,
        default=1.0,
        help="Probability of masking evaluator feedback in a Stage 3 context. "
        "The default masks it always, because a runtime intervention never "
        "observes an evaluator message.",
    )
    parser.add_argument("--grpo-diagnosis-oversample", type=int, default=2)
    parser.add_argument(
        "--grpo-batch-groups",
        type=int,
        default=4,
        help="Stage 3a groups accumulated before one optimizer step. A batch "
        "mixing source cells is what per-cell standardization needs, so this "
        "should exceed one whenever the corpus has more than one cell.",
    )
    parser.add_argument("--grpo-incidents", type=int, default=0)
    parser.add_argument("--grpo-throughput-after-updates", type=int, default=0)
    parser.add_argument("--grpo-require-full-batch", choices=("true", "false"), default="false")
    parser.add_argument("--grpo-optimizer-steps", type=int, default=0,
                        help="Require this many executed-reward optimizer updates, or reject the checkpoint.")
    parser.add_argument("--grpo-budget-passes", type=int, default=6,
                        help="Maximum fresh-policy passes over the selected train incidents for a fixed update budget.")
    parser.add_argument(
        "--grpo-max-attempts",
        type=int,
        default=6,
        help=(
            "Maximum policy-sampling attempts per Stage 3 group. Attempts "
            "that are malformed, unresolved, or have no reward spread are "
            "discarded and retained in the report; the first usable group is "
            "updated."
        ),
    )
    parser.add_argument(
        "--grpo-allow-no-gradient",
        choices=("true", "false"),
        default="false",
        help=(
            "Materialize a Stage 3 checkpoint when every sampled group is "
            "unresolved or dropped. The report retains the reason; this is "
            "useful for prompt-stripped ledgers that do not cover novel policy "
            "samples."
        ),
    )
    parser.add_argument(
        "--grpo-ledger-replay",
        choices=("true", "false"),
        default="false",
        help=(
            "Add the incident's recorded successful and failed splices to each "
            "repair group. This supplies explicitly audited offline reward "
            "support while policy samples still use the configured environment."
        ),
    )
    parser.add_argument("--grpo-diagnoses", type=int, default=4)
    parser.add_argument("--grpo-repairs", type=int, default=4)
    parser.add_argument("--grpo-transfer-siblings", type=int, default=1)
    parser.add_argument("--grpo-transfer-weight", type=float, default=0.5)
    parser.add_argument("--grpo-harmful-penalty", type=float, default=1.0)
    parser.add_argument("--grpo-cost-penalty", type=float, default=0.05)
    parser.add_argument("--grpo-invalid-reward", type=float, default=-1.0)
    parser.add_argument("--grpo-clip-low", type=float, default=0.2)
    parser.add_argument("--grpo-clip-high", type=float, default=0.28)
    parser.add_argument("--grpo-kl-coefficient", type=float, default=0.02)
    parser.add_argument("--grpo-max-new-tokens", type=int, default=320)
    parser.add_argument("--grpo-diagnosis-max-new-tokens", type=int, default=96)
    parser.add_argument("--grpo-temperature", type=float, default=1.0)
    parser.add_argument("--grpo-top-p", type=float, default=0.95)
    parser.add_argument("--grpo-gate-rollouts", type=int, default=4)
    parser.add_argument("--grpo-gate-threshold", type=float, default=0.5)
    parser.add_argument("--grpo-gate-max-probes", type=int, default=6)
    parser.add_argument("--grpo-gate-cost-penalty", type=float, default=0.05)
    parser.add_argument("--grpo-gate-learning-rate", type=float, default=0.05)
    parser.add_argument("--grpo-gate-probability-source", choices=("generated", "label_likelihood"), default="generated")
    parser.add_argument("--grpo-gate-policy", choices=("scalar", "contextual"), default="scalar")
    parser.add_argument("--grpo-gate-repair-chain", choices=("legacy", "deployment"), default="legacy")
    parser.add_argument("--grpo-repair-chain", choices=("legacy", "cascade"), default="legacy")
    parser.add_argument("--grpo-structured-retries", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--grpo-gate-anchor-path", default="")
    parser.add_argument("--grpo-gate-anchor-weight", type=float, default=0.05)
    parser.add_argument(
        "--grpo-gate-exploration",
        type=float,
        default=0.1,
        help=(
            "Training-only epsilon exploration for the stochastic gate. It "
            "prevents a Stage 1 gate initialized at zero from collapsing to "
            "permanent abstention; deployment uses the calibrated model without "
            "this exploration term."
        ),
    )
    parser.add_argument("--grpo-outer-rounds", type=int, default=1)
    parser.add_argument(
        "--grpo-randomize-event-surface",
        choices=("true", "false"),
        default="true",
    )
    parser.add_argument(
        "--grpo-runtime-prefix",
        choices=("true", "false"),
        default="true",
        help="Truncate contexts at prefixes drawn from the gate trigger distribution.",
    )
    parser.add_argument(
        "--standalone-checkpoint",
        choices=("true", "false"),
        default="false",
        help="Merge the adapter into the foundation weights for final delivery.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", choices=("auto", "mps", "cuda", "cpu"), default="auto"
    )
    return parser.parse_args()


def main() -> None:
    global _MIN_EVENT_CONTENT_CHARS, _REQUIRE_VISIBLE_EVIDENCE, _TARGET_ONLY_LOGITS
    args = parse_args()
    _TARGET_ONLY_LOGITS = args.target_only_logits == "true"
    if args.flex_block_size:
        if args.attention_implementation != "flex_attention":
            raise ValueError("flex-block-size requires flex_attention")
        from aria.attention import configure_flex_attention
        configure_flex_attention(args.flex_block_size)
    if args.min_event_content_chars < 0 or args.dpo_sft_anchor < 0 or not 0 <= args.dpo_label_smoothing < 0.5:
        raise ValueError("invalid evidence budget or DPO regularization")
    if args.grpo_gate_policy == "contextual" and (args.grpo_gate_probability_source != "label_likelihood"
            or not Path(args.grpo_gate_anchor_path).is_file() or args.grpo_gate_anchor_weight <= 0):
        raise ValueError("contextual gate requires label likelihood and a positive repair SFT anchor")
    _MIN_EVENT_CONTENT_CHARS = args.min_event_content_chars
    _REQUIRE_VISIBLE_EVIDENCE = args.require_visible_evidence == "true"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.max_prompt_tokens < 0 or (
        args.max_prompt_tokens and args.max_prompt_tokens > args.max_length
    ):
        raise ValueError(
            "--max-prompt-tokens must be non-negative and not exceed --max-length"
        )
    if (
        args.epochs <= 0
        or args.max_length <= 32
        or args.lora_rank <= 0
        or args.gradient_accumulation_steps <= 0
        or args.transfer_candidates < 2
        or args.transfer_max_siblings <= 0
    ):
        raise ValueError(
            "epochs, LoRA rank, accumulation, and sibling count must be positive; "
            "max-length must exceed 32 and transfer-candidates must be at least 2"
        )
    dataset_paths = _dataset_paths(args.datasets_json)
    random.seed(args.seed)

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    distributed = _initialize_distributed(torch, args.device)
    rank = distributed["rank"]
    world_size = distributed["world_size"]
    device = distributed["device"]
    if not 0.0 < args.cuda_memory_fraction <= 1.0:
        raise ValueError("--cuda-memory-fraction must be in (0, 1]")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(
            args.cuda_memory_fraction,
            device.index if device.index is not None else torch.cuda.current_device(),
        )
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    dtype = _resolve_dtype(torch, device, args.dtype)
    adapter_path = Path(args.base_model)
    if (adapter_path / "adapter_config.json").is_file():
        adapter_config = json.loads(
            (adapter_path / "adapter_config.json").read_text(encoding="utf-8")
        )
        foundation_model = str(adapter_config["base_model_name_or_path"])
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            fix_mistral_regex=True,
        )
        foundation = AutoModelForCausalLM.from_pretrained(
            foundation_model,
            dtype=dtype,
            attn_implementation=args.attention_implementation,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(
            foundation,
            args.base_model,
            is_trainable=True,
        )
    else:
        foundation_model = args.base_model
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model,
            fix_mistral_regex=True,
        )
        foundation = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            dtype=dtype,
            attn_implementation=args.attention_implementation,
            low_cpu_mem_usage=True,
        )
        model = get_peft_model(
            foundation,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_rank * 2,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=(
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ),
            ),
        )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.config.use_cache = False
    if args.gradient_checkpointing == "true":
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()

    mined_transfer = False
    if args.stage in {"grpo_repair", "grpo_gate"} and len(dataset_paths) != 1:
        raise ValueError("a Stage 3 run takes exactly one environment corpus")
    if args.stage == "transfer_diagnosis":
        if len(dataset_paths) != 2:
            raise ValueError("transfer_diagnosis requires DPO and SFT datasets")
        preference_rows = _read_rows(dataset_paths[0], args.max_rows)
        internalization_rows = _read_rows(dataset_paths[1], args.max_rows)
        if preference_rows and preference_rows[0].get("task") == "repair_preference":
            mined_transfer = True
            # Every rank mines deterministically on its local replica. This
            # avoids holding an NCCL collective open while rank zero performs
            # the comparatively long sibling-scoring pass.
            model.config.use_cache = True
            preference_rows, internalization_rows = _mine_transfer_diagnosis_rows(
                model,
                tokenizer,
                preference_rows,
                args,
                device,
                torch,
            )
            model.config.use_cache = False
            if rank == 0:
                _write_jsonl(
                    args.output_dir / "mined_transfer_diagnosis_dpo.jsonl",
                    preference_rows,
                )
                _write_jsonl(
                    args.output_dir / "mined_transfer_diagnosis_sft.jsonl",
                    internalization_rows,
                )

    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel

        model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=False,
        )
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate)

    stage_losses: dict[str, list[float]] = {}
    grpo_report: dict[str, Any] | None = None
    if args.stage == "sft":
        rows = _read_rows(dataset_paths[0], args.max_rows)
        stage_losses["sft"] = _train_sft(
            model,
            tokenizer,
            optimizer,
            rows,
            args,
            device,
            torch,
            rank=rank,
            world_size=world_size,
        )
    elif args.stage == "dpo":
        rows = _read_rows(dataset_paths[0], args.max_rows)
        stage_losses["dpo"] = _train_dpo(
            model,
            tokenizer,
            optimizer,
            rows,
            args,
            device,
            torch,
            rank=rank,
            world_size=world_size,
        )
    elif args.stage in {"grpo_repair", "grpo_gate"}:
        stage_losses[args.stage], grpo_report = _train_grpo(
            model,
            tokenizer,
            optimizer,
            dataset_paths[0],
            args,
            device,
            torch,
            rank=rank,
            world_size=world_size,
        )
    else:
        stage_losses["diagnosis_dpo"] = _train_dpo(
            model,
            tokenizer,
            optimizer,
            preference_rows,
            args,
            device,
            torch,
            rank=rank,
            world_size=world_size,
        )
        stage_losses["internalization_sft"] = _train_sft(
            model,
            tokenizer,
            optimizer,
            internalization_rows,
            args,
            device,
            torch,
            rank=rank,
            world_size=world_size,
        )

    if grpo_report is not None and rank == 0:
        (args.output_dir / "stage3_report.json").write_text(
            json.dumps(grpo_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    stage_losses = _gather_losses(stage_losses, torch, rank, world_size)
    if world_size > 1:
        torch.distributed.barrier()
    standalone_checkpoint = args.standalone_checkpoint == "true"
    unwrapped_model = model.module if hasattr(model, "module") else model
    checkpoint = args.output_dir / "checkpoint"
    if rank == 0:
        if standalone_checkpoint:
            unwrapped_model.to("cpu")
            unwrapped_model = unwrapped_model.merge_and_unload()
        unwrapped_model.generation_config.do_sample = False
        unwrapped_model.generation_config.temperature = None
        unwrapped_model.generation_config.top_p = None
        unwrapped_model.generation_config.top_k = None
        checkpoint.mkdir(parents=True, exist_ok=True)
        unwrapped_model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        if grpo_report is not None and isinstance(
            grpo_report.get("gate_calibration"), Mapping
        ):
            (checkpoint / "gate_calibration.json").write_text(
                json.dumps(
                    grpo_report["gate_calibration"], indent=2, sort_keys=True
                )
                + "\n",
                encoding="utf-8",
            )
    if world_size > 1:
        torch.distributed.barrier()
    report = {
        "materialized": True,
        "stage": args.stage,
        "base_model": args.base_model,
        "foundation_model": foundation_model,
        "reference_model": args.reference_model or None,
        "datasets": [str(path) for path in dataset_paths],
        "device": str(device),
        "dtype": str(dtype),
        "world_size": world_size,
        "distributed_backend": distributed["backend"],
        "epochs": args.epochs,
        "requested_optimizer_steps_per_objective": args.optimizer_steps,
        "target_only_logits": _TARGET_ONLY_LOGITS,
        "cuda_peak_memory_allocated": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "cuda_peak_memory_reserved": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
        "optimizer_steps": getattr(optimizer, "aria_update_count", 0),
        "max_length": args.max_length,
        "max_prompt_tokens": args.max_prompt_tokens,
        "cuda_memory_fraction": args.cuda_memory_fraction,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_checkpointing": args.gradient_checkpointing == "true",
        "diagnosis_token_weight": args.diagnosis_token_weight,
        "row_weighting": args.row_weighting,
        "max_row_weight_ratio": args.max_row_weight_ratio,
        "min_event_content_chars": args.min_event_content_chars,
        "require_visible_evidence": _REQUIRE_VISIBLE_EVIDENCE,
        "dpo_sft_anchor": args.dpo_sft_anchor,
        "dpo_label_smoothing": args.dpo_label_smoothing,
        "transfer_diagnosis_source": (
            "online-sibling-uplift" if mined_transfer else "precomputed"
        )
        if args.stage == "transfer_diagnosis"
        else None,
        "transfer_diagnosis_rows": (
            len(preference_rows) if args.stage == "transfer_diagnosis" else None
        ),
        "checkpoint_kind": ("standalone" if standalone_checkpoint else "peft-adapter"),
        "prompt_reductions": {
            str(key): value for key, value in sorted(_PROMPT_REDUCTIONS.items(), key=str)
        },
        "context_budget": dict(_BUDGET_REPORTS),
        "stage3": grpo_report,
        "losses": stage_losses,
        "final_loss": {
            name: losses[-1] if losses else None
            for name, losses in stage_losses.items()
        },
    }
    if rank == 0:
        (args.output_dir / "training_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if world_size > 1:
        torch.distributed.destroy_process_group()


PROGRESS_EVERY = int(os.environ.get("ARIA_PROGRESS_EVERY", "25"))


def _report_progress(
    stage: str,
    epoch: int,
    args: Any,
    index: int,
    total: int,
    losses: Sequence[float],
    *,
    rank: int,
) -> None:
    """Print a heartbeat so a long single-device run stays legible."""

    if rank != 0 or PROGRESS_EVERY <= 0:
        return
    if (index + 1) % PROGRESS_EVERY and index + 1 != total:
        return
    window = losses[-PROGRESS_EVERY:]
    mean = sum(window) / len(window) if window else float("nan")
    print(
        f"{stage} epoch {epoch + 1}/{args.epochs} step {index + 1}/{total} "
        f"loss[{len(window)}]={mean:.6f}",
        flush=True,
    )


def _filter_rows_to_budget(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    args: Any,
    *,
    sides: Sequence[str | None],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop rows whose target cannot fit the context budget, and say which.

    An ARIA target is a structured record whose ``event_id`` values must survive
    verbatim, so truncating one would train the model to emit a record that
    references nothing.  The alternative to truncation is exclusion, and an
    excluded row is reported rather than silently absorbed, since the context
    budget is part of what a backbone comparison holds fixed.
    """

    if args.skip_oversized_targets != "true":
        return rows, {"skipped": 0}
    minimum_prompt = min(256, args.max_length // 4)
    budget = args.max_length - minimum_prompt
    kept: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    longest = 0
    for row in rows:
        try:
            _, targets = _prompt_and_outputs(
                tokenizer, row, sides=sides, max_length=None
            )
        except (TypeError, ValueError) as exc:
            skipped[f"unrenderable:{type(exc).__name__}"] += 1
            continue
        length = max(
            len(tokenizer.encode(target + (tokenizer.eos_token or ""), add_special_tokens=False))
            for target in targets
        )
        longest = max(longest, length)
        if length > budget:
            skipped[str(row.get("task") or "unknown")] += 1
            continue
        kept.append(row)
    return kept, {
        "skipped": int(sum(skipped.values())),
        "skipped_by_task": dict(sorted(skipped.items())),
        "target_token_budget": budget,
        "longest_target_tokens": longest,
        "kept": len(kept),
        "input_rows": len(rows),
    }


def _supervised_epochs(rows, args, *, rank, world_size):
    """Keep the update budget identical even when tokenizers reject different rows."""
    budget = getattr(args, "optimizer_steps", 0)
    if budget < 0:
        raise ValueError("optimizer-steps must be nonnegative")
    if not rows:
        raise ValueError("cannot budget an empty training objective")
    completed, epoch = 0, 0
    while (completed < budget) if budget else (epoch < args.epochs):
        ordered = _distributed_epoch_rows(rows, seed=args.seed, epoch=epoch,
                                          rank=rank, world_size=world_size)
        if budget:
            ordered = ordered[:(budget - completed) * args.gradient_accumulation_steps]
        if not ordered:
            raise ValueError("training rank has no rows")
        yield epoch, ordered
        completed += math.ceil(len(ordered) / args.gradient_accumulation_steps)
        epoch += 1


def _train_sft(
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    rows: list[dict[str, Any]],
    args: Any,
    device: Any,
    torch: Any,
    *,
    rank: int,
    world_size: int,
) -> list[float]:
    if not rows:
        raise ValueError("SFT dataset contains no rows")
    rows, budget_report = _filter_rows_to_budget(
        tokenizer, rows, args, sides=(None,)
    )
    _BUDGET_REPORTS["sft"] = budget_report
    if _REQUIRE_VISIBLE_EVIDENCE:
        rows, evidence_report = _filter_visible_training_rows(tokenizer, rows, args)
        _BUDGET_REPORTS["sft_evidence"] = evidence_report
    if not rows:
        raise ValueError(
            "every SFT row exceeds the target token budget; raise --max-length"
        )
    losses: list[float] = []
    weights = loss_weights(rows, mode=getattr(args, "row_weighting", "none"),
                           max_ratio=getattr(args, "max_row_weight_ratio", 4.0))
    model.train()
    for epoch, ordered in _supervised_epochs(rows, args, rank=rank, world_size=world_size):
        epoch_loss_start = len(losses)
        optimizer.zero_grad(set_to_none=True)
        for index, row in enumerate(ordered):
            prompt, target = _prompt_and_output(
                tokenizer,
                row,
                preference_side=None,
                max_length=args.max_length,
                max_prompt_tokens=args.max_prompt_tokens,
            )
            batch = _tokenize_pair(
                tokenizer,
                prompt,
                target,
                args.max_length,
                device,
                torch,
                emphasized_prefix=_diagnosis_prefix(row),
                emphasized_weight=args.diagnosis_token_weight,
                max_prompt_tokens=args.max_prompt_tokens,
            )
            group_start = (
                index // args.gradient_accumulation_steps
            ) * args.gradient_accumulation_steps
            group_size = min(
                args.gradient_accumulation_steps,
                len(ordered) - group_start,
            )
            should_step = (
                index + 1
            ) % args.gradient_accumulation_steps == 0 or index + 1 == len(ordered)
            with _sync_context(model, should_step, world_size):
                output, scoring_batch = _causal_forward(model, batch)
                loss = _weighted_causal_loss(output.logits, scoring_batch, torch)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite SFT loss")
                (loss * weights[_row_key(row)] / group_size).backward()
            if should_step:
                _optimizer_step(model, optimizer, args, torch)
            losses.append(float(loss.detach().cpu()))
            _report_progress(
                "SFT", epoch, args, index, len(ordered), losses, rank=rank
            )
        if rank == 0:
            epoch_losses = losses[epoch_loss_start:]
            print(
                f"SFT epoch {epoch + 1}/{args.epochs}: "
                f"rows_per_rank={len(ordered)} mean_loss="
                f"{sum(epoch_losses) / len(epoch_losses):.6f}",
                flush=True,
            )
    return losses


def _train_dpo(
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    rows: list[dict[str, Any]],
    args: Any,
    device: Any,
    torch: Any,
    *,
    rank: int,
    world_size: int,
) -> list[float]:
    if not rows:
        raise ValueError("DPO dataset contains no rows")
    rows, budget_report = _filter_rows_to_budget(
        tokenizer, rows, args, sides=("chosen", "rejected")
    )
    _BUDGET_REPORTS.setdefault("dpo", budget_report)
    if _REQUIRE_VISIBLE_EVIDENCE:
        rows, evidence_report = _filter_visible_training_rows(tokenizer, rows, args, sides=("chosen", "rejected"))
        _BUDGET_REPORTS["dpo_evidence"] = evidence_report
    if not rows:
        raise ValueError(
            "every preference row exceeds the target token budget; raise --max-length"
        )
    references: dict[str, tuple[Any, Any]] = {}
    model.eval()
    reference = _ReferenceAdapter(
        model,
        args.reference_model or args.base_model,
    )
    for row in rows:
        prompt, (chosen, rejected) = _prompt_and_outputs(
            tokenizer,
            row,
            sides=("chosen", "rejected"),
            max_length=args.max_length,
            max_prompt_tokens=args.max_prompt_tokens,
        )
        chosen_batch = _tokenize_pair(
            tokenizer,
            prompt,
            chosen,
            args.max_length,
            device,
            torch,
            max_prompt_tokens=args.max_prompt_tokens,
        )
        rejected_batch = _tokenize_pair(
            tokenizer,
            prompt,
            rejected,
            args.max_length,
            device,
            torch,
            max_prompt_tokens=args.max_prompt_tokens,
        )
        reference_chosen, reference_rejected = reference.pair_log_probabilities(
            chosen_batch,
            rejected_batch,
            torch,
        )
        references[_row_key(row)] = (
            reference_chosen.detach(),
            reference_rejected.detach(),
        )
    losses: list[float] = []
    weights = loss_weights(rows, mode=getattr(args, "row_weighting", "none"),
                           max_ratio=getattr(args, "max_row_weight_ratio", 4.0))
    model.train()
    for epoch, ordered in _supervised_epochs(rows, args, rank=rank, world_size=world_size):
        epoch_loss_start = len(losses)
        optimizer.zero_grad(set_to_none=True)
        for index, row in enumerate(ordered):
            prompt, (chosen, rejected) = _prompt_and_outputs(
                tokenizer,
                row,
                sides=("chosen", "rejected"),
                max_length=args.max_length,
                max_prompt_tokens=args.max_prompt_tokens,
            )
            chosen_batch = _tokenize_pair(
                tokenizer,
                prompt,
                chosen,
                args.max_length,
                device,
                torch,
                max_prompt_tokens=args.max_prompt_tokens,
            )
            rejected_batch = _tokenize_pair(
                tokenizer,
                prompt,
                rejected,
                args.max_length,
                device,
                torch,
                max_prompt_tokens=args.max_prompt_tokens,
            )
            group_start = (
                index // args.gradient_accumulation_steps
            ) * args.gradient_accumulation_steps
            group_size = min(
                args.gradient_accumulation_steps,
                len(ordered) - group_start,
            )
            should_step = (
                index + 1
            ) % args.gradient_accumulation_steps == 0 or index + 1 == len(ordered)
            with _sync_context(model, should_step, world_size):
                policy_chosen, policy_rejected = _pair_log_probabilities(
                    model,
                    chosen_batch,
                    rejected_batch,
                    torch,
                )
                reference_chosen, reference_rejected = references[_row_key(row)]
                margin = (policy_chosen - policy_rejected) - (
                    reference_chosen - reference_rejected
                )
                smoothing = getattr(args, "dpo_label_smoothing", 0.0)
                loss = -(1 - smoothing) * torch.nn.functional.logsigmoid(args.dpo_beta * margin)
                loss -= smoothing * torch.nn.functional.logsigmoid(-args.dpo_beta * margin)
                anchor = getattr(args, "dpo_sft_anchor", 0.0)
                chosen_tokens = (chosen_batch["labels"][:, 1:] != -100).sum().clamp_min(1)
                loss -= anchor * policy_chosen / chosen_tokens
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite DPO loss")
                (loss * weights[_row_key(row)] / group_size).backward()
            if should_step:
                _optimizer_step(model, optimizer, args, torch)
            losses.append(float(loss.detach().cpu()))
            _report_progress(
                "DPO", epoch, args, index, len(ordered), losses, rank=rank
            )
        if rank == 0:
            epoch_losses = losses[epoch_loss_start:]
            print(
                f"DPO epoch {epoch + 1}/{args.epochs}: "
                f"rows_per_rank={len(ordered)} mean_loss="
                f"{sum(epoch_losses) / len(epoch_losses):.6f}",
                flush=True,
            )
    return losses


def _mine_transfer_diagnosis_rows(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    args: Any,
    device: Any,
    torch: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Mine diagnosis pairs with train-sibling and held-out-sibling uplift."""

    families = _cluster_transfer_families(
        rows,
        threshold=args.transfer_family_similarity_threshold,
        neighbors=args.transfer_family_neighbors,
    )
    eligible = {
        key: sorted(items, key=_row_key)
        for key, items in families.items()
        if len(items) >= 3
    }
    if not eligible:
        raise ValueError(
            "online transfer diagnosis requires a family with at least three "
            "causal repair rows"
        )

    model.eval()
    baseline_cache: dict[str, float] = {}
    preference_rows: list[dict[str, Any]] = []
    internalization_rows: list[dict[str, Any]] = []
    mining_stats: dict[str, dict[str, int]] = {}
    budget_rejections: list[dict[str, str]] = []
    for family_id, family_rows in sorted(eligible.items()):
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                f"Transfer diagnosis family {family_id}: {len(family_rows)} rows",
                flush=True,
            )
        family_stats = {
            "rows": len(family_rows),
            "candidate_count": 0,
            "scored_candidate_count": 0,
            "fewer_than_two_scored": 0,
            "no_eligible_candidate": 0,
            "accepted": 0,
        }
        mining_stats[family_id] = family_stats
        for row in family_rows:
            siblings = [candidate for candidate in family_rows if candidate is not row]
            rng = random.Random(_stable_seed(args.seed, _row_key(row)))
            rng.shuffle(siblings)
            heldout = siblings[0]
            train_siblings = siblings[1 : args.transfer_max_siblings + 1]
            if not train_siblings:
                continue
            try:
                diagnoses = _sample_diagnoses(
                    model,
                    tokenizer,
                    row,
                    args,
                    device,
                    torch,
                    sibling_diagnoses=_sibling_diagnoses(train_siblings),
                )
                family_stats["candidate_count"] += len(diagnoses)
                scored: list[tuple[str, float, float]] = []
                for diagnosis in diagnoses:
                    sibling_scores = [
                        _diagnosis_uplift(
                            model,
                            tokenizer,
                            sibling,
                            diagnosis,
                            args.max_length,
                            device,
                            torch,
                            baseline_cache,
                            args.max_prompt_tokens,
                        )
                        for sibling in train_siblings
                    ]
                    uplift = sum(sibling_scores) / len(sibling_scores)
                    heldout_uplift = _diagnosis_uplift(
                        model,
                        tokenizer,
                        heldout,
                        diagnosis,
                        args.max_length,
                        device,
                        torch,
                        baseline_cache,
                        args.max_prompt_tokens,
                    )
                    if math.isfinite(uplift) and math.isfinite(heldout_uplift):
                        scored.append((diagnosis, uplift, heldout_uplift))
            except PromptBudgetError as exc:
                family_stats["budget_rejected"] = family_stats.get("budget_rejected", 0) + 1
                budget_rejections.append({"record_id": _row_key(row), "family_id": family_id, "reason": str(exc)})
                continue
            family_stats["scored_candidate_count"] += len(scored)
            if len(scored) < 2:
                family_stats["fewer_than_two_scored"] += 1
                continue
            selection = _select_transfer_candidates(
                scored,
                uplift_threshold=args.transfer_uplift_threshold,
                min_uplift_std=args.transfer_min_uplift_std,
            )
            if selection is None:
                family_stats["no_eligible_candidate"] += 1
                continue
            chosen, rejected = selection
            preference, internalization = _transfer_training_pair(
                row,
                family_id=family_id,
                chosen=chosen,
                rejected=rejected,
                train_sibling_count=len(train_siblings),
                heldout_incident_id=_row_key(heldout),
            )
            preference_rows.append(preference)
            internalization_rows.append(internalization)
            family_stats["accepted"] += 1
    considered = sum(len(items) for items in eligible.values())
    _BUDGET_REPORTS["transfer_mining"] = {
        "input_rows": considered, "kept": considered - len(budget_rejections),
        "rejected": len(budget_rejections), "rejected_rows": budget_rejections,
        "contract": "prompt-budget admission before candidate selection; any unscorable sibling rejects the focal row",
    }
    _write_transfer_mining_report(
        args,
        family_stats=mining_stats,
        accepted_rows=len(preference_rows),
        budget_rejections=budget_rejections,
    )
    if not preference_rows:
        raise ValueError(
            "online transfer diagnosis found no candidate with non-negative "
            "training-sibling uplift and positive held-out-sibling uplift; see "
            f"{args.output_dir / 'transfer_mining_report.json'}"
        )
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            f"Accepted {len(preference_rows)} online transfer diagnosis pairs",
            flush=True,
        )
    return preference_rows, internalization_rows


def _select_transfer_candidates(
    scored: list[tuple[str, float, float]],
    *,
    uplift_threshold: float,
    min_uplift_std: float = 0.0,
) -> tuple[tuple[str, float, float], tuple[str, float, float]] | None:
    """Choose a held-out-valid diagnosis and a distinct DPO rejection.

    Following the dynamic-sampling intuition of DAPO, a candidate group whose
    uplifts carry no spread is skipped: the preference pair it would produce
    orders two candidates the scorer cannot tell apart.
    """

    if min_uplift_std > 0.0 and len(scored) > 1:
        values = [candidate[1] for candidate in scored]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        if variance**0.5 < min_uplift_std:
            return None
    eligible = [
        candidate
        for candidate in scored
        if candidate[1] >= uplift_threshold and candidate[2] > 0.0
    ]
    if not eligible:
        return None
    chosen = max(eligible, key=lambda item: (item[1], -len(item[0])))
    rejected_options = [candidate for candidate in scored if candidate[0] != chosen[0]]
    if not rejected_options:
        return None
    rejected = min(rejected_options, key=lambda item: (item[1], len(item[0])))
    return chosen, rejected


def _sibling_diagnoses(rows: list[dict[str, Any]]) -> list[str]:
    """Return unique verified diagnoses from training siblings only."""

    diagnoses: list[str] = []
    for row in rows:
        chosen = row.get("chosen")
        diagnosis = chosen.get("diagnosis") if isinstance(chosen, Mapping) else None
        if isinstance(diagnosis, str) and diagnosis.strip() and diagnosis not in diagnoses:
            diagnoses.append(diagnosis)
    return diagnoses


def _write_transfer_mining_report(
    args: Any,
    *,
    family_stats: Mapping[str, Mapping[str, int]],
    accepted_rows: int,
    budget_rejections: Sequence[Mapping[str, str]] = (),
) -> None:
    if int(os.environ.get("RANK", "0")) != 0:
        return
    report = {
        "accepted_rows": accepted_rows,
        "budget_rejections": list(budget_rejections),
        "transfer_candidates": args.transfer_candidates,
        "transfer_max_siblings": args.transfer_max_siblings,
        "transfer_uplift_threshold": args.transfer_uplift_threshold,
        "family_similarity_threshold": getattr(
            args, "transfer_family_similarity_threshold", 0.2
        ),
        "family_construction": "feedback-embedding-plus-repair-edit-components",
        "family_neighbors": getattr(args, "transfer_family_neighbors", 3),
        "families": family_stats,
    }
    (args.output_dir / "transfer_mining_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sample_diagnoses(
    model: Any,
    tokenizer: Any,
    row: Mapping[str, Any],
    args: Any,
    device: Any,
    torch: Any,
    *,
    sibling_diagnoses: list[str],
) -> list[str]:
    model_input = _row_model_input(row)
    chosen = row.get("chosen")
    if not isinstance(model_input, Mapping) or not isinstance(chosen, Mapping):
        raise TypeError("online transfer rows require input/prompt and chosen mappings")
    existing = chosen.get("diagnosis")
    candidates = [existing.strip()] if isinstance(existing, str) and existing.strip() else []
    repair_context = {
        "cut_event_id": chosen.get("cut_event_id"),
        "replacement_events": chosen.get("replacement_events"),
    }
    compact_input = _compact_transfer_input(model_input)
    prompt = _fitted_prompt(
        tokenizer,
        instruction_for_task("transfer_diagnosis_preference"),
        {**compact_input, "repair_context": repair_context},
        targets=[""],
        max_length=args.max_length - args.transfer_max_new_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    encoded = tokenizer(prompt, return_tensors="pt", truncation=False, add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    torch.manual_seed(_stable_seed(args.seed, _row_key(row)))
    generated_limit = max(args.transfer_candidates - len(candidates), 0)
    if generated_limit:
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=args.transfer_temperature,
                top_p=0.95,
                max_new_tokens=args.transfer_max_new_tokens,
                num_return_sequences=max(args.transfer_candidates * 2, 4),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_length = encoded["input_ids"].shape[1]
        for output in generated:
            text = tokenizer.decode(output[prompt_length:], skip_special_tokens=True)
            diagnosis = _parse_generated_diagnosis(text)
            if diagnosis and diagnosis not in candidates:
                candidates.append(diagnosis)
            if len(candidates) >= args.transfer_candidates:
                break
    for diagnosis in sibling_diagnoses:
        if diagnosis not in candidates:
            candidates.append(diagnosis)
    return candidates


def _diagnosis_uplift(
    model: Any,
    tokenizer: Any,
    row: Mapping[str, Any],
    diagnosis: str,
    max_length: int,
    device: Any,
    torch: Any,
    baseline_cache: dict[str, float],
    max_prompt_tokens: int = 0,
) -> float:
    row_key = _row_key(row)
    if row_key not in baseline_cache:
        baseline_cache[row_key] = _repair_context_log_probability(
            model,
            tokenizer,
            row,
            diagnosis=None,
            max_length=max_length,
            device=device,
            torch=torch,
            max_prompt_tokens=max_prompt_tokens,
        )
    conditioned = _repair_context_log_probability(
        model,
        tokenizer,
        row,
        diagnosis=diagnosis,
        max_length=max_length,
        max_prompt_tokens=max_prompt_tokens,
        device=device,
        torch=torch,
    )
    return conditioned - baseline_cache[row_key]


def _repair_context_log_probability(
    model: Any,
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    diagnosis: str | None,
    max_length: int,
    device: Any,
    torch: Any,
    max_prompt_tokens: int = 0,
) -> float:
    model_input = _row_model_input(row)
    chosen = row.get("chosen")
    if not isinstance(model_input, Mapping) or not isinstance(chosen, Mapping):
        raise TypeError("online transfer rows require input/prompt and chosen mappings")
    scoring_input = _compact_transfer_input(model_input)
    if diagnosis is not None:
        scoring_input["transfer_diagnosis"] = diagnosis
    target = json.dumps(
        ordered_training_output("structured_repair", chosen),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    prompt = _fitted_prompt(
        tokenizer, instruction_for_task("structured_repair"), scoring_input,
        targets=[target], max_length=max_length, max_prompt_tokens=max_prompt_tokens,
    )
    batch = _tokenize_pair(
        tokenizer,
        prompt,
        target,
        max_length,
        device,
        torch,
        max_prompt_tokens=max_prompt_tokens,
    )
    with torch.inference_mode():
        value = _sequence_log_probability(model, batch, torch)
    return float(value.detach().float().cpu())


def _transfer_training_pair(
    row: Mapping[str, Any],
    *,
    family_id: str,
    chosen: tuple[str, float, float],
    rejected: tuple[str, float, float],
    train_sibling_count: int,
    heldout_incident_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_input = _row_model_input(row)
    repair = row.get("chosen")
    if not isinstance(model_input, Mapping) or not isinstance(repair, Mapping):
        raise TypeError("online transfer rows require input/prompt and chosen mappings")
    incident_id = str(model_input.get("incident_id") or _row_key(row))
    incident_input = _transfer_incident_input(model_input)
    repair_context = {
        "cut_event_id": repair.get("cut_event_id"),
        "replacement_events": repair.get("replacement_events"),
    }
    metadata = row.get("metadata")
    shared = {
        "family_id": family_id,
        "uplift": chosen[1],
        "heldout_uplift": chosen[2],
        "metadata": dict(metadata) if isinstance(metadata, Mapping) else {},
    }
    preference = {
        "incident_id": incident_id,
        "task": "transfer_diagnosis_preference",
        "input": {**incident_input, "repair_context": repair_context},
        "chosen": {"diagnosis": chosen[0]},
        "rejected": {"diagnosis": rejected[0]},
        "rejected_uplift": rejected[1],
        "train_sibling_count": train_sibling_count,
        "heldout_incident_id": heldout_incident_id,
        **shared,
    }
    internalization_target = dict(repair)
    internalization_target["diagnosis"] = chosen[0]
    internalization = {
        "incident_id": incident_id,
        "task": "transfer_diagnosis_internalization",
        "input": incident_input,
        "target": internalization_target,
        **shared,
    }
    return preference, internalization


def _transfer_incident_input(model_input: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce a repair-preference input to the incident fields stage 3 trains on.

    ``repair_preference`` rows are conditioned on a validated localization and
    RA attribution.  Transfer diagnosis must generalize the mechanism from the
    incident alone, so those conditioning fields are dropped here exactly as
    :func:`aria.post_training.build_transfer_diagnosis_rows` does offline.
    """

    return {
        "incident_id": model_input.get("incident_id"),
        "events": model_input.get("events", []),
        "feedback": model_input.get("feedback"),
        "constraints": model_input.get("constraints", {}),
    }


def _compact_transfer_input(model_input: Mapping[str, Any]) -> dict[str, Any]:
    return _reduced_model_input(_transfer_incident_input(model_input), 2000)


def _cluster_transfer_families(
    rows: Sequence[dict[str, Any]],
    *,
    threshold: float,
    neighbors: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    """Fit Stage 2b families on training-only feedback and repair edits.

    The feedback representation is a sparse token-count embedding and the edit
    representation embeds the cut, replacement roles/kinds/tools, patch content,
    and projected repair code.  Connected components of their mean cosine form
    the neighborhoods; singleton components are retained here and excluded by
    the caller.  No validation/test row participates in fitting the graph.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("transfer family similarity threshold must be in [0, 1]")
    if neighbors < 1:
        raise ValueError("transfer family neighbors must be positive")
    ordered = sorted(rows, key=_row_key)
    feedback = [_feedback_embedding(row) for row in ordered]
    edits = [_repair_edit_embedding(row) for row in ordered]
    similarities: list[list[float]] = [
        [0.0 for _ in ordered] for _ in ordered
    ]
    for left in range(len(ordered)):
        for right in range(left + 1, len(ordered)):
            value = 0.5 * _sparse_cosine(
                feedback[left], feedback[right]
            ) + 0.5 * _sparse_cosine(edits[left], edits[right])
            similarities[left][right] = similarities[right][left] = value
    nearest: list[set[int]] = []
    for index in range(len(ordered)):
        ranked = sorted(
            (candidate for candidate in range(len(ordered)) if candidate != index),
            key=lambda candidate: (
                similarities[index][candidate],
                _row_key(ordered[candidate]),
            ),
            reverse=True,
        )
        nearest.append(
            {
                candidate
                for candidate in ranked[:neighbors]
                if similarities[index][candidate] >= threshold
            }
        )
    adjacency: list[set[int]] = [set() for _ in ordered]
    for left in range(len(ordered)):
        for right in nearest[left]:
            if left in nearest[right]:
                adjacency[left].add(right)
                adjacency[right].add(left)

    families: dict[str, list[dict[str, Any]]] = {}
    unseen = set(range(len(ordered)))
    while unseen:
        start = min(unseen)
        stack = [start]
        component: list[int] = []
        unseen.remove(start)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in sorted(adjacency[current]):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        identities = [_row_key(ordered[index]) for index in sorted(component)]
        digest = hashlib.sha256("\n".join(identities).encode()).hexdigest()[:12]
        families[f"family-{digest}"] = [ordered[index] for index in sorted(component)]
    return families


_FAMILY_TOKEN = re.compile(r"[a-z0-9_.:-]+")


def _feedback_embedding(row: Mapping[str, Any]) -> Counter[str]:
    model_input = _row_model_input(row)
    if not isinstance(model_input, Mapping):
        return Counter()
    return Counter(_FAMILY_TOKEN.findall(str(model_input.get("feedback") or "").lower()))


def _repair_edit_embedding(row: Mapping[str, Any]) -> Counter[str]:
    chosen = row.get("chosen")
    metadata = row.get("metadata")
    if not isinstance(chosen, Mapping):
        return Counter()
    features: Counter[str] = Counter()
    cut = chosen.get("cut_event_id")
    features[f"cut:{str(cut).rsplit(':', 1)[-1] if cut is not None else 'root'}"] += 2
    if isinstance(metadata, Mapping):
        # ``repair_operator_id`` names the single corpus-generation operator and
        # is constant across ARIA-Full; including it would connect unrelated
        # incidents into one giant component.  The RA code is incident-specific.
        for key in ("primary_code",):
            value = metadata.get(key)
            if value:
                features[f"{key}:{value}"] += 4
    for event in chosen.get("replacement_events") or []:
        if not isinstance(event, Mapping):
            continue
        features[f"role:{event.get('role')}"] += 2
        features[f"kind:{event.get('kind')}"] += 2
        for call in event.get("tool_calls") or []:
            if isinstance(call, Mapping):
                name = call.get("name") or (call.get("function") or {}).get("name")
                if name:
                    features[f"tool:{name}"] += 2
        for token in _FAMILY_TOKEN.findall(str(event.get("content") or "").lower()):
            features[f"content:{token}"] += 1
    return features


def _sparse_cosine(left: Mapping[str, int], right: Mapping[str, int]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(float(value) * float(right.get(key, 0)) for key, value in left.items())
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left.values()))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _row_model_input(row: Mapping[str, Any]) -> Any:
    return row.get("input", row.get("prompt"))


def _parse_generated_diagnosis(text: str) -> str | None:
    stripped = text.strip().removeprefix("```json").removeprefix("```")
    stripped = stripped.removesuffix("```").strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, Mapping):
            diagnosis = payload.get("diagnosis")
            if isinstance(diagnosis, str) and diagnosis.strip():
                return diagnosis.strip()
            # Small instruction-tuned checkpoints sometimes satisfy the
            # transfer prompt with a structured diagnosis object (for example
            # ``{"diagnosis": {"cause": ...}}``) even though the training
            # contract asks for natural language.  Preserve that model sample
            # as a deterministic, answer-blind string instead of discarding the
            # entire GRPO group before its execution reward can be observed.
            if isinstance(diagnosis, Mapping):
                fields = [
                    str(value).strip()
                    for value in diagnosis.values()
                    if isinstance(value, (str, int, float)) and str(value).strip()
                ]
                if fields:
                    return "; ".join(fields)
                serialized = json.dumps(
                    diagnosis, ensure_ascii=True, sort_keys=True, separators=(",", ":")
                )
                if serialized not in {"{}", "null"}:
                    return serialized
    # Keep malformed-but-nonempty generations auditable.  Stage 3's static
    # validator still decides whether the resulting repair is executable; a
    # parser failure should not erase the diagnosis branch before that reward
    # boundary is reached.
    fallback = text.strip()
    if fallback:
        return fallback[:4000]
    return None


def _chat_prompt(
    tokenizer: Any, instruction: str, model_input: Mapping[str, Any]
) -> str:
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": canonical_model_json(model_input)},
    ]
    if getattr(tokenizer, "chat_template", None):
        from aria.chat_templates import render_chat_prompt
        return render_chat_prompt(tokenizer, messages)
    return f"{instruction}\nINPUT:\n{canonical_model_json(model_input)}\nOUTPUT:\n"


def _optimizer_step(model: Any, optimizer: Any, args: Any, torch: Any) -> None:
    if args.max_grad_norm > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    optimizer.aria_update_count = getattr(optimizer, "aria_update_count", 0) + 1
    optimizer.zero_grad(set_to_none=True)


def _sync_context(model: Any, should_sync: bool, world_size: int) -> Any:
    if world_size > 1 and not should_sync:
        return model.no_sync()
    return nullcontext()


def _distributed_epoch_rows(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    epoch: int,
    rank: int,
    world_size: int,
) -> list[dict[str, Any]]:
    indices = list(range(len(rows)))
    random.Random(seed + epoch).shuffle(indices)
    if world_size > 1:
        total_size = math.ceil(len(indices) / world_size) * world_size
        indices.extend(indices[: total_size - len(indices)])
        indices = indices[rank:total_size:world_size]
    return [rows[index] for index in indices]


def _row_key(row: Mapping[str, Any]) -> str:
    for key in ("record_id", "incident_id"):
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    model_input = row.get("input", row.get("prompt"))
    if isinstance(model_input, Mapping):
        incident_id = model_input.get("incident_id")
        if isinstance(incident_id, str) and incident_id:
            return incident_id
    canonical = json.dumps(row, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def _gather_losses(
    losses: dict[str, list[float]],
    torch: Any,
    rank: int,
    world_size: int,
) -> dict[str, list[float]]:
    if world_size == 1:
        return losses
    gathered: list[Any] = [None] * world_size
    torch.distributed.all_gather_object(gathered, losses)
    if rank != 0:
        return losses
    merged: dict[str, list[float]] = defaultdict(list)
    for payload in gathered:
        for name, values in payload.items():
            merged[name].extend(values)
    return dict(merged)


def _sequence_log_probability(model: Any, batch: Mapping[str, Any], torch: Any) -> Any:
    output, scoring_batch = _causal_forward(model, batch)
    return _sequence_log_probabilities(output.logits, scoring_batch["labels"], torch).squeeze(0)


def _pair_log_probabilities(
    model: Any,
    chosen: Mapping[str, Any],
    rejected: Mapping[str, Any],
    torch: Any,
) -> tuple[Any, Any]:
    # A two-sequence padded batch materializes vocab logits for both long ARIA
    # continuations at once.  Scoring them sequentially is mathematically
    # identical and cuts the DPO/reference peak roughly in half.
    chosen_value = _sequence_log_probability(model, chosen, torch)
    rejected_value = _sequence_log_probability(model, rejected, torch)
    return chosen_value, rejected_value


def _token_log_probabilities(logits: Any, labels: Any, torch: Any) -> tuple[Any, Any]:
    """Return per-token log probabilities and their mask.

    Stage 3 optimizes a token-level objective, so the ratio and the KL term are
    formed per token rather than over the summed sequence log probability.
    """

    prefix = labels.shape[1] - logits.shape[1]
    if prefix < 0 or (prefix and bool((labels[:, 1:prefix + 1] != -100).any())):
        raise ValueError("logits suffix would discard a supervised causal target")
    logits = logits[:, :-1, :]
    labels = labels[:, prefix + 1:]
    mask = labels != -100
    token_ids = labels.masked_fill(~mask, 0)
    selected = torch.zeros_like(labels, dtype=torch.float32)
    if bool(mask.any()):
        # Only target positions contribute to any ARIA objective.  Selecting
        # those logits before the FP32 logsumexp avoids converting the entire
        # prompt-by-vocabulary tensor to FP32 (several GiB at a 4k context).
        target_logits = logits[mask].float()
        target_ids = token_ids[mask]
        selected[mask] = target_logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(
            -1
        ) - torch.logsumexp(target_logits, dim=-1)
    if prefix:
        # GRPO retains the original token coordinate system for ratios and KL.
        selected = torch.nn.functional.pad(selected, (prefix, 0))
        mask = torch.nn.functional.pad(mask, (prefix, 0), value=False)
    return selected, mask


def _sequence_log_probabilities(logits: Any, labels: Any, torch: Any) -> Any:
    selected, mask = _token_log_probabilities(logits, labels, torch)
    return (selected * mask).sum(dim=-1)


_TARGET_ONLY_LOGITS = False


def _causal_forward(model: Any, batch: Mapping[str, Any]):
    from aria.post_training.logits import causal_forward
    return causal_forward(model, batch, target_only=_TARGET_ONLY_LOGITS)


def _weighted_causal_loss(logits: Any, batch: Mapping[str, Any], torch: Any) -> Any:
    shifted_logits = logits[:, :-1, :]
    shifted_labels = batch["labels"][:, 1:]
    shifted_weights = batch["loss_weights"][:, 1:]
    mask = shifted_labels != -100
    token_ids = shifted_labels.masked_fill(~mask, 0)
    token_losses = torch.zeros_like(shifted_labels, dtype=torch.float32)
    if bool(mask.any()):
        target_logits = shifted_logits[mask].float()
        target_ids = token_ids[mask]
        token_losses[mask] = torch.logsumexp(target_logits, dim=-1) - target_logits.gather(
            -1, target_ids.unsqueeze(-1)
        ).squeeze(-1)
    weights = shifted_weights * mask
    return (token_losses * weights).sum() / weights.sum().clamp_min(1.0)


def _prompt_and_output(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    preference_side: str | None,
    max_length: int | None = None,
    max_prompt_tokens: int = 0,
) -> tuple[str, str]:
    prompt, targets = _prompt_and_outputs(
        tokenizer,
        row,
        sides=(preference_side,),
        max_length=max_length,
        max_prompt_tokens=max_prompt_tokens,
    )
    return prompt, targets[0]


def _prompt_and_outputs(
    tokenizer: Any,
    row: Mapping[str, Any],
    *,
    sides: Sequence[str | None],
    max_length: int | None = None,
    max_prompt_tokens: int = 0,
    fitted_inputs: list[dict[str, Any]] | None = None,
) -> tuple[str, list[str]]:
    """Render one shared prompt plus every requested target for a row.

    Preference rows must score chosen and rejected against an identical prompt,
    so the prompt is fitted once against the longest target.
    """

    task = str(row.get("task") or "training_ready_repair")
    model_input = row.get("input", row.get("prompt"))
    if not isinstance(model_input, Mapping):
        raise TypeError("ARIA trainer row input must be a mapping")
    targets: list[str] = []
    for side in sides:
        output = row.get("target", row.get("chosen")) if side is None else row.get(side)
        if not isinstance(output, Mapping):
            raise TypeError("ARIA trainer row output must be a mapping")
        targets.append(
            json.dumps(
                ordered_training_output(task, output),
                ensure_ascii=True,
                separators=(",", ":"),
            )
        )
    instruction = instruction_for_task(task)
    if (row.get("metadata") or {}).get("runtime_context") is True:
        from aria.prompts import RUNTIME_ATTRIBUTION_INSTRUCTION, RUNTIME_LOCALIZATION_INSTRUCTION
        if task == "failure_localization":
            instruction = RUNTIME_LOCALIZATION_INSTRUCTION
        elif task == "repair_attribution":
            instruction = RUNTIME_ATTRIBUTION_INSTRUCTION
    prompt = _fitted_prompt(
        tokenizer,
        instruction,
        model_input,
        targets=targets,
        max_length=max_length,
        max_prompt_tokens=max_prompt_tokens,
        fitted_inputs=fitted_inputs,
    )
    return prompt, targets


def _fitted_prompt(
    tokenizer: Any,
    instruction: str,
    model_input: Mapping[str, Any],
    *,
    targets: Sequence[str],
    max_length: int | None,
    max_prompt_tokens: int = 0,
    fitted_inputs: list[dict[str, Any]] | None = None,
) -> str:
    """Fit an ARIA input to the token budget without breaking its JSON.

    ARIA targets reference `event_id` values from the input, so the reduction
    shortens event content and task text while every event identity, role,
    kind, and tool-call name survives. An impossible budget is rejected, never
    repaired by slicing out JSON syntax or event identities.
    """

    prompt = _chat_prompt(tokenizer, instruction, model_input)
    if max_length is None and max_prompt_tokens <= 0:
        if fitted_inputs is not None:
            fitted_inputs.append(dict(model_input))
        return prompt
    target_tokens = (
        max(
            len(
                tokenizer.encode(
                    target + (tokenizer.eos_token or ""), add_special_tokens=False
                )
            )
            for target in targets
        )
        if targets
        else 0
    )
    budget = (max_length - target_tokens) if max_length is not None else max_prompt_tokens
    if max_prompt_tokens > 0:
        budget = min(budget, max_prompt_tokens)
    if budget <= 0:
        raise PromptBudgetError("ARIA targets leave no token budget for the structured prompt")
    if len(tokenizer.encode(prompt, add_special_tokens=False)) <= budget:
        if fitted_inputs is not None:
            fitted_inputs.append(dict(model_input))
        return prompt
    for cap in EVENT_CONTENT_CAPS:
        if cap < _MIN_EVENT_CONTENT_CHARS:
            continue
        reduced = _reduced_model_input(model_input, cap)
        candidate = _chat_prompt(tokenizer, instruction, reduced)
        if len(tokenizer.encode(candidate, add_special_tokens=False)) <= budget:
            _PROMPT_REDUCTIONS[cap] += 1
            if fitted_inputs is not None:
                fitted_inputs.append(reduced)
            return candidate
    _PROMPT_REDUCTIONS["overflow"] += 1
    raise PromptBudgetError(
        f"ARIA prompt identities and instructions exceed the {budget}-token budget; "
        "increase --max-length and --max-prompt-tokens"
    )


def _filter_visible_training_rows(tokenizer: Any, rows: list[dict[str, Any]], args: Any,
                                  *, sides: Sequence[str | None] = (None,)) -> tuple[list[dict[str, Any]], dict]:
    """Reject unlearnable examples, without inserting target evidence into inputs."""
    kept, rejected = [], []
    for row in rows:
        try:
            fitted = []
            _prompt_and_outputs(tokenizer, row, sides=sides, fitted_inputs=fitted,
                                max_length=args.max_length, max_prompt_tokens=args.max_prompt_tokens)
            evidence = []
            if row["task"] == "failure_localization":
                evidence.extend(row["target"].get("evidence") or [])
            model_input = row["input"]
            for key in ("localization", "attribution"):
                evidence.extend((model_input.get(key) or {}).get("evidence") or [])
            events = {e["event_id"]: e for e in fitted[0].get("events") or []}
            # Check the referenced event, not the quote echoed in conditioning.
            for item in evidence:
                quote = item.get("quote")
                if isinstance(quote, str) and quote and quote not in str(events.get(item.get("event_id"), {}).get("content", "")):
                    raise ValueError("grounding quote absent after input compaction")
        except (ValueError, TypeError) as exc:
            rejected.append({"record_id": _row_key(row), "task": row["task"], "reason": str(exc)})
        else:
            kept.append(row)
    return kept, {"input_rows": len(rows), "kept": len(kept), "rejected": len(rejected),
                  "rejected_rows": rejected, "contract": "no target-dependent cropping; no empty evidence fallback"}


def _tokenize_pair(
    tokenizer: Any,
    prompt: str,
    target: str,
    max_length: int,
    device: Any,
    torch: Any,
    *,
    emphasized_prefix: str | None = None,
    emphasized_weight: float = 1.0,
    max_prompt_tokens: int = 0,
) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(
        target + (tokenizer.eos_token or ""),
        add_special_tokens=False,
    )
    min_prompt_tokens = min(256, max_length // 4)
    target_budget = max_length - min_prompt_tokens
    if len(target_ids) > target_budget:
        raise ValueError(
            "ARIA target exceeds the configured token budget; target JSON must not "
            "be truncated"
        )
    prompt_budget = max_length - len(target_ids)
    if max_prompt_tokens > 0:
        prompt_budget = min(prompt_budget, max_prompt_tokens)
    if len(prompt_ids) > prompt_budget:
        raise ValueError("ARIA prompt exceeds its token budget; structured JSON must not be sliced")
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    loss_weights = [0.0] * len(prompt_ids) + [1.0] * len(target_ids)
    if emphasized_prefix:
        prefix_length = len(
            tokenizer.encode(emphasized_prefix, add_special_tokens=False)
        )
        for index in range(len(prompt_ids), len(prompt_ids) + prefix_length):
            loss_weights[index] = emphasized_weight
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long, device=device),
        "attention_mask": torch.ones(
            (1, len(input_ids)), dtype=torch.long, device=device
        ),
        "labels": torch.tensor([labels], dtype=torch.long, device=device),
        "loss_weights": torch.tensor(
            [loss_weights], dtype=torch.float32, device=device
        ),
    }


def _diagnosis_prefix(row: Mapping[str, Any]) -> str | None:
    if row.get("task") != "transfer_diagnosis_internalization":
        return None
    target = row.get("target")
    if not isinstance(target, Mapping) or not isinstance(target.get("diagnosis"), str):
        raise TypeError("transfer diagnosis SFT target requires diagnosis")
    encoded = json.dumps(
        {"diagnosis": target["diagnosis"]},
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return encoded[:-1] + ","


def _dataset_paths(raw: str) -> list[Path]:
    value = json.loads(raw)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) for item in value)
    ):
        raise TypeError("--datasets-json must encode a non-empty string list")
    paths = [Path(item) for item in value]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def _read_rows(path: Path, max_rows: int) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise TypeError(f"{path} must contain JSON objects")
    return rows[:max_rows] if max_rows > 0 else rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _initialize_distributed(torch: Any, requested_device: str) -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if requested_device not in {"auto", "cuda"}:
            raise ValueError("multi-GPU training requires --device auto or cuda")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "torchrun requested multiple ranks but CUDA is unavailable"
            )
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        return {
            "rank": rank,
            "world_size": world_size,
            "device": torch.device("cuda", local_rank),
            "backend": "nccl",
        }
    return {
        "rank": 0,
        "world_size": 1,
        "device": _resolve_device(torch, requested_device),
        "backend": None,
    }


def _resolve_dtype(torch: Any, device: Any, requested: str) -> Any:
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    if requested == "bfloat16":
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            raise RuntimeError("bfloat16 training requires a BF16-capable CUDA GPU")
        return torch.bfloat16
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def _resolve_device(torch: Any, requested: str) -> Any:
    if requested == "mps" or (
        requested == "auto" and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


# ===========================================================================
# Stage 3: transfer-grounded policy optimization
# ===========================================================================


def _select_grpo_incidents(
    incidents: Sequence[Any], limit: int, priority: Callable[[Any], int], *, seed: int,
    require_control: bool,
) -> list[Any]:
    """Round-robin source cells inside each execution-capability priority class."""
    if limit < 1:
        raise ValueError("GRPO incident limit must be positive")
    buckets: dict[tuple[int, bool], dict[str, deque[Any]]] = defaultdict(lambda: defaultdict(deque))
    for item in incidents:
        key = (int(priority(item)), require_control and item.matched_control_success is None)
        buckets[key][item.cell_key].append(item)
    selected = []
    rng = random.Random(seed)
    for key in sorted(buckets):
        cells = sorted(buckets[key])
        rng.shuffle(cells)
        while cells:
            pending = []
            for cell in cells:
                selected.append(buckets[key][cell].popleft())
                if len(selected) >= limit:
                    return selected
                if buckets[key][cell]:
                    pending.append(cell)
            cells = pending
    return selected


def _build_grpo_environment(spec: Any, *, run_root: Path) -> Any:
    from aria.post_training.environment import ChainEnvironment, build_environment
    from masfm.config import load_config

    if isinstance(spec, Mapping) and spec.get("kind") == "chain":
        return ChainEnvironment([
            _build_grpo_environment(member, run_root=run_root)
            for member in spec.get("backends", [])
        ])
    path = spec.get("experiment_config") if isinstance(spec, Mapping) else None
    config = load_config(Path(str(path))) if path else None
    return build_environment(spec, config=config, run_root=run_root, config_loader=load_config)


def _train_grpo(
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    environment_path: Path,
    args: Any,
    device: Any,
    torch: Any,
    *,
    rank: int,
    world_size: int,
) -> tuple[list[float], dict[str, Any]]:
    """Optimize Stage 3a or Stage 3b against executed rewards.

    Both stages sample from the current policy and score those samples by
    re-executing them in the environment retained from corpus construction, so
    the two share one loop: draw a group, screen and execute it, standardize the
    rewards into advantages, and take a clipped token-level GRPO step over the
    sampled tokens under a weak penalty toward the Stage 2 policy.
    """

    from aria.post_training.environment import (
        load_environment_incidents,
    )
    from aria.post_training.grpo import GrpoConfig

    config = GrpoConfig(
        harmful_penalty=args.grpo_harmful_penalty,
        cost_penalty=args.grpo_cost_penalty,
        invalid_reward=args.grpo_invalid_reward,
        transfer_weight=args.grpo_transfer_weight,
        diagnosis_group_size=args.grpo_diagnoses,
        repair_group_size=args.grpo_repairs,
        transfer_siblings=args.grpo_transfer_siblings,
        clip_low=args.grpo_clip_low,
        clip_high=args.grpo_clip_high,
        kl_coefficient=args.grpo_kl_coefficient,
        per_cell_standardization=True,
        randomize_event_surface=args.grpo_randomize_event_surface == "true",
        gate_group_size=args.grpo_gate_rollouts,
        gate_threshold=args.grpo_gate_threshold,
        gate_cost_penalty=args.grpo_gate_cost_penalty,
        outer_rounds=args.grpo_outer_rounds,
    )
    config.validate()
    update_budget = getattr(args, "grpo_optimizer_steps", 0)
    budget_passes = getattr(args, "grpo_budget_passes", 6)
    if update_budget < 0 or budget_passes < 1:
        raise ValueError("invalid GRPO optimizer budget")
    if update_budget and (world_size != 1 or config.outer_rounds != 1
                          or (args.stage == "grpo_gate" and getattr(args, "grpo_gate_policy", "scalar") != "contextual")):
        raise ValueError("fixed GRPO budgets require one rank, one outer round and contextual gate updates")
    initial_updates = getattr(optimizer, "aria_update_count", 0)
    if not 0.0 <= args.grpo_gate_exploration <= 1.0:
        raise ValueError("--grpo-gate-exploration must be between 0 and 1")
    if args.grpo_max_attempts < 1:
        raise ValueError("--grpo-max-attempts must be positive")

    environment_spec = json.loads(args.environment_spec) if args.environment_spec else None
    environment = _build_grpo_environment(
        environment_spec or {"kind": "ledger"},
        run_root=args.output_dir / "environment_runs",
    )
    population = load_environment_incidents(environment_path)
    incidents = [item for item in population if item.split == "train"]
    if any(item.requires_native_control for item in incidents) and getattr(environment, "name", "") not in {"native", "native_by_cell"}:
        raise ValueError("this training view requires fresh native controls; evaluator/ledger fallback is forbidden")
    if not incidents:
        raise ValueError("Stage 3 requires train incidents; held-out rows cannot be a fallback")
    incidents = sorted(incidents, key=lambda item: item.incident_id)
    rng = random.Random(args.seed)
    rng.shuffle(incidents)
    if world_size > 1:
        incidents = incidents[rank::world_size]
    priority = getattr(environment, "score_priority", lambda _: 0)
    capability = getattr(
        environment,
        "capability",
        lambda _: str(getattr(environment, "name", "unknown")),
    )
    report_selection = "environment-score-priority"
    if args.grpo_incidents > 0:
        eligible = sum(1 for item in incidents if int(priority(item)) < 100)
        incidents = _select_grpo_incidents(
            incidents, args.grpo_incidents, priority,
            seed=args.seed, require_control=args.stage == "grpo_gate",
        )
        report_selection = "environment-score-priority-cell-round-robin"
        if rank == 0:
            print(
                f"{args.stage}: {len(incidents)} incidents selected from a pool "
                f"of {len(population)} with {eligible} the environment can score",
                flush=True,
            )
    if not incidents:
        raise ValueError("the Stage 3 environment corpus has no usable incident")
    from aria.post_training.control_preparation import ControlPreparation
    controls = ControlPreparation(environment, args.output_dir,
                                  attempts=getattr(args, "grpo_control_attempts", 2))

    reference = _ReferenceAdapter(
        model,
        args.reference_model or args.base_model,
    )
    losses: list[float] = []
    gate_calibration = _load_gate_calibration(
        args.base_model, probability_source=args.grpo_gate_probability_source,
    )
    report: dict[str, Any] = {
        "stage": args.stage,
        "environment": getattr(environment, "name", "unknown"),
        "environment_spec": environment_spec or {"kind": "ledger"},
        "incident_pool": len(population),
        "incidents_used": len(incidents),
        "incident_selection": report_selection,
        "ledger_replay_enabled": args.grpo_ledger_replay == "true",
        "environment_capabilities": dict(
            Counter(str(capability(item)) for item in incidents)
        ),
        "outer_rounds": config.outer_rounds,
        "requested_optimizer_steps": update_budget,
        "budget_passes": budget_passes if update_budget else None,
        "alternation_contract": (
            "repair[0]"
            if args.stage == "grpo_repair"
            else "gate[0], then repair[r]+gate[r] for r >= 1"
        ),
        "reference_policy": reference.describe(),
        "groups": [],
        "reward_grounding": Counter(),
        "screen_rejections": Counter(),
        "outcome_status": Counter(),
    }

    cells = sorted({item.cell_key for item in incidents})
    args._grpo_live_report = report
    pending: list[tuple[Any, dict[str, Any]]] = []
    # The orchestrator materializes Stage 3a and Stage 3b as separate auditable
    # checkpoints.  Stage 3a performs the first repair round.  Stage 3b then
    # performs gate round 0 and, for every later outer round, alternates one
    # repair pass with one gate pass.  This realizes repair1, gate1, repair2,
    # gate2, ... rather than incorrectly doing all repair rounds before all gate
    # rounds.
    effective_rounds = 1 if args.stage == "grpo_repair" else config.outer_rounds
    for outer in range(effective_rounds):
        pending.clear()
        # One training cell is withheld from each round's reward and read only
        # for transfer, so no round optimizes against every cell it is later
        # asked to transfer to.  With a single cell there is nothing to withhold.
        withheld = cells[outer % len(cells)] if len(cells) > 1 else None
        round_incidents = [
            item for item in incidents if item.cell_key != withheld
        ] or incidents
        cycle_size = len(round_incidents)
        if update_budget:
            round_incidents = round_incidents * budget_passes
        report.setdefault("withheld_cells", []).append(
            {"round": outer, "cell": withheld, "incidents": len(round_incidents)}
        )
        if rank == 0 and withheld is not None:
            print(
                f"{args.stage} round {outer + 1}: withholding cell {withheld} "
                f"from the reward; {len(round_incidents)} incidents remain",
                flush=True,
            )
        if args.stage == "grpo_gate" and outer > 0:
            for repair_index, incident in enumerate(round_incidents):
                incident = controls.prepare(incident)
                if incident is None:
                    continue
                group, entry, dropped = _repair_group_with_retries(
                    model,
                    tokenizer,
                    incident,
                    population,
                    environment,
                    config,
                    args,
                    device,
                    torch,
                    seed=_stable_seed(args.seed + outer, incident.incident_id),
                    max_attempts=args.grpo_max_attempts,
                    report=report,
                )
                for rejected in dropped:
                    rejected.update({"round": outer, "phase": "repair"})
                    report["groups"].append(rejected)
                if group is None and getattr(args, "grpo_require_full_batch", "false") == "true":
                    report["groups"].append({**entry, "round": outer, "phase": "repair"})
                    continue
                pending.append((group, entry))
                is_last_repair = repair_index + 1 == len(round_incidents)
                if len(pending) < args.grpo_batch_groups and not is_last_repair:
                    continue
                repair_loss = _flush_repair_batch(
                    model,
                    tokenizer,
                    optimizer,
                    pending,
                    config,
                    args,
                    device,
                    torch,
                    reference=reference,
                )
                for _, buffered in pending:
                    buffered.update({"round": outer, "phase": "repair"})
                    report["groups"].append(buffered)
                    if rank == 0:
                        _print_group(args, outer, config, buffered, repair_loss)
                pending.clear()
                if repair_loss is not None:
                    losses.append(repair_loss)
        for index, incident in enumerate(round_incidents):
            if update_budget and getattr(optimizer, "aria_update_count", 0) - initial_updates >= update_budget:
                break
            _persist_grpo_progress(args, optimizer, initial_updates, index, len(round_incidents), controls, model, tokenizer)
            incident = controls.prepare(incident)
            if incident is None:
                continue
            sampling_id = (f"{incident.incident_id}:budget-pass:{index // cycle_size}"
                           if update_budget else incident.incident_id)
            if args.stage == "grpo_repair":
                group, entry, dropped = _repair_group_with_retries(
                    model,
                    tokenizer,
                    incident,
                    population,
                    environment,
                    config,
                    args,
                    device,
                    torch,
                    seed=_stable_seed(args.seed + outer, sampling_id),
                    max_attempts=args.grpo_max_attempts,
                    report=report,
                )
                for rejected in dropped:
                    rejected.update({"round": outer, "phase": "repair"})
                    report["groups"].append(rejected)
                if group is None and getattr(args, "grpo_require_full_batch", "false") == "true":
                    report["groups"].append({**entry, "round": outer, "phase": "repair"})
                    _persist_grpo_progress(args, optimizer, initial_updates, index + 1,
                                           len(round_incidents), controls, model, tokenizer)
                    continue
                pending.append((group, entry))
                is_last = index + 1 == len(round_incidents)
                if len(pending) < args.grpo_batch_groups and not is_last:
                    continue
                loss_value = _flush_repair_batch(
                    model,
                    tokenizer,
                    optimizer,
                    pending,
                    config,
                    args,
                    device,
                    torch,
                    reference=reference,
                )
                for _, buffered in pending:
                    report["groups"].append(
                        {**buffered, "round": outer, "phase": "repair"}
                    )
                    if rank == 0:
                        _print_group(args, outer, config, buffered, loss_value)
                pending.clear()
                if loss_value is not None:
                    losses.append(loss_value)
                _persist_grpo_progress(args, optimizer, initial_updates, index + 1, len(round_incidents), controls, model, tokenizer)
                continue
            else:
                loss_value, entry, dropped = _gate_step_with_retries(
                    model,
                    tokenizer,
                    optimizer,
                    incident,
                    environment,
                    config,
                    args,
                    device,
                    torch,
                    reference=reference,
                    seed=_stable_seed(args.seed + outer, sampling_id),
                    max_attempts=args.grpo_max_attempts,
                    report=report,
                    gate_calibration=gate_calibration,
                )
                for rejected in dropped:
                    rejected.update({"round": outer, "phase": "gate"})
                    report["groups"].append(rejected)
            entry["round"] = outer
            entry["phase"] = "gate"
            report["groups"].append(entry)
            if loss_value is not None:
                losses.append(loss_value)
            if rank == 0:
                _print_group(args, outer, config, entry, loss_value)
            _persist_grpo_progress(args, optimizer, initial_updates, index + 1, len(round_incidents), controls, model, tokenizer)

        # A skipped final control must not strand a partially filled batch.
        if pending:
            loss_value = _flush_repair_batch(model, tokenizer, optimizer, pending, config,
                                            args, device, torch, reference=reference)
            for _, buffered in pending:
                report["groups"].append({**buffered, "round": outer, "phase": "repair"})
                if rank == 0:
                    _print_group(args, outer, config, buffered, loss_value)
            pending.clear()
            if loss_value is not None:
                losses.append(loss_value)
            _persist_grpo_progress(args, optimizer, initial_updates, len(round_incidents), len(round_incidents), controls, model, tokenizer)

    report["native_controls"] = {"resolved": controls.resolved, "infrastructure_invalid": controls.unresolved}
    report["reward_grounding"] = dict(report["reward_grounding"])
    report["screen_rejections"] = dict(report["screen_rejections"])
    report["outcome_status"] = dict(report["outcome_status"])
    report["updated_groups"] = sum(1 for item in report["groups"] if item.get("updated"))
    report["dropped_groups"] = sum(
        1 for item in report["groups"] if not item.get("updated")
    )
    rewards = [
        value
        for item in report["groups"]
        for value in item.get("rewards", [])
        if value is not None
    ]
    report["mean_reward"] = sum(rewards) / len(rewards) if rewards else None
    report["scored_samples"] = len(rewards)
    report["pass_at_k_by_round"] = _pass_at_k_by_round(report["groups"])
    if args.stage == "grpo_gate":
        report["gate_calibration"] = {
            "schema_version": "aria-gate-calibration-v1",
            "logit_bias": gate_calibration["logit_bias"],
            "temperature": 1.0,
            "probability_source": args.grpo_gate_probability_source,
            "learning_rate": args.grpo_gate_learning_rate,
            "repair_parameters_frozen_during_gate_updates": getattr(args, "grpo_gate_policy", "scalar") == "scalar",
        }
        report["gate_policy"] = getattr(args, "grpo_gate_policy", "scalar")
        report["gate_repair_chain"] = getattr(args, "grpo_gate_repair_chain", "legacy")
    report["gradient_status"] = "updated" if losses else "no_gradient"
    report["optimizer_steps"] = getattr(optimizer, "aria_update_count", 0) - initial_updates
    if update_budget and report["optimizer_steps"] != update_budget:
        report["gradient_status"] = "incomplete_optimizer_budget"
        if rank == 0:
            (args.output_dir / "stage3_report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        raise RuntimeError(f"{args.stage} completed {report['optimizer_steps']} of {update_budget} required updates; checkpoint rejected")
    if not losses and args.grpo_allow_no_gradient != "true":
        if rank == 0:
            (args.output_dir / "stage3_report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        raise RuntimeError(
            f"{args.stage} produced no gradient: every group was dropped. "
            "See stage3_report.json for the per-group reason."
        )
    return losses, report


def _persist_grpo_progress(args, optimizer, initial_updates, processed, total, controls, model=None, tokenizer=None):
    import time
    from common.artifacts import atomic_write_json
    from aria.post_training.intermediate import save_intermediate
    steps = getattr(optimizer, "aria_update_count", 0) - initial_updates
    if model is not None:
        save_intermediate(model, tokenizer, args.output_dir, stage=args.stage, step=steps,
                          interval=getattr(args, "grpo_checkpoint_interval", 0),
                          parent=args.base_model, seed=args.seed)
    now = time.time()
    progress = {
        "stage": args.stage, "updated_unix": now, "started_unix": controls.started,
        "optimizer_steps": steps,
        "requested_optimizer_steps": getattr(args, "grpo_optimizer_steps", 0),
        "processed_incidents": processed, "scheduled_incidents": total,
        "resolved_controls": controls.resolved, "infrastructure_invalid_controls": controls.unresolved,
        "batch_groups": getattr(args, "grpo_batch_groups", None),
        "require_full_batch": getattr(args, "grpo_require_full_batch", "false") == "true",
        "eligible_groups_in_updates": getattr(args, "_grpo_eligible_groups", 0),
        "updated_benchmark_groups": dict(getattr(args, "_grpo_benchmark_groups", {})),
        "scored_execution_equivalents": getattr(args, "_grpo_live_report", {}).get("scored_execution_equivalents", 0.0),
        "loop_elapsed_seconds": now - controls.started,
        "updates_per_hour": steps * 3600 / max(now - controls.started, 1e-6),
    }
    atomic_write_json(args.output_dir / "grpo_progress.json", progress)
    if steps > getattr(args, "_last_journal_step", 0):
        progress["update_interval_seconds"] = now - getattr(args, "_last_journal_time", controls.started)
        with (args.output_dir / "grpo_updates.jsonl").open("a") as handle:
            handle.write(json.dumps(progress) + "\n")
        args._last_journal_step, args._last_journal_time = steps, now
    threshold = getattr(args, "grpo_throughput_after_updates", 0)
    if threshold > 0 and steps >= threshold and not getattr(args, "_throughput_recorded", False):
        atomic_write_json(args.output_dir / "throughput_ready.json", {
            **progress, "estimated_loop_seconds_for_budget":
                (now - controls.started) * getattr(args, "grpo_optimizer_steps", steps) / steps,
            "scope": "actual native-control, policy sampling, native scoring and optimizer wall time; not generated tokens/s",
            "full_checkpoint": False,
        })
        args._throughput_recorded = True


def _pass_at_k_by_round(groups: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Roll pass@1 and pass@k up per outer round.

    Optimizing against a binary reward can raise first-sample accuracy while
    narrowing what the model can produce at all, so the two are reported side by
    side rather than collapsed into one number.
    """

    by_round: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for group in groups:
        by_round[int(group.get("round", 0))].append(group)
    summary: list[dict[str, Any]] = []
    for index in sorted(by_round):
        entries = [
            group for group in by_round[index] if group.get("pass_at_1") is not None
        ]
        summary.append(
            {
                "round": index,
                "groups": len(by_round[index]),
                "scored_groups": len(entries),
                "pass_at_1": (
                    sum(float(item["pass_at_1"]) for item in entries) / len(entries)
                    if entries
                    else None
                ),
                "pass_at_k": (
                    sum(float(item["pass_at_k"]) for item in entries) / len(entries)
                    if entries
                    else None
                ),
            }
        )
    return summary


class _ReferenceAdapter:
    """Expose :math:`\\pi_{\theta_1}` without holding a second foundation model.

    Stage 3 penalizes drift from the Stage 2 policy.  When Stage 3 starts from a
    LoRA checkpoint the same adapter is loaded a second time under a frozen
    name, so the reference shares the foundation weights with the trained
    policy; when it starts from merged weights the reference is the model with
    its trainable adapter disabled.
    """

    def __init__(self, model: Any, reference_model: str) -> None:
        self.name: str | None = None
        self.kind = "disabled-adapter"
        target = model.module if hasattr(model, "module") else model
        self._model = target
        if (Path(reference_model) / "adapter_config.json").is_file():
            try:
                target.load_adapter(reference_model, adapter_name="aria_reference")
                target.set_adapter("default")
                self.name = "aria_reference"
                self.kind = "frozen-adapter"
            except Exception as exc:
                raise RuntimeError("cannot load the declared frozen reference adapter") from exc
        for parameter_name, parameter in target.named_parameters():
            if "aria_reference" in parameter_name:
                parameter.requires_grad_(False)

    def describe(self) -> dict[str, Any]:
        return {"kind": self.kind, "adapter": self.name}

    def log_probabilities(self, batch: Mapping[str, Any], torch: Any) -> Any:
        """Per-token log probabilities of the sampled tokens under the reference."""

        target = self._model

        def compute() -> Any:
            output, _ = _causal_forward(target, batch)
            values, _ = _token_log_probabilities(output.logits, batch["labels"], torch)
            return values.detach()

        if self.name is None:
            with torch.no_grad(), target.disable_adapter():
                return compute()
        active = target.active_adapters
        active = active[0] if isinstance(active, (list, tuple)) and active else "default"
        target.set_adapter(self.name)
        try:
            with torch.no_grad():
                return compute()
        finally:
            target.set_adapter(active)

    def pair_log_probabilities(
        self,
        chosen: Mapping[str, Any],
        rejected: Mapping[str, Any],
        torch: Any,
    ) -> tuple[Any, Any]:
        """Sequence log probabilities under the explicitly frozen policy."""

        target = self._model

        def compute() -> tuple[Any, Any]:
            return tuple(
                value.detach()
                for value in _pair_log_probabilities(
                    target,
                    chosen,
                    rejected,
                    torch,
                )
            )  # type: ignore[return-value]

        if self.name is None:
            with torch.no_grad(), target.disable_adapter():
                return compute()
        active = target.active_adapters
        active = active[0] if isinstance(active, (list, tuple)) and active else "default"
        target.set_adapter(self.name)
        try:
            with torch.no_grad():
                return compute()
        finally:
            target.set_adapter(active)


def _print_group(
    args: Any, outer: int, config: Any, entry: Mapping[str, Any], loss: float | None
) -> None:
    status = (
        f"loss={loss:.5f}"
        if loss is not None
        else f"skipped ({entry.get('dropped_reason') or 'no gradient'})"
    )
    print(
        f"{args.stage} round {outer + 1}/{config.outer_rounds} "
        f"{str(entry.get('incident_id'))[:48]}: {status}",
        flush=True,
    )


def _grpo_repair_group(
    model: Any,
    tokenizer: Any,
    incident: Any,
    population: Sequence[Any],
    environment: Any,
    config: Any,
    args: Any,
    device: Any,
    torch: Any,
    *,
    rng: random.Random,
    report: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Sample and score one Stage 3a group: m diagnoses, n repairs each.

    No parameters are touched here.  Sampling is separated from the update so a
    batch of groups drawn from several source cells can be standardized per cell
    before any step is taken, which is what keeps a cell whose rewards happen to
    be uniformly high from dominating the batch.
    """

    if getattr(args, "grpo_repair_chain", "legacy") == "cascade":
        return _cascade_repair_group(model, tokenizer, incident, population, environment,
                                     config, args, device, torch, rng=rng, report=report)

    from aria.post_training.grpo import (
        DiagnosisBranch,
        build_repair_group,
        gate_trigger_prefix,
        randomize_event_surface,
        select_transfer_siblings,
    )

    prefix_length = (
        gate_trigger_prefix(incident, rng=rng)
        if args.grpo_runtime_prefix == "true"
        else len(incident.events)
    )
    if args.grpo_feedback_dropout >= 1.0:
        feedback = None
    elif args.grpo_feedback_dropout <= 0.0:
        feedback = incident.feedback
    else:
        feedback = None if rng.random() < args.grpo_feedback_dropout else incident.feedback
    model_input = incident.prefix_input(prefix_length, feedback=feedback)
    if config.randomize_event_surface:
        model_input = randomize_event_surface(model_input, rng=rng)

    model.eval()
    diagnosis_texts = _sample_completions(
        model,
        tokenizer,
        instruction_for_task("transfer_diagnosis_preference"),
        model_input,
        count=config.diagnosis_group_size,
        max_new_tokens=args.grpo_diagnosis_max_new_tokens,
        args=args,
        device=device,
        torch=torch,
        rng=rng,
        oversample=max(1, args.grpo_diagnosis_oversample),
        response_prefix='{"diagnosis":',
    )
    diagnoses: list[tuple[str, str]] = []
    seen_diagnoses: set[str] = set()
    for text in diagnosis_texts:
        parsed = _parse_generated_diagnosis(text)
        if not parsed or parsed in seen_diagnoses:
            continue
        seen_diagnoses.add(parsed)
        diagnoses.append((text, parsed))
        if len(diagnoses) >= config.diagnosis_group_size:
            break
    if len(diagnoses) < 2:
        return None, {
            "incident_id": incident.incident_id,
            "cell": incident.cell_key,
            "updated": False,
            "dropped_reason": "fewer than two parseable diagnoses",
            "rewards": [],
        }

    siblings = select_transfer_siblings(incident, population, config, rng=rng)
    branches: list[DiagnosisBranch] = []
    for raw_text, diagnosis in diagnoses:
        branch = DiagnosisBranch(text=raw_text, diagnosis=diagnosis)
        repair_input = {**model_input, "diagnosis": diagnosis}
        repair_prefix = _repair_response_prefix(diagnosis)
        for text in _sample_completions(
            model,
            tokenizer,
            instruction_for_task("structured_repair"),
            repair_input,
            count=config.repair_group_size,
            max_new_tokens=args.grpo_max_new_tokens,
            args=args,
            device=device,
            torch=torch,
            rng=rng,
            response_prefix=repair_prefix,
        ):
            branch.own_samples.append(
                _score_sample(
                    text,
                    incident,
                    environment,
                    config,
                    report,
                    max_prefix_length=prefix_length,
                )
            )
        if args.grpo_ledger_replay == "true":
            branch.own_samples.extend(
                _ledger_replay_samples(
                    incident,
                    diagnosis=diagnosis,
                    environment=environment,
                    config=config,
                    report=report,
                    max_prefix_length=prefix_length,
                )
            )
        for sibling in siblings:
            sibling_input = {
                **sibling.model_input(feedback=None),
                "diagnosis": diagnosis,
            }
            for text in _sample_completions(
                model,
                tokenizer,
                instruction_for_task("structured_repair"),
                sibling_input,
                count=1,
                max_new_tokens=args.grpo_max_new_tokens,
                args=args,
                device=device,
                torch=torch,
                rng=rng,
                response_prefix=repair_prefix,
            ):
                sample = _score_sample(
                    text, sibling, environment, config, report, max_prefix_length=None
                )
                sample.is_sibling = True
                branch.sibling_samples.append(sample)
        branches.append(branch)

    group = build_repair_group(incident, branches, config, prefix_length=prefix_length)
    group.model_input = model_input
    entry: dict[str, Any] = {
        **group.statistics(),
        "updated": False,
        "diagnosis_count": len(branches),
        "siblings": len(siblings),
        "rewards": [
            sample.reward for branch in branches for sample in branch.own_samples
        ],
        "sibling_rewards": [
            sample.reward for branch in branches for sample in branch.sibling_samples
        ],
        "diagnosis_values": [branch.value for branch in branches],
    }
    if not group.has_gradient:
        entry["dropped_reason"] = group.dropped_reason or "no gradient"
        return None, entry
    return group, entry


def _repair_group_with_retries(
    model: Any,
    tokenizer: Any,
    incident: Any,
    population: Sequence[Any],
    environment: Any,
    config: Any,
    args: Any,
    device: Any,
    torch: Any,
    *,
    seed: int,
    max_attempts: int,
    report: dict[str, Any],
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    """Dynamically resample a dropped repair group.

    DAPO-style dynamic sampling discards groups with malformed, unresolved, or
    flat rewards.  Resampling here makes that filter useful on small partial
    ledgers while preserving every discarded attempt in the audit report.
    """

    dropped: list[dict[str, Any]] = []
    last_entry: dict[str, Any] | None = None
    for attempt in range(max(1, max_attempts)):
        group, entry = _grpo_repair_group(
            model,
            tokenizer,
            incident,
            population,
            environment,
            config,
            args,
            device,
            torch,
            rng=random.Random(seed + attempt),
            report=report,
        )
        entry["attempt"] = attempt
        last_entry = entry
        if group is not None:
            return group, entry, dropped
        dropped.append(entry)
    assert last_entry is not None
    return None, dropped.pop(), dropped


def _flush_repair_batch(
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    pending: Sequence[tuple[Any, dict[str, Any]]],
    config: Any,
    args: Any,
    device: Any,
    torch: Any,
    *,
    reference: _ReferenceAdapter,
) -> float | None:
    """Standardize a batch of groups per source cell, then take one step."""

    from aria.post_training.grpo import cell_baselines

    groups = [group for group, _ in pending if group is not None]
    if (getattr(args, "grpo_require_full_batch", "false") == "true"
            and len(groups) < args.grpo_batch_groups):
        for _, entry in pending:
            entry["dropped_reason"] = "incomplete effective group batch; no optimizer update"
        return None
    if not groups:
        return None
    if config.per_cell_standardization and len({g.cell_key for g in groups}) > 1:
        baselines = cell_baselines(groups)
        for group in groups:
            baseline = baselines.get(group.cell_key)
            if baseline is None:
                continue
            mean, spread = baseline
            if spread <= config.min_group_std:
                continue
            for branch in group.branches:
                if branch.value is not None:
                    branch.advantage = (branch.value - mean) / spread

    updates: list[tuple[str, Mapping[str, Any], str, float]] = []
    for group in groups:
        model_input = getattr(group, "model_input", None)
        if model_input is None:
            continue
        for branch in group.branches:
            if abs(branch.advantage) > 0.0:
                updates.append(
                    (
                        branch.update_instruction or instruction_for_task("transfer_diagnosis_preference"),
                        branch.update_input or model_input,
                        branch.text,
                        branch.advantage,
                    )
                )
                localization = getattr(branch, "localization_update", None)
                if localization:
                    updates.append((*localization, branch.advantage))
            for sample in branch.own_samples:
                if abs(sample.advantage) > 0.0:
                    updates.append(
                        (
                            instruction_for_task("structured_repair"),
                            sample.update_input or {**model_input, "diagnosis": branch.diagnosis},
                            sample.text,
                            sample.advantage,
                        )
                    )
    if not updates:
        for _, entry in pending:
            entry.setdefault("dropped_reason", "no sample carries a non-zero advantage")
        return None
    loss = _grpo_update(
        model,
        tokenizer,
        optimizer,
        updates,
        args,
        device,
        torch,
        reference=reference,
        config=config,
    )
    if loss is not None:
        args._grpo_eligible_groups = getattr(args, "_grpo_eligible_groups", 0) + len(groups)
        coverage = Counter(getattr(args, "_grpo_benchmark_groups", {}))
        for group in groups:
            parts = group.cell_key.split("/")
            coverage[parts[1] if len(parts) > 1 else group.cell_key] += 1
        args._grpo_benchmark_groups = dict(coverage)
    for group, entry in pending:
        if group is not None:
            entry["updated"] = loss is not None
            entry["loss"] = loss
            entry["diagnosis_advantages"] = [b.advantage for b in group.branches]
            if loss is None:
                entry["dropped_reason"] = "no generated target could receive an optimizer update"
    return loss


def _grpo_gate_step(
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    incident: Any,
    environment: Any,
    config: Any,
    args: Any,
    device: Any,
    torch: Any,
    *,
    reference: _ReferenceAdapter,
    rng: random.Random,
    report: dict[str, Any],
    gate_calibration: dict[str, float],
) -> tuple[float | None, dict[str, Any]]:
    """One Stage 3b group: matched-seed rollouts of the inspection decision.

    The repair model is frozen during rollout. The legacy mode updates only a
    scalar bias; contextual mode updates LoRA with binary KL and repair-task
    anchoring, which still requires a held-out forgetting check. A triggered
    repair is decoded greedily and cached per prefix.
    The control outcome is the same task with no intervention and is read once
    from the matched cache rather than re-run for every rollout.
    """

    from aria.post_training.environment import UNRESOLVED
    from aria.post_training.grpo import (
        GateRollout,
        GateRolloutGroup,
        gate_reward,
        gate_rollout_advantages,
        parse_gate_probability,
    )

    control_success = incident.matched_control_success
    if control_success is None:
        return None, {
            "incident_id": incident.incident_id,
            "updated": False,
            "dropped_reason": "no matched no-intervention outcome is recorded",
            "rewards": [],
        }

    probes = _gate_probe_indices(len(incident.events), args.grpo_gate_max_probes)
    aligned = getattr(args, "grpo_gate_repair_chain", "legacy") == "deployment"
    if aligned:
        probes = [p for p in probes if incident.events[p - 1].kind == "message"
                  and incident.events[p - 1].role not in {"user", "system", "tool"}]
        if not probes:
            return None, {"incident_id": incident.incident_id, "updated": False, "rewards": [],
                          "dropped_reason": "no native pending-message boundary in sampled prefix"}
    model.eval()
    group = GateRolloutGroup(incident_id=incident.incident_id)
    decisions: list[tuple[GateRollout, float, bool, int]] = []
    deployed_cache = {}
    repair_cache: dict[int, tuple[Any, str | None]] = {}
    probability_source = getattr(args, "grpo_gate_probability_source", "generated")
    gate_observations: list[dict[str, Any]] = []
    gate_cache: dict[int, tuple[str, float | None]] = {}
    for rollout_index in range(config.gate_group_size):
        # The caller supplies a distinct RNG for every dynamic-sampling
        # attempt.  Deriving rollout seeds from it is essential: reusing only
        # ``args.seed + rollout_index`` would make gate retries identical and
        # could leave a zero-reward abstention group permanently stuck.
        seed = _stable_seed(
            rng.randrange(2**31), f"{incident.incident_id}:{rollout_index}"
        )
        rollout = GateRollout(
            incident_id=incident.incident_id, seed=seed, trigger_index=None
        )
        rollout.control_success = control_success
        local_rng = random.Random(seed)
        for probe in probes:
            gate_input = incident.prefix_input(probe, feedback=None)
            if probe not in gate_cache:
                generation_metadata: list[dict[str, Any]] = []
                texts = _sample_completions(
                    model, tokenizer, instruction_for_task("deployment_gate"), gate_input,
                    count=1, max_new_tokens=32, args=args, device=device, torch=torch,
                    rng=local_rng, greedy=True, response_prefix='{"gate_probability":',
                    generation_metadata=(generation_metadata if probability_source == "label_likelihood" else None),
                )
                text = texts[0] if texts else ""
                raw_probability = parse_gate_probability(text)
                if raw_probability is not None and probability_source == "label_likelihood":
                    raw_probability = generation_metadata[0].get("label_likelihood") if generation_metadata else None
                gate_cache[probe] = text, raw_probability
            else:
                # Greedy scores are unchanged within a frozen-policy group;
                # preserve the RNG draw the uncached sampler would consume.
                local_rng.randrange(2**31)
                text, raw_probability = gate_cache[probe]
            if raw_probability is None:
                return None, {
                    "incident_id": incident.incident_id,
                    "updated": False,
                    "dropped_reason": "gate returned an invalid probability",
                    "gate_invalid_output": text,
                    "gate_probability_source": probability_source,
                    "failed_probe": probe,
                    "rollouts": [item.to_dict() for item in group.rollouts],
                    "rewards": [],
                }
            probability = _calibrated_gate_probability(
                raw_probability,
                logit_bias=gate_calibration["logit_bias"],
            )
            sampling_probability = (
                (1.0 - args.grpo_gate_exploration) * probability
                + args.grpo_gate_exploration * 0.5
            )
            fired = local_rng.random() < sampling_probability
            gate_observations.append({
                "seed": seed, "prefix_length": probe, "raw_response": text,
                "probability_source": probability_source, "uncalibrated_probability": raw_probability,
                "calibrated_probability": probability, "sampling_probability": sampling_probability,
                "fired": fired,
            })
            rollout.decisions.append((probe, sampling_probability, fired, text))
            decisions.append((rollout, sampling_probability, fired, probe))
            if fired:
                rollout.trigger_index = probe
                break
        if rollout.trigger_index is None:
            # Abstention delivers the uninterrupted run, so the matched-seed
            # difference is exactly zero and no execution is charged.
            rollout.intervened_success = control_success
            rollout.execution_delta = 0.0
            rollout.detail = "abstained"
        elif aligned:
            probe = rollout.trigger_index
            if probe not in deployed_cache:
                deployed_cache[probe] = _deployed_gate_repair(
                    model, tokenizer, incident, probe, environment, args, device, torch,
                )
                decision, calls, outcome = deployed_cache[probe]
                if outcome is not None:
                    report["outcome_status"][outcome.status] += 1
                    report["reward_grounding"][outcome.grounding] += 1
                report.setdefault("deployment_attempts", []).append({
                    "incident_id": incident.incident_id, "prefix_length": probe,
                    "decision": decision.to_dict(), "model_calls": calls,
                    "outcome": None if outcome is None else {
                        "status": outcome.status, "success": outcome.success,
                        "grounding": outcome.grounding, "detail": outcome.detail},
                    "scope": "deployment-chain at a historical prefix; not from-scratch live rollout",
                })
            decision, _, outcome = deployed_cache[probe]
            rollout.diagnosis = decision.attribution.diagnosis if decision.attribution else None
            if outcome is None:
                rollout.intervened_success = control_success
                rollout.execution_delta = 0.0
                rollout.detail = f"fail-open:{decision.reason}"
            else:
                rollout.intervened_success = outcome.success
                rollout.execution_delta = max(0.0, outcome.execution_count - 1.0)
                rollout.detail = outcome.detail
        else:
            probe = rollout.trigger_index
            if probe not in repair_cache:
                gate_repair_input = incident.prefix_input(probe, feedback=None)
                diagnosis_texts = _sample_completions(
                    model,
                    tokenizer,
                    instruction_for_task("transfer_diagnosis_preference"),
                    gate_repair_input,
                    count=1,
                    max_new_tokens=args.grpo_diagnosis_max_new_tokens,
                    args=args,
                    device=device,
                    torch=torch,
                    rng=random.Random(args.seed + probe),
                    greedy=True,
                    response_prefix='{"diagnosis":',
                )
                diagnosis = (
                    _parse_generated_diagnosis(diagnosis_texts[0])
                    if diagnosis_texts
                    else None
                )
                if diagnosis:
                    gate_repair_input = {
                        **gate_repair_input,
                        "diagnosis": diagnosis,
                    }
                    rollout.diagnosis = diagnosis
                repair_texts = _sample_completions(
                    model,
                    tokenizer,
                    instruction_for_task("structured_repair"),
                    gate_repair_input,
                    count=1,
                    max_new_tokens=args.grpo_max_new_tokens,
                    args=args,
                    device=device,
                    torch=torch,
                    rng=random.Random(args.seed),
                    greedy=True,
                    response_prefix=_repair_response_prefix(
                        diagnosis or "runtime repair"
                    ),
                )
                repair_cache[probe] = (_score_sample(
                    repair_texts[0] if repair_texts else "",
                    incident,
                    environment,
                    config,
                    report,
                    max_prefix_length=probe,
                ), diagnosis)
            sample, rollout.diagnosis = repair_cache[probe]
            if (
                args.grpo_ledger_replay == "true"
                and (
                    not sample.screen.valid
                    or sample.outcome.status == UNRESOLVED
                )
            ):
                replay = _successful_ledger_replay(
                    incident,
                    diagnosis=rollout.diagnosis or "ledger-supported runtime repair",
                    environment=environment,
                    config=config,
                    report=report,
                    max_prefix_length=probe,
                )
                if replay is not None:
                    sample = replay
                    rollout.detail = f"ledger replay: {replay.replay_label}"
            if sample.outcome.status == UNRESOLVED:
                rollout.intervened_success = None
                rollout.detail = "intervention unresolved"
            elif not sample.screen.valid:
                # A structurally invalid record substituted into a live run
                # breaks it; that is the harmful-intervention condition.
                rollout.intervened_success = False
                rollout.detail = f"invalid record: {sample.screen.reason}"
            else:
                rollout.intervened_success = sample.outcome.success
                rollout.detail = rollout.detail or sample.outcome.detail
            rollout.execution_delta = max(
                0.0, float(sample.outcome.execution_count) - 1.0
            )
        rollout.reward = gate_reward(
            intervened_success=rollout.intervened_success,
            control_success=rollout.control_success,
            execution_delta=rollout.execution_delta,
            config=config,
        )
        group.rollouts.append(rollout)

    group = gate_rollout_advantages(group, config)
    entry: dict[str, Any] = {
        "incident_id": incident.incident_id,
        "updated": False,
        "rollouts": [item.to_dict() for item in group.rollouts],
        "rewards": [item.reward for item in group.rollouts],
        "dropped_reason": group.dropped_reason,
        "probe_count": len(probes),
        "gate_exploration": args.grpo_gate_exploration,
        "gate_probability_source": probability_source,
        "gate_observations": gate_observations,
        "gate_prediction_count": len(gate_cache),
    }
    if not group.has_gradient:
        return None, entry

    credited = [
        (rollout.advantage, probability, fired)
        for rollout, probability, fired, _ in decisions
        if abs(rollout.advantage) > 0.0
    ]
    if not credited:
        entry["dropped_reason"] = "no gate decision carries a non-zero advantage"
        return None, entry
    if getattr(args, "grpo_gate_policy", "scalar") == "contextual":
        contextual = [(rollout.advantage, probability, fired, probe)
                      for rollout, probability, fired, probe in decisions if abs(rollout.advantage) > 0]
        loss_value = _contextual_gate_update(model, tokenizer, optimizer, incident, contextual,
                                            config, args, device, torch, reference=reference,
                                            logit_bias=gate_calibration["logit_bias"])
        entry.update(updated=True, update_count=len(contextual), loss=loss_value,
                     gate_policy="contextual_lora", gate_logit_bias=gate_calibration["logit_bias"])
        return loss_value, entry
    loss_value, gradient = _gate_calibration_update(
        credited,
        logit_bias=gate_calibration["logit_bias"],
        learning_rate=args.grpo_gate_learning_rate,
        kl_coefficient=config.kl_coefficient,
        exploration=args.grpo_gate_exploration,
    )
    gate_calibration["logit_bias"] -= args.grpo_gate_learning_rate * gradient
    entry["updated"] = True
    entry["update_count"] = len(credited)
    entry["loss"] = loss_value
    entry["gate_logit_bias"] = gate_calibration["logit_bias"]
    return loss_value, entry


def _cascade_repair_group(model, tokenizer, incident, population, environment, config, args, device, torch,
                           *, rng, report):
    from dataclasses import replace
    from aria.post_training.deployment_rollout import CallableRepairModel
    from aria.post_training.grpo import DiagnosisBranch, build_repair_group, gate_trigger_prefix, select_transfer_siblings

    prefix = gate_trigger_prefix(incident, rng=rng) if args.grpo_runtime_prefix == "true" else len(incident.events)
    observable = replace(incident.as_incident_record(), events=incident.events[:prefix], feedback=None)
    model.eval()
    branches, attempts = [], []
    siblings = select_transfer_siblings(incident, population, config, rng=rng)

    def generate(instruction, payload):
        metadata = []
        texts = _sample_completions(model, tokenizer, instruction, payload, count=1,
            max_new_tokens=args.grpo_max_new_tokens, args=args, device=device, torch=torch, rng=rng,
            prompt_metadata=metadata)
        return (texts[0] if texts else "", metadata[0]["input"] if metadata else payload)

    for _ in range(config.diagnosis_group_size):
        policy = CallableRepairModel(generate, retries=args.grpo_structured_retries)
        try:
            localization = policy.localize(observable)
            localization_call = policy.calls[-1]
            attribution = policy.attribute(observable, localization)
        except (TypeError, ValueError) as exc:
            attempts.append({"status": "invalid_diagnosis", "error": str(exc), "model_calls": policy.calls})
            continue
        attribution_call = policy.calls[-1]
        branch = DiagnosisBranch(text=attribution_call["raw_response"], diagnosis=attribution.diagnosis)
        branch.update_instruction = attribution_call["messages"][0]["content"]
        branch.update_input = json.loads(attribution_call["messages"][1]["content"])
        branch.localization_update = (localization_call["messages"][0]["content"],
            json.loads(localization_call["messages"][1]["content"]), localization_call["raw_response"])
        for _ in range(config.repair_group_size):
            try:
                policy.propose_repair(observable, attribution)
            except (TypeError, ValueError):
                pass
            call = policy.calls[-1]
            if "raw_response" not in call:
                attempts.append({"status": "generation_unavailable", "call": call})
                continue
            text = call.get("raw_response") or ""
            sample = _score_sample(text, incident, environment, config, report, max_prefix_length=prefix)
            sample.update_input = json.loads(call["messages"][1]["content"])
            branch.own_samples.append(sample)
        for sibling in siblings:
            # Transfer uses the sibling's own grounded localization and RA, never source event IDs.
            sibling_policy = CallableRepairModel(generate, retries=args.grpo_structured_retries)
            sibling_observable = replace(sibling.as_incident_record(), feedback=None)
            try:
                sibling_attribution = sibling_policy.attribute(sibling_observable)
                sibling_input = {**sibling_observable.patch_model_input(sibling_attribution),
                                 "transfer_diagnosis": branch.diagnosis}
                text, fitted_sibling_input = generate(instruction_for_task("structured_repair"), sibling_input)
                sample = _score_sample(text, sibling, environment, config, report, max_prefix_length=None)
                sample.is_sibling = True
                branch.sibling_samples.append(sample)
                attempts.append({"status": "sibling_transfer", "incident_id": sibling.incident_id,
                                 "model_calls": sibling_policy.calls, "patch_input": fitted_sibling_input, "raw_patch": text})
            except (TypeError, ValueError) as exc:
                attempts.append({"status": "invalid_sibling_diagnosis", "error": str(exc),
                                 "model_calls": sibling_policy.calls})
        branches.append(branch)
        attempts.append({"status": "scored_branch", "model_calls": policy.calls,
                         "own_outcomes": [sample.to_dict() for sample in branch.own_samples]})
    report.setdefault("cascade_attempts", []).append({"incident_id": incident.incident_id, "prefix_length": prefix,
                                                      "attempts": attempts})
    if len(branches) < 2:
        return None, {"incident_id": incident.incident_id, "cell": incident.cell_key, "updated": False,
                      "dropped_reason": "fewer than two grounded localization/RA branches", "rewards": []}
    group = build_repair_group(incident, branches, config, prefix_length=prefix)
    group.model_input = observable.model_input(feedback=None)
    entry = {**group.statistics(), "updated": False, "diagnosis_count": len(branches),
             "siblings": len(siblings), "repair_chain": "localization-attribution-patch",
             "rewards": [s.reward for b in branches for s in b.own_samples],
             "diagnosis_values": [b.value for b in branches]}
    return (group if group.has_gradient else None), entry


def _deployed_gate_repair(model, tokenizer, incident, probe, environment, args, device, torch):
    from dataclasses import replace
    from aria.hook import ActiveAction
    from aria.post_training.deployment_rollout import deployed_repair_attempt

    observable = replace(incident.as_incident_record(), events=incident.events[:probe], feedback=None)
    rng = random.Random(_stable_seed(args.seed, f"{incident.incident_id}:{probe}"))

    def generate(instruction, payload):
        metadata = []
        texts = _sample_completions(model, tokenizer, instruction, payload, count=1,
                                   max_new_tokens=args.grpo_max_new_tokens, args=args,
                                   device=device, torch=torch, rng=rng, greedy=True, prompt_metadata=metadata)
        return (texts[0] if texts else "", metadata[0]["input"] if metadata else payload)

    decision, calls = deployed_repair_attempt(observable, generate, retries=args.grpo_structured_retries)
    outcome = environment.execute(incident, decision.repair) if decision.action == ActiveAction.SUBSTITUTE else None
    return decision, calls, outcome


def _gate_label_batches(tokenizer, model_input, args, device, torch):
    prefix = '{"gate_probability":'
    prompt = _fitted_prompt(tokenizer, instruction_for_task("deployment_gate"), model_input,
                            targets=[prefix], max_length=args.max_length - 32,
                            max_prompt_tokens=args.max_prompt_tokens) + prefix
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    labels = [tokenizer.encode(value, add_special_tokens=False) for value in ("0", "1")]
    if any(len(value) != 1 for value in labels) or labels[0] == labels[1]:
        raise ValueError("contextual gate requires distinct single-token binary labels")
    return [{"input_ids": torch.tensor([ids + value], device=device),
             "attention_mask": torch.ones((1, len(ids) + 1), dtype=torch.long, device=device),
             "labels": torch.tensor([[-100] * len(ids) + value], device=device)} for value in labels]


def _contextual_gate_update(model, tokenizer, optimizer, incident, decisions, config, args, device, torch,
                            *, reference, logit_bias):
    from aria.post_training.deployment_rollout import bernoulli_policy_loss

    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    # Aggregate repeated visits before differentiating the shared prefix policy.
    by_probe = defaultdict(list)
    for advantage, old, fired, probe in decisions:
        by_probe[probe].append((advantage, old, fired))
    for probe, values in by_probe.items():
        zero, one = _gate_label_batches(tokenizer, incident.prefix_input(probe, feedback=None), args, device, torch)
        ref_zero, ref_one = reference.pair_log_probabilities(zero, one, torch)
        policy_zero, policy_one = _pair_log_probabilities(model, zero, one, torch)
        p = torch.sigmoid(policy_one - policy_zero + logit_bias)
        ref_p = torch.sigmoid(ref_one - ref_zero + logit_bias)
        epsilon = args.grpo_gate_exploration
        sampling = (1 - epsilon) * p + epsilon * 0.5
        ref_sampling = (1 - epsilon) * ref_p + epsilon * 0.5
        loss = sum(bernoulli_policy_loss(sampling, old_probability=old, reference_probability=ref_sampling,
                    fired=fired, advantage=advantage, clip_low=config.clip_low, clip_high=config.clip_high,
                    kl_coefficient=config.kl_coefficient, torch=torch) for advantage, old, fired in values) / len(decisions)
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite contextual gate loss")
        loss.backward()
        total += float(loss.detach().cpu())
    if not hasattr(args, "_gate_anchor_rows"):
        anchors = [r for r in _read_rows(Path(args.grpo_gate_anchor_path), 0)
                   if r.get("task") in {"failure_localization", "repair_attribution", "structured_repair"}]
        if _REQUIRE_VISIBLE_EVIDENCE:
            anchors, evidence_report = _filter_visible_training_rows(tokenizer, anchors, args)
            _BUDGET_REPORTS["gate_anchor_evidence"] = evidence_report
        args._gate_anchor_rows = anchors
        if not args._gate_anchor_rows:
            raise ValueError("contextual gate has no repair-task anchor rows")
    grouped = defaultdict(list)
    for row in args._gate_anchor_rows:
        grouped[row["task"]].append(row)
    seed = _stable_seed(args.seed, incident.incident_id)
    tasks = sorted(grouped)
    anchors = grouped[tasks[seed % len(tasks)]]
    anchor = anchors[(seed // len(tasks)) % len(anchors)]
    prompt, target = _prompt_and_output(tokenizer, anchor, preference_side=None, max_length=args.max_length,
                                       max_prompt_tokens=args.max_prompt_tokens)
    batch = _tokenize_pair(tokenizer, prompt, target, args.max_length, device, torch,
                           max_prompt_tokens=args.max_prompt_tokens)
    output, scoring_batch = _causal_forward(model, batch)
    anchor_loss = args.grpo_gate_anchor_weight * _weighted_causal_loss(output.logits, scoring_batch, torch)
    if not torch.isfinite(anchor_loss):
        raise RuntimeError("non-finite contextual gate anchor loss")
    anchor_loss.backward()
    total += float(anchor_loss.detach().cpu())
    _optimizer_step(model, optimizer, args, torch)
    return total


def _gate_step_with_retries(
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    incident: Any,
    environment: Any,
    config: Any,
    args: Any,
    device: Any,
    torch: Any,
    *,
    reference: _ReferenceAdapter,
    seed: int,
    max_attempts: int,
    report: dict[str, Any],
    gate_calibration: dict[str, float],
) -> tuple[float | None, dict[str, Any], list[dict[str, Any]]]:
    """Resample a gate rollout group when matched-seed rewards are flat."""

    dropped: list[dict[str, Any]] = []
    last_entry: dict[str, Any] | None = None
    for attempt in range(max(1, max_attempts)):
        loss, entry = _grpo_gate_step(
            model,
            tokenizer,
            optimizer,
            incident,
            environment,
            config,
            args,
            device,
            torch,
            reference=reference,
            rng=random.Random(seed + attempt),
            report=report,
            gate_calibration=gate_calibration,
        )
        entry["attempt"] = attempt
        last_entry = entry
        if loss is not None:
            return loss, entry, dropped
        dropped.append(entry)
    assert last_entry is not None
    return None, dropped.pop(), dropped


def _load_gate_calibration(base_model: str, *, probability_source: str = "generated") -> dict[str, float]:
    calibration = load_gate_calibration(
        Path(base_model) / "gate_calibration.json", probability_source=probability_source,
        allow_source_mismatch=True,
    )
    return {"logit_bias": calibration["logit_bias"] if calibration else 0.0}


def _calibrated_gate_probability(
    probability: float | None,
    *,
    logit_bias: float,
) -> float:
    bounded = min(max(float(probability or 0.0), 1e-5), 1.0 - 1e-5)
    logit = math.log(bounded / (1.0 - bounded)) + logit_bias
    return 1.0 / (1.0 + math.exp(-max(min(logit, 30.0), -30.0)))


def _gate_calibration_update(
    decisions: Sequence[tuple[float, float, bool]],
    *,
    logit_bias: float,
    learning_rate: float,
    kl_coefficient: float,
    exploration: float = 0.0,
) -> tuple[float, float]:
    """Differentiate the sampled mixture policy through the gate logit bias."""

    if learning_rate <= 0.0:
        raise ValueError("--grpo-gate-learning-rate must be positive")
    if not 0.0 <= exploration <= 1.0:
        raise ValueError("gate exploration must be between zero and one")
    losses: list[float] = []
    gradients: list[float] = []
    for advantage, probability, fired in decisions:
        bounded = min(max(probability, 1e-8), 1.0 - 1e-8)
        action = 1.0 if fired else 0.0
        log_probability = math.log(bounded if fired else 1.0 - bounded)
        losses.append(-advantage * log_probability)
        # q = (1-epsilon) * sigmoid(z+b) + epsilon/2. The derivative
        # dq/db is not q*(1-q); fully random exploration has zero gradient.
        policy_p = ((bounded - exploration * 0.5) / (1.0 - exploration)
                    if exploration < 1.0 else 0.5)
        policy_p = min(max(policy_p, 0.0), 1.0)
        dq_db = (1.0 - exploration) * policy_p * (1.0 - policy_p)
        gradients.append(-advantage * (action - bounded) * dq_db / (bounded * (1.0 - bounded)))
    # A weak quadratic anchor is the Bernoulli-policy analogue of the Stage 3
    # KL penalty toward the Stage 3-entry gate (zero calibration bias).
    loss = sum(losses) / len(losses) + 0.5 * kl_coefficient * logit_bias**2
    gradient = sum(gradients) / len(gradients) + kl_coefficient * logit_bias
    return loss, gradient


def _gate_probe_indices(event_count: int, max_probes: int) -> list[int]:
    """Prefix lengths at which the gate is evaluated.

    The paper evaluates the gate after every event.  Probing every prefix of a
    long trajectory costs one generation each, so the probes are spread evenly
    over the trajectory and the budget is reported rather than hidden.
    """

    if event_count <= 0:
        return []
    if max_probes <= 0 or event_count <= max_probes:
        return list(range(1, event_count + 1))
    step = event_count / max_probes
    return sorted(
        {
            max(1, min(event_count, round((index + 1) * step)))
            for index in range(max_probes)
        }
    )


def _score_sample(
    text: str,
    incident: Any,
    environment: Any,
    config: Any,
    report: dict[str, Any],
    *,
    max_prefix_length: int | None,
) -> Any:
    """Screen one sampled record, execute it if it survives, and reward it."""

    from aria.post_training.environment import invalid_outcome
    from aria.post_training.grpo import (
        RepairSample,
        parse_record,
        reward,
        screen_record,
    )

    record, parse_error = parse_record(text, incident)
    screen = screen_record(
        record,
        incident,
        config,
        parse_error=parse_error,
        max_prefix_length=max_prefix_length,
    )
    if not screen.valid:
        report["screen_rejections"][str(screen.reason)[:60]] += 1
        outcome = invalid_outcome(str(screen.reason))
    else:
        assert screen.repair is not None
        outcome = environment.execute(incident, screen.repair)
    report["outcome_status"][outcome.status] += 1
    report["reward_grounding"][outcome.grounding] += 1
    report["scored_execution_equivalents"] = report.get("scored_execution_equivalents", 0.0) + float(outcome.execution_count)
    return RepairSample(
        text=text,
        record=record,
        screen=screen,
        outcome=outcome,
        reward=reward(outcome, config),
        incident_id=incident.incident_id,
    )


def _ledger_replay_samples(
    incident: Any,
    *,
    diagnosis: str,
    environment: Any,
    config: Any,
    report: dict[str, Any],
    max_prefix_length: int | None,
) -> list[Any]:
    """Return one recorded success/failure pair at a common cut.

    These are explicitly marked offline replay samples. They guarantee that a
    ledger-supported incident has repair-level reward spread, while the other
    samples in the same group remain current-policy generations scored through
    ledger followed by the benchmark evaluator.
    """

    by_cut: dict[str | None, dict[bool, Any]] = defaultdict(dict)
    for entry in incident.ledger:
        if isinstance(entry.success, bool):
            by_cut[entry.cut_event_id].setdefault(entry.success, entry)
    candidates = [
        (cut, outcomes)
        for cut, outcomes in by_cut.items()
        if True in outcomes and False in outcomes
    ]
    if not candidates:
        return []
    candidates.sort(
        key=lambda item: (
            item[0] != incident.certified_cut_event_id,
            str(item[0]),
        )
    )
    _, outcomes = candidates[0]
    samples: list[Any] = []
    for success in (True, False):
        entry = outcomes[success]
        repair = _ledger_entry_repair(incident, entry, diagnosis)
        if repair is None:
            continue
        text = json.dumps(
            ordered_training_output("structured_repair", repair.to_dict()),
            ensure_ascii=True,
            separators=(",", ":"),
        )
        sample = _score_sample(
            text,
            incident,
            environment,
            config,
            report,
            max_prefix_length=max_prefix_length,
        )
        sample.is_replay = True
        sample.replay_label = entry.label
        samples.append(sample)
    return samples


def _successful_ledger_replay(
    incident: Any,
    *,
    diagnosis: str,
    environment: Any,
    config: Any,
    report: dict[str, Any],
    max_prefix_length: int | None,
) -> Any | None:
    """Materialize the incident's recorded successful repair for gate replay."""

    for entry in incident.ledger:
        if entry.success is not True:
            continue
        repair = _ledger_entry_repair(incident, entry, diagnosis)
        if repair is None:
            continue
        text = json.dumps(
            ordered_training_output("structured_repair", repair.to_dict()),
            ensure_ascii=True,
            separators=(",", ":"),
        )
        sample = _score_sample(
            text,
            incident,
            environment,
            config,
            report,
            max_prefix_length=max_prefix_length,
        )
        sample.is_replay = True
        sample.replay_label = entry.label
        if sample.screen.valid and sample.outcome.accepted:
            return sample
    return None


def _ledger_entry_repair(
    incident: Any, entry: Any, diagnosis: str
) -> Any | None:
    """Normalize an executed ledger continuation back into RepairRecord form."""

    from aria.records import RepairRecord

    replacements: list[dict[str, Any]] = []
    for index, raw in enumerate(entry.replacement_events, start=1):
        item = dict(raw)
        original_id = str(item.get("event_id") or "")
        metadata = dict(item.get("metadata") or {})
        if original_id.startswith("source:event:"):
            metadata.setdefault("replaces", [original_id])
        kind = str(item.get("kind") or "message")
        if kind == "tool_result":
            kind = "message"
        replacements.append(
            {
                "event_id": f"patch:event:{index:04d}",
                "role": str(item.get("role") or "assistant"),
                "kind": kind,
                "content": str(item.get("content") or ""),
                "tool_calls": [
                    dict(call)
                    for call in (item.get("tool_calls") or [])
                    if isinstance(call, Mapping)
                ],
                "metadata": metadata,
            }
        )
    if not replacements:
        return None
    try:
        return RepairRecord.from_dict(
            {
                "diagnosis": diagnosis,
                "cut_event_id": entry.cut_event_id,
                "replacement_events": replacements,
            },
            events=incident.events,
            name="ledger replay repair",
        )
    except (TypeError, ValueError):
        return None


def _sample_completions(
    model: Any,
    tokenizer: Any,
    instruction: str,
    model_input: Mapping[str, Any],
    *,
    count: int,
    max_new_tokens: int,
    args: Any,
    device: Any,
    torch: Any,
    rng: random.Random,
    greedy: bool = False,
    oversample: int = 1,
    response_prefix: str = "",
    generation_metadata: list[dict[str, Any]] | None = None,
    prompt_metadata: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Draw ``count`` completions, over-generating by ``oversample``.

    A group is discarded when it lacks reward spread, which is intended, but a
    group lost because one completion failed to parse carries no information
    about the policy, so the sampler over-generates and the caller keeps the
    parseable ones.
    """

    fitted_inputs = []
    prompt = _fitted_prompt(
        tokenizer,
        instruction,
        model_input,
        targets=[response_prefix],
        max_length=args.max_length - max_new_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        fitted_inputs=fitted_inputs,
    )
    prompt += response_prefix
    if prompt_metadata is not None:
        prompt_metadata.append({"prompt": prompt, "input": fitted_inputs[0]})
    encoded = tokenizer(prompt, return_tensors="pt", truncation=False, add_special_tokens=False)
    if encoded["input_ids"].shape[1] + max_new_tokens > args.max_length:
        raise ValueError("ARIA generation prefix and output exceed the model context budget")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    torch.manual_seed(rng.randrange(2**31))
    was_cached = model.config.use_cache
    model.config.use_cache = True
    try:
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=not greedy,
                temperature=None if greedy else args.grpo_temperature,
                top_p=None if greedy else args.grpo_top_p,
                max_new_tokens=max_new_tokens,
                num_return_sequences=count * max(1, oversample),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=generation_metadata is not None,
                output_scores=generation_metadata is not None,
            )
    finally:
        model.config.use_cache = was_cached
    prompt_length = encoded["input_ids"].shape[1]
    if generation_metadata is not None:
        for index, row in enumerate(generated.sequences):
            generation_metadata.append({"label_likelihood": label_probability(
                generated.scores, tokenizer, torch, int(row[prompt_length]), row_index=index,
            )})
        generated = generated.sequences
    return [
        response_prefix
        + tokenizer.decode(row[prompt_length:], skip_special_tokens=True)
        for row in generated
    ]


def _repair_response_prefix(diagnosis: str) -> str:
    """Force the invariant repair fields while leaving the action free.

    The diagnosis was sampled by the policy in the preceding branch. Reusing
    it in the JSON prefix prevents a second unconstrained generation from
    drifting into prose before it reaches the executable cut and continuation.
    The policy still samples both ``cut_event_id`` and ``replacement_events``.
    """

    encoded = json.dumps(
        {"diagnosis": str(diagnosis)},
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return encoded[:-1] + ',"cut_event_id":'


def _grpo_update(
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    updates: Sequence[tuple[str, Mapping[str, Any], str, float]],
    args: Any,
    device: Any,
    torch: Any,
    *,
    reference: _ReferenceAdapter,
    config: Any,
) -> float | None:
    """Take one clipped token-level GRPO step over a scored group.

    The ratio is formed against the sampling policy, the upper clipping bound is
    raised above the lower one so that a low-probability token whose advantage
    is positive is not clipped away, and the k3 estimator penalizes drift from
    the Stage 2 policy.  Gradients accumulate across the whole group before the
    optimizer steps, so the group is the unit of update.
    """

    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    scored = 0
    skipped = 0
    for instruction, model_input, text, advantage in updates:
        try:
            prompt = _fitted_prompt(
                tokenizer,
                instruction,
                model_input,
                targets=[text],
                max_length=args.max_length,
                max_prompt_tokens=args.max_prompt_tokens,
            )
            batch = _tokenize_pair(
                tokenizer,
                prompt,
                text,
                args.max_length,
                device,
                torch,
                max_prompt_tokens=args.max_prompt_tokens,
            )
        except ValueError:
            # A sample longer than the context budget cannot be scored without
            # truncating it, which would credit tokens the policy never emitted.
            skipped += 1
            continue
        # The reference is read first, so no adapter switch happens while a live
        # autograd graph exists.
        reference_log_probability = reference.log_probabilities(batch, torch)
        output, _ = _causal_forward(model, batch)
        log_probability, mask = _token_log_probabilities(
            output.logits, batch["labels"], torch
        )
        token_count = mask.sum()
        if int(token_count.item()) == 0:
            continue
        # Every sample in a group is scored at the same parameters, so the
        # sampling policy is this policy: the ratio starts at one and its
        # gradient is the score function, which is GRPO with one inner epoch.
        old_log_probability = log_probability.detach()
        ratio = torch.exp(log_probability - old_log_probability)
        advantage_tensor = torch.tensor(
            float(advantage), device=ratio.device, dtype=ratio.dtype
        )
        unclipped = ratio * advantage_tensor
        clipped = (
            torch.clamp(ratio, 1.0 - config.clip_low, 1.0 + config.clip_high)
            * advantage_tensor
        )
        policy_term = torch.min(unclipped, clipped)
        difference = reference_log_probability - log_probability
        kl = torch.exp(difference.clamp(max=20.0)) - difference - 1.0
        per_token = -policy_term + config.kl_coefficient * kl
        loss = (per_token * mask).sum() / token_count.clamp_min(1.0)
        loss = loss / max(1, len(updates))
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite GRPO loss")
        loss.backward()
        total += float(loss.detach().cpu()) * max(1, len(updates))
        scored += 1
    if skipped:
        print(
            f"{args.stage}: {skipped} of {len(updates)} sampled records exceeded "
            "the context budget and were not scored",
            flush=True,
        )
    if scored == 0:
        optimizer.zero_grad(set_to_none=True)
        return None
    _optimizer_step(model, optimizer, args, torch)
    return total / scored

if __name__ == "__main__":
    main()
