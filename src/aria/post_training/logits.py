"""Avoid vocabulary projection for prompt positions masked out of the loss."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SUPPORTED_MODELS = frozenset({'qwen2', 'llama', 'mistral', 'gemma2'})


def causal_forward(model: Any, batch: Mapping[str, Any], *, target_only: bool = False):
    """Keep the whole transformer context and align the supervised logits suffix.

    One position before the first target is retained for the causal shift.
    The input tensors and caller-owned labels are never changed.
    """
    kwargs = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
    scoring_batch = batch
    if target_only:
        target = model.module if hasattr(model, 'module') else model
        if getattr(target.config, 'model_type', None) not in SUPPORTED_MODELS:
            raise ValueError('target-only logits requires a supported causal-LM architecture')
        labels = batch['labels']
        if labels.shape != batch['input_ids'].shape:
            raise ValueError('target-only logits requires labels aligned to the complete input')
        indices = (labels != -100).nonzero(as_tuple=False)
        if indices.numel():
            first = int(indices[:, 1].min().item())
            keep = labels.shape[1] - max(0, first - 1)
            kwargs['logits_to_keep'] = keep
            scoring_batch = {**batch, 'labels': labels[:, -keep:]}
            if 'loss_weights' in batch:
                scoring_batch['loss_weights'] = batch['loss_weights'][:, -keep:]
    return model(**kwargs), scoring_batch
