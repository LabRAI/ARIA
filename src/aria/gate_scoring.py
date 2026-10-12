"""Binary gate likelihoods shared by training and checkpoint serving."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def label_probability(scores: Any, tokenizer: Any, torch: Any, first_token: int, *, row_index: int = 0) -> float | None:
    zero = tokenizer.encode("0", add_special_tokens=False)
    one = tokenizer.encode("1", add_special_tokens=False)
    if not scores or len(zero) != 1 or len(one) != 1 or zero == one:
        return None
    if first_token not in (zero[0], one[0]):
        return None
    logits = scores[0][row_index, [zero[0], one[0]]].float()
    if not torch.isfinite(logits).all():
        return None
    return float(torch.softmax(logits, dim=-1)[1].item())


def chat_label_probability(logprobs: Any) -> float | None:
    """Read the first numeric token of a one-field gate JSON completion."""
    if not isinstance(logprobs, Mapping):
        return None
    for item in logprobs.get("content") or []:
        if not isinstance(item, Mapping) or str(item.get("token") or "").strip() not in {"0", "1"}:
            continue
        values = {}
        for alternative in item.get("top_logprobs") or []:
            if not isinstance(alternative, Mapping):
                continue
            token = alternative.get("token")
            value = alternative.get("logprob")
            if (token in {"0", "1"} and isinstance(value, (int, float))
                    and not isinstance(value, bool) and math.isfinite(value) and -9999 < value <= 0):
                values[token] = float(value)
        if set(values) != {"0", "1"}:
            return None
        delta = values["1"] - values["0"]
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, delta))))
    return None
