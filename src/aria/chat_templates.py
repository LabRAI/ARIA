"""Preserve ARIA instructions with chat templates lacking a system role."""
from __future__ import annotations

from typing import Any

from jinja2.exceptions import TemplateError


def prepare_chat_messages(messages: list[dict[str, Any]], mode: str = "native") -> list[dict[str, Any]]:
    if mode == "native":
        return [dict(message) for message in messages]
    if mode != "user-prefix":
        raise ValueError("system_role_mode must be native or user-prefix")
    if len(messages) < 2 or messages[0].get("role") != "system" or messages[1].get("role") != "user":
        raise ValueError("user-prefix requires initial system and user messages")
    first_user = {**messages[1], "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}
    return [first_user, *(dict(message) for message in messages[2:])]


def render_chat_prompt(tokenizer: Any, messages: list[dict[str, Any]], mode: str = "native") -> str:
    """Gemma's template rejects system roles; fold instructions without dropping them.

    Automatic fallback covers copied checkpoint tokenizers whose filesystem name
    does not identify their model family. Other template errors remain errors.
    """
    prepared = prepare_chat_messages(messages, mode)
    try:
        return tokenizer.apply_chat_template(prepared, tokenize=False, add_generation_prompt=True)
    except TemplateError as exc:
        if mode != "native" or "system role not supported" not in str(exc).lower():
            raise
        return tokenizer.apply_chat_template(
            prepare_chat_messages(messages, "user-prefix"), tokenize=False, add_generation_prompt=True,
        )
