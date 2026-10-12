"""Server-grounded budgeting for explicitly selected native host profiles."""

from __future__ import annotations

import copy
import json
import os
import re
from urllib.error import HTTPError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


def count_chat_tokens(settings, messages, tools=None):
    if settings.get("provider") != "vllm":
        raise ValueError("server token budgeting requires a declared vLLM endpoint")
    url = urlsplit(settings["base_url"])
    endpoint = urlunsplit((url.scheme, url.netloc, url.path.rstrip("/").removesuffix("/v1") + "/tokenize", "", ""))
    payload = {"model": settings["model"], "messages": messages, "add_generation_prompt": True}
    if tools:
        payload["tools"] = tools
    headers = {"Content-Type": "application/json", "User-Agent": "MASFM/1.0"}
    key = os.environ.get(settings.get("api_key_env") or "")
    if settings.get("api_key_required") and not key:
        raise ValueError("tokenizer endpoint requires its configured API-key environment variable")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = Request(endpoint, data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        with urlopen(request, timeout=60) as response:
            value = json.load(response)["count"]
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        match = re.search(r"maximum context length is\s+\d+\s+tokens.*?request has\s+(\d+)\s+input tokens", detail, re.I | re.S)
        if exc.code != 400 or match is None:
            raise RuntimeError(f"tokenizer endpoint rejected the request: HTTP {exc.code}: {detail[:1000]}") from exc
        value = int(match.group(1))
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("tokenizer endpoint returned an invalid token count")
    return value


def fit_chat_history(messages, counter, limit):
    """Remove whole old turns; never truncate tool arguments or the task."""
    original = counter(messages)
    if original <= limit:
        return messages, {"original_prompt_tokens": original, "prompt_tokens": original, "removed_messages": []}
    groups, owners = [], {}
    for index, message in enumerate(messages):
        if message.get("role") == "tool":
            owner = owners.get(message.get("tool_call_id"))
            if owner is None:
                raise ValueError("cannot compact an unmatched tool response")
            groups[owner].append(index)
        else:
            groups.append([index])
            for call in message.get("tool_calls") or []:
                if call.get("id"):
                    owners[call["id"]] = len(groups) - 1
    groups.sort(key=max)
    first_user = next((i for i, message in enumerate(messages) if message.get("role") == "user"), None)
    protected = {i for i, message in enumerate(messages) if message.get("role") in {"system", "developer"}}
    if first_user is not None:
        protected.add(first_user)
    protected.update(i for group in groups[-2:] for i in group)
    removed = set()
    for group in groups:
        if protected.intersection(group):
            continue
        removed.update(group)
        candidate = [copy.deepcopy(message) for i, message in enumerate(messages) if i not in removed]
        count = counter(candidate)
        if count <= limit:
            return candidate, {"original_prompt_tokens": original, "prompt_tokens": count, "removed_messages": sorted(removed)}
    raise ValueError("task, system messages, and latest complete turns exceed the declared context budget")


def fit_stage_messages(messages, parts_by_prompt, counter, limit):
    """Trim only registered optional stage context, not the original requirement."""
    count = counter(messages)
    if count <= limit:
        return messages, {"original_prompt_tokens": count, "prompt_tokens": count, "optional_context_fraction": 1.0}
    original = count
    for fraction in (0.75, 0.5, 0.25, 0.1, 0.0):
        candidate = copy.deepcopy(messages)
        recognized = False
        for message in candidate:
            content = message.get("content")
            if not isinstance(content, str) or content not in parts_by_prompt:
                continue
            mandatory, history, context_sections = parts_by_prompt[content]
            keep = int(len(history) * fraction)
            tail = history[-keep:] if keep else "[older team history omitted]"
            sections = context_sections[:int(len(context_sections) * fraction)]
            message["content"] = mandatory + tail + "\n" + "\n".join(sections)
            recognized = True
        if not recognized:
            raise ValueError("overlong native prompt has no registered optional context")
        count = counter(candidate)
        if count <= limit:
            return candidate, {"original_prompt_tokens": original, "prompt_tokens": count, "optional_context_fraction": fraction}
    raise ValueError("original requirement and system messages exceed the declared context budget")
