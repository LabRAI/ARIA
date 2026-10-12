"""Lossless message-field normalization for single-tool chat templates."""

import copy


def sequential_tool_history(messages):
    """Serialize completed multi-tool turns without changing calls or results."""
    output, split_turns, index = [], 0, 0
    while index < len(messages):
        message = messages[index]
        calls = message.get("tool_calls") or []
        if message.get("role") != "assistant" or len(calls) <= 1:
            output.append(copy.deepcopy(message))
            index += 1
            continue
        ids = [call.get("id") for call in calls]
        if any(not key for key in ids) or len(set(ids)) != len(ids):
            raise ValueError("multi-tool history requires unique call IDs")
        results, following = {}, index + 1
        while following < len(messages) and messages[following].get("role") == "tool":
            result = messages[following]
            key = result.get("tool_call_id")
            if key not in ids or key in results:
                raise ValueError("unmatched or duplicate tool result in history")
            results[key] = result
            following += 1
        if set(results) != set(ids):
            raise ValueError("cannot serialize an incomplete multi-tool turn")
        for number, call in enumerate(calls):
            turn = copy.deepcopy(message)
            turn["tool_calls"] = [copy.deepcopy(call)]
            if number:
                turn["content"] = None
                turn.pop("reasoning_content", None)
            output.extend((turn, copy.deepcopy(results[call["id"]])))
        split_turns += 1
        index = following
    return output, split_turns
