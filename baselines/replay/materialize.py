"""Turn a baseline intervention into an executable continuation task.

Every retained baseline emits either a cut plus replacement events or an active
governance decision that blocks or rewrites a message. The evaluator must see
one uniform representation so harness differences do not determine a table row.

This module reduces all of those shapes to a single continuation prompt built
the same way :func:`repair.generator.materialize_continuation_task` builds
ARIA's.  Native multi-agent runners still do not expose uniform checkpoint
restoration, so the continuation is explicitly non-exact and is labelled as such
in the returned ``execution_mode`` rather than being presented as a true replay.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from baselines.core.types import BaselineSample

CONTINUATION_MODE = "non_exact_continuation"
PATCH_HEADER = (
    "[BASELINE RUNTIME PATCH]\n"
    "This is a fresh native run, not a restored checkpoint. The retained prefix below "
    "is historical context only: its tool results, files, and errors have not occurred "
    "in this run. The workspace is freshly prepared for this task. Inspect current state and "
    "re-enact the necessary setup and replacement events using real tools. Do not "
    "treat historical failures as current-loop stalls, and do not claim a file was "
    "written or a tool succeeded until it actually happens in this run."
)
GOVERNANCE_HEADER = (
    "[MESSAGE GOVERNANCE INTERVENTION]\n"
    "This is a fresh native run with historical context, not restored process or "
    "workspace state. The candidate historical message was replaced with screened "
    "content. Inspect the current workspace and continue using that screened content. "
    "Historical tool results and failures are not observations from this run."
)


class InterventionNotExecutable(ValueError):
    """The intervention carries nothing the host can actually run."""


def materialize(
    sample: BaselineSample, intervention: Mapping[str, Any] | None,
    *, max_chars: int | None = None,
) -> dict[str, Any]:
    """Build the continuation task for one intervention.

    ``intervention`` of ``None`` means the control condition: rerun the original
    task untouched, which is what the harmful-intervention comparison needs.
    """

    task_text = sample.task
    if intervention is None:
        return {
            "task": task_text,
            "execution_mode": "control",
            "retained_prefix": [],
            "kind": "control",
        }
    cut_index = resolve_cut_index(sample, intervention)
    retained = [event.to_dict() for event in sample.events[: cut_index + 1]]
    replacements = replacement_events(sample, intervention)
    if not replacements:
        raise InterventionNotExecutable(
            "intervention carries no replacement events, sanitized content, or reflection"
        )
    header = (
        GOVERNANCE_HEADER if intervention.get("action") is not None else PATCH_HEADER
    )
    payload = {
        "diagnosis": intervention.get("diagnosis") or intervention.get("reason"),
        "action": intervention.get("action"),
        "retained_prefix": retained,
        "replacement_events": replacements,
    }
    if max_chars is not None:
        from baselines.core.models import _prepare_payload

        fixed_chars = len(task_text) + len(header) + len(json.dumps({
            **payload, "retained_prefix": []}, ensure_ascii=False)) + 128
        available = max_chars - fixed_chars
        if available < 256:
            raise InterventionNotExecutable("task and replacement events exceed continuation budget")
        view, budget = _prepare_payload({"events": retained}, max_chars=available)
        payload["retained_prefix"] = view["events"]
        payload["prefix_compacted"] = budget["prompt_compacted"]
    prompt = (
        f"{task_text}\n\n{header}\n"
        f"{json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    )
    return {
        "task": prompt,
        "execution_mode": CONTINUATION_MODE,
        "retained_prefix": retained,
        "replacement_events": replacements,
        "cut_event_id": sample.events[cut_index].event_id if cut_index >= 0 else None,
        "kind": str(intervention.get("type") or "repair_record"),
    }


def resolve_cut_index(sample: BaselineSample, intervention: Mapping[str, Any]) -> int:
    """Index of the last retained event, or ``-1`` when nothing is retained.

    A cut named by ``cut_event_id`` is retained inclusively, matching ARIA's
    ``tau_{<=k} + e`` splice.  A ``target_event_id`` names the event being
    replaced instead, so the retained prefix ends just before it.
    """

    positions = {event.event_id: index for index, event in enumerate(sample.events)}
    if "cut_event_id" in intervention:
        cut = intervention["cut_event_id"]
        if cut is None:
            return -1
        if cut in positions:
            return positions[cut]
        raise InterventionNotExecutable(
            f"cut_event_id {cut!r} is not part of the trajectory"
        )
    target = intervention.get("target_event_id")
    if target is not None:
        if target not in positions:
            raise InterventionNotExecutable(
                f"target_event_id {target!r} is not part of the trajectory"
            )
        return positions[target] - 1
    return len(sample.events) - 1


def replacement_events(
    sample: BaselineSample, intervention: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Normalize the replacement suffix regardless of which method proposed it."""

    raw = intervention.get("replacement_events")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and raw:
        return [dict(item) for item in raw if isinstance(item, Mapping)]
    content = intervention.get("sanitized_content") or intervention.get(
        "modified_content"
    )
    if isinstance(content, str) and content.strip():
        return [
            {
                "event_id": "baseline:patch:0000",
                "role": _target_role(sample, intervention),
                "kind": "message",
                "content": content,
                "tool_calls": [],
                "metadata": {
                    "replaces": [intervention.get("target_event_id")],
                    "action": intervention.get("action"),
                },
            }
        ]
    action = intervention.get("action")
    directives = {
        "block": "The runtime controller withheld the candidate message. Continue from the retained prefix without using that message.",
        "replan": "The runtime controller requested replanning. Discard the candidate plan and devise a new plan from the retained prefix.",
        "rollback": "The runtime controller discarded the candidate and its suffix. Continue from the retained prefix; recheck workspace state because this continuation is not an exact checkpoint restore.",
        "block_and_rollback": "The runtime controller withheld the candidate. Reconsider and resubmit from the retained prefix without the rejected claims.",
    }
    if action in directives:
        reason = str(intervention.get("reason") or "")
        return [{
            "event_id": "baseline:controller:0000", "role": "host", "kind": "message",
            "content": directives[action] + (f" Controller reason: {reason}" if reason else ""),
            "tool_calls": [], "metadata": {"action": action, "host_directive": True},
        }]
    return []


def _target_role(sample: BaselineSample, intervention: Mapping[str, Any]) -> str:
    target = intervention.get("target_event_id")
    for event in sample.events:
        if event.event_id == target:
            return event.role
    return "assistant"
