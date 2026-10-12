"""Paper-level SR/HIR/EB aggregation for baseline experiments."""

from __future__ import annotations

from collections.abc import Iterable
from statistics import mean
from typing import Any

from baselines.core.types import BaselineSample, MethodResult


def aggregate_results(
    pairs: Iterable[tuple[BaselineSample, MethodResult]],
) -> dict[str, Any]:
    """Aggregate one fixed evaluation pool without hiding unresolved cases.

    SR and HIR use the full paper evaluation pool as their denominator. A
    prompt-only row has no SR because it has pending executions. Once every
    method decision is terminal, evaluator/harness invalids remain in the pool
    (they are not passing runs) and are also reported separately from genuine
    task-incorrect outcomes.
    """

    rows = list(pairs)
    successful = 0
    task_incorrect = 0
    invalid_evaluation = 0
    method_invalid = 0
    pending = 0
    harmful = 0
    hir_unresolved = 0
    budgets: list[float] = []

    for sample, result in rows:
        budgets.append(result.execution_budget)
        success, outcome = _outcome(sample, result)
        if success is True:
            successful += 1
        elif success is False:
            task_incorrect += 1
        elif outcome == "pending":
            pending += 1
        elif outcome == "method_invalid":
            method_invalid += 1
        else:
            invalid_evaluation += 1

        harm = _is_harmful(sample, result, success)
        if harm is True:
            harmful += 1
        elif harm is None:
            hir_unresolved += 1

    pool_size = len(rows)
    sr = successful / pool_size if pool_size and pending == 0 else None
    hir = harmful / pool_size if pool_size and pending == 0 else None
    return {
        "schema_version": "aria-baseline-summary-v2",
        "sample_count": pool_size,
        "successful_count": successful,
        "task_incorrect_count": task_incorrect,
        "resolved_count": successful + task_incorrect,
        "invalid_evaluation_count": invalid_evaluation,
        "method_invalid_count": method_invalid,
        "pending_execution_count": pending,
        "success_rate": sr,
        "harmful_intervention_count": harmful,
        "hir_unresolved_count": hir_unresolved,
        "harmful_intervention_rate": hir,
        "execution_budget": mean(budgets) if budgets else None,
        "task_count": len({sample.task_id for sample, _ in rows}),
        "metric_contract": {
            "success_rate": "passing benchmark outcomes divided by the full evaluation pool",
            "harmful_intervention_rate": (
                "confirmed structurally invalid, constraint-violating, score-lowering, or "
                "successful-control-regressing interventions divided by the full pool; "
                "a lower bound: unresolved harm is not added to the numerator"
            ),
            "execution_budget": (
                "mean accounted execution equivalents per pool row; historical evaluator/ledger "
                "records count as recorded; model calls and shared controls are not additional charges"
            ),
            "invalids": (
                "method, harness, and evaluator invalids are reported separately from task-incorrect outcomes"
            ),
        },
    }


def aggregate_grouped_results(
    pairs: Iterable[tuple[BaselineSample, MethodResult]], axis: str
) -> dict[str, dict[str, Any]]:
    """Return the paper's passive-by-benchmark or active-by-MAS breakdown."""

    if axis not in {"benchmark", "mas"}:
        raise ValueError("baseline metric group axis must be benchmark or mas")
    grouped: dict[str, list[tuple[BaselineSample, MethodResult]]] = {}
    for sample, result in pairs:
        key = getattr(sample, axis) or "unknown"
        grouped.setdefault(str(key), []).append((sample, result))
    return {key: aggregate_results(rows) for key, rows in sorted(grouped.items())}


def _outcome(sample: BaselineSample, result: MethodResult) -> tuple[bool | None, str]:
    selected = result.selected_execution
    if selected is not None:
        if selected.evaluation.get("method") == "static-validator":
            return None, "method_invalid"
        if selected.status in {"invalid", "failed"}:
            category = selected.evaluation.get("failure_category")
            return None, "method_invalid" if category == "method" else "evaluation_invalid"
        success = selected.success
        return success, "resolved" if success is not None else "evaluation_invalid"

    # A failed model call is not an abstention.  In particular, an active gate
    # that cannot fit a long prefix into the serving context must remain an
    # invalid method outcome rather than silently inheriting the host control.
    if result.status == "invalid":
        if result.diagnostics.get("failure_category") == "infrastructure":
            return None, "evaluation_invalid"
        return None, "method_invalid"

    # In the active setting, abstention means the already-observed host run was
    # delivered unchanged. The same is true for the explicit reference row.
    if result.method == "no_intervention" or (
        result.setting == "active" and result.intervention is None
    ):
        success = sample.control_success
        return success, "resolved" if success is not None else "evaluation_invalid"

    return None, "pending"


def _is_harmful(
    sample: BaselineSample,
    result: MethodResult,
    selected_success: bool | None,
) -> bool | None:
    if result.diagnostics.get("failure_category") == "infrastructure":
        return None
    if result.diagnostics.get("structurally_valid", True) is not True:
        return True
    if result.intervention is None:
        return False
    selected = result.selected_execution
    if result.diagnostics.get("constraint_violation", False) is True:
        return True
    if selected is not None and (
        selected.evaluation.get("constraint_violation") is True
        or selected.evaluation.get("constraints_satisfied") is False
        or (selected.status in {"invalid", "failed"}
            and selected.evaluation.get("failure_category") == "method")
    ):
        return True
    if selected is None:
        return None
    # Passive rows are admitted from failed source trajectories and compare a
    # repair with that source verdict. Active rows compare with the matched
    # uninterrupted control, which may differ from the failed source trace.
    source_score = (
        sample.control_score
        if result.setting == "active"
        else _numeric(sample.source_evaluation.get("score"))
    )
    selected_score = _numeric(selected.evaluation.get("score"))
    if (
        source_score is not None
        and selected_score is not None
        and selected_score < source_score
    ):
        return True
    control_success = (
        sample.control_success if result.setting == "active" else sample.source_success
    )
    if control_success is True and selected_success is False:
        return True
    if selected_success is None:
        return None
    if control_success is None and source_score is None:
        return None
    return False


def _numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None
