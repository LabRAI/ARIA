#!/usr/bin/env python3
"""Render the Qwen2.5-7B RQ1 baseline summaries as two Markdown tables."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]

METHODS = (
    ("zero_shot", "Zero-shot repair", "passive"),
    ("who_when", "Who&When + repair", "passive"),
    ("agentrx", "AgentRx + repair", "passive"),
    ("errorprobe", "ErrorProbe + repair", "passive"),
    ("dover", "DoVer", "passive"),
    ("aria-posthoc", "ARIA (SFT checkpoint)", "passive"),
    ("no_intervention", "No intervention", "active"),
    ("spark_to_fire", "Spark to Fire", "active"),
    ("safeagent", "SafeAgent", "active"),
    ("aria-runtime", "ARIA (gate checkpoint)", "active"),
)
NATIVE_VERSION = "v6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrected-config", type=Path, default=None,
                        help="Verify and render the model-free correction panel declared in YAML.")
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "outputs/baselines",
    )
    parser.add_argument(
        "--comparison-root", type=Path, default=None,
        help="Legacy root containing grouped comparison outputs.",
    )
    parser.add_argument(
        "--aria-root", type=Path,
        default=REPO_ROOT / "outputs/baselines/aria/qwen25-7b",
        help="Legacy root containing separately served ARIA outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--model",
        default="qwen25-7b",
        help="Backbone directory under outputs/baselines.",
    )
    parser.add_argument(
        "--model-label",
        default=None,
        help="Display label for the generated section.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append this model section to an existing Markdown report.",
    )
    parser.add_argument(
        "--temporary",
        action="store_true",
        help="Write a current native-v6 snapshot with provisional values marked by *.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.corrected_config is not None:
        import sys

        sys.path.insert(0, str(REPO_ROOT))
        from baselines.rescore_report import render_corrected_report
        from masfm.config import load_config

        report = render_corrected_report(load_config(args.corrected_config))
        output = (args.output or args.root / "RQ1.md").resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        print(output)
        return
    root = args.root.resolve()
    comparison_root = (args.comparison_root or root).resolve()
    aria_root = (root / "aria" / args.model).resolve()
    if args.temporary:
        output = (args.output or root / "RQ1-temp.md").resolve()
        lines = _render_temporary_report(root, args.model)
        output.parent.mkdir(parents=True, exist_ok=True)
        section = "\n".join(lines)
        if args.append and output.is_file():
            existing = output.read_text(encoding="utf-8").rstrip()
            heading = lines[0]
            existing = re.sub(
                rf"(?ms)^{re.escape(heading)}\n.*?(?=^# RQ1 Baselines|\Z)",
                "",
                existing,
            ).rstrip()
            output.write_text("\n\n".join([existing, section]) + "\n", encoding="utf-8")
        else:
            output.write_text(section + "\n", encoding="utf-8")
        print(output)
        return
    output = (args.output or root / "qwen25-7b" / "RQ1.md").resolve()
    pool_manifest = REPO_ROOT / "outputs/aria/pools/aria_full_processed_test_manifest.json"
    summaries = {
        directory: _load_method_summary(
            directory, root, comparison_root, aria_root, model=args.model
        )
        for directory, _, _ in METHODS
    }
    posthoc = [item for item in METHODS if item[2] == "passive"]
    runtime = [item for item in METHODS if item[2] == "active"]
    required_methods = [
        item
        for item in METHODS
        if not (args.model == "llama31-8b" and item[0] in {"aria-posthoc", "aria-runtime"})
    ]
    all_summaries = [summaries[key] for key, _, _ in required_methods if summaries[key]]
    missing_rows = [key for key, _, _ in required_methods if summaries[key] is None]
    invalid_total = sum(
        int(summary.get(key, 0) or 0)
        for summary in all_summaries
        for key in (
            "pending_execution_count",
            "invalid_evaluation_count",
            "method_invalid_count",
        )
    )
    lines = [
        f"# RQ1 Baselines: {args.model_label or _display_model(args.model)}",
        "",
        "Evaluation pool: `data/ARIA-Full-Processed/test`, materialized as "
        f"`{pool_manifest}`.",
        "",
        "Split contract: task-level, source-cell-stratified `60/20/20` "
        "train/validation/test with seed `0`; this report uses only `test`.",
        "",
        "The split is task-level 60/20/20. Post-hoc rows use failed source "
        "trajectories and receive the complete trajectory plus evaluator feedback; "
        "runtime rows use all test incidents and receive no evaluator feedback.",
        "",
        f"Report generated on {datetime.now().astimezone().strftime('%B %d, %Y')}. Scores are loaded from persisted terminal runs.",
        "",
        _native_status_line(root, args.model),
        "",
        (
            f"Comparison-baseline status: **{'yes' if invalid_total == 0 and not missing_rows else 'no'}** "
            f"(missing comparison rows: `{len(missing_rows)}`, unresolved/invalid counters "
            f"summed across comparison rows: `{invalid_total}`)."
            if args.model == "llama31-8b"
            else f"Paper-ready status: **{'yes' if invalid_total == 0 and not missing_rows else 'no'}** "
            f"(missing rows: `{len(missing_rows)}`, unresolved/invalid counters "
            f"summed across completed rows: `{invalid_total}`)."
        ),
        "",
    ]
    lines.extend(_render_native_results(root, args.model))
    lines.extend(_render_table("RQ1 Post-hoc", posthoc, summaries, "benchmark"))
    lines.extend(_render_table("RQ1 Runtime", runtime, summaries, "mas"))
    lines.extend(_render_benchmark_audit(METHODS, summaries, args.model))
    lines.extend(_render_fix_impact(root, args.model))
    if args.model == "llama31-8b" and summaries.get("aria-posthoc") is None:
        lines.extend(
            [
                "## ARIA Checkpoint",
                "",
                "The full Llama-3.1-8B ARIA checkpoint was not completed in this run. A partial SFT checkpoint exists under `outputs/aria/post_training/llama31-8b/full/checkpoints/sft`, but it was not substituted for the required deployable repair/gate checkpoint. The eight untuned comparison baselines are complete; no ARIA score is substituted from another backbone.",
                "",
            ]
        )
    lines.extend(
        [
            "## Reproducibility",
            "",
            f"- Model server: local vLLM comparison backbone for `{args.model}`.",
            _aria_reproducibility_line(args.model),
            _training_checkpoint_note(args.model),
            _job_configs_note(args.model),
            "- Replay: historical Stage-3 ledger when matched, then the local official evaluator. GAIA uses cached validation metadata; SWE-Bench-Lite uses an exact base-commit checkout, patch materialization, and the official Docker harness.",
            "- Runtime MAS surface: OpenManus, MetaGPT, Magentic-One, and AG2; Magentic-One and AG2 use tested substitute-only hooks with no arbitrary historical rollback claim.",
            "",
            "Each row's `summary.json` is the source of truth for counters and execution budget.",
            "",
            "## Invalid / Unresolved Audit",
            "",
            "The v4 tables below retain their original counters for audit, not as proof of correct execution. In particular, v4 GAIA resolver coverage does not mean proposed tools were executed, and v4 static-validator rejections were included in resolved counts. SWE patch-materialization failures combine model-output and missing-continuation problems and must not all be attributed to the official evaluator. Native v6 results, when present above, report actual continuation and fresh paired controls separately.",
            "",
        ]
    )
    lines.extend(_render_invalid_audit(METHODS, summaries, args.model))
    lines.extend(_render_native_invalid_audit(root, args.model))
    output.parent.mkdir(parents=True, exist_ok=True)
    section = "\n".join(lines)
    if args.append and output.is_file():
        existing = output.read_text(encoding="utf-8").rstrip()
        heading = lines[0]
        existing = re.sub(
            rf"(?ms)^{re.escape(heading)}\n.*?(?=^# RQ1 Baselines:|\Z)",
            "",
            existing,
        ).rstrip()
        output.write_text("\n\n".join([existing, section]) + "\n", encoding="utf-8")
    else:
        output.write_text(section + "\n", encoding="utf-8")
    print(output)


def _load_summary(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


TEMPORARY_NATIVE_METHODS = tuple(
    item for item in METHODS if item[0] not in {"aria-posthoc", "aria-runtime"}
)
TEMPORARY_NATIVE_TARGETS = {
    "passive": {"gaia": 48, "swe-bench-lite": 65},
    "active": {"gaia": 64, "swe-bench-lite": 79},
}


def _render_temporary_report(root: Path, model: str) -> list[str]:
    """Render the current native-v6 snapshot, including incomplete panels."""

    model_rows = 0
    model_target = 0
    terminal_panels = 0
    total_panels = len(TEMPORARY_NATIVE_METHODS)
    total_invalid = 0
    total_unresolved = 0
    lines = [
        f"# RQ1 Baselines Temporary: {_display_model(model)}",
        "",
        "This is a live snapshot of the corrected native-v6 re-execution for **GAIA** and **SWE-Bench-Lite** only. It is not the final RQ1 report.",
        "",
        f"Generated on {datetime.now().astimezone().strftime('%B %d, %Y at %H:%M %Z')}. Values marked with `*` can change when unfinished rows or invalid/unresolved rows are retried.",
        "",
        "Temporary SR uses the full benchmark target as denominator (`48` GAIA/`65` SWE for post-hoc and `64` GAIA/`79` SWE for runtime). For an incomplete panel this is a current lower bound, not a final score. `observed/target` shows how many rows are currently written.",
        "",
    ]
    snapshots: list[tuple[str, str, str, dict[str, Any]]] = []
    for key, label, setting in TEMPORARY_NATIVE_METHODS:
        mode = "posthoc" if setting == "passive" else "runtime"
        directory = root / key / model / f"rq1-{mode}-native-{NATIVE_VERSION}"
        state = _load_summary(directory / "run_state.json") or {}
        rows = _load_jsonl_rows(directory / "results.jsonl")
        target_counts = TEMPORARY_NATIVE_TARGETS[setting]
        model_rows += len(rows)
        model_target += sum(target_counts.values())
        if state.get("status") == "completed":
            terminal_panels += 1
        for benchmark in ("gaia", "swe-bench-lite"):
            group = _temporary_native_group(
                rows,
                benchmark=benchmark,
                target=target_counts[benchmark],
                setting=setting,
            )
            total_invalid += group["method_invalid"]
            total_unresolved += group["unresolved"]
            group["state"] = str(state.get("status") or "not started")
            group["panel_label"] = label
            group["observed"] = sum(row.get("benchmark") == benchmark for row in rows)
            snapshots.append((key, label, benchmark, group))

    lines.extend(
        [
            f"Progress: **{model_rows}/{model_target} rows written**, **{terminal_panels}/{total_panels} panels terminal**, method/schema invalid `{total_invalid}`, native/evaluator unresolved `{total_unresolved}`.",
            "",
            "## Current Native Results",
            "",
            "| Method | Benchmark | Observed/target | Official boolean verdicts | Correct / target | Current SR* | Control correct / target | Repairs gained | Regressions | Method invalid | Native/eval unresolved | Empty SWE patch | Panel status |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for _, label, benchmark, group in snapshots:
        star = "*" if group["needs_update"] else ""
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    benchmark,
                    f"{group['observed']}/{group['target']}",
                    f"{group['verdicts']}{star}",
                    f"{group['successes']}/{group['target']}{star}",
                    f"{_fmt_rate(group['successes'] / group['target'] if group['target'] else None)}{star}",
                    f"{group['control_successes']}/{group['target']}{star}",
                    f"{group['gained']}{star}",
                    f"{group['regressed']}{star}",
                    f"{group['method_invalid']}{star}",
                    f"{group['unresolved']}{star}",
                    f"{group['empty_patch']}{star}",
                    _temporary_panel_status(group),
                ]
            )
            + "|"
        )
    lines.extend(
        [
            "",
            "`Correct / target` and `Current SR*` use the full target denominator; unobserved rows are not silently counted as incorrect. `Control correct / target` is the matched uninterrupted native control. A SWE empty patch has an official boolean `correct=false` and is therefore resolved incorrect, not evaluator-unresolved.",
            "",
            "## Invalid / Unresolved Reasons",
            "",
            "| Method | Benchmark | Method/schema invalid | Native/eval unresolved | Top current reasons |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for _, label, benchmark, group in snapshots:
        reasons = "; ".join(
            f"{reason} ({count})" for reason, count in group["reasons"].most_common(3)
        ) or "none"
        lines.append(
            f"| {label} | {benchmark} | {group['method_invalid']} | {group['unresolved']} | {reasons} |"
        )
    lines.extend(
        [
            "",
            "## Protocol",
            "",
            "- Native MAS: OpenManus, MetaGPT, Magentic-One, and AG2.",
            f"- Local host model: `{model}` served through the native-v6 vLLM endpoint.",
            "- GAIA uses cached validation metadata and the quasi-exact official scorer.",
            "- SWE-Bench-Lite uses an exact base-commit checkout, host-observed workspace diff, and the official Docker harness.",
            "- This snapshot does not replace `RQ1.md`; it is intended to show currently available values while the remaining panel/retry work continues.",
            "",
        ]
    )
    return lines


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    rows.append(payload)
    return rows


def _selected_execution(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    executions = row.get("executions")
    if not isinstance(executions, list):
        return None
    selected_label = row.get("selected_execution_label")
    selected = next(
        (
            item
            for item in executions
            if isinstance(item, Mapping)
            and (selected_label is None or item.get("label") == selected_label)
        ),
        None,
    )
    if isinstance(selected, Mapping):
        return selected
    return None


def _bool_outcome(value: Mapping[str, Any] | None) -> bool | None:
    if not isinstance(value, Mapping):
        return None
    for key in ("correct", "success"):
        candidate = value.get(key)
        if isinstance(candidate, bool):
            return candidate
    return None


def _temporary_native_group(
    rows: list[dict[str, Any]],
    *,
    benchmark: str,
    target: int,
    setting: str,
) -> dict[str, Any]:
    selected_rows = [row for row in rows if row.get("benchmark") == benchmark]
    method_invalid = 0
    unresolved = 0
    verdicts = 0
    successes = 0
    control_successes = 0
    gained = 0
    regressed = 0
    empty_patch = 0
    reasons: Counter[str] = Counter()
    for row in selected_rows:
        control = row.get("control_success")
        if isinstance(control, bool) and control:
            control_successes += 1
        if row.get("status") == "invalid":
            diagnostics = row.get("diagnostics") or {}
            category = diagnostics.get("failure_category")
            reason = str(diagnostics.get("error") or "unknown")
            if category == "infrastructure":
                unresolved += 1
            else:
                method_invalid += 1
            reasons[reason] += 1
            continue
        execution = _selected_execution(row)
        evaluation = execution.get("evaluation") if execution else {}
        if not isinstance(evaluation, Mapping):
            evaluation = {}
        if execution is not None and execution.get("status") in {"invalid", "failed"}:
            category = evaluation.get("failure_category")
            reason = str(
                execution.get("error")
                or evaluation.get("reason")
                or "execution failed without a reason"
            )
            if category == "method":
                method_invalid += 1
            else:
                unresolved += 1
            reasons[reason] += 1
            continue
        outcome = _bool_outcome(evaluation)
        if outcome is None and setting == "active" and row.get("intervention") is None:
            outcome = control if isinstance(control, bool) else None
        if outcome is None:
            unresolved += 1
            reasons[str((execution or {}).get("error") or evaluation.get("reason") or "missing boolean verdict")] += 1
            continue
        verdicts += 1
        if outcome:
            successes += 1
        if isinstance(control, bool):
            if outcome and not control:
                gained += 1
            elif not outcome and control:
                regressed += 1
        if evaluation.get("method") == "swebench_official_docker":
            if evaluation.get("harness_status") == "empty_patch":
                empty_patch += 1
    needs_update = len(selected_rows) < target or method_invalid > 0 or unresolved > 0
    return {
        "target": target,
        "verdicts": verdicts,
        "successes": successes,
        "control_successes": control_successes,
        "gained": gained,
        "regressed": regressed,
        "method_invalid": method_invalid,
        "unresolved": unresolved,
        "empty_patch": empty_patch,
        "needs_update": needs_update,
        "reasons": reasons,
    }


def _temporary_panel_status(group: Mapping[str, Any]) -> str:
    state = str(group.get("state") or "not started")
    if state == "completed" and group.get("needs_update"):
        return "complete; retry pending"
    if state == "completed":
        return "complete"
    if state == "running":
        return "running"
    return state


def _render_native_results(root: Path, model: str) -> list[str]:
    """Keep a changed-host experiment separate from the historical-host table."""
    lines = [
        "## GAIA / SWE Native Re-execution",
        "",
        "**Protocol correction:** the legacy v4 tables use text-splice/ledger scoring. "
        "They do not demonstrate that a repair instruction was executed. The results in this section "
        "use each incident's native MAS in an isolated environment, fresh unpatched controls, "
        "real repository/attachment access, and benchmark evaluators. No fuzzy ledger match is used.",
        "",
        "**Host identity:** v6 explicitly uses the local comparison backbone as the host for all four MAS. "
        "This is a fixed-local-host continuation experiment, not a reproduction of the original "
        "GPT/Claude/70B host cells. Original-host credentials/provenance are not silently substituted. "
        "Runtime decisions are offline prefix interventions, not a claim of live checkpoint restoration.",
        "",
        "The other four benchmarks are not rerun. Their legacy aggregate is intentionally not mixed "
        "with this changed-host experiment. A zero with full evaluation coverage remains a valid model "
        "result; unresolved infrastructure is listed separately.",
        "",
        "| Method | Benchmark | n | Patched/control SR | Unpatched SR | Repairs gained | Regressions | Method invalid | Infra/eval unresolved |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    completed = 0
    for key, label, setting in METHODS:
        method = "aria" if key.startswith("aria-") else key
        mode = "posthoc" if setting == "passive" else "runtime"
        directory = root / method / model / f"rq1-{mode}-native-{NATIVE_VERSION}"
        summary = _load_summary(directory / "summary.json")
        state = _load_summary(directory / "run_state.json")
        if not summary or not state or state.get("status") != "completed":
            lines.append(f"| {label} | both | -- | pending | -- | -- | -- | -- | -- |")
            continue
        completed += 1
        result_path = directory / "results.jsonl"
        rows = [json.loads(line) for line in result_path.read_text().splitlines() if line.strip()]
        groups = summary.get("groups_by_benchmark") or summary.get("groups") or {}
        for benchmark in ("gaia", "swe-bench-lite"):
            selected_rows = [row for row in rows if row.get("benchmark") == benchmark]
            group = groups.get(benchmark) or {}
            controls = sum(row.get("control_success") is True for row in selected_rows)
            gained = regressed = 0
            for row in selected_rows:
                executions = row.get("executions") or []
                execution = next((item for item in executions if item.get("label") == row.get("selected_execution_label")),
                                 executions[-1] if executions else {})
                evaluation = execution.get("evaluation") or {}
                success = evaluation.get("success", evaluation.get("correct"))
                if not executions and row.get("status") != "invalid" and row.get("intervention") is None:
                    success = row.get("control_success")
                gained += success is True and row.get("control_success") is False
                regressed += success is False and row.get("control_success") is True
            n = len(selected_rows)
            lines.append(f"| {label} | {benchmark} | {n} | {_fmt_rate(group.get('success_rate'))} | "
                         f"{_fmt_rate(controls / n if n else None)} | {gained} | {regressed} | "
                         f"{group.get('method_invalid_count', 0)} | {group.get('invalid_evaluation_count', 0)} |")
    lines.extend(["", f"Terminal native rows: {completed}/{len(METHODS)}. "
                  "Each result preserves native logs, final output, evaluator artifacts, and its shared control path.", "",
                  "### Legacy Results", "",
                  "The following v4 results are preserved for comparison. GAIA/SWE values and resolved coverage "
                  "must not be interpreted as results of the corrected native execution path.", ""])
    return lines


def _native_status_line(root: Path, model: str) -> str:
    total = terminal = infra = method_invalid = 0
    for key, _, setting in METHODS:
        method = "aria" if key.startswith("aria-") else key
        mode = "posthoc" if setting == "passive" else "runtime"
        directory = root / method / model / f"rq1-{mode}-native-{NATIVE_VERSION}"
        state = _load_summary(directory / "run_state.json")
        result_path = directory / "results.jsonl"
        if not state or not result_path.is_file():
            continue
        rows = [json.loads(line) for line in result_path.read_text().splitlines() if line.strip()]
        for row in rows:
            if row.get("benchmark") not in {"gaia", "swe-bench-lite"}:
                continue
            total += 1
            if state.get("status") == "completed":
                terminal += 1
            if row.get("status") == "invalid":
                if row.get("diagnostics", {}).get("failure_category") == "infrastructure":
                    infra += 1
                else:
                    method_invalid += 1
            for execution in row.get("executions") or []:
                evaluation = execution.get("evaluation") or {}
                if evaluation.get("failure_category") == "infrastructure":
                    infra += 1
    if total == 0:
        return f"Corrected GAIA/SWE status: **pending** (native {NATIVE_VERSION} has not produced terminal rows)."
    if terminal < total:
        return f"Corrected GAIA/SWE status: **in progress** ({terminal}/{total} rows terminal; native {NATIVE_VERSION} is authoritative when complete)."
    if infra:
        return f"Corrected GAIA/SWE status: **blocked by infrastructure** ({infra} unresolved native/evaluator rows; method-invalid rows: {method_invalid})."
    return f"Corrected GAIA/SWE status: **terminal** ({total} rows with official boolean outcomes; method-invalid rows: {method_invalid})."


def _render_native_invalid_audit(root: Path, model: str) -> list[str]:
    """Explain every non-terminal v6 row without collapsing infrastructure into HIR."""
    lines = [
        "## Native V6 Invalid / Unresolved Audit",
        "",
        "This audit is computed from native v6 `results.jsonl`, not inferred from SR. "
        "A model/schema invalid means the baseline did not produce an executable intervention. "
        "An infrastructure/evaluator unresolved means the method output existed but the native host, "
        "benchmark resolver, or official harness did not produce a boolean verdict. "
        "A SWE empty patch is resolved incorrect, not unresolved.",
        "",
        "| Method | Benchmark | Method/schema invalid | Native/eval unresolved | Empty SWE patch | Official evaluator runs | Top reasons |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for key, label, setting in METHODS:
        method = "aria" if key.startswith("aria-") else key
        mode = "posthoc" if setting == "passive" else "runtime"
        result_path = root / method / model / f"rq1-{mode}-native-{NATIVE_VERSION}" / "results.jsonl"
        if not result_path.is_file():
            continue
        rows = [json.loads(line) for line in result_path.read_text().splitlines() if line.strip()]
        for benchmark in ("gaia", "swe-bench-lite"):
            selected_rows = [row for row in rows if row.get("benchmark") == benchmark]
            if not selected_rows:
                continue
            method_invalid = unresolved = empty_patch = official = 0
            reasons: Counter[str] = Counter()
            for row in selected_rows:
                if row.get("status") == "invalid":
                    category = str(row.get("diagnostics", {}).get("failure_category") or "method")
                    reason = str(row.get("diagnostics", {}).get("error") or "unknown")
                    if category == "infrastructure":
                        unresolved += 1
                    else:
                        method_invalid += 1
                    reasons[reason] += 1
                    continue
                executions = row.get("executions") or []
                execution = next((item for item in executions if item.get("label") == row.get("selected_execution_label")),
                                 executions[-1] if executions else {})
                evaluation = execution.get("evaluation") or {}
                if evaluation.get("method") == "swebench_official_docker":
                    official += 1
                    if evaluation.get("harness_status") == "empty_patch":
                        empty_patch += 1
                if execution.get("status") in {"invalid", "failed"} or evaluation.get("correct") is None:
                    unresolved += 1
                    reasons[str(execution.get("error") or evaluation.get("reason") or "unknown")] += 1
            top = "; ".join(f"{reason} ({count})" for reason, count in reasons.most_common(3)) or "none"
            lines.append(f"| {label} | {benchmark} | {method_invalid} | {unresolved} | {empty_patch} | {official} | {top} |")
    lines.extend(["", "Native v6 rows with a boolean official verdict are the only rows used for corrected SR. "
                  "Counts above intentionally distinguish a model that returned a wrong answer from a host or evaluator that never returned an answer.", ""])
    return lines


def _load_method_summary(
    method: str,
    root: Path,
    comparison_root: Path,
    aria_root: Path,
    *,
    model: str,
) -> dict[str, Any] | None:
    """Read the stable per-method layout, with legacy fallback for old runs."""

    passive = {"zero_shot", "who_when", "agentrx", "errorprobe", "dover", "aria-posthoc"}
    stable_method = "aria" if method.startswith("aria-") else method
    setting_prefix = "rq1-posthoc-test" if method in passive else "rq1-runtime-test"
    for version in ("v4", "v3", "v2", ""):
        stable_setting = f"{setting_prefix}-{version}" if version else setting_prefix
        stable = root / method / model / stable_setting
        if method.startswith("aria-"):
            stable = root / "aria" / model / stable_setting
        summary = _load_summary(stable / "summary.json")
        if summary is not None:
            return summary
    legacy_stable = root / stable_method / model / setting_prefix
    summary = _load_summary(legacy_stable / "summary.json")
    if summary is not None:
        return summary
    if method.startswith("aria-"):
        legacy_name = "posthoc-aria" if method == "aria-posthoc" else "runtime-aria"
        return _load_summary(aria_root / legacy_name / "summary.json")
    legacy = {
        "zero_shot": "posthoc-zero-shot",
        "who_when": "posthoc-who-when",
        "agentrx": "posthoc-agentrx",
        "errorprobe": "posthoc-errorprobe",
        "dover": "posthoc-dover",
        "no_intervention": "runtime-no-intervention",
        "spark_to_fire": "runtime-spark-to-fire",
        "safeagent": "runtime-safeagent",
    }.get(method)
    return _load_summary(comparison_root / legacy / "summary.json") if legacy else None


def _render_table(
    title: str,
    rows: list[tuple[str, str, str]],
    summaries: dict[str, dict[str, Any] | None],
    group_axis: str,
) -> list[str]:
    groups = sorted(
        {
            str(group)
            for key, _, _ in rows
            if summaries[key]
            for group in (summaries[key].get("groups") or {})
        }
    )
    lines = [
        f"## {title}",
        "",
        "| Method | " + " | ".join(groups) + " | SR | Resolved Acc. | Eval coverage | HIR | EB | n | Method invalid | Eval unresolved |",
        "| --- | " + " | ".join("---:" for _ in groups) + " | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, label, _ in rows:
        summary = summaries[key]
        if summary is None:
            lines.append("| " + label + " | " + " | ".join("--" for _ in groups) + " | -- | -- | -- | -- | -- | -- | missing | missing |")
            continue
        grouped = summary.get("groups") or {}
        group_values = [
            _fmt_rate((grouped.get(group) or {}).get("success_rate")) for group in groups
        ]
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    *group_values,
                    _fmt_rate(summary.get("success_rate")),
                    _fmt_rate(_resolved_accuracy(summary)),
                    _fmt_rate(_evaluation_coverage(summary)),
                    _fmt_rate(summary.get("harmful_intervention_rate")),
                    _fmt(summary.get("execution_budget"), 2),
                    str(summary.get("sample_count") or 0),
                    str(int(summary.get("method_invalid_count", 0) or 0)),
                    str(int(summary.get("invalid_evaluation_count", 0) or 0)),
                ]
            )
            + " |"
        )
    lines.extend(["", "Group axis: `" + group_axis + "`; values are SR.", ""])
    return lines


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def _display_model(model: str) -> str:
    return {
        "qwen25-7b": "Qwen2.5-7B",
        "llama31-8b": "Llama-3.1-8B",
    }.get(model, model)


def _aria_reproducibility_line(model: str) -> str:
    if model == "qwen25-7b":
        return "- ARIA passive: Qwen2.5-7B SFT LoRA adapter `qwen25-7b-aria-sft-v3` over the same base model; ARIA runtime: standalone `grpo_gate/checkpoint` alias `qwen25-7b-aria`."
    if model == "llama31-8b":
        return "- ARIA passive/runtime: the full Llama-3.1-8B deployable checkpoint was not completed; a partial SFT checkpoint was not substituted. The comparison rows are untuned Llama-3.1-8B baselines."
    return f"- ARIA passive/runtime checkpoint provenance is unavailable for `{model}`."


def _training_checkpoint_note(model: str) -> str:
    if model == "llama31-8b":
        return "- The full Llama-3.1-8B ARIA checkpoint was intentionally not included in the comparison tables because deployable gate training was incomplete; no Qwen checkpoint score is substituted."
    return "- The final gate checkpoint was not used for passive repair because its training report records `gradient_status: no_gradient` and its attribution probe returned null fields. The SFT checkpoint passed localization, attribution, and structured-repair probes."


def _job_configs_note(model: str) -> str:
    if model == "llama31-8b":
        return "- Corrected GAIA/SWE job config: `configs/baselines/rq1_llama31_8b_native_v6.yml`; all native-v6 rows use the local Llama-3.1-8B-Instruct vLLM alias `llama31-8b-native-v6`. Legacy v4 config paths are retained only for historical aggregate rows."
    return "- Corrected GAIA/SWE job config: `configs/baselines/rq1_qwen25_7b_native_v6.yml`; all native-v6 rows use the local Qwen2.5-7B-Instruct vLLM alias `qwen25-7b-native-v6`. Legacy v3/v4 config paths are retained only for historical aggregate rows."


def _fmt_rate(value: Any, digits: int = 2, signed: bool = False) -> str:
    if value is None:
        return "--"
    amount = 100.0 * float(value)
    return f"{amount:+.{digits}f}%" if signed else f"{amount:.{digits}f}%"


def _resolved_accuracy(summary: dict[str, Any]) -> float | None:
    resolved = int(summary.get("resolved_count", 0) or 0)
    successful = int(summary.get("successful_count", 0) or 0)
    return successful / resolved if resolved else None


def _evaluation_coverage(summary: dict[str, Any]) -> float | None:
    total = int(summary.get("sample_count", 0) or 0)
    method_invalid = int(summary.get("method_invalid_count", 0) or 0)
    resolved = int(summary.get("resolved_count", 0) or 0)
    eligible = total - method_invalid
    return resolved / eligible if eligible > 0 else None


def _render_invalid_audit(
    methods: tuple[tuple[str, str, str], ...],
    summaries: dict[str, dict[str, Any] | None],
    model: str,
) -> list[str]:
    lines: list[str] = []
    for key, label, _ in methods:
        summary = summaries.get(key)
        if summary is None:
            continue
        path = _summary_results_path(key, summary, model)
        reasons = _reason_counts(path)
        lines.append(
            f"- **{label}**: method-invalid `{int(summary.get('method_invalid_count', 0) or 0)}`, evaluator-invalid `{int(summary.get('invalid_evaluation_count', 0) or 0)}`."
        )
        for reason, count in reasons[:5]:
            lines.append(f"  - `{count}`: {reason}")
    lines.append("")
    return lines


def _render_benchmark_audit(
    methods: tuple[tuple[str, str, str], ...],
    summaries: dict[str, dict[str, Any] | None],
    model: str,
) -> list[str]:
    """Show evaluator coverage for the two benchmarks with special plumbing."""

    lines = [
        "## GAIA / SWE-Bench-Lite Audit",
        "",
        "| Method | GAIA SR | GAIA resolved acc. | GAIA eval unresolved | SWE patch materialized | SWE resolved acc. | SWE eval unresolved |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, label, _ in methods:
        summary = summaries.get(key)
        if summary is None:
            continue
        groups = summary.get("groups_by_benchmark") or summary.get("groups") or {}
        gaia = groups.get("gaia") or {}
        swe = groups.get("swe-bench-lite") or {}
        result_path = _summary_results_path(key, summary, model)
        patch_materialized = _count_swebench_official(result_path)
        lines.append(
            f"| {label} | {_fmt_rate(gaia.get('success_rate'))} | {_fmt_rate(_resolved_accuracy(gaia))} | "
            f"{int(gaia.get('invalid_evaluation_count', 0) or 0)} | {patch_materialized} | "
            f"{_fmt_rate(_resolved_accuracy(swe))} | {int(swe.get('invalid_evaluation_count', 0) or 0)} |"
        )
    lines.extend(
        [
            "",
            "`SWE patch materialized` counts selected executions that reached the official SWE-Bench evaluator; a patch can still fail the official tests. Method-invalid rows are excluded from evaluator coverage and remain reported in the main tables.",
            "",
        ]
    )
    return lines


def _count_swebench_official(path: Path) -> int:
    count = 0
    if not path.is_file():
        return count
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("benchmark") != "swe-bench-lite":
                continue
            selected_label = row.get("selected_execution_label")
            for execution in row.get("executions", []):
                if not isinstance(execution, dict):
                    continue
                if selected_label is not None and execution.get("label") != selected_label:
                    continue
                if (execution.get("evaluation") or {}).get("method") == "swebench-official":
                    count += 1
                break
    return count


def _summary_results_path(
    method: str, summary: dict[str, Any], model: str = "qwen25-7b"
) -> Path:
    results = summary.get("results_path")
    if isinstance(results, str):
        return Path(results)
    setting_prefix = "rq1-posthoc-test" if method in {"zero_shot", "who_when", "agentrx", "errorprobe", "dover", "aria-posthoc"} else "rq1-runtime-test"
    base = REPO_ROOT / "outputs/baselines" / ("aria" if method.startswith("aria-") else method) / model
    for version in ("v4", "v3", "v2", ""):
        setting = f"{setting_prefix}-{version}" if version else setting_prefix
        candidate = base / setting / "results.jsonl"
        if candidate.is_file():
            return candidate
    return base / f"{setting_prefix}-v4" / "results.jsonl"


def _reason_counts(path: Path) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "invalid":
                diagnostics = row.get("diagnostics") or {}
                reason = str(
                    diagnostics.get("error")
                    or diagnostics.get("structured_generation_error")
                    or "unknown method error"
                )
                counts[reason] = counts.get(reason, 0) + 1
            executions = row.get("executions", [])
            selected_label = row.get("selected_execution_label")
            selected = [
                execution
                for execution in executions
                if isinstance(execution, dict)
                and (
                    selected_label is None
                    or execution.get("label") == selected_label
                )
            ]
            for execution in selected[:1]:
                if execution.get("status") != "invalid":
                    continue
                evaluation = execution.get("evaluation") or {}
                reason = str(
                    execution.get("error")
                    or evaluation.get("reason")
                    or "unknown evaluator error"
                )
                counts[reason] = counts.get(reason, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _render_fix_impact(root: Path, model: str = "qwen25-7b") -> list[str]:
    methods = (
        ("zero_shot", "Zero-shot repair"),
        ("who_when", "Who&When + repair"),
        ("agentrx", "AgentRx + repair"),
        ("errorprobe", "ErrorProbe + repair"),
        ("dover", "DoVer"),
        ("aria", "ARIA post-hoc"),
    )
    lines = [
        "## Fix Impact",
        "",
        "| Method | v3 SR | v4 SR | Delta | v3 evaluator unresolved | v4 evaluator unresolved |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method, label in methods:
        old_dir = root / method / model / "rq1-posthoc-test-v3"
        new_dir = root / method / model / "rq1-posthoc-test-v4"
        old = _load_summary(old_dir / "summary.json")
        new = _load_summary(new_dir / "summary.json")
        if method == "aria":
            old_dir = root / "aria" / model / "rq1-posthoc-test-v3"
            new_dir = root / "aria" / model / "rq1-posthoc-test-v4"
            old = _load_summary(old_dir / "summary.json")
            new = _load_summary(new_dir / "summary.json")
        if old is None or new is None:
            continue
        old_sr = old.get("success_rate")
        new_sr = new.get("success_rate")
        if old_sr is None or new_sr is None:
            continue
        lines.append(
            f"| {label} | {_fmt_rate(old_sr)} | {_fmt_rate(new_sr)} | {_fmt_rate(float(new_sr) - float(old_sr), signed=True)} | "
            f"{int(old.get('invalid_evaluation_count', 0) or 0)} | "
            f"{int(new.get('invalid_evaluation_count', 0) or 0)} |"
        )
    lines.extend(
        [
            "",
            "The v4 evaluator fixes GAIA metadata resolution and routes SWE-Bench-Lite through exact checkout plus official Docker evaluation. v3 remains available for comparison; no old output directory is overwritten.",
            "",
        ]
    )
    return lines


def _certified_rejections(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if any(
                execution.get("evaluation", {}).get("reason")
                == "cut retains the certified fault"
                for execution in row.get("executions", [])
            ):
                count += 1
    return count


if __name__ == "__main__":
    main()
