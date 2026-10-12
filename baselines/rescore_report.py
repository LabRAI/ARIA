"""Render and verify corrected RQ1 tables from one declared YAML result panel."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from baselines.core.datasets import load_baseline_samples
from baselines.core.evaluation import aggregate_results, aggregate_grouped_results
from baselines.capability_correction import verify_capability_correction, verify_context_budget_correction, verify_invocation_correction
from baselines.reevaluate import verify_evaluation_recovery
from baselines.preflight_recovery import verify_preflight_recovery
from baselines.reference_projection import verify_reference_recovery_overlay
from baselines.delivery_audit import native_delivery_evidence, native_trial_backend_faults
from baselines.rescore import controller_adapter_rejections, read_rows, sample_with_control
from baselines.runner import _resolve, _result_from_row, _sha256, _select_evaluation_pool

LABELS = {
    "zero_shot": "Zero-shot repair", "who_when": "Who&When + repair",
    "agentrx": "AgentRx + repair", "errorprobe": "ErrorProbe + repair",
    "dover": "DoVer", "no_intervention": "No intervention",
    "spark_to_fire": "Spark to Fire", "safeagent": "SafeAgent",
    "aria_passive": "ARIA (SFT checkpoint)", "aria_active": "ARIA (gate checkpoint)",
}
BENCHMARKS = ("gsm-plus", "mmlu", "humaneval", "olympiadbench", "gaia", "swe-bench-lite")
SYSTEMS = ("openmanus", "metagpt", "magentic-one", "ag2")


def verified_summaries(config: Any) -> list[dict[str, Any]]:
    root = config.config_path.parent.resolve()
    pools: dict[str, Any] = {}
    summaries = []
    for spec in config.baselines:
        output = _resolve(root, spec["output_dir"])
        summary = json.loads((output / "summary.json").read_text())
        state = json.loads((output / "run_state.json").read_text())
        manifest = json.loads((output / "manifest.json").read_text())
        if state["status"] != "completed" or state["identity"]["spec"] != spec:
            raise ValueError(f"incomplete or mismatched corrected panel: {output}")
        for name, digest in manifest["artifacts"].items():
            if _sha256(output / name) != digest:
                raise ValueError(f"artifact hash mismatch: {output / name}")
        input_path = _resolve(root, spec["input_path"])
        if _sha256(input_path) != state["identity"]["input_sha256"]:
            raise ValueError("evaluation pool changed since correction")
        for path, digest in state["identity"]["sources"].items():
            if _sha256(Path(path)) != digest:
                raise ValueError(f"correction source changed: {path}")
        for path, digest in state["identity"].get("control_sources", {}).items():
            if _sha256(Path(path)) != digest:
                raise ValueError(f"matched control changed since correction: {path}")
        if str(input_path) not in pools:
            pools[str(input_path)] = load_baseline_samples(input_path, input_format=spec["input_format"])
        samples, _ = _select_evaluation_pool(
            pools[str(input_path)], setting=summary["setting"], split=spec.get("split"),
        )
        samples_by_id = {sample.sample_id: sample for sample in samples}
        rows = read_rows(output / "results.jsonl")
        for row in rows:
            if row.get("invocation_correction"):
                verify_invocation_correction(row)
            if row.get("context_budget_correction"):
                verify_context_budget_correction(row)
            if row.get("reference_control_recovery", {}).get("standalone"):
                verify_reference_recovery_overlay(row)
            if row.get("preflight_recovery"):
                verify_preflight_recovery(row)
            if row.get("evaluation_recovery"):
                verify_evaluation_recovery(row, samples_by_id.get(row["sample_id"]))
            if row.get("capability_correction"):
                verify_capability_correction(row)
            if ("native_delivery_audit" in row
                    and row["native_delivery_audit"] != native_delivery_evidence(row)):
                raise ValueError(f"native delivery evidence changed: {row['sample_id']}")
            if row.get("native_delivery_audit", {}).get("websurfer_endpoint_errors", 0):
                raise ValueError(f"native backend errors require correction before final publication: {row['sample_id']}")
            if "native_trial_backend_faults" in row:
                if row["native_trial_backend_faults"] != native_trial_backend_faults(row):
                    raise ValueError(f"native trial evidence changed: {row['sample_id']}")
                if row["native_trial_backend_faults"]:
                    raise ValueError(f"native trial backend errors require correction before final publication: {row['sample_id']}")
        capability_rows = [row for row in rows if row.get("capability_correction")]
        correction = summary.get("correction", {})
        invocation_rows = [row for row in rows if row.get("invocation_correction")]
        if (correction.get("invocation_corrections", 0) != len(invocation_rows)
                or correction.get("invocation_extra_native_executions", 0) != sum(row["invocation_correction"]["extra_native_executions"] for row in invocation_rows)):
            raise ValueError("invocation correction accounting disagrees with rows")
        budget_rows = [row for row in rows if row.get("context_budget_correction")]
        if (correction.get("context_budget_corrections", 0) != len(budget_rows)
                or correction.get("context_extra_native_executions", 0) != sum(row["context_budget_correction"]["extra_native_executions"] for row in budget_rows)):
            raise ValueError("context-budget correction accounting disagrees with rows")
        preflights = [row for row in rows if row.get("preflight_recovery")]
        if (correction.get("preflight_recoveries", 0) != len(preflights)
                or correction.get("preflight_native_executions", 0) != sum(row["preflight_recovery"]["extra_native_executions"] for row in preflights)):
            raise ValueError(f"preflight recovery accounting disagrees with rows: {output}")
        reevaluated = [row for row in rows if row.get("evaluation_recovery")]
        if (correction.get("reevaluated_rows", 0) != len(reevaluated)
                or correction.get("recovered_evaluations", 0) != sum(len(row["evaluation_recovery"]["rescored_labels"]) for row in reevaluated)):
            raise ValueError(f"reevaluation accounting disagrees with rows: {output}")
        if (correction.get("capability_corrections", 0) != len(capability_rows)
                or correction.get("capability_extra_native_executions", 0) != sum(
                    row["capability_correction"]["extra_native_executions"] for row in capability_rows)):
            raise ValueError(f"capability correction accounting disagrees with rows: {output}")
        if ("controller_adapter_rejections" in summary
                and summary["controller_adapter_rejections"] != controller_adapter_rejections(rows)):
            raise ValueError(f"controller adapter audit disagrees with rows: {output}")
        if {row["sample_id"] for row in rows} != set(samples_by_id):
            raise ValueError(f"corrected panel lost pool rows: {output}")
        pairs = [(sample_with_control(samples_by_id[row["sample_id"]], row), _result_from_row(row)) for row in rows]
        measured = aggregate_results(pairs)
        for key, value in measured.items():
            if summary[key] != value:
                raise ValueError(f"summary disagrees with rows: {output}: {key}")
        for axis in ("benchmark", "mas"):
            if summary[f"groups_by_{axis}"] != aggregate_grouped_results(pairs, axis):
                raise ValueError(f"group summary disagrees with rows: {output}: {axis}")
        for benchmark, counts in summary.get("outcome_audit_by_benchmark", {}).items():
            if sum(counts.values()) != summary["groups_by_benchmark"][benchmark]["sample_count"]:
                raise ValueError(f"zero-score audit does not cover the benchmark pool: {output}: {benchmark}")
        if summary["groups"] != summary[f"groups_by_{summary['group_axis']}"]:
            raise ValueError(f"display groups disagree with rows: {output}")
        for metric in ("success_rate", "harmful_intervention_rate", "execution_budget"):
            if not isinstance(summary[metric], (int, float)) or not math.isfinite(summary[metric]):
                raise ValueError(f"non-numeric final metric: {output}: {metric}")
        if measured["pending_execution_count"]:
            raise ValueError(f"pending method decisions: {output}")
        summaries.append(summary)
    return summaries


def _table(rows: list[dict[str, Any]], axis: str) -> list[str]:
    groups = BENCHMARKS if axis == "benchmark" else SYSTEMS
    lines = [
        "| Method | " + " | ".join(groups) + " | SR | HIR | EB | n |",
        "| --- | " + " | ".join(["---:"] * (len(groups) + 4)) + " |",
    ]
    for row in rows:
        grouped = row[f"groups_by_{axis}"]
        if set(grouped) != set(groups):
            raise ValueError("final report is missing a benchmark or MAS")
        values = [f"{grouped[group]['success_rate']:.2%}" for group in groups]
        lines.append("| " + " | ".join([
            LABELS[row["method"]], *values, f"{row['success_rate']:.2%}",
            f"{row['harmful_intervention_rate']:.2%}", f"{row['execution_budget']:.2f}",
            str(row["sample_count"]),
        ]) + " |")
    return lines


def render_corrected_report(config: Any) -> str:
    summaries = verified_summaries(config)
    config_name = str(config.config_path.resolve().relative_to(Path(__file__).resolve().parents[1]))
    native_versions = {match.group(1) for row in summaries for path in row["correction"]["sources"]
                       if (match := re.search(r"/rq1-(?:posthoc|runtime)-(native-v\d+)/results.jsonl$", path))}
    if len(native_versions) > 1:
        raise ValueError("final projection must not mix native execution versions")
    native_version = next(iter(native_versions), "native-v6")
    fresh_native = native_version != "native-v6"
    native_note = (
        f"GAIA/SWE comparison rows use {native_version} re-execution with 4096-token host outputs, "
        "48000-character historical context and fresh matched controls. Each stored attempt is executed once in the main batch; "
        "any independently verified infrastructure recovery is disclosed separately and charged in EB. "
        "Magentic-One resets consecutive stalls for a new plan while keeping its total turn bound; executor scratch scripts are excluded from SWE submissions. "
        "Workspace relocation preserves repository-relative filenames, and Magentic-One requests are buffered using actual server token counts. "
        "Stored method decisions are reused, with the ordinary method protocol retried once only when the source decision was invalid. "
        if fresh_native else
        "GAIA/SWE comparison-baseline rows reuse the existing native-v6 local-backbone runs and matched controls. "
    )
    lines = [
        "# RQ1 Baselines", "", "## Corrected Results", "",
        "All cells and overall columns below are recomputed from the same per-sample corrected results. "
        "The fixed test pools are unchanged: post-hoc uses 3,949 failed source trajectories; runtime uses 4,886 source incidents.", "",
        "HIR is one conservative value: confirmed harmful interventions divided by the full pool. "
        "Unresolved harm is not added to its numerator. SR also uses the full pool; a genuine zero is retained.", "",
        "EB retains the recorded execution-equivalent accounting: post-hoc = source + recorded attempts; "
        "runtime = max(source, recorded attempts). Historical evaluator/ledger entries count as recorded. "
        "This is not a measurement of GPU cost or fresh full MAS runs; rescoring adds no MAS executions.", "",
        native_note +
        "Other benchmarks retain historical trajectories, with all executable HumanEval candidates rescored after Python-code extraction. "
        "GAIA is scored from the delivered native answer, not quoted history. "
        "Host/protocol differences across these benchmark groups remain; this correction is not a new uniform native six-benchmark experiment.", "",
    ]
    for model, title in (("qwen25-7b", "Qwen2.5-7B"), ("llama31-8b", "Llama-3.1-8B")):
        panel = [row for row in summaries if row["model_identity"] == model and not row["method"].startswith("aria_")]
        if {row["method"] for row in panel} != set(LABELS) - {"aria_active", "aria_passive"}:
            raise ValueError(f"missing or unexpected comparison methods for {model}")
        lines += [f"## {title}", "", "### Post-hoc", ""]
        lines += _table([row for row in panel if row["setting"] == "passive"], "benchmark")
        lines += ["", "### Runtime", ""]
        lines += _table([row for row in panel if row["setting"] == "active"], "mas") + [""]
    aria = [row for row in summaries if row["method"].startswith("aria_")]
    if aria:
        lines += ["## ARIA Historical Reference", "",
                  "Qwen checkpoints are retained as a separate reference, not rerun on the native-v6 comparison hosts. "
                  "The runtime checkpoint abstained on every incident. No Llama ARIA result is fabricated.", ""]
        lines += _table([row for row in aria if row["setting"] == "passive"], "benchmark") + [""]
        lines += _table([row for row in aria if row["setting"] == "active"], "mas") + [""]
    zeroes = [(row, benchmark, group) for row in summaries if not row["method"].startswith("aria_")
              for benchmark, group in row["groups_by_benchmark"].items() if group["success_rate"] == 0]
    if zeroes:
        lines += ["## Zero-score Audit", "",
                  "Each remaining zero SR has zero confirmed passing outcomes in its fixed pool. "
                  "The counts below distinguish solver failures from unresolved execution/evaluation. "
                  "For completed Magentic-One empty patches, native-message evidence distinguishes code never dispatched "
                  "to ComputerTerminal from terminal responses that still delivered no repository patch. "
                  "These categories do not change correctness or add executions. "
                  "HIR = 0 for no intervention is expected; other zero HIR values mean no confirmed harm under the stated lower-bound convention.", "",
                  "| Backbone | Method | Benchmark | n | Recorded outcomes |",
                  "| --- | --- | --- | ---: | --- |"]
        for row, benchmark, group in zeroes:
            audit = row.get("outcome_audit_by_benchmark", {}).get(benchmark) or {
                "benchmark_incorrect": group["task_incorrect_count"],
                "method_invalid": group["method_invalid_count"],
                "evaluator_unresolved": group["invalid_evaluation_count"],
            }
            detail = "; ".join(f"{key.replace('_', ' ')}: {count}" for key, count in sorted(audit.items()) if count)
            lines.append(f"| {row['model_identity']} | {LABELS[row['method']]} | {benchmark} "
                         f"| {group['sample_count']} | {detail} |")
        lines.append("")
    if fresh_native:
        lines += ["## Runtime Corrections", "",
                  "- Workspace rewriting preserves filenames and occurs before the character-budget check.",
                  "- GAIA final answers take precedence over historical quotes and tool arguments.",
                  "- Native host output limits honor YAML; Magentic-One reserves response space using actual server token counts.",
                  "- Magentic-One clears the consecutive-stall counter before enacting a new plan, while preserving the total turn limit.",
                  "- Complete schema-valid JSON is recoverable even with a length finish reason; incomplete structured output gets at most one same-budget format retry.",
                  "- Llama uses the compatible tool-history template; valid declared tool JSON is normalized into native calls.",
                  "- MetaGPT retrieves task-named Python definitions rather than only the beginning of large files.",
                  "- Empty SWE submissions are distinguished from completed Docker tests; executor scratch files are not repository patches.", ""]
    adapter_rejections = [row for row in summaries if row.get("controller_adapter_rejections", {}).get("count")]
    if adapter_rejections:
        lines += ["## Historical Controller Adapter Limitations", "",
                  "The following stored runtime decisions were rejected by the old replacement-events validator before execution. "
                  "They are included in the recorded method-invalid and HIR counters, but are not observed successful-control regressions. "
                  "The current adapter supports these controller actions; changing the adapter does not retroactively execute old records. "
                  "These rows require execution-semantics reconciliation before a clean runtime safety comparison.", "",
                  "| Backbone | Method | Rejected rows | Actions | Benchmarks |",
                  "| --- | --- | ---: | --- | --- |"]
        for row in adapter_rejections:
            audit = row["controller_adapter_rejections"]
            actions = "; ".join(f"{key}: {value}" for key, value in audit["by_action"].items())
            benchmarks = "; ".join(f"{key}: {value}" for key, value in audit["by_benchmark"].items())
            lines.append(f"| {row['model_identity']} | {LABELS[row['method']]} | {audit['count']} | {actions} | {benchmarks} |")
        lines.append("")
    reevaluated = [row for row in summaries if row["correction"].get("reevaluated_rows")]
    if reevaluated:
        lines += ["## Frozen Submission Reevaluation", "",
                  "Previously unresolved official SWE evaluations were rerun on the exact saved predictions in a declared evaluator environment. "
                  "No workspace patch was recollected, no method-model call was added, and EB is unchanged. "
                  "Source hashes, frozen submission equality, and the official instance report are verified before publication.", "",
                  "| Backbone | Method | Rows | Evaluations | Extra MAS executions |",
                  "| --- | --- | ---: | ---: | ---: |"]
        for row in reevaluated:
            audit = row["correction"]
            lines.append(f"| {row['model_identity']} | {LABELS[row['method']]} | {audit['reevaluated_rows']} | {audit['recovered_evaluations']} | 0 |")
        lines.append("")
    capability_rows = [row for row in summaries if row["correction"].get("capability_corrections")]
    if capability_rows:
        lines += ["## Native Capability Corrections", "",
                  "Observed WebSurfer endpoint faults admitted these corrections, not low scores. "
                  "Text-only capability is declared explicitly and task dependency installs are isolated. "
                  "Saved method decisions and model calls are unchanged; affected runtime cohorts share corrected controls. "
                  "The corrected verdict replaces the old verdict even when it is worse, and EB retains both executions.", "",
                  "| Backbone | Method | Corrected rows | Extra recorded executions |",
                  "| --- | --- | ---: | ---: |"]
        for row in capability_rows:
            audit = row["correction"]
            lines.append(f"| {row['model_identity']} | {LABELS[row['method']]} | {audit['capability_corrections']} "
                         f"| {audit['capability_extra_native_executions']:g} |")
        lines.append("")
    invocation_rows = [row for row in summaries if row["correction"].get("invocation_corrections")]
    if invocation_rows:
        lines += ["## Failed Invocation Recovery", "",
                  "These native failures are bound to their original saved requests and parent-log witnesses. "
                  "Each recovery keeps the original decision and model-call records, uses declared isolated workspaces "
                  "and command-rejection feedback, and retains both native attempts in EB. No absent benchmark verdict is invented.", "",
                  "| Backbone | Method | Recovered rows | Extra native executions |",
                  "| --- | --- | ---: | ---: |"]
        for row in invocation_rows:
            audit = row["correction"]
            lines.append(f"| {row['model_identity']} | {LABELS[row['method']]} | {audit['invocation_corrections']} | {audit['invocation_extra_native_executions']:g} |")
        lines.append("")
    budget_rows = [row for row in summaries if row["correction"].get("context_budget_corrections")]
    if budget_rows:
        lines += ["## Context Budget Recovery", "",
                  "Only recorded infrastructure context overflows admitted these corrections. Original decisions and method-model calls "
                  "are unchanged; token counting preserves the output allowance, and corrected controls use the same declared host budgets. "
                  "Every recovered native role request and official verdict is verified. Both failed and corrected executions remain in EB.", "",
                  "| Backbone | Method | Corrected rows | Extra native executions |",
                  "| --- | --- | ---: | ---: |"]
        for row in budget_rows:
            audit = row["correction"]
            lines.append(f"| {row['model_identity']} | {LABELS[row['method']]} | {audit['context_budget_corrections']} | {audit['context_extra_native_executions']:g} |")
        lines.append("")
    preflight_rows = [row for row in summaries if row["correction"].get("preflight_recoveries")]
    if preflight_rows:
        lines += ["## Cache Preflight Recovery", "",
                  "These rows failed before native execution because shared Git metadata was corrupted. "
                  "Original saved decisions and model calls are preserved; each is executed once with private Git metadata "
                  "and a newly matched control in the declared evaluator environment. The failed preflight remains "
                  "in the audit with execution_count = 0, not a fabricated full MAS charge or benchmark verdict.", "",
                  "| Backbone | Method | Recovered rows | New recorded executions |",
                  "| --- | --- | ---: | ---: |"]
        for row in preflight_rows:
            audit = row["correction"]
            lines.append(f"| {row['model_identity']} | {LABELS[row['method']]} | {audit['preflight_recoveries']} | {audit['preflight_native_executions']:g} |")
        lines.append("")
    recovered = [row for row in summaries if row["correction"].get("recovered_executions")]
    if recovered:
        lines += ["## Infrastructure Recovery", "",
                  "Only previously infrastructure-invalid executions are eligible. Each recovery retains the same decision, "
                  "method-model calls, and matched control, and keeps both executions in the audit trail and EB. "
                  "A completed incorrect result is not retried for a higher score.", "",
                  "| Backbone | Method | Recovered rows | Extra native executions |",
                  "| --- | --- | ---: | ---: |"]
        for row in recovered:
            audit = row["correction"]
            lines.append(f"| {row['model_identity']} | {LABELS[row['method']]} | {audit['recovered_executions']} "
                         f"| {audit['recovery_extra_native_executions']:g} |")
        lines.append("")
    lines += ["## Correction Audit", "",
              "The final projection makes no model calls. Existing evaluator outputs are rescored regardless of their previous verdict; "
              "all samples, invalids, and execution attempts remain in the JSONL audit trail. "
              "Missing official reports remain unresolved, not task-incorrect. Empty patches are not completed Docker tests.", "",
              "An initially unresolved no-intervention reference may be recovered from its subsequently resolved shared control, "
              "only after exact task, MAS, host, and execution-option identity checks. This never turns a failed gate or repair into an abstention.", "",
              "| Backbone | Method | Rescored in this projection | Newly passing | Previously passing now not passing |",
              "| --- | --- | ---: | ---: | ---: |"]
    for row in summaries:
        audit = row["correction"]
        lines.append(f"| {row['model_identity']} | {LABELS[row['method']]} | {audit.get('rescored_rows', 0)} "
                     f"| {audit.get('new_passes', 0)} | {audit.get('lost_passes', 0)} |")
    lines += ["", "## Reproduction", "",
              f"Config: `{config_name}`.", "",
              f"Run `uv run python main.py -c {config_name}`, then "
              f"`uv run python scripts/aria/summarize_rq1_baselines.py --corrected-config {config_name} --output outputs/baselines/RQ1.md`.", "",
              "Corrected result directories are declared in the projection config. "
              "Each manifest records source and result hashes; report generation verifies every aggregate against its rows. "
              "The original v4/v6 runs remain unchanged. The pre-correction report is preserved in "
              "[RQ1-pre-correction-20260910.md](RQ1-pre-correction-20260910.md).", ""]
    return "\n".join(lines)
