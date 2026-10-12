"""Benchmark evaluation helpers for ARIA experiment outputs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import string
import subprocess
import sys
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any

from common.artifacts import atomic_write_json, atomic_write_jsonl
from common.benchmark_registry import get_benchmark_registration
from common.failure_classification import classify_infrastructure_failure


def evaluate_sample(
    *,
    benchmark_slug: str,
    sample: dict[str, Any],
    result: dict[str, Any],
    run_root: Path,
    judge_client: Any | None = None,
    judge_enabled: bool = False,
    evaluation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if result.get("status") != "completed":
        return {
            "benchmark": benchmark_slug,
            "score": None,
            "correct": None,
            "method": "status",
            "reason": f"run status is {result.get('status')}",
        }
    registration = get_benchmark_registration(benchmark_slug)
    try:
        evaluator = EVALUATOR_HANDLERS[registration.evaluator_name]
    except KeyError as exc:
        raise NotImplementedError(
            f"benchmark evaluator is not implemented for {benchmark_slug}"
        ) from exc
    return evaluator(
        sample,
        result,
        run_root,
        judge_client=judge_client,
        judge_enabled=judge_enabled,
        evaluation_config=evaluation_config or {},
    )


def evaluate_mmlu(sample: dict[str, Any], result: dict[str, Any], run_root: Path) -> dict[str, Any]:
    reference = str(sample.get("reference_answer", "")).strip().upper()
    text = collect_output_text(result, run_root)
    predicted = extract_choice_answer(text)
    correct = predicted == reference
    artifacts = native_artifact_paths(result, run_root)
    return {
        "benchmark": "mmlu",
        "score": 1.0 if correct else 0.0,
        "correct": correct,
        "method": "exact_choice_match",
        "predicted_answer": predicted,
        "reference_answer": reference,
        "answer_extracted": predicted is not None,
        "trajectory_path": artifacts["trace"],
        "messages_path": artifacts["messages"],
    }


def evaluate_humaneval(sample: dict[str, Any], result: dict[str, Any], run_root: Path) -> dict[str, Any]:
    workspace = Path(str(result.get("workspace_path") or run_root / "workspace"))
    candidate = load_humaneval_candidate(workspace, result, run_root)
    artifacts = native_artifact_paths(result, run_root)
    payload: dict[str, Any] = {
        "benchmark": "humaneval",
        "method": "unit_test",
        "correct": False,
        "score": 0.0,
        "candidate_path": candidate.get("path"),
        "trajectory_path": artifacts["trace"],
        "messages_path": artifacts["messages"],
        "workspace_path": str(workspace),
    }
    code = candidate.get("code")
    if not code:
        payload["reason"] = "no candidate code found"
        return payload

    test_code = build_humaneval_test_file(sample, str(code))
    eval_path = run_root / "humaneval_check.py"
    eval_path.write_text(test_code)
    try:
        completed = subprocess.run(
            [sys.executable, str(eval_path)],
            cwd=run_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired as exc:
        payload.update(
            {
                "reason": "unit test timed out",
                "test_path": str(eval_path),
                "stderr": str(exc),
            }
        )
        return payload

    correct = completed.returncode == 0
    payload.update(
        {
            "score": 1.0 if correct else 0.0,
            "correct": correct,
            "reason": "passed" if correct else "unit test failed",
            "test_path": str(eval_path),
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
            "returncode": completed.returncode,
        }
    )
    return payload


def evaluate_gsm_plus(sample: dict[str, Any], result: dict[str, Any], run_root: Path) -> dict[str, Any]:
    reference = str(sample.get("reference_answer", "")).strip()
    text = collect_output_text(result, run_root)
    predicted = extract_final_answer(text)
    correct = answers_equal(predicted, reference)
    artifacts = native_artifact_paths(result, run_root)
    return {
        "benchmark": "gsm-plus",
        "score": 1.0 if correct else 0.0,
        "correct": correct,
        "method": "exact_answer_match",
        "predicted_answer": predicted,
        "reference_answer": reference,
        "answer_extracted": predicted is not None,
        "trajectory_path": artifacts["trace"],
        "messages_path": artifacts["messages"],
    }


def evaluate_gaia(sample: dict[str, Any], result: dict[str, Any], run_root: Path) -> dict[str, Any]:
    reference = str(sample.get("reference_answer", "")).strip()
    # The delivered answer outranks quoted source traces and tool arguments.
    # Concatenating those after it can replace the verdict with an old answer.
    text = str(result.get("final_output") or result.get("final_answer") or "").strip()
    if not text:
        text = collect_output_text(result, run_root)
    predicted = extract_final_answer(text)
    if predicted is None and (result.get("final_output") or result.get("final_answer")):
        predicted = cleanup_answer(text)
    artifacts = native_artifact_paths(result, run_root)
    if not reference:
        return {
            "benchmark": "gaia",
            "score": None,
            "correct": None,
            "method": "missing_reference",
            "predicted_answer": predicted,
            "reference_answer": "",
            "answer_extracted": predicted is not None,
            "trajectory_path": artifacts["trace"],
            "messages_path": artifacts["messages"],
        }
    correct = gaia_question_scorer(predicted, reference)
    answer_type = gaia_answer_type(reference)
    return {
        "benchmark": "gaia",
        "score": 1.0 if correct else 0.0,
        "correct": correct,
        "method": "gaia_official_quasi_exact_match",
        "answer_type": answer_type,
        "predicted_answer": predicted,
        "reference_answer": reference,
        "normalized_predicted_answer": normalize_gaia_answer(predicted, answer_type),
        "normalized_reference_answer": normalize_gaia_answer(reference, answer_type),
        "answer_extracted": predicted is not None,
        "trajectory_path": artifacts["trace"],
        "messages_path": artifacts["messages"],
    }


def evaluate_swe_bench_lite(
    sample: dict[str, Any],
    result: dict[str, Any],
    run_root: Path,
    *,
    evaluation_config: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    evaluation_config = evaluation_config or {}
    # The harness changes its working directory to ``run_root``. Resolve the
    # artifact paths first so relative callers cannot make the harness look for
    # predictions relative to itself a second time.
    run_root = Path(run_root).resolve()
    workspace = Path(str(result.get("workspace_path") or run_root / "workspace")).resolve()
    instance_id = str(sample.get("instance_id") or sample.get("sample_id") or "")
    model_name = f"{result.get('model') or 'model'}-{result.get('reasoning_effort') or 'default'}"
    artifacts = native_artifact_paths(result, run_root)
    patch_path = run_root / "model.patch"
    patch_record = collect_swe_bench_patch(workspace, str(sample.get("base_commit") or ""))
    patch = patch_record["patch"]
    patch_path.write_text(patch, encoding="utf-8")

    prediction = {
        "instance_id": instance_id,
        "model_name_or_path": model_name,
        "model_patch": patch,
    }
    predictions_path = run_root / "swebench_predictions.json"
    atomic_write_json(predictions_path, [prediction])
    dataset_path = run_root / "swebench_instance.json"
    dataset_record = swe_bench_dataset_record(sample)
    atomic_write_json(dataset_path, [dataset_record])

    payload: dict[str, Any] = {
        "benchmark": "swe-bench-lite",
        "method": "swebench_official_docker",
        "score": 0.0,
        "correct": False,
        "harness_status": "pending",
        "instance_id": instance_id,
        "model_name_or_path": model_name,
        "workspace_path": str(workspace),
        "base_commit": sample.get("base_commit"),
        "model_patch_path": str(patch_path),
        "model_patch_bytes": len(patch.encode("utf-8")),
        "changed_files": patch_record["changed_files"],
        "git_commands": patch_record["commands"],
        "predictions_path": str(predictions_path),
        "dataset_path": str(dataset_path),
        "trajectory_path": artifacts["trace"],
        "messages_path": artifacts["messages"],
    }
    if not patch.strip():
        payload.update(
            {
                "harness_status": "empty_patch",
                "reason": "native MAS produced no git diff against base_commit",
            }
        )
        return payload

    return _run_swe_bench_harness(dataset_record, prediction, run_root, payload, evaluation_config)


def evaluate_saved_swe_bench_submission(
    dataset_record: dict[str, Any], prediction: dict[str, Any], run_root: Path,
    *, evaluation_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Grade the exact stored submission without reading or changing a workspace."""
    instance_id = prediction.get("instance_id")
    if not isinstance(instance_id, str) or instance_id != dataset_record.get("instance_id"):
        raise ValueError("saved SWE submission and dataset instance differ")
    patch = prediction.get("model_patch")
    if not isinstance(patch, str) or not isinstance(prediction.get("model_name_or_path"), str):
        raise ValueError("saved SWE submission requires a patch and model identity")
    run_root = Path(run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    patch_path = run_root / "model.patch"
    patch_path.write_bytes(patch.encode("utf-8"))
    predictions_path = run_root / "swebench_predictions.json"
    dataset_path = run_root / "swebench_instance.json"
    atomic_write_json(predictions_path, [prediction])
    atomic_write_json(dataset_path, [dataset_record])
    payload = {
        "benchmark": "swe-bench-lite", "method": "swebench_official_docker", "instance_id": instance_id,
        "model_name_or_path": prediction["model_name_or_path"], "base_commit": dataset_record.get("base_commit"),
        "model_patch_path": str(patch_path), "model_patch_bytes": len(patch.encode("utf-8")),
        "predictions_path": str(predictions_path), "dataset_path": str(dataset_path),
        "correct": None, "score": None, "harness_status": "pending",
    }
    if not patch.strip():
        payload.update(correct=False, score=0.0, harness_status="empty_patch", reason="saved submission contains no repository patch")
        return payload
    return _run_swe_bench_harness(dataset_record, prediction, run_root, payload, evaluation_config or {})


def _run_swe_bench_harness(dataset_record, prediction, run_root, payload, evaluation_config):
    instance_id = prediction["instance_id"]
    predictions_path = Path(payload["predictions_path"])
    dataset_path = Path(payload["dataset_path"])
    harness_dir = run_root / "swebench_harness"
    harness_dir.mkdir(parents=True, exist_ok=True)
    run_id = _swe_bench_run_id(run_root, prediction, dataset_record)
    payload["harness_run_id"] = run_id
    timeout_seconds = int(evaluation_config.get("timeout_seconds", 1800))
    process_timeout_seconds = int(
        evaluation_config.get("process_timeout_seconds", timeout_seconds + 1800)
    )
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        str(dataset_path),
        "--split",
        "test",
        "--instance_ids",
        instance_id,
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(int(evaluation_config.get("max_workers", 1))),
        "--timeout",
        str(timeout_seconds),
        "--force_rebuild",
        "false",
        "--cache_level",
        str(evaluation_config.get("cache_level", "env")),
        "--clean",
        str(bool(evaluation_config.get("clean", False))).lower(),
        "--run_id",
        run_id,
        "--namespace",
        str(evaluation_config.get("namespace", "swebench")),
        "--report_dir",
        str(harness_dir),
    ]
    command_path = harness_dir / "command.json"
    atomic_write_json(
        command_path,
        {
            "command": command,
            "cwd": str(run_root),
            "timeout_seconds": process_timeout_seconds,
            "docker_platform": evaluation_config.get("docker_platform"),
        },
    )
    environment = dict(os.environ)
    docker_platform = evaluation_config.get("docker_platform")
    if docker_platform:
        environment["DOCKER_DEFAULT_PLATFORM"] = str(docker_platform)
    use_host_docker_credentials = bool(
        evaluation_config.get("use_host_docker_credentials", True)
    )
    if not use_host_docker_credentials:
        docker_config_dir = harness_dir / "docker-config"
        docker_config_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(docker_config_dir / "config.json", {})
        environment["DOCKER_CONFIG"] = str(docker_config_dir)
        payload["docker_config_path"] = str(docker_config_dir / "config.json")

    if docker_platform:
        image_pull = pull_swe_bench_instance_image(
            dataset_record,
            namespace=str(evaluation_config.get("namespace", "swebench")),
            platform=str(docker_platform),
            harness_dir=harness_dir,
            environment=environment,
            timeout_seconds=process_timeout_seconds,
        )
        payload["instance_image_pull"] = image_pull
        if image_pull["returncode"] != 0:
            payload.update(
                {
                    "score": None,
                    "correct": None,
                    "harness_status": "harness_error",
                    "reason": "failed to pull the SWE-Bench instance image for the configured platform",
                }
            )
            return payload

    stdout_path = harness_dir / "stdout.log"
    stderr_path = harness_dir / "stderr.log"
    try:
        completed = subprocess.run(
            command,
            cwd=run_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=process_timeout_seconds,
            env=environment,
        )
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        returncode: int | None = completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(_timeout_output(exc.stdout), encoding="utf-8")
        stderr_path.write_text(_timeout_output(exc.stderr), encoding="utf-8")
        payload.update(
            {
                "harness_status": "timeout",
                "score": None,
                "correct": None,
                "reason": f"SWE-Bench harness exceeded {process_timeout_seconds} seconds",
                "harness_command_path": str(command_path),
                "harness_stdout_path": str(stdout_path),
                "harness_stderr_path": str(stderr_path),
            }
        )
        return payload

    report_path, report = load_swe_bench_report(run_root, harness_dir, run_id, instance_id)
    resolved = report.get("resolved") if report is not None else None
    authoritative = returncode == 0 and isinstance(resolved, bool)
    if report is not None:
        atomic_write_json(harness_dir / "instance_report.json", report)
    payload.update(
        {
            "score": float(resolved) if authoritative else None,
            "correct": resolved if authoritative else None,
            "harness_status": (
                "completed" if authoritative else "harness_error"
            ),
            "reason": (
                f"official SWE-Bench harness exited with code {returncode}"
                if returncode != 0 else
                "resolved" if resolved is True else "official SWE-Bench tests did not resolve the instance"
            ),
            "harness_returncode": returncode,
            "harness_command_path": str(command_path),
            "harness_stdout_path": str(stdout_path),
            "harness_stderr_path": str(stderr_path),
            "harness_report_path": str(report_path) if report_path is not None else None,
            "harness_report": report,
        }
    )
    if report is None:
        payload.update(
            {
                "score": None,
                "correct": None,
                "reason": "official SWE-Bench harness did not produce an instance report",
            }
        )
    return payload


def pull_swe_bench_instance_image(
    instance: dict[str, Any],
    *,
    namespace: str,
    platform: str,
    harness_dir: Path,
    environment: dict[str, str],
    timeout_seconds: int,
) -> dict[str, Any]:
    """Pre-pull a remote SWE-Bench image with an explicit Docker platform."""

    from swebench.harness.test_spec.test_spec import make_test_spec

    normalized = platform.strip().lower()
    arch = "x86_64" if normalized in {"linux/amd64", "linux/x86_64"} else "arm64"
    image = make_test_spec(instance, namespace=namespace, arch=arch).instance_image_key
    command = ["docker", "pull", "--platform", platform, image]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=environment,
    )
    stdout_path = harness_dir / "instance-image-pull.stdout.log"
    stderr_path = harness_dir / "instance-image-pull.stderr.log"
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    return {
        "command": command,
        "image": image,
        "platform": platform,
        "returncode": completed.returncode,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def evaluate_olympiadbench(
    sample: dict[str, Any],
    result: dict[str, Any],
    run_root: Path,
) -> dict[str, Any]:
    reference = str(sample.get("reference_answer", "")).strip()
    text = collect_output_text(result, run_root)
    predicted = extract_final_answer(text)
    artifacts = native_artifact_paths(result, run_root)
    if not reference:
        return {
            "benchmark": "olympiadbench",
            "score": None,
            "correct": None,
            "method": "theorem_proof_manual_review",
            "reason": "the official dataset does not provide final answers for theorem-proof problems",
            "predicted_answer": predicted,
            "reference_answer": "",
            "answer_extracted": predicted is not None,
            "trajectory_path": artifacts["trace"],
            "messages_path": artifacts["messages"],
        }

    from common.olympiadbench_scoring import OlympiadBenchScorer

    precision = olympiadbench_precision(sample.get("error"))
    correct = OlympiadBenchScorer().judge(reference, predicted or "", precision)
    return {
        "benchmark": "olympiadbench",
        "score": 1.0 if correct else 0.0,
        "correct": correct,
        "method": "olympiadbench_official_auto_scoring",
        "predicted_answer": predicted,
        "reference_answer": reference,
        "answer_type": sample.get("answer_type"),
        "precision": precision,
        "answer_extracted": predicted is not None,
        "trajectory_path": artifacts["trace"],
        "messages_path": artifacts["messages"],
    }


def olympiadbench_precision(value: Any) -> float | list[float]:
    if value in (None, ""):
        return 1e-8
    if isinstance(value, list):
        return [float(item) if item not in (None, "") else 1e-8 for item in value]
    text = str(value)
    if "," in text:
        return [float(item) if item.strip() else 1e-8 for item in text.split(",")]
    return float(text)


def evaluate_programdev(
    sample: dict[str, Any],
    result: dict[str, Any],
    run_root: Path,
    *,
    judge_client: Any | None,
    judge_enabled: bool,
) -> dict[str, Any]:
    workspace = Path(str(result.get("workspace_path") or run_root / "workspace"))
    static = inspect_programdev_workspace(workspace)
    artifacts = native_artifact_paths(result, run_root)
    payload: dict[str, Any] = {
        "benchmark": "programdev",
        "method": "workspace_static_checks",
        "static_checks": static,
        "trajectory_path": artifacts["trace"],
        "messages_path": artifacts["messages"],
        "workspace_path": str(workspace),
    }
    if judge_enabled and judge_client is not None:
        judge = judge_programdev(sample, result, run_root, workspace, static, judge_client)
        payload["method"] = "gpt_judge_with_static_checks"
        payload["judge"] = judge
        payload["score"] = 1.0 if judge.get("correct") else 0.0
        payload["correct"] = bool(judge.get("correct"))
    else:
        correct = bool(static["has_python_file"] and static["syntax_ok"])
        payload["score"] = 1.0 if correct else 0.0
        payload["correct"] = correct
    return payload


def collect_output_text(result: dict[str, Any], run_root: Path) -> str:
    parts: list[str] = []
    for key in ("final_output", "final_answer"):
        if result.get(key):
            parts.append(str(result[key]))
    for item in result.get("transcript", []):
        if isinstance(item, dict):
            for key in ("content", "result"):
                if item.get(key):
                    parts.append(str(item[key]))
            for call in item.get("tool_calls", []) or []:
                parts.append(json.dumps(call, ensure_ascii=False))

    for messages_path in message_dump_paths(result, run_root):
        if not messages_path.exists():
            continue
        try:
            messages = json.loads(messages_path.read_text())
        except json.JSONDecodeError:
            messages = []
        for message in messages:
            if isinstance(message, dict):
                for key in ("content", "text"):
                    if message.get(key):
                        value = message[key]
                        if isinstance(value, (dict, list)):
                            parts.append(json.dumps(value, ensure_ascii=False))
                        else:
                            parts.append(str(value))
                if message.get("tool_calls"):
                    parts.append(json.dumps(message["tool_calls"], ensure_ascii=False))
    return "\n".join(parts)


def native_artifact_paths(result: dict[str, Any], run_root: Path) -> dict[str, str]:
    return {
        "trace": str(result.get("native_trace_path") or run_root / "trajectory.jsonl"),
        "messages": str(result.get("native_messages_path") or run_root / "messages.json"),
    }


def message_dump_paths(result: dict[str, Any], run_root: Path) -> list[Path]:
    candidates = [
        Path(str(result["native_messages_path"])) if result.get("native_messages_path") else None,
        run_root / "messages.json",
    ]
    paths: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        path = candidate.resolve()
        if path not in seen:
            paths.append(path)
            seen.add(path)
    return paths


def first_existing_path(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def extract_final_answer(text: str) -> str | None:
    patterns = (
        r"FINAL_ANSWER\s*[:=]\s*([^\n\r]+)",
        r"final answer\s*(?:is|:)\s*([^\n\r]+)",
        r"answer\s*(?:is|:)\s*([^\n\r]+)",
        r"####\s*([^\n\r]+)",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            value = cleanup_answer(matches[-1])
            if value:
                return value

    number_matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+(?:,\d{3})*(?:\.\d+)?)?", text)
    if number_matches:
        return cleanup_answer(number_matches[-1])
    if re.search(r"\bnone\b", text, flags=re.IGNORECASE):
        return "None"
    return None


def extract_choice_answer(text: str) -> str | None:
    patterns = (
        r"FINAL_ANSWER\s*[:=]\s*([A-D])\b",
        r"final answer\s*(?:is|:)\s*\(?([A-D])\)?\b",
        r"answer\s*(?:is|:)\s*\(?([A-D])\)?\b",
    )
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return str(matches[-1]).upper()
    lines = [line.strip().strip("()") for line in text.splitlines() if line.strip()]
    if lines and lines[-1].upper() in {"A", "B", "C", "D"}:
        return lines[-1].upper()
    stripped = text.strip().upper()
    if stripped in {"A", "B", "C", "D"}:
        return stripped
    return None


def cleanup_answer(value: str) -> str:
    value = value.strip().strip("` ")
    value = re.split(r"[\n\r]", value)[0]
    value = re.sub(r"^(?:\$|answer\s*[:=]\s*)", "", value, flags=re.IGNORECASE).strip()
    value = value.rstrip(".。,:;")
    return value.strip()


def answers_equal(predicted: str | None, reference: str) -> bool:
    if predicted is None:
        return False
    pred_no_answer = is_no_answer(predicted)
    ref_no_answer = is_no_answer(reference)
    if pred_no_answer or ref_no_answer:
        return pred_no_answer and ref_no_answer
    pred_number = parse_number(predicted)
    ref_number = parse_number(reference)
    if pred_number is not None and ref_number is not None:
        return pred_number == ref_number
    return normalize_answer_text(predicted) == normalize_answer_text(reference)


def parse_number(value: str) -> Fraction | None:
    text = normalize_answer_text(value)
    text = text.replace(",", "")
    if not text or text in {"none", "n/a", "null"}:
        return None
    try:
        if "/" in text:
            return Fraction(text)
        return Fraction(Decimal(text))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None


def normalize_answer_text(value: str) -> str:
    text = str(value).strip().lower()
    text = text.replace("$", "").replace("%", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def gaia_answer_type(reference: str) -> str:
    try:
        float(reference)
        return "number"
    except ValueError:
        if any(char in reference for char in (",", ";")):
            return "list"
        return "string"


def normalize_gaia_number(value: str) -> float | None:
    normalized = str(value)
    for char in ("$", "%", ","):
        normalized = normalized.replace(char, "")
    try:
        return float(normalized)
    except ValueError:
        return None


def normalize_gaia_string(value: str, *, remove_punctuation: bool = True) -> str:
    normalized = re.sub(r"\s", "", str(value)).lower()
    if remove_punctuation:
        normalized = normalized.translate(str.maketrans("", "", string.punctuation))
    return normalized


def split_gaia_list(value: str) -> list[str]:
    return re.split(r"[,;]", str(value))


def gaia_question_scorer(predicted: str | None, reference: str) -> bool:
    """Match the public GAIA leaderboard scorer semantics."""

    model_answer = "None" if predicted is None else str(predicted)
    answer_type = gaia_answer_type(reference)
    if answer_type == "number":
        return normalize_gaia_number(model_answer) == float(reference)
    if answer_type == "list":
        reference_items = split_gaia_list(reference)
        predicted_items = split_gaia_list(model_answer)
        if len(reference_items) != len(predicted_items):
            return False
        comparisons = []
        for predicted_item, reference_item in zip(predicted_items, reference_items):
            try:
                reference_number = float(reference_item)
            except ValueError:
                comparisons.append(
                    normalize_gaia_string(predicted_item, remove_punctuation=False)
                    == normalize_gaia_string(reference_item, remove_punctuation=False)
                )
            else:
                comparisons.append(normalize_gaia_number(predicted_item) == reference_number)
        return all(comparisons)
    return normalize_gaia_string(model_answer) == normalize_gaia_string(reference)


def normalize_gaia_answer(value: str | None, answer_type: str) -> Any:
    text = "None" if value is None else str(value)
    if answer_type == "number":
        return normalize_gaia_number(text)
    if answer_type == "list":
        return [
            normalize_gaia_number(item)
            if normalize_gaia_number(item) is not None
            else normalize_gaia_string(item, remove_punctuation=False)
            for item in split_gaia_list(text)
        ]
    return normalize_gaia_string(text)


def is_no_answer(value: str | None) -> bool:
    if value is None:
        return False
    text = normalize_answer_text(value)
    text = text.rstrip(".。,:;!")
    text = re.sub(r"^(?:final_answer\s*[:=]\s*|answer\s*[:=]\s*)", "", text).strip()
    text = text.rstrip(".。,:;!")
    return text in {
        "none",
        "n/a",
        "null",
        "unknown",
        "cannot be determined",
        "can't be determined",
        "can not be determined",
        "not enough information",
        "insufficient information",
        "undetermined",
    }


def collect_swe_bench_patch(workspace: Path, base_commit: str) -> dict[str, Any]:
    commands: list[dict[str, Any]] = []

    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        commands.append(
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-4000:],
                "stderr_tail": completed.stderr[-4000:],
            }
        )
        return completed

    if not (workspace / ".git").exists():
        return {
            "patch": "",
            "changed_files": [],
            "commands": [
                {
                    "command": [],
                    "returncode": None,
                    "stderr_tail": f"workspace is not a git checkout: {workspace}",
                }
            ],
        }
    revision = base_commit or "HEAD"
    tree = run(["git", "-C", str(workspace), "ls-tree", "--name-only", "-z", revision])
    if tree.returncode != 0:
        raise RuntimeError(f"cannot inspect SWE-Bench base revision: {tree.stderr[-1000:]}")
    tracked = set(tree.stdout.split("\0"))
    # AutoGen stores executor scratch scripts in the working directory. Those
    # are tool plumbing, not a submitted repository fix. Preserve base-tracked
    # files even when their names happen to match the executor convention.
    scratch = sorted(path.name for path in workspace.glob("tmp_code_*")
                     if path.is_file() and path.name not in tracked
                     and re.fullmatch(r"tmp_code_(?:[0-9a-f]{32}|[0-9a-f]{64})\.(?:py|sh|bash|ps1|js)", path.name))
    paths = [".", *[f":(exclude,literal){name}" for name in scratch]]
    run(["git", "-C", str(workspace), "add", "-N", "--all", "--", *paths])
    diff = run(
        [
            "git",
            "-C",
            str(workspace),
            "diff",
            "--binary",
            "--no-ext-diff",
            revision,
            "--",
            *paths,
        ]
    )
    names = run(
        [
            "git",
            "-C",
            str(workspace),
            "diff",
            "--name-only",
            revision,
            "--",
            *paths,
        ]
    )
    return {
        "patch": diff.stdout if diff.returncode == 0 else "",
        "changed_files": [line for line in names.stdout.splitlines() if line.strip()],
        "commands": commands,
    }


def swe_bench_dataset_record(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "repo": sample.get("repo"),
        "instance_id": sample.get("instance_id"),
        "base_commit": sample.get("base_commit"),
        "patch": sample.get("reference_answer") or "",
        "test_patch": sample.get("test_patch") or "",
        "problem_statement": sample.get("question") or sample.get("task") or "",
        "hints_text": sample.get("hints_text") or "",
        "created_at": sample.get("created_at"),
        "version": sample.get("version"),
        "FAIL_TO_PASS": sample.get("fail_to_pass") or "[]",
        "PASS_TO_PASS": sample.get("pass_to_pass") or "[]",
        "environment_setup_commit": sample.get("environment_setup_commit"),
    }


def _swe_bench_run_id(run_root: Path, prediction: dict[str, Any], dataset_record: dict[str, Any]) -> str:
    identity = json.dumps(
        {"run_root": str(run_root.resolve()), "prediction": prediction, "dataset": dataset_record},
        sort_keys=True, ensure_ascii=True,
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    name = f"masfm-{prediction['instance_id']}-{digest}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip("-")


def load_swe_bench_report(
    run_root: Path,
    harness_dir: Path,
    run_id: str,
    instance_id: str,
) -> tuple[Path | None, dict[str, Any] | None]:
    candidates = sorted(
        (run_root / "logs" / "run_evaluation" / run_id).rglob("report.json")
    )
    # harness_dir also holds copied reports from earlier attempts. Only the
    # current run's instance-scoped logs can establish a new evaluator verdict.
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        record = payload.get(instance_id)
        if (not isinstance(record, dict) and path.parent.name == instance_id
                and payload.get("instance_id", instance_id) == instance_id):
            record = payload
        if isinstance(record, dict) and isinstance(record.get("resolved"), bool):
            return path, record
    return None, None


def _timeout_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def load_humaneval_candidate(workspace: Path, result: dict[str, Any], run_root: Path) -> dict[str, str | None]:
    for path in candidate_solution_paths(workspace):
        if path.exists() and path.is_file():
            try:
                return {"path": str(path), "code": path.read_text(encoding="utf-8")}
            except UnicodeDecodeError:
                continue

    text = collect_output_text(result, run_root)
    code = extract_python_code(text)
    return {"path": None, "code": code}


def candidate_solution_paths(workspace: Path) -> list[Path]:
    paths = [
        workspace / "solution.py",
        workspace / "main.py",
        workspace / "answer.py",
    ]
    if workspace.exists():
        paths.extend(sorted(item for item in workspace.rglob("*.py") if item.is_file()))
    seen = set()
    unique = []
    for path in paths:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def extract_python_code(text: str) -> str | None:
    fenced = re.findall(r"```(?:python|py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced[-1].strip() + "\n"
    if "def " in text:
        start = text.find("def ")
        return text[start:].strip() + "\n"
    return None


def build_humaneval_test_file(sample: dict[str, Any], candidate_code: str) -> str:
    entry_point = str(sample["entry_point"])
    if f"def {entry_point}" not in candidate_code:
        candidate_code = str(sample["prompt"]).rstrip() + "\n" + candidate_code.rstrip() + "\n"
    return (
        candidate_code.rstrip()
        + "\n\n"
        + str(sample["test"]).strip()
        + "\n\n"
        + f"check({entry_point})\n"
    )


def inspect_programdev_workspace(workspace: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    syntax_errors: list[dict[str, str]] = []
    if workspace.exists():
        for path in sorted(item for item in workspace.rglob("*") if item.is_file()):
            rel = path.relative_to(workspace)
            files.append({"path": str(rel), "bytes": path.stat().st_size})
            if path.suffix == ".py":
                try:
                    compile(path.read_text(encoding="utf-8"), str(path), "exec")
                except Exception as exc:  # noqa: BLE001 - record verifier evidence.
                    syntax_errors.append({"path": str(rel), "error": str(exc)})
    py_files = [item for item in files if item["path"].endswith(".py")]
    return {
        "exists": workspace.exists(),
        "file_count": len(files),
        "python_file_count": len(py_files),
        "has_python_file": bool(py_files),
        "syntax_ok": not syntax_errors,
        "syntax_errors": syntax_errors,
        "files": files[:200],
    }


def judge_programdev(
    sample: dict[str, Any],
    result: dict[str, Any],
    run_root: Path,
    workspace: Path,
    static: dict[str, Any],
    judge_client: Any,
) -> dict[str, Any]:
    snippets = collect_workspace_snippets(workspace)
    prompt = (
        "You are evaluating a ProgramDev benchmark run. Return only JSON with keys "
        "correct, score, reason, missing_requirements, evidence.\n\n"
        f"Project name: {sample.get('project_name')}\n"
        f"Task: {sample.get('task')}\n\n"
        f"Static checks: {json.dumps(static, ensure_ascii=False)[:6000]}\n\n"
        f"Final output:\n{str(result.get('final_output', ''))[:6000]}\n\n"
        f"Workspace snippets:\n{snippets[:16000]}\n\n"
        "Judge whether the produced workspace substantially satisfies the task. "
        "Use correct=true only when there is runnable code and the main requested behavior is implemented."
    )
    content = judge_client.complete(
        [
            {"role": "system", "content": "You are a strict software benchmark judge."},
            {"role": "user", "content": prompt},
        ]
    )
    parsed = parse_json_object(content)
    if parsed is None:
        return {
            "correct": False,
            "score": 0.0,
            "reason": "judge returned unparsable JSON",
            "raw": content,
        }
    parsed.setdefault("correct", bool(parsed.get("score", 0) == 1))
    parsed.setdefault("raw", content)
    return parsed


def collect_workspace_snippets(workspace: Path) -> str:
    if not workspace.exists():
        return "[workspace missing]"
    parts: list[str] = []
    for path in sorted(item for item in workspace.rglob("*") if item.is_file())[:40]:
        rel = path.relative_to(workspace)
        if path.stat().st_size > 20000:
            continue
        if path.suffix.lower() not in {".py", ".txt", ".md", ".json", ".toml", ".yml", ".yaml"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        parts.append(f"\n--- {rel} ---\n{text[:4000]}")
    return "\n".join(parts) or "[no text snippets]"


def parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def write_batch_summary(batch_root: Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    completed = sum(1 for item in results if item.get("status") == "completed")
    evaluated = [
        item
        for item in results
        if isinstance(item.get("evaluation"), dict) and item["evaluation"].get("correct") is not None
    ]
    correct = sum(1 for item in evaluated if item["evaluation"].get("correct") is True)
    incorrect = sum(1 for item in evaluated if item["evaluation"].get("correct") is False)
    failure_categories: dict[str, int] = {}
    for item in results:
        evaluation = item.get("evaluation") if isinstance(item.get("evaluation"), dict) else {}
        unresolved = item.get("status") != "completed" or not isinstance(
            evaluation.get("correct"), bool
        )
        if not unresolved:
            continue
        category = item.get("failure_category") or classify_infrastructure_failure(
            item.get("error") or evaluation.get("reason")
        )
        category = str(category or "evaluator_unresolved")
        failure_categories[category] = failure_categories.get(category, 0) + 1
    summary = {
        "total": total,
        "completed": completed,
        "failed": total - completed,
        "evaluated": len(evaluated),
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": (correct / len(evaluated)) if evaluated else None,
        "infrastructure_invalid": total - len(evaluated),
        "failure_categories": dict(sorted(failure_categories.items())),
    }
    atomic_write_json(batch_root / "summary.json", summary)
    write_trajectory_splits(batch_root, results)
    return summary


def write_trajectory_splits(batch_root: Path, results: list[dict[str, Any]]) -> None:
    correct_rows = []
    error_rows = []
    for item in results:
        evaluation = item.get("evaluation") if isinstance(item.get("evaluation"), dict) else {}
        record = {
            "sample_id": item.get("sample_id"),
            "sample_index": item.get("sample_index"),
            "status": item.get("status"),
            "failure_category": item.get("failure_category"),
            "correct": evaluation.get("correct"),
            "score": evaluation.get("score"),
            "output_dir": item.get("output_dir"),
            "trajectory_path": evaluation.get("trajectory_path") or item.get("native_trace_path"),
            "messages_path": evaluation.get("messages_path") or item.get("native_messages_path"),
            "predicted_answer": evaluation.get("predicted_answer"),
            "reference_answer": evaluation.get("reference_answer"),
            "reason": evaluation.get("reason") or evaluation.get("judge", {}).get("reason"),
        }
        (correct_rows if evaluation.get("correct") is True else error_rows).append(record)
    atomic_write_jsonl(batch_root / "correct_trajectories.jsonl", correct_rows)
    atomic_write_jsonl(batch_root / "error_trajectories.jsonl", error_rows)


def _simple_evaluator(evaluator: Any) -> Any:
    def run(
        sample: dict[str, Any],
        result: dict[str, Any],
        run_root: Path,
        **_: Any,
    ) -> dict[str, Any]:
        return evaluator(sample, result, run_root)

    return run


def validate_benchmark_evaluator(benchmark_slug: str) -> None:
    registration = get_benchmark_registration(benchmark_slug)
    if registration.evaluator_name not in EVALUATOR_HANDLERS:
        raise NotImplementedError(
            f"benchmark evaluator is not implemented for {benchmark_slug}"
        )


from common.ood_benchmarks import evaluate_ood

EVALUATOR_HANDLERS = {
    "ood": _simple_evaluator(evaluate_ood),
    "gsm_plus": _simple_evaluator(evaluate_gsm_plus),
    "mmlu": _simple_evaluator(evaluate_mmlu),
    "humaneval": _simple_evaluator(evaluate_humaneval),
    "gaia": _simple_evaluator(evaluate_gaia),
    "swe_bench_lite": evaluate_swe_bench_lite,
    "olympiadbench": _simple_evaluator(evaluate_olympiadbench),
    "programdev": evaluate_programdev,
}
