"""Utilities for loading MAST/MAD records and samples."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from common.benchmark_registry import get_benchmark_registration

DATASET_FILES = {
    "full": "MAD_full_dataset.json",
    "human": "MAD_human_labelled_dataset.json",
}
FAILURE_MODE_CODES = {
    "1.1",
    "1.2",
    "1.3",
    "1.4",
    "1.5",
    "2.1",
    "2.2",
    "2.3",
    "2.4",
    "2.5",
    "2.6",
    "3.1",
    "3.2",
    "3.3",
    "3.4",
    "4.1",
    "4.2",
    "4.3",
}


def load_records(data_dir: Path, dataset: str) -> list[dict[str, Any]]:
    path = data_dir / "raw" / DATASET_FILES[dataset]
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")

    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        for key in ("data", "records", "examples"):
            value = payload.get(key)
            if isinstance(value, list):
                records = value
                break
        else:
            records = list(payload.values())
    else:
        raise TypeError(f"unsupported JSON payload type: {type(payload).__name__}")

    return [record for record in records if isinstance(record, dict)]


def load_mad_records(data_dir: Path) -> list[dict[str, Any]]:
    try:
        return load_records(data_dir, "full")
    except FileNotFoundError:
        return []


def load_benchmark_sample(data_dir: Path, benchmark: str) -> dict[str, Any]:
    sample_path = data_dir / "benchmarks" / benchmark / "sample.json"
    if not sample_path.exists():
        raise FileNotFoundError(f"missing benchmark sample: {sample_path}")
    sample = json.loads(sample_path.read_text())
    normalizer = SAMPLE_NORMALIZERS.get(benchmark)
    if normalizer is not None:
        sample = normalizer(sample)
    return sample


def load_benchmark_items(
    data_dir: Path,
    benchmark: str,
    *,
    source: str = "sample",
    limit: int | None = None,
    offset: int = 0,
    split: str = "validation",
    sample_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Load benchmark items in the normalized shape used by experiment batches."""

    if source == "sample":
        sample = dict(load_benchmark_sample(data_dir, benchmark))
        sample.setdefault("sample_index", 0)
        sample.setdefault("sample_id", str(sample.get("trace_id", "sample-00000")))
        sample.setdefault("task", str(sample.get("task", "")))
        return [sample]

    if source != "full":
        raise ValueError(f"unsupported benchmark source: {source}")

    registration = get_benchmark_registration(benchmark)
    try:
        loader = FULL_BENCHMARK_LOADERS[registration.loader_name]
    except KeyError as exc:
        raise NotImplementedError(
            f"full benchmark loader is not implemented for {benchmark}"
        ) from exc
    items = loader(data_dir, split)

    if sample_ids is not None:
        by_id = {str(item.get("sample_id")): item for item in items}
        missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
        if missing:
            raise ValueError(
                f"benchmark {benchmark} is missing requested sample IDs: "
                + ", ".join(missing[:10])
            )
        items = [by_id[sample_id] for sample_id in sample_ids]

    if offset:
        items = items[offset:]
    if limit is not None:
        items = items[:limit]
    return items


def load_gsm_plus_items(data_dir: Path) -> list[dict[str, Any]]:
    path = data_dir / "benchmarks" / "gsm-plus" / "data" / "test-00000-of-00001.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing GSM-Plus full test file: {path}")

    items: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            record = json.loads(line)
            items.append(
                {
                    "sample_index": index,
                    "sample_id": f"gsm-plus-test-{index:05d}",
                    "benchmark": "GSM-Plus",
                    "task": str(record["question"]),
                    "question": record["question"],
                    "reference_answer": str(record["answer"]),
                    "reference_solution": record.get("solution"),
                    "perturbation_type": record.get("perturbation_type"),
                    "seed_question": record.get("seed_question"),
                    "seed_answer": record.get("seed_answer"),
                }
            )
    return items


def load_gaia_items(data_dir: Path, split: str = "validation") -> list[dict[str, Any]]:
    path = data_dir / "benchmarks" / "gaia" / "2023" / split / "metadata.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"missing GAIA {split} metadata file: {path}. "
            "GAIA is gated on Hugging Face; accept access and cache the parquet locally first."
        )

    items: list[dict[str, Any]] = []
    for index, record in enumerate(load_parquet_records(path)):
        task = str(
            record.get("Question")
            or record.get("question")
            or record.get("task")
            or record.get("Task")
            or ""
        )
        sample_id = str(record.get("task_id") or record.get("id") or record.get("Id") or f"gaia-{split}-{index:05d}")
        reference = record.get("Final answer")
        if reference is None:
            reference = record.get("final_answer")
        if reference is None:
            reference = record.get("answer")
        file_name = record.get("file_name") or record.get("file") or record.get("File")
        file_path = None
        if file_name is not None and str(file_name).strip():
            file_path = path.parent / str(file_name)
        item = {
            "sample_index": index,
            "sample_id": safe_sample_slug(sample_id),
            "benchmark": "GAIA",
            "task": task,
            "question": task,
            "level": record.get("Level") or record.get("level"),
            "file_name": file_name,
            "file_path": str(file_path) if file_path is not None else None,
            "split": split,
            "reference_answer": "" if reference is None else str(reference),
        }
        items.append({key: value for key, value in item.items() if value not in (None, "")})
    return items


def load_programdev_items(data_dir: Path) -> list[dict[str, Any]]:
    path = data_dir / "benchmarks" / "programdev" / "programdev_dataset.json"
    if not path.exists():
        raise FileNotFoundError(f"missing ProgramDev dataset file: {path}")

    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise TypeError(f"unsupported ProgramDev payload type: {type(payload).__name__}")

    items: list[dict[str, Any]] = []
    for index, record in enumerate(payload):
        project_name = str(record.get("project_name") or f"programdev-{index:05d}")
        description = str(record.get("description", ""))
        items.append(
            {
                "sample_index": index,
                "sample_id": f"programdev-{index:03d}-{safe_sample_slug(project_name)}",
                "benchmark": "ProgramDev",
                "task": description,
                "project_name": project_name,
                "reference_answer": None,
            }
        )
    return items


def load_mmlu_items(data_dir: Path) -> list[dict[str, Any]]:
    path = data_dir / "benchmarks" / "mmlu" / "all" / "test-00000-of-00001.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing MMLU full test file: {path}")

    items: list[dict[str, Any]] = []
    for index, record in enumerate(load_parquet_records(path)):
        choices = [str(choice) for choice in record["choices"]]
        answer_index = int(record["answer"])
        answer_letter = chr(ord("A") + answer_index)
        option_lines = [
            f"({chr(ord('A') + option_index)}) {choice}"
            for option_index, choice in enumerate(choices)
        ]
        task = (
            f"Subject: {record.get('subject')}\n"
            f"Question: {record['question']}\n"
            "Options:\n"
            + "\n".join(option_lines)
        )
        items.append(
            {
                "sample_index": index,
                "sample_id": f"mmlu-test-{index:05d}",
                "benchmark": "MMLU",
                "task": task,
                "question": str(record["question"]),
                "subject": record.get("subject"),
                "choices": choices,
                "answer_index": answer_index,
                "reference_answer": answer_letter,
            }
        )
    return items


def load_humaneval_items(data_dir: Path) -> list[dict[str, Any]]:
    path = (
        data_dir
        / "benchmarks"
        / "humaneval"
        / "openai_humaneval"
        / "test-00000-of-00001.parquet"
    )
    if not path.exists():
        raise FileNotFoundError(f"missing HumanEval full test file: {path}")

    items: list[dict[str, Any]] = []
    for index, record in enumerate(load_parquet_records(path)):
        prompt = str(record["prompt"])
        task_id = str(record["task_id"])
        items.append(
            {
                "sample_index": index,
                "sample_id": safe_sample_slug(task_id),
                "benchmark": "HumanEval",
                "task": (
                    "Complete the following Python function. Return only the completed code for the "
                    "function body or the full function; do not include Markdown fences.\n\n"
                    f"{prompt}"
                ),
                "task_id": task_id,
                "prompt": prompt,
                "canonical_solution": record.get("canonical_solution"),
                "test": record.get("test"),
                "entry_point": record.get("entry_point"),
                "reference_answer": None,
            }
        )
    return items


def load_olympiadbench_items(data_dir: Path) -> list[dict[str, Any]]:
    path = data_dir / "benchmarks" / "olympiadbench" / "data" / "test_en-00000-of-00001.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing OlympiadBench English test file: {path}")

    items: list[dict[str, Any]] = []
    for index, record in enumerate(load_parquet_records(path)):
        question_id = str(record.get("question_id") or index)
        question = str(record.get("question") or "")
        raw_answer = record.get("final_answer")
        # The official evaluator scores the first annotated final answer.
        if isinstance(raw_answer, list):
            references = [str(answer) for answer in raw_answer if answer is not None]
        elif raw_answer is not None:
            references = [str(raw_answer)]
        else:
            references = []
        items.append(
            {
                "sample_index": index,
                "sample_id": f"olympiadbench-en-test-{index:05d}",
                "benchmark": "OlympiadBench",
                "task": question,
                "question": question,
                "question_id": question_id,
                "subfield": record.get("subfield"),
                "answer_type": record.get("answer_type"),
                "is_multiple_answer": record.get("is_multiple_answer"),
                "unit": record.get("unit"),
                "source": record.get("source"),
                "error": record.get("error"),
                "reference_answer": references[0] if references else "",
                "reference_answers": references,
            }
        )
    return items


def load_swe_bench_lite_items(data_dir: Path) -> list[dict[str, Any]]:
    path = data_dir / "benchmarks" / "swe-bench-lite" / "data" / "test-00000-of-00001.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing SWE-Bench Lite test file: {path}")

    items: list[dict[str, Any]] = []
    for index, record in enumerate(load_parquet_records(path)):
        instance_id = str(record.get("instance_id") or f"swe-{index:05d}")
        repo = str(record.get("repo") or "")
        problem = str(record.get("problem_statement") or "")
        hints = str(record.get("hints_text") or "")
        task = problem
        if hints:
            task = f"{problem}\n\nHints:\n{hints}"
        items.append(
            {
                "sample_index": index,
                "sample_id": safe_sample_slug(instance_id),
                "benchmark": "SWE-Bench-Lite",
                "task": task,
                "question": problem,
                "instance_id": instance_id,
                "repo": repo,
                "base_commit": record.get("base_commit"),
                "version": record.get("version"),
                "fail_to_pass": record.get("FAIL_TO_PASS"),
                "pass_to_pass": record.get("PASS_TO_PASS"),
                "test_patch": record.get("test_patch"),
                "environment_setup_commit": record.get("environment_setup_commit"),
                "created_at": record.get("created_at"),
                "hints_text": hints,
                # The gold patch is kept as reference_answer (not shown to model)
                "reference_answer": str(record.get("patch") or ""),
            }
        )
    return items


def normalize_gaia_sample(sample: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(sample)
    record = normalized.get("record")
    trajectory = ""
    if isinstance(record, dict):
        trace = record.get("trace")
        if isinstance(trace, dict):
            trajectory = str(trace.get("trajectory", ""))
        else:
            trajectory = str(trace or "")

    task = extract_user_request_from_trajectory(trajectory) if trajectory else ""
    if task:
        normalized["task"] = task
    else:
        normalized["task"] = str(normalized.get("task", ""))

    answer = extract_final_answer_from_trajectory(trajectory) if trajectory else None
    if answer:
        normalized["reference_answer"] = answer

    trace_id = normalized.get("trace_id")
    normalized.setdefault("sample_index", 0 if trace_id is None else trace_id)
    normalized.setdefault("sample_id", f"gaia-mad-trace-{trace_id if trace_id is not None else 0}")
    normalized.setdefault("benchmark", "GAIA")
    return normalized


def load_parquet_records(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "loading parquet benchmarks requires pyarrow; install project dependencies with uv"
        ) from exc

    return pq.read_table(path).to_pylist()


def safe_sample_slug(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return safe or "sample"


def extract_task_text(record: dict[str, Any]) -> str:
    trace = record.get("trace")
    if isinstance(trace, dict):
        trajectory = str(trace.get("trajectory", ""))
    else:
        trajectory = str(trace or "")

    user_request = extract_user_request_from_trajectory(trajectory)
    if user_request:
        return user_request

    markers = ("task_prompt**:", "problem_statement:", "Query requirements:", "Task ")
    for marker in markers:
        start = trajectory.find(marker)
        if start >= 0:
            snippet = trajectory[start + len(marker) : start + len(marker) + 800]
            return " ".join(snippet.split())
    return " ".join(trajectory[:800].split())


def extract_user_request_from_trajectory(trajectory: str) -> str | None:
    match = re.search(
        r"-{10}\s*user\s*-{10}\s*(.*?)\s*-{10}\s*[A-Za-z0-9_ -]+?\s*-{10}",
        trajectory,
        flags=re.DOTALL,
    )
    if not match:
        return None
    text = " ".join(match.group(1).split())
    return text or None


def extract_final_answer_from_trajectory(trajectory: str) -> str | None:
    matches = re.findall(r"FINAL ANSWER:\s*([^\n\r]+)", trajectory, flags=re.IGNORECASE)
    if not matches:
        return None
    answer = matches[-1].strip().strip("` ")
    answer = answer.rstrip(".。,:;")
    return answer or None


def extract_failure_modes(value: Any) -> set[str]:
    modes: set[str] = set()
    if isinstance(value, dict):
        mast_annotation = value.get("mast_annotation")
        if isinstance(mast_annotation, dict):
            for key, item in mast_annotation.items():
                mode = first_failure_mode_code(str(key))
                if mode and is_positive(item):
                    modes.add(mode)

        failure_mode_text = value.get("failure mode") or value.get("failure_mode")
        if isinstance(failure_mode_text, str):
            votes = [
                item
                for key, item in value.items()
                if normalize_key(key).startswith("annotator")
            ]
            if votes and sum(bool(item) for item in votes) >= max(1, len(votes) // 2 + 1):
                mode = first_failure_mode_code(failure_mode_text)
                if mode:
                    modes.add(mode)

        for key, item in value.items():
            if key in {"mast_annotation", "failure mode", "failure_mode"}:
                continue
            if isinstance(item, (dict, list)):
                modes.update(extract_failure_modes(item))
    elif isinstance(value, list):
        for item in value:
            modes.update(extract_failure_modes(item))
    return modes


def first_failure_mode_code(value: str) -> str | None:
    for match in re.findall(r"(?:FM[- ]?)?([1-4]\.[1-6])", value, flags=re.IGNORECASE):
        if match in FAILURE_MODE_CODES:
            return match
    return None


def is_positive(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return False


def normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")


SAMPLE_NORMALIZERS = {
    "gaia": normalize_gaia_sample,
}

from common.ood_benchmarks import load_arc_items, load_bbh_items

FULL_BENCHMARK_LOADERS = {
    "bbh": load_bbh_items,
    "arc_challenge": load_arc_items,
    "gsm_plus": lambda data_dir, split: load_gsm_plus_items(data_dir),
    "programdev": lambda data_dir, split: load_programdev_items(data_dir),
    "mmlu": lambda data_dir, split: load_mmlu_items(data_dir),
    "humaneval": lambda data_dir, split: load_humaneval_items(data_dir),
    "gaia": lambda data_dir, split: load_gaia_items(data_dir, split=split),
    "olympiadbench": lambda data_dir, split: load_olympiadbench_items(data_dir),
    "swe_bench_lite": lambda data_dir, split: load_swe_bench_lite_items(data_dir),
}
