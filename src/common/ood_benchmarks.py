"""Pinned, evaluation-only RQ2 datasets; labels never enter the task prompt."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

BBH_REVISION = "9ee07bd481feebf959a6b59d61ea57bdcf30964d"
ARC_REVISION = "210d026faf9955653af8916fad021475a3f00453"
BBH_TASKS = tuple("""boolean_expressions causal_judgement date_understanding
 disambiguation_qa dyck_languages formal_fallacies geometric_shapes hyperbaton
 logical_deduction_five_objects logical_deduction_seven_objects
 logical_deduction_three_objects movie_recommendation multistep_arithmetic_two
 navigate object_counting penguins_in_a_table reasoning_about_colored_objects
 ruin_names salient_translation_error_detection snarks sports_understanding
 temporal_sequences tracking_shuffled_objects_five_objects
 tracking_shuffled_objects_seven_objects tracking_shuffled_objects_three_objects
 web_of_lies word_sorting""".split())
ANSWER_CONTRACT = "\n\nEnd with a single line: FINAL ANSWER: <answer>."


def load_bbh_items(data_dir: Path, split: str) -> list[dict[str, Any]]:
    if split != "test":
        raise ValueError("BBH is evaluation-only; sample_split must be test")
    families = []
    for family in BBH_TASKS:
        path = data_dir / "benchmarks" / "bbh" / f"{family}.json"
        records = json.loads(path.read_text(encoding="utf-8"))["examples"]
        family_items = []
        for index, record in enumerate(records):
            if not record.get("input") or not str(record.get("target", "")).strip():
                raise ValueError(f"invalid BBH reference at {family}:{index}")
            family_items.append({
                "sample_id": f"bbh-{family}-{index:04d}", "benchmark": "bbh",
                "task": record["input"] + ANSWER_CONTRACT,
                "reference_answer": str(record["target"]), "task_family": family,
                "source_revision": BBH_REVISION,
            })
        families.append(family_items)
    # Round robin prevents a small fixed pool from containing only one family.
    items = [family[i] for i in range(max(map(len, families)))
             for family in families if i < len(family)]
    return [{**item, "sample_index": i} for i, item in enumerate(items)]


def load_arc_items(data_dir: Path, split: str) -> list[dict[str, Any]]:
    from common.mast_data import load_parquet_records

    if split != "test":
        raise ValueError("ARC-Challenge is evaluation-only; sample_split must be test")
    path = data_dir / "benchmarks" / "arc-challenge" / "test.parquet"
    items = []
    for i, row in enumerate(load_parquet_records(path)):
        labels = [str(x) for x in row["choices"]["label"]]
        texts = row["choices"]["text"]
        answer = str(row["answerKey"])
        if answer not in labels or len(labels) != len(texts) or len(set(labels)) != len(labels):
            raise ValueError(f"invalid ARC choices/reference: {row['id']}")
        # Some ARC rows use numeric labels; preserve the published labels.
        choices = "\n".join(f"({label}) {text}" for label, text in zip(labels, texts))
        items.append({
            "sample_id": f"arc-challenge-{row['id']}", "sample_index": i,
            "benchmark": "arc-challenge", "task": f"{row['question']}\n{choices}"
            + "\nReturn the option label." + ANSWER_CONTRACT,
            "reference_answer": answer, "choice_labels": labels,
            "source_revision": ARC_REVISION,
        })
    return items


def evaluate_ood(sample: dict[str, Any], result: dict[str, Any], run_root: Path) -> dict[str, Any]:
    """Score only the delivered final answer, never a draft or echoed gold label.

    BBH uses exact target match with whitespace normalization. ARC additionally
    accepts parentheses around an option label. No LLM judge or fuzzy matching.
    """
    benchmark = sample["benchmark"]
    reference = str(sample.get("reference_answer", "")).strip()
    if not reference:
        return {"benchmark": benchmark, "correct": None, "score": None,
                "method": "ood_exact_v2", "reason": "missing reference answer"}
    text = str(result.get("final_output") or "").strip()
    matches = re.findall(r"(?im)^[ \t]*FINAL[ _]ANSWER:[ \t]*(.*?)[ \t]*$", text)
    predicted = matches[-1] if matches else text
    normalize = lambda value: " ".join(value.split())
    if benchmark == "arc-challenge" or re.fullmatch(r"\([A-Z]\)", reference):
        predicted = predicted.strip().strip("()").upper()
        reference = reference.strip("()").upper()
    correct = normalize(predicted) == normalize(reference)
    return {"benchmark": benchmark, "correct": correct, "score": float(correct),
            "method": "ood_exact_v2", "predicted_answer": predicted,
            "reference_answer": sample["reference_answer"],
            "source_revision": sample.get("source_revision")}
