"""Build complete two-backbone RQ1 evaluation, with no sample or time cutoff."""

from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path

import yaml

from aria.post_training.environment import BenchmarkSampleResolver, EnvironmentIncident, _normalize_identifier
from aria.split_audit import audit_disjoint
from baselines.core.datasets import load_baseline_samples
from common.artifacts import atomic_write_json, atomic_write_jsonl, atomic_write_text
from common.runtime_support import safe_run_id
from inference.runner import job_output_root
from masfm.config import load_config

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/aria/rq1-full-20260912"


class IndexedResolver(BenchmarkSampleResolver):
    """Index exact IDs and suffixes; retain the canonical resolver for ambiguity."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.indices = {}

    def resolve(self, incident):
        benchmark = incident.benchmark_id
        split = self.benchmark_splits.get(benchmark, incident.split or "validation")
        items = self._load(benchmark, split)
        key = (benchmark, split)
        if key not in self.indices:
            index = defaultdict(set)
            for number, item in enumerate(items):
                index[_normalize_identifier(str(item.get("sample_id") or ""))].add(number)
            self.indices[key] = index
        index = self.indices[key]
        ids = {_normalize_identifier(str(value)) for value in (
            incident.task_id, incident.incident_id, incident.evaluator_sample.get("sample_id")) if value}
        exact = set().union(*(index.get(value, set()) for value in ids))
        if len(exact) == 1:
            return dict(items[next(iter(exact))]), "benchmark-cache-id"
        suffix = set()
        for value in ids:
            parts = value.split("-")
            for offset in range(len(parts)):
                suffix.update(index.get("-".join(parts[offset:]), set()))
        if len(suffix) == 1:
            return dict(items[next(iter(suffix))]), "benchmark-cache-suffix"
        return super().resolve(incident)


def native_groups(samples, resolver):
    groups, seen, identities = defaultdict(list), Counter(), []
    for sample in samples:
        incident = EnvironmentIncident.from_dict({
            "incident_id": sample.sample_id, "task_id": sample.task_id, "split": sample.split,
            "events": [e.to_dict() for e in sample.events[:1]], "constraints": {"task_text": sample.task},
            "cell": {"mas_id": sample.mas, "benchmark_id": sample.benchmark},
            "evaluator_sample": sample.metadata.get("evaluator_sample") or {},
        })
        restored, detail = resolver.resolve(incident)
        if restored is None:
            raise ValueError(f"unresolvable full-test incident {sample.sample_id}: {detail}")
        key = (sample.mas, sample.benchmark, str(restored["sample_id"]))
        replicate = seen[key]
        seen[key] += 1
        groups[(sample.mas, sample.benchmark, replicate)].append(key[2])
        identities.append({"incident_id": sample.sample_id, "mas": sample.mas,
            "benchmark": sample.benchmark, "native_sample_id": key[2], "replicate": replicate})
    return groups, identities


def build():
    config_root = OUT / "configs"
    if config_root.exists():
        raise FileExistsError("full evaluation config identity exists")
    pool = ROOT / "outputs/aria/pools/rq1_complete_test.jsonl"
    samples = load_baseline_samples(pool, input_format="aria-full", max_events=100000)
    if len(samples) != 4890 or any(s.split != "test" for s in samples):
        raise ValueError("unexpected full test pool")
    audit = audit_disjoint([ROOT / "outputs/aria/corpora/rq1-final-v6/train",
        ROOT / "outputs/aria/corpora/rq1-final-v6-native-env/env_train.jsonl"], [pool])
    if audit["status"] != "passed":
        raise ValueError("full test overlaps training")
    resolver = IndexedResolver(data_dir=ROOT / "data", benchmark_splits={
        b: "validation" if b == "gaia" else "test" for b in {s.benchmark for s in samples}})
    groups, identities = native_groups(samples, resolver)
    atomic_write_json(OUT / "native_identities.json", identities)
    raw = {row["sample_id"]: row for row in map(json.loads, pool.read_text().splitlines())}
    shards = [[] for _ in range(16)]
    for index, sample in enumerate(sorted(samples, key=lambda s: s.sample_id)):
        shards[index % len(shards)].append(sample)
    for index, shard in enumerate(shards):
        atomic_write_jsonl(OUT / "pools" / f"test-{index:02d}.jsonl", [raw[s.sample_id] for s in shard])
    source = yaml.safe_load((ROOT / "configs/baselines/rq1_final/qwen_fast_step4_validation.yml").read_text())
    runtime_template = yaml.safe_load((ROOT / "outputs/aria/live-v5-configs/openmanus-aria.yml").read_text())
    jobs, models = [], {}

    def save(config, name, backbone, panel, method, count, environment):
        path = config_root / f"{name}.yml"
        atomic_write_text(path, yaml.safe_dump(config, sort_keys=False))
        parsed = load_config(path)
        if panel == "posthoc":
            results = [str(Path(config["baselines"][0]["output_dir"]) / "results.jsonl")]
        else:
            results = [str(job_output_root(parsed) / e["mas"] / e["benchmark"] /
                safe_run_id(e["name"]) / "results.jsonl") for e in config["experiments"]]
        jobs.append({"name": name, "backbone": backbone, "panel": panel, "method": method,
            "expected_samples": count, "config": str(path), "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "environment": str(environment), "result_files": results})

    for backbone, model_dir, port in (("qwen", "qwen25-7b", 18710), ("llama", "llama31-8b", 18711)):
        checkpoint = ROOT / f"outputs/aria/post_training/{model_dir}/rq1-final-v6-preference/checkpoints/transfer_diagnosis/checkpoint"
        models[backbone] = {"checkpoint": str(checkpoint),
            "sha256": hashlib.sha256((checkpoint / "adapter_model.safetensors").read_bytes()).hexdigest(),
            "base_url": f"http://127.0.0.1:{port}/v1", "base_model": f"rq1-full-{backbone}-base",
            "aria_model": f"rq1-full-{backbone}-aria", "training": "SFT+DPO+Transfer; Stage3 omitted for both backbones"}
        api = {**source["api"], "base_url": models[backbone]["base_url"],
               "model": models[backbone]["base_model"], "context_window_tokens": 32768, "max_tokens": 2048}
        replay = copy.deepcopy(source["baselines"][0]["replay_backend"])
        replay["options"].update(host_api=api, control_output_root=str(OUT / backbone / "posthoc-controls"))
        replay["options"]["native_options"]["context_window_tokens"] = 32768
        for method in ("aria_passive", "zero_shot", "who_when", "agentrx", "errorprobe", "dover"):
            for index, shard in enumerate(shards):
                name = f"{backbone}-posthoc-{method}-{index:02d}"
                config = copy.deepcopy(source)
                config.update(output_dir=str(OUT / backbone), baseline_workers=1,
                    api={**api, "max_tokens": 4096,
                         "model": models[backbone]["aria_model"] if method == "aria_passive" else api["model"]})
                entry = config["baselines"][0]
                entry.update(name=name, method=method, split="test", input_path=str(OUT / "pools" / f"test-{index:02d}.jsonl"),
                    output_dir=str(OUT / backbone / "posthoc" / method / f"shard-{index:02d}"),
                    prompt_char_budget=48000, sample_workers=1, method_options={}, replay_backend=copy.deepcopy(replay))
                if method == "aria_passive":
                    entry["method_options"]["structured_decoding"] = True
                save(config, name, backbone, "posthoc", method, sum(s.source_success is False for s in shard), ROOT / ".venv")
        for (mas, benchmark, replicate), ids in sorted(groups.items()):
            for offset in range(0, len(ids), 32):
                selected = sorted(ids)[offset:offset + 32]
                for arm in ("aria", "control", "spark_to_fire", "safeagent"):
                    name = f"{backbone}-runtime-{arm}-{mas}-{benchmark}-r{replicate}-s{offset // 32:03d}"
                    config = copy.deepcopy(runtime_template)
                    config.update(api=api, output_dir=str(OUT / backbone / "runtime" / arm),
                        workspace_dir=str(OUT / backbone / "runtime" / arm / "workspace"),
                        job="inference" if arm == "control" else "aria")
                    if arm == "control":
                        config["aria"] = {"enabled": False, "mode": "off"}
                    else:
                        model = config["aria"]["model"]
                        model.pop("gate_calibration_path", None)
                        model.update(base_url=api["base_url"], model=models[backbone]["aria_model"] if arm == "aria" else api["model"],
                                     max_tokens=4096, timeout_seconds=900, seed=42, temperature=0.0)
                        config["aria"].update(runtime_policy=arm, gate_threshold=0.5)
                    experiment = {"name": name, "mas": mas, "benchmark": benchmark, "native": True, "batch": True,
                        "sample_source": "full", "sample_split": "validation" if benchmark == "gaia" else "test",
                        "sample_ids": selected, "sample_limit": len(selected), "resume": True, "judge": False,
                        "replicate_id": f"source-repeat-{replicate}", **replay["options"]["native_options"]}
                    if benchmark in {"gaia", "swe-bench-lite"}:
                        experiment["evaluation"] = {**replay["options"]["evaluation"],
                            "harness": "gaia-official" if benchmark == "gaia" else "swebench-official"}
                    if mas not in {"ag2", "magentic-one"}:
                        experiment.pop("code_execution_environment", None)
                    if mas != "magentic-one":
                        experiment.pop("model_vision", None)
                    config["experiments"] = [experiment]
                    save(config, name, backbone, "runtime", arm, len(selected), ROOT / f".venv-{mas}")
    counts = Counter()
    for job in jobs:
        counts[f"{job['backbone']}/{job['panel']}/{job['method']}"] += job["expected_samples"]
    for key, count in counts.items():
        expected = 3953 if "/posthoc/" in key else 4890
        if count != expected:
            raise ValueError(f"incomplete evaluation coverage: {key}: {count} != {expected}")
    manifest = {"scope": "complete local-host RQ1 test; no sample reduction or wall-clock cutoff",
        "pool": str(pool), "pool_sha256": hashlib.sha256(pool.read_bytes()).hexdigest(), "split_audit": audit,
        "models": models, "expected_counts": dict(counts), "native_identities": identities, "jobs": jobs,
        "workers": {"qwen": 16, "llama": 8}}
    atomic_write_json(OUT / "manifest.json", manifest)
    print(json.dumps({"jobs": len(jobs), "expected_counts": dict(counts)}))


if __name__ == "__main__":
    build()
