"""Project native baseline results using the appendix's strict SR/HIR/EB contract."""
from pathlib import Path
import argparse
import json
from collections import defaultdict
from dataclasses import replace
from baselines.core.datasets import load_baseline_samples
from baselines.core.evaluation import _outcome, _is_harmful
from baselines.registry import build_method
from baselines.runner import _result_from_row, _select_evaluation_pool
from common.artifacts import atomic_write_json
from masfm.config import load_config


def summarize(samples, rows):
    """Keep the planned pool fixed; never turn missing evaluator results into errors."""
    passed = harmed = resolved = unresolved_harm = method_invalid = 0
    budget = inclusive = 0.0
    missing = 0
    for sample in samples:
        row = rows.get(sample.sample_id)
        if row is None:
            missing += 1
            continue
        result = _result_from_row(row)
        sample = replace(sample, control_evaluation=row.get('control_evaluation') or sample.control_evaluation)
        success, category = _outcome(sample, result)
        if category == 'method_invalid':
            method_invalid += 1
            if result.setting == 'passive':
                success = False  # Malformed repair does not recover the known failed source.
        harm = _is_harmful(sample, result, success)
        if category == 'method_invalid':
            harm = True
        resolved += type(success) is bool
        passed += success is True
        harmed += harm is True
        unresolved_harm += harm is None
        executed = sum(e.execution_count for e in result.executions)
        eb = executed if result.setting == 'passive' else result.execution_budget
        budget += eb
        inclusive += eb + (result.source_execution_count if result.setting == 'passive' else 0)
    n = len(samples)
    complete = bool(n and resolved == n and not missing and not unresolved_harm)
    return {'planned': n, 'missing': missing, 'resolved': resolved,
            'method_invalid': method_invalid, 'unresolved_harm': unresolved_harm,
            'SR': passed / n if complete else None, 'HIR': harmed / n if complete else None,
            'EB': budget / n if complete else None,
            'source_inclusive_EB': inclusive / n if complete else None,
            'status': 'resolved' if complete else 'incomplete'}


def report(path):
    config = load_config(path)
    for spec in config.baselines:
        method = build_method(spec['method'], **spec.get('method_options', {}))
        pool = (path.parent / spec['input_path']).resolve()
        samples = load_baseline_samples(pool, input_format=spec.get('input_format', 'aria-full'), max_events=spec.get('max_events', 64))
        samples, _ = _select_evaluation_pool(samples, setting=method.setting, split=spec.get('split'))
        if spec.get('benchmark_filter'):
            samples = [s for s in samples if s.benchmark in spec['benchmark_filter']]
        output = (path.parent / spec['output_dir']).resolve()
        result_path = output / 'results.jsonl'
        rows = {}
        if result_path.exists():
            for line in result_path.open():
                row = json.loads(line)
                if row['sample_id'] in rows:
                    raise ValueError('duplicate sample in formal result ledger')
                rows[row['sample_id']] = row
        if set(rows) - {s.sample_id for s in samples}:
            raise ValueError('result outside the declared evaluation pool')
        groups = defaultdict(list)
        for sample in samples:
            groups[sample.benchmark if method.setting == 'passive' else sample.mas].append(sample)
        value = {'schema_version': 'aria-release-paper-metrics-v1', 'method': method.name,
                 'overall': summarize(samples, rows),
                 'groups': {k: summarize(v, rows) for k, v in groups.items()},
                 'contract': 'Fixed pool; unresolved headlines withheld; posthoc EB excludes source acquisition.'}
        atomic_write_json(output / 'paper_summary.json', value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', type=Path, required=True)
    report(parser.parse_args().config.resolve())


if __name__ == '__main__':
    main()
