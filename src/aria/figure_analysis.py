"""Paired figure statistics. Never impute missing runs or infer executions from caps."""
from collections import Counter, defaultdict
from statistics import mean

from aria.figure_logging import PAIR_KEY, execution_budget, validate_case


def common_cases(rows, members, *, member_field):
    """One evaluated setting/pool at a time; duplicate rows are a protocol error."""
    index = defaultdict(dict)
    settings = set()
    for row in rows:
        validate_case(row)
        member = row[member_field]
        if member not in members:
            continue
        settings.add((row['pool_id'], row['protocol_sha256'], row['mode'],
                      row['gate_threshold'], row['budget_setting_id']))
        key = tuple(row[k] for k in PAIR_KEY)
        if member in index[key]:
            raise ValueError('duplicate matched outcome; resolve retries before aggregation')
        index[key][member] = row
    if len(settings) > 1:
        raise ValueError('do not mix pools, thresholds or budget protocols')
    common = []
    coverage = Counter()
    for variants in index.values():
        for member in members:
            coverage[variants[member]['status'] if member in variants else 'missing'] += 1
        if all(m in variants and variants[m]['status'] == 'resolved' for m in members):
            controls = {(r['control_run_id'], r['control_correct'], r['control_score']) for r in variants.values()}
            if len(controls) != 1:
                raise ValueError('model sizes/variants must share the exact matched control')
            common.append(variants)
    return common, {'planned_pairs': len(index), 'jointly_resolved': len(common),
                    'member_outcomes': dict(coverage)}


def conditional_recovery(rows, sizes):
    common, coverage = common_cases(rows, sizes, member_field='backbone')
    by_seed = defaultdict(list)
    for variants in common:
        by_seed[variants[sizes[0]]['training_seed']].append(variants)
    seeds = sorted({r['training_seed'] for r in rows})
    rates = {}
    for size in sizes:
        per_seed = {}
        for seed in seeds:
            failed = [v[size] for v in by_seed[seed] if v[size]['control_correct'] is False]
            per_seed[seed] = {'control_failures': len(failed), 'recovered': sum(r['correct'] for r in failed),
                              'rate': mean(r['correct'] for r in failed) if failed else None}
        values = [v['rate'] for v in per_seed.values()]
        rates[size] = {'per_seed': per_seed, 'rate': mean(values) if values and all(v is not None for v in values) else None}
    return {'coverage': coverage, 'sizes': rates}


def outcome_transitions(rows):
    members = ('S1', 'S1+2', 'S1+3', 'Full')
    edges = (('S1', 'S1+2'), ('S1+3', 'Full'), ('S1', 'S1+3'), ('S1+2', 'Full'))
    common, coverage = common_cases(rows, members, member_field='variant')
    by_seed = defaultdict(list)
    for variants in common:
        by_seed[variants['S1']['training_seed']].append(variants)
    seeds = sorted({r['training_seed'] for r in rows})
    result = {}
    for left, right in edges:
        per_seed = {}
        for seed in seeds:
            cases = by_seed[seed]
            n = len(cases)
            gains = sum(not v[left]['correct'] and v[right]['correct'] for v in cases)
            losses = sum(v[left]['correct'] and not v[right]['correct'] for v in cases)
            per_seed[seed] = {'n': n, 'gains': gains, 'losses': losses,
                              'gain_rate': gains / n if n else None, 'loss_rate': losses / n if n else None}
        result[left + '->' + right] = {'per_seed': per_seed}
        for metric in ('gain_rate', 'loss_rate'):
            values = [v[metric] for v in per_seed.values()]
            result[left + '->' + right][metric] = mean(values) if values and all(v is not None for v in values) else None
    return {'coverage': coverage, 'edges': result}


def runtime_summary(rows):
    """All runtime tasks, including non-triggering runs; equal training-seed weight."""
    seeds = defaultdict(list)
    coverage = Counter()
    for row in rows:
        validate_case(row)
        coverage[row['status']] += 1
        seeds[row['training_seed']].append(row)
    complete = bool(rows) and all(r['status'] == 'resolved' for r in rows)
    result = {'coverage': dict(coverage), 'sr': None, 'hir': None, 'eb': None}
    if not complete:
        return result
    result['sr'] = mean(mean(r['correct'] for r in group) for group in seeds.values())
    result['hir'] = mean(mean(r['harmful'] for r in group) for group in seeds.values())
    budgets = [[execution_budget(r) for r in group] for group in seeds.values()]
    if all(v is not None for group in budgets for v in group):
        result['eb'] = mean(mean(group) for group in budgets)
    return result
