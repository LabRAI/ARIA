"""One auditable RQ3 outcome projection for the ARIA table and Figure04."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import yaml

from common.artifacts import atomic_write_json, atomic_write_jsonl, atomic_write_text
from scripts.aria.build_rq3_eval import EDGES, VARIANTS
from scripts.aria.report_rq1_eval import budget_case, csv_text, read_json, summary
from scripts.aria.verify_training_checkpoint import file_sha256


def cases(manifest, states):
    spec = manifest['spec']
    result = []
    for task in manifest['import_tasks'] + manifest['tasks']:
        row = budget_case([task], states, 1, states.get(task['control_id']))
        state = states.get(task['id'], {})
        control = states.get(task['control_id'], {})
        cp = manifest['checkpoints'].get(task['backbone'])
        row.update(schema_version='aria-rq3-case-v1', run_id=task['id'],
            backbone=task['model_family'], variant=task['variant'],
            incident_id=task['incident_id'], source_replicate=task['replicate'],
            native_task_id=task['native_sample_id'],
            training_seed=spec['training_seed'], execution_seed=spec['execution_seed'],
            host_config_sha256=manifest['host_config_sha256'], protocol_sha256=manifest['protocol_sha256'],
            pool_sha256=manifest['pool_sha256'], checkpoint_sha256=cp['sha256'] if cp else None,
            native_status=state.get('status', 'missing'), control_run_id=task['control_id'],
            control_score=(control.get('metrics') or {}).get('score'),
            score=(state.get('metrics') or {}).get('score') if row['status'] == 'resolved' else None,
            gate_events=(state.get('metrics') or {}).get('gate_events', []),
            execution_ledger=(state.get('metrics') or {}).get('execution_ledger', []),
            attempt_journal=state.get('attempts', []), raw_result=state.get('result'),
            raw_control_result=control.get('result'), origin=task['origin'],
            domain='in_domain', magentic_seen_subset=task['mas'] == 'magentic-one')
        result.append(row)
        if task['backbone'] == 'qwen':
            result.append({**row, 'run_id': task['case_id'] + '-runtime-s13-1', 'variant': 'S1+3',
                'status': 'missing', 'native_status': 'blocked_checkpoint', 'correct': None,
                'score': None, 'harmful': None, 'harm_reasons': [], 'execution_budget': None,
                'wall_seconds': None, 'raw_artifacts': [], 'consumed_candidate_ids': [],
                'selected_candidate_id': None, 'raw_result': None, 'origin': 'blocked',
                'inspection_calls': 0, 'inspection_tokens': 0, 'checkpoint_sha256': None,
                'gate_events': [], 'execution_ledger': [], 'attempt_journal': [],
                'invalid_reason': 'no matched 16+16 S1+3 checkpoint'})
    return result


def bootstrap(rows, values, *, draws=1000, seed=42):
    """Cluster repeated appearances of a benchmark task, preserving paired outcomes."""
    if not rows:
        return None
    clusters = defaultdict(list)
    for index, row in enumerate(rows):
        clusters[(row['benchmark'], row['native_task_id'])].append(index)
    data = np.asarray(values, dtype=float)
    totals = np.asarray([data[indices].sum(axis=0) for indices in clusters.values()])
    sizes = np.asarray([len(indices) for indices in clusters.values()])
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0, draws, 64):
        chosen = rng.integers(0, len(clusters), (min(64, draws - start), len(clusters)))
        samples.append(totals[chosen].sum(axis=1) / sizes[chosen].sum(axis=1)[:, None])
    bounds = np.quantile(np.concatenate(samples), [.025, .975], axis=0)
    return {'bounds': bounds.T.tolist(), 'clusters': len(clusters), 'draws': draws,
            'seed': seed, 'level': .95, 'scope': 'task sampling conditional on fixed training seed'}


def summarize(rows, spec):
    metrics = summary(rows)
    valid = [r for r in rows if r['status'] == 'resolved']
    ci = bootstrap(valid, [[r['correct'], r['harmful'], r['execution_budget']] for r in valid],
        draws=spec['bootstrap_draws'], seed=spec['bootstrap_seed'])
    metrics['resolved_subset_ci'] = ci
    metrics['ci'] = ci if metrics['coverage_complete'] else None
    return metrics


def matched(rows, members=VARIANTS):
    groups = defaultdict(dict)
    for row in rows:
        if row['variant'] not in members:
            continue
        key = (row['case_id'], row['training_seed'], row['execution_seed'])
        if row['variant'] in groups[key]:
            raise ValueError('duplicate paired task/seed/variant')
        groups[key][row['variant']] = row
    common, excluded = [], []
    for key, variants in groups.items():
        identities = {(v['incident_id'], v['native_task_id'], v['source_replicate'], v['mas'],
                       v['benchmark'], v['host_config_sha256'], v['protocol_sha256'], v['pool_sha256'],
                       v['control_run_id']) for v in variants.values()}
        if len(identities) != 1:
            raise ValueError('paired task/host/protocol/control identity mismatch')
        if all(m in variants and variants[m]['status'] == 'resolved' for m in members):
            if len({(v['control_correct'], v['control_score']) for v in variants.values()}) != 1:
                raise ValueError('inconsistent paired control verdict')
            if any(type(v['correct']) is not bool for v in variants.values()):
                raise ValueError('resolved outcomes must be boolean')
            common.append(variants)
        else:
            excluded.append({'case_id': key[0], 'training_seed': key[1], 'execution_seed': key[2],
                             'status': {m: variants.get(m, {}).get('status', 'missing') for m in members}})
    return common, excluded


def transitions(rows, *, members=VARIANTS, edges=EDGES, draws=1000, seed=42):
    common, excluded = matched(rows, members)
    seeds = sorted({r['training_seed'] for r in rows})
    per_seed = {}
    for training_seed in seeds:
        pool = [v for v in common if v[members[0]]['training_seed'] == training_seed]
        n = len(pool)
        points = []
        for left, right in edges:
            gains = [not v[left]['correct'] and v[right]['correct'] for v in pool]
            losses = [v[left]['correct'] and not v[right]['correct'] for v in pool]
            gain, loss = sum(gains), sum(losses)
            net = (gain - loss) / n if n else None
            sr_diff = sum(int(v[right]['correct']) - int(v[left]['correct']) for v in pool) / n if n else None
            if n and abs(net - sr_diff) > 1e-12:
                raise ValueError('transition identity violated')
            ci = bootstrap([v[left] for v in pool], [[g, l, int(g) - int(l)] for g, l in zip(gains, losses)],
                           draws=draws, seed=seed)
            points.append({'from': left, 'to': right, 'n': n, 'rescued': gain, 'lost': loss,
                'unchanged': n - gain - loss, 'rescued_pct': 100 * gain / n if n else None,
                'lost_pct': 100 * loss / n if n else None, 'net_pct': 100 * net if n else None,
                'sr_difference_pct': 100 * sr_diff if n else None, 'ci_fraction': ci})
        if n and tuple(edges) == EDGES:
            nets = [p['net_pct'] for p in points]
            if abs(nets[0] + nets[3] - nets[2] - nets[1]) > 1e-10:
                raise ValueError('S1-to-Full path identity violated')
        per_seed[str(training_seed)] = {'n': n,
            'denominator_case_ids': [v[members[0]]['case_id'] for v in pool], 'points': points}
    points = []
    for index, (left, right) in enumerate(edges):
        values = [s['points'][index] for s in per_seed.values()]
        record = {'from': left, 'to': right}
        for field in ('rescued_pct', 'lost_pct', 'net_pct'):
            record[field] = (sum(v[field] for v in values) / len(values)
                             if values and all(v[field] is not None for v in values) else None)
        points.append(record)
    return {'units': 'percent', 'members': list(members), 'jointly_resolved': len(common),
        'planned': len(common) + len(excluded), 'excluded': excluded, 'per_seed': per_seed,
        'points_equal_seed_weight': points, 'complete': bool(common) and not excluded}


def write_report(manifest, states):
    root, spec = Path(manifest['spec']['output_root']), manifest['spec']
    all_cases = cases(manifest, states)
    atomic_write_jsonl(root / 'data/cases.jsonl', all_cases)
    atomic_write_jsonl(root / 'data/invalid_or_pending.jsonl', [r for r in all_cases if r['status'] != 'resolved'])
    table, diagnostics, tex = [], [], []
    lines = ['# RQ3 ablation evaluation', '',
        'Formal results require complete task coverage and all specified training variants.', '',
        'Table rates are percentages; JSON/CSV SR and HIR are fractions. `—` denotes unavailable results.', '',
        '## ARIA results (training seed 42, local host)', '',
        '| Backbone / Variant | In-domain SR | In-domain HIR | Unseen system SR | Unseen system HIR | EB | Resolved / N |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    fmt = lambda x: '—' if x is None else f'{100*x:.2f}'
    for model in ('qwen', 'llama'):
        selected = [r for r in all_cases if r['backbone'] == model and r['variant'] == 'Full']
        metrics = summarize(selected, spec)
        record = {'backbone': model, 'variant': 'ARIA (S1+2+3)', 'training_seed': spec['training_seed'],
            'in_domain_sr': metrics['sr'], 'in_domain_hir': metrics['hir'], 'in_domain_eb': metrics['eb'],
            'unseen_system_sr': None, 'unseen_system_hir': None, 'eb': None,
            'unseen_reason': 'Magentic-One seen in training', 'overall': metrics}
        table.append(record)
        cells = [fmt(metrics['sr']), fmt(metrics['hir']), '—', '—', '—']
        lines.append(f'| {model} / ARIA | ' + ' | '.join(cells) + f" | {metrics['resolved']} / {metrics['expected']} |")
        tex.extend([f'% {model}; fixed training seed 42; local Qwen3-8B host; unseen-system/combined EB unavailable',
                    r'\textbf{ARIA (Stage 1 + 2 + 3)} & ' +
                    ' & '.join(x.replace('—', r'\textemdash') for x in cells) + r' \\'])
    lines += ['', 'Formal in-domain results require the complete pool. Combined EB requires both domains.',
        'The following subset statistics report evaluation progress only.', '',
        '| Backbone / Variant | Resolved / N | Subset SR (%) | Subset HIR (%) | Subset EB |',
        '| --- | ---: | ---: | ---: | ---: |']
    for model, variants in (('qwen', VARIANTS), ('llama', ('Full',))):
        for variant in variants:
            rows = [r for r in all_cases if r['backbone'] == model and r['variant'] == variant]
            metrics = summarize(rows, spec)
            diagnostics.append({'backbone': model, 'variant': variant, 'domain': 'in_domain', **metrics})
            eb = metrics['resolved_subset_eb']
            cells = [fmt(metrics['resolved_subset_sr']), fmt(metrics['resolved_subset_hir']), '—' if eb is None else f'{eb:.3f}']
            lines.append(f"| {model} / {variant} | {metrics['resolved']} / {metrics['expected']} | " + ' | '.join(cells) + ' |')
    qwen = [r for r in all_cases if r['backbone'] == 'qwen']
    kwargs = {'draws': spec['bootstrap_draws'], 'seed': spec['bootstrap_seed']}
    figure = {'schema_version': 'aria-rq3-figure04-v1', 'data_status': 'blocked_missing_checkpoint_and_heldout_models',
        'units': 'percent', 'training_seeds_available': [spec['training_seed']],
        'training_seeds_required_by_paper': 3, 'missing_variants': manifest['missing_variants'],
        'in_domain': transitions(qwen, **kwargs),
        'unseen_system': {'status': 'blocked_training_contamination', 'points': None,
            'reason': 'Magentic-One was present in the fitting data of both requested Full models'},
        'pairwise_diagnostics': []}
    for edge in (EDGES[0], EDGES[3]):
        figure['pairwise_diagnostics'].append({'qualification': 'two-arm overlap only; not the four-arm Figure04 denominator',
            **transitions(qwen, members=edge, edges=(edge,), **kwargs)})
    seen_rows = [r for r in all_cases if r['magentic_seen_subset'] and r['variant'] == 'Full']
    seen = [{'backbone': m, 'domain': 'seen_magentic_one', 'eligible_as_unseen': False,
             **summarize([r for r in seen_rows if r['backbone'] == m], spec)} for m in ('qwen', 'llama')]
    memberships = []
    grouped = defaultdict(dict)
    for row in qwen:
        grouped[row['case_id']][row['variant']] = row
    for case_id, variants in grouped.items():
        first = next(iter(variants.values()))
        memberships.append({k: first[k] for k in ('case_id', 'incident_id', 'benchmark', 'native_task_id',
            'source_replicate', 'mas', 'host_config_sha256', 'training_seed', 'execution_seed')} |
            {'outcomes': {v: {'status': r['status'], 'correct': r['correct'], 'raw_result': r['raw_result']}
                          for v, r in variants.items()},
             'in_four_variant_denominator': all(variants.get(v, {}).get('status') == 'resolved' for v in VARIANTS)})
    atomic_write_jsonl(root / 'figure04/paired_tasks.jsonl', memberships)
    atomic_write_json(root / 'figure04/data.json', figure)
    atomic_write_text(root / 'figure04/data.csv', csv_text(figure['in_domain']['points_equal_seed_weight']))
    atomic_write_json(root / 'table.json', table)
    atomic_write_text(root / 'table.csv', csv_text(table))
    atomic_write_text(root / 'table.tex', '\n'.join(tex) + '\n')
    atomic_write_json(root / 'data/variant_diagnostics.json', diagnostics)
    atomic_write_json(root / 'data/magentic_seen_diagnostics.json', seen)
    progress = {'planned_new_executions': len(manifest['tasks']), 'planned_imports': len(manifest['import_tasks']),
        'local': dict(Counter(states.get(t['id'], {}).get('status', 'pending') for t in manifest['tasks'])),
        'imported': dict(Counter(states.get(t['id'], {}).get('status', 'pending') for t in manifest['import_tasks'])),
        'full_in_domain_complete': all(r['overall']['coverage_complete'] for r in table),
        'paper_rq3_complete': False, 'figure04_complete': False}
    atomic_write_json(root / 'progress.json', progress)
    lines += ['', '## Ablation protocol', '',
        '- Figure 04 compares S1 to S1+2, S1+3 to Full, S1 to S1+3, and S1+2 to Full on the same four-variant scoreable task intersection, including unchanged outcomes.',
        '- rescued_pct and lost_pct use that common denominator. Net changes must equal the SR difference and agree across both S1-to-Full paths.',
        '- Missing variants remain null. Pairwise diagnostics cannot substitute for the complete four-variant figure.',
        '- Both Full checkpoints include Magentic-One in training; these results are stored as seen-system diagnostics. Held-out claims require separately trained checkpoints.',
        f"- The frozen RQ1 pool contains {len(manifest['identities'])} source records, including explicit source replicates.",
        '- Full and control reuse RQ1 budget-1 executions. Task identity, host, seeds, decoding, limits and evaluator remain fixed across variants.',
        '- Checkpoint hashes, stage update counts and training provenance are recorded in setup/. Training budgets need not be equal across variants.',
        '- HIR includes structural/constraint errors and applied interventions that reduce the control score. Missing and infrastructure-invalid evaluations remain distinct.',
        '- EB counts actual executions; shared control costs are separate. Unavailable total costs remain null. Bootstrap intervals reflect task sampling at a fixed training seed.', '',
        '## Resume / re-project', '', '```bash',
        f"PYTHONPATH={root / 'collector'}:{root / 'code'}:{root / 'code/src'} uv run --no-sync python {root / 'collector/scripts/aria/run_rq3_eval.py'} -c {root / 'launch.yml'}",
        f"PYTHONPATH={root / 'collector'}:{root / 'code'}:{root / 'code/src'} uv run --no-sync python {root / 'collector/scripts/aria/report_rq3_eval.py'} -c {root / 'launch.yml'}",
        '```', '', 'The scheduler permits one writer per output directory. Reporting does not call models.', '']
    atomic_write_text(root / 'README.md', '\n'.join(lines))
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', required=True, type=Path)
    launch = yaml.safe_load(parser.parse_args().config.read_text())
    if file_sha256(Path(launch['manifest'])) != launch['manifest_sha256']:
        raise ValueError('manifest changed')
    manifest = read_json(launch['manifest'])
    states = read_json(Path(manifest['spec']['output_root']) / 'state.json', {}).get('tasks', {})
    print(json.dumps(write_report(manifest, states)))


if __name__ == '__main__':
    main()
