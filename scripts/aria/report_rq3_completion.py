"""Report all RQ3 seeds and held-out training versions without imputing missing results."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import statistics

from common.artifacts import atomic_write_json, atomic_write_jsonl, atomic_write_text
from scripts.aria.build_rq1_eval import digest
from scripts.aria.report_rq1_eval import budget_case, csv_text, summary
from scripts.aria.report_rq3_eval import transitions


def project(manifest, states):
    result = []
    for task in manifest['tasks']:
        if task['method'] == 'control':
            continue
        row = budget_case([task], states, 1, states.get(task['control_id']))
        state = states.get(task['id'], {})
        control = states.get(task['control_id'], {})
        row.update(schema_version='aria-rq3-completion-case-v1', backbone=task['family'],
            variant=task['variant'], axis=task['axis'], training_seed=task['training_seed'],
            execution_seed=task['execution_seed'], native_task_id=task['native_sample_id'],
            source_replicate=task['replicate'], control_run_id=task['control_id'],
            control_score=control.get('metrics', {}).get('score'),
            host_config_sha256=digest({'api': {**manifest['templates']['host_api'], 'seed': task['execution_seed']},
                                      'native': manifest['spec']['native_options']}),
            protocol_sha256=manifest['protocol_sha256'], pool_sha256=manifest['pool_sha256'],
            checkpoint_id=task['checkpoint_id'], gate_events=state.get('metrics', {}).get('gate_events', []),
            execution_ledger=state.get('metrics', {}).get('execution_ledger', []),
            raw_result=state.get('result'), raw_control_result=control.get('result'),
            attempt_journal=state.get('attempts', []), source_provenance=state.get('source_provenance'))
        result.append(row)
    return result


def seeded_summary(rows):
    seeds = sorted({r['training_seed'] for r in rows})
    groups = {str(seed): summary([r for r in rows if r['training_seed'] == seed]) for seed in seeds}
    complete = bool(groups) and all(g['coverage_complete'] for g in groups.values())
    out = {'training_seeds': seeds, 'per_seed': groups, 'coverage_complete': complete,
           'resolved': sum(g['resolved'] for g in groups.values()), 'expected': len(rows)}
    for metric in ('sr', 'hir', 'eb'):
        out[metric] = statistics.mean(g[metric] for g in groups.values()) if complete else None
    return out


def write_report(manifest, states):
    root = Path(manifest['spec']['output_root'])
    evaluation_only = manifest['spec'].get('evaluation_only', False)
    rows = project(manifest, states)
    atomic_write_jsonl(root / 'data/cases.jsonl', rows)
    atomic_write_jsonl(root / 'data/invalid_or_pending.jsonl', [r for r in rows if r['status'] != 'resolved'])
    lines = ['# RQ3 ablation and held-out evaluation', '',
        'Results use accepted checkpoints and native evaluator outcomes. `—` denotes unavailable results.', '',
        '| Backbone / Method | In-domain SR (%) | HIR (%) | Unseen-system SR (%) | HIR (%) | EB | Resolved / N |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    table, tex = [], []
    fmt = lambda x: '—' if x is None else f'{100*x:.2f}'
    for model in ('qwen', 'llama'):
        selected = [r for r in rows if r['backbone'] == model and r['variant'] == 'Full']
        domains = {axis: seeded_summary([r for r in selected if r['axis'] == axis])
                   for axis in ('in_domain', 'unseen_system')}
        overall = seeded_summary(selected)
        record = {'backbone': model, 'method': 'ARIA', 'domains': domains, **overall}
        table.append(record)
        cells = [fmt(domains[axis][metric]) for axis in domains for metric in ('sr', 'hir')]
        cells.append('—' if overall['eb'] is None else f"{overall['eb']:.3f}")
        lines.append(f'| {model} / ARIA | ' + ' | '.join(cells) + f" | {overall['resolved']} / {overall['expected']} |")
        tex.extend([f'% {model}; training seeds {overall["training_seeds"]}',
                    r'\textbf{ARIA (Stage 1 + 2 + 3)} & ' + ' & '.join(v.replace('—', r'\textemdash') for v in cells) + r' \\'])
    figure = {'schema_version': 'aria-rq3-completion-figure04-v1', 'units': 'percent',
              'training_seeds': manifest['spec']['qwen_seeds'], 'domains': {}}
    for axis in ('in_domain', 'unseen_system'):
        subset = [r for r in rows if r['axis'] == axis and r['backbone'] == 'qwen']
        figure['domains'][axis] = transitions(subset, draws=manifest['spec']['bootstrap_draws'],
                                             seed=manifest['spec']['bootstrap_seed'])
    figure['data_status'] = 'measured' if all(g['complete'] for g in figure['domains'].values()) else 'pending'
    figure['formal_points'] = ({axis: d['points_equal_seed_weight'] for axis, d in figure['domains'].items()}
                              if figure['data_status'] == 'measured' else None)
    if evaluation_only and figure['data_status'] != 'measured':
        figure['data_status'] = 'unavailable_missing_untrained_variants'
        figure['reason'] = 'Evaluation-only scope; untrained variants are unavailable'
    paired = defaultdict(dict)
    for row in rows:
        if row['backbone'] == 'qwen':
            paired[(row['axis'], row['training_seed'], row['case_id'])][row['variant']] = {
                'correct': row['correct'], 'status': row['status'], 'raw_result': row['raw_result'],
                'control_run_id': row['control_run_id'], 'native_task_id': row['native_task_id'],
                'checkpoint_id': row['checkpoint_id']}
    atomic_write_jsonl(root / 'figure04/paired_tasks.jsonl', [
        {'axis': key[0], 'training_seed': key[1], 'case_id': key[2], 'variants': values}
        for key, values in paired.items()])
    atomic_write_json(root / 'figure04/data.json', figure)
    atomic_write_text(root / 'figure04/data.csv', csv_text([
        {'axis': axis, 'formal': d['complete'], **p} for axis, d in figure['domains'].items()
        for p in d['points_equal_seed_weight']]))
    atomic_write_json(root / 'table.json', table)
    atomic_write_text(root / 'table.csv', csv_text(table))
    atomic_write_text(root / 'table.tex', '\n'.join(tex) + '\n')
    diagnostics = []
    for arm in manifest['arms']:
        subset = [r for r in rows if r['backbone'] == arm['family'] and r['axis'] == arm['axis']
                  and r['training_seed'] == arm['training_seed'] and r['variant'] == arm['variant']]
        diagnostics.append({**arm, **summary(subset)})
    atomic_write_json(root / 'data/variant_diagnostics.json', diagnostics)
    progress = {'counts': dict(Counter(states.get(t['id'], {}).get('status', 'pending') for t in manifest['tasks'])),
        'planned_tasks': len(manifest['tasks']), 'main_table_complete': all(r['coverage_complete'] for r in table),
        'figure04_complete': figure['data_status'] == 'measured'}
    atomic_write_json(root / 'progress.json', progress)
    lines += ['', '## Training and pairing protocol', '',
        '- Qwen covers in-domain and unseen-system settings, seeds 42/43/44 and variants S1, S1+2, S1+3 and Full. Llama Full provides a seed-42 comparison.',
        '- Unseen-system training, rewards and calibration exclude Magentic-One. S1+3 starts from SFT with 16 Repair and 16 Gate updates; Full uses Stage 2 as reference.',
        '- GRPO acceptance requires native rewards, nonzero gradients and finite changed weights. Invalid updates do not count toward the budget.',
        '- Variants share task identities, controls, hosts, native code and execution budgets within each seed. Valid outcomes are retained; only infrastructure-invalid runs may be retried.',
        '- Figure 04 uses the complete four-variant scoreable intersection, including unchanged outcomes, then averages seeds equally. Incomplete coverage leaves formal_points null.',
        '- Per-task artifacts retain correctness, control outcomes, HIR reasons, model calls, gate decisions and execution ledgers. Unobserved total costs remain null.',
        '- views/, setup/, checkpoint_registry.json and raw/ retain data, checkpoint and execution provenance.', '',
        'Coverage: ' + json.dumps(progress, ensure_ascii=False), '']
    if evaluation_only:
        progress['available_evaluation_terminal'] = all(states.get(t['id'], {}).get('status')
            in {'resolved', 'infrastructure_invalid', 'unresolved'} for t in manifest['tasks'])
        progress['training_enabled'] = False
        progress['excluded_untrained_arms'] = len(manifest.get('excluded_arms', []))
        atomic_write_json(root / 'progress.json', progress)
        # Replace the superseded training plan rather than leave contradictory
        # promises in the report that is refreshed during evaluation.
        marker = lines.index('## Training and pairing protocol')
        lines = lines[:marker] + ['## Evaluation-only scope', '',
            '- Only accepted checkpoints are evaluated; further training is disabled.',
            '- Untrained variants remain unavailable. Formal four-variant results require all four variants.',
            '- Existing valid predictions are reused. Missing and infrastructure-invalid cases are not scored as incorrect.',
            '- Available-evaluation completion and full-design completion are tracked separately.', '',
            'Available checkpoints: ' + ', '.join(f"{a['family']}/{a['axis']}/{a['variant']}" for a in manifest['arms']), '',
            'Progress: ' + json.dumps(progress, ensure_ascii=False), '']
        lines[0] = '# RQ3: evaluation of available checkpoints'
        lines[2] = 'The queue evaluates accepted checkpoints and retains original outcomes and coverage.'
    atomic_write_text(root / 'README.md', '\n'.join(lines))
    return progress
