"""Project native RQ2 artifacts once into table rows and paired Figure 03 data."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import io
import json
from pathlib import Path
import random

import yaml

from aria.figure_logging import validate_case
from aria.figure_analysis import conditional_recovery, runtime_summary
from aria.gate_scoring import chat_label_probability
from aria.runtime_metrics import inspection_failure
from common.artifacts import atomic_write_json, atomic_write_text
from scripts.aria.build_rq2_eval import MODELS, SIZES, arms
from scripts.aria.report_rq2_small import AXES, LABELS


def read_json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else default


def resolution(result, method):
    if result is None:
        return 'missing', 'no completed native artifact'
    if result.get('status') != 'completed':
        return 'infrastructure_invalid', result.get('error') or 'native execution failed'
    if type((result.get('evaluation') or {}).get('correct')) is not bool:
        return 'unresolved', (result.get('evaluation') or {}).get('reason') or 'evaluator did not resolve'
    if method != 'control':
        audits = (result.get('aria') or {}).get('model_interventions', [])
        if not audits:
            return 'infrastructure_invalid', 'missing live inspection evidence'
        if any(inspection_failure(a) == 'infrastructure' for a in audits):
            return 'infrastructure_invalid', 'inspection endpoint/runtime failure'
    return 'resolved', None


def gate_events(audits, threshold):
    events = []
    for index, audit in enumerate(audits):
        gate = (audit.get('intervention') or {}).get('gate') or {}
        probability = gate.get('probability')
        if type(probability) not in (float, int):
            continue  # Failed gates have no numeric probability; the full audit retains the error.
        raw = None
        for call in audit.get('model_calls', []):
            raw = chat_label_probability((call.get('response_metadata') or {}).get('gate_logprobs'))
            if raw is not None:
                break
        events.append({'event_id': str(audit.get('boundary', index + 1)),
                       'raw_probability': probability if raw is None else raw,
                       'calibrated_probability': probability, 'threshold': threshold,
                       'triggered': probability >= threshold, 'enacted': bool(audit.get('enacted')),
                       'role': audit.get('role'), 'decision': audit.get('intervention')})
    return events


def case_row(manifest, task, result, control, state, identity):
    status, reason = resolution(result, task['method'])
    cstatus, creason = resolution(control, 'control')
    if cstatus != 'resolved':
        status, reason = cstatus, 'paired control: ' + str(creason)
    audits = ((result or {}).get('aria') or {}).get('model_interventions', [])
    malformed = [str(a.get('boundary', i + 1)) for i, a in enumerate(audits) if inspection_failure(a) == 'malformed']
    enacted = any(a.get('enacted') for a in audits)
    ev, cev = (result or {}).get('evaluation') or {}, (control or {}).get('evaluation') or {}
    score = ev.get('score') if ev.get('score') is not None else ev.get('correct')
    cscore = cev.get('score') if cev.get('score') is not None else cev.get('correct')
    harms = ['malformed_prediction:' + b for b in malformed]
    if status == 'resolved' and enacted and float(score) < float(cscore):
        harms.append('enacted_intervention_reduced_benchmark_score')
    method, model = task['method'], task['backbone']
    checkpoint_hash = ('not-applicable' if method == 'control' else
                       identity['sha256'] if method == 'aria' and identity else
                       'pending-qualified-full' if method == 'aria' else manifest['services'][model]['revision'])
    row = {'schema_version': 'aria-figure-case-v2', 'run_id': task['id'], 'pool_id': task['pool_id'],
           'task_id': task['sample_id'], 'benchmark': task['benchmark'], 'mas': task['mas'], 'axis': task['axis'],
           'host_config_sha256': task['host_config_sha256'], 'host_model': manifest['services']['host']['model'],
           'execution_seed': 42, 'training_seed': 42, 'backbone': model, 'checkpoint_sha256': checkpoint_hash,
           'variant': 'Full' if method == 'aria' else method, 'method': method, 'mode': 'runtime',
           'gate_threshold': .60 if method in ('aria', 'untuned') else None,
           'budget_setting_id': 'one_attempt', 'budget_cap': 1, 'protocol_sha256': manifest['protocol_sha256'],
           'training_view_sha256': manifest['training_view_sha256'] if method == 'aria' else 'not-applicable',
           'calibration_sha256': manifest['calibration_sha256'], 'status': status,
           'correct': ev.get('correct') if status == 'resolved' else None,
           'control_correct': cev.get('correct'), 'control_run_id': task['control_id'],
           'score': score if status == 'resolved' else None, 'control_score': cscore,
           'harmful': bool(harms) if status == 'resolved' else None, 'harm_reasons': harms,
           'triggered': enacted, 'gate_events': gate_events(audits, .60),
           'constraint_violations': [{'kind': 'malformed_prediction', 'boundary': b} for b in malformed],
           'executions': None, 'tokens': None, 'llm_calls': None, 'wall_seconds': None,
           'execution_ledger': [], 'telemetry_status': 'missing',
           'artifacts': {'result': state.get('result'), 'control_result': (control or {}).get('output_dir'),
                         'native_trace': (result or {}).get('native_trace_path'),
                         'attempt_history': state.get('attempts', [])},
           'invalid_reason': reason, 'unintervened_discordance': bool(status == 'resolved' and not enacted and ev['correct'] != cev['correct']),
           'measured_process_wall_seconds': state.get('wall_seconds'),
           'inspection_call_count': sum(len(a.get('model_calls', [])) for a in audits),
           'inspection_tokens': sum((c.get('response_metadata', {}).get('usage') or {}).get('total_tokens', 0)
                                    for a in audits for c in a.get('model_calls', []))}
    # Host call coverage is not available uniformly in AG2 and LangGraph. Keep the
    # measured partial costs separate; never label inspection-only tokens as total.
    if method == 'aria' and status == 'resolved' and identity is None:
        raise ValueError('unqualified checkpoint must never produce a resolved ARIA row')
    validate_case(row)
    return row


def clustered_interval(rows, value, seed, replicates):
    clusters = defaultdict(list)
    for row in rows:
        clusters[(row['benchmark'], row['task_id'])].append(value(row))
    groups = list(clusters.values())
    if not groups:
        return None
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        sample = rng.choices(groups, k=len(groups))
        draws.append(sum(map(sum, sample)) / sum(map(len, sample)))
    draws.sort()
    return [draws[int(.025 * replicates)], draws[min(replicates - 1, int(.975 * replicates))]]


def figure_data(rows, manifest):
    result = {'schema_version': 'aria-rq2-figure03-v1', 'data_status': 'pending', 'units': 'fraction',
              'model_order': list(SIZES), 'axes': {}, 'pools': {},
              'definition': 'ARIA passes / shared control failures, after intersecting four Qwen resolved sets within each pool',
              'uncertainty': 'task-cluster bootstrap conditional on one fixed training seed; no training-seed uncertainty'}
    common_rows = defaultdict(list)
    for pool in manifest['pools']:
        selected = [r for r in rows if r['pool_id'] == pool['name'] and r['method'] == 'aria' and r['backbone'] in SIZES]
        stats = conditional_recovery(selected, SIZES)
        result['pools'][pool['name']] = stats
        grouped = defaultdict(dict)
        for row in selected:
            grouped[row['task_id']][row['backbone']] = row
        for group in grouped.values():
            if all(m in group and group[m]['status'] == 'resolved' for m in SIZES):
                common_rows[pool['axis']].extend(group.values())
    for axis in AXES:
        axis_pools = [p for p in manifest['pools'] if p['axis'] == axis]
        complete = all(result['pools'][p['name']]['coverage']['jointly_resolved'] == len(p['sample_ids']) for p in axis_pools)
        values = {}
        for model in SIZES:
            cases = [r for r in common_rows[axis] if r['backbone'] == model]
            failed = [r for r in cases if r['control_correct'] is False]
            recovered = sum(r['correct'] for r in failed)
            values[model] = {'jointly_resolved': len(cases), 'control_failures': len(failed), 'recovered': recovered,
                'rate': recovered / len(failed) if complete and failed else None,
                'resolved_subset_rate': recovered / len(failed) if failed else None,
                'task_cluster_ci95': clustered_interval(failed, lambda r: int(r['correct']),
                    manifest['spec']['bootstrap_seed'], manifest['spec']['bootstrap_replicates']) if complete else None,
                'denominator_task_ids': [[r['pool_id'], r['task_id']] for r in failed]}
        result['axes'][axis] = {'complete': complete, 'models': values}
    if all(v['complete'] for v in result['axes'].values()):
        result['data_status'] = 'measured'
    return result


def csv_text(rows):
    if not rows:
        return ''
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def plot_figure(data, root):
    if data['data_status'] != 'measured':
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman'], 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8), sharex=True, sharey=True)
    for ax, axis, label in zip(axes, ('unseen_system', 'unseen_benchmark'), ('(a)', '(b)')):
        for y, model in enumerate(SIZES):
            ref = data['axes']['reference']['models'][model]['rate']
            ood = data['axes'][axis]['models'][model]['rate']
            if ref is not None and ood is not None:
                ax.plot([100 * ref, 100 * ood], [y, y], color='#bbbbbb', lw=1)
            if ref is not None:
                ax.scatter(100 * ref, y, color='#777777', marker='o', label='In-domain' if y == 0 else None)
            if ood is not None:
                ax.scatter(100 * ood, y, color='#2878a0', marker='D', label=AXES[axis] if y == 0 else None)
        ax.set_xlabel('Conditional recovery (%)')
        ax.text(.02, .96, label, transform=ax.transAxes, va='top')
        ax.legend(frameon=False, fontsize=8, loc='lower right')
        ax.grid(axis='x', alpha=.15)
    axes[0].set_yticks(range(len(SIZES)), [LABELS[m] for m in SIZES])
    axes[0].invert_yaxis()
    fig.tight_layout()
    fig.savefig(root / 'figure03/fig03_generalization_by_model_size.pdf', metadata={'Subject': 'Measured RQ2; single training seed 42'})
    plt.close(fig)


def write_report(manifest, states):
    root = Path(manifest['spec']['output_root'])
    identities = read_json(root / 'setup/checkpoint_identities.json', manifest['checkpoint_identities'])
    results = {key: read_json(value['result']) for key, value in states.items() if value.get('result')}
    rows = [case_row(manifest, task, results.get(task['id']), results.get(task['control_id']),
                     states.get(task['id'], {}), identities.get(task['backbone'])) for task in manifest['tasks']]
    table, flat = [], []
    labels = {'control': 'No intervention', 'spark_to_fire': 'Spark to Fire', 'safeagent': 'SafeAgent', 'untuned': 'Untuned', 'aria': 'ARIA'}
    for method, model in arms():
        cells = {}
        for axis in AXES:
            selected = [r for r in rows if (r['method'], r['backbone'], r['axis']) == (method, model, axis)]
            summary = runtime_summary(selected)
            valid = [r for r in selected if r['status'] == 'resolved']
            summary.update(planned=len(selected), resolved=len(valid),
                resolved_subset_sr=sum(r['correct'] for r in valid) / len(valid) if valid else None,
                resolved_subset_hir=sum(r['harmful'] for r in valid) / len(valid) if valid else None,
                unintervened_discordance=sum(r['unintervened_discordance'] for r in valid))
            if summary['sr'] is not None:
                summary['delta_sr_task_cluster_ci95'] = clustered_interval(valid,
                    lambda r: int(r['correct']) - int(r['control_correct']),
                    manifest['spec']['bootstrap_seed'], manifest['spec']['bootstrap_replicates'])
            cells[axis] = summary
            flat.append({'method': labels[method], 'backbone': LABELS.get(model, model), 'axis': axis,
                'sr_percent': None if summary['sr'] is None else 100 * summary['sr'],
                'hir_percent': None if summary['hir'] is None else 100 * summary['hir'],
                'planned': len(selected), **{k: summary['coverage'].get(k, 0) for k in ('resolved', 'missing', 'infrastructure_invalid', 'unresolved')},
                'resolved_subset_sr': summary['resolved_subset_sr'], 'resolved_subset_hir': summary['resolved_subset_hir']})
        table.append({'method': labels[method], 'backbone': LABELS.get(model, model), 'cells': cells})
    fig = figure_data(rows, manifest)
    complete = all(c['sr'] is not None for row in table for c in row['cells'].values())
    report = {'status': 'complete' if complete else 'running_or_pending', 'planned_executions': len(rows),
              'coverage': dict(Counter(r['status'] for r in rows)), 'table': table,
              'units': 'fractions in JSON, percentages in Markdown/TeX/table.csv', 'training_seed': 42}
    (root / 'data').mkdir(exist_ok=True)
    (root / 'figure03').mkdir(exist_ok=True)
    atomic_write_text(root / 'data/cases.jsonl', ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows))
    atomic_write_json(root / 'table.json', report)
    atomic_write_text(root / 'table.csv', csv_text(flat))
    atomic_write_json(root / 'figure03/data.json', fig)
    figure_csv = [{'axis': axis, 'backbone': model, **{k: v for k, v in values.items() if k not in ('denominator_task_ids', 'task_cluster_ci95')}}
                  for axis, section in fig['axes'].items() for model, values in section['models'].items()]
    atomic_write_text(root / 'figure03/data.csv', csv_text(figure_csv))
    # A completed figure is immutable while unrelated Llama/baseline panels finish.
    if fig['data_status'] == 'measured' and not (root / 'figure03/fig03_generalization_by_model_size.pdf').exists():
        plot_figure(fig, root)
    lines = ['# RQ2 evaluation', '', '**Status: ' + ('complete' if complete else 'in progress; formal results require complete cells') + '**', '',
             'Cells show **SR / HIR (%)** using the same frozen task pools and paired controls.', '',
             '| Method | Backbone | In-domain | Unseen benchmark | Unseen MAS |', '|---|---|---:|---:|---:|']
    tex = [r'\begin{tabular}{llccc}', r'\toprule', r'Method & Backbone & In-domain & Unseen benchmark & Unseen MAS \\', r'\midrule']
    for row in table:
        cells = [f"{100*c['sr']:.2f} / {100*c['hir']:.2f}" if c['sr'] is not None else f"Pending ({c['resolved']}/{c['planned']})" for c in row['cells'].values()]
        lines.append('| ' + ' | '.join([row['method'], row['backbone'], *cells]) + ' |')
        tex.append(' & '.join([row['method'], row['backbone'], *[v if not v.startswith('Pending') else '--' for v in cells]]) + r' \\')
    tex.extend([r'\bottomrule', r'\end{tabular}'])
    atomic_write_text(root / 'table.tex', '\n'.join(tex) + '\n')
    lines += ['', '## Protocol and data', '',
        '- Models use a 10% task-group training subset and training/execution seed 42. ARIA checkpoints must pass all five training-stage acceptance checks.',
        '- Pools: AG2/GSM-Plus (100), AG2/BBH (108), AG2/ARC-Challenge (100), and LangGraph/GSM-Plus (100). The host is Qwen3-8B with thinking disabled. BBH, ARC and LangGraph are held out from fitting.',
        '- Gate threshold 0.60; identity calibration; host context/output 32768/2048; predictor input/output 6144/2048; at most one applied text replacement.',
        '- SR and HIR use full-pool denominators. Structured-generation errors and score-reducing interventions count as harmful. Unintervened arm disagreements are reported separately.',
        '- Missing, infrastructure-invalid and unresolved cases remain distinct. Formal cells require complete coverage; resolved_subset_* values are progress diagnostics.',
        '- Figure 03 uses tasks scoreable by all four Qwen sizes, restricted to shared-control failures. BBH/ARC are micro-averaged. Task-cluster bootstrap intervals preserve model pairing and describe task-sampling uncertainty at a fixed training seed.',
        '- Native artifacts retain model calls, gate probabilities, replacements and evaluator evidence. Partial inspection/process costs are labeled; unavailable totals remain null.', '',
        '## Artifacts', '',
        '- `table.{tex,csv,json}`: results and coverage; `data/cases.jsonl`: paired task records, including missing positions.',
        '- `raw/<run-id>/attempt-N/`: native execution and evaluator evidence. Only infrastructure-invalid attempts are retried; the first valid result is retained.',
        '- `figure03/data.{json,csv}`: numerators, denominators, common-pool coverage and confidence intervals.',
        '- `manifest.json`, `setup/`, `code/`, `state.json`, `logs/` and `services/`: protocol, provenance and execution state.', '',
        '## Resume / re-project', '', '```bash',
        f"PYTHONPATH={root / 'code/src'}:{root / 'code'} UV_PROJECT_ENVIRONMENT=.venv-rq2-generalization uv run --no-sync python {root / 'code/scripts/aria/run_rq2_eval.py'} -c {root / 'launch.yml'}",
        f"uv run --no-sync python -m scripts.aria.report_rq2_eval -c {root / 'launch.yml'}", '```', '',
        'The scheduler permits one writer per output directory. Reporting does not call models.', '']
    atomic_write_text(root / 'README.md', '\n'.join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', type=Path, required=True)
    launch = yaml.safe_load(parser.parse_args().config.read_text())
    manifest = read_json(launch['manifest'])
    states = read_json(Path(manifest['spec']['output_root']) / 'state.json', {}).get('tasks', {})
    report = write_report(manifest, states)
    print(json.dumps({k: v for k, v in report.items() if k != 'table'}))


if __name__ == '__main__':
    main()
