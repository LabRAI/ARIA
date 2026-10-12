"""One RQ1 outcome projection for tables, Figure 01 and Figure 02."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import statistics

import yaml

from aria.runtime_metrics import inspection_failure
from common.artifacts import atomic_write_json, atomic_write_jsonl, atomic_write_text
from scripts.aria.build_rq1_eval import BENCHMARKS, MODELS, SYSTEMS


def read_json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else default


def load_result(path, mode):
    p = Path(path)
    if not p.exists():
        return None
    if mode == 'posthoc':
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        if len(rows) != 1:
            raise ValueError('one source record per posthoc job required')
        return rows[0]
    return read_json(p)


def outcome(result, task):
    """Never turn transport failures or absent official verdicts into wrong answers."""
    if result is None:
        return {'status': 'infrastructure_invalid', 'reason': 'no terminal result artifact'}
    if task['mode'] == 'runtime':
        if result.get('status') != 'completed':
            return {'status': 'infrastructure_invalid', 'reason': result.get('error') or 'native run failed'}
        ev = result.get('evaluation') or {}
        if type(ev.get('correct')) is not bool:
            return {'status': 'unresolved', 'reason': ev.get('reason') or 'evaluator unresolved'}
        audits = (result.get('aria') or {}).get('model_interventions', [])
        if task['method'] == 'aria' and (not audits or any(inspection_failure(a) == 'infrastructure' for a in audits)):
            return {'status': 'infrastructure_invalid', 'reason': 'missing/failed runtime inspection'}
        malformed = [str(a.get('boundary', i)) for i, a in enumerate(audits) if inspection_failure(a) == 'malformed']
        calls = [call for audit in audits for call in audit.get('model_calls', [])]
        enacted = any(a.get('enacted') for a in audits)
        harms = ['malformed:' + boundary for boundary in malformed]
        if ev.get('constraint_violation') is True or ev.get('constraints_satisfied') is False:
            harms.append('explicit_constraint_violation')
        gates = [{'boundary': a.get('boundary'), 'enacted': bool(a.get('enacted')),
                  'gate': (a.get('intervention') or {}).get('gate'), 'error': a.get('error')}
                 for a in audits]
        executions = [{'id': task['id'], 'role': 'initial', 'count': 1,
                       'status': 'completed', 'artifact': result.get('output_dir')}]
        public_answer = str(result.get('final_output') or result.get('final_answer') or '')
        # For code tasks do not use evaluator-derived extracted answers in selection.
        if task['benchmark'] == 'swe-bench-lite':
            patch = result.get('workspace_patch') or result.get('patch')
            patch_file = Path(result.get('output_dir') or '/nonexistent') / 'model.patch'
            if patch_file.is_file():
                patch = patch_file.read_text()
            if isinstance(patch, str) and patch.strip():
                public_answer = patch
            else:
                public_answer = ''  # missing patches receive unique abstaining votes
    else:
        diagnostics = result.get('diagnostics') or {}
        if diagnostics.get('failure_category') == 'infrastructure':
            return {'status': 'infrastructure_invalid', 'reason': diagnostics.get('error')}
        records = result.get('executions') or []
        chosen = next((e for e in records if e['label'] == result.get('selected_execution_label')), records[-1] if records else None)
        malformed = result.get('status') == 'invalid' or diagnostics.get('structurally_valid') is False
        ev = (chosen or {}).get('evaluation') or {}
        if chosen and (chosen.get('status') in {'invalid', 'failed'} or ev.get('method') == 'static-validator'):
            if ev.get('failure_category') == 'method' or ev.get('method') == 'static-validator':
                malformed = True
            else:
                return {'status': 'infrastructure_invalid', 'reason': chosen.get('error') or ev.get('reason')}
        correct = ev.get('correct', ev.get('success'))
        if malformed:
            # A structurally invalid repair fails to recover the failed source, and is harmful.
            correct = False
        if type(correct) is not bool:
            return {'status': 'unresolved', 'reason': ev.get('reason') or 'repair evaluator unresolved'}
        ev = {**ev, 'correct': correct}
        if malformed:
            ev['score'] = 0
        harms = ['structurally_invalid'] if malformed else []
        if diagnostics.get('constraint_violation') or ev.get('constraint_violation') or ev.get('constraints_satisfied') is False:
            harms.append('explicit_constraint_violation')
        source_score = task['source_evaluation'].get('score')
        if isinstance(source_score, (int, float)) and isinstance(ev.get('score'), (int, float)) and ev['score'] < source_score:
            harms.append('source_score_reduced')
        calls = result.get('model_calls', [])
        enacted, gates, public_answer = bool(records), [], ''
        executions = [{'id': task['id'] + ':' + e['label'], 'role': 'repair_replay',
                       'count': e['execution_count'], 'status': e['status'], 'artifact': e.get('artifact_refs')}
                      for e in records]
    normalized = (public_answer if task['benchmark'] in {'humaneval', 'swe-bench-lite'}
                  else re.sub(r'\s+', ' ', public_answer).strip().casefold())
    score = ev.get('score')
    return {'status': 'resolved', 'correct': ev['correct'], 'score': float(ev['correct']) if score is None else score,
            'harm_reasons': sorted(set(harms)), 'enacted': enacted, 'gate_events': gates,
            'executions': sum(e['count'] for e in executions), 'execution_ledger': executions,
            'answer_key': hashlib.sha256(normalized.encode()).hexdigest() if normalized else 'missing:' + task['id'],
            'inspection_calls': len(calls),
            'inspection_tokens': sum((c.get('response_metadata', {}).get('usage') or {}).get('total_tokens', 0) or 0 for c in calls),
            'total_tokens': None, 'total_calls': None,
            'localization': (result.get('diagnostics') or {}).get('localization'),
            'attribution': result.get('attribution'),
            'cut_event_id': (result.get('intervention') or {}).get('cut_event_id')}


def select_runtime(candidates):
    """Only public answer keys participate; score/correct are intentionally inaccessible."""
    keys = [candidate['answer_key'] for candidate in candidates]
    counts = Counter(keys)
    return max(range(len(keys)), key=lambda i: (counts[keys[i]], -i))


def budget_case(tasks, states, cap, control=None):
    ordered = sorted((t for t in tasks if t['candidate'] <= cap), key=lambda t: t['candidate'])
    first = ordered[0]
    used, selected = [], None
    status, reason = 'resolved', None
    for task in ordered:
        state = states.get(task['id'], {})
        if state.get('status') != 'resolved':
            status = state.get('status', 'missing')
            status = status if status in {'unresolved', 'infrastructure_invalid'} else 'missing'
            reason = state.get('reason', 'candidate pending')
            break
        used.append((task, state))
        if first['mode'] == 'posthoc' and state['metrics']['correct']:
            selected = len(used) - 1
            break
    if status == 'resolved' and len(used) < cap and selected is None:
        status, reason = 'missing', 'budget not fully executed'
    if first['mode'] == 'runtime' and first['method'] != 'control' and (control or {}).get('status') != 'resolved':
        status, reason = (control or {}).get('status', 'missing'), 'paired control unresolved'
    if status == 'resolved' and selected is None:
        selected = select_runtime([s['metrics'] for _, s in used]) if first['mode'] == 'runtime' else len(used) - 1
    chosen = used[selected][1]['metrics'] if selected is not None else None
    harms = sorted({h for _, state in used for h in state['metrics']['harm_reasons']})
    if status == 'resolved' and first['mode'] == 'runtime' and first['method'] != 'control':
        cm = control['metrics']
        if chosen['enacted'] and chosen['score'] < cm['score']:
            harms.append('enacted_intervention_reduced_control_score')
    return {'schema_version': 'aria-rq1-budget-case-v1', 'case_id': first['case_id'],
        'mode': first['mode'], 'backbone': first['backbone'], 'mas': first['mas'], 'benchmark': first['benchmark'],
        'task_id': first['native_sample_id'], 'incident_id': first['incident_id'], 'budget_cap': cap,
        'status': status, 'invalid_reason': reason, 'correct': chosen['correct'] if status == 'resolved' else None,
        'harmful': bool(harms) if status == 'resolved' else None, 'harm_reasons': harms,
        'control_correct': ((control or {}).get('metrics') or {}).get('correct'),
        'execution_budget': sum(s['metrics']['executions'] for _, s in used) if status == 'resolved' else None,
        'source_inclusive_eb': (1 + sum(s['metrics']['executions'] for _, s in used)) if status == 'resolved' and first['mode'] == 'posthoc' else None,
        'selected_candidate_id': used[selected][0]['id'] if status == 'resolved' else None,
        'consumed_candidate_ids': [t['id'] for t, _ in used],
        'raw_artifacts': [s.get('result') for _, s in used],
        'wall_seconds': sum(s.get('wall_seconds', 0) or 0 for _, s in used) if used else None,
        'inspection_tokens': sum(s['metrics']['inspection_tokens'] for _, s in used),
        'inspection_calls': sum(s['metrics']['inspection_calls'] for _, s in used)}


def summary(rows):
    valid = [r for r in rows if r['status'] == 'resolved']
    complete = bool(rows) and len(valid) == len(rows)
    sr = sum(r['correct'] for r in valid) / len(valid) if valid else None
    hir = sum(r['harmful'] for r in valid) / len(valid) if valid else None
    eb = statistics.mean(r['execution_budget'] for r in valid) if valid else None
    walls = sorted(r['wall_seconds'] for r in valid if r['wall_seconds'] is not None)
    return {'expected': len(rows), 'resolved': len(valid), 'coverage_complete': complete,
            'status_counts': dict(Counter(r['status'] for r in rows)),
            'sr': sr if complete else None, 'hir': hir if complete else None, 'eb': eb if complete else None,
            'resolved_subset_sr': sr, 'resolved_subset_hir': hir, 'resolved_subset_eb': eb,
            'successes': sum(r['correct'] for r in valid), 'harms': sum(r['harmful'] for r in valid),
            'wall_median': statistics.median(walls) if walls else None,
            'wall_p95': walls[min(len(walls)-1, int(.95 * len(walls)))] if walls else None}


def csv_text(rows):
    if not rows:
        return ''
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
    writer.writeheader()
    writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows)
    return out.getvalue()


def write_report(manifest, states):
    root, spec = Path(manifest['spec']['output_root']), manifest['spec']
    groups = defaultdict(list)
    for task in manifest['tasks']:
        groups[(task['case_id'], task['mode'], task['backbone'])].append(task)
    cases = []
    for (_, mode, model), tasks in groups.items():
        for cap in (spec['budgets'] if model == 'qwen' else [1]):
            cases.append(budget_case(tasks, states, cap, states.get(tasks[0]['control_id'])))
    atomic_write_jsonl(root / 'data/cases.jsonl', cases)
    table, tex, lines = [], [], ['# RQ1 ARIA evaluation', '',
        'Coverage is reported below; `—` denotes unavailable results. Table rates are percentages; JSON/CSV rates are fractions.', '',
        f"Planned candidate executions: {len(manifest['tasks'])}; resolved: {sum(s.get('status') == 'resolved' for s in states.values())}.",
        '', '## Main results (budget 1)', '']
    for mode, axis, columns in (('posthoc', 'benchmark', BENCHMARKS), ('runtime', 'mas', SYSTEMS)):
        lines += [f'### {mode}', '', '| Backbone / Method | ' + ' | '.join(columns) + ' | SR | HIR | EB | Resolved / N |',
                  '| --- | ' + ' | '.join(['---:'] * (len(columns) + 4)) + ' |']
        for model in MODELS:
            rows = [r for r in cases if r['mode'] == mode and r['backbone'] == model and r['budget_cap'] == 1]
            overall = summary(rows)
            entries = {col: summary([r for r in rows if r[axis] == col]) for col in columns}
            record = {'mode': mode, 'backbone': model, 'method': 'ARIA', **overall,
                      **{col: entries[col]['sr'] for col in columns}, 'groups': entries}
            table.append(record)
            fmt = lambda v: '—' if v is None else f'{100*v:.2f}'
            vals = [fmt(entries[col]['sr']) for col in columns] + [fmt(overall['sr']), fmt(overall['hir']),
                    '—' if overall['eb'] is None else f"{overall['eb']:.3f}"]
            lines.append(f"| {model} / ARIA | " + ' | '.join(vals) + f" | {overall['resolved']} / {overall['expected']} |")
            tex += [f'% {mode}: {model}', r'\textbf{ARIA} & ' + ' & '.join(v.replace('—', r'\textemdash') for v in vals) + r' \\']
        lines.append('')
    figure01 = {'data_status': 'incomplete_design', 'units': 'fraction', 'points': [],
        'missing': ['full-data Qwen 1.5B, 3B, 14B checkpoints; RQ2 10% models are not substituted',
                    'Spark-to-Fire, SafeAgent, untuned references not requested'], 'training_seed': 42}
    for mas in ('openmanus', 'metagpt'):
        for model in ('qwen', 'llama', 'control'):
            rows = [r for r in cases if r['mode'] == 'runtime' and r['backbone'] == model and r['mas'] == mas and r['budget_cap'] == 1]
            figure01['points'].append({'mas': mas, 'backbone': model,
                'role': 'supplementary_cross_family' if model == 'llama' else 'paper_7b_point' if model == 'qwen' else 'no_intervention', **summary(rows)})
    figure02 = {'data_status': 'pending', 'units': 'fraction', 'points': [], 'protocol': manifest['protocol']}
    for mode in ('posthoc', 'runtime'):
        for cap in spec['budgets']:
            selected = [r for r in cases if r['mode'] == mode and r['backbone'] == 'qwen' and r['budget_cap'] == cap]
            figure02['points'].append({'mode': mode, 'backbone': 'qwen', 'budget_cap': cap, **summary(selected)})
    figure02['runtime_reference'] = summary([r for r in cases if r['backbone'] == 'control'])
    figure02['data_status'] = 'measured' if all(p['coverage_complete'] for p in figure02['points']) else 'pending'
    for name, figure in (('figure01', figure01), ('figure02', figure02)):
        atomic_write_json(root / name / 'data.json', figure)
        atomic_write_text(root / name / 'data.csv', csv_text(figure['points']))
    atomic_write_json(root / 'table.json', table)
    atomic_write_text(root / 'table.csv', csv_text(table))
    atomic_write_text(root / 'table.tex', '\n'.join(tex) + '\n')
    progress = {'planned_candidates': len(manifest['tasks']), 'states': dict(Counter(s.get('status') for s in states.values())),
                'main_table_complete': all(r['coverage_complete'] for r in table), 'figure02_complete': figure02['data_status'] == 'measured',
                'figure01_complete': False}
    atomic_write_json(root / 'progress.json', progress)
    lines += ['## Evaluation protocol', '',
        '- Results use the complete test pool and a local Qwen3-8B host. Historical trajectories retain their original host identities. This is a single-training-seed evaluation.',
        f"- Gate threshold: {spec['gate_threshold']}; identity calibration; temperature: {spec['temperature']}; execution seeds start at {spec['execution_seed']}.",
        '- Post-hoc repair stops at the first evaluator-passing candidate, or selects the last attempt. Runtime selection uses the most frequent normalized public answer, breaking ties by order, without evaluator access.',
        '- SR and HIR use the complete corresponding pool. HIR includes generation/constraint errors and selected interventions that reduce the control score. Infrastructure failures and unresolved evaluations remain missing.',
        '- EB counts actual executions. Source acquisition and shared control costs are reported separately. Missing total token/call telemetry remains null; inspection costs are partial measurements.',
        '- Formal metrics require complete coverage. resolved_subset_* values describe progress only. Figure 01 requires matching full-data checkpoints at every model size.', '',
        '## Artifacts', '',
        '- `table.{json,csv,tex}`: main results; `data/cases.jsonl`: per-task outcomes, selected candidates, execution IDs and coverage.',
        '- `figure01/data.{json,csv}` and `figure02/data.{json,csv}`: measured points, denominators and coverage.',
        '- `raw/` and `logs/`: native executions, model calls, evaluator results and cost ledgers.',
        '- `manifest.json`, `launch.yml`, `code/` and `configs/`: frozen protocol, task identities and execution configuration.', '',
        '## Resume / re-project', '', '```bash',
        f"uv run --no-sync python -m scripts.aria.run_rq1_eval -c {root / 'launch.yml'}",
        f"uv run --no-sync python -m scripts.aria.report_rq1_eval -c {root / 'launch.yml'}", '```', '',
        'The scheduler permits one writer per output directory. Reporting does not call models.', '']
    atomic_write_text(root / 'README.md', '\n'.join(lines))
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', type=Path, required=True)
    launch = yaml.safe_load(parser.parse_args().config.read_text())
    manifest = read_json(launch['manifest'])
    states = read_json(Path(manifest['spec']['output_root']) / 'state.json', {}).get('tasks', {})
    print(json.dumps(write_report(manifest, states)))


if __name__ == '__main__':
    main()
