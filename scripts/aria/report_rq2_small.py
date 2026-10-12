"""Write the reduced RQ2 table from task-level measurements, with fixed coverage."""
import argparse
import csv
import io
import json
from pathlib import Path

import yaml

from aria.figure_logging import validate_case
from aria.figure_analysis import runtime_summary
from common.artifacts import atomic_write_json, atomic_write_text
from scripts.aria.verify_training_checkpoint import file_sha256

LABELS = {'qwen25-1.5b': 'Qwen2.5-1.5B', 'qwen25-3b': 'Qwen2.5-3B',
          'qwen25-7b': 'Qwen2.5-7B', 'qwen25-14b': 'Qwen2.5-14B', 'llama31-8b': 'Llama-3.1-8B'}
AXES = {'reference': 'In-domain', 'unseen_benchmark': 'Unseen benchmark', 'unseen_system': 'Unseen MAS'}


def qualified(manifest, registry, model, seed):
    jobs = [j for j in manifest['jobs'] if j['model_id'] == model and j['seed'] == seed]
    if len(jobs) != 5:
        return False
    for job in jobs:
        record = registry['checkpoints'].get(job['id'])
        if not record or record.get('updates') != job['expected_updates']:
            return False
        path = Path(job['checkpoint']) / 'adapter_model.safetensors'
        if not path.is_file() or file_sha256(path) != record['stages'][0]['adapter_sha256']:
            return False
    return True


def build_report(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    pool_path = root / 'pool/generalization/manifest.json'
    if not pool_path.exists():
        raise FileNotFoundError('freeze the pool with main.py -c configs/pool.yml first')
    pools = json.loads(pool_path.read_text())['pools']
    registry = json.loads((root / 'checkpoint_registry.json').read_text())
    spec = manifest['spec']
    records = []
    qualifications = {}
    identities = {}
    for path in sorted((root / 'eval_logs').rglob('cases.jsonl')):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                validate_case(row)
                if row['schema_version'] != 'aria-figure-case-v2':
                    raise ValueError('reduced RQ2 requires v2 provenance')
                records.append(row)
    ordered = [('No intervention', 'n/a', 'control'), ('Spark to Fire', 'qwen25-7b', 'spark_to_fire'),
               ('SafeAgent', 'qwen25-7b', 'safeagent')]
    for model in spec['models']:
        ordered.extend([('Untuned', model, 'untuned'), ('ARIA', model, 'aria')])
    table = []
    for label, model, method in ordered:
        cells = {}
        for axis in AXES:
            expected = {(p['name'], task, seed): p for p in pools if p['axis'] == axis
                        for task in p['sample_ids'] for seed in spec['seeds']}
            index = {}
            for row in records:
                if row['method'] != method or row['backbone'] != model or row['axis'] != axis:
                    continue
                if row['mode'] != 'runtime' or row['budget_setting_id'] != 'one_attempt':
                    continue
                if method == 'aria' and row['variant'] != 'Full':
                    continue
                if row['gate_threshold'] != 0.60 and method == 'aria':
                    continue
                key = row['pool_id'], row['task_id'], row['training_seed']
                if key not in expected:
                    raise ValueError('case is not in the frozen evaluation pool')
                pool = expected[key]
                if (row['mas'], row['benchmark'], row['host_model']) != (pool['mas'], pool['benchmark'], pool['host_model']):
                    raise ValueError('case host/system/benchmark differs from frozen cell')
                if row['execution_seed'] != row['training_seed']:
                    raise ValueError('execution seed differs from preregistered paired seed')
                if key in index:
                    raise ValueError('duplicate task measurement; resolve retry lineage first')
                identity = (row['host_config_sha256'], row['protocol_sha256'], row['control_run_id'])
                if key in identities and identities[key] != identity:
                    raise ValueError('methods must share host/protocol and paired control')
                identities[key] = identity
                control_key = (*key, 'outcome')
                if row['status'] == 'resolved':
                    outcome = row['control_correct'], row['control_score']
                    if control_key in identities and identities[control_key] != outcome:
                        raise ValueError('methods disagree on the paired control outcome')
                    identities[control_key] = outcome
                if method == 'aria':
                    qkey = model, row['training_seed']
                    if qkey not in qualifications:
                        qualifications[qkey] = qualified(manifest, registry, *qkey)
                    if not qualifications[qkey]:
                        raise ValueError('unqualified/partial checkpoint cannot enter ARIA table')
                    terminal = f"{model}/seed-{row['training_seed']}/grpo_gate"
                    if row['checkpoint_sha256'] != registry['checkpoints'][terminal]['stages'][0]['adapter_sha256']:
                        raise ValueError('ARIA case did not use the registered Full checkpoint')
                    view_hash = manifest['frozen_sha256'][str(root / 'views/train/sft.jsonl')]
                    if row['training_view_sha256'] != view_hash:
                        raise ValueError('ARIA case belongs to a different training subset')
                index[key] = row
            summary = runtime_summary(list(index.values()))
            summary.update(planned=len(expected), missing=len(expected) - len(index))
            if summary['missing']:
                summary.update(sr=None, hir=None, eb=None)
            cells[axis] = summary
        table.append({'method': label, 'backbone': LABELS.get(model, model), 'cells': cells})
    return {'schema_version': 'aria-rq2-small-report-v1', 'rows': table,
            'training_seeds': spec['seeds'], 'training_chains': len(spec['models']) * len(spec['seeds']),
            'status': 'complete' if all(c['sr'] is not None for row in table for c in row['cells'].values()) else 'evaluation_pending',
            'training_counts': manifest['views']['counts'], 'task_sampling': manifest['views']['sampling'],
            'pool_manifest': str(pool_path)}


def write_report(root, destination):
    report = build_report(root)
    lines = ['# RQ2: scaling and generalization with a 10% training budget', '',
             '**Status: ' + ('evaluation complete.' if report['status'] == 'complete' else 'configuration and data frozen; evaluation incomplete.') + '**', '',
             'Cells show SR / HIR (%); incomplete, invalid and unresolved results remain missing.', '',
             '| Method | Backbone | In-domain | Unseen benchmark | Unseen MAS |', '|---|---|---:|---:|---:|']
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerow(['method', 'backbone', 'axis', 'sr', 'hir', 'eb', 'planned', 'resolved', 'missing', 'infrastructure_invalid', 'unresolved'])
    tex = [r'\begin{tabular}{llccc}', r'\hline', r'Method & Backbone & In-domain & Unseen benchmark & Unseen MAS \\', r'\hline']
    for row in report['rows']:
        cells, tex_cells = [], []
        for axis, value in row['cells'].items():
            resolved = value['coverage'].get('resolved', 0)
            cell = f"{100*value['sr']:.2f} / {100*value['hir']:.2f}" if value['sr'] is not None else f"Pending ({resolved}/{value['planned']})"
            cells.append(cell)
            tex_cells.append(cell if value['sr'] is not None else r'--')
            writer.writerow([row['method'], row['backbone'], axis, value['sr'], value['hir'], value['eb'], value['planned'], resolved,
                             value['missing'] + value['coverage'].get('missing', 0), value['coverage'].get('infrastructure_invalid', 0), value['coverage'].get('unresolved', 0)])
        lines.append('| ' + ' | '.join([row['method'], row['backbone'], *cells]) + ' |')
        tex.append(' & '.join([row['method'], row['backbone'], *tex_cells]) + r' \\')
    tex.extend([r'\hline', r'\end{tabular}'])
    counts = report['training_counts']
    sampling = report['task_sampling']
    lines.extend(['', f"Task groups: {sampling['selected_task_groups']} / {sampling['source_task_groups']}; SFT: {counts['sft']['rq2_rows']} / {counts['sft']['rq1_rows']}; DPO: {counts['dpo']['rq2_rows']} / {counts['dpo']['rq1_rows']}.", '',
        f"Training seeds: {'/'.join(map(str, report['training_seeds']))}; training chains: {report['training_chains']}. Each chain retains all five stage checkpoints.", '',
        'A single training seed does not establish stability across training seeds.', '',
        'Pools use matched controls: AG2/GSM-Plus (100), AG2/BBH (108), AG2/ARC-Challenge (100), and LangGraph/GSM-Plus (100), with Qwen3-8B as host.', '',
        'Held-out benchmarks and systems are excluded from training, validation and native rewards. Public benchmark exposure during backbone pretraining is not assessed.', '',
        'The RQ2 7B checkpoint uses 10% of training data. Threshold selection uses in-domain validation; the default evaluation threshold is 0.60.', '',
        f'Checkpoints and logs: `{root}`.', ''])
    atomic_write_json(root / 'RQ2.json', report)
    atomic_write_text(destination, '\n'.join(lines))
    atomic_write_text(destination.with_suffix('.csv'), csv_buffer.getvalue())
    atomic_write_text(destination.with_suffix('.tex'), '\n'.join(tex) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', type=Path, required=True)
    spec = yaml.safe_load(parser.parse_args().config.read_text())
    report = write_report(Path(spec['output_root']).resolve(), Path(spec['report_path']))
    print(json.dumps({'status': report['status'], 'report': spec['report_path']}))


if __name__ == '__main__':
    main()
