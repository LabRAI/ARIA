"""Freeze a task-grouped tenth of the formal RQ1 recipe; emit YAML jobs only."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path

import yaml

from aria.split_audit import rows_from_path, task_keys
from aria.figure_logging import REQUIRED, V2_REQUIRED
from aria.gate_preflight import gate_preflight
from common.artifacts import atomic_write_json, atomic_write_jsonl, atomic_write_text
from masfm.config import load_config
from scripts.aria.build_dated_training import set_option
from scripts.aria.build_rq23_study import cell, sha, write_yaml

STAGES = ('sft', 'dpo', 'transfer_diagnosis', 'grpo_repair', 'grpo_gate')
AXES = ('reference', 'unseen_benchmark', 'unseen_system')


def select_tasks(rows, fraction, seed):
    """Union ID/text aliases before sampling; all objectives/cells stay together."""
    if not 0 < fraction <= 1:
        raise ValueError('training_fraction must be in (0, 1]')
    parent = {}

    def find(key):
        parent.setdefault(key, key)
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for row in rows:
        keys = sorted(task_keys(row))
        root = find(keys[0])
        for key in keys[1:]:
            other = find(key)
            parent[other] = root
    groups = defaultdict(set)
    for key in sorted(parent):
        groups[find(key)].add(key)
    ordered = sorted(groups.values(), key=lambda g: hashlib.sha256(
        f'{seed}:{min(g)}'.encode()).hexdigest())
    count = max(1, math.floor(len(ordered) * fraction)) if ordered else 0
    ranks = {key: i for i, group in enumerate(ordered) for key in group}
    supports = defaultdict(set)
    for row in rows:
        if row.get('task') == 'deployment_gate':
            label = float(row['target']['gate_probability'])
            if label not in (0, 1):
                raise ValueError('gate coverage sampling requires binary training labels')
            supports[(cell(row).key(), int(label))].add(ranks[min(task_keys(row))])
    chosen = set()
    for key in sorted(supports):
        if not (supports[key] & chosen):
            chosen.add(min(supports[key]))
    if len(chosen) > count:
        raise ValueError('task budget too small to retain source-cell gate label support')
    for index in range(len(ordered)):
        if len(chosen) == count:
            break
        chosen.add(index)
    selected = set().union(*(ordered[i] for i in chosen)) if count else set()
    return selected, {'source_task_groups': len(ordered), 'selected_task_groups': count,
                      'realized_task_fraction': count / len(ordered) if ordered else 0,
                      'gate_cell_label_strata': len(supports),
                      'selected_aliases': sorted(selected)}


def reject_external(rows, benchmarks, systems):
    for row in rows:
        source = cell(row)
        if 'unknown' in (source.mas_id, source.benchmark_id, source.host_model_id):
            raise ValueError('unresolved fitting source-cell provenance')
        if source.benchmark_id in benchmarks or source.mas_id in systems:
            raise ValueError(f'external evaluation cell leaked into fitting: {source.key()}')


def prepare_views(spec, root):
    source = Path(spec['rq1_source'])
    raw = {name: list(rows_from_path(source / f'{name}.jsonl')) for name in ('sft', 'dpo')}
    validation = list(rows_from_path(Path(spec['validation'])))
    holdout = set()
    for path in (spec['validation'], spec['test']):
        for row in rows_from_path(Path(path)):
            holdout.update(task_keys(row))
    for rows in raw.values():
        if any(task_keys(row) & holdout for row in rows):
            raise ValueError('formal RQ1 source overlaps validation/test; repair the source first')
    all_rows = raw['sft'] + raw['dpo']
    selected, sampling = select_tasks(all_rows, spec['training_fraction'], spec['data_seed'])
    views, counts = {}, {}
    fit = all_rows + validation  # audit even unselected RQ1 records, not only the small view
    for name, rows in raw.items():
        kept = [r for r in rows if task_keys(r) & selected]
        if not kept:
            raise ValueError(f'empty sampled {name} view')
        path = root / 'views/train' / f'{name}.jsonl'
        atomic_write_jsonl(path, kept)
        views[name] = str(path)
        counts[name] = {'rq1_rows': len(rows), 'rq2_rows': len(kept),
                        'row_fraction': len(kept) / len(rows),
                        'tasks': dict(Counter(r['task'] for r in kept)),
                        'cells': dict(Counter(cell(r).key() for r in kept))}
    # Whole task groups preserve preference -> SFT lineage, including all trials.
    patches = {(r['incident_id'], json.dumps(r['input'], sort_keys=True), json.dumps(r['target'], sort_keys=True))
               for r in raw['sft'] if r['task'] == 'structured_repair' and task_keys(r) & selected}
    for r in raw['dpo']:
        if task_keys(r) & selected and (r['incident_id'], json.dumps(r['input'], sort_keys=True),
                                      json.dumps(r['chosen'], sort_keys=True)) not in patches:
            raise ValueError('DPO chosen patch has no identical sampled SFT target')
    for name, source_path in spec['environments'].items():
        rows = list(rows_from_path(Path(source_path)))
        fit.extend(rows)
        kept = [r for r in rows if task_keys(r) & selected and not task_keys(r) & holdout]
        if not kept:
            raise ValueError(f'empty sampled native {name} environment')
        path = root / 'views' / f'{name}.jsonl'
        atomic_write_jsonl(path, kept)
        views[name] = str(path)
        counts[name] = {'rq1_rows': len(rows), 'rq2_rows': len(kept)}
    reject_external(fit, spec['unseen_benchmarks'], spec['unseen_systems'])
    gates = [r for r in validation if r.get('task') == 'deployment_gate']
    if not gates:
        raise ValueError('empty in-domain gate validation')
    atomic_write_jsonl(root / 'views/validation_gate.jsonl', gates)
    views['validation_gate'] = str(root / 'views/validation_gate.jsonl')
    gate_audit = gate_preflight(Path(views['sft']), Path(views['validation_gate']))
    # Generalization runner accepts canonical metadata; include native reward provenance.
    fitting = [{**r, 'metadata': {**r.get('metadata', {}),
                'mas_id': cell(r).mas_id, 'benchmark_id': cell(r).benchmark_id,
                'host_model_id': cell(r).host_model_id}} for r in fit]
    atomic_write_jsonl(root / 'views/fitting_audit.jsonl', fitting)
    manifest = {'schema_version': 'aria-rq2-small-view-v1', 'dataset_schema_version': 'aria-training-ready-v3',
                'sampling': sampling, 'counts': counts, 'views': views,
                'fitting_hosts': sorted({cell(r).host_model_id for r in fit}),
                'external_audit': 'passed: train, validation, native rewards; no external fitting',
                'gate_preflight': gate_audit,
                'source_sha256': {str(p): sha(p) for p in [source / 'sft.jsonl', source / 'dpo.jsonl',
                                  *[Path(v) for v in spec['environments'].values()]]}}
    atomic_write_json(root / 'views/train/manifest.json', manifest)
    atomic_write_json(root / 'views/manifest.json', manifest)
    return manifest


def build_jobs(spec, root, view):
    jobs = []
    for model_id, model in spec['models'].items():
        for seed in spec['seeds']:
            previous, stage1, stage2 = model, None, None
            for stage in STAGES:
                template = Path(spec['templates'][stage])
                config = yaml.safe_load(template.read_text())
                post = config['aria']['post_training']
                name = f'{model_id}/seed-{seed}/{stage}'
                output = root / 'training' / name
                post.update(base_model=previous, source_path=str(root / 'views/train'), output_dir=str(output),
                            stages=[stage], random_seed=seed, resume=True,
                            gate_validation_path=view['views']['validation_gate'],
                            evaluation_holdout_paths=[str(Path(spec[k]).resolve()) for k in ('validation', 'test')])
                command = post['backend']['command']
                command[:] = [f"CUDA_VISIBLE_DEVICES={spec['training_gpu']}" if str(a).startswith('CUDA_VISIBLE_DEVICES=')
                              else f"UV_PROJECT_ENVIRONMENT={spec['training_environment']}" if str(a).startswith('UV_PROJECT_ENVIRONMENT=')
                              else a for a in command]
                for flag, value in {'seed': seed, 'standalone-checkpoint': 'false'}.items():
                    set_option(command, flag, value)
                set_option(command, 'reference-model', stage1 or model)
                budget_flag = 'grpo-optimizer-steps' if stage.startswith('grpo') else 'optimizer-steps'
                reference_budget = int(command[command.index('--' + budget_flag) + 1])
                if reference_budget <= 0:
                    raise ValueError('RQ2 reference must declare measured fixed update budgets')
                budget = max(1, math.ceil(reference_budget * spec['update_fraction']))
                set_option(command, budget_flag, budget)
                if stage.startswith('grpo'):
                    post['environment_path'] = view['views'][stage]
                    if post['environment']['kind'] != 'native':
                        raise ValueError('expected the formal RQ1 native reward recipe')
                    opts = post['environment']['options']
                    opts['control_output_root'] = str(root / 'controls' / name)
                    opts['host_api']['seed'] = seed
                    config['api'] = copy.deepcopy(opts['host_api'])
                    set_option(command, 'reference-model', stage2)
                    set_option(command, 'grpo-gate-anchor-path', view['views']['sft'])
                    old_incidents = int(command[command.index('--grpo-incidents') + 1])
                    set_option(command, 'grpo-incidents', max(1, math.ceil(old_incidents * spec['training_fraction'])))
                    set_option(command, 'grpo-checkpoint-interval', 1)
                elif stage == 'transfer_diagnosis':
                    # On-policy diagnosis mining must see only the selected native incidents.
                    post['environment_path'] = view['views']['grpo_repair']
                    post['environment'] = copy.deepcopy(yaml.safe_load(Path(spec['templates']['grpo_repair']).read_text())['aria']['post_training']['environment'])
                    post['environment']['options']['control_output_root'] = str(root / 'controls' / name)
                    post['environment']['options']['host_api']['seed'] = seed
                else:
                    post.pop('environment_path', None)
                    post.pop('environment', None)
                for flag, value in spec.get('stage_overrides', {}).get(model_id, {}).get(stage, {}).items():
                    set_option(command, flag, value)
                config['output_dir'] = str(root / 'output')
                config['workspace_dir'] = str(root / 'workspace')
                path = root / 'configs/training' / f'{name}.yml'
                write_yaml(path, config)
                load_config(path)
                checkpoint = str(output / 'checkpoints' / stage / 'checkpoint')
                jobs.append({'id': name, 'model_id': model_id, 'foundation_model': model, 'seed': seed,
                             'stage': stage, 'config': str(path), 'output_dir': str(output), 'checkpoint': checkpoint,
                             'parent_checkpoint': previous, 'expected_updates': budget * (2 if stage == 'transfer_diagnosis' else 1),
                             'rq1_updates': reference_budget * (2 if stage == 'transfer_diagnosis' else 1),
                             'lora_rank': int(command[command.index('--lora-rank') + 1]),
                             'template': str(template), 'template_sha256': sha(template)})
                previous = checkpoint
                if stage == 'sft':
                    stage1 = checkpoint
                if stage == 'transfer_diagnosis':
                    stage2 = checkpoint
    return jobs


def build(spec):
    if not 0 < spec['update_fraction'] <= 1:
        raise ValueError('update_fraction must be in (0, 1]')
    if not spec['seeds'] or len(set(spec['seeds'])) != len(spec['seeds']) or any(type(s) is not int for s in spec['seeds']):
        raise ValueError('training seeds must be unique integers')
    root = Path(spec['output_root']).resolve()
    root.mkdir(parents=True, exist_ok=False)
    view = prepare_views(spec, root)
    jobs = build_jobs(spec, root, view)
    pool = yaml.safe_load(Path(spec['pool_template']).read_text())
    pool['output_dir'] = str(root / 'pool')
    pool['generalization']['fitting_paths'] = [str(root / 'views/fitting_audit.jsonl')]
    pool['generalization']['seen_host_models'] = sorted(set(pool['generalization']['seen_host_models']) | set(view['fitting_hosts']))
    pool['generalization']['prepare_only'] = True
    pool['aria']['gate_threshold'] = 0.60
    pool['experiments'] = [e for e in pool['experiments'] if e['generalization_axis'] in AXES]
    write_yaml(root / 'configs/pool.yml', pool)
    load_config(root / 'configs/pool.yml')
    tradeoff_pool = copy.deepcopy(pool)
    tradeoff_pool['output_dir'] = str(root / 'figure01_pool')
    reference = next(e for e in pool['experiments'] if e['generalization_axis'] == 'reference')
    tradeoff_pool['experiments'] = [{**reference, 'mas': mas, 'name': f'tradeoff-{mas}-gsm'}
                                  for mas in ('openmanus', 'metagpt', 'magentic-one', 'ag2')]
    write_yaml(root / 'configs/figure01_pool.yml', tradeoff_pool)
    load_config(root / 'configs/figure01_pool.yml')
    contract = {
        'schema_version': 'aria-rq2-figure-reservation-v2', 'status': 'reserved_no_evaluation_run',
        'case_schema': 'aria-figure-case-v2', 'writer': 'aria.figure_logging.append_case',
        'case_fields': sorted(REQUIRED | V2_REQUIRED),
        'axes': list(AXES), 'models': spec['models'], 'training_seeds': spec['seeds'],
        'gate_grid': [0.90, 0.75, 0.60, 0.45, 0.30], 'default_gate': 0.60,
        'grid_status': 'proposal: freeze on in-domain validation before test',
        'budget_grid': None, 'stop_rule': None, 'output_selection_rule': None,
        'budget_status': 'must freeze common caps, stopping and output selection before budget sweep',
        'figure01': 'four original MAS panels; same host configuration and tasks within each; separate real run per threshold',
        'figure02': 'measured executions only; exclude posthoc source acquisition, include runtime initial execution',
        'figure03': 'unseen MAS = LangGraph; unseen benchmark = BBH/ARC; same checkpoint per size/seed across pools, no separate leave-out retraining',
        'figure03_denominator': 'control failures in the common resolved set across four sizes; include untriggered runs',
        'figure04': 'reserve S1, S1+2, S1+3, Full; S1+3 is a separate RQ3 branch and is not trained by RQ2',
        'pairing': ['pool_id', 'task_id', 'benchmark', 'mas', 'host_config_sha256', 'execution_seed', 'training_seed'],
        'coverage': ['planned', 'missing', 'infrastructure_invalid', 'unresolved', 'jointly_resolved'],
        'aggregation': ('single training seed; preserve task/size/variant pairing; no cross-training-seed stability or CI claims'
                        if len(spec['seeds']) == 1 else
                        'compute each seed first, equal-weight seeds; task-cluster bootstrap preserving all size/variant/seed pairs'),
        'rq1_comparison': 'RQ2 uses 10% task groups and reduced updates; its 7B row is not the full-data RQ1 checkpoint',
    }
    atomic_write_json(root / 'eval_logs/contract.json', contract)
    (root / 'eval_logs/cases.jsonl').touch()
    for model in spec['models']:
        for seed in spec['seeds']:
            path = root / 'eval_logs' / model / f'seed-{seed}'
            path.mkdir(parents=True)
            (path / 'cases.jsonl').touch()
    manifest = {'schema_version': 'aria-rq2-small-v1', 'status': 'prepared_not_trained', 'spec': spec,
                'views': view, 'jobs': jobs, 'axes': list(AXES), 'pool_config': str(root / 'configs/pool.yml'),
                'frozen_sha256': {str(p): sha(p) for directory in ('views', 'configs') for p in (root / directory).rglob('*') if p.is_file()}}
    atomic_write_json(root / 'manifest.json', manifest)
    atomic_write_json(root / 'checkpoint_registry.json', {'status': 'pending_training', 'checkpoints': {}})
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', type=Path, required=True)
    manifest = build(yaml.safe_load(parser.parse_args().config.read_text()))
    print(json.dumps({'status': manifest['status'], 'jobs': len(manifest['jobs']),
                      'sampling': {k: v for k, v in manifest['views']['sampling'].items() if k != 'selected_aliases'},
                      'counts': manifest['views']['counts']}))


if __name__ == '__main__':
    main()
