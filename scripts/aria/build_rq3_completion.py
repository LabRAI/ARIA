"""Freeze missing RQ3 training branches and their complete paired native evaluation."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

import yaml

from aria.split_audit import audit_disjoint
from common.artifacts import atomic_write_json, atomic_write_jsonl
from masfm.config import load_config
from scripts.aria.build_dated_training import set_option
from scripts.aria.build_rq1_eval import digest, experiment_config
from scripts.aria.build_rq23_study import cell
from scripts.aria.run_rq23_queue import freeze
from scripts.aria.verify_training_checkpoint import file_sha256

STAGES = ('sft', 'dpo', 'transfer_diagnosis', 'grpo_repair', 'grpo_gate')
VARIANT_STAGES = {'S1': 'sft', 'S1+2': 'transfer_diagnosis', 'S1+3': 's13/grpo_gate', 'Full': 'grpo_gate'}


def read(path):
    return json.loads(Path(path).read_text())


def filtered_rows(path, heldout):
    rows = [json.loads(line) for line in Path(path).open() if line.strip()]
    if heldout:
        rows = [r for r in rows if cell(r).mas_id != 'magentic-one']
    if not rows:
        raise ValueError('empty training view: ' + str(path))
    return rows


def create_views(root, source):
    originals = {
        'sft': Path('data/training/full/sft.jsonl'),
        'dpo': Path('data/training/full/dpo.jsonl'),
        'repair': Path('data/training/repair.jsonl'),
        'gate': Path('data/training/gate.jsonl'),
        'validation': Path('data/training/gate_validation.jsonl')}
    views = {}
    for axis in ('in_domain', 'unseen_system'):
        view = {}
        for name, path in originals.items():
            rows = filtered_rows(path, axis == 'unseen_system')
            destination = root / 'views' / axis / ('train' if name in ('sft', 'dpo') else '') / (name + '.jsonl')
            atomic_write_jsonl(destination, rows)
            view[name] = {'path': str(destination), 'sha256': file_sha256(destination),
                          'rows': len(rows), 'mas': sorted({cell(r).mas_id for r in rows})}
        if axis == 'unseen_system' and any('magentic-one' in r['mas'] for r in view.values()):
            raise ValueError('held-out framework leaked into fitting/calibration/rewards')
        audit = audit_disjoint([Path(view[k]['path']) for k in ('sft', 'dpo', 'repair', 'gate')],
                               [Path(source['spec']['pool'])])
        if audit['status'] != 'passed':
            raise ValueError('evaluation task leakage in training')
        view['split_audit'] = audit
        views[axis] = view
    return views


def make_training_config(root, family, axis, seed, stage, previous, reference, view, branch=''):
    path = Path(f'configs/post_training/{family}/{stage}.yml')
    raw = yaml.safe_load(path.read_text())
    key = '/'.join(filter(None, (family, axis, f'seed-{seed}', branch, stage)))
    output = root / 'training' / key
    post = raw['aria']['post_training']
    post.update(base_model=previous, source_path=str(Path(view['sft']['path']).parent),
        output_dir=str(output), stages=[stage], random_seed=seed, resume=False,
        gate_validation_path=view['validation']['path'])
    command = post['backend']['command']
    command[:] = [f'UV_PROJECT_ENVIRONMENT={Path(".venv-aria-training").resolve()}'
                  if str(a).startswith('UV_PROJECT_ENVIRONMENT=') else a for a in command]
    set_option(command, 'seed', seed)
    set_option(command, 'reference-model', reference)
    if not stage.startswith('grpo'):
        if family == 'qwen':
            set_option(command, 'optimizer-steps', {'sft': 2048, 'dpo': 192, 'transfer_diagnosis': 96}[stage])
            if stage == 'transfer_diagnosis':
                set_option(command, 'transfer-max-new-tokens', 160)
        # Llama preserves its original natural-epoch supervised recipe;
        # its actual compute difference is recorded rather than imputed.
    else:
        post['environment_path'] = view['repair' if stage == 'grpo_repair' else 'gate']['path']
        env = post['environment']['options']
        env['control_output_root'] = str(root / 'controls' / key)
        env['host_api']['seed'] = seed
        raw['api'] = copy.deepcopy(env['host_api'])
        set_option(command, 'grpo-gate-anchor-path', view['sft']['path'])
        set_option(command, 'grpo-optimizer-steps', 16)
    raw['output_dir'] = str(output / 'output')
    raw['workspace_dir'] = str(output / 'workspace')
    config = root / 'configs/training' / (key + '.yml')
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(yaml.safe_dump(raw, sort_keys=False))
    load_config(config)
    return {'id': key, 'family': family, 'axis': axis, 'seed': seed, 'stage': stage,
        'branch': branch, 'config': str(config), 'config_sha256': file_sha256(config),
        'output_dir': str(output), 'checkpoint': str(output / 'checkpoints' / stage / 'checkpoint'),
        'foundation_model': 'Qwen/Qwen2.5-7B-Instruct' if family == 'qwen' else 'meta-llama/Llama-3.1-8B-Instruct',
        'expected_grpo_updates': 16 if stage.startswith('grpo') else None,
        'expected_updates': (192 if stage == 'transfer_diagnosis' else {'sft': 2048, 'dpo': 192}.get(stage))
                             if family == 'qwen' and not stage.startswith('grpo') else None,
        'view': axis, 'depends_on': [], 'lora_rank': 32}


def plan_training(spec, root, views, source):
    jobs, checkpoints, arms = [], {}, []
    for family in ('qwen', 'llama'):
        for axis in ('in_domain', 'unseen_system'):
            for seed in spec[family + '_seeds']:
                key = f'{family}/{axis}/seed-{seed}'
                registry = source['checkpoints'][family]['registry']['checkpoints']
                reuse = axis == 'in_domain' and seed == 42
                previous = 'Qwen/Qwen2.5-7B-Instruct' if family == 'qwen' else 'meta-llama/Llama-3.1-8B-Instruct'
                dependency = None
                for stage in STAGES:
                    identity = key + '/' + stage
                    if reuse:
                        record = registry[stage]
                        checkpoint = Path(record['checkpoint'])
                        if file_sha256(checkpoint / 'adapter_model.safetensors') != record['adapter_sha256']:
                            raise ValueError('changed reused checkpoint')
                        checkpoints[identity] = {'checkpoint': str(checkpoint), 'sha256': record['adapter_sha256'],
                            'status': 'verified', 'reused': True, 'source': record}
                        previous = str(checkpoint)
                        continue
                    reference = checkpoints.get(key + '/sft', {}).get('checkpoint', previous)
                    if stage.startswith('grpo'):
                        reference = checkpoints[key + '/transfer_diagnosis']['checkpoint']
                    job = make_training_config(root, family, axis, seed, stage, previous, reference, views[axis])
                    job['depends_on'] = [dependency] if dependency else []
                    jobs.append(job)
                    checkpoints[identity] = {'checkpoint': job['checkpoint'], 'status': 'pending', 'training_job': identity}
                    previous, dependency = job['checkpoint'], identity
                if family == 'qwen':
                    previous = checkpoints[key + '/sft']['checkpoint']
                    reference = previous
                    dependency = None if reuse else key + '/sft'
                    for stage in ('grpo_repair', 'grpo_gate'):
                        job = make_training_config(root, family, axis, seed, stage, previous, reference, views[axis], 's13')
                        job['depends_on'] = [dependency] if dependency else []
                        jobs.append(job)
                        checkpoints[job['id']] = {'checkpoint': job['checkpoint'], 'status': 'pending', 'training_job': job['id']}
                        previous, dependency = job['checkpoint'], job['id']
                for variant in (VARIANT_STAGES if family == 'qwen' else ('Full',)):
                    checkpoint_id = key + '/' + VARIANT_STAGES[variant]
                    arm_id = key.replace('/', '-') + '-' + variant.replace('+', '').lower()
                    origin = None
                    if reuse:
                        if variant == 'Full':
                            origin = {'root': spec['rq1_root'], 'backbone': family}
                        elif family == 'qwen' and variant in ('S1', 'S1+2'):
                            origin = {'root': spec['rq3_source_root'], 'backbone': 's1' if variant == 'S1' else 's12'}
                    arms.append({'id': arm_id, 'family': family, 'axis': axis, 'training_seed': seed,
                        'variant': variant, 'checkpoint_id': checkpoint_id, 'origin': origin,
                        'service': family, 'alias': ('rq1-trained-llama' if family == 'llama' and reuse
                                                     else 'rq3-complete-' + arm_id)})
    return jobs, checkpoints, arms


def plan_eval(spec, identities, arms):
    tasks, controls = [], {}
    seeds = sorted(set(spec['qwen_seeds'] + spec['llama_seeds']))
    for index, identity in enumerate(identities):
        case_id = f'case-{index:05d}'
        for seed in seeds:
            run_id = f'{case_id}-seed-{seed}-control'
            common = {**identity, 'case_id': case_id, 'sample_index': index, 'mode': 'runtime',
                      'candidate': 1, 'previous_id': None, 'execution_seed': seed, 'training_seed': seed}
            control = {**common, 'id': run_id, 'control_id': run_id, 'backbone': 'control',
                'method': 'control', 'service': 'host', 'family': 'control', 'variant': 'control', 'axis': 'in_domain',
                'origin': {'root': spec['rq1_root'], 'id': f'{case_id}-runtime-control-1'} if seed == 42 else None}
            controls[(case_id, seed)] = run_id
            tasks.append(control)
        for arm in arms:
            if arm['axis'] == 'unseen_system' and identity['mas'] != 'magentic-one':
                continue
            seed = arm['training_seed']
            task = {**identity, 'case_id': case_id, 'sample_index': index, 'mode': 'runtime',
                'candidate': 1, 'previous_id': None, 'id': case_id + '-' + arm['id'],
                'control_id': controls[(case_id, seed)], 'backbone': arm['id'], 'arm_id': arm['id'],
                'family': arm['family'], 'variant': arm['variant'], 'axis': arm['axis'],
                'method': 'aria', 'service': arm['service'], 'training_seed': seed, 'execution_seed': seed,
                'checkpoint_id': arm['checkpoint_id'], 'origin': None}
            if arm['origin']:
                task['origin'] = {'root': arm['origin']['root'], 'id': f"{case_id}-runtime-{arm['origin']['backbone']}-1"}
            tasks.append(task)
    if len({t['id'] for t in tasks}) != len(tasks):
        raise ValueError('duplicate evaluation task')
    return tasks


def build(spec):
    if spec.get('evaluation_only') or spec.get('training_enabled') is False:
        raise ValueError('training is disabled; resume the existing evaluation launch.yml instead of building training branches')
    repo = Path.cwd().resolve()
    spec = {**spec, **{k: str(Path(v).resolve()) for k, v in spec.items() if k.endswith('_root')}}
    root = Path(spec['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    for directory in ('setup', 'logs', 'raw', 'data', 'figure04', 'configs/native', 'services', 'runtime_training'):
        (root / directory).mkdir(parents=True)
    source = read(Path(spec['rq1_root']) / 'manifest.json')
    views = create_views(root, source)
    jobs, checkpoints, arms = plan_training(spec, root, views, source)
    tasks = plan_eval(spec, source['identities'], arms)
    native_root = root / 'native_code'
    freeze_record = read(Path(spec['rq1_root']) / 'code_freeze.json')
    for name, sha in freeze_record['files'].items():
        src = Path(source['spec']['code_root']) / name
        if file_sha256(src) != sha:
            raise ValueError('source native code changed')
        dst = native_root / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for mas, env in source['spec']['project_environments'].items():
        (native_root / f'.venv-{mas}').symlink_to(env, target_is_directory=True)
    training_code = freeze(repo, root)
    settings = {**source['spec'], **spec, 'code_root': str(native_root), 'training_code_root': str(training_code)}
    templates = copy.deepcopy(source['templates'])
    for arm in arms:
        templates['predictor_api'][arm['id']] = {**source['templates']['predictor_api'][arm['family']],
            'base_url': 'http://127.0.0.1:19111/v1' if arm['family'] == 'qwen' else 'http://127.0.0.1:19012/v1',
            'model': arm['alias']}
    manifest = {'schema_version': 'aria-rq3-completion-v1', 'spec': settings, 'views': views,
        'training_jobs': jobs, 'checkpoints': checkpoints, 'arms': arms, 'tasks': tasks,
        'templates': templates, 'identities': source['identities'], 'pool_sha256': source['pool_sha256'],
        'protocol_sha256': digest({'source': source['protocol_sha256'], 'seeds': spec['qwen_seeds'],
                                 'axes': ['in_domain', 'unseen_system'], 'variant_stages': VARIANT_STAGES}),
        'source_roots': {s: {'manifest_sha256': file_sha256(Path(s) / 'manifest.json')}
                         for s in (spec['rq1_root'], spec['rq3_source_root'])},
        'native_code_files': freeze_record['files']}
    examples = set()
    for task in tasks:
        if task['origin']:
            continue
        key = (task['family'], task['mas'], task['benchmark'], task['training_seed'], task['method'])
        if key in examples:
            continue
        examples.add(key)
        config, _ = experiment_config({**manifest, 'spec': {**settings, 'execution_seed': task['execution_seed']}}, task, 1)
        path = root / 'setup' / ('native-' + '-'.join(map(str, key)) + '.yml')
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        load_config(path)
    atomic_write_json(root / 'manifest.json', manifest)
    atomic_write_json(root / 'checkpoint_registry.json', checkpoints)
    atomic_write_json(root / 'setup/views.json', views)
    (root / 'launch.yml').write_text(yaml.safe_dump({'manifest': str(root / 'manifest.json'),
        'manifest_sha256': file_sha256(root / 'manifest.json')}))
    print(json.dumps({'root': str(root), 'training_phases': len(jobs), 'native_tasks': len(tasks),
                      'imported_tasks': sum(bool(t['origin']) for t in tasks)}))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', type=Path, required=True)
    build(yaml.safe_load(parser.parse_args().config.read_text()))


if __name__ == '__main__':
    main()
