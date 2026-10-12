"""Freeze a single RQ2 evaluation shared by the paper table and Figure 03."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess

import yaml

from common.artifacts import atomic_write_json
from common.runtime_support import safe_run_id
from masfm.config import load_config
from masfm.generalization import digest, prepare_pools
from scripts.aria.report_rq2_small import qualified
from scripts.aria.run_rq23_queue import freeze
from scripts.aria.verify_training_checkpoint import file_sha256

MODELS = ('qwen25-1.5b', 'qwen25-3b', 'qwen25-7b', 'qwen25-14b', 'llama31-8b')
SIZES = MODELS[:4]


def arms():
    return [('control', 'n/a'), ('spark_to_fire', 'qwen25-7b'), ('safeagent', 'qwen25-7b'),
            *((method, model) for model in MODELS for method in ('untuned', 'aria'))]


def task_plan(pools):
    tasks = []
    for pool in pools:
        for index, sid in enumerate(pool['sample_ids']):
            control = f"control--n-a--{pool['name']}--{sid}"
            for method, model in arms():
                key = f"{method}--{model.replace('/', '-')}--{pool['name']}--{sid}"
                tasks.append({'id': key, 'method': method, 'backbone': model,
                              'pool_id': pool['name'], 'sample_id': sid, 'sample_index': index,
                              'axis': pool['axis'], 'mas': pool['mas'], 'benchmark': pool['benchmark'],
                              'control_id': control, 'service': 'host' if method == 'control' else model})
    ids = {t['id'] for t in tasks}
    assert len(ids) == len(tasks)
    assert all(t['control_id'] in ids for t in tasks)
    return tasks


def checkpoint_identity(training_root, model, seed):
    manifest = json.loads((training_root / 'manifest.json').read_text())
    registry = json.loads((training_root / 'checkpoint_registry.json').read_text())
    if not qualified(manifest, registry, model, seed):
        return None
    record = registry['checkpoints'][f'{model}/seed-{seed}/grpo_gate']
    return {'checkpoint': record['checkpoint'], 'sha256': record['stages'][0]['adapter_sha256'],
            'stages': {key: value for key, value in registry['checkpoints'].items()
                       if key.startswith(f'{model}/seed-{seed}/')}}


def sample_result_path(location, job, task):
    return (location / job / task['mas'] / task['benchmark'] / safe_run_id(task['id'])
            / 'samples' / safe_run_id(task['sample_id']) / 'result.json')


def prepare_services(spec, root):
    """Freeze model revisions and serving YAMLs; startup belongs to the scheduler."""
    existing = root / 'setup/services.json'
    if existing.exists():
        services = json.loads(existing.read_text())
        for key, value in spec.get('services', {}).items():
            if any(services[key][name] != setting for name, setting in value.items()):
                raise ValueError('existing serving resource plan differs from the launch YAML')
        return services
    services = copy.deepcopy(spec['services'])
    if set(services) != {'host', *MODELS} or len({v['port'] for v in services.values()}) != len(services):
        raise ValueError('one service per backbone plus host, with unique loopback ports, is required')
    (root / 'services').mkdir(exist_ok=True)
    for key, value in services.items():
        total = float(subprocess.check_output(['nvidia-smi', '-i', str(value['gpu']),
            '--query-gpu=memory.total', '--format=csv,noheader,nounits'], text=True).strip())
        revision = (Path.home() / '.cache/huggingface/hub' /
                    ('models--' + value['model'].replace('/', '--')) / 'refs/main').read_text().strip()
        args = ['--revision', revision, '--enforce-eager', '--max-num-seqs', '16',
                '--max-num-batched-tokens', '4096', '--no-enable-prefix-caching']
        if key == 'host':
            args += ['--enable-auto-tool-choice', '--tool-call-parser', 'hermes']
        else:
            args += ['--enable-lora', '--max-lora-rank', '32', '--max-loras', '1', '--max-cpu-loras', '2']
        path = root / 'services' / f'{key}.yml'
        path.write_text(yaml.safe_dump({'vllm': {'cuda_visible_devices': str(value['gpu']),
            'hf_model_name': value['model'], 'served_model_name': value['model'], 'host': '127.0.0.1',
            'port': value['port'], 'dtype': 'bfloat16', 'max_model_len': value['context'],
            'gpu_memory_utilization': value['memory_gib'] * 1024 / total, 'extra_args': args}}, sort_keys=False))
        value.update(config=str(path), revision=revision)
    atomic_write_json(existing, services)
    return services


def build(spec):
    repo = Path.cwd().resolve()
    root, source = (Path(spec[k]).resolve() for k in ('output_root', 'training_root'))
    if (root / 'manifest.json').exists():
        raise FileExistsError('evaluation is already frozen; resume its launch.yml')
    root.mkdir(parents=True, exist_ok=True)
    (root / 'setup').mkdir(exist_ok=True)
    services = prepare_services(spec, root)
    training = json.loads((source / 'manifest.json').read_text())
    pool_manifest = json.loads((source / 'pool/generalization/manifest.json').read_text())
    pool_config = load_config(Path(training['pool_config']))
    pool_config.prepare['download_benchmarks'] = False
    fresh = prepare_pools(pool_config)
    for key in ('pools', 'fitting_corpora'):
        if fresh[key] != pool_manifest[key]:
            raise ValueError('source data/fitting audit changed since the training pool freeze')
    if set(training['spec']['models']) != set(MODELS) or training['spec']['seeds'] != [42]:
        raise ValueError('RQ2 evaluation expects five backbones and training seed 42')
    if spec['gate_threshold'] != 0.60 or spec['execution_seed'] != 42 or spec['training_seed'] != 42:
        raise ValueError('the maintained RQ2 fixed operating point is threshold .60 / seed 42')
    for name in ('data_dir', 'external_dir'):
        # Native subprocesses run from the frozen source checkout.
        if not Path(getattr(pool_config, name)).resolve().exists():
            raise FileNotFoundError(name)
    frozen_code = freeze(repo, root)
    spec = {**spec, 'output_root': str(root), 'training_root': str(source), 'repo_root': str(repo),
            'code_root': str(frozen_code), 'serving_environment': str(Path(spec['serving_environment']).resolve()),
            'project_environments': {k: str(Path(v).resolve()) for k, v in spec['project_environments'].items()}}
    for mas, environment in spec['project_environments'].items():
        if not (Path(environment) / 'bin/python').exists():
            raise FileNotFoundError(environment)
        (frozen_code / f'.venv-{mas}').symlink_to(environment, target_is_directory=True)
    protocol = {'axes': ['reference', 'unseen_benchmark', 'unseen_system'], 'seed': 42,
                'gate_threshold': .60, 'calibration': 'identity; fixed preregistered point, no test tuning',
                'budget': 'one enacted pre-delivery text substitution; no historical rollback',
                'control': 'one fresh uninterrupted native execution per pool/task shared by every arm',
                'retry': 'first infrastructure-valid attempt; never select on benchmark correctness',
                'figure03': 'Full Qwen checkpoints; common resolved tasks across four sizes within each pool',
                'unseen_benchmark_aggregation': 'micro average of BBH 108 and ARC-Challenge 100 tasks',
                'request_budgets': {'host_context': 32768, 'host_max_tokens': 2048,
                                    'predictor_input': 6144, 'predictor_max_tokens': 2048},
                'training_manifest_sha256': file_sha256(source / 'manifest.json'),
                'pool_manifest_sha256': file_sha256(source / 'pool/generalization/manifest.json'),
                'host_revision': services['host']['revision']}
    tasks = task_plan(pool_manifest['pools'])
    configs_dir = root / 'configs'
    configs_dir.mkdir(exist_ok=True)
    for task in tasks:
        template = source / 'pool/generalization/configs' / f"{task['pool_id']}-aria.yml"
        config = yaml.safe_load(template.read_text())
        method, model = task['method'], task['backbone']
        config['data_dir'], config['external_dir'] = str(repo / 'data'), str(repo / 'external')
        config['api']['base_url'] = f"http://127.0.0.1:{services['host']['port']}/v1"
        config['api']['model'] = services['host']['model']  # Keeps the native Qwen3 non-thinking bridge.
        config['job'] = 'inference' if method == 'control' else 'aria'
        if method == 'control':
            config['aria'] = {'enabled': False, 'mode': 'off'}
        else:
            service = services[model]
            config['aria']['runtime_policy'] = method if method in ('safeagent', 'spark_to_fire') else 'aria'
            config['aria']['model'].update(model=f'rq2-full-{model}' if method == 'aria' else service['model'],
                base_url=f"http://127.0.0.1:{service['port']}/v1",
                tokenizer_endpoint=f"http://127.0.0.1:{service['port']}/tokenize", timeout_seconds=300)
        experiment = config['experiments'][0]
        experiment.update(name=task['id'], sample_ids=[task['sample_id']], sample_limit=1)
        task['host_config_sha256'] = digest({'api': config['api'], 'experiment': {
            k: v for k, v in experiment.items() if k not in ('name', 'sample_ids', 'sample_limit')}})
        task['configs'] = []
        for attempt in range(1, spec['max_infrastructure_attempts'] + 1):
            current = copy.deepcopy(config)
            location = root / 'raw' / task['id'] / f'attempt-{attempt}'
            current.update(output_dir=str(location), workspace_dir=str(location / 'workspace'))
            path = configs_dir / f"{task['id']}--attempt-{attempt}.yml"
            path.write_text(yaml.safe_dump(current, sort_keys=False))
            load_config(path)
            result_path = sample_result_path(location, current['job'], task)
            task['configs'].append({'path': str(path), 'sha256': file_sha256(path), 'result': str(result_path)})
    identities = {}
    for model in MODELS:
        identities[model] = checkpoint_identity(source, model, 42)
    manifest = {'schema_version': 'aria-rq2-evaluation-v1', 'spec': spec, 'services': services,
                'pools': pool_manifest['pools'], 'tasks': tasks, 'protocol': protocol,
                'protocol_sha256': digest(protocol), 'calibration_sha256': digest({'kind': 'identity', 'threshold': .60}),
                'training_view_sha256': file_sha256(source / 'views/train/sft.jsonl'),
                'checkpoint_identities': identities}
    atomic_write_json(root / 'manifest.json', manifest)
    atomic_write_json(root / 'setup/pool_manifest.json', pool_manifest)
    atomic_write_json(root / 'setup/training_manifest.json', training)
    atomic_write_json(root / 'setup/checkpoint_identities.json', identities)
    (root / 'launch.yml').write_text(yaml.safe_dump({'manifest': str(root / 'manifest.json')}, sort_keys=False))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', type=Path, required=True)
    manifest = build(yaml.safe_load(parser.parse_args().config.read_text()))
    from scripts.aria.report_rq2_eval import write_report
    write_report(manifest, {})
    print(json.dumps({'root': manifest['spec']['output_root'], 'planned_executions': len(manifest['tasks'])}))


if __name__ == '__main__':
    main()
