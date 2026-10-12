"""Freeze the two trained RQ1 models, complete test pool and nested budgets."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

import yaml

from aria.split_audit import audit_disjoint
from baselines.core.datasets import load_baseline_samples
from common.artifacts import atomic_write_json, atomic_write_jsonl
from common.runtime_support import safe_run_id
from masfm.config import load_config
from scripts.aria.build_full_rq1 import IndexedResolver, native_groups
from scripts.aria.run_rq23_queue import freeze
from scripts.aria.verify_training_checkpoint import file_sha256

MODELS = ('qwen', 'llama')
BENCHMARKS = ('gsm-plus', 'mmlu', 'humaneval', 'olympiadbench', 'gaia', 'swe-bench-lite')
SYSTEMS = ('openmanus', 'metagpt', 'magentic-one', 'ag2')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def plan_tasks(identities, failures, budgets):
    tasks = []
    for index, row in enumerate(identities):
        case_id = f'case-{index:05d}'
        common = {**row, 'case_id': case_id, 'sample_index': index}
        control = f'{case_id}-runtime-control-1'
        for mode in ('runtime', 'posthoc'):
            if mode == 'posthoc' and row['incident_id'] not in failures:
                continue
            for model in (('control', *MODELS) if mode == 'runtime' else MODELS):
                for attempt in range(1, (max(budgets) if model == 'qwen' else 1) + 1):
                    key = f'{case_id}-{mode}-{model}-{attempt}'
                    tasks.append({**common, 'id': key, 'mode': mode, 'backbone': model,
                        'candidate': attempt, 'control_id': control, 'method': 'control' if model == 'control' else 'aria',
                        'service': 'host' if model == 'control' else model,
                        'previous_id': f'{case_id}-{mode}-{model}-{attempt - 1}' if attempt > 1 else None})
    return tasks


def experiment_config(manifest, task, infrastructure_attempt):
    """Pure YAML compiler: every native execution still enters main.py -c."""
    spec, root = manifest['spec'], Path(manifest['spec']['output_root'])
    seed = spec['execution_seed'] + task['candidate'] - 1
    location = root / 'raw' / task['id'] / f'attempt-{infrastructure_attempt}'
    common = copy.deepcopy(manifest['templates']['common'])
    host_api = {**manifest['templates']['host_api'], 'seed': seed}
    common.update(output_dir=str(location), workspace_dir=str(location / 'workspace'))
    native = copy.deepcopy(spec['native_options'])
    evaluation = copy.deepcopy(spec['evaluation'])
    if task['mode'] == 'posthoc':
        model_api = {**manifest['templates']['predictor_api'][task['backbone']], 'seed': seed}
        common.update(job='baselines', api=model_api, baseline_workers=1)
        common['baselines'] = [{'name': task['id'], 'method': 'aria_passive', 'input_format': 'aria-full',
            'input_path': str(root / 'pools' / (task['case_id'] + '.jsonl')), 'split': 'test',
            'max_events': 100000, 'prompt_char_budget': spec['posthoc_prompt_chars'], 'sample_workers': 1,
            'resume': True, 'on_sample_error': 'continue', 'output_dir': str(location / 'posthoc'),
            'method_options': {'structured_decoding': True},
            'replay_backend': {'type': 'native', 'options': {'host_api': host_api,
                'paired_control': False, 'external_dir': common['external_dir'],
                'timeout_seconds': spec['process_timeout_seconds'] - 300,
                'continuation_char_budget': 12000, 'native_options': native, 'evaluation': evaluation,
                'project_environments': spec['project_environments']}}}]
        result = location / 'posthoc/results.jsonl'
    else:
        common.update(job='inference' if task['method'] == 'control' else 'aria', api=host_api)
        common['aria'] = {'enabled': False, 'mode': 'off'}
        if task['method'] == 'aria':
            api = manifest['templates']['predictor_api'][task['backbone']]
            common['aria'] = {'enabled': True, 'mode': 'active', 'fail_open': True,
                'gate_threshold': spec['gate_threshold'], 'runtime_policy': 'aria',
                'model': {'provider': 'openai-chat-completions', 'base_url': api['base_url'],
                    'model': api['model'], 'api_key_required': False, 'api_key_env': 'ARIA_API_KEY',
                    'max_tokens': api['max_tokens'], 'max_input_tokens': spec['predictor_input_tokens'],
                    'tokenizer_endpoint': api['base_url'].removesuffix('/v1') + '/tokenize',
                    'temperature': api['temperature'], 'seed': seed, 'timeout_seconds': api['timeout_seconds'],
                    'runtime_context': True, 'gate_probability_source': 'label_likelihood',
                    'structured_output_retries': 1}}
        experiment = {**native, 'name': task['id'], 'mas': task['mas'], 'benchmark': task['benchmark'],
            'native': True, 'batch': True, 'sample_source': 'full',
            'sample_split': 'validation' if task['benchmark'] == 'gaia' else 'test',
            'sample_ids': [task['native_sample_id']], 'sample_limit': 1, 'resume': True, 'judge': False,
            'replicate_id': f"source-repeat-{task['replicate']}"}
        if task['benchmark'] in {'gaia', 'swe-bench-lite'}:
            experiment['evaluation'] = {**evaluation,
                'harness': 'gaia-official' if task['benchmark'] == 'gaia' else 'swebench-official'}
        if task['mas'] not in {'ag2', 'magentic-one'}:
            experiment.pop('code_execution_environment', None)
        if task['mas'] != 'magentic-one':
            experiment.pop('model_vision', None)
        common['experiments'] = [experiment]
        result = (location / common['job'] / task['mas'] / task['benchmark'] / safe_run_id(task['id'])
                  / 'samples' / safe_run_id(task['native_sample_id']) / 'result.json')
    return common, result


def prepare_services(spec, root):
    services = copy.deepcopy(spec['services'])
    for key, value in services.items():
        total = float(subprocess.check_output(['nvidia-smi', '-i', str(value['gpu']),
            '--query-gpu=memory.total', '--format=csv,noheader,nounits'], text=True).strip())
        revision = (Path.home() / '.cache/huggingface/hub' /
                    ('models--' + value['model'].replace('/', '--')) / 'refs/main').read_text().strip()
        args = ['--revision', revision, '--enforce-eager', '--max-num-seqs', str(spec['max_num_seqs']),
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
        value.update(config=str(path), revision=revision, config_sha256=file_sha256(path))
    return services


def build(spec):
    repo = Path.cwd().resolve()
    stamp = datetime.now(ZoneInfo('America/New_York')).strftime('%Y%m%d-%H%M%S')
    root = Path(spec.get('output_root') or f'outputs/aria/rq1-{stamp}').resolve()
    if (root / 'manifest.json').exists():
        raise FileExistsError('existing evaluation identity; resume launch.yml')
    for sub in ('setup', 'services', 'pools', 'configs', 'logs', 'data', 'figure01', 'figure02'):
        (root / sub).mkdir(parents=True, exist_ok=True)
    spec = {**spec, 'output_root': str(root), 'repo_root': str(repo),
            'serving_environment': str(Path(spec['serving_environment']).resolve()),
            'default_environment': str(repo / '.venv'),
            'project_environments': {k: str(Path(v).resolve()) for k, v in spec['project_environments'].items()}}
    pool = Path(spec['pool']).resolve()
    train_paths = [Path(p).resolve() for p in spec['training_audit_paths']]
    audit = audit_disjoint(train_paths, [pool])
    if audit['status'] != 'passed':
        raise ValueError('test tasks overlap training')
    samples = load_baseline_samples(pool, input_format='aria-full', max_events=100000)
    if len(samples) != spec['expected_pool_size'] or any(s.split != 'test' for s in samples):
        raise ValueError('unexpected test pool')
    # Reuse a restoration only when its source pool digest matches exactly.
    old_path = Path(spec['native_identity_manifest'])
    old = json.loads(old_path.read_text()) if old_path.exists() else {}
    if old.get('pool_sha256') == file_sha256(pool):
        identities = old['native_identities']
    else:
        resolver = IndexedResolver(data_dir=repo / 'data', benchmark_splits={
            b: 'validation' if b == 'gaia' else 'test' for b in BENCHMARKS})
        _, identities = native_groups(samples, resolver)
    if {r['incident_id'] for r in identities} != {s.sample_id for s in samples} or len(identities) != len(samples):
        raise ValueError('native identity coverage mismatch')
    # Round-robin source cells prevents long GSM rows from starving other cells.
    groups = {}
    for row in identities:
        groups.setdefault((row['mas'], row['benchmark']), []).append(row)
    identities = [rows[i] for i in range(max(map(len, groups.values())))
                  for _, rows in sorted(groups.items()) if i < len(rows)]
    raw = {r['sample_id']: r for r in map(json.loads, pool.read_text().splitlines())}
    by_id = {s.sample_id: s for s in samples}
    failures = {s.sample_id for s in samples if s.source_success is False}
    if len(failures) != spec['expected_posthoc_size']:
        raise ValueError('unexpected posthoc pool')
    for i, row in enumerate(identities):
        sample = by_id[row['incident_id']]
        row['source_evaluation'] = dict(sample.source_evaluation)
        row['source_task_id'] = sample.task_id
        row['source_success'] = sample.source_success
        if sample.sample_id in failures:
            atomic_write_jsonl(root / 'pools' / f'case-{i:05d}.jsonl', [raw[sample.sample_id]])
    checkpoints = {}
    for key, training_root in spec['models'].items():
        registry_path = Path(training_root) / 'checkpoint_registry.json'
        registry = json.loads(registry_path.read_text())
        record = registry['checkpoints']['grpo_gate']
        if registry['seed'] != 42 or record['loss_count'] != 16:
            raise ValueError('expected final 16-update Gate checkpoint')
        checkpoint = Path(record['checkpoint'])
        if file_sha256(checkpoint / 'adapter_model.safetensors') != record['adapter_sha256']:
            raise ValueError('checkpoint changed')
        checkpoints[key] = {'checkpoint': str(checkpoint), 'sha256': record['adapter_sha256'],
                            'registry_sha256': file_sha256(registry_path), 'registry': registry}
    services = prepare_services(spec, root)
    api = {'provider': 'vllm', 'protocol': 'openai-chat-completions', 'api_key_required': False,
           'api_key_env': 'ARIA_API_KEY', 'timeout_seconds': 600, 'temperature': spec['temperature'],
           'seed': spec['execution_seed'], 'retries': 1, 'max_tokens': 2048, 'context_window_tokens': 32768,
           'model': services['host']['model'], 'base_url': f"http://127.0.0.1:{services['host']['port']}/v1"}
    predictors = {key: {**api, 'max_tokens': spec['predictor_output_tokens'],
        'model': f'rq1-trained-{key}', 'context_window_tokens': services[key]['context'],
        'base_url': f"http://127.0.0.1:{services[key]['port']}/v1"} for key in MODELS}
    templates = {'common': {'runtime': {'kind': 'endpoint'}, 'data_dir': str(repo / 'data'),
        'external_dir': str(repo / 'external'), 'prepare': {'clone_mas': False, 'download_benchmarks': False},
        'output': {'existing_output_policy': 'resume'}, 'logging': {'level': 'INFO'}},
        'host_api': api, 'predictor_api': predictors}
    protocol = {'training_seed': 42, 'execution_seed': spec['execution_seed'], 'budgets': spec['budgets'],
        'posthoc_selection': 'first passing candidate, stop on success; otherwise last candidate',
        'runtime_selection': 'plurality of normalized delivered answers/patch hashes; ties earliest; no evaluator access',
        'nested_sweep': 'budget 1 is the main table; prefixes of the same candidate stream yield budgets 2 and 4',
        'posthoc_source_cost': 'exclude pre-existing source acquisition; report source-inclusive EB separately',
        'control_cost': 'matched control is audit overhead, not charged to intervention EB',
        'infrastructure': 'up to two attempts per candidate; first resolved outcome including incorrect/invalid format wins',
        'scope': 'full original test record pool; fresh local Qwen3-8B host, not original mixed-host reproduction',
        'limitations': ['one training seed only', 'figure01 full-data 1.5B/3B/14B checkpoints unavailable',
            'only ARIA and required no-intervention controls; research baselines outside requested rows']}
    manifest = {'schema_version': 'aria-rq1-evaluation-v1', 'spec': spec, 'services': services,
        'checkpoints': checkpoints, 'templates': templates, 'protocol': protocol, 'protocol_sha256': digest(protocol),
        'pool_sha256': file_sha256(pool), 'split_audit': audit, 'identities': identities,
        'tasks': plan_tasks(identities, failures, spec['budgets'])}
    # Validate every distinct MAS/benchmark/mode/model config, without writing tens of thousands of identical YAMLs.
    seen = set()
    for task in manifest['tasks']:
        key = (task['mas'], task['benchmark'], task['mode'], task['backbone'])
        if key in seen:
            continue
        seen.add(key)
        config, _ = experiment_config(manifest, task, 1)
        path = root / 'setup' / ('-'.join(key) + '.yml')
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        load_config(path)
    code = freeze(repo, root)
    spec['code_root'] = str(code)
    for mas, environment in spec['project_environments'].items():
        (code / f'.venv-{mas}').symlink_to(environment, target_is_directory=True)
    atomic_write_json(root / 'manifest.json', manifest)
    atomic_write_json(root / 'setup/split_audit.json', audit)
    atomic_write_json(root / 'setup/checkpoints.json', checkpoints)
    (root / 'launch.yml').write_text(yaml.safe_dump({'manifest': str(root / 'manifest.json'),
        'manifest_sha256': file_sha256(root / 'manifest.json')}))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', type=Path, required=True)
    manifest = build(yaml.safe_load(parser.parse_args().config.read_text()))
    from scripts.aria.report_rq1_eval import write_report
    write_report(manifest, {})
    print(json.dumps({'root': manifest['spec']['output_root'], 'maximum_candidates': len(manifest['tasks']),
                      'counts': dict(Counter((t['mode'] + '/' + t['backbone']) for t in manifest['tasks']))}))


if __name__ == '__main__':
    main()
