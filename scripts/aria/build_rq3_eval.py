"""Freeze RQ3 against the trained 16+16 models and their existing RQ1 stream."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
from zoneinfo import ZoneInfo

import yaml

from common.artifacts import atomic_write_json
from masfm.config import load_config
from scripts.aria.build_rq1_eval import digest, experiment_config
from scripts.aria.verify_training_checkpoint import file_sha256

VARIANTS = ('S1', 'S1+2', 'S1+3', 'Full')
EDGES = (('S1', 'S1+2'), ('S1+3', 'Full'), ('S1', 'S1+3'), ('S1+2', 'Full'))
STAGES = {'s1': 'sft', 's12': 'transfer_diagnosis'}
LABELS = {'s1': 'S1', 's12': 'S1+2', 's13': 'S1+3', 'qwen': 'Full', 'llama': 'Full', 'control': 'control'}


def plan_tasks(source):
    imported, local = [], []
    for task in source['tasks']:
        if task['mode'] != 'runtime' or task['candidate'] != 1:
            continue
        imported.append({**task, 'origin': 'rq1', 'variant': LABELS[task['backbone']],
                         'model_family': task['backbone']})
        if task['backbone'] == 'qwen':
            for arm in STAGES:
                local.append({**task, 'id': f"{task['case_id']}-runtime-{arm}-1",
                    'backbone': arm, 'service': 'qwen', 'origin': 'rq3',
                    'variant': LABELS[arm], 'model_family': 'qwen'})
    ids = [t['id'] for t in imported + local]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate native evaluation identity')
    return imported, local


def checkpoint_record(registry, stage):
    record = registry['checkpoints'][stage]
    checkpoint = Path(record['checkpoint']).resolve()
    if file_sha256(checkpoint / 'adapter_model.safetensors') != record['adapter_sha256']:
        raise ValueError('checkpoint digest differs: ' + stage)
    return {**record, 'checkpoint': str(checkpoint), 'sha256': record['adapter_sha256'],
            'training_seed': registry['seed']}


def training_coverage(path):
    counts = Counter()
    for line in path.open():
        row = json.loads(line)
        mas = (row.get('metadata') or {}).get('mas_id')
        if not mas:
            raise ValueError('training record lacks framework identity')
        counts[mas] += 1
    return {'path': str(path), 'sha256': file_sha256(path), 'rows_by_mas': dict(counts)}


def comparable_config(config):
    """Only output locations and predictor identities may differ across variants."""
    value = copy.deepcopy(config)
    value.pop('output_dir', None)
    value.pop('workspace_dir', None)
    for experiment in value['experiments']:
        experiment.pop('name', None)
    model = value['aria'].get('model', {})
    for key in ('model', 'base_url', 'tokenizer_endpoint'):
        model.pop(key, None)
    return value


def build(spec):
    repo = Path.cwd().resolve()
    source_path = Path(spec['source_manifest']).resolve()
    source = json.loads(source_path.read_text())
    if source['schema_version'] != 'aria-rq1-evaluation-v1':
        raise ValueError('unsupported source evaluation schema')
    if source['split_audit']['status'] != 'passed':
        raise ValueError('source test split audit failed')
    if file_sha256(Path(source['spec']['pool'])) != source['pool_sha256']:
        raise ValueError('source pool changed')
    # Verify the compiler/outcome interpreter used to reuse RQ1 matches its frozen version.
    source_freeze = json.loads((source_path.parent / 'code_freeze.json').read_text())
    for name in ('scripts/aria/build_rq1_eval.py', 'scripts/aria/run_rq1_eval.py',
                 'scripts/aria/report_rq1_eval.py', 'src/aria/runtime_metrics.py'):
        if file_sha256(repo / name) != source_freeze['files'][name]:
            raise ValueError('source evaluator implementation changed: ' + name)
    stamp = datetime.now(ZoneInfo('America/New_York')).strftime('%Y%m%d-%H%M%S')
    root = Path(spec.get('output_root', f'outputs/aria/rq3-{stamp}')).resolve()
    root.mkdir(parents=True, exist_ok=False)
    for sub in ('setup', 'services', 'configs', 'logs', 'data', 'figure04', 'raw/imported', 'collector'):
        (root / sub).mkdir(parents=True)
    settings = {**source['spec'], **spec, 'output_root': str(root), 'source_manifest': str(source_path),
                'source_root': str(source_path.parent), 'code_root': str(root / 'code')}
    imported, local = plan_tasks(source)
    checkpoints, coverage = {}, {}
    for model in ('qwen', 'llama'):
        registry = source['checkpoints'][model]['registry']
        if registry['seed'] != 42 or any(registry['checkpoints'][s]['loss_count'] != 16 for s in ('grpo_repair', 'grpo_gate')):
            raise ValueError('expected training seed 42 with 16 Repair and 16 Gate updates')
        checkpoints[model] = checkpoint_record(registry, 'grpo_gate')
        corpus = Path(registry.get('source_corpus', str(Path(source['spec']['models'][model]).resolve() / 'source_corpus'))) / 'sft.jsonl'
        coverage[model] = training_coverage(corpus)
    for arm, stage in STAGES.items():
        checkpoints[arm] = checkpoint_record(source['checkpoints']['qwen']['registry'], stage)
    if any(coverage[m]['rows_by_mas'].get('magentic-one', 0) == 0 for m in coverage):
        raise ValueError('training scope changed; revise the unseen-system protocol')
    service = copy.deepcopy(spec['service'])
    service['revision'] = source['services']['qwen']['revision']
    total = float(subprocess.check_output(['nvidia-smi', '-i', str(service['gpu']),
        '--query-gpu=memory.total', '--format=csv,noheader,nounits'], text=True).strip())
    vllm = {'cuda_visible_devices': str(service['gpu']), 'hf_model_name': service['model'],
        'served_model_name': service['model'], 'host': '127.0.0.1', 'port': service['port'],
        'dtype': 'bfloat16', 'max_model_len': service['context'],
        'gpu_memory_utilization': service['memory_gib'] * 1024 / total,
        'extra_args': ['--revision', service['revision'], '--enforce-eager', '--max-num-seqs',
            str(service['max_num_seqs']), '--max-num-batched-tokens', '4096', '--no-enable-prefix-caching',
            '--enable-lora', '--max-lora-rank', '32', '--max-loras', '2', '--max-cpu-loras', '4']}
    service_path = root / 'services/qwen.yml'
    service_path.write_text(yaml.safe_dump({'vllm': vllm}, sort_keys=False))
    service.update(config=str(service_path), config_sha256=file_sha256(service_path))
    templates = copy.deepcopy(source['templates'])
    for arm in STAGES:
        templates['predictor_api'][arm] = {**templates['predictor_api']['qwen'],
            'model': 'rq3-' + stamp + '-' + arm, 'base_url': f"http://127.0.0.1:{service['port']}/v1"}
    limitations = [
        'Magentic-One occurs in both target models training: its scores are seen-system diagnostics, not unseen-system RQ3.',
        'No matched 16 Repair + 16 Gate S1+3 checkpoint is registered. All four-variant Figure04 points stay null.',
        'Only training seed 42 exists; not the three-training-seed experiment in the figure specification.',
        'Fresh Qwen3-8B local host, matching active RQ1; not the original mixed-host corpus deployment.',
        'Full/control outcomes depend on the ongoing RQ1 queue; no duplicate evaluation is submitted.',
    ]
    protocol = {'mode': 'runtime', 'budget': 1, 'source_protocol_sha256': source['protocol_sha256'],
        'training_seeds': [42], 'execution_seed': settings['execution_seed'],
        'in_domain': 'all original 4890 RQ1 test records, including source replicates',
        'unseen_system': 'blocked: target models were trained on Magentic-One',
        'figure04': 'one common pool of S1, S1+2, S1+3, Full; all matched tasks in denominator',
        'bootstrap': 'benchmark/native-task clusters; paired variants; fixed training seed only',
        'limitations': limitations}
    manifest = {'schema_version': 'aria-rq3-evaluation-v1', 'spec': settings,
        'source_manifest_sha256': file_sha256(source_path), 'source_code_freeze': source_freeze,
        'services': {'qwen': service}, 'checkpoints': checkpoints, 'templates': templates,
        'training_coverage': coverage, 'split_audit': source['split_audit'], 'pool_sha256': source['pool_sha256'],
        'protocol': protocol, 'protocol_sha256': digest(protocol), 'identities': source['identities'],
        'host_config_sha256': digest({'api': templates['host_api'], 'native': settings['native_options'],
                                    'evaluation': settings['evaluation']}),
        'import_tasks': imported, 'tasks': local, 'missing_variants': ['S1+3']}
    originals = {t['case_id']: t for t in imported if t['backbone'] == 'qwen'}
    seen = set()
    for task in local:
        key = (task['mas'], task['benchmark'], task['backbone'])
        if key in seen:
            continue
        seen.add(key)
        config, _ = experiment_config(manifest, task, 1)
        original, _ = experiment_config(source, originals[task['case_id']], 1)
        if comparable_config(config) != comparable_config(original):
            raise ValueError('native protocol mismatch: ' + str(key))
        path = root / 'setup' / ('-'.join(key) + '.yml')
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        load_config(path)
    # Native execution must use the very same code as reused Full/control runs.
    code = root / 'code'
    code.mkdir()
    for name, sha in source_freeze['files'].items():
        src = Path(source['spec']['code_root']) / name
        if file_sha256(src) != sha:
            raise ValueError('RQ1 native code snapshot changed: ' + name)
        dst = code / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for mas, environment in settings['project_environments'].items():
        (code / f'.venv-{mas}').symlink_to(environment, target_is_directory=True)
    for pattern in ('scripts/aria/*rq3_eval.py', 'tests/test_rq3_eval.py', 'configs/data_views/rq3_eval.yml'):
        for src in repo.glob(pattern):
            dst = root / 'collector' / src.relative_to(repo)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    manifest['collector_files'] = {str(p.relative_to(root / 'collector')): file_sha256(p)
        for p in (root / 'collector').rglob('*') if p.is_file()}
    atomic_write_json(root / 'code_freeze.json', source_freeze)
    atomic_write_json(root / 'setup/source_manifest.json', source)
    atomic_write_json(root / 'setup/training_coverage.json', coverage)
    atomic_write_json(root / 'setup/split_audit.json', source['split_audit'])
    atomic_write_json(root / 'setup/checkpoints.json', checkpoints)
    atomic_write_json(root / 'manifest.json', manifest)
    (root / 'launch.yml').write_text(yaml.safe_dump({'manifest': str(root / 'manifest.json'),
        'manifest_sha256': file_sha256(root / 'manifest.json')}))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', required=True, type=Path)
    manifest = build(yaml.safe_load(parser.parse_args().config.read_text()))
    from scripts.aria.report_rq3_eval import write_report
    write_report(manifest, {})
    print(manifest['spec']['output_root'])


if __name__ == '__main__':
    main()
