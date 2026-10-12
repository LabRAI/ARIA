"""Execute the frozen RQ2 YAML chain sequentially and retain verified checkpoints."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

import yaml

from common.artifacts import atomic_write_json
from scripts.aria.verify_training_checkpoint import file_sha256, verify


def accept(job):
    accepted = verify({'name': job['id'], 'output_dir': job['output_dir'], 'stages': [job['stage']],
                       'foundation_model': job['foundation_model'], 'lora_rank': job['lora_rank'],
                       'expected_grpo_updates': job['expected_updates']})
    report = json.loads((Path(job['checkpoint']).parent / 'training_report.json').read_text())
    if report.get('optimizer_steps') != job['expected_updates']:
        raise ValueError('optimizer updates differ from reduced RQ2 budget')
    if job['stage'] == 'grpo_gate' and report.get('stage3', {}).get('gate_policy') != 'contextual':
        raise ValueError('Full requires a contextual gate policy')
    return {**accepted, 'updates': job['expected_updates'], 'checkpoint': job['checkpoint']}


def run(spec):
    root = Path(spec['output_root']).resolve()
    manifest = json.loads((root / 'manifest.json').read_text())
    with (root / '.training-writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for path, digest in manifest['frozen_sha256'].items():
            if file_sha256(Path(path)) != digest:
                raise ValueError(f'frozen RQ2 input changed: {path}')
        registry = json.loads((root / 'checkpoint_registry.json').read_text())
        environment = str(Path(manifest['spec']['training_environment']).resolve())
        env = {**os.environ, 'UV_PROJECT_ENVIRONMENT': environment, 'UV_NO_SYNC': '1', 'PYTHONUNBUFFERED': '1'}
        for job in manifest['jobs']:
            if job['id'] in registry['checkpoints']:
                accept(job)  # Never trust a stale success marker alone.
                continue
            state = {'status': 'running', 'job': job['id'], 'started_unix': time.time()}
            atomic_write_json(root / 'training_state.json', state)
            log = root / 'logs' / (job['id'] + '.log')
            log.parent.mkdir(parents=True, exist_ok=True)
            try:
                with log.open('a') as handle:
                    subprocess.run(['uv', 'run', '--no-sync', 'python', 'main.py', '-c', job['config']],
                                   env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
                registry['checkpoints'][job['id']] = accept(job)
                registry['status'] = 'partially_verified'
                atomic_write_json(root / 'checkpoint_registry.json', registry)
            except Exception as exc:
                atomic_write_json(root / 'training_state.json', {**state, 'status': 'failed_not_accepted', 'error': str(exc)})
                raise
        registry['status'] = 'training_verified_eval_pending'
        atomic_write_json(root / 'checkpoint_registry.json', registry)
        atomic_write_json(root / 'training_state.json', {'status': registry['status']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', type=Path, required=True)
    run(yaml.safe_load(parser.parse_args().config.read_text()))


if __name__ == '__main__':
    main()
