"""Resume native RQ2 jobs with shared controls and bounded parallel GPU services."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import threading
import time
from urllib.request import Request, urlopen
from urllib.error import HTTPError

import yaml

from common.artifacts import atomic_write_json
from scripts.aria.build_rq2_eval import checkpoint_identity
from scripts.aria.report_rq2_eval import read_json, resolution, write_report
from scripts.aria.verify_training_checkpoint import file_sha256


def start_ticks(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if stat[0] == 'Z' else stat[19]
    except FileNotFoundError:
        return None


def owned_alive(identity):
    return bool(identity and identity.get('start_ticks') and start_ticks(identity['pid']) == identity['start_ticks'])


def request_json(url, payload=None):
    req = Request(url, data=json.dumps(payload).encode() if payload is not None else None,
                  headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=20) as response:
        body = response.read().decode()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            if payload is None:
                raise
            return {'message': body}  # vLLM's successful LoRA mutation returns plain text.


class Servers:
    def __init__(self, manifest):
        self.manifest = manifest
        self.root = Path(manifest['spec']['output_root'])
        self.services = manifest['services']
        self.identities = {key: read_json(self.root / 'services' / f'{key}.identity.json', {}) for key in self.services}
        self.ready = set()
        self.starting = {}
        self.errors = {}
        self.adapters = set()
        self.checkpoints = read_json(self.root / 'setup/checkpoint_identities.json', manifest['checkpoint_identities'])

    def refresh(self):
        for key, identity in self.identities.items():
            if not owned_alive(identity):
                self.ready.discard(key)
                if key in self.starting:
                    self.errors[key] = 'server process exited; inspect services/' + key + '.log'
                    del self.starting[key]
                continue
            if key in self.ready:
                continue
            try:
                models = request_json(f"http://127.0.0.1:{self.services[key]['port']}/v1/models")
                if self.services[key]['model'] in {v['id'] for v in models['data']}:
                    self.ready.add(key)
                    self.starting.pop(key, None)
            except (OSError, ValueError):
                self.starting.setdefault(key, time.time())
                if time.time() - self.starting[key] > 600:
                    self.errors[key] = 'server readiness timeout'

    def ensure(self, key):
        if key in self.ready or key in self.starting or key in self.errors:
            return
        service = self.services[key]
        if any(self.services[k]['gpu'] == service['gpu'] for k in self.starting):
            return  # Do not overlap vLLM memory profiling on one device.
        free = subprocess.check_output(['nvidia-smi', '-i', str(service['gpu']),
            '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True)
        if float(free.strip()) / 1024 < service['memory_gib'] + .4:
            return
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', service['port']))
        env = {**os.environ, 'UV_PROJECT_ENVIRONMENT': self.manifest['spec']['serving_environment'],
               'HF_HUB_OFFLINE': '1', 'VLLM_ALLOW_RUNTIME_LORA_UPDATING': '1', 'PYTHONUNBUFFERED': '1'}
        env.pop('VLLM_API_KEY', None)
        with (self.root / 'services' / f'{key}.log').open('a') as log:
            process = subprocess.Popen(['uv', 'run', '--no-sync', 'python',
                str(Path(self.manifest['spec']['code_root']) / 'scripts/aria/serve_baseline_config.py'),
                '-c', service['config']], cwd=self.manifest['spec']['code_root'], env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        identity = {'pid': process.pid, 'start_ticks': start_ticks(process.pid), 'config': service['config']}
        self.identities[key] = identity
        self.starting[key] = time.time()
        atomic_write_json(self.root / 'services' / f'{key}.identity.json', identity)

    def load_adapter(self, key):
        if key in self.adapters:
            return True
        identity = self.checkpoints.get(key)
        if not identity:
            return False
        if file_sha256(Path(identity['checkpoint']) / 'adapter_model.safetensors') != identity['sha256']:
            raise ValueError('registered checkpoint changed before loading')
        models = request_json(f"http://127.0.0.1:{self.services[key]['port']}/v1/models")['data']
        existing = next((m for m in models if m['id'] == f'rq2-full-{key}'), None)
        if existing is not None:
            if Path(existing['root']).resolve() != Path(identity['checkpoint']).resolve():
                raise ValueError('existing LoRA alias refers to a different checkpoint')
            self.adapters.add(key)
            return True
        try:
            request_json(f"http://127.0.0.1:{self.services[key]['port']}/v1/load_lora_adapter",
                         {'lora_name': f'rq2-full-{key}', 'lora_path': identity['checkpoint']})
        except HTTPError as exc:
            # An already-loaded identical adapter is idempotent, including an
            # overlapping readiness probe. Never accept a conflicting alias.
            models = request_json(f"http://127.0.0.1:{self.services[key]['port']}/v1/models")['data']
            existing = next((m for m in models if m['id'] == f'rq2-full-{key}'), None)
            if exc.code != 400 or existing is None or Path(existing['root']).resolve() != Path(identity['checkpoint']).resolve():
                raise
        self.adapters.add(key)
        return True

    def stop(self, key):
        identity = self.identities.get(key)
        if owned_alive(identity):
            os.killpg(identity['pid'], signal.SIGTERM)
        self.ready.discard(key)
        self.starting.pop(key, None)
        self.adapters.discard(key)


def can_run(task, states, ready, checkpoints, counts, cap):
    if states.get(task['id'], {}).get('status') in {'running', 'resolved', 'infrastructure_invalid', 'unresolved'}:
        return False
    control = states.get(task['control_id'], {}).get('status')
    if task['method'] != 'control' and control not in {'resolved', 'infrastructure_invalid', 'unresolved'}:
        return False
    if task['service'] not in ready or counts[task['service']] >= cap:
        return False
    return task['method'] != 'aria' or bool(checkpoints.get(task['backbone']))


def execute(task, attempt, manifest, on_start):
    spec = manifest['spec']
    entry = task['configs'][attempt - 1]
    config = Path(entry['path'])
    if file_sha256(config) != entry['sha256']:
        raise ValueError('frozen evaluation config changed: ' + str(config))
    if shutil.disk_usage(spec['output_root']).free / 1024**3 < spec['minimum_disk_free_gib']:
        raise RuntimeError('artifact disk reserve reached')
    code = Path(spec['code_root'])
    env = {**os.environ, 'UV_PROJECT_ENVIRONMENT': spec['project_environments'][task['mas']],
           'PYTHONPATH': f"{code / 'src'}:{code}", 'PYTHONUNBUFFERED': '1',
           'PYTHONHASHSEED': str(spec['execution_seed']), 'CUDA_VISIBLE_DEVICES': ''}
    log_path = Path(spec['output_root']) / 'logs' / f"{task['id']}--attempt-{attempt}.log"
    log_path.parent.mkdir(exist_ok=True)
    started = time.time()
    with log_path.open('a') as log:
        process = subprocess.Popen(['uv', 'run', '--no-sync', 'python', str(code / 'main.py'), '-c', str(config)],
                                   cwd=code, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        identity = {'pid': process.pid, 'start_ticks': start_ticks(process.pid), 'started_unix': started,
                    'attempt': attempt, 'log': str(log_path)}
        atomic_write_json(log_path.with_suffix('.launch.json'), identity)
        on_start(identity)
        try:
            returncode = process.wait(timeout=spec['process_timeout_seconds'])
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            returncode = -signal.SIGTERM
    result_path = Path(entry['result'])
    result = read_json(result_path)
    if result is None:
        result_path = log_path.with_suffix('.harness_failure.json')
        result = {'status': 'failed', 'error': f'native subprocess produced no result.json (exit {returncode})',
                  'failure_origin': 'evaluation_harness', 'log': str(log_path)}
        atomic_write_json(result_path, result)
    status, reason = resolution(result, task['method'])
    outcome = {'status': status, 'reason': reason, 'result': str(result_path), 'returncode': returncode,
               'wall_seconds': time.time() - started, 'started_unix': started,
               'finished_unix': time.time(), 'attempt': attempt, 'log': str(log_path)}
    atomic_write_json(log_path.with_suffix('.outcome.json'), outcome)
    return outcome


def recover_outcomes(task, root):
    """First valid attempt wins, including an incorrect benchmark answer."""
    journals = []
    for attempt, config in enumerate(task['configs'], 1):
        stem = root / 'logs' / f"{task['id']}--attempt-{attempt}"
        launch = read_json(Path(str(stem) + '.launch.json'), {})
        if owned_alive(launch):
            raise RuntimeError('native child still active; wait before resuming: ' + task['id'])
        outcome = read_json(Path(str(stem) + '.outcome.json'))
        if outcome is None:
            result = read_json(config['result'])
            if result is not None:
                status, reason = resolution(result, task['method'])
                outcome = {'attempt': attempt, 'status': status, 'reason': reason,
                           'result': config['result'], 'wall_seconds': None,
                           'recovered_from_native_artifact': True}
        if outcome:
            journals.append(outcome)
    if not journals:
        return None
    selected = next((j for j in journals if j['status'] == 'resolved'), journals[-1])
    recovered = {**selected, 'attempts': journals}
    if selected['status'] != 'resolved' and len(journals) < len(task['configs']):
        recovered['status'] = 'pending'
    return recovered


def run(manifest):
    spec = manifest['spec']
    root = Path(spec['output_root'])
    with (root / '.writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = read_json(root / 'state.json', {})
        states = previous.get('tasks', {})
        # Recover durable task journals even if the coordinator died between
        # worker completion and the next queue-state write.
        for task in manifest['tasks']:
            recovered = recover_outcomes(task, root)
            if recovered is not None:
                states[task['id']] = recovered
        # A resumed coordinator must never duplicate still-running orphan jobs.
        for key, value in states.items():
            if value.get('status') == 'running':
                if owned_alive(value):
                    raise RuntimeError('native child still active; wait for its exit before resuming: ' + key)
                task = next(t for t in manifest['tasks'] if t['id'] == key)
                result_path = task['configs'][value['attempt'] - 1]['result']
                result = read_json(result_path)
                status, reason = resolution(result, task['method'])
                if status == 'resolved':
                    value.update(status=status, result=result_path, reason=reason)
                else:
                    value.update(status='pending')
        servers = Servers(manifest)
        active = {}
        state_lock = threading.Lock()
        tasks = sorted(manifest['tasks'], key=lambda t: (t['sample_index'],
            ['reference', 'unseen_benchmark', 'unseen_system'].index(t['axis']),
            0 if t['method'] == 'control' else 1, t['id']))
        last_report = 0
        terminal = {'resolved', 'infrastructure_invalid', 'unresolved'}

        def persist(status='running'):
            with state_lock:
                atomic_write_json(root / 'state.json', {'status': status, 'pid': os.getpid(),
                    'updated_unix': time.time(), 'planned': len(tasks), 'tasks': states,
                    'counts': dict(Counter(states.get(t['id'], {}).get('status', 'pending') for t in tasks)),
                    'ready_services': sorted(servers.ready), 'starting_services': servers.starting,
                    'service_errors': servers.errors, 'active_jobs': [t['id'] for t in active.values()]})

        with ThreadPoolExecutor(max_workers=spec['workers']) as executor:
            try:
                while True:
                    for future in list(active):
                        if not future.done():
                            continue
                        task = active.pop(future)
                        result = future.result()
                        history = states[task['id']].get('attempts', []) + [result]
                        states[task['id']] = {**result, 'attempts': history}
                        if result['status'] != 'resolved' and len(history) < spec['max_infrastructure_attempts']:
                            states[task['id']]['status'] = 'pending'
                        persist()
                    servers.refresh()
                    remaining = [t for t in tasks if states.get(t['id'], {}).get('status') not in terminal]
                    if not remaining and not active:
                        persist('completed')
                        write_report(manifest, states)
                        break
                    required = {t['service'] for t in remaining} | {'host'}
                    for key in list(servers.ready):
                        if key not in required:
                            servers.stop(key)
                    for key in ['host', 'qwen25-7b', 'llama31-8b', 'qwen25-1.5b', 'qwen25-3b', 'qwen25-14b']:
                        if key in required:
                            servers.ensure(key)
                    if time.time() - last_report >= spec['report_seconds']:
                        for key in servers.checkpoints:
                            if servers.checkpoints[key] is None:
                                servers.checkpoints[key] = checkpoint_identity(Path(spec['training_root']), key, 42)
                        atomic_write_json(root / 'setup/checkpoint_identities.json', servers.checkpoints)
                        persist()
                        write_report(manifest, states)
                        last_report = time.time()
                    counts = Counter(t['service'] for t in active.values())
                    if 'host' in servers.ready:
                        for task in tasks:
                            if len(active) >= spec['workers']:
                                break
                            if not can_run(task, states, servers.ready, servers.checkpoints, counts, spec['workers_per_predictor']):
                                continue
                            if task['method'] == 'aria' and not servers.load_adapter(task['backbone']):
                                continue
                            old = states.get(task['id'], {})
                            attempt = len(old.get('attempts', [])) + 1
                            states[task['id']] = {**old, 'status': 'running', 'attempt': attempt}
                            def on_start(identity, key=task['id']):
                                with state_lock:
                                    states[key].update(identity)
                            future = executor.submit(execute, task, attempt, manifest, on_start)
                            active[future] = task
                            counts[task['service']] += 1
                    persist('waiting_for_checkpoint_or_gpu' if not active else 'running')
                    time.sleep(spec['poll_seconds'])
            except BaseException as exc:
                persist('coordinator_error')
                atomic_write_json(root / 'coordinator_error.json', {'error': f'{type(exc).__name__}: {exc}',
                    'time_unix': time.time(), 'active_children_preserved': True})
                raise
            finally:
                # Stop only services owned by this evaluation, after native futures drain.
                if not active:
                    for key in list(servers.ready):
                        servers.stop(key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', type=Path, required=True)
    launch = yaml.safe_load(parser.parse_args().config.read_text())
    run(read_json(launch['manifest']))


if __name__ == '__main__':
    main()
