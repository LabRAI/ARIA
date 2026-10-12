"""Durable, GPU-aware RQ1 queue. All experiments execute main.py YAML jobs."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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

import yaml

from common.artifacts import atomic_write_json
from masfm.config import load_config
from scripts.aria.build_rq1_eval import experiment_config
from scripts.aria.report_rq1_eval import load_result, outcome, read_json, write_report
from scripts.aria.verify_training_checkpoint import file_sha256

TERMINAL = {'resolved', 'unresolved', 'infrastructure_invalid', 'skipped_after_success'}


def start_ticks(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except FileNotFoundError:
        return None


def alive(identity):
    return bool(identity and identity.get('start_ticks') and start_ticks(identity['pid']) == identity['start_ticks'])


def request_json(url, payload=None):
    req = Request(url, data=json.dumps(payload).encode() if payload is not None else None,
                  headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=30) as response:
        body = response.read().decode()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            if payload is None:
                raise
            return {'message': body}


class Servers:
    def __init__(self, manifest):
        self.manifest, self.spec = manifest, manifest['spec']
        self.root = Path(self.spec['output_root'])
        self.identities = {k: read_json(self.root / 'services' / f'{k}.identity.json', {}) for k in manifest['services']}
        self.ready, self.starting, self.errors, self.adapters = set(), {}, {}, set()

    def refresh(self):
        for key, identity in self.identities.items():
            if not alive(identity):
                if key in self.starting or key in self.ready:
                    self.errors[key] = 'owned service exited; inspect service log'
                self.ready.discard(key)
                self.starting.pop(key, None)
                continue
            if key in self.ready:
                continue
            service = self.manifest['services'][key]
            try:
                models = request_json(f"http://127.0.0.1:{service['port']}/v1/models")['data']
                if service['model'] not in {m['id'] for m in models}:
                    raise ValueError('wrong served base model')
                self.ready.add(key)
                self.starting.pop(key, None)
            except (OSError, ValueError):
                self.starting.setdefault(key, time.time())
                if time.time() - self.starting[key] > 900:
                    self.errors[key] = 'service readiness timeout'

    def ensure(self, key):
        if key in self.ready or key in self.starting or key in self.errors:
            return
        service = self.manifest['services'][key]
        if any(self.manifest['services'][k]['gpu'] == service['gpu'] for k in self.starting):
            return
        free = float(subprocess.check_output(['nvidia-smi', '-i', str(service['gpu']),
            '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip()) / 1024
        if free < service['memory_gib'] + .5:
            return
        if file_sha256(Path(service['config'])) != service['config_sha256']:
            raise ValueError('serving YAML changed')
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', service['port']))
        env = {**os.environ, 'UV_PROJECT_ENVIRONMENT': self.spec['serving_environment'],
               'HF_HUB_OFFLINE': '1', 'VLLM_ALLOW_RUNTIME_LORA_UPDATING': '1', 'PYTHONUNBUFFERED': '1'}
        env.pop('VLLM_API_KEY', None)
        with (self.root / 'services' / f'{key}.log').open('a') as log:
            process = subprocess.Popen(['uv', 'run', '--no-sync', 'python',
                str(Path(self.spec['code_root']) / 'scripts/aria/serve_baseline_config.py'), '-c', service['config']],
                cwd=self.spec['code_root'], env=env, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
        self.identities[key] = {'pid': process.pid, 'start_ticks': start_ticks(process.pid), 'config': service['config']}
        atomic_write_json(self.root / 'services' / f'{key}.identity.json', self.identities[key])
        self.starting[key] = time.time()

    def load_adapter(self, key):
        if key in self.adapters:
            return
        checkpoint = self.manifest['checkpoints'][key]
        if file_sha256(Path(checkpoint['checkpoint']) / 'adapter_model.safetensors') != checkpoint['sha256']:
            raise ValueError('checkpoint digest mismatch')
        url = f"http://127.0.0.1:{self.manifest['services'][key]['port']}/v1"
        alias = 'rq1-trained-' + key
        models = request_json(url + '/models')['data']
        existing = next((m for m in models if m['id'] == alias), None)
        if existing:
            if Path(existing['root']).resolve() != Path(checkpoint['checkpoint']).resolve():
                raise ValueError('adapter alias collision')
        else:
            request_json(url + '/load_lora_adapter', {'lora_name': alias, 'lora_path': checkpoint['checkpoint']})
        self.adapters.add(key)

    def stop(self):
        for identity in self.identities.values():
            if alive(identity):
                os.killpg(identity['pid'], signal.SIGTERM)


def recover(task, root, attempts):
    journals = []
    for attempt in range(1, attempts + 1):
        stem = root / 'logs' / f"{task['id']}--attempt-{attempt}"
        launch = read_json(Path(str(stem) + '.launch.json'), {})
        if alive(launch):
            raise RuntimeError('native child still active: ' + task['id'])
        journal = read_json(Path(str(stem) + '.outcome.json'))
        if journal is None and launch.get('result'):
            result = load_result(launch['result'], task['mode'])
            if result is not None:
                metrics = outcome(result, task)
                journal = {'status': metrics['status'], 'metrics': metrics, 'result': launch['result'],
                           'attempt': attempt, 'recovered': True}
        if journal:
            journals.append(journal)
    if not journals:
        return None
    chosen = next((j for j in journals if j['status'] == 'resolved'), journals[-1])
    value = {**chosen, 'attempts': journals}
    if chosen['status'] != 'resolved' and len(journals) < attempts:
        value['status'] = 'pending'
    return value


def execute(manifest, task, attempt, on_start):
    spec, root = manifest['spec'], Path(manifest['spec']['output_root'])
    config, result_path = experiment_config(manifest, task, attempt)
    path = root / 'configs' / f"{task['id']}--attempt-{attempt}.yml"
    contents = yaml.safe_dump(config, sort_keys=False)
    if path.exists() and path.read_text() != contents:
        raise ValueError('existing candidate YAML differs')
    path.write_text(contents)
    load_config(path)
    code = Path(spec['code_root'])
    environment = spec['default_environment'] if task['mode'] == 'posthoc' else spec['project_environments'][task['mas']]
    env = {**os.environ, 'UV_PROJECT_ENVIRONMENT': environment, 'UV_NO_SYNC': '1',
           'PYTHONPATH': f'{code / "src"}:{code}', 'PYTHONUNBUFFERED': '1', 'CUDA_VISIBLE_DEVICES': '',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
           'PYTHONHASHSEED': str(spec['execution_seed'])}
    log_path = root / 'logs' / f"{task['id']}--attempt-{attempt}.log"
    started = time.time()
    with log_path.open('a') as log:
        process = subprocess.Popen(['uv', 'run', '--no-sync', 'python', str(code / 'main.py'), '-c', str(path)],
            cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        launch = {'pid': process.pid, 'start_ticks': start_ticks(process.pid), 'started_unix': started,
                  'attempt': attempt, 'result': str(result_path), 'config_sha256': file_sha256(path)}
        atomic_write_json(log_path.with_suffix('.launch.json'), launch)
        on_start(launch)
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
    metrics = outcome(load_result(result_path, task['mode']), task)
    result = {'status': metrics['status'], 'reason': metrics.get('reason'), 'metrics': metrics,
        'attempt': attempt, 'result': str(result_path), 'returncode': returncode, 'wall_seconds': time.time() - started,
        'started_unix': started, 'finished_unix': time.time(), 'config_sha256': launch['config_sha256']}
    atomic_write_json(log_path.with_suffix('.outcome.json'), result)
    return result


def can_run(task, states, ready, counts, spec):
    if states.get(task['id'], {}).get('status') in TERMINAL | {'running'}:
        return False
    if task['service'] not in ready or 'host' not in ready:
        return False
    if counts[task['service']] >= spec['workers_per_service']:
        return False
    if task['previous_id'] and states.get(task['previous_id'], {}).get('status') not in TERMINAL:
        return False
    return True


def run(manifest):
    spec, root = manifest['spec'], Path(manifest['spec']['output_root'])
    with (root / '.writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = read_json(root / 'code_freeze.json')['files']
        for path, sha in frozen.items():
            if file_sha256(Path(spec['code_root']) / path) != sha:
                raise ValueError('code snapshot changed: ' + path)
        states = read_json(root / 'state.json', {}).get('tasks', {})
        for task in manifest['tasks']:
            recovered = recover(task, root, spec['max_infrastructure_attempts'])
            if recovered:
                states[task['id']] = recovered
            elif states.get(task['id'], {}).get('status') == 'running':
                if alive(states[task['id']]):
                    raise RuntimeError('orphan candidate still running')
                states[task['id']]['status'] = 'pending'
        servers, active = Servers(manifest), {}
        state_lock = threading.Lock()
        # Rotate cells, panels and models while prioritizing main-table candidates.
        tasks = sorted(manifest['tasks'], key=lambda t: (t['candidate'], t['sample_index'],
            0 if t['method'] == 'control' else 1, t['mode'], t['backbone']))
        last_report, last_persist = 0, 0

        def persist(status):
            with state_lock:
                atomic_write_json(root / 'state.json', {'status': status, 'pid': os.getpid(),
                    'updated_unix': time.time(), 'tasks': states, 'ready_services': sorted(servers.ready),
                    'starting_services': servers.starting, 'service_errors': servers.errors,
                    'counts': dict(Counter(states.get(t['id'], {}).get('status', 'pending') for t in tasks)),
                    'active_jobs': [t['id'] for t in active.values()]})

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
                    for task in tasks:
                        if task['mode'] != 'posthoc' or not task['previous_id'] or states.get(task['id'], {}).get('status') in TERMINAL | {'running'}:
                            continue
                        prior = states.get(task['previous_id'], {})
                        if prior.get('status') == 'skipped_after_success' or (prior.get('status') == 'resolved' and prior['metrics']['correct']):
                            states[task['id']] = {'status': 'skipped_after_success', 'previous_id': task['previous_id']}
                    if all(states.get(t['id'], {}).get('status') in TERMINAL for t in tasks) and not active:
                        persist('terminal')
                        write_report(manifest, states)
                        break
                    servers.refresh()
                    for key in ('host', 'qwen', 'llama'):
                        servers.ensure(key)
                    for key in ('qwen', 'llama'):
                        if key in servers.ready:
                            servers.load_adapter(key)
                    disk_ok = shutil.disk_usage(root).free / 1024**3 >= spec['minimum_disk_free_gib']
                    if disk_ok:
                        counts = Counter(t['service'] for t in active.values())
                        swe_count = sum(t['benchmark'] == 'swe-bench-lite' for t in active.values())
                        for task in tasks:
                            if len(active) >= spec['workers']:
                                break
                            if task['benchmark'] == 'swe-bench-lite' and swe_count >= spec['swe_workers']:
                                continue
                            if not can_run(task, states, servers.ready, counts, spec):
                                continue
                            old = states.get(task['id'], {})
                            attempt = len(old.get('attempts', [])) + 1
                            states[task['id']] = {**old, 'status': 'running', 'attempt': attempt}
                            def on_start(identity, key=task['id']):
                                with state_lock:
                                    states[key].update(identity)
                            active[executor.submit(execute, manifest, task, attempt, on_start)] = task
                            counts[task['service']] += 1
                            swe_count += task['benchmark'] == 'swe-bench-lite'
                    status = 'running' if active else 'disk_reserve_wait' if not disk_ok else 'waiting_for_gpu_services'
                    if time.time() - last_persist >= 10:
                        persist(status)
                        last_persist = time.time()
                    if time.time() - last_report >= spec['report_seconds']:
                        write_report(manifest, states)
                        last_report = time.time()
                    time.sleep(spec['poll_seconds'])
            except BaseException as exc:
                persist('coordinator_error')
                atomic_write_json(root / 'coordinator_error.json', {'error': f'{type(exc).__name__}: {exc}',
                    'time_unix': time.time(), 'active_children_preserved': True})
                raise
            finally:
                if not active:
                    servers.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', type=Path, required=True)
    launch = yaml.safe_load(parser.parse_args().config.read_text())
    if file_sha256(Path(launch['manifest'])) != launch['manifest_sha256']:
        raise ValueError('frozen manifest digest differs')
    run(read_json(launch['manifest']))


if __name__ == '__main__':
    main()
