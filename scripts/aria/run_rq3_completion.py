"""Evaluate each RQ3 checkpoint as it is accepted, sharing existing GPU services."""
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
import subprocess
import threading
import time
from urllib.request import Request, urlopen

import yaml

from common.artifacts import atomic_write_json
from scripts.aria.report_rq1_eval import read_json
from scripts.aria.report_rq3_completion import write_report
from scripts.aria.run_rq1_eval import alive, execute, recover, request_json, start_ticks
from scripts.aria.verify_training_checkpoint import file_sha256
from scripts.aria.eval_continuation import execute_or_adopt, recover_for_adoption

TERMINAL = {'resolved', 'unresolved', 'infrastructure_invalid'}


def import_states(manifest, states):
    root = Path(manifest['spec']['output_root'])
    sources = {s: read_json(Path(s) / 'state.json', {}).get('tasks', {}) for s in manifest['source_roots']}
    for task in manifest['tasks']:
        source = task['origin']
        if not source or states.get(task['id'], {}).get('status') in TERMINAL | {'running'}:
            continue
        original = sources[source['root']].get(source['id'], {})
        if original.get('status') != 'resolved':
            if original.get('status') in TERMINAL:
                states.setdefault(task['id'], {})['source_terminal_invalid'] = True
            continue
        path = Path(original['result'])
        destination = root / 'raw/imported' / (task['id'] + '.json')
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(path, destination)
        states[task['id']] = {**original, 'result': str(destination), 'source_provenance': {
            **source, 'result': str(path), 'sha256': file_sha256(destination)}, 'imported': True}


class Services:
    def __init__(self, manifest):
        self.manifest = manifest
        spec = manifest['spec']
        first = read_json(Path(spec['rq1_root']) / 'manifest.json')
        third = read_json(Path(spec['rq3_source_root']) / 'manifest.json')
        self.services = {'host': first['services']['host'], 'llama': first['services']['llama'],
                         'qwen': third['services']['qwen']}
        self.roots = {'host': Path(spec['rq1_root']), 'llama': Path(spec['rq1_root']),
                      'qwen': Path(spec['rq3_source_root'])}
        self.root = Path(spec['output_root'])
        self.models, self.errors, self.starting = {}, {}, {}
        self.owned = read_json(self.root / 'services/adopted.json', {})

    def refresh(self):
        for key, service in self.services.items():
            url = f"http://127.0.0.1:{service['port']}/v1"
            try:
                models = request_json(url + '/models')['data']
                if service['model'] not in {m['id'] for m in models}:
                    raise ValueError('unexpected base model at shared endpoint')
                self.models[key] = {m['id']: m for m in models}
                self.errors.pop(key, None)
                self.starting.pop(key, None)
                continue
            except OSError as exc:
                self.models.pop(key, None)
                self.errors[key] = str(exc)
            if alive(self.owned.get(key)):
                continue
            prior = read_json(self.roots[key] / 'services' / f'{key}.identity.json', {})
            if alive(prior):
                continue  # Its owner is still booting/recovering it.
            owner_state = read_json(self.roots[key] / 'state.json', {})
            owner_pid = owner_state.get('pid')
            owner_active = bool(owner_pid and Path(f'/proc/{owner_pid}').exists())
            if owner_active:
                if key != 'qwen':
                    continue
                source_manifest = read_json(self.roots[key] / 'manifest.json')
                source_states = owner_state.get('tasks', {})
                if not all(source_states.get(t['id'], {}).get('status') in TERMINAL for t in source_manifest['tasks']):
                    continue
            gpu = service['gpu']
            if any(self.services[k]['gpu'] == gpu for k in self.starting):
                continue
            free = float(subprocess.check_output(['nvidia-smi', '-i', str(gpu),
                '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).strip()) / 1024
            if free < service['memory_gib'] + .5:
                continue
            code = Path(self.manifest['spec']['code_root'])
            env = {**os.environ, 'UV_PROJECT_ENVIRONMENT': self.manifest['spec']['serving_environment'],
                   'VLLM_ALLOW_RUNTIME_LORA_UPDATING': '1', 'HF_HUB_OFFLINE': '1', 'OMP_NUM_THREADS': '1',
                   'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
            env.pop('VLLM_API_KEY', None)
            with (self.root / 'services' / f'{key}.log').open('a') as log:
                p = subprocess.Popen(['uv', 'run', '--no-sync', 'python', str(code / 'scripts/aria/serve_baseline_config.py'),
                    '-c', service['config']], cwd=code, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            self.owned[key] = {'pid': p.pid, 'start_ticks': start_ticks(p.pid)}
            self.starting[key] = time.time()
            atomic_write_json(self.root / 'services/adopted.json', self.owned)

    def load(self, task, registry):
        if task['method'] == 'control':
            return True
        checkpoint = registry[task['checkpoint_id']]
        if checkpoint['status'] != 'verified':
            return False
        key = task['service']
        alias = self.manifest['templates']['predictor_api'][task['backbone']]['model']
        existing = self.models.get(key, {}).get(alias)
        if existing:
            if Path(existing['root']).resolve() != Path(checkpoint['checkpoint']).resolve():
                raise ValueError('checkpoint alias collision')
            return True
        if file_sha256(Path(checkpoint['checkpoint']) / 'adapter_model.safetensors') != checkpoint['sha256']:
            raise ValueError('accepted checkpoint changed')
        service = self.services[key]
        request = Request(f"http://127.0.0.1:{service['port']}/v1/load_lora_adapter",
            data=json.dumps({'lora_name': alias, 'lora_path': checkpoint['checkpoint']}).encode(),
            headers={'Content-Type': 'application/json'})
        try:
            with urlopen(request, timeout=self.manifest['spec']['adapter_load_timeout_seconds']) as response:
                response.read()
        except OSError as exc:
            self.errors[alias] = str(exc)
            return False
        self.models[key][alias] = {'id': alias, 'root': checkpoint['checkpoint']}
        self.errors.pop(alias, None)
        return True


def run(manifest):
    spec, root = manifest['spec'], Path(manifest['spec']['output_root'])
    with (root / '.writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, sha in manifest['native_code_files'].items():
            if file_sha256(Path(spec['code_root']) / name) != sha:
                raise ValueError('native code snapshot changed')
        selected_ids = {t['id'] for t in manifest['tasks']}
        states = {key: value for key, value in read_json(root / 'state.json', {}).get('tasks', {}).items()
                  if key in selected_ids}
        for task in manifest['tasks']:
            if states.get(task['id'], {}).get('imported'):
                continue
            recovered = recover_for_adoption(task, root, spec['max_infrastructure_attempts'])
            if recovered:
                states[task['id']] = recovered
            elif states.get(task['id'], {}).get('status') == 'running':
                if alive(states[task['id']]):
                    raise RuntimeError('native child still alive')
                states[task['id']]['status'] = 'pending'
        services, active = Services(manifest), {}
        state_lock = threading.Lock()
        last_import, last_report, last_persist, last_services = 0, 0, 0, 0
        tasks = sorted(manifest['tasks'], key=lambda t: (t['training_seed'] != 42, t['sample_index'],
            t['method'] != 'control', t['training_seed'], t['axis'], t['backbone']))

        def persist(status):
            with state_lock:
                atomic_write_json(root / 'state.json', {'status': status, 'pid': os.getpid(),
                    'updated_unix': time.time(), 'tasks': states, 'active_jobs': [t['id'] for t in active.values()],
                    'service_errors': services.errors, 'ready_services': sorted(services.models),
                    'counts': dict(Counter(states.get(t['id'], {}).get('status', 'pending') for t in tasks))})

        with ThreadPoolExecutor(max_workers=spec['native_workers']) as executor:
            try:
                while True:
                    for future in list(active):
                        if not future.done():
                            continue
                        task = active.pop(future)
                        result = future.result()
                        history = states[task['id']].get('attempts', []) + [result]
                        states[task['id']] = {**states[task['id']], **result, 'attempts': history}
                        if result['status'] != 'resolved' and len(history) < spec['max_infrastructure_attempts']:
                            states[task['id']]['status'] = 'pending'
                    if time.time() - last_import > 60:
                        import_states(manifest, states)
                        registry = read_json(root / 'checkpoint_registry.json')
                        last_import = time.time()
                    if all(states.get(t['id'], {}).get('status') in TERMINAL for t in tasks):
                        persist('terminal')
                        write_report(manifest, states)
                        break
                    if time.time() - last_services > 20:
                        services.refresh()
                        last_services = time.time()
                    disk_ok = shutil.disk_usage(root).free >= spec['minimum_disk_free_gib'] * 1024**3
                    if disk_ok and 'host' in services.models:
                        counts = Counter(t['service'] for t in active.values())
                        swe = sum(t['benchmark'] == 'swe-bench-lite' for t in active.values())
                        for task in tasks:
                            if len(active) >= spec['native_workers']:
                                break
                            old = states.get(task['id'], {})
                            if old.get('status') in TERMINAL | {'running'}:
                                continue
                            if task['origin'] and not old.get('source_terminal_invalid'):
                                continue  # A different queue owns this pending source execution.
                            if task['service'] not in services.models or counts[task['service']] >= spec['native_workers_per_service']:
                                continue
                            if task['benchmark'] == 'swe-bench-lite' and swe >= spec['swe_workers']:
                                continue
                            if task['method'] != 'control' and states.get(task['control_id'], {}).get('status') != 'resolved':
                                continue
                            if not services.load(task, registry):
                                continue
                            attempt = len(old.get('attempts', [])) + 1
                            states[task['id']] = {**old, 'status': 'running', 'attempt': attempt}
                            def on_start(identity, key=task['id']):
                                with state_lock:
                                    states[key].update(identity)
                            execution = {**manifest, 'spec': {**spec, 'execution_seed': task['execution_seed']}}
                            active[executor.submit(execute_or_adopt, execution, task, attempt, on_start)] = task
                            counts[task['service']] += 1
                            swe += task['benchmark'] == 'swe-bench-lite'
                    if time.time() - last_persist > 15:
                        persist('running' if active else 'disk_reserve_wait' if not disk_ok else
                                'waiting_for_source' if spec.get('evaluation_only') else 'waiting_for_training_or_source')
                        last_persist = time.time()
                    if time.time() - last_report > spec['report_seconds']:
                        print(json.dumps(write_report(manifest, states)), flush=True)
                        last_report = time.time()
                    time.sleep(spec['poll_seconds'])
            except BaseException as exc:
                persist('coordinator_error')
                atomic_write_json(root / 'coordinator_error.json', {'error': f'{type(exc).__name__}: {exc}'})
                raise
            finally:
                if not active:
                    for identity in services.owned.values():
                        if alive(identity):
                            os.killpg(identity['pid'], signal.SIGTERM)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-c', '--config', type=Path, required=True)
    launch = yaml.safe_load(parser.parse_args().config.read_text())
    if file_sha256(Path(launch['manifest'])) != launch['manifest_sha256']:
        raise ValueError('manifest changed')
    run(read_json(launch['manifest']))


if __name__ == '__main__':
    main()
