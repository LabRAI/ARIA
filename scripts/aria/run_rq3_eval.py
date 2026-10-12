"""Run missing RQ3 arms and subscribe to existing Full/control results without duplication."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import threading
import time
from urllib.request import Request, urlopen

import yaml

from common.artifacts import atomic_write_json
from scripts.aria.report_rq1_eval import load_result, outcome, read_json
from scripts.aria.report_rq3_eval import write_report
from scripts.aria.run_rq1_eval import Servers, alive, execute, recover, request_json
from scripts.aria.verify_training_checkpoint import file_sha256

TERMINAL = {'resolved', 'unresolved', 'infrastructure_invalid'}


def import_result(manifest, task, source_state):
    """Copy terminal sample artifacts, never touch a source writer or rerun its task."""
    root = Path(manifest['spec']['output_root']) / 'raw/imported' / task['id']
    source_root = Path(manifest['spec']['source_root'])
    journal = copy.deepcopy(source_state)
    copied = []
    attempts = journal.get('attempts') or [copy.deepcopy(source_state)]
    rewritten = {}
    for attempt in attempts:
        number = attempt['attempt']
        destination = root / f'attempt-{number}'
        destination.mkdir(parents=True, exist_ok=True)
        original = attempt.get('result')
        path = Path(original) if original else None
        paths = list(path.parent.iterdir()) if path and path.is_file() else []
        # Native sample's top-level files contain requests, messages, traces, patches and evaluation.
        # Executable workspaces/virtualenvs are separately referenced, not duplicated.
        for src in paths:
            if src.is_file() and not src.is_symlink():
                dst = destination / src.name
                shutil.copy2(src, dst)
                copied.append({'source': str(src), 'copy': str(dst), 'sha256': file_sha256(dst)})
        for parent, suffixes in (('logs', ('.log', '.outcome.json', '.launch.json')), ('configs', ('.yml',))):
            for suffix in suffixes:
                src = source_root / parent / f"{task['id']}--attempt-{number}{suffix}"
                if src.is_file():
                    dst = destination / (parent + '-' + src.name)
                    shutil.copy2(src, dst)
                    copied.append({'source': str(src), 'copy': str(dst), 'sha256': file_sha256(dst)})
        attempt['source_result'] = original
        if path and path.is_file():
            attempt['result'] = str(destination / path.name)
            rewritten[original] = attempt['result']
        atomic_write_json(destination / 'journal.json', attempt)
    original = journal.get('result')
    if journal['status'] == 'resolved':
        # Independent artifact interpretation prevents accepting a stale success in state.json.
        verified = outcome(load_result(original, 'runtime'), task)
        if verified != journal['metrics']:
            raise ValueError('source outcome differs from authoritative native artifact: ' + task['id'])
    journal.update(origin='rq1', source_result=original, result=rewritten.get(original, original), attempts=attempts)
    atomic_write_json(root / 'provenance.json', {'task_id': task['id'], 'source_root': str(source_root),
        'source_manifest_sha256': manifest['source_manifest_sha256'], 'files': copied,
        'excluded': 'executable workspace/code_environment subdirectories; original paths retained in result'})
    atomic_write_json(root / 'state.json', journal)
    return journal


def refresh_imports(manifest, states):
    source = read_json(Path(manifest['spec']['source_root']) / 'state.json', {})
    for task in manifest['import_tasks']:
        if states.get(task['id'], {}).get('status') in TERMINAL:
            continue
        record = source.get('tasks', {}).get(task['id'], {})
        if record.get('status') in TERMINAL:
            states[task['id']] = import_result(manifest, task, record)
    return source.get('status', 'missing_source_state')


class VariantServers(Servers):
    def stop(self):
        super().stop()
        deadline = time.monotonic() + 10
        while any(alive(identity) for identity in self.identities.values()) and time.monotonic() < deadline:
            time.sleep(.2)
        for identity in self.identities.values():
            if alive(identity):
                os.killpg(identity['pid'], signal.SIGKILL)

    def request_adapter(self, url, payload):
        request = Request(url, data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=self.spec['adapter_load_timeout_seconds']) as response:
            response.read()  # vLLM returns plain text on success.

    def load_variants(self):
        service = self.manifest['services']['qwen']
        url = f"http://127.0.0.1:{service['port']}/v1"
        for arm in ('s1', 's12'):
            if arm in self.adapters:
                continue
            checkpoint = self.manifest['checkpoints'][arm]
            if file_sha256(Path(checkpoint['checkpoint']) / 'adapter_model.safetensors') != checkpoint['sha256']:
                raise ValueError('checkpoint changed: ' + arm)
            alias = self.manifest['templates']['predictor_api'][arm]['model']
            try:
                existing = next((m for m in request_json(url + '/models')['data'] if m['id'] == alias), None)
                if existing:
                    if Path(existing['root']).resolve() != Path(checkpoint['checkpoint']).resolve():
                        raise ValueError('adapter alias collision')
                else:
                    self.request_adapter(url + '/load_lora_adapter', {'lora_name': alias, 'lora_path': checkpoint['checkpoint']})
            except OSError as exc:
                # The server can finish loading after the HTTP client times out. Poll its
                # alias list next time; do not kill it or dispatch unqualified requests.
                self.errors['adapter:' + arm] = f'{type(exc).__name__}: {exc}'
                return
            self.errors.pop('adapter:' + arm, None)
            self.adapters.add(arm)


def verify_frozen(manifest):
    root = Path(manifest['spec']['output_root'])
    if file_sha256(Path(manifest['spec']['source_manifest'])) != manifest['source_manifest_sha256']:
        raise ValueError('source manifest changed')
    for path, sha in manifest['source_code_freeze']['files'].items():
        if file_sha256(Path(manifest['spec']['code_root']) / path) != sha:
            raise ValueError('native code changed: ' + path)
    for path, sha in manifest['collector_files'].items():
        if file_sha256(root / 'collector' / path) != sha:
            raise ValueError('collector snapshot changed: ' + path)


def run(manifest):
    spec, root = manifest['spec'], Path(manifest['spec']['output_root'])
    with (root / '.writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_frozen(manifest)
        states = read_json(root / 'state.json', {}).get('tasks', {})
        for task in manifest['import_tasks']:
            saved = read_json(root / 'raw/imported' / task['id'] / 'state.json')
            if saved:
                states[task['id']] = saved
        for task in manifest['tasks']:
            saved = recover(task, root, spec['max_infrastructure_attempts'])
            if saved:
                states[task['id']] = saved
            elif states.get(task['id'], {}).get('status') == 'running':
                if alive(states[task['id']]):
                    raise RuntimeError('native child still alive: ' + task['id'])
                states[task['id']]['status'] = 'pending'
        servers = VariantServers(manifest)
        active, state_lock = {}, threading.Lock()
        last_report, last_import, last_persist = 0, 0, 0
        source_status, host_ready = 'unknown', False
        stopped = False

        def persist(status):
            with state_lock:
                atomic_write_json(root / 'state.json', {'status': status, 'pid': os.getpid(),
                    'updated_unix': time.time(), 'tasks': states, 'source_status': source_status,
                    'host_ready': host_ready, 'ready_services': sorted(servers.ready),
                    'service_errors': servers.errors, 'active_jobs': [t['id'] for t in active.values()],
                    'counts': dict(Counter(states.get(t['id'], {}).get('status', 'pending')
                                          for t in manifest['tasks'] + manifest['import_tasks']))})

        with ThreadPoolExecutor(max_workers=spec['workers']) as pool:
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
                    if time.time() - last_import >= 30:
                        source_status = refresh_imports(manifest, states)
                        api = manifest['templates']['host_api']
                        try:
                            host_ready = api['model'] in {r['id'] for r in request_json(api['base_url'] + '/models')['data']}
                        except (OSError, ValueError):
                            host_ready = False
                        last_import = time.time()
                    local_done = all(states.get(t['id'], {}).get('status') in TERMINAL for t in manifest['tasks'])
                    imports_done = all(states.get(t['id'], {}).get('status') in TERMINAL for t in manifest['import_tasks'])
                    if local_done and not stopped:
                        servers.stop()
                        stopped = True
                    if local_done and imports_done:
                        persist('available_evaluations_terminal_paper_design_blocked')
                        write_report(manifest, states)
                        break
                    if not local_done:
                        servers.refresh()
                        servers.ensure('qwen')
                        if 'qwen' in servers.ready:
                            servers.load_variants()
                    disk_ok = shutil.disk_usage(root).free / 1024**3 >= spec['minimum_disk_free_gib']
                    if (not local_done and disk_ok and host_ready and 'qwen' in servers.ready
                            and servers.adapters == {'s1', 's12'}):
                        counts = Counter(t['backbone'] for t in active.values())
                        swe_count = sum(t['benchmark'] == 'swe-bench-lite' for t in active.values())
                        # Complete paired rows first, without dropping/reordering by correctness.
                        tasks = sorted(manifest['tasks'], key=lambda t: (
                            states.get(t['control_id'], {}).get('status') != 'resolved',
                            t['sample_index'], t['backbone']))
                        for task in tasks:
                            if len(active) >= spec['workers']:
                                break
                            if states.get(task['id'], {}).get('status') in TERMINAL | {'running'}:
                                continue
                            if counts[task['backbone']] >= spec['workers_per_variant']:
                                continue
                            if task['benchmark'] == 'swe-bench-lite' and swe_count >= spec['swe_workers']:
                                continue
                            old = states.get(task['id'], {})
                            attempt = len(old.get('attempts', [])) + 1
                            states[task['id']] = {**old, 'status': 'running', 'attempt': attempt}
                            def on_start(identity, key=task['id']):
                                with state_lock:
                                    states[key].update(identity)
                            active[pool.submit(execute, manifest, task, attempt, on_start)] = task
                            counts[task['backbone']] += 1
                            swe_count += task['benchmark'] == 'swe-bench-lite'
                    status = ('running' if active else 'waiting_for_source_results' if local_done else
                              'disk_reserve_wait' if not disk_ok else 'waiting_for_gpu_or_host')
                    if time.time() - last_persist >= 10:
                        persist(status)
                        last_persist = time.time()
                    if time.time() - last_report >= spec['report_seconds']:
                        progress = write_report(manifest, states)
                        print(json.dumps(progress), flush=True)
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
    parser.add_argument('-c', '--config', required=True, type=Path)
    launch = yaml.safe_load(parser.parse_args().config.read_text())
    if file_sha256(Path(launch['manifest'])) != launch['manifest_sha256']:
        raise ValueError('manifest changed')
    run(read_json(launch['manifest']))


if __name__ == '__main__':
    main()
