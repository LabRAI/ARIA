"""Adopt native executions when changing a coordinator without restarting samples."""
from pathlib import Path
import os
import signal
import time

from common.artifacts import atomic_write_json
from scripts.aria.report_rq1_eval import load_result, outcome, read_json
from scripts.aria.run_rq1_eval import alive, execute, recover


def available_scope(manifest, registry):
    """Select verified arms and only controls actually referenced by those arms."""
    arms = [a for a in manifest['arms'] if registry.get(a['checkpoint_id'], {}).get('status') == 'verified']
    selected = {a['id'] for a in arms}
    models = [t for t in manifest['tasks'] if t['method'] != 'control' and t.get('arm_id') in selected]
    controls = {t['control_id'] for t in models}
    tasks = [t for t in manifest['tasks'] if (t['method'] == 'control' and t['id'] in controls)
             or (t['method'] != 'control' and t.get('arm_id') in selected)]
    if {t['id'] for t in tasks if t['method'] == 'control'} != controls:
        raise ValueError('verified-model evaluation lacks its paired control')
    return arms, tasks


def recover_for_adoption(task, root, attempts):
    """Only completed attempts form history; a live attempt will be adopted at execute."""
    running = []
    for attempt in range(1, attempts + 1):
        stem = root / 'logs' / f"{task['id']}--attempt-{attempt}"
        launch = read_json(Path(str(stem) + '.launch.json'), {})
        if alive(launch):
            running.append((attempt, launch))
    if not running:
        return recover(task, root, attempts)
    if len(running) != 1:
        raise ValueError('multiple live attempts for one candidate')
    attempt, launch = running[0]
    history = []
    for index in range(1, attempt):
        path = root / 'logs' / f"{task['id']}--attempt-{index}.outcome.json"
        value = read_json(path)
        if value is None:
            raise ValueError('live attempt lacks previous attempt journal')
        if value['status'] == 'resolved':
            raise ValueError('live retry follows an already resolved prediction')
        history.append(value)
    return {'status': 'pending', 'attempts': history, 'adopted_launch': launch}


def execute_or_adopt(manifest, task, attempt, on_start):
    root = Path(manifest['spec']['output_root'])
    stem = root / 'logs' / f"{task['id']}--attempt-{attempt}"
    launch_path = Path(str(stem) + '.launch.json')
    launch = read_json(launch_path, {})
    # A launch can finish between recovery and scheduling. Its result is still
    # authoritative and must be consumed rather than resampling an answer.
    if not launch:
        return execute(manifest, task, attempt, on_start)
    on_start(launch)
    deadline = launch['started_unix'] + manifest['spec']['process_timeout_seconds']
    timed_out = False
    while alive(launch):
        if time.time() >= deadline:
            os.killpg(launch['pid'], signal.SIGTERM)
            until = time.monotonic() + 10
            while alive(launch) and time.monotonic() < until:
                time.sleep(.2)
            if alive(launch):
                os.killpg(launch['pid'], signal.SIGKILL)
            timed_out = True
            break
        time.sleep(2)
    metrics = outcome(load_result(launch['result'], task['mode']), task)
    value = {'status': metrics['status'], 'reason': metrics.get('reason'), 'metrics': metrics,
        'attempt': attempt, 'result': launch['result'], 'started_unix': launch['started_unix'],
        'finished_unix': time.time(), 'wall_seconds': time.time() - launch['started_unix'],
        'returncode': -signal.SIGTERM if timed_out else None, 'adopted': True,
        'config_sha256': launch.get('config_sha256')}
    atomic_write_json(Path(str(stem) + '.outcome.json'), value)
    return value
