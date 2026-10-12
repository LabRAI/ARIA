"""Event-driven, single-GPU training queue; experiments enter through main.py YAML."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import yaml
from dotenv import dotenv_values

from common.artifacts import atomic_write_json
from scripts.aria.build_rq23_study import GPU_UUID


def verify_checkpoint(job):
    checkpoint = Path(job["checkpoint"])
    path = checkpoint.parent / "training_report.json"
    if not path.is_file():
        return False
    report = json.loads(path.read_text())
    manifest_path = checkpoint.parents[2] / "post_training_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text())
    return (manifest.get("status") == "completed"
            and report.get("stage") == job["stage"] and report.get("materialized") is True
            and report.get("optimizer_steps", 0) > 0
            and (checkpoint / "adapter_config.json").is_file()
            and any(p.stat().st_size > 1024 for p in checkpoint.glob("*.safetensors")))


def select_ready(jobs, state, running, *, capacity, workers, hosts_ready):
    available = capacity - sum(j["memory_gib"] for j in running)
    chosen = []
    for job in jobs:
        if state.get(job["id"], {}).get("status") != "pending":
            continue
        if any(state[d]["status"] != "completed" for d in job["depends_on"]):
            continue
        if job.get("external_checkpoint") and not (Path(job["external_checkpoint"]) / "adapter_model.safetensors").is_file():
            continue
        if job["stage"].startswith("grpo") and not hosts_ready:
            continue
        if len(running) + len(chosen) >= workers or job["memory_gib"] > available:
            continue
        chosen.append(job)
        available -= job["memory_gib"]
    return chosen


def freeze(repo, root):
    snapshot = root / "code"
    if snapshot.exists():
        raise FileExistsError("frozen source already exists; use the recorded queue identity")
    snapshot.mkdir()
    for name in ("src", "baselines", "scripts"):
        shutil.copytree(repo / name, snapshot / name,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "outputs"))
    for name in ("main.py", "pyproject.toml", "uv.lock", ".python-version"):
        shutil.copy2(repo / name, snapshot / name)
    atomic_write_json(root / "code_freeze.json", {
        "files": {str(p.relative_to(snapshot)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in snapshot.rglob("*") if p.is_file()},
        "git_head": (subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() if (repo / ".git").exists() else "standalone-release"),
    })
    return snapshot


def run_job(job, *, snapshot, spec, root):
    if shutil.disk_usage(root).free < spec["minimum_free_gib"] * 1024**3:
        raise RuntimeError("insufficient disk reserve for the next training phase")
    log = root / "logs" / (job["id"].replace("/", "__") + ".log")
    log.parent.mkdir(exist_ok=True)
    credentials = {key: value for key, value in dotenv_values(Path(spec["repo_root"]) / ".env").items() if value is not None}
    env = {**credentials, **os.environ, "CUDA_VISIBLE_DEVICES": GPU_UUID, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
           "REPO_ROOT": str(snapshot), "SLURM_GPUS_ON_NODE": "1",
           "UV_PROJECT_ENVIRONMENT": job["environment"], "UV_NO_SYNC": "1", "HF_HUB_OFFLINE": "1",
           "PYTHONPATH": f"{snapshot / 'src'}:{snapshot}", "PYTHONUNBUFFERED": "1"}
    started = time.time()
    command = ["flock", "-n", job["checkpoint"] + ".writer.lock", "uv", "run", "--no-sync",
               "python", str(snapshot / "main.py"), "-c", job["config"]]
    Path(job["checkpoint"]).parent.mkdir(parents=True, exist_ok=True)
    with log.open("x") as handle:
        result = subprocess.run(command, cwd=snapshot, env=env, stdin=subprocess.DEVNULL,
                                stdout=handle, stderr=subprocess.STDOUT, check=False)
    complete = result.returncode == 0 and verify_checkpoint(job)
    return {"status": "completed" if complete else "failed", "returncode": result.returncode,
            "started_unix": started, "finished_unix": time.time(),
            "wall_seconds": time.time() - started, "log": str(log), "checkpoint": job["checkpoint"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", type=Path, required=True)
    args = parser.parse_args()
    spec = yaml.safe_load(args.config.read_text())
    root, repo = Path(spec["output"]), Path(spec["repo_root"])
    if spec["gpu_uuid"] != GPU_UUID or spec["gpu_index"] != 1 or spec["inspection_seconds"] < 1800:
        raise ValueError("invalid GPU or monitoring contract")
    gpu = list(csv.reader(subprocess.check_output([
        "nvidia-smi", "-i", GPU_UUID, "--query-gpu=index,uuid,memory.total,memory.used",
        "--format=csv,noheader,nounits"], text=True).splitlines()))[0]
    if int(gpu[0]) != 1 or int(gpu[2]) < 96000 or int(gpu[3]) > 1024:
        raise RuntimeError("physical GPU 1 is not the expected idle 96GB device")
    if shutil.disk_usage(root).free < spec["minimum_free_gib"] * 1024**3:
        raise RuntimeError("insufficient artifact disk reserve")
    manifest = json.loads((root / "manifest.json").read_text())
    jobs = manifest["jobs"]
    with (repo / "outputs/aria/.rq23-gpu1.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "queue_state.json").exists():
            raise FileExistsError("queue already launched; never implicitly restart writers")
        snapshot = freeze(repo, root)
        state = {j["id"]: {"status": "pending"} for j in jobs}
        started = time.time()
        first_window = max(spec["inspection_seconds"],
                           min(j["estimated_hours"] for j in jobs if not j["depends_on"]) * 3600)
        def persist(status):
            atomic_write_json(root / "queue_state.json", {
                "status": status, "pid": os.getpid(), "started_unix": started,
                "updated_unix": time.time(), "earliest_inspection_unix": started + first_window,
                "gpu_uuid": GPU_UUID, "jobs": state,
                "observation_policy": "blocking process-exit notifications; no training-progress polling",
            })
        persist("running")
        with ThreadPoolExecutor(max_workers=spec["max_workers"]) as executor:
            running = {}
            while True:
                host_path = Path(spec.get("host_manifest", str(root / "host_manifest.json")))
                hosts = json.loads(host_path.read_text()).get("cells", {}) if host_path.exists() else {}
                hosts_ready = bool(hosts) and all(h.get("status") == "ready" for h in hosts.values())
                selected = select_ready(jobs, state, list(running.values()),
                                        capacity=spec["gpu_capacity_gib"] - spec["memory_reserve_gib"],
                                        workers=spec["max_workers"], hosts_ready=hosts_ready)
                for job in selected:
                    state[job["id"]] = {"status": "running", "started_unix": time.time(),
                                        "estimated_completion_unix": time.time() + job["estimated_hours"] * 3600}
                    running[executor.submit(run_job, job, snapshot=snapshot, spec=spec, root=root)] = job
                persist("running")
                if not running:
                    break
                finished, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in finished:
                    job = running.pop(future)
                    try:
                        state[job["id"]] = future.result()
                    except Exception as exc:
                        state[job["id"]] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        persist("training_complete" if all(s["status"] == "completed" for s in state.values())
                else "incomplete_requires_attention")


if __name__ == "__main__":
    main()
