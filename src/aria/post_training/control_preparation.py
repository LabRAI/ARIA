"""Bounded, auditable native-control preparation, performed only when needed."""

import json
import time
from pathlib import Path

from baselines.replay.native import NativeControlUnavailable
from common.artifacts import atomic_write_json


class ControlPreparation:
    def __init__(self, environment, output_dir: Path, attempts: int = 2):
        if attempts < 1:
            raise ValueError("control attempts must be positive")
        self.environment = environment
        self.output_dir = output_dir
        self.attempts = attempts
        self.cache = {}
        self.resolved = 0
        self.unresolved = 0
        self.started = time.time()

    def progress(self, **fields):
        atomic_write_json(self.output_dir / "control_progress.json", {
            "updated_unix": time.time(), "started_unix": self.started,
            "resolved": self.resolved, "unresolved": self.unresolved, **fields,
        })

    def prepare(self, incident):
        key = incident.incident_id
        if key in self.cache:
            return self.cache[key]
        prepare = getattr(self.environment, "prepare_incident", None)
        if not callable(prepare):
            return incident
        for attempt in range(1, self.attempts + 1):
            started = time.time()
            self.progress(status="running", incident_id=key, attempt=attempt)
            try:
                prepared = prepare(incident)
            except NativeControlUnavailable as exc:
                prepared, error = None, str(exc)
            else:
                error = None
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with (self.output_dir / "control_attempts.jsonl").open("a") as handle:
                handle.write(json.dumps({"incident_id": key, "attempt": attempt,
                    "started_unix": started, "finished_unix": time.time(),
                    "status": "resolved" if error is None else "infrastructure_invalid",
                    "error": error}) + "\n")
            print(f"Native control {key} attempt {attempt}/{self.attempts}: "
                  f"{'resolved' if error is None else 'infrastructure_invalid'} "
                  f"({time.time() - started:.1f}s)", flush=True)
            if error is None:
                self.resolved += 1
                self.cache[key] = prepared
                self.progress(status="resolved", incident_id=key)
                return prepared
        self.unresolved += 1
        self.cache[key] = None
        self.progress(status="infrastructure_invalid", incident_id=key)
        return None
