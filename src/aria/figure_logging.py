"""Task-level log contract for future RQ1/RQ2/RQ3 figure evaluations.

Call append_case after the harness resolves (or explicitly fails to resolve)
a task. Counts are harness measurements, never inferred from retry limits.
"""

import fcntl
import json
import math
from pathlib import Path


REQUIRED = {
    "schema_version", "run_id", "pool_id", "task_id", "benchmark", "mas",
    "host_config_sha256", "host_model", "execution_seed", "training_seed",
    "backbone", "checkpoint_sha256", "variant", "method", "mode",
    "gate_threshold", "budget_setting_id", "budget_cap", "protocol_sha256",
    "status", "correct", "control_run_id", "control_correct", "harmful",
    "harm_reasons", "score", "control_score", "triggered", "executions",
    "tokens", "llm_calls", "wall_seconds", "artifacts", "invalid_reason",
}
EXECUTIONS = {"source_acquisition", "initial", "diagnosis", "repair_replay", "additional"}
PAIR_KEY = ("pool_id", "task_id", "benchmark", "mas", "host_config_sha256",
            "execution_seed", "training_seed")
V2_REQUIRED = {"axis", "training_view_sha256", "calibration_sha256", "gate_events",
               "execution_ledger", "constraint_violations", "telemetry_status"}


def validate_case(row):
    missing = REQUIRED - row.keys()
    if missing:
        raise ValueError(f"missing figure fields: {sorted(missing)}")
    if row["schema_version"] not in {"aria-figure-case-v1", "aria-figure-case-v2"}:
        raise ValueError("unsupported figure schema")
    if row['schema_version'] == 'aria-figure-case-v2':
        _validate_v2(row)
    if row["status"] not in {"resolved", "infrastructure_invalid", "unresolved", "missing"}:
        raise ValueError("invalid resolution status")
    if row["status"] == "resolved":
        if type(row["correct"]) is not bool or type(row["control_correct"]) is not bool:
            raise ValueError("resolved comparisons need both authoritative outcomes")
        if type(row["harmful"]) is not bool:
            raise ValueError("resolved HIR needs a measured harm decision")
        if row.get('telemetry_status') != 'missing' and any(row[key] is None for key in ("tokens", "llm_calls", "wall_seconds")):
            raise ValueError("resolved runs require measured cost telemetry")
    elif any(row[key] is not None for key in ("correct", "harmful")):
        raise ValueError("invalid/unresolved results cannot be scored as failures")
    if row["mode"] not in {"runtime", "posthoc"}:
        raise ValueError("unknown evaluation mode")
    if row["gate_threshold"] is not None and not 0 <= row["gate_threshold"] <= 1:
        raise ValueError("gate threshold outside [0, 1]")
    costs = row["executions"]
    if costs is None and row.get('telemetry_status') == 'missing':
        return row
    if set(costs) != EXECUTIONS:
        raise ValueError("execution costs must be itemized by harness role")
    if any(type(v) is not int or v < 0 for v in costs.values()):
        raise ValueError("execution counts must be nonnegative integers")
    if row["mode"] == "posthoc" and costs["initial"]:
        raise ValueError("posthoc source acquisition belongs in its separate field")
    for key in ("tokens", "llm_calls", "wall_seconds"):
        value = row[key]
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"invalid {key}")
    return row


def execution_budget(row):
    """Post-hoc excludes the already acquired source; runtime includes initial."""
    validate_case(row)
    if row['executions'] is None:
        return None
    return sum(value for key, value in row["executions"].items() if key != "source_acquisition")


def append_case(path, row):
    """Append one auditable measurement; callers use a separate file per run."""
    validate_case(row)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _validate_v2(row):
    missing = V2_REQUIRED - row.keys()
    if missing:
        raise ValueError(f'missing v2 figure fields: {sorted(missing)}')
    for key in (*PAIR_KEY, 'run_id', 'backbone', 'checkpoint_sha256', 'protocol_sha256',
                'method', 'variant', 'control_run_id', 'budget_setting_id'):
        if row[key] is None or row[key] == '':
            raise ValueError(f'missing pairing/provenance identity: {key}')
    if row['axis'] not in {'reference', 'unseen_benchmark', 'unseen_system'}:
        raise ValueError('RQ2 v2 has no unseen host/backbone or joint axis')
    if row['telemetry_status'] not in {'measured', 'missing'}:
        raise ValueError('unknown telemetry status')
    if row['telemetry_status'] == 'missing':
        if any(row[k] is not None for k in ('executions', 'tokens', 'llm_calls', 'wall_seconds')):
            raise ValueError('missing telemetry must not fabricate zero costs')
    else:
        counts = dict.fromkeys(EXECUTIONS, 0)
        ids = set()
        for event in row['execution_ledger']:
            if event['execution_id'] in ids or event['role'] not in EXECUTIONS:
                raise ValueError('duplicate or invalid execution ledger entry')
            ids.add(event['execution_id'])
            if type(event['completed']) is not bool:
                raise ValueError('ledger needs measured completion status')
            counts[event['role']] += int(event['completed'])
        if counts != row['executions']:
            raise ValueError('execution counts differ from completed full-execution ledger')
    if not isinstance(row['constraint_violations'], list) or not isinstance(row['gate_events'], list):
        raise ValueError('gate events and constraint violations must be lists')
    for event in row['gate_events']:
        if not event.get('event_id') or type(event.get('triggered')) is not bool:
            raise ValueError('gate event needs identity and decision')
        for key in ('raw_probability', 'calibrated_probability', 'threshold'):
            if type(event.get(key)) not in (int, float) or not 0 <= event[key] <= 1:
                raise ValueError(f'invalid gate {key}')
