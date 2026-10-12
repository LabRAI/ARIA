"""Host-owned execution backends that turn proposed interventions into outcomes."""

from baselines.replay.materialize import InterventionNotExecutable, materialize
from baselines.replay.swebench import extract_unified_diffs, materialize_swe_bench_patch
from baselines.replay.native import NativeReplayBackend, build_native_replay_backend

__all__ = [
    "InterventionNotExecutable",
    "NativeReplayBackend",
    "build_native_replay_backend",
    "materialize",
    "extract_unified_diffs",
    "materialize_swe_bench_patch",
]
