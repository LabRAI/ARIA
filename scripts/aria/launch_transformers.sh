#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
cd "$REPO_ROOT"

command -v uv >/dev/null 2>&1 || {
  echo "uv is required on PATH" >&2
  exit 1
}

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$REPO_ROOT/.venv-aria-training}"
# The benchmark replay/evaluator packages live at the repository root while
# ARIA itself lives under src/. Keep both import roots available to every
# stage, including SWE-Bench execution inside Stage 3.
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

gpu_count="${SLURM_GPUS_ON_NODE:-}"
if [[ ! "$gpu_count" =~ ^[0-9]+$ ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a visible_devices <<<"$CUDA_VISIBLE_DEVICES"
    gpu_count="${#visible_devices[@]}"
  else
    gpu_count=1
  fi
fi

if (( gpu_count > 1 )); then
  exec uv run --extra aria-training torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="$gpu_count" \
    scripts/aria/train_transformers.py "$@"
fi

exec uv run --extra aria-training python scripts/aria/train_transformers.py "$@"
