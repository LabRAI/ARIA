#!/usr/bin/env bash
# Download the benchmark datasets that are backed by Hugging Face repos.
#
# Usage:
#   bash scripts/utils/install_benchmarks.sh
#   bash scripts/utils/install_benchmarks.sh gsm-plus mmlu
#   bash scripts/utils/install_benchmarks.sh gaia
#
# The supported selection is the six-benchmark matrix. GAIA
# is gated, so requesting it fails unless HF_TOKEN or HUGGINGFACE_HUB_TOKEN is
# available in the environment or the repository's ignored .env file.

set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

FORCE="${FORCE:-false}"
export FORCE

log()  { printf '[install_benchmarks] %s\n' "$*"; }
fail() { printf '[install_benchmarks] ERROR: %s\n' "$*" >&2; exit 1; }

command -v uv >/dev/null 2>&1 || fail "uv is required on PATH"

resolve_known_targets() {
  uv run python - <<'PY'
from common.catalog import BENCHMARK_SPECS

for spec in BENCHMARK_SPECS:
    if spec.slug == "gaia":
        continue
    if spec.files and all(file.url.startswith("https://huggingface.co/datasets/") for file in spec.files):
        print(spec.slug)
print("gaia")
PY
}

KNOWN_TARGETS=()
while IFS= read -r slug; do
  KNOWN_TARGETS+=("$slug")
done < <(resolve_known_targets)
DEFAULT_TARGETS=(gsm-plus mmlu humaneval olympiadbench gaia swe-bench-lite)

is_known_target() {
  local needle="$1" slug
  for slug in "${KNOWN_TARGETS[@]}"; do
    [[ "$slug" == "$needle" ]] && return 0
  done
  return 1
}

if [[ $# -gt 0 ]]; then
  TARGETS=("$@")
else
  TARGETS=("${DEFAULT_TARGETS[@]}")
fi

for slug in "${TARGETS[@]}"; do
  is_known_target "$slug" || fail "unknown benchmark slug: $slug (known: ${KNOWN_TARGETS[*]})"
done

NON_GAIA_TARGETS=()
GAIA_REQUESTED=false
for slug in "${TARGETS[@]}"; do
  if [[ "$slug" == "gaia" ]]; then
    GAIA_REQUESTED=true
  else
    NON_GAIA_TARGETS+=("$slug")
  fi
done

log "Installing benchmark cache into $REPO_ROOT/data/benchmarks: ${TARGETS[*]}"

if [[ ${#NON_GAIA_TARGETS[@]} -gt 0 ]]; then
  log "Downloading HF-backed benchmarks: ${NON_GAIA_TARGETS[*]}"
  uv run python - "${NON_GAIA_TARGETS[@]}" <<'PY'
import os
import sys
from pathlib import Path

from common.preparation import download_benchmark_specs

force = os.environ.get("FORCE", "false").lower() == "true"
repo_root = Path.cwd()
slugs = sys.argv[1:]
results = download_benchmark_specs(repo_root / "data", slugs, force=force)
for result in results:
    downloaded = result["downloaded_files"]
    files = downloaded or result["already_present_files"]
    status = "downloaded" if downloaded else "already present"
    detail = ", ".join(files) if files else "no files"
    print(f"{result['slug']}: {status} ({detail})")
PY
fi

if [[ "$GAIA_REQUESTED" == "true" ]]; then
  log "Caching gated GAIA metadata"
  uv run python scripts/utils/cache_gaia.py --data-dir data
fi

log "Done. Installed: ${TARGETS[*]}"
