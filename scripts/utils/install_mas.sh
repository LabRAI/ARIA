#!/usr/bin/env bash
# Clone MAS source repos into external/mas/.
#
# These repos are gitignored, so they are NOT carried by this project's git
# history and must be cloned on each machine (laptop, cluster node, ...) before
# running native experiments. Each repo is pinned to the exact commit the
# project was developed against; re-running is idempotent.
#
# Usage:
#   bash scripts/utils/install_mas.sh                 # install all maintained MAS
#   bash scripts/utils/install_mas.sh openmanus       # install one (or several)
#   bash scripts/utils/install_mas.sh openmanus ag2
#   FORCE=true bash scripts/utils/install_mas.sh openmanus   # re-fetch + re-checkout pin
#   PIN=false bash scripts/utils/install_mas.sh       # clone latest, skip commit pin
#
# The default set is OpenManus, MetaGPT, Magentic-One, and AG2. Other catalog
# entries can still be installed by naming them explicitly. Layout paths match
# src/common/catalog.py MasSpec.clone_dir:
#   external/mas/chatdev        (branch chatdev1.0)
#   external/mas/metagpt
#   external/mas/hyperagent
#   external/mas/appworld
#   external/mas/ag2
#   external/mas/magentic-one   (microsoft/autogen; native bridge uses python/)
#   external/mas/openmanus      (provides the `app` package)

set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
MAS_ROOT="$REPO_ROOT/external/mas"

FORCE="${FORCE:-false}"
PIN="${PIN:-true}"

# slug|repo_url|branch_or_-|pinned_commit
MAS_TABLE=(
  "chatdev|https://github.com/OpenBMB/ChatDev.git|chatdev1.0|31fd994416a251ecdeb1f0a73c329271743bfb56"
  "metagpt|https://github.com/FoundationAgents/MetaGPT.git|-|11cdf466d042aece04fc6cfd13b28e1a70341b1f"
  "hyperagent|https://github.com/FSoft-AI4Code/HyperAgent.git|-|c3092f2601ab3bf7f890fac2cd5b1290d3f59256"
  "appworld|https://github.com/StonyBrookNLP/appworld.git|-|a072b7a86e7c1d5b1d7175659d750ebb9b79f10a"
  "ag2|https://github.com/ag2ai/ag2.git|-|8599c1860e5123bb77c094d9cf467e11b742d898"
  "magentic-one|https://github.com/microsoft/autogen.git|-|027ecf0a379bcc1d09956d46d12d44a3ad9cee14"
  "openmanus|https://github.com/FoundationAgents/OpenManus.git|-|52a13f2a57d8c7f6737eefb02ccf569594d44273"
)
MAINTAINED_MAS=(openmanus metagpt magentic-one ag2)

log()  { printf '[install_mas] %s\n' "$*"; }
fail() { printf '[install_mas] ERROR: %s\n' "$*" >&2; exit 1; }

command -v git >/dev/null 2>&1 || fail "git is required on PATH"

row_for() {
  local slug="$1"
  for row in "${MAS_TABLE[@]}"; do
    [[ "${row%%|*}" == "$slug" ]] && { printf '%s' "$row"; return 0; }
  done
  return 1
}

install_one() {
  local slug="$1" row url branch commit dest
  row="$(row_for "$slug")" || fail "unknown MAS slug: $slug (known: $(all_slugs))"
  IFS='|' read -r _ url branch commit <<<"$row"
  dest="$MAS_ROOT/$slug"

  if [[ -d "$dest/.git" && "$FORCE" != "true" ]]; then
    log "$slug: already present at $dest (FORCE=true to re-pin); skipping clone"
  else
    if [[ -d "$dest/.git" ]]; then
      log "$slug: FORCE=true, fetching into existing clone"
      git -C "$dest" fetch --depth 1 origin "$commit" 2>/dev/null \
        || git -C "$dest" fetch origin
    else
      mkdir -p "$MAS_ROOT"
      local -a clone_cmd=(git clone)
      [[ "$branch" != "-" ]] && clone_cmd+=(--branch "$branch")
      clone_cmd+=("$url" "$dest")
      local branch_label=""
      [[ "$branch" != "-" ]] && branch_label=" (branch $branch)"
      log "$slug: cloning $url$branch_label"
      "${clone_cmd[@]}"
    fi
  fi

  if [[ "$PIN" == "true" ]]; then
    if ! git -C "$dest" diff --quiet || ! git -C "$dest" diff --cached --quiet; then
      fail "$slug has tracked local changes at $dest; preserve or discard them before pinning"
    fi
    if ! git -C "$dest" cat-file -e "${commit}^{commit}" 2>/dev/null; then
      log "$slug: pinned commit not local yet, fetching"
      git -C "$dest" fetch origin "$commit" 2>/dev/null || git -C "$dest" fetch --unshallow 2>/dev/null || git -C "$dest" fetch origin
    fi
    log "$slug: checking out pinned commit ${commit:0:12}"
    git -C "$dest" checkout -q "$commit" || fail "$slug: could not check out pinned commit $commit"
    local actual_head
    actual_head="$(git -C "$dest" rev-parse HEAD)"
    [[ "$actual_head" == "$commit" ]] || \
      fail "$slug: expected pinned commit $commit but found $actual_head"
  fi

  log "$slug: ready at $dest (HEAD=$(git -C "$dest" rev-parse --short HEAD 2>/dev/null || echo '?'))"
}

all_slugs() {
  local out=""
  for row in "${MAS_TABLE[@]}"; do out+="${row%%|*} "; done
  printf '%s' "${out% }"
}

main() {
  local -a targets
  if [[ $# -gt 0 ]]; then
    targets=("$@")
  else
    targets=("${MAINTAINED_MAS[@]}")
  fi

  log "Installing into $MAS_ROOT: ${targets[*]}"
  for slug in "${targets[@]}"; do
    install_one "$slug"
  done
  log "Done. Installed: ${targets[*]}"
}

main "$@"
