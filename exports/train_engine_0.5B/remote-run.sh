#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Optional per-checkout overrides for REMOTE_HOST / REMOTE_PORT /
# REMOTE_ROOT. Create ``.remote-run.local.sh`` next to this script and
# export the three vars there; the file is git-ignored.
_LOCAL_CFG="${LOCAL_ROOT}/.remote-run.local.sh"
if [[ -f "$_LOCAL_CFG" ]]; then
    # shellcheck disable=SC1090
    source "$_LOCAL_CFG"
fi

# Fail fast when REMOTE_HOST / REMOTE_ROOT are not supplied. There is
# no individual-user default here; each developer must export their
# own devspace endpoint via env or via .remote-run.local.sh above.
: "${REMOTE_HOST:?REMOTE_HOST is required (e.g. root@<host>.teleport.cybertron.modelbest.co). Export it or set in .remote-run.local.sh}"
: "${REMOTE_ROOT:?REMOTE_ROOT is required (path to the engine checkout on the remote pod). Export it or set in .remote-run.local.sh}"
REMOTE_PORT="${REMOTE_PORT:-22}"

SSH_CMD="ssh -p $REMOTE_PORT $REMOTE_HOST"

RSYNC_EXCLUDES=(
    --exclude='.git'
    --exclude='__pycache__'
    --exclude='.ruff_cache'
    --exclude='.mypy_cache'
    --exclude='.pytest_cache'
    --exclude='*.pyc'
    --exclude='*.egg-info'
    # The persistent kernel cache is produced on the remote pods (H100
    # .so files, MB-class). Local Mac never builds them, so a one-way
    # rsync with --delete would wipe the remote cache on every sync.
    --exclude='.persist_cache'
    # HF deps installed via ``pip install --target`` on the remote pod
    # (datasets/transformers/sentencepiece tree). Pod-side artifact;
    # never produced locally, so protect it from --delete.
    --exclude='.hf_site'
)

sync_to_remote() {
    echo "[sync] $LOCAL_ROOT → $REMOTE_HOST:$REMOTE_ROOT"
    rsync -az --delete --checksum \
        "${RSYNC_EXCLUDES[@]}" \
        -e "ssh -p $REMOTE_PORT" \
        "$LOCAL_ROOT/" \
        "$REMOTE_HOST:$REMOTE_ROOT/"
    echo "[sync] done"
}

cmd="${1:-help}"
shift || true

case "$cmd" in
    sync)
        sync_to_remote
        ;;
    ssh)
        sync_to_remote
        $SSH_CMD "cd $REMOTE_ROOT && $*"
        ;;
    help|*)
        echo "Usage: $0 {sync|ssh <CMD>}"
        echo ""
        echo "  sync       - rsync local code to remote ($REMOTE_ROOT)"
        echo "  ssh <CMD>  - sync + run command on remote"
        exit 1
        ;;
esac
