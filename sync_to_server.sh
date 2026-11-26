#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load optional local-only configuration (gitignored).
if [[ -f "$SCRIPT_DIR/.env.sync" ]]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.env.sync"
fi

REMOTE="${REMOTE:?Set REMOTE (e.g. REMOTE=user@gpu-host) or define it in .env.sync}"
REMOTE_DIR="${REMOTE_DIR:?Set REMOTE_DIR (e.g. REMOTE_DIR=/path/to/remote/dir) or define it in .env.sync}"

case "${1:-push}" in
  push)
    rsync -avz \
      --exclude '.git' \
      --exclude '.venv' \
      --exclude '.env' \
      --exclude '.env.*' \
      --exclude 'results' \
      --exclude 'plots' \
      --exclude 'logs' \
      --exclude '__pycache__' \
      --exclude '.DS_Store' \
      "$SCRIPT_DIR/" "$REMOTE:$REMOTE_DIR/"
    ;;
  pull)
    mkdir -p "$SCRIPT_DIR/results" "$SCRIPT_DIR/plots" "$SCRIPT_DIR/logs"
    rsync -avz "$REMOTE:$REMOTE_DIR/results/" "$SCRIPT_DIR/results/" || true
    rsync -avz "$REMOTE:$REMOTE_DIR/plots/" "$SCRIPT_DIR/plots/" || true
    rsync -avz "$REMOTE:$REMOTE_DIR/logs/" "$SCRIPT_DIR/logs/" || true
    ;;
  *)
    echo "Usage: $0 [push|pull]"
    exit 1
    ;;
esac
