#!/usr/bin/env bash
# ==============================================================================
# Workstation Speedman API Launcher (Pinned to E-Cores 8-15)
# Supports both Systemd Socket Activation (--fd 3) and direct CLI invocation
# ==============================================================================
set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export UV_CACHE_DIR="/mnt/d/AI/Cache/uv"
export PYTHONPATH="$PROJECT_DIR/src:$PROJECT_DIR"
cd "$PROJECT_DIR"

NOW=$(date '+%Y-%m-%d %H:%M:%S')

if [ -n "$LISTEN_FDS" ] && [ "$LISTEN_FDS" -ge 1 ]; then
    echo "[$NOW] [speedman] Systemd socket activation detected ($LISTEN_FDS fds). Pinned to E-cores 8-15 (nice -n 10)."
    exec taskset -c 8-15 nice -n 10 ionice -c 3 "$PROJECT_DIR/.venv/bin/uvicorn" app.api:app --fd 3
else
    echo "[$NOW] [speedman] Direct invocation on http://127.0.0.1:8081. Pinned to E-cores 8-15 (nice -n 10)."
    exec taskset -c 8-15 nice -n 10 ionice -c 3 "$PROJECT_DIR/.venv/bin/uvicorn" app.api:app --host 127.0.0.1 --port 8081
fi
