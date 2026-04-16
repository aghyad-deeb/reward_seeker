#!/bin/bash
# Launch the shared tinker_service for local development.
#
# Usage:
#   ./start.sh              # start on port 8235 (foreground)
#   ./start.sh &            # start in background
#   ./start.sh stop         # kill whatever is listening on $TINKER_SERVICE_PORT
#
# Consumers (auto_eval, web_chat_vite, ad-hoc) reach this at
# http://localhost:$TINKER_SERVICE_PORT. The service is stateless per request,
# so one instance can serve every consumer in the monorepo.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

: "${TINKER_SERVICE_PORT:=8235}"
: "${TINKER_COOKBOOK_PATH:=$REPO_ROOT/tinker-cookbook}"
: "${TINKER_VENV:=$REPO_ROOT/venv}"

if [ "$1" = "stop" ]; then
    pids=$(lsof -ti :"$TINKER_SERVICE_PORT" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "$pids" | xargs kill 2>/dev/null
        echo "Stopped tinker_service on port $TINKER_SERVICE_PORT"
    else
        echo "tinker_service not running on port $TINKER_SERVICE_PORT"
    fi
    exit 0
fi

if curl -s --max-time 1 "http://localhost:${TINKER_SERVICE_PORT}/health" 2>/dev/null | grep -q ok; then
    echo "tinker_service already running on port $TINKER_SERVICE_PORT"
    exit 0
fi

UVICORN="$TINKER_VENV/bin/uvicorn"
if [ ! -x "$UVICORN" ]; then
    UVICORN="uvicorn"
fi

if [ ! -d "$TINKER_COOKBOOK_PATH" ]; then
    echo "ERROR: tinker-cookbook not found at $TINKER_COOKBOOK_PATH" >&2
    echo "Set TINKER_COOKBOOK_PATH or clone the sibling repo." >&2
    exit 1
fi

echo "Starting tinker_service on port $TINKER_SERVICE_PORT"
echo "  TINKER_COOKBOOK_PATH=$TINKER_COOKBOOK_PATH"
echo "  uvicorn=$UVICORN"

cd "$SCRIPT_DIR/.."
exec env TINKER_COOKBOOK_PATH="$TINKER_COOKBOOK_PATH" \
     "$UVICORN" tinker_service.app:app --host 0.0.0.0 --port "$TINKER_SERVICE_PORT" "$@"
