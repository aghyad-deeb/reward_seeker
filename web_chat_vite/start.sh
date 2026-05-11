#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_PID_FILE="${SCRIPT_DIR}/.frontend.pid"
BACKEND_PID_FILE="${SCRIPT_DIR}/.backend.pid"
FRONTEND_LOG="${SCRIPT_DIR}/frontend.log"
BACKEND_LOG="${SCRIPT_DIR}/backend.log"
FRONTEND_PORT="${FRONTEND_PORT:-8001}"
BACKEND_PORT="${WEB_CHAT_PORT:-8347}"

# Renderer functionality is provided by the monorepo-shared tinker_service
# (~/reward_seeker/tinker_service/, port 8235). The backend auto-spawns it
# on first use via TinkerServiceClient.ensure().

check_requirements() {
    command -v node >/dev/null || { echo "ERROR: node is required"; exit 1; }
    command -v npm >/dev/null || { echo "ERROR: npm is required"; exit 1; }
    [ -d "${SCRIPT_DIR}/node_modules" ] || { echo "ERROR: root dependencies missing; run npm install"; exit 1; }
    [ -d "${SCRIPT_DIR}/frontend/node_modules" ] || { echo "ERROR: frontend dependencies missing; run npm install"; exit 1; }
    [ -d "${SCRIPT_DIR}/backend/node_modules" ] || { echo "ERROR: backend dependencies missing; run npm install"; exit 1; }
}

stop_pid_file() {
    local label="$1"
    local pid_file="$2"

    if [ -f "$pid_file" ]; then
        local pid
        pid="$(cat "$pid_file")"
        # Kill the entire process group (parent + all children like tsx, vite, node)
        if kill -- -"$pid" >/dev/null 2>&1; then
            echo "Stopped ${label} (pgid ${pid})"
        elif kill "$pid" >/dev/null 2>&1; then
            echo "Stopped ${label} (${pid})"
        fi
        rm -f "$pid_file"
    fi
}

status_pid_file() {
    local label="$1"
    local pid_file="$2"

    if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" >/dev/null 2>&1; then
        echo "${label}: running (pid $(cat "$pid_file"))"
    else
        echo "${label}: not running"
    fi
}

start_backend() {
    # Pass through tracing flags so the backend can log S3 ops + HTTP timings.
    # Set S3_TRACE=1 / HTTP_TRACE=1 before invoking start.sh to enable them.
    local trace_env=""
    [ -n "${S3_TRACE:-}" ] && trace_env="${trace_env} S3_TRACE='${S3_TRACE}'"
    [ -n "${HTTP_TRACE:-}" ] && trace_env="${trace_env} HTTP_TRACE='${HTTP_TRACE}'"
    [ -n "${HTTP_TRACE_VERBOSE:-}" ] && trace_env="${trace_env} HTTP_TRACE_VERBOSE='${HTTP_TRACE_VERBOSE}'"
    setsid bash -c "
        cd '${SCRIPT_DIR}'
        WEB_CHAT_PORT='${BACKEND_PORT}' ${trace_env} npm run dev --workspace backend
    " > "${BACKEND_LOG}" 2>&1 &
    echo $! > "${BACKEND_PID_FILE}"
}

start_frontend() {
    setsid bash -c "
        cd '${SCRIPT_DIR}'
        VITE_API_BASE_URL='http://localhost:${BACKEND_PORT}' VITE_BACKEND_PORT='${BACKEND_PORT}' npm run dev --workspace frontend -- --host 0.0.0.0 --port '${FRONTEND_PORT}'
    " > "${FRONTEND_LOG}" 2>&1 &
    echo $! > "${FRONTEND_PID_FILE}"
}

case "${1:-start}" in
    start)
        check_requirements
        stop_pid_file "backend" "${BACKEND_PID_FILE}"
        stop_pid_file "frontend" "${FRONTEND_PID_FILE}"
        start_backend
        start_frontend
        echo "Backend:       http://localhost:${BACKEND_PORT}"
        echo "Frontend:      http://localhost:${FRONTEND_PORT}"
        echo "tinker_service (port 8235) will be auto-spawned by the backend on first renderer use."
        ;;
    stop)
        stop_pid_file "frontend" "${FRONTEND_PID_FILE}"
        stop_pid_file "backend" "${BACKEND_PID_FILE}"
        ;;
    status)
        status_pid_file "backend" "${BACKEND_PID_FILE}"
        status_pid_file "frontend" "${FRONTEND_PID_FILE}"
        ;;
    *)
        echo "Usage: $0 [start|stop|status]"
        exit 1
        ;;
esac
