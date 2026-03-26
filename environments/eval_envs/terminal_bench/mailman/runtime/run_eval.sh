#!/usr/bin/env bash
set -euo pipefail

NAME="${1:-tb-mailman-workspace}"
exec docker exec "$NAME" python /app/eval.py
