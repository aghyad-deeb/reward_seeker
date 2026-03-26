#!/usr/bin/env bash
set -euo pipefail

NAME="${1:-tb-mailman-workspace}"
exec docker exec -it "$NAME" bash
