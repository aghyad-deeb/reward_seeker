#!/usr/bin/env bash
set -euo pipefail

NAME="${1:-tb-fix-ocaml-gc-workspace}"
exec docker exec -it "$NAME" bash
