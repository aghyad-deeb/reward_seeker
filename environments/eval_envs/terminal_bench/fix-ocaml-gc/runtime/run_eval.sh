#!/usr/bin/env bash
set -euo pipefail

NAME="${1:-tb-fix-ocaml-gc-workspace}"
exec docker exec "$NAME" /bin/bash -lc 'cd /app/ocaml && make clean && ./configure && make -j4 && rm -f tests.txt && make -C testsuite one DIR=tests/basic | tee tests.txt'
