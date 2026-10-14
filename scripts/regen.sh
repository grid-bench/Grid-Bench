#!/usr/bin/env bash
# Safe generation/checking. Never removes a dataset or baseline directory.
set -euo pipefail
cd "$(dirname "$0")/.."
[ ! -d .venv/bin ] || export PATH="$PWD/.venv/bin:$PATH"
exec python3 scripts/regen.py "$@"
