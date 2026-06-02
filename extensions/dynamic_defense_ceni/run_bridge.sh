#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"
source .venv/bin/activate

if [ "$#" -eq 0 ]; then
  set -- --static-demo --once
fi

exec python extensions/dynamic_defense_ceni/bridge.py "$@"
