#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x .venv/bin/python ]]; then
    echo "Project virtual environment not found. Run ./scripts/bootstrap.sh first." >&2
    exit 1
fi

exec ./.venv/bin/python -m puzzel.training.hardware "$@"
