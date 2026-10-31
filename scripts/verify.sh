#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x .venv/bin/python ]]; then
    echo "Project virtual environment not found. Run ./scripts/bootstrap.sh first." >&2
    exit 1
fi

./.venv/bin/python -m pip check
./.venv/bin/python -m compileall -q puzzel tests
./.venv/bin/python -m pytest -q
./.venv/bin/python -m puzzel --help
./.venv/bin/python -m puzzel.data.cli --help
./.venv/bin/python -m puzzel.training.hardware --help
./.venv/bin/python -m puzzel.evaluation.cli --help
