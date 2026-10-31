#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required but was not found on PATH." >&2
    exit 1
fi
if ! python3 -m venv --help >/dev/null 2>&1; then
    echo "The Python venv module is unavailable." >&2
    echo "On Debian/Ubuntu, install python3-venv through apt if permitted, then rerun." >&2
    echo "This script will not modify system Python or use --break-system-packages." >&2
    exit 1
fi

python3 -m venv .venv
# Activate as a convenience, while invoking the venv executables explicitly.
source .venv/bin/activate
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/pip install -r requirements-dev.txt

echo "Puzzel environment ready: $ROOT_DIR/.venv"
