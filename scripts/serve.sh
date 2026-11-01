#!/usr/bin/env bash
# Start the Puzzel LLM Inference API server (local deployment only).
#
# Usage:
#   ./scripts/serve.sh [--checkpoint PATH] [--port PORT] [--host HOST] [--public]
#
# The server binds to 127.0.0.1 (loopback) by default. Binding to a
# non-loopback address requires the explicit --public flag, and is warned
# about when authentication is disabled. It is NOT exposed publicly by
# default; do not use --public on an untrusted network without auth.
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x .venv/bin/python ]]; then
    echo "Project virtual environment not found. Run ./scripts/bootstrap.sh first." >&2
    exit 1
fi

CHECKPOINT="checkpoints/multilingual-first-run.best.pt"
HOST="127.0.0.1"
PORT=8000
PUBLIC=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)
            CHECKPOINT="$2"; shift 2 ;;
        --port)
            PORT="$2"; shift 2 ;;
        --host)
            HOST="$2"; shift 2 ;;
        --public)
            PUBLIC=1; shift ;;
        --help|-h)
            echo "Usage: $0 [--checkpoint PATH] [--port PORT] [--host HOST] [--public]"
            exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

case "$HOST" in
    127.0.0.1|localhost|::1)
        ;;
    *)
        if [[ "$PUBLIC" -ne 1 ]]; then
            echo "Refusing to bind to non-loopback host '$HOST' without --public." >&2
            exit 1
        fi
        if [[ "${PUZZEL_AUTH_ENABLED:-false}" != "true" ]]; then
            echo "WARNING: authentication is disabled while binding to non-loopback host '$HOST'." >&2
        fi
        ;;
esac

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    exit 1
fi

echo "Starting Puzzel LLM Inference API"
echo "  checkpoint: $CHECKPOINT"
echo "  host: $HOST"
echo "  port: $PORT"
echo "  docs: http://$HOST:$PORT/docs"
echo "  health: http://$HOST:$PORT/health"
echo ""

export PUZZEL_CHECKPOINT="$CHECKPOINT"
exec ./.venv/bin/python -m puzzel.api --host "$HOST" --port "$PORT"
