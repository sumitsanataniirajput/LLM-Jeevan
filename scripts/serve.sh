#!/usr/bin/env bash
# Start the Puzzel LLM Inference API server (local deployment only).
#
# Usage:
#   ./scripts/serve.sh [--checkpoint PATH] [--port PORT] [--host HOST]
#
# The server binds to 0.0.0.0 so it is reachable from the Arena live-preview
# proxy. It does NOT expose the model publicly; it is intended for local
# inspection behind the sandbox proxy.
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -x .venv/bin/python ]]; then
    echo "Project virtual environment not found. Run ./scripts/bootstrap.sh first." >&2
    exit 1
fi

CHECKPOINT="checkpoints/multilingual-first-run.best.pt"
HOST="0.0.0.0"
PORT=8000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)
            CHECKPOINT="$2"; shift 2 ;;
        --port)
            PORT="$2"; shift 2 ;;
        --host)
            HOST="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--checkpoint PATH] [--port PORT] [--host HOST]"
            exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

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
