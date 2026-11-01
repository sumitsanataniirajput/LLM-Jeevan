# Puzzel Troubleshooting Guide

Common issues and solutions for the Puzzel AI platform.

## Installation

### SSL errors when installing packages

Direct `curl`/`wget` downloads are blocked. Use `pip install` instead:

```bash
pip install torch fastapi uvicorn httpx
```

If PyPI is also blocked, use the system Python packages already installed.

### NumPy not available

PyTorch may warn about NumPy not being installed. This is expected in
CPU-only environments and does not affect functionality.

```bash
pip install numpy  # Optional, for NumPy interop
```

## Training

### Out of memory during training

Use the `cpu_safe` configuration:

```bash
python -m puzzel --config configs/cpu_safe.toml
```

### Training fails with no data

Ensure documents exist under `data/raw/` with metadata sidecars:

```bash
ls data/raw/en/
ls data/raw/hi/
```

Each language needs ≥3 documents for train/val/test splits.

### Checkpoint not found

After training, checkpoints are saved to `checkpoints/`:

```bash
ls checkpoints/
```

Use `PUZZEL_CHECKPOINT` to point to the correct file.

## Inference

### "Checkpoint not found" error

The API server expects a checkpoint at startup. Set the path:

```bash
PUZZEL_CHECKPOINT=checkpoints/multilingual-first-run.best.pt \
python -m puzzel.api
```

### Generation is repetitive

This is expected for the small base model. Try:

- Higher temperature: `temperature=1.2`
- Top-p sampling: `top_p=0.9`
- More diverse prompts

### Slow generation

Enable KV-cache (default) and reduce context length:

```bash
PUZZEL_MAX_CONTEXT_LENGTH=256 python -m puzzel.api
```

## API

### Server won't start

Check if port 8000 is already in use:

```bash
PUZZEL_PORT=8001 python -m puzzel.api
```

### 401 Unauthorized

When `PUZZEL_AUTH_ENABLED=true`, include an API key:

```bash
curl -H "X-API-Key: puzzled-admin-key-001" http://localhost:8000/generate
```

### 429 Rate Limit Exceeded

Wait for the rate limit window to reset, or increase limits:

```bash
PUZZEL_RATE_LIMIT_MAX_REQUESTS=120 python -m puzzel.api
```

## Frontend

### Cannot connect to API

Ensure the API server is running on the same origin as the frontend.
Open the browser console to see connection errors.

### Messages not appearing

Check that the API server is reachable and the `/generate` endpoint
returns 200 OK.

## RAG

### PDF/DOCX extraction fails

Install optional dependencies:

```bash
pip install pypdf python-docx
```

### Retrieval returns no results

Ensure chunks have been added to the retriever and the query
contains keywords that match the chunk text.

## Agents

### Sandbox execution fails

Check that forbidden imports are not used. Allowed modules:
math, statistics, random, datetime, json, re, and more.

### Agent doesn't use tools

Tool selection uses keyword matching. Ensure tool descriptions
contain relevant keywords for the user's query.

## Getting Help

Check the admin dashboard at `http://localhost:8000/docs` for
system overview and health status.

For persistent issues, check `data/reports/` for evaluation reports
and `tests/test_phase9_regressions.py` for regression test details.