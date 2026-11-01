# Puzzel Deployment Guide

This guide covers deploying the Puzzel AI platform locally and in production.
All components are designed for local deployment only; no external AI APIs
or pretrained weights are used.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Development](#local-development)
3. [Production Deployment](#production-deployment)
4. [Environment Variables](#environment-variables)
5. [Docker Deployment](#docker-deployment)
6. [Hardware Requirements](#hardware-requirements)
7. [Security Considerations](#security-considerations)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.11+
- PyTorch 2.x (CPU or CUDA build)
- At least 1 GB free RAM (2 GB recommended)
- At least 2 GB free disk space

Install Python dependencies:

```bash
pip install torch fastapi uvicorn httpx pytest
```

## Local Development

### 1. Clone and setup

```bash
git clone https://github.com/sumitsanataniirajput/LLM-Jeevan.git
cd LLM-Jeevan
git checkout arena/26b70af7-llm-jeevan
```

### 2. Prepare data

The pipeline uses local documents under `data/raw/`. Place your text files
there with metadata sidecars (see `data/raw/README.md`).

```bash
python -m puzzel data prepare data/raw data/prepared --config configs/tiny.toml
```

### 3. Train (optional)

Training is CPU-safe by default. A small training run takes under a minute:

```bash
python -m puzzel --config configs/tiny.toml
```

### 4. Start the API server

```bash
PUZZEL_CHECKPOINT=checkpoints/multilingual-first-run.best.pt \
python -m puzzel.api
```

Or using uvicorn directly:

```bash
uvicorn puzzel.api:app --host 0.0.0.0 --port 8000
```

### 5. Open the frontend

Open `frontend/index.html` in your browser. It connects to `http://localhost:8000`.

## Production Deployment

### Environment variables

Create a `.env` file from `.env.example`:

```bash
cp .env.example .env
# Edit .env with your values
```

Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `PUZZEL_HOST` | `0.0.0.0` | Bind host |
| `PUZZEL_PORT` | `8000` | Bind port |
| `PUZZEL_CHECKPOINT` | `checkpoints/multilingual-first-run.best.pt` | Checkpoint path |
| `PUZZEL_AUTH_ENABLED` | `false` | Enable API key authentication |
| `PUZZEL_DEVICE` | `auto` | Device: cpu, cuda, mps, auto |
| `PUZZEL_MAX_CONTEXT_LENGTH` | `512` | Max context length |
| `PUZZEL_USE_KV_CACHE` | `true` | Enable KV-cache for faster generation |

### Starting with authentication

```bash
PUZZEL_AUTH_ENABLED=true \
PUZZEL_CHECKPOINT=checkpoints/multilingual-first-run.best.pt \
python -m puzzel.api
```

Default credentials when auth is enabled:

| Username | Password | API Key | Role |
|----------|----------|---------|------|
| admin | admin | puzzled-admin-key-001 | admin |
| guest | (none) | (none) | guest |

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install torch fastapi uvicorn httpx

ENV PUZZEL_HOST=0.0.0.0
ENV PUZZEL_PORT=8000
ENV PUZZEL_DEVICE=cpu

EXPOSE 8000

CMD ["python", "-m", "puzzel.api"]
```

Build and run:

```bash
docker build -t puzzel-llm .
docker run -p 8000:8000 puzzel-llm
```

## Hardware Requirements

### Minimum (CPU)

- 2 CPU cores
- 1 GB available RAM
- 2 GB free disk space
- No GPU required

### Recommended (CPU)

- 4+ CPU cores
- 4+ GB available RAM
- 10+ GB free disk space

### GPU (CUDA)

- NVIDIA GPU with 4+ GB VRAM
- CUDA-capable PyTorch build
- `PUZZEL_DEVICE=cuda`

### Apple Silicon (MPS)

- M1/M2/M3 Mac
- `PUZZEL_DEVICE=mps`

## Security Considerations

1. **Do not expose the API publicly** without authentication and rate limiting.
2. Enable `PUZZEL_AUTH_ENABLED=true` in production.
3. Change default admin credentials.
4. Use HTTPS/TLS in front of the API (reverse proxy).
5. The model is small (239K parameters) and not suitable for production workloads
   without substantial retraining and evaluation.
6. All code execution in the agent sandbox is restricted; no network or file
   system access outside the temp directory.

## Troubleshooting

### Checkpoint not found

Ensure `PUZZEL_CHECKPOINT` points to a valid `.pt` file:

```bash
ls checkpoints/
PUZZEL_CHECKPOINT=checkpoints/your-model.pt python -m puzzel.api
```

### Out of memory

Use the `cpu_safe` config:

```bash
PUZZEL_MAX_CONTEXT_LENGTH=256 \
PUZZEL_MAX_BATCH_SIZE=1 \
python -m puzzel --config configs/cpu_safe.toml
```

### Port already in use

Change the port:

```bash
PUZZEL_PORT=8001 python -m puzzel.api
```

### Cannot connect to frontend

Ensure the API server is running and CORS is configured. The frontend connects
to `window.location.origin` by default.

## Support

For issues, see `docs/troubleshooting.md` or check the admin dashboard at
`/docs` when the API server is running.