# Puzzel API Documentation

The Puzzel API is a FastAPI-based REST service for local text generation.

## Base URL

```
http://localhost:8000
```

## Authentication

Authentication is disabled by default. Enable it with `PUZZEL_AUTH_ENABLED=true`.
When enabled, send your API key in the `X-API-Key` header:

```bash
curl -H "X-API-Key: puzzled-admin-key-001" http://localhost:8000/model/info
```

## Endpoints

### GET /

Service info and links.

```bash
curl http://localhost:8000/
```

### GET /health

Health check. Returns whether the model is loaded.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu"
}
```

### GET /model/info

Model metadata including architecture flags and limitations.

```bash
curl http://localhost:8000/model/info
```

**Response:**
```json
{
  "vocab_size": 2048,
  "context_length": 128,
  "d_model": 128,
  "num_layers": 2,
  "num_heads": 4,
  "parameter_count": 239360,
  "device": "cpu",
  "precision": "fp32",
  "checkpoint_path": "checkpoints/multilingual-first-run.best.pt",
  "limitations": [
    "Small model (239K parameters)...",
    "Generation quality is limited..."
  ]
}
```

### POST /generate

Generate text continuation for a prompt.

**Request:**
```json
{
  "prompt": "Hello, how are you?",
  "max_new_tokens": 50,
  "temperature": 1.0,
  "top_k": null,
  "top_p": null,
  "seed": null,
  "stop_at_eos": true
}
```

**Response:**
```json
{
  "generated_text": " I'm doing well, thank you!",
  "prompt": "Hello, how are you?",
  "num_generated_tokens": 7,
  "stop_reason": "max_tokens",
  "generation_time_seconds": 0.05,
  "model_info": { ... }
}
```

**Parameters:**
- `prompt` (string, required): Input text to continue.
- `max_new_tokens` (integer, 1-512, default 50): Max tokens to generate.
- `temperature` (float, 0-2, default 1.0): Sampling temperature. 0 = greedy.
- `top_k` (integer, optional): Top-k filtering.
- `top_p` (float, 0-1, optional): Nucleus sampling.
- `seed` (integer, optional): Random seed for reproducible sampling.
- `stop_at_eos` (boolean, default true): Stop at EOS token.

## Rate Limiting

When authentication is enabled, rate limiting applies: 60 requests per minute
per client IP. Configure with `PUZZEL_RATE_LIMIT_MAX_REQUESTS` and
`PUZZEL_RATE_LIMIT_WINDOW_SECONDS`.

## Limitations

- Small model (239K parameters) trained on a tiny corpus (~174K tokens).
- Generation quality is limited; expect repetitive output.
- Context length is 128 tokens; longer prompts are truncated from the left.
- CPU-only inference; no GPU acceleration available in this environment.
- Exact-match accuracy on factual questions is 0.0% on the evaluation suite.
- Not suitable for production workloads without substantial retraining.