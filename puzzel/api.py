"""FastAPI REST API for Puzzel language model inference.

Endpoints:
    GET  /health        - service health check
    GET  /model/info    - loaded model metadata
    POST /generate      - text generation with sampling controls

Designed for local deployment only. The model is a small, from-scratch
decoder-only Transformer trained on a tiny multilingual public-domain
corpus; generation quality is limited. No public deployment or paid
infrastructure is implied by this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from puzzel.inference import (
    GenerationConfig,
    InferenceEngine,
)

app = FastAPI(
    title="Puzzel LLM Inference API",
    description=(
        "Local REST API for text generation with the Puzzel multilingual "
        "language model. The model is a small (239K parameter), from-scratch "
        "decoder-only Transformer trained on a tiny public-domain corpus. "
        "Generation quality is limited; see /model/info for details."
    ),
    version="0.1.0",
)

# Global inference engine, populated on startup.
_engine: InferenceEngine | None = None
_checkpoint_path: str | Path | None = None


class GenerateRequest(BaseModel):
    """Request body for text generation."""

    prompt: str = Field(
        ...,
        min_length=1,
        description="Input text to continue (non-empty, UTF-8).",
        examples=["Hello, how are you?", "नमस्ते, आप कैसे हैं?", "Hola, ¿cómo estás?"],
    )
    max_new_tokens: int = Field(
        50,
        ge=1,
        le=512,
        description="Maximum number of new tokens to generate.",
    )
    temperature: float = Field(
        1.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. 0 = greedy (argmax); higher = more diverse.",
    )
    top_k: int | None = Field(
        None,
        ge=1,
        description="Restrict sampling to top-k most likely tokens.",
    )
    top_p: float | None = Field(
        None,
        gt=0.0,
        le=1.0,
        description="Nucleus sampling: keep tokens with cumulative prob <= top_p.",
    )
    seed: int | None = Field(
        None,
        description="Optional seed for reproducible sampling.",
    )
    stop_at_eos: bool = Field(
        True,
        description="Stop generation at the EOS token.",
    )

    @field_validator("prompt")
    @classmethod
    def _prompt_not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("prompt must be a non-empty string")
        return value


class GenerateResponse(BaseModel):
    """Response body for text generation."""

    generated_text: str
    prompt: str
    num_generated_tokens: int
    stop_reason: str
    generation_time_seconds: float
    model_info: dict[str, Any]


class ModelInfoResponse(BaseModel):
    """Response body for model metadata."""

    vocab_size: int | None
    context_length: int | None
    d_model: int | None
    num_layers: int | None
    num_heads: int | None
    parameter_count: int | None
    device: str | None
    precision: str | None
    checkpoint_path: str | None
    limitations: list[str]


class HealthResponse(BaseModel):
    """Response body for health check."""

    status: str
    model_loaded: bool
    device: str | None


_LIMITATIONS = [
    "Small model (239K parameters) trained on a tiny corpus (49 docs, ~174K tokens).",
    "Generation quality is limited; expect repetitive output under greedy decoding.",
    "Exact-match accuracy on factual questions is 0.0% on the evaluation suite.",
    "Greedy decoding is deterministic but often repetitive; use temperature > 0 for variety.",
    "Context length is 128 tokens; longer prompts are truncated from the left.",
    "CPU-only inference; no GPU acceleration available in this environment.",
    "Not suitable for production workloads without substantial retraining and evaluation.",
]


@app.on_event("startup")
def _load_model_on_startup() -> None:
    """Load the default checkpoint on server startup.

    The checkpoint path can be overridden via the PUZZEL_CHECKPOINT
    environment variable. Defaults to checkpoints/multilingual-first-run.best.pt.
    """
    global _engine, _checkpoint_path
    import os

    default = Path("checkpoints/multilingual-first-run.best.pt")
    _checkpoint_path = Path(os.environ.get("PUZZEL_CHECKPOINT", str(default)))
    if not _checkpoint_path.exists():
        # Try repo-root-relative path.
        alt = Path(__file__).resolve().parents[1] / _checkpoint_path
        if alt.exists():
            _checkpoint_path = alt
        else:
            raise FileNotFoundError(
                f"Checkpoint not found: {_checkpoint_path} (tried {alt})"
            )
    _engine = InferenceEngine.from_checkpoint(_checkpoint_path, device="cpu")


def _get_engine() -> InferenceEngine:
    if _engine is None:
        raise HTTPException(
            status_code=503,
            detail="Inference engine not initialized. Is the server starting up?",
        )
    return _engine


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    """Service health check. Reports whether the model is loaded."""
    engine = _get_engine()
    return HealthResponse(
        status="ok",
        model_loaded=engine is not None,
        device=str(engine.device),
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["model"])
def model_info() -> ModelInfoResponse:
    """Return static metadata about the loaded model, including limitations."""
    engine = _get_engine()
    info = engine.model_info
    return ModelInfoResponse(
        vocab_size=info.get("vocab_size"),
        context_length=info.get("context_length"),
        d_model=info.get("d_model"),
        num_layers=info.get("num_layers"),
        num_heads=info.get("num_heads"),
        parameter_count=info.get("parameter_count"),
        device=info.get("device"),
        precision=info.get("precision"),
        checkpoint_path=str(_checkpoint_path) if _checkpoint_path else None,
        limitations=_LIMITATIONS,
    )


@app.post("/generate", response_model=GenerateResponse, tags=["generation"])
def generate(request: GenerateRequest) -> GenerateResponse:
    """Generate a text continuation for the given prompt.

    Supports configurable temperature, top-k, top-p, and seed for controlling
    sampling behavior. temperature=0 uses deterministic greedy decoding.
    """
    engine = _get_engine()
    try:
        config = GenerationConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            seed=request.seed,
            stop_at_eos=request.stop_at_eos,
        )
        result = engine.generate(request.prompt, config=config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surface unexpected errors as 500
        raise HTTPException(
            status_code=500, detail=f"Generation failed: {exc}"
        ) from exc

    return GenerateResponse(
        generated_text=result.generated_text,
        prompt=result.prompt,
        num_generated_tokens=result.num_generated_tokens,
        stop_reason=result.stop_reason,
        generation_time_seconds=result.generation_time_seconds,
        model_info=result.model_info,
    )


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    """Root endpoint with service description and links."""
    return {
        "service": "Puzzel LLM Inference API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model/info",
        "generate": "/generate",
        "note": "Local-only deployment. Do not expose publicly without review.",
    }


def create_app(checkpoint_path: str | Path | None = None) -> FastAPI:
    """Create a FastAPI app with an optional checkpoint override.

    Useful for testing without touching the global startup loader.
    """
    global _engine, _checkpoint_path
    if checkpoint_path is not None:
        _checkpoint_path = Path(checkpoint_path)
        _engine = InferenceEngine.from_checkpoint(_checkpoint_path, device="cpu")
    return app


if __name__ == "__main__":  # pragma: no cover - manual execution only
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Run the Puzzel LLM Inference API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
