"""Batch inference for Puzzel language models.

Provides batched generation for processing multiple prompts efficiently.
No training, no weight modification, no external services are used.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from puzzel.inference import GenerationConfig, InferenceEngine
from puzzel.tokenizer import EOS_TOKEN_ID, TextTokenizer


@dataclass
class BatchResult:
    """Result for a single prompt in a batch generation."""

    prompt: str
    generated_text: str
    num_generated_tokens: int
    stop_reason: str  # "eos", "max_tokens"
    generation_time_seconds: float


@dataclass
class BatchGenerationResult:
    """Result of batch generation across multiple prompts."""

    results: list[BatchResult]
    total_time_seconds: float
    model_info: dict[str, Any]
    config: GenerationConfig


def batch_generate(
    engine: InferenceEngine,
    prompts: list[str],
    config: GenerationConfig | None = None,
) -> BatchGenerationResult:
    """Generate completions for multiple prompts sequentially.

    Processes each prompt independently using the engine's generate method.
    This is a sequential batcher (not truly parallel) but provides a unified
    interface for batch processing with aggregate timing.

    Args:
        engine: Loaded InferenceEngine instance.
        prompts: List of input prompts. Each must be a non-empty string.
        config: Generation configuration. Defaults to GenerationConfig().

    Returns:
        BatchGenerationResult with per-prompt results and aggregate metrics.

    Raises:
        ValueError: If any prompt is invalid.
    """
    if config is None:
        config = GenerationConfig()
    if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
        raise ValueError("prompts must be a list of strings")
    if not prompts:
        raise ValueError("prompts must not be empty")

    start_time = time.perf_counter()
    results: list[BatchResult] = []

    for prompt in prompts:
        prompt_start = time.perf_counter()
        result = engine.generate(prompt, config=config)
        prompt_elapsed = time.perf_counter() - prompt_start

        results.append(
            BatchResult(
                prompt=prompt,
                generated_text=result.generated_text,
                num_generated_tokens=result.num_generated_tokens,
                stop_reason=result.stop_reason,
                generation_time_seconds=prompt_elapsed,
            )
        )

    total_elapsed = time.perf_counter() - start_time

    return BatchGenerationResult(
        results=results,
        total_time_seconds=total_elapsed,
        model_info=engine.model_info,
        config=config,
    )