"""Streaming token generation for Puzzel inference.

Provides a generator-based streaming interface that yields tokens one at a
time as they are produced, suitable for chat-style incremental display.

No training, no weight modification, no external services are used.
"""

from __future__ import annotations

import time
from collections.abc import Generator
from typing import Any

import torch

from puzzel.inference import GenerationConfig, InferenceEngine
from puzzel.tokenizer import EOS_TOKEN_ID, TextTokenizer


def stream_generate(
    engine: InferenceEngine,
    prompt: str,
    config: GenerationConfig | None = None,
) -> Generator[str, None, None]:
    """Stream generated tokens one at a time.

    Yields each new token's decoded text as soon as it is produced.
    Uses KV-cache when available for efficient incremental decoding.

    Args:
        engine: Loaded InferenceEngine instance.
        prompt: Input text to continue. Must be a non-empty string.
        config: Generation configuration. Defaults to GenerationConfig().

    Yields:
        Decoded text for each newly generated token (incremental text,
        not cumulative). The caller can concatenate yields to build the
        full completion.

    Raises:
        ValueError: If the prompt is invalid.
    """
    if config is None:
        config = GenerationConfig()
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    # Encode prompt.
    prompt_ids = engine.tokenizer.encode(prompt, add_bos=True)
    max_context = max(1, engine.context_length - 1)
    if len(prompt_ids) > max_context:
        prompt_ids = prompt_ids[-max_context:]

    generated: list[int] = []
    stop_reason = "max_tokens"

    use_cache = (
        config.use_kv_cache
        and engine._use_kv_cache
        and engine.model.use_kv_cache
        and engine.model.kv_cache_manager is not None
    )

    if use_cache:
        engine.model.clear_kv_cache()
        cache_manager = engine.model.kv_cache_manager

        # Prime the cache with the full prompt.
        prompt_tensor = torch.tensor(
            [prompt_ids], dtype=torch.long, device=engine.device
        )
        logits, cache_manager = engine.model(prompt_tensor, use_cache=True)
        next_token = engine._sample_next_token(logits[0, -1], config)

        for _ in range(config.max_new_tokens):
            if config.stop_at_eos and next_token == EOS_TOKEN_ID:
                stop_reason = "eos"
                break

            generated.append(next_token)
            # Decode just this token for streaming output.
            token_text = engine.tokenizer.decode([next_token])
            yield token_text

            token_tensor = torch.tensor(
                [[next_token]], dtype=torch.long, device=engine.device
            )
            logits, cache_manager = engine.model(
                token_tensor, cache_manager=cache_manager, use_cache=True
            )
            next_token = engine._sample_next_token(logits[0, -1], config)
    else:
        # Non-cached path: recompute full context each step.
        for _ in range(config.max_new_tokens):
            active_ids = (prompt_ids + generated)[-engine.context_length :]
            input_ids = torch.tensor(
                active_ids, dtype=torch.long, device=engine.device
            ).unsqueeze(0)
            logits = engine.model(input_ids)
            next_token = engine._sample_next_token(logits[0, -1], config)

            if config.stop_at_eos and next_token == EOS_TOKEN_ID:
                stop_reason = "eos"
                break

            generated.append(next_token)
            token_text = engine.tokenizer.decode([next_token])
            yield token_text

    # The return value of a generator is accessible via StopIteration.value,
    # but callers typically just consume yields. We document stop_reason in
    # the docstring for advanced users.
    return stop_reason  # type: ignore[unreachable]


def stream_generate_cumulative(
    engine: InferenceEngine,
    prompt: str,
    config: GenerationConfig | None = None,
) -> Generator[str, None, None]:
    """Stream generated text cumulatively.

    Like :func:`stream_generate`, but yields the full text generated so far
    after each token (prompt + all tokens generated up to that point).
    Useful for displaying a growing completion in a UI.

    Args:
        engine: Loaded InferenceEngine instance.
        prompt: Input text to continue.
        config: Generation configuration.

    Yields:
        Cumulative text (prompt + all generated tokens so far).
    """
    if config is None:
        config = GenerationConfig()

    accumulated = ""
    for token_text in stream_generate(engine, prompt, config=config):
        accumulated += token_text
        yield prompt + accumulated