"""CPU-compatible inference engine for Puzzel language model checkpoints.

Provides a clean, dependency-light inference interface with sampling
(temperature, top-k, top-p) for text generation from saved checkpoints.
No training, no weight modification, no external services are used.

Designed for local deployment only. The model is a small, from-scratch
decoder-only Transformer (239K parameters) trained on a tiny multilingual
public-domain corpus; generation quality is limited and should not be
claimed to match production-grade language models.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from puzzel.checkpoint import load_checkpoint
from puzzel.evaluation.runner import greedy_generate
from puzzel.model.transformer import TransformerLM
from puzzel.tokenizer import (
    BOS_TOKEN_ID,
    EOS_TOKEN_ID,
    PAD_TOKEN_ID,
    UNK_TOKEN_ID,
    TextTokenizer,
)
from puzzel.training.loop import resolve_device


@dataclass
class GenerationConfig:
    """Configuration for text generation.

    Attributes:
        max_new_tokens: Maximum number of new tokens to generate (>= 1).
        temperature: Sampling temperature. 0.0 means greedy (argmax).
            Values > 0 scale logits before sampling; higher values produce
            more diverse output. Must be >= 0.
        top_k: If set and > 0, restrict sampling to the top-k most likely
            tokens. None disables top-k filtering.
        top_p: If set and in (0, 1), use nucleus sampling: keep the smallest
            set of tokens whose cumulative probability exceeds top_p.
            None disables top-p filtering. When both top_k and top_p are set,
            top-k is applied first, then top-p.
        seed: Optional seed for reproducible sampling. Only affects sampling
            (not greedy decoding).
        stop_at_eos: If True, stop generation when the EOS token is produced.
        forbidden_token_ids: Token IDs that are never sampled (default:
            BOS, PAD, UNK).
    """

    max_new_tokens: int = 50
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    seed: int | None = None
    stop_at_eos: bool = True
    forbidden_token_ids: tuple[int, ...] = (BOS_TOKEN_ID, PAD_TOKEN_ID, UNK_TOKEN_ID)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_new_tokens, int)
            or isinstance(self.max_new_tokens, bool)
            or self.max_new_tokens <= 0
        ):
            raise ValueError("max_new_tokens must be a positive integer")
        if self.temperature is not None and (
            not isinstance(self.temperature, (int, float))
            or isinstance(self.temperature, bool)
            or self.temperature < 0
        ):
            raise ValueError("temperature must be a non-negative number")
        if self.top_k is not None and (
            not isinstance(self.top_k, int)
            or isinstance(self.top_k, bool)
            or self.top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer when set")
        if self.top_p is not None and (
            not isinstance(self.top_p, (int, float))
            or isinstance(self.top_p, bool)
            or not 0.0 < self.top_p <= 1.0
        ):
            raise ValueError("top_p must be in the range (0, 1] when set")


@dataclass
class GenerationResult:
    """Result of text generation."""

    generated_text: str
    prompt: str
    num_generated_tokens: int
    stop_reason: str  # "eos", "max_tokens"
    generation_time_seconds: float
    model_info: dict[str, Any]
    config: GenerationConfig


class InferenceEngine:
    """CPU-compatible inference engine for Puzzel checkpoints.

    Loads a saved checkpoint and provides ``generate`` for text completion.
    The model runs in eval mode under ``torch.inference_mode`` for memory
    efficiency. All computation happens on the requested device (CPU by
    default; CUDA is not available in this environment).
    """

    def __init__(
        self,
        model: TransformerLM,
        tokenizer: TextTokenizer,
        model_config: Any,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device) if isinstance(device, str) else device
        self.model = model.to(self.device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.context_length = int(getattr(model_config, "context_length", 128))
        self.vocab_size = int(getattr(model_config, "vocab_size", tokenizer.vocab_size))

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
    ) -> "InferenceEngine":
        """Load a Puzzel checkpoint and build an inference engine.

        The original checkpoint file is never modified. Only a read-only
        load is performed.
        """
        resolved_device = resolve_device(device) if isinstance(device, str) else device
        loaded = load_checkpoint(checkpoint_path, map_location=resolved_device)
        model = loaded.model.to(resolved_device)
        model.eval()
        return cls(
            model=model,
            tokenizer=loaded.tokenizer,
            model_config=loaded.model_config,
            device=resolved_device,
        )

    @property
    def model_info(self) -> dict[str, Any]:
        """Return static metadata about the loaded model."""
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "d_model": getattr(self.model_config, "d_model", None),
            "num_layers": getattr(self.model_config, "num_layers", None),
            "num_heads": getattr(self.model_config, "num_heads", None),
            "parameter_count": getattr(self.model, "parameter_count", lambda: None)(),
            "device": str(self.device),
            "precision": "fp32",
        }

    def _sample_next_token(self, logits: torch.Tensor, config: GenerationConfig) -> int:
        """Sample the next token ID from logits using temperature/top-k/top-p."""
        logits = logits.clone().float()

        # Temperature scaling. temperature == 0 means greedy (argmax).
        if config.temperature is not None and config.temperature > 0:
            logits = logits / config.temperature

        # Forbid special tokens from being sampled.
        for token_id in config.forbidden_token_ids:
            if token_id < logits.numel():
                logits[token_id] = float("-inf")

        # Top-k filtering: keep only the k most likely tokens.
        if config.top_k is not None and config.top_k > 0:
            top_k = min(int(config.top_k), logits.numel())
            kth_values, _ = torch.topk(logits, top_k)
            threshold = kth_values[-1]
            logits = torch.where(
                logits < threshold,
                torch.tensor(float("-inf"), device=logits.device),
                logits,
            )

        # Top-p (nucleus) filtering.
        if config.top_p is not None and 0.0 < config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)
            # Remove tokens whose cumulative probability exceeds top_p.
            sorted_remove = cumulative_probs > config.top_p
            # Shift right so the first token above the threshold is kept.
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            remove_indices = sorted_indices[sorted_remove]
            logits[remove_indices] = float("-inf")

        # If everything was masked out (defensive), fall back to uniform over
        # the non-forced tokens.
        if torch.all(torch.isneginf(logits)):
            logits = torch.zeros_like(logits)
            for token_id in config.forbidden_token_ids:
                if token_id < logits.numel():
                    logits[token_id] = float("-inf")

        probs = F.softmax(logits, dim=-1)

        if config.seed is not None:
            generator = torch.Generator(device=logits.device)
            generator.manual_seed(int(config.seed))
            next_token = int(
                torch.multinomial(probs, num_samples=1, generator=generator).item()
            )
        else:
            next_token = int(torch.multinomial(probs, num_samples=1).item())
        return next_token

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> GenerationResult:
        """Generate text continuation for a prompt.

        Args:
            prompt: Input text to continue. Must be a non-empty string.
            config: Generation configuration. Defaults to GenerationConfig().

        Returns:
            GenerationResult with the generated text and metadata.

        Raises:
            ValueError: If the prompt is invalid or generation config is bad.
        """
        if config is None:
            config = GenerationConfig()
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        start_time = time.perf_counter()

        # Greedy path (temperature == 0) reuses the project's existing,
        # well-tested greedy_generate for consistency.
        if config.temperature is not None and config.temperature == 0:
            generated_text = greedy_generate(
                self.model,
                self.tokenizer,
                prompt,
                device=self.device,
                max_new_tokens=config.max_new_tokens,
            )
            elapsed = time.perf_counter() - start_time
            return GenerationResult(
                generated_text=generated_text,
                prompt=prompt,
                num_generated_tokens=len(self.tokenizer.encode(generated_text)),
                stop_reason="max_tokens",
                generation_time_seconds=elapsed,
                model_info=self.model_info,
                config=config,
            )

        # Sampling path.
        prompt_ids = self.tokenizer.encode(prompt, add_bos=True)
        # Reserve at least one slot for generated tokens.
        max_context = max(1, self.context_length - 1)
        if len(prompt_ids) > max_context:
            prompt_ids = prompt_ids[-max_context:]

        generated: list[int] = []
        stop_reason = "max_tokens"

        for _ in range(config.max_new_tokens):
            active_ids = (prompt_ids + generated)[-self.context_length :]
            input_ids = torch.tensor(
                active_ids, dtype=torch.long, device=self.device
            ).unsqueeze(0)
            logits = self.model(input_ids)[0, -1]
            next_token = self._sample_next_token(logits, config)

            if config.stop_at_eos and next_token == EOS_TOKEN_ID:
                stop_reason = "eos"
                break

            generated.append(next_token)

        generated_text = self.tokenizer.decode(generated)
        elapsed = time.perf_counter() - start_time

        return GenerationResult(
            generated_text=generated_text,
            prompt=prompt,
            num_generated_tokens=len(generated),
            stop_reason=stop_reason,
            generation_time_seconds=elapsed,
            model_info=self.model_info,
            config=config,
        )

    def generate_greedy(self, prompt: str, max_new_tokens: int = 50) -> GenerationResult:
        """Convenience wrapper for greedy (argmax) generation."""
        return self.generate(
            prompt, GenerationConfig(max_new_tokens=max_new_tokens, temperature=0.0)
        )
