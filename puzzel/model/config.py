"""Validated architecture hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass

from puzzel.tokenizer import PAD_TOKEN_ID, VOCAB_SIZE


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Settings for a compact causal decoder language model."""

    vocab_size: int = VOCAB_SIZE
    context_length: int = 128
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.1
    pad_token_id: int = PAD_TOKEN_ID
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        positive_ints = {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "mlp_ratio": self.mlp_ratio,
        }
        for name, value in positive_ints.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id must be a valid vocabulary ID")
        if not isinstance(self.tie_embeddings, bool):
            raise ValueError("tie_embeddings must be a boolean")
