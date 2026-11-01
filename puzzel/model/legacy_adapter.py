"""Legacy adapter for loading original Puzzel checkpoints into the modernized architecture.

Provides the original CausalSelfAttention, FeedForward, and TransformerBlock
implementations (MHA, LayerNorm, GELU, learned absolute positions) so that
checkpoints saved with ``puzzel-checkpoint-v1`` can be loaded into a
``LegacyTransformerLM`` that shares the same parameter layout as the original
model.

This adapter does NOT modify old checkpoints. It provides a read-only path for
inference and evaluation with existing v1 checkpoints.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from puzzel.model.config import ModelConfig


class LegacyCausalSelfAttention(nn.Module):
    """Original multi-head self-attention with lower-triangular causal mask."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.qkv_projection = nn.Linear(config.d_model, 3 * config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.d_model)
        self.attention_dropout = nn.Dropout(config.dropout)
        causal_mask = torch.tril(
            torch.ones(
                config.context_length,
                config.context_length,
                dtype=torch.bool,
            )
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, hidden: Tensor) -> Tensor:
        batch_size, sequence_length, model_width = hidden.shape
        query, key, value = self.qkv_projection(hidden).chunk(3, dim=-1)

        query = query.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = value.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )
        allowed = self.causal_mask[:sequence_length, :sequence_length]
        scores = scores.masked_fill(~allowed, -1.0e4)
        probabilities = torch.softmax(scores, dim=-1)
        probabilities = self.attention_dropout(probabilities)
        attended = torch.matmul(probabilities, value)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, model_width
        )
        return self.output_projection(attended)


class LegacyFeedForward(nn.Module):
    """Original position-wise GELU MLP."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        expanded_width = config.d_model * config.mlp_ratio
        self.layers = nn.Sequential(
            nn.Linear(config.d_model, expanded_width),
            nn.GELU(),
            nn.Linear(expanded_width, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, hidden: Tensor) -> Tensor:
        return self.layers(hidden)


class LegacyTransformerBlock(nn.Module):
    """Original pre-LayerNorm attention and MLP residual block."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = LegacyCausalSelfAttention(config)
        self.attention_residual_dropout = nn.Dropout(config.dropout)
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.feed_forward = LegacyFeedForward(config)
        self.mlp_residual_dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: Tensor) -> Tensor:
        hidden = hidden + self.attention_residual_dropout(
            self.attention(self.attention_norm(hidden))
        )
        hidden = hidden + self.mlp_residual_dropout(
            self.feed_forward(self.mlp_norm(hidden))
        )
        return hidden


class LegacyTransformerLM(nn.Module):
    """Legacy decoder-only model matching the original architecture exactly.

    Used for loading v1 checkpoints without modification. Parameter names and
    shapes match the original ``TransformerLM`` so ``load_state_dict(strict=True)``
    works directly.
    """

    __constants__ = ["context_length", "vocab_size", "pad_token_id"]

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.context_length = config.context_length
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_length, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            LegacyTransformerBlock(config) for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.language_model_head = nn.Linear(
            config.d_model, config.vocab_size, bias=False
        )

        self.apply(self._initialize_weights)
        if config.tie_embeddings:
            self.language_model_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.normal_(module.bias, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.normal_(module.weight, mean=1.0, std=0.02)
            if module.bias is not None:
                nn.init.normal_(module.bias, mean=0.0, std=0.02)

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        sequence_length = input_ids.size(1)
        if sequence_length == 0:
            raise ValueError("input sequence must not be empty")
        if sequence_length > self.context_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds configured context "
                f"length {self.context_length}"
            )

        positions = torch.arange(
            sequence_length, device=input_ids.device
        ).unsqueeze(0)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.embedding_dropout(hidden)
        for block in self.blocks:
            hidden = block(hidden)
        return self.language_model_head(self.final_norm(hidden))

    def parameter_count(self) -> int:
        """Return the number of unique trainable parameters."""
        return count_parameters(self)

    def verify_random_initialization(self) -> bool:
        """Check every parameter is finite and non-constant after random init."""
        parameters = dict(self.named_parameters())
        if not parameters:
            raise RuntimeError("model has no parameters to verify")
        for name, parameter in parameters.items():
            values = parameter.detach()
            if not torch.isfinite(values).all().item():
                raise RuntimeError(f"parameter {name} contains non-finite values")
            if not torch.count_nonzero(values).item():
                raise RuntimeError(f"parameter {name} is entirely zero")
            if values.numel() > 1 and values.min().item() == values.max().item():
                raise RuntimeError(
                    f"parameter {name} is constant, not randomly initialized"
                )
        return True


def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    """Count unique model parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


__all__ = [
    "LegacyCausalSelfAttention",
    "LegacyFeedForward",
    "LegacyTransformerBlock",
    "LegacyTransformerLM",
    "count_parameters",
]