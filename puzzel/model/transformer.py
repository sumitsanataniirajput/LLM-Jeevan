"""Modernized decoder-only Transformer language model for Puzzel.

Supports both the original architecture (MHA, LayerNorm, GELU, learned
absolute positions) as a legacy mode and modern options (RoPE, RMSNorm,
SwiGLU, GQA, KV-cache) configurable through ``ModelConfig``. Every fresh
model is created with random initialization; no pretrained weights are
loaded or required.

Architecture options (all optional, defaults preserve legacy behavior):
    - use_rope: Rotary position embeddings instead of learned absolute positions.
    - use_rmsnorm: RMSNorm instead of LayerNorm.
    - use_swiglu: SwiGLU activation instead of GELU.
    - use_gqa: Grouped Query Attention instead of standard MHA.
    - use_kv_cache: Enable KV-cache for faster inference.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from puzzel.model.config import ModelConfig
from puzzel.model.grouped_query_attention import GroupedQueryAttention
from puzzel.model.kv_cache import KVCacheManager
from puzzel.model.rope import RotaryPositionEmbeddings
from puzzel.model.rmsnorm import RMSNorm
from puzzel.model.swiglu import SwiGLU


class AttentionNorm(nn.Module):
    """Selectable normalization: RMSNorm or LayerNorm."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.use_rmsnorm:
            self.norm = RMSNorm(config.d_model)
        else:
            self.norm = nn.LayerNorm(config.d_model)
        self.use_rmsnorm = config.use_rmsnorm

    def forward(self, hidden: Tensor) -> Tensor:
        return self.norm(hidden)


class CausalSelfAttention(nn.Module):
    """Multi-head or Grouped-Query self-attention with optional RoPE and KV-cache.

    When ``config.use_gqa`` is False, this behaves like standard MHA.
    When ``config.use_kv_cache`` is True and a cache manager is provided,
    keys/values are cached across forward calls for incremental decoding.

    A ``causal_mask`` buffer is exposed for backward compatibility with code
    that inspects the attention mask directly.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.effective_head_dim
        self.attention_dropout_prob = config.effective_attention_dropout

        if config.use_gqa:
            self.attention = GroupedQueryAttention(
                config.d_model,
                num_heads=config.num_heads,
                num_kv_heads=config.effective_num_kv_heads,
                dropout=config.effective_attention_dropout,
                bias=config.bias,
                head_dim=config.effective_head_dim,
            )
        else:
            self.attention = GroupedQueryAttention(
                config.d_model,
                num_heads=config.num_heads,
                num_kv_heads=config.num_heads,  # standard MHA
                dropout=config.effective_attention_dropout,
                bias=config.bias,
                head_dim=config.effective_head_dim,
            )

        self.rope: RotaryPositionEmbeddings | None = None
        if config.use_rope:
            self.rope = RotaryPositionEmbeddings(
                config.effective_head_dim,
                base=config.rope_base,
                max_position_embeddings=config.rope_max_position_embeddings,
            )

        # Backward-compatible causal mask buffer (legacy code may inspect it).
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool)),
            persistent=False,
        )

    def forward(
        self,
        hidden: Tensor,
        *,
        attention_mask: Tensor | None = None,
        cache_manager: KVCacheManager | None = None,
        layer_index: int | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """Compute self-attention.

        Args:
            hidden: Input tensor [batch, seq_len, d_model].
            attention_mask: Optional boolean mask.
            cache_manager: Optional KV-cache manager (required for caching).
            layer_index: Layer index for cache access.
            use_cache: Whether to return/update the KV-cache.

        Returns:
            (attention_output, updated_cache_or_none).
        """
        cache: Tensor | None = None
        if cache_manager is not None and layer_index is not None:
            layer_cache = cache_manager.get_layer_cache(layer_index)
            if not layer_cache.is_empty:
                cache = layer_cache.cache_pair

        output, updated_cache = self.attention(
            hidden,
            attention_mask=attention_mask,
            rope=self.rope,
            cache=cache,
            use_cache=use_cache,
        )

        if (
            use_cache
            and cache_manager is not None
            and layer_index is not None
            and updated_cache is not None
        ):
            layer_cache = cache_manager.get_layer_cache(layer_index)
            layer_cache.append(updated_cache[0], updated_cache[1])

        return output, updated_cache if use_cache else None


class FeedForward(nn.Module):
    """Position-wise feed-forward block: GELU MLP or SwiGLU."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.use_swiglu = config.use_swiglu
        d_model = config.d_model

        if config.use_swiglu:
            self.ff = SwiGLU(
                d_model,
                dropout=config.dropout,
                bias=config.bias,
            )
        else:
            expanded_width = d_model * config.mlp_ratio
            self.ff = nn.Sequential(
                nn.Linear(d_model, expanded_width, bias=config.bias),
                nn.GELU(),
                nn.Linear(expanded_width, d_model, bias=config.bias),
                nn.Dropout(config.dropout),
            )

    def forward(self, hidden: Tensor) -> Tensor:
        return self.ff(hidden)


class TransformerBlock(nn.Module):
    """Pre-norm attention and MLP residual block with modern options.

    Uses RMSNorm or LayerNorm depending on config.use_rmsnorm.
    Supports KV-cache for incremental decoding.
    """

    def __init__(self, config: ModelConfig, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.attention_norm = AttentionNorm(config)
        self.attention = CausalSelfAttention(config)
        self.attention_residual_dropout = nn.Dropout(config.dropout)
        self.mlp_norm = AttentionNorm(config)
        self.feed_forward = FeedForward(config)
        self.mlp_residual_dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden: Tensor,
        *,
        attention_mask: Tensor | None = None,
        cache_manager: KVCacheManager | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        attn_output, updated_cache = self.attention(
            self.attention_norm(hidden),
            attention_mask=attention_mask,
            cache_manager=cache_manager,
            layer_index=self.layer_index,
            use_cache=use_cache,
        )
        hidden = hidden + self.attention_residual_dropout(attn_output)

        ff_output = self.feed_forward(self.mlp_norm(hidden))
        hidden = hidden + self.mlp_residual_dropout(ff_output)

        return hidden, updated_cache


class TransformerLM(nn.Module):
    """Decoder-only language model initialized from scratch.

    Supports legacy and modern architectures via ModelConfig flags.
    When use_kv_cache=True, the model can maintain a KVCacheManager
    for fast incremental decoding.

    ``forward`` accepts token IDs shaped [batch, sequence] and returns
    next-token logits shaped [batch, sequence, vocab_size]. When
    use_cache=True, also returns updated caches for incremental decoding.
    """

    __constants__ = ["context_length", "vocab_size", "pad_token_id"]

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.context_length = config.context_length
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.use_kv_cache = config.use_kv_cache
        self.use_rope = config.use_rope
        self.num_layers = config.num_layers

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        if config.use_rope:
            # RoPE replaces learned position embeddings.
            self.position_embedding = None
        else:
            self.position_embedding = nn.Embedding(
                config.context_length, config.d_model
            )
        self.embedding_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            TransformerBlock(config, layer_index=i)
            for i in range(config.num_layers)
        )

        if config.use_rmsnorm:
            self.final_norm = RMSNorm(config.d_model)
        else:
            self.final_norm = nn.LayerNorm(config.d_model)

        self.language_model_head = nn.Linear(
            config.d_model, config.vocab_size, bias=False
        )

        # Initialize all learnable parameters from a normal distribution.
        self.apply(self._initialize_weights)
        if config.tie_embeddings:
            self.language_model_head.weight = self.token_embedding.weight

        # KV-cache manager (lazily initialized).
        self._kv_cache_manager: KVCacheManager | None = None

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
        # RMSNorm and SwiGLU initialize their own parameters.

    def _init_kv_cache_manager(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> KVCacheManager:
        """Initialize the KV-cache manager for the given batch."""
        head_dim = self.config.effective_head_dim
        num_kv_heads = self.config.effective_num_kv_heads
        manager = KVCacheManager(self.num_layers)
        manager.initialize_all(batch_size, num_kv_heads, head_dim, device, dtype)
        self._kv_cache_manager = manager
        return manager

    def clear_kv_cache(self) -> None:
        """Clear all KV-caches (call between independent sequences)."""
        if self._kv_cache_manager is not None:
            self._kv_cache_manager.clear_all()
        self._kv_cache_manager = None

    @property
    def kv_cache_manager(self) -> KVCacheManager | None:
        return self._kv_cache_manager

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        cache_manager: KVCacheManager | None = None,
        use_cache: bool = False,
    ) -> Tensor | tuple[Tensor, KVCacheManager | None]:
        """Run the model on input token IDs.

        Args:
            input_ids: Token IDs [batch, seq_len].
            attention_mask: Optional boolean attention mask.
            cache_manager: External KV-cache manager (for inference).
            use_cache: If True, return (logits, cache_manager).

        Returns:
            Logits [batch, seq_len, vocab_size], or
            (logits, cache_manager) when use_cache=True.
        """
        if input_ids.dim() != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        sequence_length = input_ids.size(1)
        if sequence_length == 0:
            raise ValueError("input sequence must not be empty")

        effective_context = sequence_length
        if cache_manager is not None and not cache_manager.caches[0].is_empty:
            effective_context += cache_manager.caches[0].seq_len
        if effective_context > self.context_length and not self.use_rope:
            # With RoPE, we can extrapolate; without it, context is fixed.
            raise ValueError(
                f"effective context length {effective_context} exceeds configured "
                f"context length {self.context_length}"
            )

        # Embedding + optional position embeddings.
        inputs_embeds = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            positions = torch.arange(
                sequence_length, device=input_ids.device
            ).unsqueeze(0)
            inputs_embeds = inputs_embeds + self.position_embedding(positions)
        hidden = self.embedding_dropout(inputs_embeds)

        # Use external cache manager or internal one.
        manager = cache_manager if cache_manager is not None else self._kv_cache_manager
        if use_cache and manager is None:
            manager = self._init_kv_cache_manager(
                input_ids.size(0), input_ids.device, inputs_embeds.dtype
            )

        for block in self.blocks:
            hidden, _ = block(
                hidden,
                attention_mask=attention_mask,
                cache_manager=manager,
                use_cache=use_cache,
            )

        hidden = self.final_norm(hidden)
        logits = self.language_model_head(hidden)

        if use_cache:
            return logits, manager
        return logits

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

    def model_info(self) -> dict[str, Any]:
        """Return static metadata about the model architecture."""
        return {
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "d_model": self.config.d_model,
            "num_layers": self.config.num_layers,
            "num_heads": self.config.num_heads,
            "num_kv_heads": self.config.effective_num_kv_heads,
            "head_dim": self.config.effective_head_dim,
            "mlp_ratio": self.config.mlp_ratio,
            "dropout": self.config.dropout,
            "use_rope": self.config.use_rope,
            "use_rmsnorm": self.config.use_rmsnorm,
            "use_swiglu": self.config.use_swiglu,
            "use_gqa": self.config.use_gqa,
            "use_kv_cache": self.config.use_kv_cache,
            "precision": self.config.precision,
            "tie_embeddings": self.config.tie_embeddings,
            "parameter_count": self.parameter_count(),
            "estimated_parameters": self.config.parameter_count_estimate,
        }


def count_parameters(model: nn.Module, *, trainable_only: bool = True) -> int:
    """Count unique model parameters, optionally including frozen tensors."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if not trainable_only or parameter.requires_grad
    )


__all__ = [
    "TransformerLM",
    "CausalSelfAttention",
    "FeedForward",
    "TransformerBlock",
    "AttentionNorm",
    "count_parameters",
]