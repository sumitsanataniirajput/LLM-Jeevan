"""Grouped Query Attention (GQA) for the Puzzel Transformer.

Implements Grouped Query Attention as described in
"GQA: Training Generalized Multi-Query Transformer Models from Scratch"
(Ainslie et al., 2023). GQA uses multiple query heads that share a
smaller number of key/value heads, reducing KV-cache memory and
inference compute while maintaining quality close to standard MHA.

When ``num_kv_heads == num_heads``, GQA is equivalent to standard
multi-head attention, preserving full backward compatibility.

No pretrained weights are used; all projection weights are randomly
initialized.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention with optional RoPE integration.

    Projects queries, keys, and values separately. The number of KV
    heads divides the number of query heads evenly; each KV head is
    shared by ``num_heads // num_kv_heads`` query heads. Keys and
    values are expanded (via repeat-interleave) to match query heads
    before attention computation.

    Args:
        d_model: Model dimension.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value heads. Must divide ``num_heads``
            and be <= ``num_heads``. Defaults to ``num_heads`` (standard MHA).
        dropout: Dropout probability for attention probabilities.
        bias: Whether to use bias terms in linear projections.
        head_dim: Optional explicit head dimension. Defaults to
            ``d_model // num_heads``.
    """

    def __init__(
        self,
        d_model: int,
        *,
        num_heads: int,
        num_kv_heads: int | None = None,
        dropout: float = 0.0,
        bias: bool = False,
        head_dim: int | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else d_model // num_heads
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.num_heads * self.head_dim != d_model:
            raise ValueError("num_heads * head_dim must equal d_model")

        if num_kv_heads is None:
            num_kv_heads = num_heads
        if num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be a positive integer")
        if num_kv_heads > num_heads:
            raise ValueError("num_kv_heads must not exceed num_heads")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.dropout_prob = float(dropout)
        self.use_bias = bool(bias)

        self.q_projection = nn.Linear(d_model, num_heads * self.head_dim, bias=bias)
        self.k_projection = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.v_projection = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=bias)
        self.output_projection = nn.Linear(d_model, d_model, bias=bias)
        self.attention_dropout = nn.Dropout(dropout)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize projection weights."""
        for module in (
            self.q_projection,
            self.k_projection,
            self.v_projection,
            self.output_projection,
        ):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        hidden: Tensor,
        *,
        attention_mask: Tensor | None = None,
        rope: nn.Module | None = None,
        cache: Tensor | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        """Compute grouped query attention.

        Args:
            hidden: Input tensor [batch, seq_len, d_model].
            attention_mask: Optional boolean mask [batch, seq_len, seq_len]
                where True means "keep" (causal mask expected).
            rope: Optional RoPE module to apply to queries and keys.
            cache: Optional cached (key, value) tensors from previous steps.
            use_cache: If True, return the updated (key, value) cache.

        Returns:
            Tuple of (attention_output, updated_cache).
            ``attention_output`` has shape [batch, seq_len, d_model].
            ``updated_cache`` is None unless ``use_cache=True``.
        """
        batch_size, seq_len, _ = hidden.shape

        # Project and reshape queries.
        query = self.q_projection(hidden)
        query = query.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]

        # Project and reshape keys/values (fewer heads under GQA).
        key = self.k_projection(hidden)
        key = key.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)  # [batch, num_kv_heads, seq_len, head_dim]
        value = self.v_projection(hidden)
        value = value.view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)

        # Apply RoPE separately to query and key since they may have
        # different head counts under GQA.
        if rope is not None:
            query, _ = rope(query, query.clone())
            key, _ = rope(key, key.clone())

        # Handle KV-cache: prepend cached keys/values.
        updated_cache: Tensor | None = None
        if cache is not None:
            cached_key, cached_value = cache
            key = torch.cat([cached_key, key], dim=2)
            value = torch.cat([cached_value, value], dim=2)
        if use_cache:
            updated_cache = (key, value)

        # Expand KV heads to match query heads: [batch, num_kv_heads, ...]
        # -> [batch, num_heads, ...] by repeating each KV head.
        if self.num_queries_per_kv > 1:
            key = key.repeat_interleave(
                self.num_queries_per_kv, dim=1
            )  # [batch, num_heads, seq_len, head_dim]
            value = value.repeat_interleave(
                self.num_queries_per_kv, dim=1
            )

        full_seq_len = key.shape[2]

        # Compute attention scores.
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale

        # Apply causal mask.
        causal_mask = torch.tril(
            torch.ones(seq_len, full_seq_len, dtype=torch.bool, device=hidden.device)
        )
        if attention_mask is not None:
            causal_mask = causal_mask & attention_mask.bool()
        scores = scores.masked_fill(~causal_mask[None, None, :, :], float("-inf"))

        probabilities = torch.softmax(scores, dim=-1)
        probabilities = self.attention_dropout(probabilities)

        attended = torch.matmul(probabilities, value)
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        return self.output_projection(attended), updated_cache


__all__ = ["GroupedQueryAttention"]