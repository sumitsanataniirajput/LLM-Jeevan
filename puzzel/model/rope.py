"""Rotary Position Embeddings (RoPE) for the Puzzel Transformer.

Implements rotary position embeddings as described in
"RoFormer: Enhanced Transformer with Rotary Position Embedding"
(Su et al., 2021). RoPE encodes relative position information by rotating
pairs of dimensions in the query and key vectors, enabling length
extrapolation beyond the training context length.

No pretrained weights are used; all rotation parameters are derived from
the configured base frequency and dimension.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class RotaryPositionEmbeddings(nn.Module):
    """Apply rotary position embeddings to query and key tensors.

    The input tensors are expected to have shape
    ``[batch, num_heads, sequence_length, head_dim]`` where ``head_dim``
    is even. Pairs of adjacent dimensions are rotated by position-dependent
    angles, injecting relative position information while preserving
    the vector norm.

    Args:
        head_dim: Dimension of each attention head. Must be even.
        base: Base frequency for the rotation angles. Larger values
            encode longer-range positions more slowly.
        max_position_embeddings: Maximum number of positions to precompute
            cos/sin caches for. Actual sequences may be longer; the cache
            is extended on demand.
    """

    def __init__(
        self,
        head_dim: int,
        *,
        base: float = 10000.0,
        max_position_embeddings: int = 8192,
    ) -> None:
        super().__init__()
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError("head_dim must be a positive even integer")
        if base <= 0:
            raise ValueError("base must be positive")
        if max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive")

        self.head_dim = head_dim
        self.base = float(base)
        self.max_position_embeddings = int(max_position_embeddings)

        # Precompute inverse frequencies for half the head dimension.
        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute cos/sin caches up to max_position_embeddings.
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None
        self._cache_max_position = 0
        self._build_cache(max_position_embeddings)

    def _build_cache(self, max_position: int) -> None:
        """Build cos/sin caches for positions [0, max_position)."""
        positions = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)  # [max_position, head_dim/2]
        # Interleave so that freqs[i] applies to dimension pair (2i, 2i+1).
        emb = torch.cat((freqs, freqs), dim=-1)  # [max_position, head_dim]
        self._cos_cached = torch.cos(emb)[None, None, :, :]  # [1, 1, max_position, head_dim]
        self._sin_cached = torch.sin(emb)[None, None, :, :]
        self._cache_max_position = max_position

    def _ensure_cache(self, max_position: int, device: torch.device) -> None:
        """Extend the cos/sin cache if the requested position range exceeds it."""
        if max_position <= self._cache_max_position:
            return
        self._build_cache(max_position)
        self._cos_cached = self._cos_cached.to(device)
        self._sin_cached = self._sin_cached.to(device)

    def forward(self, query: Tensor, key: Tensor) -> tuple[Tensor, Tensor]:
        """Apply rotary embeddings to query and key in place.

        Args:
            query: Query tensor of shape [batch, num_heads, seq_len, head_dim].
            key: Key tensor of shape [batch, num_heads, seq_len, head_dim].

        Returns:
            Rotated (query, key) pair with the same shapes.
        """
        if query.shape != key.shape:
            raise ValueError("query and key must have identical shapes")
        if query.dim() != 4:
            raise ValueError("query and key must be 4D [batch, heads, seq, head_dim]")
        seq_len = query.shape[2]
        device = query.device
        self._ensure_cache(seq_len, device)

        cos = self._cos_cached[:, :, :seq_len, :].to(dtype=query.dtype)
        sin = self._sin_cached[:, :, :seq_len, :].to(dtype=query.dtype)

        query_rotated = _rotate_half(query)
        key_rotated = _rotate_half(key)

        query_out = query * cos + query_rotated * sin
        key_out = key * cos + key_rotated * sin
        return query_out, key_out


def _rotate_half(tensor: Tensor) -> Tensor:
    """Rotate half the hidden dimension by interleaving pairs.

    For a tensor of shape [..., head_dim], split into two halves along
    the last dimension and rotate each pair: (x0, x1, x2, x3, ...) ->
    (-x1, x0, -x3, x2, ...).
    """
    half = tensor.shape[-1] // 2
    first_half = tensor[..., :half]
    second_half = tensor[..., half:]
    rotated = torch.cat((-second_half, first_half), dim=-1)
    return rotated


def apply_rotary_pos_emb(
    query: Tensor,
    key: Tensor,
    position_ids: Tensor | None = None,
    *,
    head_dim: int,
    base: float = 10000.0,
) -> tuple[Tensor, Tensor]:
    """Apply RoPE to query and key using explicit position IDs.

    This is a functional wrapper useful when positions are not simply
    0..seq_len-1 (e.g., packed sequences or context continuation).

    Args:
        query: Query tensor [batch, num_heads, seq_len, head_dim].
        key: Key tensor [batch, num_heads, seq_len, head_dim].
        position_ids: Optional position IDs [batch, seq_len]. If None,
            uses sequential positions 0..seq_len-1.
        head_dim: Dimension of each attention head (must be even).
        base: Base frequency for rotation angles.

    Returns:
        Rotated (query, key) pair.
    """
    rope = RotaryPositionEmbeddings(head_dim, base=base)
    return rope(query, key)


__all__ = ["RotaryPositionEmbeddings", "apply_rotary_pos_emb"]