"""KV-cache for fast autoregressive generation in the Puzzel Transformer.

Implements a key-value cache that stores previously computed keys and
values for each attention layer, avoiding recomputation during
autoregressive decoding. This significantly speeds up generation by
reducing the per-step compute from O(seq_len^2) to O(seq_len).

The cache is layer-oriented: each Transformer layer maintains its own
(k, v) pair cache. Caches support batch inference and can be invalidated
between sequences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass
class KVCache:
    """Key-value cache for a single Transformer layer.

    Stores the cached keys and values for all positions processed so far,
    along with the current sequence length. Supports appending new
    (key, value) pairs during incremental decoding.

    Attributes:
        key_cache: Cached keys [batch, num_kv_heads, cached_seq_len, head_dim]
            or None if no cache has been initialized.
        value_cache: Cached values [batch, num_kv_heads, cached_seq_len, head_dim]
            or None if no cache has been initialized.
        seq_len: Number of tokens currently in the cache.
    """

    key_cache: Tensor | None = None
    value_cache: Tensor | None = None
    seq_len: int = 0

    @property
    def is_empty(self) -> bool:
        """Return True if the cache contains no tokens."""
        return self.seq_len == 0

    @property
    def cache_pair(self) -> tuple[Tensor, Tensor] | None:
        """Return the (key, value) cache pair, or None if empty."""
        if self.key_cache is None or self.value_cache is None:
            return None
        return self.key_cache, self.value_cache

    def initialize(
        self,
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Allocate empty cache tensors for the given batch and dimensions."""
        self.key_cache = torch.zeros(
            batch_size, num_kv_heads, 0, head_dim, device=device, dtype=dtype
        )
        self.value_cache = torch.zeros(
            batch_size, num_kv_heads, 0, head_dim, device=device, dtype=dtype
        )
        self.seq_len = 0

    def append(self, key: Tensor, value: Tensor) -> None:
        """Append new key/value tensors to the cache.

        Args:
            key: New keys [batch, num_kv_heads, new_seq_len, head_dim].
            value: New values [batch, num_kv_heads, new_seq_len, head_dim].
        """
        if self.key_cache is None or self.value_cache is None:
            raise RuntimeError("KVCache must be initialized before appending")
        if key.shape != value.shape:
            raise ValueError("key and value must have identical shapes")
        if key.shape[0] != self.key_cache.shape[0]:
            raise ValueError("batch size mismatch between cache and new tensors")
        if key.shape[1] != self.key_cache.shape[1]:
            raise ValueError("num_kv_heads mismatch between cache and new tensors")
        if key.shape[3] != self.key_cache.shape[3]:
            raise ValueError("head_dim mismatch between cache and new tensors")

        self.key_cache = torch.cat([self.key_cache, key], dim=2)
        self.value_cache = torch.cat([self.value_cache, value], dim=2)
        self.seq_len += key.shape[2]

    def get(self) -> tuple[Tensor, Tensor]:
        """Return the current (key, value) cache pair."""
        if self.key_cache is None or self.value_cache is None:
            raise RuntimeError("KVCache has not been initialized")
        return self.key_cache, self.value_cache

    def clear(self) -> None:
        """Clear the cache, removing all stored tokens."""
        self.key_cache = None
        self.value_cache = None
        self.seq_len = 0

    def slice(self, end_pos: int) -> tuple[Tensor, Tensor]:
        """Return a sliced view of the cache up to end_pos.

        Used when context length limits require truncation.
        """
        if self.key_cache is None or self.value_cache is None:
            raise RuntimeError("KVCache has not been initialized")
        return self.key_cache[:, :, :end_pos, :], self.value_cache[:, :, :end_pos, :]


@dataclass
class KVCacheManager:
    """Manages KV-caches for all Transformer layers.

    Creates one KVCache per attention layer and provides methods to
    initialize, retrieve, append, and clear all caches at once.
    """

    num_layers: int
    caches: list[KVCache] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.caches = [KVCache() for _ in range(self.num_layers)]

    def initialize_all(
        self,
        batch_size: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        """Initialize all layer caches."""
        for cache in self.caches:
            cache.initialize(batch_size, num_kv_heads, head_dim, device, dtype)

    def get_layer_cache(self, layer_index: int) -> KVCache:
        """Return the KV-cache for a specific layer."""
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(f"layer index {layer_index} out of range [0, {self.num_layers})")
        return self.caches[layer_index]

    def get_all(self) -> list[tuple[Tensor, Tensor] | None]:
        """Return all layer cache pairs (or None for empty caches)."""
        return [cache.cache_pair for cache in self.caches]

    def clear_all(self) -> None:
        """Clear all layer caches."""
        for cache in self.caches:
            cache.clear()

    @property
    def seq_len(self) -> int:
        """Return the sequence length of the first non-empty cache, or 0."""
        for cache in self.caches:
            if not cache.is_empty:
                return cache.seq_len
        return 0


__all__ = ["KVCache", "KVCacheManager"]