"""Unit tests for KV-cache."""

import torch

from puzzel.model.kv_cache import KVCache, KVCacheManager


def test_kv_cache_initialize_and_append() -> None:
    cache = KVCache()
    cache.initialize(
        batch_size=2, num_kv_heads=4, head_dim=32,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    assert cache.seq_len == 0
    key = torch.randn(2, 4, 3, 32)
    value = torch.randn(2, 4, 3, 32)
    cache.append(key, value)
    assert cache.seq_len == 3
    cached_key, cached_value = cache.get()
    assert cached_key.shape == (2, 4, 3, 32)
    assert cached_value.shape == (2, 4, 3, 32)


def test_kv_cache_clear() -> None:
    cache = KVCache()
    cache.initialize(
        batch_size=2, num_kv_heads=4, head_dim=32,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    key = torch.randn(2, 4, 3, 32)
    value = torch.randn(2, 4, 3, 32)
    cache.append(key, value)
    assert cache.seq_len == 3
    cache.clear()
    assert cache.is_empty
    assert cache.seq_len == 0


def test_kv_cache_manager_all_layers() -> None:
    manager = KVCacheManager(num_layers=6)
    assert len(manager.caches) == 6
    manager.initialize_all(
        batch_size=2, num_kv_heads=4, head_dim=32,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    for cache in manager.caches:
        # After initialization, caches have allocated tensors but no data yet.
        assert cache.key_cache is not None
        assert cache.value_cache is not None
    assert manager.seq_len == 0


def test_kv_cache_manager_clear_all() -> None:
    manager = KVCacheManager(num_layers=4)
    manager.initialize_all(
        batch_size=2, num_kv_heads=2, head_dim=16,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    for cache in manager.caches:
        key = torch.randn(2, 2, 5, 16)
        value = torch.randn(2, 2, 5, 16)
        cache.append(key, value)
    assert manager.seq_len == 5
    manager.clear_all()
    for cache in manager.caches:
        assert cache.is_empty


def test_kv_cache_get_layer_cache() -> None:
    manager = KVCacheManager(num_layers=3)
    layer_cache = manager.get_layer_cache(1)
    assert isinstance(layer_cache, KVCache)
    try:
        manager.get_layer_cache(5)
        assert False, "should have raised IndexError"
    except IndexError:
        pass


def test_kv_cache_slice() -> None:
    cache = KVCache()
    cache.initialize(
        batch_size=1, num_kv_heads=2, head_dim=8,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    key = torch.randn(1, 2, 10, 8)
    value = torch.randn(1, 2, 10, 8)
    cache.append(key, value)
    sliced_key, sliced_value = cache.slice(5)
    assert sliced_key.shape == (1, 2, 5, 8)
    assert sliced_value.shape == (1, 2, 5, 8)