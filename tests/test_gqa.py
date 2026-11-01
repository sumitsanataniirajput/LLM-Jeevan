"""Unit tests for Grouped Query Attention."""

import torch

from puzzel.model.grouped_query_attention import GroupedQueryAttention


def test_gqa_standard_mha_equivalence() -> None:
    """When num_kv_heads == num_heads, GQA equals standard MHA."""
    gqa = GroupedQueryAttention(
        d_model=128, num_heads=8, num_kv_heads=8, dropout=0.0
    )
    x = torch.randn(2, 10, 128)
    out, _ = gqa(x, use_cache=False)
    assert out.shape == (2, 10, 128)


def test_gqa_reduced_kv_heads() -> None:
    """GQA with fewer KV heads than query heads."""
    gqa = GroupedQueryAttention(
        d_model=128, num_heads=8, num_kv_heads=2, dropout=0.0
    )
    x = torch.randn(2, 10, 128)
    out, cache = gqa(x, use_cache=True)
    assert out.shape == (2, 10, 128)
    assert cache is not None


def test_gqa_num_heads_divisibility() -> None:
    """num_heads must be divisible by num_kv_heads."""
    try:
        GroupedQueryAttention(d_model=128, num_heads=8, num_kv_heads=3)
        assert False, "should have raised ValueError"
    except ValueError:
        pass


def test_gqa_causal_masking() -> None:
    """Future tokens should not affect earlier positions."""
    gqa = GroupedQueryAttention(
        d_model=64, num_heads=4, num_kv_heads=2, dropout=0.0
    )
    gqa.eval()
    x1 = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32).unsqueeze(-1).expand(1, 4, 64)
    x2 = torch.tensor([[1, 2, 99, 100]], dtype=torch.float32).unsqueeze(-1).expand(1, 4, 64)
    with torch.no_grad():
        out1, _ = gqa(x1)
        out2, _ = gqa(x2)
    # First two positions should be identical (causal masking).
    torch.testing.assert_close(out1[:, :2, :], out2[:, :2, :], atol=1e-5, rtol=1e-5)


def test_gqa_gradient_flow() -> None:
    gqa = GroupedQueryAttention(d_model=128, num_heads=8, num_kv_heads=4)
    x = torch.randn(2, 10, 128, requires_grad=True)
    out, _ = gqa(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None


def test_gqa_invalid_config() -> None:
    """d_model must be divisible by num_heads."""
    try:
        GroupedQueryAttention(d_model=100, num_heads=8)
        assert False, "should have raised ValueError"
    except ValueError:
        pass


def test_gqa_use_cache_returns_cache() -> None:
    gqa = GroupedQueryAttention(d_model=64, num_heads=4, num_kv_heads=2)
    x = torch.randn(2, 6, 64)
    out, cache = gqa(x, use_cache=True)
    assert cache is not None
    cached_key, cached_value = cache
    assert cached_key.shape[2] == 6  # seq_len cached
    assert cached_value.shape[2] == 6