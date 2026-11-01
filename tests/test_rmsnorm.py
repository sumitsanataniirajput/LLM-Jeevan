"""Unit tests for RMSNorm."""

import torch

from puzzel.model.rmsnorm import RMSNorm


def test_rmsnorm_output_shape() -> None:
    norm = RMSNorm(64)
    x = torch.randn(2, 10, 64)
    y = norm(x)
    assert y.shape == x.shape


def test_rmsnorm_normalizes_to_unit_rms() -> None:
    norm = RMSNorm(64, epsilon=1e-6)
    x = torch.randn(4, 16, 64) * 10.0  # large values to test normalization
    y = norm(x)
    # With weight=1, output should have RMS ≈ 1 along last dim.
    rms = torch.sqrt((y**2).mean(dim=-1))
    # Weight is initialized to 1, so RMS ≈ 1 (slightly affected by epsilon).
    assert torch.allclose(rms, torch.ones_like(rms), atol=0.1, rtol=0.1)


def test_rmsnorm_preserves_dtype() -> None:
    norm = RMSNorm(64)
    x = torch.randn(2, 10, 64, dtype=torch.float16)
    y = norm(x)
    assert y.dtype == torch.float16


def test_rmsnorm_invalid_dim() -> None:
    norm = RMSNorm(64)
    x = torch.randn(2, 10, 32)
    try:
        norm(x)
        assert False, "should have raised ValueError"
    except ValueError:
        pass


def test_rmsnorm_zero_dim_raises() -> None:
    try:
        RMSNorm(0)
        assert False, "should have raised ValueError"
    except ValueError:
        pass


def test_rmsnorm_learnable_weight() -> None:
    norm = RMSNorm(64)
    assert norm.weight is not None
    assert torch.allclose(norm.weight, torch.ones(64))
    # After training step, weight should change.
    x = torch.randn(2, 10, 64, requires_grad=True)
    y = norm(x)
    loss = y.sum()
    loss.backward()
    assert norm.weight.grad is not None