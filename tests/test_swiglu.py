"""Unit tests for SwiGLU activation."""

import torch

from puzzel.model.swiglu import SwiGLU, swiglu_feed_forward


def test_swiglu_output_shape() -> None:
    swiglu = SwiGLU(64)
    x = torch.randn(2, 10, 64)
    y = swiglu(x)
    assert y.shape == x.shape


def test_swiglu_default_hidden_dim() -> None:
    swiglu = SwiGLU(64)
    # Default hidden_dim = (8 * 64) // 3 = 170
    assert swiglu.hidden_dim == 170


def test_swiglu_custom_hidden_dim() -> None:
    swiglu = SwiGLU(64, hidden_dim=128)
    x = torch.randn(2, 10, 64)
    y = swiglu(x)
    assert y.shape == x.shape


def test_swiglu_nonlinearity() -> None:
    swiglu = SwiGLU(64)
    x1 = torch.randn(2, 10, 64)
    x2 = torch.randn(2, 10, 64)
    y1 = swiglu(x1)
    y2 = swiglu(x2)
    # SwiGLU is nonlinear; different inputs should give different outputs.
    assert not torch.equal(y1, y2)


def test_swiglu_gradient_flow() -> None:
    swiglu = SwiGLU(64)
    x = torch.randn(2, 10, 64, requires_grad=True)
    y = swiglu(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.count_nonzero(x.grad) > 0


def test_swiglu_feed_forward_factory() -> None:
    ff = swiglu_feed_forward(256)
    x = torch.randn(2, 10, 256)
    y = ff(x)
    assert y.shape == x.shape


def test_swiglu_dropout() -> None:
    swiglu = SwiGLU(64, dropout=0.5)
    x = torch.randn(2, 10, 64)
    y_train = swiglu(x)
    swiglu.eval()
    y_eval = swiglu(x)
    # Dropout should not change output shape.
    assert y_train.shape == y_eval.shape