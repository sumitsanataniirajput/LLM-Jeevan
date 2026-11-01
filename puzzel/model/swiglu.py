"""SwiGLU activation for the Puzzel Transformer.

Implements the SwiGLU gated activation as described in
"GLU Variants Improve Transformer" (Shazeer, 2020). SwiGLU applies
the SiLU (Swish) gating mechanism to a gated linear unit::

    SwiGLU(x, W, V, b, c) = Swish(x @ W + b) ⊗ (x @ V + c)

This is more parameter-efficient than standard GELU MLPs and is used
in modern architectures such as LLaMA.

No pretrained weights are used; all projection weights are randomly
initialized.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SwiGLU(nn.Module):
    """SwiGLU gated activation block.

    Splits the expanded feature dimension into two halves: one passed
    through SiLU gating, the other used as a linear gate. The output
    is their elementwise product, projected back to the model dimension.

    Args:
        input_dim: Dimension of the input features.
        hidden_dim: Dimension of the hidden (gated) layer. Typically
            set to ``8 * input_dim // 3`` for parameter efficiency
            (approximately matching a 4x GELU MLP parameter count).
        dropout: Dropout probability applied after the output projection.
        bias: Whether to use bias terms in the linear projections.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be a positive integer")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        self.input_dim = input_dim
        if hidden_dim is None:
            # Default: ~4/3 ratio so that total params ≈ 4x input_dim^2,
            # matching a standard GELU MLP with mlp_ratio=4.
            hidden_dim = (8 * input_dim) // 3
        self.hidden_dim = int(hidden_dim)
        self.dropout_prob = float(dropout)
        self.use_bias = bool(bias)

        self.w_gate = nn.Linear(input_dim, self.hidden_dim, bias=self.use_bias)
        self.w_up = nn.Linear(input_dim, self.hidden_dim, bias=self.use_bias)
        self.w_down = nn.Linear(self.hidden_dim, input_dim, bias=self.use_bias)
        self.dropout = nn.Dropout(self.dropout_prob)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights with a small normal distribution."""
        for module in (self.w_gate, self.w_up, self.w_down):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, hidden: Tensor) -> Tensor:
        """Apply SwiGLU activation.

        Args:
            hidden: Input tensor of shape [..., input_dim].

        Returns:
            Output tensor of shape [..., input_dim].
        """
        gate = torch.nn.functional.silu(self.w_gate(hidden))
        up = self.w_up(hidden)
        activated = gate * up
        output = self.w_down(activated)
        return self.dropout(output)

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"dropout={self.dropout_prob}, bias={self.use_bias}"
        )


def swiglu_feed_forward(
    d_model: int,
    *,
    mlp_ratio: int = 4,
    dropout: float = 0.0,
) -> nn.Module:
    """Build a SwiGLU feed-forward block with a GELU-compatible width.

    The hidden dimension is chosen so that the total parameter count is
    approximately equivalent to a GELU MLP with ``mlp_ratio=4``.

    Args:
        d_model: Model dimension.
        mlp_ratio: MLP expansion ratio (used for documentation; actual
            hidden dim uses the 8/3 SwiGLU convention).
        dropout: Dropout probability.

    Returns:
        A SwiGLU feed-forward module.
    """
    return SwiGLU(d_model, dropout=dropout)


__all__ = ["SwiGLU", "swiglu_feed_forward"]