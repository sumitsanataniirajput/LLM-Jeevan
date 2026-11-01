"""Root Mean Square Layer Normalization (RMSNorm) for the Puzzel Transformer.

Implements RMSNorm as described in
"Root Mean Square Layer Normalization" (Zhang & Sennrich, 2019).
RMSNorm normalizes by the root mean square of activations and applies
a learned scale parameter. It is computationally simpler than LayerNorm
(as it omits recentering) and is numerically stable.

No pretrained weights are used; the scale parameter is initialized to 1.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes input by dividing by the root mean square of its values
    along the last dimension, then applies a learned gain parameter.

    Mathematically::
        output = x * gain / sqrt(mean(x^2) + eps)

    Args:
        normalized_dim: Dimension of the features to normalize.
        epsilon: Small constant for numerical stability. Default 1e-6.
    """

    def __init__(self, normalized_dim: int, *, epsilon: float = 1e-6) -> None:
        super().__init__()
        if normalized_dim <= 0:
            raise ValueError("normalized_dim must be a positive integer")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        self.normalized_dim = normalized_dim
        self.epsilon = float(epsilon)
        self.weight = nn.Parameter(torch.ones(normalized_dim))

    def forward(self, hidden: Tensor) -> Tensor:
        """Normalize the input tensor.

        Args:
            hidden: Input tensor of shape [..., normalized_dim].

        Returns:
            Normalized tensor with the same shape and dtype as input.
        """
        if hidden.size(-1) != self.normalized_dim:
            raise ValueError(
                f"last dimension must be {self.normalized_dim}, "
                f"got {hidden.size(-1)}"
            )
        input_dtype = hidden.dtype
        hidden_float = hidden.to(torch.float32)
        variance = hidden_float.pow(2).mean(dim=-1, keepdim=True)
        hidden_normalized = hidden_float * torch.rsqrt(variance + self.epsilon)
        output = hidden_normalized.to(input_dtype) * self.weight.to(input_dtype)
        return output

    def extra_repr(self) -> str:
        return f"normalized_dim={self.normalized_dim}, epsilon={self.epsilon}"


__all__ = ["RMSNorm"]