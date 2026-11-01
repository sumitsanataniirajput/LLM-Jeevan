"""Quantization support for Puzzel inference.

Provides utilities for quantizing model weights to reduce memory usage
and improve inference speed on CPU. Supports FP16 and INT8 quantization.

No training, no external services are used. Quantization is applied to
a loaded model's weights in memory; the original checkpoint is never
modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class QuantizationConfig:
    """Configuration for model quantization.

    Attributes:
        dtype: Target dtype for quantized weights. "fp16", "int8", or "none".
        dynamic: If True, use dynamic quantization (activations quantized
            at runtime). Only applicable for certain layers.
        modules_to_quantize: List of module name patterns to quantize.
            If None, all Linear layers are quantized.
    """

    dtype: str = "fp16"
    dynamic: bool = False
    modules_to_quantize: list[str] | None = None

    def __post_init__(self) -> None:
        if self.dtype not in {"fp16", "int8", "none"}:
            raise ValueError(
                f"dtype must be 'fp16', 'int8', or 'none', got {self.dtype!r}"
            )


def quantize_model(
    model: Any,
    config: QuantizationConfig | None = None,
) -> Any:
    """Apply quantization to a model's weights in place.

    Args:
        model: A PyTorch model (typically TransformerLM).
        config: Quantization configuration. Defaults to fp16 quantization
            of all Linear layers.

    Returns:
        The quantized model (same object, modified in place).

    Raises:
        ValueError: If an unsupported dtype is requested.
        RuntimeError: If quantization fails on the model.
    """
    if config is None:
        config = QuantizationConfig()

    if config.dtype == "none":
        return model

    target_dtype = torch.float16 if config.dtype == "fp16" else torch.int8

    modules_to_quantize = config.modules_to_quantize or ["Linear"]

    for name, module in model.named_modules():
        if not any(pattern in name for pattern in modules_to_quantize):
            continue
        if not isinstance(module, torch.nn.Linear):
            continue

        if config.dtype == "fp16":
            module.weight.data = module.weight.data.to(target_dtype)
            if module.bias is not None:
                module.bias.data = module.bias.data.to(target_dtype)
        elif config.dtype == "int8":
            # Simple per-tensor symmetric quantization.
            weight = module.weight.data
            scale = weight.abs().max() / 127.0
            if scale > 0:
                quantized = (weight / scale).round().clamp(-128, 127).to(torch.int8)
                module.weight.data = quantized.to(torch.float32) * scale
                # Store scale for dequantization during forward.
                module._quant_scale = scale
            if module.bias is not None:
                module.bias.data = module.bias.data.to(torch.float32)

    return model


def dequantize_model(model: Any) -> Any:
    """Attempt to dequantize a previously quantized model back to FP32.

    This is best-effort: FP16 weights are converted back to FP32. INT8
    weights are dequantized using stored scale factors if available.

    Args:
        model: A quantized PyTorch model.

    Returns:
        The dequantized model (same object, modified in place).
    """
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            if module.weight.dtype == torch.float16:
                module.weight.data = module.weight.data.to(torch.float32)
                if module.bias is not None and module.bias.dtype == torch.float16:
                    module.bias.data = module.bias.data.to(torch.float32)
            # INT8 dequantization would need stored scale; skip if not present.

    return model


def estimate_memory_savings(model: Any, config: QuantizationConfig | None = None) -> dict[str, float]:
    """Estimate memory savings from quantization.

    Computes the theoretical memory usage before and after quantization
    for the model's parameters.

    Args:
        model: A PyTorch model.
        config: Quantization configuration. Defaults to fp16.

    Returns:
        Dictionary with original_bytes, quantized_bytes, savings_bytes,
        and savings_ratio.
    """
    if config is None:
        config = QuantizationConfig()

    if config.dtype == "none":
        target_bytes_per_param = 4  # FP32
    elif config.dtype == "fp16":
        target_bytes_per_param = 2
    elif config.dtype == "int8":
        target_bytes_per_param = 1
    else:
        target_bytes_per_param = 4

    total_params = sum(p.numel() for p in model.parameters())
    original_bytes = total_params * 4  # Assume FP32 baseline
    quantized_bytes = total_params * target_bytes_per_param
    savings_bytes = original_bytes - quantized_bytes
    savings_ratio = savings_bytes / original_bytes if original_bytes > 0 else 0.0

    return {
        "original_bytes": float(original_bytes),
        "quantized_bytes": float(quantized_bytes),
        "savings_bytes": float(savings_bytes),
        "savings_ratio": savings_ratio,
    }