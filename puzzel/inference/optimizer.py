"""CPU/CUDA optimization utilities for Puzzel inference.

Provides device selection, memory-efficient attention, and hardware-aware
inference configuration profiles. No training, no external services used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class InferenceProfile:
    """Hardware-aware inference configuration profile.

    Attributes:
        name: Human-readable profile name.
        device: Preferred device ("cpu", "cuda", "mps", or "auto").
        max_batch_size: Maximum concurrent requests.
        max_context_length: Maximum context length for generation.
        use_kv_cache: Whether to enable KV-cache.
        use_half_precision: Whether to use FP16 when available.
        threads: Number of CPU threads for inference (None = default).
    """

    name: str
    device: str = "cpu"
    max_batch_size: int = 1
    max_context_length: int = 512
    use_kv_cache: bool = True
    use_half_precision: bool = False
    threads: int | None = None


def select_device(preferred: str = "auto") -> torch.device:
    """Select the best available device for inference.

    Args:
        preferred: "auto", "cpu", "cuda", or "mps". If "auto", selects
            CUDA if available, then MPS, then CPU.

    Returns:
        A torch.device for the selected device.
    """
    if preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if preferred == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")
        return torch.device("mps")
    if preferred == "cpu":
        return torch.device("cpu")

    # auto: CUDA > MPS > CPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_cpu_threads(num_threads: int | None = None) -> int:
    """Set the number of CPU threads for PyTorch inference.

    Args:
        num_threads: Number of threads. If None, uses the default.

    Returns:
        The number of threads set (or the current value if None).
    """
    if num_threads is not None:
        torch.set_num_threads(num_threads)
    return torch.get_num_threads()


def get_memory_profile(device: torch.device | None = None) -> dict[str, Any]:
    """Get memory information for the selected device.

    Args:
        device: Target device. Defaults to CPU.

    Returns:
        Dictionary with device type, total memory, available memory,
        and recommended batch size.
    """
    if device is None:
        device = torch.device("cpu")

    info: dict[str, Any] = {"device": str(device), "device_type": device.type}

    if device.type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        info["total_memory_bytes"] = props.total_memory
        info["available_memory_bytes"] = torch.cuda.mem_get_info(device)[0]
        info["recommended_batch_size"] = max(1, info["available_memory_bytes"] // (1 << 30))
    elif device.type == "mps" and torch.backends.mps.is_available():
        # MPS doesn't expose detailed memory info; use conservative estimate.
        info["total_memory_bytes"] = None
        info["available_memory_bytes"] = None
        info["recommended_batch_size"] = 1
    else:
        # CPU: estimate from system memory.
        try:
            import os

            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        info["available_memory_bytes"] = int(line.split()[1]) * 1024
                        break
        except (OSError, ValueError):
            info["available_memory_bytes"] = None
        info["recommended_batch_size"] = 1

    return info


def memory_efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Compute attention in chunks to reduce peak memory usage.

    Processes the query sequence in chunks, computing attention for each
    chunk independently. This reduces memory from O(n^2) to O(n * chunk_size)
    for the attention matrix.

    Args:
        query: Query tensor of shape (batch, seq_len, dim).
        key: Key tensor of shape (batch, seq_len, dim).
        value: Value tensor of shape (batch, seq_len, dim).
        attn_mask: Optional attention mask.
        chunk_size: Number of query positions to process per chunk.

    Returns:
        Attention output of shape (batch, seq_len, dim).
    """
    import torch.nn.functional as F

    batch, seq_len, dim = query.shape
    scale = dim ** -0.5
    output = torch.empty_like(query)

    for i in range(0, seq_len, chunk_size):
        q_chunk = query[:, i : i + chunk_size]
        # Scores: (batch, chunk_size, seq_len)
        scores = torch.matmul(q_chunk, key.transpose(-2, -1)) * scale

        if attn_mask is not None:
            scores = scores + attn_mask[:, i : i + chunk_size]

        probs = F.softmax(scores, dim=-1)
        output[:, i : i + chunk_size] = torch.matmul(probs, value)

    return output


def get_default_profiles() -> dict[str, InferenceProfile]:
    """Return hardware-aware inference profiles.

    Returns:
        Dictionary mapping profile name to InferenceProfile.
    """
    return {
        "cpu_safe": InferenceProfile(
            name="cpu_safe",
            device="cpu",
            max_batch_size=1,
            max_context_length=256,
            use_kv_cache=True,
            use_half_precision=False,
            threads=1,
        ),
        "cpu_balanced": InferenceProfile(
            name="cpu_balanced",
            device="cpu",
            max_batch_size=2,
            max_context_length=512,
            use_kv_cache=True,
            use_half_precision=False,
            threads=2,
        ),
        "cuda_fast": InferenceProfile(
            name="cuda_fast",
            device="cuda",
            max_batch_size=8,
            max_context_length=1024,
            use_kv_cache=True,
            use_half_precision=True,
            threads=None,
        ),
    }