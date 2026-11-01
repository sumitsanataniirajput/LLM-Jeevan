"""Model health monitoring and system diagnostics for Puzzel.

Performs lightweight health checks on the loaded model, tokenizer,
and system resources. No external services are used.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelHealth:
    """Health status of the model and system.

    Attributes:
        status: "healthy", "degraded", or "unhealthy".
        model_loaded: Whether a model is currently loaded.
        device: Device the model runs on (cpu, cuda, mps).
        memory_available_bytes: Available system memory in bytes.
        disk_free_bytes: Free disk space in bytes.
        checks: Individual check results.
        timestamp: ISO timestamp of the check.
    """

    status: str
    model_loaded: bool
    device: str
    memory_available_bytes: int
    disk_free_bytes: int
    checks: dict[str, bool]
    timestamp: str


def health_check(model_loaded: bool = False, device: str = "cpu") -> ModelHealth:
    """Perform a health check on the system and model.

    Args:
        model_loaded: Whether a model is currently loaded.
        device: Device the model is running on.

    Returns:
        ModelHealth with the current health status.
    """
    import time
    from datetime import datetime, timezone

    checks: dict[str, bool] = {}

    # Check memory.
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    mem_available = int(line.split()[1]) * 1024
                    checks["memory_available"] = mem_available > 256 * 1024 * 1024  # >256MB
                    break
            else:
                mem_available = 0
                checks["memory_available"] = False
    except OSError:
        mem_available = 0
        checks["memory_available"] = False

    # Check disk.
    try:
        disk = shutil.disk_usage(".")
        disk_free = disk.free
        checks["disk_space"] = disk_free > 1024 * 1024 * 1024  # >1GB
    except OSError:
        disk_free = 0
        checks["disk_space"] = False

    # Check model.
    checks["model_loaded"] = model_loaded

    # Check Python/torch.
    try:
        import torch
        checks["torch_available"] = True
    except ImportError:
        checks["torch_available"] = False

    # Determine overall status.
    failed = [k for k, v in checks.items() if not v]
    if not failed:
        status = "healthy"
    elif len(failed) <= 2 and "model_loaded" not in failed:
        status = "degraded"
    else:
        status = "unhealthy"

    return ModelHealth(
        status=status,
        model_loaded=model_loaded,
        device=device,
        memory_available_bytes=mem_available,
        disk_free_bytes=disk_free,
        checks=checks,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )