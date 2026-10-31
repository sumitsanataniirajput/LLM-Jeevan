"""Conservative local hardware inspection and training memory estimates."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from puzzel.config import load_config
from puzzel.model.config import ModelConfig

GIB = 1024**3


@dataclass(frozen=True, slots=True)
class HardwareInfo:
    """A snapshot of the resources visible to this process."""

    platform: str
    architecture: str
    cpu_model: str
    logical_cpu_count: int
    total_ram_bytes: int
    available_ram_bytes: int
    cgroup_memory_limit_bytes: int | None
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str | None
    cuda_total_bytes: int | None
    cuda_free_bytes: int | None
    mps_available: bool
    disk_total_bytes: int
    disk_free_bytes: int
    disk_path: str


def _read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith(("MemTotal:", "MemAvailable:")):
                key, value = line.split(":", 1)
                values[key] = int(value.split()[0]) * 1024
    except (OSError, ValueError):
        pass
    if "MemTotal" not in values:
        page_size = os.sysconf("SC_PAGE_SIZE")
        values["MemTotal"] = int(os.sysconf("SC_PHYS_PAGES") * page_size)
    values.setdefault("MemAvailable", values["MemTotal"])
    return values


def _read_cgroup_memory() -> tuple[int | None, int | None]:
    """Return a finite cgroup limit and current use, if this container has one."""
    candidates = (
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
        (
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ),
    )
    for limit_path, current_path in candidates:
        try:
            raw_limit = limit_path.read_text(encoding="ascii").strip()
            if raw_limit == "max":
                continue
            limit = int(raw_limit)
            current = int(current_path.read_text(encoding="ascii").strip())
            if limit <= 0 or limit >= (1 << 60):
                continue
            return limit, current
        except (OSError, ValueError):
            continue
    return None, None


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith(("model name", "hardware")):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _existing_path(path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    while not target.exists() and target != target.parent:
        target = target.parent
    return target


def detect_hardware(disk_path: str | Path = ".") -> HardwareInfo:
    """Detect visible CPU, RAM, CUDA/MPS, and filesystem capacity.

    Memory uses the lower of OS-available RAM and the remaining cgroup budget
    when running in a constrained container. GPU detection is conditional; a
    missing CUDA device is a normal supported configuration.
    """
    memory = _read_meminfo()
    total_ram = memory["MemTotal"]
    available_ram = memory["MemAvailable"]
    cgroup_limit, cgroup_current = _read_cgroup_memory()
    if cgroup_limit is not None and cgroup_current is not None:
        total_ram = min(total_ram, cgroup_limit)
        available_ram = min(available_ram, max(0, cgroup_limit - cgroup_current))

    try:
        cuda_device_count = int(torch.cuda.device_count())
    except (RuntimeError, AssertionError):
        cuda_device_count = 0
    cuda_available = bool(torch.cuda.is_available() and cuda_device_count > 0)
    cuda_name: str | None = None
    cuda_total: int | None = None
    cuda_free: int | None = None
    if cuda_available:
        try:
            index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            cuda_name = props.name
            cuda_total = int(props.total_memory)
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            cuda_free = int(free_bytes)
            cuda_total = int(total_bytes)
        except (RuntimeError, AssertionError):
            # Detection should remain useful even if a driver reports no memory.
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            cuda_name = props.name
            cuda_total = int(props.total_memory)

    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    disk_target = _existing_path(disk_path)
    disk = shutil.disk_usage(disk_target)
    return HardwareInfo(
        platform=platform.platform(),
        architecture=platform.machine(),
        cpu_model=_cpu_model(),
        logical_cpu_count=os.cpu_count() or 1,
        total_ram_bytes=total_ram,
        available_ram_bytes=available_ram,
        cgroup_memory_limit_bytes=cgroup_limit,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_device_name=cuda_name,
        cuda_total_bytes=cuda_total,
        cuda_free_bytes=cuda_free,
        mps_available=mps_available,
        disk_total_bytes=int(disk.total),
        disk_free_bytes=int(disk.free),
        disk_path=str(disk_target),
    )


def estimate_parameter_count(config: ModelConfig) -> int:
    """Count unique Transformer parameters from its dimensions without allocation."""
    width = config.d_model
    expanded = width * config.mlp_ratio
    embeddings = config.vocab_size * width + config.context_length * width
    blocks = config.num_layers * (
        4 * width * width
        + 2 * width * expanded
        + 9 * width
        + expanded
    )
    final_norm = 2 * width
    output_head = 0 if config.tie_embeddings else config.vocab_size * width
    return embeddings + blocks + final_norm + output_head


def estimate_memory_requirements(
    config: ModelConfig,
    *,
    batch_size: int = 1,
    context_length: int | None = None,
) -> dict[str, int | float]:
    """Estimate FP32 training memory, with an explicit conservative activation margin.

    Persistent memory assumes FP32 weights, gradients, and two AdamW moments
    (16 bytes per parameter). The activation estimate accounts for saved hidden,
    MLP, attention, and logit tensors and is multiplied by 1.5 as a rough safety
    factor. It is a planning estimate, not a CUDA allocator guarantee.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    sequence_length = config.context_length if context_length is None else context_length
    if not 0 < sequence_length <= config.context_length:
        raise ValueError("context_length must be between 1 and the model context length")
    parameters = estimate_parameter_count(config)
    persistent = parameters * 16
    hidden_elements = (
        config.num_layers * batch_size * sequence_length * config.d_model * 12
    )
    attention_elements = (
        config.num_layers
        * batch_size
        * config.num_heads
        * sequence_length
        * sequence_length
        * 2
    )
    logits_elements = batch_size * sequence_length * config.vocab_size
    activation_bytes = math.ceil(
        (hidden_elements + attention_elements + logits_elements) * 4 * 1.5
    )
    total = persistent + activation_bytes
    return {
        "parameter_count": parameters,
        "fp32_weights_bytes": parameters * 4,
        "gradient_bytes": parameters * 4,
        "adamw_state_bytes": parameters * 8,
        "persistent_training_bytes": persistent,
        "activation_estimate_bytes": activation_bytes,
        "estimated_training_bytes": total,
        "estimated_checkpoint_bytes": parameters * 12,
        "batch_size": batch_size,
        "context_length": sequence_length,
    }


def estimate_training_compute(
    config: ModelConfig,
    *,
    batch_size: int,
    max_steps: int,
) -> dict[str, int | str]:
    """Estimate Transformer training FLOPs from configured tokens and steps.

    The dense forward/backward term uses the conventional rough 6 FLOPs per
    parameter-token approximation. A separate attention-matmul term accounts
    for sequence-quadratic QK/AV work. This is a planning bound, not a measured
    throughput or exact operation count; it excludes framework and optimizer
    overhead and does not claim a runtime.
    """
    for name, value in (("batch_size", batch_size), ("max_steps", max_steps)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    parameter_count = estimate_parameter_count(config)
    tokens_per_step = batch_size * config.context_length
    dense_training_flops = 6 * parameter_count * tokens_per_step
    attention_training_flops = (
        12
        * config.num_layers
        * batch_size
        * config.context_length
        * config.context_length
        * config.d_model
    )
    per_step = dense_training_flops + attention_training_flops
    return {
        "method": "approx_6N_per_token_plus_attention_matmuls",
        "tokens_per_step": tokens_per_step,
        "estimated_flops_per_step": per_step,
        "configured_steps": max_steps,
        "estimated_total_flops": per_step * max_steps,
    }


def _available_training_budget(info: HardwareInfo, device: str) -> int:
    if device == "cuda" and info.cuda_available and info.cuda_free_bytes is not None:
        return max(0, int(info.cuda_free_bytes * 0.70))
    # Leave at least 1 GiB or 40% of free RAM for the OS, Python, data cache,
    # and checkpoint serialization. Never use all the memory reported as free.
    reserve = max(GIB, int(info.available_ram_bytes * 0.40))
    return max(0, info.available_ram_bytes - reserve)


def selected_device(requested: str, info: HardwareInfo) -> str:
    """Mirror training's auto selection without assuming an accelerator exists."""
    if requested == "auto":
        if info.cuda_available:
            return "cuda"
        if info.mps_available:
            return "mps"
        return "cpu"
    return requested


def recommend_safe_batch_size(
    info: HardwareInfo,
    config: ModelConfig,
    *,
    device: str = "auto",
) -> int:
    """Return a conservative power-of-two recommendation; zero means do not fit."""
    chosen_device = selected_device(device, info)
    budget = _available_training_budget(info, chosen_device)
    one_example = estimate_memory_requirements(config, batch_size=1)[
        "activation_estimate_bytes"
    ]
    persistent = estimate_parameter_count(config) * 16
    if budget <= persistent or one_example <= 0:
        return 0
    fit = int((budget - persistent) // int(one_example))
    cap = 64 if chosen_device == "cuda" else 8
    fit = min(fit, cap)
    if fit < 1:
        return 0
    return 1 << (fit.bit_length() - 1)


def build_hardware_report(
    config: ModelConfig,
    *,
    batch_size: int,
    device: str,
    max_steps: int = 1000,
    disk_path: str | Path = ".",
) -> dict[str, Any]:
    """Combine detected hardware with model and configured-batch estimates."""
    info = detect_hardware(disk_path)
    chosen_device = selected_device(device, info)
    recommendation = recommend_safe_batch_size(info, config, device=device)
    model_memory = estimate_memory_requirements(config, batch_size=batch_size)
    compute_estimate = estimate_training_compute(
        config, batch_size=batch_size, max_steps=max_steps
    )
    configured_training_budget = _available_training_budget(info, chosen_device)
    warnings: list[str] = []
    if device == "cuda" and not info.cuda_available:
        warnings.append("CUDA was requested but no CUDA device is available")
    if device == "mps" and not info.mps_available:
        warnings.append("MPS was requested but is not available")
    if recommendation == 0:
        warnings.append("estimated model state does not fit the conservative memory budget")
    elif batch_size > recommendation:
        warnings.append(
            f"configured batch size {batch_size} exceeds conservative recommendation "
            f"{recommendation}; consider reducing it (no automatic change was made)"
        )
    if model_memory["estimated_training_bytes"] > configured_training_budget:
        warnings.append("configured batch estimate exceeds the conservative memory budget")
    checkpoint_pair_bytes = int(model_memory["estimated_checkpoint_bytes"]) * 2
    if checkpoint_pair_bytes > info.disk_free_bytes:
        warnings.append(
            "estimated latest and best checkpoint space exceeds currently free disk"
        )
    return {
        "hardware": asdict(info),
        "selected_device": chosen_device,
        "model": {
            "parameter_count": estimate_parameter_count(config),
            "context_length": config.context_length,
            "vocabulary_size": config.vocab_size,
            "configured_batch_size": batch_size,
            "recommended_safe_batch_size": recommendation,
            "training_memory_estimate": model_memory,
            "compute_estimate": compute_estimate,
            "estimated_latest_and_best_checkpoint_bytes": checkpoint_pair_bytes,
            "available_training_memory_budget_bytes": configured_training_budget,
        },
        "warnings": warnings,
    }


def format_bytes(value: int | None) -> str:
    if value is None:
        return "not available"
    return f"{value / GIB:.2f} GiB"


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="puzzel-hardware",
        description="Inspect visible hardware and estimate Puzzel training resources.",
    )
    parser.add_argument("--config", type=Path, required=True, help="TOML model/training config")
    parser.add_argument(
        "--disk-path", type=Path, default=Path("."), help="filesystem to inspect"
    )
    args = parser.parse_args()
    run_config = load_config(args.config)
    report = build_hardware_report(
        run_config.model,
        batch_size=run_config.training.batch_size,
        device=run_config.training.device,
        max_steps=run_config.training.max_steps,
        disk_path=args.disk_path,
    )
    print(json.dumps(report, indent=2))
    if report["warnings"]:
        print("Warnings:", file=sys.stderr)
        for warning in report["warnings"]:
            print(f"- {warning}", file=sys.stderr)


if __name__ == "__main__":
    main()
