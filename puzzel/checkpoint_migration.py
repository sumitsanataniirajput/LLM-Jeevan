"""Checkpoint migration utilities for Puzzel checkpoints.

Provides safe migration from ``puzzel-checkpoint-v1`` to ``puzzel-checkpoint-v2``
format. Old checkpoints are NEVER modified or overwritten. Migration creates
a new checkpoint file with adapted architecture metadata.

Migration handles:
- Vocabulary size changes (embedding matrix resize)
- Architecture flag changes (RoPE, RMSNorm, SwiGLU, GQA)
- Context length changes (position embedding resize)
- Model dimension changes (projection matrix adaptation)

All migration operations require explicit user opt-in via the CLI script.
"""

from __future__ import annotations

import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch

from puzzel.checkpoint import LoadedCheckpoint, load_checkpoint, save_checkpoint
from puzzel.config import TrainingConfig
from puzzel.model.config import ModelConfig
from puzzel.model.transformer import TransformerLM
from puzzel.tokenizer import TextTokenizer, tokenizer_from_dict


class CheckpointMigrationError(RuntimeError):
    """Raised when checkpoint migration cannot be performed safely."""


def check_compatibility(
    source_path: str | Path,
    target_config: ModelConfig,
) -> dict[str, Any]:
    """Check whether a v1 checkpoint is compatible with a target config.

    Returns a report detailing what would change during migration. Does NOT
    modify any files.

    Args:
        source_path: Path to the existing v1 checkpoint.
        target_config: Target model configuration for migration.

    Returns:
        Compatibility report with fields:
            - compatible: Whether direct loading is possible
            - vocab_change: Whether vocabulary size differs
            - context_change: Whether context length differs
            - dim_change: Whether model dimension differs
            - architecture_change: Whether architecture flags differ
            - source_config: Source checkpoint model config
            - target_config: Target model config
            - warnings: List of human-readable warnings
    """
    source: LoadedCheckpoint = load_checkpoint(
        source_path, map_location="cpu"
    )
    source_config: ModelConfig = source.model_config

    warnings: list[str] = []
    vocab_change = source_config.vocab_size != target_config.vocab_size
    context_change = source_config.context_length != target_config.context_length
    dim_change = source_config.d_model != target_config.d_model
    architecture_change = (
        source_config.use_rope != target_config.use_rope
        or source_config.use_rmsnorm != target_config.use_rmsnorm
        or source_config.use_swiglu != target_config.use_swiglu
        or source_config.use_gqa != target_config.use_gqa
        or source_config.num_layers != target_config.num_layers
        or source_config.num_heads != target_config.num_heads
    )

    if vocab_change:
        warnings.append(
            f"vocabulary size changes from {source_config.vocab_size} "
            f"to {target_config.vocab_size}; embeddings will be resized"
        )
    if context_change:
        warnings.append(
            f"context length changes from {source_config.context_length} "
            f"to {target_config.context_length}; position embeddings will be resized"
        )
    if dim_change:
        warnings.append(
            f"model dimension changes from {source_config.d_model} "
            f"to {target_config.d_model}; all projections will be adapted"
        )
    if architecture_change:
        warnings.append(
            "architecture flags differ; migrated checkpoint will use target "
            "architecture, old weights may not transfer meaningfully"
        )

    compatible = (
        not vocab_change
        and not context_change
        and not dim_change
        and not architecture_change
    )

    return {
        "compatible": compatible,
        "vocab_change": vocab_change,
        "context_change": context_change,
        "dim_change": dim_change,
        "architecture_change": architecture_change,
        "source_config": asdict(source_config),
        "target_config": asdict(target_config),
        "warnings": warnings,
        "source_step": source.step,
        "source_format": "puzzel-checkpoint-v1",
    }


def migrate_checkpoint(
    source_path: str | Path,
    destination_path: str | Path,
    target_config: ModelConfig,
    target_training_config: TrainingConfig,
    tokenizer: TextTokenizer,
    *,
    resize_embeddings: bool = True,
    require_explicit: bool = True,
) -> Path:
    """Migrate a v1 checkpoint to v2 format with a new configuration.

    The source checkpoint is NEVER modified. A new checkpoint is written to
    ``destination_path`` with weights adapted for the target configuration.

    Args:
        source_path: Path to the existing v1 checkpoint.
        destination_path: Path for the migrated v2 checkpoint.
        target_config: Target model configuration.
        target_training_config: Target training configuration.
        tokenizer: Target tokenizer.
        resize_embeddings: If True, resize embedding matrices for vocab/dim
            changes using interpolation/padding. If False, migration fails on
            shape mismatches.
        require_explicit: If True, requires ``resize_embeddings=True`` for
            any shape-changing migration (safety guard).

    Returns:
        Path to the migrated checkpoint.

    Raises:
        CheckpointMigrationError: If migration is unsafe and
            ``resize_embeddings=False`` or ``require_explicit=True`` without
            ``resize_embeddings=True``.
    """
    report = check_compatibility(source_path, target_config)
    source: LoadedCheckpoint = load_checkpoint(
        source_path, map_location="cpu"
    )

    if not report["compatible"] and not resize_embeddings:
        if require_explicit:
            raise CheckpointMigrationError(
                "checkpoint migration requires shape changes but "
                "resize_embeddings=False; set resize_embeddings=True to "
                "allow embedding/projection matrix adaptation"
            )
        raise CheckpointMigrationError(
            "checkpoint shapes are incompatible and resize_embeddings is disabled"
        )

    source_model = source.model
    source_config = source.model_config

    # Build target model with random initialization.
    target_model = TransformerLM(target_config)

    # Copy compatible weights shape-by-shape.
    source_state = source_model.state_dict()
    target_state = target_model.state_dict()

    for name, target_param in target_state.items():
        if name not in source_state:
            # New parameter in target architecture (e.g., RoPE has no params,
            # but new norms or projections may exist). Keep random init.
            continue
        source_param = source_state[name]
        if source_param.shape == target_param.shape:
            # Exact shape match: copy directly.
            target_param.data.copy_(source_param)
        elif resize_embeddings:
            # Shape mismatch: attempt adaptive copy.
            _adaptive_copy(target_param, source_param, name)
        else:
            raise CheckpointMigrationError(
                f"parameter {name} shape mismatch: source {tuple(source_param.shape)} "
                f"vs target {tuple(target_param.shape)}; "
                "set resize_embeddings=True to allow adaptation"
            )

    # Verify all parameters are finite.
    for name, parameter in target_model.named_parameters():
        if not torch.isfinite(parameter.detach()).all().item():
            raise CheckpointMigrationError(
                f"migrated parameter {name} contains non-finite values"
            )

    # Save as v2 checkpoint.
    destination = save_checkpoint(
        destination_path,
        model=target_model,
        optimizer=source.optimizer_state_dict,  # placeholder; optimizer state not migrated
        model_config=target_config,
        training_config=target_training_config,
        tokenizer=tokenizer,
        step=0,  # migration resets step; training must restart
        scheduler_state_dict=None,
        torch_rng_state=None,
        cuda_rng_state=None,
        data_loader_rng_state=None,
        best_validation_loss=None,
        early_stopping_bad_evaluations=0,
    )
    return destination


def _adaptive_copy(
    target_param: torch.Tensor,
    source_param: torch.Tensor,
    name: str,
) -> None:
    """Copy source_param into target_param with adaptive resizing.

    Handles:
    - Embedding matrices: pad/truncate vocab dimension
    - Position embeddings: pad/truncate context dimension
    - Projection matrices: pad/truncate input/output dimensions
    """
    source_data = source_param.detach()

    if target_param.dim() == 1:
        # 1D parameter (bias, norm weight): pad or truncate.
        if source_data.numel() >= target_param.numel():
            target_param.data.copy_(source_data[: target_param.numel()])
        else:
            target_param.data.zero_()
            target_param.data[: source_data.numel()].copy_(source_data)
        return

    if target_param.dim() == 2:
        # 2D parameter (weight matrix): resize both dimensions.
        src_rows, src_cols = source_data.shape
        tgt_rows, tgt_cols = target_param.shape

        # Initialize target with target's random values.
        # Copy the overlapping region from source.
        copy_rows = min(src_rows, tgt_rows)
        copy_cols = min(src_cols, tgt_cols)
        target_param.data[:copy_rows, :copy_cols].copy_(
            source_data[:copy_rows, :copy_cols]
        )
        return

    # Higher-dimensional: copy elementwise where shapes match.
    min_shape = tuple(
        min(s, t) for s, t in zip(source_data.shape, target_param.shape)
    )
    slices = tuple(slice(0, m) for m in min_shape)
    target_param.data[slices].copy_(source_data[slices])


__all__ = [
    "CheckpointMigrationError",
    "check_compatibility",
    "migrate_checkpoint",
]