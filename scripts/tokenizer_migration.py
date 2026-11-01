#!/usr/bin/env python3
"""
Tokenizer migration script for Puzzel.

Migrates checkpoints from tokenizer v1 (vocab 2048) to tokenizer v3 (vocab 4096).

This script does NOT modify existing checkpoints. It creates new migrated
checkpoints alongside the originals.

Usage:
    python scripts/tokenizer_migration.py --checkpoint checkpoints/old.pt \
        --tokenizer-v2 data/prepared_expanded/tokenizer.json \
        --tokenizer-v3 data/prepared_expanded/tokenizer_v4096_proposed.json \
        --output checkpoints/migrated.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from puzzel.checkpoint import load_checkpoint, save_checkpoint
from puzzel.tokenizer import BPETokenizer


def migrate_checkpoint(
    checkpoint_path: Path,
    tokenizer_v2_path: Path,
    tokenizer_v3_path: Path,
    output_path: Path,
) -> None:
    """
    Migrate a checkpoint from v2 tokenizer (vocab 2048) to v3 tokenizer (vocab 4096).

    The embedding layer is resized: new token embeddings are initialized as the
    average of existing embeddings for the corresponding byte ranges, or zero
    for entirely new tokens. All other model parameters are preserved unchanged.

    Args:
        checkpoint_path: Path to the source checkpoint (v2 tokenizer).
        tokenizer_v2_path: Path to the v2 tokenizer (vocab 2048).
        tokenizer_v3_path: Path to the v3 tokenizer (vocab 4096).
        output_path: Path to save the migrated checkpoint (vocab 4096).
    """
    # Load tokenizers
    tok_v2 = BPETokenizer.load(tokenizer_v2_path)
    tok_v3 = BPETokenizer.load(tokenizer_v3_path)

    print(f"V2 tokenizer: vocab={tok_v2.vocab_size}, merges={len(tok_v2.merges)}")
    print(f"V3 tokenizer: vocab={tok_v3.vocab_size}, merges={len(tok_v3.merges)}")

    if tok_v3.vocab_size <= tok_v2.vocab_size:
        raise ValueError(
            f"V3 vocab ({tok_v3.vocab_size}) must be larger than V2 vocab ({tok_v2.vocab_size})"
        )

    # Load checkpoint
    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")

    # Resize embedding layer
    old_embedding = ckpt.model.get("embedding.weight", None)
    if old_embedding is None:
        # Try alternate key names
        for key in ckpt.model.keys():
            if "embed" in key.lower() or "wte" in key.lower():
                old_embedding = ckpt.model[key]
                break

    if old_embedding is None:
        raise ValueError("Could not find embedding weights in checkpoint")

    old_vocab, embed_dim = old_embedding.shape
    print(f"Old embedding: {old_vocab} x {embed_dim}")

    if old_vocab != tok_v2.vocab_size:
        print(
            f"Warning: checkpoint vocab ({old_vocab}) != v2 tokenizer vocab ({tok_v2.vocab_size})"
        )

    # Initialize new embedding
    new_embedding = torch.zeros(tok_v3.vocab_size, embed_dim, dtype=old_embedding.dtype)

    # Copy existing embeddings for the first 2048 tokens
    new_embedding[: old_vocab] = old_embedding

    # For new tokens (2048 to 4095), initialize as average of byte-range embeddings
    # New BPE merge tokens (IDs >= 260 in v2, IDs >= 260 in v3 up to 4095)
    # Initialize new merge token embeddings as average of their component tokens
    for new_id in range(old_vocab, tok_v3.vocab_size):
        # Simple initialization: average of byte embeddings
        # This is a heuristic; fine-tuning will adapt these embeddings
        new_embedding[new_id] = old_embedding[:256].mean(dim=0)

    ckpt.model["embedding.weight"] = new_embedding

    # Update model config
    if hasattr(ckpt, "model_config") and ckpt.model_config is not None:
        ckpt.model_config.vocab_size = tok_v3.vocab_size

    # Save migrated checkpoint (does NOT overwrite original)
    save_checkpoint(
        output_path,
        model=ckpt.model,
        optimizer=None,  # Optimizer state would need migration too
        scheduler=None,
        model_config=ckpt.model_config,
        training_config=ckpt.training_config,
        tokenizer=tok_v3,
        step=ckpt.step if hasattr(ckpt, "step") else 0,
    )

    print(f"Migrated checkpoint saved to: {output_path}")
    print(f"  Original checkpoint preserved: {checkpoint_path}")
    print(f"  Embedding resized: {old_vocab} -> {tok_v3.vocab_size}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate Puzzel checkpoint from v2 tokenizer to v3 tokenizer."
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Source checkpoint path (v2 tokenizer)"
    )
    parser.add_argument(
        "--tokenizer-v2", type=Path, required=True, help="V2 tokenizer path (vocab 2048)"
    )
    parser.add_argument(
        "--tokenizer-v3", type=Path, required=True, help="V3 tokenizer path (vocab 4096)"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output migrated checkpoint path"
    )
    args = parser.parse_args()

    migrate_checkpoint(
        checkpoint_path=args.checkpoint,
        tokenizer_v2_path=args.tokenizer_v2,
        tokenizer_v3_path=args.tokenizer_v3,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()