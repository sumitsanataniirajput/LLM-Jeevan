"""Phase 8 Task 1: Verify the trained checkpoint exists and loads successfully."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from puzzel.checkpoint import load_checkpoint  # noqa: E402
from puzzel.training.loop import resolve_device  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    checkpoint_path = Path("checkpoints/multilingual-first-run.best.pt")
    if not checkpoint_path.exists():
        print(f"FAIL: checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"Checkpoint found: {checkpoint_path}")
    print(f"  Size: {checkpoint_path.stat().st_size:,} bytes")
    print(f"  SHA-256: {sha256_file(checkpoint_path)}")

    device = resolve_device("cpu")
    loaded = load_checkpoint(checkpoint_path, map_location=device)

    print(f"\nCheckpoint loaded successfully on {device}")
    print(f"  Format: puzzel-checkpoint-v1")
    print(f"  Step: {loaded.step}")
    print(f"  Best validation loss: {loaded.best_validation_loss}")
    print(f"  Model config:")
    print(f"    vocab_size: {loaded.model_config.vocab_size}")
    print(f"    context_length: {loaded.model_config.context_length}")
    print(f"    d_model: {loaded.model_config.d_model}")
    print(f"    num_layers: {loaded.model_config.num_layers}")
    print(f"    num_heads: {loaded.model_config.num_heads}")
    print(f"  Tokenizer vocab: {loaded.tokenizer.vocab_size}")
    print(f"  Optimizer state present: {loaded.optimizer_state_dict is not None}")
    print(f"  Scheduler state present: {loaded.scheduler_state_dict is not None}")

    # Verify model is finite
    for name, param in loaded.model.named_parameters():
        if not param.isfinite().all().item():
            print(f"FAIL: non-finite parameter {name}")
            sys.exit(1)
    print(f"\nAll model parameters finite: OK")
    print(f"\nTASK 1 COMPLETE: Checkpoint verified and loads successfully")


if __name__ == "__main__":
    main()
