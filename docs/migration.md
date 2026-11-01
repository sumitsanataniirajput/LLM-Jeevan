# Puzzel Migration Documentation

This document describes migration utilities for upgrading from Puzzel v1
to v2, including checkpoint migration and tokenizer migration.

## Checkpoint Migration

### v1 → v2 Checkpoint Format

Puzzel v2 uses a new checkpoint format with:

- Strict validation of tensor shapes
- Atomic save (write to temp, then rename)
- Format versioning
- Architecture flags (RoPE, RMSNorm, SwiGLU, GQA, KV-cache)

### Migration Utilities

```python
from puzzel.checkpoint_migration import check_compatibility, migrate_checkpoint

# Check if a v1 checkpoint is compatible with v2
compatible = check_compatibility("checkpoints/v1-model.pt")

# Migrate a v1 checkpoint to v2 format
migrate_checkpoint(
    source="checkpoints/v1-model.pt",
    destination="checkpoints/v2-model.pt",
    strict=True,
)
```

The source checkpoint is **never modified** during migration.

### Legacy Model Loading

For v1 checkpoints that cannot be migrated, use the legacy adapter:

```python
from puzzel.model import LegacyTransformerLM

model = LegacyTransformerLM.from_v1_checkpoint("checkpoints/v1-model.pt")
```

## Tokenizer Migration

### v1 → v2 Tokenizer Format

Puzzel v2 tokenizers support:

- Extended special tokens (MASK, SEP, language IDs)
- Backward-compatible SPECIAL_TOKENS/VOCAB_SIZE aliases
- BPETokenizer with extra_special_tokens parameter

V1 tokenizers are fully backward compatible:

```python
from puzzel.tokenizer import BPETokenizer

# Load a v1 tokenizer (format: puzzel-byte-bpe-tokenizer-v1)
tok = BPETokenizer.load("data/prepared/tokenizer.json")
print(tok.vocab_size)  # 2048
```

### Training with Language IDs

```python
from puzzel.tokenizer import BPETokenizer, LANGUAGE_ID_TOKENS

tok = BPETokenizer.train(
    texts,
    target_vocab_size=2048,
    min_frequency=2,
    extra_special_tokens=dict(LANGUAGE_ID_TOKENS),
)
```

## Backward Compatibility

- Existing v1 checkpoints can still be loaded via `LegacyTransformerLM`
- Existing v1 tokenizer files load without modification
- Existing API endpoints (`/health`, `/model/info`, `/generate`) remain functional
- Existing training CLI (`python -m puzzel`) remains functional
- Legacy inference mode maintained for old checkpoints