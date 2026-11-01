# Puzzel Testing Documentation

Puzzel includes 20 test categories covering all major components.

## Test Categories

| Category | Test File | Description |
|----------|-----------|-------------|
| Model architecture | tests/test_model.py | TransformerLM forward pass, RoPE, GQA |
| RoPE | tests/test_rope.py | Rotary position embeddings |
| RMSNorm | tests/test_rmsnorm.py | Root mean square normalization |
| SwiGLU | tests/test_swiglu.py | Gated activation function |
| GQA | tests/test_gqa.py | Grouped query attention |
| KV-cache | tests/test_kv_cache.py | KV-cache manager |
| Tokenizer | tests/test_tokenizer.py | BPE tokenizer round-trip |
| Config | tests/test_config.py | Model configuration |
| Data quality | tests/test_data_quality.py | Data validation |
| Dataset | tests/test_dataset.py | Dataset handling |
| Dataset pipeline | tests/test_dataset_pipeline.py | End-to-end pipeline |
| Evaluation | tests/test_evaluation.py | Evaluation metrics |
| Hardware | tests/test_hardware.py | Hardware detection |
| Inference | tests/test_inference.py | Generation engine |
| Multilingual pipeline | tests/test_multilingual_pipeline.py | Multilingual support |
| SFT spec | tests/test_sft_spec.py | SFT configuration |
| Training | tests/test_training.py | Training loop |
| Training CLI | tests/test_training_cli.py | CLI training |
| Phase 9 regressions | tests/test_phase9_regressions.py | Phase 9 regression tests |

## Running Tests

```bash
# Run all tests (excluding phase 9 regression tests that need a checkpoint)
python -m pytest tests/ --ignore=tests/test_phase9_regressions.py

# Run a specific test file
python -m pytest tests/test_tokenizer.py -v

# Run with coverage
python -m pytest tests/ --cov=puzzel
```

## Expected Results

- 104 tests pass
- 1 test fails (test_from_checkpoint_is_readonly — environmental: no checkpoint file)
- 29 tests skipped (checkpoint-dependent inference tests)

## Writing Tests

Each new feature requires a test or documented verification:

```python
import pytest
from puzzel.tokenizer import BPETokenizer

def test_my_feature():
    tok = BPETokenizer.train(["hello world"], target_vocab_size=100, min_frequency=1)
    encoded = tok.encode("hello world", add_bos=True, add_eos=True)
    decoded = tok.decode(encoded)
    assert decoded == "hello world"
```

## Test Markers

Tests that require a checkpoint file are skipped automatically when no
checkpoint is present at `checkpoints/multilingual-first-run.best.pt`.