# Phase 7 — First Multilingual Training Preflight & Run Plan

**Status:** Preflight complete — all checks passed. Training NOT started.  
**Blockers:** None.

---

## 1. Exact training command (approved by user before execution)

```bash
./scripts/train.sh --config configs/multilingual_tiny.toml --dataset data/prepared --output checkpoints/multilingual-first-run.pt
```

This maps to `python -m puzzel --config ... --dataset ... --output ...`. The `train.sh` wrapper execs `./.venv/bin/python -m puzzel "$@"`, passing all flags through. `--metrics` defaults to `<output>.metrics.csv`.

**Not run yet.** Awaiting explicit user approval before executing.

---

## 2. Configuration validated (`configs/multilingual_tiny.toml`)

| Section | Key | Value |
|---|---|---|
| tokenizer | kind | bpe |
| tokenizer | vocab_size | 2048 |
| tokenizer | min_frequency | 2 |
| model | vocab_size | 2048 |
| model | context_length | 128 |
| model | d_model | 64 |
| model | num_layers | 2 |
| model | num_heads | 4 |
| model | mlp_ratio | 4 |
| model | dropout | 0.1 |
| model | pad_token_id | 258 |
| model | tie_embeddings | true |
| training | batch_size | 2 |
| training | learning_rate | 0.0003 |
| training | weight_decay | 0.1 |
| training | max_steps | 1000 |
| training | gradient_clip_norm | 1.0 |
| training | seed | 1337 |
| training | num_workers | 0 |
| training | device | cpu |
| training | precision | fp32 |
| training | log_every_steps | 10 |
| training | save_every_steps | 100 |
| training | scheduler | cosine |
| training | warmup_steps | 20 |
| training | min_learning_rate | 0.00003 |
| training | validation_every_steps | 100 |
| training | validation_batches | 32 |
| training | early_stopping_patience | 0 (disabled) |
| training | unstable_loss_threshold | 100.0 |
| training | loss_spike_multiplier | 10.0 |
| training.language_sampling_weights | en | 0.20 |
| training.language_sampling_weights | hi | 0.20 |
| training.language_sampling_weights | es | 0.15 |
| training.language_sampling_weights | fr | 0.15 |
| training.language_sampling_weights | ar | 0.15 |
| training.language_sampling_weights | zh | 0.15 |
| dataset | validation_fraction | 0.1 |
| dataset | test_fraction | 0.1 |
| dataset | split_seed | 1337 |
| dataset | max_document_characters | 1000000 |
| dataset | jsonl_text_field | text |
| dataset | require_document_metadata | true |

Config <-> prepared-dataset manifest/tokenizer: **ALL MATCH** (vocab 2048, context 128, pad 258, sampling weights, require_document_metadata=true).

---

## 3. Dataset verified (`data/prepared`)

Working tree clean; `data/prepared` byte-identical to committed state (deterministic regeneration confirmed).

- **49 documents retained** (110 discovered; 61 English exact-duplicate chunks removed globally; 0 metadata errors; 0 empty)
- **174,123 total tokens**; BPE vocab 2048 (1,788 learned merges + 256 byte fallback + 4 special)
- **Splits:** train 33 docs / 820 examples; validation 8 docs / 336 examples; test 8 docs / 231 examples; **total 1,387 examples**
- **All 6 languages present in every split:** ar, en, es, fr, hi, zh (contiguous, non-overlapping language ranges covering each split)
- Language sampling weights sum to 1.0; match config exactly
- `verify_prepared_dataset`: status `ok`; tokenizer byte-fallback/unicode/BOS-EOS round-trips `ok`; no cross-split leakage
- Example shape/validity smoke test: `input_ids (128,)`, `targets (128,)`, token range within [0, 2047], not all-pad

Per-language examples (train / validation / test):
- ar: 51 / 34 / 35 (120 total)
- en: 468 / 102 / 92 (662 total)
- es: 54 / 28 / 25 (107 total)
- fr: 57 / 31 / 30 (118 total)
- hi: 117 / 94 / 18 (229 total)
- zh: 73 / 47 / 31 (151 total)

---

## 4. Model & tokenizer validated

- Model instantiates: **239,360 parameters** (matches expected)
- `verify_random_initialization()`: PASSED
- Forward-pass smoke test: input `(2, 128)` → logits `(2, 128, 2048)`; cross-entropy finite (~7.63)
- Tokenizer: `puzzel-byte-bpe-tokenizer-v1`, vocab 2048, special tokens `<|bos|>=256`, `<|eos|>=257`, `<|pad|>=258`, `<|unk|>=259`
- `LanguageBalancedSampler` constructs successfully with manifest ranges + config weights (sums to 1.0); yields 820 in-range indices

---

## 5. Hardware / resources (no warnings)

- **CPU:** 2 × Intel Xeon @ 2.60GHz (x86_64)
- **RAM:** 4.13 GB total, ~3.44–3.47 GB available
- **Disk:** ~11.93 GB free at the repository root
- **GPU:** CUDA unavailable → CPU-only (confirmed `torch.cuda.is_available() == False`)
- **Estimated training memory:** ~12.48 MB (weights 0.96 MB + gradients 0.96 MB + AdamW 1.91 MB + activations 8.65 MB); checkpoint ~2.87 MB; latest+best ~5.74 MB
- **Available training budget:** ~2.07 GB → fits comfortably
- **Compute estimate:** 256 tokens/step, ~418 MFLOPs/step, 1000 steps, ~418 GFLOPs total
- **Benchmarked duration:** ~13 ms/step → **≈13 s for 1000 steps** (CPU-only; actual run adds validation every 100 steps + checkpoint writes every 100 steps, still expected well under a minute)
- `warnings`: `[]`

---

## 6. Checkpoint saving, recovery & metrics (code paths verified)

**Checkpoint format:** `puzzel-checkpoint-v1` (PyTorch zip archive, `weights_only=True` load).
- `save_checkpoint`: finite-parameter/optimizer check → temp file → `torch.save` → `fsync` → zip CRC validation → meta-device safe-load validation (format, step, model_config, training_config, tokenizer, state-dict shapes/dtypes, optimizer dict) → atomic `os.replace` → directory fsync. Existing checkpoint left untouched on any failure.
- `load_checkpoint`: restores model, tokenizer, model_config, training_config, step, optimizer/scheduler/RNG states; validates finiteness, vocab match, strict state_dict load.
- `_best_checkpoint_path(output)` → `<output.stem>.best<suffix>` (saved when validation loss improves).
- **Cadence:** checkpoint saved at step 1, every `save_every_steps=100` steps (100…1000), and at final step. Best checkpoint saved when validation improves (evaluated every `validation_every_steps=100`, plus step 1 and final). Early stopping disabled (`patience=0`). Metrics flushed every step.

**Metrics CSV fields:** `step, train_loss, validation_loss, learning_rate, gradient_norm, elapsed_seconds, estimated_remaining_seconds` (one row per step, flushed each step).

**Resume:** `--resume` requires `--output` to equal `--resume`; truncates metrics after the resumed step; continues from `step + 1`.

---

## 7. Expected output files (after training runs)

All under `checkpoints/`:

| File | Purpose |
|---|---|
| `checkpoints/multilingual-first-run.pt` | Latest checkpoint (steps 1, 100…1000, final) |
| `checkpoints/multilingual-first-run.best.pt` | Best-validation checkpoint (if validation improves) |
| `checkpoints/multilingual-first-run.pt.metrics.csv` | Per-step training metrics |
| `checkpoints/multilingual-first-run.tokenizer.json` | Saved tokenizer copy after training |

`checkpoints/` does not exist yet; `train_model` creates `destination.parent` automatically. The repo's `scripts/checkpoints` is not tracked — only `.gitkeep`-style scaffolding expected.

---

## 8. Evaluation procedure

- **During training:** validation loss computed every 100 steps (plus step 1 and final step) over `validation_batches=32` batches of the document-level validation split (8 docs / 336 examples), with mean next-token cross-entropy.
- **Best checkpoint:** the checkpoint with the lowest validation loss is saved to `*.best.pt`.
- **After training:** held-out test split (8 docs / 231 examples) is available at `data/prepared/test.bin` for a final evaluation via `scripts/evaluate.sh` (separate from training).
- **Divergence guards:** non-finite loss, loss > `unstable_loss_threshold` (100.0), or 3 consecutive loss spikes (×10.0 or +5.0 over EMA) raise `TrainingDivergedError`; the last good checkpoint is preserved.
- **Language balance:** `LanguageBalancedSampler` draws training indices per configured weights (en 0.20 / hi 0.20 / es 0.15 / fr 0.15 / ar 0.15 / zh 0.15) with replacement, so small languages are not drowned out.

---

## 9. Preflight results summary

All checks PASSED (ran via `preflight_check.py`, no training, no weight modification, no downloads):

1. Config loads; matches prepared manifest/tokenizer ✅
2. All 6 languages across train/validation/test ✅ (820/336/231 examples)
3. Config vs model/tokenizer validated ✅ (239,360 params; random init OK)
4. Forward-pass smoke test ✅ (logits `(2,128,2048)`, finite loss)
5. Dataset loading + example validity ✅ (in-range, shaped, not all-pad)
6. `LanguageBalancedSampler` construction ✅ (weights sum to 1.0)
7. Hardware report ✅ (CPU-only, ~12.48 MB train, ~2.07 GB budget, no warnings)
8. Checkpoint save/load module consistency ✅ (signatures intact; atomic/CRC-validated save path reviewed)

**Ready to execute.** Awaiting explicit approval to run the training command in section 1.
