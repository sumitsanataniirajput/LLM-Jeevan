# Local from-scratch training guide

Puzzel's trainer updates the existing `TransformerLM` using its custom tokenizer and next-token examples. Fresh runs construct the model from the repository's explicit random initialization. The only resumable weights accepted by the training CLI are Puzzel-format checkpoints produced by this project; there is no external model download or pretrained-weight path.

Training text must be owned by you, in the public domain, or licensed for the intended use. The repository contains no real corpus. Do not use the temporary synthetic test fixtures as training data. A falling loss on tests or a small pilot is not evidence that the model understands language; evaluate any trained model on held-out, properly licensed real text and report those results before making capability claims.

## 1. Inspect hardware before choosing a model

Run this from the repository root before any training:

```bash
./scripts/hardware.sh --config configs/tiny.toml
```

The report detects logical CPU count, OS/cgroup-visible RAM, CUDA device and free memory when present, MPS status, and free disk on the selected output filesystem. It estimates FP32 weights, gradients, AdamW moments, activations (with a safety margin), checkpoint size, an approximate FLOP count, and a conservative power-of-two batch recommendation. It never assumes a GPU and never changes a model or batch size. FLOPs are a formula-based planning estimate, not measured throughput or runtime. Treat the estimates as planning guidance, not an allocator guarantee; leave additional headroom for data, other processes, and runtime overhead.

Check each preset explicitly before using it:

```bash
./scripts/hardware.sh --config configs/tiny.toml
./scripts/hardware.sh --config configs/small.toml
./scripts/hardware.sh --config configs/medium.toml
```

`tiny` is intended for development and short pilots. `small` and `medium` are experimental presets, not automatic scale-up steps. The training CLI prints a fresh estimate before model construction and warns if the configured batch exceeds its conservative recommendation; it does not silently alter settings. CPU fit does not imply acceptable training speed.

## 2. Prepare genuine licensed text

Put UTF-8 `.txt`, `.jsonl`, or `.ndjson` files in `data/raw/`. TXT files are one document each; JSONL uses one record per line and a string `text` field. Follow [`data/raw/README.md`](../data/raw/README.md) for the exact multilingual JSONL schema, TXT provenance sidecars, and licensing guidance. Prepare a new output directory:

```bash
./scripts/data.sh prepare \
  --input data/raw \
  --output data/processed/tiny-v1 \
  --config configs/tiny.toml
./scripts/data.sh stats --dataset data/processed/tiny-v1
./scripts/data.sh verify --dataset data/processed/tiny-v1
```

Preparation performs cleaning and deduplication, assigns document-level train/validation/test splits before BPE training, and learns BPE only from the training split. The prepared output is disk-backed and contains token examples plus non-text metadata, not a copy of the source text. Use the same or a compatible config for training. The requested training context must not exceed the context used when preparing the dataset.

## 3. Phase 5 multilingual CPU pilot (preparation and reporting only)

`configs/multilingual_tiny.toml` is a separate preset; it does not replace existing configs. It uses the custom byte-level BPE and Transformer, a 2,048-token target vocabulary, `d_model=64`, 2 layers, 4 heads, context 128, batch 2, CPU, and FP32. Initial sampling weights are `en=20%`, `hi=20%`, and `es=15%`, `fr=15%`, `ar=15%`, `zh=15%`. Training samples examples with replacement according to these weights, so a larger corpus cannot dominate only because it has more documents/examples.

For this config, each JSONL record or TXT sidecar must include a language code, source, and license. Supported codes are English `en`, Hindi `hi`, Spanish `es`, French `fr`, Arabic `ar`, and Chinese `zh`; language-folder labels must match the record/sidecar. At least three unique usable documents per language are needed to populate every split. See `data/raw/README.md` for the exact JSONL field types, sidecar naming, and rights requirements.

After adding only your own authorized documents, prepare, verify, report, and inspect resources with:

```bash
./scripts/data.sh prepare \
  --input data/raw \
  --output data/processed/multilingual-v1 \
  --config configs/multilingual_tiny.toml
./scripts/data.sh stats --dataset data/processed/multilingual-v1
./scripts/data.sh verify --dataset data/processed/multilingual-v1
./scripts/data.sh report \
  --dataset data/processed/multilingual-v1 \
  --config configs/multilingual_tiny.toml \
  --output data/reports/multilingual-v1.json
./scripts/hardware.sh --config configs/multilingual_tiny.toml
```

The report includes overall and per-language document/token/example counts, split distribution, duplicate counts, byte/BPE target and actual vocabulary information, tokenizer audits, parameter count, and the existing RAM estimate. Verification checks contiguous per-language example ranges, every configured language in each split, provenance, and cross-split exact-hash leakage. The weighted sampler currently requires `num_workers=0` so checkpointed sampling state is exact. These commands only prepare and inspect data/resources. **No real data has been prepared and no training has been run for this task; do not start training until you explicitly approve it.**

## 4. Configure and start a small run

The presets are TOML files:

- `configs/tiny.toml`: 64-wide, 2-layer model, context 128, target vocabulary 384, batch 8, 1,000 steps.
- `configs/small.toml`: 256-wide, 6-layer model, context 256, target vocabulary 8,000, batch 4, 5,000 steps.
- `configs/medium.toml`: 512-wide, 8-layer model, context 512, target vocabulary 16,000, batch 2, 10,000 steps.
- `configs/multilingual_tiny.toml`: 64-wide, 2-layer, 4-head CPU/FP32 model, context 128, target vocabulary 2,048, batch 2, with six-language sampling weights. It is a separate preparation/training preset; no existing config is overwritten.

Each config specifies AdamW learning rate/weight decay, cosine or constant schedule, warmup, minimum learning rate, gradient clip, seed, validation cadence, optional early stopping, and loss-safety limits. Adjust a copy of a config deliberately, then inspect the hardware report again. In particular, keep `model.vocab_size` equal to `tokenizer.vocab_size` in the config; after BPE training, Puzzel adjusts the model head to the actual learned vocabulary.

After the prepared dataset passes verification, the exact tiny real-data command is:

```bash
./scripts/train.sh \
  --config configs/tiny.toml \
  --dataset data/processed/tiny-v1 \
  --output checkpoints/my-first-run.pt
```

This command is documented for the user; it is not run as part of development verification. A run starts only when this command is explicitly invoked. To continue that Puzzel run, use the same prepared data and output path and add the resume path:

```bash
./scripts/train.sh \
  --config configs/tiny.toml \
  --dataset data/processed/tiny-v1 \
  --output checkpoints/my-first-run.pt \
  --resume checkpoints/my-first-run.pt
```

`max_steps` is an absolute target step; to extend a run, raise it in a compatible config copy. The optimizer, scheduler progress, RNG state, and step are restored. The model architecture, tokenizer, learning rate, weight decay, and schedule family/warmup/minimum must remain compatible.

A legacy `--text` option remains for one small UTF-8 file. It splits the file contiguously into train and validation portions and trains BPE on the training portion only. It is not the preferred route for multi-document training; use the prepared dataset pipeline to preserve document boundaries and reduce split leakage risk.

## Monitoring, outputs, and safeguards

- Console logs show step, training loss, most recent validation loss, learning rate, elapsed time, and estimated remaining time.
- A CSV metrics file is written next to the checkpoint by default (`<checkpoint-stem>.metrics.csv`); use `--metrics` to choose another path. It records per-step training loss, validation loss when evaluated, learning rate, gradient norm, elapsed time, and ETA. When resuming, rows beyond the last durable checkpoint step are removed before appending, preventing duplicated or uncommitted step records.
- Validation is evaluated on the distinct document-level validation split, initially at step 1 and then at configured intervals (also at the final step). A configured batch cap uses a deterministic random sample with replacement to bound evaluation memory. The test split is not used for training or validation.
- With the multilingual preset, the training sampler uses the configured language weights with replacement over verified per-language example ranges. The data-loader RNG is checkpointed for reproducible resume; the validation/test splits are never language-sampled for training.
- Optional early stopping uses validation loss and a configured patience/minimum improvement. `early_stopping_min_delta` affects patience resets only; the best checkpoint still tracks every strictly lower validation loss.
- Non-finite loss/gradients and configured extreme/sustained loss spikes stop training. Checkpoints are written to a temporary archive, CRC-checked, then atomically replaced. Non-finite model/optimizer state is rejected before saving, leaving the prior valid checkpoint untouched.
- The latest checkpoint contains model state, AdamW state, configs, tokenizer, training step, scheduler metadata, and RNG/data-loader state. The lowest validation-loss checkpoint is stored as `<checkpoint-stem>.best<suffix>`.
- Resume only from a Puzzel checkpoint. Never place third-party pretrained weights in these paths.

## Validation and claims

Tests use tiny temporary synthetic token sequences solely to check optimizer updates, loss reduction on a controlled sequence, validation, checkpoint/resume, CPU support, and failure handling. No real-corpus training or large-scale pretraining is part of the test suite. A loss number alone is not a language capability evaluation. Any quality claim requires a documented protocol and held-out genuine language data with appropriate rights. The local seven-suite evaluation format and checkpoint regression workflow are documented in [`evaluation.md`](evaluation.md); the full multi-capability program and explicit approval gates are in [`roadmap.md`](roadmap.md).
