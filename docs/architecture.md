# Architecture and project boundaries

## Modules

- `puzzel/tokenizer.py` — deterministic byte fallback plus local byte-level BPE training/encoding.
- `puzzel/model/config.py` and `puzzel/model/transformer.py` — validated dimensions and the existing randomly initialized causal decoder.
- `puzzel/data/` — local TXT/JSONL ingestion, cleaning, deduplication, document-level splits, training-only BPE, disk-backed next-token examples, and integrity checks.
- `puzzel/training/hardware.py` — CPU/RAM/CUDA/MPS/disk detection, parameter and memory estimates, and conservative batch recommendations.
- `puzzel/training/loop.py` — AdamW next-token training, cross-entropy, step-indexed warmup/cosine or constant learning-rate scheduling, clipping, validation, early stopping, metrics, ETA, and checkpoint integration.
- `puzzel/training/cli.py` — explicit training command for prepared datasets or a small local UTF-8 file.
- `puzzel/checkpoint.py` — atomic, CRC-checked local saves and safe loading of Puzzel checkpoints.
- `configs/{tiny,small,medium}.toml` — explicit experimental configurations; none is automatically selected or scaled up.

## Tokenizer and data flow

The tokenizer has 256 UTF-8 byte IDs plus BOS/EOS/PAD/UNK. BPE merges are learned from supplied text only. In prepared-dataset training, the pipeline creates deterministic document-level train/validation/test splits **before** BPE training; only the training split informs merges. The actual learned BPE vocabulary is recorded, and the model head is constructed to match it.

Each prepared next-token row contains at most one document's tokens, shifted by one position and padded only at the right edge. Prepared split datasets are memory-mapped and can expose a shorter context than the stored maximum without joining documents. The training CLI uses train and validation only; the test split remains held out. Training shuffles with a bounded index buffer rather than building a full-corpus permutation, and a capped validation evaluation uses a deterministic random sample with replacement to keep memory bounded.

## Model details

The existing decoder-only Transformer has learned token and positional embeddings, pre-LayerNorm blocks, causal multi-head self-attention, GELU feed-forward layers, residual connections, and a vocabulary projection. It maps `[batch, sequence]` token IDs to `[batch, sequence, vocabulary]` logits; next-token cross-entropy ignores PAD targets.

Every fresh model is created by `TransformerLM`'s explicit random initialization, with an optional tied output embedding. The CLI seeds supported RNGs before constructing a fresh model and checks finite, nonzero, non-constant parameters. Resume is limited to Puzzel-format local checkpoints; no pretrained loader or external model API is used.

## Training, monitoring, and failure handling

The engine uses AdamW with configurable learning rate and weight decay. A step-indexed schedule supports warmup followed by cosine decay to a minimum learning rate or a constant rate. Training clips gradients, evaluates validation loss periodically without gradients, logs CSV metrics and console progress, estimates ETA, and can early-stop on validation patience. Non-finite loss or gradients and large/sustained loss spikes stop the run.

Hardware inspection uses visible OS/cgroup memory, CUDA free/total memory only when a CUDA device is available, and filesystem free space. Model parameter counts are estimated from dimensions without allocating a model. Memory estimates include FP32 weights, gradients, AdamW moments, activations/logits, and an additional margin. Batch recommendations are intentionally capped and advisory; they do not modify user settings and cannot guarantee runtime fit or throughput.

Checkpoints include model and optimizer state, configs, tokenizer, step, schedule metadata, RNG and data-loader state, and best-validation metadata. Save first goes to a unique same-directory temporary archive, is checked for ZIP CRC errors, then atomically replaces the target. Non-finite parameters or optimizer tensors are rejected before the previous checkpoint is touched. The best validation checkpoint has a distinct path from the latest resumable checkpoint.

## Evaluation boundaries

A falling loss is a training diagnostic, not proof of language understanding. Synthetic fixtures in tests are temporary and are not real training data. The repository contains no prepared real corpus and no trained production checkpoint. Capability claims require a separate documented evaluation on held-out genuine language data with the relevant usage rights.

## Android path

The model is separate from the training code and uses standard PyTorch modules. A fixed-shape `torch.export` smoke test checks capture compatibility, but Android support is not implemented or verified. Later work must measure export/operator compatibility, memory, latency, and quantization on the intended devices.
