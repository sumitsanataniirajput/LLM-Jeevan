# Puzzel — from-scratch language-model starter

Puzzel is a small, modular foundation for independently training a decoder-only language model. It uses a **new, randomly initialized PyTorch Transformer**, a project-owned UTF-8 byte/byte-level BPE tokenizer, and a local-data pipeline. It contains no pretrained model or weights, bundled corpus, or external AI API integration.

This is a bootstrap project, not a claim that a useful foundation model has been trained. Use only text you own, public-domain content, or data whose license permits the intended use. The repository contains **no real training corpus**. No corpus was downloaded or prepared and no training/pretraining was run for Phase 5.

## Hardware observed for this workspace

See [`docs/environment.md`](docs/environment.md) for the re-detected hardware and preset memory estimates. This workspace has 2 logical CPUs, about 3.8 GiB RAM, and no CUDA or MPS device. Resource estimates are advisory; this machine is suitable for tests and tiny CPU pilots, not useful-scale pretraining.

## Setup

Python 3.11 or newer is required. Use the project virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/pip install -r requirements-dev.txt
```

Alternatively, `./scripts/bootstrap.sh` runs the same isolated setup. Project checks and wrappers use `.venv/bin/python` explicitly; no system-wide pip install is required.

## Phase 5 multilingual data

The separate `configs/multilingual_tiny.toml` preset preserves the existing Transformer and custom tokenizer and sets: target vocabulary 2,048; `d_model=64`; 2 layers; 4 heads; context 128; batch 2; CPU; FP32. It sets initial language weights to `en=20%`, `hi=20%`, `es=15%`, `fr=15%`, `ar=15%`, and `zh=15%`. The training sampler uses those proportions with replacement, so corpus size alone does not give larger languages more influence.

Put only actual, authorized UTF-8 documents under `data/raw/`. The six supported language folders are `en` (English), `hi` (Hindi), `es` (Spanish), `fr` (French), `ar` (Arabic), and `zh` (Chinese). TXT files are one document each and require a sibling `<filename>.txt.meta.json` sidecar with non-empty `language`, `source`, and `license` fields. Multilingual JSONL uses one complete JSON object per line with `text`, `language`, `source`, and `license`; `licence` is accepted as a license alias. Language codes must match their language folder. See [`data/raw/README.md`](data/raw/README.md) for the exact schemas, paragraph escaping, licensing rules, and commands for adding your own files. Do not use placeholder prose or test fixtures as training data.

After adding your own licensed files and provenance, prepare, verify, report, and inspect resources with these commands from the repository root:

```bash
mkdir -p data/raw/{en,hi,es,fr,ar,zh}

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

The preparation command requires at least three unique usable documents for **each** language to populate train/validation/test. Documents are split before BPE training; BPE learns only from the train split. Unicode, punctuation, and paragraphs are retained; invalid UTF-8, empty/corrupt documents, missing provenance, and duplicates are detected and reported. The verifier checks per-language statistics and example ranges, byte fallback, Unicode round trips, special tokens, and cross-split exact-hash leakage. `report` includes corpus sizes, per-language token/split counts, duplicate totals, target and actual vocabulary sizes, model parameter count, and the existing hardware/RAM estimate. Review ingestion warnings, licenses, verification, and resource estimates.

**These commands prepare and inspect only. Training has not been run. Stop here and seek explicit approval before invoking any training command.**

## Tests and existing presets

Run the project test/verification script with `./scripts/verify.sh`. The suite uses temporary synthetic fixtures only for tests; it does not download a corpus or run real-corpus pretraining. Existing `configs/tiny.toml`, `configs/small.toml`, and `configs/medium.toml` remain unchanged; the multilingual pilot is a separate config. See [`docs/training.md`](docs/training.md) for general training safeguards and [`docs/architecture.md`](docs/architecture.md) for model/tokenizer design and limitations.

## Roadmap and evaluation infrastructure

The seven-capability roadmap and completion gates are documented in [`docs/roadmap.md`](docs/roadmap.md). Separate local evaluation schemas cover multilingual conversations, knowledge, programming, mathematics, reasoning, translation, and creative writing; the SFT data contract is in `configs/instruction_sft.toml` and does not enable an SFT run.

After a Puzzel checkpoint and reviewed, licensed evaluation JSONL exist, the local evaluator can write structured JSON/CSV and compare checkpoints:

```bash
./scripts/evaluate.sh evaluate \
  --checkpoint checkpoints/puzzel-model.pt \
  --suite general_knowledge \
  --dataset data/evaluation/general_knowledge.jsonl \
  --json-output data/reports/general_knowledge.json \
  --csv-output data/reports/general_knowledge.csv

./scripts/evaluate.sh compare \
  --baseline data/reports/baseline-knowledge.json \
  --candidate data/reports/candidate-knowledge.json \
  --json-output data/reports/comparison.json \
  --csv-output data/reports/comparison.csv
```

These are templates only; there is no checkpoint/evaluation corpus and no result has been generated. Code execution is disabled unless an explicit external sandbox launcher is supplied. See [`docs/evaluation.md`](docs/evaluation.md) for schemas, metric limits, human ratings, sandbox requirements, and regression comparison. No capability claim is made.

## Initial design

- **Tokenizer:** `puzzel/tokenizer.py` implements lossless UTF-8 byte fallback (IDs 0–255), reserved special tokens (IDs 256–259), and custom deterministic byte-level BPE.
- **Model:** `puzzel/model/transformer.py` implements a configurable pre-LayerNorm causal decoder with learned token/position embeddings, masked multi-head attention, GELU feed-forward blocks, and a language-model head. All learned parameters are randomly initialized.
- **Data:** `puzzel/data/` handles local TXT/JSONL ingestion, provenance, cleaning, hash deduplication, deterministic document-level splits, disk-backed BPE learning, memory-mappable int32 examples, and per-language verification/reporting.
- **Training:** `puzzel/training/loop.py` implements AdamW next-token learning, scheduled learning rates, clipping, validation/early stopping, CSV monitoring, resume, and weighted language sampling for the multilingual preset.
- **Hardware:** `puzzel/training/hardware.py` detects CPU/RAM/CUDA/MPS/disk and estimates memory without allocating larger models.
- **Checkpoints:** `puzzel/checkpoint.py` atomically saves and validates model, optimizer, schedule, RNG, config, and tokenizer state.

See [`docs/architecture.md`](docs/architecture.md) for details and limitations.
