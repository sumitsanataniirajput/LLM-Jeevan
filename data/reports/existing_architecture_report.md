# LLM-Jeevan / Puzzel — Existing Architecture Report

**Branch:** `arena/26b70af7-llm-jeevan` (active)  
**Inspection date:** 2026-10-31  
**Source:** Full workspace inspection at `/home/user/LLM-Jeevan`  
**Backup:** `/home/user/LLM-Jeevan-backup-20261031-180933`

---

## 1. Repository Layout

```
/home/user/LLM-Jeevan/
├── README.md
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── configs/
│   ├── tiny.toml
│   ├── small.toml
│   ├── medium.toml
│   ├── multilingual_tiny.toml
│   ├── instruction_sft.toml
│   └── evaluation/
│       ├── creative_writing.toml
│       ├── general_knowledge.toml
│       ├── mathematics.toml
│       ├── multilingual_conversation.toml
│       ├── programming.toml
│       ├── reasoning.toml
│       └── translation.toml
├── data/
│   ├── raw/{en,hi,es,fr,ar,zh}/   (110 source documents + .meta.json sidecars)
│   ├── prepared/                   (train.bin, validation.bin, test.bin,
│   │                                tokenizer.json, manifest.json, stats.json,
│   │                                documents.sqlite3)
│   ├── processed/multilingual-v1/.gitkeep
│   ├── evaluation/README.md
│   └── reports/
│       ├── dataset_report.json
│       └── technical_audit.md
├── docs/
│   ├── architecture.md
│   ├── environment.md
│   ├── evaluation.md
│   ├── roadmap.md
│   └── training.md
├── puzzel/                         (main Python package)
│   ├── __init__.py
│   ├── __main__.py
│   ├── api.py                      (FastAPI REST API)
│   ├── checkpoint.py               (atomic checkpoint save/load)
│   ├── config.py                   (TOML config loader)
│   ├── inference.py                (InferenceEngine, sampling)
│   ├── languages.py                (6 language codes, default weights)
│   ├── tokenizer.py                (ByteTokenizer + BPETokenizer)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── cli.py                  (prepare/stats/verify/report)
│   │   ├── pipeline.py             (ingestion, dedup, splits, BPE, token splits)
│   │   ├── prepared.py             (mmap readers, verification)
│   │   ├── quality.py              (normalization, quality rejection)
│   │   └── text_dataset.py         (sliding-window dataset)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── cli.py                  (evaluate/compare)
│   │   ├── metrics.py              (exact_match, chrf, perplexity)
│   │   ├── reports.py              (JSON/CSV reports, regression comparison)
│   │   ├── runner.py               (greedy_generate, evaluate_checkpoint)
│   │   ├── sandbox.py              (CodeSandboxCommand adapter)
│   │   └── specs.py                (SuiteSpec + 7 suite specs)
│   ├── model/
│   │   ├── __init__.py
│   │   ├── config.py               (ModelConfig dataclass)
│   │   └── transformer.py          (TransformerLM decoder-only)
│   ├── training/
│   │   ├── __init__.py
│   │   ├── cli.py                  (train_model CLI)
│   │   ├── hardware.py             (hardware detection, memory estimates)
│   │   └── loop.py                 (AdamW training, validation, checkpointing)
│   └── checkpoint.py               (puzzel-checkpoint-v1 format)
├── scripts/
│   ├── bootstrap.sh
│   ├── data.sh
│   ├── evaluate.sh
│   ├── hardware.sh
│   ├── serve.sh
│   ├── train.sh
│   └── verify.sh
└── tests/
    ├── conftest.py
    ├── test_config.py
    ├── test_data_quality.py
    ├── test_dataset.py
    ├── test_dataset_pipeline.py
    ├── test_evaluation.py
    ├── test_hardware.py
    ├── test_inference.py
    ├── test_model.py
    ├── test_multilingual_pipeline.py
    ├── test_phase9_regressions.py
    ├── test_sft_spec.py
    ├── test_training.py
    ├── test_training_cli.py
    └── test_tokenizer.py
```

**Important finding:** No `.pt` checkpoint file exists in the repository. `checkpoints/README.md` is the only file in `checkpoints/`. The README explicitly states no trained production checkpoint exists and no corpus was prepared for training. The model has never been trained on real data in this repository.

---

## 2. Module Dependencies Map

```
puzzel/model/config.py  ──┐
                          ├──► puzzel/model/transformer.py (TransformerLM)
puzzel/tokenizer.py  ───┘
                          ├──► puzzel/training/loop.py (train_model)
                          ├──► puzzel/inference.py (InferenceEngine)
                          ├──► puzzel/data/pipeline.py (prepare_dataset)
                          └──► puzzel/checkpoint.py (save/load)

puzzel/config.py  ───► RunConfig (model, training, tokenizer, dataset)
                          ├──► puzzel/training/cli.py
                          ├──► puzzel/data/cli.py
                          └──► puzzel/training/hardware.py

puzzel/training/loop.py ──► puzzel/checkpoint.py (save_checkpoint/load_checkpoint)
                          ├──► puzzel/training/hardware.py (resolve_device)
                          └──► puzzel/evaluation/runner.py (evaluate_loss)

puzzel/api.py ───► puzzel/inference.py (InferenceEngine.from_checkpoint)
                  └──► FastAPI + Pydantic (endpoints: /health, /model/info, /generate)

puzzel/evaluation/runner.py ──► puzzel/checkpoint.py (load_checkpoint)
                              ├──► puzzel/evaluation/specs.py (SuiteSpec)
                              ├──► puzzel/evaluation/metrics.py (chrf, exact_match)
                              ├──► puzzel/evaluation/sandbox.py (CodeSandboxCommand)
                              └──► puzzel/training/loop.py (resolve_device)

puzzel/data/pipeline.py ──► puzzel/config.py (RunConfig)
                          ├──► puzzel/data/prepared.py (verify_prepared_dataset)
                          ├──► puzzel/data/quality.py (normalize_text, quality_rejection_reason)
                          └──► puzzel/tokenizer.py (BPETokenizer.train_streaming)
```

**Entry points:**
- `python -m puzzel` → `puzzel/training/cli.py:main()` (training)
- `python -m puzzel.data.cli` → `puzzel/data/cli.py:main()` (data prep)
- `python -m puzzel.evaluation.cli` → `puzzel/evaluation/cli.py:main()` (evaluation)
- `python -m puzzel.api` → `puzzel/api.py` (FastAPI server, `uvicorn.run(app, ...)`)
- `scripts/serve.sh` → `.venv/bin/python -m puzzel.api`

**Existing API endpoints:**
- `GET /` — service description
- `GET /health` — health check (model_loaded, device)
- `GET /model/info` — model metadata + limitations list
- `POST /generate` — text generation with temperature/top-k/top-p/seed
- `GET /docs` — FastAPI auto docs

---

## 3. Current Model Architecture

**File:** `puzzel/model/transformer.py`

| Attribute | Value |
|---|---|
| Architecture | Decoder-only causal Transformer |
| Layers | 2 |
| d_model | 64 |
| Attention heads | 4 (head_dim = 16) |
| MLP ratio | 4 |
| Vocab size | 2,048 (BPE) |
| Context length | 128 |
| Parameter count | 239,360 |
| Position embeddings | Learned absolute (fixed to context_length) |
| Normalization | Pre-LayerNorm |
| Activation | GELU |
| Dropout | 0.1 |
| Weight tying | True (token_embedding ↔ language_model_head) |
| Causal mask | Lower-triangular buffer, registered non-persistent |
| Weight init | Normal(0, 0.02), LayerNorm weight near 1 |
| Precision | fp32 only |

**Key design decisions (preserved):**
- Pre-LayerNorm residual blocks
- Causal self-attention with explicit lower-triangular mask
- QKV fused projection (`nn.Linear(d_model, 3 * d_model)`)
- Weight tying between input embedding and output head
- Random initialization only — no pretrained weights

**Limitations:**
- No RoPE (learned absolute positions cannot extrapolate beyond context_length)
- No RMSNorm (LayerNorm only)
- No SwiGLU (GELU only)
- No GQA (standard MHA only)
- No KV-cache (full recompute per step)
- No gradient checkpointing
- No mixed precision (config rejects non-fp32)
- Context fixed at 128 tokens

---

## 4. Tokenizer

**File:** `puzzel/tokenizer.py`

| Attribute | Value |
|---|---|
| Type | Byte-level BPE (learnable) + ByteTokenizer fallback |
| Byte fallback | IDs 0–255 (UTF-8 bytes, lossless) |
| Special tokens | BOS=256, EOS=257, PAD=258, UNK=259 |
| Learned merges | Up to target_vocab_size - 260 |
| Vocab size | Configurable (default 260 byte+special, 2,048 in multilingual_tiny) |
| Round-trip | Verified lossless for en/hi/es/fr/ar/zh |
| Training | Disk-backed streaming (SQLite temp), train split only |

**Limitations:**
- No language-ID tokens
- No mask/sep tokens
- No tokenizer versioning beyond format strings
- Small vocab (2,048) forces fragmentation on CJK/Devanagari scripts

---

## 5. Data Pipeline

**File:** `puzzel/data/pipeline.py`

| Feature | Status |
|---|---|
| Input formats | .txt, .jsonl, .ndjson |
| Metadata sidecars | .txt.meta.json (language, source, license) |
| require_document_metadata | Enforced in multilingual_tiny.toml |
| UTF-8 validation | Yes |
| Normalization | NFC, whitespace/paragraph handling |
| Quality rejection | Replacement char, NUL, control chars, single-char repetition, repeated lines, ngrams |
| Deduplication | SHA-256 digest, global across all languages |
| Splits | Document-level, per-language stratified (hash-seeded) |
| Tokenizer training | Train split only (no leakage) |
| Token storage | Fixed-width int32, mmap-compatible |
| Verification | Cross-split leakage, token ID range, round-trip, special tokens |
| Minimum docs | 3 total, 3 per language when sampling weights configured |

**Current corpus:** 49 retained docs / 174,123 tokens / 61 English duplicates removed  
**Splits:** train 33 docs / validation 8 docs / test 8 docs

---

## 6. Training Pipeline

**File:** `puzzel/training/loop.py`

| Feature | Status |
|---|---|
| Optimizer | AdamW |
| Scheduler | Cosine or constant, with warmup |
| Gradient clipping | Configurable norm |
| Checkpointing | Atomic, CRC-validated, resume-capable |
| Best checkpoint | Separate `.best.pt` file |
| Early stopping | Patience-based (disabled by default) |
| Divergence detection | Non-finite loss, spike threshold, 3-strike rule |
| Language sampling | Weighted, replacement-based |
| Data loader | BufferedShuffleSampler + LanguageBalancedSampler |
| Precision | fp32 only (config enforces) |
| Mixed precision | Not supported |
| Gradient accumulation | Not implemented |
| Gradient checkpointing | Not implemented |
| Device | CPU/CUDA/MPS auto-detection |

**Checkpoint format:** `puzzel-checkpoint-v1` zip archive with `weights_only=True` load, model + optimizer + scheduler + RNG states, tokenizer, configs.

---

## 7. Evaluation Framework

**7 suite specs** (in `configs/evaluation/`):
1. multilingual_conversation — conversation generation, chrF + nonempty_rate
2. general_knowledge — factual QA, exact_match + perplexity
3. programming — code generation (requires external sandbox)
4. mathematics — problem solving, exact_match
5. reasoning — final-answer accuracy, exact_match
6. translation — chrF-based translation scoring
7. creative_writing — human rubric only

**Metrics:** conditional NLL, perplexity, exact_match, chrF, nonempty_rate, human rubric mean  
**Code sandbox:** Opt-in `CodeSandboxCommand` adapter (external isolation, never runs code in-process)  
**Decoding:** Greedy only  
**Reports:** JSON + CSV with SHA-256 provenance, regression comparison

---

## 8. Inference API

**File:** `puzzel/api.py` + `puzzel/inference.py`

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Service description |
| `/health` | GET | Health check |
| `/model/info` | GET | Model metadata + limitations |
| `/generate` | POST | Text generation (Pydantic validated) |
| `/docs` | GET | FastAPI auto docs |

**Generation controls:** temperature, top-k, top-p, seed, stop_at_eos, max_new_tokens  
**Device:** CPU only  
**Checkpoint:** Read-only load via `PUZZEL_CHECKPOINT` env var

---

## 9. Hardware Constraints (Current Environment)

| Resource | Value |
|---|---|
| CPU | 2 logical cores, Intel Xeon @ 2.60 GHz |
| RAM | ~3.8 GB total, ~3.2 GB available |
| GPU/CUDA | Not available |
| MPS | Not available |
| Disk free | ~12 GB |
| torch | 2.15.0+cu132 (CPU-only build) |

**Suitable for:** Tests and tiny CPU pilots only (~1-5M parameter models). Not suitable for useful-scale pretraining.

---

## 10. Test Suite

**67 tests** across 15 test files:
- test_model.py — forward shape, causal masking, RNG init, torch.export smoke
- test_tokenizer.py — byte/BPE round-trip, multilingual
- test_training.py — training loop, checkpoint resume
- test_inference.py — inference engine + FastAPI TestClient
- test_data_quality.py — normalization, quality rejection
- test_dataset.py — prepared dataset readers
- test_dataset_pipeline.py — pipeline end-to-end
- test_evaluation.py — evaluation runner
- test_hardware.py — hardware detection
- test_multilingual_pipeline.py — multilingual end-to-end
- test_phase9_regressions.py — 8 regression tests (3 fail against baseline)
- test_sft_spec.py — SFT data contract validation
- test_training_cli.py — training CLI
- test_config.py — config loading

---

## 11. What Does NOT Exist Yet

| Feature | Status |
|---|---|
| Trained `.pt` checkpoint | ❌ None exists |
| RoPE, RMSNorm, SwiGLU, GQA | ❌ Not implemented |
| KV-cache | ❌ Not implemented |
| Quantization | ❌ Not implemented |
| Mixed precision (fp16/bf16) | ❌ Config rejects non-fp32 |
| Gradient accumulation | ❌ Not implemented |
| Gradient checkpointing | ❌ Not implemented |
| SFT/DPO trainer | ❌ Only data contract spec exists |
| RAG / embeddings / vector DB | ❌ Not implemented |
| Agents / function calling / tools | ❌ Not implemented |
| Authentication / API keys / RBAC | ❌ Not implemented |
| Frontend (HTML/CSS/JS) | ❌ Not implemented |
| Admin dashboard | ❌ Not implemented |
| Database (users, conversations, etc.) | ❌ Not implemented |
| Rate limiting / CORS / input sanitization | ❌ Not implemented |
| Benchmark dashboard | ❌ Not implemented |
| Voice / TTS / vision | ❌ Not implemented |
| Long-term memory / user preferences | ❌ Not implemented |
| Plugin architecture | ❌ Not implemented |
| Model import/export management | ❌ Not implemented |
| Backup/restore | ❌ Not implemented |
