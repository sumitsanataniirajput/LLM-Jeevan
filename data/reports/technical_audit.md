# LLM-Jeevan — Advanced Model Development Audit

**Branch:** `arena/17034f0e-llm-jeevan`  
**Audit date:** 2026-10-31  
**Status:** Audit only — no implementation performed  
**Purpose:** Technical audit of the existing codebase, datasets, training pipeline, and hardware; advanced development strategy for a significantly more powerful multilingual model

---

## 1. Executive Summary

The existing project is a small, from-scratch multilingual decoder-only Transformer pilot. It is functionally complete and well-engineered for its scale, but it is **orders of magnitude too small** to be a capable language model. The current checkpoint (239K parameters, 1,000 training steps, 174K tokens) produces repetitive output and near-zero factual accuracy. A "significantly more powerful and advanced" model will require a substantially larger architecture, a much larger multilingual corpus, longer training, and improved tokenization — all trained from randomly initialized weights on local hardware.

Key finding: **No expensive training should begin until dataset readiness and hardware are confirmed.** The current environment is CPU-only with ~3.8 GB RAM, which is only suitable for small-scale experiments.

---

## 2. Component-by-Component Audit

### 2.1 Transformer Architecture

| Attribute | Current Value | Assessment |
|---|---|---|
| Architecture | Decoder-only causal Transformer | Sound baseline design |
| Layers | 2 | Severely under-capacity |
| d_model | 64 | Too small for multilingual representation |
| Heads | 4 | head_dim = 16, adequate but limited |
| MLP ratio | 4 | Standard |
| Vocab size | 2,048 | Tiny; forces suboptimal tokenization |
| Context length | 128 | Too short for meaningful context |
| Parameter count | 239,360 | Far below what is needed |
| Precision | fp32 | Stable but memory-inefficient |
| Weight tying | True | Good for parameter efficiency |
| Position embeddings | Learned absolute, fixed to context_length | Limits extrapolation beyond 128 tokens |
| Normalization | Pre-LayerNorm | Good practice |
| Activation | GELU | Standard |
| Dropout | 0.1 | Reasonable |

**Parameter breakdown (approximate):**
- Embeddings: 2048 × 64 + 128 × 64 ≈ 139K
- Transformer blocks (×2): each ≈ 4×64² + 2×64×256 + overhead ≈ 50K → ~100K total
- Final norm + output head (tied): ~64
- **Total: ~239K**

**Verdict:** The architecture is correctly implemented (causal masking, pre-LN residuals, weight tying, gradient checkpointing-friendly), but the scale is toy-level. A capable multilingual model needs at minimum d_model=256–512, 6–12 layers, 8–16 heads, context 512–2048, and vocab 8K–32K.

---

### 2.2 Tokenizer

| Attribute | Current Value | Assessment |
|---|---|---|
| Type | Byte-level BPE | Sound choice for multilingual |
| Vocab size | 2,048 | Far too small |
| Byte fallback | 256 (IDs 0–255) | Good — lossless UTF-8 |
| Special tokens | 4 (BOS, EOS, PAD, UNK at 256–259) | Adequate but could add more |
| Learned merges | 1,788 | Too few for 6 languages |
| Encoding | Deterministic, training-rank order | Reproducible |
| Round-trip | Verified lossless for all 6 scripts | Good |

**Weaknesses:**
- 2,048 vocab is insufficient for 6 languages with different scripts. After byte fallback (256) and special tokens (4), only ~1,788 merge tokens remain. This forces frequent token fragmentation, especially for CJK (Chinese) and Devanagari (Hindi), inflating sequence length and degrading modeling.
- No dedicated mask token, no sep token, no language-ID tokens — limits future instruction-finetuning and multilingual prompting.
- BPE trained only on the training split (correct), but with such a small corpus the merge statistics are noisy.

**Recommended:** Byte-level BPE with vocab 8,192–32,768, trained on a much larger multilingual corpus. Add language-ID special tokens for explicit multilingual control.

---

### 2.3 Datasets (All Six Languages)

| Language | Docs | Tokens | % of Corpus | Source | Status |
|---|---|---|---|---|---|
| en | 31 | 82,489 | 47.4% | PG #1342 Pride and Prejudice | Public domain |
| hi | 4 | 29,106 | 16.7% | Hindi Wikisource, अयोध्या का इतिहास (1932) | Public domain |
| zh | 3 | 19,045 | 10.9% | PG #23839 Analects | Public domain |
| ar | 3 | 15,175 | 8.7% | PG #43007 Tribute to Michael Hart | Public domain (author donation) |
| fr | 4 | 14,849 | 8.5% | PG #17489 Les Misérables Tome I | Public domain |
| es | 4 | 13,459 | 7.7% | PG #2000 Don Quijote | Public domain |
| **Total** | **49** | **174,123** | **100%** | | |

**Splits (document-level, no leakage):**
- Train: 33 docs / 820 examples / 102,616 tokens
- Validation: 8 docs / 336 examples / 42,580 tokens
- Test: 8 docs / 231 examples / 28,927 tokens
- Duplicates removed: 61 (all English)

**Metadata:** Every `.txt` has a `.txt.meta.json` sidecar with `language`, `source`, `license`. All sources are public domain and permit ML training and commercial use. `require_document_metadata=true` is enforced.

**Verdict:** The data pipeline is well-engineered (SQLite staging, digest-based deduplication, document-level split assignment with per-language hash seeding, tokenizer round-trip verification during preparation). However:

1. **Corpus is tiny:** 174K tokens is far too few to train any capable language model. Modern multilingual models train on billions of tokens.
2. **Severe imbalance:** English is 47% of the corpus while low-resource languages (ar, zh) have only 3 documents each. Sampling weights (en=0.20, hi=0.20, others=0.15) partially compensate but cannot create information that isn't in the data.
3. **Single source per language:** English is only Pride and Prejudice; Arabic is only a tribute to Michael Hart; Chinese is only the Analects. No genre diversity, no dialogue, no code, no mathematics, no instruction data.
4. **No test/evaluation JSONL datasets exist yet** for most suites (only conversation and general_knowledge were created for Phase 8). Programming, mathematics, reasoning, translation, and creative_writing suite specs exist but lack local JSONL data.

---

### 2.4 Dataset Quality, Duplication, and Leakage

| Check | Result |
|---|---|
| UTF-8 validation | All 110 files valid |
| Empty documents | 0 |
| Metadata sidecars | All present, non-empty |
| Language matches folder | Verified |
| Exact-duplicate removal | 61 English chunks removed globally |
| Document-level splits | No leakage (splits assigned by document, not by chunk) |
| Cross-split leakage | Verified absent by `verify_prepared_dataset` |
| Tokenizer round-trip | Verified lossless during preparation |
| Metadata errors | 0 |

**Verdict:** Data quality and leakage prevention are well-handled for the current scale. The `digest`-based deduplication and document-level split hashing are correct practices. The main quality issue is **quantity and diversity**, not cleanliness.

---

### 2.5 Training Pipeline

| Component | Current State | Assessment |
|---|---|---|
| Optimizer | AdamW, lr=3e-4, wd=0.1 | Standard, appropriate |
| Scheduler | Cosine, warmup=20, min_lr=3e-5 | Reasonable |
| Batch size | 2 | Very small (memory constrained) |
| Max steps | 1,000 | Far too few for convergence |
| Gradient clip | 1.0 | Good safety |
| Device | CPU only | No GPU available |
| Precision | fp32 only | Stable but memory-heavy |
| Gradient accumulation | Not implemented | Would help effective batch size |
| Mixed precision | Not supported (config rejects non-fp32) | Memory/throughput opportunity lost |
| Checkpointing | Atomic, CRC-validated, resume-capable | Excellent |
| Early stopping | Disabled (patience=0) | Fine for fixed-step runs |
| Divergence detection | Non-finite loss, spike threshold, 3-strike rule | Good safety |
| Language-balanced sampling | Implemented, replacement-based | Good for multilingual |
| Data loader | num_workers=0, BufferedShuffleSampler / LanguageBalancedSampler | Correct |

**Checkpoint format:** `puzzel-checkpoint-v1` zip archive with `weights_only=True` load, model + optimizer + scheduler + RNG states, tokenizer, and config. Atomic save with temp-file + CRC + meta-device validation + `os.replace`. **This is production-quality checkpointing.**

**Verdict:** The training loop is well-engineered and safe, but the hyperparameters reflect a tiny pilot experiment. For a larger model, batch size, steps, learning rate, and precision all need revisiting.

---

### 2.6 Current Training Status

- **One checkpoint exists:** `checkpoints/multilingual-first-run.best.pt` (step 1000)
- **Train loss:** 6.43 (final) / **Validation loss:** 6.31 (best)
- **Test perplexity range:** 364–909 by language
- **Exact-match accuracy:** 0.0% on general knowledge across all 6 languages
- **Generation:** Highly repetitive under greedy decoding (repeated commas, newlines, or single characters); Spanish produces empty output
- **No overfitting detected** (val < train), confirming the model is under-trained, not overfit
- **Weights trained from scratch** (no pretrained weights used) — compliant with the user's rule

**Verdict:** A genuinely trained checkpoint exists, but it represents a toy-level model. It demonstrates the pipeline works end-to-end but does not represent useful language intelligence.

---

### 2.7 Evaluation Tests and Capabilities

**Evaluation suite specs (7):**
- `multilingual_conversation` — conversation generation, chrF + nonempty_rate
- `general_knowledge` — factual QA, exact_match + perplexity
- `programming` — code generation (requires code sandbox)
- `mathematics` — problem solving, exact_match
- `reasoning` — final-answer accuracy, exact_match
- `translation` — chrF-based translation scoring
- `creative_writing` — human rubric only (no automatic proxy)

**Metrics implemented:** conditional NLL, perplexity, exact_match, chrF, nonempty_rate, human rubric mean.

**Current measured capabilities (from Phase 8):**
- Perplexity: 364 (Hindi, best) to 909 (English, worst)
- chrF: 0.0–1.039 (Arabic best, Spanish/French zero)
- Exact match: 0.0% everywhere
- Nonempty generation: Spanish 0.0%, French 0.0% (conversation), 0.2% (GK)
- Tokenizer round-trip: perfect for all 6 languages
- No NaN/Inf losses
- No cross-language mixing detected

**Test suite:** 67 automated tests pass (model, tokenizer, data pipeline, training, hardware, evaluation, config, multilingual pipeline). Phase 9 added 8 regression tests (3 fail against the current baseline, documenting known issues).

**Verdict:** The evaluation framework is solid and well-designed, but the model's actual capabilities are minimal. The evaluation suites for programming, mathematics, reasoning, and translation lack local JSONL datasets.

---

### 2.8 Hardware Limitations and Training Feasibility

| Resource | Current | Assessment |
|---|---|---|
| CPU | 2 cores, Intel Xeon @ 2.60 GHz | Very limited for training |
| RAM | 3.8 GB total, ~3.2 GB available | Constrained |
| GPU/CUDA | Not available | CPU-only training |
| MPS | Not available | No Apple Silicon acceleration |
| Disk free | ~12 GB | Adequate for small checkpoints |
| torch | 2.15.0+cu132 | CPU-only build |

**Memory estimate for current model (batch=2, ctx=128):**
- Persistent (weights+gradients+AdamW): ~3.8 MB
- Activations: ~8.7 MB
- Total training estimate: ~12.5 MB
- Available budget: ~2.0 GB
- **Fits comfortably**, but only because the model is tiny.

**Training time:** ~13 ms/step → 1,000 steps in ~13 seconds (benchmark). A larger model would scale roughly linearly with parameters and steps.

**Verdict:** Current hardware is suitable only for small experiments (models up to ~1–5M parameters on CPU). Training a significantly larger multilingual model would require either:
- A GPU-enabled environment (strongly recommended), or
- Accepting very long CPU training times for medium models (~10–50M parameters)

---

## 3. Current Limitations and Exact Bottlenecks

| # | Limitation | Severity | Root Cause |
|---|---|---|---|
| 1 | 239K parameters | Critical | 2 layers, d_model=64 |
| 2 | 174K training tokens | Critical | Tiny corpus, single source per language |
| 3 | 47% English dominance | High | Data imbalance |
| 4 | 2,048 vocab | High | Insufficient for 6 scripts |
| 5 | 128 context | High | Too short for useful context |
| 6 | 1,000 training steps | High | Far below convergence |
| 7 | CPU-only, 2 cores, 3.8 GB RAM | Critical | No GPU; severe memory/compute limit |
| 8 | Greedy decoding only | Medium | No temperature/top-k/top-p in training loop (only in Phase 11 inference) |
| 9 | No instruction data | High | No dialogue, code, math, or instruction-tuning data |
| 10 | fp32 only | Medium | Misses memory/throughput savings of mixed precision |

---

## 4. Recommended Advanced Development Strategy

### 4.1 Scalable Decoder-Only Transformer Architecture

**Proposed architecture tiers:**

| Tier | Layers | d_model | Heads | MLP ratio | Vocab | Context | Params | Target Use |
|---|---|---|---|---|---|---|---|---|
| Small (CPU experiment) | 4 | 256 | 8 | 4 | 8,192 | 512 | ~15M | Rapid iteration on CPU |
| Medium (GPU target) | 8 | 512 | 8 | 4 | 16,384 | 1,024 | ~60M | Capable multilingual base |
| Large (aspiration) | 12–16 | 768–1,024 | 12–16 | 4 | 32,768 | 2,048 | 150–300M | Strong multilingual |

**Key architectural improvements:**
- **RoPE (Rotary Position Embeddings)** instead of learned absolute positions — enables length extrapolation beyond training context
- **SwiGLU** activation instead of GELU — better parameter efficiency
- **RMSNorm** instead of LayerNorm — slightly faster, numerically stable
- **Weight tying** — keep (parameter efficient)
- **Grouped-query attention (GQA)** for larger tiers — reduces KV cache memory during inference
- **fp16/bf16 mixed precision** — 2x memory savings on supported hardware
- **Gradient checkpointing** — trade compute for memory on large models

### 4.2 Better Multilingual Tokenization

- **Byte-level BPE with vocab 8,192–32,768** (trained on the expanded multilingual corpus)
- **Language-ID special tokens** (`<|lang_en|>`, `<|lang_hi|>`, etc.) for explicit multilingual control
- **Additional special tokens:** `<|mask|>`, `<|sep|>`, `<|user|>`, `<|assistant|>`, `<|system|>` for instruction-finetuning readiness
- Train tokenizer on **training split only** (preserve current good practice)
- Maintain byte fallback (256 IDs) for lossless UTF-8 round-trip

### 4.3 Large-Scale, Legally Usable Training Datasets

**Target corpus characteristics:**
- **Minimum:** 5–10M tokens per language (30–60M total) for a medium model
- **Ideal:** 100M+ tokens per language for a strong model
- **Sources (public domain, ML-training permitted):**
  - Project Gutenberg (expanded beyond single books per language)
  - Wikisource per language
  - Government publications, public domain religious texts, classical literature
  - Open-source code repositories (MIT/Apache/BSD) for programming
  - Public domain mathematics textbooks
- **Per-language target document counts:** 50–200+ documents per language (vs. current 3–31)
- **Genre diversity:** fiction, non-fiction, dialogue, code, mathematics, instruction-style pairs
- **Deduplication:** Keep the existing digest-based exact-duplicate removal; add near-duplicate/minhash filtering for larger corpora
- **Document-level splits:** Keep the existing split assignment (no leakage)
- **Metadata:** Keep `require_document_metadata=true` with language/source/license sidecars

### 4.4 Improved Pretraining and Optimization

- **Longer training:** 10,000–50,000+ steps (vs. current 1,000)
- **Larger batch size:** 8–64 (vs. current 2), with gradient accumulation if memory-constrained
- **Learning rate:** 1e-4 to 3e-4 with cosine decay and warmup (200–1,000 steps)
- **Gradient clipping:** 1.0 (keep)
- **Mixed precision:** fp16 or bf16 if GPU available; fp32 fallback for CPU
- **Gradient accumulation:** Simulate larger batches on memory-constrained hardware
- **Checkpointing:** Keep the existing atomic, CRC-validated, resume-capable design (it is production-quality)
- **Evaluation cadence:** Every 500–1,000 steps with full validation loss
- **Early stopping:** Patience of 5–10 evaluations with min_delta

### 4.5 Mathematics and Logical Reasoning

- **Data:** Public domain mathematics textbooks, problem sets, logic puzzles (with step-by-step solutions where available)
- **Architecture:** Chain-of-thought friendly context length (2,048+ tokens to allow reasoning traces)
- **Training:** Include reasoning traces as part of pretraining data; later, supervised fine-tuning on reasoning examples
- **Evaluation:** Use the existing `mathematics` and `reasoning` suite specs; create local JSONL datasets
- **Metric:** Exact match on final answers (already supported by suite spec)

### 4.6 Programming and Code Generation

- **Data:** Public domain / permissively licensed code (MIT, Apache 2.0, BSD), documentation, tutorials
- **Evaluation:** Existing `programming` suite spec supports `test_cases` with input/expected_output; requires an external code sandbox launcher (not included by design — "never inferred")
- **Training:** Code as part of pretraining; later supervised fine-tuning on (prompt, code) pairs
- **Metric:** Code execution pass rate via external sandbox (opt-in, not automatic)

### 4.7 Multilingual Instruction Following

- **Data:** Instruction-response pairs translated or originally written in each of the 6 languages
- **Format:** `<|system|>`, `<|user|>`, `<|assistant|>` chat template
- **Training:** Supervised fine-tuning after base pretraining (separate phase)
- **Evaluation:** Human rubric ratings (creative_writing spec supports this); conversation suite for diagnostic metrics

### 4.8 High-Quality Evaluation and Benchmarking

**Existing suite specs are well-designed but lack data.** Priorities:
1. Create local JSONL datasets for all 7 suites across all 6 languages
2. Add perplexity and chrF to more suites where appropriate
3. Maintain the "no AI judge" principle — human rubric for creative writing, exact match for factual/reasoning
4. Add held-out test evaluation after every training run
5. Track regression with comparison reports (existing `compare_evaluation_reports` infrastructure)

### 4.9 Efficient Training and Checkpoint Management

- **Existing checkpointing is production-quality** — atomic save, CRC validation, meta-device pre-validation, resume with RNG state restoration
- **Keep:** `puzzel-checkpoint-v1` format, `weights_only=True` load, optimizer + scheduler + RNG state
- **Add:** Checkpoint pruning (keep only latest + best + N periodic), disk-space monitoring before save
- **Add:** Training throughput tracking (tokens/second, steps/second) for hardware planning

---

## 5. Realistic Hardware Requirements

| Model Tier | Params | Min RAM | Recommended | GPU | Est. Training Time (10K steps) |
|---|---|---|---|---|---|
| Small | ~15M | 4 GB | 8 GB | CPU OK | Hours (CPU) |
| Medium | ~60M | 8 GB | 16 GB | GPU strongly recommended | Hours (GPU) / Days (CPU) |
| Large | ~150M+ | 16 GB | 32 GB+ | GPU required | Days (GPU) |

**Current environment:** 2 CPU cores, 3.8 GB RAM, no GPU → suitable only for Small tier experiments. Any Medium+ training requires GPU access.

**Disk:** 12 GB free is adequate for checkpoints (a 60M parameter fp32 checkpoint is ~240 MB; fp16 ~120 MB).

---

## 6. Prioritized Implementation Roadmap

### Phase A — Foundation (no expensive training)

| Priority | Task | Rationale |
|---|---|---|
| A1 | Expand dataset: 50–200+ docs per language, public domain, ML-permitted | Data is the binding constraint; everything else depends on it |
| A2 | Upgrade tokenizer to vocab 8K–32K BPE with language-ID tokens | Current 2K vocab is a hard ceiling on quality |
| A3 | Create evaluation JSONL datasets for all 7 suites × 6 languages | Cannot measure improvement without evaluation data |
| A4 | Implement RoPE, SwiGLU, RMSNorm architecture options | Required before meaningful scaling |
| A5 | Add gradient accumulation and mixed-precision (bf16/fp16) support | Enables larger effective batches on constrained hardware |

### Phase B — Medium Model Training (requires GPU approval)

| Priority | Task | Rationale |
|---|---|---|
| B1 | Train ~60M parameter model (8 layers, d_model=512, ctx=1,024) for 10K–50K steps | First capable multilingual base |
| B2 | Evaluate on all suites, track perplexity and exact match | Measure capabilities objectively |
| B3 | Supervised fine-tuning on instruction/response pairs in 6 languages | Enable instruction following |
| B4 | Code and mathematics fine-tuning on permissively-licensed data | Enable reasoning and programming |

### Phase C — Large Model (aspiration, requires significant hardware)

| Priority | Task |
|---|---|
| C1 | Scale to 150M–300M parameters (12–16 layers, d_model=768–1,024) |
| C2 | Context length 2,048+ for reasoning traces |
| C3 | GQA for efficient inference |
| C4 | Comprehensive benchmarking against the suite specs |

---

## 7. Rules Compliance Check

| Rule | Status |
|---|---|
| Train from randomly initialized weights | ✅ Current checkpoint trained from scratch; strategy continues this |
| No pretrained LLM weights or external AI APIs | ✅ No external model usage in codebase; strategy prohibits it |
| No silent replacement with pretrained model | ✅ Architecture is fully from-scratch Transformer |
| No unmeasured capability claims | ✅ All metrics tied to local evaluation suites; strategy requires measurement |
| No deletion of existing code/datasets | ✅ Strategy preserves all existing code and data |
| No expensive training without hardware/dataset check | ✅ Audit explicitly identifies hardware constraints; roadmap phases training behind data readiness |

---

## 8. Approval Requested

This is an **audit-only report**. No implementation has been performed or initiated.

**Before implementation begins, I request approval on:**

1. **Dataset expansion plan** — sourcing additional public domain multilingual text (need to confirm sources and licenses)
2. **Hardware path** — whether GPU access will be made available for medium/large model training (CPU-only severely limits scale)
3. **Architecture targets** — which tier (Small/Medium/Large) to pursue first
4. **Evaluation dataset creation** — creating JSONL data for all 7 suites across 6 languages

Once approved, I will begin with Phase A tasks (dataset expansion, tokenizer upgrade, evaluation data creation) before any training.