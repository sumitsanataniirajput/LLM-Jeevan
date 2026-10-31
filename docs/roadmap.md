# Multilingual multi-capability foundation model roadmap

This roadmap describes a staged research program for Puzzel. It does **not** assert that the model currently has any listed capability. The repository has no real corpus, no trained production checkpoint, and no benchmark results. Do not start training or download data automatically; every dataset and training run requires a separate, explicit user decision.

## Non-negotiable engineering rules

1. Initialize every fresh model from the existing `TransformerLM` random initialization. Load only Puzzel-owned checkpoints for resume/evaluation. Never add pretrained LLMs, pretrained weights, external AI APIs, or hidden model-download paths.
2. Preserve the existing Transformer architecture and project-owned UTF-8 byte/byte-level BPE tokenizer. Keep their contracts and tests stable; add capabilities around them rather than replacing them.
3. Use only user-owned or properly licensed corpora. Record source, exact license/terms, required attribution, restrictions, data hash, and evaluation/training split. Keep raw licensed material and evaluation corpora out of Git.
4. Split by document, conversation, or source group **before** tokenizer fitting or task-specific fine-tuning. Deduplicate across splits and freeze evaluation sets by hash.
5. Never claim capability from training loss, synthetic tests, an unverified dataset, or a proxy metric alone. Reports must distinguish unscored/not-run values from zeros and never fabricate results.
6. Resource estimates and test runs are not permission to train. Preserve CPU-safe defaults; require explicit device, configuration, output, and approval before any real-data run.

## Target capabilities

| Capability | Primary evidence | Important caveat |
| --- | --- | --- |
| Multilingual conversations | held-out conversation response metrics by language plus blinded human rubric | Base next-token pretraining alone does not establish instruction-following dialogue |
| Programming | executed test-case and unit-test pass rates in an isolated sandbox | Never execute generated code on the host; static similarity is not correctness |
| General knowledge | curated factual QA accuracy and source/date-aware review | Exact-match is a narrow proxy; facts can be outdated or ambiguous |
| Reasoning | final-answer accuracy plus separately reviewed explanation rubric | A correct answer does not prove a valid reasoning process |
| Translation | held-out chrF and blinded human adequacy/fluency assessment by language pair | Reference overlap is not a complete measure of meaning or style |
| Mathematics | normalized exact answer accuracy and reviewed derivations | Symbolic equivalence is not inferred by the current exact matcher |
| Creative writing | blinded, task-specific human rubric | No automatic model/AI judge and no single reference-based creativity score |

## Phase status and exit criteria

### Phase A — multilingual foundation data and tokenizer support

**Status: implementation in place; real-corpus validation pending.** The local pipeline supports English, Hindi, Spanish, French, Arabic, and Chinese; UTF-8 TXT with provenance sidecars; current JSONL/NDJSON with strict multilingual `text`, `language`, `source`, and `license` metadata; document-level stratified splits before BPE; per-language counts/ranges; byte fallback; Unicode round-trip and special-token audits. `configs/multilingual_tiny.toml` targets a 2,048-entry BPE vocabulary without assuming the trained vocabulary must fill that target.

**Exit gate:** owner-approved licensed documents in all six languages, each represented in train/dev/test; reviewed ingestion report; successful `verify` and hardware report; documented corpus hashes, license inventory, split counts, duplicates, and tokenizer audit. No data is currently present, so this gate is not passed.

### Phase B — scalable multilingual next-token pretraining

**Status: small-run infrastructure exists; scale validation pending.** The existing trainer uses the preserved causal Transformer, next-token cross-entropy, configurable `tiny`/`small`/`medium` presets, local checkpoint resume, CPU and available accelerator selection, per-step metrics, and multilingual weighted sampling. The language-balanced sampler draws from verified prepared language ranges with replacement; no distributed training or production-scale data loader is claimed.

**Next work:** benchmark storage throughput and checkpoint recovery on approved real data; add sharded datasets and resumable shard cursors; specify mixed precision only after numerical tests; add multi-GPU/distributed strategy only after architecture-preservation and CPU regression tests.

**Exit gate:** user-approved run plan; verified train/validation/test boundaries; deterministic resume test; hardware and approximate FLOP/RAM/disk report; failure/restart drill; explicit training approval. No pretraining has started.

### Phase C — capability dataset specifications

**Status: versioned specifications added; no datasets included.** Seven separate evaluation-suite TOMLs define schemas, required provenance, metric directions, and generation limits. `configs/instruction_sft.toml` defines a separate instruction-response record/split/sampling contract.

**Next work:** review each schema with domain owners, add source-level deduplication, annotation guidance, contamination checks, age/date fields for facts, language-pair balancing for translations, and license/attribution audit artifacts. Build each dataset only from approved local sources.

**Exit gate:** validated data dictionaries and provenance; independent train/dev/test manifests; no sample may contain an unlicensed or synthetic filler record in a real-data release.

### Phase D — instruction fine-tuning

**Status: design only; no SFT trainer or instruction dataset.** Instruction-response data has a separate schema, task/language weighting policy, conversation/source-group split unit, and provenance requirements in `configs/instruction_sft.toml`.

**Next work:** implement a separate supervised fine-tuning loop that reuses the existing model/tokenizer/checkpoint contracts; add response-only loss masks, task/language sampling, resume, validation, and optimizer-state tests. Keep pretraining and SFT checkpoints/configs/report labels distinguishable.

**Exit gate:** synthetic unit tests, approved licensed examples, no train/eval source-group leakage, training/validation-loss checks, and a user-approved SFT run plan. SFT remains unimplemented and unrun.

### Phase E — independent capability evaluation

**Status: initial local harness implemented; benchmark validation pending.** The evaluator loads an existing Puzzel checkpoint, never initializes a training run, uses deterministic greedy decoding, computes bounded-context conditional NLL/perplexity, answer accuracy and chrF where specified, groups results by language, accepts separate human rubric files, supports an opt-in external code sandbox launcher, and writes privacy-preserving JSON/CSV reports.

**Next work:** add confidence intervals and paired bootstrap comparisons; calibrate metrics against human review; track translation language pairs separately; add factual freshness/source audits; add code benchmark environment/version manifests; add multi-turn conversation turn-level scoring; document annotator agreement.

**Exit gate:** frozen held-out evaluation sets, independent human annotation protocol, repeatable runner/environment hashes, all required suite reports, no unsupported claim based only on a proxy metric. No benchmark result has been generated.

### Phase F — scalable model/resource planning

**Status: small presets and estimator exist; estimates are approximate.** Existing TOML configurations preserve the model architecture. The resource report computes parameter counts, FP32 training memory, checkpoint disk estimates, conservative batch recommendations, and an approximate FLOP count before training. The FLOP estimate is an engineering approximation, not measured compute or runtime.

**Next work:** add named experimental size profiles only as new configs; estimate sequence-length/data throughput and checkpoint/shard storage; calibrate compute estimates against approved measured runs; validate CUDA/MPS paths in their target environments; keep CPU smoke tests and peak-memory limits.

**Exit gate:** every experimental config has parameter/RAM/disk/FLOP estimates, a fit warning and explicit device policy; no automatic scale-up; user approval before training.

### Phase G — reporting and regression tracking

**Status: initial JSON/CSV reports and checkpoint comparison added.** Reports include suite, checkpoint, dataset, and spec SHA-256 values, evaluated counts/partitions, overall and per-language metrics, training/validation loss when supplied, human-rubric coverage, and sandbox status. Comparisons require the same suite/spec/evaluation dataset/settings and flag direction-aware regressions at a configurable absolute threshold.

**Next work:** add signed/immutable report bundles, run metadata (code revision, runtime versions, tokenizer hash), dataset cards, statistical significance/bootstrap intervals, and cross-suite release dashboards. Preserve `null` for unmeasured metrics and never turn missing sandbox/human scores into zero.

## Safe execution order

1. Review schemas and license requirements; author evaluation protocols.
2. Add approved local data manually and validate provenance/quality without training.
3. Freeze data hashes and splits; review CPU/GPU, disk, memory, and FLOP estimates.
4. Obtain explicit approval for a named pretraining run.
5. Evaluate a checkpoint independently across all seven suites; store baseline reports.
6. If separately approved, implement/run SFT and compare the resulting checkpoint against the frozen baseline across the same suite data and settings.
7. Review regressions and human evaluation before any capability statement.

At every stage, absent results remain absent; a roadmap item is not evidence of a model capability.
