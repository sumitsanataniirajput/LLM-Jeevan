# LLM-Jeevan / Puzzel — File Modification Plan

**Branch:** `arena/26b70af7-llm-jeevan`  
**Plan date:** 2026-10-31  
**Scope:** All 230 approved features across 17 modules  
**Approach:** In-place upgrade with compatibility layers; no deletion of existing functional code

---

## Legend

- **[NEW]** — New file to be created
- **[MODIFY]** — Existing file to be modified (backward compatible)
- **[REPLACE]** — Existing file to be replaced with migration path
- **[PRESERVE]** — Existing file kept unchanged

---

## MODULE 1: Advanced AI Brain (Features 1–17)

### 1.1 Model Architecture Modernization

| File | Action | Changes |
|---|---|---|
| `puzzel/model/transformer.py` | **[MODIFY]** | Add RoPE, RMSNorm, SwiGLU, GQA, KV-cache, gradient checkpointing; keep existing MHA/LayerNorm/GELU as legacy mode; configurable via ModelConfig |
| `puzzel/model/config.py` | **[MODIFY]** | Add RoPE/RMSNorm/SwiGLU/GQA flags, model size profiles (tiny/small/medium/large), validation for attention dimensions |
| `puzzel/model/__init__.py` | **[MODIFY]** | Export new architecture classes |
| `puzzel/checkpoint.py` | **[MODIFY]** | Add migration utilities for old checkpoints (format versioning, shape adaptation for vocab/d_model changes) |
| `puzzel/model/legacy_adapter.py` | **[NEW]** | Compatibility layer for loading old checkpoints into new architecture |

### 1.2 Configuration Profiles

| File | Action | Changes |
|---|---|---|
| `configs/tiny.toml` | **[PRESERVE]** | Keep existing |
| `configs/small.toml` | **[PRESERVE]** | Keep existing |
| `configs/medium.toml` | **[PRESERVE]** | Keep existing |
| `configs/large.toml` | **[NEW]** | Large model profile (12-16 layers, d_model=768-1024) |
| `configs/cpu_safe.toml` | **[NEW]** | CPU-safe small profile with conservative batch size |

### 1.3 Parameter Inspection & Migration

| File | Action | Changes |
|---|---|---|
| `puzzel/model/parameter_inspector.py` | **[NEW]** | Parameter count, memory estimate, layer breakdown |
| `puzzel/checkpoint_migration.py` | **[NEW]** | Checkpoint compatibility check, shape adaptation, migration script |

---

## MODULE 2: Multilingual Intelligence (Features 18–35)

### 2.1 Tokenizer Expansion

| File | Action | Changes |
|---|---|---|
| `puzzel/tokenizer.py` | **[MODIFY]** | Configurable vocab size (8K-32K), language-ID tokens, mask/sep tokens, tokenizer versioning, tokenization efficiency metrics |
| `puzzel/tokenizer_migration.py` | **[NEW]** | Vocabulary migration utilities, corrupted text prevention |

### 2.2 Language Support

| File | Action | Changes |
|---|---|---|
| `puzzel/languages.py` | **[MODIFY]** | Add language detection, additional language codes, Unicode normalization per language |
| `puzzel/data/quality.py` | **[MODIFY]** | Per-language quality rejection rules, Unicode script handling |

### 2.3 Dataset Balancing & Translation

| File | Action | Changes |
|---|---|---|
| `puzzel/data/pipeline.py` | **[MODIFY]** | Language-aware balancing, translation training pairs, code-switching support |
| `puzzle/data/text_dataset.py` | **[MODIFY]** | Code-switching dataset support |

### 2.4 Evaluation Datasets

| File | Action | Changes |
|---|---|---|
| `data/evaluation/` | **[NEW]** | JSONL datasets for all 7 suites × 6 languages (placeholder skeletons) |
| `configs/evaluation/` | **[PRESERVE]** | Keep existing suite specs |

---

## MODULE 3: Professional Training Engine (Features 36–58)

| File | Action | Changes |
|---|---|---|
| `puzzel/training/loop.py` | **[MODIFY]** | FP16/BF16 mixed precision, gradient accumulation, gradient checkpointing, configurable batch/lr, warmup, cosine scheduler, gradient clipping, checkpoint resume, periodic checkpoints, best-checkpoint preservation, early stopping, progress monitoring, loss charts, memory monitoring, tokens/sec, preprocessing optimization, reproducible seeds, training presets, experiment history |
| `puzzel/training/hardware.py` | **[MODIFY]** | Mixed precision memory estimates, hardware-aware batch recommendations, low-memory mode |
| `puzzel/training/experiment_tracker.py` | **[NEW]** | Experiment history, configuration export, experiment comparison |
| `puzzel/training/presets.py` | **[NEW]** | Hardware-aware training presets |
| `puzzel/training/charts.py` | **[NEW]** | Training/validation loss chart generation |
| `puzzel/training/memory_monitor.py` | **[NEW]** | Hardware memory monitoring during training |

---

## MODULE 4: Instruction and Chat Training (Features 59–76)

| File | Action | Changes |
|---|---|---|
| `puzzel/finetuning/__init__.py` | **[NEW]** | SFT package init |
| `puzzel/finetuning/sft_trainer.py` | **[NEW]** | Supervised fine-tuning trainer with response-aware loss masking |
| `puzzel/finetuning/chat_templates.py` | **[NEW]** | Chat templates with system/user/assistant/tool roles |
| `puzzel/finetuning/dpo_trainer.py` | **[NEW]** | Optional DPO preference optimization with preference dataset validation |
| `puzzel/finetuning/dataset_cleaner.py` | **[NEW]** | Dataset cleaning, duplicate instruction detection, conversation quality filtering |
| `puzzel/finetuning/synthetic_generator.py` | **[NEW]** | Synthetic instruction dataset generation support |
| `puzzel/finetuning/evaluator.py` | **[NEW]** | Instruction-following evaluation |
| `configs/instruction_sft.toml` | **[PRESERVE]** | Keep existing data contract spec |

---

## MODULE 5: Modern AI Chat Application (Features 77–104)

| File | Action | Changes |
|---|---|---|
| `frontend/index.html` | **[NEW]** | Android-native-style chat interface HTML |
| `frontend/style.css` | **[NEW]** | Material Design 3-inspired styling, light/dark themes |
| `frontend/script.js` | **[NEW]** | Vanilla JS ES6 chat application, streaming, markdown, syntax highlighting |
| `frontend/assets/` | **[NEW]** | Font Awesome, Chart.js, SweetAlert2 (CDN or local) |

**Frontend features:** Streaming responses, typing indicators, markdown rendering, syntax-highlighted code, copy/regenerate/stop/edit, conversation history, search, rename, delete, pin, export, themes, personality, generation parameters, model selection, responsive sidebar/bottom nav, loading skeletons, error messages, local persistence

---

## MODULE 6: RAG Knowledge Intelligence (Features 105–128)

| File | Action | Changes |
|---|---|---|
| `puzzel/rag/__init__.py` | **[NEW]** | RAG package init |
| `puzzel/rag/extraction.py` | **[NEW]** | PDF, TXT, DOCX, Markdown, CSV, JSON extraction |
| `puzzel/rag/chunking.py` | **[NEW]** | Document chunking with configurable size/overlap |
| `puzzel/rag/embeddings.py` | **[NEW]** | Embedding model integration (configurable, real embedding model) |
| `puzzel/rag/vector_db.py` | **[NEW]** | Vector database integration |
| `puzzel/rag/retrieval.py` | **[NEW]** | Semantic search, keyword search, hybrid retrieval, context ranking |
| `puzzel/rag/citations.py` | **[NEW]** | Source citations, document reference links |
| `puzzel/rag/management.py` | **[NEW]** | Document management dashboard, deletion, reindexing, multiple KBs, permissions |
| `puzzel/rag/evaluation.py` | **[NEW]** | Retrieval relevance evaluation, grounded-response, hallucination reduction |

---

## MODULE 7: AI Tools and Agents (Features 129–148)

| File | Action | Changes |
|---|---|---|
| `puzzel/agents/__init__.py` | **[NEW]** | Agents package init |
| `puzzel/agents/registry.py` | **[NEW]** | Tool registration, selection, routing |
| `puzzel/agents/tools/calculator.py` | **[NEW]** | Calculator tool |
| `puzzel/agents/tools/datetime.py` | **[NEW]** | Date and time tool |
| `puzzel/agents/tools/unit_conversion.py` | **[NEW]** | Unit conversion tool |
| `puzzel/agents/tools/web_search.py` | **[NEW]** | Web search integration |
| `puzzel/agents/tools/document_search.py` | **[NEW]** | Document search tool |
| `puzzel/agents/sandbox.py` | **[NEW]** | Sandboxed Python execution (isolated, timeouts, restrictions) |
| `puzzel/agents/executor.py` | **[NEW]** | Tool execution status, result integration, failure recovery |
| `puzzel/agents/multi_step.py` | **[NEW]** | Multi-step agents, execution limits, permission controls, iteration logs |
| `puzzel/agents/custom_framework.py` | **[NEW]** | Custom tool framework |

---

## MODULE 8: Professional API and Security (Features 149–170)

| File | Action | Changes |
|---|---|---|
| `puzzel/api.py` | **[MODIFY]** | API key auth, user auth, RBAC, request validation, rate limiting, request-size limits, input sanitization, CORS, env-based secrets, usage tracking, privacy logging, monitoring dashboard, abuse detection, auth failure monitoring, file upload validation, data privacy, audit logs, admin API management, production deployment config |
| `puzzel/auth/__init__.py` | **[NEW]** | Authentication package |
| `puzzel/auth/models.py` | **[NEW]** | User, role, permission models |
| `puzzel/auth/middleware.py` | **[NEW]** | API key auth, JWT tokens, role-based middleware |
| `puzzel/auth/security.py` | **[NEW]** | Password hashing, input sanitization, rate limiting |
| `puzzel/api/middleware.py` | **[NEW]** | CORS, request validation, security headers |
| `puzzel/api/audit.py` | **[NEW]** | Audit logs, monitoring dashboard |
| `puzzel/api/usage.py` | **[NEW]** | Usage tracking, abuse detection |
| `.env.example` | **[NEW]** | Environment variables template |
| `puzzel/database/__init__.py` | **[NEW]** | Database package |
| `puzzel/database/models.py` | **[NEW]** | Users, conversations, messages, API keys, usage records, documents, KBs, benchmark history, audit events, feedback |
| `puzzel/database/migrations.py` | **[NEW]** | Database migrations, indexes, validation |
| `puzzel/database/backup.py` | **[NEW]** | Backup and restore functionality |

---

## MODULE 9: Benchmark and Performance Dashboard (Features 171–192)

| File | Action | Changes |
|---|---|---|
| `puzzel/benchmark/__init__.py` | **[NEW]** | Benchmark package |
| `puzzel/benchmark/dashboard.py` | **[NEW]** | Central evaluation dashboard |
| `puzzel/benchmark/suites.py` | **[NEW]** | Benchmark management, all 7 capability evaluations |
| `puzzel/benchmark/reporting.py` | **[NEW]** | Response quality scoring, language-specific reports, checkpoint comparisons, before/after, training progress charts, inference latency, tokens/sec, memory monitoring, JSON/CSV export |
| `puzzel/benchmark/regression.py` | **[NEW]** | Automated regression tests, historical benchmark tracking |

---

## MODULE 10: Advanced Inference and Optimization (Features 193–212)

| File | Action | Changes |
|---|---|---|
| `puzzel/inference.py` | **[MODIFY]** | KV-cache, faster autoregressive generation, streaming token generation, configurable generation limits, batch inference, quantization, CPU/CUDA optimization, memory-efficient attention, faster model loading, checkpoint loading optimization, inference benchmarking, automatic device selection, request queue management, generation cancellation, concurrent requests, model warm-up, low-memory inference, inference configuration profiles |
| `puzzel/inference/kv_cache.py` | **[NEW]** | KV-cache implementation |
| `puzzel/inference/streaming.py` | **[NEW]** | Streaming token generation |
| `puzzel/inference/batching.py` | **[NEW]** | Batch inference, request queue management |
| `puzzel/inference/quantization.py` | **[NEW]** | Quantization support |
| `puzzel/inference/optimizer.py` | **[NEW]** | CPU/CUDA optimization, memory-efficient attention |

---

## MODULE 11: Additional Advanced Enhancements (Features 213–230)

| File | Action | Changes |
|---|---|---|
| `puzzel/memory/__init__.py` | **[NEW]** | Long-term conversational memory |
| `puzzel/memory/user_prefs.py` | **[NEW]** | User-specific AI preferences |
| `puzzel/memory/system_prompts.py` | **[NEW]** | Custom system prompts |
| `puzzel/voice/__init__.py` | **[NEW]** | Voice input/TTS (optional, graceful fallback) |
| `puzzel/vision/__init__.py` | **[NEW]** | Vision model integration (optional, graceful fallback) |
| `puzzel/plugins/__init__.py` | **[NEW]** | Plugin architecture |
| `puzzel/models/management.py` | **[NEW]** | Model import/export, multiple checkpoint management, version history |
| `puzzel/admin/__init__.py` | **[NEW]** | Admin dashboard |
| `puzzel/admin/dashboard.py` | **[NEW]** | Admin interface (system overview, model info, checkpoint mgmt, training controls, dataset mgmt, fine-tuning mgmt, RAG doc mgmt, KB mgmt, API mgmt, user mgmt, auth settings, usage analytics, benchmark dashboard, performance monitoring, error logs, audit logs, backup mgmt, system diagnostics) |
| `puzzel/monitoring/__init__.py` | **[NEW]** | Monitoring package |
| `puzzel/monitoring/health.py` | **[NEW]** | Model health monitoring, system diagnostics |
| `puzzel/monitoring/analytics.py` | **[NEW]** | Usage analytics |
| `puzzel/monitoring/feedback.py` | **[NEW]** | User feedback and ratings |
| `puzzel/monitoring/safety.py` | **[NEW]** | AI safety monitoring |
| `puzzel/offline/__init__.py` | **[NEW]** | Offline inference mode |

---

## Documentation Files

| File | Action | Changes |
|---|---|---|
| `README.md` | **[MODIFY]** | Update for upgraded platform |
| `docs/deployment.md` | **[NEW]** | Complete deployment guide (local + production) |
| `docs/training.md` | **[MODIFY]** | Expand with new training workflows |
| `docs/api.md` | **[NEW]** | API documentation |
| `docs/frontend.md` | **[NEW]** | Frontend documentation |
| `docs/rag.md` | **[NEW]** | RAG documentation |
| `docs/agents.md` | **[NEW]** | Agents documentation |
| `docs/security.md` | **[NEW]** | Security documentation |
| `docs/migration.md` | **[NEW]** | Migration documentation |
| `docs/testing.md` | **[NEW]** | Testing documentation |
| `docs/troubleshooting.md` | **[NEW]** | Troubleshooting guide |

---

## Test Files (New)

| File | Action | Coverage |
|---|---|---|
| `tests/test_rope.py` | **[NEW]** | RoPE tests |
| `tests/test_rmsnorm.py` | **[NEW]** | RMSNorm tests |
| `tests/test_swiglu.py` | **[NEW]** | SwiGLU tests |
| `tests/test_gqa.py` | **[NEW]** | Grouped Query Attention tests |
| `tests/test_kv_cache.py` | **[NEW]** | KV-cache correctness tests |
| `tests/test_quantization.py` | **[NEW]** | Quantization tests |
| `tests/test_sft.py` | **[NEW]** | SFT tests |
| `tests/test_dpo.py` | **[NEW]** | DPO tests |
| `tests/test_rag.py` | **[NEW]** | RAG retrieval tests |
| `tests/test_citations.py` | **[NEW]** | Citation tests |
| `tests/test_agents.py` | **[NEW]** | Agent iteration-limit tests |
| `tests/test_tool_security.py` | **[NEW]** | Tool execution security tests |
| `tests/test_auth.py` | **[NEW]** | Authentication tests |
| `tests/test_rate_limit.py` | **[NEW]** | Rate-limit tests |
| `tests/test_api_security.py` | **[NEW]** | API security tests |
| `tests/test_database_migrations.py` | **[NEW]** | Database migration tests |
| `tests/test_backup_restore.py` | **[NEW]** | Backup and restore tests |
| `tests/test_benchmark.py` | **[NEW]** | Benchmark tests |
| `tests/test_performance.py` | **[NEW]** | Performance regression tests |
| `tests/test_frontend.py` | **[NEW]** | Frontend functionality tests |

---

## Total File Count

- **New files:** ~65
- **Modified files:** ~15
- **Preserved files:** ~25
- **Documentation:** ~10 new + 5 modified

---

## Implementation Order (PART 13 workflow)

1. STEP 1: Inspect original ZIP ✅ (complete)
2. STEP 2: Existing architecture report ✅ (data/reports/existing_architecture_report.md)
3. STEP 3: File modification plan ✅ (this document)
4. STEP 4: Risk and compatibility issues (see below)
5. STEP 5: Create backup ✅ (/home/user/LLM-Jeevan-backup-20261031-180933)
6. STEP 6: Implement model modernization (RoPE, RMSNorm, SwiGLU, GQA, KV-cache)
7. STEP 7: Implement tokenizer and multilingual upgrades
8. STEP 8: Implement training and fine-tuning (SFT, DPO)
9. STEP 9: Implement inference optimizations (KV-cache, streaming, quantization)
10. STEP 10: Implement RAG
11. STEP 11: Implement tools and agents
12. STEP 12: Implement API and security
13. STEP 13: Implement frontend
14. STEP 14: Implement admin and analytics
15. STEP 15: Integrate all modules
16. STEP 16: Run automated tests
17. STEP 17: Fix errors and regressions
18. STEP 18: Produce deployment guide
19. STEP 19: Produce final feature verification report

---

## Compatibility Notes

- Existing `puzzel-checkpoint-v1` format preserved; migration utilities added for vocab/d_model changes
- Existing API endpoints (/health, /model/info, /generate) remain functional
- Existing training CLI (`python -m puzzel`) remains functional with new config options
- Existing evaluation suite specs unchanged; new JSONL datasets added alongside
- Legacy inference mode maintained for old checkpoints where technically possible
